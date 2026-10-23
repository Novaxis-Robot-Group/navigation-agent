"""找物任务的目标跟进；只保存证据引用，不另存场景记忆。"""

import copy
import math


class TargetTracking:
    def __init__(self, reacquire_limit=4):
        self.active = None
        self.reacquire_limit = reacquire_limit

    def begin(self, candidate, view):
        if self.active is None:
            self.active = {
                'candidate_id': candidate['id'], 'description': candidate['description'],
                'first_seen_step': view['step'], 'last_seen_step': view['step'],
                'phase': 'approaching', 'confirmation_turns': 0,
            }
        self.active['candidate_id'] = candidate['id']
        self.active['last_seen_anchor'] = {
            'step': view['step'], 'image': view.get('image'),
            'observer_position': copy.deepcopy(view.get('observer_position')),
            'heading_deg': view.get('heading_deg'),
            'description': candidate['description'], 'evidence': candidate['evidence'],
        }

    def defers_local_followup(self, decision):
        """暂缓的是本次局部跟进，不否定历史目击，也不结束找物任务。"""
        return bool(
            self.active and decision.target_review is not None
            and decision.target_review.status == 'temporarily_lost'
            and decision.subgoal_review is not None and decision.subgoal_review.status == 'abandoned'
        )

    def apply_review(self, decision, view):
        review = decision.target_review
        if self.active is None:
            if review is not None:
                # 首次目击可以同时提供本帧复核，但不能伪造旧目标经历。
                if (review.observation_step != view['step']
                        or review.status not in ('visible', 'arrived')
                        or decision.scene.target_seen != 'yes'):
                    raise ValueError('首次目标复核须引用当前帧的明确目击，不能填写旧目标状态')
                if review.status == 'arrived' and decision.action != 'finish':
                    raise ValueError('已确认到达应 finish，不继续接近')
            return False
        if review is None:
            if decision.action in ('navigate', 'finish'):
                raise ValueError('接近后先填写 target_review，按当前帧确认目标或说明暂时不可见，不能重复盲走')
            return False
        if review.observation_step != view['step']:
            raise ValueError('target_review 必须引用当前 observation_step，不能把历史目标图当成当前可见')
        if review.status in ('visible', 'arrived') and decision.scene.target_seen == 'no':
            raise ValueError('目标复核为可见或到达时，当前 scene.target_seen 不能为 no')
        if review.status == 'arrived' and decision.action != 'finish':
            raise ValueError('已确认到达应 finish，不再继续 scan 或接近')
        if decision.action == 'finish' and review.status != 'arrived':
            raise ValueError('完成目标须提供当前到达证据，暂时不可见或误识别不能 finish')
        if (review.status == 'temporarily_lost' and decision.subgoal_review is not None
                and decision.subgoal_review.status == 'abandoned'):
            if decision.action == 'give_up':
                raise ValueError('暂缓局部跟进后应按当前证据继续探索，不能仅因此结束整个任务')
            self.active = None
            return True
        if review.status == 'rejected':
            self.active = None
            return True
        self.active['review'] = review.model_dump()
        if review.status in ('visible', 'arrived'):
            self.active['reacquire_actions'] = 0
            self.active['last_seen_anchor'] = {
                'step': view['step'], 'image': view.get('image'),
                'observer_position': copy.deepcopy(view.get('observer_position')),
                'heading_deg': view.get('heading_deg'), 'evidence': review.evidence,
            }
            self.active['last_seen_step'] = view['step']
            self.active['phase'] = 'arrived' if review.status == 'arrived' else 'verifying_arrival'
        else:
            self.active['phase'] = 'temporarily_lost'
        return False

    def navigation_result(self, result):
        if self.active is None:
            return
        # 运动结束不证明目标重新出现，丢失状态须由新视觉证据解除。
        if self.active['phase'] != 'temporarily_lost':
            self.active['phase'] = 'verifying_arrival'
        self.active['last_navigation_result'] = result['reason']

    def record_reacquire_action(self):
        if self.active and self.active["phase"] == "temporarily_lost":
            self.active["reacquire_actions"] = self.active.get("reacquire_actions", 0) + 1

    def record_turn(self):
        self.active['confirmation_turns'] += 1

    def evidence_steps(self):
        if self.active is None:
            return []
        return list(dict.fromkeys([
            self.active['first_seen_step'], self.active['last_seen_step'],
        ]))

    def context(self, view=None):
        if self.active is None:
            return None
        motion = None
        anchor = self.active.get('last_seen_anchor') or {}
        start, current = anchor.get('observer_position'), (view or {}).get('observer_position')
        headings = [anchor.get('heading_deg'), (view or {}).get('heading_deg')]
        if (start is not None and current is not None
                and all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in [*start, *current, *headings])):
            motion = {
                'translation_m': round(math.dist(start, current), 2),
                'heading_change_deg': round((headings[1] - headings[0] + 180) % 360 - 180, 1),
                'meaning': '相机相对最后目击的运动，不是目标当前位置或方向；历史左右不能直接沿用',
            }
        return {
            'camera_motion_since_sighting': motion,
            **copy.deepcopy(self.active),
            'reacquire_actions_remaining': max(0, self.reacquire_limit - self.active.get('reacquire_actions', 0)),
            'reacquire_rule': '目标丢失后先定向观察或短距离换观察点；未暂缓跟进前，不能把新门当成原目标路线。走偏或继续检查无收益时可提前暂缓，不必耗尽局部预算；保持 target_review=temporarily_lost，填写 subgoal_review=abandoned 说明暂缓原因，保留历史线索并恢复探索；不能仅因此 give_up。',
            'scope': '待验证目标线索，先核对类别与特征再判断距离；当前不可见不等于不存在。'
                     '历史目标图仅用于身份对照，不证明当前可见，也不是目标坐标。',
        }
