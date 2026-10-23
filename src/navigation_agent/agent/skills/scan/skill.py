"""按需转向观察与历史方向引用；覆盖状态不控制导航权限。"""

import copy
import math

from navigation_agent.agent.tools.registry import ToolCall


class ScanSkill:
    def __init__(self, *, policy, same_view_angle_deg, look_angle_deg):
        self.policy = policy
        self.same_view_angle_deg = same_view_angle_deg
        self.look_angle_deg = look_angle_deg
        self.scan_anchor = None
        self.scan_headings = []
        self.start_step = None
        self.direction = None
        self.complete = False
        self.look_count = 0
        self.yaw = 0.0
        self.pause_reason = None
        self.number = 1
        self.views = []
        self.actions = []
        self.selected_view_id = None
        self.checked_views = set()
        self.current_view_id = None
        self.phase = 'observing'
        self.last_outcome = None

    def allowed_look_directions(self):
        return [
            direction for direction, delta in (
                ('left', self.look_angle_deg), ('right', -self.look_angle_deg)
            )
            if not self.reinspection_exhausted(self.observed_view(self.yaw + delta))
        ]

    def required(self):
        return False

    def unobserved_look_directions(self):
        return [
            direction for direction, delta in (
                ('left', self.look_angle_deg), ('right', -self.look_angle_deg)
            )
            if not any(angle_distance(self.yaw + delta, heading) <= self.same_view_angle_deg
                       for heading in self.scan_headings)
        ]

    def observed_view(self, heading):
        """按本次拍摄位置和预计朝向匹配最近证据，不把转向小幅漂移当成新区域。"""
        if self.pending_view['heading_deg'] is None:
            return None
        return next((
            view for view in reversed(self.views)
            if view['heading_deg'] is not None and self.near_current_position(view)
            and angle_distance(heading, view['heading_deg']) <= self.same_view_angle_deg
        ), None)

    def look_options(self):
        options = {}
        pose_known = (self.pending_view['observer_position'] is not None
                      and self.pending_view['heading_deg'] is not None)
        for direction in ('left', 'right'):
            delta = self.look_angle_deg if direction == 'left' else -self.look_angle_deg
            previous = self.observed_view(self.yaw + delta) if pose_known else None
            if not pose_known:
                note = '位姿缺失，不能可靠判断是否看过；按当前图说明观察目的'
            elif self.reinspection_exhausted(previous):
                note = '该方向已完成一次复核；不能再转回，选择未看方向或当前可见的观察点'
            elif previous:
                note = f'会回到附近已看方向 {previous["id"]}；需要具体复核问题和新依据'
            else:
                note = '当前附近没有该方向的观察记录；仅在有具体问题时观察，当前有可用路线可直接前往'
            options[direction] = {
                'heading_deg': (self.yaw + delta + 180) % 360 - 180,
                'repeat_of': previous['id'] if previous else None,
                'position_known': pose_known,
                'allowed': not self.reinspection_exhausted(previous),
                'returned_times': self.return_count(previous) if previous else 0,
                'note': note,
            }
        return options

    def unseen_directions_after(self, view):
        """旧方向可作转向中间点；只有实际拍到该方向才算看过。"""
        if view['heading_deg'] is None or not self.near_current_position(view):
            return []
        return [
            direction for direction, sign in (('left', 1), ('right', -1))
            if self.observed_view(view['heading_deg'] + sign * self.look_angle_deg) is None
        ]

    def unseen_via_orient(self):
        routes = []
        for view in self.choices([], self.yaw, self.same_view_angle_deg):
            directions = self.unseen_directions_after(view)
            if directions and not self.reinspection_exhausted(view):
                routes.append({'view_id': view['id'], 'then_look': directions})
        return routes[-2:]

    def direction_visits(self, view):
        """只统计本次停留实际到过的朝向，重复取帧不增加次数。"""
        if view is None or view.get('heading_deg') is None:
            return 0
        nearby = []
        for observed in reversed(self.views):
            if not self.near_current_position(observed):
                break
            if observed['heading_deg'] is not None:
                nearby.append(observed)
        visits = 0
        previous_heading = None
        for observed in reversed(nearby):
            heading = observed['heading_deg']
            if (previous_heading is not None
                    and angle_distance(heading, previous_heading) <= self.same_view_angle_deg):
                continue
            previous_heading = heading
            if angle_distance(heading, view['heading_deg']) <= self.same_view_angle_deg:
                visits += 1
        return visits

    def return_count(self, view):
        return max(0, self.direction_visits(view) - 1)

    def reinspection_exhausted(self, view):
        return self.direction_visits(view) >= 2

    def direction_candidate_ids(self, view):
        """保留原目击；仅在转回复核明确未确认时撤销这个方向的可用候选。"""
        if view['heading_deg'] is None:
            return []
        candidate_ids = {}
        for earlier in self.views:
            if (earlier['heading_deg'] is None or not self.near_current_position(earlier)
                    or angle_distance(earlier['heading_deg'], view['heading_deg']) > self.same_view_angle_deg):
                continue
            candidate_ids.update(dict.fromkeys(earlier['candidate_ids']))
            for candidate_id in earlier.get('unconfirmed_candidate_ids', []):
                candidate_ids.pop(candidate_id, None)
        return list(candidate_ids)

    def orient_options(self, candidates, yaw, tolerance):
        """转回选项同时说明后续用途和真实重复次数，帮助连续观察。"""
        states = {candidate['id']: candidate for candidate in candidates}
        options = []
        for view in self.choices(candidates, yaw, tolerance):
            route_ids = self.direction_candidate_ids(view)
            routes = [candidate_id for candidate_id in route_ids
                      if candidate_id in states
                      and states[candidate_id]['status'] != 'avoid_from_here'
                      and states[candidate_id].get('no_new_information_count', 0) < 2]
            options.append({
                'view_id': view['id'],
                'turn_deg': round((view['heading_deg'] - yaw + 180) % 360 - 180, 1),
                'returned_times': self.return_count(view),
                'allowed': not self.reinspection_exhausted(view),
                'candidate_ids': routes,
                'unseen_then_look': self.unseen_directions_after(view),
            })
        return options

    def repeated_view(self, decision):
        if decision.action == 'look':
            delta = self.look_angle_deg if decision.direction == 'left' else -self.look_angle_deg
            return self.observed_view(self.yaw + delta)
        if decision.action == 'orient':
            return next((view for view in self.views if view['id'] == decision.view_id), None)
        return None

    def validate_reinspection(self, decision):
        previous = self.repeated_view(decision)
        if previous is None:
            return
        if self.reinspection_exhausted(previous):
            raise ValueError(
                '已从当前位置转回这个方向并完成一次复核；首次观察加一次复核的次数已用尽。'
                'look/orient 共用次数，换 O 编号、措辞、候选或 progress 都不能重置。'
                '请选择未看方向、当前可见的观察点，或 action_options 中已允许的 recover。'
            )
        intent = decision.observation_intent
        # orient 的实际转向由已校验的 view_id 决定；参考线索可以来自另一张旧图。
        if intent is None or (decision.action == 'look' and intent.reference_view_id != previous['id']):
            raise ValueError(
                f'本次转向会重复附近已看方向 {previous["id"]}。如需复核，填写 observation_intent：'
                f'reference_view_id={previous["id"]}、question（具体疑问）、new_evidence（为何再看有用）。'
                '未找到目标或扩大视野不是重复依据；也可选未看方向、转回准备接近或当前可见候选'
            )
        if decision.action == 'orient':
            return
        has_visual_progress = (
            decision.previous_action_review is not None
            and decision.previous_action_review.outcome == 'progress'
        )
        has_target_clue = (self.pause_reason == 'target' or decision.target_review is not None
                           or any(candidate.kind == 'target' for candidate in decision.candidates))
        if decision.action == 'look' and not has_target_clue and not has_visual_progress:
            raise ValueError(
                '该方向已有观察，尚未找到目标不是新依据。选择未看方向或当前可见路线；'
                '若已选中历史路线，使用 orient 转回准备接近；环境变化须给出 progress 视觉复核'
            )

    def coverage_complete(self):
        return self.complete

    def heading_gap(self):
        return heading_gap(self.scan_headings)

    def prepare_call(self, decision):
        if decision.action == 'look':
            arguments = {'direction': decision.direction}
        else:
            arguments = {'heading_deg': self.target_heading(decision.view_id)}
        return ToolCall(decision.action, arguments), {'action': decision.action, **arguments}

    def observe(self, view):
        # 每帧都有独立身份，重复朝向也不覆盖旧图或借用旧编号。
        self.pending_view = copy.deepcopy(view)
        self.current_view_id = f'O{view["step"]}'
        position = view['observer_position']
        if self.start_step is None:
            self.start_step = view['step']
            self.scan_anchor = copy.deepcopy(position)
        if self.scan_anchor is None and position is not None:
            self.scan_anchor = copy.deepcopy(position)
        self.yaw = view['heading_deg'] if view['heading_deg'] is not None else 0.0
        previous = self.observed_view(self.yaw) if view['heading_deg'] is not None else None
        turned_back = (
            previous is not None and self.views
            and self.views[-1]['heading_deg'] is not None
            and self.near_current_position(self.views[-1])
            and angle_distance(self.views[-1]['heading_deg'], self.yaw) > self.same_view_angle_deg
        )
        self.views.append({
            'id': self.current_view_id, 'candidate_ids': [],
            'reinspection_of': previous['id'] if turned_back else None,
            'step': view['step'], 'heading_deg': view['heading_deg'],
            'image': view.get('image'),
            'observer_position': copy.deepcopy(view['observer_position']),
        })
        self.scan_headings = [
            item['heading_deg'] for item in self.views
            if item['heading_deg'] is not None
            and self.near_current_position(item)
        ]
        # 覆盖只作当前附近的观察摘要，不触发或阻止运动。
        self.complete = self.heading_gap() <= self.policy.scan_max_heading_gap_deg
        if self.phase == 'aligning':
            self.phase = 'confirming'
        elif self.phase == 'exploring':
            self.phase = 'observing'

    def near_current_position(self, view):
        position = self.pending_view['observer_position']
        previous = view['observer_position']
        return (
            position is not None and previous is not None
            and math.dist(position, previous) < self.policy.scan_reset_distance_m
        )

    def describe(self, decision, candidates):
        item = next((item for item in self.views if item['id'] == self.current_view_id), None)
        if item is not None:
            item['candidate_ids'] = [
                candidate['id'] for candidate in candidates
                if candidate.get('last_seen_step') == item['step']
            ]
            if item.get('reinspection_of') and 'candidates' in decision.model_fields_set:
                states = {candidate['id']: candidate for candidate in candidates}
                visible_origins = {
                    candidate.get('origin_candidate_id', candidate['id'])
                    for candidate in candidates if candidate['id'] in item['candidate_ids']
                }
                item['unconfirmed_candidate_ids'] = [
                    candidate_id for candidate_id in self.direction_candidate_ids(item)
                    if states.get(candidate_id, {}).get('origin_candidate_id', candidate_id)
                    not in visible_origins
                ]

    def choices(self, candidates, yaw, tolerance):
        # 只提供当前位置附近的历史方向；相近朝向优先保留最新证据。
        if yaw is None:
            return []
        selected = []
        for item in reversed(self.views):
            if (not item.get('image') or item['heading_deg'] is None
                    or not self.near_current_position(item)
                    or angle_distance(item['heading_deg'], yaw) <= tolerance):
                continue
            if any(angle_distance(item['heading_deg'], chosen['heading_deg']) <= tolerance
                   for chosen in selected):
                continue
            selected.append(item)
        return list(reversed(selected))

    def useful_choices(self, candidates, yaw, tolerance):
        """已复核方向不强迫重访；未尝试候选提醒模型继续寻找证据。"""
        options = self.orient_options(candidates, yaw, tolerance)
        useful_ids = {
            option['view_id'] for option in options
            if option['allowed'] and option['candidate_ids']
        }
        return [view for view in self.choices(candidates, yaw, tolerance) if view['id'] in useful_ids]

    def select(self, view_id):
        self.selected_view_id = view_id
        self.checked_views.add(view_id)
        self.phase = 'aligning'

    def target_heading(self, view_id):
        return next(item['heading_deg'] for item in self.views if item['id'] == view_id)

    def record_result(self, action, result, *, candidate_id=None, view_id=None):
        self.actions.append({
            'action': action, 'candidate_id': candidate_id, 'view_id': view_id,
            'result': copy.deepcopy(result),
            'observer_position': copy.deepcopy(self.pending_view['observer_position']),
            'heading_deg': self.target_heading(view_id) if view_id else self.yaw,
        })
        self.actions = self.actions[-12:]
        if action == 'navigate':
            self.phase = 'exploring'

    def finish(self, reason):
        outcome = {
            'session_id': f'S{self.number}', 'reason': reason,
            'selected_view_id': self.selected_view_id,
            'last_action': copy.deepcopy(self.actions[-1]) if self.actions else None,
            'directions': [
                {'id': view['id'], 'heading_deg': view['heading_deg'],
                 'candidate_ids': list(view['candidate_ids']), 'step': view['step']}
                for view in self.views
            ],
        }
        self.last_outcome = outcome
        self.views = []
        self.actions = []
        self.selected_view_id = None
        self.checked_views = set()
        self.current_view_id = None
        self.phase = 'observing'
        return outcome

    def context(self, *, candidates, yaw, tolerance):
        choices = self.choices(candidates, yaw, tolerance)
        current = next((view for view in self.views if view['id'] == self.current_view_id), None)
        return {
            'session_id': f'S{self.number}',
            'phase': 'target_confirmation' if self.pause_reason else self.phase,
            'current_view_id': self.current_view_id,
            'selected_view_id': self.selected_view_id,
            'orient_view_ids': [item['id'] for item in choices if not self.reinspection_exhausted(item)],
            'views': copy.deepcopy([*choices, *([current] if current else [])]),
            'reviewed_view_ids': sorted(self.checked_views),
            'actions': copy.deepcopy(self.actions),
            'previous_result': copy.deepcopy(self.last_outcome),
            'scope': '仅提供当前位置附近的方向证据；每帧编号固定，历史图不证明当前可见。',
        }


def angle_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


def heading_gap(headings):
    values = sorted(set(heading % 360 for heading in headings))
    if len(values) < 2:
        return 360.0
    return max((b - a) % 360 for a, b in zip(values, values[1:] + values[:1]))
