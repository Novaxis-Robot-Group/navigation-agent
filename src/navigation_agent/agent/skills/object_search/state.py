"""当前任务的候选、运动约束与探索子任务上下文。"""

import copy
import math
from navigation_agent.agent.config import ExplorationSettings, ObservationSettings, ExecutionSettings

from navigation_agent.agent.skills.object_search import rules as exploration_policy
from navigation_agent.agent.skills.scan.skill import ScanSkill
from navigation_agent.agent.skills.object_search.target import TargetTracking


class ExecutionState:
    def __init__(self, *, look_angle_deg, exploration: ExplorationSettings,
                 observation: ObservationSettings, execution: ExecutionSettings, navigation_policy=None, scene_validator=None):
        self.navigation_policy = navigation_policy
        self.scene_validator = scene_validator
        if navigation_policy is not None and scene_validator is None:
            raise ValueError("场景策略必须由部署入口提供校验器")
        self.policy = exploration
        self.same_view_angle_deg = observation.same_view_angle_deg
        self.max_unfinished_candidates = execution.max_unfinished_candidates
        self.target = TargetTracking(exploration.max_target_reacquire_actions)
        self.recent_views = []
        self.current = None
        self.step = 0
        self.yaw = 0.0
        self.candidates = []
        self.recovery_required = False
        self.recovery_attempts = 0
        self.must_change_position = False
        self.last_action = None
        self.scan = ScanSkill(policy=exploration, same_view_angle_deg=observation.same_view_angle_deg,
                              look_angle_deg=look_angle_deg)
        self.last_navigation = None
        self.subgoal = None
        self.subgoal_history = []
        self.reviewed_action_step = None
        self.recovery_anchor = None

    def validate_scene(self, decision):
        if self.scene_validator is not None:
            self.scene_validator(decision)

    def observe(self, view):
        # 地点与参考图由 SpatialMemory 查询，只缓存当前观察的标识。
        self.recent_views = (self.recent_views + [copy.deepcopy(view)])[-3:]
        self.scan.observe(view)
        self.current = view
        self.step = view['step']
        self.yaw = view['heading_deg'] if view['heading_deg'] is not None else 0.0
        position = view['observer_position']
        # 脱困位置约束按实际平移解除，不依赖模型的新区域判断。
        if (self.recovery_anchor is not None and position is not None
                and math.dist(position, self.recovery_anchor) >= self.policy.scan_reset_distance_m):
            self.recovery_required = False
            self.recovery_attempts = 0
            self.must_change_position = False
            self.recovery_anchor = None

    def navigation_started(self, candidate_id, *, continue_inspection=False):
        candidate = next(item for item in self.candidates if item['id'] == candidate_id)
        origin = candidate.get('origin_candidate_id', candidate['id'])
        if candidate['kind'] != 'target':
            # 换观察点可延续同一个检查问题，但不把不同入口合并为同一条路线。
            continuing = (continue_inspection and candidate['kind'] == 'viewpoint'
                          and self.subgoal is not None)
            if self.subgoal is None or (self.subgoal['origin_candidate_id'] != origin and not continuing):
                self._end_subgoal('switched', '选择新的探索路线')
                self.subgoal = {
                    'origin_candidate_id': origin, 'started_step': self.step,
                    'status': 'active', 'no_new_information_count': 0,
                    'description': candidate['description'],
                    'question': candidate.get('expected_new_view', ''),
                    'evidence': candidate['evidence'],
                }
            origins = self.subgoal.setdefault(
                'inspection_candidate_origins', [self.subgoal['origin_candidate_id']],
            )
            if origin not in origins:
                origins.append(origin)
            self.subgoal.update(candidate_id=candidate['id'], local_destination=candidate['description'],
                                local_question=candidate.get('expected_new_view', ''),
                                evidence_step=self.step)
        else:
            self._end_subgoal('target_found', '当前已见具体目标，转入目标跟进')
        if candidate['kind'] == 'target' and self.current.get('image'):
            self.target.begin(candidate, self.current)
            self.scan.pause_reason = 'target'
        self.last_navigation = {
            'step': self.step,
            'candidate_id': candidate_id,
            'intent': candidate.get('expected_new_view') or candidate['description'],
            'image': self.current.get('image'),
            'heading_deg': self.current.get('heading_deg'),
            'observer_position': copy.deepcopy(self.current['observer_position']),
        }

    def _end_subgoal(self, status, evidence):
        if self.subgoal is not None:
            self.subgoal_history.append({**self.subgoal, 'status': status,
                                         'ended_step': self.step, 'evidence': evidence})
            self.subgoal_history = self.subgoal_history[-4:]
            self.subgoal = None

    def _apply_review(self, decision):
        if decision.subgoal_review is not None and self.subgoal is not None:
            review = decision.subgoal_review
            if review.status in ('completed', 'abandoned'):
                self._end_subgoal(review.status, review.evidence)
            else:
                self.subgoal['review'] = review.model_dump()
        deferred = copy.deepcopy(self.target.active) if self.target.defers_local_followup(decision) else None
        if self.target.apply_review(decision, self.current):
            self.scan.pause_reason = None
        if deferred is not None:
            # 复用已有局部目的历史和拓扑结论，不另建目标记忆。
            candidate = next(item for item in self.candidates if item['id'] == deferred['candidate_id'])
            self.subgoal_history.append({
                'origin_candidate_id': candidate.get('origin_candidate_id', candidate['id']),
                'started_step': deferred['first_seen_step'], 'ended_step': self.step,
                'status': 'abandoned', 'description': deferred['description'],
                'question': '定向找回并核实该目标线索',
                'evidence': decision.subgoal_review.evidence + '；保留O'
                            + str(deferred['last_seen_step']) + '的历史目击，暂缓不代表目标不存在。',
            })
            self.subgoal_history = self.subgoal_history[-4:]
        review = decision.previous_action_review
        if review is None or self.last_action is None:
            return
        if (self.last_action['action'] in ('look', 'orient') and review.outcome == 'progress'
                and self.subgoal is not None):
            origin = self.subgoal['origin_candidate_id']
            for item in self.candidates:
                if item.get('origin_candidate_id', item['id']) == origin:
                    item['no_new_information_count'] = 0
            self.subgoal['no_new_information_count'] = 0
        if (self.last_action['action'] != 'navigate'
                or self.reviewed_action_step == self.last_action['step']):
            return
        self.reviewed_action_step = self.last_action['step']
        candidate_id = self.last_action.get('candidate_id')
        selected = next((item for item in self.candidates if item['id'] == candidate_id), None)
        if selected:
            origin_id = selected.get('origin_candidate_id', selected['id'])
            count = selected.get('no_new_information_count', 0)
            result = self.last_action.get('result') or {}
            detection_pause = (result.get('reason') == 'agent_interrupt'
                               and result.get('detail') == 'target_detected')
            # 途中检测暂停没有完成原路线，不用未走完的路累计无收益。
            if review.outcome == 'no_new_information' and not detection_pause:
                count += 1
            elif review.outcome == 'progress':
                count = 0
            for item in self.candidates:
                if item.get('origin_candidate_id', item['id']) == origin_id:
                    item['no_new_information_count'] = count
                    item['visual_review'] = review.model_dump()
        if self.subgoal is not None and selected is not None:
            self.subgoal['no_new_information_count'] = count
        if self.last_navigation:
            self.last_navigation['visual_review'] = review.model_dump()

    def resolve(self, decision):
        """先按本轮观察校验，再在副本上复核；拒绝动作不改变执行状态。"""
        proposed = copy.deepcopy(self)
        return proposed._resolve(decision)

    def _resolve(self, decision):
        """仅在副本上处理复核和候选；动作条件先按原观察检查。"""
        self.validate_scene(decision)
        exploration_policy.validate_action(self, decision)
        self._apply_review(decision)

        # 先在副本上校验，失败时不污染执行状态
        candidates = copy.deepcopy(self.candidates)
        # 结束任务不需要选择导航候选，附带的旧编号不应阻止停车收尾。
        if decision.action in ('finish', 'give_up'):
            return candidates, None
        aliases = {}

        # 重新看见历史候选时，按观察点分别保留接近记录。
        for proposed in decision.candidates:
            if not exploration_policy.candidate_is_current(proposed, self.scan.current_view_id):
                continue
            if proposed.id in aliases:
                raise ValueError('候选编号不能重复')
            local = [
                candidate for candidate in candidates if candidate['place_id'] == self.current['id']
            ]
            origin_id = None
            item = None
            source_id = proposed.id
            if proposed.kind == 'target' and self.target.active and source_id.startswith('new_'):
                source_id = self.target.active['candidate_id']
            # 新编号表示新探索入口；延续同一入口须由模型显式沿用 C 编号。
            if source_id.startswith('C'):
                source = next(
                    (candidate for candidate in self.candidates if candidate['id'] == source_id),
                    None,
                )
                if source is None:
                    raise ValueError('候选编号不存在；当前图中新发现的候选请使用 new_1 至 new_4')
                # 关联仅代表模型本轮的重识别，不证明两个目击属于同一物体。
                origin_id = source.get('origin_candidate_id', source['id'])
                item = next(
                    (
                        candidate for candidate in local
                        if candidate.get('origin_candidate_id', candidate['id']) == origin_id
                    ),
                    None,
                )
            if item is None:
                item = {
                    'id': f'C{len(candidates)+1}',
                    'place_id': self.current['id'],
                    'attempt_count': 0,
                    'failed_attempts': 0,
                    'status': 'untried',
                }
                if origin_id is not None:
                    item['origin_candidate_id'] = origin_id
                    # 接近失败仍按观察点隔离；累计探索记录随候选关联延续。
                    for key in ('total_attempt_count', 'no_new_information_count', 'visual_review'):
                        if key in source:
                            item[key] = copy.deepcopy(source[key])
                candidates.append(item)
            if item['id'] in aliases.values():
                raise ValueError('同一候选不能在本轮使用多个编号重复提交')
            item.update(
                kind=proposed.kind,
                description=proposed.description,
                evidence=proposed.evidence,
                expected_new_view=proposed.expected_new_view,
                last_seen_step=self.step,
                view_id=self.scan.current_view_id,
                source='model_visual_report',
            )
            aliases[proposed.id] = item['id']
        selected = None
        if decision.action == 'navigate':
            selected_id = aliases.get(decision.candidate_id)
            if selected_id is None and decision.candidate_id in aliases.values():
                selected_id = decision.candidate_id
            if selected_id is None:
                raise ValueError(
                    'navigate 必须引用本次 candidates 中当前可见的候选；不能盲走向历史入口或用历史图授权导航'
                )
            selected = next(
                candidate
                for candidate in candidates
                if candidate['id'] == selected_id
            )
            exploration_policy.validate_candidate(self, selected)
        return candidates, selected['id'] if selected else None

    def accept(self, decision):
        proposed = copy.deepcopy(self)
        candidates, selected = proposed._resolve(decision)
        self.__dict__.update(proposed.__dict__)
        self.candidates = candidates
        if decision.action not in ('finish', 'give_up'):
            self.scan.describe(decision, candidates)
        if decision.action == 'orient':
            self.scan.select(decision.view_id)
        if decision.action in ('look', 'orient') and self.target.active:
            self.target.record_turn()
        if decision.action == 'look':
            self.scan.look_count += 1
        if decision.action == 'recover':
            self.recovery_attempts += 1
            if self.recovery_anchor is None:
                self.recovery_anchor = copy.deepcopy(self.current['observer_position'])
        return selected

    def record_result(self, candidate_id, result):
        if result['reason'].startswith('recover'):
            self.recovery_required = False
            self.must_change_position = result['reason'] != 'recovered'
            return
        if candidate_id is None:
            return
        self.target.navigation_result(result)
        item = next(candidate for candidate in self.candidates if candidate['id'] == candidate_id)
        displacement = result.get('motion', {}).get('displacement_m')

        # 检测停车仅暂停原路线；保留真实结果，但不消耗路线尝试次数。
        if result['reason'] == 'agent_interrupt' and result.get('detail') == 'target_detected':
            item['last_attempt'] = {
                'step': self.step, 'reason': result['reason'], 'detail': result['detail'],
                'displacement_m': displacement, 'failed_approach': False, 'interrupted': True,
            }
            return

        # 程序判断是否受阻，累计尝试次数并保留最后一次结果
        failed, stuck = exploration_policy.classify_result(result, self.policy)
        if stuck:
            self.recovery_required = self.recovery_attempts < self.policy.max_recovery_attempts
            self.must_change_position = True
            self.recovery_anchor = copy.deepcopy(self.current['observer_position'])
        item['attempt_count'] += 1
        origin_id = item.get('origin_candidate_id', item['id'])
        total = item.get('total_attempt_count', 0) + 1
        for related in self.candidates:
            if related.get('origin_candidate_id', related['id']) == origin_id:
                related['total_attempt_count'] = total
        item['last_attempt'] = {
            'step': self.step,
            'reason': result['reason'],
            'displacement_m': displacement,
            'failed_approach': failed,
        }
        item['failed_attempts'] += int(failed)
        item['status'] = (
            'avoid_from_here'
            if item['failed_attempts'] >= self.policy.max_candidate_failures
            else ('approach_failed' if failed else 'attempted_unverified')
        )

    def allowed_look_directions(self):
        return self.scan.allowed_look_directions()

    def scan_state(self):
        return {
            'anchor': self.scan.scan_anchor,
            'observed_headings_deg': self.scan.scan_headings,
            'look_angle_deg': self.scan.look_angle_deg,
            'look_count': self.scan.look_count,
            'coverage_complete': self.scan.coverage_complete(),
            'required': self.scan.required(),
            'direction': self.scan.direction,
            'largest_heading_gap_deg': self.scan.heading_gap(),
            'allowed_look_directions': self.allowed_look_directions(),
            'unobserved_look_directions': self.scan.unobserved_look_directions(),
            'recovery_required': self.recovery_required,
            'recovery_attempts': self.recovery_attempts,
            'must_change_position': self.must_change_position,
        }

    def can_leave_observation_dead_end(self):
        """周围已看却没有可用路线时，允许由设备沿刚走过的路径退回。"""
        target = self.target.active
        local_followup_exhausted = (target is not None and target['phase'] == 'temporarily_lost'
                                    and target.get('reacquire_actions', 0) >= self.target.reacquire_limit)
        if (target and not local_followup_exhausted) or self.must_change_position or not self.scan.coverage_complete():
            return False
        navigation = self.last_navigation or {}
        result = navigation.get('result') or {}
        start, current = navigation.get('observer_position'), self.current['observer_position']
        if (not result or result.get('startup_failed') or start is None or current is None
                or math.dist(start, current) < self.policy.scan_reset_distance_m):
            return False
        local_ids = {
            candidate_id for view in self.scan.views if self.scan.near_current_position(view)
            for candidate_id in self.scan.direction_candidate_ids(view)
        }
        return not any(candidate['id'] in local_ids and candidate['status'] != 'avoid_from_here'
                       and candidate.get('no_new_information_count', 0) < 2
                       for candidate in self.candidates)

    def action_options(self):
        """上下文与校验使用同一观察状态；视觉复核不会隐式开启新一轮。"""
        yaw = self.current['heading_deg'] if self.current['observer_position'] is not None else None
        return {
            'look_directions': self.allowed_look_directions(),
            'look_options': self.scan.look_options(),
            'scan_required': self.scan.required(),
            'unseen_via_orient': self.scan.unseen_via_orient(),
            'orient_options': self.scan.orient_options(self.candidates, yaw, self.same_view_angle_deg),
            'reinspection_rule': '同一位置和朝向只允许首次观察加一次复核，look/orient 共用次数；progress、候选或换 O 编号不能重置。复核未见入口后换未看方向或可见观察点',
            'orient_view_ids': [
                view['id'] for view in self.scan.choices(
                    self.candidates, yaw, self.same_view_angle_deg,
                ) if not self.scan.reinspection_exhausted(view)
            ],
            'navigate': '可立即前往当前可见候选；有进展时延续当前目的，路径须通过部署场景校验',
            'viewpoint_motion': {
                'max_displacement_m': self.policy.viewpoint_max_displacement_m or None,
                'meaning': '局部换位观察，支持实时位姿时到位移上限就停车复核；停车不表示问题已解决，仍需按当前画面决定继续检查或换目的',
            },
            'recover_allowed': (
                (self.recovery_required or self.can_leave_observation_dead_end())
                and self.recovery_attempts < self.policy.max_recovery_attempts
            ),
            'finish': '须有当前图的完成证据；跟进目标时还须当前帧 arrived 复核',
            'give_up': '说明现有证据为什么无法继续，不能因观察次数放弃',
        }

    def snapshot(self):
        return copy.deepcopy(self.context())

    def context(self, *, excluded_route_origins=()):
        place_id = self.current['id']
        return {
            'current_place_id': place_id,
            'target_tracking': self.target.context(self.current),
            'exploration_session': self.scan.context(
                candidates=self.candidates,
                yaw=self.current['heading_deg'] if self.current['observer_position'] is not None else None,
                tolerance=self.same_view_angle_deg,
            ),
            'scan': self.scan_state(),
            'last_action': copy.deepcopy(self.last_action),
            'last_navigation': copy.deepcopy(self.last_navigation),
            'active_subgoal': copy.deepcopy(self.subgoal),
            'recent_subgoals': copy.deepcopy(self.subgoal_history),
            'action_options': self.action_options(),
            'local_candidates': [copy.deepcopy(c) for c in self.candidates if c['place_id'] == place_id],
            'unfinished_elsewhere': sorted([
                copy.deepcopy(c) for c in self.candidates
                if c['place_id'] != place_id
                and c.get('origin_candidate_id', c['id']) not in excluded_route_origins
                and (c['kind'] != 'viewpoint' or c.get('total_attempt_count', 0) > 0)
                and c['status'] in ('untried', 'approach_failed', 'attempted_unverified')
            ], key=lambda item: (
                item['id'] == (self.last_navigation or {}).get('candidate_id'),
                item.get('no_new_information_count', 0) > 0 or item['failed_attempts'] > 0,
                item['last_seen_step'],
            ), reverse=True)[:self.max_unfinished_candidates],
            'limits': '历史入口只有在当前图重新可见时才可导航；接近失败不等于入口不可通行。',
        }
