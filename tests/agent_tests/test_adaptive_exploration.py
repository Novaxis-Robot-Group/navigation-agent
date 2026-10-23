"""按需观察、持续目的与区域证据的行为回归。"""

import copy
from pathlib import Path
import tempfile
import unittest

from navigation_agent.agent.schemas import Decision
from navigation_agent.agent.memory.topology import TopologyMemory
from tests.config_support import execution_state, spatial_memory


def view(step, x=0., heading=0.):
    return dict(id='P1' if x < .75 else 'P2', observer_position=[x, 0.], position=[x, 0.],
                step=step, heading_deg=heading, image=f'images/{step:03d}.jpg')


def navigation(candidate='new_1', **extra):
    return Decision(action='navigate', reason='到入口看清后面的通道', instruction='go',
                    candidate_id=candidate,
                    scene={'target_seen': 'no', 'visible_ground': '到入口地面连续可见'},
                    candidates=[dict(id=candidate, evidence_source='current', kind='open_passage', description='入口',
                                     evidence='当前可见入口及连续地面', expected_new_view='确认门后通道')],
                    **extra)


class AdaptiveExplorationTests(unittest.TestCase):
    def test_unattempted_viewpoint_does_not_remain_pending_after_leaving(self):
        state = execution_state()
        state.observe(view(1))
        proposal = Decision(action='look', direction='left', reason='查看左侧',
                            candidates=[
                                dict(id='new_1', kind='viewpoint', description='向左转头',
                                     evidence='左侧尚未看清', expected_new_view='看清左侧'),
                                dict(id='new_2', kind='open_passage', description='白色入口',
                                     evidence='开口内地面可见', expected_new_view='查看门后'),
                            ])
        state.accept(proposal)
        state.observe(view(2, x=2.))
        pending = state.context()['unfinished_elsewhere']
        self.assertEqual([item['kind'] for item in pending], ['open_passage'])
        self.assertEqual(len(state.candidates), 2)

    def test_closed_routes_are_filtered_before_the_pending_limit(self):
        state = execution_state()
        state.max_unfinished_candidates = 1
        state.observe(view(3, x=2.))
        state.candidates = [
            dict(id='C1', place_id='P1', kind='open_passage', status='untried',
                 failed_attempts=0, last_seen_step=2),
            dict(id='C2', place_id='P1', kind='open_passage', status='untried',
                 failed_attempts=0, last_seen_step=1),
        ]
        context = state.context(excluded_route_origins={'C1'})
        self.assertEqual([item['id'] for item in context['unfinished_elsewhere']], ['C2'])
        self.assertEqual(len(state.candidates), 2)

    def test_missing_review_number_binds_to_request_but_explicit_old_number_is_preserved(self):
        payload = dict(action='finish', reason='看见沙发', evidence='当前沙发轮廓明确',
                       scene={'target_seen': 'yes'},
                       target_review={'status': 'arrived', 'evidence': '当前图可见沙发'})
        decision = Decision.model_validate(payload, context={'observation_step': 7})
        self.assertEqual(decision.target_review.observation_step, 7)
        payload['target_review']['observation_step'] = 2
        old = Decision.model_validate(payload, context={'observation_step': 7})
        self.assertEqual(old.target_review.observation_step, 2)
        state = execution_state()
        state.observe(view(7))
        with self.assertRaisesRegex(ValueError, '当前'):
            state.accept(old)

    def test_misnested_review_is_validated_without_changing_its_content(self):
        intent = dict(reference_view_id='O1', question='入口是否仍可走', new_evidence='准备接近')
        decision = Decision.model_validate(dict(action='look', direction='left', reason='复核',
                                                scene=dict(target_seen='no', observation_intent=intent)))
        self.assertEqual(decision.observation_intent.model_dump(), intent)
        with self.assertRaises(ValueError):
            Decision.model_validate(dict(action='look', direction='left', reason='复核',
                                         scene=dict(observation_intent={'question': '缺少证据'})))
        with self.assertRaises(ValueError):
            Decision.model_validate(dict(action='look', direction='left', reason='复核',
                                         scene=dict(observation_intent=intent), observation_intent=intent))

    def test_unused_observation_intent_does_not_change_finish_but_look_remains_strict(self):
        raw = dict(action='finish', reason='当前确认', evidence='当前图有目标类别特征',
                   scene={'target_seen': 'yes'}, observation_intent='确认目标')
        self.assertIsNone(Decision.model_validate(raw).observation_intent)
        raw.update(action='look', direction='left')
        with self.assertRaises(ValueError):
            Decision.model_validate(raw)

    def test_visible_route_does_not_require_a_scan(self):
        state = execution_state()
        state.observe(view(1))
        before = state.snapshot()
        state.resolve(navigation())
        self.assertEqual(state.snapshot(), before)
        self.assertFalse(state.action_options()['scan_required'])
        self.assertEqual(state.accept(navigation()), 'C1')

    def test_moving_keeps_goal_and_does_not_reopen_scan(self):
        state = execution_state()
        state.observe(view(1))
        candidate = state.accept(navigation())
        state.navigation_started(candidate)
        expected = copy.deepcopy(state.subgoal)
        state.last_action = dict(step=1, action='navigate', candidate_id=candidate,
                                 result={'reason': 'timeout'})
        state.observe(view(2, x=.5))
        self.assertFalse(state.scan.required())
        self.assertEqual(state.subgoal, expected)
        self.assertEqual(state.scan.views[0]['id'], 'O1')
        state.resolve(navigation(candidate, previous_action_review={
            'outcome': 'progress', 'view_change': 'same_area', 'evidence': '入口变大'}))

    def test_nested_action_arguments_are_validated_and_conflicts_rejected(self):
        raw = navigation().model_dump()
        raw['navigate'] = {key: raw.pop(key) for key in ('instruction', 'candidate_id')}
        self.assertEqual(Decision.model_validate(raw).candidate_id, 'new_1')
        raw['instruction'] = 'conflicting instruction'
        with self.assertRaises(ValueError):
            Decision.model_validate(raw)

    def test_sole_candidate_selection_does_not_guess_between_alternatives(self):
        raw = navigation().model_dump()
        raw.pop('candidate_id')
        self.assertEqual(Decision.model_validate(raw).candidate_id, 'new_1')
        raw['candidates'].append({**raw['candidates'][0], 'id': 'new_2'})
        self.assertIsNone(Decision.model_validate(raw).candidate_id)

    def test_canonical_candidate_id_still_requires_current_evidence(self):
        state = execution_state()
        state.observe(view(1))
        proposal = navigation()
        proposal.candidate_id = 'C1'
        self.assertEqual(state.resolve(proposal)[1], 'C1')
        proposal.candidates[0].evidence = '旧图 O9 有入口'
        with self.assertRaisesRegex(ValueError, '历史图'):
            state.resolve(proposal)

    def test_first_target_review_requires_current_explicit_sighting(self):
        state = execution_state()
        state.observe(view(1))
        proposal = Decision(action='finish', reason='已找到', evidence='当前沙发靠背清晰可见',
                            scene={'target_seen': 'yes'},
                            target_review={'observation_step': 1, 'status': 'arrived', 'evidence': '当前图确认'})
        state.resolve(proposal)
        proposal.target_review.observation_step = 2
        with self.assertRaisesRegex(ValueError, '当前帧'):
            state.resolve(proposal)

    def test_new_viewpoints_continue_explicit_inspection_without_merging_routes(self):
        state = execution_state()
        state.observe(view(1))
        first = state.accept(navigation())
        state.navigation_started(first)
        question = state.subgoal['question']
        for step in (2, 3):
            state.observe(view(step, x=float(step)))
            decision = navigation(subgoal_review={
                'status': 'continue', 'evidence': '换一个位置核实原来的疑似物体'})
            decision.candidates[0].kind = 'viewpoint'
            selected = state.accept(decision)
            self.assertNotEqual(selected, first)
            self.assertNotIn('origin_candidate_id', state.candidates[-1])
            state.navigation_started(selected, continue_inspection=True)
        self.assertEqual(state.subgoal['question'], question)
        self.assertEqual(state.subgoal['started_step'], 1)
        self.assertEqual(state.subgoal['inspection_candidate_origins'], ['C1', 'C2', 'C3'])
        state.observe(view(4, x=3.))
        state.accept(Decision(action='look', direction='left', reason='去看未查区域',
                              subgoal_review={'status': 'completed', 'evidence': '已看清是靠垫，不是球'}))
        self.assertIsNone(state.subgoal)
        self.assertEqual(state.subgoal_history[-1]['inspection_candidate_origins'], ['C1', 'C2', 'C3'])

    def test_local_waypoint_does_not_replace_the_persistent_exploration_question(self):
        state = execution_state()
        state.observe(view(1))
        candidate = state.accept(navigation())
        state.navigation_started(candidate)
        original_question = state.subgoal['question']
        state.observe(view(2, x=1.))
        onward = navigation(candidate)
        onward.candidates[0].description = '门内左侧当前可见的空地'
        onward.candidates[0].expected_new_view = '从局部空地看清通道下一段'
        candidate = state.accept(onward)
        state.navigation_started(candidate)
        self.assertEqual(state.subgoal['question'], original_question)
        self.assertEqual(state.subgoal['local_destination'], '门内左侧当前可见的空地')
        self.assertEqual(state.subgoal['started_step'], 1)

    def test_explicit_candidate_keeps_original_exploration_question(self):
        state = execution_state()
        state.observe(view(1))
        candidate = state.accept(navigation())
        state.navigation_started(candidate)
        state.observe(view(2))
        selected = state.accept(navigation(candidate, subgoal_review={'status': 'continue', 'evidence': '还在接近入口'}))
        self.assertEqual(selected, candidate)
        self.assertEqual(len(state.candidates), 1)
        state.observe(view(3))
        self.assertEqual(state.accept(navigation(candidate)), candidate)

    def test_two_no_information_reviews_require_a_different_viewpoint(self):
        state = execution_state()
        state.observe(view(1))
        candidate = state.accept(navigation())
        for step in (1, 2):
            state.navigation_started(candidate)
            state.last_action = dict(step=step, action='navigate', candidate_id=candidate,
                                     result={'reason': 'model_stop'})
            state.observe(view(step + 1))
            proposal = navigation(candidate, previous_action_review={
                'outcome': 'no_new_information', 'view_change': 'same_area', 'evidence': '入口与细节均未变化'})
            if step == 1:
                state.accept(proposal)
            else:
                before = state.snapshot()
                with self.assertRaisesRegex(ValueError, '连续两次'):
                    state.accept(proposal)
                self.assertEqual(state.snapshot(), before)
        state.observe(view(4, x=1.))
        with self.assertRaisesRegex(ValueError, '连续两次'):
            state.resolve(proposal)
        # 新视觉进展可以解除限制，位移本身不可以。
        proposal.previous_action_review.outcome = 'progress'
        state.resolve(proposal)

    def test_repeat_without_question_rejected_but_new_direction_allowed(self):
        state = execution_state()
        state.observe(view(1))
        state.observe(view(2, heading=70))
        with self.assertRaisesRegex(ValueError, '复核'):
            state.resolve(Decision(action='look', direction='right', reason='还没找到'))
        state.resolve(Decision(action='look', direction='left', reason='检查左侧新入口'))

    def test_detector_pauses_leave_route_available_but_real_failures_still_count(self):
        state = execution_state()
        state.observe(view(1))
        candidate = state.accept(navigation())
        for step in (1, 2, 3):
            state.navigation_started(candidate)
            result = {'reason': 'agent_interrupt', 'detail': 'target_detected',
                      'motion': {'displacement_m': 0.}}
            state.record_result(candidate, result)
            state.last_action = dict(step=step, action='navigate', candidate_id=candidate, result=result)
            state.last_navigation['result'] = result
            state.observe(view(step + 1))
            state.accept(navigation(candidate, previous_action_review={
                'outcome': 'no_new_information', 'evidence': '误报停车，尚未到门口'}))
        route = state.candidates[0]
        self.assertEqual(route['attempt_count'], 0)
        self.assertEqual(route.get('total_attempt_count', 0), 0)
        self.assertEqual(route.get('no_new_information_count', 0), 0)
        self.assertEqual(route['failed_attempts'], 0)
        self.assertTrue(route['last_attempt']['interrupted'])
        self.assertIsNotNone(state.subgoal)

        state.record_result(candidate, {'reason': 'blocked', 'motion': {
            'displacement_m': 0., 'blocked_samples': 1}})
        self.assertEqual(route['attempt_count'], 1)
        self.assertEqual(route['failed_attempts'], 1)
        self.assertTrue(state.recovery_required)

    def test_other_interrupts_keep_their_exploration_accounting(self):
        state = execution_state()
        state.observe(view(1))
        candidate = state.accept(navigation())
        state.navigation_started(candidate)
        state.record_result(candidate, {'reason': 'agent_interrupt',
                                        'detail': 'sustained_heading_deviation'})
        self.assertEqual(state.candidates[0]['attempt_count'], 1)

    def test_interrupt_is_not_failed_or_stuck(self):
        from navigation_agent.agent.skills.object_search.rules import classify_result
        state = execution_state()
        self.assertEqual(classify_result({'reason': 'agent_interrupt', 'motion': {'displacement_m': 0}},
                                        state.policy), (False, False))


class TopologyTests(unittest.TestCase):
    def test_only_executed_transition_creates_connection_and_labels_do_not_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            spatial = spatial_memory(Path(directory) / 'memory.sqlite3', '找到沙发')
            topology = TopologyMemory(spatial)
            state = execution_state()
            state.observe(view(1))
            candidate = state.accept(navigation())
            state.navigation_started(candidate)
            state.last_action = dict(step=1, action='navigate', candidate_id=candidate,
                                     result={'reason': 'model_stop', 'motion': {'displacement_m': 1.}})
            state.observe(view(2, x=1.))
            decision = navigation(candidate, previous_action_review={
                'outcome': 'progress', 'view_change': 'new_area', 'evidence': '越过门框进入另一房间'},
                spatial_update={'region_id': 'new', 'transition': 'entered', 'label': '办公室',
                                'evidence': '连续移动后的当前图显示门框在身后'})
            topology.update(view(2, x=1.), state, decision)
            result = topology.query()
            self.assertEqual(result['current_region'], 'R2')
            self.assertEqual(len(result['connections']), 1)
            self.assertEqual(TopologyMemory(spatial).query()['current_region'], 'R2')
            self.assertEqual(result['connections'][0]['start_step'], 1)
            state.last_action['action'] = 'look'
            topology.update(view(3, x=1.), state, decision)
            self.assertEqual(topology.query()['current_region'], 'R2')
            self.assertEqual(len(topology.query()['connections']), 1)

    def test_visual_transition_is_sufficient_but_uncertainty_and_motion_are_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            topology = TopologyMemory(spatial_memory(Path(directory) / 'memory.sqlite3', '找到沙发'))
            state = execution_state()
            state.observe(view(1))
            candidate = state.accept(navigation())
            state.navigation_started(candidate)
            state.last_action = dict(step=1, action='navigate', candidate_id=candidate,
                                     result={'reason': 'model_stop', 'motion': {'displacement_m': 1.}})
            state.observe(view(2, x=1.))
            decision = navigation(previous_action_review={
                'outcome': 'progress', 'view_change': 'new_area', 'evidence': '实际穿过门框进入另一房间'})
            uncertain = decision.model_copy(deep=True)
            from navigation_agent.agent.schemas import SpatialUpdate
            uncertain.spatial_update = SpatialUpdate(region_id='new', transition='uncertain', evidence='门框方向仍待确认')
            topology.update(view(2, x=1.), state, uncertain)
            self.assertEqual(topology.query()['connections'], [])
            topology.update(view(2, x=1.), state, decision)
            self.assertEqual(topology.query()['current_region'], 'R2')
            self.assertEqual(len(topology.query()['connections']), 1)
            state.last_action['result']['motion']['displacement_m'] = 0
            topology.update(view(3, x=2.), state, decision)
            self.assertEqual(len(topology.query()['connections']), 1)
            self.assertEqual(TopologyMemory(topology.spatial).current_region, 'R2')

    def test_pose_change_without_executed_motion_is_not_a_traversed_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            topology = TopologyMemory(spatial_memory(Path(directory) / 'memory.sqlite3', '找到沙发'))
            state = execution_state()
            state.observe(view(1))
            candidate = state.accept(navigation())
            state.navigation_started(candidate)
            state.last_action = dict(step=1, action='navigate', candidate_id=candidate,
                                     result={'reason': 'timeout', 'motion': {'displacement_m': 0.}})
            state.observe(view(2, x=2.))
            decision = navigation(candidate, previous_action_review={
                'outcome': 'progress', 'view_change': 'new_area', 'evidence': '当前图像像另一个房间'},
                spatial_update={'region_id': 'new', 'transition': 'entered', 'label': '房间',
                                'evidence': '位置和画面都改变'})
            topology.update(view(2, x=2.), state, decision)
            self.assertEqual(topology.query()['connections'], [])
            self.assertEqual(topology.query()['current_region'], 'R1')

    def test_unknown_region_and_invalid_pose_do_not_invent_links(self):
        with tempfile.TemporaryDirectory() as directory:
            topology = TopologyMemory(spatial_memory(Path(directory) / 'memory.sqlite3', '找到沙发'))
            state = execution_state()
            current = view(1)
            current['observer_position'] = None
            state.observe(current)
            decision = navigation(spatial_update={
                'region_id': 'R99', 'label': '相同名字', 'transition': 'entered', 'evidence': '推测'})
            topology.update(current, state, decision)
            self.assertEqual(topology.query()['current_region'], 'R1')
            self.assertEqual(topology.query()['connections'], [])
            self.assertEqual(topology.query()['regions'][0]['label'], '')


class TopologyRecallTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.spatial = spatial_memory(Path(self.temp.name) / 'memory.sqlite3', '找到沙发')
        self.topology = TopologyMemory(self.spatial)
        self.state = SimpleNamespace(last_navigation=None, last_action=None,
                                     candidates=[], subgoal_history=[])

    def observe(self, step, x=0., *, accepted=True):
        observation = {'pose': {'x': x, 'y': 0., 'yaw': 0.}, 'pose_valid': True}
        current = self.spatial.observe(observation, f'images/{step:03d}.jpg', step)
        report = Decision(action='look', direction='left', reason='检查入口',
                          scene={'target_seen': 'no', 'landmarks': ['白色门框', '黄色椅子'],
                                 'visible_ground': '黄色椅子旁的连续地面'})
        self.spatial.describe(current['id'], step, report, action_accepted=accepted)
        return current

    def update(self, current, *, region=None, crossed=False):
        payload = {'action': 'look', 'direction': 'left', 'reason': '检查当前房间'}
        if region is not None:
            payload['spatial_update'] = {
                'region_id': region, 'transition': 'entered' if crossed else 'same',
                'label': '有黄色椅子的房间', 'evidence': '门框和椅子的相对位置与历史一致',
            }
        if crossed:
            payload['previous_action_review'] = {
                'outcome': 'progress', 'view_change': 'new_area', 'evidence': '实际穿过可见门框',
            }
        self.topology.update(current, self.state, Decision(**payload))

    def motion_from(self, start, distance):
        self.state.last_navigation = {'step': start['step'], 'observer_position': start['observer_position']}
        self.state.last_action = {
            'step': start['step'], 'action': 'navigate',
            'result': {'reason': 'model_stop', 'motion': {'displacement_m': distance}},
        }

    @staticmethod
    def candidate(identifier, step=1):
        return {
            'id': identifier, 'origin_candidate_id': identifier, 'kind': 'open_passage',
            'description': '白色门框入口', 'evidence': '当前可见连续地面',
            'expected_new_view': '检查白色门框后的通道', 'last_seen_step': step,
            'status': 'untried', 'total_attempt_count': 0, 'no_new_information_count': 0,
        }

    def test_confirmed_same_room_ends_navigation_review_before_a_stationary_turn(self):
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 2.)
        current = self.observe(2, x=2.)
        self.assertTrue(self.topology.region_comparison_needed(current, self.state))
        self.update(current, region='R1')
        self.state.last_navigation['result'] = self.state.last_action['result']
        self.state.last_action = {'step': 2, 'action': 'look', 'result': {'reason': 'turned'}}
        turned = self.observe(3, x=2.)
        self.assertFalse(self.topology.region_comparison_needed(turned, self.state))
        # 即使后续转头误报新房间，旧导航也不能再支持一条虚假的跨区连接。
        self.update(turned, region='new', crossed=True)
        self.assertEqual(self.topology.current_region, 'R1')
        self.assertEqual(self.topology.query()['connections'], [])
        self.motion_from(turned, 1.)
        next_stop = self.observe(4, x=3.)
        self.assertTrue(self.topology.region_comparison_needed(next_stop, self.state))
        self.update(next_stop, region='new', crossed=True)
        self.assertEqual(self.topology.current_region, 'R2')
        self.assertEqual(len(self.topology.query()['connections']), 1)

    def test_uncertain_room_review_stays_pending_until_a_clear_view(self):
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 2.)
        unclear = self.observe(2, x=2.)
        decision = Decision(action='look', direction='left', reason='转向看清房间')
        self.topology.update(unclear, self.state, decision, region_match={
            'region_id': 'uncertain', 'label': '墙角', 'evidence': '缺少完整布局'})
        self.state.last_navigation['result'] = self.state.last_action['result']
        self.state.last_action = {'step': 2, 'action': 'look', 'result': {'reason': 'turned'}}
        clear = self.observe(3, x=2.)
        self.assertTrue(self.topology.region_comparison_needed(clear, self.state))
        self.topology.update(clear, self.state, decision, region_match={
            'region_id': 'new', 'label': '床和窗户', 'evidence': '转向后明确看到另一房间布局'})
        self.assertEqual(self.topology.current_region, 'R2')
        self.assertFalse(self.topology.region_comparison_needed(self.observe(4, x=2.), self.state))

    def test_actual_room_crossing_records_the_passage_destination_and_removes_frontier(self):
        candidate = self.candidate('C1')
        self.state.candidates = [candidate]
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 2.)
        self.state.last_navigation['candidate_id'] = 'C1'
        candidate.update(total_attempt_count=1, last_attempt={'step': 1})
        self.update(self.observe(2, x=2.), region='new', crossed=True)
        context = self.topology.query()
        self.assertEqual(context['routes'], [])
        passage = context['reviewed_routes'][0]
        self.assertEqual((passage['region_id'], passage['destination_region']), ('R1', 'R2'))
        self.assertEqual(passage['exploration_status'], 'traversed')
        self.assertEqual(passage['destination_step'], 2)
        self.assertTrue(all(region['route_counts']['traversed'] == 1 for region in context['regions']))
        self.assertEqual(self.topology.closed_route_origins(), {'C1'})
        self.assertNotEqual(context['regions'][0]['search']['status'], 'checked')

        # 同一入口的新编号和新尝试不抹掉已知去向，也不把入口移到终点房间内部。
        alias = dict(self.candidate('C2', step=3), origin_candidate_id='C1')
        self.state.candidates.append(alias)
        self.update(self.observe(3, x=2.), region='R2')
        alias['last_attempt'] = {'step': 4}
        alias['total_attempt_count'] = 1
        self.update(self.observe(5, x=2.), region='R2')
        passage = self.topology.query()['reviewed_routes'][0]
        self.assertEqual((passage['region_id'], passage['destination_region']), ('R1', 'R2'))
        self.assertEqual(self.topology.query()['routes'], [])
        self.assertEqual(self.topology.closed_route_origins(), {'C1'})

    def test_visual_revisit_recalls_matched_room_routes_before_accepting_a_decision(self):
        from navigation_agent.agent.context.builder import exploration_context

        self.state.candidates = [self.candidate('C1'), self.candidate('C2')]
        first = self.observe(1)
        self.topology.update(first, self.state, Decision(
            action='look', direction='left', reason='明显区域已查',
            spatial_update={'region_id': 'R1', 'transition': 'same', 'label': '窗户与木桌',
                            'evidence': '主要可达地面已看过，门后另行探索',
                            'search_status': 'checked', 'remaining': []},
        ))
        self.motion_from(first, 2.)
        self.state.last_navigation['candidate_id'] = 'C1'
        self.state.candidates[0].update(total_attempt_count=1, last_attempt={'step': 1})
        self.update(self.observe(2, x=2.), region='new', crossed=True)
        self.state.candidates.append(self.candidate('C3', step=3))
        self.update(self.observe(3, x=2.), region='R2')
        current = self.observe(4, x=2.)
        original = self.topology.query(current)
        self.assertEqual(original['next_frontier']['id'], 'C3')
        comparison = {'region_id': 'R1', 'label': '窗户与木桌', 'evidence': '当前回到已查房间'}
        context = exploration_context(
            self.topology, current, {'local_candidates': [], 'unfinished_elsewhere': []},
            comparison=comparison,
        )
        self.assertEqual(context['previous_region'], 'R2')
        self.assertEqual(context['focus_region'], 'R1')
        self.assertEqual(context['regions'][0]['search']['status'], 'checked')
        self.assertEqual(context['regions'][0]['reference_step'], 1)
        self.assertEqual(context['next_frontier']['id'], 'C2')
        self.assertEqual(context['next_frontier']['return_path'], ['R1'])
        self.assertEqual(context['routes'][0]['id'], 'C2')
        self.assertEqual(self.topology.current_region, 'R2')
        self.assertEqual(self.topology.query(current), original)
        for unsupported in ('new', 'uncertain', 'R99'):
            self.assertEqual(self.topology.query(current, focus_region=unsupported)['focus_region'], 'R2')

    def test_room_comparison_keeps_anchor_and_recalls_a_confirmed_matching_direction(self):
        first = self.observe(1)
        self.update(first, region='R1')
        for step, yaw, accepted in ((2, 1., True), (3, 3., True), (4, 3.1, False)):
            current = self.spatial.observe(
                {'pose': {'x': 0., 'y': 0., 'yaw': yaw}, 'pose_valid': True},
                f'images/{step:03d}.jpg', step)
            self.spatial.describe(current['id'], step,
                                  Decision(action='look', direction='left', reason='观察',
                                           scene={'target_seen': 'no', 'landmarks': ['另一侧电视柜']}),
                                  action_accepted=accepted)
            self.update(current, region='R1')
        current = self.spatial.observe({'pose': {'x': 0., 'y': 0., 'yaw': 3.1}, 'pose_valid': True},
                                       'images/005.jpg', 5)
        frames = self.topology.comparison_references(current, self.topology.query()['regions'])
        self.assertEqual([frame['step'] for frame in frames], [1, 3])
        self.assertTrue(all(frame['region_id'] == 'R1' for frame in frames))
        self.assertEqual(self.topology.query()['regions'][0]['reference_step'], 1)

    def test_stopped_comparison_owns_identity_but_never_copies_mismatched_search_status(self):
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 2.)
        current = self.observe(2, x=2.)
        proposal = Decision(action='look', direction='left', reason='查看',
                            spatial_update={'region_id': 'R1', 'transition': 'same', 'label': '旧房间',
                                            'evidence': '沿用了旧房间文字',
                                            'search_status': 'checked', 'remaining': []})
        self.topology.update(current, self.state, proposal, region_match={
            'region_id': 'new', 'label': '床铺与内侧门框', 'evidence': '停车原图显示不同房间布局'})
        context = self.topology.query()
        self.assertEqual(context['current_region'], 'R2')
        self.assertEqual(len(context['connections']), 1)
        self.assertEqual(context['regions'][0]['search']['status'], 'visited')
        self.assertEqual(proposal.spatial_update.region_id, 'R1')
        with self.spatial.connect() as db:
            import json
            report = json.loads(db.execute('SELECT report_json FROM region_evidence WHERE step=2').fetchone()[0])
        self.assertEqual(report['source'], 'stopped_image_comparison')
        self.motion_from(current, 2.)
        self.topology.update(self.observe(3, x=4.), self.state, proposal, region_match={
            'region_id': 'uncertain', 'label': '墙角', 'evidence': '缺少可辨认的家具组合'})
        self.assertEqual(self.topology.query()['current_region'], 'R2')
        self.assertEqual(len(self.topology.query()['connections']), 1)

    def test_untried_frontier_survives_room_return_and_ignores_already_traversed_entry(self):
        self.state.candidates = [self.candidate('C1'), self.candidate('C2')]
        first = self.observe(1)
        self.update(first, region='R1')
        self.assertEqual(self.topology.query()['next_frontier']['id'], 'C1')
        self.motion_from(first, 2.)
        self.state.last_navigation['candidate_id'] = 'C2'
        self.state.candidates[1].update(total_attempt_count=1, last_attempt={'step': 1})
        second = self.observe(2, x=2.)
        self.update(second, region='new', crossed=True)
        context = self.topology.query()
        self.assertEqual(context['next_frontier']['id'], 'C1')
        self.assertEqual(context['next_frontier']['return_path'], ['R2', 'R1'])
        self.assertEqual(context['recent_region_path'], ['R1', 'R2'])
        self.motion_from(second, 2.)
        self.state.last_navigation['candidate_id'] = 'C2'
        self.update(self.observe(3), region='R1', crossed=True)
        context = self.topology.query()
        self.assertEqual(context['recent_region_path'], ['R1', 'R2', 'R1'])
        self.assertEqual(context['next_frontier']['return_path'], ['R1'])
        self.state.candidates[0].update(total_attempt_count=1, last_attempt={'step': 4})
        self.update(self.observe(4), region='R1')
        self.assertIsNone(self.topology.query()['next_frontier'])

    def test_visual_entry_claim_without_motion_does_not_mark_passage_traversed(self):
        self.state.candidates = [self.candidate('C1')]
        first = self.observe(1)
        self.update(first)
        self.motion_from(first, 0.)
        self.state.last_navigation['candidate_id'] = 'C1'
        self.update(self.observe(2), region='new', crossed=True)
        self.assertNotIn('destination_region', self.topology.query()['routes'][0])
        self.assertEqual(self.topology.closed_route_origins(), set())

    def test_completed_inspection_closes_its_viewpoints_without_merging_entrances(self):
        self.state.candidates = [self.candidate(f'C{i}') for i in range(1, 5)]
        self.update(self.observe(1))
        self.state.subgoal_history = [{
            'origin_candidate_id': 'C1', 'inspection_candidate_origins': ['C1', 'C2', 'C3'],
            'status': 'completed', 'ended_step': 2, 'evidence': '换位后确认疑似物是靠垫',
        }]
        self.update(self.observe(2))
        self.state.subgoal_history.clear()
        self.update(self.observe(3))
        result = self.topology.query()
        self.assertEqual({route['id'] for route in result['reviewed_routes']}, {'C1', 'C2', 'C3'})
        self.assertEqual([route['id'] for route in result['routes']], ['C4'])
        self.assertEqual(self.topology.closed_route_origins(), {'C1', 'C2', 'C3'})
        self.assertEqual(result['connections'], [])

    def test_resolved_question_survives_short_history_eviction_and_reopen(self):
        self.state.candidates = [self.candidate('C1')]
        self.update(self.observe(1))
        self.state.subgoal_history = [{
            'origin_candidate_id': 'C1', 'status': 'completed',
            'ended_step': 2, 'evidence': '已看清门后通道，没有更多可通行入口',
        }]
        self.update(self.observe(2))
        self.state.subgoal_history.clear()
        self.topology = TopologyMemory(self.spatial)
        self.state.candidates.append(self.candidate('C2', step=3))
        self.update(self.observe(3))
        result = self.topology.query()
        checked = result['reviewed_routes'][0]
        self.assertEqual(checked['id'], 'C1')
        self.assertEqual(checked['resolved_step'], 2)
        self.assertIn('已看清门后通道', checked['resolution'])
        self.assertEqual([route['id'] for route in result['routes']], ['C2'])
        self.assertEqual(result['regions'][0]['route_counts'], {'resolved': 1, 'pending': 1, 'deferred': 0, 'traversed': 0, 'blocked': 0})
        self.assertIn('不证明整间房已搜完', result['limits'])

    def test_abandoned_route_is_not_pending_and_an_actual_new_attempt_reopens_it(self):
        self.state.candidates = [self.candidate('C1')]
        self.update(self.observe(1))
        self.state.subgoal_history = [{
            'origin_candidate_id': 'C1', 'status': 'abandoned',
            'ended_step': 2, 'evidence': '靠近后只看到墙面，暂缓这个入口假设',
        }]
        self.update(self.observe(2))
        self.state.subgoal_history.clear()
        self.topology = TopologyMemory(self.spatial)
        self.update(self.observe(3))
        memory = self.topology.query()
        self.assertEqual(memory['routes'], [])
        self.assertEqual(memory['reviewed_routes'][0]['exploration_status'], 'deferred')
        self.assertEqual(memory['regions'][0]['route_counts']['pending'], 0)
        self.state.candidates[0]['last_attempt'] = {'step': 4}
        self.state.candidates[0]['total_attempt_count'] = 1
        self.update(self.observe(5))
        memory = self.topology.query()
        self.assertEqual(memory['reviewed_routes'], [])
        self.assertEqual(memory['routes'][0]['exploration_status'], 'approached')

    def test_return_hint_requires_visual_confirmation_and_visit_counts_survive_reopen(self):
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 3.)
        second = self.observe(2, x=3.)
        self.update(second, region='new', crossed=True)
        third = self.observe(3)
        recalled = self.topology.query(third)
        self.assertEqual(recalled['current_region'], 'R2')
        self.assertEqual(recalled['revisit_candidates'][0]['region_id'], 'R1')
        self.assertEqual(recalled['revisit_candidates'][0]['observation_step'], 1)
        self.assertTrue(recalled['revisit_candidates'][0]['similar_heading'])
        self.assertIn('黄色椅子', recalled['revisit_candidates'][0]['description'])
        self.assertEqual(len(recalled['connections']), 1)
        self.assertEqual(self.topology.current_region, 'R2')
        self.motion_from(second, 3.)
        self.update(third, region='R1', crossed=True)
        restored = TopologyMemory(self.spatial).query()
        self.assertEqual(restored['current_region'], 'R1')
        regions = {region['id']: region for region in restored['regions']}
        self.assertEqual(regions['R1']['visits'], 2)
        self.assertEqual(regions['R2']['visits'], 1)
        self.assertEqual(regions['R1']['last_observed_step'], 3)
        self.assertEqual(len(restored['connections']), 2)

    def test_close_positions_and_identical_labels_never_merge_rooms(self):
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 3.)
        second = self.observe(2, x=3.)
        self.update(second, region='new', crossed=True)
        self.motion_from(second, 2.9)
        third = self.observe(3, x=.1)
        self.update(third, region='new', crossed=True)
        fourth = self.observe(4, x=.1)
        result = self.topology.query(fourth)
        self.assertEqual(result['current_region'], 'R3')
        self.assertEqual({item['region_id'] for item in result['revisit_candidates']}, {'R1', 'R3'})
        with self.spatial.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM regions').fetchone()[0], 3)
        self.assertEqual(len(result['connections']), 2)

    def test_inherited_region_is_recalled_without_inventing_visual_report(self):
        self.update(self.observe(1))
        current = {**self.observe(2), 'id': 'another_place_cluster'}
        recalled = self.topology.query(current)
        self.assertEqual(recalled['revisit_candidates'][0]['region_id'], 'R1')
        with self.spatial.connect() as db:
            self.assertEqual(db.execute('SELECT report_json FROM region_evidence WHERE step=1').fetchone()[0], '{}')

    def test_invalid_pose_and_rejected_visual_report_are_not_revisit_evidence(self):
        self.update(self.observe(1, accepted=False))
        current = self.observe(2)
        self.assertEqual(self.topology.query(current)['revisit_candidates'], [])
        self.update(current)
        missing_pose = {**self.observe(3), 'observer_position': None}
        self.assertEqual(self.topology.query(missing_pose)['revisit_candidates'], [])

    def test_later_scene_does_not_replace_the_rooms_identity_landmarks(self):
        self.update(self.observe(1), region='R1')
        self.topology.update(self.observe(2), self.state, Decision(
            action='look', direction='left', reason='核对当前区域',
            spatial_update={'region_id': 'R1', 'transition': 'same',
                            'label': '灰色柜体和墙面', 'evidence': '当前只看见柜体和墙面'},
        ))
        region = self.topology.query()['regions'][0]
        self.assertEqual(region['label'], '有黄色椅子的房间')
        self.assertEqual(region['source_step'], 1)
        with self.spatial.connect() as db:
            latest = db.execute('SELECT report_json FROM region_evidence WHERE step=2').fetchone()[0]
        self.assertIn('灰色柜体和墙面', latest)

    def test_room_reference_keeps_first_supported_accepted_view(self):
        self.update(self.observe(1, accepted=False), region='R1')
        self.update(self.observe(2), region='R1')
        self.update(self.observe(3), region='R1')
        result = self.topology.query()
        self.assertEqual(result['regions'][0]['reference_step'], 2)

    def test_region_window_keeps_current_region_with_many_neighbors(self):
        with self.spatial.connect() as db:
            for number in range(2, 13):
                db.execute('INSERT INTO regions VALUES (?, ?, ?, ?)',
                           (f'R{number}', '房间', number, 'model_supported'))
                db.execute('INSERT INTO region_connections VALUES (?, ?, ?, ?, ?, ?)',
                           (number, 'R1', f'R{number}', number, number + 1, '实际穿过入口'))
        result = self.topology.query()
        self.assertEqual(len(result['regions']), 8)
        self.assertEqual(result['regions'][0]['id'], 'R1')

    def test_completed_aliases_leave_pending_context_and_actual_retry_reopens_them(self):
        from navigation_agent.agent.context.builder import compact_execution_memory
        first = self.candidate('C1')
        second = dict(self.candidate('C2', step=2), origin_candidate_id='C1')
        self.state.candidates = [first, second, self.candidate('C3', step=2)]
        self.update(self.observe(1))
        self.update(self.observe(2))
        self.state.subgoal_history = [{
            'origin_candidate_id': 'C1', 'status': 'completed',
            'ended_step': 3, 'evidence': '已看清入口',
        }]
        self.update(self.observe(3))
        execution = {'unfinished_elsewhere': copy.deepcopy(self.state.candidates),
                     'recent_subgoals': copy.deepcopy(self.state.subgoal_history)}
        compact_execution_memory(execution, self.topology.closed_route_origins())
        self.assertEqual([item['id'] for item in execution['unfinished_elsewhere']], ['C3'])
        self.assertEqual(execution['recent_subgoals'], [])
        self.assertEqual(len(self.topology.query()['reviewed_routes']), 1)
        self.assertEqual(len(self.state.candidates), 3)
        self.state.candidates[1]['last_attempt'] = {'step': 4}
        self.update(self.observe(5))
        self.assertNotIn('C1', self.topology.closed_route_origins())
        routes = self.topology.query()['routes']
        self.assertEqual(len([r for r in routes if r['origin_candidate_id'] == 'C1']), 1)

    def test_fresh_sighting_does_not_reopen_closed_origin_after_history_eviction(self):
        self.state.candidates = [self.candidate('C1')]
        self.update(self.observe(1))
        self.state.subgoal_history = [{
            'origin_candidate_id': 'C1', 'status': 'completed',
            'ended_step': 2, 'evidence': '已看清入口后没有更多路线',
        }]
        self.update(self.observe(2))
        self.state.subgoal_history.clear()
        self.state.candidates.append(dict(self.candidate('C2', step=3), origin_candidate_id='C1'))
        self.update(self.observe(3))
        self.assertEqual(self.topology.closed_route_origins(), {'C1'})
        self.assertEqual(self.topology.query()['routes'], [])
        self.state.candidates[1]['last_attempt'] = {'step': 4}
        self.update(self.observe(5))
        self.assertEqual(self.topology.closed_route_origins(), set())
        self.assertEqual(len(self.topology.query()['routes']), 1)

    def test_blocked_viewpoint_does_not_close_an_entrance_globally(self):
        candidate = self.candidate('C1')
        candidate['status'] = 'avoid_from_here'
        self.state.candidates = [candidate]
        self.update(self.observe(1))
        self.assertEqual(self.topology.closed_route_origins(), set())

    def test_context_reuses_candidate_details_without_mutating_execution_or_stored_routes(self):
        from navigation_agent.agent.context.builder import exploration_context
        self.state.candidates = [self.candidate('C1'), self.candidate('C2')]
        self.update(self.observe(1))
        current = self.observe(2)
        execution = {'local_candidates': [self.state.candidates[0]], 'unfinished_elsewhere': []}
        before = copy.deepcopy(execution)
        context = exploration_context(self.topology, current, execution)
        self.assertEqual(context['previous_region'], 'R1')
        self.assertNotIn('current_region', context)
        self.assertEqual(self.topology.current_region, 'R1')
        routes = {route['id']: route for route in context['routes']}
        self.assertNotIn('description', routes['C1'])
        self.assertEqual(routes['C1']['region_id'], 'R1')
        self.assertEqual(routes['C1']['return_path'], ['R1'])
        self.assertEqual(routes['C2']['description'], '白色门框入口')
        self.assertEqual(execution, before)
        stored = {route['id']: route for route in self.topology.query()['routes']}
        self.assertEqual(stored['C1']['description'], '白色门框入口')

    def test_closed_navigation_path_is_a_failed_attempt_but_not_a_collision_recovery(self):
        state = execution_state()
        state.observe(view(1))
        selected = state.accept(navigation())
        state.navigation_started(selected)
        state.record_result(selected, {'reason': 'path_repeated',
                                      'motion': {'displacement_m': .1, 'blocked_samples': 0}})
        self.assertFalse(state.recovery_required)
        self.assertEqual(state.candidates[0]['failed_attempts'], 1)

    def test_passage_instruction_cannot_target_the_solid_door_frame(self):
        state = execution_state()
        state.observe(view(1))
        decision = navigation()
        decision.instruction = 'Walk straight toward the dark red door frame on the right and stop.'
        with self.assertRaisesRegex(ValueError, '门框实体'):
            state.resolve(decision)
        decision.instruction = 'Walk straight toward the open doorway with a red frame on the right and stop.'
        state.resolve(decision)

    def test_new_entrance_does_not_inherit_old_candidate_or_failure_counts(self):
        state = execution_state()
        state.observe(view(1))
        first = state.accept(navigation())
        state.navigation_started(first)
        state.candidates[0]['no_new_information_count'] = 2
        state.observe(view(2, x=1.))
        second = state.accept(navigation(subgoal_review={
            'status': 'continue', 'evidence': '仍要寻找目标，但当前选的是另一扇门'}))
        self.assertNotEqual(first, second)
        state.navigation_started(second)
        self.assertEqual(state.subgoal['origin_candidate_id'], second)
        self.assertEqual(state.subgoal_history[-1]['status'], 'switched')

    def test_room_transition_can_be_confirmed_after_a_stopped_look(self):
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 3.)
        self.state.last_navigation['result'] = self.state.last_action['result']
        self.update(self.observe(2, x=3.))
        self.state.last_action = {'step': 2, 'action': 'look', 'result': {'reason': 'turned'}}
        current = self.observe(3, x=3.)
        self.update(current, region='new', crossed=True)
        result = self.topology.query()
        self.assertEqual(result['current_region'], 'R2')
        self.assertEqual(result['connections'][0]['action_step'], 1)
        self.update(self.observe(4, x=3.), region='new', crossed=True)
        self.assertEqual(self.topology.current_region, 'R2')

    def test_checked_room_keeps_coverage_evidence_through_an_uncertain_view(self):
        current = self.observe(1)
        self.topology.update(current, self.state, Decision(
            action='look', direction='left', reason='转向出口',
            spatial_update={'region_id': 'R1', 'transition': 'same', 'label': '木桌和灰沙发',
                            'search_status': 'checked', 'remaining': ['门后地面'],
                            'evidence': '当前房间可达地面已检查，门后尚未进入'},
        ))
        self.topology.update(self.observe(2), self.state, Decision(
            action='look', direction='left', reason='找出口',
            spatial_update={'region_id': 'R1', 'transition': 'uncertain',
                            'search_status': 'partial', 'evidence': '当前只见墙面，无法判断区域'},
        ))
        search = TopologyMemory(self.spatial).query()['regions'][0]['search']
        self.assertEqual(search['status'], 'checked')
        self.assertEqual(search['step'], 1)
        self.assertEqual(search['remaining'], ['门后地面'])


    def test_nearby_room_outside_place_cluster_is_recalled_without_merging(self):
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 8.)
        self.update(self.observe(2, x=8.), region='new', crossed=True)
        current = self.observe(3, x=1.25)
        context = self.topology.query(current)
        self.assertEqual(context['current_region'], 'R2')
        self.assertEqual(context['revisit_candidates'][0]['region_id'], 'R1')
        self.assertEqual(context['revisit_candidates'][0]['distance_m'], 1.25)
        self.assertEqual(len(context['connections']), 1)

    def test_comparison_can_recall_three_rooms_with_bounded_references(self):
        regions = []
        previous = None
        for step in range(1, 4):
            if previous:
                self.motion_from(previous, 4.)
            current = self.observe(step, x=4. * (step - 1))
            self.update(current, region='new' if previous else 'R1', crossed=previous is not None)
            regions.append({'id': self.topology.current_region, 'reference_step': step})
            previous = current
        current = self.observe(4, x=5.)
        frames = self.topology.comparison_references(current, regions)
        self.assertEqual({frame['region_id'] for frame in frames}, {'R1', 'R2', 'R3'})
        self.assertLessEqual(len(frames), 6)

    def test_continuing_candidate_does_not_overwrite_first_known_connection(self):
        original = self.candidate('C1')
        self.state.candidates = [original]
        first = self.observe(1)
        self.update(first, region='R1')
        self.motion_from(first, 2.)
        self.state.last_navigation['candidate_id'] = 'C1'
        original.update(total_attempt_count=1, last_attempt={'step': 1})
        second = self.observe(2, x=2.)
        self.update(second, region='new', crossed=True)
        alias = dict(self.candidate('C2', step=2), origin_candidate_id='C1')
        self.state.candidates.append(alias)
        self.motion_from(second, 2.)
        self.state.last_navigation['candidate_id'] = 'C2'
        alias.update(total_attempt_count=2, last_attempt={'step': 2})
        self.update(self.observe(3, x=4.), region='new', crossed=True)
        context = self.topology.query()
        route = context['reviewed_routes'][0]
        self.assertEqual((route['region_id'], route['destination_region']), ('R1', 'R2'))
        self.assertEqual(route['source_step'], 1)
        with self.spatial.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM region_connections').fetchone()[0], 2)


    def test_transit_retains_checked_coverage_and_new_inspection_keeps_its_evidence(self):
        for step, status, remaining in ((1, 'checked', []), (2, 'visited', []),
                                        (3, 'surveyed', ['入口旁的小片地面']),
                                        (4, 'partial', ['确认入口旁的疑似物体'])):
            current = self.observe(step)
            self.topology.update(current, self.state, Decision(
                action='look', direction='left', reason='查看当前范围',
                spatial_update={'region_id': 'R1', 'transition': 'same', 'search_status': status,
                                'remaining': remaining, 'evidence': f'第{step}次当前画面判断'}))
            search = self.topology.query()['regions'][0]['search']
            if status == 'visited':
                self.assertEqual(search['status'], 'checked')
                self.assertEqual(search['step'], 1)
                self.assertEqual(search['last_activity'], 'visited')
            if status in ('surveyed', 'partial'):
                self.assertEqual(search['status'], status)
                self.assertEqual(search['remaining'], remaining)
                self.assertEqual(search['previous_check']['step'], 1)
                self.assertEqual(search['previous_check']['evidence'], '第1次当前画面判断')
