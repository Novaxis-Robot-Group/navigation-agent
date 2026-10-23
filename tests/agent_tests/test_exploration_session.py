"""探索 session 生命周期、方向选择和扫描覆盖的离线验证。"""

from navigation_agent.agent.schemas import ActionReview

import unittest
from tests.config_support import execution_state
from navigation_agent.agent.schemas import Decision


def observe(state, step, yaw, x=0):
    state.observe({
        'id': 'P1', 'observer_position': [x, 0], 'position': [x, 0],
        'step': step, 'heading_deg': yaw, 'image': f'images/{step:03d}.jpg',
    })


def scan(state, step, yaw):
    observe(state, step, yaw)
    state.accept(Decision(
        action='look', direction='left', reason='检查下一方向',
        scene={'visible_ground': '开阔地面'},
        candidates=[{'id': 'new_1', 'kind': 'viewpoint',
                     'description': f'方向 {yaw} 的空地', 'evidence': '地面可见'}],
    ))


class SessionTests(unittest.TestCase):
    def test_fixed_scan_uses_actual_coverage_instead_of_turn_counts(self):
        state = execution_state(look_angle_deg=75)
        for step, yaw in enumerate((0, 52, 104, 156, 208), 1):
            scan(state, step, yaw)
        observe(state, 6, 263)
        self.assertFalse(state.scan_state()['coverage_complete'])
        self.assertEqual(state.scan_state()['largest_heading_gap_deg'], 97)
        self.assertNotIn('look_budget_exhausted', state.scan_state())
        self.assertIn('left', state.allowed_look_directions())
        state.scan.look_count = 100
        with self.assertRaisesRegex(ValueError, '复核'):
            state.resolve(Decision(action='look', direction='left', reason='看清尚未覆盖的方向'))

    def test_completed_scan_stops_look_but_allows_orient_for_confirmation(self):
        state = execution_state(look_angle_deg=75)
        for step, yaw in enumerate((0, 60, 120, 180, 240, 300), 1):
            observe(state, step, yaw)
        self.assertTrue(state.scan_state()['coverage_complete'])
        self.assertEqual(state.scan.unobserved_look_directions(), [])
        self.assertEqual(state.allowed_look_directions(), ['left', 'right'])
        with self.assertRaisesRegex(ValueError, '复核'):
            state.resolve(Decision(action='look', direction='left', reason='再转一圈'))
        state.resolve(Decision(action='orient', view_id='O5', reason='复核右边缘目标局部',
                              observation_intent={'reference_view_id': 'O5', 'question': '圆柱物顶部是否是桶口',
                                    'new_evidence': '当前图边缘露出新的银色圆柱顶部'}))

    def test_observation_dead_end_can_retreat_only_after_a_real_navigation(self):
        state = execution_state(look_angle_deg=60)
        for step, yaw in enumerate((0, 60, 120, 180, 240, 300), 2):
            observe(state, step, yaw, x=3)
        self.assertFalse(state.action_options()['recover_allowed'])
        state.last_navigation = {'step': 1, 'observer_position': [0, 0],
                                 'result': {'reason': 'model_stop'}}
        self.assertTrue(state.action_options()['recover_allowed'])
        state.accept(Decision(action='recover', reason='周围已看且没有路线，退出观察死角'))
        self.assertEqual(state.recovery_attempts, 1)
        self.assertEqual(state.recovery_anchor, [3, 0])
        state.record_result(None, {'reason': 'recovery_unavailable'})
        self.assertFalse(state.action_options()['recover_allowed'])

    def test_unseen_directions_or_visible_routes_do_not_offer_dead_end_recovery(self):
        state = execution_state(look_angle_deg=60)
        state.last_navigation = {'step': 1, 'observer_position': [0, 0],
                                 'result': {'reason': 'model_stop'}}
        for step, yaw in enumerate((0, 60, 120), 2):
            observe(state, step, yaw, x=3)
        self.assertFalse(state.action_options()['recover_allowed'])
        for step, yaw in enumerate((180, 240, 300), 5):
            observe(state, step, yaw, x=3)
        state.candidates = [{'id': 'C1', 'status': 'untried'}]
        state.scan.views[-1]['candidate_ids'] = ['C1']
        self.assertFalse(state.action_options()['recover_allowed'])
        state.candidates = []
        state.last_navigation['result']['startup_failed'] = True
        self.assertFalse(state.action_options()['recover_allowed'])

    def test_final_gap_can_be_scanned_even_near_initial_heading(self):
        state = execution_state(look_angle_deg=75)
        for step, yaw in enumerate((0, 68, 136, 204, 272), 1):
            observe(state, step, yaw)
        self.assertFalse(state.scan_state()['coverage_complete'])
        self.assertIn('left', state.allowed_look_directions())
        with self.assertRaisesRegex(ValueError, '复核'):
            state.resolve(Decision(action='look', direction='left', reason='补齐最后一段环视'))
        observe(state, 6, 342)
        self.assertTrue(state.scan.coverage_complete())
        self.assertEqual(state.allowed_look_directions(), ['left', 'right'])


    def test_turn_back_survives_scan_limit_and_requires_a_fresh_view(self):
        state = execution_state(look_angle_deg=75)
        scan(state, 1, 0)
        observe(state, 2, 150)
        state.scan.look_count = 5
        with self.assertRaisesRegex(ValueError, '未复核'):
            state.resolve(Decision(action='give_up', reason='当前图没有路'))
        state.accept(Decision(action='orient', view_id='O1', reason='回到过道确认',
                              observation_intent={'reference_view_id': 'O1', 'question': '原方向空地的当前接近路径是否通畅',
                                    'new_evidence': '比较当前方向后选择 O1 空地作为下一处观察位置，准备转回接近'}))
        self.assertEqual(state.scan.look_count, 5)
        self.assertEqual(state.scan.phase, 'aligning')
        observe(state, 3, 5)
        self.assertEqual(state.scan.phase, 'confirming')
        self.assertEqual(state.scan.current_view_id, 'O3')
        with self.assertRaises(ValueError):
            state.resolve(Decision(action='orient', view_id='O1', reason='重复转回'))

    def test_error_without_translation_keeps_session_and_original_images(self):
        state = execution_state()
        scan(state, 1, 0)
        scan(state, 2, 70)
        state.scan.record_result('navigate', {'reason': 'navigation_error'}, candidate_id='C1')
        observe(state, 3, 70)
        self.assertEqual(state.scan.number, 1)
        self.assertEqual(state.scan.views[0]['image'], 'images/001.jpg')
        self.assertEqual(state.scan.actions[-1]['result']['reason'], 'navigation_error')
        self.assertIsNone(state.scan.last_outcome)

    def test_navigation_translation_starts_scan_before_decision_without_losing_records(self):
        state = execution_state()
        scan(state, 1, 0)
        state.scan.look_count = 100
        state.navigation_started('C1')
        state.last_action = {'step': 1, 'action': 'navigate', 'candidate_id': 'C1'}
        observe(state, 2, 70, x=.5)
        options = state.action_options()
        self.assertEqual(options['orient_view_ids'], [])
        self.assertEqual(state.scan.scan_headings, [70])
        self.assertEqual(state.scan.views[0]['image'], 'images/001.jpg')
        for change in ('same_area', 'uncertain', 'new_area'):
            decision = Decision(action='look', direction='left', reason='看清走廊旁的遮挡',
                                previous_action_review={'outcome': 'progress',
                                    'view_change': change, 'evidence': '门框后出现可探索空间'})
            before = state.snapshot()
            state.resolve(decision)
            self.assertEqual(state.snapshot(), before)
        state.accept(decision)
        self.assertEqual(state.scan.number, 1)
        self.assertEqual(state.scan.look_count, 101)
        self.assertFalse(options['scan_required'])
        self.assertEqual(state.action_options()['look_directions'], ['left', 'right'])
        self.assertIsNone(state.scan.last_outcome)

    def test_candidate_descriptions_are_bound_to_each_view(self):
        state = execution_state()
        scan(state, 1, 0)
        original = list(state.scan.views[0]['candidate_ids'])
        scan(state, 2, 70)
        self.assertEqual(state.scan.views[0]['candidate_ids'], original)
        self.assertNotIn('scene', state.scan.views[0])
        self.assertNotEqual(state.scan.views[0]['id'], state.scan.views[1]['id'])
        self.assertEqual(state.scan.views[0]['heading_deg'], 0)

    def test_orient_cannot_use_unknown_position(self):
        state = execution_state()
        scan(state, 1, 0)
        observe(state, 2, 140)
        state.current['observer_position'] = None
        with self.assertRaisesRegex(ValueError, '缺少'):
            state.resolve(Decision(action='orient', view_id='O1', reason='转回'))
        self.assertEqual(state.context()['exploration_session']['orient_view_ids'], [])
        state.resolve(Decision(action='give_up', reason='定位缺失，无法安全继续'))

    def test_exploration_requires_scan_but_current_target_can_interrupt(self):
        state = execution_state()
        observe(state, 1, 0)
        proposal = Decision(
            action='navigate', reason='选择未探索开放通道', instruction='go', candidate_id='new_1',
            scene={'target_seen': 'no'},
            candidates=[{'id': 'new_1', 'kind': 'open_passage', 'description': '开放过道',
                         'evidence': '清楚可见入口', 'expected_new_view': '门框后面的走廊'}],
        )
        state.resolve(proposal)
        self.assertEqual(state.candidates, [])
        target = proposal.model_copy(deep=True)
        target.scene.target_seen = 'yes'
        target.candidates[0].kind = 'target'
        target.candidates[0].expected_new_view = ''
        state.resolve(target)
        target.scene.target_seen = 'uncertain'
        target.candidates[0].evidence = '柜旁露出银色圆柱物，桶口被挡，需接近确认'
        state.resolve(target)
        state.resolve(Decision(action='finish', reason='已找到', evidence='当前图看到银色垃圾桶'))
        state.scan.look_count = 100
        state.resolve(proposal)
        for step, yaw in enumerate((70, 140, 210, 280, 350), 2):
            observe(state, step, yaw)
        state.resolve(proposal)
        proposal.candidates[0].expected_new_view = ''
        with self.assertRaisesRegex(ValueError, 'expected_new_view'):
            state.resolve(proposal)
        state.scan.look_count = 0
        state.must_change_position = True
        proposal.candidates[0].kind = 'viewpoint'
        proposal.candidates[0].expected_new_view = '绕开受阻位置观察过道'
        state.resolve(proposal)

    def test_invalid_new_area_decision_does_not_consume_review_or_clear_views(self):
        state = execution_state()
        scan(state, 1, 0)
        state.scan.look_count = 5
        state.navigation_started('C1')
        state.last_action = {'step': 1, 'action': 'navigate', 'candidate_id': 'C1'}
        observe(state, 2, 70, x=.5)
        before = state.snapshot()
        invalid = Decision(
            action='orient', view_id='O99', reason='非法旧方向',
            previous_action_review={'outcome': 'progress', 'view_change': 'new_area',
                                    'evidence': '出现新走廊'},
        )
        with self.assertRaises(ValueError):
            state.accept(invalid)
        self.assertEqual(state.snapshot(), before)
        self.assertIsNone(state.reviewed_action_step)

    def test_area_review_never_changes_available_actions(self):
        for distance in (0, .01, 2.):
            for change in ('same_area', 'uncertain', 'new_area'):
                with self.subTest(distance=distance, change=change):
                    state = execution_state()
                    scan(state, 1, 0)
                    state.navigation_started('C1')
                    state.last_action = {'step': 1, 'action': 'navigate', 'candidate_id': 'C1'}
                    observe(state, 2, 70, x=distance)
                    before = state.action_options()
                    decision = Decision(action='look', direction='left', reason='检查遮挡',
                                        previous_action_review={'outcome': 'progress',
                                            'view_change': change, 'evidence': '保留图像比较结论'})
                    state.resolve(decision)
                    self.assertEqual(state.action_options(), before)
                    state.accept(decision)
                    self.assertEqual(state.action_options()['look_directions'], ['left', 'right'])
                    self.assertEqual(state.scan.number, 1)

    def test_new_area_review_only_applied_once(self):
        state = execution_state()
        scan(state, 1, 0)
        state.scan.look_count = 5
        state.navigation_started('C1')
        state.last_action = {'step': 1, 'action': 'navigate', 'candidate_id': 'C1'}
        observe(state, 2, 70, x=.5)
        decision = Decision(action='look', direction='left', reason='新区域环视',
                            previous_action_review={'outcome': 'progress', 'view_change': 'new_area',
                                                    'evidence': '越过门框看到新走廊'})
        state.accept(decision)
        observe(state, 3, 140, x=.5)
        state.accept(decision)
        self.assertEqual(state.scan.number, 1)
        self.assertEqual(state.scan.look_count, 7)

    def test_candidate_history_survives_new_place_without_forcing_useless_revisit(self):
        state = execution_state()
        scan(state, 1, 0)
        state.scan.look_count = 5
        for step in (1, 2):
            state.step = step
            state.navigation_started('C1')
            state.record_result('C1', {'reason': 'model_stop', 'motion': {'displacement_m': .5}})
            state.last_action = {'step': step, 'action': 'navigate', 'candidate_id': 'C1'}
            observe(state, step + 1, 70, x=.5 * step)
            # 复核后无有效方向可以结束，不要求再转回无新信息的候选。
            review = Decision(action='finish', reason='测试复核', evidence='当前图目标明确',
                              previous_action_review={'outcome': 'no_new_information',
                                  'view_change': 'same_area', 'evidence': '仍被同一柜体遮挡'})
            state.accept(review)
        self.assertEqual(state.candidates[0]['no_new_information_count'], 2)
        self.assertEqual(state.scan.useful_choices(state.candidates, 70, 25), [])
        self.assertEqual(state.scan.choices(state.candidates, 70, 25), [])
        # 不强迫返回旧候选；当前新位置仍有未看方向时，不能因此宣称无路可走。
        with self.assertRaisesRegex(ValueError, '仍有未看的方向'):
            state.resolve(Decision(action='give_up', reason='无有效探索方向'))
        state.current['id'] = 'P2'
        proposal = Decision(action='navigate', reason='从左侧不同路径绕行', instruction='go',
                            candidate_id='C1', candidates=[{'id': 'C1', 'evidence_source': 'current', 'kind': 'viewpoint',
                                'description': '旧柜子左侧开口', 'evidence': '当前看到新的左侧绕行路径',
                                'expected_new_view': '此前柜体挡住的左侧过道'}])
        # 换位置本身不解除重复限制；转向看见新路径后才允许再接近。
        state.scan.complete = True
        with self.assertRaisesRegex(ValueError, '连续两次'):
            state.accept(proposal)
        state.last_action = {'step': 3, 'action': 'look'}
        proposal.previous_action_review = ActionReview(outcome='progress', view_change='same_area',
                                                       evidence='转向后看见柜子左侧此前被遮挡的路径')
        selected = state.accept(proposal)
        related = next(item for item in state.candidates if item['id'] == selected)
        self.assertEqual(related['total_attempt_count'], 2)
        self.assertEqual(related['no_new_information_count'], 0)
        self.assertEqual(related['attempt_count'], 0)
        self.assertEqual(related['origin_candidate_id'], 'C1')
        self.assertEqual(related['visual_review']['view_change'], 'same_area')

    def test_safety_position_constraint_clears_without_resetting_scan(self):
        state = execution_state()
        scan(state, 1, 0)
        state.scan.look_count = 5
        state.record_result('C1', {'reason': 'blocked', 'motion': {'displacement_m': 0}})
        observe(state, 2, 70, x=.5)
        self.assertFalse(state.must_change_position)
        self.assertFalse(state.recovery_required)
        self.assertEqual(state.scan.look_count, 5)

    def test_old_candidates_cannot_be_used_without_current_visual_registration(self):
        state = execution_state()
        scan(state, 1, 0)
        state.scan.look_count = 5
        observe(state, 2, 140, x=.5)
        with self.assertRaisesRegex(ValueError, '当前可见'):
            state.resolve(Decision(action='navigate', reason='盲走旧方向', instruction='go',
                                   candidate_id='C1'))

    def test_recorded_positions_keep_only_local_heading_evidence(self):
        # 2026-09-22 银色垃圾桶任务：四轮重复环视的实际观察位置与朝向。
        trace = [
            (1, -0.668, -0.055, 154.9),
            (2, -0.637, -0.11, 87.4),
            (3, -0.67, -0.146, 21.5),
            (4, -0.715, -0.131, -45.4),
            (5, -0.702, -0.08, -113.5),
            (6, -0.655, -0.066, 177.0),
            (7, -0.623, -0.134, 92.8),
            (8, -0.789, 1.769, 64.3),
            (9, -0.739, 1.732, 133.0),
            (10, -0.693, 1.767, -158.8),
            (11, -0.713, 1.825, -89.6),
            (12, -0.774, 1.826, -22.1),
            (13, -0.804, 1.768, 47.0),
            (14, -0.921, 1.703, -83.4),
            (15, -1.104, -0.262, -89.0),
            (16, -1.166, -0.245, -21.3),
            (17, -1.208, -0.289, 46.0),
            (18, -1.181, -0.346, 113.8),
            (19, -1.117, -0.334, -178.3),
            (20, -1.109, -0.272, -110.8),
            (21, -1.197, -0.238, -27.7),
            (22, -0.553, -0.85, -26.4),
            (23, -0.637, -0.813, 40.8),
            (24, -0.59, -0.83, 108.3),
            (25, -0.537, -0.857, 176.7),
            (26, -0.491, -0.797, -114.7),
            (27, -0.534, -0.781, -45.8),
            (28, -0.529, -0.812, 100.5),
        ]
        state = execution_state(look_angle_deg=75)
        for step, x, y, yaw in trace:
            if step in (8, 15, 22):
                state.last_action = {'step': step - 1, 'action': 'navigate'}
            state.observe({'id': 'P' + str(step), 'step': step,
                           'observer_position': [x, y], 'heading_deg': yaw,
                           'image': f'images/{step:03d}.jpg'})
            if step <= 5:
                proposal = Decision(action='look', direction='right', reason='首轮环视')
                if step == 5:
                    with self.assertRaisesRegex(ValueError, '复核'):
                        state.accept(proposal)
                else:
                    state.accept(proposal)
            if step in (8, 15, 22):
                self.assertEqual(state.scan.current_view_id, f'O{step}')
                self.assertEqual(state.scan.scan_headings, [yaw])
                self.assertEqual(state.context()['exploration_session']['orient_view_ids'], [])
                state.resolve(Decision(action='look', direction='left', reason='确认新的局部遮挡'))
                self.assertEqual(state.scan.look_count, 4)

    def test_old_fields_default_to_uncertain_and_review_uncertainty_is_accepted(self):
        from navigation_agent.agent.schemas import ActionReview, CandidateObservation
        self.assertEqual(ActionReview(outcome='progress', evidence='旧日志').view_change, 'uncertain')
        self.assertEqual(CandidateObservation(id='C1', kind='viewpoint', description='旧候选',
                                             evidence='旧证据').expected_new_view, '')
        review = ActionReview(outcome='uncertain', view_change='new_area', evidence='疑似新区域')
        self.assertEqual(review.outcome, 'uncertain')

    def test_known_side_views_can_lead_to_unseen_rear_directions(self):
        state = execution_state(look_angle_deg=70)
        for step, yaw in enumerate((0, 70, -70, 0), 1):
            observe(state, step, yaw)
        options = state.action_options()
        self.assertTrue(all(item['repeat_of'] for item in options['look_options'].values()))
        routes = {item['view_id']: item['then_look'] for item in options['unseen_via_orient']}
        self.assertEqual(routes['O2'], ['left'])
        self.assertEqual(routes['O3'], ['right'])
        with self.assertRaisesRegex(ValueError, '仍有未看的方向'):
            state.resolve(Decision(action='give_up', reason='左右都看过且没找到出口'))
        state.accept(Decision(
            action='orient', view_id='O2', reason='转到左侧已知方向，继续检查后方',
            observation_intent={'reference_view_id': 'O2', 'question': '左后方是否有出口',
                                'new_evidence': '方向记忆显示左后方尚未检查'},
        ))
        observe(state, 5, 70)
        state.accept(Decision(action='look', direction='left', reason='检查左后方尚未看的出口'))
        observe(state, 6, 140)
        self.assertEqual(state.scan.unseen_directions_after(state.scan.views[1]), [])

    def test_give_up_remains_available_after_nearby_directions_are_checked(self):
        state = execution_state(look_angle_deg=70)
        for step, yaw in enumerate((0, 70, 140, 210, 280), 1):
            observe(state, step, yaw)
        self.assertEqual(state.action_options()['unseen_via_orient'], [])
        state.resolve(Decision(action='give_up', reason='周围各方向均为墙面，没有可继续的路线'))

    def test_first_return_can_continue_scanning_but_second_return_is_rejected(self):
        state = execution_state(look_angle_deg=70)
        observe(state, 1, 0)
        observe(state, 2, -70)
        restore = Decision(
            action='orient', view_id='O1', reason='恢复开阔方向以继续检查另一侧',
            observation_intent={'reference_view_id': 'O1', 'question': '如何找到剩余出口',
                                'new_evidence': '当前朝向被墙挡住，需要恢复开阔方向'},
        )
        state.accept(restore)
        observe(state, 3, 0)
        state.accept(Decision(action='look', direction='left', reason='查看左侧未看范围'))
        observe(state, 4, 70)
        restore.view_id = 'O3'
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            state.accept(restore)
        self.assertEqual(state.scan.selected_view_id, 'O1')
        state.resolve(Decision(action='look', direction='left', reason='继续检查左后方'))

    def test_orient_destination_is_independent_of_the_supporting_reference(self):
        state = execution_state()
        observe(state, 1, 0, x=0)
        observe(state, 2, 20, x=2)
        observe(state, 3, 90, x=2)
        decision = Decision(
            action='orient', view_id='O2', reason='转回最近的开阔方向继续找入口',
            observation_intent={'reference_view_id': 'O1', 'question': '转回后如何找到之前的入口',
                                'new_evidence': '当前被墙挡住，先恢复到最近可见地面的方向'},
        )
        state.accept(decision)
        self.assertEqual(state.scan.selected_view_id, 'O2')
        observe(state, 4, 20, x=2)
        decision.view_id = 'O1'
        with self.assertRaisesRegex(ValueError, '旧位置'):
            state.resolve(decision)

    def test_unregistered_view_can_be_oriented_without_spending_scan_budget(self):
        state = execution_state()
        observe(state, 1, 0)
        state.accept(Decision(action='look', direction='left', reason='继续环视', candidates=[]))
        observe(state, 2, 70)
        before = state.scan.look_count
        self.assertEqual(state.candidates, [])
        self.assertIn('O1', state.context()['exploration_session']['orient_view_ids'])
        state.accept(Decision(action='orient', view_id='O1', reason='旧图有疑似目标，转回确认',
                              observation_intent={'reference_view_id': 'O1', 'question': '原方向空地的当前接近路径是否通畅',
                                    'new_evidence': '比较当前方向后选择 O1 空地作为下一处观察位置，准备转回接近'}))
        self.assertEqual(state.scan.look_count, before)
        self.assertEqual(state.scan.selected_view_id, 'O1')

    def test_navigation_cannot_use_historical_evidence_or_skip_fresh_observation(self):
        state = execution_state()
        observe(state, 1, 0)
        observe(state, 2, 140)
        state.scan.complete = True
        proposal = Decision(
            action='navigate', reason='探索门后', instruction='go', candidate_id='new_1',
            candidates=[{'id': 'new_1', 'kind': 'open_passage', 'description': '开放门洞',
                         'evidence': 'O1图中可见门后地面', 'expected_new_view': '门后走廊'}],
        )
        with self.assertRaisesRegex(ValueError, '历史图'):
            state.resolve(proposal)
        proposal.candidates[0].evidence = '当前图门前地面连续可通行'
        proposal.view_id = 'O1'
        with self.assertRaisesRegex(ValueError, '历史图'):
            state.resolve(proposal)
        proposal.view_id = None
        state.accept(Decision(action='orient', view_id='O1', reason='转回门洞',
                              observation_intent={'reference_view_id': 'O1', 'question': '原方向空地的当前接近路径是否通畅',
                                    'new_evidence': '比较当前方向后选择 O1 空地作为下一处观察位置，准备转回接近'}))
        with self.assertRaisesRegex(ValueError, '新观察'):
            state.resolve(proposal)
        observe(state, 3, 0)
        proposal.view_id = 'O3'
        state.resolve(proposal)

    def test_uncertain_review_does_not_authorize_purposeless_reinspection(self):
        state = execution_state()
        scan(state, 1, 0)
        state.navigation_started('C1')
        state.last_action = {'step': 1, 'action': 'navigate', 'candidate_id': 'C1'}
        observe(state, 2, 70, x=.75)
        state.accept(Decision(action='look', direction='right', reason='看清门框后方',
                             previous_action_review={'outcome': 'uncertain', 'evidence': '尚未看清'}))
        observe(state, 3, 0, x=.75)
        proposal = Decision(action='look', direction='left', reason='继续扩大视野',
                            previous_action_review={'outcome': 'uncertain', 'evidence': '尚未看清'})
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '复核'):
            state.accept(proposal)
        self.assertEqual(state.snapshot(), before)

    def test_delayed_area_review_does_not_rewrite_navigation_or_change_options(self):
        state = execution_state()
        scan(state, 1, 0)
        state.navigation_started('C1')
        state.last_action = {'step': 1, 'action': 'navigate', 'candidate_id': 'C1'}
        observe(state, 2, 70, x=.75)
        state.accept(Decision(action='look', direction='right', reason='看清门框后方',
                              previous_action_review={'outcome': 'uncertain', 'evidence': '尚未看清'}))
        state.last_action = {'step': 2, 'action': 'look'}
        observe(state, 3, 0, x=.75)
        before = state.action_options()
        state.accept(Decision(action='look', direction='right', reason='继续同向环视',
                              observation_intent={'reference_view_id': 'O2', 'question': '门框旁露出的缝隙是否可通行',
                                    'new_evidence': '当前左边缘出现此前被柜体遮住的地面连接'},
                              previous_action_review={'outcome': 'progress', 'view_change': 'new_area',
                                                      'evidence': '补看见独立走廊'}))
        self.assertEqual(state.action_options(), before)
        self.assertEqual(state.last_navigation['visual_review']['outcome'], 'uncertain')
        self.assertIsNone(state.scan.last_outcome)

    def test_current_frame_identity_survives_many_views_and_missing_pose(self):
        state = execution_state()
        for step in range(1, 21):
            observe(state, step, 0)
        self.assertEqual(len(state.scan.views), 20)
        self.assertEqual(state.scan.current_view_id, 'O20')
        self.assertEqual(state.scan.views[0]['image'], 'images/001.jpg')
        missing = {'id': 'P2', 'step': 21, 'heading_deg': None,
                   'observer_position': None, 'image': 'images/021.jpg'}
        state.observe(missing)
        self.assertEqual(state.scan.current_view_id, 'O21')
        self.assertEqual(state.action_options()['orient_view_ids'], [])
        missing.update(step=22, observer_position=[0, 0])
        state.observe(missing)
        self.assertEqual(state.action_options()['orient_view_ids'], [])
        self.assertEqual(state.context()['exploration_session']['orient_view_ids'], [])

    def test_nearby_direction_context_prefers_latest_frame_and_excludes_old_positions(self):
        state = execution_state()
        observe(state, 1, 0)
        observe(state, 2, 3)
        observe(state, 3, 140)
        self.assertEqual(state.action_options()['orient_view_ids'], ['O2'])
        self.assertEqual([v['id'] for v in state.context()['exploration_session']['views']], ['O2', 'O3'])
        observe(state, 4, 140, x=2)
        self.assertEqual([v['id'] for v in state.context()['exploration_session']['views']], ['O4'])
        observe(state, 5, 140)
        self.assertIn('O2', state.action_options()['orient_view_ids'])

    def test_old_position_direction_cannot_be_reused_after_navigation_translation(self):
        state = execution_state()
        scan(state, 1, 0)
        observe(state, 2, 140)
        state.accept(Decision(action='orient', view_id='O1', reason='转回',
                              observation_intent={'reference_view_id': 'O1', 'question': '原方向空地的当前接近路径是否通畅',
                                    'new_evidence': '比较当前方向后选择 O1 空地作为下一处观察位置，准备转回接近'}))
        observe(state, 3, 0)
        self.assertIn('O1', state.scan.checked_views)
        state.navigation_started('C1')
        state.last_action = {'step': 3, 'action': 'navigate', 'candidate_id': 'C1'}
        observe(state, 4, 140, x=.5)
        self.assertNotIn('O1', state.context()['exploration_session']['orient_view_ids'])
        self.assertEqual(state.scan.number, 1)


    def test_recovered_pose_can_anchor_the_next_navigation_scan(self):
        state = execution_state()
        state.observe({'id': 'P1', 'step': 1, 'observer_position': None,
                       'heading_deg': None, 'image': 'images/001.jpg'})
        observe(state, 2, 0)
        self.assertEqual(state.scan.scan_anchor, [0, 0])
        self.assertEqual(state.scan.number, 1)
        state.last_action = {'step': 2, 'action': 'navigate'}
        observe(state, 3, 0, x=1)
        self.assertEqual(state.scan.number, 1)
        self.assertFalse(state.scan.required())

    def test_small_navigation_and_turn_drift_do_not_restart_completed_scan(self):
        state = execution_state(look_angle_deg=75)
        for step, yaw in enumerate((0, 70, 140, 210, 280, 350), 1):
            observe(state, step, yaw)
        self.assertTrue(state.scan.coverage_complete())
        state.last_action = {'step': 6, 'action': 'navigate'}
        observe(state, 7, 350, x=.1)
        self.assertTrue(state.scan.coverage_complete())
        self.assertEqual(state.scan.number, 1)
        state.last_action = {'step': 7, 'action': 'orient'}
        observe(state, 8, 70, x=.3)
        self.assertFalse(state.scan.required())
        self.assertEqual(state.scan.number, 1)
        self.assertEqual(state.allowed_look_directions(), ['left', 'right'])


class ReinspectionTests(unittest.TestCase):
    def test_specific_new_evidence_allows_revisit_but_reusing_it_does_not(self):
        state = execution_state(look_angle_deg=75)
        state.scan.pause_reason = 'target'
        observe(state, 1, 0)
        observe(state, 2, 70, x=.05)
        intention = {
            'reference_view_id': 'O1',
            'question': '桌脚旁的圆柱物是否有桶口',
            'new_evidence': '当前左边缘新露出银色圆柱物，上次该位置被椅子挡住',
        }
        proposal = Decision(action='look', direction='right', reason='确认新出现的目标局部',
                            observation_intent=intention)
        before = state.snapshot()
        state.resolve(proposal)
        self.assertEqual(state.snapshot(), before)
        state.accept(proposal)
        observe(state, 3, 2, x=.07)
        observe(state, 4, 72, x=.08)
        reused = proposal.model_copy(deep=True)
        reused.observation_intent.reference_view_id = 'O3'
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            state.accept(reused)
        self.assertEqual(state.snapshot(), before)
        reused.observation_intent.new_evidence = '换种说法继续看相同方向'
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            state.resolve(reused)
        reused.observation_intent.new_evidence = '当前椅子已移开，圆柱物顶部露出的范围扩大'
        from navigation_agent.agent.schemas import ActionReview
        reused.previous_action_review = ActionReview(
            outcome='progress', view_change='same_area', evidence='当前图显示遮挡椅子移开并露出桶口')
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            state.resolve(reused)

    def test_reinspection_reference_must_match_the_actual_turn(self):
        state = execution_state(look_angle_deg=75)
        state.scan.pause_reason = 'target'
        observe(state, 1, 0)
        observe(state, 2, 70)
        with self.assertRaisesRegex(ValueError, 'reference_view_id=O1'):
            state.resolve(Decision(action='look', direction='right', reason='确认遮挡',
                observation_intent={'reference_view_id': 'O2', 'question': '桶口是否可见',
                                    'new_evidence': '椅子旁出现银色边缘'}))

    def test_unknown_pose_does_not_invent_unseen_or_repeated_directions(self):
        state = execution_state(look_angle_deg=75)
        observe(state, 1, 0)
        state.observe({'id': 'P1', 'observer_position': None, 'heading_deg': None,
                       'step': 2, 'image': 'images/002.jpg'})
        for option in state.action_options()['look_options'].values():
            self.assertFalse(option['position_known'])
            self.assertIsNone(option['repeat_of'])
        state.resolve(Decision(action='look', direction='right', reason='查看右侧遮挡'))


class CoherentObservationTests(unittest.TestCase):
    def setUp(self):
        self.state = execution_state()
        observe(self.state, 1, 0)
        observe(self.state, 2, -70)
        self.state.accept(self.orient('O1'))
        self.state.scan.record_result('orient', {'reason': 'oriented'}, view_id='O1')
        observe(self.state, 3, 0)
        observe(self.state, 4, 70)

    @staticmethod
    def orient(view_id, **extra):
        return Decision(action='orient', view_id=view_id, reason='转回后决定下一步',
                        observation_intent={'reference_view_id': view_id,
                            'question': '另一侧是否还有路线', 'new_evidence': '当前只看到墙面'},
                        **extra)

    def test_returning_to_an_equivalent_new_image_does_not_reset_repeat_count(self):
        option = next(item for item in self.state.action_options()['orient_options']
                      if item['view_id'] == 'O3')
        self.assertEqual(option['returned_times'], 1)
        self.assertEqual(option['unseen_then_look'], [])
        self.assertIsNone(self.state.action_options()['look_options']['left']['repeat_of'])
        before = self.state.snapshot()
        with self.assertRaisesRegex(ValueError, '已从当前位置转回'):
            self.state.resolve(self.orient('O3'))
        self.assertEqual(self.state.snapshot(), before)
        self.state.resolve(Decision(action='look', direction='left', reason='继续查看后方未知区域'))

    def test_known_route_and_new_target_clue_do_not_reset_completed_reinspection(self):
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            self.state.resolve(self.orient('O3', scene={'target_seen': 'uncertain'}))
        self.state.candidates = [{'id': 'C1', 'status': 'untried', 'kind': 'open_passage'}]
        self.state.scan.views[2]['candidate_ids'] = ['C1']
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            self.state.resolve(self.orient('O3'))

    def test_new_position_does_not_inherit_turn_back_count(self):
        observe(self.state, 5, 0, x=2)
        observe(self.state, 6, 70, x=2)
        option = next(item for item in self.state.action_options()['orient_options']
                      if item['view_id'] == 'O5')
        self.assertEqual(option['returned_times'], 0)
        self.state.resolve(self.orient('O5'))

    def test_latest_view_without_candidates_keeps_known_route_in_same_direction(self):
        self.state.candidates = [{'id': 'C1', 'status': 'untried', 'kind': 'open_passage'}]
        self.state.scan.views[0]['candidate_ids'] = ['C1']
        option = next(item for item in self.state.action_options()['orient_options']
                      if item['view_id'] == 'O3')
        self.assertEqual(option['candidate_ids'], ['C1'])
        self.assertFalse(option['allowed'])
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            self.state.resolve(self.orient('O3'))

    def test_blocked_route_cannot_justify_repeated_return(self):
        self.state.candidates = [{'id': 'C1', 'status': 'avoid_from_here', 'kind': 'open_passage'}]
        self.state.scan.views[0]['candidate_ids'] = ['C1']
        with self.assertRaisesRegex(ValueError, '已从当前位置转回'):
            self.state.resolve(self.orient('O3'))
