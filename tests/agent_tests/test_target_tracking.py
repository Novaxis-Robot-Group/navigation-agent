"""目标接近后的证据保留、局部复核与请求附图回归。"""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from navigation_agent.agent.schemas import Decision, TaskRequest
from navigation_agent.agent.context.builder import build_context
from navigation_agent.agent.context.images import image_manifest
from navigation_agent.agent.skills.object_search.context import ObjectSearchContext
from tests.config_support import agent_settings, execution_state, spatial_memory
from tests.agent_tests import test_loop
from tests.agent_tests.test_loop import navigation, target_review


def view(step, x=0, yaw=0):
    return {
        'id': 'P1' if x == 0 else 'P2', 'step': step,
        'observer_position': [x, 0], 'position': [x, 0],
        'heading_deg': yaw, 'image': f'images/{step:03d}.jpg',
    }


def begin_target():
    state = execution_state()
    state.observe(view(1))
    selected = state.accept(Decision.model_validate(navigation()))
    state.navigation_started(selected)
    result = {'reason': 'model_stop', 'motion': {'displacement_m': 2.52}}
    state.record_result(selected, result)
    state.last_action = {'step': 1, 'action': 'navigate', 'candidate_id': selected, 'result': result}
    state.observe(view(2, 2.52))
    return state


class TargetTrackingTests(unittest.TestCase):
    def test_lost_target_cannot_be_reassigned_to_another_doorway(self):
        state = begin_target()
        proposal = navigation(review_step=2)
        proposal['target_review']['status'] = 'temporarily_lost'
        proposal['scene']['target_seen'] = 'no'
        proposal['candidates'][0]['kind'] = 'open_passage'
        proposal['candidates'][0]['expected_new_view'] = '猜测另一扇门后有目标'
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '另一个门洞'):
            state.accept(Decision.model_validate(proposal))
        self.assertEqual(state.snapshot(), before)

    def test_reused_finish_evidence_cannot_bypass_stale_observation_rejection(self):
        state = begin_target()
        decision = Decision(action='finish', reason='认为已到达', scene={'target_seen': 'yes'},
                            target_review={'observation_step': 1, 'status': 'arrived',
                                           'evidence': '旧图中的目标'})
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '当前 observation_step'):
            state.accept(decision)
        self.assertEqual(state.snapshot(), before)

    def test_lost_phase_and_anchor_survive_motion_and_budget_resets_only_when_seen(self):
        state = begin_target()
        anchor = copy.deepcopy(state.target.active['last_seen_anchor'])
        lost = Decision(action='look', direction='left', reason='找回左侧目标',
                        scene={'target_seen': 'no'}, target_review=target_review(2, 'temporarily_lost'))
        state.accept(lost)
        state.target.record_reacquire_action()
        state.target.navigation_result({'reason': 'agent_interrupt'})
        self.assertEqual(state.target.active['phase'], 'temporarily_lost')
        self.assertEqual(state.target.active['last_seen_anchor'], anchor)
        self.assertEqual(state.target.context()['reacquire_actions_remaining'], 3)
        state.observe(view(3, 2.52, yaw=70))
        state.target.apply_review(Decision(action='look', direction='right', reason='已重新看见',
                                 scene={'target_seen': 'yes'}, target_review=target_review(3, 'visible')), state.current)
        self.assertEqual(state.target.active['reacquire_actions'], 0)
        self.assertEqual(state.target.active['last_seen_anchor']['step'], 3)

    def test_camera_motion_explains_changed_left_right_without_guessing_target_position(self):
        state = begin_target()
        state.target.active['last_seen_anchor'].update(observer_position=[0, 0], heading_deg=170)
        state.observe({**view(3, 3, -170), 'observer_position': [3, 4]})
        motion = state.context()['target_tracking']['camera_motion_since_sighting']
        self.assertEqual(motion['translation_m'], 5.)
        self.assertEqual(motion['heading_change_deg'], 20.)
        self.assertIn('不是目标当前位置', motion['meaning'])
        state.current['heading_deg'] = None
        self.assertIsNone(state.context()['target_tracking']['camera_motion_since_sighting'])

    def test_lost_target_allows_only_short_visible_viewpoint_motion(self):
        state = begin_target()
        proposal = navigation(review_step=2)
        proposal['target_review']['status'] = 'temporarily_lost'
        proposal['scene']['target_seen'] = 'no'
        proposal['candidates'][0]['kind'] = 'viewpoint'
        proposal['candidates'][0]['expected_new_view'] = '向侧面挪动以看清原遮挡物后方'
        decision = Decision.model_validate(proposal)
        selected = state.accept(decision)
        skill = ObjectSearchContext(agent_settings(), 70)
        skill.state = state
        call, event = skill.prepare_call(decision, selected, 15.)
        self.assertEqual(call.arguments['seconds'], 3.)
        self.assertEqual(call.max_seconds, 3.)
        self.assertTrue(event['reacquiring_target'])

    def test_only_viewpoints_receive_explicit_local_distance_limit(self):
        settings = agent_settings()
        settings = settings.model_copy(update={
            'exploration': settings.exploration.model_copy(update={'viewpoint_max_displacement_m': 1.5}),
        })
        for kind, expected in [('target', None), ('open_passage', None), ('viewpoint', 1.5)]:
            with self.subTest(kind=kind):
                skill = ObjectSearchContext(settings, 70)
                skill.observe(view(1))
                proposal = navigation(kind=kind)
                decision = Decision.model_validate(proposal)
                selected = skill.state.accept(decision)
                call, _ = skill.prepare_call(decision, selected, 15.)
                self.assertEqual(call.arguments.get('max_displacement_m'), expected)

    def test_legacy_fixture_disables_distance_but_new_schema_defaults_to_1_5m(self):
        settings = agent_settings()
        self.assertEqual(settings.exploration.viewpoint_max_displacement_m, 0.)
        exploration = settings.exploration.model_dump()
        exploration.pop('viewpoint_max_displacement_m')
        defaults = type(settings.exploration).model_validate(exploration)
        self.assertEqual(defaults.viewpoint_max_displacement_m, 1.5)
        skill = ObjectSearchContext(settings, 70)
        skill.observe(view(1))
        decision = Decision.model_validate(navigation(kind='viewpoint'))
        selected = skill.state.accept(decision)
        call, _ = skill.prepare_call(decision, selected, 15.)
        self.assertIsNone(call.arguments.get('max_displacement_m'))

    def test_reacquire_budget_prevents_unbounded_local_search(self):
        state = begin_target()
        state.target.active['phase'] = 'temporarily_lost'
        state.target.active['reacquire_actions'] = state.target.reacquire_limit
        with self.assertRaisesRegex(ValueError, '预算'):
            state.resolve(Decision(action='look', direction='left', reason='继续查找'))
        with self.assertRaisesRegex(ValueError, '预算'):
            state.resolve(Decision(action='give_up', reason='局部找回预算用尽，保留最后目击'))
        state.accept(Decision(action='look', direction='left', reason='暂缓局部跟进，检查未看方向',
                              scene={'target_seen': 'no'}, target_review=target_review(2, 'temporarily_lost'),
                              subgoal_review={'status': 'abandoned', 'evidence': '多次局部检查未能复核，先查未看方向'}))
        self.assertIsNone(state.target.active)
        self.assertIsNone(state.scan.pause_reason)
        self.assertEqual(state.subgoal_history[-1]['status'], 'abandoned')
        self.assertIn('保留O1', state.subgoal_history[-1]['evidence'])
        self.assertIn('不代表目标不存在', state.subgoal_history[-1]['evidence'])

    def test_deferred_followup_can_choose_new_visible_route_but_cannot_blindly_reapproach_target(self):
        state = begin_target()
        state.target.active.update(phase='temporarily_lost', reacquire_actions=state.target.reacquire_limit)
        proposal = navigation(review_step=2)
        proposal['target_review']['status'] = 'temporarily_lost'
        proposal['scene']['target_seen'] = 'no'
        proposal['subgoal_review'] = {'status': 'abandoned', 'evidence': '局部跟进没有新证据，保留线索后探索'}
        with self.assertRaisesRegex(ValueError, '暂时不可见'):
            state.resolve(Decision.model_validate(proposal))
        proposal['candidates'][0]['kind'] = 'open_passage'
        proposal['candidates'][0]['expected_new_view'] = '检查当前可见入口后的未知区域，不声称旧目标在那里'
        decision = Decision.model_validate(proposal)
        before = state.snapshot()
        state.resolve(decision)
        self.assertEqual(state.snapshot(), before)
        selected = state.accept(decision)
        self.assertIsNone(state.target.active)
        self.assertNotEqual(selected, state.subgoal_history[-1]['origin_candidate_id'])
        self.assertEqual(state.candidates[0]['description'], before['target_tracking']['description'])

    def test_wrong_room_can_defer_followup_before_spending_the_action_budget(self):
        state = begin_target()
        decision = Decision(
            action='look', direction='left', reason='已停在另一房间，重新确认出口',
            target_review=target_review(2, 'temporarily_lost'),
            subgoal_review={'status': 'abandoned', 'evidence': '当前已是厨房，原目击在门外；暂缓跟进后重新定位'},
        )
        before = state.snapshot()
        state.resolve(decision)
        self.assertEqual(state.snapshot(), before)
        state.accept(decision)
        self.assertIsNone(state.target.active)
        self.assertIn('保留O1', state.subgoal_history[-1]['evidence'])
        self.assertEqual(state.subgoal_history[-1]['status'], 'abandoned')

    def test_early_deferral_does_not_allow_giving_up_the_whole_task(self):
        state = begin_target()
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '不能仅因此结束'):
            state.accept(Decision(
                action='give_up', reason='走偏后结束任务',
                target_review=target_review(2, 'temporarily_lost'),
                subgoal_review={'status': 'abandoned', 'evidence': '走错房间，暂缓当前目标跟进'},
            ))
        self.assertEqual(state.snapshot(), before)

    def test_exhausted_local_followup_can_retreat_after_visible_directions_are_checked(self):
        state = begin_target()
        state.last_navigation['result'] = {'reason': 'model_stop'}
        for step, heading in enumerate((0, 70, 140, 210, 280, 350), 2):
            state.observe(view(step, 2.52, heading))
        self.assertFalse(state.action_options()['recover_allowed'])
        state.target.active.update(phase='temporarily_lost', reacquire_actions=state.target.reacquire_limit)
        self.assertTrue(state.action_options()['recover_allowed'])
        state.accept(Decision(action='recover', reason='暂缓局部跟进后沿已走路径退回换位置',
                              target_review=target_review(7, 'temporarily_lost'),
                              subgoal_review={'status': 'abandoned', 'evidence': '各方向已查但受遮挡，退回后继续探索'}))
        self.assertIsNone(state.target.active)

    def test_deferred_target_evidence_is_retained_in_existing_topology_memory(self):
        from navigation_agent.agent.memory.topology import TopologyMemory

        with tempfile.TemporaryDirectory() as folder:
            memory = spatial_memory(Path(folder) / 'memory.sqlite3', '找到目标')
            first = memory.observe({'pose': {'x': 0., 'y': 0.}, 'yaw': 0.}, 'images/001.jpg', 1)
            second = memory.observe({'pose': {'x': 2.52, 'y': 0.}, 'yaw': 0.}, 'images/002.jpg', 2)
            state = begin_target()
            topology = TopologyMemory(memory)
            sighting = Decision.model_validate(navigation())
            memory.describe(first['id'], 1, sighting)
            topology.update(first, state, sighting)
            state.target.active.update(phase='temporarily_lost', reacquire_actions=state.target.reacquire_limit)
            deferred = Decision(action='look', direction='left', reason='暂缓当前跟进，继续探索',
                                target_review=target_review(2, 'temporarily_lost'),
                                subgoal_review={'status': 'abandoned', 'evidence': '局部多次未能复核该线索'})
            state.accept(deferred)
            topology.update(second, state, deferred)
            recalled = TopologyMemory(memory).query(second)['reviewed_routes']
            route = next(item for item in recalled if item['kind'] == 'target')
            self.assertEqual(route['exploration_status'], 'deferred')
            self.assertEqual(route['source_step'], 1)
            self.assertIn('保留O1', route['resolution'])
            self.assertEqual(memory.frames([1])[0]['scene']['target_seen'], 'yes')

    def test_navigation_stop_enters_confirmation_without_reopening_scan(self):
        state = begin_target()
        self.assertEqual(state.target.active['phase'], 'verifying_arrival')
        self.assertEqual(state.allowed_look_directions(), ['left', 'right'])
        proposal = Decision(
            action='look', direction='right', reason='检查目标是否在右侧边缘',
            scene={'target_seen': 'no'},
            previous_action_review={'outcome': 'progress', 'view_change': 'new_area',
                                    'evidence': '已移动，但目标暂时不在视野'},
            target_review=target_review(2, 'temporarily_lost'),
        )
        state.accept(proposal)
        self.assertEqual(state.scan.number, 1)
        self.assertEqual(state.target.active['first_seen_step'], 1)
        self.assertEqual(state.target.active['phase'], 'temporarily_lost')

    def test_target_confirmation_has_no_turn_quota_and_invalid_review_is_transactional(self):
        state = begin_target()
        for step in range(2, 6):
            state.observe(view(step, 2.52, yaw=-(step - 2) * 70))
            state.accept(Decision(action='look', direction='right', reason='复核目标近处局部',
                                  target_review=target_review(step, 'temporarily_lost')))
        self.assertEqual(state.target.active['confirmation_turns'], 4)
        self.assertNotIn('confirmation_turns_remaining', state.target.context())
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '当前'):
            state.accept(Decision(action='look', direction='right', reason='错误地使用旧帧',
                                  target_review=target_review(2, 'temporarily_lost')))
        self.assertEqual(state.snapshot(), before)
        self.assertEqual(state.target.evidence_steps(), [1])

    def test_orient_and_look_keep_target_evidence_without_changing_action_options(self):
        state = begin_target()
        state.accept(Decision(action='look', direction='left', reason='检查近处局部'))
        state.observe(view(3, 2.52, yaw=70))
        options = state.action_options()
        state.accept(Decision(action='orient', view_id='O2', reason='转回复核目标边缘',
                              observation_intent={'reference_view_id': 'O2', 'question': '目标机身在边缘是否仍可辨认',
                                    'new_evidence': '当前右边缘露出与此前目标一致的关节，需要转回确认'}))
        state.observe(view(4, 2.52, yaw=0))
        self.assertEqual(state.target.active['confirmation_turns'], 2)
        state.resolve(Decision(action='look', direction='right', reason='确认另一侧遮挡'))
        self.assertEqual(state.target.evidence_steps(), [1])
        self.assertIn('O2', options['orient_view_ids'])

    def test_repeated_navigation_requires_current_frame_review(self):
        state = begin_target()
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, 'target_review'):
            state.accept(Decision.model_validate(navigation()))
        stale = navigation(review_step=1)
        with self.assertRaisesRegex(ValueError, '当前'):
            state.accept(Decision.model_validate(stale))
        self.assertEqual(state.snapshot(), before)
        lost = navigation(review_step=2)
        lost['target_review']['status'] = 'temporarily_lost'
        with self.assertRaisesRegex(ValueError, '暂时不可见'):
            state.accept(Decision.model_validate(lost))

    def test_history_or_stop_without_current_review_cannot_finish(self):
        state = begin_target()
        for proposal in (
            Decision(action='finish', reason='模型停车了', evidence='以前看到过目标'),
            Decision(action='finish', reason='当前没看到', evidence='历史图有目标',
                     scene={'target_seen': 'no'}, target_review=target_review(2, 'arrived')),
        ):
            with self.assertRaises(ValueError):
                state.resolve(proposal)

    def test_partial_current_target_can_finish_without_more_scanning(self):
        state = begin_target()
        decision = Decision(
            action='finish', reason='已到目标旁边',
            evidence='当前图右边近处可见与目标一致的机身和关节',
            scene={'target_seen': 'yes'},
            target_review={'observation_step': 2, 'status': 'arrived',
                           'evidence': '机身局部占据当前图右边缘，已在旁边'},
        )
        state.accept(decision)
        self.assertEqual(state.scan.look_count, 0)

    def test_rejection_restores_exploration_and_keeps_old_observation_refs(self):
        state = begin_target()
        old = copy.deepcopy(state.scan.views)
        state.accept(Decision(action='look', direction='left', reason='已否定旧线索，查看新方向',
                              target_review=target_review(2, 'rejected')))
        self.assertIsNone(state.target.active)
        self.assertEqual(state.scan.number, 1)
        self.assertIsNone(state.scan.last_outcome)
        self.assertEqual(old[0]['image'], 'images/001.jpg')

    def test_same_direction_after_translation_cannot_replace_original_frame(self):
        state = begin_target()
        self.assertEqual(state.scan.views[0]['image'], 'images/001.jpg')
        self.assertEqual(state.scan.views[1]['image'], 'images/002.jpg')
        self.assertNotEqual(state.scan.views[0]['id'], state.scan.views[1]['id'])
        state.observe(view(3, 2.52, yaw=70))
        self.assertNotIn('O1', state.context()['exploration_session']['orient_view_ids'])

    def test_new_candidate_name_keeps_target_identity_and_attempt_history(self):
        state = begin_target()
        state.target.active['confirmation_turns'] = 2
        first = state.target.active['candidate_id']
        state.current['id'] = 'P1'
        selected = state.accept(Decision.model_validate(navigation(review_step=2)))
        self.assertEqual(first, selected)
        state.navigation_started(selected)
        state.record_result(selected, {'reason': 'timeout', 'motion': {'displacement_m': .01}})
        self.assertEqual(state.target.active['confirmation_turns'], 2)
        self.assertEqual(state.allowed_look_directions(), ['left', 'right'])
        self.assertEqual(state.target.active['first_seen_step'], 1)

    def test_target_references_are_loaded_from_memory_and_attached_as_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'images').mkdir()
            settings = agent_settings()
            skill = ObjectSearchContext(settings, 75)
            memory = spatial_memory(root / 'memory.sqlite3', '走到机器狗旁边')
            first = memory.observe({'pose': {'x': 0, 'y': 0}, 'yaw': 0}, 'images/001.jpg', 1)
            skill.observe(first)
            decision = Decision.model_validate(navigation())
            selected = skill.accept(decision)
            memory.describe(first['id'], 1, decision)
            skill.before_execute(decision, selected)
            skill.record_result(decision, selected, {'reason': 'model_stop', 'motion': {'displacement_m': 2.52}}, 1)
            # 中间的无关报告不能挤掉目标证据，也不依赖中文目标匹配英文名称。
            for step in range(2, 8):
                observation = {'pose': {'x': 2.52, 'y': 0}, 'yaw': 0}
                current = memory.observe(observation, f'images/{step:03d}.jpg', step)
                if step < 7:
                    memory.describe(current['id'], step, Decision(action='look', direction='right',
                                    reason='记录观察', scene={'target_seen': 'no'}))
                (root / f'images/{step:03d}.jpg').write_bytes(b'image')
            (root / 'images/001.jpg').write_bytes(b'target')
            skill.observe(current)
            request = TaskRequest(goal='走到机器狗旁边', navigation_seconds=15.)
            context = build_context(request, observation, memory, current, skill, settings)
            evidence = context['execution']['target_tracking']['evidence_frames']
            self.assertEqual([item['step'] for item in evidence], [1])
            manifest = image_manifest(
                root, current, context['execution']['exploration_session']['views'],
                skill.state.last_navigation, memory_frames=context['spatial_memory']['semantic_frames'],
                target_frames=evidence,
            )
            self.assertEqual([item['image'] for item in manifest], ['images/001.jpg', 'images/007.jpg'])
            self.assertNotIn('scene', skill.state.scan.views[0])
            self.assertEqual(memory.frames([1])[0]['scene']['target_seen'], 'yes')


class TargetLoopTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_loop.LoopTests.asyncSetUp
    asyncTearDown = test_loop.LoopTests.asyncTearDown
    run_loop = test_loop.LoopTests.run_loop

    async def test_recorded_robot_approach_then_partial_view_confirmation(self):
        # 2026-09-23 日志第 4—6 步：导航 2.52m，桌柜画面，再见右边近距离局部。
        poses = iter([
            (1.369142, .314008, -125.687),
            (-.64, -.96, -125.493),
            (-.64, -.96, -195.5),
        ])

        async def observe():
            x, y, heading = next(poses)
            return b'frame', {'pose': {'x': x, 'y': y}, 'yaw': math.radians(heading)}

        async def navigate(instruction, *, seconds):
            self.tools.actions.append(('navigate', instruction))
            return {'reason': 'model_stop', 'motion': {'displacement_m': 2.52}, 'arrival_verified': False}

        self.tools.observe = observe
        self.tools.navigate = navigate
        self.responses = [
            navigation('Walk straight toward the silver quadruped robot on the right and stop.'),
            {'action': 'look', 'direction': 'right', 'reason': '在附近确认目标',
             'scene': {'target_seen': 'no'}, 'target_review': target_review(2, 'temporarily_lost')},
            {'action': 'finish', 'reason': '已在机器狗旁边', 'scene': {'target_seen': 'yes'},
             'evidence': '当前右边缘出现近距离机身和关节',
             'target_review': {'observation_step': 3, 'status': 'arrived',
                               'evidence': '当前局部与原目标一致，已在旁边'}},
        ]
        record = await self.run_loop(steps=3)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual([action[0] for action in self.tools.actions], ['navigate', 'look'])
        last = json.loads(self.requests[-1]['messages'][1]['content'][0]['text'])
        self.assertEqual(last['execution']['target_tracking']['first_seen_step'], 1)
        self.assertEqual(last['execution']['scan']['allowed_look_directions'], ['left', 'right'])
        metadata = [json.loads(item['text']) for item in self.requests[-1]['messages'][1]['content'][1:]
                    if item['type'] == 'text']
        self.assertIn('images/001.jpg', [item['image'] for item in metadata])
        self.assertIn('近处物体可能只露部分部件', self.requests[-1]['messages'][0]['content'])

    async def test_old_target_claim_cannot_repeat_navigation_without_review(self):
        self.responses = [
            navigation(),
            navigation(),
            {'action': 'give_up', 'reason': '当前证据不足，停止等待人工确认'},
        ]
        record = await self.run_loop(steps=2)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual([action[0] for action in self.tools.actions], ['navigate'])
        self.assertIn('target_review', self.requests[-1]['messages'][-1]['content'])
