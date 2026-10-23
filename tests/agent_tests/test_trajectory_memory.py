"""轨迹记录、有限上下文和重复尝试提示的离线回归。"""

import math
from pathlib import Path
import tempfile
import unittest

from navigation_agent.agent.memory.trajectory import TrajectoryMemory
from tests.config_support import spatial_memory


class TrajectoryMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.spatial = spatial_memory(Path(self.temp.name) / 'memory.sqlite3', '找到沙发')
        self.memory = TrajectoryMemory(self.spatial)

    def observe(self, step, x=0, yaw=0, valid=True, *, y=0):
        view = self.spatial.observe(
            {'pose': {'x': x, 'y': y}, 'yaw': math.radians(yaw), 'pose_valid': valid},
            f'images/{step:03d}.jpg', step,
        )
        self.memory.observe(step)
        return view

    def attempt(self, step, *, x=0, yaw=0, valid=True, outcome='no_new_information',
                origin='C1', intent='检查左侧门洞', reason='model_stop'):
        self.observe(step, x, yaw, valid)
        self.memory.start(step, 'navigate', intent, {'instruction': 'Walk toward the doorway.'},
                          {'id': f'C{step}', 'origin_candidate_id': origin})
        self.memory.result(step, {'reason': reason})
        self.observe(step + 1, x, yaw, valid)
        if outcome:
            self.memory.review(step + 1, {'outcome': outcome, 'evidence': '门洞景象未改变'})

    def test_action_uses_actual_observation_coordinates_and_separate_review(self):
        self.observe(1, x=.2)
        self.memory.start(1, 'navigate', '接近沙发', {'instruction': 'Approach the sofa.'})
        self.memory.result(1, {'reason': 'model_stop', 'motion': {'displacement_m': 2.52}})
        self.observe(2, x=2.72)
        before_review = self.memory.query(2)['recent'][0]
        self.assertIn('到达未确认', before_review)
        self.assertIn('尚未视觉复核', before_review)
        self.memory.review(2, {'outcome': 'uncertain', 'evidence': '右边缘仅见局部'},
                           {'status': 'temporarily_lost', 'evidence': '当前无法确认身份'})
        entry = self.memory._read(3)[0]
        self.assertEqual(entry['start']['observer_position'], [.2, 0])
        self.assertEqual(entry['end']['observer_position'], [2.72, 0])
        self.assertEqual(entry['result']['reason'], 'model_stop')
        summary = self.memory.query(3)['recent'][0]
        self.assertIn('执行净位移2.52m', summary)
        self.assertIn('进展不确定', summary)
        self.assertIn('目标暂时不可见', summary)
        self.assertNotIn('当前证据支持完成', summary)

    def test_pending_action_and_last_step_never_invent_result_or_end(self):
        self.observe(1)
        self.memory.start(1, 'look', '检查左侧', {'direction': 'left'})
        self.assertIn('执行结果未返回', self.memory.query(2)['recent'][0])
        self.memory.result(1, {'reason': 'turned'})
        self.assertIn('待观察', self.memory.query(2)['recent'][0])
        self.assertIsNone(self.memory._read(2)[0]['end'])

    def test_all_tools_retained_and_recent_context_is_bounded(self):
        actions = [('look', 'turned'), ('orient', 'oriented'), ('recover', 'recovered')]
        for step in range(1, 13):
            self.observe(step)
            action, reason = actions[(step - 1) % len(actions)]
            self.memory.start(step, action, '查看周围', {})
            self.memory.result(step, {'reason': reason})
        context = self.memory.query(13)
        self.assertEqual(len(context['recent']), 5)
        self.assertTrue(context['recent'][0].startswith('8→9'))
        with self.spatial.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM trajectory_actions').fetchone()[0], 12)
        self.assertEqual(context['repeated_attempts'], [])
        self.assertNotIn('images/', str(context))

    def test_old_repetition_survives_recent_window_and_reopen(self):
        self.attempt(1)
        self.attempt(3)
        for step in range(5, 12):
            self.observe(step)
            self.memory.start(step, 'look', '继续观察', {'direction': 'left'})
            self.memory.result(step, {'reason': 'turned'})
        reopened = TrajectoryMemory(self.spatial)
        context = reopened.query(12)
        self.assertEqual(len(context['recent']), 5)
        self.assertEqual(len(context['repeated_attempts']), 1)
        self.assertIn('2次模型报告无新信息', context['repeated_attempts'][0])

    def test_progress_clears_old_hint_and_new_evidence_starts_fresh(self):
        self.attempt(1)
        self.attempt(3)
        self.assertEqual(len(self.memory.query(5)['repeated_attempts']), 1)
        self.attempt(5, outcome='progress')
        self.attempt(7)
        self.assertEqual(self.memory.query(9)['repeated_attempts'], [])

    def test_local_confirmation_can_resolve_previous_navigation_hint(self):
        self.attempt(1)
        self.attempt(3)
        self.assertEqual(len(self.memory.query(5)['repeated_attempts']), 1)
        self.observe(5)
        self.memory.start(5, 'look', '确认目标身份', {'direction': 'right'})
        self.memory.result(5, {'reason': 'turned'})
        self.observe(6, yaw=70)
        self.memory.review(6, target_review={'status': 'rejected', 'evidence': '局部图表明是椅子'})
        self.assertEqual(self.memory.query(7)['repeated_attempts'], [])

    def test_normal_return_or_new_position_direction_and_invalid_pose_not_flagged(self):
        cases = [
            {'x': 5}, {'yaw': 100}, {'valid': False},
            {'origin': 'C2', 'intent': '查看右侧窗户'},
        ]
        for index, kwargs in enumerate(cases):
            with self.subTest(kwargs=kwargs):
                spatial = spatial_memory(Path(self.temp.name) / f'case{index}.sqlite3', '找到沙发')
                self.spatial, self.memory = spatial, TrajectoryMemory(spatial)
                self.attempt(1)
                self.attempt(3, **kwargs)
                self.assertEqual(self.memory.query(5)['repeated_attempts'], [])
        self.spatial = spatial_memory(Path(self.temp.name) / 'progress.sqlite3', '找到沙发')
        self.memory = TrajectoryMemory(self.spatial)
        self.attempt(1, outcome='progress')
        self.attempt(3, outcome='progress')
        self.assertEqual(self.memory.query(5)['repeated_attempts'], [])

    def test_failure_without_visual_review_can_hint_but_service_start_failure_cannot(self):
        self.attempt(1, outcome=None, reason='blocked')
        self.attempt(3, outcome=None, reason='timeout')
        hint = self.memory.query(5)['repeated_attempts'][0]
        self.assertIn('0次模型报告无新信息', hint)
        self.assertIn('2次执行受阻/超时/错误', hint)
        for step in (1, 3):
            self.memory.result(step, {'reason': 'navigation_error', 'startup_failed': True})
        context = self.memory.query(5)
        self.assertEqual(context['repeated_attempts'], [])
        self.assertIn('启动失败，未执行接近', context['recent'][0])

    def test_stopped_review_selects_two_actual_intermediate_navigation_frames(self):
        self.observe(1, x=0)
        self.memory.start(1, 'navigate', '通过入口', {})
        for step in range(2, 7):
            self.spatial.observe({'pose': {'x': step - 1, 'y': 0}, 'yaw': 0, 'pose_valid': True},
                                 f'images/{step:03d}.jpg', step)
        self.assertEqual(self.memory.navigation_frames(6), [])
        self.memory.result(1, {'reason': 'model_stop'})
        self.observe(7, x=6)
        references = self.memory.navigation_frames(7)
        self.assertEqual([frame['step'] for frame in references], [3, 5])
        self.assertEqual([frame['image'] for frame in references], ['images/003.jpg', 'images/005.jpg'])
        self.assertTrue(all(frame['scene'] is None for frame in references))
        self.memory.result(1, {'reason': 'navigation_error', 'startup_failed': True})
        self.assertEqual(self.memory.navigation_frames(7), [])

    def test_stationary_navigation_has_no_intermediate_room_evidence(self):
        self.observe(1)
        self.memory.start(1, 'navigate', '接近入口', {})
        self.spatial.observe({'pose': {'x': 0, 'y': 0}, 'yaw': 1., 'pose_valid': True},
                             'images/002.jpg', 2)
        self.memory.result(1, {'reason': 'no_progress'})
        self.observe(3)
        self.assertEqual(self.memory.navigation_frames(3), [])

    def test_return_to_an_old_viewpoint_is_a_hint_without_forbidding_transit(self):
        self.observe(1, x=0)
        self.memory.start(1, 'navigate', '前往下一间房', {})
        self.memory.result(1, {'reason': 'model_stop'})
        self.observe(2, x=3)
        self.memory.start(2, 'navigate', '途经原房间去另一入口', {})
        self.memory.result(2, {'reason': 'model_stop'})
        self.observe(3, x=0)
        context = self.memory.query(4)
        self.assertEqual(context['repeated_attempts'], [])
        self.assertEqual(len(context['returned_places']), 1)
        self.assertIn('O3 回到 O1', context['returned_places'][0])
        self.assertIn('不禁止正常途经', context['returned_places'][0])
        self.memory.start(3, 'look', '确认出口方向', {})
        self.memory.result(3, {'reason': 'turned'})
        self.observe(4, x=0)
        self.assertEqual(len(self.memory.query(5)['returned_places']), 1)

    def test_return_recalls_passed_positions_across_place_id_boundaries(self):
        self.observe(1, x=0)
        self.observe(2, x=1)
        passed = self.spatial.observe({'pose': {'x': .4, 'y': 0}, 'yaw': 0},
                                      'images/003.jpg', 3)
        self.observe(4, x=3)
        self.memory.start(4, 'navigate', '返回先前经过的入口', {})
        self.memory.result(4, {'reason': 'model_stop'})
        end = self.observe(5, x=.6)
        self.assertNotEqual(end['id'], passed['id'])
        hints = self.memory.query(6)['returned_places']
        self.assertEqual(len(hints), 1)
        self.assertIn('O5 回到 O3', hints[0])
        self.assertIn('不代表已经搜完', hints[0])

    def test_long_detour_is_reported_without_overriding_model_stop(self):
        self.observe(1)
        self.memory.start(1, 'navigate', '检查入口', {})
        self.memory.result(1, {'reason': 'model_stop',
                              'motion': {'path_length_m': 10.3, 'displacement_m': .79}})
        self.observe(2, x=.79)
        summary = self.memory.query(3)['recent'][0]
        self.assertIn('实际行程10.30m', summary)
        self.assertIn('有绕行或折返', summary)
        self.assertIn('底层停止，到达未确认', summary)
        self.memory.result(1, {'reason': 'model_stop',
                              'motion': {'path_length_m': 4., 'displacement_m': 3.8}})
        self.assertNotIn('有绕行或折返', self.memory.query(3)['recent'][0])

    def test_invalid_pose_is_unknown_and_new_task_is_empty(self):
        self.attempt(1, valid=False)
        entry = self.memory._read(3)[0]
        self.assertIsNone(entry['start']['observer_position'])
        self.assertIsNone(entry['end']['observer_position'])
        self.assertNotIn('观测位移', self.memory.query(3)['recent'][0])
        other = TrajectoryMemory(spatial_memory(Path(self.temp.name) / 'other.sqlite3', '找到沙发'))
        self.assertEqual(other.query(1)['recent'], [])

    def test_configured_item_limits_and_reference_summary_lengths(self):
        self.memory.settings = self.memory.settings.model_copy(update={
            'max_trajectory_actions': 2, 'max_trajectory_hints': 1,
        })
        for step, origin in ((1, 'A'), (3, 'A'), (5, 'B'), (7, 'B')):
            self.attempt(step, origin=origin, intent=origin * 1000)
        context = self.memory.query(9)
        self.assertEqual(len(context['recent']), 2)
        self.assertEqual(len(context['repeated_attempts']), 1)
        self.assertIn('B' * 64 + '…', context['recent'][0])
        self.assertIn('B' * 40 + '…', context['repeated_attempts'][0])
        self.assertNotIn('A' * 1000, str(context['recent']))
        self.assertNotIn('xy', context)


    def test_recent_and_local_path_summaries_match_reference_truncation(self):
        intent = '检查入口及门框后方的空间。' * 8 + '不要进入右侧已查房间。'
        evidence = '当前看到墙体和另一扇门。' * 8 + '未达成进入左侧房间的目标。'
        target = '画面中有同色的家具局部。' * 8 + '实际是椅子，不是沙发。'
        self.recorded_path([[i / 4, 0.] for i in range(9)])
        with self.spatial.connect() as db:
            db.execute('UPDATE trajectory_actions SET intent=? WHERE action_step=1', (intent,))
        self.memory.review(2, {'outcome': 'no_new_information', 'evidence': evidence},
                           {'status': 'rejected', 'evidence': target})
        context = self.memory.query(2)
        for value in (intent, evidence, target):
            self.assertIn(value[:64] + '…', context['recent'][0])
        segment = context['local_paths']['segments'][0]
        self.assertEqual(segment['intent'], intent[:64] + '…')
        self.assertEqual(segment['visual_review']['evidence'], evidence[:64] + '…')
        stored = self.memory._read(2)[0]
        self.assertEqual(stored['intent'], intent)
        self.assertEqual(stored['review']['action']['evidence'], evidence)


    def recorded_path(self, points, *, heading=0):
        self.observe(1, x=points[0][0], y=points[0][1])
        self.memory.start(1, 'navigate', '检查门后的通道', {}, {'id': 'C1'})
        self.memory.result(1, {'reason': 'model_stop', 'motion': {'sampled_positions': points}})
        self.observe(2, x=points[-1][0], y=points[-1][1], yaw=heading)

    def test_path_links_stop_image_heading_review_and_reorients_after_turning(self):
        points = [[i / 4, 0.] for i in range(9)]
        self.recorded_path(points, heading=90)
        self.memory.review(2, {'outcome': 'progress', 'evidence': '门后可见走廊'})
        self.memory = TrajectoryMemory(self.spatial)
        entry = self.memory._read(2)[0]
        self.assertEqual(entry['result']['motion']['sampled_positions'], points)
        self.assertEqual(entry['end']['image'], 'images/002.jpg')
        self.assertEqual(entry['end']['heading_deg'], 90)
        segment = self.memory.query(2)['local_paths']['segments'][0]
        self.assertEqual(segment['start_observation'], 'O1')
        self.assertEqual(segment['end_observation'], 'O2')
        self.assertEqual(segment['origin_candidate_id'], 'C1')
        self.assertEqual(segment['visual_review']['outcome'], 'progress')
        self.assertEqual(segment['directions'][0]['relative_bearing_deg'], 90)
        self.assertEqual(segment['directions'][0]['direction'], '左侧')
        for step, heading, expected in ((3, 180, 0), (4, 270, -90), (5, 0, -180)):
            self.memory.start(step - 1, 'look', '查看侧面', {'direction': 'left'})
            self.memory.result(step - 1, {'reason': 'turned'})
            self.observe(step, x=2, yaw=heading)
            context = self.memory.query(step)['local_paths']
            self.assertEqual(context['observation_id'], f'O{step}')
            segment = context['segments'][0]
            self.assertEqual(segment['action_step'], 1)
            self.assertEqual(segment['stop_heading_deg'], 90)
            self.assertEqual(segment['directions'][0]['relative_bearing_deg'], expected)

    def test_path_recall_uses_all_history_and_both_sides_at_an_old_midpoint(self):
        self.recorded_path([[i / 4, 0.] for i in range(17)])
        self.memory.review(2, {'outcome': 'progress', 'evidence': '获得新观察'})
        for step in range(2, 10):
            self.memory.start(step, 'look', '检查附近', {})
            self.memory.result(step, {'reason': 'turned'})
            self.observe(step + 1, x=2)
        context = self.memory.query(10)
        self.assertTrue(all(not item.startswith('1→') for item in context['recent']))
        segment = context['local_paths']['segments'][0]
        self.assertEqual(segment['action_step'], 1)
        directions = {item['order']: item['relative_bearing_deg'] for item in segment['directions']}
        self.assertEqual(directions, {'toward_start': -180, 'toward_end': 0})
        self.assertNotIn('sampled_positions', str(context))
        self.observe(11, x=8)
        self.assertNotIn('local_paths', self.memory.query(11))

    def test_path_direction_follows_local_curve_instead_of_bearing_to_old_start(self):
        points = [[i / 4, 0.] for i in range(9)]
        points += [[2., i / 4] for i in range(1, 9)]
        self.recorded_path(points)
        direction = self.memory.query(2)['local_paths']['segments'][0]['directions'][0]
        self.assertEqual(direction['relative_bearing_deg'], -90)
        self.assertEqual(direction['direction'], '右侧')

    def test_missing_path_pose_or_result_does_not_invent_a_direction(self):
        self.recorded_path([[i / 4, 0.] for i in range(9)])
        for motion in (
            {}, {'sampled_positions': [[0., 0.], [2., 0.]]},
            {'sampled_positions': [[0., 0.], None, [1.75, 0.], [2., 0.]]},
            {'sampled_positions': [[float('nan'), 0.], [True, 0.], [2.]]},
        ):
            with self.subTest(motion=motion):
                self.memory.result(1, {'reason': 'model_stop', 'motion': motion})
                self.assertNotIn('local_paths', self.memory.query(2))
        self.memory.result(1, {
            'reason': 'navigation_error', 'startup_failed': True,
            'motion': {'sampled_positions': [[i / 4, 0.] for i in range(9)]},
        })
        self.assertNotIn('local_paths', self.memory.query(2))
        self.memory.result(1, {'reason': 'model_stop',
                              'motion': {'sampled_positions': [[i / 4, 0.] for i in range(9)]}})
        self.observe(3, x=2, valid=False)
        self.assertNotIn('local_paths', self.memory.query(3))
        self.observe(4, x=2, yaw=float('nan'))
        self.assertNotIn('local_paths', self.memory.query(4))
        self.assertNotIn('local_paths', self.memory.query(99))

    def test_nearby_segments_are_bounded_without_erasing_persisted_paths(self):
        self.recorded_path([[i / 4, 0.] for i in range(9)])
        points = [[i / 4, 0.] for i in range(9)]
        for step in range(2, 7):
            self.memory.start(step, 'navigate', '经过已走通道', {})
            self.memory.result(step, {'reason': 'model_stop', 'motion': {'sampled_positions': points}})
            self.observe(step + 1, x=2)
        self.memory.settings = self.memory.settings.model_copy(update={'max_trajectory_hints': 2})
        segments = self.memory.query(7)['local_paths']['segments']
        self.assertEqual([segment['action_step'] for segment in segments], [6, 5])
        self.assertEqual(len(self.memory._read(7)), 6)


if __name__ == '__main__':
    unittest.main()
