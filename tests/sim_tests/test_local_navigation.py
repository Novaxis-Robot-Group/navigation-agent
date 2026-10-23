"""短程规划必须留足车身空间，且不能跨墙或扩大成全局导航。"""

import math
import unittest
import tempfile
from concurrent.futures import Future
from threading import RLock
from types import SimpleNamespace
from pathlib import Path
import numpy as np

from navigation_agent.robots.mujoco import local_navigation as navigation
from navigation_agent.robots.mujoco.mujoco_safe import SafeTurtleBot
from vln_mujoco.simulation import Simulation


class LocalRouteTests(unittest.TestCase):
    def test_off_center_doorway_has_safe_detour(self):
        # 已扣除车身半径的门口，直线会擦到右侧门框。
        def free(point):
            x, y = point
            return not (0.5 < x < 0.8 and not -0.30 < y < 0.15)

        route = navigation.local_route([0, 0.20], [1.2, 0.20], free)
        self.assertGreater(len(route), 2)
        for start, end in zip(route, route[1:]):
            self.assertTrue(navigation.segment_free(start, end, free))
        np.testing.assert_allclose(route[-1], [1.2, 0.20])

    def test_solid_wall_and_occupied_goal_stop(self):
        self.assertEqual(navigation.local_route([0, 0], [1, 0],
                         lambda point: not 0.4 < point[0] < 0.7), [])
        self.assertEqual(navigation.local_route([0, 0], [1, 0],
                         lambda point: point[0] < 0.9), [])

    def test_no_global_route_or_thin_wall_shortcut(self):
        self.assertEqual(navigation.local_route([0, 0], [3, 0], lambda point: True), [])
        self.assertFalse(navigation.segment_free(np.array([0, 0]), np.array([1, 0]),
                         lambda point: not 0.48 < point[0] < 0.52))

    def test_real_body_crosses_door_without_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'assets/scenes/test/scene.xml'
            scene.parent.mkdir(parents=True)
            scene.write_text('''<mujoco><worldbody>
              <geom name="floor" type="plane" size="20 20 .1"/>
              <geom name="lower_jamb" type="box" pos="7.5 12.6 .5" size=".04 .8 .5"/>
              <geom name="upper_jamb" type="box" pos="7.5 14.8 .5" size=".04 .9 .5"/>
            </worldbody></mujoco>''')
            robot = SafeTurtleBot(scene)
            tracker = navigation.LocalNavigationTracker(Simulation(robot))
            start = robot.state().pose
            trajectory = np.column_stack((np.linspace(start[0], 8.5, 30),
                                          np.full(30, start[1]), np.zeros(30)))
            command = (0, 0)
            try:
                for step in range(1800):
                    if step % 20 == 0:
                        x, y, _, yaw = robot.state().pose
                        result = tracker._solve_local(
                            1, np.array([x, y, yaw]), trajectory,
                            np.array([start[0], start[1], start[3]]), command,
                        )
                        command = result.command
                    robot.step(command)
                self.assertGreater(robot.state().pose[0], 8.25)
                self.assertEqual(robot.blocked_steps, 0)
                live_pose = robot.data.qpos.copy()
                tracker.collision_map.free([7.5, 14.0])
                np.testing.assert_array_equal(robot.data.qpos, live_pose)
                tracker.collision_map.refresh()
                self.assertGreater(len(tracker.collision_map.cache), 0)

                # 停止之后，旧一轮计算结果不能重新启动运动。
                tracker.reset()
                tracker._future = Future()
                tracker._future.set_result(navigation.MpcSolveResult(
                    -1, (0.35, 0), trajectory, trajectory, 1.0
                ))
                self.assertIsNone(tracker.poll())
                self.assertEqual(tracker.command, (0.0, 0.0))
                self.assertEqual(tracker.local_status, 'idle')
            finally:
                tracker.close()

    def test_scene_motion_invalidates_cached_clearance(self):
        collision_map = navigation.CollisionMap.__new__(navigation.CollisionMap)
        collision_map.robot = SimpleNamespace(_base_qpos=0, _left_qpos=7, _right_qpos=8)
        collision_map.simulation = SimpleNamespace(
            _lock=RLock(), data=SimpleNamespace(qpos=np.zeros(10))
        )
        collision_map.data = SimpleNamespace(qpos=np.zeros(10))
        collision_map.cache = {}
        collision_map.scene_pose = None
        collision_map.refresh()
        collision_map.cache[(1, 2)] = True
        collision_map.simulation.data.qpos[0] = 1
        collision_map.refresh()
        self.assertTrue(collision_map.cache[(1, 2)])
        collision_map.simulation.data.qpos[9] = 1
        collision_map.refresh()
        self.assertEqual(collision_map.cache, {})


class LocalTrackingTests(unittest.TestCase):
    def setUp(self):
        self.pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.tracker = navigation.LocalNavigationTracker.__new__(navigation.LocalNavigationTracker)
        navigation.MpcTracker.__init__(self.tracker)
        self.tracker.collision_map = SimpleNamespace(
            simulation=SimpleNamespace(snapshot=lambda: {'pose': self.pose.copy()}),
            refresh=lambda: None,
            free=lambda point: True,
        )
        self.tracker.local_status = 'idle'
        self.tracker._capture_pose = None
        self.addCleanup(self.tracker.close)

    def solve(self, pose, points, previous=(0.0, 0.0), capture=(0.0, 0.0, 0.0)):
        return self.tracker._solve_local(
            1, np.asarray(pose), np.asarray(points), np.asarray(capture), previous,
        )

    def test_old_view_after_turn_holds_until_fresh_prediction(self):
        self.tracker.command = self.tracker.previous_command = (0.3, 1.0)
        self.tracker._future = Future()
        self.tracker._future.set_result(navigation.MpcSolveResult(
            self.tracker._generation, (0.3, 1.0), np.ones((2, 3)), np.ones((2, 3)), 1.0,
        ))
        self.pose['yaw'] = math.radians(70)
        points = [(0, 0, 0), (1, 0.3, 0.2)]
        self.tracker.set_body_path(points, (0, 0, 0))
        self.assertEqual(self.tracker.command, (0.0, 0.0))
        self.assertEqual(self.tracker.local_status, 'waiting_for_fresh_path')
        self.assertIsNone(self.tracker._trajectory)
        self.assertEqual(len(self.tracker.reference), 0)
        self.assertIsNone(self.tracker.poll())

        # 新画面的模型输出仍按原始采图位姿转换，不改写模型路线。
        capture = (0, 0, self.pose['yaw'])
        self.tracker.set_body_path(points, capture)
        expected = np.array([
            [0, 0, capture[2]],
            [math.cos(capture[2]) - .3 * math.sin(capture[2]),
             math.sin(capture[2]) + .3 * math.cos(capture[2]), capture[2] + .2],
        ])
        np.testing.assert_allclose(self.tracker._trajectory, expected)
        self.tracker.submit(capture)
        self.tracker._future.result(timeout=2)
        self.assertGreater(self.tracker.poll()[0], 0)
        self.assertEqual(self.tracker.local_status, 'tracking')

    def test_heading_wrap_and_intended_large_turn_are_accepted(self):
        self.pose['yaw'] = math.radians(-179)
        capture = (0, 0, math.radians(179))
        self.tracker.set_body_path([(0, 0, math.pi / 2)], capture)
        self.assertIsNotNone(self.tracker._trajectory)
        self.tracker.submit((0, 0, self.pose['yaw']))
        self.tracker._future.result(timeout=2)
        self.assertGreater(self.tracker.poll()[1], 0)

    def test_position_reached_keeps_remaining_heading(self):
        result = self.solve([.3, 0, 0], [
            [.15, 0, -.5], [.3, 0, -1], [.3, 0, -math.pi / 2],
        ])
        self.assertEqual(result.command[0], 0)
        self.assertLess(result.command[1], 0)

    def test_corner_and_model_headings_remain_in_reference(self):
        points = [[.15, 0, 0], [.3, 0, 0], [.3, .15, math.pi / 2],
                  [.3, .3, math.pi / 2]]
        result = self.solve([0, 0, 0], points)
        self.assertGreater(result.command[0], 0)
        self.assertEqual(result.command[1], 0)
        np.testing.assert_allclose(result.reference[1:], points)

    def test_initial_rotation_is_completed_before_translation(self):
        result = self.solve([0, 0, 0], [
            [0, 0, .5], [0, 0, 1.0], [.1, .2, 1.0],
        ])
        self.assertEqual(result.command[0], 0)
        self.assertGreater(result.command[1], 0)

    def test_last_in_place_turn_is_not_skipped_before_translation(self):
        result = self.solve([0, 0, .5], [
            [0, 0, .5], [0, 0, 1.0], [.1, .2, 1.0],
        ])
        self.assertEqual(result.command[0], 0)
        self.assertGreater(result.command[1], 0)

    def test_repeated_forward_endpoint_is_translation(self):
        result = self.solve([0, 0, 0], [[.3, 0, 0]] * 10)
        self.assertGreater(result.command[0], 0)
        self.assertAlmostEqual(result.command[1], 0)

    def test_finished_path_does_not_turn_back_to_old_endpoint(self):
        paths = [[[0, 0, 0], [.5, 0, 0], [1, 0, 0]], [[1, 0, 0]] * 10]
        for points in paths:
            for pose in ([1.02, 0, 0], [1.3, 0, 0]):
                with self.subTest(pose=pose, points=points):
                    result = self.solve(pose, points, (.3, 1))
                    self.assertEqual(result.command, (0.0, 0.0))

    def test_turn_reversal_is_limited_and_reset_stops_immediately(self):
        result = self.solve([0, 0, 0], [[0, 0, 0], [1, -1, -.7]], (0, 1))
        self.assertGreater(result.command[1], 0)
        self.assertLess(result.command[1], 1)
        self.assertLessEqual(
            1 - result.command[1],
            navigation.ANGULAR_ACCELERATION / navigation.CONTROL_RATE_HZ + 1e-9,
        )
        self.assertEqual(result.command[0], 0)
        self.tracker.command = result.command
        self.tracker.reset()
        self.assertEqual(self.tracker.command, (0.0, 0.0))
        self.assertIsNone(self.tracker._capture_pose)

    def test_collision_is_checked_after_turn_smoothing(self):
        # 目标路线向右安全，但残余左转会进入上方障碍，必须暂停平移。
        self.tracker.collision_map.free = lambda point: point[1] < .001
        result = self.solve([0, 0, 0], [[0, 0, 0], [1, 0, 0]], (.35, 1))
        self.assertEqual(result.command[0], 0)

    def test_blocked_route_stops_without_acceleration_delay(self):
        self.tracker.collision_map.free = lambda point: point[0] < .2
        result = self.solve([0, 0, 0], [[0, 0, 0], [1, 0, 0]], (.35, 1))
        self.assertEqual(result.command, (0.0, 0.0))

    def test_fixed_turn_converges_with_limited_acceleration(self):
        pose = np.zeros(3)
        command = (0.0, 0.0)
        total_turn = 0.0
        for _ in range(80):
            result = self.solve(pose, [[0, 0, math.pi / 2]], command)
            self.assertEqual(result.command[0], 0)
            self.assertLessEqual(
                abs(result.command[1] - command[1]),
                navigation.ANGULAR_ACCELERATION / navigation.CONTROL_RATE_HZ + 1e-9,
            )
            command = result.command
            turn = command[1] / navigation.CONTROL_RATE_HZ
            pose[2] += turn
            total_turn += abs(turn)
        self.assertAlmostEqual(pose[2], math.pi / 2, delta=.02)
        self.assertLess(total_turn, math.pi / 2 + .1)
