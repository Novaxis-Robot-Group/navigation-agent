"""替代模型和执行器验证公共闭环，不加载官方包、场景或 GPU。"""

import unittest

from navigation_agent.contracts.robot import CameraFrame
from navigation_agent.contracts.vln import Prediction
from navigation_agent.navigation.runtime import NavigationRuntime


class Model:
    def __init__(self):
        self.accepting = True
        self.offered = []
        self.results = []
        self.stopped = False

    def offer_frame(self, stamp_ns, jpeg):
        self.offered.append((stamp_ns, jpeg))
        return self.accepting

    def take_results(self):
        results, self.results = self.results, []
        return results

    def pause(self):
        self.accepting = False

    def stop(self):
        self.stopped = True


class Executor:
    control_period_s = 0.05

    def __init__(self):
        self.current_frame = CameraFrame(10, b'image', (1, 2, 0.3))
        self.paths = []
        self.commands = []
        self.resets = 0
        self.error = ''
        self.submitted = 0

    def frame(self):
        return self.current_frame

    def accept_path(self, waypoints, capture_pose):
        self.paths.append((waypoints, capture_pose))
        return (0.2, 0.0)

    def poll(self):
        return None

    def drive(self, command):
        self.commands.append(command)

    def reset(self):
        self.resets += 1

    def fail(self, message):
        self.error = message

    def submit(self):
        self.submitted += 1


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.executor = Executor()
        self.runtime = NavigationRuntime(self.model, self.executor)

    def test_prediction_uses_original_capture_pose_after_robot_moves(self):
        self.runtime.step(True)
        self.executor.current_frame = CameraFrame(20, b'later', (9, 9, 1.5))
        points = ((1.0, 0.0, 0.0),)
        self.model.results = [Prediction(1, 10, points, False, 300)]
        self.runtime.step(True)
        self.assertEqual(self.executor.paths, [(points, (1, 2, 0.3))])
        self.assertEqual(self.runtime.diagnostic['capture_pose'], (1, 2, 0.3))

    def test_latest_frame_replacements_keep_inflight_capture_pose(self):
        stamp = 1_000_000_000
        original_pose = (1, 2, .3)
        self.executor.current_frame = CameraFrame(stamp, b'inflight', original_pose)
        self.runtime.step(True)
        for index in range(1, 61):
            self.executor.current_frame = CameraFrame(
                stamp + index * 40_000_000, b'pending', (index, 0, 0),
            )
            self.runtime.step(True)
        self.model.results = [Prediction(1, stamp, ((1, 0, 0),), False, 2400)]
        self.runtime.step(True)
        self.assertEqual(self.executor.paths[-1][1], original_pose)
        self.assertEqual(self.executor.error, '')
        self.executor.current_frame = CameraFrame(stamp + 20_000_000_000, b'new', (0, 0, 0))
        self.runtime.step(True)
        self.assertEqual(len(self.runtime.capture_poses), 1)

    def test_rejected_frame_does_not_change_capture_history(self):
        self.runtime.step(False)
        self.model.accepting = False
        self.executor.current_frame = CameraFrame(20, b'later', (9, 9, 1.5))
        self.runtime.step(False)
        self.assertEqual(self.runtime.capture_poses, {10: (1, 2, 0.3)})
        self.assertEqual(self.runtime.last_frame_stamp, 10)

    def test_model_stop_stops_motion_and_retains_result_for_observer(self):
        self.model.results = [Prediction(1, 10, (), True, 300)]
        self.assertTrue(self.runtime.step(True))
        self.assertEqual(self.executor.commands[-1], (0.0, 0.0))
        self.assertEqual(self.executor.submitted, 0)
        self.assertEqual(self.runtime.capture_poses, {})
        self.assertTrue(self.model.stopped)
        self.assertTrue(self.runtime.last_prediction.stop)

    def test_missing_capture_pose_never_executes_prediction(self):
        self.model.results = [Prediction(1, 99, ((1, 0, 0),), False, 300)]
        self.runtime.step(True)
        self.assertFalse(self.executor.paths)
        self.assertIn('capture timestamp', self.executor.error)
        self.assertEqual(self.executor.commands[-1], (0.0, 0.0))

    def test_reset_invalidates_previous_capture_poses(self):
        self.runtime.step(True)
        self.runtime.stop()
        self.assertEqual(self.runtime.capture_poses, {})
        self.assertEqual(self.runtime.waypoints, ())
        self.assertIsNone(self.runtime.last_prediction)
        self.assertEqual(self.executor.commands[-1], (0.0, 0.0))



class PauseRuntimeTests(unittest.TestCase):
    def test_pause_stops_motion_frames_and_late_results(self):
        model, executor = Model(), Executor()
        runtime = NavigationRuntime(model, executor)
        runtime.step(True)
        runtime.pause()
        offered = list(model.offered)
        model.results = [Prediction(1, 10, ((1., 0., 0.),), True, 300)]
        self.assertFalse(runtime.step(False))
        self.assertEqual(model.offered, offered)
        self.assertEqual(executor.paths, [])
        self.assertEqual(executor.commands[-1], (0., 0.))
        self.assertFalse(model.stopped)
        runtime.reset()
        model.accepting = True
        executor.current_frame = CameraFrame(20, b'new', (1, 2, .3))
        runtime.step(True)
        self.assertEqual(model.offered[-1], (20, b'new'))
