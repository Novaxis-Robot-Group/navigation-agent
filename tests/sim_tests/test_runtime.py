"""验证渲染变慢时物理时间仍跟随现实时间。"""

import unittest

from navigation_agent.robots.mujoco.runtime import due_steps
from vln_mujoco.model import PHYSICS_DT


class RuntimeTests(unittest.TestCase):
    def test_slow_frames_do_not_starve_physics(self):
        next_step = 0
        steps = 0
        for frame in range(101):
            now = frame * 0.08
            count, next_step = due_steps(now, next_step)
            steps += count
        self.assertAlmostEqual(steps * PHYSICS_DT, 8, delta=PHYSICS_DT * 2)

    def test_future_step_waits(self):
        self.assertEqual(due_steps(0, 0.005), (0, 0.005))

    def test_long_pause_has_bounded_catchup(self):
        count, next_step = due_steps(60, 0)
        self.assertLessEqual(count * PHYSICS_DT, 0.255)
        self.assertGreater(next_step, 60)


if __name__ == '__main__':
    unittest.main()
