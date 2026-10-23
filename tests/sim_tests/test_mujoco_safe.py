"""检查真实 MuJoCo 接触：薄墙、家具、倒退和无效指令。"""

import math
import asyncio
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from navigation_agent.robots.mujoco.mujoco_safe import SafeTurtleBot
from tests.backend_support import backend_class

SimTools = backend_class()


class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        scene = Path(cls.directory.name) / 'assets/scenes/test/scene.xml'
        scene.parent.mkdir(parents=True)
        scene.write_text('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="20 20 .1"/>
          <geom name="thin_wall" type="box" pos="7.5 13.8 .5" size=".0005 2 .5"/>
          <geom name="furniture" type="box" pos="5.5 13.8 .17" size=".15 .5 .04"/>
        </worldbody></mujoco>''')
        cls.robot = SafeTurtleBot(scene)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.robot.reset()

    def assert_no_penetration(self):
        self.assertTrue(
            all(c.dist >= 0 for c in self.robot.data.contact if self.robot.envelope in c.geom)
        )

    def test_configured_camera_height_preserves_body_and_reset(self):
        scene = Path(self.directory.name) / 'assets/scenes/test/scene.xml'
        for height in (.198, .8):
            robot = SafeTurtleBot(scene, camera_height_m=height)
            camera = robot.model.camera('robot_rgb').id
            self.assertAlmostEqual(robot.data.cam_xpos[camera, 2], height)
            np.testing.assert_allclose(robot.model.geom_size[robot.envelope],
                                       self.robot.model.geom_size[self.robot.envelope])
            robot.reset()
            self.assertAlmostEqual(robot.data.cam_xpos[camera, 2], height)
        for height in (math.nan, math.inf, 0, 3):
            with self.assertRaises(ValueError):
                SafeTurtleBot(scene, camera_height_m=height)

    def test_thin_wall_repeated_maximum_forward(self):
        for _ in range(700):
            self.robot.step((1000, 0))
            self.assert_no_penetration()
        self.assertTrue(self.robot.blocked)
        self.assertLess(self.robot.state().pose[0], 7.39)
        self.assertEqual(self.robot.state().velocity[0], 0)
        position = self.robot.state().pose[0]
        for _ in range(100):
            self.robot.step((1.5, 0))
        self.assertEqual(self.robot.state().pose[0], position)
        for _ in range(40):
            self.robot.step((-1, 0))
        self.assertLess(self.robot.state().pose[0], position - 0.1)

    def test_camera_height_furniture_blocks_reverse(self):
        for _ in range(400):
            self.robot.step((-1.5, 0))
            self.assert_no_penetration()
        self.assertTrue(self.robot.blocked)
        self.assertIn('furniture', self.robot.last_obstacle)
        self.assertGreater(self.robot.state().pose[0], 5.75)

    def test_turn_can_escape_after_block(self):
        for _ in range(400):
            self.robot.step((1.5, 0))
        position = np.array(self.robot.state().pose[:2])
        for _ in range(100):
            self.robot.step((0, 2))
        np.testing.assert_allclose(self.robot.state().pose[:2], position)
        self.assertAlmostEqual(self.robot.state().pose[3], 1)
        for _ in range(40):
            self.robot.step((-1, 0))
        self.assertGreater(np.linalg.norm(np.array(self.robot.state().pose[:2]) - position), 0.1)

    def test_detector_positive_control(self):
        self.robot.data.qpos[self.robot._base_qpos] = 7.5
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.assertTrue(self.robot._collision())
        self.assertTrue(
            any(c.dist < 0 for c in self.robot.data.contact if self.robot.envelope in c.geom)
        )

    def test_nonfinite_is_stopped(self):
        start = self.robot.state().pose
        for command in [(math.nan, 0), (1, math.inf), (math.inf, 0)]:
            self.robot.step(command)
        self.assertEqual(self.robot.state().pose, start)
        self.assertEqual(self.robot.state().velocity, (0, 0))

    def test_recover_retraces_path_after_real_wall_collision(self):
        async def run():
            client = SimTools('http://unused', 'ws://unused').robot

            async def snapshot():
                x, y, z, yaw = self.robot.state().pose
                client.state = {
                    'control': {'manual': True},
                    'simulation': {
                        'pose': dict(x=x, y=y, z=z, yaw=yaw),
                        'backend': {'collision_guard': True, 'blocked': self.robot.blocked},
                    },
                }

            async def hold():
                self.robot.step((0, 0))
                await snapshot()

            class Socket:
                async def send_json(_, command):
                    self.assertLessEqual(abs(command['linear']), 0.12)
                    for _ in range(20):
                        self.robot.step((command['linear'], command['angular']))
                        self.assert_no_penetration()

            client.channel.send = Socket().send_json
            client.hold = hold
            client._snapshot = snapshot
            await snapshot()
            client.remember_pose()
            for _ in range(700):
                self.robot.step((0.5, 0))
                await snapshot()
                client.remember_pose()
            self.assertTrue(self.robot.blocked)
            start = self.robot.state().pose[0]
            result = await client.recover()
            self.assertEqual(result['reason'], 'recovered')
            self.assertGreater(start - self.robot.state().pose[0], 0.5)
            self.assertEqual(self.robot.state().velocity, (0, 0))

        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
