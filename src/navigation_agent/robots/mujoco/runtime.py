"""按现实时间补齐固定物理步，避免慢渲染饿死物理更新。"""

import time

import mujoco
import numpy as np
from vln_mujoco.model import CAMERA_HEIGHT, CAMERA_WIDTH, PHYSICS_DT
from vln_mujoco.simulation import FRAME_PERIOD_S, Simulation


def due_steps(now, next_step):
    """最多补齐 250ms，长时间暂停后不回放无限积压的运动。"""
    if now < next_step:
        return 0, next_step
    next_step = max(next_step, now - 0.25)
    count = int((now - next_step) / PHYSICS_DT) + 1
    return count, next_step + count * PHYSICS_DT


class RealtimeSimulation(Simulation):
    def _run(self):
        renderer = mujoco.Renderer(
            self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
        )
        third_person_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(third_person_camera)
        next_step = time.monotonic()
        next_frame = next_step
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                count, next_step = due_steps(now, next_step)
                if not count:
                    self._stop.wait(min(next_step - now, 0.002))
                    continue

                # 每次渲染前补齐物理步，保留每步碰撞检查与命令超时保护。
                for _ in range(count):
                    if self._stop.is_set():
                        break
                    with self._lock:
                        self._advance(time.monotonic())

                if now >= next_frame and not self._stop.is_set():
                    # 两个视角取自同一物理状态；沿用官方帧格式和真实采图位姿。
                    with self._lock:
                        renderer.update_scene(self.data, camera=self.robot.camera_name)
                        rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
                        camera = self.robot.third_person_camera(third_person_camera)
                        renderer.update_scene(self.data, camera=camera)
                        third_rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
                        x, y, _, yaw = self.robot.state().pose
                        pose = (x, y, yaw)
                    stamp_ns = time.time_ns()
                    frame = self._encode_frame(stamp_ns, rgb, pose)
                    third_frame = self._encode_frame(stamp_ns, third_rgb, pose)
                    with self._lock:
                        self._frame = frame
                        self._third_person_frame = third_frame
                        self._frame_count += 1
                    next_frame = now + FRAME_PERIOD_S
        finally:
            renderer.close()
