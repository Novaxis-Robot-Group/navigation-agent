"""服务侧图像、预测与运动闭环，只依赖公共模型和执行器契约。"""

import logging
import time

from navigation_agent.contracts.robot import MotionExecutor
from navigation_agent.contracts.vln import Prediction, VlnClient


class NavigationRuntime:
    def __init__(self, model: VlnClient, executor: MotionExecutor):
        self.model = model
        self.executor = executor
        self.paused = False
        self.last_frame_stamp = 0
        self.capture_poses = {}
        self.waypoints = ()
        self.last_prediction: Prediction | None = None
        self.command = (0.0, 0.0)
        self.next_control_s = time.monotonic()
        self.diagnostic = None
        self.on_prediction = None

    def reset(self):
        self.paused = False
        self.command = (0.0, 0.0)
        self.waypoints = ()
        self.last_prediction = None
        self.capture_poses.clear()
        self.executor.reset()
        self.executor.drive(self.command)

    def pause(self):
        self.model.pause()
        self.reset()
        self.paused = True

    def stop(self):
        self.reset()
        self.model.stop()

    def step(self, active: bool) -> bool:
        """执行一次原有控制节拍；返回模型是否发出了停止信号。"""
        frame = self.executor.frame()
        if not self.paused and frame is not None and frame.stamp_ns != self.last_frame_stamp:
            # 每次接受的帧都保留采图位姿，包含推理中帧和可替换的待发帧。
            if self.model.offer_frame(frame.stamp_ns, frame.jpeg):
                self.last_frame_stamp = frame.stamp_ns
                self.capture_poses[frame.stamp_ns] = frame.pose
                while (self.capture_poses and next(iter(self.capture_poses))
                       < frame.stamp_ns - 10_000_000_000):
                    self.capture_poses.pop(next(iter(self.capture_poses)))

        stopped = False
        for result in self.model.take_results():
            if self.paused:
                continue
            # 诊断只读取数据，不参与位姿转换或执行。
            self.diagnostic = {
                'sequence': result.sequence,
                'capture_stamp_ns': result.stamp_ns,
                'capture_pose': self.capture_poses.get(result.stamp_ns),
                'received_ns': time.time_ns(),
                'waypoints': [list(point) for point in result.waypoints],
                'stop': result.stop,
                'latency_ms': result.latency_ms,
                'history_elapsed_s': result.metadata.get('history_elapsed_s'),
                'frame_age_ms': (time.time_ns() - result.stamp_ns) / 1_000_000,
            }
            self.waypoints = result.waypoints
            self.last_prediction = result
            if result.stop is True:
                self.notify_prediction("model_stop")
                self.command = (0.0, 0.0)
                self.executor.drive(self.command)
                active = False
                stopped = True
                self.capture_poses.clear()
                self.executor.reset()
                self.model.stop()
                break
            capture_pose = self.capture_poses.pop(result.stamp_ns, None)
            if capture_pose is None:
                self.notify_prediction("missing_pose")
                self.executor.reset()
                self.executor.fail('missing robot pose for VLN capture timestamp')
                self.command = (0.0, 0.0)
                self.executor.drive(self.command)
                continue
            self.command = self.executor.accept_path(result.waypoints, capture_pose)
            self.notify_prediction("submitted")

        solved = self.executor.poll()
        if solved is not None:
            self.command = solved
            if active:
                self.executor.drive(self.command)

        now_s = time.monotonic()
        if active and now_s >= self.next_control_s:
            self.executor.submit()
            self.executor.drive(self.command)
            while self.next_control_s <= now_s:
                self.next_control_s += self.executor.control_period_s
        elif not active:
            self.next_control_s = now_s
        return stopped


    def notify_prediction(self, execution):
        """只旁路记录当前结果；日志故障不得改变控制节拍与模型会话。"""
        if self.on_prediction is None:
            return
        try:
            self.on_prediction({**self.diagnostic, 'execution': execution})
        except Exception:
            logging.getLogger(__name__).exception('预测日志回调失败')
