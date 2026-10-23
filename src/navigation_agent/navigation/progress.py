"""用统一运动采样判断进展，不读取机器人原始协议。"""

import math

from navigation_agent.contracts.robot import MotionSample
from navigation_agent.contracts.runtime import ProtectionSettings


class MotionProgress:
    def __init__(self):
        self.settings = ProtectionSettings()
        self.samples: list[MotionSample] = []
        self.path_points = []
        self.path_distance = 0.
        self.path_repeated = False

    def reset(self):
        self.samples = []
        self.path_points = []
        self.path_distance = 0.
        self.path_repeated = False

    def record(self, sample: MotionSample):
        self.samples.append(sample)
        position = tuple(sample.pose.get(axis) for axis in ('x', 'y'))
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) for value in position):
            self.path_points = []
            self.path_distance = 0.
            return
        if self.path_points:
            distance = math.dist(position, self.path_points[-1]['position'])
            if distance < .2:
                return
            self.path_distance += distance
        # 同一指令下离开旧点至少 1.5m、行走至少 4m 后回到 0.35m 内，才视为闭环。
        for point in self.path_points:
            distance = math.dist(position, point['position'])
            point['departure'] = max(point['departure'], distance)
            if (self.path_distance - point['distance'] >= 4.
                    and point['departure'] >= 1.5 and distance <= .35):
                self.path_repeated = True
        self.path_points.append({'position': position, 'distance': self.path_distance, 'departure': 0.})

    def stalled(self):
        return _navigation_stalled(self.samples, self.settings)

    def inactive(self, seconds):
        """一段时间内既未平移也未转向；正常原地转弯仍算运动进展。"""
        if not self.samples:
            return False
        end = self.samples[-1].monotonic_s
        start_index = len(self.samples) - 1
        while start_index > 0 and end - self.samples[start_index].monotonic_s < seconds:
            start_index -= 1
        window = self.samples[start_index:]
        if end - window[0].monotonic_s < seconds:
            return False
        poses = [sample.pose for sample in window]
        if not all(all(axis in pose for axis in ('x', 'y', 'yaw')) for pose in poses):
            return False
        origin = poses[0]
        for pose in poses:
            distance = math.hypot(pose['x'] - origin['x'], pose['y'] - origin['y'])
            yaw_delta = pose['yaw'] - origin['yaw']
            heading_change = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
            if distance >= self.settings.stall_distance_m or heading_change >= math.radians(5):
                return False
        return True

    def summary(self):
        # 汇总实际运动，模型停止不代表已经到达目标
        samples = self.samples
        poses = [sample.pose for sample in samples]
        turns = [
            math.atan2(math.sin(b['yaw'] - a['yaw']), math.cos(b['yaw'] - a['yaw']))
            for a, b in zip(poses, poses[1:])
            if 'yaw' in a and 'yaw' in b
        ]
        positions = [pose for pose in poses if 'x' in pose and 'y' in pose]
        signs = [1 if turn > 0 else -1 for turn in turns if abs(turn) > 0.002]
        summary = {
            'samples': len(samples),
            'displacement_m': (
                round(
                    math.hypot(
                        positions[-1]['x'] - positions[0]['x'],
                        positions[-1]['y'] - positions[0]['y'],
                    ),
                    3,
                )
                if positions
                else None
            ),
            'path_length_m': (
                round(
                    sum(
                        math.hypot(b['x'] - a['x'], b['y'] - a['y'])
                        for a, b in zip(positions, positions[1:])
                    ),
                    3,
                )
                if positions
                else None
            ),
            'absolute_turn_deg': round(math.degrees(sum(abs(turn) for turn in turns)), 1),
            'turn_direction_changes': sum(a != b for a, b in zip(signs, signs[1:])),
            'blocked_samples': sum(sample.blocked for sample in samples),
            'interpretation': 'Motion statistics only; doorway crossing and goal arrival require visual verification.',
        }
        # 复用约每 0.2 米保留的实测点，并补上不足采样间隔的最后位置。
        points = [list(point['position']) for point in self.path_points]
        if points and samples:
            final = [samples[-1].pose.get(axis) for axis in ('x', 'y')]
            valid = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in final)
            if valid and final != points[-1]:
                points.append(final)
        if points:
            summary['sampled_positions'] = points
        return summary


def _navigation_stalled(samples, settings=None):
    settings = settings or ProtectionSettings()
    if not samples:
        return False

    # 只看最近约三秒，排除更早的移动进展
    now = samples[-1].monotonic_s
    window = [sample for sample in samples if now - sample.monotonic_s <= settings.stall_window_s + .2]
    if now - window[0].monotonic_s < settings.stall_window_s:
        return False
    poses = [sample.pose for sample in window]
    if not all(all(axis in pose for axis in ('x', 'y', 'yaw')) for pose in poses):
        return False
    origin = (poses[0]['x'], poses[0]['y'])
    if max(math.dist(origin, (pose['x'], pose['y'])) for pose in poses) >= settings.stall_distance_m:
        return False

    # 位移不足时，再检查碰撞和累计转向
    turning = sum(
        abs(math.atan2(math.sin(b['yaw'] - a['yaw']), math.cos(b['yaw'] - a['yaw'])))
        for a, b in zip(poses, poses[1:])
    )
    return any(sample.blocked for sample in window) or turning >= math.radians(settings.stall_turn_deg)
