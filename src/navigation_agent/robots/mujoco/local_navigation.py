"""仿真专用的短程避障；只修正 VLN 路线，不读取目标或房间语义。"""

from __future__ import annotations

import heapq
import math
import time

import mujoco
import numpy as np
from vln_mujoco.mpc import CONTROL_RATE_HZ, MpcSolveResult, MpcTracker, wrap_angle

GRID_STEP = 0.10
LOCAL_RADIUS = 2.0
EXTRA_CLEARANCE = 0.035
MAX_CAPTURE_HEADING_CHANGE = math.radians(35)
ANGULAR_ACCELERATION = 1.5
GOAL_TOLERANCE = 0.06
HEADING_TOLERANCE = 0.01


class CollisionMap:
    """用独立物理状态查询车身占用，避免改动正在运行的仿真。"""

    def __init__(self, simulation):
        self.simulation = simulation
        self.robot = simulation.robot
        self.data = mujoco.MjData(self.robot.model)
        self.cache = {}
        self.scene_pose = None

    def refresh(self):
        with self.simulation._lock:
            self.data.qpos[:] = self.simulation.data.qpos
        # 圆形车身的占用与自身朝向无关；只在场景物体移动时清除查询缓存。
        scene_pose = self.data.qpos.copy()
        offset = self.robot._base_qpos
        scene_pose[offset:offset + 7] = 0
        scene_pose[self.robot._left_qpos] = 0
        scene_pose[self.robot._right_qpos] = 0
        if not np.array_equal(scene_pose, self.scene_pose) or len(self.cache) > 20000:
            self.cache.clear()
        self.scene_pose = scene_pose

    def free(self, point):
        key = tuple(round(float(value), 5) for value in point[:2])
        if key in self.cache:
            return self.cache[key]
        # 四周额外留出余量，防止跟踪误差导致门框擦碰。
        for dx, dy in ((0, 0), (EXTRA_CLEARANCE, 0), (-EXTRA_CLEARANCE, 0),
                       (0, EXTRA_CLEARANCE), (0, -EXTRA_CLEARANCE)):
            offset = self.robot._base_qpos
            self.data.qpos[offset:offset + 2] = (key[0] + dx, key[1] + dy)
            mujoco.mj_kinematics(self.robot.model, self.data)
            mujoco.mj_collision(self.robot.model, self.data)
            if any(self.robot.envelope in contact.geom and contact.dist < 0.005
                   for contact in self.data.contact):
                self.cache[key] = False
                return False
        self.cache[key] = True
        return True


def segment_free(start, end, free):
    distance = float(np.linalg.norm(np.asarray(end) - start))
    steps = max(1, math.ceil(distance / 0.035))
    return all(free(np.asarray(start) + (np.asarray(end) - start) * i / steps)
               for i in range(1, steps + 1))


def local_route(start, goal, free, *, deadline=math.inf):
    """在当前位置两米内绕开障碍，不跨越未知远处寻找任务目标。"""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if np.linalg.norm(goal - start) > LOCAL_RADIUS or not free(goal):
        return []
    if segment_free(start, goal, free):
        return [start, goal]

    frontier = [(float(np.linalg.norm(goal - start)), 0.0, (0, 0))]
    cost = {(0, 0): 0.0}
    parent = {}
    visited = 0
    while frontier and visited < 500:
        if time.perf_counter() >= deadline:
            return []
        _, travelled, node = heapq.heappop(frontier)
        if travelled > cost[node]:
            continue
        visited += 1
        point = start + np.asarray(node) * GRID_STEP
        if np.linalg.norm(goal - point) < 0.25 and segment_free(point, goal, free):
            route = [goal, point]
            while node in parent:
                node = parent[node]
                route.append(start + np.asarray(node) * GRID_STEP)
            return list(reversed(route))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (node[0] + dx, node[1] + dy)
            position = start + np.asarray(neighbor) * GRID_STEP
            if np.linalg.norm(position - start) > LOCAL_RADIUS:
                continue
            next_cost = travelled + math.hypot(dx, dy) * GRID_STEP
            if next_cost >= cost.get(neighbor, math.inf):
                continue
            if not free(position) or not segment_free(point, position, free):
                continue
            cost[neighbor] = next_cost
            parent[neighbor] = node
            priority = next_cost + float(np.linalg.norm(goal - position))
            heapq.heappush(frontier, (priority, next_cost, neighbor))
    return []


class LocalNavigationTracker(MpcTracker):
    """沿用官方轨迹坐标转换与异步接口，在仿真执行端增加局部通行检查。"""

    def __init__(self, simulation):
        super().__init__()
        self.collision_map = CollisionMap(simulation)
        self.local_status = 'idle'
        self._capture_pose = None

    def reset(self):
        super().reset()
        self.local_status = 'idle'
        self._capture_pose = None

    def set_body_path(self, waypoints, capture_pose):
        super().set_body_path(waypoints, capture_pose)
        self._capture_pose = np.asarray(capture_pose, dtype=float).copy()
        pose = self.collision_map.simulation.snapshot()['pose']
        heading_change = wrap_angle(pose['yaw'] - self._capture_pose[2])
        # 画面拍摄后已大幅转向时，停车等新预测，不回头追旧视角的修正。
        if abs(heading_change) > MAX_CAPTURE_HEADING_CHANGE:
            self.reset()
            self.local_status = 'waiting_for_fresh_path'

    def poll(self):
        command = super().poll()
        if command is not None:
            if len(self.reference):
                self.local_status = 'tracking'
            else:
                self.local_status = 'turning' if abs(command[1]) > 0.01 else 'no_safe_path'
        return command

    def submit(self, current_pose):
        if self._future is not None or self._trajectory is None:
            return
        self._future = self._pool.submit(
            self._solve_local, self._generation,
            np.asarray(current_pose, dtype=float), self._trajectory.copy(),
            self._capture_pose.copy(), self.command,
        )

    def _solve_local(self, generation, pose, trajectory, capture_pose, previous_command):
        started = time.perf_counter()
        self.collision_map.refresh()
        free = self.collision_map.free
        command = (0.0, 0.0)
        route = []
        if len(trajectory):
            distances = np.linalg.norm(trajectory[:, :2] - pose[:2], axis=1)
            nearest = int(np.argmin(distances))
            endpoint = trajectory[-1, :2]
            previous_points = np.vstack((capture_pose[:2], trajectory[:-1, :2]))
            distinct = previous_points[
                np.linalg.norm(previous_points - endpoint, axis=1) > 0.01
            ]
            passed_endpoint = (
                distances[-1] <= distances[nearest] + 1e-6
                and len(distinct) > 0
                and np.dot(pose[:2] - endpoint, endpoint - distinct[-1]) > 0
            )
            translation = np.linalg.norm(trajectory[:, :2] - capture_pose[:2], axis=1)
            at_endpoint = distances[-1] <= GOAL_TOLERANCE or passed_endpoint
            if np.max(translation) < 0.05 or at_endpoint:
                # 到达位置后仍须完成模型给出的终点朝向，不把它当作 STOP。
                angle = wrap_angle(trajectory[-1, 2] - pose[2])
                if abs(angle) > HEADING_TOLERANCE:
                    command = (0.0, float(np.clip(1.8 * angle, -1.0, 1.0)))
            else:
                # 从最近的未通过点继续，保留模型的拐点，不直连最远端点。
                index = nearest
                while index < len(trajectory) - 1:
                    point = trajectory[index]
                    segment = point[:2] - previous_points[index]
                    passed = np.dot(pose[:2] - point[:2], segment) >= 0
                    reached = distances[index] <= 0.035
                    if not (passed or reached):
                        break
                    # 原地转向段后才允许平移；同位置用当前最接近的朝向接续。
                    next_point = trajectory[index + 1]
                    if np.linalg.norm(segment) < 0.01:
                        current_error = abs(wrap_angle(point[2] - pose[2]))
                        next_error = abs(wrap_angle(next_point[2] - pose[2]))
                        next_turn_is_closer = (
                            np.linalg.norm(next_point[:2] - point[:2]) < 0.01
                            and next_error < current_error
                        )
                        if current_error > HEADING_TOLERANCE and not next_turn_is_closer:
                            break
                    index += 1

                # 直线段可向前看 35 cm；遇到转角或原地转向就停止前视。
                while index < len(trajectory) - 1:
                    here, ahead = trajectory[index:index + 2]
                    incoming = here[:2] - previous_points[index]
                    outgoing = ahead[:2] - here[:2]
                    if np.linalg.norm(incoming) < 0.01 or np.linalg.norm(outgoing) < 0.01:
                        break
                    turn = wrap_angle(math.atan2(outgoing[1], outgoing[0])
                                      - math.atan2(incoming[1], incoming[0]))
                    if (abs(turn) > 0.10
                            or abs(wrap_angle(ahead[2] - here[2])) > 0.10
                            or np.linalg.norm(ahead[:2] - pose[:2]) > 0.35):
                        break
                    index += 1

                target = trajectory[index]
                distance = float(np.linalg.norm(target[:2] - pose[:2]))
                if distance <= 0.035:
                    angle = wrap_angle(target[2] - pose[2])
                    if abs(angle) > HEADING_TOLERANCE:
                        command = (0.0, float(np.clip(1.8 * angle, -1.0, 1.0)))
                else:
                    # 障碍内的点不能执行；按顺序找最近的可通行接续点。
                    for candidate in range(index, len(trajectory)):
                        target = trajectory[candidate]
                        distance = float(np.linalg.norm(target[:2] - pose[:2]))
                        if distance > LOCAL_RADIUS or time.perf_counter() >= started + 0.20:
                            break
                        if distance <= 0.035 or not free(target[:2]):
                            continue
                        route = local_route(
                            pose[:2], target[:2], free, deadline=started + 0.20
                        )
                        if route:
                            index = candidate
                            break
                    if route:
                        steering_point = route[1]
                        if len(route) > 2:
                            for point in route[2:]:
                                if np.linalg.norm(point - pose[:2]) > 0.35:
                                    break
                                if segment_free(pose[:2], point, free):
                                    steering_point = point
                        delta = steering_point - pose[:2]
                        angle = wrap_angle(math.atan2(delta[1], delta[0]) - pose[2])
                        # 位置误差负责回到轨迹，模型朝向负责沿轨迹转动。
                        heading_error = wrap_angle(target[2] - pose[2])
                        if len(route) > 2:
                            heading_error = angle
                        angular = float(np.clip(2.0 * angle + 0.6 * heading_error, -1.0, 1.0))
                        remaining_distance = distance + float(np.sum(np.linalg.norm(
                            np.diff(trajectory[index:, :2], axis=0), axis=1
                        )))
                        linear = min(0.35, remaining_distance)
                        linear *= max(0.0, math.cos(angle)) ** 2
                        if abs(angle) > 0.65:
                            linear = 0.0
                        command = (linear, angular)
                        # 日志保留模型剩余的 SE(2) 参考；绕障段单独保留几何路径。
                        if len(route) == 2:
                            route = np.vstack((pose, trajectory[nearest:]))
                        else:
                            route = np.array([[p[0], p[1], target[2]] for p in route])

        linear, angular = command
        if len(route) or angular:
            # 新路径不得让转向速度瞬间反号；停车和无路可走仍立即生效。
            angular_step = ANGULAR_ACCELERATION / CONTROL_RATE_HZ
            requested_angular = angular
            angular = float(np.clip(
                angular,
                previous_command[1] - angular_step,
                previous_command[1] + angular_step,
            ))
            if angular * requested_angular < 0:
                linear = 0.0
            # 用限加速度后的实际命令检查碰撞，不能沿用平滑前的预测。
            for seconds in np.linspace(0.05, 0.5, 10):
                heading = pose[2] + angular * seconds / 2
                probe = pose[:2] + linear * seconds * np.array([math.cos(heading), math.sin(heading)])
                if not free(probe):
                    linear = 0.0
                    break
        command = (linear, angular)
        reference = np.asarray(route).reshape((-1, 3))
        prediction = []
        linear, angular = command
        for seconds in np.linspace(0, 0.5, 6):
            heading = pose[2] + angular * seconds / 2
            position = pose[:2] + linear * seconds * np.array([math.cos(heading), math.sin(heading)])
            prediction.append([*position, pose[2] + angular * seconds])
        return MpcSolveResult(generation, command, reference, np.asarray(prediction),
                              (time.perf_counter() - started) * 1000)
