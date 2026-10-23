"""在仿真碰撞保护下，沿实际轨迹退回。"""

import asyncio
import math
import time


def remember_pose(tools):
    # 只记录实际走过的位置，位姿失效就清空轨迹
    pose = tools.state.get('simulation', {}).get('pose', {})
    if not all(
        isinstance(pose.get(axis), (int, float)) and math.isfinite(pose[axis])
        for axis in ('x', 'y', 'yaw')
    ):
        tools.breadcrumbs.clear()
        return
    point = (pose['x'], pose['y'])
    if tools.breadcrumbs and math.dist(point, tools.breadcrumbs[-1]) > 0.20:
        tools.breadcrumbs.clear()  # 位姿发生跳变时清空轨迹，不跨越未知路段。
    if not tools.breadcrumbs or math.dist(point, tools.breadcrumbs[-1]) >= 0.02:
        tools.breadcrumbs.append(point)


def recovery_path(tools):
    simulation = tools.state.get('simulation', {})
    if not simulation.get('backend', {}).get('collision_guard'):
        return []
    pose = simulation.get('pose', {})
    if not all(
        isinstance(pose.get(axis), (int, float)) and math.isfinite(pose[axis])
        for axis in ('x', 'y', 'yaw')
    ):
        return []
    current = (pose['x'], pose['y'])
    if not tools.breadcrumbs or math.dist(current, tools.breadcrumbs[-1]) > 0.12:
        return []
    path, distance, previous = [], 0.0, current

    # 沿记录反向取一段退路，最长约六十厘米
    for point in reversed(tools.breadcrumbs):
        distance += math.dist(previous, point)
        path.append(point)
        previous = point
        if distance >= 0.60:
            break
    return path if distance >= 0.20 else []


async def recover(tools):
    """沿近期走过的位置退回，每个运动子步由仿真检查碰撞。"""
    tools.require_manual()
    await tools.hold()

    # 先停车，再确认有连续退路和碰撞保护
    path = tools.recovery_path()
    if not path:
        return {
            'reason': 'recovery_unavailable',
            'detail': 'No continuous recent path or collision guard',
        }
    start = dict(tools.state['simulation']['pose'])
    tools.begin_motion('recover')
    deadline, reason = time.monotonic() + 8, 'recovery_timeout'
    try:
        while time.monotonic() < deadline:
            tools.require_manual()
            simulation = tools.state['simulation']
            if not simulation.get('backend', {}).get('collision_guard'):
                reason = 'recovery_unavailable'
                break
            if simulation['backend'].get('blocked'):
                reason = 'recovery_blocked'
                break
            pose = simulation['pose']

            # 依次移除已退回的轨迹点
            while path and math.dist((pose['x'], pose['y']), path[0]) < 0.025:
                reached = path.pop(0)
                while tools.breadcrumbs and tools.breadcrumbs[-1] != reached:
                    tools.breadcrumbs.pop()
            if not path:
                reason = 'recovered'
                break

            # 按轨迹点逐段退回，不抄近路跨过拐角。
            dx, dy = path[0][0] - pose['x'], path[0][1] - pose['y']
            error = math.atan2(
                math.sin(math.atan2(dy, dx) - pose['yaw']),
                math.cos(math.atan2(dy, dx) - pose['yaw']),
            )

            # 按当前朝向选择前进或倒退，先对齐再移动
            reverse = abs(error) > math.pi / 2
            if reverse:
                error = math.atan2(math.sin(error - math.pi), math.cos(error - math.pi))
            if abs(error) < 0.20:
                linear = -0.12 if reverse else 0.12
            else:
                linear = 0.0
            angular = max(-0.6, min(0.6, 2 * error))
            await tools.channel.send({'type': 'twist', 'linear': linear, 'angular': angular})
            await tools._snapshot()
            await asyncio.sleep(0.03)
        end = tools.state['simulation']['pose']
        moved = math.hypot(end['x'] - start['x'], end['y'] - start['y'])
        if reason == 'recovered' and moved < 0.15:
            reason = 'recovery_no_progress'
        return {
            'reason': reason,
            'displacement_m': round(moved, 3),
        }
    finally:
        try:
            # 仍持有控制权时停车，取消或异常也走这里
            if tools.state.get('control', {}).get('manual') and not tools.error:
                await tools.hold()
        finally:
            tools.motion_action = None
