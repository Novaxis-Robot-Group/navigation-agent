"""在原版完整 MuJoCo 场景中重复验证碰撞。"""

import argparse
import json
import math
from pathlib import Path
import time

import mujoco

from navigation_agent.robots.mujoco.mujoco_safe import SafeTurtleBot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    robot = SafeTurtleBot()
    started = time.perf_counter()
    report = {'steps': 0, 'blocked': 0, 'penetrations': 0, 'obstacles': {}}
    for direction in range(32):
        robot.reset()
        robot._yaw = direction * 2 * math.pi / 32
        robot.data.qpos[robot._base_qpos + 3 : robot._base_qpos + 7] = [
            math.cos(robot._yaw / 2),
            0,
            0,
            math.sin(robot._yaw / 2),
        ]
        mujoco.mj_forward(robot.model, robot.data)
        for _ in range(1200):
            robot.step((1.5 if direction % 2 else -1.5, 0))
            report['steps'] += 1
            if robot.blocked:
                report['blocked'] += 1
                name = robot.last_obstacle
                report['obstacles'][name] = report['obstacles'].get(name, 0) + 1
            if any(c.dist < 0 for c in robot.data.contact if robot.envelope in c.geom):
                report['penetrations'] += 1
    report['elapsed_s'] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert report['penetrations'] == 0 and report['blocked'] > 0, report
    assert any('sofa' in name for name in report['obstacles']), report
    assert any('diningtable' in name for name in report['obstacles']), report
    assert any('wall' in name for name in report['obstacles']), report


if __name__ == '__main__':
    main()
