"""独立测量真实场景的仿真速度，不连接 Agent 或 LightNav。"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import mujoco
from navigation_agent.robots.mujoco.mujoco_safe import SafeTurtleBot
from vln_mujoco.simulation import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', choices=('upstream', 'project'), default='upstream')
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    timings = {'step': [], 'render': []}
    render = mujoco.Renderer.render

    def timed_render(renderer, *args, **kwargs):
        started = time.perf_counter()
        result = render(renderer, *args, **kwargs)
        timings['render'].append(time.perf_counter() - started)
        return result

    mujoco.Renderer.render = timed_render
    robot = SafeTurtleBot()
    step = robot.step

    def timed_step(command):
        started = time.perf_counter()
        step(command)
        timings['step'].append(time.perf_counter() - started)

    robot.step = timed_step
    runtime = Simulation
    if args.runtime == 'project':
        from navigation_agent.robots.mujoco.runtime import RealtimeSimulation
        runtime = RealtimeSimulation
    simulation = runtime(robot)
    simulation.start()
    try:
        deadline = time.monotonic() + 90
        while simulation.frame() is None:
            if time.monotonic() > deadline:
                raise RuntimeError('Camera did not start')
            time.sleep(0.1)
        started = time.monotonic()
        initial = simulation.snapshot()
        frame_count = simulation._frame_count
        # 固定低速直行，避免把模型决策和控制器耗时混入基准。
        while time.monotonic() - started < args.seconds:
            simulation.set_velocity(0.10, 0)
            time.sleep(0.05)
        simulation.set_velocity(0, 0)
        final = simulation.snapshot()
        elapsed = time.monotonic() - started
        report = {
            'runtime': args.runtime,
            'wall_seconds': elapsed,
            'sim_seconds': final['sim_time'] - initial['sim_time'],
            'realtime_ratio': (final['sim_time'] - initial['sim_time']) / elapsed,
            'camera_fps': (simulation._frame_count - frame_count) / elapsed,
            'distance_m': final['pose']['x'] - initial['pose']['x'],
            'blocked': final['backend']['blocked'],
            'timing_ms': {
                name: {'median': statistics.median(values) * 1000,
                       'max': max(values) * 1000}
                for name, values in timings.items() if values
            },
        }
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        simulation.stop()


if __name__ == '__main__':
    main()
