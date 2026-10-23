"""在独占的 MuJoCo 实例比较找物策略；场景真值只用于评测。"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import mujoco
import numpy as np
from aiohttp import web
from PIL import Image

from navigation_agent.agent.core.runner import run_task
from navigation_agent.agent.context.prompts import read_prompt
from navigation_agent.agent.schemas import TaskRequest
from navigation_agent.agent.storage.logs import LogStore
from navigation_agent.agent.tools import NavigationTools
from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.deployments.settings import load_settings
from navigation_agent.app.mujoco.console import ConsoleServer
from navigation_agent.robots.mujoco.mujoco_safe import SafeTurtleBot
from navigation_agent.robots.mujoco.runtime import RealtimeSimulation

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    dict(id='front', goal='找到沙发', pose=[6.5, 11.6, 90], target='sofa_', visible=True),
    dict(id='behind', goal='找到沙发', pose=[6.5, 11.6, -90], target='sofa_', visible=False),
    dict(id='passage', goal='找到冰箱', pose=[6.5, 14.0, 90], target='refrigerator_', visible=False),
    dict(id='occlusion', goal='走到沙发旁边', pose=[7.7, 11.6, 90], target='sofa_', distance=1.5),
    dict(id='detour', goal='走到电视前', pose=[6.5, 13.8, -90], target='television_', distance=1.5),
    dict(id='absent', goal='找到紫色独角兽雕像', pose=[6.5, 11.6, 0], target=None),
]


def set_pose(simulation, pose):
    """只重置本脚本创建的仿真；不通过公共接口改变他人场景。"""
    with simulation._lock:
        robot = simulation.robot
        robot.reset()
        x, y, degrees = pose
        yaw = math.radians(degrees)
        robot.data.qpos[robot._base_qpos:robot._base_qpos + 2] = [x, y]
        robot.data.qpos[robot._base_qpos + 3:robot._base_qpos + 7] = [
            math.cos(yaw / 2), 0, 0, math.sin(yaw / 2),
        ]
        robot._yaw = yaw
        mujoco.mj_forward(robot.model, robot.data)
        if robot._collision():
            raise ValueError(f'评测起点与障碍重叠：{pose}')


def oracle(simulation, renderer, case, output):
    """分割真值不进入 Agent；64 像素只作可见性最低门槛，另保留原图复核。"""
    with simulation._lock:
        model, data = simulation.model, simulation.data
        renderer.update_scene(data, camera=simulation.robot.camera_name)
        renderer.disable_segmentation_rendering()
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        target_bodies = [
            i for i in range(model.nbody)
            if case['target'] and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').startswith(case['target'])
        ]
        geoms_by_body = {body: [] for body in target_bodies}
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body:
                if body in geoms_by_body:
                    geoms_by_body[body].append(geom)
                    break
                body = int(model.body_parentid[body])
        pose = simulation.robot.state().pose
        instances = []
        for body, geom_ids in geoms_by_body.items():
            mask = np.isin(segmentation[:, :, 0], geom_ids) & (
                segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            )
            instances.append(dict(
                body=model.body(body).name, visible_pixels=int(mask.sum()),
                distance_m=float(np.linalg.norm(data.xpos[body, :2] - np.asarray(pose[:2]))),
            ))
        visible = [instance for instance in instances if instance['visible_pixels'] >= 64]
        # 可见性和距离必须来自同一个实例，不能把近处遮挡目标与远处可见目标拼成到达。
        distance = min((instance['distance_m'] for instance in visible), default=None)
        pixels = sum(instance['visible_pixels'] for instance in instances)
    Image.fromarray(rgb).save(output)
    return dict(visible_pixels=pixels, target_visible=bool(visible), distance_m=distance,
                instances=instances, pose=list(pose),
                collision_guard=hasattr(simulation.robot, 'envelope'))


def summarize(directory, record, case, truth, elapsed):
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    actions = Counter(e['decision']['action'] for e in events if e['type'] == 'decision')
    completed = record['status'] == 'completed'
    grounded = truth['target_visible'] and (
        'distance' not in case or truth['distance_m'] <= case['distance']
    )
    valid = any(e['type'] == 'decision' for e in events)
    success = (completed and grounded) if case['target'] else None
    motion = [e.get('result', {}).get('motion', {}) for e in events if e['type'] == 'action_result']
    turns = Counter()
    invalid = Counter(e.get('category', 'unknown') for e in events if e['type'] == 'decision_error')
    durations = Counter()
    pending = {}
    for e in events:
        t = datetime.fromisoformat(e['time']).timestamp()
        if e['type'] == 'action_input':
            pending[e['step']] = (e.get('action', 'unknown'), t)
        if e['type'] == 'action_result' and e['step'] in pending:
            action, started = pending.pop(e['step'])
            durations[action] += t - started
            turns[action] += (e.get('result', {}).get('motion') or {}).get('absolute_turn_deg', 0)
    return dict(case=case['id'], task_id=record['id'], directory=str(directory), status=record['status'],
                summary=record.get('summary'), success=success, false_finish=completed and not grounded,
                valid=valid, elapsed_s=elapsed, actions=dict(actions), action_seconds=dict(durations),
                turn_by_action_deg=dict(turns), decision_errors=dict(invalid),
                negative_case=case['target'] is None,
                turn_deg=sum(x.get('absolute_turn_deg', 0) for x in motion),
                path_length_m=sum(x.get('path_length_m') or 0 for x in motion), oracle=truth)


async def evaluate(args):
    settings = load_settings(args.agent_config, args.backend_config)
    if not settings.api_key:
        raise RuntimeError('尚未配置 Agent 模型密钥')
    output = ROOT / 'runs' / 'adaptive-evaluation' / args.label
    output.mkdir(parents=True, exist_ok=False)
    # 一批任务固定同一份 Prompt，运行期间仓库编辑不会改变后续重复的条件。
    prompt_snapshot = output / 'prompt.md'
    prompt_snapshot.write_text(read_prompt(settings.agent.prompt), encoding='utf-8')
    prompt_settings = settings.agent.prompt.model_copy(update={
        'path': prompt_snapshot, 'reload_each_step': False,
    })
    if prompt_settings.scene is not None:
        raise ValueError('MuJoCo 固定评测不加载真机场景规则')
    agent = settings.agent.model_copy(update={
        'storage': settings.agent.storage.model_copy(update={'runs_dir': output}),
        'prompt': prompt_settings,
    })
    backend = settings.backend.model_copy(update={
        'robot': settings.backend.robot.model_copy(update={'base_url': f'http://127.0.0.1:{args.port}'})
    })
    cases = json.loads(args.cases_file.read_text()) if args.cases_file else CASES
    selected = [case for case in cases if not args.case or case['id'] in args.case]
    if not selected:
        raise ValueError('没有匹配的评测案例')
    provenance = dict(
        revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        configuration=agent.model_dump(mode='json'), backend=backend.model_dump(mode='json'),
        cases=selected, repeats=args.repeats,
        max_steps=args.steps, max_seconds=args.seconds,
        navigation_seconds=agent.execution.navigation_seconds,
        scene='procthor-val-2', thresholds=dict(visible_pixels=64, arrival_center_distance_m=1.5),
    )
    (output / 'manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    # 保存评测实际使用的源文件摘要，脏工作区也能追溯。
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*(ROOT / 'src/navigation_agent').rglob('*.py'),
                        *(ROOT / 'configs/prompts').glob('*.md')]}
    hashes['tests/evaluate_object_search.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / 'source-hashes.json').write_text(json.dumps(hashes, indent=2))
    patch = subprocess.check_output([
        'git', '-C', str(ROOT), 'diff', '--', 'src/navigation_agent', 'configs',
        'tests/evaluate_object_search.py',
    ], text=True)
    (output / 'source.patch').write_text(patch)
    simulation = RealtimeSimulation(SafeTurtleBot(
        camera_height_m=backend.simulation.camera_height_m,
    ))
    server = ConsoleServer(agent_url='http://127.0.0.1:18092', simulation=simulation,
                           vln_settings=backend.vln,
                           recording=backend.simulation.recording.model_copy(update={
                               'directory': output / 'recordings',
                           }))
    runner = web.AppRunner(server.app())
    await runner.setup()
    renderer = mujoco.Renderer(simulation.model, height=270, width=480)
    results = []
    try:
        await web.TCPSite(runner, '127.0.0.1', args.port).start()
        await asyncio.sleep(1)
        for repeat in range(args.repeats):
            for case in selected:
                server.stop_vln()
                simulation.set_velocity(0, 0)
                set_pose(simulation, case['pose'])
                await asyncio.sleep(.3)
                initial = oracle(simulation, renderer, case, output / f"{case['id']}-{repeat}-initial.png")
                if 'visible' in case and initial['target_visible'] != case['visible']:
                    raise ValueError(f"案例初始可见性不符：{case['id']} {initial}")
                if args.prepare:
                    print(json.dumps(dict(case=case['id'], initial=initial)), flush=True)
                    continue
                request = TaskRequest(goal=case['goal'], max_steps=args.steps, max_seconds=args.seconds,
                                      navigation_seconds=agent.execution.navigation_seconds)
                record = dict(id=uuid4().hex, goal=request.goal, status='running', phase='starting',
                              step=0, max_steps=args.steps, max_seconds=args.seconds,
                              navigation_seconds=agent.execution.navigation_seconds,
                              configuration=agent.model_dump(mode="json"), model=agent.llm.model,
                              started_at=datetime.now(timezone.utc).isoformat())
                store = LogStore(agent.storage)
                (store.run_directory(record) / 'images').mkdir(parents=True)
                store.event(record, 'started', goal=request.goal, evaluation_case=case['id'])
                # 评测直接调用 runner，也要使用部署同一录像器保存连续画面。
                record['recording'] = await server.recorder.start(record)
                started = time.monotonic()
                try:
                    await run_task(record, request, NavigationTools(create_backend(backend)),
                                   settings=agent, api_key=settings.api_key, store=store)
                finally:
                    elapsed = time.monotonic() - started
                    record['recording'] = await server.recorder.stop(record['id'])
                    store.save_result(record)
                truth = oracle(simulation, renderer, case, output / f"{case['id']}-{repeat}-final.png")
                result = summarize(store.run_directory(record), record, case, truth, elapsed)
                result.update(repeat=repeat, initial=initial)
                results.append(result)
                (output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print('EVALUATION ' + json.dumps(result, ensure_ascii=False), flush=True)
                if record['status'] == 'cancelled':
                    return
    finally:
        renderer.close()
        await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--port', type=int, default=18088)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--agent-config', type=Path, default=ROOT / 'configs/agent.toml')
    parser.add_argument('--backend-config', type=Path, default=ROOT / 'configs/mujoco-lightnav.toml')
    parser.add_argument('--cases-file', type=Path, help='自定义案例 JSON；场景真值仅用于评测')
    parser.add_argument('--case', action='append')
    parser.add_argument('--prepare', action='store_true')
    asyncio.run(evaluate(parser.parse_args()))
