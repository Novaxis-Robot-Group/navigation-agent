"""验证调度状态、完整核验请求及部署限制；固定输入不调用远端模型。"""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from navigation_agent.agent.core import scheduling
from navigation_agent.agent.model.completion import verify_completion
from navigation_agent.agent.schemas import AgentError, Decision
from navigation_agent.app.agent.settings import load_agent


class Model:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, body):
        self.requests.append(copy.deepcopy(body))
        return {'choices': [{'finish_reason': 'stop',
                            'message': {'content': json.dumps(self.responses.pop(0))}}]}


def verdict(status='yes'):
    return {name: {'status': status if name == 'arrival' else 'yes', 'evidence': '当前图证据'}
            for name in ('identity', 'arrival', 'facing')}


async def trace(controller_class=scheduling.SelectiveController, verifier=verify_completion):
    """参考源码与迁移实现共用输入；返回全部请求、动作、事件和状态。"""
    settings = load_agent()
    events = []
    controller = controller_class(settings, 'Turn left and go to the rightmost gray pole.',
                                  lambda kind, **data: events.append((kind, copy.deepcopy(data))))
    context = {'goal': controller.goal, 'observation_step': 1, 'observation_id': 'O1',
               'observation': {}, 'execution': {}, 'task_contract': {'original_instruction': controller.goal}}
    images = [{'image': 'images/001.jpg', 'observation_id': 'O1', 'jpeg': b'initial'},
              {'image': 'images/003.jpg', 'observation_id': 'O3', 'jpeg': b'current'}]
    model = Model([verdict()])
    decision, usage = await controller.decide(model, copy.deepcopy(context), images, settings=settings.llm)
    actions = [(decision.model_dump(), usage)]
    controller.result(decision, {'reason': 'agent_interrupt', 'detail': 'target_detected'}, 1)
    controller.result(Decision(action='look', direction='left', reason='确认剩余阶段',
                               resume_instruction='Go to the gray pole beside the doorway.'),
                      {'reason': 'turned'}, 2)
    decision, usage = await controller.decide(model, copy.deepcopy(context), images, settings=settings.llm)
    actions.append((decision.model_dump(), usage))
    controller.result(decision, {'reason': 'model_stop'}, 3)
    decision, usage = await controller.decide(model, copy.deepcopy(context), images, settings=settings.llm)
    actions.append((decision.model_dump(), usage))
    check_model = Model([verdict('uncertain')])
    report, check_usage = await verifier(check_model, context, images, settings.llm, evidence_mode=True)
    return {'requests': model.requests + check_model.requests, 'actions': actions,
            'events': events, 'state': controller.context(),
            'report': report.model_dump(), 'usage': check_usage}


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = load_agent()
        self.events = []
        self.controller = scheduling.SelectiveController(
            self.settings, 'Go to the door.', lambda kind, **data: self.events.append((kind, data)))

    async def test_direct_and_three_criteria_completion(self):
        output = await trace()
        self.assertEqual(output['actions'][0][1], {'planner_calls': 0})
        self.assertEqual(output['actions'][0][0]['target_labels'], [])
        self.assertEqual(output['actions'][-1][0]['action'], 'finish')
        self.assertEqual(len(output['requests']), 2)
        self.assertFalse(output['report']['confirmed'])
        self.assertEqual(output['state']['interventions'], 1)
        self.assertEqual(output['requests'][0]['max_tokens'], 900)

    async def test_failed_verification_reuses_same_frame_report(self):
        self.controller.transition('VERIFY', 'model_stop')
        model = Model([verdict('uncertain')])
        reports = []
        async def planner(*args, **kwargs):
            reports.append(await kwargs['confirm_completion']())
            reports.append(await kwargs['confirm_completion']())
            return Decision(action='look', direction='left', reason='补充证据'), {}
        with patch.object(scheduling, 'plan', planner):
            decision, _ = await self.controller.decide(model, {'execution': {}}, [], settings=self.settings.llm)
        self.assertIs(reports[0], reports[1])
        self.assertFalse(reports[0].confirmed)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(self.controller.mode, 'STAGING')

    async def test_bad_verification_cannot_finish(self):
        self.controller.transition('VERIFY', 'model_stop')
        with self.assertRaises(AgentError):
            await self.controller.decide(Model([{'confirmed': True, 'evidence': '旧协议'}]),
                                         {'execution': {}}, [], settings=self.settings.llm)

    def test_intervention_counts_transitions_not_planner_calls(self):
        self.controller.transition('STAGING', 'target')
        self.controller.transition('STAGING', 'target')
        self.assertEqual(self.controller.interventions, 1)
        self.controller.transition('DIRECT', 'handoff')
        self.controller.interventions = self.settings.selective.max_interventions
        with self.assertRaises(AgentError):
            self.controller.transition('STAGING', 'target')

    def test_failed_turn_does_not_apply_resume(self):
        self.controller.transition('STAGING', 'test')
        decision = Decision(action='look', direction='left', reason='补充证据', resume_instruction='Go to the chair.')
        self.controller.result(decision, {'reason': 'turn_timeout'}, 1)
        self.assertEqual(self.controller.instruction, 'Go to the door.')
        self.assertEqual(self.controller.mode, 'STAGING')

    def test_motion_requires_distinct_samples_and_excludes_turning(self):
        calls = []
        tools = SimpleNamespace(navigation_id=1,
            request_navigation_interrupt=lambda identity, reason: calls.append((identity, reason)) or True)
        def sample(second, yaw=0.):
            return {'camera_stamp_ns': int((100 + second) * 1e9), 'pose_valid': True,
                    'pose': {'x': 0., 'y': 0., 'yaw': yaw}}
        for _ in range(20):
            self.controller.observe_motion(sample(0), tools)
        self.assertEqual(calls, [])
        for second in range(1, 10):
            self.controller.observe_motion(sample(second, second * .1), tools)
        self.assertEqual(calls, [])
        tools.navigation_id = 2
        for second in range(10):
            self.controller.observe_motion(sample(second), tools)
        self.assertEqual(calls, [(2, 'selective_no_motion')])

    def test_navigation_failures_are_not_successful_handoffs(self):
        for reason in ('navigation_error', 'upstream_cancelled', 'timeout'):
            with self.subTest(reason=reason), self.assertRaises(AgentError):
                self.controller.result(Decision(action='navigate', reason='执行', instruction='Go to the door.'),
                                       {'reason': reason}, 1)


class DefaultLoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_default(self, goal, model, *, settings=None, backend=None, scene_validator=None, steps=5):
        from datetime import datetime, timezone
        from pathlib import Path
        import tempfile
        from navigation_agent.agent.core.runner import run_task
        from navigation_agent.agent.schemas import TaskRequest
        from navigation_agent.agent.storage.logs import LogStore
        from navigation_agent.agent.tools import NavigationTools
        from navigation_agent.app.agent.demo import DemoBackend
        with tempfile.TemporaryDirectory() as directory:
            settings = settings or load_agent()
            settings = settings.model_copy(update={
                'storage': settings.storage.model_copy(update={'runs_dir': Path(directory)}),
                'memory': settings.memory.model_copy(update={'compare_regions_when_stopped': False}),
            })
            request = TaskRequest(goal=goal, max_steps=steps, max_seconds=30, navigation_seconds=15.)
            record = {'id': 'd' * 32, 'goal': goal, 'status': 'running', 'step': 0,
                      'started_at': datetime.now(timezone.utc).isoformat()}
            store = LogStore(settings.storage)
            (store.run_directory(record) / 'images').mkdir(parents=True)
            backend = backend or DemoBackend()
            await run_task(record, request, NavigationTools(backend), settings=settings,
                           api_key='offline', model=model, store=store, scene_validator=scene_validator)
            events = [json.loads(line) for line in
                      (store.run_directory(record) / 'events.jsonl').read_text().splitlines()]
            return record, backend, events

    async def test_english_default_loop_has_no_initial_planner_request(self):
        from navigation_agent.app.agent.demo import DemoModel
        model = DemoModel()
        goal = 'Turn left and go to the rightmost gray pole.'
        record, backend, events = await self.run_default(goal, model)
        self.assertEqual(record['status'], 'completed', record.get('summary'))
        self.assertEqual(backend.actions, [('navigate', goal, 15.)])
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(record['execution_goal'], goal)
        self.assertEqual(next(e for e in events if e['type'] == 'decision')['usage'], {'planner_calls': 0})

    async def test_chinese_translates_once_then_uses_full_english_instruction(self):
        from navigation_agent.app.agent.demo import DemoModel
        model = DemoModel()
        record, backend, events = await self.run_default('运行离线接入示例', model)
        self.assertEqual(record['status'], 'completed', record.get('summary'))
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(record['original_goal'], '运行离线接入示例')
        self.assertEqual(backend.actions[0][1], record['execution_goal'])
        self.assertEqual(sum(e['type'] == 'execution_goal' for e in events), 1)

    async def test_bad_translation_precedes_backend_connection(self):
        from navigation_agent.app.agent.demo import DemoBackend, DemoModel
        class Backend(DemoBackend):
            entered = False
            async def __aenter__(self):
                self.entered = True
                return self
        class InvalidModel(DemoModel):
            async def complete(self, body):
                return {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}
        backend = Backend()
        record, backend, _ = await self.run_default('找到右侧第二个灰色柱子', InvalidModel(), backend=backend)
        self.assertEqual(record['status'], 'failed')
        self.assertFalse(backend.entered)
        self.assertEqual(backend.actions, [])

    async def test_scene_validator_prevents_direct_navigation(self):
        from navigation_agent.app.agent.demo import DemoModel
        model = DemoModel()
        record, backend, _ = await self.run_default(
            'Go to the door.', model, scene_validator=lambda decision: None, steps=1)
        self.assertEqual(backend.actions, [('look', 'left')])
        self.assertEqual(len(model.requests), 1)
        context = json.loads(model.requests[0]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['execution']['selective']['mode'], 'STAGING')

    async def test_missing_history_capability_fails_without_motion(self):
        from navigation_agent.app.agent.demo import DemoBackend, DemoModel
        class Unsupported(DemoBackend):
            configure_history = None
        record, backend, _ = await self.run_default('Go to the door.', DemoModel(), backend=Unsupported())
        self.assertEqual(record['status'], 'failed')
        self.assertIn('历史延续', record['summary'])
        self.assertEqual(backend.actions, [])

    async def test_explicit_history_disable_accepts_custom_backend(self):
        from navigation_agent.app.agent.demo import DemoBackend, DemoModel
        class Unsupported(DemoBackend):
            configure_history = None
        settings = load_agent()
        settings = settings.model_copy(update={'harness': settings.harness.model_copy(
            update={'preserve_same_instruction_history': False})})
        record, _, _ = await self.run_default('Go to the door.', DemoModel(),
                                              settings=settings, backend=Unsupported())
        self.assertEqual(record['status'], 'completed', record.get('summary'))


    async def test_passive_frames_keep_observation_order_without_consuming_decisions(self):
        import asyncio
        import time
        from navigation_agent.app.agent.demo import DemoBackend, DemoModel
        class Backend(DemoBackend):
            active_navigation_id = None
            supports_streaming_observation = True
            sampled = 0
            async def snapshot_observation(self):
                self.sampled += 1
                jpeg, observation = await super().observe()
                observation['camera_stamp_ns'] = time.time_ns()
                return jpeg, observation

            async def navigate(self, instruction, seconds):
                self.active_navigation_id = 1
                async with asyncio.timeout(2):
                    while self.sampled < 3:
                        await asyncio.sleep(.005)
                self.active_navigation_id = None
                return await super().navigate(instruction, seconds=seconds)

        settings = load_agent()
        settings = settings.model_copy(update={'harness': settings.harness.model_copy(
            update={'capture_interval_s': .01})})
        model = DemoModel()
        record, backend, events = await self.run_default('Go to the door.', model,
                                                         settings=settings, backend=Backend(), steps=2)
        self.assertEqual(record['status'], 'completed', record.get('summary'))
        self.assertEqual(record['step'], 2)
        passive = [e for e in events if e['type'] == 'passive_observation']
        self.assertGreaterEqual(len(passive), 3)
        observations = [e['step'] for e in events if e['type'] in ('observation', 'passive_observation')]
        self.assertEqual(observations, list(range(1, len(observations) + 1)))
        self.assertEqual(len(model.requests), 1)


class SceneHandoffTests(unittest.TestCase):
    def test_successful_turn_still_requires_scene_validation(self):
        controller = scheduling.SelectiveController(load_agent(), 'Go to the door.', lambda *a, **k: None,
                                                     scene_guarded=True)
        self.assertEqual(controller.mode, 'STAGING')
        controller.result(Decision(action='look', direction='left', reason='调整视角',
                                   resume_instruction='Go to the gray door.'), {'reason': 'turned'}, 1)
        self.assertEqual(controller.instruction, 'Go to the gray door.')
        self.assertEqual(controller.mode, 'STAGING')
