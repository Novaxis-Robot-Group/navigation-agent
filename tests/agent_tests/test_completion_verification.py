"""独立完成核验须接收当前图和历史证据，并遵守拒绝、取消与预算边界。"""

import asyncio
import base64
import json
from pathlib import Path
import tempfile
import unittest

from tests.config_support import legacy_search_run_task as run_task
from navigation_agent.agent.model.completion import verify_completion
from navigation_agent.agent.schemas import TaskRequest
from navigation_agent.agent.storage.logs import LogStore
from navigation_agent.agent.tools import NavigationTools
from tests.agent_tests.test_loop import FakeTools
from tests.config_support import agent_settings


def response(value, finish_reason='stop'):
    return {'choices': [{'finish_reason': finish_reason, 'message': {'content': json.dumps(value)}}]}


class Model:
    def __init__(self, confirmed=True, error=None, wait=False):
        self.confirmed, self.error, self.wait = confirmed, error, wait
        self.requests = []
        self.checking = asyncio.Event()
        self.closed = False
        self.cancelled = False
        self.decisions = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    async def complete(self, body):
        self.requests.append(body)
        query = json.loads(body['messages'][1]['content'][0]['text'])
        if query.get('operation') == 'verify_completion':
            self.checking.set()
            if self.error:
                raise self.error
            if self.wait:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            return response({'confirmed': self.confirmed, 'evidence': '当前图已独立检查类别'})
        self.decisions += 1
        if self.decisions == 1:
            return response({'action': 'finish', 'reason': '认为已看到目标',
                             'evidence': '当前有座面与靠背', 'scene': {'target_seen': 'yes'}})
        return response({'action': 'look', 'direction': 'left', 'reason': '检查左侧是否存在实际目标',
                         'scene': {'target_seen': 'no'}})


class CompletionVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def run_loop(self, model, *, cancel=False, expire=False, steps=1, backend=None, goal='找到沙发'):
        with tempfile.TemporaryDirectory() as directory:
            settings = agent_settings()
            settings = settings.model_copy(update={
                'storage': settings.storage.model_copy(update={'runs_dir': Path(directory)}),
                'execution': settings.execution.model_copy(update={'verify_completion': True}),
            })
            request = TaskRequest(goal=goal, max_steps=steps, max_seconds=30, navigation_seconds=15)
            if expire:
                request = request.model_copy(update={'max_seconds': .2})
            record = dict(id='c' * 32, goal=request.goal, status='running', step=0,
                          started_at='2026-09-30T00:00:00+00:00')
            store = LogStore(settings.storage)
            (store.run_directory(record) / 'images').mkdir(parents=True)
            backend = backend or FakeTools()
            task = asyncio.create_task(run_task(record, request, NavigationTools(backend),
                settings=settings, api_key='test', model=model, store=store))
            if cancel:
                await asyncio.wait_for(model.checking.wait(), 5)
                task.cancel()
            await task
            events = [json.loads(line) for line in
                      (store.run_directory(record) / 'events.jsonl').read_text().splitlines()]
            return record, events, backend

    async def test_no_history_still_checks_current_frame_and_confirms_finish(self):
        model = Model()
        record, events, backend = await self.run_loop(model)
        self.assertEqual(record['status'], 'completed', record.get('summary'))
        self.assertEqual(backend.actions, [])
        check = model.requests[1]
        self.assertEqual(len(check['messages']), 2)
        content = check['messages'][1]['content']
        self.assertEqual(len(content), 3)
        context = json.loads(content[0]['text'])
        self.assertEqual(context['operation'], 'verify_completion')
        self.assertEqual(context['goal'], '找到沙发')
        self.assertEqual(context['trajectory_memory']['recent'], [])
        self.assertEqual(context['observation_id'], 'O1')
        self.assertEqual(json.loads(content[1]['text'])['uses'][0]['view_id'], 'O1')
        self.assertEqual(base64.b64decode(content[2]['image_url']['url'].split(',')[1]), b'\xff\xd8test-image')
        self.assertTrue(next(e for e in events if e['type'] == 'completion_check')['confirmed'])

    async def test_reused_arrival_evidence_still_requires_independent_check(self):
        class ArrivalModel(Model):
            async def complete(self, body):
                result = await super().complete(body)
                payload = json.loads(result['choices'][0]['message']['content'])
                if payload.get('action') == 'finish':
                    payload['target_review'] = {'observation_step': 1, 'status': 'arrived',
                                                'evidence': payload.pop('evidence')}
                    return response(payload)
                return result

        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed):
                model = ArrivalModel(confirmed=confirmed)
                record, events, backend = await self.run_loop(model)
                self.assertEqual(sum(e['type'] == 'completion_check' for e in events), 1)
                errors = [e['category'] for e in events if e['type'] == 'decision_error']
                self.assertNotIn('output_format', errors)
                if confirmed:
                    self.assertEqual(record['status'], 'completed')
                    self.assertEqual(model.decisions, 1)
                    self.assertEqual(backend.actions, [])
                else:
                    self.assertEqual(record['status'], 'budget_exhausted')
                    self.assertEqual(errors, ['completion_verification'])
                    self.assertEqual(backend.actions, [('look', 'left')])

    async def test_approach_passes_memory_motion_and_original_images_to_independent_check(self):
        class ApproachTools(FakeTools):
            async def observe(self):
                moved = bool(self.actions)
                return (b'close-up' if moved else b'full-target'), {
                    'yaw': 0, 'pose': {'x': 1.2 if moved else 0, 'y': 0, 'yaw': 0},
                    'blocked': False,
                }

            async def navigate(self, instruction, *, seconds):
                await super().navigate(instruction, seconds=seconds)
                return {'reason': 'model_stop', 'motion': {
                    'displacement_m': 1.2, 'path_length_m': 1.2,
                    'sampled_positions': [[0, 0], [1.2, 0]],
                }}

        class ApproachModel(Model):
            async def complete(self, body):
                query = json.loads(body['messages'][1]['content'][0]['text'])
                if query.get('operation') == 'verify_completion':
                    return await super().complete(body)
                self.requests.append(body)
                self.decisions += 1
                if self.decisions == 1:
                    return response({
                        'action': 'navigate', 'skill': 'approach_target',
                        'reason': '接近前方绿色垃圾桶', 'instruction': 'Go to the green trash can.',
                        'scene': {'target_seen': 'yes', 'landmarks': ['绿色方形垃圾桶']},
                        'candidate_id': 'new_1',
                        'candidates': [{
                            'id': 'new_1', 'kind': 'target', 'description': '绿色垃圾桶',
                            'evidence': '前方可见绿色桶身和桶口', 'evidence_source': 'current',
                        }],
                    })
                if self.decisions == 2:
                    return response({
                        'action': 'finish', 'reason': '接近后当前只见桶口，结合上一帧确认到达',
                        'scene': {'target_seen': 'yes', 'landmarks': ['绿色桶口局部']},
                        'previous_action_review': {'outcome': 'progress', 'evidence': '接近后目标变大'},
                        'target_review': {'observation_step': 2, 'status': 'arrived',
                                          'evidence': '当前近距离桶口与历史目标一致'},
                    })
                return response({'action': 'look', 'direction': 'left',
                                 'reason': '补充到达证据', 'scene': {'target_seen': 'uncertain'}})

        # 历史身份和 STOP 已存在，是否完成仍取决于独立核验的响应。
        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed):
                model = ApproachModel(confirmed=confirmed)
                record, events, backend = await self.run_loop(
                    model, steps=2, backend=ApproachTools(), goal='走到绿色垃圾桶旁边',
                )
                check = model.requests[2]['messages'][1]['content']
                context = json.loads(check[0]['text'])
                decision_input = next(e for e in events if e['type'] == 'decision_input' and e['step'] == 2)
                original = decision_input['model_context']
                for name in ('spatial_memory', 'trajectory_memory', 'exploration_memory', 'observation'):
                    self.assertEqual(context[name], original[name])
                for name in ('target_tracking', 'last_action', 'last_navigation'):
                    self.assertEqual(context['execution'][name], original['execution'][name])
                self.assertEqual(context['observation_id'], 'O2')
                self.assertTrue(context['trajectory_memory']['recent'])
                self.assertEqual(context['execution']['last_navigation']['result']['motion']['displacement_m'], 1.2)
                self.assertEqual(context['execution']['target_tracking']['last_seen_step'], 1)
                self.assertNotIn('available_tools', context)
                self.assertNotIn('action_options', context['execution'])
                self.assertNotIn('接近后当前只见桶口', check[0]['text'])
                metadata = [json.loads(item['text']) for item in check[1:] if item['type'] == 'text']
                self.assertEqual(metadata, decision_input['images'])
                pixels = [base64.b64decode(item['image_url']['url'].split(',')[1])
                          for item in check if item['type'] == 'image_url']
                self.assertEqual(pixels, [b'full-target', b'close-up'])
                self.assertEqual(record['status'], 'completed' if confirmed else 'budget_exhausted', record.get('summary'))
                self.assertEqual(sum(e['type'] == 'completion_check' for e in events), 1)
                if not confirmed:
                    self.assertEqual(backend.actions[-1], ('look', 'left'))
                    self.assertIn('未确认到达不等于目标身份错误', model.requests[-1]['messages'][-1]['content'])

    async def test_rejected_identity_retries_without_recording_false_visual_claim(self):
        model = Model(confirmed=False)
        record, events, backend = await self.run_loop(model)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(backend.actions, [('look', 'left')])
        self.assertEqual([e['decision']['action'] for e in events if e['type'] == 'decision'], ['look'])
        self.assertFalse(any(e['type'] == 'observation_report' for e in events))
        self.assertEqual(next(e for e in events if e['type'] == 'decision_error')['category'], 'completion_verification')
        self.assertEqual(sum(e['type'] == 'observation' for e in events), 1)

    async def test_format_repair_does_not_spend_the_only_chance_to_reconsider_wrong_identity(self):
        class SequenceModel(Model):
            async def complete(self, body):
                query = json.loads(body['messages'][1]['content'][0]['text'])
                if query.get('operation') == 'verify_completion':
                    return await super().complete(body)
                self.requests.append(body)
                self.decisions += 1
                if self.decisions == 1:
                    return response({'action': 'finish', 'reason': '误认目标，漏填证据'})
                if self.decisions == 2:
                    return response({'action': 'finish', 'reason': '补齐格式',
                                     'evidence': '仍误认目标', 'scene': {'target_seen': 'yes'}})
                return response({'action': 'look', 'direction': 'left',
                                 'reason': '核验未确认，改看未查方向', 'scene': {'target_seen': 'no'}})

        model = SequenceModel(confirmed=False)
        record, events, backend = await self.run_loop(model)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(backend.actions, [('look', 'left')])
        self.assertEqual(model.decisions, 3)
        self.assertEqual([event['category'] for event in events if event['type'] == 'decision_error'],
                         ['output_format', 'completion_verification'])
        self.assertFalse(any(event['type'] == 'observation_report' for event in events))
        self.assertIn('不再 finish', model.requests[-1]['messages'][-1]['content'])

    async def test_repeated_finish_on_rejected_frame_is_bounded_and_does_not_rerun_verifier(self):
        class PersistentModel(Model):
            async def complete(self, body):
                query = json.loads(body['messages'][1]['content'][0]['text'])
                if query.get('operation') == 'verify_completion':
                    return await super().complete(body)
                self.requests.append(body)
                self.decisions += 1
                return response({'action': 'finish', 'reason': '重复旧判断', 'evidence': '相同画面'})

        model = PersistentModel(confirmed=False)
        record, events, backend = await self.run_loop(model)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(backend.actions, [])
        self.assertEqual(model.decisions, agent_settings().llm.format_retries + 2)
        self.assertEqual(sum(event['type'] == 'completion_check' for event in events), 1)
        self.assertFalse(any(event['type'] in ('decision', 'observation_report') for event in events))

    async def test_verifier_error_cannot_complete_or_move(self):
        record, events, backend = await self.run_loop(Model(error=TimeoutError()))
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(backend.actions, [])
        self.assertFalse(any(e['type'] == 'decision' for e in events))
        self.assertFalse(next(e for e in events if e['type'] == 'completion_check')['confirmed'])

    async def test_cancellation_closes_inflight_verifier(self):
        model = Model(wait=True)
        record, events, backend = await self.run_loop(model, cancel=True)
        self.assertEqual(record['status'], 'cancelled')
        self.assertTrue(model.cancelled and model.closed and backend.closed)
        self.assertEqual(backend.actions, [])
        self.assertFalse(any(e['type'] == 'decision' for e in events))

    async def test_task_budget_cancels_inflight_verifier(self):
        model = Model(wait=True)
        record, events, backend = await self.run_loop(model, expire=True)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertTrue(model.cancelled and model.closed and backend.closed)
        self.assertEqual(backend.actions, [])

    async def test_strict_boolean_and_truncated_response_fail_closed(self):
        class InvalidModel:
            async def complete(self, body):
                return self.value
        model = InvalidModel()
        for value in (response({'confirmed': 'true', 'evidence': '沙发'}),
                      response({'confirmed': True, 'evidence': '沙发'}, 'length')):
            model.value = value
            with self.assertRaises(ValueError):
                await verify_completion(model, {'goal': '找到沙发'}, [{'jpeg': b'jpeg'}], agent_settings().llm)
