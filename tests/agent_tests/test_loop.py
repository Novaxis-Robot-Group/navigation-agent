import asyncio
from contextlib import closing
import json
import math
import os
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import HTTPException

from navigation_agent.app.agent import api as main
from tests.support import patch_runs, clear_navigation_check
from tests.backend_support import patch_tools


def target_review(step, status='visible'):
    return {'observation_step': step, 'status': status, 'evidence': '当前图已复核目标和可见路径'}


def navigation(instruction='Walk straight toward the doorway ahead and stop.', candidate='new_1', *, kind='target', review_step=None):
    # 通用导航测试使用当前已见目标；探索顺序测试显式使用 open_passage。
    proposal = {
        'action': 'navigate',
        'skill': 'approach_target' if kind == 'target' else 'search_navigation',
        'navigation_check': clear_navigation_check(),
        'instruction': instruction,
        'reason': '接近当前可见目标或探索所选通道',
        'scene': {'target_seen': 'yes' if kind == 'target' else 'no'},
        'candidate_id': candidate,
        'candidates': [
            {
                'id': candidate,
                'evidence_source': 'current',
                'kind': kind,
                'expected_new_view': '门框后方尚未观察的走廊',
                'description': '开放门洞',
                'evidence': '可见门后地面',
            }
        ],
    }
    if review_step is not None:
        proposal['target_review'] = target_review(review_step)
    return proposal



class FakeTools:
    look_angle_deg = 70
    def __init__(self, *_):
        self.actions = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    async def observe(self):
        return b'\xff\xd8test-image', {
            'yaw': 0,
            'pose': {'x': 0, 'y': 0, 'yaw': 0},
            'blocked': False,
        }

    def require_manual(self):
        pass

    def require_control(self):
        self.require_manual()

    async def look(self, direction):
        self.actions.append(('look', direction))
        return {'reason': 'turned'}

    async def navigate(self, instruction, *, seconds):
        self.actions.append(('navigate', instruction))
        return {'reason': 'model_stop'}

    async def orient(self, heading_deg):
        self.actions.append(('orient', heading_deg))
        return {'reason': 'oriented'}

    async def recover(self):
        self.actions.append(('recover',))
        return {'reason': 'recovered', 'displacement_m': 0.5}


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.responses = []
        self.requests = []
        self.tools = FakeTools()
        self.delay = 0
        self.http_status = 200
        self.finish_reason = 'stop'
        app = web.Application()

        async def cloud(request):
            self.requests.append(await request.json())
            await asyncio.sleep(self.delay)
            if self.http_status != 200:
                return web.json_response({'error': 'test-secret'}, status=self.http_status)
            value = self.responses.pop(0)
            content = json.dumps(value) if isinstance(value, dict) else value
            return web.json_response(
                {
                    'choices': [
                        {
                            'finish_reason': self.finish_reason,
                            'message': {'content': content},
                        }
                    ],
                    'usage': {'total_tokens': 30},
                }
            )

        app.router.add_post('/chat/completions', cloud)
        self.server = TestServer(app)
        await self.server.start_server()
        self.patches = [
            patch_runs(Path(self.temp.name)),
            patch_tools(self.tools),
            patch.dict(
                os.environ,
                {
                    'DASHSCOPE_API_KEY': 'test-secret',
                    'DASHSCOPE_MODEL': 'qwen3.8-27b',
                    'DASHSCOPE_BASE_URL': str(self.server.make_url('/')),
                },
            ),
        ]
        for item in self.patches:
            item.start()
        main.worker = None
        main.records.clear()

    async def asyncTearDown(self):
        if main.worker is not None and not main.worker.done():
            main.worker.cancel()
            await main.worker
        await self.server.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def run_loop(self, steps=5):
        record = await main.start_task(main.TaskRequest(goal='找到棕色沙发', max_steps=steps))
        await main.worker
        return await main.get_task(record['id'])

    async def test_invalid_backend_configuration_does_not_create_running_task(self):
        with patch.object(main.deployment, 'create_tools', side_effect=ValueError('unknown backend')):
            with self.assertRaises(HTTPException) as error:
                await main.start_task(main.TaskRequest(goal='找到沙发'))
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(main.records, {})
        self.assertIsNone(main.worker)

    async def test_observe_look_navigate_then_verify_finish(self):
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '看看门口'},
            navigation('Walk straight toward the doorway on your left and stop.'),
            {'action': 'finish', 'reason': '找到', 'evidence': '新画面中可见棕色沙发',
             'target_review': target_review(3, 'arrived')},
        ]
        record = await self.run_loop()
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['step'], 3)
        self.assertEqual(self.tools.actions, [('look', 'left'), ('navigate', 'Walk straight toward the doorway on your left and stop.')])
        self.assertTrue(self.tools.closed)
        self.assertEqual(self.requests[0]['model'], 'qwen3.8-27b')
        self.assertFalse(self.requests[0]['enable_thinking'])
        self.assertIn('data:image/jpeg;base64,', json.dumps(self.requests[0]))
        self.assertIn('model_stop', json.dumps(self.requests[2]))
        directory = main.log_store().run_directory(record)
        self.assertEqual(len(list((directory / 'images').glob('*.jpg'))), 3)
        self.assertNotIn('test-secret', (directory / 'events.jsonl').read_text(encoding='utf-8'))

    async def test_invalid_cloud_action_never_moves(self):
        self.responses = [{'action': 'teleport', 'reason': 'invalid'}] * 2
        # 单轮校验基线；额外重新观察由专门用例验证。
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(self.tools.actions, [])
        self.assertTrue(self.tools.closed)

    async def test_navigation_stop_does_not_complete_task(self):
        self.responses = [navigation()]
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(record['last_result']['reason'], 'model_stop')

    async def test_cancel_during_cloud_and_reject_concurrent_task(self):
        self.responses = [{'action': 'finish', 'reason': 'done', 'evidence': 'test'}]
        self.delay = 0.5
        record = await main.start_task(main.TaskRequest(goal='test'))
        for _ in range(100):
            if self.requests:
                break
            await asyncio.sleep(0.01)
        with self.assertRaises(HTTPException) as error:
            await main.start_task(main.TaskRequest(goal='another'))
        self.assertEqual(error.exception.status_code, 409)
        result = await main.cancel_task(record['id'])
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(self.tools.closed)
        self.assertEqual(self.tools.actions, [])

    async def test_cloud_failure_does_not_leak_body(self):
        self.http_status = 401
        record = await self.run_loop()
        self.assertEqual(record['status'], 'failed')
        self.assertIn('401', record['summary'])
        self.assertNotIn('test-secret', json.dumps(record))

    async def test_missing_key_rejected_before_any_connection(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': ''}):
            with self.assertRaises(HTTPException) as error:
                await main.start_task(main.TaskRequest(goal='test'))
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.requests, [])

    async def test_finish_requires_evidence(self):
        for evidence in (None, '   '):
            with self.subTest(evidence=evidence):
                self.requests.clear()
                proposal = {
                    'action': 'finish', 'reason': '当前图已看到灰色沙发',
                    'scene': {'target_seen': 'yes'},
                    'previous_action_review': {'outcome': 'progress', 'evidence': '接近后看到沙发'},
                }
                if evidence is not None:
                    proposal['evidence'] = evidence
                self.responses = [proposal] * 2
                record = await self.run_loop(steps=1)
                self.assertEqual(record['status'], 'failed')
                self.assertIn('finish 缺少最外层 evidence', record['summary'])
                self.assertIn('共 2 次', record['summary'])
                self.assertEqual(self.tools.actions, [])
                self.assertEqual(len(self.requests), 2)
                events = [json.loads(line) for line in (
                    main.log_store().run_directory(record) / 'events.jsonl'
                ).read_text(encoding='utf-8').splitlines()]
                errors = [event for event in events if event['type'] == 'decision_error']
                self.assertEqual(len(errors), 2)
                for event in errors:
                    error = event['validation_errors'][0]
                    self.assertEqual(error['field'], ['evidence'])
                    self.assertIn('没有可复用的 arrived 复核证据', error['message'])

    async def test_finish_correction_preserves_observation_and_completes_without_motion(self):
        # 灰色沙发任务的失败形态：完成证据写在场景和复核里，遗漏最外层字段。
        proposal = {
            'action': 'finish', 'reason': '右侧玻璃隔断内可见灰色沙发，已满足找到目标',
            'scene': {'target_seen': 'yes', 'objects': [
                {'name': '灰色沙发', 'description': '右侧玻璃隔断内可见浅灰色沙发'},
            ]},
            'candidates': [],
            'previous_action_review': {
                'outcome': 'progress', 'view_change': 'same_area',
                'evidence': '接近后沙发形状和颜色更清楚',
            },
        }
        corrected = {**proposal, 'evidence': '当前图右侧玻璃隔断内可见灰色沙发的座面和靠背'}
        self.responses = [proposal, corrected]
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['step'], 1)
        self.assertEqual(record['last_decision']['evidence'], corrected['evidence'])
        self.assertEqual(self.tools.actions, [])
        self.assertEqual(len(self.requests), 2)
        original, retried = [request['messages'] for request in self.requests]
        self.assertEqual(original[1], retried[1])
        self.assertEqual(json.loads(retried[-2]['content']), proposal)
        self.assertIn('finish 缺少最外层 evidence', retried[-1]['content'])
        self.assertIn('不能用 reason 或 previous_action_review.evidence 代替', retried[-1]['content'])

    async def test_validation_details_redact_unexpected_field_names(self):
        self.responses = [{
            'action': 'finish', 'reason': '已找到', 'evidence': '当前图可见目标',
            'test-secret': 'unexpected field',
        }] * 2
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'failed')
        self.assertNotIn('test-secret', record['summary'])
        self.assertIn('[REDACTED]', record['summary'])
        self.assertNotIn('test-secret', self.requests[1]['messages'][-1]['content'])
        events = (main.log_store().run_directory(record) / 'events.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('test-secret', events)

    async def test_output_token_limit_has_specific_error_and_cannot_complete(self):
        self.finish_reason = 'length'
        self.responses = [
            {'action': 'finish', 'reason': '已找到', 'evidence': '当前图看到目标'},
        ] * 2
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'failed')
        self.assertIn('输出 token 上限', record['summary'])
        self.assertEqual(self.tools.actions, [])
        self.assertEqual(len(self.requests), 2)

    async def test_invalid_output_logged_and_retried_without_motion(self):
        self.responses = [
            {
                'action': 'navigate',
                'direction': 'up',
                'instruction': 'forward',
                'reason': 'test-secret',
            },
            {'action': 'give_up', 'reason': 'Closed door; no available route'},
        ]
        record = await self.run_loop(steps=1)
        self.assertEqual(self.tools.actions, [])
        self.assertEqual(len(self.requests), 2)
        events = (main.log_store().run_directory(record) / 'events.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('test-secret', events)
        error = next(
            e for e in map(json.loads, events.splitlines()) if e['type'] == 'decision_error'
        )
        self.assertEqual(error['finish_reason'], 'stop')
        self.assertEqual(error['validation_errors'][0]['field'], ['direction'])
        self.assertIn('[REDACTED]', error['raw_output'])

    async def test_malformed_json_can_recover(self):
        self.responses = [
            '{"action":',
            {'action': 'look', 'direction': 'left', 'reason': 'observe'},
        ]
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(self.tools.actions, [('look', 'left')])
        with closing(sqlite3.connect(main.log_store().run_directory(record) / 'memory.sqlite3')) as db:
            rows = db.execute('SELECT action, observed_step FROM trajectory_actions').fetchall()
        self.assertEqual(rows, [('look', None)])

    async def test_visible_ground_without_useful_candidates_is_allowed_without_retry(self):
        # 普通地面不强制登记成探索候选，也不为此浪费模型重试。
        blank = {
            'action': 'look',
            'direction': 'left',
            'reason': '环视寻找目标',
            'scene': {'visible_ground': '前方可见浅色地砖', 'target_seen': 'no'},
        }
        self.responses = [blank]
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(self.tools.actions, [('look', 'left')])
        self.assertEqual(len(self.requests), 1)
        events = [json.loads(line) for line in (
            main.log_store().run_directory(record) / 'events.jsonl'
        ).read_text(encoding='utf-8').splitlines()]
        errors = [event for event in events if event['type'] == 'decision_error']
        self.assertEqual(errors, [])

    async def test_historical_candidate_on_look_does_not_create_sighting_or_gain(self):
        first = {
            'action': 'look', 'direction': 'left', 'reason': '先看左侧',
            'scene': {'target_seen': 'no'},
            'candidates': navigation(kind='open_passage')['candidates'],
        }
        historical = {
            **first,
            'candidates': [{**first['candidates'][0], 'id': 'C1', 'evidence_source': 'history',
                            'evidence': '根据历史记录提及先前的门洞，当前不可见'}],
        }
        self.responses = [first, historical]
        record = await self.run_loop(steps=2)
        directory = main.log_store().run_directory(record)
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        gain = next(event for event in events if event['type'] == 'observation_gain')
        self.assertEqual(gain['current_candidates'], [])
        self.assertEqual(self.tools.actions, [('look', 'left'), ('look', 'left')])
        with closing(sqlite3.connect(directory / 'memory.sqlite3')) as db:
            routes = db.execute('SELECT id, source_step, candidate_json FROM exploration_routes').fetchall()
        self.assertEqual([(row[0], row[1]) for row in routes], [('C1', 1)])
        self.assertEqual(json.loads(routes[0][2])['view_id'], 'O1')

    async def test_same_instruction_from_new_positions_is_allowed(self):
        count = 0

        async def observe_moving():
            nonlocal count
            count += 1
            return b'\xff\xd8image', {'yaw': 0, 'pose': {'x': count * 2, 'y': 0}}

        self.tools.observe = observe_moving
        self.responses = [
            navigation('Walk straight toward the doorway ahead and stop.',
                       review_step=step if step > 1 else None)
            for step in range(1, 4)
        ]
        record = await self.run_loop(steps=3)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(len(self.tools.actions), 3)

    async def test_visible_target_can_continue_after_moving_then_finish(self):
        count = 0

        async def observe_moving():
            nonlocal count
            count += 1
            return b'\xff\xd8image', {'yaw': 0, 'pose': {'x': count * 4.56, 'y': 0}}

        self.tools.observe = observe_moving
        self.responses = [
            navigation('Walk straight toward the silver trash bin in front-right and stop.'),
            navigation('Walk straight toward the silver trash bin on your right and stop.', candidate='C1', review_step=2),
            {
                'action': 'finish', 'reason': '当前图已确认目标',
                'evidence': '当前图清楚可见大的银色垃圾筐',
                'target_review': target_review(3, 'arrived'),
                'candidates': navigation(candidate='C1')['candidates'],
            },
        ]
        record = await self.run_loop()
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(self.tools.actions), 2)
        events = [json.loads(line) for line in (
            main.log_store().run_directory(record) / 'events.jsonl'
        ).read_text(encoding='utf-8').splitlines()]
        self.assertFalse(any(event['type'] == 'decision_error' for event in events))
        actions = [event for event in events if event['type'] == 'action_input']
        self.assertEqual([event['candidate_id'] for event in actions], ['C1', 'C2'])
        execution = json.loads((main.log_store().run_directory(record) / 'execution.json').read_text(encoding='utf-8'))
        candidates = execution['unfinished_elsewhere']
        self.assertEqual([candidate['attempt_count'] for candidate in candidates], [1, 1])

    async def test_revisit_keeps_history_in_storage_without_attaching_irrelevant_old_image(self):
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '检查桌子后方',
             'scene': {'landmarks': ['desk'], 'target_seen': 'no', 'occluded': ['behind desk']}},
            {'action': 'give_up', 'reason': 'No reachable new viewpoint'},
        ]
        record = await self.run_loop()
        content = self.requests[1]['messages'][1]['content']
        self.assertEqual(sum(p['type'] == 'image_url' for p in content), 1)
        context = json.loads(content[0]['text'])['spatial_memory']
        self.assertEqual(context['same_view_previous_observations'], 1)
        self.assertEqual(context['reference_image'], 'images/001.jpg')
        self.assertEqual(context['semantic_frames'], [])
        with closing(sqlite3.connect(main.log_store().run_directory(record) / 'memory.sqlite3')) as db:
            scene = db.execute('SELECT scene_json FROM semantic_frames WHERE step=1').fetchone()[0]
            self.assertEqual(json.loads(scene)['occluded'], ['behind desk'])
        session = json.loads(content[0]['text'])['execution']['exploration_session']
        self.assertEqual(session['current_view_id'], 'O2')
        self.assertEqual([v['id'] for v in session['views']], ['O2'])

    async def test_reworded_failed_approach_is_rejected_then_replanned(self):
        async def blocked(instruction, *, seconds):
            self.tools.actions.append(('navigate', instruction))
            return {'reason': 'blocked', 'motion': {'displacement_m': 0}}

        self.tools.navigate = blocked
        self.responses = [
            navigation('Walk straight toward the first doorway ahead and stop.'),
            navigation('Walk straight toward the second doorway ahead and stop.', candidate='C1', review_step=3),
            navigation('Walk straight toward the third doorway ahead and stop.', candidate='C1', review_step=5),
            {'action': 'look', 'direction': 'right', 'reason': 'Find another direction'},
        ]
        record = await self.run_loop(steps=5)
        self.assertEqual(
            self.tools.actions,
            [
                ('navigate', 'Walk straight toward the first doorway ahead and stop.'),
                ('recover',),
                ('navigate', 'Walk straight toward the second doorway ahead and stop.'),
                ('recover',),
                ('look', 'right'),
            ],
        )
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(len(self.requests), 4)
        context = json.loads(self.requests[2]['messages'][1]['content'][0]['text'])['execution']
        self.assertEqual(context['local_candidates'][0]['failed_attempts'], 2)
        self.assertEqual(context['local_candidates'][0]['status'], 'avoid_from_here')

    async def test_missing_candidate_does_not_execute_and_memory_resets_per_task(self):
        self.responses = [
            {'action': 'navigate', 'instruction': 'forward', 'reason': 'test'},
            {'action': 'give_up', 'reason': 'no visible route'},
        ]
        first = await self.run_loop()
        self.assertEqual(self.tools.actions, [])
        self.responses = [{'action': 'give_up', 'reason': 'new task'}]
        second = await self.run_loop()
        context = json.loads(self.requests[-1]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['spatial_memory']['same_view_previous_observations'], 0)
        self.assertEqual(context['execution']['local_candidates'], [])
        self.assertNotEqual(first['id'], second['id'])

    async def test_failures_on_new_candidates_do_not_abort_exploration(self):
        async def blocked(instruction, *, seconds):
            self.tools.actions.append(('navigate', instruction))
            return {'reason': 'blocked', 'motion': {'displacement_m': 0}}

        self.tools.navigate = blocked
        self.responses = []
        for step, name in zip((1, 3, 5), ('left doorway', 'white table', 'right doorway')):
            proposal = navigation(f'Walk straight toward the {name} ahead and stop.')
            proposal['candidates'][0]['description'] = name
            if step > 1:
                proposal['target_review'] = target_review(step, 'rejected')
            self.responses.append(proposal)
        record = await self.run_loop(steps=5)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(record['exploration_summary']['candidates'], 3)
        self.assertEqual(record['exploration_summary']['avoided_approaches'], 0)

    async def test_door_after_turn_reaches_navigation_despite_legacy_angles(self):
        # 复现失败日志：右转后门在左前方，旧格式仍输出负角度。
        headings = iter((99.735, 29.605, 29.605))

        async def observe_turn():
            return b'\xff\xd8test-image', {
                'pose': {'x': 0, 'y': 0},
                'yaw': math.radians(next(headings)),
            }

        self.tools.observe = observe_turn
        first = navigation()
        first.update(action='look', direction='right')
        first.pop('candidate_id')
        first.pop('instruction')
        first['candidates'][0]['bearing_deg'] = -60
        self.responses = [first]
        instruction = 'Walk straight toward the open doorway in front-left and stop.'
        for step, bearing in enumerate((-30, -25), 2):
            proposal = navigation(instruction, candidate='C1', review_step=step if step > 2 else None)
            proposal['candidates'][0]['bearing_deg'] = bearing
            self.responses.append(proposal)

        record = await self.run_loop(steps=3)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(self.tools.actions, [
            ('look', 'right'), ('navigate', instruction), ('navigate', instruction),
        ])
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(record['exploration_summary']['candidates'], 1)
        context = json.loads(self.requests[1]['messages'][1]['content'][0]['text'])
        self.assertNotIn('heading_deg', context['execution']['local_candidates'][0])

    async def test_total_deadline_cleans_up(self):
        self.delay = 0.5
        self.responses = [{'action': 'finish', 'reason': 'done', 'evidence': 'test'}]
        request = main.TaskRequest.model_construct(goal='test', max_steps=2, max_seconds=0.1, navigation_seconds=15.)
        with patch.object(main, 'resolve_task', return_value=request):
            record = await main.start_task(request)
        await main.worker
        self.assertEqual((await main.get_task(record['id']))['status'], 'budget_exhausted')
        self.assertTrue(self.tools.closed)

    async def test_immediate_cancel_is_persisted(self):
        record = await main.start_task(main.TaskRequest(goal='test'))
        result = await main.cancel_task(record['id'])
        self.assertEqual(result['status'], 'cancelled')
        stored = json.loads((main.log_store().run_directory(result) / 'result.json').read_text())
        self.assertEqual(stored['status'], 'cancelled')

    async def test_external_stop_during_cloud_prevents_finish(self):
        self.responses = [{'action': 'finish', 'reason': 'done', 'evidence': 'test'}]

        def lost_control():
            raise main.ToolError('Robot control was interrupted')

        self.tools.require_manual = lost_control
        self.assertEqual((await self.run_loop())['status'], 'failed')

    async def test_spatial_memory_reaches_model_and_survives_run_rename(self):
        observation_count = 0

        async def moving_observation():
            nonlocal observation_count
            observation_count += 1
            return b'\xff\xd8image', {'pose': {'x': observation_count * 2, 'y': 0}, 'yaw': 0}

        self.tools.observe = moving_observation
        first = navigation('Walk straight toward the doorway ahead and stop.')
        first['scene'] = {
            'place_label': '疑似客厅',
            'objects': [{'name': '沙发', 'description': '右侧棕色沙发'}],
        }
        self.responses = [first, {
            'action': 'give_up', 'reason': '无新的可见通路',
            'previous_action_review': {'outcome': 'uncertain', 'evidence': '不能确认穿过门洞'},
        }]
        record = await self.run_loop(steps=2)
        content = self.requests[1]['messages'][1]['content']
        context = json.loads(content[0]['text'])['spatial_memory']
        self.assertEqual(context['objects'][0]['name'], '沙发')
        self.assertEqual(context['reference_reason'], 'previous_action_start')
        self.assertEqual(sum(part['type'] == 'image_url' for part in content), 2)
        events = [json.loads(line) for line in
                  (main.log_store().run_directory(record) / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        logged = next(item for item in events if item['type'] == 'decision_input' and item['step'] == 2)
        self.assertEqual(logged['model_context'], json.loads(content[0]['text']))
        self.assertNotIn('memory', logged['model_context'])
        self.assertNotIn('recent_actions', logged['model_context'])
        self.assertNotIn('places', logged['model_context']['execution'])
        self.assertFalse((main.log_store().run_directory(record) / 'exploration.json').exists())
        self.assertTrue((main.log_store().run_directory(record) / 'execution.json').exists())
        memory_page = main.log_store().run_directory(record) / 'memory_visualize.html'
        self.assertTrue(memory_page.exists())
        self.assertIn('沙发', memory_page.read_text(encoding='utf-8'))
        self.assertNotIn('log_rename_error', record)
        with closing(sqlite3.connect(main.log_store().run_directory(record) / 'memory.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM semantic_frames').fetchone()[0], 2)
            review = db.execute('SELECT review_json FROM semantic_frames WHERE step=2').fetchone()[0]
            self.assertEqual(json.loads(review)['outcome'], 'uncertain')

    async def test_startup_retry_exhausted_reports_service_error_without_candidate_failure(self):
        async def failed(instruction, *, seconds):
            return {
                'reason': 'navigation_error', 'startup_failed': True,
                'motion': {'displacement_m': 0},
                'attempts': [{'attempt': n, 'detail': 'VLN ERROR'} for n in range(1, 4)],
            }
        self.tools.navigate = failed
        self.responses = [navigation()]
        record = await self.run_loop()
        self.assertEqual(record['status'], 'failed')
        self.assertIn('导航服务启动失败', record['summary'])
        self.assertEqual(len(self.requests), 1)
        directory = main.log_store().run_directory(record)
        execution = json.loads((directory / 'execution.json').read_text())
        candidate = execution['local_candidates'][0]
        self.assertEqual(candidate['failed_attempts'], 0)
        self.assertEqual(candidate['attempt_count'], 0)
        self.assertEqual(candidate['status'], 'untried')
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        action = next(event for event in events if event['type'] == 'action_result')
        self.assertEqual(len(action['result']['attempts']), 3)

    async def test_exploration_session_scans_selects_turns_confirms_and_moves(self):
        pose = {'x': 0., 'y': 0., 'yaw': 0.}
        async def observe():
            return b'\xff\xd8view', {'pose': dict(pose), 'yaw': pose['yaw'], 'pose_valid': True, 'blocked': False}
        async def look(direction):
            pose['yaw'] += math.radians(75)
            self.tools.actions.append(('look', direction))
            return {'reason': 'turned'}
        async def orient(heading_deg):
            pose['yaw'] = math.radians(heading_deg)
            self.tools.actions.append(('orient', heading_deg))
            return {'reason': 'oriented'}
        async def navigate(instruction, *, seconds):
            pose['x'] += 1.
            self.tools.actions.append(('navigate', instruction))
            return {'reason': 'model_stop', 'motion': {'displacement_m': 1.}}
        self.tools.observe, self.tools.look = observe, look
        self.tools.orient, self.tools.navigate = orient, navigate
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '收集方向', 'candidates': []}
            for _ in range(4)
        ] + [
            {'action': 'orient', 'view_id': 'O1', 'reason': '转回准备接近首个开放过道',
             'observation_intent': {'reference_view_id': 'O1', 'question': '原过道当前路径是否仍可接近',
                                    'new_evidence': '比较当前可用方向后决定接近 O1 的开放过道'}},
            navigation(kind='open_passage'),
            {'action': 'look', 'direction': 'left', 'reason': '已进入新区域，开始新一轮',
             'previous_action_review': {'outcome': 'progress', 'view_change': 'new_area',
                                       'evidence': '经过门框，新图出现独立走廊'}},
            {'action': 'finish', 'reason': '找到', 'evidence': '当前图看到垃圾桶'},
        ]
        record = await self.run_loop(steps=8)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual([action[0] for action in self.tools.actions],
                         ['look'] * 4 + ['orient', 'navigate', 'look'])
        contexts = [json.loads(request['messages'][1]['content'][0]['text']) for request in self.requests]
        self.assertEqual(contexts[4]['execution']['exploration_session']['session_id'], 'S1')
        self.assertEqual(contexts[4]['execution']['local_candidates'], [])
        self.assertIn('O1', contexts[4]['execution']['exploration_session']['orient_view_ids'])
        self.assertEqual(contexts[6]['execution']['exploration_session']['session_id'], 'S1')
        self.assertEqual(contexts[7]['execution']['exploration_session']['session_id'], 'S1')
        events = [json.loads(line) for line in
                  (main.log_store().run_directory(record) / 'events.jsonl').read_text().splitlines()]
        self.assertFalse(any(event['type'] == 'exploration_session_finished'
                             and event['reason'] == 'new_area' for event in events))
        inputs = [event for event in events if event['type'] == 'decision_input']
        for event, request in zip(inputs, self.requests):
            content = request['messages'][1]['content']
            metadata = [json.loads(part['text']) for part in content[1:] if part['type'] == 'text']
            self.assertEqual(event['images'], metadata)
            self.assertEqual(len(event['images']), sum(part['type'] == 'image_url' for part in content))
        # 原地观察保留最多两个旧方向的原图，不要求旧方向已有候选，也不重发整圈。
        self.assertEqual([item['image'] for item in inputs[4]['images']],
                         ['images/003.jpg', 'images/002.jpg', 'images/005.jpg'])
        self.assertNotIn('coverage_complete', contexts[4]['execution']['scan'])
        navigation_image = next(item for item in inputs[6]['images']
                                if any(use['purpose'] == '真实上次导航起始图' for use in item['uses']))
        self.assertEqual(navigation_image['image'], 'images/006.jpg')
        report = (main.log_store().run_directory(record) / 'index.html').read_text()
        self.assertIn('src="images/006.jpg"', report)
        self.assertIn('决策时画面', report)
        self.assertNotIn('真实上次导航起始图', report)

    async def test_reverse_turn_is_corrected_to_continue_scan_on_the_same_frame(self):
        pose = {'x': 0., 'y': 0., 'yaw': 0.}

        async def observe():
            return b'frame', {'pose': dict(pose), 'yaw': pose['yaw'], 'pose_valid': True}

        async def look(direction):
            pose['yaw'] += math.radians(70 if direction == 'left' else -70)
            self.tools.actions.append(('look', direction))
            return {'reason': 'turned'}

        self.tools.observe, self.tools.look = observe, look
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '开始环视'},
            {'action': 'look', 'direction': 'right', 'reason': '未见目标，继续扩大视野',
             'scene': {'visible_ground': '前方有可接近的位置'}},
            {'action': 'look', 'direction': 'left', 'reason': '沿原方向继续环视'},
            {'action': 'finish', 'reason': '找到', 'evidence': '当前图看到垃圾桶的桶身和桶口'},
        ]
        record = await self.run_loop(steps=3)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(self.tools.actions, [('look', 'left'), ('look', 'left')])
        before_retry = self.requests[1]['messages'][1]
        self.assertEqual(self.requests[2]['messages'][1], before_retry)
        context = json.loads(before_retry['content'][0]['text'])
        self.assertEqual(context['execution']['action_options']['look_directions'], ['left', 'right'])
        self.assertFalse(context['execution']['scan']['required'])
        self.assertNotIn('look_count', context['execution']['scan'])
        self.assertIn('复核', self.requests[2]['messages'][-1]['content'])
        directory = main.log_store().run_directory(record)
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(len([e for e in events if e['type'] == 'observation']), 3)
        self.assertEqual([e['action'] for e in events if e['type'] == 'action_input'], ['look', 'look'])

    async def test_suspected_target_can_interrupt_scan_and_approach_for_confirmation(self):
        proposal = navigation('Walk straight toward the silver cylindrical object in front-left and stop.')
        proposal['scene']['target_seen'] = 'uncertain'
        proposal['candidates'][0].update(
            description='柜旁银色圆柱物',
            evidence='露出银色圆柱轮廓，桶口遮挡，需接近确认',
            expected_new_view='',
        )
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '环视寻找目标'},
            proposal,
            {'action': 'finish', 'reason': '接近后确认', 'evidence': '新图可见银色桶身和桶口',
             'target_review': target_review(3, 'arrived')},
        ]
        record = await self.run_loop()
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(self.requests), 3)
        self.assertEqual([action[0] for action in self.tools.actions], ['look', 'navigate'])

    async def test_rejected_early_exploration_keeps_visual_clue_in_correction_context(self):
        proposal = navigation(kind='open_passage')
        # 现在允许直接探索；用非法历史引用继续覆盖视觉报告纠错。
        proposal['view_id'] = 'O99'
        # 环视尚未完成时，拒绝提前探索仍须保留有效视觉报告。
        proposal['scene'] = {'target_seen': 'no', 'visible_ground': '左前方有开放门洞',
                             'objects': [{'name': '门洞', 'description': '左前方可见门框'}]}
        proposal['candidates'][0]['description'] = '左前方开放门洞'
        self.responses = [
            proposal,
            {'action': 'look', 'direction': 'left', 'reason': '保留门洞线索，继续环视',
             'candidates': proposal['candidates']},
        ]
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'budget_exhausted')
        self.assertEqual(self.tools.actions, [('look', 'left')])
        messages = self.requests[1]['messages']
        self.assertEqual(json.loads(messages[-2]['content']), proposal)
        self.assertIn('上次动作未执行', messages[-1]['content'])
        self.assertIn('"orient_view_ids": []', messages[-1]['content'])
        context = json.loads(messages[1]['content'][0]['text'])['execution']
        self.assertEqual(context['local_candidates'], [])
        directory = main.log_store().run_directory(record)
        execution = json.loads((directory / 'execution.json').read_text())
        self.assertEqual(len(execution['local_candidates']), 1)
        with closing(sqlite3.connect(directory / 'memory.sqlite3')) as db:
            scene, source = db.execute('SELECT scene_json, report_status FROM semantic_frames').fetchone()
            self.assertEqual(json.loads(scene)['visible_ground'], '左前方有开放门洞')
            self.assertEqual(source, 'rejected_action')
            self.assertEqual(db.execute('SELECT action FROM trajectory_actions').fetchall(), [('look',)])
            self.assertEqual(db.execute('SELECT count(*) FROM objects').fetchone()[0], 1)

    async def test_exhausted_actions_get_one_fresh_observation_before_continuing(self):
        invalid = {'action': 'orient', 'view_id': 'O99', 'reason': '无效历史引用'}
        self.responses = [invalid, invalid,
                          {'action': 'look', 'direction': 'left', 'reason': '重新查看左侧'},
                          {'action': 'finish', 'reason': '看到了', 'evidence': '当前目标可见'}]
        record = await self.run_loop(steps=3)
        self.assertEqual(record['status'], 'completed', record)
        self.assertEqual(self.tools.actions, [('look', 'left')])
        fresh = json.loads(self.requests[2]['messages'][1]['content'][0]['text'])
        self.assertEqual(fresh['observation_step'], 2)
        self.assertIn('decision_recovery', fresh)
        self.assertIsNone(fresh['execution']['last_action'])
        directory = main.log_store().run_directory(record)
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(event['type'] == 'decision_replan' for event in events), 1)
        with closing(sqlite3.connect(directory / 'memory.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT action FROM trajectory_actions').fetchall(), [('look',)])

    async def test_replanning_exhaustion_does_not_loop_forever(self):
        invalid = {'action': 'orient', 'view_id': 'O99', 'reason': '无效历史引用'}
        self.responses = [invalid] * 4
        record = await self.run_loop(steps=8)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(self.tools.actions, [])

    async def test_local_give_up_gets_one_reconsideration_without_forced_motion(self):
        async def unlocated_observation():
            return b'image', {'pose_valid': False}
        self.tools.observe = unlocated_observation
        give_up = {'action': 'give_up', 'reason': '当前暂未找到可继续的方向'}
        self.responses = [navigation(kind='open_passage'), give_up, give_up]
        record = await self.run_loop(steps=5)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['summary'], give_up['reason'])
        self.assertEqual(len(self.tools.actions), 1)
        fresh = json.loads(self.requests[2]['messages'][1]['content'][0]['text'])
        self.assertIn('decision_recovery', fresh)
        self.assertEqual(fresh['execution']['last_action']['action'], 'navigate')

    async def test_exhausted_action_retries_preserve_report_without_executed_motion(self):
        invalid = navigation(kind='open_passage')
        invalid['candidate_id'] = 'new_2'
        invalid['scene'] = {'visible_ground': '当前左侧可见门前空地 test-secret'}
        self.responses = [invalid, {'action': 'orient', 'view_id': 'O99', 'reason': '无效历史引用'}]
        record = await self.run_loop(steps=1)
        self.assertEqual(record['status'], 'failed')
        self.assertIn('当前 action_options', record['summary'])
        self.assertEqual(self.tools.actions, [])
        self.assertEqual(len(self.requests), 2)
        directory = main.log_store().run_directory(record)
        with closing(sqlite3.connect(directory / 'memory.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM trajectory_actions').fetchone()[0], 0)
            report, status = db.execute('SELECT scene_json, report_status FROM semantic_frames').fetchone()
        self.assertEqual(json.loads(report)['visible_ground'], '当前左侧可见门前空地 [REDACTED]')
        self.assertEqual(status, 'rejected_action')
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        errors = [event for event in events if event['type'] == 'decision_error']
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[-1]['action_options']['orient_view_ids'], [])
        self.assertNotIn('test-secret', (directory / 'events.jsonl').read_text())

    async def test_trajectory_reaches_model_without_extra_images_or_rejected_actions(self):
        points = [[i / 4, 0.] for i in range(9)]

        async def observe():
            moved = any(action[0] == 'navigate' for action in self.tools.actions)
            return b'\xff\xd8test-image', {
                'pose': {'x': 2. if moved else 0., 'y': 0., 'yaw': 0.},
                'yaw': 0., 'pose_valid': True, 'blocked': False,
            }

        async def navigate(instruction, *, seconds):
            self.tools.actions.append(('navigate', instruction))
            return {'reason': 'model_stop', 'motion': {'sampled_positions': points}}

        self.tools.observe, self.tools.navigate = observe, navigate
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '查看左侧'},
            navigation(review_step=None),
            {'action': 'teleport', 'reason': '无效动作'},
            {'action': 'look', 'direction': 'right', 'reason': '局部确认右边缘',
             'target_review': target_review(3, 'temporarily_lost'),
             'previous_action_review': {'outcome': 'uncertain', 'evidence': '仅见目标局部'}},
            {'action': 'finish', 'reason': '确认到达', 'evidence': '当前可见沙发靠背和座面',
             'target_review': target_review(4, 'arrived')},
        ]
        record = await self.run_loop(steps=4)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(self.tools.actions), 3)
        contexts = [json.loads(request['messages'][1]['content'][0]['text']) for request in self.requests]
        self.assertEqual(contexts[0]['trajectory_memory']['recent'], [])
        self.assertEqual(len(contexts[1]['trajectory_memory']['recent']), 1)
        self.assertIn('底层停止，到达未确认', contexts[2]['trajectory_memory']['recent'][-1])
        self.assertEqual(contexts[2]['trajectory_memory'], contexts[3]['trajectory_memory'])
        path = contexts[2]['trajectory_memory']['local_paths']['segments'][0]
        self.assertEqual(path['action_step'], 2)
        self.assertEqual(path['directions'][0]['direction'], '后方')
        self.assertIn('当前身体正前方', self.requests[2]['messages'][0]['content'])
        self.assertNotIn('sampled_positions', json.dumps(contexts))
        self.assertIn('目标暂时不可见', contexts[4]['trajectory_memory']['recent'][-2])
        self.assertIn('进展不确定', contexts[4]['trajectory_memory']['recent'][-2])
        for context in contexts:
            self.assertNotIn('actions', context['execution']['exploration_session'])
            self.assertNotIn('recent_reviews', context['spatial_memory'])
        directory = main.log_store().run_directory(record)
        with closing(sqlite3.connect(directory / 'memory.sqlite3')) as db:
            rows = db.execute(
                'SELECT action_step, action, observed_step, review_json FROM trajectory_actions ORDER BY action_step'
            ).fetchall()
        self.assertEqual([(row[0], row[1], row[2]) for row in rows],
                         [(1, 'look', 2), (2, 'navigate', 3), (3, 'look', 4)])
        self.assertEqual(json.loads(rows[-1][3])['target']['status'], 'arrived')
        with closing(sqlite3.connect(directory / 'memory.sqlite3')) as db:
            result = json.loads(db.execute(
                'SELECT result_json FROM trajectory_actions WHERE action_step=2',
            ).fetchone()[0])
        self.assertEqual(result['motion']['sampled_positions'], points)
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        inputs = [event for event in events if event['type'] == 'decision_input']
        self.assertEqual(len(inputs), 4)
        for event in inputs:
            self.assertFalse(any('轨迹' in use['purpose']
                                 for image in event['images'] for use in image['uses']))
        execution = json.loads((directory / 'execution.json').read_text())
        self.assertEqual(execution['exploration_session']['previous_result']['last_action']['action'], 'look')
        self.assertEqual(execution['last_navigation']['result']['motion']['sampled_positions'], points)

    async def test_navigation_starts_scan_before_new_area_review(self):
        pose = {'x': 0., 'y': 0., 'yaw': 0.}

        async def observe():
            return b'\xff\xd8view', {'pose': dict(pose), 'yaw': pose['yaw'], 'pose_valid': True}

        async def look(direction):
            pose['yaw'] += math.radians(75 if direction == 'left' else -75)
            self.tools.actions.append(('look', direction))
            return {'reason': 'turned'}

        async def navigate(instruction, *, seconds):
            pose['x'] += 1.
            self.tools.actions.append(('navigate', instruction))
            return {'reason': 'model_stop', 'motion': {'displacement_m': 1.}}

        self.tools.observe, self.tools.look, self.tools.navigate = observe, look, navigate
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '收集方向'} for _ in range(4)
        ] + [
            navigation(kind='open_passage'),
            {'action': 'look', 'direction': 'right', 'reason': '补看入口旁边',
             'previous_action_review': {'outcome': 'uncertain', 'evidence': '停止后暂时无法判断区域'}},
            {'action': 'look', 'direction': 'right', 'reason': '继续同向查看新区域',
             'observation_intent': {'reference_view_id': 'O6', 'question': '门框旁新露出的开口是否连接过道',
                                    'new_evidence': '补看后发现柜侧有新的地面连接，需转回确认与原门框的关系'},
             'previous_action_review': {'outcome': 'progress', 'view_change': 'new_area',
                                       'evidence': '对比导航前图，已越过门框，补看见独立走廊'}},
            {'action': 'finish', 'reason': '发现目标', 'evidence': '当前可见棕色沙发'},
        ]
        record = await self.run_loop(steps=8)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(self.requests), 8)
        contexts = [json.loads(request['messages'][1]['content'][0]['text']) for request in self.requests]
        self.assertEqual(contexts[6]['execution']['exploration_session']['session_id'], 'S1')
        self.assertNotIn('translation_for_review', contexts[6]['execution'])
        self.assertEqual(contexts[7]['execution']['exploration_session']['session_id'], 'S1')
        self.assertNotIn('look_count', contexts[7]['execution']['scan'])
        execution = json.loads((main.log_store().run_directory(record) / 'execution.json').read_text())
        self.assertEqual(execution['scan']['look_count'], 6)
        recent = contexts[7]['trajectory_memory']['recent']
        navigation_summary = next(item for item in recent if item.startswith('5→6'))
        confirmation_summary = next(item for item in recent if item.startswith('6→7'))
        self.assertIn('进展不确定', navigation_summary)
        self.assertIn('有进展', confirmation_summary)
        content = self.requests[6]['messages'][1]['content']
        metadata = [json.loads(item['text']) for item in content[1:] if item['type'] == 'text']
        self.assertTrue(any(item['image'] == 'images/005.jpg'
                            and any(use['purpose'] == '真实上次导航起始图' for use in item['uses'])
                            for item in metadata))


if __name__ == '__main__':
    unittest.main()
