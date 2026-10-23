"""保留原动作基线，并逐项检查新上下文、图片与日志证据的一致性。"""

import json
from pathlib import Path
import unittest
from tests.agent_tests import test_loop


def load_tests(loader, tests, pattern):
    baseline = json.loads(Path(__file__).with_name('refactor_behavior.json').read_text())['cases']

    class BehaviorCase(test_loop.LoopTests):
        async def asyncTearDown(self):
            try:
                record = next(iter(test_loop.main.records.values()))
                directory = test_loop.main.log_store().run_directory(record)
                events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
                for event in events:
                    event.pop('time', None)
                actual = {
                    'actions': self.tools.actions,
                    'result': {key: record.get(key) for key in (
                        'status', 'summary', 'step', 'last_decision', 'last_result', 'exploration_result')},
                }
                expected = baseline[self._testMethodName]
                if self._testMethodName == 'test_invalid_cloud_action_never_moves':
                    # 删除独立监测动作后，只允许枚举错误提示发生这一处变化。
                    expected = {**expected, 'result': dict(expected['result'])}
                    summary = expected['result']['summary']
                    self.assertIn("'finish', 'give_up' or 'watch_target'", summary)
                    expected['result']['summary'] = summary.replace(
                        "'finish', 'give_up' or 'watch_target'", "'finish' or 'give_up'")
                self.assertEqual(json.loads(json.dumps(actual['actions'])), expected['actions'])
                for key in ('status', 'summary', 'step', 'last_result'):
                    self.assertEqual(actual['result'][key], expected['result'][key])
                decision = dict(actual['result']['last_decision'] or {})
                decision.pop('target_review', None)
                if decision.get('action') == 'navigate':
                    self.assertIn(decision.pop('skill'), ('search_navigation', 'approach_target'))
                if decision.get('navigation_check'):
                    self.assertEqual(decision['navigation_check'].pop('space'), 'normal')
                # 新协议显式标注当前可见；动作、结果及原候选内容仍与原基线逐项一致。
                for candidate in decision.get('candidates', []):
                    self.assertEqual(candidate.pop('evidence_source'), 'current')
                self.assertEqual(decision or None, expected['result']['last_decision'])
                # 请求结构有意调整：用可解释的证据对应断言代替重录整包 hash。
                inputs = {event['step']: event for event in events if event['type'] == 'decision_input'}
                for body in self.requests:
                    content = body['messages'][1]['content']
                    context = json.loads(content[0]['text'])
                    event = inputs[context['observation_step']]
                    self.assertEqual(context, event['model_context'])
                    metadata = [json.loads(item['text']) for item in content[1:] if item['type'] == 'text']
                    self.assertEqual(metadata, event['images'])
                    self.assertEqual(sum(item['type'] == 'image_url' for item in content), len(metadata))
                    # 当前图有意放在历史之后；请求与日志仍使用同一清单。
                    self.assertEqual(metadata[-1]['image'], event['image'])
                    for frame in context['spatial_memory']['semantic_frames']:
                        self.assertIn(frame['image'], [item['image'] for item in metadata])
            finally:
                await super().asyncTearDown()

    return unittest.TestSuite(BehaviorCase(name) for name in baseline)
