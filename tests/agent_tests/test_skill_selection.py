"""验证模型可见的技能选择、同轮工具执行与监测生命周期。"""

import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone

from tests.config_support import legacy_search_run_task as run_task
from navigation_agent.agent.schemas import Decision, TaskRequest
from navigation_agent.agent.skills.registry import SkillRegistry
from navigation_agent.agent.tools import NavigationTools
from navigation_agent.app.agent.demo import DemoBackend, DemoModel
from tests.agent_tests.test_loop import navigation
from tests.agent_tests import test_target_watch as watch_tests
from tests.config_support import agent_settings


class SkillExecutionTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = watch_tests.WatchTests.asyncSetUp
    asyncTearDown = watch_tests.WatchTests.asyncTearDown
    navigate = watch_tests.WatchTests.navigate

    async def test_search_starts_monitor_in_same_navigation_and_approach_stops_it(self):
        skills = SkillRegistry()
        decision = Decision.model_validate({
            **navigation(kind='open_passage'), 'target_labels': ['sofa'],
        })
        await self.watch.configure(False, [])
        skills.validate(decision, detector_available=True)
        monitoring = await skills.prepare(decision, self.watch)
        self.assertTrue(monitoring)
        await self.navigate(search=monitoring)
        self.assertIsNotNone(self.watch.pending)
        calls = self.client.calls

        # 相同目的地类型也可由模型明确选择核实；不靠候选类型或 scene 来路由。
        approach = Decision.model_validate({
            **navigation(kind='open_passage'), 'skill': 'approach_target',
            'target_labels': ['sofa'],
        })
        skills.validate(approach, detector_available=True)
        monitoring = await skills.prepare(approach, self.watch)
        self.assertFalse(monitoring)
        await self.navigate(search=monitoring)
        self.assertEqual(self.client.calls, calls)
        self.assertEqual(len(self.tools.interrupts), 1)

        monitoring = await skills.prepare(decision, self.watch)
        await self.navigate(search=monitoring)
        self.assertEqual(len(self.tools.interrupts), 2)

    async def test_observation_does_not_inherit_search_monitor(self):
        await self.navigate()
        decision = Decision(action='look', direction='left', reason='查看侧方',
                            skill='search_navigation', target_labels=['sofa'])
        skills = SkillRegistry()
        skills.validate(decision, detector_available=True)
        await skills.prepare(decision, self.watch)
        calls = self.client.calls
        await self.navigate(search=False)
        self.assertEqual(self.client.calls, calls)
        self.assertFalse(self.watch.enabled)
        self.assertIsNone(self.watch.task)

    async def test_unsupported_goal_or_unconfigured_service_keeps_navigation(self):
        skills = SkillRegistry()
        for configured, labels in ((True, []), (False, ['sofa'])):
            with self.subTest(configured=configured):
                self.watch.settings = self.watch.settings.model_copy(update={
                    'server_url': 'http://localhost:8060' if configured else '',
                })
                decision = Decision.model_validate({
                    **navigation(kind='open_passage'), 'target_labels': labels,
                })
                skills.validate(decision, detector_available=configured)
                monitoring = await skills.prepare(decision, self.watch)
                calls = self.client.calls
                self.assertEqual((await self.navigate(search=monitoring))['reason'], 'model_stop')
                self.assertEqual(self.client.calls, calls)


class SelectionValidationTests(unittest.TestCase):
    def test_missing_or_unknown_skill_does_not_silently_pick_a_workflow(self):
        skills = SkillRegistry()
        for name in (None, 'object_search', 'made_up_skill'):
            with self.subTest(name=name):
                proposal = {**navigation(), 'skill': name}
                with self.assertRaises(ValueError):
                    skills.validate(Decision.model_validate(proposal), detector_available=True)

    def test_search_requires_labels_only_when_detector_is_configured(self):
        skills = SkillRegistry()
        decision = Decision.model_validate(navigation(kind='open_passage'))
        with self.assertRaisesRegex(ValueError, 'target_labels'):
            skills.validate(decision, detector_available=True)
        skills.validate(decision, detector_available=False)

    def test_unused_labels_do_not_block_the_selected_skill_or_public_tool(self):
        skills = SkillRegistry()
        skills.validate(Decision.model_validate({**navigation(), 'target_labels': ['sofa']}),
                        detector_available=True)
        for name in (None, 'search_navigation', 'approach_target'):
            with self.subTest(skill=name):
                decision = Decision(action='look', direction='left', reason='继续观察',
                                    skill=name, target_labels=['toilet'])
                skills.validate(decision, detector_available=True)

    def test_old_standalone_monitor_action_is_no_longer_in_model_protocol(self):
        with self.assertRaises(ValueError):
            Decision(action='watch_target', watch={'enabled': True, 'labels': ['sofa']}, reason='旧格式')


class SelectionLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_skill_is_corrected_on_same_frame_before_any_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = agent_settings()
            settings = settings.model_copy(update={
                'storage': settings.storage.model_copy(update={'runs_dir': Path(directory)}),
            })
            model, backend = DemoModel(), DemoBackend()
            model.responses = [
                {**navigation(kind='open_passage'), 'skill': 'made_up_skill'},
                navigation(kind='open_passage'),
                {'action': 'finish', 'reason': '找到', 'evidence': '当前图支持目标身份'},
            ]
            record = {'id': 'skills', 'goal': '找到沙发', 'status': 'running',
                      'started_at': datetime.now(timezone.utc).isoformat()}
            request = TaskRequest(goal='找到沙发', max_steps=2, max_seconds=10, navigation_seconds=5.)
            (Path(directory) / record['id'] / 'images').mkdir(parents=True)
            await run_task(record, request, NavigationTools(backend), settings=settings,
                           api_key='', model=model)
            self.assertEqual(record['status'], 'completed', record)
            self.assertEqual(len(backend.actions), 1)
            self.assertEqual(model.requests[0]['messages'][1], model.requests[1]['messages'][1])
            first = json.loads(model.requests[0]['messages'][1]['content'][0]['text'])
            final = json.loads(model.requests[-1]['messages'][1]['content'][0]['text'])
            self.assertEqual({item['name'] for item in first['available_tools']},
                             {'navigate', 'look', 'orient', 'recover'})
            self.assertTrue(all(item['instructions'] for item in first['available_skills']))
            self.assertEqual(final['previous_skill'], 'search_navigation')
            self.assertEqual(final['execution']['last_action']['action'], 'navigate')
            events = [json.loads(line) for path in Path(directory).rglob('events.jsonl')
                      for line in path.read_text().splitlines()]
            selections = [event for event in events if event['type'] == 'skill_selected']
            self.assertEqual([item['name'] for item in selections], ['search_navigation', None])
            self.assertFalse(any(event.get('action') == 'watch_target' for event in events))
