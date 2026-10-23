"""通过非默认值验证消费者，避免只测试配置能被解析。"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from navigation_agent.app.agent import api
from navigation_agent.agent.skills.object_search.state import ExecutionState
from navigation_agent.agent.context.prompts import navigation_prompt
from navigation_agent.agent.config import ScenePromptSettings
from navigation_agent.agent.memory.spatial import SpatialMemory
from navigation_agent.agent.skills.object_search.rules import classify_result
from navigation_agent.agent.schemas import TaskRequest, Decision
from navigation_agent.agent.storage.logs import LogStore
from tests.config_support import agent_settings, test_runtime
from tests.agent_tests import test_loop as loop_fixtures
from tests.agent_tests.test_loop import navigation, target_review
from tests.agent_tests.test_exploration import observe, decision


class ConfigurationLoopTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = loop_fixtures.LoopTests.asyncSetUp
    asyncTearDown = loop_fixtures.LoopTests.asyncTearDown
    run_loop = loop_fixtures.LoopTests.run_loop

    async def test_guide_instruction_reaches_navigation_without_rewriting(self):
        from navigation_agent.vln.lightnav.instructions import INSTRUCTION_GUIDE

        for instruction in ('Approach the doorway.', 'Turn right, then walk to the doorway and stop.'):
            with self.subTest(instruction=instruction):
                self.requests.clear()
                self.tools.actions.clear()
                self.responses = [
                    navigation(instruction, kind='open_passage'),
                    {'action': 'finish', 'reason': '找到', 'evidence': '当前图确认棕色沙发'},
                ]
                runtime = self.runtime(prompt={'instruction_guide': INSTRUCTION_GUIDE})
                with patch.object(api, 'current_settings', return_value=runtime):
                    record = await self.run_loop(steps=2)
                self.assertEqual(record['status'], 'completed')
                self.assertEqual(len(self.requests), 2)
                self.assertIn(INSTRUCTION_GUIDE, self.requests[0]['messages'][0]['content'])
                self.assertIn(('navigate', instruction), self.tools.actions)

    def runtime(self, **groups):
        runtime = test_runtime(Path(self.temp.name))
        agent = runtime.agent.model_copy(update={name:getattr(runtime.agent,name).model_copy(update=values)
                                               for name,values in groups.items()})
        return replace(runtime,agent=agent)

    async def test_nondefault_request_options_task_budget_and_log_timezone(self):
        runtime = self.runtime(llm={'model':'another-model','max_tokens':321,'enable_thinking':True},
            execution={'navigation_seconds':7.5},task_defaults={'max_steps':2,'max_seconds':40},
            storage={'utc_offset_minutes':-300})
        self.responses = [navigation('Walk straight toward the corridor ahead and stop.'),{'action':'finish','reason':'完成','evidence':'看到沙发','target_review':target_review(2,'arrived')}]
        budgets=[]
        original=self.tools.navigate
        async def navigate_with_budget(instruction,*,seconds):
            budgets.append(seconds)
            return await original(instruction,seconds=seconds)
        self.tools.navigate=navigate_with_budget
        with patch.object(api,'current_settings',return_value=runtime):
            record=await api.start_task(TaskRequest(goal='找到沙发'))
            await api.worker
            result=await api.get_task(record['id'])
            self.assertEqual(result['status'],'completed')
            self.assertEqual((result['max_steps'],result['max_seconds']),(2,40))
            self.assertEqual(budgets,[7.5])
            payload=self.requests[0]
            self.assertEqual((payload['model'],payload['max_tokens'],payload['enable_thinking']),('another-model',321,True))
            context=json.loads(payload['messages'][1]['content'][0]['text'])
            self.assertEqual(context['execution']['navigation_seconds'],7.5)
            self.assertEqual(LogStore(runtime.agent.storage).log_time('2026-01-01T00:00:00+00:00'),'2025-12-31_19-00-00')
            self.assertIn('git_revision',result['configuration']['version'])
            self.assertNotIn('test-secret',json.dumps(result))

    async def test_room_comparison_uses_stopped_images_then_supplies_decision_context(self):
        runtime = self.runtime(memory={'compare_regions_when_stopped': True})
        first = navigation(kind='open_passage')
        first['spatial_update'] = {'region_id': 'R1', 'transition': 'same',
                                  'label': '木桌与窗户', 'evidence': '当前可见木桌和窗户'}
        finish = {'action': 'finish', 'reason': '当前找到', 'evidence': '当前地面可见目标沙发',
                  'scene': {'target_seen': 'yes'},
                  'spatial_update': {'region_id': 'new', 'transition': 'entered',
                                     'label': '沙发与门框', 'evidence': '实际穿门后看到不同布局'}}
        comparison = {'region_id': 'new', 'label': '沙发与门框', 'evidence': '当前布局与木桌房间不同'}
        self.responses = [first, comparison, finish]
        moved = False
        async def observe():
            return b'\\xff\\xd8image', {'pose': {'x': 2. if moved else 0., 'y': 0., 'yaw': 0.},
                                        'pose_valid': True, 'yaw': 0., 'blocked': False}
        async def navigate_with_motion(instruction, *, seconds):
            nonlocal moved
            self.tools.actions.append(('navigate', instruction))
            moved = True
            return {'reason': 'model_stop', 'motion': {'displacement_m': 2.}}
        self.tools.observe = observe
        self.tools.navigate = navigate_with_motion
        with patch.object(api, 'current_settings', return_value=runtime):
            record = await self.run_loop(steps=2)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(self.requests), 3)
        comparison_request = self.requests[1]
        self.assertEqual(comparison_request['max_tokens'], 400)
        self.assertNotIn('active_subgoal', json.dumps(comparison_request))
        context = json.loads(self.requests[2]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['exploration_memory']['visual_match'], comparison)
        events = [json.loads(line) for line in
                  (api.log_store().run_directory(record) / 'events.jsonl').read_text().splitlines()]
        types = [event['type'] for event in events]
        comparison_index = types.index('region_comparison')
        self.assertEqual(events[comparison_index]['step'], 2)
        self.assertEqual(events[comparison_index]['reference_images'], ['images/001.jpg'])
        self.assertLess(next(i for i, event in enumerate(events)
                             if event['type'] == 'observation' and event['step'] == 2), comparison_index)
        self.assertGreater(comparison_index, types.index('action_latency'))

    async def test_old_passage_match_reaches_decision_after_a_real_recorded_crossing(self):
        runtime = self.runtime(memory={'compare_regions_when_stopped': True})
        first = navigation(kind='open_passage')
        first['spatial_update'] = {'region_id': 'R1', 'transition': 'same',
                                  'label': '木桌与窗户', 'evidence': '当前房间特征'}
        comparison = {'region_id': 'new', 'label': '床与书柜', 'evidence': '门后是另一房间'}
        turn = {'action': 'look', 'direction': 'left', 'reason': '查看未看区域',
                'subgoal_review': {'status': 'completed', 'evidence': '已经进入门后房间'},
                'spatial_update': {'region_id': 'new', 'transition': 'entered',
                                   'label': '床与书柜', 'evidence': '已实际进入房间'}}
        route_match = {'matches': [{'route_id': 'C1', 'evidence': '当前左侧是刚走过的木框入口'}]}
        finish = {'action': 'finish', 'reason': '当前找到', 'evidence': '当前看见目标',
                  'scene': {'target_seen': 'yes'}}
        self.responses = [first, comparison, turn, route_match, finish]
        moved = False
        async def observe():
            yaw = math.radians(70 * sum(action[0] == 'look' for action in self.tools.actions))
            return b'image', {'pose': {'x': 2. if moved else 0., 'y': 0., 'yaw': yaw},
                              'pose_valid': True, 'yaw': yaw, 'blocked': False}
        async def navigate_with_motion(instruction, *, seconds):
            nonlocal moved
            moved = True
            return {'reason': 'model_stop', 'motion': {'displacement_m': 2.}}
        self.tools.observe = observe
        self.tools.navigate = navigate_with_motion
        with patch.object(api, 'current_settings', return_value=runtime):
            record = await self.run_loop(steps=3)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(len(self.requests), 5)
        context = json.loads(self.requests[-1]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['exploration_memory']['visible_known_routes'], route_match['matches'])
        self.assertIn('相同入口沿用该 C 编号', self.requests[-1]['messages'][0]['content'])
        self.assertEqual(self.tools.actions, [('look', 'left')])

    async def test_two_occluded_views_do_not_prevent_comparison_when_the_room_becomes_visible(self):
        runtime = self.runtime(memory={'compare_regions_when_stopped': True})
        first = navigation(kind='open_passage')
        first['spatial_update'] = {'region_id': 'R1', 'transition': 'same',
                                  'label': '木桌与窗户', 'evidence': '起始房间特征'}
        unclear = {'region_id': 'uncertain', 'label': '墙角', 'evidence': '近处墙面遮挡视野'}
        turn = {'action': 'look', 'direction': 'left', 'reason': '转向寻找房间特征',
                'spatial_update': {'region_id': 'new', 'transition': 'uncertain',
                                   'evidence': '当前是墙角'}}
        visible = {'region_id': 'new', 'label': '床与书柜', 'evidence': '第三次观察看清不同房间布局'}
        finish = {'action': 'finish', 'reason': '当前找到', 'evidence': '当前看见目标',
                  'scene': {'target_seen': 'yes'},
                  'spatial_update': {'region_id': 'new', 'transition': 'entered',
                                     'evidence': '已进入床与书柜所在的新房间'}}
        self.responses = [first, unclear, turn, unclear, turn, visible, finish]
        moved = False
        async def observe():
            yaw = math.radians(70 * sum(action[0] == 'look' for action in self.tools.actions))
            return b'image', {'pose': {'x': 2. if moved else 0., 'y': 0., 'yaw': yaw},
                              'pose_valid': True, 'yaw': yaw, 'blocked': False}
        async def navigate_with_motion(instruction, *, seconds):
            nonlocal moved
            moved = True
            return {'reason': 'model_stop', 'motion': {'displacement_m': 2.}}
        self.tools.observe = observe
        self.tools.navigate = navigate_with_motion
        with patch.object(api, 'current_settings', return_value=runtime):
            record = await self.run_loop(steps=4)
        self.assertEqual(record['status'], 'completed')
        context = json.loads(self.requests[-1]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['exploration_memory']['visual_match'], visible)
        events = [json.loads(line) for line in
                  (api.log_store().run_directory(record) / 'events.jsonl').read_text().splitlines()]
        comparisons = [event for event in events if event['type'] == 'region_comparison']
        self.assertEqual([event['step'] for event in comparisons], [2, 3, 4])

    async def test_unavailable_room_comparison_falls_back_without_inventing_a_match(self):
        runtime = self.runtime(memory={'compare_regions_when_stopped': True})
        first = navigation(kind='open_passage')
        first['spatial_update'] = {'region_id': 'R1', 'transition': 'same',
                                  'label': '木桌与窗户', 'evidence': '当前可见木桌和窗户'}
        self.responses = [first, {'action': 'finish', 'reason': '找到', 'evidence': '当前看到沙发',
                                  'scene': {'target_seen': 'yes'}}]
        observed = 0
        async def observe():
            nonlocal observed
            observed += 1
            return b'image', {'pose': {'x': float(observed), 'y': 0., 'yaw': 0.},
                              'pose_valid': True, 'yaw': 0., 'blocked': False}
        async def navigate_with_motion(instruction, *, seconds):
            return {'reason': 'model_stop', 'motion': {'displacement_m': 1.}}
        self.tools.observe = observe
        self.tools.navigate = navigate_with_motion
        from unittest.mock import AsyncMock
        with patch.object(api, 'current_settings', return_value=runtime), patch(
                'navigation_agent.agent.core.loop.compare_regions',
                AsyncMock(side_effect=TimeoutError('private details'))) as comparison:
            record = await self.run_loop(steps=2)
        self.assertEqual(record['status'], 'completed')
        comparison.assert_awaited_once()
        context = json.loads(self.requests[-1]['messages'][1]['content'][0]['text'])
        self.assertNotIn('visual_match', context['exploration_memory'])
        events = (api.log_store().run_directory(record) / 'events.jsonl').read_text()
        self.assertIn('region_comparison_error', events)
        self.assertNotIn('private details', events)

    async def test_continuous_navigation_prompt_and_context_keep_tool_parameters(self):
        runtime = self.runtime(
            prompt={'navigation_time_limit_enabled': False},
            execution={'navigation_seconds': 7.5},
        )
        self.responses = [
            navigation(),
            {'action': 'finish', 'reason': '完成', 'evidence': '看到沙发',
             'target_review': target_review(2, 'arrived')},
        ]
        budgets = []
        original = self.tools.navigate

        async def navigate_with_budget(instruction, *, seconds):
            budgets.append(seconds)
            return await original(instruction, seconds=seconds)

        self.tools.navigate = navigate_with_budget
        with patch.object(api, 'current_settings', return_value=runtime):
            record = await self.run_loop(steps=2)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(budgets, [7.5])
        for payload in self.requests:
            prompt = payload['messages'][0]['content']
            self.assertIn('单段导航时限：不设固定上限', prompt)
            self.assertNotIn('最多 7.5 秒', prompt)
            self.assertNotIn('{{', prompt)
            context = json.loads(payload['messages'][1]['content'][0]['text'])
            self.assertFalse(context['navigation_settings']['time_limit_enabled'])
            self.assertIsNone(context['navigation_settings']['seconds'])
            self.assertIsNone(context['execution']['navigation_seconds'])
            self.assertIn(runtime.agent.prompt.instruction_guide, prompt)
            self.assertNotIn('instruction_guide', context['navigation_settings'])

    async def test_format_retry_count_and_redacted_truncated_diagnostics(self):
        self.responses=['test-secret ' + 'x'*100]*3
        runtime=self.runtime(llm={'format_retries':2},storage={'max_error_output_chars':20})
        with patch.object(api,'current_settings',return_value=runtime):
            record=await self.run_loop(steps=1)
            self.assertEqual(record['status'],'failed')
            self.assertEqual(len(self.requests),3)
            events=[json.loads(line) for line in (api.log_store().run_directory(record)/'events.jsonl').read_text().splitlines()]
            errors=[item for item in events if item['type']=='decision_error']
            self.assertEqual(len(errors),3)
            for error in errors:
                self.assertEqual(len(error['raw_output']),20)
                self.assertTrue(error['truncated'])
                self.assertNotIn('test-secret',str(error))

    async def test_request_timeout_does_not_retry_motion_or_network(self):
        self.delay=.2
        self.responses=[{'action':'give_up','reason':'测试'}]
        runtime=self.runtime(llm={'request_timeout_s':.03,'format_retries':4})
        with patch.object(api,'current_settings',return_value=runtime):
            record=await self.run_loop(steps=1)
        self.assertEqual(record['status'],'budget_exhausted')
        self.assertEqual(len(self.requests),1)
        self.assertEqual(self.tools.actions,[])

    async def test_prompt_hot_read_and_fixed_modes_record_exact_content(self):
        prompt=Path(self.temp.name)/'custom prompt.md'
        scene=Path(self.temp.name)/'scene.md'
        original=self.tools.look
        async def change_prompt(direction):
            prompt.write_text('第二版提示')
            scene.write_text('第二版场景')
            return await original(direction)
        self.tools.look=change_prompt
        for reload in (True,False):
            with self.subTest(reload=reload):
                prompt.write_text('第一版提示')
                scene.write_text('第一版场景')
                self.requests.clear()
                self.responses=[{'action':'look','direction':'left','reason':'环视'},
                                {'action':'finish','reason':'完成','evidence':'看到目标'}]
                runtime=self.runtime(prompt={'path':prompt,'reload_each_step':reload,'scene':ScenePromptSettings(path=scene)})
                with patch.object(api,'current_settings',return_value=runtime):
                    record=await self.run_loop(steps=2)
                    expected=[navigation_prompt(text, seconds=runtime.agent.execution.navigation_seconds,
                        guide=runtime.agent.prompt.instruction_guide)
                        for text in ['第一版提示\n\n第一版场景','第二版提示\n\n第二版场景' if reload else '第一版提示\n\n第一版场景']]
                    from navigation_agent.agent.context.prompts import decision_reminder
                    expected = [text + decision_reminder(json.loads(request['messages'][1]['content'][0]['text']))
                                for text, request in zip(expected, self.requests)]
                    self.assertEqual([x['messages'][0]['content'] for x in self.requests],expected)
                    events=[json.loads(line) for line in (api.log_store().run_directory(record)/'events.jsonl').read_text().splitlines()]
                    used=[event for event in events if event['type']=='prompt']
                    self.assertEqual(len(used),2)
                    for event,content in zip(used,expected):
                        self.assertEqual(event['sha256'],hashlib.sha256(content.encode()).hexdigest())
                        self.assertEqual((api.log_store().run_directory(record)/event['file']).read_text(),content)

    async def test_office_gate_is_selected_by_configuration_in_model_loop(self):
        from navigation_agent.app.deployments.settings import load_backend, PROJECT_ROOT
        scene=load_backend(PROJECT_ROOT/'configs/go2-lightnav.toml').prompt
        for enabled in (False,True):
            with self.subTest(enabled=enabled):
                self.requests.clear()
                self.tools.actions.clear()
                decision=navigation('Walk straight toward the corridor ahead and stop.')
                decision.pop('navigation_check')
                finish={'action':'finish','reason':'完成','evidence':'看到目标'}
                if not enabled:
                    finish['target_review']=target_review(2,'arrived')
                self.responses=[decision,finish]
                runtime=self.runtime(prompt={'scene':scene if enabled else None})
                with patch.object(api,'current_settings',return_value=runtime):
                    record=await self.run_loop(steps=2)
                    self.assertEqual(record['status'],'completed')
                    events=[json.loads(line) for line in (api.log_store().run_directory(record)/'events.jsonl').read_text().splitlines()]
                moved=any(event['type']=='action_input' and event.get('action')=='navigate' for event in events)
                self.assertEqual(moved,not enabled)
                self.assertEqual('GO2 办公场景边界规则' in self.requests[0]['messages'][0]['content'],enabled)
                self.assertEqual(any(event.get('category')=='scene_rule' for event in events),enabled)

    async def test_consecutive_navigation_errors_use_configured_limit(self):
        self.responses=[navigation('Walk straight toward the corridor ahead and stop.')]
        async def fail_navigation(instruction,*,seconds):
            return {'reason':'navigation_error'}
        self.tools.navigate=fail_navigation
        with patch.object(api,'current_settings',return_value=self.runtime(execution={'max_consecutive_navigation_errors':1})):
            record=await self.run_loop(steps=4)
        self.assertEqual(record['status'],'failed')
        self.assertIn('1 consecutive',record['summary'])
        self.assertEqual(len(self.requests),1)


class ConfigurationConsumerTests(unittest.TestCase):
    def setUp(self):
        from navigation_agent.app.deployments.runtime import Deployment
        deployment = patch.object(api, 'deployment', Deployment(api))
        deployment.start()
        self.addCleanup(deployment.stop)

    def test_task_precedence_limits_and_two_page_presets(self):
        from navigation_agent.app.deployments.settings import load_backend, PROJECT_ROOT
        runtime=test_runtime()
        agent=runtime.agent.model_copy(update={'task_defaults':runtime.agent.task_defaults.model_copy(update={'max_steps':6,'max_seconds':75}),
            'execution':runtime.agent.execution.model_copy(update={'navigation_seconds':8.0})})
        with patch.object(api,'current_settings',return_value=replace(runtime,agent=agent)):
            self.assertEqual(api.resolve_task(TaskRequest(goal='测试')).max_steps,6)
            request=api.resolve_task(TaskRequest(goal='测试',max_steps=9,max_seconds=90,navigation_seconds=4.0))
            self.assertEqual((request.max_steps,request.max_seconds,request.navigation_seconds),(9,90,4))
            from fastapi import HTTPException
            for value in (0.5,61.):
                with self.assertRaises(HTTPException):
                    api.resolve_task(TaskRequest(goal='测试',navigation_seconds=value))
        for kind,steps,seconds in [('go2',300,600),('mujoco',300,600)]:
            backend=load_backend(PROJECT_ROOT/f'configs/{kind}-lightnav.toml')
            # 验证未保存设置时的页面预设，不读取开发机的运行状态。
            with patch.object(api,'current_settings',return_value=replace(runtime,backend=backend)), \
                 patch.object(api.deployment,'saved_motion',return_value=None):
                self.assertEqual(api.get_agent_settings()['ui_task_defaults'],{'max_steps':steps,'max_seconds':seconds})

    def test_exploration_thresholds_and_shared_view_angle(self):
        settings=agent_settings()
        policy=settings.exploration.model_copy(update={'max_look_actions':1,'max_candidate_failures':1,
            'max_recovery_attempts':1,'scan_reset_distance_m':1.0,'failed_displacement_m':.5,'stuck_displacement_m':.3})
        state=ExecutionState(look_angle_deg=60,exploration=policy,
            observation=settings.observation.model_copy(update={'same_view_angle_deg':40.0}),execution=settings.execution)
        observe(state,yaw=95)
        observe(state,yaw=0,step=2)
        self.assertNotIn('left',state.scan.unobserved_look_directions())
        state.scan.look_count=1
        self.assertEqual(state.allowed_look_directions(),['left','right'])
        observe(state,x=.4,step=3)
        self.assertEqual(state.scan.look_count,1)
        observe(state,x=1.1,step=4)
        self.assertEqual(state.scan.look_count,1)
        self.assertEqual(classify_result({'reason':'model_stop','motion':{'displacement_m':.2,'blocked_samples':1}},policy),(True,True))
        cid=state.accept(decision())
        state.record_result(cid,{'reason':'timeout'})
        with self.assertRaisesRegex(ValueError,'失败1次'):
            state.resolve(decision(candidate=cid))
        state.recovery_required=True
        state.recovery_attempts=1
        with self.assertRaises(ValueError):
            state.resolve(Decision(action='recover',reason='测试'))

    def test_memory_radius_view_matching_and_retrieval_limits(self):
        settings=agent_settings()
        with tempfile.TemporaryDirectory() as directory:
            memory=SpatialMemory(Path(directory)/'memory.sqlite3','沙发',
                settings=settings.memory.model_copy(update={'place_radius_m':.1,'max_places':1,
                    'max_semantic_frames':1,'max_objects':1,'max_reviews':1,'same_view_angle_deg':5.0}),
                observation=settings.observation.model_copy(update={'same_view_angle_deg':5.0}))
            for step,x in enumerate((0,.2,.2),1):
                view=memory.observe({'pose':{'x':x,'y':0,'yaw':math.radians(step*10)}},f'{step}.jpg',step)
                if step==2:self.assertEqual(view['id'],'P2')
                if step==3:self.assertEqual(view['same_view_previous_observations'],0)
                memory.describe(view['id'],step,Decision(action='look',direction='left',reason='测试',
                    scene={'objects':[{'name':f'物体{step}','description':'可见'}]},
                    previous_action_review={'outcome':'uncertain','evidence':'尚不能确认'}))
            result=memory.query(view)
            self.assertEqual(len(result['places']),1)
            self.assertEqual(len(result['semantic_frames']),1)
            self.assertEqual(len(result['objects']),1)
            self.assertEqual(len(result['recent_reviews']),1)

    def test_memory_heading_buckets_and_nearby_ranking(self):
        settings=agent_settings()
        with tempfile.TemporaryDirectory() as directory:
            memory=SpatialMemory(Path(directory)/'m.sqlite3','目标',settings=settings.memory,
                                 observation=settings.observation)
            for step,x,yaw in [(1,2,0),(2,0,0),(3,0,10),(4,5,0),(5,10,0)]:
                view=memory.observe({'pose':{'x':x,'y':0,'yaw':math.radians(yaw)}},f'{step}.jpg',step)
                memory.describe(view['id'],step,Decision(action='look',direction='left',reason='测试',
                    scene={'visible_ground':'当前可见地面'}))
            query={**view,'position':[0,0]}
            narrow=memory.query(query)
            memory.settings=settings.memory.model_copy(update={'heading_bin_deg':5.0,'nearby_radius_m':20.0})
            wide=memory.query(query)
            self.assertEqual(len(narrow['semantic_frames']),3)
            self.assertEqual(len(wide['semantic_frames']),4)
            self.assertEqual(narrow['semantic_frames'][0]['step'],3)
            self.assertEqual(wide['semantic_frames'][0]['step'],4)

    def test_unfinished_candidate_limit(self):
        settings=agent_settings()
        state=ExecutionState(look_angle_deg=60,exploration=settings.exploration,
            observation=settings.observation,execution=settings.execution.model_copy(update={'max_unfinished_candidates':1}))
        for step,x in enumerate((0,2,4),1):
            observe(state,x=x,step=step)
            state.current['id'] = f'P{step}'
            state.accept(decision())
        self.assertEqual(len(state.context()['unfinished_elsewhere']),1)

    def test_agent_has_no_environment_or_application_imports(self):
        import ast
        from navigation_agent.app.deployments.settings import PROJECT_ROOT
        for path in (PROJECT_ROOT/'src/navigation_agent/agent').rglob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(x.name.split('.')[0] in ('app','robots','vln','os','dotenv') for x in node.names),path)
                if isinstance(node,ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0],('app','robots','vln','os','dotenv'),path)


class RecordingLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """录像启动先于任务；失败与取消都必须封装或停止录像。"""

    async def asyncSetUp(self):
        import asyncio
        from aiohttp import web
        from aiohttp.test_utils import TestServer
        from types import SimpleNamespace
        from unittest.mock import Mock
        from navigation_agent.app.deployments.runtime import Deployment
        self.calls = []
        self.failure = False
        self.block_start = False
        self.started = asyncio.Event()

        async def recording(request):
            action = request.match_info['action']
            self.calls.append(action)
            if action == 'start':
                self.started.set()
                if self.block_start:
                    await asyncio.sleep(.3)
            if self.failure:
                raise web.HTTPServiceUnavailable()
            return web.json_response({'status': 'recording' if action == 'start' else 'ready'})

        app = web.Application()
        app.router.add_post('/api/recordings/{task}/{action}', recording)
        self.server = TestServer(app)
        await self.server.start_server()
        settings = SimpleNamespace(
            backend=SimpleNamespace(robot=SimpleNamespace(backend='mujoco', base_url=str(self.server.make_url('/')).rstrip('/')),
                                    simulation=SimpleNamespace(recording=SimpleNamespace(enabled=True))),
            agent=SimpleNamespace(storage=None))
        self.deployment = Deployment(SimpleNamespace(current_settings=lambda: settings))
        self.record = {'id': 'a' * 32}
        self.store = Mock()
        self.patch = patch('navigation_agent.app.deployments.runtime.LogStore', return_value=self.store)
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        await self.server.close()

    async def test_recording_brackets_task_and_survives_encoder_failure(self):
        for fail in (False, True):
            self.failure = fail
            self.calls.clear()
            async def runner(record):
                self.assertEqual(self.calls, ['start'])
                self.calls.append('runner')
                record['status'] = 'completed'
            await self.deployment.execute_task(runner, self.record)
            self.assertEqual(self.calls, ['start', 'runner', 'stop'])
            self.assertEqual(self.record['status'], 'completed')
            self.assertEqual(self.record['recording']['status'], 'failed' if fail else 'ready')

    async def test_cancellation_stops_recording_and_does_not_leave_task_running(self):
        import asyncio
        from unittest.mock import AsyncMock
        self.block_start = True
        runner = AsyncMock()
        worker = asyncio.create_task(self.deployment.execute_task(runner, self.record))
        await self.started.wait()
        worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await worker
        runner.assert_not_awaited()
        self.assertEqual(self.record['status'], 'cancelled')
        self.assertEqual(self.calls, ['start', 'stop'])
        self.store.finish_log_directory.assert_called_once_with(self.record)


class RegionComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_reference_or_incomplete_response_is_rejected(self):
        from unittest.mock import AsyncMock
        from navigation_agent.agent.model.observation import compare_regions
        references = [{'region_id': 'R1', 'jpeg': b'old'}]
        settings = agent_settings().llm
        for identifier, finish, valid in (
                ('R1', 'stop', True), ('new', 'stop', True), ('uncertain', 'stop', True),
                ('R9', 'stop', False), ('new', 'length', False)):
            client = AsyncMock()
            client.complete.return_value = {'choices': [{'finish_reason': finish, 'message': {
                'content': json.dumps({'region_id': identifier, 'label': '门框与木桌',
                                       'evidence': '比较当前图与参考图'})}}]}
            with self.subTest(identifier=identifier, finish=finish):
                if valid:
                    report, _ = await compare_regions(client, b'current', references, settings)
                    self.assertEqual(report.region_id, identifier)
                else:
                    with self.assertRaises(ValueError):
                        await compare_regions(client, b'current', references, settings)


    async def test_third_room_reference_and_pose_distance_reach_the_comparison(self):
        from unittest.mock import AsyncMock
        from navigation_agent.agent.model.observation import compare_regions
        client = AsyncMock()
        client.complete.return_value = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'region_id': 'R3', 'label': '入口和窗户', 'evidence': '与第三组参照一致'})}}]}
        frames = [dict(region_id=f'R{index // 2 + 1}', step=index + 1, jpeg=b'image',
                       observer_position=[0., 0.]) for index in range(6)]
        report, _ = await compare_regions(client, b'current', frames, agent_settings().llm,
                                          current_view={'observer_position': [1.25, 0.]})
        self.assertEqual(report.region_id, 'R3')
        content = client.complete.call_args.args[0]['messages'][1]['content']
        self.assertEqual(sum(item['type'] == 'image_url' for item in content), 7)
        self.assertIn('1.25', content[0]['text'])


class RoomCrossingEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_navigation_sequence_reaches_room_comparison_without_old_intent(self):
        from navigation_agent.agent.model.observation import compare_regions
        from types import SimpleNamespace

        requests = []
        class Client:
            async def complete(self, body):
                requests.append(body)
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
                    {'region_id': 'new', 'label': '门后区域', 'evidence': '相机已穿过门框'})}}]}

        refs = [{'region_id': 'R1', 'step': 1, 'jpeg': b'old-room'}]
        frames = [{'step': 1, 'jpeg': b'old-room', 'intent': '不应发送的旧目的'},
                  {'step': 3, 'jpeg': b'doorway'}, {'step': 5, 'jpeg': b'after-door'}]
        settings = SimpleNamespace(model='test', max_tokens=1000)
        await compare_regions(Client(), b'current', refs, settings, navigation_frames=frames)
        content = requests[0]['messages'][1]['content']
        texts = [item['text'] for item in content if item['type'] == 'text']
        self.assertEqual(sum(item['type'] == 'image_url' for item in content), 4)
        self.assertLess(next(i for i, text in enumerate(texts) if '连续原图 O3' in text),
                        next(i for i, text in enumerate(texts) if '连续原图 O5' in text))
        self.assertIn('当前停车图 A', texts[-1])
        self.assertNotIn('不应发送的旧目的', json.dumps(requests, ensure_ascii=False))


class StoppedRecallPromptTests(unittest.TestCase):
    def test_known_room_match_recalls_its_own_remaining_work_without_changing_memory(self):
        import copy
        from navigation_agent.agent.context.prompts import decision_reminder

        context = {'goal': '找到目标', 'observation_step': 20, 'observation_id': 'O20',
                   'execution': {'action_options': {'recover_allowed': False, 'look_options': {},
                                                    'orient_view_ids': []}},
                   'exploration_memory': {'previous_region': 'R2',
                       'visual_match': {'region_id': 'R1', 'label': '旧房间', 'evidence': '入口与地标一致'},
                       'connections': [{'source': 'R2', 'destination': 'R1', 'start_step': 5, 'end_step': 10}],
                       'regions': [
                           {'id': 'R2', 'visits': 1, 'search': {'status': 'surveyed', 'remaining': ['岛台后面']}},
                           {'id': 'R1', 'visits': 1, 'search': {'status': 'checked', 'remaining': ['北侧入口']}}]}}
        original = copy.deepcopy(context)
        prompt = decision_reminder(context)
        self.assertIn('本轮重点召回区域 R1 的历史检查状态=checked', prompt)
        self.assertIn('历史具体未查项=["北侧入口"]', prompt)
        self.assertIn('此前已访问1次', prompt)
        self.assertIn('历史已走来路：R2→R1，起点O5、停车O10', prompt)
        self.assertEqual(context, original)
        context['exploration_memory']['regions'][1]['search']['status'] = 'visited'
        self.assertIn('不能视为已查完', decision_reminder(context))
        context['exploration_memory']['visual_match']['region_id'] = 'uncertain'
        self.assertIn('本轮重点召回区域 R2', decision_reminder(context))


class KnownRouteComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_route_evidence_uses_current_rgb_without_old_action_intent(self):
        from unittest.mock import AsyncMock
        from navigation_agent.agent.model.observation import compare_routes

        client = AsyncMock()
        client.complete.return_value = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'matches': [{'route_id': 'C2', 'evidence': '当前右侧门框与置物架关系一致'}]})}}]}
        frames = [dict(route_id='C2', region_id='R1', destination_region='R2', step=3,
                       jpeg=b'old-entrance', old_intent='不应发送的旧目的'),
                  dict(route_id='C2', region_id='R1', destination_region='R2', step=8,
                       jpeg=b'after-crossing', at_destination=True)]
        report, _ = await compare_routes(client, b'current', frames, agent_settings().llm)
        self.assertEqual(report.matches[0].route_id, 'C2')
        body = client.complete.call_args.args[0]
        content = body['messages'][1]['content']
        self.assertEqual(sum(item['type'] == 'image_url' for item in content), 2)
        self.assertIn('当前停车图 A', content[-2]['text'])
        # 通过后的图可能正对另一入口，不能用作同一门的身份参照。
        import base64
        self.assertNotIn(base64.b64encode(b'after-crossing').decode(), json.dumps(body))
        self.assertNotIn('不应发送的旧目的', json.dumps(body, ensure_ascii=False))

    async def test_empty_is_uncertain_and_unknown_duplicate_or_truncated_match_is_rejected(self):
        from unittest.mock import AsyncMock
        from navigation_agent.agent.model.observation import compare_routes

        frames = [dict(route_id='C2', region_id='R1', destination_region='R2', step=3, jpeg=b'old')]
        for ids, finish, valid in [([], 'stop', True), (['C2'], 'stop', True),
                                   (['C9'], 'stop', False), (['C2', 'C2'], 'stop', False),
                                   (['C2'], 'length', False)]:
            with self.subTest(ids=ids, finish=finish):
                client = AsyncMock()
                client.complete.return_value = {'choices': [{'finish_reason': finish, 'message': {
                    'content': json.dumps({'matches': [{'route_id': identifier, 'evidence': '当前图入口匹配'}
                                                       for identifier in ids]})}}]}
                if valid:
                    report, _ = await compare_routes(client, b'current', frames, agent_settings().llm)
                    self.assertEqual([item.route_id for item in report.matches], ids)
                else:
                    with self.assertRaises(ValueError):
                        await compare_routes(client, b'current', frames, agent_settings().llm)
