"""办公场景规则仅在部署显式启用时校验，不能被网页参数或缺失报告绕过。"""
import unittest
from pydantic import ValidationError
from tests.config_support import execution_state as ExecutionState
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from navigation_agent.agent.context.prompts import navigation_prompt
from navigation_agent.app.deployments.scene import validate_navigation_scene
from navigation_agent.agent.schemas import Decision
from tests.support import clear_navigation_check


def proposal(check):
    return Decision(action='navigate',reason='探索',instruction='Walk straight toward the sign ahead and stop.',
                    navigation_check=check,candidate_id='new_1',candidates=[
                        {'id':'new_1','kind':'viewpoint','description':'标志前空地','evidence':'当前图可见'}])


class ScenePolicyTests(unittest.TestCase):
    def test_forbidden_or_unknown_assessments_are_rejected_before_state_changes(self):
        for field, values in {'target_area':['carpeted_office','uncertain'],
                              'path_area':['carpeted_office','uncertain'],
                              'passage':['glass_or_closed_door','uncertain']}.items():
            for value in values:
                with self.subTest(field=field,value=value):
                    check=clear_navigation_check();check[field]=value
                    state=ExecutionState(navigation_policy="go2_office")
                    state.observe({'id':'P1','step':1,'observer_position':[0,0],'heading_deg':0})
                    with self.assertRaises(ValueError):state.accept(proposal(check))
                    self.assertEqual(state.candidates,[])

    def test_missing_assessment_cannot_default_to_clear(self):
        with self.assertRaisesRegex(ValueError,'缺少 navigation_check'):
            validate_navigation_scene(proposal(None), "go2_office")
        check=clear_navigation_check();check['evidence']='  '
        with self.assertRaises(ValidationError):proposal(check)

    def test_allowed_path_passes_and_stationary_actions_need_no_assessment(self):
        validate_navigation_scene(proposal(clear_navigation_check()), "go2_office")
        for decision in (Decision(action='look',direction='left',reason='检查玻璃'),
                         Decision(action='give_up',reason='目标在禁区'),
                         Decision(action='finish',reason='只需看到目标',evidence='从允许区域已清楚看到目标')):
            validate_navigation_scene(decision, "go2_office")

    def test_local_viewpoint_can_inspect_unknown_distant_door_without_crossing_it(self):
        state = ExecutionState(navigation_policy='go2_office')
        state.observe({'id': 'P1', 'step': 6, 'observer_position': [0, 0], 'heading_deg': 0})
        decision = proposal({
            'target_area': 'allowed', 'path_area': 'allowed', 'passage': 'clear', 'space': 'normal',
            'evidence': '终点为玻璃门近侧空地，到此地面可见且无人员和障碍，不经过玻璃门',
        })
        decision.instruction = 'Walk straight toward the open floor area ahead and stop.'
        decision.candidates[0].description = '玻璃门近侧的空地'
        decision.candidates[0].evidence = '近侧空地及接近路径可见，远处门的开合仍不清楚'
        decision.candidates[0].expected_new_view = '接近后看清入口开合及两侧连接，不确定能否继续'
        state.scan.complete = True
        state.resolve(decision)
        # 如果不确定的是本次实际要走的路段，仍必须拒绝。
        decision.navigation_check.passage = 'uncertain'
        with self.assertRaisesRegex(ValueError, '到本次终点'):
            state.resolve(decision)
        self.assertEqual(state.candidates, [])

    def test_fixed_rules_are_present_with_custom_navigation_guide(self):
        prompt=navigation_prompt((PROJECT_ROOT/'configs/prompts/common.md').read_text() + '\n\n' + (PROJECT_ROOT/'configs/prompts/go2-office.md').read_text(), seconds=20,
            guide='使用 Approach 描述明确目标，方向和停止要求按需要添加。')
        self.assertIn('禁区只限有人正在使用的工位',prompt)
        self.assertIn('不能因为画面里有办公桌椅或地毯，就把整片区域判成禁区',prompt)
        self.assertIn('GO2 办公场景边界规则',prompt)
        self.assertIn('navigation_check',prompt)

    def test_common_navigation_has_no_office_gate(self):
        state=ExecutionState()
        state.observe({'id':'P1','step':1,'observer_position':[0,0],'heading_deg':0})
        decision = proposal(None)
        decision.candidates[0].expected_new_view = '接近后看清入口是否开放，门后情况尚不确定'
        state.scan.complete = True
        state.accept(decision)
        self.assertEqual(len(state.candidates),1)
        prompt=navigation_prompt((PROJECT_ROOT/'configs/prompts/common.md').read_text(),
            seconds=20,guide='使用 Approach 描述明确目标，方向和停止要求按需要添加。')
        self.assertNotIn('地毯',prompt)
        self.assertNotIn('navigation_check',prompt)
        self.assertNotIn('办公',prompt)

    def test_narrow_space_requires_a_current_visible_target(self):
        check = clear_navigation_check()
        check['space'] = 'narrow'
        decision = proposal(check)
        for seen in ('no', 'uncertain', 'yes'):
            decision.scene.target_seen = seen
            with self.assertRaisesRegex(ValueError, '狭窄空间'):
                validate_navigation_scene(decision, 'go2_office')
        decision.candidates[0].kind = 'target'
        for seen in ('no', 'uncertain'):
            decision.scene.target_seen = seen
            with self.assertRaisesRegex(ValueError, '明确看到目标'):
                validate_navigation_scene(decision, 'go2_office')
        decision.scene.target_seen = 'yes'
        validate_navigation_scene(decision, 'go2_office')
        decision.navigation_check.passage = 'glass_or_closed_door'
        with self.assertRaisesRegex(ValueError, '玻璃'):
            validate_navigation_scene(decision, 'go2_office')

    def test_missing_space_assessment_is_not_assumed_normal(self):
        check = clear_navigation_check()
        check.pop('space')
        with self.assertRaisesRegex(ValueError, 'space'):
            validate_navigation_scene(proposal(check), 'go2_office')
        check['space'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, 'space'):
            validate_navigation_scene(proposal(check), 'go2_office')

    def test_narrow_rule_is_general_and_does_not_rank_roads_by_width(self):
        prompt = (PROJECT_ROOT / 'configs/prompts/common.md').read_text()
        self.assertIn('只有当前画面中明确看到目标位于其中', prompt)
        self.assertIn('不能仅为了确认里面有没有目标而钻进去', prompt)
        self.assertNotIn('探索时优先选择通道中部', prompt)
        self.assertNotIn('钻入桌椅夹缝', prompt)
        self.assertNotIn('靠近柜侧', prompt)
