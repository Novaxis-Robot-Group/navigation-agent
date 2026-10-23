from pathlib import Path
"""Runtime settings change the actual prompt, policy and navigation time budget."""
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError
from navigation_agent.contracts.runtime import PromptSettings
from navigation_agent.agent.context.prompts import navigation_prompt
from navigation_agent.agent.tools.navigation import validate_instruction
from navigation_agent.agent.skills.object_search.state import ExecutionState
from tests.config_support import agent_settings
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT = (PROJECT_ROOT / 'configs/prompts/common.md').read_text()
from navigation_agent.agent.tools import NavigationTools


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_nondefault_time_budget_reaches_prompt_and_navigation(self):
        backend=AsyncMock()
        tools=NavigationTools(backend)
        prompt=navigation_prompt(PROMPT, seconds=27, guide=PromptSettings().instruction_guide)
        self.assertIn('最多 27 秒',prompt)
        self.assertIn('单段导航时限：最多 27 秒',prompt)
        self.assertNotIn('15 秒',prompt)
        self.assertNotIn('{{',prompt)
        self.assertIn(PromptSettings().instruction_guide,prompt)
        await tools.navigate('Walk straight toward the green sign on the wall ahead and stop.', seconds=27)
        backend.navigate.assert_awaited_once_with(
            'Walk straight toward the green sign on the wall ahead and stop.',seconds=27)

    def test_instruction_format_accepts_guide_variations(self):
        instructions = (
            'Go to the front-left TV',
            'Go to the desk on your right and stop.',
            'Turn right, then walk to the chair and stop.',
            'Walk towards the trash can next to the green lawn',
            'Walk to the vase on the dining table ahead.',
            'Head to the plant behind you on the right.',
            'Go to the leftmost TV in front',
            'Turn left and walk to the second stone bench from the left.',
            'Approach the red sofa.',
        )
        for instruction in instructions:
            with self.subTest(instruction=instruction):
                validate_instruction(instruction)
        for instruction in ('', ' ', 'Go to {target}.', '走向门洞', 'Go to the door.\nStop.',
                            'Go to the door.\t', 'x' * 501):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                validate_instruction(instruction)

    def test_guide_validation_and_rendering_preserve_rules(self):
        guide = '使用 Approach 描述目标。\n方向和停止要求可选。'
        settings = PromptSettings(instruction_guide=guide)
        prompt = navigation_prompt(PROMPT, seconds=9, guide=settings.instruction_guide)
        self.assertIn(guide, prompt)
        self.assertNotIn('{{', prompt)
        self.assertNotIn('严格使用模板', prompt)
        self.assertNotIn('不写连续动作', prompt)
        for invalid in ('', ' ', 'x' * 4001, 'rules\x00'):
            with self.subTest(guide=invalid), self.assertRaises(ValidationError):
                PromptSettings(instruction_guide=invalid)

    def test_retired_observation_limits_are_ignored_in_old_configs_and_saved_settings(self):
        from navigation_agent.agent.config import ExplorationSettings
        from navigation_agent.app.go2.motion_settings import MotionSettings
        settings = agent_settings()
        data = settings.exploration.model_dump()
        data.update(max_look_actions=1, max_target_confirmation_looks=1)
        policy = ExplorationSettings.model_validate(data)
        self.assertNotIn('max_look_actions', policy.model_dump())
        self.assertNotIn('max_target_confirmation_looks', policy.model_dump())
        motion = MotionSettings.model_validate({'exploration': {'max_looks': 1}})
        self.assertNotIn('max_looks', motion.exploration.model_dump())
        self.assertNotIn('max_looks', type(motion.exploration).model_json_schema()['properties'])
        state = ExecutionState(look_angle_deg=60, exploration=policy,
                               observation=settings.observation, execution=settings.execution)
        state.observe({'id':'P1', 'observer_position':[0,0], 'step':1, 'heading_deg':0})
        state.scan.look_count = 100
        self.assertEqual(state.allowed_look_directions(), ['left', 'right'])
