"""执行约束、运动统计及当前接口格式的回归快照。"""

import json
from pathlib import Path
import unittest

from tests.support import clear_navigation_check
from navigation_agent.agent.schemas import Decision, TaskRequest
from navigation_agent.contracts.robot import MotionSample
from tests.backend_support import backend_class

SimTools = backend_class()
from tests.config_support import execution_state as ExecutionState


def replay():
    memory = ExecutionState()
    trace = []

    def observe(x, yaw, step):
        memory.observe({'id': 'P1', 'observer_position': [x, 0], 'step': step, 'heading_deg': yaw * 180 / 3.141592653589793})
        trace.append(memory.snapshot())

    observe(0, 0, 1)
    decision = Decision(
        action='navigate', navigation_check=clear_navigation_check(),
        reason='接近已见目标', scene={'target_seen': 'yes'},
        instruction='走到门口',
        candidate_id='new_1',
        candidates=[
            dict(
                id='new_1',
                kind='target',
                description='门',
                evidence='可见地面',
            )
        ],
    )
    selected = memory.accept(decision)
    memory.record_result(
        selected, {'reason': 'blocked', 'motion': {'displacement_m': 0, 'blocked_samples': 20}}
    )
    trace.append(memory.snapshot())
    memory.accept(Decision(action='recover', reason='脱困'))
    memory.record_result(None, {'reason': 'recovered', 'displacement_m': 0.6})
    observe(-0.6, 0, 2)
    memory.accept(Decision(action='look', reason='看看', direction='left'))
    observe(-0.6, 1.22, 3)
    errors = []
    for action in (Decision(action='look', direction='right', reason='回看'), decision):
        try:
            memory.resolve(action)
            errors.append(None)
        except ValueError as exc:
            errors.append(str(exc))
    tools = SimTools('http://unused', 'ws://unused')
    tools.robot.state = {
        'simulation': {'pose': {'x': 0.6, 'y': 0, 'yaw': 0}, 'backend': {'collision_guard': True}}
    }
    tools.robot.breadcrumbs.extend((i * 0.02, 0) for i in range(31))
    tools.progress.samples = [
        MotionSample(i * 0.1, {'x': 0, 'y': 0, 'yaw': i * 0.2}, i % 3 == 0)
        for i in range(32)
    ]
    return {
        'trace': trace,
        'errors': errors,
        'context': memory.context(),
        'path': tools.robot.recovery_path(),
        'motion': tools.progress.summary(),
        'stalled': tools.progress.stalled(),
        'decision_schema': Decision.model_json_schema(),
        'task_schema': TaskRequest.model_json_schema(),
    }


class ContractTests(unittest.TestCase):
    def test_execution_and_schema_match_snapshot(self):
        expected = json.loads((Path(__file__).resolve().parents[1] / 'agent_tests' / 'behavior.json').read_text(encoding='utf-8'))
        actual = json.loads(json.dumps(replay(), ensure_ascii=False))
        # 明确列出本次有意改变的观察行为；运动保护和其他契约仍对照原快照。
        actual_contexts = [*actual['trace'], actual['context']]
        expected_contexts = [*expected['trace'], expected['context']]
        for index, (current, old) in enumerate(zip(actual_contexts, expected_contexts)):
            step = (1, 1, 2, 3, 3)[index]
            self.assertEqual(current['exploration_session']['current_view_id'], f'O{step}')
            options = current.pop('action_options')
            directions = ['left', 'right']
            self.assertEqual(options['look_directions'], directions)
            self.assertFalse(options['scan_required'])
            self.assertEqual(options['orient_view_ids'], current['exploration_session']['orient_view_ids'])
            self.assertEqual(options['recover_allowed'], index == 1)
            scan = current['scan']
            self.assertEqual(scan['allowed_look_directions'], directions)
            self.assertFalse(scan.pop('required'))
            self.assertIsNone(scan.pop('direction'))
            self.assertEqual(scan.pop('unobserved_look_directions'),
                             ['left', 'right'] if index < 3 else ['left'])
            for retired in ('new_scan_requires', 'translation_for_review', 'new_area_look_directions'):
                self.assertNotIn(retired, current)
                old.pop(retired)
            old['scan'].pop('look_budget_exhausted')
            old['scan'].pop('confirmation_look_directions')
            old['scan']['allowed_look_directions'] = directions
            if index >= 2:
                # 夹具没有实际 navigation_started/record_result，原地观察不重设锚点。
                old['scan']['anchor'] = [0, 0]
                old['scan']['observed_headings_deg'] = old['scan']['observed_headings_deg'][1:]
            for candidate in old['local_candidates']:
                candidate['view_id'] = 'O1'
        self.assertIn('复核', actual['errors'][0])
        self.assertIsNone(actual['errors'][1])
        expected['errors'][0] = actual['errors'][0]
        self.assertEqual(actual['decision_schema']['properties']['view_id']['anyOf'][0]['pattern'],
                         r'^O[1-9]\d*$')
        expected['decision_schema']['properties']['view_id']['anyOf'][0]['pattern'] = r'^O[1-9]\d*$'
        for value in (actual, expected):
            for context in [*value['trace'], value['context']]:
                context.pop('exploration_session')
                context.pop('target_tracking', None)
                context.pop('active_subgoal', None)
                context.pop('recent_subgoals', None)
        schema = actual['decision_schema']
        # 模型明确选择技能；运动动作不变，独立监测开关已移到技能内部。
        self.assertNotIn('watch_target', schema['properties']['action']['enum'])
        self.assertNotIn('watch', schema['properties'])
        detection = schema['$defs'].pop('DetectionReview')
        self.assertEqual(detection['required'], ['candidate_id', 'verdict', 'evidence'])
        for name in ('skill', 'target_labels', 'detection_review'):
            self.assertIsNone(schema['properties'].pop(name)['default'])

        source = schema['$defs']['CandidateObservation']['properties'].pop('evidence_source')
        self.assertEqual(source['anyOf'], [{'enum': ['current', 'history'], 'type': 'string'}, {'type': 'null'}])
        self.assertIsNone(source['default'])

        space = schema['$defs']['NavigationCheck']['properties'].pop('space')
        self.assertEqual(space['enum'], ['normal', 'narrow', 'uncertain'])
        self.assertEqual(space['default'], 'uncertain')
        review = schema['$defs'].pop('TargetReview')
        self.assertEqual(review['required'], ['observation_step', 'status', 'evidence'])
        self.assertEqual(review['properties']['status']['enum'],
                         ['visible', 'arrived', 'temporarily_lost', 'rejected'])
        schema['properties'].pop('target_review')
        intent = schema['$defs'].pop('ObservationIntent')
        self.assertEqual(intent['required'], ['reference_view_id', 'question', 'new_evidence'])
        self.assertEqual(intent['properties']['reference_view_id']['pattern'], r'^O[1-9]\d*$')
        self.assertIn('observation_intent', schema['properties'])
        schema['properties'].pop('observation_intent')
        subgoal = schema['$defs'].pop('SubgoalReview')
        self.assertEqual(subgoal['properties']['status']['enum'], ['continue', 'completed', 'abandoned'])
        schema['properties'].pop('subgoal_review')
        spatial = schema['$defs'].pop('SpatialUpdate')
        self.assertEqual(spatial['properties']['transition']['enum'], ['same', 'entered', 'uncertain'])
        schema['properties'].pop('spatial_update')
        # 新增的任务开关单独核对；既有动作、请求与图片顺序继续使用原快照。
        mode = actual['task_schema']['properties'].pop('navigation_time_limit_enabled')
        self.assertEqual(mode, {
            'anyOf': [{'type': 'boolean'}, {'type': 'null'}],
            'default': None,
            'title': 'Navigation Time Limit Enabled',
        })
        resume = schema['properties'].pop('resume_instruction')
        self.assertEqual(resume['anyOf'][0]['maxLength'], 500)
        self.assertIsNone(resume['default'])
        self.assertEqual(actual, expected)
