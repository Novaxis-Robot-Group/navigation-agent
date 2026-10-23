"""无关空字段不应触发模型重新决策，必填错误仍须拒绝。"""

import unittest
from pydantic import ValidationError
from navigation_agent.agent.schemas import Decision


class EmptyFieldTests(unittest.TestCase):
    def test_look_preserves_action_with_empty_candidate(self):
        decision = Decision(action='look', direction='left', candidate_id=' ', reason='观察')
        self.assertEqual(decision.action, 'look')
        self.assertEqual(decision.direction, 'left')
        self.assertIsNone(decision.candidate_id)

    def test_navigate_can_omit_empty_direction(self):
        decision = Decision(action='navigate', direction='', candidate_id='new_1',
                            instruction='Go to the sofa.', reason='靠近')
        self.assertIsNone(decision.direction)

    def test_required_or_nonempty_invalid_values_are_rejected(self):
        for fields in ({'action': 'look', 'direction': ''},
                       {'action': 'navigate', 'instruction': 'Go.', 'candidate_id': ''},
                       {'action': 'finish', 'evidence': '看到了', 'direction': 'up'}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                Decision(reason='检查', **fields)


    def test_finish_reuses_arrival_evidence_without_requiring_duplicate_text(self):
        for fields in ({}, {'evidence': ''}, {'evidence': ' '}):
            with self.subTest(fields=fields):
                decision = Decision(action='finish', reason='已到达', scene={'target_seen': 'yes'},
                                    target_review={'observation_step': 7, 'status': 'arrived',
                                                   'evidence': '当前图可见近处白色马桶'}, **fields)
                self.assertEqual(decision.evidence, '当前图可见近处白色马桶')
                self.assertEqual(decision.target_review.observation_step, 7)

    def test_finish_does_not_promote_weak_or_missing_evidence_to_completion(self):
        for status, seen, evidence in [('visible', 'yes', '看见目标'),
                                       ('temporarily_lost', 'no', '历史目标被遮挡'),
                                       ('arrived', 'no', '当前未见'),
                                       ('arrived', 'yes', ' ')]:
            with self.subTest(status=status, seen=seen), self.assertRaises(ValidationError):
                Decision(action='finish', reason='想结束', scene={'target_seen': seen},
                         target_review={'observation_step': 7, 'status': status, 'evidence': evidence})
        with self.assertRaises(ValidationError):
            Decision(action='finish', reason='已到达',
                     previous_action_review={'outcome': 'progress', 'evidence': '上一段导航有位移'})

    def test_explicit_completion_evidence_is_never_overwritten(self):
        decision = Decision(action='finish', reason='当前核实', evidence='当前图的新证据',
                            scene={'target_seen': 'yes'},
                            target_review={'observation_step': 7, 'status': 'arrived', 'evidence': '另一处描述'})
        self.assertEqual(decision.evidence, '当前图的新证据')


    def test_action_review_accepts_only_unambiguous_status_alias(self):
        decision = Decision(action='look', direction='left', reason='核实',
                            previous_action_review={'status': 'no_new_information',
                                                    'evidence': '仍在同一门边'})
        self.assertEqual(decision.previous_action_review.outcome, 'no_new_information')
        self.assertNotIn('status', decision.previous_action_review.model_dump())
        for review in (
            {'status': 'completed', 'evidence': '工具返回'},
            {'status': 'progress', 'outcome': 'uncertain', 'evidence': '矛盾结果'},
            '上一段没有获得新信息',
        ):
            with self.subTest(review=review), self.assertRaises(ValidationError):
                Decision(action='look', direction='left', reason='核实',
                         previous_action_review=review)



class RepairRequirementsTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_retry_reports_all_missing_navigation_fields_without_inventing_candidate(self):
        import copy
        import json
        from navigation_agent.agent.model.decision import decide
        from tests.config_support import agent_settings

        requests, errors = [], []
        outputs = [{'action': 'navigate'},
                   {'action': 'finish', 'reason': '当前图已经确认目标', 'evidence': '当前图可见棕色沙发'}]
        class Client:
            async def complete(self, body):
                requests.append(copy.deepcopy(body))
                return {'choices': [{'finish_reason': 'stop', 'message': {
                    'content': json.dumps(outputs[len(requests) - 1])}}]}

        decision, _ = await decide(
            Client(), {'execution': {'action_options': {}}}, [],
            diagnostic=lambda **event: errors.append(event), settings=agent_settings().llm,
            api_key='unused', prompt_text='test', max_error_output_chars=10000,
        )
        fields = {tuple(error.get('field', [])) for error in errors[0]['validation_errors']}
        self.assertTrue({('reason',), ('instruction',), ('candidates',), ('candidate_id',)} <= fields)
        self.assertTrue({('candidates', '*', field)
                         for field in ('id', 'kind', 'description', 'evidence', 'expected_new_view')} <= fields)
        self.assertEqual(decision.action, 'finish')
        self.assertEqual(len(requests), 2)
        self.assertIn('上次动作未执行', requests[1]['messages'][-1]['content'])
        self.assertIn('新发现候选用 new_1 至 new_4', requests[1]['messages'][-1]['content'])
        self.assertEqual(json.loads(requests[1]['messages'][-2]['content']), {'action': 'navigate'})

    async def test_finish_reports_completion_evidence_and_active_target_review_together(self):
        import json
        from navigation_agent.agent.model.decision import missing_action_fields
        fields = missing_action_fields(json.dumps({'action': 'finish', 'reason': '找到'}),
                                       {'execution': {'target_tracking': {'phase': 'verifying_arrival'}}})
        self.assertIn(['evidence'], fields)
        self.assertIn(['target_review'], fields)

    async def test_normalized_single_candidate_is_not_reported_as_missing(self):
        import json
        from navigation_agent.agent.model.decision import missing_action_fields
        payload = {'action': 'navigate', 'reason': '查看门后',
                   'navigate': {'instruction': 'Go to the doorway in front.'},
                   'candidates': [{'id': 'new_1', 'kind': 'open_passage',
                                   'description': '正前方入口', 'evidence': '当前图可见门内地面',
                                   'expected_new_view': '确认门后区域'}]}
        self.assertEqual(missing_action_fields(json.dumps(payload), {}), [])

    async def test_malformed_action_does_not_break_existing_format_diagnostic(self):
        from navigation_agent.agent.model.decision import missing_action_fields
        for content in ('not-json', '[]', '{"action": []}', '{"action": {}}'):
            with self.subTest(content=content):
                self.assertEqual(missing_action_fields(content, {}), [])


class RepairSchemaTests(unittest.TestCase):
    def test_repair_includes_nested_contract_and_only_referenced_definitions(self):
        from navigation_agent.agent.model.decision import repair_field_schema
        schema = repair_field_schema([
            {'field': ['previous_action_review'], 'type': 'model_type'},
            {'field': ['candidates', '*', 'evidence'], 'type': 'missing_action_field'},
            {'field': ['reasoning'], 'type': 'extra_forbidden'},
        ])
        self.assertEqual(set(schema['properties']), {'previous_action_review', 'candidates'})
        self.assertEqual(set(schema['$defs']), {'ActionReview', 'CandidateObservation'})
        review = schema['$defs']['ActionReview']
        self.assertEqual(review['required'], ['outcome', 'evidence'])
        self.assertEqual(review['properties']['outcome']['enum'],
                         ['progress', 'no_new_information', 'uncertain'])
        self.assertIn('expected_new_view', schema['$defs']['CandidateObservation']['properties'])
        self.assertFalse(review['additionalProperties'])
