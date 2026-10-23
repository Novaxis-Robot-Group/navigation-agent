import math
import unittest

from tests.config_support import execution_state as ExecutionState
from tests.support import clear_navigation_check
from navigation_agent.agent.schemas import Decision


def observe(state, x=0, y=0, yaw=0, step=1):
    state.observe({
        'id': 'P1' if abs(x) <= .75 else 'P2',
        'observer_position': [x, y], 'position': [x, y],
        'step': step, 'heading_deg': yaw,
    })


def decision(name='door', candidate='new_1'):
    return Decision(
        action='navigate', navigation_check=clear_navigation_check(),
        reason='接近当前目标', scene={'target_seen': 'yes'},
        instruction=name,
        candidate_id=candidate,
        candidates=[
            {
                'id': candidate,
                'evidence_source': 'current',
                'kind': 'target',
                'expected_new_view': '绕过遮挡后查看通道',
                'description': name,
                'evidence': 'visible opening',
            }
        ],
    )


class ExplorationTests(unittest.TestCase):
    def test_scan_prediction_uses_backend_angle(self):
        state = ExecutionState(look_angle_deg=60)
        observe(state, yaw=95)
        observe(state, yaw=0, step=2)
        # 左转 60° 离已观察的 95° 还有 35°；默认 70° 则提示为已看方向，但不禁止有目的的复核。
        self.assertIn('left', state.scan.unobserved_look_directions())
        self.assertEqual(state.scan_state()['look_angle_deg'], 60)
        state.scan.look_angle_deg = 70
        self.assertNotIn('left', state.scan.unobserved_look_directions())

    def setUp(self):
        self.memory = ExecutionState()
        observe(self.memory)

    def fail_twice(self):
        candidate = 'new_1'
        for _ in range(2):
            cid = self.memory.accept(decision(candidate=candidate))
            self.memory.record_result(cid, {'reason': 'blocked', 'motion': {'displacement_m': 0}})
            self.memory.record_result(None, {'reason': 'recovered'})
            candidate = cid

    def test_existing_candidate_keeps_failures_when_reworded(self):
        self.fail_twice()
        with self.assertRaises(ValueError):
            self.memory.resolve(decision('white passage', candidate='C1'))
        self.assertEqual(len(self.memory.candidates), 1)
        self.assertEqual(self.memory.candidates[0]['failed_attempts'], 2)

    def test_rotation_does_not_reset_but_new_position_allows_new_approach(self):
        self.fail_twice()
        observe(self.memory, yaw=70, step=2)
        with self.assertRaises(ValueError):
            self.memory.resolve(decision(candidate='C1'))
        observe(self.memory, x=2, step=3)
        self.assertEqual(self.memory.accept(decision()), 'C2')

    def test_new_candidate_is_not_merged_by_estimated_direction(self):
        self.fail_twice()
        self.assertEqual(self.memory.accept(decision()), 'C2')

    def test_visible_historical_candidate_gets_local_attempt_record(self):
        old_id = self.memory.accept(decision())
        self.memory.record_result(old_id, {'reason': 'timeout', 'motion': {'displacement_m': 4.56}})
        previous = self.memory.snapshot()['local_candidates'][0]
        observe(self.memory, x=4.56, step=2)

        candidates, selected = self.memory.resolve(decision(candidate=old_id))
        self.assertEqual(selected, 'C2')
        self.assertEqual(len(self.memory.candidates), 1)
        self.assertEqual(candidates[0], previous)
        self.assertEqual(candidates[1]['place_id'], 'P2')
        self.assertEqual(candidates[1]['origin_candidate_id'], old_id)
        self.assertEqual(candidates[1]['failed_attempts'], 0)
        self.assertEqual(self.memory.accept(decision(candidate=old_id)), 'C2')

    def test_historical_alias_cannot_reset_local_failures(self):
        self.memory.accept(decision())
        observe(self.memory, x=2, step=2)
        for _ in range(2):
            cid = self.memory.accept(decision(candidate='C1'))
            self.memory.record_result(cid, {'reason': 'timeout', 'motion': {'displacement_m': .2}})
        for alias in ('C1', 'C2'):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                self.memory.resolve(decision(candidate=alias))
        self.assertEqual(len(self.memory.candidates), 2)

    def test_return_to_old_place_preserves_its_failures(self):
        self.fail_twice()
        observe(self.memory, x=2, step=2)
        self.assertEqual(self.memory.accept(decision(candidate='C1')), 'C2')
        observe(self.memory, x=0, step=3)
        with self.assertRaises(ValueError):
            self.memory.resolve(decision(candidate='C2'))
        self.assertEqual(self.memory.candidates[0]['failed_attempts'], 2)

    def test_historical_candidate_still_requires_current_visual_report(self):
        self.memory.accept(decision())
        observe(self.memory, x=2, step=2)
        with self.assertRaises(ValueError):
            self.memory.resolve(Decision(
                action='navigate', navigation_check=clear_navigation_check(), reason='探索', instruction='Approach the doorway.',
                candidate_id='C1',
            ))
        self.assertEqual(len(self.memory.candidates), 1)

    def test_duplicate_aliases_are_rejected_without_state_changes(self):
        self.memory.accept(decision())
        observe(self.memory, x=2, step=2)
        self.memory.accept(decision(candidate='C1'))
        proposal = decision(candidate='C1')
        proposal.candidates.append(decision(candidate='C2').candidates[0])
        previous = self.memory.snapshot()
        with self.assertRaises(ValueError):
            self.memory.resolve(proposal)
        self.assertEqual(self.memory.snapshot(), previous)

    def test_newly_allocated_id_cannot_be_guessed_in_same_response(self):
        proposal = decision()
        proposal.candidates.append(decision(candidate='C1').candidates[0])
        with self.assertRaises(ValueError):
            self.memory.resolve(proposal)
        self.assertEqual(self.memory.candidates, [])

    def test_finish_does_not_depend_on_candidate_resolution(self):
        self.fail_twice()
        observe(self.memory, x=2, step=2)
        previous = self.memory.snapshot()
        proposal = Decision(
            action='finish', reason='已找到', evidence='当前图清楚可见目标',
            candidates=decision(candidate='C99').candidates,
        )
        self.assertIsNone(self.memory.accept(proposal))
        self.assertEqual(self.memory.snapshot(), previous)

    def test_same_door_after_rotation_needs_no_angle(self):
        observe(self.memory, yaw=99.735, step=1)
        self.assertEqual(self.memory.accept(decision('右侧门洞')), 'C1')

        observe(self.memory, yaw=29.605, step=2)
        proposal = decision('左前方门洞', candidate='C1')
        proposal.instruction = 'Approach the open doorway in front-left.'
        self.assertEqual(self.memory.accept(proposal), 'C1')
        self.assertNotIn('heading_deg', self.memory.candidates[0])

    def test_legacy_candidate_angles_do_not_block_navigation(self):
        self.memory.accept(decision())
        observe(self.memory, yaw=29.605, step=2)
        for bearing in (-30, -25):
            with self.subTest(bearing=bearing):
                payload = decision(candidate='C1').model_dump()
                payload['candidates'][0]['bearing_deg'] = bearing
                proposal = Decision.model_validate(payload)
                self.assertEqual(self.memory.accept(proposal), 'C1')
                self.assertNotIn('bearing_deg', proposal.candidates[0].model_dump())

    def test_model_stop_with_no_displacement_is_no_progress_not_arrival(self):
        cid = self.memory.accept(decision())
        self.memory.record_result(cid, {'reason': 'model_stop', 'motion': {'displacement_m': 0.01}})
        self.assertEqual(self.memory.candidates[0]['failed_attempts'], 1)
        self.assertEqual(self.memory.candidates[0]['status'], 'approach_failed')

    def test_model_stop_after_movement_is_unverified(self):
        cid = self.memory.accept(decision())
        self.memory.record_result(cid, {'reason': 'model_stop', 'motion': {'displacement_m': 2}})
        self.assertEqual(self.memory.candidates[0]['status'], 'attempted_unverified')
        self.assertEqual(self.memory.candidates[0]['failed_attempts'], 0)

    def test_unknown_or_not_currently_seen_candidate_rejected(self):
        self.memory.accept(decision())
        with self.assertRaises(ValueError):
            self.memory.resolve(decision(candidate='C99'))
        with self.assertRaises(ValueError):
            self.memory.resolve(
                Decision(action='navigate', navigation_check=clear_navigation_check(), reason='test', instruction='go', candidate_id='C1')
            )

    def test_resolve_is_transactional_and_snapshots_are_detached(self):
        invalid = decision()
        invalid.candidate_id = 'C99'
        with self.assertRaises(ValueError):
            self.memory.resolve(invalid)
        self.assertEqual(self.memory.candidates, [])
        saved = self.memory.snapshot()
        saved['scan']['observed_headings_deg'].clear()
        self.assertEqual(len(self.memory.scan.scan_headings), 1)
        self.assertFalse(hasattr(self.memory, 'places'))

    def test_scan_keeps_direction_until_coverage_is_complete(self):
        self.memory.accept(Decision(action='look', direction='left', reason='检查左侧'))
        observe(self.memory, yaw=70, step=2)
        self.assertEqual(self.memory.scan.unobserved_look_directions(), ['left'])
        self.assertEqual(self.memory.allowed_look_directions(), ['left', 'right'])
        with self.assertRaisesRegex(ValueError, '复核'):
            self.memory.resolve(Decision(action='look', direction='right', reason='继续环视'))
        self.memory.resolve(Decision(action='look', direction='left', reason='继续同向环视'))

    def test_turn_counts_do_not_limit_new_or_repeated_observations(self):
        self.memory.scan.look_count = 100
        observe(self.memory, x=0.01, yaw=90, step=2)
        self.assertEqual(self.memory.allowed_look_directions(), ['left', 'right'])
        observe(self.memory, x=0.3, yaw=90, step=3)
        self.assertEqual(self.memory.allowed_look_directions(), ['left', 'right'])
        self.assertEqual(self.memory.scan.scan_headings, [90])

    def test_blocked_requires_position_change_not_new_door_name(self):
        cid = self.memory.accept(decision())
        self.memory.record_result(cid, {'reason': 'blocked', 'motion': {'displacement_m': 0}})
        self.assertTrue(self.memory.recovery_required)
        with self.assertRaises(ValueError):
            self.memory.resolve(decision('different door'))
        recovery = Decision(action='recover', reason='backtrack')
        self.memory.accept(recovery)
        self.memory.record_result(None, {'reason': 'recovery_unavailable'})
        self.assertFalse(self.memory.recovery_required)
        self.assertTrue(self.memory.must_change_position)
        with self.assertRaises(ValueError):
            self.memory.resolve(recovery)
        viewpoint = decision('visible floor')
        viewpoint.candidates[0].kind = 'viewpoint'
        self.memory.resolve(viewpoint)
