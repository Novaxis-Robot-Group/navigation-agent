from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tests.config_support import spatial_memory as SpatialMemory
from tests.config_support import execution_state as ExecutionState
from tests.support import clear_navigation_check
from navigation_agent.agent.schemas import Decision


def report(name='沙发', review=None):
    return Decision(
        action='look', direction='left', reason='检查遮挡',
        scene={'place_label': '疑似客厅', 'landmarks': [name], 'occluded': ['家具后方'],
               'objects': [{'name': name, 'description': '棕色家具', 'bearing_deg': 20}]},
        previous_action_review=review,
    )


class SpatialMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'memory.sqlite3'
        self.memory = SpatialMemory(self.path, '找到沙发')

    def observe(self, step, x=0, yaw=0, valid=True):
        return self.memory.observe(
            {'pose': {'x': x, 'y': 0}, 'yaw': math.radians(yaw), 'pose_valid': valid},
            f'images/{step:03d}.jpg', step,
        )

    def test_places_and_reference_frames_are_matched_from_sqlite(self):
        first = self.observe(1, yaw=179)
        self.memory.describe(first['id'], 1, report())
        second = self.observe(2, x=.2, yaw=-179)
        self.assertEqual(second['id'], first['id'])
        self.assertEqual(second['reference_image'], 'images/001.jpg')
        self.assertEqual(second['same_view_previous_observations'], 1)
        third = self.observe(3, yaw=90)
        self.assertEqual(third['reference_reason'], 'previous_action_start')
        self.assertEqual(third['same_view_previous_observations'], 0)
        fourth = self.observe(4, x=3)
        self.assertEqual(fourth['place_count'], 2)
        self.memory = SpatialMemory(self.path, '找到沙发')
        self.assertEqual(self.observe(5, yaw=179)['id'], first['id'])

    def test_objects_link_to_actual_observer_position_not_place_anchor(self):
        first = self.observe(1)
        self.memory.describe(first['id'], 1, report())
        second = self.observe(2, x=.5)
        self.memory.describe(second['id'], 2, report())
        context = self.memory.query(self.observe(3, x=3))
        sighting = context['objects'][0]
        self.assertEqual(sighting['seen_from_position'], [.5, 0])
        self.assertEqual(sighting['seen_from_place'], first['id'])
        self.assertEqual(sighting['image'], 'images/002.jpg')
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM objects').fetchone()[0], 2)
            scene = json.loads(db.execute('SELECT scene_json FROM semantic_frames WHERE step=1').fetchone()[0])
            self.assertNotIn('objects', scene)
            self.assertEqual(scene['occluded'], ['家具后方'])

    def test_invalid_pose_never_merges_places_or_invents_coordinates(self):
        first = self.observe(1, valid=False)
        self.memory.describe(first['id'], 1, report())
        second = self.observe(2, valid=False)
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(second['same_view_previous_observations'], 0)
        self.assertIsNone(self.memory.query(second)['objects'][0]['seen_from_position'])

    def test_rejected_action_leaves_only_raw_observation(self):
        view = self.observe(1)
        state = ExecutionState()
        state.observe(view)
        invalid = Decision(action='navigate', navigation_check=clear_navigation_check(), instruction='Go forward.', reason='测试',
                           candidate_id='C99', scene=report().scene)
        with self.assertRaises(ValueError):
            state.accept(invalid)
            self.memory.describe(view['id'], 1, invalid)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM objects').fetchone()[0], 0)
            self.assertIsNone(db.execute('SELECT scene_json FROM semantic_frames').fetchone()[0])

    def test_review_retains_uncertainty_and_links_previous_step(self):
        self.observe(1)
        second = self.observe(2, x=2)
        self.memory.describe(second['id'], 2, report(review={
            'outcome': 'uncertain', 'evidence': '仍见沙发，不能确认穿门',
        }))
        context = self.memory.query(self.observe(3, x=4))
        self.assertEqual(context['recent_reviews'][0]['action_step'], 1)
        self.assertEqual(context['recent_reviews'][0]['outcome'], 'uncertain')

    def test_bounded_retrieval_keeps_old_goal_evidence(self):
        for step in range(1, 13):
            view = self.observe(step, x=step * 4)
            self.memory.describe(view['id'], step, report('沙发' if step == 1 else '桌子'))
        context = self.memory.query(self.observe(13, x=60))
        self.assertEqual(context['objects'][0]['name'], '沙发')
        self.assertEqual(context['semantic_frames'][0]['step'], 1)
        self.assertLessEqual(len(context['objects']), 6)
        self.assertLessEqual(len(context['semantic_frames']), 4)
        self.assertLessEqual(len(context['places']), 8)

    def test_new_run_has_no_old_memories(self):
        view = self.observe(1)
        self.memory.describe(view['id'], 1, report())
        other = SpatialMemory(Path(self.temp.name) / 'another.sqlite3', '找到沙发')
        view = other.observe({'pose': {'x': 0, 'y': 0}, 'yaw': 0}, 'images/001.jpg', 1)
        self.assertEqual(other.query(view)['objects'], [])
        self.assertEqual(other.query(view)['semantic_frames'], [])

    def test_rejected_reports_remain_claims_and_corrections_do_not_duplicate_objects(self):
        view = self.observe(1)
        proposal = report()
        self.memory.describe(view['id'], 1, proposal, action_accepted=False)
        first = self.memory.frames([1])[0]
        self.assertEqual(first['observation_id'], 'O1')
        self.assertEqual(first['action_status'], 'rejected_action')
        self.assertEqual(first['source'], 'model_visual_report')
        # 没有填写场景的纠错不能抹掉之前的线索。
        self.memory.describe(view['id'], 1, Decision(action='look', direction='left', reason='复核'))
        self.assertEqual(self.memory.frames([1])[0], first)
        self.memory.describe(view['id'], 1, report('柜子'))
        accepted = self.memory.frames([1])[0]
        self.assertEqual(accepted['action_status'], 'accepted')
        self.assertEqual(accepted['scene']['objects'][0]['name'], '柜子')
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM objects').fetchone()[0], 1)

    def test_existing_database_without_report_status_remains_readable(self):
        view = self.observe(1)
        self.memory.describe(view['id'], 1, report())
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('ALTER TABLE semantic_frames DROP COLUMN report_status')
            db.commit()
        self.memory = SpatialMemory(self.path, '找到沙发')
        frame = self.memory.frames([1])[0]
        self.assertEqual(frame['action_status'], 'accepted')
        self.assertEqual(frame['scene']['landmarks'], ['沙发'])

    def test_only_three_tables_and_no_legacy_text_memory(self):
        with closing(sqlite3.connect(self.path)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {'objects', 'places', 'semantic_frames'})
        self.assertNotIn('memory', Decision.model_json_schema()['properties'])
        state = ExecutionState()
        state.observe(self.observe(1))
        self.assertNotIn('places', state.context())
        self.assertFalse(hasattr(state, 'places'))


if __name__ == '__main__':
    unittest.main()
