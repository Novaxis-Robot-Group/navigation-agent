import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from navigation_agent.app.agent import api as main
from tests.support import patch_runs


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_id = 'b' * 32
        directory = self.root / self.run_id
        (directory / 'images').mkdir(parents=True)
        (directory / 'result.json').write_text(
            json.dumps({'id': self.run_id, 'status': 'running', 'goal': '历史任务'}),
            encoding='utf-8',
        )
        (directory / 'events.jsonl').write_text(
            '{"type":"started"}\n{"type":"observation"}\n', encoding='utf-8'
        )
        (directory / 'images/001.jpg').write_bytes(b'\xff\xd8image')
        self.patches = [patch_runs(self.root), patch.object(main, 'records', {})]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def test_recovers_history_without_resuming_work(self):
        task = await main.get_task(self.run_id)
        self.assertEqual(task['status'], 'interrupted')
        self.assertEqual((await main.list_tasks(limit=30))['tasks'][0]['id'], self.run_id)
        self.assertEqual((await main.cancel_task(self.run_id))['status'], 'interrupted')

    async def test_events_cursor_and_image_allowlist(self):
        result = await main.get_events(self.run_id, after=1)
        self.assertEqual(result['next'], 2)
        self.assertEqual(result['events'][0]['type'], 'observation')
        self.assertEqual((await main.task_image(self.run_id, '001.jpg')).media_type, 'image/jpeg')
        with self.assertRaises(HTTPException):
            await main.task_image(self.run_id, '../../.env')
        with self.assertRaises(HTTPException):
            await main.get_task('../agent')

    async def test_timestamp_directory_keeps_id_routes_after_restart(self):
        record = {
            'id': self.run_id,
            'status': 'completed',
            'goal': 'test',
            'started_at': '2026-09-14T03:11:01.123456+00:00',
            'ended_at': '2026-09-14T03:15:43.654321+00:00',
        }
        main.log_store().finish_log_directory(record)
        main.log_store().save_result(record)
        self.assertEqual(
            record['log_directory'], '2026-09-14_11-11-01__2026-09-14_11-15-43_success'
        )
        self.assertFalse((self.root / self.run_id).exists())
        self.assertEqual((await main.get_task(self.run_id))['status'], 'completed')
        self.assertEqual((await main.list_tasks(limit=30))['tasks'][0]['id'], self.run_id)
        self.assertEqual((await main.get_events(self.run_id, after=0))['next'], 2)
        self.assertTrue(Path((await main.task_image(self.run_id, '001.jpg')).path).is_file())

    async def test_timestamp_collision_does_not_overwrite_existing_logs(self):
        record = {
            'id': self.run_id,
            'status': 'budget_exhausted',
            'started_at': '2026-09-14T03:00:00+00:00',
            'ended_at': '2026-09-14T03:01:00+00:00',
        }
        occupied = self.root / (
            main.log_store().log_time(record['started_at']) + '__' + main.log_store().log_time(record['ended_at']) + '_failed'
        )
        occupied.mkdir()
        (occupied / 'marker').write_text('keep')
        main.log_store().finish_log_directory(record)
        self.assertEqual((occupied / 'marker').read_text(), 'keep')
        self.assertTrue(record['log_directory'].endswith(self.run_id + '_failed'))


if __name__ == '__main__':
    unittest.main()
