import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from navigation_agent.app.agent import api as main
from tests.support import patch_runs

from navigation_agent.agent.storage import logs, report
from functools import partial
from tests.config_support import agent_settings
write_report = partial(report.write_report, utc_offset_minutes=agent_settings().storage.utc_offset_minutes)


class LogReportTests(unittest.TestCase):
    def test_key_nodes_keep_failures_without_capture_and_retry_noise(self):
        events = [
            {'type': 'started', 'goal': '找到床'},
            {'type': 'passive_observation', 'image': 'images/001.jpg'},
            {'type': 'decision_input', 'context': {'private': '诊断'}},
            {'type': 'decision_error', 'step': 2, 'category': 'output_format'},
            {'type': 'decision_error', 'step': 2, 'category': 'output_format'},
            {'type': 'decision', 'step': 2, 'decision': {'action': 'navigate', 'reason': '去门口看看'}},
            {'type': 'action_result', 'result': {'reason': 'model_stop'}},
            {'type': 'action_result', 'result': {'reason': 'blocked'}},
            {'type': 'target_detection_review', 'verdict': 'rejected', 'evidence': '这是椅子'},
            {'type': 'completion_check', 'step': 3, 'confirmed': False, 'evidence': '仍看不清'},
            {'type': 'decision_error', 'step': 3, 'category': 'completion_verification'},
            {'type': 'finished', 'status': 'failed', 'summary': '没有找到目标'},
        ]
        nodes = list(report.key_events(events))
        self.assertEqual([n['type'] for n in nodes],
                         ['started', 'decision_error', 'decision', 'action_result',
                          'target_detection_review', 'completion_check', 'finished'])
        self.assertIn('受阻', nodes[3]['description'])
        self.assertEqual(nodes[4]['label'], '排除误检')
        self.assertEqual(nodes[-1]['label'], '未完成')
        self.assertEqual(len(events), 12)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
            page = write_report(root, {'goal': '找到床', 'status': 'failed'}).read_text()
            visible = page.split('<div id="events">', 1)[1].split('</section>', 1)[0]
            self.assertNotIn('<pre>', visible)
            self.assertNotIn('passive_observation', visible)
            self.assertIn('id="step-2"', visible)
            self.assertIn('没有找到目标', visible)
            self.assertNotIn('<details', page)
            self.assertNotIn('<pre', page)

    def test_detection_retry_and_sampler_stop_remain_distinct_visible_events(self):
        events = [
            {'type': 'target_watch_degraded', 'stage': 'observation',
             'reason': 'TimeoutError', 'retrying': True},
            {'type': 'target_watch_degraded', 'stage': 'sampler',
             'reason': 'AttributeError', 'retrying': False},
            {'type': 'target_detection_frame'},
            {'type': 'action_result', 'result': {'reason': 'agent_interrupt',
                                               'detail': 'observation_unavailable'}},
        ]
        nodes = report.key_events(events)
        self.assertEqual([node['label'] for node in nodes],
                         ['行进采图暂不可用', '本段途中检测已停止', '目标检测已恢复', '位置观测失效，已停车'])
        legacy = report.key_event({'type': 'target_watch_degraded', 'reason': 'AttributeError'})
        self.assertEqual(legacy['label'], '本段途中检测已停止')
        unsupported = report.key_event({'type': 'action_result', 'result': {
            'reason': 'model_stop', 'motion_limit_unavailable': True}})
        self.assertEqual(unsupported['label'], '位移限制未生效')

    def test_navigation_instruction_comes_from_executed_tool_input(self):
        events = [
            {'type': 'decision', 'step': 1, 'decision': {'action': 'navigate', 'instruction': 'Draft.', 'reason': '去门口'}},
            {'type': 'action_input', 'step': 1, 'action': 'navigate', 'instruction': 'Go through the <open> doorway.'},
            {'type': 'decision', 'step': 2, 'decision': {'action': 'navigate', 'instruction': 'Unsent.'}},
            {'type': 'decision', 'step': 3, 'decision': {'action': 'look'}},
        ]
        nodes = report.key_events(events)
        self.assertEqual(nodes[0]['instruction'], events[1]['instruction'])
        self.assertNotIn('instruction', nodes[1])
        self.assertNotIn('instruction', nodes[2])
        html = report.render_key_event(nodes[0], 480)
        self.assertIn('VLN 指令：', html)
        self.assertIn('Go through the &lt;open&gt; doorway.', html)
        self.assertNotIn('Draft.', html)

    def test_offline_report_escapes_content_and_skips_partial_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            events = [
                {'type': 'observation', 'step': 1, 'image': 'images/001.jpg'},
                {
                    'type': 'decision', 'step': 1,
                    'decision': {
                        'action': 'look',
                        'reason': '<script>alert(1)</script>',
                        'memory': '沙发在左侧',
                    },
                },
                {'type': 'observation', 'image': 'images/001.jpg'},
                {'type': 'observation', 'image': 'javascript:alert(1)'},
            ]
            (root / 'events.jsonl').write_text(
                '\n'.join(json.dumps(e) for e in events) + '\n{"unfinished":', encoding='utf-8'
            )
            page = write_report(root, {'goal': '寻找沙发', 'status': 'completed'}).read_text(
                encoding='utf-8'
            )
            self.assertIn('寻找沙发', page)
            self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', page)
            self.assertNotIn('<script>alert(1)</script>', page)
            self.assertIn('src="images/001.jpg"', page)
            self.assertNotIn('href="javascript:', page)
            self.assertIn('有 1 行日志不完整', page)

    def test_event_updates_report_and_report_failure_preserves_json(self):
        with tempfile.TemporaryDirectory() as folder, patch_runs(Path(folder)):
            record = {'id': 'a' * 32, 'goal': '寻找沙发', 'status': 'running'}
            directory = main.log_store().run_directory(record)
            directory.mkdir()
            main.log_store().event(record, 'started', goal=record['goal'])
            self.assertTrue((directory / 'index.html').is_file())
            record['status'] = 'completed'
            main.log_store().event(record, 'finished', status='completed', summary='找到沙发')
            self.assertIn('找到沙发', (directory / 'index.html').read_text(encoding='utf-8'))
            with patch.object(
                logs, 'write_report', side_effect=OSError('locked')
            ), self.assertLogs('uvicorn.error', level='ERROR'):
                main.log_store().save_result(record)
            self.assertEqual(
                json.loads((directory / 'result.json').read_text(encoding='utf-8'))['status'],
                'completed',
            )

    def test_manifest_deduplicates_images_without_losing_purpose_or_old_report_support(self):
        from navigation_agent.agent.context.images import image_manifest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'images').mkdir()
            for name in ('001.jpg', '002.jpg'):
                (root / 'images' / name).write_bytes(name.encode())
            view = {'step': 2, 'image': 'images/002.jpg', 'reference_image': 'images/001.jpg',
                    'reference_reason': 'similar_view', 'heading_deg': 70,
                    'observer_position': [.5, 0]}
            frames = [{'id': 'O1', 'step': 1, 'image': 'images/001.jpg', 'heading_deg': 0,
                       'observer_position': [0, 0], 'candidate_ids': ['C1']},
                      {'id': 'O2', 'step': 2, 'image': 'images/002.jpg', 'heading_deg': 70,
                       'observer_position': [.5, 0]}]
            manifest = image_manifest(root, view, frames, frames[0])
            self.assertEqual([item['image'] for item in manifest], ['images/001.jpg', 'images/002.jpg'])
            self.assertEqual(len(manifest[0]['uses']), 2)
            self.assertEqual(len(manifest[1]['uses']), 1)
            manifest[0]['uses'][0]['purpose'] += '<script>'
            nodes = report.key_events([
                {'type': 'observation', 'step': 2, 'image': 'images/002.jpg'},
                {'type': 'decision_input', 'step': 2, 'images': manifest},
                {'type': 'decision', 'step': 2, 'decision': {'action': 'look', 'reason': '核实目标'}},
            ])
            self.assertEqual(nodes[0]['frames'][0]['image'], 'images/002.jpg')

    def test_room_references_share_the_two_historical_direction_slots(self):
        from navigation_agent.agent.context.images import image_manifest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = []
            for step in range(1, 7):
                image = f'{step}.jpg'
                (root / image).write_bytes(b'image')
                frames.append({'step': step, 'id': f'O{step}', 'image': image,
                               'heading_deg': 0, 'candidate_ids': [f'C{step}']})
            regions = [{**frames[0], 'region_id': 'R1'}, {**frames[1], 'region_id': 'R2'}]
            manifest = image_manifest(root, frames[-1], frames[:-1], frames[-2], region_frames=regions)
            self.assertEqual([item['image'] for item in manifest], ['5.jpg', '1.jpg', '4.jpg', '6.jpg'])
            self.assertIn('区域 R1', manifest[1]['uses'][0]['purpose'])
            self.assertEqual(manifest[-1]['uses'][0]['purpose'], '当前图像 A')
            self.assertIn('历史方向', manifest[2]['uses'][0]['purpose'])

    def test_route_references_keep_identity_and_deduplicate_existing_images(self):
        from navigation_agent.agent.context.images import image_manifest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for image in ('1.jpg', '2.jpg'):
                (root / image).write_bytes(b'image')
            current = {'step': 2, 'image': '2.jpg'}
            route = {'step': 1, 'image': '1.jpg', 'region_id': 'R1', 'route_id': 'C1'}
            manifest = image_manifest(root, current, [], None,
                                      region_frames=[route],
                                      route_frames=[route, {**route, 'destination_region': 'R2'}])
            self.assertEqual([frame['image'] for frame in manifest], ['1.jpg', '2.jpg'])
            purposes = [use['purpose'] for use in manifest[0]['uses']]
            self.assertTrue(any('尚未尝试入口 C1' in value for value in purposes))
            self.assertTrue(any('R1 通向 R2' in value for value in purposes))
            self.assertEqual(manifest[-1]['uses'][0]['purpose'], '当前图像 A')

    def test_nearby_direction_image_does_not_require_a_candidate(self):
        from navigation_agent.agent.context.images import image_manifest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = []
            for step, heading in ((1, 60), (2, 180), (3, 0)):
                image = f'{step}.jpg'
                (root / image).write_bytes(b'image')
                frames.append({'step': step, 'id': f'O{step}', 'image': image,
                               'heading_deg': heading, 'candidate_ids': [],
                               'observer_position': [0, 0]})
            manifest = image_manifest(root, frames[-1], frames, None)
            self.assertEqual(manifest[0]['image'], '2.jpg')
            self.assertEqual(manifest[-1]['image'], '3.jpg')
            self.assertEqual(len(manifest), 3)

    def test_report_shows_target_loss_without_debug_panels(self):
        from tests.agent_tests.test_memory_visualize import memory_run
        from navigation_agent.agent.storage.memory_visualize import write_memory_visualization
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            memory_run(directory)
            write_memory_visualization(directory)
            html = write_report(directory).read_text()
            self.assertIn('目标暂时不可见', html)
            self.assertIn('src="images/002.jpg"', html)
            self.assertIn('id="decision-input-2"', html)
            self.assertNotIn('memory_visualize.html', html)
            self.assertNotIn('<details', html)
            self.assertNotIn('<pre', html)
            self.assertNotIn('<script>不能执行</script>', html)

    def test_images_match_recorded_decision_detection_and_review_times(self):
        events = [
            {'type': 'observation', 'step': 1, 'image': 'images/001.jpg'},
            {'type': 'decision', 'step': 1, 'decision': {'action': 'navigate', 'reason': '去门口'}},
            {'type': 'target_watch_degraded', 'reason': 'TimeoutError'},
            {'type': 'target_detection_frame', 'inference_ms': 80},
            {'type': 'target_detection_hit', 'step': 2, 'candidate_id': 'a',
             'image': 'images/002.jpg', 'boxed_image': 'images/detection-002.jpg'},
            {'type': 'observation', 'step': 3, 'image': 'images/003.jpg'},
            {'type': 'target_detection_review', 'candidate_id': 'a', 'verdict': 'rejected', 'evidence': '这不是床'},
            {'type': 'decision', 'step': 9, 'decision': {'action': 'look', 'reason': '旧日志缺图'}},
            {'type': 'observation', 'step': 9, 'image': 'images/009.jpg'},
            {'type': 'finished', 'status': 'failed', 'summary': '没有找到'},
        ]
        nodes = report.key_events(events)
        self.assertEqual(nodes[0]['frames'][0]['image'], 'images/001.jpg')
        self.assertEqual(nodes[1]['label'], '本次目标检测超时')
        self.assertEqual(nodes[2]['label'], '目标检测已恢复')
        review = next(n for n in nodes if n['type'] == 'target_detection_review')
        self.assertEqual([f['image'] for f in review['frames']],
                         ['images/detection-002.jpg', 'images/003.jpg'])
        missing = next(n for n in nodes if n.get('step') == 9 and n['type'] == 'decision')
        self.assertEqual(missing['frames'], [])
        self.assertEqual(nodes[-1]['frames'][0]['image'], 'images/009.jpg')
        self.assertEqual(nodes[-1]['label'], '未完成')
        self.assertIsNone(report.event_frame('../secret.jpg', '不能越界'))
        self.assertIsNone(report.event_frame('images/001.jpg" onerror="alert(1)', '不能注入'))

    def test_live_snapshot_exists_before_log_link_and_failure_keeps_events(self):
        from tests.config_support import spatial_memory
        with tempfile.TemporaryDirectory() as folder, patch_runs(Path(folder)):
            record = {'id': 'b' * 32, 'goal': '找沙发', 'status': 'running'}
            store = main.log_store()
            directory = store.run_directory(record)
            directory.mkdir()
            spatial_memory(directory / 'memory.sqlite3', record['goal'])
            store.event(record, 'decision_input', step=1, model_context={})
            self.assertTrue((directory / 'memory_visualize.html').is_file())
            self.assertNotIn('href="memory_visualize.html"', (directory / 'index.html').read_text())
            with patch.object(logs, 'write_memory_visualization', side_effect=OSError('locked')):
                with self.assertLogs('uvicorn.error', level='ERROR'):
                    store.event(record, 'action_result', step=1, result={'reason': 'turned'})
            events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['result']['reason'], 'turned')

    def test_cli_export_keeps_memory_file_separate_from_reader(self):
        from navigation_agent.app.reports.main import main as export
        from tests.agent_tests.test_memory_visualize import memory_run
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            memory_run(directory)
            with patch('sys.argv', ['report', str(directory)]), patch('builtins.print'):
                export()
            self.assertTrue((directory / 'memory_visualize.html').is_file())
            self.assertNotIn('href="memory_visualize.html"', (directory / 'index.html').read_text())


class EntranceEvidenceTests(unittest.TestCase):
    def test_frontier_and_both_sides_of_known_passage_are_available_without_duplicating_current(self):
        from pathlib import Path
        import tempfile
        from navigation_agent.agent.context.images import image_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(1, 5):
                (root / f'{number}.jpg').write_bytes(b'image')
            current = {'step': 4, 'image': '4.jpg'}
            routes = [
                {'step': 1, 'image': '1.jpg', 'route_id': 'C2', 'region_id': 'R1'},
                {'step': 2, 'image': '2.jpg', 'route_id': 'C1', 'region_id': 'R1',
                 'destination_region': 'R2'},
                {'step': 3, 'image': '3.jpg', 'route_id': 'C1', 'region_id': 'R1',
                 'destination_region': 'R2', 'at_destination': True},
            ]
            manifest = image_manifest(root, current, [], None, route_frames=routes)
            self.assertEqual([item['image'] for item in manifest], ['1.jpg', '2.jpg', '3.jpg', '4.jpg'])
            self.assertIn('通过入口 C1 后在 R2', manifest[2]['uses'][0]['purpose'])
