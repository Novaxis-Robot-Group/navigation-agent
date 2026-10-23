from pathlib import Path
import tempfile
import unittest

from tests.config_support import spatial_memory as SpatialMemory
from navigation_agent.agent.schemas import Decision
from navigation_agent.agent.storage.memory_visualize import write_memory_visualization

def memory_run(directory):
    """生成包含决策前快照与后续复核的离线记录，不连接模型或机器人。"""
    import json
    from navigation_agent.agent.memory.trajectory import TrajectoryMemory

    memory = SpatialMemory(directory / 'memory.sqlite3', '走到棕色沙发旁边')
    trajectory = TrajectoryMemory(memory)
    memory.observe({'pose': {'x': 0, 'y': 1}, 'yaw': 0}, 'images/001.jpg', 1)
    trajectory.start(1, 'navigate', '接近左侧沙发', {'instruction': 'Approach the sofa.'})
    trajectory.result(1, {'reason': 'model_stop', 'motion': {'displacement_m': 2}})
    view = memory.observe({'pose': {'x': 2, 'y': 1}, 'yaw': 0}, 'images/002.jpg', 2)
    trajectory.observe(2)
    event = {
        'type': 'decision_input', 'step': 2,
        'model_context': {
            'observation_step': 2,
            'spatial_memory': memory.query(view), 'trajectory_memory': trajectory.query(2),
            'execution': {
                'scan': {'look_count': 1},
                'exploration_session': {'phase': 'target_confirmation', 'views': []},
                'target_tracking': {'description': '棕色沙发', 'phase': 'verifying_arrival',
                                    'confirmation_turns': 2},
            },
        },
        'images': [{'image': 'images/002.jpg', 'uses': [{'purpose': '当前图像 A'}]},
                   {'image': 'images/001.jpg', 'uses': [{'purpose': '目标历史证据'}]}],
    }
    event['model_context']['trajectory_memory']['repeated_attempts'] = ['重复提示 <script>不能执行</script>']
    # 后续复核只能出现在经历详情，不能回填进上面的决策前快照。
    review = {'status': 'temporarily_lost', 'observation_step': 2, 'evidence': '后续复核：沙发局部暂不可确认'}
    trajectory.review(2, target_review=review)
    trajectory.start(2, 'look', '检查右边缘', {'direction': 'right'})
    events = [
        {'type': 'observation', 'step': 2, 'image': 'images/002.jpg'}, event,
        {'type': 'decision', 'step': 2, 'decision': {
            'action': 'look', 'reason': '检查近处边缘', 'target_review': review}},
        {'type': 'action_result', 'step': 1, 'result': {'reason': 'model_stop'}},
    ]
    (directory / 'events.jsonl').write_text(
        '\n'.join(json.dumps(item, ensure_ascii=False) for item in events), encoding='utf-8',
    )
    (directory / 'result.json').write_text(json.dumps({
        'id': 'offline-visual-check', 'goal': '走到棕色沙发旁边（离线示例）', 'status': 'running', 'step': 2,
    }, ensure_ascii=False), encoding='utf-8')
    return event


class MemoryVisualizationTests(unittest.TestCase):
    def test_export_preserves_database_and_escapes_model_text(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            database = directory / 'memory.sqlite3'
            memory = SpatialMemory(database, '找桌子')
            view = memory.observe({'pose': {'x': 0, 'y': 1}, 'yaw': 0}, '../outside.jpg', 1)
            memory.describe(view['id'], 1, Decision(
                action='look', direction='left', reason='观察',
                scene={'place_label': '<script>alert(1)</script>',
                       'objects': [{'name': '桌子', 'description': '白色桌子'}]},
            ))
            memory.observe({'pose_valid': False}, 'images/002.jpg', 2)
            before = database.read_bytes()
            output = write_memory_visualization(database)
            html = output.read_text(encoding='utf-8')
            self.assertEqual(output.name, 'memory_visualize.html')
            self.assertEqual(database.read_bytes(), before)
            self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', html)
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertNotIn('src="../outside.jpg"', html)
            self.assertIn('白色桌子', html)
            self.assertIn('仅原始观察', html)
            self.assertIn('定位无效', html)
            self.assertIn('id="frame-1"', html)
            # 导出连接必须关闭，Windows 才能在收尾时移动数据库。
            database.rename(directory / 'moved.sqlite3')

    def test_rejected_report_is_labeled_without_claiming_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            memory = SpatialMemory(directory / 'memory.sqlite3', '找沙发')
            view = memory.observe({'pose': {'x': 0, 'y': 0}, 'yaw': 0}, 'images/001.jpg', 1)
            memory.describe(view['id'], 1, Decision(
                action='look', direction='left', reason='复核',
                scene={'visible_ground': '当前图左侧有可见地面'},
            ), action_accepted=False)
            html = write_memory_visualization(directory).read_text()
            self.assertIn('对应动作被拒绝，未执行', html)
            self.assertIn('当前图左侧有可见地面', html)

    def test_empty_database_and_missing_database(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.assertIsNone(write_memory_visualization(directory))
            SpatialMemory(directory / 'memory.sqlite3', '目标')
            output = write_memory_visualization(directory)
            self.assertIn('暂无记忆', output.read_text(encoding='utf-8'))

    def test_trajectory_and_actual_inputs_are_distinct_and_preserve_database(self):
        import json
        from navigation_agent.agent.storage.memory_sections import render_memory_input
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            event = memory_run(directory)
            database = directory / 'memory.sqlite3'
            before = database.read_bytes()
            html = write_memory_visualization(directory).read_text(encoding='utf-8')
            self.assertEqual(database.read_bytes(), before)
            self.assertIn('id="action-1"', html)
            self.assertIn('href="#frame-2"', html)
            self.assertIn('净位移 2.00 m', html)
            self.assertIn('执行结果未返回，状态未知', html)
            self.assertIn('终点：尚无对应观察', html)
            self.assertIn('后续复核：沙发局部暂不可确认', html)
            self.assertIn('href="index.html#decision-input-2"', html)
            self.assertIn('id="input-2"', html)
            self.assertIn('&lt;script&gt;不能执行&lt;/script&gt;', html)
            self.assertNotIn('<script>不能执行</script>', html)
            snapshot = render_memory_input(event)
            self.assertIn('尚未视觉复核', snapshot)
            self.assertNotIn('后续复核：', snapshot)
            # 无效尾行不能阻止导出已有记忆，且不修改事件源。
            with (directory / 'events.jsonl').open('a') as stream:
                stream.write('\n{"incomplete":')
            events_before = (directory / 'events.jsonl').read_bytes()
            html = write_memory_visualization(directory).read_text(encoding='utf-8')
            self.assertIn('1 行事件不完整', html)
            self.assertEqual((directory / 'events.jsonl').read_bytes(), events_before)

    def test_old_memory_and_forced_recovery_do_not_claim_model_read(self):
        from navigation_agent.agent.storage.memory_sections import render_memory_input
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            SpatialMemory(directory / 'memory.sqlite3', '目标')
            html = write_memory_visualization(directory).read_text(encoding='utf-8')
            self.assertIn('旧版记录未保存轨迹表', html)
            self.assertIn('未记录决策输入快照', html)
        self.assertIn('旧日志未记录', render_memory_input({}))
        snapshot = render_memory_input({
            'model_context': {'execution': {'scan': {'recovery_required': True}},
                              'trajectory_memory': {'recent': ['不能展示为模型已读']}},
        })
        self.assertIn('本轮未调用模型', snapshot)
        self.assertNotIn('不能展示为模型已读', snapshot)
