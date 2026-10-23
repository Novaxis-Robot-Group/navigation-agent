"""验证摘要不泄漏大段上下文，动态状态正确收尾。"""

from io import StringIO
import unittest
from unittest.mock import patch

from navigation_agent.agent.storage import terminal
from navigation_agent.agent.storage.terminal import TerminalLog, concise

try:
    from rich.console import Console
except ImportError:
    Console = None


@unittest.skipIf(Console is None, '当前环境未安装 Rich')
class TerminalLogTests(unittest.TestCase):
    def make_log(self, interactive=False):
        stream = StringIO()
        console = Console(file=stream, force_terminal=interactive, width=100, color_system=None)
        return TerminalLog(console), stream

    def test_redirected_output_keeps_progress_without_raw_payloads(self):
        log, stream = self.make_log()
        record = {'id': 'task-a', 'step': 1}
        log.event(record, 'connecting', {}, 480)
        log.event(record, 'skill', {'name': 'object_search'}, 480)
        log.event(record, 'decision_input', {'model_context': 'SECRET_CONTEXT'}, 480)
        log.event(record, 'prompt', {'sha256': 'SECRET_HASH'}, 480)
        log.event(record, 'decision', {'decision': {'action': 'navigate', 'reason': '[red]去右侧'}}, 480)
        log.event(record, 'action_input', {'action': 'navigate', 'instruction': 'Walk right.'}, 480)
        log.event(record, 'action_result', {'result': {'reason': 'timeout', 'motion': {'displacement_m': .03}}}, 480)
        log.event(record, 'finished', {'status': 'budget_exhausted', 'summary': 'Maximum decision steps reached'}, 480)
        output = stream.getvalue()
        for expected in ('object_search', '正在分析画面', '[red]去右侧', 'navigate', '导航超时', '0.03 m', '已达到最大决策轮数'):
            self.assertIn(expected, output)
        for hidden in ('SECRET_CONTEXT', 'SECRET_HASH', '\x1b'):
            self.assertNotIn(hidden, output)
        self.assertFalse(log.active)
        self.assertIsNone(log.live)

    def test_live_status_survives_one_task_finishing_and_cleans_up_last(self):
        log, stream = self.make_log(interactive=True)
        first = {'id': 'task-a', 'step': 1}
        second = {'id': 'task-b', 'step': 2}
        try:
            log.event(first, 'observing', {}, 480)
            log.event(second, 'prompt', {}, 480)
            self.assertIsNotNone(log.live)
            with patch.object(terminal, 'monotonic', return_value=log.active['task-b']['since'] + 4):
                log.console.print(log.render())
            log.event(first, 'finished', {'status': 'cancelled'}, 480)
            self.assertIsNotNone(log.live)
            self.assertEqual(set(log.active), {'task-b'})
            log.event(second, 'finished', {'status': 'failed', 'summary': '离线错误'}, 480)
            self.assertIsNone(log.live)
            self.assertFalse(log.active)
            self.assertIn('4 s', stream.getvalue())
        finally:
            log.stop(first)
            log.stop(second)

    def test_stop_signal_is_not_reported_as_target_found(self):
        log, stream = self.make_log()
        log.event({'id': 'a'}, 'action_result', {'result': {'reason': 'model_stop'}}, 480)
        self.assertIn('待重新观察', stream.getvalue())
        self.assertNotIn('找到目标', stream.getvalue())
        log.stop({'id': 'a'})


class PlainTerminalTests(unittest.TestCase):
    def test_missing_rich_still_prints_and_finishes(self):
        with patch.object(terminal, 'Console', None), patch.object(terminal.sys, 'stderr', StringIO()) as stream:
            log = TerminalLog()
            log.event({'id': 'a'}, 'connecting', {}, 480)
            log.event({'id': 'a'}, 'finished', {'status': 'cancelled'}, 480)
            self.assertIn('正在连接后端', stream.getvalue())
            self.assertIn('任务已取消', stream.getvalue())
            self.assertFalse(log.active)

    def test_model_text_is_bounded_and_control_characters_removed(self):
        text = concise('a\x1b[2J\r\n' + '字' * 300)
        self.assertLessEqual(len(text), 160)
        self.assertNotIn('\x1b', text)
        self.assertNotIn('\n', text)
        self.assertTrue(text.endswith('…'))
