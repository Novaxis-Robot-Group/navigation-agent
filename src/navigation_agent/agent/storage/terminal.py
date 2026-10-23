"""终端执行摘要；完整事件仍由 LogStore 原样保存。"""

from datetime import datetime, timedelta, timezone
import sys
from time import monotonic
import unicodedata

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
except ImportError:
    # 旧环境未安装新增依赖时，仍输出简短文本。
    Console = None

STYLES = {
    '任务': '#b49a80', 'SKILL': '#b49a80', 'TOOL': '#83a7ac',
    '观察': 'dim', '判断': 'default', '结果': '#9cad91',
    '等待': 'dim', '警告': '#d1b477', '失败': '#cb8580', '结束': '#b49a80',
}


def concise(value, limit=160):
    """限制长度并过滤控制字符，模型文本不作为终端指令。"""
    clean = ''.join(
        char if not unicodedata.category(char).startswith('C') else ' '
        for char in str(value or '')
    )
    clean = ' '.join(clean.split())
    return clean if len(clean) <= limit else clean[:limit - 1] + '…'


class TerminalLog:
    """进程共用一个状态区，同时运行的任务分别显示。"""

    def __init__(self, console=None):
        self.console = console if console is not None else (
            Console(stderr=True, highlight=False) if Console is not None else None
        )
        self.active = {}
        self.live = None

    def write(self, record, label, message, offset):
        timestamp = datetime.now(timezone(timedelta(minutes=offset))).strftime('%H:%M:%S')
        identity = f" [{concise(record['id'], 8)}]" if len(self.active) > 1 else ''
        prefix = f"  {timestamp}{identity}  "
        message = concise(message)
        label_width = sum(2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1 for char in label)
        padded_label = label + ' ' * max(0, 6 - label_width)
        if self.console is None:
            print(f'{prefix}{padded_label} {message}', file=sys.stderr, flush=True)
            return
        line = Text(prefix, style='dim')
        line.append(f'{padded_label} ', style=STYLES[label])
        line.append(message)
        self.console.print(line)

    def render(self):
        rows = []
        for task_id, state in list(self.active.items()):
            elapsed = monotonic() - state['since']
            identity = f' [{concise(task_id, 8)}]' if len(self.active) > 1 else ''
            step = f" · 第 {state['step']} 轮" if state['step'] else ''
            skill = f" · {state['skill']}" if state['skill'] else ''
            text = Text(f" {state['message']}  {elapsed:.0f} s{step}{skill}{identity}")
            state['spinner'].update(text=text)
            rows.append(state['spinner'])
        return Group(*rows)

    def waiting(self, record, message, offset, *, skill=None, announce=True):
        previous = self.active.get(record['id'], {})
        self.active[record['id']] = {
            'message': concise(message), 'since': monotonic(),
            'step': record.get('step', 0),
            'skill': concise(skill) if skill is not None else previous.get('skill', ''),
            'spinner': Spinner('dots', style='#b49a80') if Console is not None else None,
        }
        interactive = self.console is not None and self.console.is_terminal and not self.console.is_dumb_terminal
        if interactive:
            if self.live is None:
                self.live = Live(
                    console=self.console, get_renderable=self.render,
                    refresh_per_second=4, transient=True,
                )
                self.live.start()
            else:
                self.live.refresh()
        elif announce:
            step = f"第 {record['step']} 轮 · " if record.get('step') else ''
            self.write(record, '等待', step + message, offset)

    def stop(self, record):
        self.active.pop(record['id'], None)
        if self.live is not None:
            if self.active:
                self.live.refresh()
            else:
                self.live.stop()
                self.live = None

    def event(self, record, kind, data, offset):
        if kind == 'started':
            self.write(record, '任务', data.get('goal', record.get('goal', '')), offset)
        elif kind == 'connecting':
            self.waiting(record, '正在连接后端…', offset)
        elif kind == 'finishing':
            self.waiting(record, '正在保存任务结果…', offset, announce=False)
        elif kind == 'skill':
            self.write(record, 'SKILL', data['name'], offset)
            self.waiting(record, '准备观察环境…', offset, skill=data['name'], announce=False)
        elif kind == 'observing':
            self.waiting(record, '正在观察环境…', offset)
        elif kind == 'observation':
            self.write(record, '观察', f"O{data['step']} · 已获取画面", offset)
            self.waiting(record, '正在整理观察与记忆…', offset, announce=False)
        elif kind == 'prompt':
            self.waiting(record, '正在分析画面…', offset)
        elif kind == 'decision':
            decision = data['decision']
            self.write(record, '判断', decision.get('reason') or decision.get('evidence') or decision['action'], offset)
            self.waiting(record, '正在准备执行…', offset, announce=False)
        elif kind == 'action_input':
            action = data['action']
            if action == 'navigate':
                detail = data.get('instruction', '')
            elif action == 'look':
                detail = {'left': '向左观察', 'right': '向右观察'}.get(data.get('direction'), '')
            elif action == 'orient':
                detail = f"转向 {data['heading_deg']:.0f}°"
            else:
                detail = '沿实际路径退回'
            self.write(record, 'TOOL', f'{action} · {detail}', offset)
            self.waiting(record, f'正在执行 {action}…', offset, announce=False)
        elif kind == 'action_result':
            result = data['result']
            reason = result.get('reason', 'unknown')
            descriptions = {
                'turned': '转向完成', 'oriented': '朝向调整完成',
                'recovered': '退回完成', 'model_stop': '导航模型已停止，待重新观察',
                'agent_interrupt': '新观察触发重规划，已停车待复核',
                'timeout': '导航超时', 'blocked': '前方受阻',
                'path_repeated': '导航绕回旧位置，停车复核',
                'no_progress': '未取得明显进展', 'navigation_error': '导航服务异常',
                'recovery_unavailable': '当前无法安全退回',
            }
            message = descriptions.get(reason, f'执行反馈：{reason}')
            displacement = result.get('motion', {}).get('displacement_m', result.get('displacement_m'))
            if displacement is not None:
                message += f' · 位移 {displacement:.2f} m'
            previous = self.active.get(record['id'])
            if previous:
                message += f" · 耗时 {monotonic() - previous['since']:.1f} s"
            warning = reason in ('timeout', 'blocked', 'no_progress', 'navigation_error', 'recovery_unavailable')
            if result.get('startup_failed'):
                message = '导航服务启动失败，未获得有效推理结果'
                warning = True
            self.write(record, '警告' if warning else '结果', message, offset)
            self.waiting(record, '正在整理执行结果…', offset, announce=False)
        elif kind == 'region_comparison':
            report = data.get('report', {})
            self.write(record, '区域', str(report.get('region_id')) + ' · '
                       + report.get('evidence', ''), offset)
        elif kind == 'region_comparison_error':
            self.write(record, '区域', '对照不可用，继续普通停车决策 · ' + data.get('error', ''), offset)
        elif kind == 'completion_check':
            outcome = '独立核验确认完成' if data.get('confirmed') else '独立核验未确认完成'
            self.write(record, '判断', outcome + ' · ' + (data.get('evidence') or data.get('error', '')), offset)
        elif kind == 'decision_error':
            category = {
                'scene_rule': '场景规则', 'exploration_rule': '探索规则',
                'output_format': '输出格式', 'completion_verification': '完成证据',
            }.get(data.get('category'), '规则')
            errors = data.get('validation_errors') or []
            reason = errors[0].get('message', '') if errors else ''
            self.write(record, '警告',
                       f"模型回答未通过{category}校验 · 第 {data['attempt']} 次"
                       + (f" · {reason}" if reason else ''), offset)
            self.waiting(record, '正在处理模型校验结果…', offset, announce=False)
        elif kind == 'finished':
            status = data.get('status')
            description = {
                'completed': '任务完成', 'failed': '任务失败',
                'cancelled': '任务已取消', 'budget_exhausted': '任务预算耗尽',
            }.get(status, status or '任务结束')
            summary = {
                'Maximum decision steps reached': '已达到最大决策轮数',
                'Task or operation timed out': '任务或操作等待超时',
                'Cancelled by user or service shutdown': '用户取消或服务关闭',
            }.get(data.get('summary'), data.get('summary', ''))
            try:
                label = '失败' if status == 'failed' else '结束'
                self.write(record, label, description + (' · ' + summary if summary else ''), offset)
            finally:
                self.stop(record)


terminal_log = TerminalLog()
