"""保存任务结果、运动采样与离线报告。"""

from datetime import datetime, timezone, timedelta
import json
import hashlib
import logging
from pathlib import Path

from navigation_agent.agent.config import StorageSettings
from navigation_agent.agent.storage.report import write_report
from navigation_agent.agent.storage.terminal import terminal_log
from navigation_agent.agent.storage.memory_visualize import write_memory_visualization

logger = logging.getLogger('uvicorn.error')


class LogStore:
    """同一任务的写入和历史读取使用同一套路径配置。"""

    def __init__(self, settings: StorageSettings):
        self.settings = settings

    def run_directory(self, record):
        name = record.get('log_directory', record['id'])
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid log directory')
        return self.settings.runs_dir / name


    def log_time(self, value):
        return (
            datetime.fromisoformat(value)
            .astimezone(timezone(timedelta(minutes=self.settings.utc_offset_minutes)))
            .strftime('%Y-%m-%d_%H-%M-%S')
        )


    def finish_log_directory(self, record):
        # 完成目标记为 success，其余结束状态记为 failed，具体原因保留在日志中。
        source = self.run_directory(record).resolve()
        name = self.log_time(record['started_at']) + '__' + self.log_time(record['ended_at'])
        status = 'success' if record['status'] == 'completed' else 'failed'
        destination = (self.settings.runs_dir / (name + '_' + status)).resolve()
        if destination.exists() and destination != source:
            destination = (self.settings.runs_dir / (name + '__' + record['id'] + '_' + status)).resolve()
        if source.parent != self.settings.runs_dir.resolve() or destination.parent != self.settings.runs_dir.resolve():
            raise ValueError('Log directory must remain inside runs')
        if source != destination:
            source.rename(destination)
        record['log_directory'] = destination.name


    def event(self, record, kind, **data):
        payload = {'time': datetime.now(timezone.utc).isoformat(), 'type': kind, **data}
        with (self.run_directory(record) / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + '\n')
        if kind in ('decision_input', 'action_result', 'finished'):
            # 决策和动作后更新只读记忆快照；展示失败不能影响任务执行。
            try:
                write_memory_visualization(self.run_directory(record))
            except Exception:
                logger.exception('Could not export memory HTML for task=%s', record['id'])
        self.save_result(record)
        self.progress(record, kind, **data)

    def progress(self, record, kind, **data):
        # 展示故障不影响任务执行和已经落盘的完整记录。
        try:
            terminal_log.event(record, kind, data, self.settings.utc_offset_minutes)
        except Exception:
            logger.warning('终端摘要显示失败，完整记录请查看 events.jsonl')



    def append_telemetry(self, record, sample):
        # 高频运动采样单独保存，不塞进模型历史和页面事件。
        with (self.run_directory(record) / 'navigation.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'step': record['step'], **sample}, ensure_ascii=False) + '\n')


    def save_result(self, record):
        destination = self.run_directory(record) / 'result.json'

        # 先写临时文件，再替换，避免读到半份结果
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        temporary.replace(destination)

        # 报告生成失败不能中断运动或 JSON 日志写入。
        try:
            write_report(self.run_directory(record), record, utc_offset_minutes=self.settings.utc_offset_minutes)
        except Exception:
            logger.exception('Could not update HTML log for task=%s', record['id'])


    def save_execution(self, record, execution):
        destination = self.run_directory(record) / 'execution.json'
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(
            json.dumps(execution.snapshot(), ensure_ascii=False, indent=2), encoding='utf-8'
        )
        temporary.replace(destination)
        record['exploration_summary'] = {
            'observation_points': execution.current['place_count'],
            'current_place_id': execution.current['id'],
            'same_view_previous_observations': execution.current['same_view_previous_observations'],
            'candidates': len(execution.candidates),
            'avoided_approaches': sum(c['status'] == 'avoid_from_here' for c in execution.candidates),
        }

    def save_image(self, record, name, jpeg):
        (self.run_directory(record) / 'images' / name).write_bytes(jpeg)

    def load_images(self, record, manifest):
        return [
            {**item, 'jpeg': (self.run_directory(record) / item['image']).read_bytes()}
            for item in manifest
        ]

    def save_prompt(self, record, step, prompt_text):
        digest = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
        prompt_dir = self.run_directory(record) / 'prompts'
        prompt_dir.mkdir(exist_ok=True)
        prompt_file = prompt_dir / (digest + '.md')
        if not prompt_file.exists():
            prompt_file.write_text(prompt_text, encoding='utf-8')
        self.event(record, 'prompt', step=step, sha256=digest, file='prompts/' + prompt_file.name)
