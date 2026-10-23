"""离线导出日志，使用记录中的时区；旧记录回退到显式选择的配置。"""
import argparse
import json
from pathlib import Path
from navigation_agent.app.agent.settings import DEFAULT_AGENT_CONFIG, load_agent
from navigation_agent.agent.storage.report import write_report
from navigation_agent.agent.storage.memory_visualize import write_memory_visualization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path)
    parser.add_argument('--agent-config', default=DEFAULT_AGENT_CONFIG)
    args = parser.parse_args()
    settings = load_agent(args.agent_config)
    root = args.path or settings.storage.runs_dir
    paths = [root / 'result.json'] if (root / 'result.json').is_file() else sorted(root.rglob('result.json'))
    for path in paths:
        record = json.loads(path.read_text(encoding='utf-8'))
        # 实验评分结果不是 Agent 记录，不能传给决策报告生成器。
        if not record.get('id') or 'goal' not in record:
            continue
        offset = record.get('configuration', {}).get('agent', {}).get('storage', {}).get('utc_offset_minutes', settings.storage.utc_offset_minutes)
        memory = write_memory_visualization(path.parent)
        if memory is not None:
            print(memory)
        print(write_report(path.parent, record, utc_offset_minutes=offset))


if __name__ == '__main__':
    main()
