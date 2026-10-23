"""从固定评测输出生成可审阅的对照，失败变快不计入完成时间收益。"""

import argparse
from collections import Counter
from datetime import datetime
from html import escape
import json
import hashlib
import sqlite3
from pathlib import Path
from statistics import median


def timing(record):
    events = [json.loads(line) for line in (Path(record['directory']) / 'events.jsonl').read_text().splitlines()]
    totals = Counter()
    prompt_started = None
    monitor_started = None
    action_started = {}
    selected_kinds = {}
    reasons = Counter()
    for event in events:
        stamp = datetime.fromisoformat(event['time']).timestamp()
        kind = event['type']
        if kind == 'prompt' and prompt_started is None:
            prompt_started = stamp
        if kind in ('decision', 'finished') and prompt_started is not None:
            totals['model_wait_s'] += stamp - prompt_started
            prompt_started = None
        if kind == 'action_input':
            action_started[event['step']] = (stamp, event['action'])
        if kind in ('monitor_report', 'monitor_error', 'action_result', 'finished') and monitor_started is not None:
            totals['monitor_wait_s'] += stamp - monitor_started
            monitor_started = None
        if kind == 'action_result' and event['step'] in action_started:
            start, action = action_started.pop(event['step'])
            duration = stamp - start
            totals[action + '_s'] += duration
            reason = event['result']['reason']
            if action == 'navigate':
                reasons[reason] += 1
                purpose = 'target' if selected_kinds.get(event['step']) == 'target' else 'exploration'
                totals[purpose + '_navigation_s'] += duration
                totals['navigation_' + reason + '_s'] += duration
                if reason in ('blocked', 'no_progress', 'timeout', 'navigation_error'):
                    totals['failed_navigation_s'] += duration
            if event.get('result', {}).get('motion'):
                totals[action + '_turn_deg'] += event['result']['motion'].get('absolute_turn_deg', 0)
        if kind == 'stopped_observation_latency':
            totals['stop_and_observe_s'] += event.get('seconds', 0)
        if kind == 'decision':
            decision = event['decision']
            candidates = decision.get('candidates', [])
            selected = next((item for item in candidates if item['id'] == decision.get('candidate_id')), None)
            if selected is None and len(candidates) == 1:
                selected = candidates[0]
            selected_kinds[event['step']] = (selected or {}).get('kind')
            if (decision.get('target_review') or {}).get('status') == 'temporarily_lost':
                totals['target_lost_reports'] += 1
            if (decision.get('previous_action_review') or {}).get('view_change') == 'new_area':
                totals['new_area_reports'] += 1
        if kind == 'navigation_interrupt_requested' and event.get('accepted'):
            totals['accepted_interrupts'] += 1
        if kind == 'monitor_error':
            totals['monitor_errors'] += 1
        if kind == 'monitor_request':
            monitor_started = stamp
            totals['monitor_calls'] += 1
        if kind == 'monitor_report':
            totals['monitor_model_s'] += event.get('model_seconds', 0)
            totals['monitor_reports'] += 1
        if kind == 'completion_check':
            totals['completion_check_s'] += event.get('seconds', 0)
            totals['completion_checks'] += 1
            totals['completion_rejections'] += not event.get('confirmed', False)
        if kind == 'decision_error':
            totals['decision_errors'] += 1
    totals['turn_s'] = totals['look_s'] + totals['orient_s']
    return dict(totals), dict(reasons)


def load(root, name):
    rows = json.loads((root / name / 'results.json').read_text())
    manifest = json.loads((root / name / 'manifest.json').read_text())
    negatives = {case['id'] for case in manifest['cases'] if case.get('target') is None}
    for row in rows:
        row['negative_case'] = row['case'] in negatives
        row['timing'], row['navigation_reasons'] = timing(row)
        with sqlite3.connect('file:' + str(Path(row['directory']) / 'memory.sqlite3') + '?mode=ro', uri=True) as db:
            tables = {item[0] for item in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            row['topology'] = {table: db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                               for table in ('regions', 'region_connections', 'exploration_routes') if table in tables}
    positives = [row for row in rows if not row['negative_case']]
    negatives = [row for row in rows if row['negative_case']]
    expected = len(manifest['cases']) * manifest['repeats']
    hashes = json.loads((root / name / 'source-hashes.json').read_text())
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    configuration = manifest['configuration']
    configuration.get('storage', {}).pop('runs_dir', None)
    monitor_enabled = configuration.get('monitor', {}).pop('enabled', None)
    configuration.get('prompt', {}).pop('path', None)
    conditions = {key: manifest.get(key) for key in ('cases', 'max_steps', 'max_seconds', 'navigation_seconds', 'scene', 'thresholds')}
    conditions['monitor_enabled'] = monitor_enabled
    return dict(source_digest=digest, comparable_configuration=configuration, backend=manifest.get('backend'),
                evaluation_conditions=conditions,
                name=name, complete=(len(rows) == expected and all(row.get('valid', True)
                                        and row['status'] != 'cancelled' for row in rows)),
                runs=len(rows), expected=expected,
                positive_success=sum(row['success'] for row in positives), positive_runs=len(positives),
                false_finishes=sum(row['false_finish'] for row in rows),
                negative_false_finishes=sum(row['false_finish'] for row in negatives),
                negative_runs=len(negatives), rows=rows)


def comparison(baseline, new):
    before = {(row['case'], row['repeat']): row for row in baseline['rows']}
    matched = [(before[(row['case'], row['repeat'])], row) for row in new['rows']
               if (row['case'], row['repeat']) in before and not row.get('negative_case', row['case'] == 'absent')
               and row['success'] and before[(row['case'], row['repeat'])]['success']]
    result = dict(common_success_count=len(matched), new_false_finish_pairs=[])
    for old, current in [('elapsed_s', 'elapsed_s'), ('turn_s', 'turn_s')]:
        def value(row):
            return row[old] if old == 'elapsed_s' else row['timing'][old]
        if matched:
            a, b = median(value(a) for a, _ in matched), median(value(b) for _, b in matched)
            result[current] = dict(baseline_median=a, new_median=b,
                                   reduction=1 - b / a if a else None)
    result['new_false_finish_pairs'] = [
        dict(case=row['case'], repeat=row['repeat']) for row in new['rows']
        if row['false_finish'] and not before.get((row['case'], row['repeat']), {}).get('false_finish', False)
    ]
    result['same_cases_and_complete'] = baseline['complete'] and new['complete'] and {
        (row['case'], row['repeat']) for row in baseline['rows']
    } == {(row['case'], row['repeat']) for row in new['rows']}
    result['identical_evaluation_conditions'] = baseline['evaluation_conditions'] == new['evaluation_conditions']
    result['identical_runtime_configuration'] = (baseline['comparable_configuration'] == new['comparable_configuration']
                                                  and baseline['backend'] == new['backend'])
    result['success_rate_gate'] = (new['positive_success'] / max(1, new['positive_runs'])
                                   >= baseline['positive_success'] / max(1, baseline['positive_runs']))
    result['no_new_false_finish_gate'] = not result['new_false_finish_pairs']
    return result


def render(root, groups, comparisons):
    names = {'baseline': '旧版', 'adaptive-monitor': '新版 · 行进观察开启',
             'adaptive-adopt': '新版 · 行进观察关闭', 'adaptive-corrected-monitor': '修正版 · 行进观察开启',
             'adaptive-corrected': '修正版 · 行进观察关闭',
             'monitor_vs_off': '同代码行进观察开关对照'}
    name = lambda value: names.get(value, value)
    complete = all(group['complete'] for group in groups)
    default_result = comparisons[groups[-1]['name']]
    gates = (default_result.get('success_rate_gate') and default_result.get('no_new_false_finish_gate')
             and default_result.get('same_cases_and_complete')
             and default_result.get('identical_evaluation_conditions')
             and default_result.get('identical_runtime_configuration'))
    verdict = ('对照仍在进行，不能据此采用' if not complete else
               '可比性、成功率或新增误报门槛未通过，不建议采用' if not gates else
               '成功率与误报粗筛通过；仍须核查运动保护、覆盖和最终画面')
    verdict = name(groups[-1]['name']) + '：' + verdict
    esc = lambda value: escape(str(value), quote=True)
    summary = ''.join(
        '<tr><td>' + esc(name(group['name'])) + '</td><td>'
        + f"{group['runs']}/{group['expected']}</td><td>{group['positive_success']}/{group['positive_runs']}"
        + f"</td><td>{group['false_finishes']}</td><td>{group['negative_false_finishes']}/{group['negative_runs']}</td></tr>"
        for group in groups)
    sections = []
    for group in groups:
        rows = []
        evidence = []
        for row in group['rows']:
            link = Path(row['directory']).relative_to(root).as_posix() + '/index.html'
            outcome = '误报完成' if row['false_finish'] else (
                '成功' if row['success'] and not row.get('negative_case', row['case'] == 'absent') else {'failed': '失败', 'budget_exhausted': '预算耗尽', 'cancelled': '取消'}.get(row['status'], row['status']))
            times = row['timing']
            picture = f"{group['name']}/{row['case']}-{row['repeat']}-final.png"
            evidence.append(f"<figure><figcaption>{esc(row['case'])} · {row['repeat'] + 1} — {esc(outcome)}"
                            f"<br>目标像素 {row['oracle']['visible_pixels']}；中心距离 {esc(row['oracle']['distance_m'])}"
                            f"<br>{esc(row['summary'])}</figcaption><a href='{esc(link)}'>"
                            f"<img loading='lazy' src='{esc(picture)}' alt='最终画面'></a></figure>")
            rows.append(
                f"<tr><td><a href='{esc(link)}'>{esc(row['case'])} · {row['repeat'] + 1}</a></td>"
                f"<td>{esc(outcome)}</td><td>{row['elapsed_s']:.1f}</td>"
                f"<td>{times.get('model_wait_s', 0):.1f}</td><td>{times.get('turn_s', 0):.1f}</td>"
                f"<td>{times.get('navigate_s', 0):.1f}</td><td>{esc(row['navigation_reasons'])}</td></tr>")
        aggregate = Counter()
        for record in group['rows']:
            aggregate.update(record['timing'])
        details = (f"<p>后台观察：{int(aggregate['monitor_calls'])} 次请求，"
                   f"{int(aggregate['monitor_reports'])} 份报告，{int(aggregate['monitor_errors'])} 次错误/超时，"
                   f"{int(aggregate['accepted_interrupts'])} 次中断请求；等待 {aggregate['monitor_wait_s']:.1f} 秒（与运动重叠）。"
                   f"前台模型等待 {aggregate['model_wait_s']:.1f} 秒；其中独立完成核验 {aggregate['completion_check_s']:.1f} 秒。"
                   f"探索导航 {aggregate['exploration_navigation_s']:.1f} 秒，目标接近 {aggregate['target_navigation_s']:.1f} 秒。"
                   f"按停止原因统计的导航失败/超时段 {aggregate['failed_navigation_s']:.1f} 秒，"
                   f"它与前两类导航时间重叠，不能相加，也不等同于全部无效探索时间。"
                   f"机器人总转角（含导航）{sum(row['turn_deg'] for row in group['rows']):.0f}°，"
                   f"总路程 {sum(row['path_length_m'] for row in group['rows']):.1f} 米；减少 look 不代表底层不再转圈。"
                   f"模型自报目标暂时丢失 {int(aggregate['target_lost_reports'])} 次，须结合图片核查。"
                   f"记录的跨区域连接 {sum(row['topology'].get('region_connections', 0) for row in group['rows'])} 条。</p>")
        sections.append(f"<h2>{esc(name(group['name']))}</h2>" + details +
                        "<table><thead><tr><th>案例</th><th>结果</th>"
                        "<th>总耗时 s</th><th>模型等待 s</th><th>观察/转回 s</th><th>导航 s</th><th>停止原因</th>"
                        "</tr></thead><tbody>" + ''.join(rows) + '</tbody></table>'
                        + '<details><summary>查看最终画面与完成依据</summary><div class=gallery>'
                        + ''.join(evidence) + '</div></details>')
    paired_rows = []
    for key, values in comparisons.items():
        elapsed = values.get('elapsed_s') or {}
        turns = values.get('turn_s') or {}
        def display(metric):
            if not metric:
                return '暂无共同成功案例'
            reduction = metric['reduction']
            change = f"{reduction * 100:.1f}%" if reduction is not None else '不适用'
            return f"{metric['baseline_median']:.1f} → {metric['new_median']:.1f} 秒；减少 {change}"
        pairs = values['new_false_finish_pairs']
        new_errors = '、'.join(f"{item['case']} #{item['repeat'] + 1}" for item in pairs) or '无'
        paired_rows.append(
            f"<tr><td>{esc(name(key))}</td><td>{values['common_success_count']}</td>"
            f"<td>{esc(display(elapsed))}</td><td>{esc(display(turns))}</td><td>{esc(new_errors)}</td></tr>")
    paired = ("<table><tr><th>对照</th><th>共同成功次数</th><th>总耗时中位数</th>"
              "<th>观察/转回中位数</th><th>新增误报案例</th></tr>" + ''.join(paired_rows) + '</table>'
              + '<details><summary>查看口径与版本一致性</summary><pre>'
              + esc(json.dumps(comparisons, ensure_ascii=False, indent=2)) + '</pre></details>')
    return ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Agent 探索效果对照</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;'
            'background:#f4f1e8;color:#383630;font:16px/1.65 system-ui}a{color:#785b43}h1,h2{font-family:serif}'
            'table{width:100%;border-collapse:collapse;margin:20px 0}th,td{border-bottom:1px solid #d8d0c2;'
            'padding:9px;text-align:left}th{background:#e9e2d6}pre{white-space:pre-wrap;font-size:14px}'
            '.note{padding:16px;background:#e7e6db}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}figure{margin:8px 0}img{width:100%}figcaption{font-size:14px}</style><h1>Agent 探索效果对照</h1>'
            '<p class="note">固定 ProcTHOR val_2 起点、目标与预算。按最终画面可见性和到达距离粗筛；'
            '不能仅凭 Agent finish 判成功。失败变快不计收益。不存在目标只统计误报，不能计入正目标成功率。'
            '遮挡、受阻案例还须核对实际触发情况。小样本不证明普遍改善。</p>'
            '<table><tr><th>版本</th><th>已记录/计划</th><th>正目标成功</th><th>全部误报</th><th>不存在目标误报</th></tr>'
            + summary + '</table><p class=note>' + esc(verdict) + '</p><h2>共同成功案例</h2><p>模型等待由 prompt 到 decision/finished 事件估算，'
            '含格式纠错；后台模型耗时与导航重叠，不应直接加到总耗时。转向为 look + orient 执行时间。</p>'
            + paired + ''.join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/adaptive-evaluation'))
    parser.add_argument('--baseline', default='baseline')
    parser.add_argument('--variant', action='append', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = load(root, args.baseline)
    groups = [baseline, *(load(root, name) for name in args.variant)]
    comparisons = {group['name']: comparison(baseline, group) for group in groups[1:]}
    if len(groups) >= 3:
        comparisons['monitor_vs_off'] = {
            'reference': groups[1]['name'], 'variant': groups[2]['name'],
            'identical_source': groups[1]['source_digest'] == groups[2]['source_digest'],
            'identical_other_configuration': (groups[1]['comparable_configuration'] == groups[2]['comparable_configuration']
                                              and groups[1]['backend'] == groups[2]['backend']),
            **comparison(groups[1], groups[2]),
        }
    output = dict(groups=groups, comparisons=comparisons)
    (root / 'comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    (root / 'comparison.html').write_text(render(root, groups, comparisons))
    print(json.dumps(dict(groups=[{k: v for k, v in group.items() if k not in ('rows', 'comparable_configuration', 'backend')} for group in groups],
                          comparisons=comparisons), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
