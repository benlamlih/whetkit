"""Self-contained HTML rendering of a ComparisonReport.

No external assets: all CSS is inline, no scripts, no fonts, no CDN. The
file works offline and can be attached to a PR or shared as-is.
"""

import html

from whetkit.report.builder import ComparisonReport, SideView, TaskComparison

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem auto;
       max-width: 70rem; padding: 0 1rem; line-height: 1.5; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.2rem; margin-top: 2.5rem; }
.meta { opacity: .75; font-size: .9rem; }
.cards { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
.card { border: 1px solid rgba(127,127,127,.35); border-radius: .5rem;
        padding: .8rem 1.2rem; min-width: 11rem; }
.card .label { font-size: .8rem; opacity: .75; }
.card .value { font-size: 1.6rem; font-weight: 700; }
.delta-up { color: #1a7f37; } .delta-down { color: #cf222e; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { border: 1px solid rgba(127,127,127,.35); padding: .45rem .6rem;
         text-align: left; vertical-align: top; }
th { background: rgba(127,127,127,.12); }
.scroll { overflow-x: auto; }
.pass { color: #1a7f37; font-weight: 700; } .miss { color: #cf222e; font-weight: 700; }
.improved { background: rgba(26,127,55,.10); } .regressed { background: rgba(207,34,46,.10); }
code { background: rgba(127,127,127,.15); padding: .1rem .3rem; border-radius: .25rem;
       font-size: .85em; }
.tools { font-family: ui-monospace, monospace; font-size: .8rem; }
"""


def _e(value: object) -> str:
    return html.escape(str(value))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _delta_pct(before: float, after: float, higher_is_better: bool = True) -> str:
    delta = after - before
    if abs(delta) < 1e-9:
        return '<span class="meta">±0</span>'
    good = (delta > 0) == higher_is_better
    cls = "delta-up" if good else "delta-down"
    return f'<span class="{cls}">{"+" if delta > 0 else ""}{delta:.0%}</span>'


def _metric_card(label: str, before: float, after: float) -> str:
    return (
        f'<div class="card"><div class="label">{_e(label)}</div>'
        f'<div class="value">{_pct(before)} → {_pct(after)}</div>'
        f"{_delta_pct(before, after)}</div>"
    )


def _side_cell(side: SideView) -> str:
    mark = '<span class="pass">HIT</span>' if side.hit else '<span class="miss">MISS</span>'
    tools = " → ".join(_e(t) for t in side.tools_called) or "(no tool calls)"
    extra = ""
    if side.missing_slots:
        missing = ", ".join(" / ".join(slot) for slot in side.missing_slots)
        extra += f'<br><span class="meta">never called: {_e(missing)}</span>'
    if side.judge_passed is not None:
        verdict = "pass" if side.judge_passed else "fail"
        extra += f'<br><span class="meta">judge: {verdict} — {_e(side.judge_rationale)}</span>'
    return f'{mark}<br><span class="tools">{tools}</span>{extra}'


def _task_rows(tasks: list[TaskComparison]) -> str:
    rows = []
    for task in tasks:
        cls = task.outcome if task.outcome in ("improved", "regressed") else ""
        rows.append(
            f'<tr class="{cls}"><td><strong>{_e(task.task_id)}</strong>'
            f'<br><span class="meta">{_e(task.prompt)}</span></td>'
            f"<td>{_side_cell(task.before)}</td><td>{_side_cell(task.after)}</td>"
            f"<td>{_e(task.outcome)}</td></tr>"
        )
    return "".join(rows)


def _plan_rows(report: ComparisonReport) -> str:
    impacts_by_original = {i.override.original_name: i for i in report.action_impacts}
    rows = []
    for override in report.plan.overrides:
        impact = impacts_by_original.get(override.original_name)
        presented = "(hidden)" if override.hidden else override.presented_name
        helped = ", ".join(impact.improved_tasks) if impact and impact.improved_tasks else "—"
        rows.append(
            f"<tr><td><code>{_e(override.original_name)}</code></td>"
            f"<td><code>{_e(presented)}</code></td>"
            f"<td>{_e(impact.action if impact else '')}</td>"
            f"<td>{_e(override.new_description or '')}</td>"
            f"<td>{_e(override.reason)}</td><td>{_e(helped)}</td></tr>"
        )
    return "".join(rows)


def render_html(report: ComparisonReport) -> str:
    before, after = report.before, report.after
    token_delta = (after.input_tokens + after.output_tokens) - (
        before.input_tokens + before.output_tokens
    )
    latency_delta_s = (after.latency_ms - before.latency_ms) / 1000

    improved = len(report.improved)
    regressed = len(report.regressed)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(report.title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{_e(report.title)}</h1>
<p class="meta">server: <code>{_e(report.server)}</code> · agent model:
<code>{_e(report.model)}</code> · {len(report.tasks)} tasks ·
{improved} improved / {regressed} regressed</p>

<div class="cards">
{_metric_card("Hit-rate", before.hit_rate, after.hit_rate)}
{_metric_card("Tool-selection hit-rate", before.tool_hit_rate, after.tool_hit_rate)}
{_metric_card("Tool precision (avg)", before.avg_precision, after.avg_precision)}
{_metric_card("Tool recall (avg)", before.avg_recall, after.avg_recall)}
<div class="card"><div class="label">Tokens (in+out)</div>
<div class="value">{before.input_tokens + before.output_tokens:,} →
{after.input_tokens + after.output_tokens:,}</div>
<span class="meta">{"+" if token_delta > 0 else ""}{token_delta:,}</span></div>
<div class="card"><div class="label">Total latency</div>
<div class="value">{before.latency_ms / 1000:,.1f}s → {after.latency_ms / 1000:,.1f}s</div>
<span class="meta">{"+" if latency_delta_s > 0 else ""}{latency_delta_s:,.1f}s</span></div>
</div>

<h2>Per-task breakdown</h2>
<div class="scroll">
<table>
<tr><th>Task</th><th>Before (origin tools)</th><th>After (curated overlay)</th><th>Outcome</th></tr>
{_task_rows(report.tasks)}
</table>
</div>

<h2>Curation plan ({len(report.plan.overrides)} actions)</h2>
<p class="meta">{_e(report.plan.notes)}</p>
<div class="scroll">
<table>
<tr><th>Origin tool</th><th>Presented as</th><th>Action</th><th>New description</th>
<th>Why</th><th>Helped tasks</th></tr>
{_plan_rows(report)}
</table>
</div>

<p class="meta">Generated by whetkit. The origin server was never modified —
the “after” column measures the same server behind a reversible overlay.</p>
</body>
</html>
"""
