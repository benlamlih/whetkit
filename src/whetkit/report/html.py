"""Render a ComparisonReport as the designed self-contained HTML report.

Layout and styling follow the Claude-design handoff (whetkit Report.html,
2026-07), with one deliberate substitution: the design's inline expand/
collapse script is replaced by CSS-only <details> accordions so the file
ships with zero scripts — the strongest form of "self-contained, no external
requests" for an artifact people open from disk and attach to PRs.

Honesty rules carried from the CLI: warning strips and noise caveats render
only when there is something to say; multi-run headlines show mean [min–max];
per-task cells show hits across repetitions (✓ 3/3, ⚡ 2/3, ✗ 0/3).
"""

import html

from whetkit import __version__
from whetkit.report.builder import ComparisonReport, SideView, TaskComparison

GREEN = "#5AD8A0"
AMBER = "#E9B24E"
PURPLE = "#C79BE8"
BLUE = "#8FB4E8"
MUTED = "#656C74"
SOFT = "#9AA1A9"

_CSS = """
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: #08090A;
    color: #EAECEE;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  .mono { font-family: ui-monospace, "SF Mono", "Menlo", "Consolas", "Liberation Mono", monospace; }
  a { color: #5AD8A0; text-decoration: none; }
  a:hover { color: #7BE6B8; }
  ::selection { background: #5AD8A0; color: #05100B; }
  .card { border: 1px solid rgba(255,255,255,0.09); border-radius: 16px; background: linear-gradient(180deg,#121519,#0C0E10); }
  details.row > summary { list-style: none; cursor: pointer; }
  details.row > summary::-webkit-details-marker { display: none; }
  details.row > summary:hover { background: rgba(255,255,255,0.02); }
  details.row .chev { transition: transform 0.18s ease; display: inline-block; color: #656C74; text-align: center; }
  details.row[open] .chev { transform: rotate(180deg); }
"""

_TASK_GRID = "grid-template-columns:24px 1fr 148px 78px 78px 58px 18px;"


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float) -> int:
    return round(value * 100)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _meta_cell(label: str, value_html: str) -> str:
    return (
        f'<div><div class="mono" style="font-size:11.5px; color:{MUTED}; '
        f'margin-bottom:6px;">{label}</div>'
        f'<div class="mono" style="font-size:16px;">{value_html}</div></div>'
    )


def _header(report: ComparisonReport) -> str:
    cells = [
        _meta_cell("TASKS", str(len(report.tasks))),
        _meta_cell(
            "RUNS",
            f'{report.runs_per_side} <span style="color:{MUTED};">× per side</span>',
        ),
        _meta_cell("MODEL", _e(report.model) or "—"),
        _meta_cell("DATE", report.generated_at.strftime("%Y-%m-%d")),
    ]
    if report.est_cost_usd is not None:
        cells.append(
            _meta_cell(
                "EST. COST",
                f'≈ ${report.est_cost_usd:.2f} <span style="color:{MUTED};">(est.)</span>',
            )
        )
    cells.append(_meta_cell("RUN ID", report.run_id))
    return f"""
  <header class="card" style="overflow:hidden;">
    <div style="display:flex; align-items:center; justify-content:space-between; padding:20px 26px; border-bottom:1px solid rgba(255,255,255,0.07);">
      <div style="display:flex; align-items:center; gap:11px;">
        <div style="width:20px; height:20px; background:{GREEN}; transform:rotate(45deg); border-radius:4px;"></div>
        <span class="mono" style="font-weight:600; font-size:15px;">whetkit</span>
        <span class="mono" style="font-size:12px; color:{MUTED};">curate report</span>
      </div>
      <span class="mono" style="display:inline-flex; align-items:center; gap:8px; font-size:12px; color:{GREEN}; border:1px solid rgba(90,216,160,0.28); background:rgba(90,216,160,0.07); padding:5px 12px; border-radius:100px;">
        <span style="width:6px;height:6px;border-radius:50%;background:{GREEN};"></span>curate complete
      </span>
    </div>
    <div style="padding:26px;">
      <div class="mono" style="font-size:12px; color:{MUTED}; letter-spacing:0.06em; margin-bottom:8px;">MCP SERVER EVALUATED</div>
      <div class="mono" style="font-size:18px; font-weight:600; letter-spacing:-0.01em; margin-bottom:24px; word-break:break-all;">{_e(_shorten(report.server, 120))}</div>
      <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:20px 24px;">
        {"".join(cells)}
      </div>
    </div>
  </header>"""


def _warnings_strip(report: ComparisonReport) -> str:
    if not report.warnings:
        return ""
    rows = "".join(f"<div>{_e(warning)}</div>" for warning in report.warnings)
    return f"""
  <section class="mono" style="margin-top:16px; border:1px solid rgba(233,178,78,0.22); background:rgba(233,178,78,0.05); border-radius:12px; padding:12px 18px; display:flex; flex-direction:column; gap:8px; font-size:12.5px; color:{AMBER};">
    {rows}
  </section>"""


def _side_headline(label: str, stats: dict, color: str, task_count: int, runs: int) -> str:
    mean, low, high = stats["mean"], stats["low"], stats["high"]
    range_note = f"[{_pct(low)}–{_pct(high)}%]" if low != high else ""
    correct = round(mean * task_count)
    spread_note = (
        "shaded band = run-to-run spread"
        if low != high
        else ("no spread — stable across runs" if runs > 1 else "single run")
    )
    band = ""
    if low != high:
        band = (
            f'<div style="position:absolute; left:{_pct(low)}%; right:{100 - _pct(high)}%; '
            'top:0; bottom:0; background:rgba(233,178,78,0.18);"></div>'
        )
    runs_note = f"(mean of {runs} runs)" if runs > 1 else "(single run)"
    range_html = (
        f'<span class="mono" style="font-size:15px; color:{SOFT};">{range_note}</span>'
        if range_note
        else ""
    )
    return f"""
      <div>
        <div class="mono" style="font-size:13px; color:{SOFT}; margin-bottom:10px;">{label}</div>
        <div style="display:flex; align-items:baseline; gap:8px; color:{color};">
          <span class="mono" style="font-size:60px; font-weight:600; letter-spacing:-0.04em; line-height:0.9;">{_pct(mean)}<span style="font-size:26px;">%</span></span>
          {range_html}
        </div>
        <div class="mono" style="font-size:12px; color:{MUTED}; margin-top:10px;">{correct} / {task_count} tasks correct {runs_note}</div>
        <div style="height:8px; background:rgba(255,255,255,0.06); border-radius:100px; margin-top:12px; overflow:hidden; position:relative;">
          {band}
          <div style="height:100%; width:{_pct(mean)}%; background:{color}; border-radius:100px; position:relative;"></div>
        </div>
        <div class="mono" style="font-size:10.5px; color:#5A6069; margin-top:6px;">{spread_note}</div>
      </div>"""


def _delta_box(label: str, before: str, after: str, delta_note: str) -> str:
    return f"""
      <div style="border:1px solid rgba(255,255,255,0.08); border-radius:11px; padding:16px;">
        <div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:8px;">{label}</div>
        <div class="mono" style="font-size:20px;">{before} <span style="color:{MUTED};">→</span> <span style="color:{GREEN};">{after}</span></div>
        <div class="mono" style="font-size:11.5px; color:{GREEN}; margin-top:6px;">{delta_note}</div>
      </div>"""


def _headline(report: ComparisonReport) -> str:
    task_count = len(report.tasks) or 1
    runs = report.runs_per_side
    delta = _pct(report.after_stats["mean"]) - _pct(report.before_stats["mean"])
    delta_color = GREEN if delta >= 0 else AMBER
    caveat = ""
    if report.noise_caveat:
        caveat = f"""
    <div class="mono" style="margin-top:22px; border:1px solid rgba(233,178,78,0.28); background:rgba(233,178,78,0.06); border-radius:10px; padding:12px 16px; font-size:12.5px; color:{AMBER}; line-height:1.5;">
      {_e(report.noise_caveat)}
    </div>"""

    boxes = []
    if report.tools_before and report.tools_after is not None:
        change = (
            f"−{round(100 * (report.tools_before - report.tools_after) / report.tools_before)}%"
        )
        boxes.append(
            _delta_box("TOOLS EXPOSED", str(report.tools_before), str(report.tools_after), change)
        )
    denominator = max(task_count * runs, 1)
    tokens_before = (report.before.input_tokens + report.before.output_tokens) // denominator
    tokens_after = (report.after.input_tokens + report.after.output_tokens) // denominator
    if tokens_before:
        signed = tokens_before - tokens_after
        change = f"{'−' if signed >= 0 else '+'}{round(100 * abs(signed) / tokens_before)}%"
        boxes.append(_delta_box("TOKENS / TASK", f"{tokens_before:,}", f"{tokens_after:,}", change))

    return f"""
  <section class="card" style="margin-top:16px; padding:34px 30px;">
    <div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:24px; flex-wrap:wrap; gap:8px;">
      <div class="mono" style="font-size:12px; color:{MUTED}; letter-spacing:0.08em;">TOOL-SELECTION ACCURACY</div>
      <div class="mono" style="font-size:12px; color:{MUTED};">Runs: {runs} × {task_count} tasks · mean [min–max]</div>
    </div>
    <div style="display:grid; grid-template-columns:1fr auto 1fr auto; gap:28px; align-items:center;">
      {_side_headline("before curation", report.before_stats, AMBER, task_count, runs)}
      <div style="color:{MUTED}; font-size:30px; padding:0 4px;">→</div>
      {_side_headline("after curation", report.after_stats, GREEN, task_count, runs)}
      <div style="text-align:center; padding-left:8px;">
        <div class="mono" style="font-size:32px; font-weight:600; color:{delta_color}; letter-spacing:-0.03em;">{"+" if delta >= 0 else ""}{delta}</div>
        <div class="mono" style="font-size:12px; color:{MUTED}; margin-top:4px;">pts (mean)</div>
      </div>
    </div>
    {caveat}
    <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-top:22px; padding-top:24px; border-top:1px solid rgba(255,255,255,0.07);">
      {"".join(boxes)}
    </div>
  </section>"""


def _state_cell(side: SideView) -> str:
    state = TaskComparison.side_state(side)
    style = {"pass": (GREEN, "✓"), "miss": (AMBER, "✗"), "flaky": (PURPLE, "⚡")}[state]
    return (
        f'<span style="text-align:center; color:{style[0]};">'
        f"{style[1]} {side.hits}/{side.runs}</span>"
    )


def _judge_cell(side: SideView) -> str:
    if side.judge_passed is None:
        return f'<span style="text-align:center; color:{MUTED};">—</span>'
    color, mark = (GREEN, "✓") if side.judge_passed else (AMBER, "✗")
    return f'<span style="text-align:center; color:{color};">{mark}</span>'


def _task_row(index: int, comparison: TaskComparison, start_open: bool = False) -> str:
    improved_bg = " background:rgba(90,216,160,0.04);" if comparison.outcome == "improved" else ""
    has_spec_gap = comparison.after.spec_gap or comparison.before.spec_gap
    spec_tag = (
        f' <span style="color:{AMBER}; font-size:11px;">⚠ spec-gap</span>' if has_spec_gap else ""
    )
    detail_lines = []
    if has_spec_gap:
        detail_lines.append(
            f'<span style="color:{AMBER};">⚠ correct outcome via unlisted tools — '
            "expected_tools may be incomplete.</span><br>"
        )
    rationale = comparison.after.judge_rationale or comparison.before.judge_rationale
    if rationale:
        detail_lines.append(
            f'<span style="color:{MUTED};">judge ›</span> {_e(_shorten(rationale, 420))}<br>'
        )
    before_calls = " → ".join(comparison.before.tools_called) or "(no calls)"
    after_calls = " → ".join(comparison.after.tools_called) or "(no calls)"
    detail_lines.append(
        f'<span style="color:{MUTED};">before ›</span> {_e(before_calls)}<br>'
        f'<span style="color:{MUTED};">after ›</span> {_e(after_calls)}'
    )
    border_color = "rgba(233,178,78,0.35)" if has_spec_gap else "rgba(90,216,160,0.3)"
    return f"""
      <details class="row"{" open" if start_open else ""} style="border-bottom:1px solid rgba(255,255,255,0.05);{improved_bg}">
        <summary class="mono" style="display:grid; {_TASK_GRID} gap:10px; padding:13px 18px; font-size:12.5px; align-items:center;">
          <span style="color:{MUTED};">{index:02d}</span>
          <span style="color:#EAECEE; font-family:ui-sans-serif,system-ui,sans-serif;">{_e(_shorten(comparison.prompt, 60))}{spec_tag}</span>
          <span style="color:{SOFT};">{_e(_shorten(" / ".join(slot[0] for slot in comparison.expected_slots), 22))}</span>
          {_state_cell(comparison.before)}
          {_state_cell(comparison.after)}
          {_judge_cell(comparison.after)}
          <span class="chev">⌄</span>
        </summary>
        <div style="padding:0 18px 16px 52px;">
          <div class="mono" style="font-size:12px; color:{SOFT}; line-height:1.6; border-left:2px solid {border_color}; padding:2px 0 2px 16px;">
            {"".join(detail_lines)}
          </div>
        </div>
      </details>"""


def _task_table(report: ComparisonReport) -> str:
    first_improved = next((c.task_id for c in report.improved), None)
    rows = "".join(
        _task_row(index, comparison, start_open=comparison.task_id == first_improved)
        for index, comparison in enumerate(report.tasks, 1)
    )
    flaky_stabilized = sum(
        1
        for comparison in report.tasks
        if TaskComparison.side_state(comparison.before) == "flaky"
        and TaskComparison.side_state(comparison.after) == "pass"
    )
    spec_gaps = sum(
        1 for comparison in report.tasks if comparison.after.spec_gap or comparison.before.spec_gap
    )
    summary = (
        f"{flaky_stabilized} flaky tasks stabilized · {len(report.regressed)} regressions · "
        f"{spec_gaps} spec-gap flagged"
    )
    return f"""
  <section style="margin-top:34px;">
    <div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
      <h2 style="font-size:18px; font-weight:600; margin:0;">Per-task breakdown</h2>
      <div class="mono" style="font-size:11.5px; color:{MUTED};">
        <span style="color:{GREEN};">✓ pass</span> · <span style="color:{AMBER};">✗ miss</span> · <span style="color:{PURPLE};">⚡ flaky</span> · click a row for the judge's rationale
      </div>
    </div>
    <div class="card" style="background:none; overflow:hidden;">
      <div class="mono" style="display:grid; {_TASK_GRID} gap:10px; padding:11px 18px; background:#101316; font-size:11px; color:{MUTED}; letter-spacing:0.04em; border-bottom:1px solid rgba(255,255,255,0.07);">
        <div>#</div><div>TASK</div><div>EXPECTED TOOL</div><div style="text-align:center;">BEFORE</div><div style="text-align:center;">AFTER</div><div style="text-align:center;">JUDGE</div><div></div>
      </div>
      {rows}
      <div class="mono" style="display:flex; justify-content:space-between; padding:12px 18px; font-size:12px; background:#101316; color:{SOFT}; flex-wrap:wrap; gap:8px;">
        <span>{summary}</span>
        <span><span style="color:{AMBER};">{_pct(report.before_stats["mean"])}% mean</span> <span style="color:{MUTED};">→</span> <span style="color:{GREEN};">{_pct(report.after_stats["mean"])}% mean</span></span>
      </div>
    </div>
  </section>"""


_ACTION_STYLES = {
    "prune": ("PRUNED", AMBER, "rgba(233,178,78,0.3)"),
    "rename": ("RENAMED", BLUE, "rgba(143,180,232,0.3)"),
    "rename + rewrite": ("RENAMED", BLUE, "rgba(143,180,232,0.3)"),
    "rewrite": ("REWRITTEN", PURPLE, "rgba(199,155,232,0.3)"),
}


def _plan_section(report: ComparisonReport) -> str:
    groups: dict[str, list] = {}
    styles: dict[str, tuple[str, str]] = {}
    for impact in report.action_impacts:
        label, color, border = _ACTION_STYLES.get(impact.action, ("CHANGED", SOFT, MUTED))
        groups.setdefault(label, []).append(impact)
        styles[label] = (color, border)
    if not groups:
        return ""
    notes = (
        f'<p class="mono" style="font-size:12px; color:{MUTED}; margin:0 0 16px;">'
        f"optimizer › {_e(_shorten(report.plan.notes, 320))}</p>"
        if report.plan.notes
        else ""
    )
    cards = []
    for label, impacts in groups.items():
        color, border = styles[label]
        lines = []
        improved: set[str] = set()
        for impact in impacts:
            override = impact.override
            reason = (
                f' <span style="color:{MUTED}; font-size:11px;">— {_e(_shorten(override.reason, 90))}</span>'
                if override.reason
                else ""
            )
            if override.new_name:
                lines.append(
                    f'<div><span style="color:{SOFT};">{_e(override.original_name)}</span> '
                    f'<span style="color:{MUTED};">→</span> '
                    f'<span style="color:{GREEN};">{_e(override.new_name)}</span>{reason}</div>'
                )
            else:
                lines.append(
                    f'<div style="color:{MUTED};">{_e(override.original_name)}{reason}</div>'
                )
            improved.update(impact.improved_tasks)
        improved_note = (
            f'<div style="color:{MUTED}; margin-top:8px;">improved on {len(improved)} task(s)</div>'
            if improved
            else ""
        )
        cards.append(f"""
      <div style="border:1px solid rgba(255,255,255,0.09); border-radius:13px; padding:18px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
          <span class="mono" style="font-size:11px; color:{color}; border:1px solid {border}; padding:3px 9px; border-radius:6px;">{label} · {len(impacts)}</span>
        </div>
        <div class="mono" style="font-size:12px; line-height:2.1;">{"".join(lines)}{improved_note}</div>
      </div>""")
    return f"""
  <section style="margin-top:34px;">
    <h2 style="font-size:18px; font-weight:600; margin:0 0 6px;">What curation changed</h2>
    <p style="font-size:14px; color:{SOFT}; margin:0 0 10px;">Non-destructive overlay — source tools untouched, fully reversible. {len(report.plan.overrides)} change(s) across {report.tools_before or "?"} tools.</p>
    {notes}
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px;">
      {"".join(cards)}
    </div>
  </section>"""


def _trace_side(label: str, color: str, side: SideView) -> str:
    chips = []
    for call in side.calls[:6]:
        chip_color = AMBER if call.is_error else color
        chip_bg = "rgba(233,178,78,0.08)" if call.is_error else "rgba(90,216,160,0.08)"
        chip_border = "rgba(233,178,78,0.22)" if call.is_error else "rgba(90,216,160,0.22)"
        chips.append(
            f'<div style="background:{chip_bg}; border:1px solid {chip_border}; '
            f"border-radius:8px; padding:10px 12px; color:{chip_color}; margin-bottom:8px; "
            f'word-break:break-all;">{_e(call.name)}({_e(call.args)})</div>'
        )
    verdict_color = GREEN if side.hit else AMBER
    verdict = "✓ hit" if side.hit else "✗ miss"
    final = (
        f'<div style="margin-top:10px; color:{SOFT};">final › {_e(_shorten(side.final_text, 180))}</div>'
        if side.final_text
        else ""
    )
    calls_html = "".join(chips) or f'<div style="color:{MUTED};">(no tool calls)</div>'
    return f"""
          <div style="padding:20px;">
            <div class="mono" style="font-size:11px; color:{color}; letter-spacing:0.06em; margin-bottom:16px;">{label}</div>
            <div class="mono" style="font-size:12px; line-height:1.7; color:{SOFT};">
              <div style="margin-bottom:6px;"><span style="color:{MUTED};">calls ›</span></div>
              {calls_html}
              <div style="margin-top:14px; color:{verdict_color};">{verdict}</div>
              {final}
            </div>
          </div>"""


def _trace_section(report: ComparisonReport) -> str:
    showcase = next(iter(report.improved), None) or next(iter(report.regressed), None)
    if showcase is None:
        return ""
    return f"""
  <section style="margin-top:34px;">
    <h2 style="font-size:18px; font-weight:600; margin:0 0 6px;">Reasoning-path trace</h2>
    <p style="font-size:14px; color:{SOFT}; margin:0 0 16px;">Task “{_e(_shorten(showcase.prompt, 80))}” — the last repetition of each side. The tools the agent reached for are the whole story.</p>
    <div class="card" style="background:none; overflow:hidden;">
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0;">
        <div style="border-right:1px solid rgba(255,255,255,0.07);">
          {_trace_side(f"BEFORE · raw tool set ({report.tools_before or '?'} tools)", AMBER, showcase.before)}
        </div>
        <div>
          {_trace_side(f"AFTER · curated tool set ({report.tools_after or '?'} tools)", GREEN, showcase.after)}
        </div>
      </div>
    </div>
  </section>"""


def render_html(report: ComparisonReport) -> str:
    generated = report.generated_at.strftime("%Y-%m-%dT%H:%MZ")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(report.title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div style="max-width:940px; margin:0 auto; padding:48px 28px 64px;">
{_header(report)}
{_warnings_strip(report)}
{_headline(report)}
{_task_table(report)}
{_plan_section(report)}
{_trace_section(report)}
  <footer style="margin-top:44px; padding-top:24px; border-top:1px solid rgba(255,255,255,0.08); display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;">
    <div class="mono" style="font-size:12.5px; color:{MUTED};">generated by <span style="color:{SOFT};">whetkit v{_e(__version__)}</span> · {generated} · self-contained, no external requests</div>
    <a href="https://github.com/benlamlih/whetkit" class="mono" style="font-size:12.5px;">github.com/benlamlih/whetkit ↗</a>
  </footer>
</div>
</body>
</html>
"""
