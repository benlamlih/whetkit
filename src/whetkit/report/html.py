"""Self-contained HTML rendering of a ComparisonReport.

Implements the whetkit report design (dark, mono, before→after headline).
No external assets and no scripts: all CSS is inline and the trace panels
use CSS-only <details> accordions, so the file works offline anywhere.
"""

import html
from statistics import median

from whetkit import __version__
from whetkit.report.builder import ComparisonReport, SideView

GREEN = "#5AD8A0"
AMBER = "#E9B24E"
BLUE = "#8FB4E8"
PURPLE = "#C79BE8"
MUTED = "#656C74"
SUBTLE = "#9AA1A9"
FAINT = "#5A6069"

_CSS = f"""
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  background: #08090A;
  color: #EAECEE;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}}
.mono {{ font-family: ui-monospace, "SF Mono", "Menlo", "Consolas", "Liberation Mono", monospace; }}
a {{ color: {GREEN}; text-decoration: none; }}
a:hover {{ color: #7BE6B8; }}
::selection {{ background: {GREEN}; color: #05100B; }}
.panel {{ border: 1px solid rgba(255,255,255,0.09); border-radius: 16px;
         background: linear-gradient(180deg,#121519,#0C0E10); }}
.statcard {{ border: 1px solid rgba(255,255,255,0.08); border-radius: 11px; padding: 16px; }}
.bar {{ height: 8px; background: rgba(255,255,255,0.06); border-radius: 100px;
       margin-top: 12px; overflow: hidden; }}
.bar > div {{ height: 100%; border-radius: 100px; }}
.grid-row {{ display: grid; grid-template-columns: 26px 1fr 190px 60px 60px; gap: 12px;
            padding: 12px 18px; font-size: 12.5px;
            border-bottom: 1px solid rgba(255,255,255,0.05); }}
details > summary {{ list-style: none; cursor: pointer; }}
details > summary::-webkit-details-marker {{ display: none; }}
details .chev {{ display: inline-block; transition: transform .2s ease; }}
details[open] .chev {{ transform: rotate(180deg); }}
.callbox {{ border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; word-break: break-all; }}
.call-ok {{ background: rgba(90,216,160,0.08); border: 1px solid rgba(90,216,160,0.22);
           color: {GREEN}; }}
.call-bad {{ background: rgba(233,178,78,0.08); border: 1px solid rgba(233,178,78,0.22);
            color: {AMBER}; }}
.badge {{ font-size: 11px; padding: 3px 9px; border-radius: 6px; }}
@media (max-width: 720px) {{
  .cols2 {{ grid-template-columns: 1fr !important; }}
  .headline-grid {{ grid-template-columns: 1fr !important; }}
}}
"""


def _e(value: object) -> str:
    return html.escape(str(value))


def _pct_num(value: float) -> int:
    return round(value * 100)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _delta_chip(before: float, after: float, lower_is_better: bool = False) -> str:
    if before <= 0:
        return f'<span style="color:{MUTED};">—</span>'
    change = (after - before) / before
    good = (change < 0) == lower_is_better if change != 0 else True
    color = GREEN if good else AMBER
    return f'<span style="color:{color};">{change:+.0%}</span>'


def _header(report: ComparisonReport) -> str:
    date = report.generated_at.strftime("%Y-%m-%d")
    return f"""
  <header class="panel" style="overflow:hidden;">
    <div style="display:flex; align-items:center; justify-content:space-between; padding:20px 26px; border-bottom:1px solid rgba(255,255,255,0.07); flex-wrap:wrap; gap:10px;">
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
      <div class="mono" style="font-size:22px; font-weight:600; letter-spacing:-0.01em; margin-bottom:22px; word-break:break-all;">{_e(report.server)}</div>
      <div class="cols2" style="display:grid; grid-template-columns:repeat(4,1fr); gap:20px;">
        <div><div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:6px;">TASKS</div><div class="mono" style="font-size:16px;">{len(report.tasks)}</div></div>
        <div><div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:6px;">MODEL</div><div class="mono" style="font-size:16px;">{_e(report.model or "—")}</div></div>
        <div><div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:6px;">DATE</div><div class="mono" style="font-size:16px;">{date}</div></div>
        <div><div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:6px;">RUN ID</div><div class="mono" style="font-size:16px;">{report.run_id}</div></div>
      </div>
    </div>
  </header>"""


def _headline(report: ComparisonReport) -> str:
    before_pct = _pct_num(report.before.hit_rate)
    after_pct = _pct_num(report.after.hit_rate)
    total = len(report.tasks)
    before_hits = sum(t.before.hit for t in report.tasks)
    after_hits = sum(t.after.hit for t in report.tasks)
    delta = after_pct - before_pct
    delta_color = GREEN if delta >= 0 else AMBER

    def tokens_per_task(side) -> int:
        return round((side.input_tokens + side.output_tokens) / total) if total else 0

    tok_before, tok_after = tokens_per_task(report.before), tokens_per_task(report.after)
    p50_before = median([t.before.latency_ms for t in report.tasks]) / 1000 if total else 0
    p50_after = median([t.after.latency_ms for t in report.tasks]) / 1000 if total else 0

    if report.tools_before is not None and report.tools_after is not None:
        tools_card = f"""
      <div class="statcard">
        <div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:8px;">TOOLS EXPOSED</div>
        <div class="mono" style="font-size:20px;">{report.tools_before} <span style="color:{MUTED};">→</span> <span style="color:{GREEN};">{report.tools_after}</span></div>
        <div class="mono" style="font-size:11.5px; margin-top:6px;">{_delta_chip(report.tools_before, report.tools_after, lower_is_better=True)}</div>
      </div>"""
    else:
        tools_card = f"""
      <div class="statcard">
        <div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:8px;">TOOLS EXPOSED</div>
        <div class="mono" style="font-size:20px; color:{MUTED};">—</div>
        <div class="mono" style="font-size:11.5px; color:{MUTED}; margin-top:6px;">n/a</div>
      </div>"""

    return f"""
  <section class="panel" style="margin-top:20px; padding:34px 30px;">
    <div class="mono" style="font-size:12px; color:{MUTED}; letter-spacing:0.08em; margin-bottom:24px;">TOOL-SELECTION ACCURACY</div>
    <div class="headline-grid" style="display:grid; grid-template-columns:1fr auto 1fr auto; gap:28px; align-items:center;">
      <div>
        <div class="mono" style="font-size:13px; color:{SUBTLE}; margin-bottom:10px;">before curation</div>
        <div style="display:flex; align-items:baseline; gap:4px; color:{AMBER};"><span class="mono" style="font-size:64px; font-weight:600; letter-spacing:-0.04em; line-height:0.9;">{before_pct}</span><span class="mono" style="font-size:26px;">%</span></div>
        <div class="mono" style="font-size:12px; color:{MUTED}; margin-top:10px;">{before_hits} / {total} tasks correct</div>
        <div class="bar"><div style="width:{before_pct}%; background:{AMBER};"></div></div>
      </div>
      <div style="color:{MUTED}; font-size:30px; padding:0 4px;">→</div>
      <div>
        <div class="mono" style="font-size:13px; color:{SUBTLE}; margin-bottom:10px;">after curation</div>
        <div style="display:flex; align-items:baseline; gap:4px; color:{GREEN};"><span class="mono" style="font-size:64px; font-weight:600; letter-spacing:-0.04em; line-height:0.9;">{after_pct}</span><span class="mono" style="font-size:26px;">%</span></div>
        <div class="mono" style="font-size:12px; color:{MUTED}; margin-top:10px;">{after_hits} / {total} tasks correct</div>
        <div class="bar"><div style="width:{after_pct}%; background:{GREEN};"></div></div>
      </div>
      <div style="text-align:center; padding-left:8px;">
        <div class="mono" style="font-size:34px; font-weight:600; color:{delta_color}; letter-spacing:-0.03em;">{delta:+d}</div>
        <div class="mono" style="font-size:12px; color:{MUTED}; margin-top:4px;">points</div>
      </div>
    </div>
    <div class="cols2" style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:30px; padding-top:26px; border-top:1px solid rgba(255,255,255,0.07);">
      {tools_card}
      <div class="statcard">
        <div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:8px;">TOKENS / TASK</div>
        <div class="mono" style="font-size:20px;">{tok_before:,} <span style="color:{MUTED};">→</span> <span style="color:{GREEN};">{tok_after:,}</span></div>
        <div class="mono" style="font-size:11.5px; margin-top:6px;">{_delta_chip(tok_before, tok_after, lower_is_better=True)}</div>
      </div>
      <div class="statcard">
        <div class="mono" style="font-size:11.5px; color:{MUTED}; margin-bottom:8px;">LATENCY p50</div>
        <div class="mono" style="font-size:20px;">{p50_before:.1f}s <span style="color:{MUTED};">→</span> <span style="color:{GREEN};">{p50_after:.1f}s</span></div>
        <div class="mono" style="font-size:11.5px; margin-top:6px;">{_delta_chip(p50_before, p50_after, lower_is_better=True)}</div>
      </div>
    </div>
  </section>"""


def _mark(hit: bool) -> tuple[str, str]:
    return ("✓", GREEN) if hit else ("✗", AMBER)


def _task_table(report: ComparisonReport) -> str:
    rows = []
    for i, task in enumerate(report.tasks):
        before_mark, before_color = _mark(task.before.hit)
        after_mark, after_color = _mark(task.after.hit)
        if task.outcome == "improved":
            row_bg = "rgba(90,216,160,0.04)"
        elif task.outcome == "regressed":
            row_bg = "rgba(233,178,78,0.05)"
        else:
            row_bg = "rgba(255,255,255,0.012)" if i % 2 else "transparent"
        expected = " · ".join(" / ".join(slot) for slot in task.expected_slots)
        rows.append(f"""
      <div class="grid-row mono" style="background:{row_bg};">
        <div style="color:{MUTED};">{i + 1:02d}</div>
        <div style="color:#EAECEE; font-family:ui-sans-serif,system-ui,sans-serif;">{_e(_shorten(task.prompt, 90))}</div>
        <div style="color:{SUBTLE}; word-break:break-all;">{_e(expected)}</div>
        <div style="text-align:center; color:{before_color};">{before_mark}</div>
        <div style="text-align:center; color:{after_color};">{after_mark}</div>
      </div>""")

    improved = len(report.improved)
    still_failing = sum(1 for t in report.tasks if not t.after.hit)
    before_hits = sum(t.before.hit for t in report.tasks)
    after_hits = sum(t.after.hit for t in report.tasks)
    return f"""
  <section style="margin-top:34px;">
    <div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:16px; flex-wrap:wrap; gap:8px;">
      <h2 style="font-size:18px; font-weight:600; margin:0;">Per-task breakdown</h2>
      <div class="mono" style="font-size:12px; color:{MUTED};">✓ hit · ✗ miss</div>
    </div>
    <div style="border:1px solid rgba(255,255,255,0.09); border-radius:14px; overflow:hidden;">
      <div class="grid-row mono" style="padding:11px 18px; background:#101316; font-size:11px; color:{MUTED}; letter-spacing:0.04em; border-bottom:1px solid rgba(255,255,255,0.07);">
        <div>#</div><div>TASK</div><div>EXPECTED TOOL</div><div style="text-align:center;">BEFORE</div><div style="text-align:center;">AFTER</div>
      </div>
      {"".join(rows)}
      <div class="mono" style="display:flex; justify-content:space-between; padding:12px 18px; font-size:12px; background:#101316; color:{SUBTLE}; flex-wrap:wrap; gap:8px;">
        <span>improved on {improved} task{"s" if improved != 1 else ""} · {still_failing} still failing</span>
        <span><span style="color:{AMBER};">{before_hits} ✓</span> <span style="color:{MUTED};">→</span> <span style="color:{GREEN};">{after_hits} ✓</span></span>
      </div>
    </div>
  </section>"""


def _names_line(names: list[str], limit: int = 8) -> str:
    shown = " · ".join(_e(n) for n in names[:limit])
    more = len(names) - limit
    if more > 0:
        shown += f' · <span style="color:{FAINT};">+ {more} more</span>'
    return shown


def _curation_section(report: ComparisonReport) -> str:
    pruned = [o for o in report.plan.overrides if o.hidden]
    renamed = [o for o in report.plan.overrides if not o.hidden and o.new_name]
    rewritten = [
        o for o in report.plan.overrides if not o.hidden and o.new_description and not o.new_name
    ]

    cards = []
    if pruned:
        cards.append(f"""
      <div style="border:1px solid rgba(255,255,255,0.09); border-radius:13px; padding:18px; grid-column:1 / -1;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px; flex-wrap:wrap;">
          <span class="mono badge" style="color:{AMBER}; border:1px solid rgba(233,178,78,0.3);">PRUNED · {len(pruned)}</span>
          <span style="font-size:13px; color:{SUBTLE};">Hidden from the exposed set; still present on the origin server.</span>
        </div>
        <div class="mono" style="font-size:12px; color:{MUTED}; line-height:1.9;">{_names_line([o.original_name for o in pruned])}</div>
      </div>""")
    if renamed:
        rows = "".join(
            f'<div><span style="color:{SUBTLE};">{_e(o.original_name)}</span> '
            f'<span style="color:{MUTED};">→</span> '
            f'<span style="color:{GREEN};">{_e(o.new_name)}</span></div>'
            for o in renamed[:6]
        )
        if len(renamed) > 6:
            rows += (
                f'<div class="mono" style="color:{FAINT}; font-size:11.5px;">'
                f"+ {len(renamed) - 6} more</div>"
            )
        cards.append(f"""
      <div style="border:1px solid rgba(255,255,255,0.09); border-radius:13px; padding:18px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
          <span class="mono badge" style="color:{BLUE}; border:1px solid rgba(143,180,232,0.3);">RENAMED · {len(renamed)}</span>
        </div>
        <div class="mono" style="font-size:12px; line-height:2.1; word-break:break-all;">{rows}</div>
      </div>""")
    if rewritten:
        rows = "".join(
            f'<div><span style="color:{GREEN};">{_e(o.original_name)}</span> '
            f'<span style="color:{MUTED};">— {_e(_shorten(o.new_description or "", 64))}</span></div>'
            for o in rewritten[:6]
        )
        if len(rewritten) > 6:
            rows += (
                f'<div class="mono" style="color:{FAINT}; font-size:11.5px;">'
                f"+ {len(rewritten) - 6} more</div>"
            )
        cards.append(f"""
      <div style="border:1px solid rgba(255,255,255,0.09); border-radius:13px; padding:18px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
          <span class="mono badge" style="color:{PURPLE}; border:1px solid rgba(199,155,232,0.3);">REWRITTEN · {len(rewritten)}</span>
        </div>
        <div class="mono" style="font-size:12px; line-height:2.1;">{rows}</div>
      </div>""")

    if not cards:
        cards.append(f"""
      <div style="border:1px solid rgba(255,255,255,0.09); border-radius:13px; padding:18px; grid-column:1 / -1;">
        <span class="mono" style="font-size:12px; color:{MUTED};">The optimizer proposed no changes.</span>
      </div>""")

    notes = (
        f'<p style="font-size:14px; color:{SUBTLE}; margin:0 0 16px;">{_e(report.plan.notes)}</p>'
        if report.plan.notes
        else ""
    )
    return f"""
  <section style="margin-top:34px;">
    <h2 style="font-size:18px; font-weight:600; margin:0 0 6px;">What curation changed</h2>
    <p style="font-size:14px; color:{SUBTLE}; margin:0 0 6px;">Non-destructive overlay — origin tools untouched. {len(report.plan.overrides)} change{"s" if len(report.plan.overrides) != 1 else ""}.</p>
    {notes}
    <div class="cols2" style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">
      {"".join(cards)}
    </div>
  </section>"""


def _trace_side(side: SideView, label: str, color: str, expected: set[str]) -> str:
    calls = []
    for call in side.calls:
        ok = not call.is_error and call.name in expected
        box_class = "call-ok" if ok else "call-bad"
        calls.append(
            f'<div class="callbox mono {box_class}">{_e(call.name)}({_e(call.args)})</div>'
        )
    if not calls:
        calls.append(
            f'<div class="mono" style="color:{MUTED}; font-size:12px;">(no tool calls)</div>'
        )

    mark, mark_color = _mark(side.hit)
    verdict = f'<div class="mono" style="margin-top:14px; color:{mark_color}; font-size:12px;">{mark} {"hit" if side.hit else "miss"}'
    if side.missing_slots:
        missing = ", ".join(" / ".join(slot) for slot in side.missing_slots)
        verdict += f" — never called: {_e(missing)}"
    if side.judge_passed is False and side.judge_rationale:
        verdict += f" — judge: {_e(_shorten(side.judge_rationale, 100))}"
    verdict += "</div>"

    answer = ""
    if side.final_text:
        answer = (
            f'<div class="mono" style="margin-top:10px; color:{MUTED}; font-size:11.5px;">'
            f"final answer › {_e(_shorten(side.final_text, 160))}</div>"
        )
    return f"""
        <div style="padding:20px;">
          <div class="mono" style="font-size:11px; color:{color}; letter-spacing:0.06em; margin-bottom:16px;">{label}</div>
          <div class="mono" style="font-size:12px; line-height:1.7; color:{SUBTLE};">
            {"".join(calls)}
            {verdict}
            {answer}
          </div>
        </div>"""


def _trace_section(report: ComparisonReport) -> str:
    if not report.tasks:
        return ""
    first_improved = next((t.task_id for t in report.improved), None)
    blocks = []
    for i, task in enumerate(report.tasks):
        expected = {name for slot in task.expected_slots for name in slot}
        # calls in the "after" run use curated names — accept those too
        curated_expected = expected | {
            o.presented_name
            for o in report.plan.overrides
            if not o.hidden and o.original_name in expected
        }
        before_mark, before_color = _mark(task.before.hit)
        after_mark, after_color = _mark(task.after.hit)
        open_attr = " open" if task.task_id == first_improved else ""
        blocks.append(f"""
    <details{open_attr} style="border:1px solid rgba(255,255,255,0.09); border-radius:14px; overflow:hidden; margin-bottom:12px;">
      <summary style="display:flex; align-items:center; justify-content:space-between; gap:12px; padding:16px 20px; background:#101316;">
        <span style="display:flex; align-items:center; gap:14px; min-width:0;">
          <span class="mono" style="font-size:12px; color:{MUTED};">#{i + 1:02d}</span>
          <span style="font-size:14px; color:#EAECEE; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">“{_e(_shorten(task.prompt, 80))}”</span>
        </span>
        <span style="display:flex; align-items:center; gap:14px; flex-shrink:0;">
          <span class="mono" style="font-size:12px; color:{before_color};">{before_mark} before</span>
          <span class="mono" style="font-size:12px; color:{after_color};">{after_mark} after</span>
          <span class="mono chev" style="font-size:16px; color:{MUTED};">⌄</span>
        </span>
      </summary>
      <div class="cols2" style="display:grid; grid-template-columns:1fr 1fr; gap:0; border-top:1px solid rgba(255,255,255,0.07);">
        <div style="border-right:1px solid rgba(255,255,255,0.07);">
          {_trace_side(task.before, "BEFORE · raw tool set", AMBER, expected)}
        </div>
        {_trace_side(task.after, "AFTER · curated tool set", GREEN, curated_expected)}
      </div>
    </details>""")

    return f"""
  <section style="margin-top:34px;">
    <h2 style="font-size:18px; font-weight:600; margin:0 0 6px;">Reasoning-path traces</h2>
    <p style="font-size:14px; color:{SUBTLE}; margin:0 0 16px;">Every task's tool calls, before and after curation. Click a row to expand.</p>
    {"".join(blocks)}
  </section>"""


def render_html(report: ComparisonReport) -> str:
    timestamp = report.generated_at.strftime("%Y-%m-%dT%H:%MZ")
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
{_headline(report)}
{_task_table(report)}
{_curation_section(report)}
{_trace_section(report)}
  <footer style="margin-top:44px; padding-top:24px; border-top:1px solid rgba(255,255,255,0.08); display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;">
    <div class="mono" style="font-size:12.5px; color:{MUTED};">generated by <span style="color:{SUBTLE};">whetkit v{__version__}</span> · {timestamp} · origin server never modified</div>
    <a href="https://github.com/benlamlih/whetkit" class="mono" style="font-size:12.5px;">github.com/benlamlih/whetkit ↗</a>
  </footer>
</div>
</body>
</html>
"""
