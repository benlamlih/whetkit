"""Export a curation plan into shareable formats.

A plan is most useful when it leaves the machine it was made on: as a
Markdown fix report you can paste into an upstream issue or PR, or as
neutral JSON for gateways and scripts that apply metadata overrides
themselves.
"""

import json

from whetkit.curation.plan import CurationPlan, ToolOverride


def _action(override: ToolOverride) -> str:
    if override.hidden:
        return "hide"
    if override.new_name and override.new_description:
        return "rename + rewrite"
    if override.new_name:
        return "rename"
    return "rewrite"


def plan_to_markdown(plan: CurationPlan) -> str:
    """A fix report for humans — upstream-issue / PR-description ready."""
    lines = [
        "## Proposed tool-surface fixes",
        "",
        "Metadata-only changes — no schemas or behavior are touched. "
        "Measured with [whetkit](https://github.com/benlamlih/whetkit); "
        "each change is reversible (it can be served as an overlay without "
        "modifying the server).",
        "",
    ]
    if plan.notes:
        lines += [f"> {plan.notes.strip()}", ""]
    lines += [
        "| Tool | Change | Proposal | Why |",
        "|---|---|---|---|",
    ]
    for override in plan.overrides:
        if override.hidden:
            proposal = "hide from the tool list"
        else:
            parts = []
            if override.new_name:
                parts.append(f"rename to `{override.new_name}`")
            if override.new_description:
                parts.append(f'description: "{override.new_description.strip()}"')
            proposal = "; ".join(parts)
        cell = lambda s: " ".join(str(s).split()).replace("|", "\\|")  # noqa: E731
        lines.append(
            f"| `{override.original_name}` | {_action(override)} "
            f"| {cell(proposal)} | {cell(override.reason) or '—'} |"
        )
    return "\n".join(lines) + "\n"


def plan_to_json(plan: CurationPlan) -> str:
    """Neutral override list for gateways/scripts that apply their own
    metadata overrides (Cloudflare portal aliases, MetaMCP namespaces, …)."""
    overrides = [
        {
            "original_name": o.original_name,
            "action": _action(o),
            "new_name": o.new_name,
            "new_description": o.new_description,
            "hidden": o.hidden,
            "reason": o.reason,
        }
        for o in plan.overrides
    ]
    return json.dumps({"notes": plan.notes, "overrides": overrides}, indent=2) + "\n"
