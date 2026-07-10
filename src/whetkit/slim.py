"""whetkit slim: audit and shrink the union tool surface of an MCP client config.

A Claude Code / Cursor / Claude Desktop config unions every configured
server's tool definitions into every single message. This module reads the
``mcpServers`` object those clients share, inventories the union, prices it,
finds tools duplicated ACROSS servers, and can emit per-server hide plans
plus a rewritten config that serves each slimmed server through
``whetkit overlay`` — the original config is never modified.
"""

import json
import shutil
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel

from whetkit.curation.plan import CurationPlan, ToolOverride, save_plan
from whetkit.doctor import (
    DUPLICATE_DESCRIPTION_SIMILARITY,
    DUPLICATE_NAME_SIMILARITY,
    _is_crud_family,
    _similarity,
)
from whetkit.mcp.introspect import ServerInventory
from whetkit.mcp.transport import ServerSpec, spec_from_dict

KNOWN_CONFIG_LOCATIONS = (
    "~/.claude.json (Claude Code, global)",
    ".mcp.json (Claude Code, project)",
    "~/.cursor/mcp.json (Cursor)",
    "~/Library/Application Support/Claude/claude_desktop_config.json (Claude Desktop)",
)


class SkippedServer(BaseModel):
    name: str
    reason: str


class ClientConfig(BaseModel):
    """The servers a client config declares, mapped to whetkit specs."""

    path: str
    servers: dict[str, ServerSpec]
    skipped: list[SkippedServer] = []
    raw_entries: dict[str, dict] = {}
    standalone: bool = True
    """False when the source file holds more than mcpServers (a full
    ~/.claude.json with settings/projects): the slimmed output is then a
    fragment to merge, not a drop-in replacement for the original file."""


def _entry_to_spec(entry: dict) -> ServerSpec:
    """One mcpServers entry -> ServerSpec. Client configs use the same shape
    across Claude Code, Cursor, and Claude Desktop."""
    entry_type = entry.get("type", "stdio" if "command" in entry else "http")
    if entry_type == "sse":
        raise ValueError("sse transport is not supported yet (audit skips it)")
    if entry_type in ("http", "streamable-http", "streamable_http"):
        data = {"kind": "http", "url": entry["url"]}
        if headers := entry.get("headers"):
            data["headers"] = headers
        return spec_from_dict(data)
    data = {"kind": "stdio", "command": entry["command"]}
    for key in ("args", "env", "cwd"):
        if entry.get(key) is not None:
            data[key] = entry[key]
    return spec_from_dict(data)


def parse_client_config(path: str | Path) -> ClientConfig:
    """Read every server out of a client config file.

    Accepts any JSON document with an ``mcpServers`` object; for Claude
    Code's ``~/.claude.json`` the per-project ``mcpServers`` blocks under
    ``projects`` are merged in too (global entries win on name clashes)."""
    path = Path(path).expanduser()
    if not path.is_file():
        locations = "\n  ".join(KNOWN_CONFIG_LOCATIONS)
        raise ValueError(f"no config file at {path} — known client locations:\n  {locations}")
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    merged: dict[str, dict] = {}
    for project in (document.get("projects") or {}).values():
        if isinstance(project, dict):
            merged.update(project.get("mcpServers") or {})
    merged.update(document.get("mcpServers") or {})
    if not merged:
        raise ValueError(
            f"{path} has no mcpServers entries — nothing to audit. "
            "Point --config at a client config that declares MCP servers."
        )

    servers: dict[str, ServerSpec] = {}
    skipped: list[SkippedServer] = []
    for name, entry in merged.items():
        if not isinstance(entry, dict):
            skipped.append(SkippedServer(name=name, reason="entry is not an object"))
            continue
        try:
            servers[name] = _entry_to_spec(entry)
        except (ValueError, KeyError) as exc:
            skipped.append(SkippedServer(name=name, reason=str(exc)))
    standalone = set(document) <= {"mcpServers"}
    return ClientConfig(
        path=str(path),
        servers=servers,
        skipped=skipped,
        raw_entries=merged,
        standalone=standalone,
    )


class CrossServerDuplicate(BaseModel):
    """Two tools on DIFFERENT servers that look interchangeable to a model."""

    keep_server: str
    keep_tool: str
    hide_server: str
    hide_tool: str
    name_similarity: float
    description_similarity: float

    def describe(self) -> str:
        return (
            f"{self.hide_server}.{self.hide_tool} ↔ {self.keep_server}.{self.keep_tool} "
            f"look interchangeable (name {self.name_similarity:.0%}, "
            f"description {self.description_similarity:.0%}) — agents split "
            "attention across servers"
        )


def cross_server_duplicates(
    inventories: dict[str, ServerInventory],
) -> list[CrossServerDuplicate]:
    """Near-duplicate pairs across servers only (each server's own internal
    duplicates are doctor's job). Winner = the more informative description;
    tie -> the server listed first in the config."""
    duplicates: list[CrossServerDuplicate] = []
    for (name_a, inv_a), (name_b, inv_b) in combinations(inventories.items(), 2):
        for tool_a in inv_a.tools:
            for tool_b in inv_b.tools:
                name_sim = _similarity(tool_a.name.lower(), tool_b.name.lower())
                desc_sim = (
                    _similarity(tool_a.description.lower(), tool_b.description.lower())
                    if tool_a.description.strip() and tool_b.description.strip()
                    else 0.0
                )
                if _is_crud_family(tool_a.name, tool_b.name):
                    continue
                if (
                    name_sim < DUPLICATE_NAME_SIMILARITY
                    and desc_sim < DUPLICATE_DESCRIPTION_SIMILARITY
                ):
                    continue
                if tool_b.description_tokens > tool_a.description_tokens:
                    keep, hide = (name_b, tool_b), (name_a, tool_a)
                else:
                    keep, hide = (name_a, tool_a), (name_b, tool_b)
                duplicates.append(
                    CrossServerDuplicate(
                        keep_server=keep[0],
                        keep_tool=keep[1].name,
                        hide_server=hide[0],
                        hide_tool=hide[1].name,
                        name_similarity=name_sim,
                        description_similarity=desc_sim,
                    )
                )
    return duplicates


def build_dedupe_plans(
    inventories: dict[str, ServerInventory],
    duplicates: list[CrossServerDuplicate],
    hide_servers: set[str] = frozenset(),
    keep_servers: set[str] = frozenset(),
) -> dict[str, CurationPlan]:
    """Per-server hide plans: duplicate losers, plus every tool of servers
    the user asked to hide outright. Servers in ``keep_servers`` are never
    touched. Only servers that end up with overrides get a plan."""
    hides: dict[str, dict[str, str]] = {}
    for duplicate in duplicates:
        if duplicate.hide_server in keep_servers:
            continue
        hides.setdefault(duplicate.hide_server, {})[duplicate.hide_tool] = (
            f"Duplicate of {duplicate.keep_server}.{duplicate.keep_tool} "
            "(kept there); hidden to stop split attention."
        )
    for server in hide_servers - keep_servers:
        inventory = inventories.get(server)
        if inventory is None:
            continue
        for tool in inventory.tools:
            hides.setdefault(server, {})[tool.name] = (
                "Whole server hidden by --hide; delete the plan to restore."
            )

    # A keeper named in a reason can itself be hidden by ANOTHER pair (ties
    # chain: c hidden "kept b" while b is hidden "kept a"). Re-point every
    # duplicate reason at a copy that actually stays visible.
    def visible_holder(tool: str) -> str | None:
        for server, inventory in inventories.items():
            if tool in hides.get(server, {}):
                continue
            if any(t.name == tool for t in inventory.tools):
                return server
        return None

    for tool_reasons in hides.values():
        for tool, reason in list(tool_reasons.items()):
            if not reason.startswith("Duplicate of "):
                continue
            survivor = visible_holder(tool)
            if survivor is not None:
                tool_reasons[tool] = (
                    f"Duplicate of {survivor}.{tool} (kept there); hidden to stop split attention."
                )

    plans: dict[str, CurationPlan] = {}
    for server, tool_reasons in hides.items():
        plans[server] = CurationPlan(
            server=server,
            notes="whetkit slim: hide-only view plan; the origin server is never modified.",
            overrides=[
                ToolOverride(original_name=tool, hidden=True, reason=reason)
                for tool, reason in sorted(tool_reasons.items())
            ],
        )
    return plans


def write_slim_output(
    config: ClientConfig,
    plans: dict[str, CurationPlan],
    out_dir: str | Path,
) -> Path:
    """Write per-server origin spec + plan, and the rewritten client config.

    Servers with a plan are re-pointed at ``whetkit overlay``; everything
    else (including skipped entries) is copied through verbatim. Env values
    are preserved as-is — they came from a config file of the same
    sensitivity. Returns the path of the slimmed config."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    # GUI clients (Claude Desktop) launch servers with a minimal PATH where a
    # bare 'whetkit' often doesn't resolve — pin the absolute path when known.
    whetkit_command = shutil.which("whetkit") or "whetkit"

    slimmed: dict[str, dict] = {}
    for name, raw_entry in config.raw_entries.items():
        plan = plans.get(name)
        if plan is None or name not in config.servers:
            slimmed[name] = raw_entry
            continue
        server_dir = out / name
        server_dir.mkdir(parents=True, exist_ok=True)
        spec_path = server_dir / "server.json"
        plan_path = server_dir / "plan.yaml"
        spec_path.write_text(config.servers[name].model_dump_json(indent=2) + "\n")
        save_plan(plan, plan_path)
        slimmed[name] = {
            "command": whetkit_command,
            "args": [
                "overlay",
                "--server",
                str(spec_path.resolve()),
                "--plan",
                str(plan_path.resolve()),
            ],
        }

    slimmed_path = out / "mcp.slimmed.json"
    slimmed_path.write_text(json.dumps({"mcpServers": slimmed}, indent=2) + "\n")
    return slimmed_path
