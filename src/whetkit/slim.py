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
    always_load: list[str] = []
    """Servers whose entries carry ``alwaysLoad: true`` — under Claude Code's
    tool search these stay in context; everything else loads on demand."""
    defer_loading_entries: list[str] = []
    """Servers whose entries carry ``defer_loading`` — a field Claude Code
    parses and silently ignores (anthropics/claude-code#26844); the real
    mechanism is ``alwaysLoad``. Worth a loud lint."""


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
    always_load = sorted(
        name
        for name, entry in merged.items()
        if isinstance(entry, dict) and entry.get("alwaysLoad") is True
    )
    defer_loading_entries = sorted(
        name
        for name, entry in merged.items()
        if isinstance(entry, dict) and "defer_loading" in entry
    )
    return ClientConfig(
        path=str(path),
        servers=servers,
        skipped=skipped,
        raw_entries=merged,
        standalone=standalone,
        always_load=always_load,
        defer_loading_entries=defer_loading_entries,
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
    """Per-server hide plans from duplicate clusters plus whole-server hides.

    Duplicate pairs are consolidated into clusters (connected components):
    each cluster keeps exactly ONE copy — the most informative description,
    tie broken by config order — and every other member is hidden with a
    reason naming that single visible winner. Servers in ``keep_servers``
    are never touched; only servers that end up with overrides get a plan."""
    order = {name: index for index, name in enumerate(inventories)}

    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        parent[find(a)] = find(b)

    for duplicate in duplicates:
        union(
            (duplicate.keep_server, duplicate.keep_tool),
            (duplicate.hide_server, duplicate.hide_tool),
        )

    clusters: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for node in list(parent):
        clusters.setdefault(find(node), []).append(node)

    def description_tokens(server: str, tool: str) -> int:
        inventory = inventories.get(server)
        if inventory is None:
            return 0
        return next((t.description_tokens for t in inventory.tools if t.name == tool), 0)

    hides: dict[str, dict[str, str]] = {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        winner = max(
            members,
            key=lambda node: (description_tokens(*node), -order.get(node[0], 1_000_000)),
        )
        for server, tool in members:
            if (server, tool) == winner or server in keep_servers:
                continue
            hides.setdefault(server, {})[tool] = (
                f"Duplicate of {winner[0]}.{winner[1]} (the kept copy); "
                "hidden to stop split attention."
            )
    for server in hide_servers - keep_servers:
        inventory = inventories.get(server)
        if inventory is None:
            continue
        for tool in inventory.tools:
            hides.setdefault(server, {})[tool.name] = (
                "Whole server hidden by --hide; delete the plan to restore."
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
    inventories: dict[str, ServerInventory] | None = None,
) -> tuple[Path, list[str]]:
    """Write per-server origin spec + plan, and the rewritten client config.

    Servers with a plan are re-pointed at ``whetkit overlay``; a server
    whose plan hides EVERY tool is dropped from the slimmed config outright
    (no point launching an overlay to serve nothing). Everything else
    (including skipped entries) is copied through verbatim; env values are
    preserved as-is — they came from a config file of the same sensitivity.
    Returns (slimmed config path, names of dropped servers)."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    # GUI clients (Claude Desktop) launch servers with a minimal PATH where a
    # bare 'whetkit' often doesn't resolve — pin the absolute path when known.
    whetkit_command = shutil.which("whetkit") or "whetkit"

    slimmed: dict[str, dict] = {}
    removed: list[str] = []
    for name, raw_entry in config.raw_entries.items():
        plan = plans.get(name)
        if plan is None or name not in config.servers:
            slimmed[name] = raw_entry
            continue
        inventory = (inventories or {}).get(name)
        hidden = {o.original_name for o in plan.overrides if o.hidden}
        if inventory is not None and hidden >= {t.name for t in inventory.tools}:
            removed.append(name)
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
    return slimmed_path, removed


def recommend_hot_servers(
    inventories: dict[str, ServerInventory],
    trace_store_path: str | Path,
) -> tuple[set[str], set[str], list[str]]:
    """From real usage traces: which servers deserve ``alwaysLoad: true``.

    A server is hot when any of its tools was actually called. Attribution
    is by spec label when it matches, otherwise by unique tool-name lookup
    across inventories. Returns (hot, cold, warnings)."""
    from whetkit.tracing import TraceStore

    path = Path(trace_store_path).expanduser()
    if not path.is_file():
        raise ValueError(f"no trace store at {path}")

    labels = {}
    tool_owner: dict[str, set[str]] = {}
    for name, inventory in inventories.items():
        labels[inventory.server] = name
        for tool in inventory.tools:
            tool_owner.setdefault(tool.name, set()).add(name)

    hot: set[str] = set()
    warnings: list[str] = []
    unmatched_tools: set[str] = set()
    with TraceStore(path) as store:
        for run in store.load_runs(None):
            server = labels.get(run.server)
            for tool in run.called_tool_names:
                if server is not None:
                    hot.add(server)
                    continue
                owners = tool_owner.get(tool, set())
                if len(owners) == 1:
                    hot.add(next(iter(owners)))
                elif not owners:
                    unmatched_tools.add(tool)
    if unmatched_tools:
        warnings.append(
            "traces mention tools not in this config (ignored): "
            + ", ".join(sorted(unmatched_tools)[:5])
        )
    cold = set(inventories) - hot
    return hot, cold, warnings


def write_hot_config(config: ClientConfig, hot: set[str], out_dir: str | Path) -> Path:
    """``hot.mcp.json``: the user's entries with ``alwaysLoad: true`` stamped
    on the recommended servers (and removed elsewhere) — ready for Claude
    Code's tool search. Same merge semantics as the slimmed config."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stamped: dict[str, dict] = {}
    for name, entry in config.raw_entries.items():
        entry = dict(entry) if isinstance(entry, dict) else entry
        if isinstance(entry, dict):
            entry.pop("defer_loading", None)  # silently-ignored field, drop it
            if name in hot:
                entry["alwaysLoad"] = True
            else:
                entry.pop("alwaysLoad", None)
        stamped[name] = entry
    path = out / "hot.mcp.json"
    path.write_text(json.dumps({"mcpServers": stamped}, indent=2) + "\n")
    return path
