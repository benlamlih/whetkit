"""whetkit doctor: zero-setup lint of an MCP server's tool surface.

Heuristic checks over tool names, descriptions, and schemas — no eval
tasks and no API key required. Findings point at the metadata problems
that make agents pick the wrong tool; ``whetkit run``/``curate`` are how
you then measure and fix them.
"""

import re
from difflib import SequenceMatcher
from enum import StrEnum
from itertools import combinations

from pydantic import BaseModel

from whetkit.mcp.introspect import ServerInventory, ToolInfo

# Thresholds are deliberately opinionated defaults, tuned on real servers
# (official filesystem/memory/everything, Context7, Playwright): loose
# enough that a clean server reports nothing.
VAGUE_DESCRIPTION_TOKENS = 6
BLOATED_DESCRIPTION_TOKENS = 150
DUPLICATE_NAME_SIMILARITY = 0.8
DUPLICATE_DESCRIPTION_SIMILARITY = 0.8
COMPLEX_SCHEMA_SCORE = 12
LARGE_SURFACE_TOOLS = 30
LARGE_SURFACE_TOKENS = 1500

_GENERIC_NAME_RE = re.compile(
    r"(?:^|_)(do|thing|stuff|util|utils|helper|helpers|misc|tmp|temp|proc|handle|manage)(?:_|$)"
)
_VERSION_SUFFIX_RE = re.compile(r"(?:_v?\d+)$")


class Severity(StrEnum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class Finding(BaseModel):
    """One diagnosed problem with the tool surface."""

    severity: Severity
    check: str
    tools: list[str]
    message: str


def _check_descriptions(tools: list[ToolInfo]) -> list[Finding]:
    findings = []
    for tool in tools:
        if not tool.description.strip():
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    check="missing-description",
                    tools=[tool.name],
                    message=(
                        f"{tool.name} has no description — the model can only "
                        "guess from the name."
                    ),
                )
            )
        elif tool.description_tokens < VAGUE_DESCRIPTION_TOKENS:
            findings.append(
                Finding(
                    severity=Severity.WARN,
                    check="vague-description",
                    tools=[tool.name],
                    message=(
                        f"{tool.name}: description is ~{tool.description_tokens} tokens "
                        f"({tool.description.strip()!r}) — say what it does, over what data, "
                        "and when to use it."
                    ),
                )
            )
        elif tool.description_tokens > BLOATED_DESCRIPTION_TOKENS:
            findings.append(
                Finding(
                    severity=Severity.WARN,
                    check="bloated-description",
                    tools=[tool.name],
                    message=(
                        f"{tool.name}: description is ~{tool.description_tokens} tokens and is "
                        "resent on every request — tighten it and move detail into the schema."
                    ),
                )
            )
    return findings


def _check_names(tools: list[ToolInfo]) -> list[Finding]:
    findings = []
    for tool in tools:
        reasons = []
        if _GENERIC_NAME_RE.search(tool.name.lower()):
            reasons.append("generic words say nothing about behavior")
        if _VERSION_SUFFIX_RE.search(tool.name.lower()):
            reasons.append("a version suffix invites confusion with its siblings")
        if reasons:
            findings.append(
                Finding(
                    severity=Severity.WARN,
                    check="cryptic-name",
                    tools=[tool.name],
                    message=f"{tool.name}: {'; '.join(reasons)} — prefer a verb_object name.",
                )
            )
    return findings


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _is_crud_family(a: str, b: str) -> bool:
    """delete_entities / delete_relations (same verb, different objects) or
    create_relations / delete_relations (same object, different verbs) are
    deliberate CRUD families, not duplicates. Digit tails don't count —
    data_query_1 / data_query_2 is a duplicate, not a family."""
    parts_a, parts_b = a.lower().split("_"), b.lower().split("_")
    if len(parts_a) < 2 or len(parts_b) < 2:
        return False
    head_a, tail_a, head_b, tail_b = parts_a[0], parts_a[-1], parts_b[0], parts_b[-1]
    if not (tail_a.isalpha() and tail_b.isalpha() and head_a.isalpha() and head_b.isalpha()):
        return False
    return (head_a == head_b and tail_a != tail_b) or (head_a != head_b and tail_a == tail_b)


# Within a family, only a near-identical description means real duplication.
FAMILY_DESCRIPTION_SIMILARITY = 0.95


def _check_duplicates(tools: list[ToolInfo]) -> list[Finding]:
    findings = []
    for a, b in combinations(tools, 2):
        name_sim = _similarity(a.name.lower(), b.name.lower())
        desc_sim = (
            _similarity(a.description.lower(), b.description.lower())
            if a.description.strip() and b.description.strip()
            else 0.0
        )
        if _is_crud_family(a.name, b.name):
            suspicious = desc_sim >= FAMILY_DESCRIPTION_SIMILARITY
        else:
            suspicious = (
                name_sim >= DUPLICATE_NAME_SIMILARITY
                or desc_sim >= DUPLICATE_DESCRIPTION_SIMILARITY
            )
        if suspicious:
            findings.append(
                Finding(
                    severity=Severity.WARN,
                    check="possible-duplicate",
                    tools=[a.name, b.name],
                    message=(
                        f"{a.name} and {b.name} look interchangeable "
                        f"(name similarity {name_sim:.0%}, description {desc_sim:.0%}) — "
                        "near-duplicates split the model's attention."
                    ),
                )
            )
    return findings


def _check_schemas(tools: list[ToolInfo]) -> list[Finding]:
    findings = []
    for tool in tools:
        if tool.complexity > COMPLEX_SCHEMA_SCORE:
            findings.append(
                Finding(
                    severity=Severity.INFO,
                    check="complex-schema",
                    tools=[tool.name],
                    message=(
                        f"{tool.name}: schema complexity {tool.complexity} — deeply nested or "
                        "union-heavy arguments are hard for models to fill correctly."
                    ),
                )
            )
    return findings


def _check_surface(inventory: ServerInventory) -> list[Finding]:
    findings = []
    if inventory.tool_count > LARGE_SURFACE_TOOLS:
        findings.append(
            Finding(
                severity=Severity.WARN,
                check="large-surface",
                tools=[],
                message=(
                    f"{inventory.tool_count} tools — past ~{LARGE_SURFACE_TOOLS} the model's "
                    "selection accuracy degrades; consider pruning or splitting the server."
                ),
            )
        )
    if inventory.total_description_tokens > LARGE_SURFACE_TOKENS:
        findings.append(
            Finding(
                severity=Severity.WARN,
                check="large-surface",
                tools=[],
                message=(
                    f"~{inventory.total_description_tokens} description tokens are resent on "
                    "every request — that budget comes out of the agent's context window."
                ),
            )
        )
    return findings


_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}


def diagnose(inventory: ServerInventory) -> list[Finding]:
    """Run every check; findings come back most severe first."""
    findings = [
        *_check_descriptions(inventory.tools),
        *_check_names(inventory.tools),
        *_check_duplicates(inventory.tools),
        *_check_schemas(inventory.tools),
        *_check_surface(inventory),
    ]
    return sorted(findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.check, f.tools))
