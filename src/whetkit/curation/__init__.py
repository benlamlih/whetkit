"""Curation: analyze failing traces, propose tool-set fixes, apply them as a
reversible overlay in front of the origin server."""

from whetkit.curation.optimizer import OptimizerConfig, propose_plan
from whetkit.curation.overlay import CuratedMCPClient, serve_overlay
from whetkit.curation.plan import CurationPlan, ToolOverride, load_plan, save_plan

__all__ = [
    "CuratedMCPClient",
    "CurationPlan",
    "OptimizerConfig",
    "ToolOverride",
    "load_plan",
    "propose_plan",
    "save_plan",
    "serve_overlay",
]
