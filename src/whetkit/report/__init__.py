"""Before/after comparison reports (JSON + self-contained HTML)."""

from whetkit.report.builder import ComparisonReport, TaskComparison, build_report
from whetkit.report.html import render_html

__all__ = ["ComparisonReport", "TaskComparison", "build_report", "render_html"]
