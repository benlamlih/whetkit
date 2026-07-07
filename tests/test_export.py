import json
from pathlib import Path

from typer.testing import CliRunner

from whetkit.cli import app
from whetkit.curation import CurationPlan, ToolOverride, save_plan
from whetkit.curation.export import plan_to_json, plan_to_markdown

runner = CliRunner()


def plan() -> CurationPlan:
    return CurationPlan(
        notes="tidy the shop tools",
        overrides=[
            ToolOverride(
                original_name="data_query_1",
                new_name="search_products",
                new_description="Search the product catalog by name keywords.",
                reason="cryptic name",
            ),
            ToolOverride(original_name="legacy_search", hidden=True, reason="duplicate"),
            ToolOverride(
                original_name="inv_check",
                new_description="Return stock for a product id.",
                reason="vague",
            ),
        ],
    )


class TestExportFormats:
    def test_markdown_report(self) -> None:
        md = plan_to_markdown(plan())
        assert "## Proposed tool-surface fixes" in md
        assert "> tidy the shop tools" in md
        assert "| `data_query_1` | rename + rewrite |" in md
        assert "rename to `search_products`" in md
        assert "| `legacy_search` | hide | hide from the tool list |" in md
        assert "| `inv_check` | rewrite |" in md
        assert "Metadata-only changes" in md

    def test_markdown_escapes_pipes(self) -> None:
        tricky = CurationPlan(
            overrides=[
                ToolOverride(original_name="a", new_description="either x | y", reason="has | pipe")
            ]
        )
        md = plan_to_markdown(tricky)
        assert "x \\| y" in md and "has \\| pipe" in md

    def test_json_round_trips(self) -> None:
        data = json.loads(plan_to_json(plan()))
        assert data["notes"] == "tidy the shop tools"
        by_name = {o["original_name"]: o for o in data["overrides"]}
        assert by_name["data_query_1"]["new_name"] == "search_products"
        assert by_name["legacy_search"]["hidden"] is True
        assert by_name["inv_check"]["action"] == "rewrite"


class TestExportCli:
    def test_export_to_file(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        save_plan(plan(), plan_file)
        out = tmp_path / "fixes.md"
        result = runner.invoke(
            app, ["export", "--plan", str(plan_file), "--to", "markdown", "--out", str(out)]
        )
        assert result.exit_code == 0
        assert "search_products" in out.read_text()

    def test_export_stdout_json(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        save_plan(plan(), plan_file)
        result = runner.invoke(app, ["export", "--plan", str(plan_file), "--to", "json"])
        assert result.exit_code == 0
        assert json.loads(result.output)["overrides"]

    def test_missing_plan_is_friendly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["export", "--plan", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0
        assert "no curation plan" in result.output

    def test_empty_plan_exits_nonzero(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        save_plan(CurationPlan(), plan_file)
        result = runner.invoke(app, ["export", "--plan", str(plan_file)])
        assert result.exit_code == 1
        assert "nothing to export" in result.output
