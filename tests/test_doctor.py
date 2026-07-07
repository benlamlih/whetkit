from whetkit.doctor import Severity, diagnose
from whetkit.mcp.introspect import ServerInventory, ToolInfo


def tool(name: str, description: str = "A perfectly clear description of what this does.", **kw):
    return ToolInfo(name=name, description=description, **kw)


def inventory(*tools: ToolInfo) -> ServerInventory:
    return ServerInventory(server="test", tools=list(tools))


def checks_for(findings, name: str) -> set[str]:
    return {f.check for f in findings if name in f.tools}


class TestDoctorChecks:
    def test_clean_surface_reports_nothing(self) -> None:
        inv = inventory(
            tool("search_products", "Search the product catalog by name substring and max price."),
            tool("create_order", "Place an order for a customer given product id and quantity."),
        )
        assert diagnose(inv) == []

    def test_missing_description_is_error(self) -> None:
        findings = diagnose(inventory(tool("mystery", "")))
        assert findings[0].severity == Severity.ERROR
        assert findings[0].check == "missing-description"

    def test_vague_and_bloated_descriptions(self) -> None:
        findings = diagnose(
            inventory(
                tool("inv_check", "Inv."),
                tool("resolve_library", "word " * 200),
            )
        )
        assert "vague-description" in checks_for(findings, "inv_check")
        assert "bloated-description" in checks_for(findings, "resolve_library")

    def test_cryptic_names(self) -> None:
        findings = diagnose(
            inventory(
                tool("do_thing", "Does the thing for the given target and reference value."),
                tool("ord_status_check_tool_v2", "Checks the status of an order by reference."),
            )
        )
        assert "cryptic-name" in checks_for(findings, "do_thing")
        assert "cryptic-name" in checks_for(findings, "ord_status_check_tool_v2")

    def test_near_duplicates_flagged_as_pair(self) -> None:
        findings = diagnose(
            inventory(
                tool("fetch_record", "Fetches a record from the customer store by identifier."),
                tool("get_record", "Fetches a record from the customer store by its identifier."),
            )
        )
        dupes = [f for f in findings if f.check == "possible-duplicate"]
        assert len(dupes) == 1
        assert set(dupes[0].tools) == {"fetch_record", "get_record"}

    def test_crud_family_is_not_a_duplicate(self) -> None:
        findings = diagnose(
            inventory(
                tool("delete_entities", "Delete multiple entities from the knowledge graph."),
                tool("delete_relations", "Delete multiple relations from the knowledge graph."),
            )
        )
        assert not [f for f in findings if f.check == "possible-duplicate"]

    def test_digit_suffix_twins_still_flagged(self) -> None:
        findings = diagnose(
            inventory(
                tool("data_query_1", "Queries data from the primary system store."),
                tool("data_query_2", "Queries data from the system, alternate path."),
            )
        )
        assert [f for f in findings if f.check == "possible-duplicate"]

    def test_complex_schema_is_info(self) -> None:
        deep = {
            "type": "object",
            "properties": {
                f"p{i}": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "object", "properties": {"x": {"type": "string"}}},
                    ]
                }
                for i in range(4)
            },
        }
        gnarly = tool(
            "gnarly", "Does a well-described but structurally gnarly thing.", input_schema=deep
        )
        findings = diagnose(inventory(gnarly))
        assert "complex-schema" in checks_for(findings, "gnarly")
        assert all(f.severity == Severity.INFO for f in findings)

    def test_large_surface(self) -> None:
        tools = [
            tool(f"tool_number_{i:02d}", f"Performs distinct operation {i} on the data store.")
            for i in range(31)
        ]
        findings = diagnose(inventory(*tools))
        assert any(f.check == "large-surface" for f in findings)

    def test_errors_sort_first(self) -> None:
        findings = diagnose(
            inventory(
                tool("fine_tool", "Looks up a customer by id and returns the full record."),
                tool("vague", "Hm."),
                tool("undescribed", ""),
            )
        )
        assert findings[0].severity == Severity.ERROR


class TestDoctorCli:
    def test_doctor_on_messy_sample_server(self) -> None:
        from pathlib import Path

        from typer.testing import CliRunner

        from whetkit.cli import app

        sample = Path(__file__).parent.parent / "examples" / "sample-server"
        result = CliRunner().invoke(app, ["doctor", "--server", str(sample)])
        assert result.exit_code == 0
        assert "vague-description" in result.output
        assert "possible-duplicate" in result.output
        assert "whetkit curate" in result.output

    def test_fail_on_warn_exits_nonzero(self) -> None:
        from pathlib import Path

        from typer.testing import CliRunner

        from whetkit.cli import app

        sample = Path(__file__).parent.parent / "examples" / "sample-server"
        result = CliRunner().invoke(app, ["doctor", "--server", str(sample), "--fail-on", "warn"])
        assert result.exit_code == 1

    def test_json_output_parses(self) -> None:
        import json
        from pathlib import Path

        from typer.testing import CliRunner

        from whetkit.cli import app

        sample = Path(__file__).parent.parent / "examples" / "sample-server"
        result = CliRunner().invoke(app, ["doctor", "--server", str(sample), "--json"])
        assert result.exit_code == 0
        findings = json.loads(result.output)
        assert isinstance(findings, list) and findings
        assert {"severity", "check", "tools", "message"} <= set(findings[0])
