from pathlib import Path

import pytest

from whetkit.datasets import TaskSpec, load_tasks
from whetkit.mcp import MCPClient, resolve_server_spec

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_TASKS = REPO_ROOT / "examples" / "tasks"
SAMPLE_SERVER = REPO_ROOT / "examples" / "sample-server"

VALID = """
id: my-task
prompt: Do the thing.
server: http://localhost:9/mcp
expected_tools:
  - tool_a
  - [tool_b, tool_c]
success_criteria: The thing was done.
"""


def test_valid_task_parses_and_normalizes(tmp_path: Path) -> None:
    file = tmp_path / "task.yaml"
    file.write_text(VALID)
    (task,) = load_tasks(file)
    assert task.id == "my-task"
    assert task.ordered is False
    assert task.tags == []
    assert task.expected_tool_slots == [["tool_a"], ["tool_b", "tool_c"]]


def test_relative_server_resolves_against_task_file(tmp_path: Path) -> None:
    server_dir = tmp_path / "srv"
    server_dir.mkdir()
    (server_dir / "server.py").write_text("# stub\n")
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "t.yaml").write_text(VALID.replace("http://localhost:9/mcp", "../srv"))
    (task,) = load_tasks(tasks_dir)
    assert task.server == str(server_dir.resolve())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("id: My Task", "task id"),
        ("id: -bad", "task id"),
        ("prompt: ''", "prompt"),
        ("expected_tools: []", "expected_tools"),
        ("expected_tools:\n  - []", "alternatives"),
        ("success_criteria: ''", "success_criteria"),
    ],
)
def test_invalid_tasks_rejected(tmp_path: Path, mutation: str, match: str) -> None:
    lines = [
        mutation if line.split(":")[0].strip() == mutation.split(":")[0] else line
        for line in VALID.strip().splitlines()
        if not (
            mutation.startswith("expected_tools") and line.strip().startswith(("- tool", "- ["))
        )
    ]
    file = tmp_path / "task.yaml"
    file.write_text("\n".join(lines))
    with pytest.raises(ValueError, match=match):
        load_tasks(file)


def test_unknown_field_rejected_with_suggestion(tmp_path: Path) -> None:
    file = tmp_path / "task.yaml"
    file.write_text(VALID + "orderd: true\n")  # typo of 'ordered'
    with pytest.raises(ValueError) as exc_info:
        load_tasks(file)
    message = str(exc_info.value)
    assert str(file) in message
    assert "unknown field 'orderd'" in message
    assert "did you mean one of:" in message and "ordered" in message


def test_unknown_field_without_close_match_lists_valid_fields(tmp_path: Path) -> None:
    file = tmp_path / "task.yaml"
    file.write_text(VALID + "zzz_bogus: 1\n")
    with pytest.raises(ValueError, match="unknown field 'zzz_bogus'"):
        load_tasks(file)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(VALID)
    (tmp_path / "b.yaml").write_text(VALID)
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(tmp_path)


def test_list_of_tasks_in_one_file(tmp_path: Path) -> None:
    lines = VALID.strip().replace("id: my-task", "id: t1", 1).splitlines()
    doc = "- " + "\n  ".join(lines)
    file = tmp_path / "many.yaml"
    file.write_text(doc)
    tasks = load_tasks(file)
    assert [t.id for t in tasks] == ["t1"]


def test_missing_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no such file"):
        load_tasks(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .yaml"):
        load_tasks(empty)


def test_example_tasks_load() -> None:
    tasks = load_tasks(EXAMPLES_TASKS)
    assert len(tasks) == 5
    assert all(isinstance(t, TaskSpec) for t in tasks)
    assert all(t.server == str(SAMPLE_SERVER.resolve()) for t in tasks)
    ordered = {t.id: t.ordered for t in tasks}
    assert ordered["update-customer-email"] is True
    assert ordered["find-product"] is False


async def test_sample_server_tools_work() -> None:
    spec = resolve_server_spec(str(SAMPLE_SERVER))
    async with MCPClient(spec) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert len(tools) == 14
        assert {"data_query_1", "proc_ord", "do_thing", "get_rec"} <= names

        search = await client.call_tool("data_query_1", {"q": "wireless mouse"})
        assert "AeroGlide" in search.content[0].text

        order = await client.call_tool(
            "proc_ord", {"customer_id": "CUST-2", "product_id": "P-3", "quantity": 2}
        )
        assert "ORD-1003" in order.content[0].text

        stock = await client.call_tool("inv_check", {"pid": "P-3"})
        assert '"stock": 15' in stock.content[0].text

        # expected tool references in example tasks must actually exist
        for task in load_tasks(EXAMPLES_TASKS):
            for slot in task.expected_tool_slots:
                assert set(slot) <= names, f"{task.id} references unknown tools {slot}"
