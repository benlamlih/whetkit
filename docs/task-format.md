# Task file format

Eval tasks are YAML files validated by `mcp_eval.datasets.TaskSpec`
(pydantic). A file contains either one task mapping or a list of task
mappings; `mcp-eval` loads every `*.yaml` / `*.yml` file in the directory you
point it at.

## Example

```yaml
id: update-customer-email
prompt: >
  Change customer CUST-7's email address to jonas.weber@newmail.example and
  then verify the change by fetching the customer record.
server: ../sample-server
expected_tools:
  - cust_upd              # call 1: exactly this tool
  - [get_rec, fetch_record]  # call 2: either of these is acceptable
ordered: true
success_criteria: >
  The final answer confirms that customer CUST-7's email is now
  jonas.weber@newmail.example, based on a fresh read of the record.
tags: [customers, write, ordered]
```

## Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique task id: lowercase alphanumeric plus `-`/`_`, must start alphanumeric. Duplicate ids across a task directory are an error. |
| `prompt` | string | yes | What the user asks the agent to do. |
| `server` | string | yes | The MCP server the task runs against: a URL, a directory (containing `server.json` or `server.py`), or a file path. Relative paths resolve against the task file's directory. A `--server` CLI flag overrides this. |
| `expected_tools` | list | yes | The tool calls a correct run makes. Each entry is one expected call; an entry may itself be a list of acceptable alternatives for that call (any one of them counts). |
| `ordered` | bool | no (default `false`) | When `true`, the expected calls must occur in the listed order (other calls may be interleaved; see scoring docs). |
| `success_criteria` | string | yes | Natural-language rubric the LLM judge uses to grade the agent's final answer. Make it concrete and checkable — name the exact facts a correct answer contains. |
| `tags` | list of strings | no | Free-form labels for filtering and reporting. |

## Writing good tasks

- **One user intent per task.** The prompt should read like something a real
  user would type, not like instructions about which tools to use — the whole
  point is measuring whether the agent finds the right tools itself.
- **Pin the data.** Point tasks at deterministic fixtures (like the bundled
  sample server) so the same run is reproducible and the `success_criteria`
  can reference exact values.
- **Alternatives are for genuine equivalents.** If two tools legitimately
  solve the step, list both. Don't use alternatives to paper over a server
  that has duplicate tools — that duplication is exactly what
  `mcp-eval curate` is for.
