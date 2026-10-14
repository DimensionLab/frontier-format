# Frontier ACP agent: `python -m frontier_agent acp`

_Status: specification, 2026-10-11. Nothing implemented; planned for the
`frontier_agent` package. Decided with the editor-integration decision (section 8 of
that document); the Zed side (the frontier-zed extension) is built and documents the
configuration below._

## 1. Why

Zed's agent panel (and every other ACP client: JetBrains, Neovim, Emacs, VS Code plugins)
drives external agents through the Agent Client Protocol. The Frontier MCP server already
puts Frontier's tools in those panels, but an MCP server cannot ask a person for an
approval: in Zed the server runs with `--approver none` and every write is refused. The
ACP agent closes that gap. It is Frontier's own data-prep harness (`frontier_agent.session`)
speaking ACP: the client renders the conversation, tool calls, previews and diffs; the
agent asks the client for permission exactly where `prep.request_approval` is called today;
policy stays inside the agent; the session record is written as for the TUI, so
`session replay` works on a session that was driven from an editor.

## 2. Facts this specification rests on (verified 2026-10-11)

From agentclientprotocol.com, protocol v1:

- Transport: JSON-RPC 2.0 over stdio; the agent is a subprocess of the client. stdout is the
  protocol; logs go to stderr.
- `protocolVersion` is the integer `1`. `initialize` negotiates it and exchanges
  `clientCapabilities` (`fs.readTextFile`, `fs.writeTextFile`, `terminal`) and
  `agentCapabilities` (`loadSession`, `promptCapabilities.{image,audio,embeddedContext}`,
  `mcpCapabilities.{http,sse}`), plus `clientInfo` / `agentInfo` and `authMethods`.
- Client → agent: `initialize`, `authenticate`, `session/new`, `session/load` (needs
  `loadSession`), `session/prompt`, `session/set_mode`, `session/cancel` (notification),
  `logout`. Agent → client: `session/update` (notification), `session/request_permission`,
  `fs/read_text_file`, `fs/write_text_file`, `terminal/*`, `elicitation/*`.
- `session/new` takes `cwd` (absolute) and `mcpServers` (stdio entries with `name`,
  `command`, `args`, `env`; `http` / `sse` entries only when the agent advertises them).
  Every agent MUST support the stdio form. The response carries a unique `sessionId` and
  may carry `modes` (`currentModeId`, `availableModes[{id,name,description}]`).
- `session/update` variants: `user_message_chunk`, `agent_message_chunk`,
  `agent_thought_chunk`, `tool_call`, `tool_call_update`, `plan`,
  `available_commands_update`, `current_mode_update`.
- Tool calls: `toolCallId`, `title`, optional `name`, `kind` in `read | edit | delete | move
| search | execute | think | fetch | switch_mode | other`, `status` in `pending |
in_progress | completed | failed`, `content[]` of `{type: content}`, `{type: diff, path,
oldText?, newText}` or `{type: terminal, terminalId}`, `locations[{path, line?}]`,
  `rawInput`, `rawOutput`. Updates carry only the fields that change.
- `session/request_permission`: `sessionId`, `toolCall{toolCallId}`, `options[{optionId,
name, kind}]` with `kind` in `allow_once | allow_always | reject_once | reject_always`;
  the response is `{outcome: selected, optionId}` or `{outcome: cancelled}` (mandatory after
  `session/cancel`).
- Plan: `{sessionUpdate: plan, entries[{content, priority: high|medium|low, status:
pending|in_progress|completed}]}`; every update carries the full list.
- `session/prompt` ends with `stopReason` in `end_turn | max_tokens | max_turn_requests |
refusal | cancelled`. After `session/cancel` the agent MUST answer the pending prompt
  with `cancelled`.
- Content blocks: `text`, `image` (base64 `data`, `mimeType`), `audio`, `resource_link`
  (`uri`, `name`, ...), `resource` (embedded: `uri`, `mimeType?`, `text` or `blob`).
- `session/load` replays the whole conversation as `session/update` notifications, then
  answers with an empty result.
- Extensibility: custom methods and notifications are prefixed with `_` (the Python SDK
  routes them to `ext_method` / `ext_notification`); `_meta` fields are free.
- Authentication: `authMethods[]` entries have `id`, `name`, `description` and an optional
  `type` (`agent`, the default, or `terminal`); an empty list means no login.

Python SDK: `agent-client-protocol` on PyPI (0.12.1 stable 2026-08-16; 1.0.0rc2 on
2026-09-21), Apache-2.0, Python ≥ 3.10. `acp.Agent` is the base class with
`initialize`, `authenticate`, `new_session`, `load_session`, `set_session_mode`, `prompt`,
`cancel`, `ext_method`, `ext_notification`; `on_connect(conn)` hands over the client
connection (`conn.session_update(session_id, update)`, `conn.request_permission(...)`);
`acp.run_agent(agent)` serves stdio; `acp.schema` holds the generated pydantic models;
`acp.helpers` builds content blocks, tool calls and permission requests.

Zed: agents outside the ACP Registry are configured in `settings.json` under
`agent_servers.<name>` with `type: "custom"`, `command`, `args`, `env`; Zed-configured MCP
servers may be forwarded to the agent over ACP; `dev: open acp logs` shows the traffic;
extension-provided agents are deprecated since Zed 1.5 in favour of the registry.

ACP Registry (`agentclientprotocol/registry`): one directory per agent with `agent.json`
(`id`, `name`, `version`, `description`, `license`, `license_url`, `repository`,
`distribution` with at least one of `binary | npx | uvx`) and a 16×16 monochrome
`icon.svg`; package versions must exist on PyPI and match `version`; CI validates by pull
request.

## 3. Process

```
python -m frontier_agent acp [--datasets-dir D] [--record-dir DIR] [--model M] [--harness builtin]
```

- Same resolution rules as `python -m frontier_agent mcp` for datasets (`--datasets-dir`,
  `FRONTIER_DATASETS_DIR`, `./datasets` under the session `cwd`) and policy
  (`frontier.toml` chain, `FRONTIER_POLICY__AI_*`).
- Records: one `<session-id>.jsonl` per ACP session in `--record-dir` (default
  `policy.ai.session_dir`, `~/.frontier/agent-sessions`), the existing `SessionRecord`
  format. The ACP `sessionId` **is** the record id, so `session show|replay <id>` works on
  it.
- Logging: `tracing`-style structured logs on stderr only. Never print to stdout.
- The harness is `BuiltinLoop` (the Python tool-calling loop) by default; the model comes
  from `policy.ai` (`local_model` / `local_endpoint`, or `hosted_provider` /
  `hosted_model` only when `data_leaves_site` allows). `--harness pi` is not offered over
  ACP: Pi has its own ACP adapter.
- Dependency: `agent-client-protocol>=0.12,<1` in the `frontier-agent` package manifest,
  optional extra `acp` so the MCP server does not pull it. Move to 1.x when it is released
  (the 1.0 release candidates are already on PyPI; the method surface above is the same).

## 4. `initialize`

Response:

```json
{
  "protocolVersion": 1,
  "agentCapabilities": {
    "loadSession": true,
    "promptCapabilities": {
      "image": false,
      "audio": false,
      "embeddedContext": true
    },
    "mcpCapabilities": { "http": false, "sse": false }
  },
  "agentInfo": {
    "name": "frontier-agent",
    "title": "Frontier data-prep agent",
    "version": "<package version>"
  },
  "authMethods": []
}
```

- `protocolVersion` lower than 1 from the client: answer `1` anyway (the spec lets the
  agent name the version it supports; the client decides).
- `embeddedContext: true` because the client may attach a recipe, a dataset card or a
  `frontier.toml` as an embedded resource; the agent passes text resources to the model as
  context. Images and audio are refused with `-32602`.
- `authMethods: []`: Frontier credentials (`frontier auth login`) and policy come from the
  configuration chain, not from the client. If the registry's validator requires a method
  (its CONTRIBUTING mentions authentication checks), advertise one `terminal` method with
  `args: ["auth", "login"]`, which runs `frontier auth login` in the client's terminal.
- Store `clientCapabilities`; `fs.*` and `terminal` are not used in v1 (nothing the agent
  does needs the client's file system or terminal: datasets live outside the worktree and
  recipes are returned as diffs for the client to apply).

## 5. Sessions

### `session/new`

- `cwd`: the project root. Relative dataset names resolve against `<cwd>/datasets` unless
  `--datasets-dir` is given. Store `cwd` in the record's `session_start` event.
- `mcpServers`: accepted and recorded (names and commands, env values redacted). v1 does
  not connect to them: the agent's tools are Frontier's own `ToolRegistry`, and the
  judge backend is chosen by policy, never by a client-supplied server. v2 may connect to
  stdio servers whose `name` is on an allow-list in `policy.ai` to give the model extra
  read-only tools; until then a non-empty list produces one `agent_message_chunk` saying
  the servers are ignored.
- Response: `{"sessionId": "<record id>", "modes": {...}}` with the modes of section 7.
- One `SessionRecord`, one `ToolRegistry` (with the task-specific tool filter off: the user
  names the task in the prompt), one `AiPolicy` snapshot per session. Sessions are
  independent; a process serves many.

### `session/load`

- Supported (`loadSession: true`). Open `<record-dir>/<sessionId>.jsonl`; refuse with
  `-32602` when it does not exist.
- Replay in order: `message` events as `user_message_chunk` / `agent_message_chunk`;
  `tool_call` + `tool_result` as `tool_call` (status `completed` or `failed`) with the
  recorded `rawInput` / `rawOutput`; `approval_request` + `approval` as a `tool_call` of
  kind `other` titled "Approval: <summary>" with status `completed` and the decision as
  text content; `preview` as content on its tool call; then an empty result. Replay never
  re-asks for permission and never re-runs anything.
- After load, new prompts continue the same record (append) and the same dataset head.

### `session/cancel`

- Set the session's cancel flag; the loop checks it before every model request and tool
  call. A pending `session/request_permission` is answered by the client with `cancelled`;
  the agent records `approval {granted: false, note: "cancelled"}` and does not run the
  op. The prompt returns `stopReason: cancelled`. Tool calls in flight get a final
  `tool_call_update` with `status: failed` (`prep` ops are atomic: a cancelled run leaves no
  new version).

## 6. Prompt turn

`session/prompt` → the harness runs one task with the prompt text (plus embedded text
resources) as the user message, streaming the record as it is written:

| Record event (`frontier_agent.record.EVENT_KINDS`) | ACP                                                                                                                                                                                                                                                                         |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `message` (assistant text)                         | `agent_message_chunk` per chunk; model "thinking" (when the backend exposes it) as `agent_thought_chunk`                                                                                                                                                                    |
| `tool_call`                                        | `tool_call` with `toolCallId` = record seq, `name` = tool name, `title` from the table below, `status: in_progress`, `rawInput` = args                                                                                                                                      |
| `tool_result`                                      | `tool_call_update` `status: completed`, `rawOutput` = result (redacted), `content`: a text block with the server's human summary when the tool has one                                                                                                                      |
| `preview`                                          | `tool_call_update` on the `<op>.preview` call: text block with the before/after stats; each render as `{type: content, content: {type: image, data, mimeType: "image/png"}}` (renders are files the preview wrote; read and base64 them; cap at 4 renders)                  |
| `judgment`                                         | `tool_call_update` on the `judge.*` call: text content with choice, confidence and backend                                                                                                                                                                                  |
| `approval_request`                                 | `session/request_permission` (section 8)                                                                                                                                                                                                                                    |
| `approval`                                         | `tool_call_update` on the approval tool call: `completed`, text "approved by <approver>" or "refused"                                                                                                                                                                       |
| `op_run`                                           | `tool_call_update` on the `<op>.run` call: `completed`, text with input → output version; and when the session records a recipe (`--record` of `frontier prep`), a `{type: diff, path: <cwd-relative recipe path>, oldText, newText}` so the client shows the recipe change |
| `assertion`                                        | text content on the current tool call                                                                                                                                                                                                                                       |
| `error`                                            | `tool_call_update` `status: failed` with the message; policy refusals end the turn with `refusal`                                                                                                                                                                           |
| `note`                                             | `agent_message_chunk`                                                                                                                                                                                                                                                       |
| `session_end`                                      | the prompt response                                                                                                                                                                                                                                                         |

Tool call `kind` and `title` by tool name:

| Tool                                                                                                                                                                             | kind      | title                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | --------------------------------------------------------- |
| `dataset.list`, `dataset.describe`, `dataset.sample_stats`, `version.log`, `version.show`, `version.diff`, `quality.report`, `quality.check_leakage`, `prep.explain`, `prep.ops` | `read`    | "Describe <dataset>", "Diff <a>..<b>", ...                |
| `judge.*`                                                                                                                                                                        | `think`   | "Judge <family>: <question>"                              |
| `<op>.plan`                                                                                                                                                                      | `other`   | "Plan <op> on <dataset>"                                  |
| `<op>.preview`                                                                                                                                                                   | `execute` | "Preview <op> on <n> samples"                             |
| `<op>.run`, `version.revert`                                                                                                                                                     | `edit`    | "Run <op> on <dataset>" / "Revert <dataset> to <version>" |
| `prep.request_approval`                                                                                                                                                          | `other`   | "Approval: <summary>"                                     |

`locations` is set on `edit` calls when the dataset lives under `cwd`
(`{path: datasets/<name>/provenance.jsonl}`), so the client can jump to the provenance.

Plan: after the first `<op>.plan` of a turn, and after every `op_run`, send
`{sessionUpdate: plan}` listing the ops the model has planned in this turn
(`in_progress` for the one being previewed or awaiting approval, `completed` after
`op_run`, `pending` otherwise; priority `high` for ops with `requires_approval`,
`medium` otherwise).

Stop reasons: `end_turn` when the loop finishes; `max_turn_requests` when
`policy.ai.max_tool_calls` is reached (the record gets `session_end {reason:
"budget"}`); `refusal` when policy refuses a provider or an op class, or when the model
attempted a write without a token three times (the registry's existing guard); `cancelled`
per section 5. `max_tokens` is passed through from the backend.

## 7. Modes

```json
{
  "currentModeId": "diagnose",
  "availableModes": [
    {
      "id": "diagnose",
      "name": "Diagnose (read-only)",
      "description": "Describe, sample, judge, check quality and leakage. No op can run."
    },
    {
      "id": "prep",
      "name": "Prep with approvals",
      "description": "Plan, preview, then ask for approval before each op runs."
    },
    {
      "id": "autonomous",
      "name": "Autonomous within policy",
      "description": "Ops in policy.ai.autonomous_op_classes run without asking; everything else still asks."
    }
  ]
}
```

- `diagnose`: the registry gets the read-only tool filter (`evalset.task_tool_filter` with
  no ops); `prep.request_approval` and `<op>.run` are not offered to the model.
- `prep`: the default behaviour of the TUI harness.
- `autonomous`: only listed when `policy.ai.autonomous_op_classes` is non-empty; otherwise
  `availableModes` has two entries. Switching to it never widens policy: it only stops
  asking for the classes policy already allows.
- `session/set_mode` applies at the next tool call; the agent confirms with
  `current_mode_update`. The record gets a `note {mode}` event.

## 8. Approvals (the point of the exercise)

When the model calls `prep.request_approval(plan_id, summary)`:

1. The registry checks, as today, that the plan exists and was previewed; a failing check
   is a tool error, no permission request.
2. The agent sends `session/request_permission` with `toolCall.toolCallId` = the approval
   call's id and

   ```json
   "options": [
     {"optionId": "approve", "name": "Approve this op", "kind": "allow_once"},
     {"optionId": "approve_class", "name": "Approve every <class> op this session", "kind": "allow_always"},
     {"optionId": "refuse", "name": "Refuse", "kind": "reject_once"},
     {"optionId": "refuse_all", "name": "Refuse and stop proposing writes", "kind": "reject_always"}
   ]
   ```

   `approve_class` is included only when the op's class is in
   `policy.ai.autonomous_op_classes` or the mode is `autonomous`: a client cannot grant
   per-session autonomy the site policy forbids. The tool call carries the summary, the
   plan (op, params, warnings) and the preview numbers as text content, so the person sees
   what the TUI dialog shows.

3. Outcome handling:
   - `selected: approve` → `catalog.approvals.grant(plan_id, approver=<clientInfo.name>
user)`, record `approval {granted: true}`; the token is returned to the model as the
     tool result, exactly as for `--approver ui`.
   - `selected: approve_class` → grant, and remember the class for the session (no further
     requests for it; each later run still records an `approval` with `approver:
"session-grant"`).
   - `selected: refuse` → record `approval {granted: false}`; the tool result tells the
     model the person refused and that it must not retry the same plan.
   - `selected: refuse_all` → as refuse, plus the registry drops `prep.request_approval`
     and every `<op>.run` for the rest of the session (the model sees them gone).
   - `cancelled` → section 5.
4. The approval token is minted only after `selected: approve*`; nothing else can produce
   one. The model never sees the permission request, only the tool result.

`version.revert` asks the same way with "Revert <dataset> to <version>".

## 9. Policy

Unchanged and enforced inside the agent: `policy.ai.data_leaves_site`,
`allowed_providers`, `autonomous_op_classes`, `max_tool_calls`, `judge_backend`. The
client chooses nothing about models or providers. Secrets are redacted in records and in
every `rawInput` / `rawOutput` (`credentials.redact`, as today). A prompt that asks the
agent to change policy is answered with the policy summary (`AiPolicy.summary()`), never
acted on.

## 10. Extension methods

- `_frontier/session/record` (request, `{sessionId}`) → `{path}`: where the record is, for
  the client's "Open record" action.
- `_frontier/session/summary` (request, `{sessionId}`) → `SessionRecord.summary()`.
- Both are optional for clients; unknown `_frontier/*` methods get `-32601`.

## 11. Errors

JSON-RPC standard codes: `-32700` parse, `-32600` invalid request, `-32601` unknown method,
`-32602` invalid params (unknown session, unsupported content type, unknown mode), `-32603`
internal (a traceback is logged to stderr; the message is short and secret-free). Tool-level
failures are never JSON-RPC errors: they are `tool_call_update {status: failed}` so the turn
continues.

## 12. Zed configuration (until the registry entry exists)

```json
{
  "agent_servers": {
    "Frontier": {
      "type": "custom",
      "command": "/path/to/frontier/.venv/bin/python",
      "args": ["-m", "frontier_agent", "acp"],
      "env": { "FRONTIER_DATASETS_DIR": "/path/to/frontier/datasets" }
    }
  }
}
```

Open the agent panel, pick "Frontier" as the external agent; approvals appear as Zed's
permission prompt; `dev: open acp logs` shows the JSON-RPC traffic.

## 13. Registry submission

`agentclientprotocol/registry` pull request, directory `frontier-agent/`:

```json
{
  "id": "frontier-agent",
  "name": "Frontier data-prep agent",
  "version": "<frontier-agent PyPI version>",
  "description": "Diagnose and clean physics-ML datasets with Frontier's data-prep ops; every write needs your approval.",
  "repository": "https://github.com/DimensionLab/frontier-sdk",
  "license": "Apache-2.0",
  "license_url": "https://github.com/DimensionLab/frontier-sdk/blob/main/LICENSE",
  "distribution": {
    "uvx": { "package": "frontier-agent==<version>", "args": ["acp"] }
  }
}
```

Prerequisites, none of which hold today: `frontier-agent` published on PyPI (it is
currently a private package, `Private :: Do Not Upload`, so this is a licensing and
packaging decision for the founder, not a code change) with a console script named
`frontier-agent` (add `[project.scripts] frontier-agent = "frontier_agent.__main__:main"`;
`uvx` runs the script named after the package), the `acp` extra installed by default or the
dependency made non-optional, and `icon.svg` (16×16, `currentColor` only; the recipe icon of
the frontier-zed extension fits). The alternative the registry accepts is a `binary`
distribution: a PyInstaller-style bundle per platform published as GitHub Releases, which
keeps the package itself private. Until one of these exists, Zed users configure the agent
by hand (section 12).

## 14. Implementation notes for the Python owner

- New module `frontier_agent/acp_server.py`: `class FrontierAcpAgent(acp.Agent)`, one
  `_Session` dataclass per `sessionId` holding `SessionConfig`, `SessionRecord`,
  `ToolRegistry`, `AiPolicy`, mode, cancel flag, class grants, and the `asyncio.Task` of
  the running turn.
- The harness is synchronous today; run `BuiltinLoop.run` in a thread
  (`asyncio.to_thread`) and bridge events with a queue: `SessionRecord.append` gets an
  optional `listener` callback, which the ACP session uses to translate events into
  `session/update` notifications (`conn.session_update`) from the event loop. The approval
  reviewer passed to the loop blocks on a `concurrent.futures.Future` that the
  `session/request_permission` response resolves.
- `__main__.py`: `sub.add_parser("acp")` and dispatch like `mcp`; `argv[0] == "acp"` short
  path so `--help` of the ACP server is its own.
- Tests: the SDK's client side (`acp.Client`, `spawn_agent_process`) drives the server in a
  subprocess over pipes: initialize → new session → prompt with a scripted model
  (`frontier_agent.llm` fake) → assert the update sequence, a permission request for the
  rename op, token minted only after `approve`, record replayable with `session replay`;
  cancel mid-turn → `cancelled`; `diagnose` mode never requests permission; `load_session`
  replays without side effects. The eval tasks `ahmed-refused-approval` and
  `ahmed-rename-fields` get an `acp` harness variant so the same criteria hold.
- Estimate: a week for the server and tests, a day for the registry PR.
