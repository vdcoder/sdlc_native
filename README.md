# sdlc_native_multi_agents

**A local, working multi-agent SDLC pipeline: real LLM-driven agents that code, test, and review pull requests — built on native [LangGraph](https://langchain-ai.github.io/langgraph/) primitives.**

Drop a ticket (a small JSON file, optionally pointing at a git repo) into `jira_tickets/`, and an autonomous pipeline clones the repo, writes code to satisfy the requirement, tests its own work, self-heals on failure, packages the result, and runs an independent PR review — each stage a real LLM agent using real filesystem tools, sandboxed to its own workspace.

```
python main.py jira_tickets/my_ticket.json
```

or watch the folder and process tickets as they arrive, fully in parallel:

```
python watcher.py
```

---

## 1. Architecture

```
   START
     │
     ▼
 jira_ingest ──────────────────────────────────────────────┐
     │  (provisions branch, resets registers)               │
     ▼                                                       │
 git_clone   (classic node, no LLM — clones repo_url if set) │
     │                                                       │
     ▼                                                       │
 coder_agent  <───────────────────────────┐                  │
     │           self-healing retry loop  │                  │
     ▼                                    │ loop_count < 3    │
 tester_agent ──[route_validation_gate]───┘                   │
     │            │                                            │
     │ pass        └─ loop_count >= 3 ──┐                       │
     ▼                                  │                       │
 qa_package  <────────────────────────┘                       │
 (packages workspace -> github_PRS/<ticket>/{repo,QA,PR_review})
     │
     ▼
 pr_review_agent ─────────────────────────────────────────────► END
```

Any of `coder_agent` / `tester_agent` / `pr_review_agent` can call an `ask_human` tool mid-loop — a real, blocking terminal prompt — if it genuinely needs a human's input. There's no separate hardcoded approval gate; a human is pulled in only when an agent actually asks.

### Centralized state (`SDLCState`)

Every node is a pure function `SDLCState -> dict`, returning only the fields it changes; LangGraph merges the update into the checkpointed state. Concurrent tickets are isolated by `thread_id`, and every state transition is persisted by the `MemorySaver` checkpointer — `app.get_state_history(config)` gives a full, replayable audit trail of a ticket's lifecycle (printed as an ASCII table at the end of each run).

### Nodes

| Node | Responsibility |
|---|---|
| `jira_ingest` | Mocks the inbound Jira webhook, provisions a branch name, resets all working registers |
| `git_clone` | Classic node (no LLM). Clones `repo_url` into a curated per-ticket workspace, or falls back to a scratch folder if the ticket has none |
| `coder_agent` | Real LLM ReAct agent. Explores the workspace and writes code/files to satisfy `requirements` |
| `tester_agent` | Real LLM ReAct agent. Reviews the workspace, may patch small issues itself, and renders a pass/fail verdict with a stated reason |
| `qa_package` | Classic node (no LLM). Copies the workspace into `github_PRS/<ticket>/repo`, creates sibling `QA/` and `PR_review/` folders, writes `QA_results.md` from the tester's verdict |
| `pr_review_agent` | A second, independent LLM ReAct agent (read-only). Reviews the packaged repo copy as a pull request and writes `PR_review_results.md` with an approve/reject verdict |

### Control routine

```python
def route_validation_gate(state: SDLCState) -> Literal["coder_agent", "qa_package"]:
    if state["test_passed"] or state["loop_count"] >= MAX_LOOP_ATTEMPTS:
        return "qa_package"   # ship it for review either way — win or budget exhausted
    return "coder_agent"      # self-healing retry
```

`MAX_LOOP_ATTEMPTS` (default 3) is the token-cost circuit breaker: `coder_agent` ↔ `tester_agent` cannot loop forever. Once the budget is exhausted, the ticket still gets packaged and reviewed — the human/reviewer sees exactly what was attempted and why it didn't pass, instead of the pipeline silently failing.

---

## 2. Ticket files

A ticket is a small JSON file:

```json
{
  "jira_ticket_id": "JIRA-4521",
  "requirements": "Explore this repository and add a summary.txt describing what it contains.",
  "repo_url": "https://github.com/octocat/Hello-World.git"
}
```

- `jira_ticket_id` and `requirements` are required.
- `repo_url` is optional — omit it and `coder_agent` gets a plain scratch folder instead of a cloned repo.

### Where things end up

| Path | Contents |
|---|---|
| `jira_tickets/` | Inbox — drop new ticket files here (watched by `watcher.py`) |
| `jira_tickets_done/` | Ticket files moved here (unmodified) once processing completes |
| `workspace/<ticket>/` | The cloned repo (or scratch dir) `coder_agent`/`tester_agent` actually edit |
| `github_PRS/<ticket>/repo/` | A packaged copy of the workspace, frozen at review time |
| `github_PRS/<ticket>/QA/QA_results.md` | The tester's verdict + reasoning (and any self-patches it made) |
| `github_PRS/<ticket>/PR_review/PR_review_results.md` | The PR reviewer's verdict + reasoning |

`workspace/`, `github_PRS/`, and `jira_tickets_done/` are all generated at runtime and gitignored.

---

## 3. Agents & tools

`coder_agent` and `tester_agent` share the same tools, each running in a bounded ReAct loop (ask the model for one JSON tool call, execute it for real, feed the result back, repeat):

- `list_file_names` / `read_file` / `write_file` — real filesystem operations
- `ask_human` — a genuine blocking terminal prompt for when an agent needs a person
- `finish` (coder) / `finish` with `passed`+`reason` (tester) — ends the loop with a verdict

`pr_review_agent` is read-only (`list_file_names` / `read_file` / `ask_human`) and ends with `approve_pr` or `reject_pr`, each carrying a `reason`.

**Sandboxing:** every tool call is resolved through `make_sandboxed_tools(root)`, which confines all paths to the agent's intended root (the ticket's `workspace/` dir, or the packaged `github_PRS/.../repo` for the reviewer) — regardless of what path string the model actually passes in (`"."`, an absolute path, `../..`, etc.). An agent cannot read or write outside the folder it was scoped to.

Small local models occasionally emit slightly malformed JSON (e.g. `{"tool": "finish": {}}` instead of `{"tool": "finish", "args": {}}`); `parse_tool_call` tolerates this, and a genuinely unparseable reply gets one corrective nudge before falling back, rather than silently giving up.

---

## 4. LLM-agnostic model layer

Every agent is written against LangChain's `BaseChatModel` interface via `get_llm()`. Swapping providers is a `.env` change, not a code change:

```bash
LLM_PROVIDER=ollama     # default, local/offline, no API key — LLM_MODEL=qwen2.5-coder
LLM_PROVIDER=openai     # requires OPENAI_API_KEY   — LLM_MODEL=gpt-4o-mini
LLM_PROVIDER=gemini     # requires GOOGLE_API_KEY
LLM_PROVIDER=anthropic  # requires ANTHROPIC_API_KEY
```

Copy `.env.example` to `.env` and fill in the provider/model/key you want to use.

---

## 5. External agent delegation

Any node can outsource its work entirely to an external service instead of calling an LLM locally — useful for heavier compute, specialized models, or hardware a different machine owns. Set the matching env var to a URL:

```bash
EXTERNAL_AGENT_CODER=http://localhost:9000/coder
EXTERNAL_AGENT_TESTER=http://localhost:9000/tester
EXTERNAL_AGENT_PR_REVIEW=http://localhost:9000/pr-review
```

When set, `send_requirements()` POSTs `{"jira_ticket_id": ..., "requirements": ...}` to that URL and blocks for the response:

```json
{
  "tool_calls": [
    {"tool": "list_file_names", "args": {"directory": "."}},
    {"tool": "read_file", "args": {"path": "README"}},
    {"tool": "write_file", "args": {"path": "summary.txt", "content": "..."}},
    {"tool": "finish", "args": {}}
  ]
}
```

The node replays that ordered list of tool calls through its own sandboxed tools — the external agent decides *what* to do, but every filesystem effect still runs locally, through the same containment guarantees as the built-in LLM loop.

---

## 6. Running it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

cp .env.example .env   # then fill in your LLM_PROVIDER / API key

python main.py jira_tickets/my_ticket.json
```

Or run the folder watcher to process tickets as they arrive — each new file spawns a fully independent, parallel `python main.py` process:

```bash
python watcher.py
```

---

## 7. Design constraints (by intent, not oversight)

- **No real Jira/GitHub integration.** `jira_ingest` and the branch-provisioning print statements are mocks — this keeps the artifact self-contained and portable. Everything downstream of that (cloning, coding, testing, reviewing) is real.
- **`MemorySaver` checkpointer.** In-memory, per-process. Swapping to a durable backend (Postgres, SQLite) for checkpoints that survive a restart is a one-line change (`from langgraph.checkpoint.postgres import PostgresSaver`), with zero changes to graph topology or node logic.
- **Concurrency is process-level, not in-process.** `watcher.py` spawns one OS process per ticket via `subprocess.Popen` rather than running multiple tickets on one event loop — simple, robust, and trivially parallel across tickets.

---

## 8. Extending this platform

- Swap `MemorySaver` for a durable checkpointer to persist audit trails across restarts.
- Replace `jira_ingest`'s mocked webhook/branch calls with real Jira/GitHub SDK calls behind the same function signature.
- Add more conditional gates (e.g. a security-scan agent with its own bounded retry loop) by composing additional `add_conditional_edges` calls.
- Build a real external agent server against the `send_requirements` protocol to outsource a node to specialized compute.

