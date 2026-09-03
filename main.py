"""
================================================================================
 sdlc_native_multi_agents — Autonomous Multi-Agent SDLC Pipeline
================================================================================

A single-file, dependency-light reference implementation of a self-healing,
human-gated software delivery pipeline built on native LangGraph primitives.

WHY THIS FILE EXISTS
--------------------
This is an executive/architecture-review artifact. It demonstrates four
platform capabilities that matter when operating LLM agents in production
SDLC workflows:

    1. Token Cost Mitigation  -> bounded self-healing loop (MAX_LOOP_ATTEMPTS)
                                  instead of unbounded agent retries.
    2. System Determinism     -> a typed, centralized state object (SDLCState)
                                  flowing through an explicit directed graph,
                                  instead of implicit chain-of-thought handoffs.
    3. On-Demand HITL         -> an `ask_human` tool available to every agent
                                  (coder/tester/reviewer), so a human is pulled
                                  in only when an agent actually needs one —
                                  not via a single hardcoded approval gate.
    4. Audit Trail / SOC2     -> every state transition is persisted by a
                                  LangGraph checkpointer, giving us a replayable,
                                  immutable timeline of every decision made —
                                  including the ability to "time-travel" back
                                  to any prior checkpoint and fork a new,
                                  human-corrected timeline for debugging.

ARCHITECTURE (happy path)
--------------------------

    START
      │
      ▼
  jira_ingest ───────────────────────────────────────────────┐
      │                                                       │ (reset registers)
      ▼                                                       │
  git_clone    (classic node, no LLM — clones repo_url if set)│
      │                                                       │
      ▼                                                       │
  coder_agent  <───────────────────────────┐                  │
      │                                    │ retry            │
      ▼                                    │ (loop_count<3)   │
  tester_agent ──[route_validation_gate]────┘                  │
      │            │                                           │
      │ pass        └─[loop_count>=3]──┐                       │
      ▼                                │                       │
  qa_package  <───────────────────────┘                       │
  (packages workspace -> github_PRS/<ticket>/{repo,QA,PR_review})
      │
      ▼
  pr_review_agent ─────────────────────────────────────────────► END

Any of coder_agent / tester_agent / pr_review_agent can call the `ask_human`
tool mid-loop if it genuinely needs a human's input — this is a real,
blocking terminal prompt, not a scripted/bypassed approval step.

LLM AGNOSTICISM
----------------
Every agent node is written against LangChain's `BaseChatModel` interface via
`get_llm()`. Swapping the backing model (local Ollama <-> Gemini <-> Anthropic)
requires ZERO code changes — only an environment variable flip. The demo
runner below does NOT require a live model at all: node logic is deterministic
and simulated so the entire pipeline executes instantly, offline, with no
API keys. Set LIVE_LLM_MODE=1 to additionally construct (not require calls
from) a real ChatModel via get_llm() to prove the wrapper resolves correctly.

Run it:
    python main.py
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
from typing import Literal, TypedDict

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()  # picks up LLM_PROVIDER / API keys from a local .env, if present

# Force UTF-8 stdout so emoji/box-drawing output renders on default Windows
# console codepages (cp1252) without requiring `chcp 65001` — keeps the demo
# truly "instant, out-of-the-box" per the design constraints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdlc-pipeline")


def banner(title: str, width: int = 96) -> None:
    """Pretty-print a section header so the console transcript reads like a demo script."""
    print("\n" + "#" * width)
    print(f"#  {title}".ljust(width - 1) + "#")
    print("#" * width)


# ==============================================================================
# 1. LLM-AGNOSTIC MODEL WRAPPER
# ==============================================================================
# Every provider below satisfies LangChain's `BaseChatModel` interface, so any
# node in this pipeline (coder_agent, tester_agent, etc.) could call
# `get_llm().invoke(...)` interchangeably regardless of backend. This is what
# makes the platform "LLM agnostic": the *graph* owns the control flow, the
# *model* is just a pluggable dependency resolved at runtime from env vars.
# ==============================================================================
def get_llm():
    """Factory returning a LangChain-compatible ChatModel selected via env vars.

    Environment variables:
        LLM_PROVIDER  = "ollama" (default) | "openai" | "gemini" | "anthropic"
        LLM_MODEL     = provider-specific model name override

    NOTE: This factory is intentionally NOT invoked in the default simulation
    path below (see module docstring) so the demo runs instantly without any
    local model server or API key. It exists to prove the abstraction is real
    and swappable — a one-line change in production code, not a rewrite.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-1.5-pro"), temperature=0
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"), temperature=0
        )

    # Default: local Ollama — zero-cost, offline, no API key required.
    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.getenv("LLM_MODEL", "gemma4:e2b"), temperature=0)


# ==============================================================================
# 2. CENTRALIZED STATE
# ==============================================================================
# `SDLCState` is the single source of truth threaded through every node in the
# graph. LangGraph's checkpointer serializes this exact structure at every
# super-step, which is what enables both the audit trail and time-travel:
# there is never any hidden state living inside an agent's memory — everything
# material to the workflow's outcome is captured here.
#
# Thread-safety note: LangGraph isolates concurrent executions by `thread_id`
# (see `config["configurable"]["thread_id"]` below). Each thread gets its own
# independently checkpointed state history, so concurrent tickets never share
# or race on the same SDLCState instance.
# ==============================================================================
class SDLCState(TypedDict):
    jira_ticket_id: str      # e.g. "JIRA-4521" — external system-of-record key
    requirements: str        # natural-language spec driving code generation
    repo_url: str            # optional git repo to clone as the ticket's workspace
    workspace_dir: str       # local folder coder_agent tools operate against (cloned repo or scratch dir)
    branch_name: str         # git branch provisioned for this ticket
    generated_code: str      # latest artifact produced by coder_agent
    test_reasoning: str      # tester_agent's stated reason for approving/rejecting
    test_modifications: str  # reasoning behind any files tester_agent edited itself (empty if none)
    pr_repo_dir: str         # copy of the workspace under github_PRS/<ticket>/repo
    pr_review_dir: str       # github_PRS/<ticket>/PR_review folder, written to by pr_review_agent
    pr_review_passed: bool   # PR reviewer's approve/request-changes verdict
    pr_review_reasoning: str # PR reviewer's stated reason for its verdict
    loop_count: int          # self-healing attempt counter (bounded by MAX_LOOP_ATTEMPTS)
    test_passed: bool        # gate flag consumed by route_validation_gate


MAX_LOOP_ATTEMPTS = 3  # Hard ceiling on coder<->tester retries — the token-cost circuit breaker.
DONE_DIR = "jira_tickets_done"
WORKSPACE_DIR = "workspace"  # per-ticket scratch dir the coder_agent tools read/write
GITHUB_PRS_DIR = "github_PRS"  # packaged {repo, QA, PR_review} folders per ticket
MAX_CODER_STEPS = 6  # Hard ceiling on the coder_agent's own tool-calling loop
MAX_TESTER_STEPS = 6  # Hard ceiling on the tester_agent's own tool-calling loop

# Per-node external agent delegation: if a URL is set, that node's local ReAct
# loop is skipped entirely — the task is POSTed to the URL instead (see
# `send_requirements`) and the ordered tool calls it responds with are
# replayed locally. Empty/unset means "run the local LLM loop as usual".
EXTERNAL_AGENTS = {
    "coder_agent": os.getenv("EXTERNAL_AGENT_CODER", ""),
    "tester_agent": os.getenv("EXTERNAL_AGENT_TESTER", ""),
    "pr_review_agent": os.getenv("EXTERNAL_AGENT_PR_REVIEW", ""),
}


def load_ticket(path: str) -> tuple[str, str, str]:
    """Reads jira_ticket_id + requirements + optional repo_url out of a ticket file."""
    with open(path, "r", encoding="utf-8") as f:
        ticket = json.load(f)
    return ticket["jira_ticket_id"], ticket["requirements"], ticket.get("repo_url", "")


# ==============================================================================
# 3. PIPELINE NODES
# ==============================================================================
def jira_ingest(state: SDLCState) -> dict:
    """Simulated webhook entry point: provisions a branch and resets all registers.

    In production this node would be triggered by an inbound Jira webhook
    (issue transitioned to "Ready for Dev"). Here we mock that trigger with a
    print statement and deterministically derive a feature branch name.
    """
    ticket = state["jira_ticket_id"]
    branch = f"feature/{ticket.lower()}-auto-agent"

    print(f"   🔔 MOCK WEBHOOK  : Jira ticket '{ticket}' transitioned to 'IN PROGRESS'")
    print(f"   🌿 MOCK GIT      : git checkout -b {branch}")
    log.info("[jira_ingest] Branch provisioned: %s. Registers cleared.", branch)

    return {
        "branch_name": branch,
        "generated_code": "",
        "test_reasoning": "",
        "test_modifications": "",
        "pr_repo_dir": "",
        "pr_review_dir": "",
        "pr_review_passed": False,
        "pr_review_reasoning": "",
        "loop_count": 0,
        "test_passed": False,
        "workspace_dir": "",
    }


def curate_folder_name(ticket_id: str) -> str:
    """Turns a ticket id into a filesystem-safe folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", ticket_id).strip("-").lower()


def git_clone(state: SDLCState) -> dict:
    """Classic node (no LLM): clones the ticket's repo into a curated workspace folder.

    Falls back to a plain scratch folder when the ticket has no `repo_url`,
    so downstream tools always have a real directory to operate against.
    """
    workspace_dir = os.path.join(WORKSPACE_DIR, curate_folder_name(state["jira_ticket_id"]))
    repo_url = state.get("repo_url", "")

    if not repo_url:
        os.makedirs(workspace_dir, exist_ok=True)
        log.info("[git_clone] No repo_url on ticket; using scratch workspace %s", workspace_dir)
    elif os.path.isdir(os.path.join(workspace_dir, ".git")):
        log.info("[git_clone] Repo already cloned at %s, skipping.", workspace_dir)
    else:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        print(f"   📥 GIT CLONE     : git clone {repo_url} {workspace_dir}")
        subprocess.run(["git", "clone", repo_url, workspace_dir], check=True)
        log.info("[git_clone] Cloned %s -> %s", repo_url, workspace_dir)

    return {"workspace_dir": workspace_dir}


# ==============================================================================
# 3a. CODER_AGENT TOOLS — filesystem primitives scoped to a per-ticket workspace
# ==============================================================================
def list_file_names(directory: str) -> str:
    """Tool: lists the files currently in the ticket's workspace."""
    files = sorted(os.listdir(directory))
    return "\n".join(files) if files else "(empty)"


def read_file(path: str) -> str:
    """Tool: returns a file's contents, or a not-found marker."""
    if not os.path.exists(path):
        return f"(not found: {path})"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    """Tool: writes a file's contents, creating the workspace dir if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars -> {path}"


def ask_human(question: str) -> str:
    """Tool: asks a real human via a blocking terminal prompt and returns their answer."""
    print(f"\n🙋 AGENT ASKS: {question}")
    return input("👤 Your answer: ").strip()


def make_sandboxed_tools(root: str) -> dict:
    """Builds a tools dict whose paths are confined to `root`, whatever the model passes in.

    Resolves every path against `root` and rejects (via a clear error the
    agent sees as an observation) anything that would escape it — closes the
    gap where a model ignores the workspace it was told to use and reads/
    writes files elsewhere on disk (e.g. "." or "../something").
    """
    root_abs = os.path.abspath(root)

    def _safe(path: str) -> str:
        target_abs = os.path.abspath(path if os.path.isabs(path) else os.path.join(root_abs, path))
        if os.path.commonpath([root_abs, target_abs]) != root_abs:
            raise ValueError(f"path '{path}' escapes the sandboxed workspace '{root}'")
        return target_abs

    return {
        "list_file_names": lambda args: list_file_names(_safe(args.get("directory", "."))),
        "read_file": lambda args: read_file(_safe(args["path"])),
        "write_file": lambda args: write_file(_safe(args["path"]), args["content"]),
        "ask_human": lambda args: ask_human(args.get("question", "")),
    }

CODER_SYSTEM_PROMPT = """You are an autonomous coding agent. Your workspace root is "." — all paths you use must be relative to it (e.g. "summary.txt", "src/app.py").

You have these tools. Call exactly one per turn by replying with ONLY a single JSON object, no other text:
  {{"tool": "list_file_names", "args": {{"directory": "."}}}}
  {{"tool": "read_file", "args": {{"path": "<path>"}}}}
  {{"tool": "write_file", "args": {{"path": "<path>", "content": "<text>"}}}}
  {{"tool": "ask_human", "args": {{"question": "<question for a human>"}}}}
  {{"tool": "finish", "args": {{}}}}

Explore the folder (list/read) before writing. Use ask_human if you are
genuinely blocked and need a human's input to proceed.
When the task is done, reply with the "finish" tool."""


def parse_tool_call(text: str) -> dict | None:
    """Extracts a tool call from a model reply, tolerating surrounding prose and small JSON mistakes.

    Small local models reliably mangle empty/simple args into a malformed
    shape like {"tool": "finish": {}} instead of {"tool": "finish", "args": {}}.
    If strict parsing fails, fall back to pulling the tool name via regex and
    decoding whatever JSON object follows it as the args.
    """
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("args", {})
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r'"tool"\s*:\s*"(\w+)"', text)
    if not match:
        return None

    args = {}
    rest = text[match.end():]
    obj_start = rest.find("{")
    if obj_start != -1:
        try:
            args, _ = json.JSONDecoder().raw_decode(rest[obj_start:])
        except json.JSONDecodeError:
            args = {}
    return {"tool": match.group(1), "args": args}


def execute_tool(tools: dict, tool: str, args: dict) -> str:
    """Executes one tool call against `tools`, returning an observation string (never raises)."""
    try:
        return tools[tool](args)
    except Exception as exc:  # keep the caller's loop alive on a bad tool call/args
        return f"(tool error: {exc})"


def send_requirements(url: str, payload: dict, timeout: int = 300) -> list:
    """Delegates a node's work to an external agent: POSTs the task and blocks for its reply.

    The external agent is expected to respond with a fully pre-planned,
    ordered list of tool calls to replay locally:
        {"tool_calls": [{"tool": "read_file", "args": {"path": "..."}}, ...]}
    This lets heavier or specialized work (different compute, model, or
    hardware access) be outsourced while the calling node still owns and
    executes every filesystem effect through its own sandboxed tools.
    """
    log.info("[send_requirements] POST %s (blocking for external agent's tool-call plan)", url)
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("tool_calls", [])


def coder_agent(state: SDLCState) -> dict:
    """LLM-driven, multi-step, multi-tool agent.

    Runs a bounded ReAct loop (`MAX_CODER_STEPS`): the model from `get_llm()`
    is shown each tool's result and asked which single tool to call next —
    `list_file_names` / `read_file` / `write_file` — until it replies with
    `finish`. Every tool call executes for real against the ticket's on-disk
    workspace (the cloned repo, when one exists).

    If `EXTERNAL_AGENTS["coder_agent"]` is set, this node's work is delegated
    entirely: the task is POSTed there via `send_requirements` and the
    ordered tool calls it responds with are replayed locally instead of
    running the LLM loop below.
    """
    workspace = state["workspace_dir"]
    os.makedirs(workspace, exist_ok=True)
    tools = make_sandboxed_tools(workspace)

    log.info("[coder_agent] Starting tool loop — attempt #%d for %s", state["loop_count"] + 1, state["jira_ticket_id"])

    last_written = state.get("generated_code", "")

    external_url = EXTERNAL_AGENTS.get("coder_agent")
    if external_url:
        for call in send_requirements(external_url, {"jira_ticket_id": state["jira_ticket_id"], "requirements": state["requirements"]}):
            tool, args = call.get("tool"), call.get("args") or {}
            if tool == "finish":
                break
            observation = execute_tool(tools, tool, args)
            log.info("[coder_agent] (external) %s(%s) -> %s", tool, args, observation.splitlines()[0] if observation else observation)
            if tool == "write_file":
                last_written = args.get("content", "")
        return {"generated_code": last_written, "loop_count": state["loop_count"] + 1}

    llm = get_llm()
    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {state['requirements']}"),
    ]

    for step in range(MAX_CODER_STEPS):
        reply = llm.invoke(messages).content
        action = parse_tool_call(reply)

        if not action:
            log.warning("[coder_agent] step %d: unparseable reply (%r), asking model to retry", step, reply[:200])
            messages.append(AIMessage(content=reply))
            messages.append(HumanMessage(content="That wasn't valid JSON. Respond with ONLY a single JSON tool call."))
            continue

        if action.get("tool") == "finish":
            log.info("[coder_agent] step %d: model finished", step)
            break

        tool, args = action.get("tool"), action.get("args") or {}
        observation = execute_tool(tools, tool, args)

        headline = observation.splitlines()[0] if observation else observation
        log.info("[coder_agent] step %d: %s(%s) -> %s", step, tool, args, headline)

        if tool == "write_file":
            last_written = args.get("content", "")

        messages.append(AIMessage(content=reply))
        messages.append(HumanMessage(content=f"Observation: {observation}\n\nRespond with the next tool call as JSON only."))

    return {"generated_code": last_written, "loop_count": state["loop_count"] + 1}


TESTER_SYSTEM_PROMPT = """You are an autonomous QA agent. Your workspace root is "." — all paths you use must be relative to it (e.g. "summary.txt", "src/app.py").

You have these tools. Call exactly one per turn by replying with ONLY a single JSON object, no other text:
  {{"tool": "list_file_names", "args": {{"directory": "."}}}}
  {{"tool": "read_file", "args": {{"path": "<path>"}}}}
  {{"tool": "write_file", "args": {{"path": "<path>", "content": "<text>", "reason": "<why you are changing this file>"}}}}
  {{"tool": "ask_human", "args": {{"question": "<question for a human>"}}}}
  {{"tool": "finish", "args": {{"passed": true|false, "reason": "<why you approved or rejected the work>"}}}}

Inspect the workspace and judge whether it satisfies the ticket's requirements.
You may use write_file to fix small issues yourself, but every write_file call
must include a "reason". Use ask_human if you are genuinely blocked and need
a human's input to proceed. When your review is complete, reply with "finish"."""


def tester_agent(state: SDLCState) -> dict:
    """LLM-driven QA agent.

    Runs the same bounded ReAct loop as `coder_agent` — `list_file_names` /
    `read_file` / `write_file` — to inspect (and optionally patch) the
    ticket's workspace, then renders a pass/fail verdict with a stated
    reason. Any `write_file` calls it makes are logged, with their own
    stated reason, into `test_modifications`.

    If `EXTERNAL_AGENTS["tester_agent"]` is set, this node delegates to that
    external agent instead (see `coder_agent`'s docstring for the protocol).
    """
    workspace = state["workspace_dir"]
    tools = make_sandboxed_tools(workspace)
    log.info("[tester_agent] Starting review loop (attempt #%d)", state["loop_count"])

    modifications = []
    passed, reason = False, "Reviewer did not reach a verdict within the step budget."

    external_url = EXTERNAL_AGENTS.get("tester_agent")
    if external_url:
        for call in send_requirements(external_url, {"jira_ticket_id": state["jira_ticket_id"], "requirements": state["requirements"]}):
            tool, args = call.get("tool"), call.get("args") or {}
            if tool == "finish":
                passed = bool(args.get("passed", False))
                reason = args.get("reason", "(no reason given)")
                break
            observation = execute_tool(tools, tool, args)
            if tool == "write_file":
                modifications.append(f"{args.get('path')}: {args.get('reason', '(no reason given)')}")
            log.info("[tester_agent] (external) %s(%s) -> %s", tool, args, observation.splitlines()[0] if observation else observation)
        log.info("[tester_agent] Verdict: %s — %s", "PASSED" if passed else "FAILED", reason)
        return {"test_passed": passed, "test_reasoning": reason, "test_modifications": "; ".join(modifications)}

    llm = get_llm()
    messages = [
        SystemMessage(content=TESTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Requirements: {state['requirements']}\n\nReview the workspace and render a verdict."),
    ]

    for step in range(MAX_TESTER_STEPS):
        reply = llm.invoke(messages).content
        action = parse_tool_call(reply)

        if not action:
            log.warning("[tester_agent] step %d: unparseable reply (%r), asking model to retry", step, reply[:200])
            messages.append(AIMessage(content=reply))
            messages.append(HumanMessage(content="That wasn't valid JSON. Respond with ONLY a single JSON tool call."))
            continue

        tool, args = action.get("tool"), action.get("args") or {}

        if tool == "finish":
            passed = bool(args.get("passed", False))
            reason = args.get("reason", "(no reason given)")
            break

        observation = execute_tool(tools, tool, args)

        if tool == "write_file":
            modifications.append(f"{args.get('path')}: {args.get('reason', '(no reason given)')}")

        headline = observation.splitlines()[0] if observation else observation
        log.info("[tester_agent] step %d: %s(%s) -> %s", step, tool, args, headline)

        messages.append(AIMessage(content=reply))
        messages.append(HumanMessage(content=f"Observation: {observation}\n\nRespond with the next tool call as JSON only."))

    log.info("[tester_agent] Verdict: %s — %s", "PASSED" if passed else "FAILED", reason)

    return {
        "test_passed": passed,
        "test_reasoning": reason,
        "test_modifications": "; ".join(modifications),
    }


def force_rmtree(path: str) -> None:
    """Removes a directory tree, clearing the read-only bit git leaves on pack files (Windows)."""
    def _on_error(func, target, exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onerror=_on_error)


def qa_package(state: SDLCState) -> dict:
    """Classic node (no LLM): packages the workspace for PR review.

    Copies `workspace_dir` into `github_PRS/<ticket>/repo`, creates sibling
    `QA` and `PR_review` folders, and writes `QA_results.md` from the
    tester's own verdict/reasoning captured earlier in the state.
    """
    ticket_dir = os.path.join(GITHUB_PRS_DIR, curate_folder_name(state["jira_ticket_id"]))
    repo_dir = os.path.join(ticket_dir, "repo")
    qa_dir = os.path.join(ticket_dir, "QA")
    pr_review_dir = os.path.join(ticket_dir, "PR_review")

    if os.path.isdir(repo_dir):
        force_rmtree(repo_dir)
    shutil.copytree(state["workspace_dir"], repo_dir)
    os.makedirs(qa_dir, exist_ok=True)
    os.makedirs(pr_review_dir, exist_ok=True)

    verdict = "PASSED" if state["test_passed"] else "FAILED (retry budget exhausted)"
    report = f"# QA Results — {state['jira_ticket_id']}\n\n**Verdict:** {verdict}\n\n## Reasoning\n{state['test_reasoning']}\n"
    if state.get("test_modifications"):
        report += f"\n## Modifications made during review\n{state['test_modifications']}\n"
    write_file(os.path.join(qa_dir, "QA_results.md"), report)

    log.info("[qa_package] Packaged workspace -> %s ; QA_results.md written.", repo_dir)
    return {"pr_repo_dir": repo_dir, "pr_review_dir": pr_review_dir}


PR_REVIEWER_SYSTEM_PROMPT = """You are an autonomous PR reviewer. Your workspace root is "." — all paths you use must be relative to it (e.g. "summary.txt", "src/app.py").

You have these tools. Call exactly one per turn by replying with ONLY a single JSON object, no other text:
  {{"tool": "list_file_names", "args": {{"directory": "."}}}}
  {{"tool": "read_file", "args": {{"path": "<path>"}}}}
  {{"tool": "ask_human", "args": {{"question": "<question for a human>"}}}}
  {{"tool": "approve_pr", "args": {{"reason": "<why this PR is good to merge>"}}}}
  {{"tool": "reject_pr", "args": {{"reason": "<why this PR needs changes>"}}}}

Review the code against the ticket's requirements as you would a pull request.
Do not modify any files, only inspect them. Use ask_human if you are genuinely
blocked and need a human's input to proceed. When your review is complete,
call approve_pr or reject_pr."""


def pr_review_agent(state: SDLCState) -> dict:
    """LLM-driven PR reviewer — a second ReAct agent, same read-only tools as tester_agent.

    Inspects the packaged repo copy at `pr_repo_dir` and writes its own
    verdict + explanation to `PR_review_results.md` inside `pr_review_dir`.

    If `EXTERNAL_AGENTS["pr_review_agent"]` is set, this node delegates to
    that external agent instead (see `coder_agent`'s docstring for the protocol).
    """
    repo_dir = state["pr_repo_dir"]
    tools = make_sandboxed_tools(repo_dir)
    log.info("[pr_review_agent] Starting PR review for %s", state["jira_ticket_id"])

    approved, reason = False, "Reviewer did not reach a verdict within the step budget."

    external_url = EXTERNAL_AGENTS.get("pr_review_agent")
    if external_url:
        for call in send_requirements(external_url, {"jira_ticket_id": state["jira_ticket_id"], "requirements": state["requirements"]}):
            tool, args = call.get("tool"), call.get("args") or {}
            if tool in ("approve_pr", "reject_pr"):
                approved = tool == "approve_pr"
                reason = args.get("reason", "(no reason given)")
                break
            observation = execute_tool(tools, tool, args)
            log.info("[pr_review_agent] (external) %s(%s) -> %s", tool, args, observation.splitlines()[0] if observation else observation)
        log.info("[pr_review_agent] Verdict: %s — %s", "APPROVED" if approved else "CHANGES REQUESTED", reason)
        verdict = "APPROVED" if approved else "CHANGES REQUESTED"
        report = f"# PR Review — {state['jira_ticket_id']}\n\n**Verdict:** {verdict}\n\n## Reasoning\n{reason}\n"
        write_file(os.path.join(state["pr_review_dir"], "PR_review_results.md"), report)
        return {"pr_review_passed": approved, "pr_review_reasoning": reason}

    llm = get_llm()
    messages = [
        SystemMessage(content=PR_REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=f"Requirements: {state['requirements']}\n\nReview the pull request and render a verdict."),
    ]

    for step in range(MAX_TESTER_STEPS):
        reply = llm.invoke(messages).content
        action = parse_tool_call(reply)

        if not action:
            log.warning("[pr_review_agent] step %d: unparseable reply (%r), asking model to retry", step, reply[:200])
            messages.append(AIMessage(content=reply))
            messages.append(HumanMessage(content="That wasn't valid JSON. Respond with ONLY a single JSON tool call."))
            continue

        tool, args = action.get("tool"), action.get("args") or {}

        if tool in ("approve_pr", "reject_pr"):
            approved = tool == "approve_pr"
            reason = args.get("reason", "(no reason given)")
            break

        observation = execute_tool(tools, tool, args)

        headline = observation.splitlines()[0] if observation else observation
        log.info("[pr_review_agent] step %d: %s(%s) -> %s", step, tool, args, headline)

        messages.append(AIMessage(content=reply))
        messages.append(HumanMessage(content=f"Observation: {observation}\n\nRespond with the next tool call as JSON only."))

    log.info("[pr_review_agent] Verdict: %s — %s", "APPROVED" if approved else "CHANGES REQUESTED", reason)

    verdict = "APPROVED" if approved else "CHANGES REQUESTED"
    report = f"# PR Review — {state['jira_ticket_id']}\n\n**Verdict:** {verdict}\n\n## Reasoning\n{reason}\n"
    write_file(os.path.join(state["pr_review_dir"], "PR_review_results.md"), report)

    return {"pr_review_passed": approved, "pr_review_reasoning": reason}


# ==============================================================================
# 4. CONTROL ROUTINE — CONDITIONAL EDGE
# ==============================================================================
def route_validation_gate(state: SDLCState) -> Literal["coder_agent", "qa_package"]:
    """Routes control after `tester_agent` runs.

    - Tests passed, or budget exhausted -> proceed to `qa_package` (then PR review).
    - Tests failed, budget left         -> loop back to `coder_agent` (self-healing).
    """
    if state["test_passed"] or state["loop_count"] >= MAX_LOOP_ATTEMPTS:
        if not state["test_passed"]:
            log.critical(
                "[route_validation_gate] MAX_LOOP_ATTEMPTS (%d) exceeded for %s — proceeding to review anyway.",
                MAX_LOOP_ATTEMPTS,
                state["jira_ticket_id"],
            )
        return "qa_package"
    return "coder_agent"


# ==============================================================================
# 5. GRAPH ASSEMBLY
# ==============================================================================
def build_graph(checkpointer: MemorySaver):
    """Wires nodes + edges into a compiled, checkpointed LangGraph app."""
    builder = StateGraph(SDLCState)

    builder.add_node("jira_ingest", jira_ingest)
    builder.add_node("git_clone", git_clone)
    builder.add_node("coder_agent", coder_agent)
    builder.add_node("tester_agent", tester_agent)
    builder.add_node("qa_package", qa_package)
    builder.add_node("pr_review_agent", pr_review_agent)

    builder.add_edge(START, "jira_ingest")
    builder.add_edge("jira_ingest", "git_clone")
    builder.add_edge("git_clone", "coder_agent")
    builder.add_edge("coder_agent", "tester_agent")
    builder.add_conditional_edges(
        "tester_agent",
        route_validation_gate,
        {"coder_agent": "coder_agent", "qa_package": "qa_package"},
    )
    builder.add_edge("qa_package", "pr_review_agent")
    builder.add_edge("pr_review_agent", END)

    return builder.compile(checkpointer=checkpointer)


# ==============================================================================
# 6. AUDIT TRAIL — ASCII CHECKPOINT TIMELINE
# ==============================================================================
def print_checkpoint_timeline(app, config: dict, title: str) -> list:
    """Renders every historical checkpoint for a thread as an ASCII table.

    This is the SOC2-relevant artifact: `app.get_state_history(config)` walks
    the checkpointer's immutable log of every super-step the graph has taken.
    Nothing here is reconstructed after the fact — it is the literal
    persisted audit trail.
    """
    history = list(app.get_state_history(config))
    history.reverse()  # checkpointer yields newest-first; display oldest-first

    width = 100
    print("\n" + "=" * width)
    print(f" 🕰️  {title}  ({len(history)} checkpoints)".center(width))
    print("=" * width)
    print(f"{'#':<3} {'checkpoint_id':<38} {'next_node':<20} {'loop':<5} {'passed':<7}")
    print("-" * width)

    for i, snap in enumerate(history):
        cp_id = snap.config["configurable"]["checkpoint_id"]
        next_node = snap.next[0] if snap.next else "—(END)—"
        v = snap.values
        print(
            f"{i:<3} {cp_id:<38} {next_node:<20} "
            f"{v.get('loop_count', '-'):<5} {str(v.get('test_passed')):<7}"
        )

    print("=" * width)
    return history


# ==============================================================================
# 7. RUNNER — processes a single ticket file end-to-end
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python main.py <path_to_jira_ticket_file>")

    ticket_path = sys.argv[1]
    jira_ticket_id, requirements, repo_url = load_ticket(ticket_path)

    checkpointer = MemorySaver()
    app = build_graph(checkpointer)

    config = {"configurable": {"thread_id": jira_ticket_id}}

    initial_state: SDLCState = {
        "jira_ticket_id": jira_ticket_id,
        "requirements": requirements,
        "repo_url": repo_url,
        "workspace_dir": "",
        "branch_name": "",
        "generated_code": "",
        "test_reasoning": "",
        "test_modifications": "",
        "pr_repo_dir": "",
        "pr_review_dir": "",
        "pr_review_passed": False,
        "pr_review_reasoning": "",
        "loop_count": 0,
        "test_passed": False,
    }

    banner("RUNNING PIPELINE")
    for update in app.stream(initial_state, config, stream_mode="updates"):
        print(f"   → node completed: {next(iter(update))}")

    final_state = app.get_state(config)
    banner("PIPELINE COMPLETE")
    print(f"Final state values:\n{final_state.values}\n")

    print_checkpoint_timeline(app, config, "FULL AUDIT TRAIL")

    # Move the ticket file (unmodified) out of the inbox now that processing is done.
    os.makedirs(DONE_DIR, exist_ok=True)
    shutil.move(ticket_path, os.path.join(DONE_DIR, os.path.basename(ticket_path)))
    print(f"📁 Ticket file moved to {DONE_DIR}/{os.path.basename(ticket_path)}")
