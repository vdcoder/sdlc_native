# SDLC Native (Multi-Agents)

![SDLC Native Multi-Agents](SDLCNative.png)

### Before we automate the engineering organization, can we automate one engineering loop we can actually trust?

A few hours before this repository existed, I was in a job interview discussing what an **AI-native software organization** might look like.

The conversation eventually reached multiple agents: coding agents, testing agents, review agents, agents delegating work to other agents, and an engineering organization increasingly built around autonomous AI collaboration.

It is an exciting direction.

But something bothered me.

If the goal is to automate more and more of the software-development lifecycle, I don't think the first architectural step should be:

> *Give several autonomous agents a problem and let them figure out how to organize themselves.*

I think the first step should be considerably less magical:

> **Build one deterministic, observable and verifiable engineering graph — then place AI inside the steps where probabilistic reasoning is actually useful.**

That is what this repository explores.

It is a small, fully working agentic SDLC pipeline built with native LangGraph primitives.

Give it a ticket and, without pretending Jira or GitHub integrations exist where they do not, the real pipeline will:

**clone → inspect → code → test → repair → package → independently review**

The coder, tester, and reviewer are real LLM-driven agents using real filesystem tools. But they do **not** own the engineering process.

**The graph does.**

```text
                    ┌──────────── retry ────────────┐
                    │                               │
                    ▼                               │
Ticket → Clone → Coder Agent → Tester Agent ────────┘
                                │
                              PASS
                                │
                                ▼
                             Package
                                │
                                ▼
                        Independent Review
                                │
                                ▼
                               END
```

The AI is probabilistic.

The workflow is not.

---

## Why build it this way?

There are two very different things we can automate:

1. **reasoning**
2. **authority**

LLMs are increasingly excellent at the first.

That does not mean they should automatically receive the second.

In this project, an agent can inspect code, decide what should change, write files, evaluate another agent's work, and even ask a human for help.

But the surrounding application still determines:

* what role the agent occupies;
* what tools it receives;
* what filesystem it can touch;
* what state transition follows;
* how many retries it gets;
* when work is packaged;
* what another independent agent reviews;
* and when the workflow ends.

That distinction is intentional.

The architecture is not trying to eliminate autonomous agents.

It is trying to give autonomy **boundaries**.

---

# A ticket becomes a graph execution

Drop a small JSON ticket into `jira_tickets/`:

```json
{
  "jira_ticket_id": "JIRA-4521",
  "requirements": "Explore this repository and add a summary.txt describing what it contains.",
  "repo_url": "https://github.com/octocat/Hello-World.git"
}
```

`jira_ticket_id` and `requirements` are required. `repo_url` is optional — leave it out and the coder gets a plain scratch folder instead of a cloned repo.

Then run:

```bash
python main.py jira_tickets/my_ticket.json
```

Or let the watcher process incoming tickets independently and in parallel:

```bash
python watcher.py
```

The system will clone the requested repository, create an isolated workspace, let a coding agent inspect and modify it, send the result through an independent testing agent, retry failed work up to a bounded budget, package the result, and finally give a read-only copy to a separate PR-review agent. 

Everything it produces lands somewhere predictable:

| Path | Contents |
|---|---|
| `jira_tickets/` | Inbox — drop new ticket files here |
| `jira_tickets_done/` | Ticket files moved here (unmodified) once processing completes |
| `workspace/<ticket>/` | The cloned repo (or scratch dir) the coder/tester actually edit |
| `github_PRS/<ticket>/repo/` | A packaged copy of the workspace, frozen at review time |
| `github_PRS/<ticket>/QA/QA_results.md` | The tester's verdict + reasoning |
| `github_PRS/<ticket>/PR_review/PR_review_results.md` | The reviewer's verdict + reasoning |

This is not a simulated sequence of prompts.

The filesystem operations are real.

The generated code is real.

The tests and verdicts are real.

The repair loop is real.

The independent review is real.

Only the Jira/GitHub edges are mocked so the experiment remains self-contained. 

---

# The graph is the engineering contract

The topology is deliberately simple:

```text
START
  │
  ▼
jira_ingest
  │
  ▼
git_clone
  │
  ▼
coder_agent ◄─────────────────┐
  │                           │
  ▼                           │
tester_agent                  │
  │                           │
  ├── fail + budget left ─────┘
  │
  ▼
qa_package
  │
  ▼
pr_review_agent
  │
  ▼
END
```

Three nodes contain LLM-driven reasoning:

* `coder_agent`
* `tester_agent`
* `pr_review_agent`

The rest are ordinary deterministic software.

That distinction matters.

A conventional function should remain a conventional function when reasoning is unnecessary. There is little value in asking an LLM to clone a Git repository or copy a directory merely so that every rectangle in an architecture diagram can be called an agent.

**Use intelligence where intelligence adds value. Use software everywhere else.**

---

# The agents are powerful, but deliberately unequal

The coder can explore and modify its workspace.

The tester can inspect that work, execute its own reasoning, make small corrections, and decide whether the implementation passes.

The final reviewer gets a frozen packaged copy and is **read-only**.

It can approve or reject.

It cannot quietly repair the evidence it is supposed to judge.

That separation gives us something closer to an engineering process than a single giant prompt pretending to contain one.

The agents use bounded ReAct-style tool loops with real filesystem tools, and all paths are resolved through sandboxed tool implementations so an agent cannot escape the workspace it was assigned—even if the model attempts an absolute path or `../..`. 

---

# Failure is part of the architecture

An autonomous loop without a stopping condition is not autonomy.

It is a billing incident.

The coder/tester repair cycle therefore has an explicit retry budget:

```python
def route_validation_gate(state: SDLCState) -> Literal["coder_agent", "qa_package"]:
    if state["test_passed"] or state["loop_count"] >= MAX_LOOP_ATTEMPTS:
        return "qa_package"

    return "coder_agent"
```

By default, the system allows three attempts.

If the implementation still fails, the ticket does **not** disappear into an infinite self-healing fantasy.

It moves forward carrying the evidence of failure.

The QA package records what happened, and the reviewer receives the imperfect result.

A human can see:

> what was attempted,
> what failed,
> and why the system stopped.

The retry limit is therefore simultaneously a reliability boundary and a token-cost circuit breaker. 

There is a smaller version of the same idea one level down: each agent's own tool-calling loop — explore, read, write, decide — is capped too, by default at six steps per attempt. A single confused agent can't spin forever inside its own turn, either.

---

# Humans are available without becoming a mandatory ceremony

Any reasoning agent can call:

```text
ask_human
```

when it genuinely needs information.

There is intentionally no universal:

```text
AI → HUMAN APPROVAL → AI → HUMAN APPROVAL → AI
```

pipeline.

That would preserve human control by destroying most of the useful autonomy.

Instead, the human is another capability available to the agent when uncertainty actually requires escalation.

The human is not removed.

The human is **pulled into the graph deliberately**. 

---

# State should be inspectable

Every node operates against one centralized `SDLCState`.

A node receives:

```text
SDLCState
```

and returns only the fields it changed.

LangGraph merges those changes into checkpointed state.

Each concurrent ticket receives its own `thread_id`, and every transition can be inspected through state history. The result is a replayable lifecycle of the ticket rather than an opaque conversation that happened somewhere inside an agent runtime. 

For an enterprise SDLC, this becomes increasingly important.

If an AI changes production software, eventually someone will ask:

> Why did it do that?

“Because the agent decided to” is not a sufficient audit trail.

---

# Multiple agents do not have to mean uncontrolled agent-to-agent authority

This experiment also supports something I wanted to test explicitly.

Any agent node can be delegated to an external agent service:

```env
EXTERNAL_AGENT_CODER=http://localhost:9000/coder
EXTERNAL_AGENT_TESTER=http://localhost:9000/tester
EXTERNAL_AGENT_PR_REVIEW=http://localhost:9000/pr-review
```

The external agent can reason elsewhere—even on another machine, model, framework, or hardware architecture—and return the ordered tool actions it wants performed.

But those actions are replayed **locally through the same sandboxed tools**.

So the remote agent can decide:

> *what should happen*

without automatically receiving authority over:

> *where and how it happens.*

That separation may be considerably more interesting than simply connecting Agent A directly to Agent B.

It lets the deterministic graph remain the protocol between potentially heterogeneous forms of intelligence. 

---

# Model independence

The pipeline does not require one model provider.

Agents use LangChain's `BaseChatModel` abstraction, and the provider can be changed through configuration:

```env
LLM_PROVIDER=ollama
LLM_PROVIDER=openai
LLM_PROVIDER=gemini
LLM_PROVIDER=anthropic
```

The default can run locally through Ollama with a coding model, allowing the entire experiment to operate offline and without an API key. 

This becomes particularly interesting when combined with external delegation:

A cheap local model may handle one stage.

A frontier model may handle another.

Specialized compute may perform a third.

The graph doesn't have to care.

---

# Why LangGraph?

I wanted the experiment to expose the control structure rather than hide it behind a large autonomous-agent abstraction.

LangGraph gives us exactly the useful primitive:

> **state + nodes + edges + conditional transitions**

The interesting engineering question then becomes:

> Where should reasoning live?

rather than:

> Which agent framework should control my application?

That is a distinction I care about in most of my AI work.

---

# What this is not

This is not an assertion that this particular graph should run an enterprise engineering organization tomorrow.

It is deliberately small.

It has:

* mocked Jira ingestion;
* mocked GitHub publication;
* in-memory checkpointing;
* process-level ticket concurrency;
* a deliberately limited set of tools.

Those constraints make the experiment understandable.

They also create obvious next steps: durable checkpoints, real Jira/GitHub adapters, security scanning, production evaluation gates, more sophisticated test environments, richer observability, and specialized external agents. 

---

# Where I think this can go

The graph could eventually become much richer:

```text
Requirement
    │
    ▼
Requirement Analysis
    │
    ▼
Architecture Gate
    │
    ▼
Implementation
    │
    ▼
Unit / Integration Testing
    │
    ├──── repair loop
    │
    ▼
Security Review
    │
    ├──── repair loop
    │
    ▼
Performance Evaluation
    │
    ▼
PR Review
    │
    ▼
Human / Policy Gate
    │
    ▼
Deployment
    │
    ▼
Production Observation
    │
    └────► new evidence enters future evaluations
```

Some nodes may be deterministic.

Some may contain one model.

Some may contain a local agent.

Some may delegate to a specialized external agent.

Some may require a human.

The important part is that **the engineering process itself remains explicit**.

---

# The hypothesis

This repository is testing a simple hypothesis:

> **An AI-native engineering organization does not have to begin by replacing its development process with autonomous agents.**

It can begin by making the development process **machine-executable**.

Then we can gradually decide which steps deserve intelligence, which deserve autonomy, which require verification, and which still require people.

Perhaps eventually many autonomous engineering agents will negotiate, delegate, specialize, and collaborate with one another.

I suspect they will.

But before building a society of software engineers made of AI, I would like to know that **one ticket can travel through one graph and leave behind evidence that I can understand and verify.**

This repository is that first experiment.

---

## Running it

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt

# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

python main.py jira_tickets/my_ticket.json
```

Or:

```bash
python watcher.py
```

---

## Extending it

Good next experiments include:

* persistent PostgreSQL checkpoints;
* real Jira and GitHub adapters;
* security and adversarial-review nodes;
* executable test runners;
* performance gates;
* LLM evaluation and release thresholds;
* record/replay of failed agent trajectories;
* richer human escalation;
* cost/latency-aware model routing;
* specialized external agents;
* multiple implementations competing against the same evaluation gate.

---

## One final principle

**AI should be allowed to reason broadly before it is allowed to act broadly.**

That separation is cheap to design at the beginning.

It becomes very expensive to rediscover after autonomy has already spread through the system.

---

This README, and every guardrail described in it, took shape through an actual back-and-forth with GitHub Copilot (Claude Sonnet 5) — genuinely a co-creator on this one, not just an autocomplete.

---
