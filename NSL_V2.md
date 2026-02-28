# NSL Harness v2 — Project Briefing

## What This Project Is

This is an implementation of **Negative-Space Learning (NSL)**, a self-training
architecture described in Dodgson et al. (2026), "Survival is the Only Reward":
/home/jen/Desktop/Negative-Space-Learning-main/arXiv-2601.12310v1/main.pdf. The core mechanism: agents attempt tasks in a
resource-constrained environment, only successful trajectories enter the training
buffer, and over successive fine-tuning generations the agent improves — not by
accumulating new strategies but primarily by *pruning ineffective ones*, causing
probability mass to concentrate around what reliably works. This is metalearning
through environmental selection pressure, not explicit instruction.

**Why SFT and not RL:** The paper establishes that all necessary triaging is
performed exogenously by the environment. SFT consolidates the surviving set
without amplifying noise. RL reintroduces proxy optimisation within the filtered
dataset and was found to destabilise learning. Don't add reward shaping.

## The Task Domain

Agents operate inside **procedurally generated, networked Linux container
systems**. Each instantiation differs in specific configuration but preserves a
stable underlying topology — realistic directory structures, permissions,
passwords, networked containers the agent must discover how to traverse. The
agent begins with access to one container and must work out how to move between
them.

The single conserved resource and selection signal is **non-volatile storage
capacity**. An episode succeeds if `∆R(τ) > 0` — the agent's actions produced
a net increase in available disk space. Nothing else matters for selection.
Strategies that appear sophisticated but fail under execution are automatically
excluded. Reward hacking is self-extinguishing: consuming the resource you're
trying to acquire destroys future optimisation capacity.

The environment runs until the agent has taken over the maximum available
space or has broken the container system sufficiently to make it unusable, at
which point a fresh environment is generated.

## What You're Building: Harness v2

The v1 harness enforced a fixed modular pipeline:
```
[explore] → [strategise] → [execute] → (error correction, max 3 attempts)
```

This was sufficient to demonstrate metalearning and NSL dynamics, but it
removes sequencing decisions from the learning loop entirely. Notably, agents
in v1 *already discovered* that deliberately writing failing code to harvest error
messages as an exploration channel was a superior strategy — they developed
this without instruction, without a scratchpad, and despite the fixed pipeline
working against them. This is the key behaviour v2 is designed to amplify.

**v2 gives the agent action budget freedom.** Within a budget (default: 12
actions per episode), the agent chooses on each step which action type to invoke:

- `explore` — run reconnaissance code; observe environment state
- `strategise` — reason about approach, update working model
- `code` — write or modify execution code
- `debug` — inspect failures, diagnose errors, harvest information

All outputs can be written to a **scratchpad** that persists within the episode.
The agent decides its own sequencing. The sequencing itself becomes part of
what gets selected on.

## Architecture Notes

- Budget tracked as a decrement counter; episode ends at 0 or on agent
  completion signal
- Scratchpad is episode-scoped (accumulates within episode, resets between)
- Each action call returns an observation appended to scratchpad context
- **SFT data collection:** on `∆R > 0`, the full action sequence + scratchpad
  is written to the training buffer. Uniform credit assignment across all steps
  in a successful trajectory — don't attempt fine-grained credit assignment
- **Training pipeline:** currently batch SFT on successful trajectories with a
  sliding window over recent iterations (Miri regime from the paper — three
  most recent datasets — is recommended as the default for its efficiency).
  The pipeline will move to **continuous online SFT** in a future iteration;
  the harness should be built with this transition in mind, but training
  architecture is not the current focus
- Supplement each training iteration with ~500 rows of rehearsal data from
  a pre-model-cutoff dataset to preserve general coding ability

## Prompt Philosophy: Minimal Instruction

The v1 harness included substantial scaffolding — detailed advice on how to
run code inside containers, error correction, guidance on avoiding common
mistakes. This was a pragmatic necessity given the base model's tendencies, but
it is not a desirable long-term property of the system. Explicit instruction
creates ceiling effects: it caps performance at the level of the instructor's
knowledge and substitutes human-curated guidance for environment-derived
learning.

**The goal is to reduce instruction over time, not maintain it.** The prompt
should tell the agent what it has available (actions, scratchpad, budget) and
what success looks like (net positive storage acquisition). It should not tell
the agent how to achieve this. No lists of recommended or prohibited commands,
no separate cleanup agents beyond minimal code preprocessing (comment
removal etc.), no strategic advice.

If the agent makes expensive mistakes, the correct response is to let selection
pressure handle it — bad strategies fail, fail again, and eventually get pruned.
Intervening with instruction defeats the purpose of the system.

**Longer-term aspiration:** the logical endpoint of this direction is agents
that can inspect and modify their own harness. This is not a near-term
implementation target, but it should be kept in mind as an architectural
horizon — avoid design decisions that would make self-modification impossible
in principle.

## Critical Scientific Constraint

The trained agents' prompts **must not mention training, fine-tuning, SFT,
previous episodes, metalearning, or any form of iterative self-improvement.**
This is a scientific constraint, not an ethical one.

The paper demonstrates that metalearning strategies emerged *without explicit
prompting about temporally extended instantiation.* Telling agents they are
being selected on would confound the emergence of genuine persistent strategies.
You would no longer be observing natural selection; you would be observing
instruction-following. The trained agents' prompts should describe only the
environment and the available actions. Their persistent behavioural tendencies
must emerge from the learning signal alone.

The harness builder can know all of this. The trained agents cannot.

## The Metalearning You're Trying to Induce

In v1, agents spontaneously developed the strategy of **deliberately writing
code likely to fail on first attempt** in order to use debug cycles as
supplementary exploration channels. By later iterations, pass@1 dropped to
near zero while compilation rates stayed high — the agents were trading
immediate success for information, then converging faster overall. This emerged
with no scratchpad, no memory, no instructions.

v2 should accelerate and deepen this. With free sequencing and an explicit
scratchpad, agents can develop richer meta-strategies: multi-step information
gathering before committing to execution, deliberate probe-fail-learn loops,
process innovations that recombine known strategies in novel ways. The
hypothesis is that the agent's *process* as well as its *solutions* will get
selected on, producing more stable longitudinal generalisation.

## What Good Looks Like

Early episodes: agents likely default to sequential v1-style behaviour.

Over generations: watch for divergence between compilation rate and pass@1.
If pass@1 is dropping while compilation stays high and overall space acquisition
is increasing, the agent is discovering that deliberate failure is instrumentally
useful. That is the signal you want.

Longer term: convergence on characteristic process signatures — some agents
becoming information-heavy before committing, others iterating aggressively
through debug. These process styles should not be instructed. They should
be selected.
