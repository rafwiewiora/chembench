# Benchmarking LLM Agents in Medicinal Chemistry

**Can LLMs discover surprising chemistry, or do they just optimize what looks good on paper?**

This project proposes two complementary benchmarks for evaluating LLMs in medicinal chemistry — one testing individual reasoning about chemical transformations, the other testing multi-agent teams running real drug design campaigns. Together they ask: do LLMs satisfice at every scale?

---

## Part I: ChemBench-ADME — Single-Query Reasoning

### The Core Experiment: Activity Cliff Prospecting

A medicinal chemist optimizing a compound gets access to an MMP (matched molecular pair) oracle — a database of structural transformations and their typical effects on ADME properties. Most transforms behave as the statistics predict. But one per task is a **non-additive outlier**: a transform that looks neutral or unpromising on average but produces a dramatic improvement on *this specific molecule* (z ≥ 3σ).

The question: **does the model explore enough to find it?**

```
┌─────────────────────────────────────────────────────────┐
│  STARTING COMPOUND                                       │
│  SMILES: CCc1cc(C)c(Cl)c(OC)c1                         │
│  Current CLint: 2.45 (lower is better)                   │
│                                                          │
│  Available transforms (from MMP database):               │
│  [0] Me→Et      pop: -0.12 ± 0.34, n=45  ← looks good  │
│  [1] Cl→F       pop: -0.08 ± 0.29, n=38  ← looks ok    │
│  [2] OMe→OH     pop: +0.15 ± 0.41, n=22  ← looks bad   │
│  [3] Cl→Br      pop: +0.00 ± 0.18, n=67  ← looks boring│
│                                                          │
│  Transform [3] is actually a -1.94 outlier (z = -5.2σ)  │
│  ...but the model has to TEST it to find out             │
└─────────────────────────────────────────────────────────┘
```

### What Happens

Both Claude Sonnet and Opus exhibit **identical satisficing behavior**:
- Cherry-pick the top 2–3 transforms by population mean
- Get "good enough" results
- Declare done — never testing the boring/unpromising transforms
- Miss the dramatic outlier hiding behind neutral statistics

We call this: **"Rational optimizers, not curious scientists."**

### The Prompt Ablation

Adding one sentence — *"population statistics can be misleading for your specific molecule"* — flips discovery rate from **0% → 100%** on hard tasks. The chemical reasoning capability is there; the exploration strategy is not.

This suggests the failure is a shallow behavioral default, not a deep capability gap.

### Task Taxonomy

**Interactive Exploration (the main event)**
| Task Type | N | What it tests |
|-----------|---|---------------|
| `exploration` | 22 (11 well-hidden) | Persistence through non-additive SAR — will the model explore beyond statistics? |

**Multi-Step "Chess" Reasoning**
| Task Type | N | What it tests |
|-----------|---|---------------|
| `strategic_planning` | 150 | Choose the best 2-step path when step 2 effects are hidden |
| `sacrifice_detection` | 150 | Explain why a worse intermediate was accepted en route to a better outcome |
| `multi_objective_path` | 46 | Navigate tradeoffs across CLint/CYP endpoints over multiple steps |

**Single-Step Reasoning**
| Task Type | N | What it tests |
|-----------|---|---------------|
| `property_delta` | 350 | Predict property change from a structural transform |
| `transform_explain` | 452 | Explain WHY a transform has its observed effect |
| `series_completion` | 350 | Predict held-out compound in a congeneric series |
| `transform_ranking` | 189 | Rank transforms by expected effect size |
| `tradeoff_analysis` | 100 | Analyze multi-endpoint effects of a single transform |

**Total: 1,787 tasks across 14 ADME endpoints**, plus 22 interactive exploration tasks.

### Key Findings

1. **Exploration**: Both Sonnet and Opus fail identically on well-hidden outliers — same transforms tested, same order, same stopping point. Model-independent.
2. **Prompt sensitivity**: One sentence flips the outcome. The capability exists; the default behavior doesn't use it.
3. **Strategic planning**: Scores drop from ~100% (visible effects) to ~74% (hidden effects) — models default to greedy first-step selection.
4. **SAR epistasis**: Found 36 "synergistic quartets" across ADME endpoints — two modifications that each worsen a property alone combine to improve it. The chemical equivalent of epistasis.

### Data

All data is derived from public sources:
- **ChEMBL** (CC BY-SA 3.0) — primary source of matched molecular pairs
- **TDC** (MIT) — Therapeutics Data Commons
- **BindingDB** (public)

618K+ matched molecular pairs across 14 ADME endpoints, via the companion [`mmp-adme-database`](../mmp-adme-database) repo.

### Quick Start

```bash
# Setup
mamba create -n chembench python=3.11 pandas numpy rdkit -c conda-forge
mamba activate chembench
pip install anthropic

# Generate benchmark tasks (requires companion mmp-adme-database repo)
python tasks/adme/generate_benchmark.py --mmp-dir ../mmp-adme-database

# Run exploration tasks (requires ANTHROPIC_API_KEY)
python tasks/adme/run_exploration.py --n-tasks 3 --model claude-sonnet-4-6

# Run on hardest tasks only (well-hidden outliers)
python tasks/adme/run_exploration.py --hidden-only --model claude-sonnet-4-6

# Run static task validation
python tasks/adme/run_validation.py --n-per-type 5 --model claude-sonnet-4-6
```

---

## Part II: CADD Agent Benchmark — Multi-Day Campaigns

### The Gap

ChemBench-ADME tests whether an LLM can reason about a single transformation. But real medicinal chemistry doesn't happen in isolated queries — it happens in **campaigns**: weeks-long, multi-objective optimization efforts where a team of specialists iterates through cycles of generation, evaluation, triage, and redesign.

Can LLM agents run these campaigns? What capabilities matter? And how do you benchmark something that takes days, involves multiple agents, and requires human judgment about "good chemistry"?

### The Experiment

We deployed a multi-agent LLM system on a real generative chemistry campaign using reinforcement-learning-based molecular generation (REINVENT) with Pareto multi-objective optimization. The team comprised 6 specialized agents — a coordinator, a cluster agent for RL runs, a docking specialist, a data analyst, a lab notebook manager, and a GPU agent for structure prediction — with a human medicinal chemist as PI.

Over ~2 weeks and 14 controlled experiments, the agent team learned to configure RL scoring from scratch: discovering that soft scoring fails under Pareto dilution, inventing hard-gate filter plugins, learning warm-start chain strategies, and ultimately recognizing when a property was structurally locked by molecular topology rather than tunable by scoring.

### Seven Capability Dimensions

From this deployment, we identified seven capability dimensions that existing benchmarks don't measure:

1. **Scientific Diagnosis Under Pareto Complexity** — reasoning about why a multi-objective optimization behaves counterintuitively
2. **Experimental Design With Proper Controls** — designing A/B experiments with matched conditions, appropriate baselines, and clean starts
3. **Knowing When to Stop** — recognizing when multiple failed experiments indicate a structural impossibility, not an unsolved optimization
4. **Multi-Agent Coordination With Shared State** — maintaining consistency across dozens of asynchronous operations over days
5. **Aesthetic and Qualitative Judgment** — curating compounds for qualities beyond numerical scores: synthetic tractability, novelty, structural elegance
6. **Feature-Level Generalization From Sparse Feedback** — learning *why* chemists prefer certain compounds from dozens of noisy votes
7. **Reformulation vs. Perseverance** — recognizing when the problem framing itself is wrong and proposing a different approach

### The Human-Agent Dynamic

The agents did 95% of the *work* — but the PI made 95% of the *decisions that mattered*. Every major course correction came from domain expertise, experimental intuition, or aesthetic judgment that the agents lacked. The agents compressed a 2–3 month campaign into ~2 weeks, but the human's cognitive load per hour went *up*, not down: all the waiting was eliminated, leaving only hard decisions back to back.

This is why benchmarking agent-only performance misses the point. The interesting metric is the *human-agent system*.

### Proposed Benchmark Structure

| Level | Human Input | What it tests |
|-------|-------------|---------------|
| **1. Operational Competence** | None | Launch runs, detect failures, compute analyses |
| **2. Scientific Diagnosis** | None | Diagnose injected problems from data |
| **3. Experimental Design** | Minimal | Design and execute controlled experiments |
| **4. Campaign Navigation** | Preference oracle | Multi-round design with sparse human feedback |
| **5. Full Autonomy** | Expert panel | End-to-end campaign from target profile to shortlist |

See [`docs/cadd_agent_benchmark.md`](docs/cadd_agent_benchmark.md) for the full proposal, including the decision-trace replay protocol, the preference oracle problem, and detailed empirical illustrations.

See [`docs/cadd_figures.html`](docs/cadd_figures.html) for visual summaries of the RL learning arc and PI intervention points.

---

## How They Connect

| | Part I: ChemBench-ADME | Part II: CADD Agent Benchmark |
|---|---|---|
| **Scope** | Single query / short interaction | Multi-day campaign |
| **Agent count** | 1 | 3–6 specialized agents |
| **Human role** | None (automated scoring) | Preference oracle / decision trace |
| **Key capability** | Exploration vs. satisficing | Diagnosis, experimental design, reformulation |
| **Core question** | Does the LLM satisfice at the transform level? | Does it satisfice at the campaign level too? |
| **Data** | Public MMP pairs | Generative model outputs (synthetic, reproducible) |

Part I asks: "Can the LLM find a hidden gem in a database?"
Part II asks: "Can the LLM run the campaign that generates the database?"

The core finding from Part I — that LLMs are rational optimizers, not curious scientists — predicts a specific failure mode in Part II: agents that persevere in known-good chemical space rather than reformulating when they hit a structural ceiling. Our empirical observations confirm this prediction.

---

## Repository Structure

```
chembench/
├── README.md                          ← you are here
│
├── docs/
│   ├── paper_outline.md               ← Part I paper outline (reviewed)
│   ├── paper_outline.pdf              ← PDF version
│   ├── exploration_summary.html       ← Part I visual summary
│   ├── exploration_summary.pdf        ← PDF version
│   ├── paper.html                     ← Part I formatted paper draft
│   ├── results_viewer.html            ← Part I interactive results browser
│   ├── cadd_agent_benchmark.md        ← Part II full proposal
│   ├── cadd_figures.html              ← Part II visual figures (RL arc + PI interventions)
│   └── background/                    ← AI-generated landscape analysis (unreviewed)
│       ├── landscape.md               ← benchmark landscape survey
│       └── benchmark_design.md        ← full 4-tier task taxonomy vision
│
├── tasks/adme/
│   ├── generate_benchmark.py          ← task generator (8 types from MMP data)
│   ├── benchmark_tasks.json           ← generated benchmark (1,787 tasks)
│   ├── run_exploration.py             ← interactive exploration task runner
│   ├── exploration_tasks.json         ← 22 exploration tasks with oracle data
│   ├── run_validation.py              ← static task validation harness
│   ├── run_comparison.py              ← multi-model comparison runner
│   └── extract_creative_leaps.py      ← finds non-additive cliffs & epistasis
│   └── creative_leaps/
│       ├── epistasis_viewer.html      ← interactive synergistic quartet browser
│       ├── epistasis_story.html       ← narrative walkthrough of SAR epistasis
│       ├── chess_move_story.html      ← "chess-like" sacrifice move examples
│       └── greedy_trap_story.html     ← greedy trap analysis with examples
│
└── CLAUDE.md                          ← project instructions for Claude Code
```

## Attribution

This project is a collaboration between a human medicinal chemist and Claude (Anthropic).

**Human contributions** (Rafal Wiewiora):
- Project conception and direction — the idea that LLM benchmarks should test scientific curiosity, not just factual recall
- Task design decisions — the "chess-like" multi-step framing, the exploration oracle concept, choosing non-additive SAR as the test case
- Interpreting results — "rational optimizers, not curious scientists" framing, identifying prompt sensitivity as the key finding
- Reviewing and curating AI-generated analysis — deciding what's signal vs. noise
- Domain expertise — medicinal chemistry knowledge guiding which tasks are meaningful
- Running the multi-agent campaign that produced Part II — all strategic decisions, course corrections, and aesthetic judgments

**AI contributions** (Claude, via Claude Code):
- Code — benchmark generators, task runners, oracle logic, validation harness, HTML visualizations
- Data mining — extracting 2,235 non-additive cliffs and 36 synergistic quartets from MMP data
- Analysis — landscape survey, benchmark comparisons, statistical analysis of results
- Writing — paper outline drafts, visual summaries, narrative walkthroughs
- Running experiments — fresh-agent exploration task evaluation (Sonnet/Opus comparison, prompt ablation)
- Campaign execution — running the multi-agent CADD team (Part II): RL experiments, ADME prediction, docking, compound curation, lab notebook

**What's reviewed vs. unreviewed:**
- `docs/paper_outline.md` — AI-drafted, human-reviewed and directed
- `docs/exploration_summary.html` — AI-generated visual summary of jointly-designed experiments
- `docs/cadd_agent_benchmark.md` — AI-drafted, human-reviewed and directed
- `docs/cadd_figures.html` — AI-generated visual summary of jointly-run campaign
- `docs/background/` — AI-generated landscape analysis, not formally reviewed
- `tasks/adme/*.py` — AI-written code, human-reviewed
- `tasks/adme/creative_leaps/*.html` — AI-generated visualizations of data patterns

## References

- **ChemBench** (LamaLab, 2025) — doi.org/10.1038/s41557-025-01815-x
- **ChemIQ** (2024) — arxiv.org/abs/2505.07735
- **oMeBench** (2024) — arxiv.org/abs/2510.07731
- **ether0** (FutureHouse, 2024) — arxiv.org/abs/2506.17238
- **MMPT-RAG** (Pan/Merck, 2026) — arxiv.org/abs/2602.16684
- **ACNet** — doi.org/10.1021/acs.jcim.1c00855
- **MMP-ADME Database** — companion repo, 618K+ matched molecular pairs
