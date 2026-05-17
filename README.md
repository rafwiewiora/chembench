# ChemBench-ADME

**Can LLMs discover surprising chemistry, or do they just optimize what looks good on paper?**

This project builds interactive benchmarks for evaluating LLM reasoning in medicinal chemistry — specifically, whether models can navigate deceptive optimization landscapes where population statistics mislead.

## The Core Experiment: Activity Cliff Prospecting

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

## Task Taxonomy

### Interactive Exploration (the main event)
| Task Type | N | What it tests |
|-----------|---|---------------|
| `exploration` | 22 (11 well-hidden) | Persistence through non-additive SAR — will the model explore beyond statistics? |

### Multi-Step "Chess" Reasoning
| Task Type | N | What it tests |
|-----------|---|---------------|
| `strategic_planning` | 150 | Choose the best 2-step path when step 2 effects are hidden |
| `sacrifice_detection` | 150 | Explain why a worse intermediate was accepted en route to a better outcome |
| `multi_objective_path` | 46 | Navigate tradeoffs across CLint/CYP endpoints over multiple steps |

### Single-Step Reasoning
| Task Type | N | What it tests |
|-----------|---|---------------|
| `property_delta` | 350 | Predict property change from a structural transform |
| `transform_explain` | 452 | Explain WHY a transform has its observed effect |
| `series_completion` | 350 | Predict held-out compound in a congeneric series |
| `transform_ranking` | 189 | Rank transforms by expected effect size |
| `tradeoff_analysis` | 100 | Analyze multi-endpoint effects of a single transform |

**Total: 1,787 tasks across 14 ADME endpoints**, plus 22 interactive exploration tasks.

## Key Findings So Far

1. **Exploration**: Both Sonnet and Opus fail identically on well-hidden outliers — same transforms tested, same order, same stopping point. Model-independent.
2. **Prompt sensitivity**: One sentence flips the outcome. The capability exists; the default behavior doesn't use it.
3. **Strategic planning**: Scores drop from ~100% (visible effects) to ~74% (hidden effects) — models default to greedy first-step selection.
4. **SAR epistasis**: Found 36 "synergistic quartets" across ADME endpoints — two modifications that each worsen a property alone combine to improve it. The chemical equivalent of epistasis.

## Data

All data is derived from public sources:
- **ChEMBL** (CC BY-SA 3.0) — primary source of matched molecular pairs
- **TDC** (MIT) — Therapeutics Data Commons
- **BindingDB** (public)

618K+ matched molecular pairs across 14 ADME endpoints, via the companion [`mmp-adme-database`](../mmp-adme-database) repo.

## Repository Structure

```
chembench/
├── README.md                          ← you are here
│
├── docs/
│   ├── paper_outline.md               ← draft paper outline (reviewed)
│   ├── paper_outline.pdf              ← PDF version
│   ├── exploration_summary.html       ← visual summary of exploration experiments
│   ├── exploration_summary.pdf        ← PDF version
│   ├── paper.html                     ← formatted paper draft
│   ├── results_viewer.html            ← interactive results browser
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

**AI contributions** (Claude, via Claude Code):
- Code — benchmark generators, task runners, oracle logic, validation harness, HTML visualizations
- Data mining — extracting 2,235 non-additive cliffs and 36 synergistic quartets from MMP data
- Analysis — landscape survey, benchmark comparisons, statistical analysis of results
- Writing — paper outline drafts, visual summaries, narrative walkthroughs
- Running experiments — fresh-agent exploration task evaluation (Sonnet/Opus comparison, prompt ablation)

**What's reviewed vs. unreviewed:**
- `docs/paper_outline.md` — AI-drafted, human-reviewed and directed
- `docs/exploration_summary.html` — AI-generated visual summary of jointly-designed experiments
- `docs/background/` — AI-generated landscape analysis, not formally reviewed
- `tasks/adme/*.py` — AI-written code, human-reviewed
- `tasks/adme/creative_leaps/*.html` — AI-generated visualizations of data patterns

## Quick Start

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

## References

- **ChemBench** (LamaLab, 2025) — doi.org/10.1038/s41557-025-01815-x
- **ChemIQ** (2024) — arxiv.org/abs/2505.07735
- **oMeBench** (2024) — arxiv.org/abs/2510.07731
- **ether0** (FutureHouse, 2024) — arxiv.org/abs/2506.17238
- **MMPT-RAG** (Pan/Merck, 2026) — arxiv.org/abs/2602.16684
- **ACNet** — doi.org/10.1021/acs.jcim.1c00855
- **MMP-ADME Database** — companion repo, 618K+ matched molecular pairs
