# ChemBench

**A chemistry reasoning benchmark that tests what textbooks can't teach.**

Existing chemistry benchmarks overwhelmingly test factual recall. ChemBench tests the reasoning that separates a working medicinal chemist from a language model that memorized PubChem -- multi-step strategic thinking about structure-activity relationships, property tradeoffs, and molecular optimization paths.

## Why Another Benchmark?

Current benchmarks have two fundamental problems:

1. **Format inflation.** ChemIQ demonstrated that switching from short-answer to multiple-choice inflates scores by up to 10x on the same underlying questions. Most chemistry benchmarks use MCQ.

2. **Single-step ceiling.** oMeBench found models score 90% on predicting reaction products but only 24% on explaining mechanisms. ether0 (FutureHouse) tests single-step retrosynthesis and property prediction -- tasks where pattern matching suffices. Real drug discovery requires composing multiple reasoning steps under uncertainty.

ChemBench addresses both: all tasks are open-ended (no MCQ), and the most diagnostic tasks require multi-step strategic reasoning where intermediate outcomes are hidden.

## Key Finding

Claude Sonnet 4 scores 87% average on single-step ADME reasoning tasks. But on `strategic_planning` -- where the model must choose a 2-step molecular optimization path without seeing step 2 outcomes -- performance drops from 100% (with visible effects) to 74% (effects hidden). This gap measures genuine strategic reasoning versus pattern completion.

## Task Taxonomy

### Single-step reasoning (5 types)

| Task Type | Description | N |
|-----------|-------------|---|
| `property_delta` | Predict property change from a structural transform and explain the mechanism | 350 |
| `series_completion` | Given a congeneric series with SAR trend, predict the held-out compound | 350 |
| `transform_ranking` | Rank structural transforms by expected effect on an ADME endpoint | 189 |
| `tradeoff_analysis` | Analyze multi-endpoint effects of a single transform (e.g., clearance vs. CYP inhibition) | 100 |
| `transform_explain` | Explain WHY a transform produces its observed property change | 452 |

### Multi-step strategic reasoning (3 types)

| Task Type | Description | N |
|-----------|-------------|---|
| `sacrifice_detection` | Explain why a medicinal chemist accepted a worse intermediate in a 2-step optimization path | 150 |
| `strategic_planning` | Choose the best 2-step molecular path when step 2 effects are hidden | 150 |
| `multi_objective_path` | Navigate tradeoffs across clearance and CYP endpoints over multiple steps | 46 |

**Total: 1,787 tasks across 14 ADME endpoints.**

## How It Works

```
mmp-adme-database (618K+ matched molecular pairs from ChEMBL)
        |
        v
generate_benchmark.py (task generator, 8 task types)
        |
        v
benchmark_tasks.json (1,787 tasks with ground truth)
        |
        v
run_validation.py (runs tasks through LLM, auto-assesses, HTML report)
```

All tasks are grounded in real experimental data: matched molecular pairs (MMPs) where a single structural transformation produces a measured property change. This means every task has a verifiable ground truth derived from actual assay results, not expert opinion.

## Quick Start

### Requirements

```bash
mamba create -n chembench python=3.11 pandas numpy rdkit -c conda-forge
mamba activate chembench
pip install anthropic
```

### Generate benchmark tasks

```bash
# Clone the companion data repo (sibling directory)
git clone <mmp-adme-database-url> ../mmp-adme-database

# Generate all 8 task types
python tasks/adme/generate_benchmark.py --mmp-dir ../mmp-adme-database

# Generate specific task type
python tasks/adme/generate_benchmark.py --mmp-dir ../mmp-adme-database --task-type strategic_planning
```

### Run validation

```bash
# Requires ANTHROPIC_API_KEY environment variable
python tasks/adme/run_validation.py --n-per-type 5 --model claude-sonnet-4-6
```

This produces a self-contained HTML report with molecule visualizations, model responses, ground truth, and auto-assessment scores.

## Data

- **14 ADME endpoints**: microsomal clearance, microsomal half-life, hepatocyte clearance, CYP inhibition (3A4, 2C9, 2C19, 2D6) in both binary and potency formats
- **618K+ matched molecular pairs** from ChEMBL, TDC, and BindingDB
- **Transforms with statistical support**: tasks use only transforms observed across multiple molecular contexts, ensuring the expected answer is grounded in population-level SAR, not single-pair noise

## Positioning

| | ChemBench | ether0 (FutureHouse) | ChemIQ | oMeBench |
|---|---|---|---|---|
| Format | Open-ended | MCQ-heavy | Short-answer | Open-ended |
| Multi-step reasoning | Yes (3 task types) | No | No | Partial (mechanisms) |
| Verifiable ground truth | Yes (MMP data) | Yes (reactions) | Yes (expert) | Yes (textbook) |
| ADME/drug design focus | Yes | Minimal | No | No |
| Anti-memorization | Randomized SMILES | Fixed | Fixed | Fixed |
| Strategic reasoning | Yes | No | No | No |

## Repository Structure

```
chembench/
  docs/
    benchmark_design.md    # 4-tier task taxonomy (full vision)
    landscape.md           # Benchmark landscape analysis
  tasks/adme/
    generate_benchmark.py  # Task generator (8 types from MMP data)
    benchmark_tasks.json   # Generated benchmark (1,787 tasks)
    run_validation.py      # Validation harness (LLM + auto-assessment + HTML)
  results/                 # Validation reports (gitignored)
  CLAUDE.md                # Project instructions
```

## References

- **ether0** (FutureHouse, 2024) -- Verifiable chemistry tasks with automated reward; single-step focus
- **ChemIQ** (2024) -- Demonstrated 10x MCQ score inflation; established short-answer as gold standard
- **oMeBench** (2024) -- Process-based evaluation revealing the products/mechanisms gap (90% vs 24%)
- **MolQuest** (2024) -- Agentic/interactive evaluation paradigm for chemistry
- **MMP-ADME Database** -- 618K+ matched molecular pairs across 14 ADME endpoints (companion repo)
