# ChemBench

Chemistry reasoning benchmark for LLMs — focused on tasks that require genuine chemical reasoning, not factual recall.

## Project Philosophy
- Test what a bench chemist knows that a textbook can't teach
- Prioritize open-ended/generative formats over MCQ (ChemIQ showed 10x score inflation with MCQ)
- Evaluate reasoning chains, not just final answers
- Include failure cases and ambiguity — real chemistry is messy
- Design tasks resistant to memorization (representation perturbation, OOD splits)

## Key References
- ether0 (FutureHouse): our primary inspiration for verifiable task design
- ChemIQ: gold standard for format design (short-answer >> MCQ)
- oMeBench: process-based evaluation (mechanisms, not just products)
- MolQuest: agentic/interactive evaluation paradigm

## Data Dependencies
- `mmp-adme-database` repo (sibling dir) provides MMP pair/transform data
  - 618K+ matched molecular pairs across 14 ADME endpoints
  - Data from ChEMBL, TDC, BindingDB

## Repo Structure
- `docs/` — benchmark design docs, landscape analysis
- `tasks/adme/` — ADME reasoning benchmark generator and tasks
  - `generate_benchmark.py` — reads from mmp-adme-database, generates 8 task types
  - `benchmark_tasks.json` — generated benchmark (1,787 tasks)
  - `run_validation.py` — runs tasks through Claude, auto-assesses, generates HTML report
- `results/` — validation reports and artifacts (gitignored)

## Task Types
1. property_delta — predict property change from transform + explain
2. series_completion — predict held-out compound in congeneric series
3. transform_ranking — rank transforms by effect size
4. tradeoff_analysis — analyze multi-endpoint effects
5. transform_explain — explain WHY a transform has its observed effect
6. sacrifice_detection — explain why a worse intermediate was accepted in a 2-step path
7. strategic_planning — choose the best 2-step path (step 2 effects hidden)
8. multi_objective_path — navigate tradeoffs across CLint/CYP endpoints

## Auto-assessment Limitations
Keyword-based auto-scoring (in run_validation.py) is directional only — it catches
missing reasoning elements but can't distinguish good reasoning from fluent-sounding
wrong answers. The HTML report's expert rating system is the intended primary eval.
