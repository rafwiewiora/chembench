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

## Repo Structure
- `docs/` — benchmark design docs, landscape analysis
- `tasks/` — task definitions organized by tier
- `data/` — task data and examples
- `eval/` — evaluation/reward functions
