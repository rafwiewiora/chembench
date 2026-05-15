# Paper Outline: ADME Reasoning Benchmark

## Title Ideas

1. **"Rational Optimizers, Not Curious Scientists: How LLMs Fail at Chemical Exploration"**
2. **"ChemBench-ADME: Testing Strategic Reasoning and Scientific Curiosity in Large Language Models"**
3. **"Beyond Property Lookup: A Matched Molecular Pair Benchmark for Multi-Step Chemical Reasoning"**

---

## Abstract (key claims, bullet form)

- LLMs are increasingly deployed for molecular optimization, but existing benchmarks test recall/lookup, not reasoning or exploration behavior
- We introduce ChemBench-ADME: tasks across multiple types that test genuine medicinal chemistry reasoning using matched molecular pairs from ChEMBL
- Our most novel contribution is the **exploration task**: an interactive oracle-based evaluation where models query an MMP database to optimize a compound. Hidden among transforms with predictable population statistics is a single non-additive outlier (z >= 3 sigma) — the key discovery the model must find
- **Key finding**: Both Sonnet and Opus exhibit identical satisficing behavior — they cherry-pick transforms with favorable population means and stop after finding "good enough" results, never exploring transforms that look neutral or unfavorable by population statistics. They are **rational optimizers, not curious scientists.**
- A single sentence in the system prompt ("population statistics can be misleading") flips discovery rate from 0% to 100% on hard tasks — revealing that the failure is prompt sensitivity, not capability
- Models score well on single-step reasoning tasks but degrade on strategic multi-step tasks and exploration
- Open-ended format avoids the ~10x MCQ score inflation demonstrated by ChemIQ

---

## 1. Introduction

**Argument flow:**

- LLMs increasingly used in drug discovery pipelines (molecular generation, property prediction, SAR interpretation)
- Existing chemistry benchmarks primarily test factual recall or single-step transformations
  - ChemBench (LamaLab): MCQ format, no correlation between molecular complexity and accuracy -> memorization signal
  - ether0 (FutureHouse): 640K problems but limited to single-step forward/retro reactions
  - ChemIQ: demonstrated 10x score inflation from MCQ vs. short-answer format
  - oMeBench: 90% correct products but only 24% correct mechanisms -> right answer, wrong reasoning
- **Missing evaluation dimension: scientific curiosity.** Real medicinal chemistry is not just optimization — it requires:
  - Persistence through disappointing results to discover unexpected SAR
  - Recognizing that population statistics can mask molecule-specific non-additive effects
  - Exploring structural modifications that look unpromising on average but may be transformative in context
  - This is the difference between a statistical optimizer and a scientist
- Real medicinal chemistry also requires strategic multi-step reasoning:
  - Accepting worse intermediates to enable better final outcomes (sacrifice moves)
  - Navigating multi-objective tradeoffs (clearance vs. CYP inhibition)
  - Predicting downstream consequences of structural changes
- Our contributions:
  1. **Exploration benchmark**: Interactive oracle-based tasks built on 2,235 non-additive SAR cliffs — transforms that behave as extreme z-score outliers on specific molecules despite well-characterized population statistics. Tests whether models persist through disappointing results or satisfice.
  2. **First evidence of model-independent satisficing**: Sonnet and Opus produce identical exploration patterns — same transforms tested, same order, same failure modes. This is structural, not a capability gap.
  3. **Prompt ablation reveals the mechanism**: One sentence about population statistics being misleading flips 0% → 100% discovery rate, showing the failure is the model's statistical prior, not its chemical reasoning.
  4. **Multi-step strategic reasoning tasks**: 8 task types from single-step prediction to "chess-like" strategic planning, grounded in 618K+ experimentally validated MMP pairs across 14 ADME endpoints.

**Key citations:**
- ChemBench (LamaLab) - doi.org/10.1038/s41557-025-01815-x
- ChemIQ - arxiv.org/abs/2505.07735
- oMeBench - arxiv.org/abs/2510.07731
- ether0 (FutureHouse) - arxiv.org/abs/2506.17238
- MMPT-RAG (Pan/Merck) - arxiv.org/abs/2602.16684
- medchem_moves (Awale/Roche) - doi.org/10.1021/acs.jcim.0c01143
- ACNet (activity cliff prediction) - doi.org/10.1021/acs.jcim.1c00855
- SEISMO (LLM-as-optimizer) - arxiv.org/abs/2502.07773

---

## 2. Related Work

### 2.1 Comprehensive Chemistry Benchmarks
- ChemBench, ChemLLMBench, ChemEval, ChemPro, SciKnowEval, GPQA
- Common limitation: MCQ format, knowledge recall focus, no process evaluation

### 2.2 Reasoning-Focused Benchmarks
- ether0: verifiable molecular rewards, single-step only
- ChemIQ: format innovation (short-answer), exposed MCQ inflation
- oMeBench: first to evaluate mechanisms (not just products)
- ChemCoTBench: showed standard CoT fails for chemistry
- MolQuest: agentic paradigm (model requests data iteratively)
- Synthegy: LLM-guided retrosynthesis with natural language strategy

### 2.3 MMP-Based Tools and Models
- medchem_moves (Awale/Roche 2021): exhaustive MMP extraction, frequency-ranked transform playbook
- MMPT-RAG (Pan/Merck 2026): ChemT5 trained on 800K MMPTs, RAG for analog generation
- Both validate MMPs as the right unit of analysis for medchem reasoning
- Gap: neither tests whether models understand WHY transforms have their effects

### 2.4 Activity Cliff and Non-Additivity Literature
- ACNet: activity cliff prediction as pairwise binary classification (400K+ MMPs). ECFP+MLP dominates — tests pattern recognition, not reasoning
- Non-additive SAR / epistasis in medicinal chemistry: well-characterized phenomenon where the effect of modification X depends on what's at position Y
- Free-Wilson additivity model: additive SAR model that by construction cannot capture epistasis
- Our non-additive cliffs: transforms that are extreme z-score outliers on specific molecules despite being well-characterized in the population — the chemical equivalent of discovering epistatic interactions

### 2.5 LLM-as-Optimizer
- SEISMO and related work: LLMs as black-box optimizers with oracle calls
- Near-optimal in ~50 queries on smooth objectives
- Key difference: smooth objectives reward greedy gradient-following. Our landscape is deceptive — population statistics actively mislead, and the best answer hides behind unpromising averages

### 2.6 Positioning
- We are the first benchmark that:
  - Tests **exploration behavior** (curiosity vs. satisficing) in an interactive setting
  - Uses non-additive SAR to create deceptive optimization landscapes where statistics mislead
  - Tests multi-step strategic reasoning (not just single transforms)
  - Uses open-ended format with hidden quantitative answers
  - Evaluates reasoning quality, not just answer correctness

---

## 3. Benchmark Design

### 3.1 Data Source
- ChEMBL matched molecular pairs via mmp-adme-database
- 618K+ pairs across 14 ADME endpoints (7 continuous, 5 binary + 2 pIC50)
- Endpoints: microsomal CLint, microsomal t1/2, hepatocyte CLint, CYP3A4/2D6/2C9/2C19/1A2 inhibition
- Transform statistics: mean/median/std delta, number of supporting pairs
- Cross-endpoint transform effects for multi-objective tasks

### 3.2 Non-Additive Cliff Discovery
- Searched MMP database for transforms where a specific molecule's delta is >= 3 standard deviations from the population mean for that transform
- Required well-characterized transforms (n >= 10 pairs, pop_std > 0.01) to ensure outlier status is meaningful
- Found **2,235 non-additive cliffs** across all endpoints
- Population statistics completely mask non-additivity: transforms alone have avg z = +0.66 (look normal), in epistatic context avg z = -1.60 (extreme outliers)
- Oracle additive predictions miss by 0.65 log units on average

### 3.3 Task Taxonomy

#### Single-Step Tasks (Tier 1)

| Task Type | N | What it tests | Why existing benchmarks can't |
|-----------|---|---------------|-------------------------------|
| property_delta | ~350 | Predict property change + explain mechanism | ether0 tests prediction only, no explanation required |
| series_completion | ~210 | Interpolate within a congeneric series | Requires integrating SAR trends, not single-pair reasoning |
| transform_ranking | ~140 | Rank modifications by expected improvement | Multi-option comparison with relative reasoning |
| tradeoff_analysis | ~50 | Analyze multi-endpoint effects of one transform | No benchmark tests multi-property reasoning |
| transform_explain | ~240 | Explain WHY a well-supported transform has its effect | Pure reasoning task; oMeBench closest but limited to mechanisms |

[TODO: finalize task counts from benchmark_tasks.json]

#### Multi-Step "Chess" Tasks (Tier 2)

| Task Type | N | What it tests | The "chess" angle |
|-----------|---|---------------|-------------------|
| sacrifice_detection | ~60 | Explain why a worse intermediate was accepted | Post-hoc reasoning about sacrifice moves |
| strategic_planning | ~60 | Choose the best 2-step path (step 2 effects hidden) | Forward planning under uncertainty; greedy trap |
| multi_objective_path | ~80 | Navigate CLint/CYP tradeoffs across endpoints | Multi-objective sequential optimization |

[TODO: finalize counts]

#### Interactive Exploration Tasks (Tier 3)

| Task Type | N | What it tests | The exploration angle |
|-----------|---|---------------|---------------------|
| exploration | 22 (11 well-hidden) | Persistence through non-additive SAR | Model queries oracle, must explore beyond statistics |

### 3.4 Exploration Task Design

**Task structure:**
- Model receives a starting molecule with a measured ADME property value and optimization goal
- Can query an MMP oracle via text-based actions: LIST_TRANSFORMS, APPLY <id>, PROPOSE_COMBINATION <id1> <id2>, DONE
- Oracle returns population statistics (mean ± std, n) for available transforms
- When model applies a transform, oracle returns the actual delta for this specific molecule
- Most transforms behave as population statistics predict (normal z-scores)
- One transform per task is a dramatic outlier (z >= 3) — the non-additive cliff

**Difficulty calibration via outlier hiddenness:**
- Each task scored by where the outlier's population mean ranks among all transforms
- **Well-hidden** (rank > 50th percentile): outlier's pop mean looks neutral or unfavorable — a rational statistics-follower would test it last or skip it entirely
- 11/22 tasks have well-hidden outliers — these are the true persistence tests
- Remaining 11 have outliers with favorable pop means — models find these by default

**Scoring:**
- `found_outlier`: did the model apply the outlier transform?
- `outlier_turn`: how many turns before discovery?
- `gave_up_early`: declared DONE without testing majority of transforms?
- `n_transforms_applied / n_total`: exploration coverage
- `persistence_rate`: fraction of tasks where model didn't give up early

**Oracle design rationale:**
- MMP population statistics are inherently additive — they represent average behavior across diverse molecular contexts
- This is a feature, not a bug: the oracle actively misleads on non-additive cases, making it a genuine test of whether the model reasons beyond population statistics
- Combination queries return additive predictions only (no experimental data), reinforcing the additive prior

### 3.5 Design Principles
- **Open-ended format**: no MCQ, free-text + numeric predictions
- **Hidden quantitative ground truth**: strategic tasks have correct answers from population statistics that the model must derive from reasoning
- **Interactive evaluation**: exploration tasks use multi-turn oracle conversations
- **Statistical grounding**: transforms selected with min N pairs (5-15) for robust ground truth
- **Deceptive landscapes**: exploration tasks where statistics mislead, testing curiosity over satisficing
- **Anti-gaming**: SMILES representations are canonical but tasks require reasoning beyond pattern matching

---

## 4. Experiments

### 4.1 Models Tested
- Claude Sonnet 4.6
- Claude Opus 4.6
- [TODO: GPT-4o, Gemini 2.5 Pro, Llama 4 Maverick]

### 4.2 Evaluation Methodology

#### Exploration tasks
- Multi-turn oracle conversations managed programmatically (fresh agent per task, no prior context)
- Oracle responses computed from task JSON (actual deltas, population stats)
- Scored on discovery rate, coverage, and persistence

#### Static tasks (Tiers 1-2)
- Automated keyword-based scoring: checks for direction correctness, numeric accuracy, reasoning depth
- Limitation: can't distinguish good reasoning from fluent-sounding wrong answers
- Expert evaluation via interactive HTML report with molecule visualizations

### 4.3 Experimental Conditions

#### Exploration tasks
- System prompt: medicinal chemist persona with oracle action instructions
- Max 15 turns per task
- No few-shot examples
- **Ablation**: same tasks with an additional sentence nudging models to explore beyond population statistics

#### Static tasks
- System prompt: expert medicinal chemist persona
- Max tokens: 2000
- Single-shot (no few-shot examples)

---

## 5. Results

### 5.1 The Exploration Finding: Rational Optimizers, Not Curious Scientists

**Setup**: 4 tasks run on both Sonnet and Opus (2 with well-hidden outliers, 2 with easy/obvious outliers)

**Result**: Identical behavior across both models:

| Task | Outlier Hidden? | Sonnet | Opus |
|------|----------------|--------|------|
| 001 (Cl→Br, z=-5.2) | Moderate (44%) | FAIL 3/9 | FAIL 3/9 |
| 002 (remove F, z=-4.5) | Borderline (50%) | PASS 6/6 | PASS 6/6 |
| 003 (cPr→Me, z=-3.9) | Not hidden (40%) | PASS 5/5 | PASS 5/5 |
| 009 (pyr→pyrim, z=-3.8) | Very hidden (83%) | FAIL 2/6 | FAIL 2/6 |

**Key observations:**
1. **Model-independent failure pattern**: Same transforms tested, same order, same stopping point on every task. This is structural behavior, not a capability gap.
2. **Satisficing, not maximizing**: Models pick the top 2-3 transforms by population mean, get "good enough" results, and declare done. They never explore transforms with neutral or positive population means.
3. **Excellent reasoning on what they test**: When a model does encounter a surprising result (e.g., task 002 where the "best" transform disappointed), it pivots intelligently and produces strong mechanistic explanations. The chemical reasoning is not the bottleneck — the exploration strategy is.
4. **Task 003 is a false positive**: The outlier had the most favorable population mean, so any rational optimizer finds it first. Not a persistence test.

**The quote**: "They're rational optimizers, not curious scientists."

### 5.2 The Prompt Ablation: One Sentence Changes Everything

**Added to system prompt:**
> "Population statistics reflect average behavior across many different molecules. Individual molecules can deviate dramatically from these averages due to specific structural context. A transform that looks neutral or unfavorable on average may be transformative for YOUR specific molecule. Do not skip transforms just because their population statistics look unpromising — test broadly before concluding."

**Result (Sonnet):**

| Task | Baseline | With Nudge |
|------|----------|------------|
| 001 | 3/9 tested, FAIL | 9/9 tested, **PASS** |
| 009 | 2/6 tested, FAIL | 6/6 tested, **PASS** |

**What the ablation reveals:**
1. **The failure is prompt sensitivity, not capability.** The model has the chemical reasoning to interpret outlier results — it just needs "permission" to explore beyond statistics.
2. **Cascade effect**: Once the nudged model encounters one deviation from population predictions (even a small one), it becomes more curious about remaining transforms. The Me→Et result (delta -0.161 vs pop mean +0.15) catalyzed full exploration of all "unfavorable" transforms.
3. **Better science**: The nudged model produced richer SAR insights — it discovered that cyclopropyl ring expansion progressively improves clearance (cPr < cBu < cPent < cHex), a trend invisible to the satisficing model that stopped after the halogen swaps.
4. **Implications for benchmark design**: The benchmark should NOT include the nudge — it tests the model's natural disposition. The nudge is an ablation that illuminates the mechanism.

### 5.3 Per-Task-Type Breakdown (Static Tasks)
[TODO: Table -- task types x models, showing avg auto-score]
[TODO: Figure -- radar/spider chart of model capabilities across task types]

**Key expected findings:**
- Single-step tasks (property_delta, transform_explain): models score well (~70-100 auto-score)
- Strategic tasks (strategic_planning, sacrifice_detection): significant degradation
- strategic_planning score drops when step 2 effects are hidden
  - [TODO: quantify -- compare when step 2 is revealed vs hidden]
  - Preliminary signal: models default to greedy first-step selection

### 5.4 The Strategic Planning Gap
- Models that score ~100 on property_delta may score ~74 on strategic_planning [TODO: confirm exact numbers]
- Analysis: models select the "greedy best first step" option instead of the globally optimal path
- Connects to exploration finding: same underlying behavior — follow the best-looking option, don't look ahead

### 5.5 Qualitative Analysis
[TODO: Select 3-5 illustrative failure/success cases]
- **Exploration failure (task 001 baseline)**: Model dismisses Cl→Br as "neutral swap, not useful" on turn 1, never tests it. Misses a -1.94 log unit improvement hiding behind a population mean of 0.00.
- **Exploration success (task 001 nudged)**: Same model tests all 9 transforms, discovers Cl→Br outlier, and produces rich SAR narrative about cyclopropyl metabolic liability
- **Exploration reasoning quality**: Both models generate excellent mechanistic hypotheses when they DO encounter surprises (e.g., defluorination hypothesis in task 002, CYP heme coordination in task 009)

---

## 6. Discussion

### 6.1 What the Benchmark Reveals

**The exploration-exploitation tradeoff in LLM reasoning:**
- LLMs default to exploitation (optimize based on available statistics) over exploration (test everything, look for surprises)
- This mirrors a well-known tradeoff in reinforcement learning and Bayesian optimization, but here it manifests in natural language reasoning about chemistry
- The identical behavior across Sonnet and Opus suggests this is a training-time prior, not a model-specific limitation
- Prompt sensitivity (one sentence flips the outcome) suggests the prior is shallow — the exploration capability exists but is not the default behavior

**Connection to real medicinal chemistry:**
- Activity cliffs and non-additive SAR are among the most important phenomena in drug discovery
- A medicinal chemist who stops testing after finding "good enough" improvements would miss transformative modifications
- The benchmark captures this real-world failure mode: relying on MMP database averages (which are inherently additive) to guide exploration of a non-additive landscape

**The chemical reasoning is not the bottleneck:**
- Models produce excellent mechanistic explanations for surprising results when they encounter them
- The gap is in the exploration strategy, not the scientific reasoning
- This has implications for how LLMs should be deployed: as hypothesis generators that evaluate ALL options, not as optimizers that cherry-pick the statistically best ones

### 6.2 Limitations
- **Scope**: ADME only (no potency, selectivity, toxicity, DMPK) -- intentional for V1
- **Exploration task sample size**: 4 tasks x 2 models for the head-to-head comparison. Need full 11-task evaluation for robust statistics.
- **Single oracle design**: Only MMP population statistics as the oracle. Alternative oracles (QSAR models, pharmacophore filters) might elicit different exploration patterns.
- **Auto-evaluation for static tasks**: Keyword-based scoring is a weak proxy; can't detect incorrect but fluent reasoning
- **Data biases**: ChEMBL publication bias (mainly successful optimizations), endpoint coverage varies
- **MMP limitations**: Transforms assume additivity of property effects; context-dependence is partially captured by std but not fully modeled
- **Ground truth quality**: Population mean delta has inherent variance (captured in std_delta but still noisy)

### 6.3 Future Directions
- **Full exploration evaluation**: Run all 11 well-hidden tasks across multiple models + prompt variants
- **Exploration prompt engineering**: Systematically vary the nudge strength — from no hint, to subtle ("results may surprise you"), to explicit ("test everything"). Find the minimum intervention that flips behavior.
- **Agentic vs. prompted exploration**: Does wrapping the model in an agentic framework (tool use, structured exploration protocols) improve discovery rate without prompt nudging?
- **Non-additive SAR as a training signal**: Could fine-tuning on exploration tasks with non-additive rewards improve the model's natural curiosity?
- **Additional endpoints**: Potency data from ChEMBL target-based assays, selectivity panels, hERG
- **Representation perturbation**: Test with randomized SMILES, SELFIES, IUPAC names to probe memorization vs understanding
- **Temporal OOD**: Train/test split by publication date to test generalization to novel scaffolds
- **Epistasis quartets**: The 36 synergistic quartets (each modification alone worsens, together improves) as a separate "propose combinations" task variant

---

## 7. Conclusion

- Introduced ChemBench-ADME: a benchmark testing strategic reasoning and scientific curiosity in LLMs applied to medicinal chemistry
- Most novel finding: LLMs are **rational optimizers, not curious scientists** — they follow population statistics and satisfice, missing non-additive SAR outliers that a persistent explorer would discover
- This behavior is model-independent (identical across Sonnet and Opus) and prompt-sensitive (one sentence flips it), suggesting a shallow training-time prior rather than a deep capability gap
- The exploration benchmark captures a real-world failure mode: over-reliance on additive MMP statistics in a non-additive landscape
- Models excel at chemical reasoning when forced to explore — the bottleneck is the exploration strategy, not the science
- Released the benchmark, exploration task runner, and non-additive cliff dataset to enable systematic evaluation of exploration behavior in LLMs

---

## Figures Needed
- [TODO] Figure 1: Task taxonomy overview (3 tiers: single-step, strategic, exploration)
- [TODO] Figure 2: Exploration task schematic (oracle interaction loop, example transform list, outlier reveal)
- [TODO] Figure 3: Head-to-head comparison table (Sonnet vs Opus, baseline vs nudge, 4 tasks)
- [TODO] Figure 4: Exploration coverage bar chart (transforms tested / total, by task and condition)
- [TODO] Figure 5: Example exploration transcript (task 001 baseline vs nudged — same model, radically different behavior)
- [TODO] Figure 6: Non-additive cliff statistics (population z-scores vs molecule-specific z-scores, showing how statistics mask outliers)
- [TODO] Figure 7: Per-task-type performance radar chart (static tasks)
- [TODO] Figure 8: Strategic planning greedy trap analysis

## Tables Needed
- [TODO] Table 1: Benchmark statistics (tasks per type, endpoints covered, mean support per transform)
- [TODO] Table 2: Exploration task results (all 4 tasks x 2 models x 2 conditions)
- [TODO] Table 3: Non-additive cliff dataset statistics (n per endpoint, mean |z|, pop mean vs actual delta)
- [TODO] Table 4: Per-task-type model scores (static tasks)
- [TODO] Table 5: Auto-assessment vs expert agreement metrics
