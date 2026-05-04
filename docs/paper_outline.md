# Paper Outline: ADME Reasoning Benchmark

## Title Ideas

1. **"ChemBench-ADME: Testing Strategic Medicinal Chemistry Reasoning in Large Language Models"**
2. **"Beyond Property Lookup: A Matched Molecular Pair Benchmark for Multi-Step Chemical Reasoning"**
3. **"Can LLMs Think Like Medicinal Chemists? A Benchmark for Strategic ADME Optimization"**

---

## Abstract (key claims, bullet form)

- LLMs are increasingly deployed for molecular optimization, but existing benchmarks test recall/lookup, not reasoning
- We introduce ChemBench-ADME: 1,787 tasks across 8 types that test genuine medicinal chemistry reasoning using matched molecular pairs from ChEMBL
- Task types range from single-step property prediction to multi-step strategic planning where greedy optimization fails
- Key finding: models score well on single-step tasks but degrade significantly on strategic multi-step tasks (e.g., strategic_planning accuracy drops when step 2 effects are hidden) [TODO: exact numbers from full evaluation]
- Open-ended format avoids the ~10x MCQ score inflation demonstrated by ChemIQ
- We release the benchmark, generation code, and an expert evaluation framework

---

## 1. Introduction

**Argument flow:**

- LLMs increasingly used in drug discovery pipelines (molecular generation, property prediction, SAR interpretation)
- Existing chemistry benchmarks primarily test factual recall or single-step transformations
  - ChemBench (LamaLab): MCQ format, no correlation between molecular complexity and accuracy -> memorization signal
  - ether0 (FutureHouse): 640K problems but limited to single-step forward/retro reactions
  - ChemIQ: demonstrated 10x score inflation from MCQ vs. short-answer format
  - oMeBench: 90% correct products but only 24% correct mechanisms -> right answer, wrong reasoning
- Real medicinal chemistry requires strategic multi-step reasoning:
  - Accepting worse intermediates to enable better final outcomes (sacrifice moves)
  - Navigating multi-objective tradeoffs (clearance vs. CYP inhibition)
  - Predicting downstream consequences of structural changes
- Our contribution: first benchmark testing multi-step strategic medchem reasoning
  - Built on 618K+ experimentally validated matched molecular pairs across 14 ADME endpoints
  - 8 task types from single-step prediction to "chess-like" strategic planning
  - Open-ended format with quantitative ground truth from population statistics
  - Expert evaluation framework (HTML report with ratings) + automated keyword-based proxy

**Key citations:**
- ChemBench (LamaLab) - doi.org/10.1038/s41557-025-01815-x
- ChemIQ - arxiv.org/abs/2505.07735
- oMeBench - arxiv.org/abs/2510.07731
- ether0 (FutureHouse) - arxiv.org/abs/2506.17238
- MMPT-RAG (Pan/Merck) - arxiv.org/abs/2602.16684
- medchem_moves (Awale/Roche) - doi.org/10.1021/acs.jcim.0c01143

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

### 2.4 Positioning
- We are the first benchmark that:
  - Tests multi-step strategic reasoning (not just single transforms)
  - Uses open-ended format with hidden quantitative answers
  - Evaluates reasoning quality, not just answer correctness
  - Covers the full medchem workflow: predict -> explain -> rank -> strategize

---

## 3. Benchmark Design

### 3.1 Data Source
- ChEMBL matched molecular pairs via mmp-adme-database
- 618K+ pairs across 14 ADME endpoints (7 continuous, 5 binary + 2 pIC50)
- Endpoints: microsomal CLint, microsomal t1/2, hepatocyte CLint, CYP3A4/2D6/2C9/2C19/1A2 inhibition
- Transform statistics: mean/median/std delta, number of supporting pairs
- Cross-endpoint transform effects for multi-objective tasks

### 3.2 Task Taxonomy

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

### 3.3 Design Principles
- **Open-ended format**: no MCQ, free-text + numeric predictions
- **Hidden quantitative ground truth**: strategic tasks have correct answers from population statistics that the model must derive from reasoning
- **Statistical grounding**: transforms selected with min N pairs (5-15) for robust ground truth
- **Medchem workflow fidelity**: tasks mirror real optimization decisions
- **Anti-gaming**: SMILES representations are canonical but tasks require reasoning beyond pattern matching

### 3.4 Task Construction Details

**property_delta**: Select well-supported transforms (>=5 pairs). Present compound A with known value, ask to predict compound B and explain. Ground truth = actual measured value + population mean delta.

**series_completion**: Find congeneric series (4-10 compounds sharing a core). Hold one out. Ask for prediction + SAR trend identification + confidence calibration.

**transform_ranking**: Anchor molecule with 3-6 available transforms. Ask to rank by expected improvement. Ground truth = actual observed deltas.

**tradeoff_analysis**: Transforms with effects across >=3 endpoints. Some beneficial, some detrimental. Ask for unified mechanistic explanation.

**transform_explain**: Highly consistent transforms (>=15 pairs). Present the observed effect, ask for mechanistic explanation + context-dependence + reversal scenarios.

**sacrifice_detection**: A->B->C chains where A->B worsens property but B->C more than compensates. Different positions modified at each step. Ask why the chemist accepted B.

**strategic_planning**: Present molecule with multiple 2-step paths. Best path has a worse first step but better total outcome. Step 2 effects labeled "unknown -- you must predict from structure." Tests whether model falls for greedy trap.

**multi_objective_path**: Transforms that improve CLint but worsen CYP inhibition (or vice versa). Present conflict + optional fixing transform. Ask for mechanistic explanation + navigation strategy.

[TODO: Figure 1 -- task taxonomy diagram]
[TODO: Figure 2 -- example tasks for each type (one panel each)]

---

## 4. Experiments

### 4.1 Models Tested
[TODO: Run on these models]
- Claude Sonnet 4.6
- Claude Opus 4
- Claude Haiku 3.5
- GPT-4o
- Potentially: Gemini 2.5 Pro, Llama 4 Maverick

### 4.2 Evaluation Methodology

#### Automated assessment (proxy)
- Keyword-based scoring: checks for direction correctness, numeric accuracy, reasoning depth
- Limitation: can't distinguish good reasoning from fluent-sounding wrong answers
- Scoring rubric per task type (direction=40pts, value accuracy=30pts, reasoning depth=30pts, etc.)
- Grades: good (>=70), partial (40-69), weak (1-39), poor (0)

#### Expert evaluation (primary)
- Interactive HTML report with molecule visualizations (2D SVGs with MMP highlighting)
- Per-task ratings: response quality (excellent/good/acceptable/poor/wrong) + task quality
- Free-text comments persisted in localStorage, exportable as JSON
- Protocol: 2 expert chemists rate independently, discuss disagreements

### 4.3 Experimental Conditions
- System prompt: expert medicinal chemist persona
- Max tokens: 2000
- Temperature: default (not specified -- deterministic via API)
- Single-shot (no few-shot examples)
- [TODO: decide whether to test with/without few-shot for ablation]

---

## 5. Results

### 5.1 Overall Model Comparison
[TODO: Table -- models x metrics (avg score, % good, % poor)]
[TODO: Figure 3 -- grade distribution bar chart per model]

### 5.2 Per-Task-Type Breakdown
[TODO: Table -- task types x models, showing avg auto-score]
[TODO: Figure 4 -- radar/spider chart of model capabilities across task types]

**Key expected findings:**
- Single-step tasks (property_delta, transform_explain): models score well (~70-100 auto-score)
- Strategic tasks (strategic_planning, sacrifice_detection): significant degradation
- The "chess" hypothesis: strategic_planning score drops when step 2 effects are hidden
  - [TODO: quantify -- compare when step 2 is revealed vs hidden]
  - Preliminary signal: models default to greedy first-step selection

### 5.3 The Strategic Planning Gap
- Models that score ~100 on property_delta may score ~74 on strategic_planning [TODO: confirm exact numbers]
- Analysis: models select the "greedy best first step" option instead of the globally optimal path
- Interpretation: LLMs struggle with look-ahead / positional sacrifice reasoning in chemistry
- [TODO: Figure 5 -- scatter plot of single-step vs multi-step performance per model]

### 5.4 Qualitative Analysis of Model Failures
[TODO: Select 3-5 illustrative failure cases from expert review]
- Category 1: Correct direction, wrong mechanism (fluent but chemically incoherent)
- Category 2: Fails to integrate multi-position effects (treats each modification independently)
- Category 3: Overconfident on uncertain cases (poor calibration)
- Category 4: Pattern matching on SMILES substrings rather than chemical reasoning

### 5.5 Auto-Assessment vs Expert Agreement
[TODO: Compute correlation between keyword-based auto-score and expert ratings]
- Expected: moderate correlation (auto-assessment catches missing elements but misses incorrect reasoning)

---

## 6. Discussion

### 6.1 What the Benchmark Reveals
- LLMs can predict property directions but struggle to explain mechanisms correctly (oMeBench confirmation)
- Multi-step strategic reasoning is a genuine capability gap, not just task difficulty
- The "sacrifice move" pattern is particularly revealing: models understand individual steps but not strategic sequencing
- Implications for deployment: LLMs as medchem assistants may give locally optimal but globally suboptimal suggestions

### 6.2 Limitations
- **Scope**: ADME only (no potency, selectivity, toxicity, DMPK) -- intentional for V1
- **Auto-evaluation**: Keyword-based scoring is a weak proxy; can't detect incorrect but fluent reasoning
- **Data biases**: ChEMBL publication bias (mainly successful optimizations), endpoint coverage varies
- **MMP limitations**: Transforms assume additivity of property effects; context-dependence is partially captured by std but not fully modeled
- **Single-pass evaluation**: No iterative/agentic testing (cf. MolQuest); models can't ask clarifying questions
- **Ground truth quality**: Population mean delta has inherent variance (captured in std_delta but still noisy)

### 6.3 Future Directions
- **Additional endpoints**: Potency data from ChEMBL target-based assays, selectivity panels, hERG
- **LLM-as-judge evaluation**: Replace keyword matching with a calibrated LLM evaluator (but acknowledge circular evaluation risk)
- **Representation perturbation**: Test with randomized SMILES, SELFIES, IUPAC names to probe memorization vs understanding
- **Agentic evaluation**: Allow models to request additional MMP data before answering (MolQuest-style)
- **Temporal OOD**: Train/test split by publication date to test generalization to novel scaffolds
- **Difficulty scaling**: Systematically vary transform support (N pairs), effect size, and number of competing options

---

## 7. Conclusion

- Introduced ChemBench-ADME: first benchmark testing multi-step strategic medicinal chemistry reasoning
- 1,787 tasks across 8 types, grounded in 618K+ experimentally validated MMP pairs
- Key contribution: "chess-like" tasks that expose the gap between pattern matching and genuine strategic reasoning
- Models that excel at single-step prediction fall short on multi-step planning -- a critical limitation for real-world medchem applications
- Released benchmark, generation code, and expert evaluation framework to the community

---

## Supplementary Material (planned)
- Full task type specifications with examples
- Auto-assessment rubric details
- Cross-endpoint transform statistics
- Expert evaluation guidelines
- Additional model outputs and failure cases

---

## Figures Needed
- [TODO] Figure 1: Task taxonomy overview (8 types organized by tier)
- [TODO] Figure 2: Example tasks (one per type, showing prompt structure + molecule visualizations)
- [TODO] Figure 3: Grade distribution across models
- [TODO] Figure 4: Per-task-type performance radar chart
- [TODO] Figure 5: Single-step vs multi-step performance scatter
- [TODO] Figure 6: Strategic planning -- greedy trap analysis (how often models pick greedy vs optimal)

## Tables Needed
- [TODO] Table 1: Benchmark statistics (tasks per type, endpoints covered, mean support per transform)
- [TODO] Table 2: Overall model comparison (avg score, grade distribution)
- [TODO] Table 3: Per-task-type model scores
- [TODO] Table 4: Auto-assessment vs expert agreement metrics
