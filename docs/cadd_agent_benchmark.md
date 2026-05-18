# Beyond Single Queries: Benchmarking LLM Agents in Multi-Day Drug Design Campaigns

## Motivation

ChemBench-ADME tests whether LLMs can reason about individual chemical transformations — whether they explore, strategize, and explain. But real medicinal chemistry doesn't happen in isolated queries. It happens in **campaigns**: weeks-long, multi-objective optimization efforts where a team of specialists (computational chemists, data scientists, docking experts, ADME modelers) iterates through cycles of generation, evaluation, triage, and redesign.

Can LLM agents run these campaigns? What capabilities matter? And how do you benchmark something that takes days, involves multiple agents, and requires human judgment about "good chemistry"?

This document proposes a benchmark framework based on empirical observations from deploying multi-agent LLM systems on real generative chemistry campaigns.

## What a CADD Campaign Actually Requires

A typical computer-aided drug design campaign involves:

1. **Generative design**: a model (e.g., REINVENT, diffusion-based, flow-matching) proposes candidate molecules under scoring constraints
2. **Multi-objective scoring**: potency predictions, ADME models, structural filters, shape/pharmacophore constraints — often 15-25 simultaneous objectives
3. **Iterative refinement**: analyzing results, diagnosing failures, adjusting scoring parameters, launching new rounds
4. **Triage and curation**: selecting the best compounds for synthesis, balancing diversity against optimization, applying med-chem judgment
5. **Infrastructure management**: GPU jobs, cluster scheduling, API calls, file management, record-keeping

Steps 1-2 are already automated. Steps 3-5 are where LLM agents can operate — and where the interesting evaluation questions live.

## Observed Capability Dimensions

From deploying multi-agent LLM systems on generative chemistry campaigns, we identified seven distinct capability dimensions that existing benchmarks don't measure:

### 1. Scientific Diagnosis Under Pareto Complexity

When a multi-objective optimization produces unexpected results (e.g., a molecular weight penalty that makes compounds heavier instead of lighter), the agent must reason about **why** the scoring landscape behaves counterintuitively. This requires understanding how correlated objectives interact in Pareto optimization — not just predicting a property value.

**Example task**: Given a Pareto optimization with 20+ objectives, one property is trending in the wrong direction despite strong penalty weight. Diagnose whether the cause is (a) a correlated competing objective, (b) a topology constraint making the property structurally inaccessible, (c) a misconfigured scoring transform, or (d) an optimization dynamics artifact.

**What makes this hard**: The diagnosis requires integrating knowledge of optimization theory, molecular topology, and scoring function design. It's not a chemistry question or a math question — it's both simultaneously.

### 2. Experimental Design With Proper Controls

When a hypothesis needs testing (e.g., "removing scoring component X will fix the property distribution"), the agent must design a controlled experiment: matched conditions, appropriate baselines, cold starts vs warm starts, sufficient run length. We observed LLM agents making experimental design errors that any bench scientist would catch — warm-starting from a biased checkpoint when testing whether a bias exists.

**Example task**: Design an A/B experiment to test whether a specific scoring component causes a particular optimization artifact. Specify: baseline configuration, test configuration, what to hold constant, expected outcome under null and alternative hypotheses, and the decision criterion.

### 3. Knowing When to Stop

In iterative optimization, there's always "one more thing to try." The agent must recognize when a finding is definitive — when multiple experiments with different approaches converge on the same conclusion. We observed a clear capability gap: LLM agents diagnosed problems correctly but kept proposing fixes for a structurally impossible goal (reducing molecular weight below a topology-locked floor), requiring human intervention to close the experimental arc.

**Example task**: Given results from N experiments that varied different parameters, determine whether the evidence supports (a) continued optimization, (b) a fundamental constraint that no parameter change can overcome, or (c) insufficient data to decide. Justify the conclusion.

### 4. Multi-Agent Coordination With Shared State

Real campaigns involve parallel workstreams: generative runs on GPUs, ADME prediction queues, docking jobs, data analysis, record-keeping. The agent must delegate, track, and synthesize across these streams without losing coherence. File naming, deduplication, version tracking, and result merging are unglamorous but critical.

**What makes this hard**: It's not any single task — it's maintaining consistency across dozens of asynchronous operations over days. A small error (e.g., uploading undeduplicated compounds, using stale data) compounds silently.

### 5. Aesthetic and Qualitative Judgment

Medicinal chemists have strong intuitions about what makes a "good" compound beyond numerical scores: synthetic tractability, novelty, structural elegance, similarity to known drug scaffolds. We observed that when asked to curate "beautiful" compounds, LLM agents could generate reasonable thematic frameworks (grouping by structural features, interaction patterns, property profiles) but needed human calibration on project-specific preferences.

**Example task**: Given 500 compounds that all pass numerical filters (potency, ADME, structural alerts), select 20 for synthesis. Explain why each was chosen. Score against expert chemist selections.

### 6. Feature-Level Generalization From Sparse Feedback

When human feedback arrives (e.g., votes on a compound set), the agent must learn **why** certain compounds were preferred — extracting generalizable structural features, not memorizing specific molecules. This is the difference between "chemists liked compound X" and "chemists prefer compact scaffolds with 3D character that maintain key pharmacophore interactions."

**What makes this hard**: The feedback is sparse (dozens of votes, not thousands), noisy (chemists disagree), and implicit (a "dislike" vote doesn't say which feature is the problem).

### 7. Reformulation vs. Perseverance

The hardest capability: recognizing when the problem framing itself is wrong. When optimization within a scaffold hits a structural ceiling, the right move is to propose a different scaffold — not to keep tuning scoring weights. This is analogous to the exploration finding in ChemBench-ADME (satisficing within familiar space), but at the campaign level rather than the transform level.

**What makes this hard**: The agent has invested days of computation in the current approach. All its learned context is within that frame. Proposing "start over with a different scaffold" requires overriding both sunk-cost heuristics and the accumulated context.

## Proposed Benchmark Structure

### Level 1: Operational Competence (no human input needed)

Given a fully specified generative chemistry configuration:
- Launch the run, monitor progress, pull intermediate results
- Detect common failure modes (training collapse, mode collapse, filter starvation)
- Compute standard analyses (property distributions, structural diversity, filter pass rates)

**Evaluation**: automated, against known outcomes.

### Level 2: Scientific Diagnosis (no human input needed)

Given results from a run with a known injected problem:
- Diagnose the root cause from the data
- Distinguish between scoring misconfigurations, optimization artifacts, and structural constraints
- Propose a specific fix

**Evaluation**: automated, against known diagnoses. Dataset constructed by running generative models with deliberately introduced problems (e.g., correlated objectives, topology traps, transform bugs).

### Level 3: Experimental Design (minimal human input)

Given a diagnosis:
- Design a controlled experiment to test a hypothesis
- Specify baselines, controls, decision criteria
- Execute the experiment and interpret results correctly

**Evaluation**: automated for execution quality; expert review for experimental design soundness.

### Level 4: Campaign Navigation (preference oracle)

Run a multi-round generative design campaign with access to a **preference oracle** — pre-collected expert votes on compound quality. The agent must:
- Generate diverse candidates
- Query the oracle efficiently (budget-limited)
- Learn from sparse, noisy feedback
- Adapt the generative strategy across rounds
- Balance exploration (novel chemotypes) against exploitation (optimizing known good scaffolds)

**Evaluation**: final compound quality (MPO scores, structural diversity, oracle preference alignment) at fixed oracle-query budget. Different LLMs get the same oracle — fully comparable.

### Level 5: Full Autonomy (the frontier)

Given only a target profile and a generative model:
- Define the optimization strategy
- Manage the full campaign
- Recognize and respond to structural constraints
- Produce a curated shortlist with scientific rationale

**Evaluation**: expert panel review of the shortlist and the campaign narrative. This is the hardest to standardize but the most meaningful.

## The Decision-Trace Protocol

To control for human variability in Levels 4-5, we propose **decision-trace replay**:

1. A human expert runs the campaign, and every decision point is logged with timestamp and context:
   ```
   t=0h:  "Optimize for ADME + shape, scaffold X, pharmacophore constraint Y"
   t=3h:  "MW distribution is too high — investigate"
   t=6h:  "Use cold start, not warm — need clean comparison"
   t=12h: "Compounds aren't structurally appealing — curate for quality"
   ```

2. The same decision trace is replayed for different LLMs. At each decision point, the LLM receives the human's directive and must execute.

3. **Evaluation axes**:
   - **Efficiency**: how many compute-hours to reach the same conclusion?
   - **Diagnosis quality**: did the agent identify the right root cause?
   - **Execution quality**: were experiments properly controlled?
   - **Compound quality**: given identical human steering, which LLM produces better chemistry?

4. **Sensitivity analysis**: vary the decision trace (vague vs. specific directives) to measure how much the LLM depends on human specificity.

## The Preference Oracle Problem

A key challenge identified empirically: **preference oracles only cover explored space**. If the oracle is built from votes on generated compounds, it biases toward the chemotypes the generative model already produces. The agent that best optimizes the oracle is the one that perseveres in known-good space — exactly the satisficing behavior ChemBench-ADME's exploration tasks reveal at the transform level.

The fix is to evaluate at the **feature level**, not the molecule level:
- The oracle provides votes on specific compounds
- The benchmark scores the agent on preference prediction for **held-out compounds from novel scaffolds** — chemotypes the oracle never saw
- This forces the agent to learn generalizable features ("3D character + low MW + pharmacophore match") rather than memorize ("cyclohexyl linker + triazole head")

This connects directly to ChemBench-ADME's core finding: LLMs are rational optimizers, not curious scientists. The multi-agent campaign benchmark tests whether this limitation persists at the campaign scale — and whether preference feedback can break the satisficing loop.

## What We Actually Built: A Multi-Agent CADD Team

This benchmark proposal isn't theoretical — it emerged from running a real multi-agent system on a real drug design campaign over several weeks. What follows is an honest account of what worked, what broke, and what the human expert actually had to do.

### The Team

The system mirrors how a computational chemistry group actually operates. Each agent is a specialist:

| Agent | Role | Analogy |
|-------|------|---------|
| **Team lead** | Scientific reasoning, experiment design, coordination | Senior computational chemist |
| **Cluster agent** | Manages generative model runs, ADME predictions on HPC | DevOps / pipeline engineer |
| **Docking agent** | Runs and monitors molecular docking campaigns | Docking specialist |
| **Analyst** | Data processing, filtering, structural decomposition, SDF manipulation | Data scientist |
| **Archiver** | Lab notebook, memory management, session continuity | Lab manager |
| **GPU agent** | Structure prediction, 3D analysis | Structural biologist |

The human is the **PI** — setting scientific strategy, making judgment calls about what constitutes good chemistry, deciding when to change direction, and providing the deep domain expertise that no amount of compute can replace.

### What the PI Actually Does

The romantic vision is that you type "design me a drug" and come back to a curated shortlist. The reality is that the human expert works *harder*, not less, because the agents dramatically compress the iteration cycle.

In a traditional campaign, a computational chemist might launch one generative run per day, analyze results the next morning, and iterate. With agents running 24/7 — generating, scoring, docking, predicting ADME, curating — the bottleneck shifts entirely to the human. The agents can produce and analyze thousands of compounds overnight. But every strategic decision still requires a seasoned medicinal chemist:

- **"The molecular weight is too high — why?"** The agents can diagnose that it's a topology constraint vs. a scoring misconfiguration. But deciding whether to accept the constraint or redesign the scaffold requires understanding the target biology, the competitive landscape, and the synthetic feasibility of alternatives.

- **"These compounds all pass the filters but none of them are beautiful."** Numerical scores don't capture synthetic elegance, novelty, or the kind of structural features that make a med-chem team excited. The PI's aesthetic judgment is irreplaceable — and the agents need calibration from it.

- **"We've been optimizing this scaffold for a week. Is it time to try something else?"** The agents will happily run experiments forever. Recognizing that the problem framing itself needs to change — that's a human call that requires integrating business context, timeline pressure, and scientific intuition that isn't in any scoring function.

The experience felt less like "managing automation" and more like directing a team of tireless, brilliant, but sometimes literal-minded postdocs who work around the clock. The PI's job is to keep the scientific narrative coherent, catch when the agents are optimizing the wrong thing, and inject the kind of lateral thinking that comes from years at the bench.

### What Broke (Instructively)

Several failure modes emerged that directly informed the benchmark dimensions above:

1. **Warm-start contamination**: An agent designed an A/B experiment to test whether a scoring component caused a bias — but warm-started from a checkpoint that already contained the bias. A bench scientist would never make this mistake. This drove Dimension 2 (Experimental Design).

2. **The topology trap**: Agents correctly diagnosed that molecular weight was too high and correctly identified the scoring dynamics. Then they spent days proposing fix after fix for a problem that was structurally impossible — the scaffold plus pharmacophore constraints imposed a hard floor on molecular weight that no scoring change could overcome. The human had to step in and declare the experimental arc closed. This drove Dimension 3 (Knowing When to Stop) and Dimension 7 (Reformulation).

3. **Deduplication disasters**: With multiple parallel workstreams generating compounds, the same molecule would appear in multiple batches. Without careful deduplication, thousands of redundant docking calculations were launched. Unglamorous but expensive. This drove Dimension 4 (Multi-Agent Coordination).

4. **The "one more run" problem**: After establishing that a property was topology-locked, an agent proposed "what if we try weight 5.0 instead of 2.0?" — a reasonable-sounding experiment that any seasoned optimizer would know couldn't break a structural constraint. The human's role was to recognize this as sunk-cost reasoning and redirect to productive work. This reinforced Dimension 7 (Reformulation vs. Perseverance).

### The Compression Effect

The most striking observation: a campaign that would traditionally take a small team 2-3 months of calendar time compressed into roughly two weeks of intense human-agent collaboration. But "compressed" is misleading — the human's cognitive load per hour went *up*, not down. The agents eliminated all the waiting (for jobs to finish, for data to process, for analysis to complete) and left only the hard decisions, back to back, at whatever pace the human could sustain.

This has direct implications for benchmark design. Any evaluation that measures only the *agent's* performance without accounting for the quality and intensity of human steering will miss the point. The interesting question isn't "can the agent run a campaign alone?" — it's "how much does the agent amplify an expert, and what's the minimum expertise required to steer it effectively?"

## Relationship to ChemBench-ADME

| | ChemBench-ADME | CADD Agent Benchmark |
|---|---|---|
| **Scope** | Single query / short interaction | Multi-day campaign |
| **Agent count** | 1 | 3-6 specialized agents |
| **Human role** | None (automated scoring) | Preference oracle / decision trace |
| **Key capability** | Exploration vs. satisficing | Diagnosis, experimental design, reformulation |
| **Core finding** | LLMs satisfice at transform level | Do they satisfice at campaign level too? |
| **Data** | Public MMP pairs | Generative model outputs (synthetic, reproducible) |

The two benchmarks are complementary. ChemBench-ADME asks: "Can the LLM find a hidden gem in a database?" The CADD Agent Benchmark asks: "Can the LLM run the campaign that generates the database?"

## Attribution

This benchmark concept emerged from a collaboration between Rafal Wiewiora (medicinal/computational chemist) and Claude (Anthropic), based on empirical observations from deploying multi-agent LLM systems on real drug design campaigns. The specific capability dimensions, failure modes, and benchmark levels described here were identified through iterative experimentation, not theoretical analysis.
