# Benchmark Design: Chemistry Reasoning Beyond Lookup

## Design Principles

1. **Open-ended answers** — No MCQ. Short-answer, generative molecules, free-text explanations.
2. **Evaluate reasoning chains** — Score intermediate steps, not just final answers.
3. **Representation robustness** — Test with randomized SMILES, equivalent drawings, canonical/non-canonical forms.
4. **Include failure cases** — What won't work? What went wrong? Publication-bias antidote.
5. **Practical/tacit knowledge** — The stuff textbooks don't teach but bench chemists know.
6. **Compositional multi-step** — Strategic planning requiring step composition, not single transforms.
7. **Out-of-distribution by design** — Temporal splits, novel reaction classes, unusual substrates.
8. **Calibrated uncertainty** — Reward "I don't know" over confident wrong answers.

## Task Taxonomy

### Tier 1: Mechanistic & Strategic Reasoning

#### 1.1 Mechanism Elucidation
- **Input**: Reactants + reagents + conditions
- **Output**: Step-by-step mechanism (electron flow, intermediates)
- **Evaluation**: Each elementary step verified; partial credit for correct subpaths
- **Why ether0 can't do this**: Binary reward on final molecule; can't evaluate mechanism quality
- **Inspiration**: oMeBench (but we go beyond organic to organometallic, pericyclic, radical)

#### 1.2 Multi-Step Retrosynthesis
- **Input**: Complex target molecule
- **Output**: Full synthetic route (disconnections, reagents, conditions per step)
- **Evaluation**: Route feasibility (purchasable starting materials, known transforms), strategic quality (convergent vs linear, step economy, protecting group economy)
- **Why ether0 can't do this**: Only single-step retrosynthesis
- **Inspiration**: Synthegy's strategy-aware evaluation

#### 1.3 Selectivity Prediction
- **Input**: Substrate with multiple reactive sites + reagent
- **Output**: Major product + reasoning (steric, electronic, orbital arguments)
- **Evaluation**: Correct product AND correct reasoning category
- **Subtypes**: Regioselectivity, chemoselectivity, stereoselectivity, site-selectivity in complex molecules

### Tier 2: Practical / Tacit Knowledge

#### 2.1 Condition Selection
- **Input**: Desired transformation (substrate → product)
- **Output**: Solvent, catalyst, temperature, base/acid, atmosphere, time
- **Evaluation**: Conditions that would actually work (validated against literature precedent)
- **Subtypes**: Standard conditions, unusual substrates requiring modified conditions, green chemistry alternatives

#### 2.2 Reaction Troubleshooting
- **Input**: Target reaction + procedure + observed outcome (low yield, wrong product, no reaction, decomposition)
- **Output**: Diagnosis + proposed fix
- **Evaluation**: Root cause identification + feasibility of proposed solution
- **Example**: "Suzuki coupling of aryl chloride with phenylboronic acid using Pd(PPh3)4 in THF/H2O at 80°C gave 5% yield. What went wrong?"

#### 2.3 Workup & Purification
- **Input**: Reaction mixture composition (product, byproducts, excess reagents, solvent)
- **Output**: Isolation strategy
- **Evaluation**: Would the proposed workup actually separate the components?
- **Tests practical knowledge**: aqueous washes, extraction, chromatography conditions, crystallization

### Tier 3: Creative / Generative

#### 3.1 Constrained Molecular Design
- **Input**: Multiple simultaneous constraints (activity, selectivity, ADME, synthesizability, novelty)
- **Output**: Molecule + synthetic route + rationale
- **Evaluation**: Constraints satisfied + novelty + synthetic feasibility
- **Goes beyond ether0**: Multiple constraints simultaneously, not single-property prediction

#### 3.2 Bioisosteric Replacement
- **Input**: Lead molecule with identified liability (metabolic, toxicity, selectivity) + SAR data
- **Output**: Proposed replacement + predicted effect on properties
- **Evaluation**: Chemically reasonable replacement + correct property prediction direction

#### 3.3 Predicting Unexpected Reactivity
- **Input**: Proposed reaction on a complex substrate
- **Output**: Predicted side products, competing pathways, functional group incompatibilities
- **Evaluation**: Identification of non-obvious failure modes
- **This is the hardest task**: requires integrating functional group compatibility, conditions, and mechanistic reasoning

### Tier 4: Integration & Judgment

#### 4.1 Route Comparison
- **Input**: Two or more synthetic routes to the same target
- **Output**: Ranking + multi-criteria analysis (yield, cost, safety, scalability, environmental)
- **Evaluation**: Correct ranking + quality of reasoning across dimensions

#### 4.2 Experimental Design
- **Input**: Research question (e.g., "determine the substrate scope of this new catalyst")
- **Output**: Systematic experimental plan
- **Evaluation**: Completeness, efficiency, controls, information gain per experiment

#### 4.3 Calibrated Uncertainty
- **Input**: Mix of answerable and genuinely ambiguous/unknown chemistry questions
- **Output**: Answer + confidence
- **Evaluation**: Brier score; reward appropriate uncertainty, penalize overconfidence
- **Why this matters**: ChemBench showed models at maximum confidence on wrong safety answers

## Evaluation Framework

### For molecular answers
- RDKit canonicalization + exact match (ether0 approach)
- Tanimoto similarity thresholds for partial credit
- Synthesizability scores (SA score, purchasable fragment check)

### For reasoning chains
- Step-level verification against known mechanisms
- Expert-written rubrics for key reasoning elements
- LLM-as-judge with chemistry-expert calibration (but aware of limitations — LLM judges miss domain errors)

### For open-ended / creative tasks
- Multi-dimensional rubrics (feasibility, novelty, completeness, chemical soundness)
- Expert evaluation on a subset for calibration
- Automated proxy metrics where possible (SA score, property predictors, similarity to known solutions)

### Anti-gaming measures
- Randomized SMILES representations
- Temporal OOD splits (train on pre-2020, test on post-2020 reactions)
- Scaffold-based splits (hold out entire reaction classes)
- Representation perturbation testing
