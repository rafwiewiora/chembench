# Chemistry LLM Benchmark Landscape

## Existing Benchmarks

### Comprehensive
| Benchmark | Year | Size | Format | Key Finding |
|-----------|------|------|--------|-------------|
| ChemBench (LamaLab) | 2025 | 2,700 | MCQ + open | No correlation between molecular complexity and accuracy → memorization |
| ChemLLMBench | 2023 | 8 tasks | Mixed | LLMs "not competitive" on name prediction (8%), retrosynthesis (40% below Chemformer) |
| ChemEval | 2024 | 42 tasks | Mixed | General LLMs excel at lit understanding, fail at advanced chemical knowledge |
| ChemPro | 2026 | 4,100 | MCQ | 21-point drop from elementary to competitive; top models cap at 74-76% |
| SciKnowEval | 2024 | 50K+ | 5 levels | Almost no model scores >3/5 on protocol design |
| GPQA | 2024 | 448 | MCQ | PhD experts at 65-74%; designed to resist lookup |

### Multimodal
| Benchmark | Year | Key Finding |
|-----------|------|-------------|
| USNCO-V | 2025 | Removing images sometimes improves accuracy |
| MaCBench | 2025 | Near-perfect equipment ID, fundamental failures in spatial reasoning |
| RxnBench | 2024 | 96% single-figure, <50% full-document QA |
| ReactBench | 2026 | >30% gap between localization and topology reasoning |

### Reasoning-Focused (Recent)
| Benchmark | Year | Key Innovation |
|-----------|------|---------------|
| ether0 | 2025 | All answers are molecules; RL with verifiable rewards; 640K problems |
| ChemIQ | 2025 | Short-answer format; 10x gap between reasoning/non-reasoning models |
| ChemCoTBench | 2025 | Modular chemical operations; standard CoT distillation fails for chemistry |
| oMeBench | 2025 | Step-level mechanism evaluation; 90% correct products, 24% correct mechanisms |
| MolQuest | 2026 | Agentic: model requests NMR/MS/IR data iteratively |
| Synthegy | 2025 | LLM-guided retrosynthesis search with natural language strategy |

### MMP-Based Models & Tools
| Method | Year | Key Innovation |
|--------|------|---------------|
| medchem_moves (Awale/Roche) | 2021 | Exhaustive MMP extraction from ChEMBL at radius 3; frequency-ranked "playbook" of real medchem transforms |
| MMPT-RAG (Pan/Merck) | 2026 | ChemT5-based foundation model trained on 800K MMPTs from ChEMBL; RAG retrieves reference analogs for controllable generation |

**Relevance to our benchmark:** Both validate MMPs as the right unit of analysis for medchem reasoning. medchem_moves provides transform frequency priors (what chemists actually do); MMPT-RAG generates plausible analogs but can't explain *why* a transform affects a property. Our benchmark fills the reasoning gap — testing whether models understand the mechanistic basis for property changes, not just structural plausibility.

## The Core Problem

Most benchmarks test **knowledge recall**, not **reasoning**:
- MCQ format inflates scores ~10x (ChemIQ)
- Models treat SMILES as strings, not molecular graphs (40% drop with randomized SMILES)
- Correct final answers via chemically incoherent reasoning (oMeBench)
- Training data = published literature = massive publication bias toward successful reactions

## ether0 Deep Dive

### What it covers (18 task types)
- Open-answer: solubility edit, IUPAC name, SMILES completion, functional group, elucidation, retrosynthesis, reaction prediction, molecular caption, molecular formula
- MCQ: safety/GHS, scent, BBB, receptor binding, ADME, solubility, LD50, pKa, photoswitches

### What it doesn't cover (our opportunity)
- Reaction mechanisms (why, not just what)
- Multi-step synthesis planning
- Stereochemistry and 3D reasoning
- Reaction conditions (solvent, temp, catalyst)
- Selectivity prediction
- Troubleshooting / failure diagnosis
- Experimental design (iterative)
- Yield / optimization
- Process / scale-up
- Inorganic / organometallic / catalysis
- Explanation quality evaluation
- Ambiguity handling

### Key lessons from ether0
- Reward function design took ~4 months of iteration
- Model found exploits: adding inert N2 for purchasability, trivial halogen swaps, unstable structures
- "Specifying good reward functions may now be the majority of the effort" — Andrew White
- Emergent reasoning behaviors are task-dependent, not universal

## References
- ChemBench: doi.org/10.1038/s41557-025-01815-x
- ChemLLMBench: openreview.net/forum?id=1ngbR3SZHW
- ChemIQ: arxiv.org/abs/2505.07735
- ether0: arxiv.org/abs/2506.17238
- oMeBench: arxiv.org/abs/2510.07731
- MolQuest: arxiv.org/abs/2603.25253
- ChemCoTBench: arxiv.org/abs/2505.21318
- Synthegy: arxiv.org/abs/2503.08537
- MaCBench: doi.org/10.1038/s43588-025-00836-3
- ReactBench: arxiv.org/abs/2604.15994
- "Challenging Reaction Prediction": arxiv.org/abs/2501.06669
- "Chemical intuition in MOF synthesis": doi.org/10.1038/s41467-019-08483-9
- "AI for Retrosynthesis Needs Expert Knowledge": doi.org/10.1021/jacs.4c00338
- medchem_moves: doi.org/10.1021/acs.jcim.0c01143 (github.com/mahendra-awale/medchem_moves)
- MMPT-RAG: arxiv.org/abs/2602.16684
