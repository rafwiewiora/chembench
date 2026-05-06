#!/usr/bin/env python3
"""Extract 'creative leap' patterns from ChEMBL MMP data.

Finds multi-step optimization stories where medchem reasoning is evident:
1. Substituent scans: systematic hypothesis testing at one position
2. Ring bioisostere scans: phenyl → pyridine → pyrimidine etc.
3. Cross-position chains: fixing one site reveals the next bottleneck
4. Iterative refinement: progressive simplification at the same position

Each story has a testable hypothesis and verifiable ground truth from
experimental data. Output is a structured JSON database suitable for
building benchmark tasks.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# ---------------------------------------------------------------------------
# Fragment classification helpers
# ---------------------------------------------------------------------------

HALOGEN_SMARTS = {
    "F": "[F;$([F][*])]",
    "Cl": "[Cl;$([Cl][*])]",
    "Br": "[Br;$([Br][*])]",
    "I": "[I;$([I][*])]",
}

# Common bioisostere groups — fragments that medicinal chemists think of as
# related replacements. Keys are group names, values are SMARTS that match
# the variable fragment.
RING_BIOISOSTERES = {
    "phenyl":     "c1ccccc1",
    "pyridine":   "c1ccncc1",
    "pyridine_2": "c1ccccn1",
    "pyrimidine": "c1ccncn1",
    "pyrazine":   "c1cnccn1",
    "thiophene":  "c1ccsc1",
    "furan":      "c1ccoc1",
    "thiazole":   "c1cscn1",
    "oxazole":    "c1cocn1",
    "imidazole":  "c1cnc[nH]1",
    "pyrazole":   "c1cc[nH]n1",
    "triazole":   "c1cnn[nH]1",
}


def fragment_to_mol(frag_smiles: str) -> Optional[Chem.Mol]:
    """Parse a variable fragment SMILES (contains [*:1]) into an RDKit mol."""
    clean = frag_smiles.replace("[*:1]", "[H]")
    return Chem.MolFromSmiles(clean)


def fragment_fingerprint(frag_smiles: str, radius: int = 2, nbits: int = 1024):
    """Morgan fingerprint of a variable fragment."""
    mol = fragment_to_mol(frag_smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def fragment_similarity(frag1: str, frag2: str) -> float:
    """Tanimoto similarity between two variable fragments."""
    fp1 = fragment_fingerprint(frag1)
    fp2 = fragment_fingerprint(frag2)
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def classify_ring_content(frag_smiles: str) -> Optional[str]:
    """Classify a fragment by its dominant ring system."""
    mol = fragment_to_mol(frag_smiles)
    if mol is None:
        return None
    for name, smarts in RING_BIOISOSTERES.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            return name
    return None


def is_small_fragment(frag_smiles: str, max_heavy: int = 8) -> bool:
    """Check if a fragment is small enough to be a single substituent."""
    mol = fragment_to_mol(frag_smiles)
    if mol is None:
        return False
    return mol.GetNumHeavyAtoms() <= max_heavy


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_endpoint(mmp_dir: str, endpoint: str) -> pd.DataFrame:
    """Load pairs CSV for an endpoint."""
    path = os.path.join(mmp_dir, "data", "mmp_databases", f"{endpoint}_pairs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pairs file at {path}")
    return pd.read_csv(path)


def build_mol_index(df: pd.DataFrame) -> dict:
    """Build mol_id → {smiles, value, cores} index."""
    idx = {}
    for _, row in df.iterrows():
        for side in [1, 2]:
            mid = row[f"mol_id_{side}"]
            smi = row[f"smiles_{side}"]
            val = row[f"value_{side}"]
            if mid not in idx:
                idx[mid] = {"smiles": smi, "value": val, "cores": set()}
            idx[mid]["cores"].add(row["core"])
    return idx


def build_core_groups(df: pd.DataFrame) -> dict:
    """Group molecules by shared core → {core: [{mol_id, smiles, value, variable}]}."""
    groups = defaultdict(list)
    seen = defaultdict(set)
    for _, row in df.iterrows():
        core = row["core"]
        for side in [1, 2]:
            mid = row[f"mol_id_{side}"]
            if mid not in seen[core]:
                seen[core].add(mid)
                groups[core].append({
                    "mol_id": mid,
                    "smiles": row[f"smiles_{side}"],
                    "value": row[f"value_{side}"],
                    "variable": row[f"variable_{side}"],
                })
    return dict(groups)


# ---------------------------------------------------------------------------
# Pattern 1: Substituent scans
# ---------------------------------------------------------------------------

def find_substituent_scans(df: pd.DataFrame, min_compounds: int = 5,
                           max_frag_heavy: int = 8) -> list:
    """Find series where >=min_compounds share a core and differ by small substituents.

    These represent systematic hypothesis testing at one position.
    """
    core_groups = build_core_groups(df)
    stories = []

    for core, members in core_groups.items():
        # Filter to small fragments only
        small_members = [m for m in members if is_small_fragment(m["variable"], max_frag_heavy)]
        if len(small_members) < min_compounds:
            continue

        # Deduplicate by variable fragment
        by_var = {}
        for m in small_members:
            v = m["variable"]
            if v not in by_var or m["value"] < by_var[v]["value"]:
                by_var[v] = m
        if len(by_var) < min_compounds:
            continue

        sorted_members = sorted(by_var.values(), key=lambda m: m["value"])
        best = sorted_members[0]
        worst = sorted_members[-1]
        spread = worst["value"] - best["value"]

        if spread < 0.5:
            continue

        story = {
            "pattern": "substituent_scan",
            "core": core,
            "n_substituents": len(by_var),
            "spread": round(spread, 4),
            "best": {
                "variable": best["variable"],
                "mol_id": best["mol_id"],
                "smiles": best["smiles"],
                "value": round(best["value"], 4),
            },
            "worst": {
                "variable": worst["variable"],
                "mol_id": worst["mol_id"],
                "smiles": worst["smiles"],
                "value": round(worst["value"], 4),
            },
            "all_substituents": [
                {
                    "variable": m["variable"],
                    "mol_id": m["mol_id"],
                    "value": round(m["value"], 4),
                }
                for m in sorted_members
            ],
        }

        # Try to infer hypothesis from the best/worst fragments
        best_ring = classify_ring_content(best["variable"])
        worst_ring = classify_ring_content(worst["variable"])
        if best_ring and worst_ring and best_ring != worst_ring:
            story["hypothesis_hint"] = f"ring_bioisostere: {worst_ring} → {best_ring}"

        stories.append(story)

    stories.sort(key=lambda s: (-s["n_substituents"], -s["spread"]))
    return stories


# ---------------------------------------------------------------------------
# Pattern 2: Ring bioisostere scans
# ---------------------------------------------------------------------------

def find_ring_scans(df: pd.DataFrame, min_rings: int = 3) -> list:
    """Find series where the same core has different ring systems as the variable part."""
    core_groups = build_core_groups(df)
    stories = []

    for core, members in core_groups.items():
        ring_members = []
        for m in members:
            ring_type = classify_ring_content(m["variable"])
            if ring_type:
                ring_members.append({**m, "ring_type": ring_type})

        # Deduplicate by ring type (keep best value)
        by_ring = {}
        for m in ring_members:
            rt = m["ring_type"]
            if rt not in by_ring or m["value"] < by_ring[rt]["value"]:
                by_ring[rt] = m

        if len(by_ring) < min_rings:
            continue

        sorted_rings = sorted(by_ring.values(), key=lambda m: m["value"])
        spread = sorted_rings[-1]["value"] - sorted_rings[0]["value"]
        if spread < 0.3:
            continue

        stories.append({
            "pattern": "ring_bioisostere_scan",
            "core": core,
            "n_ring_types": len(by_ring),
            "spread": round(spread, 4),
            "rings": [
                {
                    "ring_type": m["ring_type"],
                    "variable": m["variable"],
                    "mol_id": m["mol_id"],
                    "value": round(m["value"], 4),
                }
                for m in sorted_rings
            ],
        })

    stories.sort(key=lambda s: (-s["n_ring_types"], -s["spread"]))
    return stories


# ---------------------------------------------------------------------------
# Pattern 3: Cross-position optimization chains
# ---------------------------------------------------------------------------

def find_cross_position_chains(df: pd.DataFrame, min_improvement: float = 0.5) -> list:
    """Find A→B→C chains where step 1 and step 2 modify DIFFERENT positions.

    This captures the "fix one soft spot, reveal the next" pattern.
    """
    adj = defaultdict(list)
    mol_values = {}
    mol_ids = {}

    for _, row in df.iterrows():
        for side in [1, 2]:
            mol_values[row[f"smiles_{side}"]] = row[f"value_{side}"]
            mol_ids[row[f"smiles_{side}"]] = row[f"mol_id_{side}"]

        adj[row["smiles_1"]].append({
            "mol": row["smiles_2"],
            "delta": row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_1"],
            "var_to": row["variable_2"],
        })
        adj[row["smiles_2"]].append({
            "mol": row["smiles_1"],
            "delta": -row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_2"],
            "var_to": row["variable_1"],
        })

    stories = []
    seen_chains = set()

    for mol_a in adj:
        if len(adj[mol_a]) < 3:
            continue

        for e1 in adj[mol_a]:
            mol_b = e1["mol"]
            core_1 = e1["core"]

            for e2 in adj[mol_b]:
                mol_c = e2["mol"]
                if mol_c == mol_a:
                    continue
                core_2 = e2["core"]

                # Different cores = different positions modified
                if core_1 == core_2:
                    continue

                total = e1["delta"] + e2["delta"]
                # Both steps should improve (for CLint, negative delta = improvement)
                if e1["delta"] > 0 or e2["delta"] > 0:
                    # Allow one slightly bad step if total is good
                    if total > -min_improvement:
                        continue
                    # But don't allow a very bad step
                    if e1["delta"] > 0.3 or e2["delta"] > 0.3:
                        continue

                chain_key = tuple(sorted([mol_a, mol_c]))
                if chain_key in seen_chains:
                    continue
                seen_chains.add(chain_key)

                # Compute fragment similarity between the two transforms
                sim = fragment_similarity(e1["var_to"], e2["var_to"])

                stories.append({
                    "pattern": "cross_position_chain",
                    "total_delta": round(total, 4),
                    "step1": {
                        "mol_from": mol_a,
                        "mol_to": mol_b,
                        "mol_id_from": mol_ids.get(mol_a, "?"),
                        "mol_id_to": mol_ids.get(mol_b, "?"),
                        "transform": e1["transform"],
                        "var_from": e1["var_from"],
                        "var_to": e1["var_to"],
                        "delta": round(e1["delta"], 4),
                        "core": core_1,
                    },
                    "step2": {
                        "mol_from": mol_b,
                        "mol_to": mol_c,
                        "mol_id_from": mol_ids.get(mol_b, "?"),
                        "mol_id_to": mol_ids.get(mol_c, "?"),
                        "transform": e2["transform"],
                        "var_from": e2["var_from"],
                        "var_to": e2["var_to"],
                        "delta": round(e2["delta"], 4),
                        "core": core_2,
                    },
                    "transform_similarity": round(sim, 3),
                    "values": {
                        "A": round(mol_values[mol_a], 4),
                        "B": round(mol_values[mol_b], 4),
                        "C": round(mol_values[mol_c], 4),
                    },
                })

    stories.sort(key=lambda s: s["total_delta"])
    return stories


# ---------------------------------------------------------------------------
# Pattern 4: Hypothesis transfer ("if that works then this should work")
# ---------------------------------------------------------------------------

def find_hypothesis_transfers(df: pd.DataFrame, min_improvement: float = 0.3,
                              min_similarity: float = 0.2) -> list:
    """Find cases where a transform at one position inspires a RELATED
    transform at a different position on the same molecule.

    This is the core SAR_brain pattern: "if replacing X with Y at position A
    improved the property, then a similar replacement at position B should
    also help." The similarity is measured by Tanimoto between the new
    fragments introduced at each step.

    Key distinction from cross_position_chains: here we specifically require
    the two transforms to be structurally RELATED (high fragment similarity),
    not just happening at different positions.
    """
    adj = defaultdict(list)
    mol_values = {}
    mol_ids = {}

    for _, row in df.iterrows():
        for side in [1, 2]:
            mol_values[row[f"smiles_{side}"]] = row[f"value_{side}"]
            mol_ids[row[f"smiles_{side}"]] = row[f"mol_id_{side}"]

        adj[row["smiles_1"]].append({
            "mol": row["smiles_2"],
            "delta": row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_1"],
            "var_to": row["variable_2"],
        })
        adj[row["smiles_2"]].append({
            "mol": row["smiles_1"],
            "delta": -row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_2"],
            "var_to": row["variable_1"],
        })

    stories = []
    seen = set()

    for mol_a in adj:
        if len(adj[mol_a]) < 3:
            continue

        for e1 in adj[mol_a]:
            mol_b = e1["mol"]
            core_1 = e1["core"]
            if e1["delta"] > 0.1:
                continue

            for e2 in adj[mol_b]:
                mol_c = e2["mol"]
                if mol_c == mol_a:
                    continue
                core_2 = e2["core"]

                # Must be different positions
                if core_1 == core_2:
                    continue

                if e2["delta"] > 0.1:
                    continue

                chain_key = tuple(sorted([mol_a, mol_c]))
                if chain_key in seen:
                    continue

                # The key signal: are the NEW fragments related?
                # e.g., both introduced a fluorine, both replaced H with Me,
                # both swapped phenyl for pyridine
                sim_new_frags = fragment_similarity(e1["var_to"], e2["var_to"])
                sim_old_frags = fragment_similarity(e1["var_from"], e2["var_from"])

                # Also check: are the transforms themselves related?
                # (same type of change applied at different positions)
                sim_transform = max(sim_new_frags, sim_old_frags)

                if sim_transform < min_similarity:
                    continue

                total = e1["delta"] + e2["delta"]
                if total > -min_improvement:
                    continue

                seen.add(chain_key)

                # Classify the type of transfer
                ring1_from = classify_ring_content(e1["var_from"])
                ring1_to = classify_ring_content(e1["var_to"])
                ring2_from = classify_ring_content(e2["var_from"])
                ring2_to = classify_ring_content(e2["var_to"])

                transfer_type = "fragment_analog"
                if ring1_from and ring1_to and ring2_from and ring2_to:
                    if ring1_to == ring2_to:
                        transfer_type = f"same_ring_swap ({ring1_to})"
                    else:
                        transfer_type = f"related_ring_swap ({ring1_to}/{ring2_to})"
                elif is_small_fragment(e1["var_to"], 3) and is_small_fragment(e2["var_to"], 3):
                    transfer_type = "small_group_analog"

                stories.append({
                    "pattern": "hypothesis_transfer",
                    "transfer_type": transfer_type,
                    "total_delta": round(total, 4),
                    "transform_similarity": round(sim_transform, 3),
                    "step1": {
                        "var_from": e1["var_from"],
                        "var_to": e1["var_to"],
                        "delta": round(e1["delta"], 4),
                        "transform": e1["transform"],
                        "core": core_1,
                        "mol_id_from": mol_ids.get(mol_a, "?"),
                        "mol_id_to": mol_ids.get(mol_b, "?"),
                    },
                    "step2": {
                        "var_from": e2["var_from"],
                        "var_to": e2["var_to"],
                        "delta": round(e2["delta"], 4),
                        "transform": e2["transform"],
                        "core": core_2,
                        "mol_id_from": mol_ids.get(mol_b, "?"),
                        "mol_id_to": mol_ids.get(mol_c, "?"),
                    },
                    "values": {
                        "A": round(mol_values[mol_a], 4),
                        "B": round(mol_values[mol_b], 4),
                        "C": round(mol_values[mol_c], 4),
                    },
                })

    # Score: reward meaningful similarity (0.3-0.8 is more interesting than 1.0),
    # penalize trivial identity matches, weight by improvement magnitude
    def quality_score(s):
        sim = s["transform_similarity"]
        # Sweet spot: moderate similarity = creative analogy, not trivial repeat
        sim_score = sim * (1.0 - 0.5 * max(0, sim - 0.8))  # peaks at 0.8
        delta_score = min(abs(s["total_delta"]), 3.0) / 3.0  # cap at 3 log units
        # Bonus for related ring swaps (most interpretable)
        type_bonus = 0.3 if "related_ring_swap" in s["transfer_type"] else 0.0
        type_bonus += 0.1 if "same_ring_swap" in s["transfer_type"] else 0.0
        return -(sim_score + delta_score + type_bonus)  # negative for sort

    stories.sort(key=quality_score)
    return stories


# ---------------------------------------------------------------------------
# Pattern 5: Iterative refinement at same position (diversity-filtered)
# ---------------------------------------------------------------------------

def find_iterative_refinement(df: pd.DataFrame, min_improvement: float = 0.5,
                              max_per_destination: int = 3) -> list:
    """Find A→B→C chains at the SAME position where each step improves.

    Captures progressive simplification / systematic deepening of a hypothesis.
    Diversity-filtered: max N stories per destination molecule to prevent
    domination by a single super-transform.
    """
    adj = defaultdict(list)
    mol_values = {}
    mol_ids = {}

    for _, row in df.iterrows():
        for side in [1, 2]:
            mol_values[row[f"smiles_{side}"]] = row[f"value_{side}"]
            mol_ids[row[f"smiles_{side}"]] = row[f"mol_id_{side}"]

        adj[row["smiles_1"]].append({
            "mol": row["smiles_2"],
            "delta": row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_1"],
            "var_to": row["variable_2"],
        })
        adj[row["smiles_2"]].append({
            "mol": row["smiles_1"],
            "delta": -row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_2"],
            "var_to": row["variable_1"],
        })

    stories = []
    seen = set()

    for mol_a in adj:
        for e1 in adj[mol_a]:
            mol_b = e1["mol"]
            core_1 = e1["core"]

            for e2 in adj[mol_b]:
                mol_c = e2["mol"]
                if mol_c == mol_a:
                    continue
                core_2 = e2["core"]

                # Same core = same position modified
                if core_1 != core_2:
                    continue

                total = e1["delta"] + e2["delta"]
                if total > -min_improvement:
                    continue

                # Both steps should improve (or at least one strong improvement)
                if e1["delta"] > 0.1 and e2["delta"] > 0.1:
                    continue

                chain_key = (mol_a, mol_b, mol_c)
                if chain_key in seen:
                    continue
                seen.add(chain_key)

                sim_consecutive = fragment_similarity(e1["var_to"], e2["var_to"])
                sim_endpoints = fragment_similarity(e1["var_from"], e2["var_to"])

                stories.append({
                    "pattern": "iterative_refinement",
                    "core": core_1,
                    "total_delta": round(total, 4),
                    "step1": {
                        "var_from": e1["var_from"],
                        "var_to": e1["var_to"],
                        "delta": round(e1["delta"], 4),
                        "mol_id_from": mol_ids.get(mol_a, "?"),
                        "mol_id_to": mol_ids.get(mol_b, "?"),
                    },
                    "step2": {
                        "var_from": e2["var_from"],
                        "var_to": e2["var_to"],
                        "delta": round(e2["delta"], 4),
                        "mol_id_from": mol_ids.get(mol_b, "?"),
                        "mol_id_to": mol_ids.get(mol_c, "?"),
                    },
                    "fragment_similarity_consecutive": round(sim_consecutive, 3),
                    "fragment_similarity_endpoints": round(sim_endpoints, 3),
                    "values": {
                        "A": round(mol_values[mol_a], 4),
                        "B": round(mol_values[mol_b], 4),
                        "C": round(mol_values[mol_c], 4),
                    },
                })

    # Diversity filter: limit per destination mol_id
    from collections import Counter
    dest_counts = Counter()
    filtered = []
    # Sort by quality first (high fragment similarity = more "hypothesis-driven")
    stories.sort(key=lambda s: (-s["fragment_similarity_consecutive"], s["total_delta"]))
    for s in stories:
        dest = s["step2"]["mol_id_to"]
        if dest_counts[dest] < max_per_destination:
            filtered.append(s)
            dest_counts[dest] += 1
    return filtered


# ---------------------------------------------------------------------------
# Pattern 5: Greedy-trap paths (sacrifice required)
# ---------------------------------------------------------------------------

def find_greedy_traps(df: pd.DataFrame, min_gap: float = 0.5) -> list:
    """Find molecules where the greedy best first step leads to a worse
    2-step outcome than a non-greedy first step.

    Unlike current strategic_planning tasks, these use the FULL graph
    neighborhood (not curated 4-option sets). The 'trap' is measured
    as the gap between the best 2-step total from greedy-first vs
    optimal-first.
    """
    adj = defaultdict(list)
    mol_values = {}
    mol_ids = {}

    for _, row in df.iterrows():
        for side in [1, 2]:
            mol_values[row[f"smiles_{side}"]] = row[f"value_{side}"]
            mol_ids[row[f"smiles_{side}"]] = row[f"mol_id_{side}"]

        adj[row["smiles_1"]].append({
            "mol": row["smiles_2"],
            "delta": row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_1"],
            "var_to": row["variable_2"],
        })
        adj[row["smiles_2"]].append({
            "mol": row["smiles_1"],
            "delta": -row["delta"],
            "transform": row["transform"],
            "core": row["core"],
            "var_from": row["variable_2"],
            "var_to": row["variable_1"],
        })

    stories = []

    for mol_a in adj:
        if len(adj[mol_a]) < 5:
            continue

        # Find all 2-step paths
        all_paths = []
        for e1 in adj[mol_a]:
            mol_b = e1["mol"]
            for e2 in adj[mol_b]:
                mol_c = e2["mol"]
                if mol_c == mol_a:
                    continue
                all_paths.append({
                    "mol_b": mol_b,
                    "mol_c": mol_c,
                    "d1": e1["delta"],
                    "d2": e2["delta"],
                    "total": e1["delta"] + e2["delta"],
                    "e1": e1,
                    "e2": e2,
                })

        if len(all_paths) < 5:
            continue

        # Find greedy first step (best single-step delta)
        greedy_edge = min(adj[mol_a], key=lambda e: e["delta"])
        greedy_mol_b = greedy_edge["mol"]

        # Best 2-step from greedy first step
        greedy_paths = [p for p in all_paths if p["mol_b"] == greedy_mol_b]
        if not greedy_paths:
            continue
        greedy_best = min(greedy_paths, key=lambda p: p["total"])

        # Overall optimal 2-step
        optimal = min(all_paths, key=lambda p: p["total"])

        gap = greedy_best["total"] - optimal["total"]
        if gap < min_gap:
            continue

        # The optimal first step should NOT be the greedy step
        if optimal["mol_b"] == greedy_mol_b:
            continue

        stories.append({
            "pattern": "greedy_trap",
            "mol_a": mol_a,
            "mol_id_a": mol_ids.get(mol_a, "?"),
            "value_a": round(mol_values[mol_a], 4),
            "n_neighbors": len(adj[mol_a]),
            "n_2step_paths": len(all_paths),
            "gap": round(gap, 4),
            "optimal_path": {
                "step1_delta": round(optimal["d1"], 4),
                "step2_delta": round(optimal["d2"], 4),
                "total": round(optimal["total"], 4),
                "step1_transform": optimal["e1"]["transform"],
                "step2_transform": optimal["e2"]["transform"],
                "step1_var": f"{optimal['e1']['var_from']} → {optimal['e1']['var_to']}",
                "step2_var": f"{optimal['e2']['var_from']} → {optimal['e2']['var_to']}",
                "mol_b": optimal["mol_b"],
                "mol_c": optimal["mol_c"],
            },
            "greedy_path": {
                "step1_delta": round(greedy_edge["delta"], 4),
                "best_total": round(greedy_best["total"], 4),
                "step1_transform": greedy_edge["transform"],
            },
        })

    stories.sort(key=lambda s: -s["gap"])
    return stories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_all(mmp_dir: str, endpoint: str = "microsomal_clint",
                output_dir: Optional[str] = None) -> dict:
    """Run all pattern extractors and return the combined database."""
    print(f"Loading {endpoint} pairs...")
    df = load_endpoint(mmp_dir, endpoint)
    print(f"  {len(df)} pairs, {df['smiles_1'].nunique() + df['smiles_2'].nunique()} molecule refs")

    results = {}

    print("\n1. Finding substituent scans...")
    scans = find_substituent_scans(df, min_compounds=5)
    print(f"   Found {len(scans)} series (top spread: {scans[0]['spread']:.2f})" if scans else "   None found")
    results["substituent_scans"] = scans

    print("\n2. Finding ring bioisostere scans...")
    rings = find_ring_scans(df, min_rings=3)
    print(f"   Found {len(rings)} series" if rings else "   None found")
    results["ring_bioisostere_scans"] = rings

    print("\n3. Finding cross-position chains...")
    cross = find_cross_position_chains(df, min_improvement=1.0)
    print(f"   Found {len(cross)} chains (best total Δ: {cross[0]['total_delta']:.2f})" if cross else "   None found")
    results["cross_position_chains"] = cross[:500]

    print("\n4. Finding hypothesis transfers (SAR_brain pattern)...")
    transfers = find_hypothesis_transfers(df, min_improvement=0.3, min_similarity=0.2)
    print(f"   Found {len(transfers)} transfers" if transfers else "   None found")
    if transfers:
        from collections import Counter
        type_counts = Counter(t["transfer_type"] for t in transfers)
        for tt, c in type_counts.most_common(5):
            print(f"     {tt}: {c}")
    results["hypothesis_transfers"] = transfers[:1000]

    print("\n5. Finding iterative refinement chains...")
    iterative = find_iterative_refinement(df, min_improvement=1.0, max_per_destination=3)
    print(f"   Found {len(iterative)} chains (diversity-filtered)" if iterative else "   None found")
    results["iterative_refinement"] = iterative[:500]

    print("\n6. Finding greedy-trap paths...")
    traps = find_greedy_traps(df, min_gap=0.5)
    # Diversity filter: limit per step2 transform
    from collections import Counter
    t2_counts = Counter()
    filtered_traps = []
    for t in traps:
        t2 = t["optimal_path"]["step2_transform"]
        if t2_counts[t2] < 5:
            filtered_traps.append(t)
            t2_counts[t2] += 1
    traps = filtered_traps
    print(f"   Found {len(traps)} traps after diversity filter (max gap: {traps[0]['gap']:.2f})" if traps else "   None found")
    results["greedy_traps"] = traps[:500]

    # Summary
    total = sum(len(v) for v in results.values())
    print(f"\n{'='*60}")
    print(f"Total creative leap examples: {total}")
    for k, v in results.items():
        print(f"  {k}: {len(v)}")

    results["metadata"] = {
        "endpoint": endpoint,
        "n_pairs": len(df),
        "n_unique_molecules": len(set(df["smiles_1"]) | set(df["smiles_2"])),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"creative_leaps_{endpoint}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmp-dir", default="../mmp-adme-database",
                        help="Path to mmp-adme-database repo")
    parser.add_argument("--endpoint", default="microsomal_clint",
                        help="ADME endpoint to analyze")
    parser.add_argument("--all-endpoints", action="store_true",
                        help="Run on all available endpoints")
    parser.add_argument("--output-dir", default="creative_leaps",
                        help="Output directory for JSON files")
    args = parser.parse_args()

    if args.all_endpoints:
        db_dir = os.path.join(args.mmp_dir, "data", "mmp_databases")
        endpoints = sorted(set(
            f.replace("_pairs.csv", "")
            for f in os.listdir(db_dir)
            if f.endswith("_pairs.csv") and "mmpdb" not in f
        ))
        print(f"Running on {len(endpoints)} endpoints: {endpoints}\n")
        for ep in endpoints:
            print(f"\n{'='*60}")
            print(f"Endpoint: {ep}")
            print(f"{'='*60}")
            try:
                extract_all(args.mmp_dir, ep, args.output_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        extract_all(args.mmp_dir, args.endpoint, args.output_dir)


if __name__ == "__main__":
    main()
