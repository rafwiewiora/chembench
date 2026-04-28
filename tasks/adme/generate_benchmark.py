#!/usr/bin/env python
"""
Generate ADME reasoning benchmark tasks from the MMP-ADME database.

Reads MMP pairs, transforms, and cross-endpoint data from the mmp-adme-database
repo and generates benchmark tasks that test LLM chemistry reasoning -- not just
property prediction, but mechanistic explanation, SAR transfer, and multi-endpoint
tradeoff analysis.

Task types:
  1. property_delta    - Predict property change from a structural transform + explain
  2. series_completion - Given a congeneric series, predict the held-out compound
  3. transform_ranking - Rank transforms by expected effect on an endpoint
  4. tradeoff_analysis - Analyze multi-endpoint effects of a transform
  5. transform_explain - Explain WHY a transform has its observed effect

Usage:
    python generate_benchmark.py --mmp-dir /path/to/mmp-adme-database
    python generate_benchmark.py --mmp-dir /path/to/mmp-adme-database --task-type property_delta --n 100
"""

import argparse
import json
import logging
import random
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_META = {
    "microsomal_clint": {
        "label": "Microsomal Intrinsic Clearance",
        "unit": "log CLint (µL/min/mg)",
        "good_direction": "negative",
        "good_label": "improved metabolic stability (lower clearance)",
        "bad_label": "reduced metabolic stability (higher clearance)",
        "property_name": "intrinsic clearance in human liver microsomes",
    },
    "microsomal_t12": {
        "label": "Microsomal Half-life",
        "unit": "log t1/2 (min)",
        "good_direction": "positive",
        "good_label": "increased half-life",
        "bad_label": "decreased half-life",
        "property_name": "half-life in human liver microsomes",
    },
    "hepatocyte_clint": {
        "label": "Hepatocyte Intrinsic Clearance",
        "unit": "log CLint (µL/min/10^6 cells)",
        "good_direction": "negative",
        "good_label": "improved hepatocyte stability",
        "bad_label": "reduced hepatocyte stability",
        "property_name": "intrinsic clearance in human hepatocytes",
    },
    "cyp3a4_binary": {
        "label": "CYP3A4 Inhibition",
        "unit": "binary (0=non-inhibitor, 1=inhibitor)",
        "good_direction": "negative",
        "good_label": "reduced CYP3A4 inhibition",
        "bad_label": "increased CYP3A4 inhibition",
        "property_name": "CYP3A4 inhibition",
    },
    "cyp3a4_pic50": {
        "label": "CYP3A4 Inhibition Potency",
        "unit": "pIC50",
        "good_direction": "negative",
        "good_label": "reduced CYP3A4 inhibition potency",
        "bad_label": "increased CYP3A4 inhibition potency",
        "property_name": "CYP3A4 inhibition potency",
    },
    "cyp1a2_binary": {
        "label": "CYP1A2 Inhibition",
        "unit": "binary (0=non-inhibitor, 1=inhibitor)",
        "good_direction": "negative",
        "good_label": "reduced CYP1A2 inhibition",
        "bad_label": "increased CYP1A2 inhibition",
        "property_name": "CYP1A2 inhibition",
    },
    "cyp2c9_binary": {
        "label": "CYP2C9 Inhibition",
        "unit": "binary (0=non-inhibitor, 1=inhibitor)",
        "good_direction": "negative",
        "good_label": "reduced CYP2C9 inhibition",
        "bad_label": "increased CYP2C9 inhibition",
        "property_name": "CYP2C9 inhibition",
    },
    "cyp2c19_binary": {
        "label": "CYP2C19 Inhibition",
        "unit": "binary (0=non-inhibitor, 1=inhibitor)",
        "good_direction": "negative",
        "good_label": "reduced CYP2C19 inhibition",
        "bad_label": "increased CYP2C19 inhibition",
        "property_name": "CYP2C19 inhibition",
    },
    "cyp2d6_binary": {
        "label": "CYP2D6 Inhibition",
        "unit": "binary (0=non-inhibitor, 1=inhibitor)",
        "good_direction": "negative",
        "good_label": "reduced CYP2D6 inhibition",
        "bad_label": "increased CYP2D6 inhibition",
        "property_name": "CYP2D6 inhibition",
    },
    "cyp2c9_pic50": {
        "label": "CYP2C9 Inhibition Potency",
        "unit": "pIC50",
        "good_direction": "negative",
        "good_label": "reduced CYP2C9 inhibition potency",
        "bad_label": "increased CYP2C9 inhibition potency",
        "property_name": "CYP2C9 inhibition potency",
    },
    "cyp2c19_pic50": {
        "label": "CYP2C19 Inhibition Potency",
        "unit": "pIC50",
        "good_direction": "negative",
        "good_label": "reduced CYP2C19 inhibition potency",
        "bad_label": "increased CYP2C19 inhibition potency",
        "property_name": "CYP2C19 inhibition potency",
    },
    "cyp2d6_pic50": {
        "label": "CYP2D6 Inhibition Potency",
        "unit": "pIC50",
        "good_direction": "negative",
        "good_label": "reduced CYP2D6 inhibition potency",
        "bad_label": "increased CYP2D6 inhibition potency",
        "property_name": "CYP2D6 inhibition potency",
    },
}

# Continuous endpoints for quantitative prediction tasks
CONTINUOUS_ENDPOINTS = [
    "microsomal_clint", "microsomal_t12", "hepatocyte_clint",
    "cyp3a4_pic50", "cyp2c9_pic50", "cyp2c19_pic50", "cyp2d6_pic50",
]


def load_pairs(mmp_dir: Path, endpoint: str) -> pd.DataFrame:
    csv_path = mmp_dir / f"{endpoint}_pairs.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    required = ["smiles_1", "smiles_2", "value_1", "value_2", "delta", "transform",
                 "core", "variable_1", "variable_2"]
    if not all(c in df.columns for c in required):
        logger.warning(f"Missing columns in {csv_path}")
        return pd.DataFrame()
    return df.dropna(subset=["value_1", "value_2", "delta"])


def load_transforms(mmp_dir: Path, endpoint: str) -> pd.DataFrame:
    csv_path = mmp_dir / f"{endpoint}_transforms.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_cross_endpoint(analysis_dir: Path) -> pd.DataFrame:
    csv_path = analysis_dir / "cross_endpoint_transforms.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Task Type 1: Property Delta Prediction
# ---------------------------------------------------------------------------

def generate_property_delta_tasks(mmp_dir, endpoints=None, n_per_endpoint=50,
                                  min_pairs_for_transform=5, seed=42):
    """
    Given compound A with known property, and a structural transform,
    predict the property of compound B and explain why.

    We select pairs where the transform has enough statistical support
    (so the "expected" answer is grounded in data, not a one-off).
    """
    rng = random.Random(seed)
    tasks = []

    if endpoints is None:
        endpoints = CONTINUOUS_ENDPOINTS

    for endpoint in endpoints:
        meta = ENDPOINT_META.get(endpoint)
        if meta is None:
            continue

        pairs_df = load_pairs(mmp_dir, endpoint)
        transforms_df = load_transforms(mmp_dir, endpoint)

        if pairs_df.empty or transforms_df.empty:
            logger.warning(f"No data for {endpoint}, skipping")
            continue

        well_supported = set(
            transforms_df[transforms_df["num_pairs"] >= min_pairs_for_transform]["transform"]
        )
        eligible = pairs_df[pairs_df["transform"].isin(well_supported)].copy()

        if len(eligible) == 0:
            continue

        transform_stats = transforms_df.set_index("transform")[
            ["num_pairs", "mean_delta", "median_delta", "std_delta"]
        ].to_dict("index")

        sampled = eligible.sample(n=min(n_per_endpoint, len(eligible)), random_state=seed)

        for _, row in sampled.iterrows():
            t_stats = transform_stats.get(row["transform"], {})

            task = {
                "task_type": "property_delta",
                "endpoint": endpoint,
                "endpoint_label": meta["label"],
                "prompt": (
                    f"You are given a matched molecular pair for {meta['property_name']}.\n\n"
                    f"Compound A: {row['smiles_1']}\n"
                    f"  {meta['label']}: {row['value_1']:.3f} {meta['unit']}\n\n"
                    f"Compound B: {row['smiles_2']}\n"
                    f"  Compound B differs from A by the structural transformation: "
                    f"{row['variable_1']} → {row['variable_2']}\n"
                    f"  (Core/scaffold: {row['core']})\n\n"
                    f"Questions:\n"
                    f"1. Predict the {meta['label']} of Compound B.\n"
                    f"2. Will this transformation result in {meta['good_label']} or "
                    f"{meta['bad_label']}?\n"
                    f"3. Explain the chemical/mechanistic reasoning behind your prediction. "
                    f"What structural or electronic features drive this change?"
                ),
                "ground_truth": {
                    "value_B": round(row["value_2"], 4),
                    "delta": round(row["delta"], 4),
                    "direction": "beneficial" if (
                        (meta["good_direction"] == "negative" and row["delta"] < 0)
                        or (meta["good_direction"] == "positive" and row["delta"] > 0)
                    ) else "detrimental",
                    "transform": row["transform"],
                    "transform_population_mean_delta": round(t_stats.get("mean_delta", 0), 4),
                    "transform_population_n_pairs": t_stats.get("num_pairs", 0),
                    "transform_population_std": round(t_stats.get("std_delta", 0), 4),
                },
                "evaluation_criteria": {
                    "prediction_accuracy": "Compare predicted value_B to actual (within 0.3 log units = good)",
                    "direction_correct": "Did the model predict the correct direction of change?",
                    "reasoning_elements": [
                        "Identifies relevant metabolic soft spots or blocking groups",
                        "Discusses electronic effects (electron-donating/withdrawing)",
                        "Mentions lipophilicity/logP changes if relevant",
                        "References steric effects if relevant",
                        "Correctly identifies CYP-relevant structural features for CYP tasks",
                    ],
                },
                "metadata": {
                    "smiles_1": row["smiles_1"],
                    "smiles_2": row["smiles_2"],
                    "core": row["core"],
                    "variable_1": row["variable_1"],
                    "variable_2": row["variable_2"],
                },
            }
            tasks.append(task)

    logger.info(f"Generated {len(tasks)} property_delta tasks")
    return tasks


# ---------------------------------------------------------------------------
# Task Type 2: Series Completion
# ---------------------------------------------------------------------------

def generate_series_completion_tasks(mmp_dir, endpoints=None, n_per_endpoint=30,
                                     min_series_size=4, max_series_size=10, seed=42):
    """
    Given a congeneric series (compounds sharing a core with different R-groups
    and measured properties), predict the property of a held-out compound.

    This tests the chemist's ability to interpolate/extrapolate within a series.
    """
    rng = random.Random(seed)
    tasks = []

    if endpoints is None:
        endpoints = CONTINUOUS_ENDPOINTS

    for endpoint in endpoints:
        meta = ENDPOINT_META.get(endpoint)
        if meta is None:
            continue

        pairs_df = load_pairs(mmp_dir, endpoint)
        if pairs_df.empty:
            continue

        cores = pairs_df.groupby("core").agg(
            n_unique_mol1=("smiles_1", "nunique"),
            n_unique_mol2=("smiles_2", "nunique"),
        )
        cores["n_compounds"] = cores["n_unique_mol1"] + cores["n_unique_mol2"]
        eligible_cores = cores[
            (cores["n_compounds"] >= min_series_size)
            & (cores["n_compounds"] <= max_series_size * 2)
        ].index.tolist()

        if not eligible_cores:
            continue

        rng.shuffle(eligible_cores)
        generated = 0

        for core in eligible_cores:
            if generated >= n_per_endpoint:
                break

            core_pairs = pairs_df[pairs_df["core"] == core]

            compounds = {}
            for _, row in core_pairs.iterrows():
                if row["smiles_1"] not in compounds:
                    compounds[row["smiles_1"]] = {
                        "smiles": row["smiles_1"],
                        "value": row["value_1"],
                        "variable": row["variable_1"],
                    }
                if row["smiles_2"] not in compounds:
                    compounds[row["smiles_2"]] = {
                        "smiles": row["smiles_2"],
                        "value": row["value_2"],
                        "variable": row["variable_2"],
                    }

            comp_list = list(compounds.values())
            if len(comp_list) < min_series_size:
                continue

            if len(comp_list) > max_series_size:
                comp_list = rng.sample(comp_list, max_series_size)

            held_out_idx = rng.randrange(len(comp_list))
            held_out = comp_list[held_out_idx]
            context = [c for i, c in enumerate(comp_list) if i != held_out_idx]

            series_text = "\n".join(
                f"  Compound {i+1}: {c['smiles']}\n"
                f"    R-group: {c['variable']}\n"
                f"    {meta['label']}: {c['value']:.3f} {meta['unit']}"
                for i, c in enumerate(context)
            )

            task = {
                "task_type": "series_completion",
                "endpoint": endpoint,
                "endpoint_label": meta["label"],
                "prompt": (
                    f"You are given a congeneric series of compounds that share the same "
                    f"molecular scaffold but differ in their R-group substitution. Each compound "
                    f"has a measured {meta['property_name']} value.\n\n"
                    f"Core scaffold: {core}\n\n"
                    f"Known compounds:\n{series_text}\n\n"
                    f"New compound: {held_out['smiles']}\n"
                    f"  R-group: {held_out['variable']}\n\n"
                    f"Questions:\n"
                    f"1. Predict the {meta['label']} of the new compound.\n"
                    f"2. Explain your reasoning based on the SAR trends in the series.\n"
                    f"3. How confident are you in this prediction? What factors could "
                    f"make it unreliable?"
                ),
                "ground_truth": {
                    "value": round(held_out["value"], 4),
                    "r_group": held_out["variable"],
                    "series_values": [round(c["value"], 4) for c in context],
                    "series_mean": round(np.mean([c["value"] for c in context]), 4),
                    "series_std": round(np.std([c["value"] for c in context]), 4),
                },
                "evaluation_criteria": {
                    "prediction_accuracy": "Within 0.3 log units of actual = good, within 0.5 = acceptable",
                    "sar_reasoning": "Does the model identify trends in the series?",
                    "calibration": "Is the stated confidence appropriate given series variance?",
                    "reasoning_elements": [
                        "Identifies SAR trend from the series data",
                        "Relates structural features of the R-group to the property",
                        "Considers similarity to other R-groups in the series",
                        "Acknowledges uncertainty appropriately",
                    ],
                },
                "metadata": {
                    "core": core,
                    "n_context_compounds": len(context),
                    "held_out_smiles": held_out["smiles"],
                },
            }
            tasks.append(task)
            generated += 1

    logger.info(f"Generated {len(tasks)} series_completion tasks")
    return tasks


# ---------------------------------------------------------------------------
# Task Type 3: Transform Ranking
# ---------------------------------------------------------------------------

def generate_transform_ranking_tasks(mmp_dir, endpoints=None, n_per_endpoint=20,
                                      min_pairs=10, seed=42):
    """
    Given a molecule with an ADME liability and several possible transforms,
    rank the transforms by expected improvement.

    Tests whether the LLM can reason about relative effects of structural changes.
    """
    rng = random.Random(seed)
    tasks = []

    if endpoints is None:
        endpoints = CONTINUOUS_ENDPOINTS

    for endpoint in endpoints:
        meta = ENDPOINT_META.get(endpoint)
        if meta is None:
            continue

        pairs_df = load_pairs(mmp_dir, endpoint)
        transforms_df = load_transforms(mmp_dir, endpoint)
        if pairs_df.empty or transforms_df.empty:
            continue

        robust = transforms_df[transforms_df["num_pairs"] >= min_pairs].copy()
        if len(robust) < 5:
            continue

        generated = 0
        mol_ids_used = set()

        for _ in range(n_per_endpoint * 10):
            if generated >= n_per_endpoint:
                break

            sample_row = pairs_df.sample(1, random_state=rng.randint(0, 1_000_000)).iloc[0]
            anchor_smi = sample_row["smiles_1"]
            anchor_val = sample_row["value_1"]
            anchor_core = sample_row["core"]

            if anchor_smi in mol_ids_used:
                continue

            available = pairs_df[
                (pairs_df["smiles_1"] == anchor_smi)
                & (pairs_df["transform"].isin(set(robust["transform"])))
            ]

            if len(available) < 3:
                continue

            if len(available) > 6:
                available = available.sample(6, random_state=rng.randint(0, 1_000_000))

            mol_ids_used.add(anchor_smi)

            options = []
            for _, r in available.iterrows():
                t_row = robust[robust["transform"] == r["transform"]]
                pop_delta = t_row.iloc[0]["mean_delta"] if len(t_row) > 0 else r["delta"]
                options.append({
                    "transform": r["transform"],
                    "variable_from": r["variable_1"],
                    "variable_to": r["variable_2"],
                    "smiles_result": r["smiles_2"],
                    "actual_delta": round(r["delta"], 4),
                    "population_mean_delta": round(pop_delta, 4),
                })

            if meta["good_direction"] == "negative":
                options.sort(key=lambda x: x["actual_delta"])
            else:
                options.sort(key=lambda x: -x["actual_delta"])
            correct_ranking = [o["transform"] for o in options]

            rng.shuffle(options)

            options_text = "\n".join(
                f"  Option {chr(65+i)}: {o['variable_from']} → {o['variable_to']}"
                for i, o in enumerate(options)
            )

            task = {
                "task_type": "transform_ranking",
                "endpoint": endpoint,
                "endpoint_label": meta["label"],
                "prompt": (
                    f"You have a compound with suboptimal {meta['property_name']}:\n\n"
                    f"Compound: {anchor_smi}\n"
                    f"  {meta['label']}: {anchor_val:.3f} {meta['unit']}\n"
                    f"  Core scaffold: {anchor_core}\n\n"
                    f"You are considering the following R-group modifications:\n"
                    f"{options_text}\n\n"
                    f"Questions:\n"
                    f"1. Rank these modifications from most likely to improve "
                    f"{meta['property_name']} to least likely.\n"
                    f"2. For the top-ranked modification, explain your chemical reasoning.\n"
                    f"3. Are there any modifications that might worsen the property? Which "
                    f"ones, and why?"
                ),
                "ground_truth": {
                    "correct_ranking": correct_ranking,
                    "options": options,
                    "best_transform": correct_ranking[0],
                    "best_actual_delta": options[0]["actual_delta"] if meta["good_direction"] == "negative"
                        else max(o["actual_delta"] for o in options),
                },
                "evaluation_criteria": {
                    "ranking_correlation": "Spearman correlation between predicted and actual ranking",
                    "top1_correct": "Did the model identify the best transform?",
                    "direction_accuracy": "Did it correctly identify beneficial vs detrimental transforms?",
                    "reasoning_elements": [
                        "Discusses structural/electronic rationale for ranking",
                        "Considers metabolic soft spots or CYP binding features",
                        "Correctly identifies detrimental transforms",
                    ],
                },
                "metadata": {
                    "anchor_smiles": anchor_smi,
                    "n_options": len(options),
                },
            }
            tasks.append(task)
            generated += 1

    logger.info(f"Generated {len(tasks)} transform_ranking tasks")
    return tasks


# ---------------------------------------------------------------------------
# Task Type 4: Multi-Endpoint Tradeoff Analysis
# ---------------------------------------------------------------------------

def generate_tradeoff_tasks(mmp_dir, analysis_dir, n=50, min_endpoints=3, seed=42):
    """
    Given a transform with effects across multiple ADME endpoints,
    analyze the tradeoffs.

    Tests multi-objective reasoning -- the hallmark of experienced med chemists.
    """
    rng = random.Random(seed)
    tasks = []

    cross_df = load_cross_endpoint(analysis_dir)
    if cross_df.empty:
        logger.warning("No cross-endpoint data available")
        return tasks

    delta_cols = [c for c in cross_df.columns if c.endswith("_delta")]
    pairs_cols = [c for c in cross_df.columns if c.endswith("_pairs")]

    eligible = cross_df[cross_df["num_endpoints"] >= min_endpoints].copy()
    if eligible.empty:
        logger.warning("No transforms with enough cross-endpoint data")
        return tasks

    sampled = eligible.sample(n=min(n, len(eligible)), random_state=seed)

    for _, row in sampled.iterrows():
        transform = row["transform"]

        effects = []
        for dc in delta_cols:
            if pd.notna(row.get(dc)):
                endpoint_label = dc.replace("_delta", "")
                pc = dc.replace("_delta", "_pairs")
                n_pairs = int(row.get(pc, 0)) if pd.notna(row.get(pc)) else 0

                ep_key = None
                for k, v in ENDPOINT_META.items():
                    if endpoint_label.lower().replace(" ", "_") in k or k in endpoint_label.lower().replace(" ", "_"):
                        ep_key = k
                        break

                direction = "unknown"
                if ep_key and ep_key in ENDPOINT_META:
                    meta = ENDPOINT_META[ep_key]
                    if (meta["good_direction"] == "negative" and row[dc] < -0.05) or \
                       (meta["good_direction"] == "positive" and row[dc] > 0.05):
                        direction = "beneficial"
                    elif (meta["good_direction"] == "negative" and row[dc] > 0.05) or \
                         (meta["good_direction"] == "positive" and row[dc] < -0.05):
                        direction = "detrimental"
                    else:
                        direction = "neutral"

                effects.append({
                    "endpoint": endpoint_label,
                    "delta": round(row[dc], 4),
                    "n_pairs": n_pairs,
                    "direction": direction,
                })

        beneficial = [e for e in effects if e["direction"] == "beneficial"]
        detrimental = [e for e in effects if e["direction"] == "detrimental"]

        effects_text = "\n".join(
            f"  {e['endpoint']}: mean delta = {e['delta']:+.4f} (n={e['n_pairs']} pairs)"
            for e in effects
        )

        task = {
            "task_type": "tradeoff_analysis",
            "prompt": (
                f"The following structural transformation has been observed across "
                f"multiple ADME endpoints:\n\n"
                f"Transformation: {transform}\n\n"
                f"Observed effects:\n{effects_text}\n\n"
                f"Questions:\n"
                f"1. Which ADME properties does this transformation improve, and which "
                f"does it worsen?\n"
                f"2. Provide a unified mechanistic explanation for why this transformation "
                f"has these different effects across endpoints.\n"
                f"3. In what drug design context would you recommend this transformation "
                f"despite the tradeoffs?\n"
                f"4. Can you suggest an alternative transformation that might achieve the "
                f"benefits without the liabilities?"
            ),
            "ground_truth": {
                "transform": transform,
                "effects": effects,
                "n_beneficial": len(beneficial),
                "n_detrimental": len(detrimental),
                "beneficial_endpoints": [e["endpoint"] for e in beneficial],
                "detrimental_endpoints": [e["endpoint"] for e in detrimental],
            },
            "evaluation_criteria": {
                "correct_classification": "Correctly identifies which endpoints improve/worsen",
                "mechanistic_coherence": "Provides a unified explanation (not just per-endpoint)",
                "practical_judgment": "Gives context-appropriate recommendation",
                "alternative_quality": "Proposed alternative is chemically reasonable",
                "reasoning_elements": [
                    "Links structural change to specific metabolic/CYP effects",
                    "Discusses lipophilicity, electronics, or steric effects as unifying theme",
                    "Identifies which liabilities are acceptable vs deal-breakers",
                    "Proposes a chemically distinct alternative (not trivial variation)",
                ],
            },
            "metadata": {
                "n_endpoints": len(effects),
            },
        }
        tasks.append(task)

    logger.info(f"Generated {len(tasks)} tradeoff_analysis tasks")
    return tasks


# ---------------------------------------------------------------------------
# Task Type 5: Transform Explanation
# ---------------------------------------------------------------------------

def generate_explanation_tasks(mmp_dir, endpoints=None, n_per_endpoint=20,
                               min_pairs=15, seed=42):
    """
    Given a well-supported transform and its observed effect, explain WHY.

    This is the purest test of chemical reasoning -- the model must connect
    structure to property through mechanism.
    """
    rng = random.Random(seed)
    tasks = []

    if endpoints is None:
        endpoints = list(ENDPOINT_META.keys())

    for endpoint in endpoints:
        meta = ENDPOINT_META.get(endpoint)
        if meta is None:
            continue

        transforms_df = load_transforms(mmp_dir, endpoint)
        if transforms_df.empty:
            continue

        robust = transforms_df[transforms_df["num_pairs"] >= min_pairs].copy()
        robust["abs_delta"] = robust["mean_delta"].abs()
        robust = robust.sort_values("abs_delta", ascending=False)

        top_transforms = robust.head(min(n_per_endpoint * 2, len(robust)))
        sampled = top_transforms.sample(
            n=min(n_per_endpoint, len(top_transforms)), random_state=seed
        )

        for _, row in sampled.iterrows():
            direction = "beneficial" if (
                (meta["good_direction"] == "negative" and row["mean_delta"] < 0)
                or (meta["good_direction"] == "positive" and row["mean_delta"] > 0)
            ) else "detrimental"

            direction_text = meta["good_label"] if direction == "beneficial" else meta["bad_label"]
            consistency = ""
            if row.get("std_delta", 0) > 0:
                effect_ratio = abs(row["mean_delta"]) / row["std_delta"]
                if effect_ratio > 1.5:
                    consistency = "highly consistent"
                elif effect_ratio > 0.8:
                    consistency = "moderately consistent"
                else:
                    consistency = "variable (context-dependent)"

            task = {
                "task_type": "transform_explain",
                "endpoint": endpoint,
                "endpoint_label": meta["label"],
                "prompt": (
                    f"A matched molecular pair analysis of {row['num_pairs']} compound pairs "
                    f"shows that the structural transformation:\n\n"
                    f"  {row['transform']}\n\n"
                    f"produces a mean change of {row['mean_delta']:+.4f} {meta['unit']} "
                    f"in {meta['property_name']}.\n"
                    f"This corresponds to {direction_text}.\n"
                    f"The effect is {consistency} across the {row['num_pairs']} pairs "
                    f"(std = {row.get('std_delta', 0):.4f}).\n\n"
                    f"Questions:\n"
                    f"1. Explain the chemical/mechanistic basis for this effect.\n"
                    f"2. Under what structural contexts would you expect this effect "
                    f"to be strongest? Weakest?\n"
                    f"3. Can you think of a scenario where this transformation would have "
                    f"the OPPOSITE effect? What would drive that reversal?"
                ),
                "ground_truth": {
                    "transform": row["transform"],
                    "mean_delta": round(row["mean_delta"], 4),
                    "std_delta": round(row.get("std_delta", 0), 4),
                    "n_pairs": row["num_pairs"],
                    "direction": direction,
                    "consistency": consistency,
                },
                "evaluation_criteria": {
                    "mechanistic_accuracy": "Is the explanation chemically correct?",
                    "context_awareness": "Does the model identify when effects would differ?",
                    "reversal_reasoning": "Is the proposed reversal scenario plausible?",
                    "reasoning_elements": [
                        "Correct identification of structural/electronic mechanism",
                        "Links to specific metabolic pathways or CYP binding modes",
                        "Discusses context-dependence (scaffold, substitution pattern)",
                        "Proposes chemically plausible reversal scenario",
                    ],
                },
                "metadata": {
                    "endpoint": endpoint,
                },
            }
            tasks.append(task)

    logger.info(f"Generated {len(tasks)} transform_explain tasks")
    return tasks


# ---------------------------------------------------------------------------
# Task Type 6: Strategic Multi-Step Optimization ("Chess" tasks)
# ---------------------------------------------------------------------------

def _build_mol_graph(mmp_dir, endpoint):
    """Build adjacency graph of molecules connected by MMP transforms."""
    from collections import defaultdict

    pairs_df = load_pairs(mmp_dir, endpoint)
    if pairs_df.empty:
        return {}, {}, pairs_df

    adj = defaultdict(list)
    mol_values = {}

    for _, row in pairs_df.iterrows():
        s1, s2 = row["smiles_1"], row["smiles_2"]
        mol_values[s1] = row["value_1"]
        mol_values[s2] = row["value_2"]
        edge = {
            "transform": row["transform"],
            "delta": row["delta"],
            "core": row["core"],
            "var_from": row["variable_1"],
            "var_to": row["variable_2"],
        }
        adj[s1].append({"mol": s2, **edge})
        adj[s2].append({"mol": s1, "transform": row["transform"],
                         "delta": -row["delta"], "core": row["core"],
                         "var_from": row["variable_2"], "var_to": row["variable_1"]})

    return adj, mol_values, pairs_df


def _find_sacrifice_chains(adj, mol_values, endpoint_meta,
                           min_sacrifice=0.15, min_payoff=0.4, min_net=0.2,
                           max_chains=50000):
    """Find A->B->C chains where step 1 worsens the property but step 2 more
    than compensates, modifying a different position (different core)."""
    good_dir = endpoint_meta["good_direction"]
    chains = []
    seen = set()

    hub_mols = sorted(adj.keys(), key=lambda m: len(adj[m]), reverse=True)[:2000]

    for a in hub_mols:
        if len(chains) >= max_chains:
            break
        for edge_ab in adj[a]:
            b = edge_ab["mol"]
            d_ab = edge_ab["delta"]
            is_sacrifice = (good_dir == "negative" and d_ab > min_sacrifice) or \
                           (good_dir == "positive" and d_ab < -min_sacrifice)
            if not is_sacrifice:
                continue

            for edge_bc in adj[b]:
                c = edge_bc["mol"]
                if c == a:
                    continue
                d_bc = edge_bc["delta"]

                is_payoff = (good_dir == "negative" and d_bc < -min_payoff) or \
                            (good_dir == "positive" and d_bc > min_payoff)
                if not is_payoff:
                    continue

                net = d_ab + d_bc
                is_net_good = (good_dir == "negative" and net < -min_net) or \
                              (good_dir == "positive" and net > min_net)
                if not is_net_good:
                    continue

                if edge_ab["core"] == edge_bc["core"]:
                    continue

                key = (a, b, c)
                if key in seen:
                    continue
                seen.add(key)

                chains.append({
                    "mol_A": a, "mol_B": b, "mol_C": c,
                    "val_A": mol_values[a], "val_B": mol_values[b], "val_C": mol_values[c],
                    "step1": edge_ab, "step2": edge_bc,
                    "step1_delta": d_ab, "step2_delta": d_bc, "net_delta": net,
                })

                if len(chains) >= max_chains:
                    break

    return chains


def generate_sacrifice_detection_tasks(mmp_dir, endpoints=None, n_per_endpoint=20, seed=42):
    """Task: Given a completed A->B->C optimization where step 1 made things
    worse, explain WHY the chemist accepted the intermediate and what step 1
    enables for step 2."""
    rng = random.Random(seed)
    tasks = []

    if endpoints is None:
        endpoints = ["microsomal_clint", "microsomal_t12", "hepatocyte_clint"]

    for endpoint in endpoints:
        meta = ENDPOINT_META.get(endpoint)
        if not meta:
            continue

        adj, mol_values, _ = _build_mol_graph(mmp_dir, endpoint)
        if not adj:
            continue

        chains = _find_sacrifice_chains(adj, mol_values, meta)
        if not chains:
            continue

        chains.sort(key=lambda c: abs(c["net_delta"]), reverse=True)
        selected = chains[:min(n_per_endpoint * 5, len(chains))]
        selected = rng.sample(selected, min(n_per_endpoint, len(selected)))

        for chain in selected:
            s1 = chain["step1"]
            s2 = chain["step2"]
            sacrifice_label = meta["bad_label"]
            payoff_label = meta["good_label"]

            task = {
                "task_type": "sacrifice_detection",
                "endpoint": endpoint,
                "endpoint_label": meta["label"],
                "prompt": (
                    f"A medicinal chemist optimized a compound in two steps for {meta['property_name']}.\n\n"
                    f"Starting compound A: {chain['mol_A']}\n"
                    f"  {meta['label']}: {chain['val_A']:.3f} {meta['unit']}\n\n"
                    f"Step 1 — Transform: {s1['var_from']} → {s1['var_to']}\n"
                    f"  (on scaffold: {s1['core']})\n"
                    f"  Result → Compound B: {chain['mol_B']}\n"
                    f"  {meta['label']}: {chain['val_B']:.3f} {meta['unit']}\n"
                    f"  This step made the property WORSE ({sacrifice_label}).\n\n"
                    f"Step 2 — Transform: {s2['var_from']} → {s2['var_to']}\n"
                    f"  (on scaffold: {s2['core']})\n"
                    f"  Result → Compound C: {chain['mol_C']}\n"
                    f"  {meta['label']}: {chain['val_C']:.3f} {meta['unit']}\n"
                    f"  This step produced a large improvement ({payoff_label}).\n\n"
                    f"Net effect: {meta['label']} went from {chain['val_A']:.3f} to {chain['val_C']:.3f} "
                    f"(net Δ = {chain['net_delta']:+.3f}).\n\n"
                    f"Questions:\n"
                    f"1. Why did the chemist accept the worse intermediate (compound B)? "
                    f"What structural feature did Step 1 introduce that enabled Step 2?\n"
                    f"2. Could the chemist have achieved the same net improvement in a single step? "
                    f"Why or why not?\n"
                    f"3. What does this sequence reveal about the structure-property relationship "
                    f"for {meta['property_name']} in this scaffold?"
                ),
                "ground_truth": {
                    "mol_A": chain["mol_A"],
                    "mol_B": chain["mol_B"],
                    "mol_C": chain["mol_C"],
                    "val_A": round(chain["val_A"], 4),
                    "val_B": round(chain["val_B"], 4),
                    "val_C": round(chain["val_C"], 4),
                    "step1_transform": s1["transform"],
                    "step2_transform": s2["transform"],
                    "step1_delta": round(chain["step1_delta"], 4),
                    "step2_delta": round(chain["step2_delta"], 4),
                    "net_delta": round(chain["net_delta"], 4),
                    "step1_core": s1["core"],
                    "step2_core": s2["core"],
                },
                "evaluation_criteria": {
                    "enablement_reasoning": "Does the model explain how step 1 enables step 2?",
                    "structural_analysis": "Does the model identify the key structural features?",
                    "single_step_analysis": "Does the model correctly assess if a single step could work?",
                    "reasoning_elements": [
                        "Identifies that different positions are modified in each step",
                        "Explains how step 1's structural change creates a context for step 2",
                        "Discusses electronic/steric/metabolic consequences of each change",
                        "Recognizes that sequential optimization is sometimes necessary",
                        "Considers whether the intermediate scaffold allows the second modification",
                    ],
                },
                "metadata": {
                    "endpoint": endpoint,
                    "n_steps": 2,
                },
            }
            tasks.append(task)

    logger.info(f"Generated {len(tasks)} sacrifice_detection tasks")
    return tasks


def generate_strategic_planning_tasks(mmp_dir, endpoints=None, n_per_endpoint=20, seed=42):
    """Task: Given molecule A with a property problem, a set of possible first-move
    transforms, and their downstream options, plan the best 2-step optimization.
    Some first moves look worse but enable better second moves (the chess analogy)."""
    rng = random.Random(seed)
    tasks = []

    if endpoints is None:
        endpoints = ["microsomal_clint", "microsomal_t12", "hepatocyte_clint"]

    for endpoint in endpoints:
        meta = ENDPOINT_META.get(endpoint)
        if not meta:
            continue

        adj, mol_values, _ = _build_mol_graph(mmp_dir, endpoint)
        if not adj:
            continue

        good_dir = meta["good_direction"]
        candidates = []

        mol_list = sorted(adj.keys(), key=lambda m: len(adj[m]), reverse=True)[:3000]
        for a in mol_list:
            neighbors = adj[a]
            if len(neighbors) < 3:
                continue

            paths = []
            for edge_ab in neighbors:
                b = edge_ab["mol"]
                best_bc = None
                for edge_bc in adj[b]:
                    c = edge_bc["mol"]
                    if c == a:
                        continue
                    if edge_ab["core"] == edge_bc["core"]:
                        continue
                    total = edge_ab["delta"] + edge_bc["delta"]
                    if best_bc is None or (
                        (good_dir == "negative" and total < best_bc["total"]) or
                        (good_dir == "positive" and total > best_bc["total"])
                    ):
                        best_bc = {"edge_bc": edge_bc, "total": total, "mol_C": c}

                if best_bc is not None:
                    paths.append({
                        "edge_ab": edge_ab, "mol_B": b,
                        "edge_bc": best_bc["edge_bc"], "mol_C": best_bc["mol_C"],
                        "step1_delta": edge_ab["delta"],
                        "best_total": best_bc["total"],
                    })

            if len(paths) < 3:
                continue

            paths.sort(key=lambda p: p["best_total"],
                       reverse=(good_dir == "positive"))

            best_path = paths[0]
            greedy_paths = sorted(paths, key=lambda p: p["step1_delta"],
                                  reverse=(good_dir == "positive"))
            greedy_best = greedy_paths[0]

            if best_path is greedy_best:
                continue

            is_sacrifice = (good_dir == "negative" and best_path["step1_delta"] > 0.1) or \
                           (good_dir == "positive" and best_path["step1_delta"] < -0.1)
            if not is_sacrifice:
                continue

            improvement = abs(best_path["best_total"] - greedy_best["best_total"])
            if improvement < 0.3:
                continue

            distractors = rng.sample(paths[1:min(6, len(paths))], min(2, len(paths)-1))

            candidates.append({
                "mol_A": a, "val_A": mol_values[a],
                "best_path": best_path, "greedy_path": greedy_best,
                "distractors": distractors, "improvement": improvement,
            })

        candidates.sort(key=lambda c: c["improvement"], reverse=True)
        selected = candidates[:min(n_per_endpoint * 3, len(candidates))]
        selected = rng.sample(selected, min(n_per_endpoint, len(selected)))

        for cand in selected:
            options = [cand["best_path"], cand["greedy_path"]] + cand["distractors"]
            rng.shuffle(options)

            options_text = ""
            for j, opt in enumerate(options):
                ab = opt["edge_ab"]
                bc = opt["edge_bc"]
                b_val = mol_values[opt["mol_B"]]
                c_val = mol_values[opt["mol_C"]]
                options_text += (
                    f"\nOption {chr(65+j)}:\n"
                    f"  Step 1: {ab['var_from']} → {ab['var_to']}  "
                    f"(Δ{meta['label']} = {opt['step1_delta']:+.3f})\n"
                    f"  Step 2: {bc['var_from']} → {bc['var_to']}  "
                    f"(Δ{meta['label']} = {bc['delta']:+.3f})\n"
                    f"  Net: Δ = {opt['best_total']:+.3f}\n"
                )

            best_idx = options.index(cand["best_path"])
            greedy_idx = options.index(cand["greedy_path"])

            task = {
                "task_type": "strategic_planning",
                "endpoint": endpoint,
                "endpoint_label": meta["label"],
                "prompt": (
                    f"You are optimizing a compound for {meta['property_name']}.\n\n"
                    f"Starting compound: {cand['mol_A']}\n"
                    f"  Current {meta['label']}: {cand['val_A']:.3f} {meta['unit']}\n"
                    f"  Goal: {meta['good_label']}\n\n"
                    f"You can make modifications at two different positions on the scaffold. "
                    f"Below are {len(options)} possible 2-step optimization paths, each "
                    f"modifying a different combination of positions.\n"
                    f"{options_text}\n"
                    f"Questions:\n"
                    f"1. Which option gives the best overall outcome? Why?\n"
                    f"2. Option {chr(65+greedy_idx)} has the best first step. Why might the "
                    f"greedy (best-first-step) strategy not lead to the best overall result?\n"
                    f"3. Explain the chemical reasoning behind why the optimal path works — "
                    f"what structural or electronic interplay between the two modifications "
                    f"drives the synergy?"
                ),
                "ground_truth": {
                    "best_option": chr(65 + best_idx),
                    "greedy_option": chr(65 + greedy_idx),
                    "best_net_delta": round(cand["best_path"]["best_total"], 4),
                    "greedy_net_delta": round(cand["greedy_path"]["best_total"], 4),
                    "improvement_over_greedy": round(cand["improvement"], 4),
                    "best_mol_B": cand["best_path"]["mol_B"],
                    "best_mol_C": cand["best_path"]["mol_C"],
                    "best_step1_transform": cand["best_path"]["edge_ab"]["transform"],
                    "best_step2_transform": cand["best_path"]["edge_bc"]["transform"],
                },
                "evaluation_criteria": {
                    "correct_choice": "Does the model select the globally optimal path?",
                    "greedy_trap": "Does the model recognize why greedy fails?",
                    "synergy_reasoning": "Does the model explain inter-position synergy?",
                    "reasoning_elements": [
                        "Identifies the globally optimal option (not the greedy one)",
                        "Explains why the best first step doesn't lead to the best outcome",
                        "Discusses how modifications at different positions interact",
                        "References electronic, steric, or metabolic interplay",
                        "Demonstrates strategic multi-step thinking",
                    ],
                },
                "metadata": {
                    "endpoint": endpoint,
                    "n_options": len(options),
                    "n_steps": 2,
                },
            }
            tasks.append(task)

    logger.info(f"Generated {len(tasks)} strategic_planning tasks")
    return tasks


def generate_multi_objective_tasks(mmp_dir, endpoints=None, n_tasks=40, seed=42):
    """Task: Reason about transform-level tradeoffs across ADME endpoints.
    Uses transform statistics (not molecule-level matching) since compounds
    rarely have data across multiple endpoints.

    Presents pairs of transforms with opposing effects on two properties and
    asks the model to reason about the mechanistic basis and how to navigate."""
    rng = random.Random(seed)
    tasks = []

    endpoint_pairs = [
        ("microsomal_clint", "cyp3a4_binary"),
        ("microsomal_clint", "cyp2d6_binary"),
        ("microsomal_clint", "cyp2c9_binary"),
        ("microsomal_clint", "cyp2c19_binary"),
        ("hepatocyte_clint", "cyp3a4_binary"),
        ("microsomal_t12", "cyp3a4_binary"),
    ]
    if endpoints:
        endpoint_pairs = [(a, b) for a, b in endpoint_pairs
                          if a in endpoints or b in endpoints]

    for ep1, ep2 in endpoint_pairs:
        meta1 = ENDPOINT_META.get(ep1)
        meta2 = ENDPOINT_META.get(ep2)
        if not meta1 or not meta2:
            continue

        t1 = load_transforms(mmp_dir, ep1)
        t2 = load_transforms(mmp_dir, ep2)
        if t1.empty or t2.empty:
            continue

        t1_supp = t1[t1["num_pairs"] >= 5].set_index("transform")
        t2_supp = t2[t2["num_pairs"] >= 5].set_index("transform")
        shared = set(t1_supp.index) & set(t2_supp.index)

        if not shared:
            continue

        good1 = meta1["good_direction"]
        good2 = meta2["good_direction"]

        conflicting = []
        for t in shared:
            d1 = t1_supp.loc[t, "mean_delta"]
            d2 = t2_supp.loc[t, "mean_delta"]
            n1 = t1_supp.loc[t, "num_pairs"]
            n2 = t2_supp.loc[t, "num_pairs"]

            good_for_1 = (good1 == "negative" and d1 < -0.1) or (good1 == "positive" and d1 > 0.1)
            bad_for_2 = (good2 == "negative" and d2 > 0.08) or (good2 == "positive" and d2 < -0.08)

            if good_for_1 and bad_for_2:
                conflicting.append({
                    "transform": t, "d1": d1, "d2": d2, "n1": n1, "n2": n2,
                    "conflict_score": abs(d1) + abs(d2),
                })

        if not conflicting:
            continue

        fixing = []
        for t in shared:
            d2 = t2_supp.loc[t, "mean_delta"]
            d1 = t1_supp.loc[t, "mean_delta"]
            n2 = t2_supp.loc[t, "num_pairs"]

            good_for_2 = (good2 == "negative" and d2 < -0.1) or (good2 == "positive" and d2 > 0.1)
            neutral_for_1 = abs(d1) < 0.15

            if good_for_2 and neutral_for_1:
                fixing.append({
                    "transform": t, "d1": d1, "d2": d2,
                    "n1": t1_supp.loc[t, "num_pairs"], "n2": n2,
                })

        conflicting.sort(key=lambda x: x["conflict_score"], reverse=True)

        n_per = max(1, n_tasks // len(endpoint_pairs))
        for conf in conflicting[:n_per]:
            fix = rng.choice(fixing) if fixing else None

            step2_text = ""
            step2_gt = {}
            if fix:
                fix_dir = meta2["good_label"]
                step2_text = (
                    f"\nA second transform is available that could fix {meta2['property_name']} "
                    f"without hurting {meta1['property_name']}:\n"
                    f"  Transform 2: {fix['transform']}\n"
                    f"  Mean Δ {meta1['label']}: {fix['d1']:+.4f} (n={fix['n1']} pairs)\n"
                    f"  Mean Δ {meta2['label']}: {fix['d2']:+.4f} (n={fix['n2']} pairs)\n\n"
                    f"Question 4: If you applied Transform 1 first and Transform 2 second "
                    f"(at a different position), would this be a viable 2-step optimization "
                    f"strategy? What are the risks?\n"
                )
                step2_gt = {
                    "fix_transform": fix["transform"],
                    "fix_d1": round(float(fix["d1"]), 4),
                    "fix_d2": round(float(fix["d2"]), 4),
                    "fix_n1": int(fix["n1"]),
                    "fix_n2": int(fix["n2"]),
                }

            task = {
                "task_type": "multi_objective_path",
                "endpoint": f"{ep1}+{ep2}",
                "endpoint_label": f"{meta1['label']} vs {meta2['label']}",
                "prompt": (
                    f"A matched molecular pair analysis reveals a tradeoff between two ADME properties.\n\n"
                    f"Transform 1: {conf['transform']}\n"
                    f"  Mean Δ {meta1['label']}: {conf['d1']:+.4f} {meta1['unit']} "
                    f"(n={conf['n1']} pairs) — {meta1['good_label']}\n"
                    f"  Mean Δ {meta2['label']}: {conf['d2']:+.4f} {meta2['unit']} "
                    f"(n={conf['n2']} pairs) — {meta2['bad_label']}\n\n"
                    f"Questions:\n"
                    f"1. Explain the mechanistic basis for why this transform improves "
                    f"{meta1['property_name']} but worsens {meta2['property_name']}.\n"
                    f"2. In what structural contexts would you expect this tradeoff to be "
                    f"most severe? When might both properties improve together?\n"
                    f"3. How would a medicinal chemist navigate this tradeoff in a real "
                    f"optimization campaign?\n"
                    f"{step2_text}"
                ),
                "ground_truth": {
                    "conflict_transform": conf["transform"],
                    "d1": round(float(conf["d1"]), 4), "d2": round(float(conf["d2"]), 4),
                    "n1": int(conf["n1"]), "n2": int(conf["n2"]),
                    "ep1": ep1, "ep2": ep2,
                    **step2_gt,
                },
                "evaluation_criteria": {
                    "tradeoff_mechanism": "Does the model explain WHY the two properties conflict?",
                    "context_awareness": "Does the model identify when the tradeoff is worst/best?",
                    "mpo_strategy": "Does the model propose a viable multi-parameter strategy?",
                    "reasoning_elements": [
                        "Explains mechanistic basis of the property conflict",
                        "Identifies specific structural/electronic features driving each effect",
                        "Discusses context-dependence of the tradeoff",
                        "Proposes practical medchem strategies (orthogonal modifications, scaffolding)",
                        "If 2-step path given: evaluates viability and risks of sequential approach",
                    ],
                },
                "metadata": {
                    "endpoint_1": ep1,
                    "endpoint_2": ep2,
                    "has_fix_transform": fix is not None,
                },
            }
            tasks.append(task)

    logger.info(f"Generated {len(tasks)} multi_objective_path tasks")
    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ADME reasoning benchmark tasks")
    parser.add_argument("--mmp-dir", type=str,
                        default="/Users/rafalwiewiora/repos/mmp-adme-database",
                        help="Path to mmp-adme-database repo")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: benchmark_tasks.json in script dir)")
    parser.add_argument("--task-type", type=str, default="all",
                        choices=["all", "property_delta", "series_completion",
                                 "transform_ranking", "tradeoff_analysis",
                                 "transform_explain", "sacrifice_detection",
                                 "strategic_planning", "multi_objective_path"],
                        help="Type of tasks to generate")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of tasks per endpoint per type")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mmp_base = Path(args.mmp_dir)
    mmp_dir = mmp_base / "data" / "mmp_databases"
    analysis_dir = mmp_base / "analysis"

    if not mmp_dir.exists():
        logger.error(f"MMP database directory not found: {mmp_dir}")
        return

    output_path = Path(args.output) if args.output else Path(__file__).parent / "benchmark_tasks.json"

    all_tasks = []

    if args.task_type in ("all", "property_delta"):
        all_tasks.extend(generate_property_delta_tasks(
            mmp_dir, n_per_endpoint=args.n, seed=args.seed))

    if args.task_type in ("all", "series_completion"):
        all_tasks.extend(generate_series_completion_tasks(
            mmp_dir, n_per_endpoint=args.n, seed=args.seed))

    if args.task_type in ("all", "transform_ranking"):
        all_tasks.extend(generate_transform_ranking_tasks(
            mmp_dir, n_per_endpoint=args.n, seed=args.seed))

    if args.task_type in ("all", "tradeoff_analysis"):
        all_tasks.extend(generate_tradeoff_tasks(
            mmp_dir, analysis_dir, n=args.n * 2, seed=args.seed))

    if args.task_type in ("all", "transform_explain"):
        all_tasks.extend(generate_explanation_tasks(
            mmp_dir, n_per_endpoint=args.n, seed=args.seed))

    if args.task_type in ("all", "sacrifice_detection"):
        all_tasks.extend(generate_sacrifice_detection_tasks(
            mmp_dir, n_per_endpoint=args.n, seed=args.seed))

    if args.task_type in ("all", "strategic_planning"):
        all_tasks.extend(generate_strategic_planning_tasks(
            mmp_dir, n_per_endpoint=args.n, seed=args.seed))

    if args.task_type in ("all", "multi_objective_path"):
        all_tasks.extend(generate_multi_objective_tasks(
            mmp_dir, n_tasks=args.n * 2, seed=args.seed))

    summary = {}
    for t in all_tasks:
        tt = t["task_type"]
        summary[tt] = summary.get(tt, 0) + 1

    logger.info(f"\nBenchmark Summary:")
    logger.info(f"  Total tasks: {len(all_tasks)}")
    for tt, count in sorted(summary.items()):
        logger.info(f"  {tt}: {count}")

    with open(output_path, "w") as f:
        json.dump({
            "benchmark": "chembench-adme-reasoning",
            "version": "0.1.0",
            "description": "ADME reasoning benchmark generated from MMP analysis of ChEMBL/TDC data",
            "n_tasks": len(all_tasks),
            "task_type_counts": summary,
            "tasks": all_tasks,
        }, f, indent=2)

    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
