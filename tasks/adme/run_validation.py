#!/usr/bin/env python
"""
Run a sample of ADME benchmark tasks through Claude, render molecule structures,
and produce a self-contained interactive HTML report for expert review.

The report includes:
- 2D molecule structure SVGs (with MMP highlighting)
- The full prompt sent to the model
- The model's response
- Ground truth data
- A comment field per task that persists in localStorage + exports to JSON

Usage:
    python run_validation.py
    python run_validation.py --n-per-type 10 --model claude-sonnet-4-6
"""

import argparse
import html
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import anthropic
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D as draw2d

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
BENCHMARK_FILE = SCRIPT_DIR / "benchmark_tasks.json"


def mol_to_svg(smiles, width=350, height=250, highlight_smarts=None):
    """Render a molecule as an SVG string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f'<div class="mol-error">Invalid SMILES: {html.escape(smiles)}</div>'

    AllChem.Compute2DCoords(mol)
    drawer = draw2d.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 2.0

    if highlight_smarts:
        pat = Chem.MolFromSmarts(highlight_smarts)
        if pat:
            matches = mol.GetSubstructMatch(pat)
            if matches:
                drawer.DrawMolecule(mol, highlightAtoms=list(matches))
            else:
                drawer.DrawMolecule(mol)
        else:
            drawer.DrawMolecule(mol)
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mmp_pair_svg(smiles_1, smiles_2, core_smiles, width=750, height=280):
    """Render two molecules side by side, highlighting the differing parts."""
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)

    if mol1 is None or mol2 is None:
        s1 = mol_to_svg(smiles_1, width // 2, height) if mol1 else f'<div class="mol-error">Invalid</div>'
        s2 = mol_to_svg(smiles_2, width // 2, height) if mol2 else f'<div class="mol-error">Invalid</div>'
        return f'<div class="mol-pair">{s1}{s2}</div>'

    AllChem.Compute2DCoords(mol1)
    AllChem.Compute2DCoords(mol2)

    core = Chem.MolFromSmarts(core_smiles) if core_smiles else None
    highlight1, highlight2 = [], []
    if core:
        match1 = mol1.GetSubstructMatch(core)
        match2 = mol2.GetSubstructMatch(core)
        if match1:
            highlight1 = [i for i in range(mol1.GetNumAtoms()) if i not in match1]
        if match2:
            highlight2 = [i for i in range(mol2.GetNumAtoms()) if i not in match2]

    drawer = draw2d.MolDraw2DSVG(width, height, width // 2, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().bondLineWidth = 2.0
    drawer.DrawMolecules(
        [mol1, mol2],
        highlightAtoms=[highlight1, highlight2],
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def series_svg(compounds, width=800, height=250):
    """Render a series of compounds in a grid."""
    mols = []
    legends = []
    for i, c in enumerate(compounds):
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            val = c.get("value", "?")
            if isinstance(val, float):
                val = f"{val:.2f}"
            legends.append(f"#{i+1}: {val}")

    if not mols:
        return '<div class="mol-error">No valid molecules</div>'

    n_per_row = min(4, len(mols))
    n_rows = (len(mols) + n_per_row - 1) // n_per_row
    cell_w = width // n_per_row
    cell_h = 200

    drawer = draw2d.MolDraw2DSVG(width, cell_h * n_rows, cell_w, cell_h)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().bondLineWidth = 1.5
    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def call_claude(prompt, model="claude-sonnet-4-6", max_tokens=2000):
    """Call Claude API with the benchmark prompt."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return call_claude_via_cli(prompt)

    client = anthropic.Anthropic(api_key=api_key)
    system = (
        "You are an expert medicinal chemist with deep knowledge of ADME properties, "
        "CYP metabolism, and structure-activity relationships. Answer precisely and "
        "provide mechanistic reasoning grounded in medicinal chemistry principles. "
        "Be specific about structural features, electronic effects, and metabolic pathways."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return f"[API ERROR: {e}]"


def call_claude_via_cli(prompt):
    """Fall back to calling Claude Code CLI as a subprocess."""
    import subprocess
    import tempfile

    system = (
        "You are an expert medicinal chemist. Answer precisely with mechanistic "
        "reasoning. Be specific about structural features, electronic effects, and "
        "metabolic pathways. Do NOT use any tools. Just answer the chemistry question directly."
    )
    full_prompt = f"{system}\n\n{prompt}"

    try:
        result = subprocess.run(
            ["claude", "-p", full_prompt, "--model", "sonnet"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            logger.error(f"CLI call failed: {result.stderr[:200]}")
            return f"[CLI ERROR: {result.stderr[:200]}]"
    except FileNotFoundError:
        return "[ERROR: claude CLI not found. Set ANTHROPIC_API_KEY or install claude CLI]"
    except subprocess.TimeoutExpired:
        return "[ERROR: CLI call timed out after 300s]"
    except Exception as e:
        return f"[ERROR: {e}]"


def sample_tasks(benchmark, n_per_type=6, seed=123):
    """Sample tasks evenly across types."""
    rng = random.Random(seed)
    by_type = {}
    for task in benchmark["tasks"]:
        tt = task["task_type"]
        by_type.setdefault(tt, []).append(task)

    sampled = []
    for tt, tasks in sorted(by_type.items()):
        n = min(n_per_type, len(tasks))
        sampled.extend(rng.sample(tasks, n))
    return sampled


def render_task_visualization(task):
    """Generate molecule SVGs appropriate for the task type."""
    meta = task.get("metadata", {})
    tt = task["task_type"]

    if tt == "property_delta":
        smi1 = meta.get("smiles_1", "")
        smi2 = meta.get("smiles_2", "")
        core = meta.get("core", "")
        return mmp_pair_svg(smi1, smi2, core)

    elif tt == "series_completion":
        gt = task.get("ground_truth", {})
        prompt_text = task["prompt"]
        compounds = []
        import re
        for m in re.finditer(
            r"Compound \d+: (.+?)\n\s+R-group: .+?\n\s+.+?: ([\d.+-]+)", prompt_text
        ):
            compounds.append({"smiles": m.group(1), "value": float(m.group(2))})
        held_out_smi = meta.get("held_out_smiles", "")
        if held_out_smi:
            compounds.append({"smiles": held_out_smi, "value": "?"})
        if compounds:
            return series_svg(compounds)
        return ""

    elif tt == "transform_ranking":
        anchor = meta.get("anchor_smiles", "")
        if anchor:
            return mol_to_svg(anchor, 400, 280)
        return ""

    elif tt in ("tradeoff_analysis", "transform_explain"):
        gt = task.get("ground_truth", {})
        transform = gt.get("transform", "")
        if ">>" in transform:
            parts = transform.split(">>")
            s1 = mol_to_svg(parts[0], 200, 160)
            s2 = mol_to_svg(parts[1], 200, 160)
            return (
                f'<div style="display:flex;align-items:center;gap:12px;">'
                f'{s1}'
                f'<span style="font-size:28px;font-weight:bold;">→</span>'
                f'{s2}'
                f'</div>'
            )
        return ""

    return ""


# ---------------------------------------------------------------------------
# Auto-assessment
# ---------------------------------------------------------------------------

def extract_numbers(text):
    """Extract all numbers from text."""
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', text)]


def assess_property_delta(task, response):
    """Auto-assess a property_delta task."""
    gt = task["ground_truth"]
    actual_delta = gt["delta"]
    actual_val = gt["value_B"]
    actual_direction = gt["direction"]

    checks = {}

    if "[ERROR" in response:
        return {"grade": "error", "score": 0, "checks": {}, "summary": "Model response timed out or errored."}

    resp_lower = response.lower()
    predicted_beneficial = any(w in resp_lower for w in ["improve", "beneficial", "better", "lower clearance", "more stable", "increased stability", "reduced clearance", "decrease in clearance"])
    predicted_detrimental = any(w in resp_lower for w in ["worsen", "detrimental", "higher clearance", "less stable", "decreased stability", "increased clearance", "increase in clearance"])

    if actual_direction == "beneficial":
        checks["direction"] = predicted_beneficial and not predicted_detrimental
    else:
        checks["direction"] = predicted_detrimental and not predicted_beneficial

    numbers = extract_numbers(response)
    closest = None
    if numbers:
        diffs = [abs(n - actual_val) for n in numbers]
        closest = numbers[diffs.index(min(diffs))]
        checks["value_within_0.3"] = min(diffs) < 0.3
        checks["value_within_0.5"] = min(diffs) < 0.5
    else:
        checks["value_within_0.3"] = False
        checks["value_within_0.5"] = False

    reasoning_keywords = ["lipophil", "metaboli", "oxidat", "cyp", "electron", "steric",
                         "soft spot", "demethyl", "dealkyl", "hydroxyl", "conjugat",
                         "glucuron", "clearance", "half-life", "binding"]
    n_keywords = sum(1 for kw in reasoning_keywords if kw in resp_lower)
    checks["has_reasoning"] = n_keywords >= 2

    score = (checks.get("direction", False) * 40 +
             checks.get("value_within_0.5", False) * 20 +
             checks.get("value_within_0.3", False) * 10 +
             checks.get("has_reasoning", False) * 30)

    if score >= 80:
        grade = "good"
    elif score >= 50:
        grade = "partial"
    elif score >= 20:
        grade = "weak"
    else:
        grade = "poor"

    predicted_str = f"{closest:.3f}" if closest else "not found"
    summary = (f"Direction {'correct' if checks.get('direction') else 'WRONG'} "
               f"(actual: {actual_direction}). "
               f"Predicted value: {predicted_str}, actual: {actual_val:.3f} "
               f"(delta={actual_delta:+.3f}). "
               f"Reasoning depth: {n_keywords} keywords found.")

    return {"grade": grade, "score": score, "checks": checks, "summary": summary}


def assess_series_completion(task, response):
    gt = task["ground_truth"]
    actual_val = gt["value"]

    if "[ERROR" in response:
        return {"grade": "error", "score": 0, "checks": {}, "summary": "Model response timed out or errored."}

    checks = {}
    numbers = extract_numbers(response)
    closest = None
    if numbers:
        diffs = [abs(n - actual_val) for n in numbers]
        closest = numbers[diffs.index(min(diffs))]
        checks["value_within_0.3"] = min(diffs) < 0.3
        checks["value_within_0.5"] = min(diffs) < 0.5
    else:
        checks["value_within_0.3"] = False
        checks["value_within_0.5"] = False

    resp_lower = response.lower()
    checks["discusses_trends"] = any(w in resp_lower for w in ["trend", "series", "pattern", "correlat", "increase", "decrease", "relationship"])
    checks["mentions_uncertainty"] = any(w in resp_lower for w in ["confiden", "uncertain", "caveat", "limit", "approximate", "error", "variab"])

    score = (checks.get("value_within_0.5", False) * 30 +
             checks.get("value_within_0.3", False) * 20 +
             checks.get("discusses_trends", False) * 30 +
             checks.get("mentions_uncertainty", False) * 20)

    if score >= 70:
        grade = "good"
    elif score >= 40:
        grade = "partial"
    else:
        grade = "weak" if score > 0 else "poor"

    predicted_str = f"{closest:.3f}" if closest else "not found"
    summary = (f"Predicted: {predicted_str}, actual: {actual_val:.3f}. "
               f"Within 0.3: {'yes' if checks.get('value_within_0.3') else 'no'}, "
               f"within 0.5: {'yes' if checks.get('value_within_0.5') else 'no'}. "
               f"Series reasoning: {'yes' if checks.get('discusses_trends') else 'no'}.")

    return {"grade": grade, "score": score, "checks": checks, "summary": summary}


def assess_transform_ranking(task, response):
    gt = task["ground_truth"]
    options = gt["options"]
    best = gt["best_transform"]

    if "[ERROR" in response:
        return {"grade": "error", "score": 0, "checks": {}, "summary": "Model response timed out or errored."}

    checks = {}
    resp_lower = response.lower()

    option_letters = [chr(65 + i) for i in range(len(options))]
    first_mentioned = None
    for letter in option_letters:
        patterns = [f"option {letter.lower()}", f"option {letter}", f"{letter})", f"{letter}."]
        for pat in patterns:
            idx = resp_lower.find(pat)
            if idx != -1 and (first_mentioned is None or idx < first_mentioned[1]):
                first_mentioned = (letter, idx)

    ranked_in_response = []
    for letter in option_letters:
        for pat in [f"option {letter.lower()}", f"option {letter}", f"{letter})", f"{letter}."]:
            if pat in resp_lower:
                ranked_in_response.append(letter)
                break

    best_idx = None
    for i, o in enumerate(options):
        if o["transform"] == best:
            best_idx = i
            break

    if best_idx is not None and first_mentioned:
        checks["top1_correct"] = first_mentioned[0] == chr(65 + best_idx)
    else:
        checks["top1_correct"] = False

    checks["has_reasoning"] = any(w in resp_lower for w in [
        "lipophil", "metaboli", "electron", "steric", "polar", "size",
        "bulk", "hydrogen bond", "cyp", "soft spot", "oxidat"])
    checks["identifies_worst"] = any(w in resp_lower for w in ["worsen", "worst", "least", "detrimental", "increase clearance"])

    score = (checks.get("top1_correct", False) * 40 +
             checks.get("has_reasoning", False) * 35 +
             checks.get("identifies_worst", False) * 25)

    if score >= 70:
        grade = "good"
    elif score >= 40:
        grade = "partial"
    else:
        grade = "weak" if score > 0 else "poor"

    summary = (f"Top-1 correct: {'yes' if checks.get('top1_correct') else 'no'}. "
               f"Has reasoning: {'yes' if checks.get('has_reasoning') else 'no'}. "
               f"Identifies worst: {'yes' if checks.get('identifies_worst') else 'no'}.")

    return {"grade": grade, "score": score, "checks": checks, "summary": summary}


def assess_tradeoff(task, response):
    gt = task["ground_truth"]
    beneficial = set(gt.get("beneficial_endpoints", []))
    detrimental = set(gt.get("detrimental_endpoints", []))

    if "[ERROR" in response:
        return {"grade": "error", "score": 0, "checks": {}, "summary": "Model response timed out or errored."}

    checks = {}
    resp_lower = response.lower()

    checks["identifies_tradeoff"] = any(w in resp_lower for w in [
        "trade", "tradeoff", "trade-off", "balance", "on the other hand",
        "however", "while", "improves.*worsen", "benefit.*cost"])

    checks["has_mechanism"] = any(w in resp_lower for w in [
        "lipophil", "electron", "polar", "metaboli", "binding",
        "aromatic", "hydrogen bond", "steric", "hydrophob", "basicity"])

    checks["suggests_alternative"] = any(w in resp_lower for w in [
        "alternative", "instead", "suggest", "consider", "could try",
        "might use", "another option", "replace with"])

    n_beneficial_mentioned = sum(1 for ep in beneficial if ep.lower().replace(" ", "") in resp_lower.replace(" ", ""))
    n_detrimental_mentioned = sum(1 for ep in detrimental if ep.lower().replace(" ", "") in resp_lower.replace(" ", ""))
    total_endpoints = len(beneficial) + len(detrimental)
    if total_endpoints > 0:
        checks["endpoint_coverage"] = (n_beneficial_mentioned + n_detrimental_mentioned) / total_endpoints > 0.5
    else:
        checks["endpoint_coverage"] = True

    score = (checks.get("identifies_tradeoff", False) * 25 +
             checks.get("has_mechanism", False) * 30 +
             checks.get("suggests_alternative", False) * 20 +
             checks.get("endpoint_coverage", False) * 25)

    if score >= 70:
        grade = "good"
    elif score >= 40:
        grade = "partial"
    else:
        grade = "weak" if score > 0 else "poor"

    summary = (f"Tradeoff identified: {'yes' if checks.get('identifies_tradeoff') else 'no'}. "
               f"Mechanistic reasoning: {'yes' if checks.get('has_mechanism') else 'no'}. "
               f"Alternative suggested: {'yes' if checks.get('suggests_alternative') else 'no'}. "
               f"Endpoint coverage: {n_beneficial_mentioned + n_detrimental_mentioned}/{total_endpoints}.")

    return {"grade": grade, "score": score, "checks": checks, "summary": summary}


def assess_explanation(task, response):
    gt = task["ground_truth"]

    if "[ERROR" in response:
        return {"grade": "error", "score": 0, "checks": {}, "summary": "Model response timed out or errored."}

    checks = {}
    resp_lower = response.lower()

    mechanism_keywords = ["lipophil", "electron", "steric", "metaboli", "oxidat",
                         "cyp", "conjugat", "glucuron", "demethyl", "hydroxyl",
                         "polar", "hydrogen bond", "hydrophob", "aromatic",
                         "resonance", "induct", "basicity", "acidity", "pka",
                         "log ?p", "clearance", "half-life", "soft spot", "binding"]
    n_mechanism = sum(1 for kw in mechanism_keywords if kw in resp_lower)
    checks["mechanistic_depth"] = n_mechanism >= 3

    checks["discusses_context"] = any(w in resp_lower for w in [
        "context", "depend", "scaffold", "environment", "substrat",
        "position", "substitut", "neighbor", "adjacent", "surround"])

    checks["discusses_reversal"] = any(w in resp_lower for w in [
        "opposite", "reversal", "reverse", "exception", "counter",
        "however", "in contrast", "paradox", "unexpected"])

    score = (checks.get("mechanistic_depth", False) * 40 +
             checks.get("discusses_context", False) * 30 +
             checks.get("discusses_reversal", False) * 30)

    if score >= 70:
        grade = "good"
    elif score >= 40:
        grade = "partial"
    else:
        grade = "weak" if score > 0 else "poor"

    summary = (f"Mechanistic depth: {n_mechanism} keywords ({checks.get('mechanistic_depth', False)}). "
               f"Context-aware: {'yes' if checks.get('discusses_context') else 'no'}. "
               f"Reversal scenario: {'yes' if checks.get('discusses_reversal') else 'no'}.")

    return {"grade": grade, "score": score, "checks": checks, "summary": summary}


def auto_assess(task, response):
    """Route to the right assessor based on task type."""
    tt = task["task_type"]
    if tt == "property_delta":
        return assess_property_delta(task, response)
    elif tt == "series_completion":
        return assess_series_completion(task, response)
    elif tt == "transform_ranking":
        return assess_transform_ranking(task, response)
    elif tt == "tradeoff_analysis":
        return assess_tradeoff(task, response)
    elif tt == "transform_explain":
        return assess_explanation(task, response)
    return {"grade": "unknown", "score": 0, "checks": {}, "summary": "Unknown task type."}


def summarize_response(response, max_sentences=4):
    """Extract the first few substantive sentences as a summary."""
    if "[ERROR" in response:
        return response

    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    substantive = [s for s in sentences if len(s) > 30 and not s.startswith("*")]
    summary = " ".join(substantive[:max_sentences])
    if len(summary) > 500:
        summary = summary[:497] + "..."
    return summary


def generate_html_report(results, output_path, model_name):
    """Generate the full interactive HTML report."""
    grade_colors = {
        "good": "#22c55e",
        "partial": "#eab308",
        "weak": "#f97316",
        "poor": "#ef4444",
        "error": "#94a3b8",
        "unknown": "#94a3b8",
    }
    grade_bg = {
        "good": "#f0fdf4",
        "partial": "#fefce8",
        "weak": "#fff7ed",
        "poor": "#fef2f2",
        "error": "#f8fafc",
        "unknown": "#f8fafc",
    }

    assessments = []
    task_blocks = []
    for i, r in enumerate(results):
        task = r["task"]
        response = r["response"]
        viz = r["visualization"]
        gt = task.get("ground_truth", {})
        eval_criteria = task.get("evaluation_criteria", {})
        tt = task["task_type"]
        endpoint = task.get("endpoint_label", task.get("endpoint", ""))

        assessment = auto_assess(task, response)
        assessments.append(assessment)
        resp_summary = summarize_response(response)

        gt_html = ""
        for k, v in gt.items():
            if isinstance(v, list) and len(v) > 5:
                v_str = f"[{len(v)} items]"
            elif isinstance(v, float):
                v_str = f"{v:.4f}"
            else:
                v_str = html.escape(str(v))
            gt_html += f'<div class="gt-row"><span class="gt-key">{html.escape(k)}:</span> <span class="gt-val">{v_str}</span></div>'

        criteria_html = ""
        for k, v in eval_criteria.items():
            if isinstance(v, list):
                items = "".join(f"<li>{html.escape(str(x))}</li>" for x in v)
                criteria_html += f'<div class="criteria-item"><strong>{html.escape(k)}:</strong><ul>{items}</ul></div>'
            else:
                criteria_html += f'<div class="criteria-item"><strong>{html.escape(k)}:</strong> {html.escape(str(v))}</div>'

        type_colors = {
            "property_delta": "#2563eb",
            "series_completion": "#7c3aed",
            "transform_ranking": "#059669",
            "tradeoff_analysis": "#dc2626",
            "transform_explain": "#d97706",
        }
        color = type_colors.get(tt, "#6b7280")
        g_color = grade_colors.get(assessment["grade"], "#94a3b8")
        g_bg = grade_bg.get(assessment["grade"], "#f8fafc")

        checks_html = ""
        for ck, cv in assessment["checks"].items():
            icon = "&#10003;" if cv else "&#10007;"
            ccolor = "#22c55e" if cv else "#ef4444"
            checks_html += f'<span class="check-item" style="color:{ccolor};">{icon} {html.escape(ck)}</span> '

        block = f"""
        <div class="task-card" data-grade="{assessment['grade']}" id="task-{i}">
            <div class="task-header" style="border-left-color: {color};">
                <span class="task-badge" style="background: {color};">{html.escape(tt)}</span>
                <span class="task-endpoint">{html.escape(endpoint)}</span>
                <span class="grade-badge" style="background: {g_color};">{assessment['grade'].upper()} ({assessment['score']})</span>
                <span class="task-number">Task {i+1}/{len(results)}</span>
            </div>

            <div class="assessment-bar" style="background: {g_bg}; border-left: 4px solid {g_color};">
                <div class="assessment-summary">{html.escape(assessment['summary'])}</div>
                <div class="assessment-checks">{checks_html}</div>
            </div>

            <div class="section summary-section">
                <h3>Response Summary</h3>
                <div class="summary-box">{html.escape(resp_summary)}</div>
            </div>

            <div class="section">
                <h3>Molecules</h3>
                <div class="mol-container">{viz}</div>
            </div>

            <div class="section collapsible">
                <h3 class="collapsible-header" onclick="toggleSection(this)">Prompt <span class="collapse-icon">&#9654;</span></h3>
                <div class="collapsible-content" style="display:none;">
                    <div class="prompt-box">{html.escape(task['prompt'])}</div>
                </div>
            </div>

            <div class="section collapsible">
                <h3 class="collapsible-header" onclick="toggleSection(this)">Full Response <span class="model-tag">{html.escape(model_name)}</span> <span class="collapse-icon">&#9654;</span></h3>
                <div class="collapsible-content" style="display:none;">
                    <div class="response-box">{html.escape(response)}</div>
                </div>
            </div>

            <div class="section gt-section">
                <h3>Ground Truth</h3>
                <div class="gt-box">{gt_html}</div>
            </div>

            <div class="section collapsible">
                <h3 class="collapsible-header" onclick="toggleSection(this)">Evaluation Criteria <span class="collapse-icon">&#9654;</span></h3>
                <div class="collapsible-content" style="display:none;">
                    <div class="criteria-box">{criteria_html}</div>
                </div>
            </div>

            <div class="section comment-section">
                <h3>Expert Review</h3>
                <div class="rating-row">
                    <label>Overall quality:</label>
                    <select id="rating-{i}" onchange="saveComment({i})">
                        <option value="">-- rate --</option>
                        <option value="excellent">Excellent</option>
                        <option value="good">Good</option>
                        <option value="acceptable">Acceptable</option>
                        <option value="poor">Poor</option>
                        <option value="wrong">Wrong</option>
                    </select>
                    <label style="margin-left:16px;">Task quality:</label>
                    <select id="task-quality-{i}" onchange="saveComment({i})">
                        <option value="">-- rate task --</option>
                        <option value="great-task">Great task</option>
                        <option value="ok-task">OK task</option>
                        <option value="needs-work">Needs work</option>
                        <option value="bad-task">Bad task</option>
                    </select>
                </div>
                <textarea id="comment-{i}" rows="3"
                    placeholder="Your comments on the response quality, reasoning accuracy, task design..."
                    oninput="saveComment({i})"></textarea>
            </div>
        </div>
        """
        task_blocks.append(block)

    tasks_html = "\n".join(task_blocks)

    # Aggregate stats for dashboard
    from collections import Counter
    grade_counts = Counter(a["grade"] for a in assessments)
    scores = [a["score"] for a in assessments if a["grade"] != "error"]
    avg_score = sum(scores) / len(scores) if scores else 0
    n_errors = grade_counts.get("error", 0)

    by_type_scores = {}
    for a, r in zip(assessments, results):
        tt = r["task"]["task_type"]
        by_type_scores.setdefault(tt, []).append(a["score"])
    type_avg_html = ""
    for tt in sorted(by_type_scores):
        s = by_type_scores[tt]
        avg = sum(s) / len(s) if s else 0
        type_avg_html += f'<div class="type-score"><span class="type-label">{html.escape(tt)}</span><span class="type-avg">{avg:.0f}</span></div>'

    grade_bar_parts = ""
    for g in ["good", "partial", "weak", "poor", "error"]:
        cnt = grade_counts.get(g, 0)
        if cnt > 0:
            pct = cnt / len(assessments) * 100
            gc = grade_colors.get(g, "#94a3b8")
            grade_bar_parts += f'<div class="gbar-seg" style="width:{pct}%;background:{gc};" title="{g}: {cnt}"></div>'

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ChemBench ADME Validation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f1f5f9; color: #1e293b; line-height: 1.5;
    max-width: 960px; margin: 0 auto; padding: 24px 16px;
  }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; margin-bottom: 24px; font-size: 0.95rem; }}
  .controls {{
    display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap;
  }}
  .controls button {{
    padding: 8px 18px; border-radius: 6px; border: 1px solid #cbd5e1;
    background: white; cursor: pointer; font-size: 0.9rem; font-weight: 500;
  }}
  .controls button:hover {{ background: #f8fafc; border-color: #94a3b8; }}
  .controls button.primary {{
    background: #2563eb; color: white; border-color: #2563eb;
  }}
  .controls button.primary:hover {{ background: #1d4ed8; }}
  .filter-bar {{
    display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap;
  }}
  .filter-btn {{
    padding: 4px 14px; border-radius: 20px; border: 2px solid #e2e8f0;
    background: white; cursor: pointer; font-size: 0.82rem; font-weight: 600;
    transition: all 0.15s;
  }}
  .filter-btn:hover {{ border-color: #94a3b8; }}
  .filter-btn.active {{ background: #1e293b; color: white; border-color: #1e293b; }}
  .dashboard {{
    background: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .dashboard h2 {{ font-size: 1rem; margin-bottom: 14px; color: #475569; }}
  .dashboard-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 16px; margin-bottom: 16px;
  }}
  .dash-stat {{ text-align: center; }}
  .dash-val {{ font-size: 1.8rem; font-weight: 800; }}
  .dash-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; }}
  .grade-bar {{
    display: flex; height: 28px; border-radius: 6px; overflow: hidden; margin-bottom: 10px;
  }}
  .gbar-seg {{ min-width: 2px; transition: width 0.3s; }}
  .grade-legend {{
    display: flex; gap: 14px; font-size: 0.78rem; color: #64748b; flex-wrap: wrap;
  }}
  .grade-legend span {{
    display: flex; align-items: center; gap: 4px;
  }}
  .grade-legend .dot {{
    width: 10px; height: 10px; border-radius: 50%; display: inline-block;
  }}
  .type-scores {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 14px; }}
  .type-score {{
    display: flex; flex-direction: column; align-items: center;
    background: #f8fafc; border-radius: 8px; padding: 8px 14px;
  }}
  .type-label {{ font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }}
  .type-avg {{ font-size: 1.3rem; font-weight: 700; }}
  .task-card {{
    background: white; border-radius: 12px; margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden;
  }}
  .task-header {{
    padding: 14px 20px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;
    border-left: 5px solid #6b7280; display: flex; align-items: center; gap: 12px;
    flex-wrap: wrap;
  }}
  .task-badge {{
    color: white; font-size: 0.75rem; font-weight: 700; padding: 3px 10px;
    border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .grade-badge {{
    color: white; font-size: 0.72rem; font-weight: 700; padding: 3px 10px;
    border-radius: 12px; letter-spacing: 0.5px;
  }}
  .task-endpoint {{ color: #64748b; font-size: 0.85rem; }}
  .task-number {{ margin-left: auto; color: #94a3b8; font-size: 0.8rem; }}
  .assessment-bar {{
    padding: 12px 20px; border-bottom: 1px solid #e2e8f0;
  }}
  .assessment-summary {{ font-size: 0.88rem; color: #1e293b; margin-bottom: 6px; }}
  .assessment-checks {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .check-item {{ font-size: 0.8rem; font-weight: 600; }}
  .summary-section .summary-box {{
    background: #eff6ff; padding: 12px; border-radius: 8px; font-size: 0.88rem;
    border: 1px solid #bfdbfe; color: #1e40af;
  }}
  .section {{ padding: 16px 20px; border-bottom: 1px solid #f1f5f9; }}
  .section h3 {{
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.8px;
    color: #94a3b8; margin-bottom: 10px;
  }}
  .collapsible-header {{
    cursor: pointer; user-select: none;
  }}
  .collapsible-header:hover {{ color: #64748b; }}
  .collapse-icon {{
    font-size: 0.7rem; display: inline-block; transition: transform 0.2s;
  }}
  .collapse-icon.open {{ transform: rotate(90deg); }}
  .mol-container {{ text-align: center; overflow-x: auto; }}
  .mol-container svg {{ max-width: 100%; height: auto; }}
  .mol-error {{ color: #dc2626; font-style: italic; padding: 12px; }}
  .prompt-box {{
    background: #f8fafc; padding: 14px; border-radius: 8px; font-size: 0.88rem;
    white-space: pre-wrap; font-family: 'SF Mono', 'Fira Code', monospace;
    border: 1px solid #e2e8f0; max-height: 350px; overflow-y: auto;
  }}
  .response-box {{
    background: #f0fdf4; padding: 14px; border-radius: 8px; font-size: 0.9rem;
    white-space: pre-wrap; border: 1px solid #bbf7d0; max-height: 500px; overflow-y: auto;
  }}
  .model-tag {{
    font-size: 0.7rem; background: #dcfce7; color: #166534; padding: 2px 8px;
    border-radius: 8px; font-weight: 600; text-transform: none; letter-spacing: 0;
  }}
  .gt-section .gt-box {{ padding: 8px 0; }}
  .gt-row {{ padding: 3px 0; font-size: 0.88rem; }}
  .gt-key {{ font-weight: 600; color: #475569; }}
  .gt-val {{ font-family: 'SF Mono', monospace; color: #1e293b; }}
  .criteria-box {{ font-size: 0.88rem; }}
  .criteria-item {{ margin-bottom: 6px; }}
  .criteria-item ul {{ margin: 4px 0 4px 20px; }}
  .criteria-item li {{ font-size: 0.85rem; color: #475569; }}
  .comment-section textarea {{
    width: 100%; padding: 10px; border: 2px solid #e2e8f0; border-radius: 8px;
    font-size: 0.9rem; font-family: inherit; resize: vertical;
    transition: border-color 0.15s;
  }}
  .comment-section textarea:focus {{
    outline: none; border-color: #2563eb;
  }}
  .rating-row {{
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
    flex-wrap: wrap;
  }}
  .rating-row select {{
    padding: 5px 10px; border: 2px solid #e2e8f0; border-radius: 6px;
    font-size: 0.88rem; background: white;
  }}
  .rating-row label {{ font-size: 0.85rem; font-weight: 500; color: #475569; }}
  .stats-bar {{
    background: white; border-radius: 10px; padding: 14px 20px; margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: flex; gap: 24px;
    flex-wrap: wrap; font-size: 0.88rem;
  }}
  .stats-bar .stat {{ text-align: center; }}
  .stats-bar .stat-val {{ font-size: 1.4rem; font-weight: 700; color: #1e293b; }}
  .stats-bar .stat-label {{ color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; }}
  .save-indicator {{
    position: fixed; bottom: 20px; right: 20px; background: #22c55e; color: white;
    padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
    opacity: 0; transition: opacity 0.3s; pointer-events: none;
  }}
  .mol-pair {{ display: flex; justify-content: center; gap: 16px; }}
</style>
</head>
<body>

<h1>ChemBench ADME Validation Report</h1>
<p class="subtitle">
  {len(results)} tasks &middot; Model: {html.escape(model_name)} &middot;
  Generated from MMP-ADME database
</p>

<div class="dashboard">
  <h2>Auto-Assessment Dashboard</h2>
  <div class="dashboard-grid">
    <div class="dash-stat"><div class="dash-val">{avg_score:.0f}</div><div class="dash-label">Avg Score</div></div>
    <div class="dash-stat"><div class="dash-val" style="color:#22c55e;">{grade_counts.get('good', 0)}</div><div class="dash-label">Good</div></div>
    <div class="dash-stat"><div class="dash-val" style="color:#eab308;">{grade_counts.get('partial', 0)}</div><div class="dash-label">Partial</div></div>
    <div class="dash-stat"><div class="dash-val" style="color:#f97316;">{grade_counts.get('weak', 0)}</div><div class="dash-label">Weak</div></div>
    <div class="dash-stat"><div class="dash-val" style="color:#ef4444;">{grade_counts.get('poor', 0)}</div><div class="dash-label">Poor</div></div>
    <div class="dash-stat"><div class="dash-val" style="color:#94a3b8;">{n_errors}</div><div class="dash-label">Errors</div></div>
  </div>
  <div class="grade-bar">{grade_bar_parts}</div>
  <div class="grade-legend">
    <span><span class="dot" style="background:#22c55e;"></span> Good (>=70)</span>
    <span><span class="dot" style="background:#eab308;"></span> Partial (40-69)</span>
    <span><span class="dot" style="background:#f97316;"></span> Weak (1-39)</span>
    <span><span class="dot" style="background:#ef4444;"></span> Poor (0)</span>
    <span><span class="dot" style="background:#94a3b8;"></span> Error</span>
  </div>
  <div class="type-scores">{type_avg_html}</div>
</div>

<div class="stats-bar">
  <div class="stat"><div class="stat-val">{len(results)}</div><div class="stat-label">Tasks</div></div>
  <div class="stat"><div class="stat-val" id="rated-count">0</div><div class="stat-label">Rated</div></div>
  <div class="stat"><div class="stat-val" id="commented-count">0</div><div class="stat-label">Commented</div></div>
</div>

<div class="filter-bar">
  <button class="filter-btn active" onclick="filterTasks('all', 'type')">All</button>
  <button class="filter-btn" onclick="filterTasks('property_delta', 'type')" style="border-color:#2563eb;">property_delta</button>
  <button class="filter-btn" onclick="filterTasks('series_completion', 'type')" style="border-color:#7c3aed;">series_completion</button>
  <button class="filter-btn" onclick="filterTasks('transform_ranking', 'type')" style="border-color:#059669;">transform_ranking</button>
  <button class="filter-btn" onclick="filterTasks('tradeoff_analysis', 'type')" style="border-color:#dc2626;">tradeoff_analysis</button>
  <button class="filter-btn" onclick="filterTasks('transform_explain', 'type')" style="border-color:#d97706;">transform_explain</button>
</div>
<div class="filter-bar">
  <button class="filter-btn grade-filter active" onclick="filterTasks('all', 'grade')">All grades</button>
  <button class="filter-btn grade-filter" onclick="filterTasks('good', 'grade')" style="border-color:#22c55e;">Good</button>
  <button class="filter-btn grade-filter" onclick="filterTasks('partial', 'grade')" style="border-color:#eab308;">Partial</button>
  <button class="filter-btn grade-filter" onclick="filterTasks('weak', 'grade')" style="border-color:#f97316;">Weak</button>
  <button class="filter-btn grade-filter" onclick="filterTasks('poor', 'grade')" style="border-color:#ef4444;">Poor</button>
  <button class="filter-btn grade-filter" onclick="filterTasks('error', 'grade')" style="border-color:#94a3b8;">Error</button>
</div>

<div class="controls">
  <button class="primary" onclick="exportComments()">Export Comments (JSON)</button>
  <button onclick="importComments()">Import Comments</button>
  <input type="file" id="import-file" style="display:none" accept=".json" onchange="handleImport(event)">
</div>

{tasks_html}

<div class="save-indicator" id="save-indicator">Saved</div>

<script>
const STORAGE_KEY = 'chembench_adme_validation_comments';
const N_TASKS = {len(results)};

const taskTypes = {json.dumps([r['task']['task_type'] for r in results])};
const taskGrades = {json.dumps([a['grade'] for a in assessments])};

let activeTypeFilter = 'all';
let activeGradeFilter = 'all';

function toggleSection(header) {{
  const content = header.nextElementSibling;
  const icon = header.querySelector('.collapse-icon');
  if (content.style.display === 'none') {{
    content.style.display = '';
    if (icon) icon.classList.add('open');
  }} else {{
    content.style.display = 'none';
    if (icon) icon.classList.remove('open');
  }}
}}

function saveComment(idx) {{
  const data = loadAll();
  data[idx] = {{
    rating: document.getElementById('rating-' + idx).value,
    taskQuality: document.getElementById('task-quality-' + idx).value,
    comment: document.getElementById('comment-' + idx).value,
    timestamp: new Date().toISOString(),
  }};
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  showSaved();
  updateCounts();
}}

function loadAll() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }}
  catch {{ return {{}}; }}
}}

function restoreAll() {{
  const data = loadAll();
  for (const [idx, entry] of Object.entries(data)) {{
    const r = document.getElementById('rating-' + idx);
    const tq = document.getElementById('task-quality-' + idx);
    const c = document.getElementById('comment-' + idx);
    if (r && entry.rating) r.value = entry.rating;
    if (tq && entry.taskQuality) tq.value = entry.taskQuality;
    if (c && entry.comment) c.value = entry.comment;
  }}
  updateCounts();
}}

function updateCounts() {{
  const data = loadAll();
  let rated = 0, commented = 0;
  for (const entry of Object.values(data)) {{
    if (entry.rating) rated++;
    if (entry.comment && entry.comment.trim()) commented++;
  }}
  document.getElementById('rated-count').textContent = rated;
  document.getElementById('commented-count').textContent = commented;
}}

function showSaved() {{
  const el = document.getElementById('save-indicator');
  el.style.opacity = '1';
  setTimeout(() => el.style.opacity = '0', 1200);
}}

function exportComments() {{
  const data = loadAll();
  const enriched = {{}};
  for (const [idx, entry] of Object.entries(data)) {{
    enriched[idx] = {{
      ...entry,
      task_type: taskTypes[parseInt(idx)] || 'unknown',
      auto_grade: taskGrades[parseInt(idx)] || 'unknown',
    }};
  }}
  const blob = new Blob([JSON.stringify(enriched, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'chembench_validation_comments.json';
  a.click();
  URL.revokeObjectURL(url);
}}

function importComments() {{
  document.getElementById('import-file').click();
}}

function handleImport(event) {{
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {{
    try {{
      const imported = JSON.parse(e.target.result);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(imported));
      restoreAll();
      showSaved();
    }} catch (err) {{
      alert('Invalid JSON file');
    }}
  }};
  reader.readAsText(file);
}}

function applyFilters() {{
  document.querySelectorAll('.task-card').forEach((card, i) => {{
    const typeMatch = activeTypeFilter === 'all' || taskTypes[i] === activeTypeFilter;
    const gradeMatch = activeGradeFilter === 'all' || taskGrades[i] === activeGradeFilter;
    card.style.display = (typeMatch && gradeMatch) ? '' : 'none';
  }});
}}

function filterTasks(value, dimension) {{
  if (dimension === 'type') {{
    activeTypeFilter = value;
    document.querySelectorAll('.filter-btn:not(.grade-filter)').forEach(b => b.classList.remove('active'));
  }} else {{
    activeGradeFilter = value;
    document.querySelectorAll('.filter-btn.grade-filter').forEach(b => b.classList.remove('active'));
  }}
  event.target.classList.add('active');
  applyFilters();
}}

restoreAll();
</script>
</body>
</html>"""
    output_path.write_text(full_html)
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-type", type=int, default=6)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else SCRIPT_DIR / "validation_report.html"

    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)

    tasks = sample_tasks(benchmark, n_per_type=args.n_per_type, seed=args.seed)
    logger.info(f"Sampled {len(tasks)} tasks across {len(set(t['task_type'] for t in tasks))} types")

    results = []
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] Running {task['task_type']}...")

        viz = render_task_visualization(task)
        response = call_claude(task["prompt"], model=args.model)
        results.append({"task": task, "response": response, "visualization": viz})

        if i < len(tasks) - 1:
            time.sleep(0.5)

    generate_html_report(results, output_path, args.model)
    logger.info(f"Done! Open {output_path} in your browser.")


if __name__ == "__main__":
    main()
