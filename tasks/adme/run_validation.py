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
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            logger.error(f"CLI call failed: {result.stderr[:200]}")
            return f"[CLI ERROR: {result.stderr[:200]}]"
    except FileNotFoundError:
        return "[ERROR: claude CLI not found. Set ANTHROPIC_API_KEY or install claude CLI]"
    except subprocess.TimeoutExpired:
        return "[ERROR: CLI call timed out after 120s]"
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


def generate_html_report(results, output_path, model_name):
    """Generate the full interactive HTML report."""
    task_blocks = []
    for i, r in enumerate(results):
        task = r["task"]
        response = r["response"]
        viz = r["visualization"]
        gt = task.get("ground_truth", {})
        eval_criteria = task.get("evaluation_criteria", {})
        tt = task["task_type"]
        endpoint = task.get("endpoint_label", task.get("endpoint", ""))

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

        block = f"""
        <div class="task-card" id="task-{i}">
            <div class="task-header" style="border-left-color: {color};">
                <span class="task-badge" style="background: {color};">{html.escape(tt)}</span>
                <span class="task-endpoint">{html.escape(endpoint)}</span>
                <span class="task-number">Task {i+1}/{len(results)}</span>
            </div>

            <div class="section">
                <h3>Molecules</h3>
                <div class="mol-container">{viz}</div>
            </div>

            <div class="section">
                <h3>Prompt</h3>
                <div class="prompt-box">{html.escape(task['prompt'])}</div>
            </div>

            <div class="section">
                <h3>Model Response <span class="model-tag">{html.escape(model_name)}</span></h3>
                <div class="response-box">{html.escape(response)}</div>
            </div>

            <div class="section gt-section">
                <h3>Ground Truth</h3>
                <div class="gt-box">{gt_html}</div>
            </div>

            <div class="section criteria-section">
                <h3>Evaluation Criteria</h3>
                <div class="criteria-box">{criteria_html}</div>
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
  .task-card {{
    background: white; border-radius: 12px; margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden;
  }}
  .task-header {{
    padding: 14px 20px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;
    border-left: 5px solid #6b7280; display: flex; align-items: center; gap: 12px;
  }}
  .task-badge {{
    color: white; font-size: 0.75rem; font-weight: 700; padding: 3px 10px;
    border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .task-endpoint {{ color: #64748b; font-size: 0.85rem; }}
  .task-number {{ margin-left: auto; color: #94a3b8; font-size: 0.8rem; }}
  .section {{ padding: 16px 20px; border-bottom: 1px solid #f1f5f9; }}
  .section h3 {{
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.8px;
    color: #94a3b8; margin-bottom: 10px;
  }}
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

<div class="stats-bar">
  <div class="stat"><div class="stat-val">{len(results)}</div><div class="stat-label">Tasks</div></div>
  <div class="stat"><div class="stat-val" id="rated-count">0</div><div class="stat-label">Rated</div></div>
  <div class="stat"><div class="stat-val" id="commented-count">0</div><div class="stat-label">Commented</div></div>
</div>

<div class="filter-bar">
  <button class="filter-btn active" onclick="filterTasks('all')">All</button>
  <button class="filter-btn" onclick="filterTasks('property_delta')" style="border-color:#2563eb;">property_delta</button>
  <button class="filter-btn" onclick="filterTasks('series_completion')" style="border-color:#7c3aed;">series_completion</button>
  <button class="filter-btn" onclick="filterTasks('transform_ranking')" style="border-color:#059669;">transform_ranking</button>
  <button class="filter-btn" onclick="filterTasks('tradeoff_analysis')" style="border-color:#dc2626;">tradeoff_analysis</button>
  <button class="filter-btn" onclick="filterTasks('transform_explain')" style="border-color:#d97706;">transform_explain</button>
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

function filterTasks(type) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.task-card').forEach((card, i) => {{
    if (type === 'all' || taskTypes[i] === type) {{
      card.style.display = '';
    }} else {{
      card.style.display = 'none';
    }}
  }});
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
