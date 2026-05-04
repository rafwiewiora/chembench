#!/usr/bin/env python
"""
Run a multi-model comparison of ADME benchmark tasks through Claude models,
auto-assess responses, and produce a comparison HTML report + CSV.

Runs the same sampled tasks through multiple models and generates:
- A summary table (rows = task types, columns = models, cells = avg score + grade)
- Overall averages per model
- Individual task cards with side-by-side model responses
- A raw results CSV for downstream analysis

Usage:
    python run_comparison.py --models sonnet,opus
    python run_comparison.py --models sonnet,opus,haiku --n-per-type 4
"""

import argparse
import csv
import html
import json
import logging
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_validation import (
    BENCHMARK_FILE,
    auto_assess,
    render_task_visualization,
    sample_tasks,
    summarize_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent

# Model name mapping for display
MODEL_DISPLAY_NAMES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


def call_claude_via_cli(prompt, model="sonnet", max_retries=3):
    """Call Claude Code CLI as a subprocess with specified model."""
    system = (
        "You are an expert medicinal chemist. Answer precisely with mechanistic "
        "reasoning. Be specific about structural features, electronic effects, and "
        "metabolic pathways. Do NOT use any tools. Just answer the chemistry question directly."
    )
    full_prompt = f"{system}\n\n{prompt}"

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["claude", "-p", full_prompt, "--model", model],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                logger.error(f"CLI call failed (attempt {attempt+1}): {result.stderr[:200]}")
                if attempt < max_retries - 1:
                    wait = 30 * (2 ** attempt)
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return f"[CLI ERROR: {result.stderr[:200]}]"
        except FileNotFoundError:
            return "[ERROR: claude CLI not found]"
        except subprocess.TimeoutExpired:
            logger.error(f"CLI call timed out (attempt {attempt+1})")
            if attempt < max_retries - 1:
                wait = 30 * (2 ** attempt)
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
                continue
            return "[ERROR: CLI call timed out after 300s]"
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 30 * (2 ** attempt)
                logger.info(f"Retrying in {wait}s after error: {e}")
                time.sleep(wait)
                continue
            return f"[ERROR: {e}]"
    return "[ERROR: max retries exceeded]"


def run_comparison(tasks, models):
    """Run all tasks through all models, returning structured results."""
    # results[task_idx][model] = {"response": ..., "assessment": ...}
    results = []

    for i, task in enumerate(tasks):
        task_result = {
            "task": task,
            "visualization": render_task_visualization(task),
            "models": {},
        }

        for model in models:
            logger.info(
                f"[{i+1}/{len(tasks)}] {task['task_type']} -> {model}"
            )
            response = call_claude_via_cli(task["prompt"], model=model)
            assessment = auto_assess(task, response)
            task_result["models"][model] = {
                "response": response,
                "assessment": assessment,
                "summary": summarize_response(response),
            }

            # Rate-limit between model calls
            time.sleep(2)

        results.append(task_result)

    # Retry failed calls
    failed = []
    for i, tr in enumerate(results):
        for model, mr in tr["models"].items():
            if "[ERROR" in mr["response"] or "[CLI ERROR" in mr["response"]:
                failed.append((i, model))

    if failed:
        logger.info(f"Retrying {len(failed)} failed calls after 60s cooldown...")
        time.sleep(60)
        for idx, model in failed:
            task = results[idx]["task"]
            logger.info(f"[retry] task {idx+1} -> {model}")
            response = call_claude_via_cli(task["prompt"], model=model)
            if "[ERROR" not in response and "[CLI ERROR" not in response:
                assessment = auto_assess(task, response)
                results[idx]["models"][model] = {
                    "response": response,
                    "assessment": assessment,
                    "summary": summarize_response(response),
                }
                logger.info("  -> retry succeeded")
            else:
                logger.warning("  -> retry still failed")
            time.sleep(2)

    return results


def write_csv(results, models, output_path):
    """Write raw results to CSV for analysis."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_idx", "task_type", "endpoint", "model", "score", "grade",
        ])
        for i, tr in enumerate(results):
            task = tr["task"]
            tt = task["task_type"]
            endpoint = task.get("endpoint_label", task.get("endpoint", ""))
            for model in models:
                mr = tr["models"][model]
                writer.writerow([
                    i, tt, endpoint, model,
                    mr["assessment"]["score"],
                    mr["assessment"]["grade"],
                ])
    logger.info(f"CSV saved to {output_path}")


def generate_comparison_html(results, models, output_path):
    """Generate a multi-model comparison HTML report."""
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
    model_colors = {
        "sonnet": "#2563eb",
        "opus": "#7c3aed",
        "haiku": "#059669",
    }

    # --- Compute summary statistics ---
    # Per model overall
    model_scores = {m: [] for m in models}
    model_grades = {m: Counter() for m in models}
    # Per model per task type
    type_model_scores = defaultdict(lambda: {m: [] for m in models})

    for tr in results:
        tt = tr["task"]["task_type"]
        for model in models:
            mr = tr["models"][model]
            score = mr["assessment"]["score"]
            grade = mr["assessment"]["grade"]
            model_scores[model].append(score)
            model_grades[model][grade] += 1
            type_model_scores[tt][model].append(score)

    # --- Build summary table ---
    task_types_sorted = sorted(type_model_scores.keys())

    summary_rows = ""
    for tt in task_types_sorted:
        summary_rows += f'<tr><td class="tt-cell">{html.escape(tt)}</td>'
        for model in models:
            scores = type_model_scores[tt][model]
            if scores:
                avg = sum(scores) / len(scores)
                # Determine dominant grade
                grades = [
                    tr["models"][model]["assessment"]["grade"]
                    for tr in results
                    if tr["task"]["task_type"] == tt
                ]
                dominant = Counter(grades).most_common(1)[0][0]
                g_color = grade_colors.get(dominant, "#94a3b8")
                summary_rows += (
                    f'<td class="score-cell">'
                    f'<span class="score-val">{avg:.0f}</span>'
                    f'<span class="mini-grade" style="background:{g_color};">{dominant}</span>'
                    f'</td>'
                )
            else:
                summary_rows += '<td class="score-cell">-</td>'
        summary_rows += "</tr>\n"

    # Overall row
    summary_rows += '<tr class="overall-row"><td class="tt-cell"><strong>OVERALL</strong></td>'
    for model in models:
        scores = model_scores[model]
        avg = sum(scores) / len(scores) if scores else 0
        summary_rows += (
            f'<td class="score-cell"><span class="score-val overall-val">{avg:.0f}</span></td>'
        )
    summary_rows += "</tr>"

    model_headers = "".join(
        f'<th style="color:{model_colors.get(m, "#1e293b")};">{html.escape(m)}<br>'
        f'<span class="model-subname">{html.escape(MODEL_DISPLAY_NAMES.get(m, m))}</span></th>'
        for m in models
    )

    # --- Build individual task cards ---
    task_cards = []
    for i, tr in enumerate(results):
        task = tr["task"]
        viz = tr["visualization"]
        tt = task["task_type"]
        endpoint = task.get("endpoint_label", task.get("endpoint", ""))
        gt = task.get("ground_truth", {})

        # Ground truth HTML
        gt_html = ""
        for k, v in gt.items():
            if isinstance(v, list) and len(v) > 5:
                v_str = f"[{len(v)} items]"
            elif isinstance(v, float):
                v_str = f"{v:.4f}"
            else:
                v_str = html.escape(str(v))
            gt_html += (
                f'<div class="gt-row">'
                f'<span class="gt-key">{html.escape(k)}:</span> '
                f'<span class="gt-val">{v_str}</span>'
                f'</div>'
            )

        # Model response columns
        model_panels = ""
        for model in models:
            mr = tr["models"][model]
            assessment = mr["assessment"]
            g_color = grade_colors.get(assessment["grade"], "#94a3b8")
            g_bg = grade_bg.get(assessment["grade"], "#f8fafc")
            m_color = model_colors.get(model, "#1e293b")

            checks_html = ""
            for ck, cv in assessment["checks"].items():
                icon = "&#10003;" if cv else "&#10007;"
                ccolor = "#22c55e" if cv else "#ef4444"
                checks_html += f'<span class="check-item" style="color:{ccolor};">{icon} {html.escape(ck)}</span> '

            model_panels += f"""
            <div class="model-panel" style="border-top: 3px solid {m_color};">
                <div class="model-panel-header">
                    <span class="model-name" style="color:{m_color};">{html.escape(model)}</span>
                    <span class="grade-badge" style="background:{g_color};">{assessment['grade'].upper()} ({assessment['score']})</span>
                </div>
                <div class="assessment-mini" style="background:{g_bg};">
                    <div class="assess-summary-text">{html.escape(assessment['summary'])}</div>
                    <div class="assess-checks">{checks_html}</div>
                </div>
                <div class="response-summary">{html.escape(mr['summary'])}</div>
                <details class="full-response-details">
                    <summary>Full response</summary>
                    <div class="response-box">{html.escape(mr['response'])}</div>
                </details>
            </div>
            """

        type_colors = {
            "property_delta": "#2563eb",
            "series_completion": "#7c3aed",
            "transform_ranking": "#059669",
            "tradeoff_analysis": "#dc2626",
            "transform_explain": "#d97706",
            "sacrifice_detection": "#0891b2",
            "strategic_planning": "#be185d",
            "multi_objective_path": "#4338ca",
        }
        color = type_colors.get(tt, "#6b7280")

        card = f"""
        <div class="task-card" data-type="{tt}" id="task-{i}">
            <div class="task-header" style="border-left-color: {color};">
                <span class="task-badge" style="background: {color};">{html.escape(tt)}</span>
                <span class="task-endpoint">{html.escape(endpoint)}</span>
                <span class="task-number">Task {i+1}/{len(results)}</span>
            </div>

            <div class="section">
                <div class="mol-container">{viz}</div>
            </div>

            <details class="prompt-details">
                <summary>Prompt</summary>
                <div class="prompt-box">{html.escape(task['prompt'])}</div>
            </details>

            <div class="model-panels-grid">
                {model_panels}
            </div>

            <details class="gt-details">
                <summary>Ground Truth</summary>
                <div class="gt-box">{gt_html}</div>
            </details>
        </div>
        """
        task_cards.append(card)

    tasks_html = "\n".join(task_cards)

    # --- Model comparison stats bar ---
    stats_blocks = ""
    for model in models:
        m_color = model_colors.get(model, "#1e293b")
        scores = model_scores[model]
        avg = sum(scores) / len(scores) if scores else 0
        gc = model_grades[model]
        stats_blocks += f"""
        <div class="model-stat-block" style="border-top: 3px solid {m_color};">
            <div class="model-stat-name" style="color:{m_color};">{html.escape(model)}</div>
            <div class="model-stat-avg">{avg:.0f}</div>
            <div class="model-stat-grades">
                <span style="color:#22c55e;">{gc.get('good',0)}G</span>
                <span style="color:#eab308;">{gc.get('partial',0)}P</span>
                <span style="color:#f97316;">{gc.get('weak',0)}W</span>
                <span style="color:#ef4444;">{gc.get('poor',0)}F</span>
            </div>
        </div>
        """

    # Task type filter buttons
    all_types = sorted(set(tr["task"]["task_type"] for tr in results))
    type_filter_buttons = '<button class="filter-btn active" onclick="filterByType(\'all\')">All</button>'
    for tt in all_types:
        color = type_colors.get(tt, "#6b7280")
        type_filter_buttons += (
            f'<button class="filter-btn" onclick="filterByType(\'{tt}\')" '
            f'style="border-color:{color};">{html.escape(tt)}</button>'
        )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ChemBench ADME Model Comparison</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f1f5f9; color: #1e293b; line-height: 1.5;
    max-width: 1200px; margin: 0 auto; padding: 24px 16px;
  }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; margin-bottom: 24px; font-size: 0.95rem; }}

  /* Summary table */
  .summary-table-wrapper {{
    background: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow-x: auto;
  }}
  .summary-table-wrapper h2 {{ font-size: 1rem; margin-bottom: 14px; color: #475569; }}
  .summary-table {{
    width: 100%; border-collapse: collapse; font-size: 0.88rem;
  }}
  .summary-table th {{
    padding: 10px 14px; text-align: center; border-bottom: 2px solid #e2e8f0;
    font-size: 0.82rem;
  }}
  .summary-table th:first-child {{ text-align: left; }}
  .summary-table td {{ padding: 8px 14px; border-bottom: 1px solid #f1f5f9; }}
  .tt-cell {{ font-weight: 600; font-size: 0.82rem; color: #475569; }}
  .score-cell {{ text-align: center; }}
  .score-val {{ font-weight: 700; font-size: 1.1rem; margin-right: 6px; }}
  .overall-val {{ font-size: 1.3rem; }}
  .mini-grade {{
    color: white; font-size: 0.65rem; font-weight: 700; padding: 2px 6px;
    border-radius: 8px; vertical-align: middle;
  }}
  .overall-row {{ background: #f8fafc; }}
  .model-subname {{ font-size: 0.65rem; color: #94a3b8; font-weight: 400; }}

  /* Model stats bar */
  .model-stats {{
    display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap;
  }}
  .model-stat-block {{
    background: white; border-radius: 10px; padding: 16px 20px; flex: 1; min-width: 160px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center;
  }}
  .model-stat-name {{ font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; }}
  .model-stat-avg {{ font-size: 2rem; font-weight: 800; }}
  .model-stat-grades {{ font-size: 0.78rem; font-weight: 600; display: flex; gap: 8px; justify-content: center; margin-top: 6px; }}

  /* Filter bar */
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

  /* Task cards */
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
  .task-endpoint {{ color: #64748b; font-size: 0.85rem; }}
  .task-number {{ margin-left: auto; color: #94a3b8; font-size: 0.8rem; }}
  .section {{ padding: 16px 20px; border-bottom: 1px solid #f1f5f9; }}
  .mol-container {{ text-align: center; overflow-x: auto; }}
  .mol-container svg {{ max-width: 100%; height: auto; }}

  /* Model panels grid */
  .model-panels-grid {{
    display: grid; grid-template-columns: repeat({len(models)}, 1fr);
    gap: 0; border-top: 1px solid #e2e8f0;
  }}
  .model-panel {{
    padding: 16px; border-right: 1px solid #f1f5f9;
  }}
  .model-panel:last-child {{ border-right: none; }}
  .model-panel-header {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
  }}
  .model-name {{ font-weight: 700; font-size: 0.88rem; }}
  .grade-badge {{
    color: white; font-size: 0.7rem; font-weight: 700; padding: 2px 8px;
    border-radius: 10px;
  }}
  .assessment-mini {{
    padding: 8px 10px; border-radius: 6px; margin-bottom: 10px; font-size: 0.8rem;
  }}
  .assess-summary-text {{ margin-bottom: 4px; color: #374151; }}
  .assess-checks {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .check-item {{ font-size: 0.75rem; font-weight: 600; }}
  .response-summary {{
    font-size: 0.85rem; color: #1e40af; background: #eff6ff; padding: 10px;
    border-radius: 6px; border: 1px solid #bfdbfe; margin-bottom: 8px;
  }}
  .full-response-details {{ font-size: 0.85rem; }}
  .full-response-details summary {{
    cursor: pointer; color: #64748b; font-weight: 500; padding: 4px 0;
  }}
  .response-box {{
    background: #f8fafc; padding: 12px; border-radius: 6px; font-size: 0.82rem;
    white-space: pre-wrap; font-family: 'SF Mono', 'Fira Code', monospace;
    border: 1px solid #e2e8f0; max-height: 400px; overflow-y: auto; margin-top: 8px;
  }}

  /* Prompt / GT details */
  .prompt-details, .gt-details {{
    padding: 8px 20px; border-bottom: 1px solid #f1f5f9;
  }}
  .prompt-details summary, .gt-details summary {{
    cursor: pointer; font-size: 0.82rem; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.5px; padding: 6px 0;
  }}
  .prompt-box {{
    background: #f8fafc; padding: 12px; border-radius: 8px; font-size: 0.85rem;
    white-space: pre-wrap; font-family: 'SF Mono', 'Fira Code', monospace;
    border: 1px solid #e2e8f0; max-height: 300px; overflow-y: auto; margin-top: 8px;
  }}
  .gt-box {{ padding: 8px 0; margin-top: 8px; }}
  .gt-row {{ padding: 3px 0; font-size: 0.85rem; }}
  .gt-key {{ font-weight: 600; color: #475569; }}
  .gt-val {{ font-family: 'SF Mono', monospace; color: #1e293b; }}

  @media (max-width: 768px) {{
    .model-panels-grid {{ grid-template-columns: 1fr; }}
    .model-panel {{ border-right: none; border-bottom: 1px solid #f1f5f9; }}
  }}
</style>
</head>
<body>

<h1>ChemBench ADME Model Comparison</h1>
<p class="subtitle">
  {len(results)} tasks &middot; Models: {', '.join(models)} &middot;
  Generated from MMP-ADME database
</p>

<div class="model-stats">
  {stats_blocks}
</div>

<div class="summary-table-wrapper">
  <h2>Scores by Task Type</h2>
  <table class="summary-table">
    <thead>
      <tr>
        <th>Task Type</th>
        {model_headers}
      </tr>
    </thead>
    <tbody>
      {summary_rows}
    </tbody>
  </table>
</div>

<div class="filter-bar">
  {type_filter_buttons}
</div>

{tasks_html}

<script>
function filterByType(type) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.task-card').forEach(card => {{
    if (type === 'all' || card.dataset.type === type) {{
      card.style.display = '';
    }} else {{
      card.style.display = 'none';
    }}
  }});
}}
</script>
</body>
</html>"""

    output_path.write_text(full_html)
    logger.info(f"Comparison report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model comparison on ADME benchmark tasks"
    )
    parser.add_argument(
        "--models", type=str, default="sonnet,opus",
        help="Comma-separated model names: sonnet, opus, haiku (default: sonnet,opus)"
    )
    parser.add_argument(
        "--n-per-type", type=int, default=4,
        help="Number of tasks to sample per type (default: 4)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML path (default: comparison_report.html)"
    )
    parser.add_argument(
        "--csv-output", type=str, default=None,
        help="Output CSV path (default: comparison_results.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task sampling (default: 42)"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in MODEL_DISPLAY_NAMES:
            logger.warning(
                f"Unknown model '{m}'. Known models: {list(MODEL_DISPLAY_NAMES.keys())}"
            )

    output_path = Path(args.output) if args.output else SCRIPT_DIR / "comparison_report.html"
    csv_path = Path(args.csv_output) if args.csv_output else SCRIPT_DIR / "comparison_results.csv"

    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)

    tasks = sample_tasks(benchmark, n_per_type=args.n_per_type, seed=args.seed)
    logger.info(
        f"Sampled {len(tasks)} tasks across "
        f"{len(set(t['task_type'] for t in tasks))} types"
    )
    logger.info(f"Running comparison with models: {models}")

    results = run_comparison(tasks, models)

    write_csv(results, models, csv_path)
    generate_comparison_html(results, models, output_path)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for model in models:
        scores = [
            tr["models"][model]["assessment"]["score"] for tr in results
        ]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {model:10s}: avg score = {avg:.1f}")
    print("=" * 60)
    print(f"Report: {output_path}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
