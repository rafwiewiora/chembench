#!/usr/bin/env python
"""
Run exploration tasks: test whether models persist through disappointing results
to discover non-additive SAR surprises.

The model gets a molecule to optimize and can query an MMP oracle.
The oracle returns population statistics for transforms. When the model
"applies" a transform, it gets the actual result — which for most transforms
matches population expectations, but for one (the outlier) deviates dramatically.

Usage:
    python run_exploration.py
    python run_exploration.py --n-tasks 5 --model claude-sonnet-4-6
    python run_exploration.py --task-id exploration_001
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
TASKS_FILE = SCRIPT_DIR / "exploration_tasks.json"
MAX_TURNS = 15


def load_tasks(path=TASKS_FILE):
    with open(path) as f:
        return json.load(f)


def build_system_prompt():
    return """You are a medicinal chemist optimizing a compound's ADME properties.

You have access to an MMP (matched molecular pair) oracle that knows about structural
transformations and their typical effects on the property you're optimizing.

You can take these actions by writing them in your response:

ACTION: LIST_TRANSFORMS
  → Shows all available single-site modifications with population statistics

ACTION: APPLY <transform_id>
  → Applies a specific transformation and returns the actual experimental result

ACTION: PROPOSE_COMBINATION <id1> <id2>
  → Asks about applying two transforms together. The oracle will return experimental
    data if available, or state that no data exists for that combination.

ACTION: DONE <explanation>
  → End the optimization with your conclusion and reasoning.

Think step by step. After each result, reason about what you learned and what to try next.
If a result surprises you (far from the population average), reason about WHY the
molecular context might cause this deviation."""


def build_initial_prompt(task):
    return f"""Optimize the following compound's {task['property_name']}.

**Starting compound:**
  SMILES: {task['start_smiles']}
  ChEMBL ID: {task['start_mol']}
  Current value: {task['start_value']:.2f} ({task['better_means']} is better)

Your goal: find modifications that improve this property. You have access to an MMP
database oracle that can tell you about available transformations and their typical effects.

Start by listing available transforms, then explore."""


def format_transform_list(task):
    lines = [
        f"Available single-site transformations for {task['start_mol']}:",
        f"(Population statistics from MMP database — mean ± std across n examples)",
        "",
    ]
    for i, tx in enumerate(task['transforms']):
        lines.append(
            f"  [{i}] {tx['var_from']} → {tx['var_to']}"
            f"    (pop: {tx['pop_mean']:+.2f} ± {tx['pop_std']:.2f}, n={tx['pop_n']})"
        )
    lines.append("")
    lines.append("Use ACTION: APPLY <id> to test a specific transformation on your molecule.")
    return "\n".join(lines)


def format_apply_result(task, transform_id):
    if transform_id < 0 or transform_id >= len(task['transforms']):
        return f"Error: transform ID {transform_id} not found. Valid range: 0-{len(task['transforms'])-1}"

    tx = task['transforms'][transform_id]
    result_value = task['start_value'] + tx['actual_delta']

    lines = [
        f"Result of applying transform [{transform_id}]: {tx['var_from']} → {tx['var_to']}",
        f"",
        f"  Starting value: {task['start_value']:.2f}",
        f"  Result value:   {result_value:.3f}  (delta = {tx['actual_delta']:+.3f})",
        f"  Population expected: {tx['pop_mean']:+.2f} ± {tx['pop_std']:.2f} (n={tx['pop_n']})",
    ]

    z = tx['z_score']
    if abs(z) >= 3:
        lines.append(f"  ⚠ This result is {abs(z):.1f} standard deviations from the population mean!")
    elif abs(z) >= 2:
        lines.append(f"  Note: this result is {abs(z):.1f}σ from the population mean.")

    return "\n".join(lines)


def format_combination_result(task, id1, id2):
    if id1 < 0 or id1 >= len(task['transforms']) or id2 < 0 or id2 >= len(task['transforms']):
        return f"Error: invalid transform IDs. Valid range: 0-{len(task['transforms'])-1}"

    tx1 = task['transforms'][id1]
    tx2 = task['transforms'][id2]

    additive_pred = tx1['actual_delta'] + tx2['actual_delta']
    pop_additive = tx1['pop_mean'] + tx2['pop_mean']

    lines = [
        f"Combination query: [{id1}] + [{id2}]",
        f"  {tx1['var_from']} → {tx1['var_to']}  AND  {tx2['var_from']} → {tx2['var_to']}",
        f"",
        f"  No direct experimental data for this exact combination.",
        f"  Additive prediction from individual results: {additive_pred:+.3f}",
        f"  Additive prediction from population means: {pop_additive:+.3f}",
        f"",
        f"  Note: real SAR can be non-additive. This prediction assumes independence.",
    ]
    return "\n".join(lines)


def parse_actions(response_text):
    actions = []
    for line in response_text.split('\n'):
        line = line.strip()
        if line.startswith('ACTION:'):
            action_str = line[7:].strip()
            if action_str == 'LIST_TRANSFORMS':
                actions.append(('LIST', None))
            elif action_str.startswith('APPLY'):
                try:
                    tid = int(action_str.split()[1])
                    actions.append(('APPLY', tid))
                except (IndexError, ValueError):
                    actions.append(('ERROR', 'APPLY requires a transform ID number'))
            elif action_str.startswith('PROPOSE_COMBINATION'):
                try:
                    parts = action_str.split()
                    actions.append(('COMBO', (int(parts[1]), int(parts[2]))))
                except (IndexError, ValueError):
                    actions.append(('ERROR', 'PROPOSE_COMBINATION requires two transform IDs'))
            elif action_str.startswith('DONE'):
                explanation = action_str[4:].strip()
                actions.append(('DONE', explanation))
            else:
                actions.append(('ERROR', f'Unknown action: {action_str}'))
    return actions


def run_task(client, task, model="claude-sonnet-4-6"):
    system = build_system_prompt()
    messages = [{"role": "user", "content": build_initial_prompt(task)}]

    transcript = []
    transforms_applied = []
    combos_proposed = []
    found_outlier = False
    done = False
    turns = 0

    log.info(f"Starting task {task['task_id']} ({task['endpoint']}, {task['start_mol']})")
    log.info(f"  Outlier: transform {task['best_outlier_idx']}, z={task['best_outlier_z']:+.1f}")

    while not done and turns < MAX_TURNS:
        turns += 1
        log.info(f"  Turn {turns}/{MAX_TURNS}")

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        assistant_text = response.content[0].text
        transcript.append({"role": "assistant", "content": assistant_text, "turn": turns})

        actions = parse_actions(assistant_text)
        if not actions:
            actions = [('LIST', None)]

        oracle_responses = []
        for action_type, action_data in actions:
            if action_type == 'LIST':
                oracle_responses.append(format_transform_list(task))
            elif action_type == 'APPLY':
                oracle_responses.append(format_apply_result(task, action_data))
                transforms_applied.append(action_data)
                if action_data == task['best_outlier_idx']:
                    found_outlier = True
                    log.info(f"  → Found the outlier on turn {turns}!")
            elif action_type == 'COMBO':
                oracle_responses.append(format_combination_result(task, *action_data))
                combos_proposed.append(action_data)
            elif action_type == 'DONE':
                done = True
                oracle_responses.append("Optimization complete.")
            elif action_type == 'ERROR':
                oracle_responses.append(f"Error: {action_data}")

        oracle_text = "\n\n---\n\n".join(oracle_responses)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": f"ORACLE RESPONSE:\n\n{oracle_text}"})
        transcript.append({"role": "oracle", "content": oracle_text, "turn": turns})

    # Score the run
    outlier_tx = task['transforms'][task['best_outlier_idx']]
    result = {
        'task_id': task['task_id'],
        'endpoint': task['endpoint'],
        'start_mol': task['start_mol'],
        'start_value': task['start_value'],
        'n_turns': turns,
        'n_transforms_applied': len(transforms_applied),
        'transforms_applied': transforms_applied,
        'n_combos_proposed': len(combos_proposed),
        'found_outlier': found_outlier,
        'outlier_turn': next((t['turn'] for t in transcript
                             if t['role'] == 'assistant' and
                             f'APPLY {task["best_outlier_idx"]}' in t.get('content', '')), None),
        'outlier_z': task['best_outlier_z'],
        'outlier_actual': outlier_tx['actual_delta'],
        'outlier_pop_mean': outlier_tx['pop_mean'],
        'gave_up_early': done and not found_outlier and len(transforms_applied) < len(task['transforms']) // 2,
        'tried_everything': len(set(transforms_applied)) >= len(task['transforms']) - 1,
        'transcript': transcript,
    }

    log.info(f"  Result: found_outlier={found_outlier}, turns={turns}, "
             f"applied={len(transforms_applied)}/{len(task['transforms'])}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run exploration tasks")
    parser.add_argument("--n-tasks", type=int, default=3)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--task-id", help="Run a specific task by ID")
    parser.add_argument("--hidden-only", action="store_true",
                        help="Only run tasks where the outlier is well-hidden (rank > 50th pct)")
    parser.add_argument("--output-dir", default="results/exploration")
    args = parser.parse_args()

    tasks = load_tasks()
    log.info(f"Loaded {len(tasks)} exploration tasks")

    if args.task_id:
        tasks = [t for t in tasks if t['task_id'] == args.task_id]
        if not tasks:
            log.error(f"Task {args.task_id} not found")
            sys.exit(1)
    else:
        if args.hidden_only:
            tasks = [t for t in tasks if t.get('outlier_well_hidden', False)]
            log.info(f"Filtered to {len(tasks)} well-hidden outlier tasks")
        tasks = tasks[:args.n_tasks]

    client = anthropic.Anthropic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for task in tasks:
        result = run_task(client, task, model=args.model)
        all_results.append(result)

        with open(output_dir / f"{task['task_id']}_result.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Summary
    n_found = sum(1 for r in all_results if r['found_outlier'])
    n_gaveup = sum(1 for r in all_results if r['gave_up_early'])
    avg_turns = sum(r['n_turns'] for r in all_results) / len(all_results) if all_results else 0
    avg_applied = sum(r['n_transforms_applied'] for r in all_results) / len(all_results) if all_results else 0

    summary = {
        'model': args.model,
        'n_tasks': len(all_results),
        'n_found_outlier': n_found,
        'n_gave_up_early': n_gaveup,
        'avg_turns': round(avg_turns, 1),
        'avg_transforms_applied': round(avg_applied, 1),
        'persistence_rate': round(1 - n_gaveup / len(all_results), 2) if all_results else 0,
        'discovery_rate': round(n_found / len(all_results), 2) if all_results else 0,
        'results': [{k: v for k, v in r.items() if k != 'transcript'} for r in all_results],
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY ({args.model})")
    log.info(f"  Tasks run: {len(all_results)}")
    log.info(f"  Found outlier: {n_found}/{len(all_results)} ({summary['discovery_rate']:.0%})")
    log.info(f"  Gave up early: {n_gaveup}/{len(all_results)}")
    log.info(f"  Avg turns: {avg_turns:.1f}")
    log.info(f"  Avg transforms tried: {avg_applied:.1f}")
    log.info(f"  Persistence rate: {summary['persistence_rate']:.0%}")


if __name__ == "__main__":
    main()
