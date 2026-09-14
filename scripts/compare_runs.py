#!/usr/bin/env python3
"""Print a side-by-side comparison table of metrics across W&B runs.

Queries the Weights & Biases API for runs in a project (optionally filtered
by group or tag) and reports the latest value logged for a fixed set of
metrics. Meant for quickly comparing several short exploratory runs (e.g. a
prediction-target or action-injection sweep) without tab-switching through
the W&B UI.

Requires being logged in to W&B (`wandb login`) with read access to the
project. Uses the same WANDB_ENTITY / WANDB_PROJECT env vars as training.

Usage:
    python3 scripts/compare_runs.py --group pred_target_sweep_2026-09-12
    python3 scripts/compare_runs.py --tag pred_target_sweep
    python3 scripts/compare_runs.py --run-id abc123 --run-id def456
"""

import argparse
import os
import sys

import wandb

METRICS = ["train_loss", "gradient_norm", "val_loss", "val_eval/psnr", "val_eval/ssim", "val_eval/lpips", "val_eval/fid", "val_eval/fvd"]


def fetch_runs(api, project, group=None, tag=None, run_ids=None):
    if run_ids:
        return [api.run(f"{project}/{run_id}") for run_id in run_ids]

    filters = {}
    if group:
        filters["group"] = group
    if tag:
        filters["tags"] = tag
    return list(api.runs(project, filters=filters))


def latest_metrics(run):
    # scan_history() walks every logged row without requiring all METRICS to
    # be present on the same step (history(keys=...) does an intersection
    # join across keys, which returns nothing when metrics are logged on
    # different cadences, e.g. train_loss every step vs. val_loss only on
    # validation checks). Take the last non-null value per metric instead.
    result = {}
    for row in run.scan_history():
        for tag in METRICS:
            if row.get(tag) is not None:
                result[tag] = (row.get("_step"), row[tag])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=None, help="entity/project, or project if WANDB_ENTITY is set. Defaults to WANDB_ENTITY/WANDB_PROJECT env vars.")
    parser.add_argument("--group", default=None, help="Only compare runs in this W&B group (see wandb.group config).")
    parser.add_argument("--tag", default=None, help="Only compare runs with this W&B tag (see wandb.tags config).")
    parser.add_argument("--run-id", action="append", dest="run_ids", default=None, help="Compare specific run IDs instead of filtering by group/tag. Repeatable.")
    args = parser.parse_args()

    project = args.project
    if project is None:
        entity = os.environ.get("WANDB_ENTITY")
        proj = os.environ.get("WANDB_PROJECT", "nano-world-model")
        project = f"{entity}/{proj}" if entity else proj

    if not args.group and not args.tag and not args.run_ids:
        parser.error("Provide --group, --tag, or one or more --run-id to select which runs to compare.")

    api = wandb.Api()
    runs = fetch_runs(api, project, group=args.group, tag=args.tag, run_ids=args.run_ids)
    if not runs:
        print(f"No runs found in {project} matching the given filter.", file=sys.stderr)
        return 1

    rows = []
    for run in runs:
        metrics = latest_metrics(run)
        row = {"run": run.name or run.id}
        for tag in METRICS:
            step, value = metrics.get(tag, (None, None))
            row[tag] = f"{value:.4f} (step {step})" if value is not None else "-"
        rows.append(row)

    headers = ["run"] + METRICS
    widths = [max(len(str(row[h])) for row in rows + [dict(zip(headers, headers))]) for h in headers]

    def fmt_row(row):
        return "  ".join(str(row[h]).ljust(w) for h, w in zip(headers, widths))

    print(fmt_row(dict(zip(headers, headers))))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
