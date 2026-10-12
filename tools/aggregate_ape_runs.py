#!/usr/bin/env python3
"""Aggregate aligned TUM APE over independent trajectory runs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate_ape_tum import align_se3, associate, load_tum


def evaluate(reference, estimate_path, max_difference):
    estimate = load_tum(estimate_path)
    matches = associate(reference, estimate, max_difference)
    if len(matches) < 3:
        raise ValueError("fewer than three timestamp matches")
    reference_positions = np.asarray([reference[i][1] for i, _ in matches])
    estimate_positions = np.asarray([estimate[j][1] for _, j in matches])
    aligned = align_se3(estimate_positions, reference_positions)
    errors = np.linalg.norm(aligned - reference_positions, axis=1)
    return {
        "matches": len(matches),
        "rmse_m": math.sqrt(float(np.mean(errors ** 2))),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "max_m": float(np.max(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("trajectories", nargs="+", type=Path)
    parser.add_argument("--max-difference", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_tum(args.reference)
    runs = []
    for path in args.trajectories:
        try:
            metrics = evaluate(reference, path, args.max_difference)
            metrics.update({"trajectory": str(path), "success": True})
        except (OSError, ValueError) as error:
            metrics = {"trajectory": str(path), "success": False,
                       "error": str(error)}
        runs.append(metrics)

    rmse = np.asarray([run["rmse_m"] for run in runs if run["success"]])
    summary = {
        "attempted_runs": len(runs),
        "successful_runs": int(rmse.size),
        "success_rate": float(rmse.size / len(runs)) if runs else 0.0,
        "rmse_mean_m": float(np.mean(rmse)) if rmse.size else None,
        "rmse_sample_std_m": float(np.std(rmse, ddof=1)) if rmse.size > 1 else 0.0,
        "rmse_median_m": float(np.median(rmse)) if rmse.size else None,
        "rmse_min_m": float(np.min(rmse)) if rmse.size else None,
        "rmse_max_m": float(np.max(rmse)) if rmse.size else None,
        "runs": runs,
    }
    serialized = json.dumps(summary, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
