#!/usr/bin/env python3
"""Compute translational APE RMSE for two TUM trajectories after SE(3) alignment."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_tum(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 4:
            continue
        rows.append((float(values[0]), np.asarray(values[1:4], dtype=float)))
    return rows


def associate(reference, estimate, max_difference):
    candidates = []
    for estimate_index, (estimate_time, _) in enumerate(estimate):
        for reference_index, (reference_time, _) in enumerate(reference):
            difference = abs(estimate_time - reference_time)
            if difference <= max_difference:
                candidates.append((difference, reference_index, estimate_index))
    candidates.sort()
    used_reference = set()
    used_estimate = set()
    matches = []
    for _, reference_index, estimate_index in candidates:
        if reference_index in used_reference or estimate_index in used_estimate:
            continue
        used_reference.add(reference_index)
        used_estimate.add(estimate_index)
        matches.append((reference_index, estimate_index))
    matches.sort(key=lambda item: estimate[item[1]][0])
    return matches


def align_se3(estimate_positions, reference_positions):
    estimate_center = estimate_positions.mean(axis=0)
    reference_center = reference_positions.mean(axis=0)
    estimate_centered = estimate_positions - estimate_center
    reference_centered = reference_positions - reference_center
    u, _, vt = np.linalg.svd(estimate_centered.T @ reference_centered)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = reference_center - rotation @ estimate_center
    return (rotation @ estimate_positions.T).T + translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("--max-difference", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_tum(args.reference)
    estimate = load_tum(args.estimate)
    matches = associate(reference, estimate, args.max_difference)
    if len(matches) < 3:
        raise SystemExit("fewer than three timestamp matches")
    reference_positions = np.asarray([reference[i][1] for i, _ in matches])
    estimate_positions = np.asarray([estimate[j][1] for _, j in matches])
    aligned = align_se3(estimate_positions, reference_positions)
    errors = np.linalg.norm(aligned - reference_positions, axis=1)
    result = {
        "matches": len(matches),
        "rmse_m": math.sqrt(float(np.mean(errors ** 2))),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "std_m": float(np.std(errors)),
        "min_m": float(np.min(errors)),
        "max_m": float(np.max(errors)),
    }
    print(f"matches {result['matches']}")
    for name in ("rmse", "mean", "median", "std", "min", "max"):
        print(f"{name} {result[name + '_m']:.9f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
