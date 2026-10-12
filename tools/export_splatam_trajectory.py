#!/usr/bin/env python3
"""Export SplaTAM relative w2c camera parameters to a TUM-format c2w trajectory."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_wxyz(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--rgb-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    params = np.load(args.params)
    rotations = params["cam_unnorm_rots"][0].T
    translations = params["cam_trans"][0].T
    raw_timestamps = [float(line.split()[0]) for line in args.rgb_list.read_text(encoding="utf-8").splitlines()
                      if line.strip() and not line.startswith("#")]
    # Match datasets/gradslam_datasets/tum.py exactly: retain a frame only when
    # it is more than 1/32 s after the last retained RGB timestamp.
    selected = [0]
    for idx in range(1, len(raw_timestamps)):
        if raw_timestamps[idx] - raw_timestamps[selected[-1]] > 1.0 / 32.0:
            selected.append(idx)
    timestamps = [raw_timestamps[idx] for idx in selected]
    timestamps = timestamps[:len(rotations)]
    if len(timestamps) != len(rotations):
        raise RuntimeError(f"timestamp/pose mismatch: {len(timestamps)} vs {len(rotations)}")
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for timestamp, q, t in zip(timestamps, rotations, translations):
        w2c = np.eye(4)
        w2c[:3, :3] = rotation_wxyz(q)
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        qxyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        values = [timestamp, *c2w[:3, 3], *qxyzw]
        lines.append(" ".join(f"{value:.9f}" for value in values))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rotations)} poses to {args.output}")


if __name__ == "__main__":
    main()
