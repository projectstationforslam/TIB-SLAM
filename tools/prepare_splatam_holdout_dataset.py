#!/usr/bin/env python3
"""Build list-only TUM views for SplaTAM while leaving source images untouched."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evaluation" / "strict_baselines_20260828" / "protocol"
ASSOCIATIONS = PROTOCOL / "cabinet_train90.txt"
OUT = PROTOCOL / "splatam_cabinet_train90"
SOURCE = Path("E:/linux0920/dataset/rgbd_dataset_freiburg3_cabinet")


def main() -> None:
    rows = [line.split() for line in ASSOCIATIONS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    OUT.mkdir(parents=True, exist_ok=True)
    rgb_lines = ["# SplaTAM strict train90 RGB list"]
    depth_lines = ["# SplaTAM strict train90 depth list"]
    for rgb_ts, rgb_path, depth_ts, depth_path in rows:
        rgb_lines.append(f"{rgb_ts} /data/rgbd_dataset_freiburg3_cabinet/{rgb_path}")
        depth_lines.append(f"{depth_ts} /data/rgbd_dataset_freiburg3_cabinet/{depth_path}")
    (OUT / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (OUT / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (OUT / "groundtruth.txt").write_text(
        (SOURCE / "groundtruth.txt").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {len(rows)} training frames to {OUT}")


if __name__ == "__main__":
    main()
