#!/usr/bin/env python3
"""Normalize ICL-NUIM RGB/depth order and frame indices to 30-Hz TUM time."""

from __future__ import annotations
import argparse
from pathlib import Path


def data_lines(path):
    return [x.strip() for x in path.read_text(encoding="utf-8",errors="replace").splitlines()
            if x.strip() and not x.lstrip().startswith("#")]


def main():
    p=argparse.ArgumentParser();p.add_argument("sequence",type=Path);a=p.parse_args()
    # The distributed ICL association uses image indices 0..N, while the
    # Freiburg-format ground truth uses 1..N.  Pair by the original timestamp
    # intersection; positional zipping would shift every RGB-D frame by one.
    source={}
    for line in data_lines(a.sequence/"associations.txt"):
        values=line.split()
        if len(values)<4:continue
        source[float(values[0])] = (values[3],values[1])  # RGB, depth
    gt={float(line.split()[0]):line.split()[1:8]
        for line in data_lines(a.sequence/"groundtruth.txt") if len(line.split())>=8}
    common=sorted(set(source)&set(gt))
    if not common:raise RuntimeError(f"no association/ground-truth timestamp overlap in {a.sequence}")
    associations=[]; groundtruth=[]
    for original in common:
        stamp=original/30.0;rgb_rel,depth_rel=source[original]
        associations.append(f"{stamp:.6f} {rgb_rel} {stamp:.6f} {depth_rel}")
        groundtruth.append(f"{stamp:.6f} "+" ".join(gt[original]))
    (a.sequence/"associate_rgbd.txt").write_text("\n".join(associations)+"\n",encoding="utf-8")
    (a.sequence/"groundtruth_30hz.txt").write_text("\n".join(groundtruth)+"\n",encoding="utf-8")
    train=[line for index,line in enumerate(associations) if index%10!=0]
    (a.sequence/"train90.txt").write_text("\n".join(train)+"\n",encoding="utf-8")
    print(f"{a.sequence}: {len(common)} timestamp-matched frames; {len(train)} train90 frames")


if __name__=="__main__":main()
