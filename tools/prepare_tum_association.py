#!/usr/bin/env python3
"""Create deterministic nearest-time RGB/depth associations for a TUM sequence."""

from __future__ import annotations
import argparse
from pathlib import Path


def read_list(path):
    rows=[]
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        stamp,name=line.split()[:2]; rows.append((float(stamp),stamp,name))
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("sequence",type=Path)
    p.add_argument("--tolerance",type=float,default=.02); p.add_argument("--output",default="associate.txt")
    a=p.parse_args(); rgb=read_list(a.sequence/"rgb.txt"); depth=read_list(a.sequence/"depth.txt")
    candidates=[]; j0=0
    for i,(rt,_,_) in enumerate(rgb):
        while j0<len(depth) and depth[j0][0]<rt-a.tolerance: j0+=1
        j=j0
        while j<len(depth) and depth[j][0]<=rt+a.tolerance:
            candidates.append((abs(rt-depth[j][0]),i,j)); j+=1
    used_r=set();used_d=set();matches=[]
    for _,i,j in sorted(candidates):
        if i not in used_r and j not in used_d:
            used_r.add(i);used_d.add(j);matches.append((i,j))
    matches.sort(key=lambda x:rgb[x[0]][0])
    lines=[f"{rgb[i][1]} {rgb[i][2]} {depth[j][1]} {depth[j][2]}" for i,j in matches]
    target=a.sequence/a.output; target.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"{target}: {len(lines)} pairs from {len(rgb)} RGB and {len(depth)} depth frames")


if __name__=="__main__":main()
