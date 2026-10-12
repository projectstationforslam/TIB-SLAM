#!/usr/bin/env python3
"""Convert NICE-SLAM Replica trajectories and image names to TUM RGB-D files."""

from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np


def rot_to_quat(r):
    # Stable branch formulation; output is qx qy qz qw.
    trace=np.trace(r)
    if trace>0:
        s=math.sqrt(trace+1.0)*2; w=.25*s
        x=(r[2,1]-r[1,2])/s; y=(r[0,2]-r[2,0])/s; z=(r[1,0]-r[0,1])/s
    else:
        i=int(np.argmax(np.diag(r)))
        if i==0:
            s=math.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2; w=(r[2,1]-r[1,2])/s
            x=.25*s; y=(r[0,1]+r[1,0])/s; z=(r[0,2]+r[2,0])/s
        elif i==1:
            s=math.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2; w=(r[0,2]-r[2,0])/s
            x=(r[0,1]+r[1,0])/s; y=.25*s; z=(r[1,2]+r[2,1])/s
        else:
            s=math.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2; w=(r[1,0]-r[0,1])/s
            x=(r[0,2]+r[2,0])/s; y=(r[1,2]+r[2,1])/s; z=.25*s
    q=np.array([x,y,z,w]);q/=np.linalg.norm(q);return q


def main():
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--fps",type=float,default=30);a=p.parse_args()
    for sequence in sorted(x for x in a.root.iterdir() if x.is_dir() and (x/"traj.txt").exists()):
        poses=np.loadtxt(sequence/"traj.txt").reshape(-1,4,4)
        frames=sorted((sequence/"results").glob("frame*.jpg"));depths=sorted((sequence/"results").glob("depth*.png"))
        count=min(len(poses),len(frames),len(depths));assoc=[];gt=[]
        for i in range(count):
            stamp=i/a.fps; assoc.append(f"{stamp:.6f} results/{frames[i].name} {stamp:.6f} results/{depths[i].name}")
            t=poses[i,:3,3];q=rot_to_quat(poses[i,:3,:3])
            gt.append(f"{stamp:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
        (sequence/"associations.txt").write_text("\n".join(assoc)+"\n",encoding="utf-8")
        (sequence/"groundtruth.txt").write_text("\n".join(gt)+"\n",encoding="utf-8")
        print(f"{sequence.name}: {count} frames")


if __name__=="__main__":main()
