#!/usr/bin/env python3
"""Unified SE(3)-aligned APE, one-step RPE, coverage, and failure metrics."""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def quat_to_rot(q):
    q = np.asarray(q, float)
    q /= max(np.linalg.norm(q), 1e-15)
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def load_tum(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        v = line.split()
        if len(v) < 8: continue
        t = np.eye(4)
        t[:3, :3] = quat_to_rot([float(x) for x in v[4:8]])
        t[:3, 3] = [float(x) for x in v[1:4]]
        rows.append((float(v[0]), t))
    return rows


def associate(reference, estimate, tolerance):
    candidates = []
    j0 = 0
    for i, (rt, _) in enumerate(reference):
        while j0 < len(estimate) and estimate[j0][0] < rt - tolerance: j0 += 1
        j = j0
        while j < len(estimate) and estimate[j][0] <= rt + tolerance:
            candidates.append((abs(rt-estimate[j][0]), i, j)); j += 1
    used_r, used_e, matches = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_r and j not in used_e:
            used_r.add(i); used_e.add(j); matches.append((i, j))
    return sorted(matches, key=lambda x: reference[x[0]][0])


def alignment(est, ref):
    ec, rc = est.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((est-ec).T @ (ref-rc))
    d = np.eye(3); d[-1,-1] = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ d @ u.T
    return r, rc-r@ec


def rotation_angle(r):
    return math.acos(float(np.clip((np.trace(r)-1)/2, -1, 1)))


def count_inputs(path: Path):
    return sum(1 for x in path.read_text(encoding="utf-8", errors="replace").splitlines()
               if x.strip() and not x.lstrip().startswith("#"))


def input_timestamps(path: Path):
    result=[]
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):continue
        values=line.split()
        if values:result.append(float(values[0]))
    return result


def loss_statistics(timestamps,estimate,tolerance):
    reference=[(stamp,np.eye(4)) for stamp in timestamps]
    matched={index for index,_ in associate(reference,estimate,tolerance)}
    losses=[index not in matched for index in range(len(timestamps))]
    episodes=longest=current=0
    for lost in losses:
        if lost:
            if current==0:episodes+=1
            current+=1;longest=max(longest,current)
        else:current=0
    return len(matched),sum(losses),episodes,longest


def relative_errors(gt, est, matches, delta_s=None, tolerance_s=0.05):
    pairs=[]
    if delta_s is None:
        pairs=list(zip(range(len(matches)-1),range(1,len(matches))))
    else:
        times=np.array([gt[gi][0] for gi,_ in matches],float)
        for i,timestamp in enumerate(times):
            target=timestamp+delta_s;j=int(np.searchsorted(times,target))
            candidates=[k for k in (j-1,j) if i<k<len(times)]
            if not candidates:continue
            best=min(candidates,key=lambda k:abs(times[k]-target))
            if abs(times[best]-target)<=tolerance_s:pairs.append((i,best))
    trans,rot=[],[]
    for first,second in pairs:
        gi0,ei0=matches[first];gi1,ei1=matches[second]
        dg=np.linalg.inv(gt[gi0][1]) @ gt[gi1][1]
        de=np.linalg.inv(est[ei0][1]) @ est[ei1][1]
        err=np.linalg.inv(dg) @ de
        trans.append(np.linalg.norm(err[:3,3]));rot.append(math.degrees(rotation_angle(err[:3,:3])))
    return trans,rot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth", required=True, type=Path)
    p.add_argument("--estimate", required=True, type=Path)
    p.add_argument("--associations", required=True, type=Path)
    p.add_argument("--tolerance", default=0.02, type=float)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    gt, est = load_tum(a.ground_truth), load_tum(a.estimate)
    matches = associate(gt, est, a.tolerance)
    if len(matches) < 3: raise SystemExit("fewer than three timestamp matches")
    gtp = np.array([gt[i][1][:3,3] for i,_ in matches])
    esp = np.array([est[j][1][:3,3] for _,j in matches])
    ar, at = alignment(esp, gtp)
    ape = np.linalg.norm((ar@esp.T).T+at-gtp, axis=1)
    trans,rot=relative_errors(gt,est,matches)
    trans_1s,rot_1s=relative_errors(gt,est,matches,delta_s=1.0,tolerance_s=0.05)
    timestamps=input_timestamps(a.associations);n_inputs=len(timestamps)
    input_matches,lost_frames,loss_episodes,longest_loss=loss_statistics(
        timestamps,est,a.tolerance)
    result = {
        "input_frames": n_inputs, "estimated_poses": len(est), "matches": len(matches),
        "tracking_input_matches":input_matches,"tracking_lost_frames":lost_frames,
        "tracking_loss_episodes":loss_episodes,"longest_tracking_loss_frames":longest_loss,
        "success_rate": input_matches/max(n_inputs,1), "failure_rate": 1-input_matches/max(n_inputs,1),
        "ape_rmse_m": math.sqrt(float(np.mean(ape*ape))), "ape_median_m": float(np.median(ape)),
        "ape_p95_m": float(np.quantile(ape,0.95)),
        "rpe_trans_rmse_m": math.sqrt(float(np.mean(np.square(trans)))),
        "rpe_rot_rmse_deg": math.sqrt(float(np.mean(np.square(rot)))),
        "rpe_1s_trans_rmse_m": math.sqrt(float(np.mean(np.square(trans_1s)))) if trans_1s else None,
        "rpe_1s_rot_rmse_deg": math.sqrt(float(np.mean(np.square(rot_1s)))) if rot_1s else None,
        "rpe_1s_pairs": len(trans_1s),
        "association_tolerance_s": a.tolerance, "alignment": "rigid_SE3"
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
