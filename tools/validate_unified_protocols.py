#!/usr/bin/env python3
"""Validate RGB-D ordering, timestamps, held-out splits, and ICL alignment."""

from __future__ import annotations
import argparse,bisect,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8",errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def nearest_distance(sorted_values,value):
    index=bisect.bisect_left(sorted_values,value);candidates=[]
    if index<len(sorted_values):candidates.append(abs(sorted_values[index]-value))
    if index:candidates.append(abs(sorted_values[index-1]-value))
    return min(candidates) if candidates else float("inf")


def validate(sequence):
    dataset=Path(sequence["path"]);association=dataset/sequence["association"]
    groundtruth=dataset/sequence["ground_truth"];train=dataset/"train90.txt";errors=[];warnings=[]
    for path in (dataset,association,groundtruth,train):
        if not path.exists():errors.append(f"missing:{path}")
    if errors:return {"id":sequence["id"],"errors":errors,"passed":False}
    assoc=lines(association);gt=lines(groundtruth);train_rows=lines(train)
    parsed=[]
    for index,row in enumerate(assoc):
        values=row.split()
        if len(values)<4:errors.append(f"association_columns:{index}");continue
        rgb_name=values[1].lower()
        if sequence["family"]=="REPLICA":
            if "frame" not in rgb_name or not rgb_name.endswith((".jpg",".jpeg",".png")):
                errors.append(f"rgb_order:{index}:{values[1]}")
        elif "rgb" not in rgb_name:errors.append(f"rgb_order:{index}:{values[1]}")
        if "depth" not in values[3].lower():errors.append(f"depth_order:{index}:{values[3]}")
        if not (dataset/values[1]).exists():errors.append(f"missing_rgb:{values[1]}")
        if not (dataset/values[3]).exists():errors.append(f"missing_depth:{values[3]}")
        parsed.append((float(values[0]),values[1],values[3],row))
    gt_times=sorted(float(row.split()[0]) for row in gt if len(row.split())>=8)
    unmatched=sum(nearest_distance(gt_times,row[0])>0.02 for row in parsed)
    matched=len(parsed)-unmatched
    gt_match_rate=matched/max(len(parsed),1)
    # Official TUM trajectories can contain mocap gaps and boundary RGB-D
    # frames without a nearby GT pose.  Evaluation legitimately associates
    # only the supported subset, so record this coverage and require a
    # substantial evaluable fraction instead of demanding impossible 100%
    # coverage.  Synthetic ICL/Replica trajectories are expected to be dense.
    if sequence["family"].startswith("TUM"):
        if unmatched:
            warnings.append(f"association_without_gt_within_20ms:{unmatched}")
        if matched<3 or gt_match_rate<0.70:
            errors.append(f"insufficient_gt_match_rate:{gt_match_rate:.6f}")
    elif unmatched:
        errors.append(f"association_without_gt_within_20ms:{unmatched}")
    expected_train=[row for index,row in enumerate(assoc) if index%10!=0]
    if train_rows!=expected_train:errors.append("train90_not_exact_modulo10_complement")
    heldout={row for index,row in enumerate(assoc) if index%10==0}
    if heldout&set(train_rows):errors.append("train90_heldout_overlap")
    if sequence["family"]=="ICL":
        original={float(row.split()[0]):(row.split()[3],row.split()[1])
                  for row in lines(dataset/"associations.txt") if len(row.split())>=4}
        original_gt={float(row.split()[0]) for row in lines(dataset/"groundtruth.txt") if len(row.split())>=8}
        common=sorted(set(original)&original_gt)
        expected=[(stamp/30.0,)+original[stamp] for stamp in common]
        observed=[(stamp,rgb,depth) for stamp,rgb,depth,_ in parsed]
        if len(expected)!=len(observed):errors.append(f"icl_count:{len(observed)}!={len(expected)}")
        else:
            for index,(left,right) in enumerate(zip(observed,expected)):
                if abs(left[0]-right[0])>5e-7 or left[1:]!=right[1:]:
                    errors.append(f"icl_timestamp_intersection_mismatch:{index}");break
    return {"id":sequence["id"],"family":sequence["family"],"association_frames":len(assoc),
            "groundtruth_poses":len(gt),"train90_frames":len(train_rows),
            "heldout_frames":len(assoc)-len(train_rows),"gt_matched_frames":matched,
            "gt_match_rate":gt_match_rate,"warnings":warnings,
            "errors":errors,"passed":not errors}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,
        default=ROOT/"evaluation/unified_benchmark/manifest.json");parser.add_argument("--output",type=Path)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    reports=[validate(sequence) for sequence in manifest["sequences"]]
    low={sequence["id"] for sequence in manifest["sequences"] if sequence.get("low_texture")}
    required={"tum_fr3_cabinet","tum_fr3_large_cabinet","tum_fr3_notex_far","tum_fr3_notex_near"}
    summary={"sequences":reports,"sequence_count":len(reports),"low_texture_ids":sorted(low),
             "required_low_texture_present":required<=low,
             "passed":all(item["passed"] for item in reports) and required<=low}
    output=args.output or args.manifest.parent/"protocol_audit.json"
    output.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"sequence_count":len(reports),"failed":[item["id"] for item in reports if not item["passed"]],
                      "required_low_texture_present":summary["required_low_texture_present"]},indent=2))
    raise SystemExit(0 if summary["passed"] else 2)


if __name__=="__main__":main()
