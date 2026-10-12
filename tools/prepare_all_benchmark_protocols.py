#!/usr/bin/env python3
"""Generate train90 association files for every currently available manifest entry."""

from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"evaluation/unified_benchmark/manifest.json").read_text())
for sequence in manifest["sequences"]:
    dataset=Path(sequence["path"]);source=dataset/sequence["association"]
    if not source.exists():
        print(f"SKIP {sequence['id']}: missing {source}");continue
    rows=[x for x in source.read_text(encoding="utf-8",errors="replace").splitlines()
          if x.strip() and not x.lstrip().startswith("#")]
    train=[x for i,x in enumerate(rows) if i%10!=0]
    target=dataset/"train90.txt";target.write_text("\n".join(train)+"\n",encoding="utf-8")
    print(f"{sequence['id']}: train={len(train)} held_out={len(rows)-len(train)}")
