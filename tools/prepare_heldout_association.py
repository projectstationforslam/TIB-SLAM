#!/usr/bin/env python3
"""Create a leakage-free train association by removing one deterministic residue."""

from __future__ import annotations
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("association",type=Path)
    p.add_argument("--modulo",type=int,default=10);p.add_argument("--residue",type=int,default=0)
    p.add_argument("--output",type=Path);a=p.parse_args()
    rows=[x for x in a.association.read_text(encoding="utf-8",errors="replace").splitlines()
          if x.strip() and not x.lstrip().startswith("#")]
    train=[x for i,x in enumerate(rows) if i%a.modulo!=a.residue]
    target=a.output or a.association.with_name("train90.txt")
    target.write_text("\n".join(train)+"\n",encoding="utf-8")
    print(f"{target}: train={len(train)} held_out={len(rows)-len(train)}")


if __name__=="__main__":main()
