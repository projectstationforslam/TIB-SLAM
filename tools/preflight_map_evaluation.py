#!/usr/bin/env python3
"""Fail early unless the strict rendering/map metric stack is operational."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluate_rendering_v8 import load_lpips, ssim


def main() -> None:
    torch, model = load_lpips(False)
    sample = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        value = float(model(sample, sample).item())
    structural = ssim(np.zeros((32, 32, 3), dtype=np.float32),
                      np.zeros((32, 32, 3), dtype=np.float32))
    if not np.isfinite(value) or not np.isfinite(structural):
        raise RuntimeError("non-finite LPIPS/SSIM preflight result")
    report = {
        "status": "complete",
        "torch_version": torch.__version__,
        "lpips_identical_image": value,
        "ssim_identical_image": structural,
        "device": "cpu",
    }
    output = (Path(__file__).resolve().parents[1] / "evaluation" /
              "unified_benchmark" / "map_evaluation_preflight.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
