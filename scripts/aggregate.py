"""Aggregate CRPS-sum across independent reproduction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()

    values = []
    for path in args.reports:
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        values.append(float(report["metrics"]["CRPS_sum"]))
    array = np.asarray(values, dtype=np.float64)
    standard_error = (
        float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    )
    print(
        json.dumps(
            {
                "runs": len(values),
                "CRPS_sum_mean": float(array.mean()),
                "CRPS_sum_standard_error": standard_error,
                "values": values,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
