"""Benchmark the standard Tetris geometries with the analytic forward model."""

import _bootstrap  # noqa: F401

import numpy as np

from gammadf import GEOMETRIES, train_and_evaluate as _train_and_evaluate
from gammadf.analytic import generate_dataset


def evaluate_geometry(name, pixels):
    """Train/evaluate a single geometry; shared by the random-search script."""
    X, angles = generate_dataset(pixels)
    result = _train_and_evaluate(X, angles)
    result["geometry"] = name
    return result


def main():
    results = sorted(
        (evaluate_geometry(name, pixels) for name, pixels in GEOMETRIES.items()),
        key=lambda r: r["mean_error_deg"],
    )

    print("\nGeometry benchmark")
    print("-" * 60)
    print(f"{'Geometry':<12} {'Mean':>10} {'Median':>10} {'Max':>10}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['geometry']:<12} "
            f"{r['mean_error_deg']:>10.3f} "
            f"{r['median_error_deg']:>10.3f} "
            f"{r['max_error_deg']:>10.3f}"
        )

    print("\nRepeated trials")
    print("-" * 60)

    for name, pixels in GEOMETRIES.items():
        all_errors = [evaluate_geometry(name, pixels)["mean_error_deg"] for _ in range(20)]
        print(f"{name:<12} {np.mean(all_errors):.3f} ± {np.std(all_errors, ddof=1):.3f}")


if __name__ == "__main__":
    main()
