"""Benchmark the standard Tetris geometries with the ray-traced forward model."""

import _bootstrap  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from gammadf import GEOMETRIES, train_and_evaluate
from gammadf.raytrace import generate_dataset


def evaluate_raytrace(name, pixels):
    print(f"Running {name}...")
    X, theta = generate_dataset(pixels)
    result = train_and_evaluate(X, theta, hidden_layer_sizes=(128, 128))
    result["geometry"] = name
    return result


def main():
    np.random.seed(42)

    results = sorted(
        (evaluate_raytrace(name, pixels) for name, pixels in GEOMETRIES.items()),
        key=lambda r: r["mean_error_deg"],
    )

    print("\nRay-traced geometry benchmark")
    print("-" * 65)
    print(f"{'Geometry':<12} {'Mean':>10} {'Median':>10} {'Max':>10}")
    print("-" * 65)

    for r in results:
        print(
            f"{r['geometry']:<12} "
            f"{r['mean_error_deg']:>10.3f} "
            f"{r['median_error_deg']:>10.3f} "
            f"{r['max_error_deg']:>10.3f}"
        )

    plt.figure(figsize=(9, 5))

    for r in results:
        order = np.argsort(r["theta_test"])
        plt.plot(np.degrees(r["theta_test"][order]), r["errors"][order], label=r["geometry"])

    plt.xlabel("True source angle (degrees)")
    plt.ylabel("Angular error (degrees)")
    plt.title("Ray-Traced Error vs Source Angle")
    plt.legend()
    plt.tight_layout()
    plt.savefig("raytrace_error_vs_angle.png", dpi=200)
    plt.show()

    print("\nSaved plot: raytrace_error_vs_angle.png")


if __name__ == "__main__":
    main()
