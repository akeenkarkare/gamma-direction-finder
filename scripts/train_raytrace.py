"""Train and evaluate an angle-recovery model on the ray-traced forward model."""

import _bootstrap  # noqa: F401

import numpy as np

from gammadf import train_and_evaluate
from gammadf.raytrace import GEOMETRY, generate_dataset


def main():
    np.random.seed(42)

    print("Generating ray-traced dataset...")
    X, theta = generate_dataset(GEOMETRY, n_angles=720, repeats=3)

    print("Training model...")
    result = train_and_evaluate(X, theta, hidden_layer_sizes=(128, 128))

    print("\nRay-traced angular recovery")
    print("-" * 40)
    print(f"Mean error:   {result['mean_error_deg']:.3f}°")
    print(f"Median error: {result['median_error_deg']:.3f}°")
    print(f"Max error:    {result['max_error_deg']:.3f}°")


if __name__ == "__main__":
    main()
