"""Generate the original toy dataset and plot the detector angular response.

This is the earliest, simplest forward model: four pixels with hand-picked
shielding directions. It writes ``toy_detector_data.npz`` consumed by
``train_angle.py``. The geometry-aware analytic model lives in
``gammadf.analytic``.
"""

import _bootstrap  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

I0 = 10000          # incoming photons
MU_LEAD = 1.2       # attenuation coefficient, arbitrary units
NOISE = True

# Hand-picked shielding direction for each of the 4 pixels.
SHIELD_DIRS = np.array([0, np.pi / 3, 2 * np.pi / 3, np.pi])


def detector_counts(theta):
    """Return 4 detector counts for a source at angle ``theta`` (radians)."""
    counts = []
    for sd in SHIELD_DIRS:
        angle_diff = np.abs(np.angle(np.exp(1j * (theta - sd))))
        # Bigger angle mismatch -> longer toy path through lead -> more shielding.
        x = 0.2 + 1.5 * (1 - np.cos(angle_diff))
        expected = I0 * np.exp(-MU_LEAD * x)
        counts.append(np.random.poisson(expected) if NOISE else expected)
    return np.array(counts)


def main():
    angles = np.linspace(0, 2 * np.pi, 360)
    data = np.array([detector_counts(theta) for theta in angles])
    data_norm = data / data.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 5))
    for i in range(data_norm.shape[1]):
        plt.plot(np.degrees(angles), data_norm[:, i], label=f"Pixel {i + 1}")

    plt.xlabel("Source angle (degrees)")
    plt.ylabel("Fraction of total counts")
    plt.title("Toy Detector Angular Response")
    plt.legend()
    plt.show()

    np.savez("toy_detector_data.npz", counts=data_norm, angles=angles)
    print("saved toy_detector_data.npz")


if __name__ == "__main__":
    main()
