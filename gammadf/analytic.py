"""Analytic toy forward model.

Each pixel is assigned a "shielding direction" pointing away from the array
center; its attenuation grows with the angular mismatch between the source
direction and that shielding direction. Fast, closed-form alternative to the
ray-traced model in :mod:`gammadf.raytrace`.
"""

import numpy as np

I0 = 10000
MU_LEAD = 1.2
NOISE = True


def make_shield_dirs(pixels):
    center = pixels.mean(axis=0)
    rel = pixels - center
    return np.arctan2(rel[:, 1], rel[:, 0]) % (2 * np.pi)


def detector_counts(theta, pixels):
    shield_dirs = make_shield_dirs(pixels)
    counts = []

    for sd in shield_dirs:
        angle_diff = np.abs(np.angle(np.exp(1j * (theta - sd))))

        # Toy "effective shielding length"
        x = 0.2 + 1.5 * (1 - np.cos(angle_diff))

        expected = I0 * np.exp(-MU_LEAD * x)

        measured = np.random.poisson(expected) if NOISE else expected
        counts.append(measured)

    return np.array(counts)


def generate_dataset(pixels, n_angles=720):
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    X = np.array([detector_counts(theta, pixels) for theta in angles])

    # normalize so the model learns relative detector pattern, not total count
    X = X / X.sum(axis=1, keepdims=True)

    return X, angles
