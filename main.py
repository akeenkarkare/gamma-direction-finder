import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Toy detector model
# -----------------------

I0 = 10000          # incoming photons
mu_lead = 1.2      # fake attenuation coefficient, arbitrary units
noise = True

# 4 detector pixels in an S-shape-ish layout
pixels = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [2, 1]
])

# Give each pixel a slightly different "shielding direction"
shield_dirs = np.array([
    0,
    np.pi/3,
    2*np.pi/3,
    np.pi
])

def detector_counts(theta):
    """
    theta = source angle in radians.
    returns 4 detector counts.
    """

    counts = []

    for sd in shield_dirs:
        # angle mismatch between incoming radiation and this pixel's shielding direction
        angle_diff = np.abs(np.angle(np.exp(1j * (theta - sd))))

        # toy path length through lead
        # bigger angle mismatch -> more shielding
        x = 0.2 + 1.5 * (1 - np.cos(angle_diff))

        # attenuation law
        expected = I0 * np.exp(-mu_lead * x)

        # optional Poisson counting noise
        if noise:
            measured = np.random.poisson(expected)
        else:
            measured = expected

        counts.append(measured)

    return np.array(counts)


# -----------------------
# Generate dataset
# -----------------------

angles = np.linspace(0, 2*np.pi, 360)
data = np.array([detector_counts(theta) for theta in angles])

# Normalize counts so patterns are easier to compare
data_norm = data / data.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 5))

for i in range(4):
    plt.plot(np.degrees(angles), data_norm[:, i], label=f"Pixel {i+1}")

plt.xlabel("Source angle (degrees)")
plt.ylabel("Fraction of total counts")
plt.title("Toy Tetris Detector Angular Response")
plt.legend()
plt.show()

np.savez("toy_detector_data.npz", counts=data_norm, angles=angles)
print("saved toy_detector_data.npz")