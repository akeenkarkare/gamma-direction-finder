"""Train an angle-recovery model on the toy dataset (``toy_detector_data.npz``)."""

import _bootstrap  # noqa: F401

import numpy as np

from gammadf import train_and_evaluate


def main():
    data = np.load("toy_detector_data.npz")
    result = train_and_evaluate(data["counts"], data["angles"])
    print("Mean angular error:", result["mean_error_deg"], "degrees")


if __name__ == "__main__":
    main()
