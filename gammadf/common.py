"""Shared geometry definitions and training utilities.

These are reused across the analytic and ray-traced pipelines so that the
list of Tetris geometries, the angular-error metric, and the MLP train/eval
routine live in exactly one place.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Standard 4-pixel Tetris geometries, on an integer grid.
GEOMETRIES = {
    "square": np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
    "S": np.array([[0, 0], [1, 0], [1, 1], [2, 1]]),
    "J": np.array([[0, 1], [0, 0], [1, 0], [2, 0]]),
    "T": np.array([[0, 0], [1, 0], [2, 0], [1, 1]]),
    "I": np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
}


def angular_error_deg(pred, true):
    """Wrap-aware angular error in degrees between predicted and true angles."""
    err = np.abs(np.angle(np.exp(1j * (pred - true))))
    return np.degrees(err)


def angles_to_sincos(angles):
    """Encode angles as (cos, sin) to avoid the 0/360 discontinuity."""
    return np.column_stack([np.cos(angles), np.sin(angles)])


def sincos_to_angle(pred):
    """Decode a (cos, sin) prediction back into an angle in [0, 2pi)."""
    return np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)


def train_and_evaluate(X, theta, hidden_layer_sizes=(64, 64), max_iter=3000):
    """Train an MLP to recover source angle from detector counts.

    X      : (n_samples, n_pixels) normalized count patterns
    theta  : (n_samples,) true source angles in radians

    Returns a dict with mean/median/max error (degrees) plus the held-out
    true angles and per-sample errors for plotting.
    """
    y = angles_to_sincos(theta)

    X_train, X_test, y_train, y_test, theta_train, theta_test = train_test_split(
        X, y, theta, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(X_train, y_train)

    theta_pred = sincos_to_angle(model.predict(X_test))
    errors = angular_error_deg(theta_pred, theta_test)

    return {
        "mean_error_deg": errors.mean(),
        "median_error_deg": np.median(errors),
        "max_error_deg": errors.max(),
        "theta_test": theta_test,
        "errors": errors,
    }
