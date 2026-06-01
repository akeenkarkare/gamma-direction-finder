import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

I0 = 10000
mu_lead = 1.2
NOISE = True

GEOMETRIES = {
    "square": np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]),
    "S": np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 1],
    ]),
    "J": np.array([
        [0, 1],
        [0, 0],
        [1, 0],
        [2, 0],
    ]),
    "T": np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [1, 1],
    ]),
    "I": np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
    ]),
}

def angle_wrap_error(pred, true):
    err = np.abs(np.angle(np.exp(1j * (pred - true))))
    return np.degrees(err)

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

        expected = I0 * np.exp(-mu_lead * x)

        measured = np.random.poisson(expected) if NOISE else expected
        counts.append(measured)

    return np.array(counts)

def generate_dataset(pixels, n_angles=720):
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    X = np.array([detector_counts(theta, pixels) for theta in angles])

    # normalize so the model learns relative detector pattern, not total count
    X = X / X.sum(axis=1, keepdims=True)

    # predict sin/cos to avoid 0/360 angle discontinuity
    y = np.column_stack([np.cos(angles), np.sin(angles)])

    return X, y, angles

def train_and_evaluate(name, pixels):
    X, y, angles = generate_dataset(pixels)

    X_train, X_test, y_train, y_test, theta_train, theta_test = train_test_split(
        X, y, angles, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        max_iter=3000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    theta_pred = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)

    errors = angle_wrap_error(theta_pred, theta_test)

    return {
        "geometry": name,
        "mean_error_deg": errors.mean(),
        "median_error_deg": np.median(errors),
        "max_error_deg": errors.max(),
    }

def main():
    results = []

    for name, pixels in GEOMETRIES.items():
        result = train_and_evaluate(name, pixels)
        results.append(result)

    results = sorted(results, key=lambda r: r["mean_error_deg"])

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

        all_errors = []

        for trial in range(20):
            result = train_and_evaluate(name, pixels)
            all_errors.append(result["mean_error_deg"])

        print(
            f"{name:<12} "
            f"{np.mean(all_errors):.3f} ± "
            f"{np.std(all_errors, ddof=1):.3f}"
        )

if __name__ == "__main__":
    main()