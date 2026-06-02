import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from ray_trace_detector import detector_counts

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


def angular_error_deg(pred, true):
    err = np.abs(np.angle(np.exp(1j * (pred - true))))
    return np.degrees(err)


def generate_dataset(pixels, n_angles=720, repeats=3):
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    X = []
    theta = []

    for angle in angles:
        for _ in range(repeats):
            counts = detector_counts(angle, pixels)
            counts = counts / counts.sum()

            X.append(counts)
            theta.append(angle)

    X = np.array(X)
    theta = np.array(theta)

    y = np.column_stack([np.cos(theta), np.sin(theta)])
    return X, y, theta


def train_and_evaluate_raytrace(name, pixels):
    print(f"Running {name}...")

    X, y, theta = generate_dataset(pixels)

    X_train, X_test, y_train, y_test, theta_train, theta_test = train_test_split(
        X, y, theta, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        max_iter=3000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    theta_pred = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)

    errors = angular_error_deg(theta_pred, theta_test)

    return {
        "geometry": name,
        "mean_error_deg": errors.mean(),
        "median_error_deg": np.median(errors),
        "max_error_deg": errors.max(),
        "theta_test": theta_test,
        "errors": errors,
    }


def main():
    np.random.seed(42)

    results = []

    for name, pixels in GEOMETRIES.items():
        results.append(train_and_evaluate_raytrace(name, pixels))

    results = sorted(results, key=lambda r: r["mean_error_deg"])

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
        theta_deg = np.degrees(r["theta_test"][order])
        err = r["errors"][order]

        plt.plot(theta_deg, err, label=r["geometry"])

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