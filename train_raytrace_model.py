import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from ray_trace_detector import detector_counts, GEOMETRY


def angular_error_deg(pred, true):
    err = np.abs(np.angle(np.exp(1j * (pred - true))))
    return np.degrees(err)



def generate_raytrace_dataset(n_angles=720, repeats=3):
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    X = []
    theta = []

    for angle in angles:
        for _ in range(repeats):
            counts = detector_counts(angle, GEOMETRY)
            counts = counts / counts.sum()

            X.append(counts)
            theta.append(angle)

    X = np.array(X)
    theta = np.array(theta)

    y = np.column_stack([np.cos(theta), np.sin(theta)])

    return X, y, theta


def main():
    np.random.seed(42)

    print("Generating ray-traced dataset...")
    X, y, theta = generate_raytrace_dataset(n_angles=720, repeats=3)

    X_train, X_test, y_train, y_test, theta_train, theta_test = train_test_split(
        X, y, theta, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        max_iter=3000,
        random_state=42,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    theta_pred = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)

    errors = angular_error_deg(theta_pred, theta_test)

    print("\nRay-traced angular recovery")
    print("-" * 40)
    print(f"Mean error:   {errors.mean():.3f}°")
    print(f"Median error: {np.median(errors):.3f}°")
    print(f"Max error:    {errors.max():.3f}°")

    worst = np.argsort(errors)[-10:][::-1]

    print("\nWorst predictions")
    print("-" * 60)
    for idx in worst:
        print(
            f"true={np.degrees(theta_test[idx]):7.2f}° | "
            f"pred={np.degrees(theta_pred[idx]):7.2f}° | "
            f"err={errors[idx]:7.2f}° | "
            f"counts={X_test[idx]}"
        )


if __name__ == "__main__":
    main()