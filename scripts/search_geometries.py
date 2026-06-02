import _bootstrap  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from gammadf import GEOMETRIES
from benchmark import evaluate_geometry as evaluate_known_geometry

GRID_SIZE = 5
N_PIXELS_LIST = [4, 5, 6]
N_RANDOM = 100
RANDOM_SEED = 7
LAMBDA_COMPACTNESS = 0.15


def random_geometry(grid_size, n_pixels):
    all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    chosen = np.random.choice(len(all_positions), size=n_pixels, replace=False)
    return np.array([all_positions[i] for i in chosen])


def geometry_to_string(pixels):
    pixels_set = set(map(tuple, pixels))
    max_x = max(x for x, y in pixels_set)
    max_y = max(y for x, y in pixels_set)

    rows = []
    for y in range(max_y, -1, -1):
        row = ""
        for x in range(max_x + 1):
            row += "□ " if (x, y) in pixels_set else ". "
        rows.append(row.rstrip())

    return "\n".join(rows)


def bounding_box_area(pixels):
    xs = pixels[:, 0]
    ys = pixels[:, 1]

    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1

    return width * height


def compactness_score(mean_error, pixels, n_pixels):
    area = bounding_box_area(pixels)
    return mean_error + LAMBDA_COMPACTNESS * (area - n_pixels)


def evaluate_geometry(name, pixels, kind, n_pixels):
    r = evaluate_known_geometry(name, pixels)

    return {
        "name": name,
        "pixels": pixels,
        "n_pixels": n_pixels,
        "mean_error_deg": r["mean_error_deg"],
        "area": bounding_box_area(pixels),
        "score": compactness_score(r["mean_error_deg"], pixels, n_pixels),
        "kind": kind,
    }


def main():
    np.random.seed(RANDOM_SEED)

    all_results = []

    for n_pixels in N_PIXELS_LIST:
        print(f"\nEvaluating {n_pixels}-pixel geometries...")

        results = []

        if n_pixels == 4:
            print("Evaluating known 4-pixel geometries...")
            for name, pixels in GEOMETRIES.items():
                results.append(evaluate_geometry(name, pixels, "known", n_pixels))

        print(f"Evaluating {N_RANDOM} random {n_pixels}-pixel geometries...")
        for i in range(N_RANDOM):
            pixels = random_geometry(GRID_SIZE, n_pixels)
            results.append(
                evaluate_geometry(
                    f"random_{n_pixels}_{i}",
                    pixels,
                    "random",
                    n_pixels,
                )
            )

        results = sorted(results, key=lambda x: x["score"])
        all_results.extend(results)

        print(f"\nTop 10 geometries for {n_pixels} pixels")
        print("-" * 60)

        for rank, r in enumerate(results[:10], start=1):
            print(
                f"\n#{rank}: {r['name']} | "
                f"error={r['mean_error_deg']:.3f}° | "
                f"area={r['area']} | "
                f"score={r['score']:.3f} | "
                f"{r['kind']}"
            )
            print(geometry_to_string(r["pixels"]))

        if n_pixels == 4:
            known_ranks = {
                r["name"]: idx + 1
                for idx, r in enumerate(results)
                if r["kind"] == "known"
            }

            print("\nKnown geometry ranks")
            print("-" * 60)
            for name in GEOMETRIES:
                print(f"{name:<10}: rank {known_ranks[name]} / {len(results)}")

    print("\nBest geometry by pixel count")
    print("-" * 60)

    best_by_pixel_count = []

    for n_pixels in N_PIXELS_LIST:
        subset = [r for r in all_results if r["n_pixels"] == n_pixels]
        best = min(subset, key=lambda x: x["score"])
        best_by_pixel_count.append(best)

        print(
            f"{n_pixels} pixels: "
            f"{best['name']} | "
            f"error={best['mean_error_deg']:.3f}° | "
            f"area={best['area']} | "
            f"score={best['score']:.3f}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["n_pixels"] for r in best_by_pixel_count],
        [r["mean_error_deg"] for r in best_by_pixel_count],
        marker="o",
    )
    plt.xlabel("Number of detector pixels")
    plt.ylabel("Best mean angular error (degrees)")
    plt.title("Best Error vs Number of Detector Pixels")
    plt.tight_layout()
    plt.savefig("best_error_vs_pixels.png", dpi=200)
    plt.show()

    print("\nSaved plot: best_error_vs_pixels.png")


if __name__ == "__main__":
    main()