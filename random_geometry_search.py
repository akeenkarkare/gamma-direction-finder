import numpy as np
import matplotlib.pyplot as plt

from benchmark_geometries import train_and_evaluate, GEOMETRIES

GRID_SIZE = 4
N_PIXELS = 4
N_RANDOM = 100
RANDOM_SEED = 7


def random_geometry(grid_size=GRID_SIZE, n_pixels=N_PIXELS):
    all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    chosen = np.random.choice(len(all_positions), size=n_pixels, replace=False)
    return np.array([all_positions[i] for i in chosen])


def geometry_to_string(pixels):
    pixels = set(map(tuple, pixels))
    max_x = max(x for x, y in pixels)
    max_y = max(y for x, y in pixels)

    rows = []
    for y in range(max_y, -1, -1):
        row = ""
        for x in range(max_x + 1):
            row += "□ " if (x, y) in pixels else ". "
        rows.append(row.rstrip())

    return "\n".join(rows)


def main():
    np.random.seed(RANDOM_SEED)

    results = []

    print("Evaluating known geometries...")
    for name, pixels in GEOMETRIES.items():
        r = train_and_evaluate(name, pixels)
        results.append({
            "name": name,
            "pixels": pixels,
            "mean_error_deg": r["mean_error_deg"],
            "kind": "known",
        })

    print("Evaluating random geometries...")
    for i in range(N_RANDOM):
        pixels = random_geometry()
        r = train_and_evaluate(f"random_{i}", pixels)

        results.append({
            "name": f"random_{i}",
            "pixels": pixels,
            "mean_error_deg": r["mean_error_deg"],
            "kind": "random",
        })

    results = sorted(results, key=lambda x: x["mean_error_deg"])

    print("\nTop 10 geometries")
    print("-" * 50)

    for rank, r in enumerate(results[:10], start=1):
        print(f"\n#{rank}: {r['name']} | {r['mean_error_deg']:.3f}° | {r['kind']}")
        print(geometry_to_string(r["pixels"]))

    known_ranks = {
        r["name"]: idx + 1
        for idx, r in enumerate(results)
        if r["kind"] == "known"
    }

    print("\nKnown geometry ranks")
    print("-" * 50)
    for name in GEOMETRIES:
        print(f"{name:<10}: rank {known_ranks[name]} / {len(results)}")

    errors = [r["mean_error_deg"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker="o", markersize=3)
    plt.xlabel("Geometry rank")
    plt.ylabel("Mean angular error (degrees)")
    plt.title("Random 4-Pixel Geometry Search")
    plt.tight_layout()
    plt.savefig("random_geometry_search.png", dpi=200)
    plt.show()

    print("\nSaved plot: random_geometry_search.png")


if __name__ == "__main__":
    main()