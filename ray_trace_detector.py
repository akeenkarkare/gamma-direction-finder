import numpy as np
import matplotlib.pyplot as plt

I0 = 10000
MU_LEAD = 4.0
LEAD_THICKNESS = 0.35
RAYS_PER_PIXEL = 500
NOISE = False
PIXEL_SIZE = 1.0
SOURCE_DISTANCE = 20.0

GEOMETRY = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [2, 1],
])  # S-shape


def normalize_pixels(pixels):
    return pixels - pixels.mean(axis=0)


def ray_box_intersection_length(ray_origin, ray_dir, box_min, box_max, max_t):
    inv_dir = 1.0 / np.where(np.abs(ray_dir) < 1e-12, 1e-12, ray_dir)

    t1 = (box_min - ray_origin) * inv_dir
    t2 = (box_max - ray_origin) * inv_dir

    t_enter = np.maximum.reduce(np.minimum(t1, t2))
    t_exit = np.minimum.reduce(np.maximum(t1, t2))

    t_enter = max(t_enter, 0.0)
    t_exit = min(t_exit, max_t)

    if t_exit <= t_enter:
        return 0.0

    return t_exit - t_enter


def make_boxes(pixels):
    pixels = normalize_pixels(pixels)

    detector_boxes = []
    lead_boxes = []
    added_leads = set()

    for x, y in pixels:
        box_min = np.array([x - PIXEL_SIZE / 2, y - PIXEL_SIZE / 2])
        box_max = np.array([x + PIXEL_SIZE / 2, y + PIXEL_SIZE / 2])
        detector_boxes.append((box_min, box_max))

    pixel_set = set(map(tuple, pixels))

    for x, y in pixels:
        current = np.array([x, y])

        neighbors = [
            np.array([x + 1, y]),
            np.array([x - 1, y]),
            np.array([x, y + 1]),
            np.array([x, y - 1]),
        ]

        for n in neighbors:
            if tuple(n) not in pixel_set:
                continue

            pair_key = tuple(sorted([tuple(current), tuple(n)]))
            if pair_key in added_leads:
                continue
            added_leads.add(pair_key)

            midpoint = (current + n) / 2
            dx, dy = n - current

            if abs(dx) > 0:
                box_min = np.array([
                    midpoint[0] - LEAD_THICKNESS / 2,
                    midpoint[1] - PIXEL_SIZE / 2,
                ])
                box_max = np.array([
                    midpoint[0] + LEAD_THICKNESS / 2,
                    midpoint[1] + PIXEL_SIZE / 2,
                ])
            else:
                box_min = np.array([
                    midpoint[0] - PIXEL_SIZE / 2,
                    midpoint[1] - LEAD_THICKNESS / 2,
                ])
                box_max = np.array([
                    midpoint[0] + PIXEL_SIZE / 2,
                    midpoint[1] + LEAD_THICKNESS / 2,
                ])

            lead_boxes.append((box_min, box_max))

    lead_boxes.append((
        np.array([-2.0, -2.0]),
        np.array([-1.5,  2.0]),
    ))
    return detector_boxes, lead_boxes


def sample_points_in_box(box_min, box_max, n):
    xs = np.random.uniform(box_min[0], box_max[0], n)
    ys = np.random.uniform(box_min[1], box_max[1], n)
    return np.column_stack([xs, ys])


def detector_counts(theta, pixels):
    detector_boxes, lead_boxes = make_boxes(pixels)

    source_position = SOURCE_DISTANCE * np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    counts = []

    for detector_min, detector_max in detector_boxes:
        sample_points = sample_points_in_box(
            detector_min,
            detector_max,
            RAYS_PER_PIXEL,
        )

        transmissions = []

        for point in sample_points:
            ray_origin = source_position
            ray_vec = point - ray_origin
            ray_length = np.linalg.norm(ray_vec)
            ray_dir = ray_vec / ray_length

            lead_length = 0.0

            for lead_min, lead_max in lead_boxes:
                lead_length += ray_box_intersection_length(
                    ray_origin,
                    ray_dir,
                    lead_min,
                    lead_max,
                    ray_length,
                )

            transmissions.append(np.exp(-MU_LEAD * lead_length))

        expected = I0 * np.mean(transmissions)

        measured = np.random.poisson(expected) if NOISE else expected
        counts.append(measured)

    return np.array(counts)


def main():
    np.random.seed(42)

    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    data = np.array([detector_counts(theta, GEOMETRY) for theta in angles])
    data_norm = data / data.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 5))

    for i in range(data_norm.shape[1]):
        plt.plot(np.degrees(angles), data_norm[:, i], label=f"Pixel {i + 1}")

    plt.xlabel("Source angle (degrees)")
    plt.ylabel("Fraction of total counts")
    plt.title("Ray-Traced Detector Angular Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ray_traced_response.png", dpi=200)
    plt.show()

    print("Saved ray_traced_response.png")


if __name__ == "__main__":
    main()