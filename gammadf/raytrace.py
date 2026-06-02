import numpy as np

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
    [0, 1],
    [1, 1],
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


def generate_dataset(pixels, n_angles=720, repeats=3):
    """Build a (counts, angle) dataset by ray tracing the source around the array."""
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    X = []
    theta = []

    for angle in angles:
        for _ in range(repeats):
            counts = detector_counts(angle, pixels)
            X.append(counts / counts.sum())
            theta.append(angle)

    return np.array(X), np.array(theta)