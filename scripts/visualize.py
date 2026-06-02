"""Draw the ray-traced detector, lead shielding, and sampled rays at fixed angles."""

import _bootstrap  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from gammadf.raytrace import (
    GEOMETRY,
    make_boxes,
    sample_points_in_box,
    ray_box_intersection_length,
    SOURCE_DISTANCE,
)


def draw_box(ax, box_min, box_max, label=None, alpha=0.35):
    width = box_max[0] - box_min[0]
    height = box_max[1] - box_min[1]

    rect = plt.Rectangle(
        box_min,
        width,
        height,
        fill=True,
        alpha=alpha,
        edgecolor="black",
        linewidth=1,
        label=label,
    )
    ax.add_patch(rect)


def visualize_angle(theta_deg, n_rays_per_pixel=12):
    theta = np.radians(theta_deg)

    detector_boxes, lead_boxes = make_boxes(GEOMETRY)

    source_position = SOURCE_DISTANCE * np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw lead boxes
    for i, (box_min, box_max) in enumerate(lead_boxes):
        draw_box(ax, box_min, box_max, label="lead" if i == 0 else None, alpha=0.75)

    # Draw detector boxes
    for i, (box_min, box_max) in enumerate(detector_boxes):
        draw_box(ax, box_min, box_max, label="detector" if i == 0 else None, alpha=0.25)

        center = (box_min + box_max) / 2
        ax.text(center[0], center[1], f"P{i+1}", ha="center", va="center", fontsize=12)

    # Draw source
    ax.scatter(source_position[0], source_position[1], marker="*", s=180, label="source")
    ax.text(source_position[0], source_position[1], f"  {theta_deg}°", fontsize=11)

    # Draw sampled rays
    total_lengths = []

    for pixel_idx, (det_min, det_max) in enumerate(detector_boxes):
        sample_points = sample_points_in_box(det_min, det_max, n_rays_per_pixel)

        pixel_lengths = []

        for point in sample_points:
            ray_vec = point - source_position
            ray_length = np.linalg.norm(ray_vec)
            ray_dir = ray_vec / ray_length

            lead_length = 0.0
            for lead_min, lead_max in lead_boxes:
                lead_length += ray_box_intersection_length(
                    source_position,
                    ray_dir,
                    lead_min,
                    lead_max,
                    ray_length,
                )

            pixel_lengths.append(lead_length)

            # Only draw a subset to avoid visual chaos
            ax.plot(
                [source_position[0], point[0]],
                [source_position[1], point[1]],
                linewidth=0.7,
                alpha=0.25,
            )

        total_lengths.append(np.mean(pixel_lengths))

    total_lengths = np.array(total_lengths)

    print(f"\nAngle {theta_deg}°")
    print("-" * 40)
    for i, length in enumerate(total_lengths):
        print(f"Pixel {i+1}: avg lead path = {length:.4f}")

    # Plot settings
    ax.set_aspect("equal")
    ax.set_title(f"S-shape ray tracing at θ = {theta_deg}°")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    all_x = [source_position[0]]
    all_y = [source_position[1]]

    for box_min, box_max in detector_boxes + lead_boxes:
        all_x.extend([box_min[0], box_max[0]])
        all_y.extend([box_min[1], box_max[1]])

    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    plt.tight_layout()
    filename = f"raytrace_angle_{int(theta_deg)}.png"
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"Saved {filename}")

    print("\nPixel attenuation summary")
    print("-" * 40)

    for i, length in enumerate(total_lengths):
        print(f"P{i+1}: {length:.4f}")

    print("\nDifferences")
    print("-" * 40)

    n = len(total_lengths)
    for i in range(n):
        for j in range(i + 1, n):
            print(
                f"P{i+1}-P{j+1}: "
                f"{abs(total_lengths[i] - total_lengths[j]):.4f}"
        )


if __name__ == "__main__":
    visualize_angle(155)
    visualize_angle(140)
    visualize_angle(245)