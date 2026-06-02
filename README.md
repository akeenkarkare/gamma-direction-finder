# gamma-direction-finder

A simulation study of how a small array of gamma-ray detector pixels can recover the **direction** of an incoming radiation source from the relative counts each pixel sees.

The idea: arrange a handful of detector pixels into a Tetris-like shape, place lead shielding between adjacent pixels, and shoot gamma rays at the array from a known angle. Because the lead self-shields some pixels more than others depending on the source direction, the *pattern* of normalized counts across pixels encodes the source angle. A small neural network is then trained to invert that pattern back into an angle, and different pixel geometries are benchmarked by how accurately they let the model recover direction.

## How it works

1. **Forward model** — given a source angle θ and a pixel geometry, compute the counts each pixel records.
2. **Dataset** — sweep θ over the full circle and record the normalized count pattern at each angle.
3. **Regression** — train an `MLPRegressor` to map count patterns → angle. To avoid the 0°/360° discontinuity, the model predicts `(cos θ, sin θ)` and angles are recovered with `arctan2`.
4. **Benchmark** — compare geometries (square, S, J, T, I, and random shapes) by mean angular recovery error.

Two forward models are implemented:

- **Analytic toy model** (`main.py`, `benchmark_geometries.py`) — each pixel has a "shielding direction" and an attenuation that grows with the angle mismatch via `I = I₀·exp(-μ·x)`. Fast and simple.
- **Ray-traced model** (`ray_trace_detector.py`) — explicitly builds 2D boxes for the detector pixels and the lead strips between adjacent pixels, samples many rays from the source to each pixel, and accumulates the lead path length through ray–box (slab) intersection. More physically faithful.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Original analytic toy detector; generates `toy_detector_data.npz` and plots the angular response of an S-shape. |
| `train_angle_model.py` | Trains the MLP on `toy_detector_data.npz` and reports mean angular error. |
| `benchmark_geometries.py` | Analytic benchmark of the standard Tetris geometries (square, S, J, T, I) over repeated trials. |
| `random_geometry_search.py` | Random search over 4/5/6-pixel geometries on a 5×5 grid, with a compactness penalty; finds the best shape per pixel count. Saves `best_error_vs_pixels.png`. |
| `ray_trace_detector.py` | Ray-traced forward model: detector + lead boxes, ray–box intersection, per-pixel counts. |
| `train_raytrace_model.py` | Trains the MLP on the ray-traced data and reports mean/median/max error plus worst predictions. |
| `raytrace_benchmark.py` | Ray-traced benchmark across geometries; saves `raytrace_error_vs_angle.png`. |
| `visualize_raytrace_angle.py` | Draws the detector, lead, source, and sampled rays for specific angles; saves `raytrace_angle_*.png`. |

## Setup

Requires Python 3.13 with `numpy`, `scikit-learn`, and `matplotlib`. The repo includes a [uv](https://github.com/astral-sh/uv) virtual environment in `.venv/`.

```bash
# using uv
uv venv
uv pip install numpy scikit-learn matplotlib

# or with pip
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scikit-learn matplotlib
```

## Usage

```bash
# 1. Generate the toy dataset and view the angular response
python main.py

# 2. Train a direction-finding model on the toy data
python train_angle_model.py

# 3. Benchmark the standard Tetris geometries (analytic model)
python benchmark_geometries.py

# 4. Search random geometries for the best direction-recovery shape
python random_geometry_search.py

# 5. Run the ray-traced pipeline
python train_raytrace_model.py     # train + evaluate
python raytrace_benchmark.py        # benchmark geometries
python visualize_raytrace_angle.py  # visualize rays at fixed angles
```

## Key parameters

Most physics knobs live at the top of `ray_trace_detector.py`:

- `I0` — incoming photon count
- `MU_LEAD` — lead attenuation coefficient
- `LEAD_THICKNESS` — width of the shielding strips between pixels
- `RAYS_PER_PIXEL` — Monte-Carlo rays sampled per pixel (accuracy vs. speed)
- `SOURCE_DISTANCE` — how far the source sits from the array
- `NOISE` — toggle Poisson counting noise
- `GEOMETRY` — the default pixel layout

## Notes

Generated artifacts (`*.png`, `*.npz`) are git-ignored, so plots and datasets are recreated when you run the scripts.
