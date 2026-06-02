# gamma-direction-finder

A simulation study of how a small array of gamma-ray detector pixels can recover the **direction** of an incoming radiation source from the relative counts each pixel sees.

The idea: arrange a handful of detector pixels into a Tetris-like shape, place lead shielding between adjacent pixels, and shoot gamma rays at the array from a known angle. Because the lead self-shields some pixels more than others depending on the source direction, the *pattern* of normalized counts across pixels encodes the source angle. A small neural network is then trained to invert that pattern back into an angle, and different pixel geometries are benchmarked by how accurately they let the model recover direction.

## How it works

1. **Forward model** — given a source angle θ and a pixel geometry, compute the counts each pixel records.
2. **Dataset** — sweep θ over the full circle and record the normalized count pattern at each angle.
3. **Regression** — train an `MLPRegressor` to map count patterns → angle. To avoid the 0°/360° discontinuity, the model predicts `(cos θ, sin θ)` and angles are recovered with `arctan2`.
4. **Benchmark** — compare geometries (square, S, J, T, I, and random shapes) by mean angular recovery error.

Two forward models are implemented:

- **Analytic model** (`gammadf/analytic.py`) — each pixel has a "shielding direction" and an attenuation that grows with the angle mismatch via `I = I₀·exp(-μ·x)`. Fast and simple.
- **Ray-traced model** (`gammadf/raytrace.py`) — explicitly builds 2D boxes for the detector pixels and the lead strips between adjacent pixels, samples many rays from the source to each pixel, and accumulates the lead path length through ray–box (slab) intersection. More physically faithful.

## Project layout

```
gammadf/                  # library: importable modules, no side effects
  common.py               # GEOMETRIES, angular-error metric, sin/cos encoding, MLP train/eval
  analytic.py             # analytic forward model + dataset generator
  raytrace.py             # ray-traced forward model + dataset generator
scripts/                  # runnable entrypoints (each has a __main__)
  generate_toy_data.py    # original toy detector; writes toy_detector_data.npz + response plot
  train_angle.py          # train on the toy dataset; report mean angular error
  train_raytrace.py       # train on the ray-traced data; report mean/median/max error
  benchmark.py            # analytic benchmark of the standard Tetris geometries
  raytrace_benchmark.py   # ray-traced benchmark; saves raytrace_error_vs_angle.png
  search_geometries.py    # random search over 4/5/6-pixel shapes; saves best_error_vs_pixels.png
  visualize.py            # draw detector, lead, source, and rays at fixed angles
```

The `scripts/` files put the repo root on `sys.path` (via `scripts/_bootstrap.py`) so they can `import gammadf` when run directly from anywhere.

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

Run the scripts from the repo root:

```bash
# 1. Generate the toy dataset and view the angular response
python scripts/generate_toy_data.py

# 2. Train a direction-finding model on the toy data
python scripts/train_angle.py

# 3. Benchmark the standard Tetris geometries (analytic model)
python scripts/benchmark.py

# 4. Search random geometries for the best direction-recovery shape
python scripts/search_geometries.py

# 5. Run the ray-traced pipeline
python scripts/train_raytrace.py        # train + evaluate
python scripts/raytrace_benchmark.py    # benchmark geometries
python scripts/visualize.py             # visualize rays at fixed angles
```

## Key parameters

Most physics knobs live at the top of `gammadf/raytrace.py`:

- `I0` — incoming photon count
- `MU_LEAD` — lead attenuation coefficient
- `LEAD_THICKNESS` — width of the shielding strips between pixels
- `RAYS_PER_PIXEL` — Monte-Carlo rays sampled per pixel (accuracy vs. speed)
- `SOURCE_DISTANCE` — how far the source sits from the array
- `NOISE` — toggle Poisson counting noise
- `GEOMETRY` — the default pixel layout

## Notes

Generated artifacts (`*.png`, `*.npz`) are git-ignored, so plots and datasets are recreated when you run the scripts.
