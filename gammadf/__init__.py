"""gammadf: simulating gamma-ray source direction finding with detector arrays."""

from .common import (
    GEOMETRIES,
    angular_error_deg,
    angles_to_sincos,
    sincos_to_angle,
    train_and_evaluate,
)

__all__ = [
    "GEOMETRIES",
    "angular_error_deg",
    "angles_to_sincos",
    "sincos_to_angle",
    "train_and_evaluate",
]
