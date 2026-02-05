from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BeamConfig:
    # Geometry (meters)
    length_m: float = 0.050   # 50 mm
    width_m: float = 0.010    # 10 mm
    thickness0_m: float = 0.005  # 5 mm baseline thickness (t=0)

    # Material
    E: float = 193e9
    nu: float = 0.29

    # Load
    force_total_N: float = -50.0
    load_radius_m: float = 0.002  # 2 mm
    load_ratio: float = 0.90      # 90% along length from fixed end

    # Thickness parameter (mm offset)
    thickness0_mm: float = 5.0
    thickness_ref_mm: float = 5.0  # for scaling


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def patch_area(cfg: BeamConfig) -> float:
    return math.pi * (cfg.load_radius_m ** 2)
