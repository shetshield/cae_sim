from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch

from pi_config import BeamConfig, lame_parameters, patch_area


# -----------------------------
# Scaling / Features
# -----------------------------

@dataclass(frozen=True)
class ScaleConfig:
    # model-space target scaling:
    #   u_model = u_phys * thickness_scale(t) * scale_u
    scale_u: float = 10000.0
    use_thickness_scaling: bool = True


def thickness_mm(t_mm: torch.Tensor, cfg: BeamConfig) -> torch.Tensor:
    return cfg.thickness0_mm + t_mm


def thickness_scale(t_mm: torch.Tensor, cfg: BeamConfig) -> torch.Tensor:
    """(h/h_ref)^3 scaling used to reduce thickness-driven amplitude variation."""
    h = thickness_mm(t_mm, cfg).clamp_min(1e-6)
    return (h / cfg.thickness_ref_mm) ** 3


def make_branch_features(t_mm: torch.Tensor, cfg: BeamConfig, t_range: Tuple[float, float]) -> torch.Tensor:
    """Branch features to improve extrapolation.

    Returns (B,4):
      [t_norm, log(h/h_ref), 1/(h/h_ref), 1/(h/h_ref)^3]
    """
    t_min, t_max = t_range
    t_norm = 2.0 * (t_mm - t_min) / max(t_max - t_min, 1e-6) - 1.0

    h = thickness_mm(t_mm, cfg).clamp_min(1e-6)
    h_ref = torch.tensor(cfg.thickness_ref_mm, device=t_mm.device, dtype=t_mm.dtype)
    h_ratio = (h / h_ref).clamp_min(1e-6)

    f1 = t_norm
    f2 = torch.log(h_ratio)
    f3 = 1.0 / h_ratio
    f4 = 1.0 / (h_ratio ** 3)

    return torch.cat([f1, f2, f3, f4], dim=1)


def apply_hard_bc_gate(u: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Hard Dirichlet BC at fixed end (x_len_hat=-1).

    u = g(x) * u_free, g = (x_len_hat + 1)/2 ∈ [0,1]
    """
    g = 0.5 * (x_hat[..., 0:1] + 1.0)
    return u * g


def model_to_physical_u(
    u_model: torch.Tensor,
    t_mm: torch.Tensor,
    cfg: BeamConfig,
    scfg: ScaleConfig,
) -> torch.Tensor:
    """Convert model-space displacement to physical displacement (meters)."""
    u = u_model / scfg.scale_u
    if scfg.use_thickness_scaling:
        ts = thickness_scale(t_mm, cfg)  # (B,1)
        u = u / ts[:, None, :]
    return u


def physical_to_model_u(
    u_phys: np.ndarray,
    t_mm: float,
    cfg: BeamConfig,
    scfg: ScaleConfig,
) -> np.ndarray:
    """Convert physical displacement (m) to model-space target."""
    u = u_phys.copy()
    if scfg.use_thickness_scaling:
        ts = ((cfg.thickness0_mm + t_mm) / cfg.thickness_ref_mm) ** 3
        u = u * ts
    return u * scfg.scale_u


# -----------------------------
# Collocation sampling in normalized canonical box [-1,1]^3
# Canonical axes: [Length, Width, Thickness]
# -----------------------------

def sample_interior_points(
    B: int,
    N: int,
    device: torch.device,
    bias_fixed: float = 0.50,
    beta_a: float = 0.5,
    beta_b: float = 3.0,
) -> torch.Tensor:
    """Sample interior points. 일부는 fixed end 근처로 bias."""
    N_bias = int(N * bias_fixed)
    N_uni = N - N_bias

    x_uni = -1.0 + 2.0 * torch.rand((B, N_uni, 3), device=device)

    if N_bias > 0:
        dist = torch.distributions.Beta(beta_a, beta_b)
        s = dist.sample((B, N_bias, 1)).to(device)
        x_len = -1.0 + 2.0 * s
        x_rest = -1.0 + 2.0 * torch.rand((B, N_bias, 2), device=device)
        x_bias = torch.cat([x_len, x_rest], dim=2)
        x_hat = torch.cat([x_uni, x_bias], dim=1)
    else:
        x_hat = x_uni

    return x_hat


def sample_fixed_face_points(B: int, N: int, device: torch.device) -> torch.Tensor:
    x = torch.empty((B, N, 3), device=device)
    x[..., 0] = -1.0
    x[..., 1:] = -1.0 + 2.0 * torch.rand((B, N, 2), device=device)
    return x


def sample_load_patch_points(
    B: int,
    N: int,
    device: torch.device,
    spans_m: torch.Tensor,
    cfg: BeamConfig,
) -> torch.Tensor:
    """Top surface(thickness=+1) 원형 패치 상의 포인트를 physical disc에서 샘플링 후 x_hat로 변환."""
    x_center_hat = -1.0 + 2.0 * cfg.load_ratio

    u = torch.rand((B, N, 1), device=device)
    r = cfg.load_radius_m * torch.sqrt(u)
    theta = 2.0 * math.pi * torch.rand((B, N, 1), device=device)
    dx = r * torch.cos(theta)
    dy = r * torch.sin(theta)

    L = spans_m[:, 0].view(B, 1, 1).clamp_min(1e-12)
    W = spans_m[:, 1].view(B, 1, 1).clamp_min(1e-12)

    dx_hat = 2.0 * dx / L
    dy_hat = 2.0 * dy / W

    x_hat = torch.empty((B, N, 3), device=device)
    x_hat[..., 0] = x_center_hat + dx_hat.squeeze(-1)
    x_hat[..., 1] = dy_hat.squeeze(-1)
    x_hat[..., 2] = 1.0
    return torch.clamp(x_hat, -1.0, 1.0)


# -----------------------------
# Physics: strain/stress/traction/energy
# -----------------------------

def jacobian_phys(u_phys: torch.Tensor, x_hat: torch.Tensor, spans_m: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian ∂u/∂x_phys using autograd + chain rule.

    u_phys: (B,N,3)
    x_hat:  (B,N,3) requires_grad=True
    spans_m: (B,3) physical spans (L,W,T)
    Returns: J (B,N,3,3)  J[...,i,j]=∂u_i/∂x_phys_j
    """
    assert x_hat.requires_grad, "x_hat must require grad"
    B, N, _ = x_hat.shape
    scale = (2.0 / spans_m.clamp_min(1e-12)).view(B, 1, 3)  # (B,1,3)

    rows = []
    for i in range(3):
        grad_i_hat = torch.autograd.grad(
            u_phys[..., i].sum(),
            x_hat,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B,N,3)
        rows.append(grad_i_hat * scale)
    J = torch.stack(rows, dim=2)  # (B,N,3,3)
    return J


def strain_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    return 0.5 * (J + J.transpose(-1, -2))


def stress_from_strain(eps: torch.Tensor, cfg: BeamConfig) -> torch.Tensor:
    lam, mu = lame_parameters(cfg.E, cfg.nu)
    I = torch.eye(3, device=eps.device, dtype=eps.dtype).view(1, 1, 3, 3)
    tr = eps[..., 0, 0] + eps[..., 1, 1] + eps[..., 2, 2]
    return 2.0 * mu * eps + lam * tr[..., None, None] * I


def traction_loss_from_model(
    model,
    bfeat: torch.Tensor,
    t_mm: torch.Tensor,
    x_hat: torch.Tensor,
    spans_m: torch.Tensor,
    cfg: BeamConfig,
    scfg: ScaleConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Traction loss σn=t on load patch (dimensionless). Uses 1st derivatives only."""
    x_hat_req = x_hat.detach().clone().requires_grad_(True)

    u_model = model(bfeat, x_hat_req)
    u_model = apply_hard_bc_gate(u_model, x_hat_req)
    u_phys = model_to_physical_u(u_model, t_mm, cfg, scfg)

    J = jacobian_phys(u_phys, x_hat_req, spans_m)
    eps = strain_from_jacobian(J)
    sigma = stress_from_strain(eps, cfg)

    sigma_n = sigma[..., :, 2]  # (B,N,3), n=e_thickness

    t_z = cfg.force_total_N / patch_area(cfg)
    t_vec = torch.zeros_like(sigma_n)
    t_vec[..., 2] = t_z

    denom = abs(t_z) + 1e-12
    res = (sigma_n - t_vec) / denom
    loss = torch.mean(res ** 2)

    return loss, {"trac_mse": float(loss.detach().cpu())}


def dirichlet_monitor_loss_from_model(
    model,
    bfeat: torch.Tensor,
    t_mm: torch.Tensor,
    x_hat: torch.Tensor,
    cfg: BeamConfig,
    scfg: ScaleConfig,
) -> torch.Tensor:
    """Monitor-only: should be ~0 due to hard BC."""
    u_model = apply_hard_bc_gate(model(bfeat, x_hat), x_hat)
    u_phys = model_to_physical_u(u_model, t_mm, cfg, scfg)
    return torch.mean(u_phys ** 2)


def energy_balance_loss_from_model(
    model,
    bfeat: torch.Tensor,
    t_mm: torch.Tensor,
    x_hat_int: torch.Tensor,
    x_hat_tr: torch.Tensor,
    spans_m: torch.Tensor,
    cfg: BeamConfig,
    scfg: ScaleConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Energy balance loss (Clapeyron theorem): 2U - W = 0.

    r_nd = (2U - W) / (|2U| + |W| + eps)
    L = mean(r_nd^2)
    """
    B = t_mm.shape[0]

    # interior: U
    x_int_req = x_hat_int.detach().clone().requires_grad_(True)
    u_int_model = apply_hard_bc_gate(model(bfeat, x_int_req), x_int_req)
    u_int_phys = model_to_physical_u(u_int_model, t_mm, cfg, scfg)

    J = jacobian_phys(u_int_phys, x_int_req, spans_m)
    eps = strain_from_jacobian(J)

    lam, mu = lame_parameters(cfg.E, cfg.nu)
    tr = eps[..., 0, 0] + eps[..., 1, 1] + eps[..., 2, 2]
    eps_sq = torch.sum(eps ** 2, dim=(-1, -2))
    psi = mu * eps_sq + 0.5 * lam * (tr ** 2)  # Pa

    V = torch.prod(spans_m.clamp_min(1e-12), dim=1)  # (B,)
    U = torch.mean(psi, dim=1) * V  # J

    # traction: W
    u_tr_model = apply_hard_bc_gate(model(bfeat, x_hat_tr), x_hat_tr)
    u_tr_phys = model_to_physical_u(u_tr_model, t_mm, cfg, scfg)

    t_z = cfg.force_total_N / patch_area(cfg)
    A = patch_area(cfg)
    W = torch.mean(t_z * u_tr_phys[..., 2], dim=1) * A  # J

    r = 2.0 * U - W
    denom = torch.abs(2.0 * U) + torch.abs(W) + 1e-12
    r_nd = r / denom
    loss = torch.mean(r_nd ** 2)

    stats = {
        "U_J": float(torch.mean(U).detach().cpu()),
        "W_J": float(torch.mean(W).detach().cpu()),
        "en_r2": float(loss.detach().cpu()),
    }
    return loss, stats


def l2_reg_loss(u_model: torch.Tensor) -> torch.Tensor:
    return torch.mean(u_model ** 2)


# -----------------------------
# Helper: spans fallback
# -----------------------------

def spans_from_t_fallback(t_mm: torch.Tensor, cfg: BeamConfig) -> torch.Tensor:
    T = (cfg.thickness0_mm + t_mm.squeeze(1)) / 1000.0
    L = torch.full_like(T, cfg.length_m)
    W = torch.full_like(T, cfg.width_m)
    return torch.stack([L, W, T], dim=1)
