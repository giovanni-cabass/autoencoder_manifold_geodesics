"""
Task 38 Phase 3 — dim-3 truncated midpoint Note-02 core (copy/adapt).

Adapted from task34_note02_core.py for the full 3D chart (e0,e1,e2).
Production path is this module — NOT a thin wrap of energy_geodesic_scripts.

Frozen algebra (design §6.2 / §0.7):

  ξ = y - x,  x_c = (x+y)/2,  L_c = sqrt(g(x_c)(ξ,ξ))
  Q4_mid = (1/12) g_μν (∂_τ Γ^ν_{ρσ} + Γ^ν_{λτ} Γ^λ_{ρσ}) ξ^μ ξ^τ ξ^ρ ξ^σ
  D_note02 = L_c + Q4_mid / (2 L_c)     (+O(ξ^5) dropped)

Index convention (Phase-1 3D util, unsliced):
  Gamma[..., ρ, μ, ν] = Γ^ρ_{μν}
  dGamma[..., α, ρ, μ, ν] = ∂_α Γ^ρ_{μν}
  g shape (N, 3, 3) — never slice g_full.

Soft tiny-L_c policy (§0.7): if L_c ≤ ε_floor, flag/report — do NOT hard-raise.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from utils.g_sigma_3d import GSigma3D

EPS_FLOOR: np.float64 = np.float64(1.0e-12)
LATENT_DIM: int = 3

ArrayLike = Union[np.ndarray, torch.Tensor]

ORACLE_FORMULA: str = "D_note02 = L_c + Q4_mid/(2 L_c)"
D_NLO_FORMULA: str = "d_nlo = Q4_mid/(2 L_c)"
NLO_LO_FORMULA: str = "NLO/LO = d_nlo/L_c"
TRUNCATION: str = "O(xi^5) dropped; NOT exact d_g"
METRIC_UTIL_ID: str = "utils.g_sigma_3d.GSigma3D / load_g_sigma_3d_bundle"


def note02_pair_decomposition(
    x: ArrayLike,
    y: ArrayLike,
    g_bundle: GSigma3D,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Truncated Note-02 LO / NLO pieces with soft tiny-L_c (Task 38 §0.7 / §6.2).

    Parameters
    ----------
    x, y : (P, 3) or (3,) float64 chart coords
    g_bundle : GSigma3D (full 3×3 g / Γ / ∂Γ)

    Returns
    -------
    L_c, d_nlo, d_note02, q4_mid, tiny_lc_mask
        Each L_c / d_nlo / d_note02 / q4_mid is () or (P,) float64.
        tiny_lc_mask is bool () or (P,) — True where L_c ≤ ε_floor.
        On tiny pairs: d_nlo and d_note02 are NaN; q4_mid and L_c remain finite.
    """
    device: torch.device = g_bundle.device
    x_t: torch.Tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    y_t: torch.Tensor = torch.as_tensor(y, dtype=torch.float64, device=device)
    squeeze_out: bool = False
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
        y_t = y_t.unsqueeze(0)
        squeeze_out = True
    if x_t.ndim != 2 or x_t.shape[1] != LATENT_DIM:
        raise ValueError(
            f"Expected x shape (P, {LATENT_DIM}) or ({LATENT_DIM},), "
            f"got {tuple(x_t.shape)}"
        )
    if y_t.shape != x_t.shape:
        raise ValueError(
            f"x/y shape mismatch: {tuple(x_t.shape)} vs {tuple(y_t.shape)}"
        )

    xi: torch.Tensor = y_t - x_t
    x_c: torch.Tensor = 0.5 * (x_t + y_t)

    with torch.no_grad():
        g_mid, gamma_mid, dgamma_mid = g_bundle.g_gamma_dgamma(x_c)

        if g_mid.shape[-2:] != (LATENT_DIM, LATENT_DIM):
            raise RuntimeError(
                f"Expected g shape (..., {LATENT_DIM}, {LATENT_DIM}), "
                f"got {tuple(g_mid.shape)} — do not slice g_full"
            )
        if gamma_mid.shape[-3:] != (LATENT_DIM, LATENT_DIM, LATENT_DIM):
            raise RuntimeError(
                f"Expected Gamma shape (..., 3, 3, 3), got {tuple(gamma_mid.shape)}"
            )
        if dgamma_mid.shape[-4:] != (
            LATENT_DIM,
            LATENT_DIM,
            LATENT_DIM,
            LATENT_DIM,
        ):
            raise RuntimeError(
                f"Expected dGamma shape (..., 3, 3, 3, 3), "
                f"got {tuple(dgamma_mid.shape)}"
            )

        # Γ^ν_{λτ} Γ^λ_{ρσ} → [M, τ, ν, ρ, σ]  (contract λ)
        gg: torch.Tensor = torch.einsum(
            "Mnlt,Mlrs->Mtnrs", gamma_mid, gamma_mid
        )
        # ∂_τ Γ^ν_{ρσ} is dGamma[..., τ, ν, ρ, σ]
        bracket: torch.Tensor = dgamma_mid + gg
        q4_mid: torch.Tensor = (1.0 / 12.0) * torch.einsum(
            "Mmn,Mtnrs,Mm,Mt,Mr,Ms->M",
            g_mid,
            bracket,
            xi,
            xi,
            xi,
            xi,
        )
        l_c_sq: torch.Tensor = torch.einsum("Mmn,Mm,Mn->M", g_mid, xi, xi)
        l_c: torch.Tensor = torch.sqrt(torch.clamp(l_c_sq, min=0.0))

    if not bool(torch.isfinite(q4_mid).all()) or not bool(torch.isfinite(l_c).all()):
        raise RuntimeError("Non-finite Q4_mid or L_c in note02_pair_decomposition")

    tiny_lc_mask: torch.Tensor = l_c <= float(EPS_FLOOR)
    d_nlo: torch.Tensor = torch.full_like(l_c, float("nan"))
    d_note02: torch.Tensor = torch.full_like(l_c, float("nan"))
    valid: torch.Tensor = ~tiny_lc_mask
    if bool(valid.any()):
        d_nlo_valid: torch.Tensor = q4_mid[valid] / (2.0 * l_c[valid])
        d_nlo[valid] = d_nlo_valid
        d_note02[valid] = l_c[valid] + d_nlo_valid
        if not bool(torch.isfinite(d_note02[valid]).all()):
            raise RuntimeError("Non-finite D_note02 on valid (non-tiny) pairs")

    if squeeze_out:
        return (
            l_c.reshape(()),
            d_nlo.reshape(()),
            d_note02.reshape(()),
            q4_mid.reshape(()),
            tiny_lc_mask.reshape(()),
        )
    return l_c, d_nlo, d_note02, q4_mid, tiny_lc_mask


def note02_pair_distance(
    x: ArrayLike,
    y: ArrayLike,
    g_bundle: GSigma3D,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Truncated Note-02 midpoint oracle D_note02 with soft tiny-L_c.

    Returns
    -------
    D_note02 : () or (P,) float64 — NaN on tiny-L_c pairs
    tiny_lc_mask : () or (P,) bool
    """
    _l_c, _d_nlo, d_note02, _q4, tiny_lc_mask = note02_pair_decomposition(
        x, y, g_bundle
    )
    return d_note02, tiny_lc_mask


def note02_frac(
    l_c: np.ndarray,
    q4_mid: np.ndarray,
    d_note02: np.ndarray,
    tiny_lc_mask: np.ndarray,
) -> np.ndarray:
    """
    frac = (D_note02 - L_c) / D_note02 when possible; soft fallback on tiny L_c.

    Never raises. Tiny / invalid → algebraically equivalent
    Q4 / (2 L_c^2 + Q4) when the denominator is usable; else NaN.
    """
    l_c_a: np.ndarray = np.asarray(l_c, dtype=np.float64).reshape(-1)
    q4_a: np.ndarray = np.asarray(q4_mid, dtype=np.float64).reshape(-1)
    d_a: np.ndarray = np.asarray(d_note02, dtype=np.float64).reshape(-1)
    tiny: np.ndarray = np.asarray(tiny_lc_mask, dtype=bool).reshape(-1)
    p: int = int(l_c_a.shape[0])
    frac: np.ndarray = np.full(p, np.nan, dtype=np.float64)

    valid: np.ndarray = (~tiny) & np.isfinite(d_a) & (np.abs(d_a) > 0.0)
    if bool(valid.any()):
        frac[valid] = (d_a[valid] - l_c_a[valid]) / d_a[valid]

    need_alt: np.ndarray = ~valid
    if bool(need_alt.any()):
        denom_alt: np.ndarray = (
            2.0 * l_c_a[need_alt] * l_c_a[need_alt] + q4_a[need_alt]
        )
        ok_alt: np.ndarray = np.isfinite(denom_alt) & (np.abs(denom_alt) > 0.0)
        idxs: np.ndarray = np.flatnonzero(need_alt)
        if bool(ok_alt.any()):
            frac[idxs[ok_alt]] = q4_a[need_alt][ok_alt] / denom_alt[ok_alt]

    return frac


def nlo_over_lo(d_nlo: np.ndarray, l_c: np.ndarray) -> np.ndarray:
    """NLO/LO = d_nlo / L_c; NaN where either input is non-finite or L_c == 0."""
    d_nlo_a: np.ndarray = np.asarray(d_nlo, dtype=np.float64)
    l_c_a: np.ndarray = np.asarray(l_c, dtype=np.float64)
    out: np.ndarray = np.full(d_nlo_a.shape, np.nan, dtype=np.float64)
    ok: np.ndarray = (
        np.isfinite(d_nlo_a) & np.isfinite(l_c_a) & (np.abs(l_c_a) > 0.0)
    )
    if bool(ok.any()):
        out[ok] = d_nlo_a[ok] / l_c_a[ok]
    return np.ascontiguousarray(out, dtype=np.float64)


def rel_err_vs_note02(d: np.ndarray, d_note02: np.ndarray) -> np.ndarray:
    """
    Locked §0.7: |D - D_note02| / max(D_note02, ε_floor), elementwise.
    NaN where either input is non-finite.
    """
    d_a: np.ndarray = np.asarray(d, dtype=np.float64)
    b: np.ndarray = np.asarray(d_note02, dtype=np.float64)
    out: np.ndarray = np.full(d_a.shape, np.nan, dtype=np.float64)
    ok: np.ndarray = np.isfinite(d_a) & np.isfinite(b)
    if bool(ok.any()):
        denom: np.ndarray = np.maximum(b[ok], float(EPS_FLOOR))
        out[ok] = np.abs(d_a[ok] - b[ok]) / denom
    return out
