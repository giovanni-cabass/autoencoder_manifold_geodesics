"""
Task 31 Phase 1 — batched differentiable g_Σ (+ Γ, ∂Γ) on fixed-ω_b Planck face.

Lineage (Phase 0 inventory `tasks/task_31.ad_inventory.md`):
  Extracted Torch AD helpers from
  `for_claude_audit_of_2D_situation.4/build_intrinsic_delaunay_mesh_fixed_omegab_48_96.py`
  (`_g_sigma_from_uv` → `_g_gamma_dgamma_at_uv`). No parallel AD stack; NumPy
  detach wrappers (`batched_eval*`) are intentionally NOT wrapped.

Pinned face / paths (parent §1.1):
  - Ambient G: geometry_outputs/induced_metric.npy
  - candidate_id = fixed_omegab_planck; FIXED_AXIS = 1 (ω_b)
  - e_Planck ≈ (-0.172, -0.032, -0.1755) → e_fixed = e_1 ≈ -0.032
  - Free chart uv = (e_0, e_2)

Index convention (already matches Phase-1 frozen shapes — no boundary permute):
  - g:      (N, 2, 2)
  - Gamma:  (N, 2, 2, 2) with Gamma[..., ρ, μ, ν] = Γ^ρ_{μν}
  - dGamma: (N, 2, 2, 2, 2) with dGamma[..., α, ρ, μ, ν] = ∂_α Γ^ρ_{μν}

Public returns are live float64 Torch tensors (no detach). Internal sub-batch
default is 256. After Phase-1 smoke PASS this module is frozen for Phases 2–3
(import-only).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

from utils.models import load_decoder
from utils.workspace_folders import geometry_outputs, main_folder

# Task 26 / §1.1 face constants (verbatim continuous Planck e-space)
E_PLANCK: np.ndarray = np.array(
    [-0.17199999999999982, -0.032000000000000806, -0.1755000000000001],
    dtype=np.float64,
)
CANDIDATE_ID: str = "fixed_omegab_planck"
FIXED_AXIS: int = 1
TANGENT_AXES: tuple[int, int] = (0, 2)
DEFAULT_SUB_BATCH: int = 256
INDUCED_METRIC_FILENAME: str = "induced_metric.npy"


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _verify_decoder_float64(decoder: nn.Module) -> None:
    all_float64: bool = all(
        p.dtype == torch.float64 for p in decoder.parameters()
    ) and all(b.dtype == torch.float64 for b in decoder.buffers())
    print("\n--- Verifying Model Dtype ---", flush=True)
    for name, param in decoder.named_parameters():
        status = "V" if param.dtype == torch.float64 else "X"
        print(f"  [{status}] param  '{name}': {param.dtype}", flush=True)
    for name, buf in decoder.named_buffers():
        status = "V" if buf.dtype == torch.float64 else "X"
        print(f"  [{status}] buffer '{name}': {buf.dtype}", flush=True)
    if not all_float64:
        raise RuntimeError(
            "MODEL DTYPE MISMATCH: not all parameters/buffers are float64."
        )
    print("SUCCESS: All parameters and buffers verified as float64.\n", flush=True)


class GSigmaFixedOmegab:
    """Loaded bundle: decoder + ambient G + e_fixed; batched g/Γ/∂Γ on free UV."""

    def __init__(
        self,
        decoder: nn.Module,
        g_metric: torch.Tensor,
        e_fixed: torch.Tensor,
        device: torch.device,
        *,
        sub_batch: int = DEFAULT_SUB_BATCH,
    ) -> None:
        self.decoder: nn.Module = decoder
        self.g_metric: torch.Tensor = g_metric
        self.e_fixed: torch.Tensor = e_fixed
        self.device: torch.device = device
        self.sub_batch: int = int(sub_batch)
        # Detach decoder weights from AD (differentiate w.r.t. free UV only).
        self.params: dict[str, torch.Tensor] = {
            k: v.detach() for k, v in decoder.named_parameters()
        }
        self.buffers: dict[str, torch.Tensor] = {
            k: v.detach() for k, v in decoder.named_buffers()
        }
        self.candidate_id: str = CANDIDATE_ID
        self.fixed_axis: int = FIXED_AXIS
        self.e_planck: np.ndarray = E_PLANCK.copy()

    # ------------------------------------------------------------------
    # Pointwise AD cores (extracted from .4 mesh builder; live Torch)
    # ------------------------------------------------------------------

    def _decoder_at_e3(
        self,
        e3: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return functional_call(
            self.decoder, {**params, **buffers}, (e3.unsqueeze(0),)
        ).squeeze(0)

    def _g_sigma_from_uv(
        self,
        uv: torch.Tensor,
        e_fixed: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> torch.Tensor:
        """Pullback 2x2 metric g_Σ at (e0,e2)=uv, e1 fixed."""

        def decoder_at_e3(e3: torch.Tensor) -> torch.Tensor:
            return self._decoder_at_e3(e3, params, buffers)

        e3: torch.Tensor = torch.stack(
            [uv[0], e_fixed.reshape(()).squeeze(), uv[1]]
        )
        jac: torch.Tensor = jacrev(decoder_at_e3)(e3)
        g_full: torch.Tensor = torch.einsum("ai,ab,bj->ij", jac, g_metric, jac)
        return torch.stack(
            [
                torch.stack([g_full[0, 0], g_full[0, 2]]),
                torch.stack([g_full[2, 0], g_full[2, 2]]),
            ]
        )

    def _christoffel_2d(self, g: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
        """Gamma[rho, mu, nu] = Gamma^rho_{mu nu}; dg[alpha, mu, nu] = d_alpha g_{mu nu}."""
        ginv: torch.Tensor = torch.linalg.inv(g)
        lowered: torch.Tensor = (
            dg
            + dg.permute(1, 0, 2)
            - dg.permute(1, 2, 0)
        )
        return 0.5 * torch.einsum("rl,mnl->rmn", ginv, lowered)

    def _gamma_at_uv(
        self,
        uv: torch.Tensor,
        e_fixed: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> torch.Tensor:
        """Christoffel Gamma[rho, mu, nu] = Gamma^rho_{mu nu}(u,v)."""

        def g_fn(u: torch.Tensor) -> torch.Tensor:
            return self._g_sigma_from_uv(u, e_fixed, params, buffers, g_metric)

        g: torch.Tensor = g_fn(uv)
        dg: torch.Tensor = jacrev(g_fn)(uv).permute(2, 0, 1)
        return self._christoffel_2d(g, dg)

    def _dgamma_at_uv(
        self,
        uv: torch.Tensor,
        e_fixed: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> torch.Tensor:
        """dGamma[tau, rho, mu, nu] = d_tau Gamma^rho_{mu nu}(u,v)."""

        def gamma_fn(u: torch.Tensor) -> torch.Tensor:
            return self._gamma_at_uv(u, e_fixed, params, buffers, g_metric)

        return jacrev(gamma_fn)(uv).permute(3, 0, 1, 2)

    def _g_gamma_dgamma_at_uv(
        self,
        uv: torch.Tensor,
        e_fixed: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single AD pass: (g, Gamma, dGamma) at one UV point."""

        def gamma_fn_with_aux(
            u: torch.Tensor,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            def g_fn(w: torch.Tensor) -> torch.Tensor:
                return self._g_sigma_from_uv(w, e_fixed, params, buffers, g_metric)

            g_local: torch.Tensor = g_fn(u)
            dg_local: torch.Tensor = jacrev(g_fn)(u).permute(2, 0, 1)
            gamma_local: torch.Tensor = self._christoffel_2d(g_local, dg_local)
            return gamma_local, (gamma_local, g_local)

        dgamma_raw, (gamma, g) = jacrev(gamma_fn_with_aux, has_aux=True)(uv)
        return g, gamma, dgamma_raw.permute(3, 0, 1, 2)

    def _g_sigma_batch_fn(self, uv_batch: torch.Tensor) -> torch.Tensor:
        e_fixed = self.e_fixed
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._g_sigma_from_uv(u, e_fixed, params, buffers, g_metric),
            randomness="same",
        )(uv_batch)

    def _gamma_batch_fn(self, uv_batch: torch.Tensor) -> torch.Tensor:
        e_fixed = self.e_fixed
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._gamma_at_uv(u, e_fixed, params, buffers, g_metric),
            randomness="same",
        )(uv_batch)

    def _dgamma_batch_fn(self, uv_batch: torch.Tensor) -> torch.Tensor:
        e_fixed = self.e_fixed
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._dgamma_at_uv(u, e_fixed, params, buffers, g_metric),
            randomness="same",
        )(uv_batch)

    def _g_gamma_dgamma_batch_fn(
        self, uv_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_fixed = self.e_fixed
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._g_gamma_dgamma_at_uv(
                u, e_fixed, params, buffers, g_metric
            ),
            randomness="same",
        )(uv_batch)

    # ------------------------------------------------------------------
    # Public batched API (no detach; sub-batch default 256)
    # ------------------------------------------------------------------

    def _ensure_uv_batch(self, uv: torch.Tensor) -> torch.Tensor:
        if uv.ndim == 1:
            uv = uv.unsqueeze(0)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"Expected uv shape (N, 2), got {tuple(uv.shape)}")
        if uv.dtype != torch.float64:
            raise ValueError(f"Expected uv dtype float64, got {uv.dtype}")
        return uv.to(self.device)

    def g_sigma(self, uv: torch.Tensor) -> torch.Tensor:
        """(N, 2) → (N, 2, 2) pullback metric; live Torch, no detach."""
        uv_b: torch.Tensor = self._ensure_uv_batch(uv)
        n: int = int(uv_b.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            chunks.append(self._g_sigma_batch_fn(uv_b[start:end]))
        return torch.cat(chunks, dim=0)

    def gamma(self, uv: torch.Tensor) -> torch.Tensor:
        """(N, 2) → (N, 2, 2, 2) Christoffel Γ^ρ_{μν}; live Torch, no detach."""
        uv_b: torch.Tensor = self._ensure_uv_batch(uv)
        n: int = int(uv_b.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            chunks.append(self._gamma_batch_fn(uv_b[start:end]))
        return torch.cat(chunks, dim=0)

    def dgamma(self, uv: torch.Tensor) -> torch.Tensor:
        """(N, 2) → (N, 2, 2, 2, 2) ∂_α Γ^ρ_{μν}; live Torch, no detach."""
        uv_b: torch.Tensor = self._ensure_uv_batch(uv)
        n: int = int(uv_b.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            chunks.append(self._dgamma_batch_fn(uv_b[start:end]))
        return torch.cat(chunks, dim=0)

    def g_gamma_dgamma(
        self, uv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused (g, Γ, ∂Γ) with shapes (N,2,2), (N,2,2,2), (N,2,2,2,2)."""
        uv_b: torch.Tensor = self._ensure_uv_batch(uv)
        n: int = int(uv_b.shape[0])
        g_chunks: list[torch.Tensor] = []
        gamma_chunks: list[torch.Tensor] = []
        dgamma_chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            g_t, gamma_t, dgamma_t = self._g_gamma_dgamma_batch_fn(uv_b[start:end])
            g_chunks.append(g_t)
            gamma_chunks.append(gamma_t)
            dgamma_chunks.append(dgamma_t)
        return (
            torch.cat(g_chunks, dim=0),
            torch.cat(gamma_chunks, dim=0),
            torch.cat(dgamma_chunks, dim=0),
        )


def load_g_sigma_bundle(
    *,
    device: Optional[torch.device] = None,
    g_path: Optional[str] = None,
    sub_batch: int = DEFAULT_SUB_BATCH,
    verify_dtype: bool = True,
) -> GSigmaFixedOmegab:
    """
    Construct the Phase-1 metric-AD bundle on the locked fixed-ω_b Planck face.

    Loads decoder via ``utils.models.load_decoder``, ambient G from pinned
    ``induced_metric.npy``, and freezes ``e_fixed = e_Planck[1]``.
    """
    if device is None:
        device = _default_device()
    if g_path is None:
        g_path = os.path.join(geometry_outputs, INDUCED_METRIC_FILENAME)

    decoder: nn.Module = load_decoder(main_folder, device=device)
    decoder.eval()
    if verify_dtype:
        _verify_decoder_float64(decoder)

    g_metric_np: np.ndarray = np.load(g_path).astype(np.float64)
    if g_metric_np.shape != (16, 16):
        raise ValueError(
            f"Expected ambient G shape (16, 16), got {g_metric_np.shape} from {g_path}"
        )
    g_metric: torch.Tensor = torch.tensor(
        g_metric_np, dtype=torch.float64, device=device
    )
    e_fixed: torch.Tensor = torch.tensor(
        float(E_PLANCK[FIXED_AXIS]), dtype=torch.float64, device=device
    )
    return GSigmaFixedOmegab(
        decoder=decoder,
        g_metric=g_metric,
        e_fixed=e_fixed,
        device=device,
        sub_batch=sub_batch,
    )
