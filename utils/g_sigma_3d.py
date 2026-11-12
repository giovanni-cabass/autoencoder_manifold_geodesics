"""
Task 38 Phase 1 — batched differentiable 3×3 decoder pullback g (+ Γ, ∂Γ).

Copy-rename-edit of ``utils/g_sigma_fixed_omegab.py`` for the full latent
volume (e0, e1, e2). **Do not slice** g_full — public g is the unsliced
3×3 pullback (design §0.15).

Index convention (same as the 2D face util):
  - g:      (N, 3, 3)
  - Gamma:  (N, 3, 3, 3) with Gamma[..., ρ, μ, ν] = Γ^ρ_{μν}
  - dGamma: (N, 3, 3, 3, 3) with dGamma[..., α, ρ, μ, ν] = ∂_α Γ^ρ_{μν}

Public returns are live float64 Torch tensors (no detach). Internal sub-batch
default is 256.

API smoke (not a §7.3 PASS gate): at e1 = e1_Planck, the (0,2) block of
g_3D matches ``g_sigma_fixed_omegab`` on the same (e0, e2). Run via
``python -m utils.g_sigma_3d`` or ``smoke_face_block_vs_fixed_omegab()``.
Tees go under parent ``out_err_folder`` with ``task38_*`` stems (not the
2D energy_campaign_2D_logs umbrella).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

from utils.g_sigma_fixed_omegab import E_PLANCK, load_g_sigma_bundle
from utils.logging_scripts import Tee
from utils.models import load_decoder
from utils.workspace_folders import geometry_outputs, main_folder, out_err_folder

DEFAULT_SUB_BATCH: int = 256
INDUCED_METRIC_FILENAME: str = "induced_metric.npy"
LATENT_DIM: int = 3


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


class GSigma3D:
    """Loaded bundle: decoder + ambient G; batched unsliced 3×3 g/Γ/∂Γ."""

    def __init__(
        self,
        decoder: nn.Module,
        g_metric: torch.Tensor,
        device: torch.device,
        *,
        sub_batch: int = DEFAULT_SUB_BATCH,
    ) -> None:
        self.decoder: nn.Module = decoder
        self.g_metric: torch.Tensor = g_metric
        self.device: torch.device = device
        self.sub_batch: int = int(sub_batch)
        # Detach decoder weights from AD (differentiate w.r.t. e only).
        self.params: dict[str, torch.Tensor] = {
            k: v.detach() for k, v in decoder.named_parameters()
        }
        self.buffers: dict[str, torch.Tensor] = {
            k: v.detach() for k, v in decoder.named_buffers()
        }
        self.e_planck: np.ndarray = E_PLANCK.copy()
        self.latent_dim: int = LATENT_DIM

    # ------------------------------------------------------------------
    # Pointwise AD cores (3D unsliced pullback)
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

    def _g_from_e(
        self,
        e: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> torch.Tensor:
        """Pullback 3×3 metric g at e=(e0,e1,e2) — do not slice g_full."""

        def decoder_at_e3(e3: torch.Tensor) -> torch.Tensor:
            return self._decoder_at_e3(e3, params, buffers)

        jac: torch.Tensor = jacrev(decoder_at_e3)(e)
        g_full: torch.Tensor = torch.einsum("ai,ab,bj->ij", jac, g_metric, jac)
        return g_full

    def _christoffel_3d(self, g: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
        """Gamma[rho, mu, nu] = Gamma^rho_{mu nu}; dg[alpha, mu, nu] = d_alpha g_{mu nu}."""
        ginv: torch.Tensor = torch.linalg.inv(g)
        lowered: torch.Tensor = (
            dg
            + dg.permute(1, 0, 2)
            - dg.permute(1, 2, 0)
        )
        return 0.5 * torch.einsum("rl,mnl->rmn", ginv, lowered)

    def _gamma_at_e(
        self,
        e: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> torch.Tensor:
        """Christoffel Gamma[rho, mu, nu] = Gamma^rho_{mu nu}(e)."""

        def g_fn(u: torch.Tensor) -> torch.Tensor:
            return self._g_from_e(u, params, buffers, g_metric)

        g: torch.Tensor = g_fn(e)
        dg: torch.Tensor = jacrev(g_fn)(e).permute(2, 0, 1)
        return self._christoffel_3d(g, dg)

    def _dgamma_at_e(
        self,
        e: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> torch.Tensor:
        """dGamma[tau, rho, mu, nu] = d_tau Gamma^rho_{mu nu}(e)."""

        def gamma_fn(u: torch.Tensor) -> torch.Tensor:
            return self._gamma_at_e(u, params, buffers, g_metric)

        return jacrev(gamma_fn)(e).permute(3, 0, 1, 2)

    def _g_gamma_dgamma_at_e(
        self,
        e: torch.Tensor,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        g_metric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single AD pass: (g, Gamma, dGamma) at one e point."""

        def gamma_fn_with_aux(
            u: torch.Tensor,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            def g_fn(w: torch.Tensor) -> torch.Tensor:
                return self._g_from_e(w, params, buffers, g_metric)

            g_local: torch.Tensor = g_fn(u)
            dg_local: torch.Tensor = jacrev(g_fn)(u).permute(2, 0, 1)
            gamma_local: torch.Tensor = self._christoffel_3d(g_local, dg_local)
            return gamma_local, (gamma_local, g_local)

        dgamma_raw, (gamma, g) = jacrev(gamma_fn_with_aux, has_aux=True)(e)
        return g, gamma, dgamma_raw.permute(3, 0, 1, 2)

    def _g_batch_fn(self, e_batch: torch.Tensor) -> torch.Tensor:
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._g_from_e(u, params, buffers, g_metric),
            randomness="same",
        )(e_batch)

    def _gamma_batch_fn(self, e_batch: torch.Tensor) -> torch.Tensor:
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._gamma_at_e(u, params, buffers, g_metric),
            randomness="same",
        )(e_batch)

    def _dgamma_batch_fn(self, e_batch: torch.Tensor) -> torch.Tensor:
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._dgamma_at_e(u, params, buffers, g_metric),
            randomness="same",
        )(e_batch)

    def _g_gamma_dgamma_batch_fn(
        self, e_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.params
        buffers = self.buffers
        g_metric = self.g_metric
        return vmap(
            lambda u: self._g_gamma_dgamma_at_e(u, params, buffers, g_metric),
            randomness="same",
        )(e_batch)

    # ------------------------------------------------------------------
    # Public batched API (no detach; sub-batch default 256)
    # ------------------------------------------------------------------

    def _ensure_e_batch(self, e: torch.Tensor) -> torch.Tensor:
        if e.ndim == 1:
            e = e.unsqueeze(0)
        if e.ndim != 2 or e.shape[1] != LATENT_DIM:
            raise ValueError(
                f"Expected e shape (N, {LATENT_DIM}), got {tuple(e.shape)}"
            )
        if e.dtype != torch.float64:
            raise ValueError(f"Expected e dtype float64, got {e.dtype}")
        return e.to(self.device)

    def g(self, e: torch.Tensor) -> torch.Tensor:
        """(N, 3) → (N, 3, 3) unsliced pullback metric; live Torch, no detach."""
        e_b: torch.Tensor = self._ensure_e_batch(e)
        n: int = int(e_b.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            chunks.append(self._g_batch_fn(e_b[start:end]))
        return torch.cat(chunks, dim=0)

    def gamma(self, e: torch.Tensor) -> torch.Tensor:
        """(N, 3) → (N, 3, 3, 3) Christoffel Γ^ρ_{μν}; live Torch, no detach."""
        e_b: torch.Tensor = self._ensure_e_batch(e)
        n: int = int(e_b.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            chunks.append(self._gamma_batch_fn(e_b[start:end]))
        return torch.cat(chunks, dim=0)

    def dgamma(self, e: torch.Tensor) -> torch.Tensor:
        """(N, 3) → (N, 3, 3, 3, 3) ∂_α Γ^ρ_{μν}; live Torch, no detach."""
        e_b: torch.Tensor = self._ensure_e_batch(e)
        n: int = int(e_b.shape[0])
        chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            chunks.append(self._dgamma_batch_fn(e_b[start:end]))
        return torch.cat(chunks, dim=0)

    def g_gamma_dgamma(
        self, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused (g, Γ, ∂Γ) with shapes (N,3,3), (N,3,3,3), (N,3,3,3,3)."""
        e_b: torch.Tensor = self._ensure_e_batch(e)
        n: int = int(e_b.shape[0])
        g_chunks: list[torch.Tensor] = []
        gamma_chunks: list[torch.Tensor] = []
        dgamma_chunks: list[torch.Tensor] = []
        for start in range(0, n, self.sub_batch):
            end = min(start + self.sub_batch, n)
            g_t, gamma_t, dgamma_t = self._g_gamma_dgamma_batch_fn(
                e_b[start:end]
            )
            g_chunks.append(g_t)
            gamma_chunks.append(gamma_t)
            dgamma_chunks.append(dgamma_t)
        return (
            torch.cat(g_chunks, dim=0),
            torch.cat(gamma_chunks, dim=0),
            torch.cat(dgamma_chunks, dim=0),
        )


def load_g_sigma_3d_bundle(
    *,
    device: Optional[torch.device] = None,
    g_path: Optional[str] = None,
    sub_batch: int = DEFAULT_SUB_BATCH,
    verify_dtype: bool = True,
) -> GSigma3D:
    """
    Construct the Task-38 3D metric-AD bundle (unsliced 3×3 pullback).

    Loads decoder via ``utils.models.load_decoder`` and ambient G from pinned
    ``induced_metric.npy``.
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
    return GSigma3D(
        decoder=decoder,
        g_metric=g_metric,
        device=device,
        sub_batch=sub_batch,
    )


def _face_smoke_uv_points() -> np.ndarray:
    """Deterministic (e0, e2) samples near Planck for face-block comparison."""
    e0_grid: np.ndarray = np.linspace(
        np.float64(-0.22), np.float64(-0.12), 3, dtype=np.float64
    )
    e2_grid: np.ndarray = np.linspace(
        np.float64(-0.23), np.float64(-0.12), 3, dtype=np.float64
    )
    pts: list[list[np.float64]] = []
    for e0 in e0_grid:
        for e2 in e2_grid:
            pts.append([np.float64(e0), np.float64(e2)])
    pts[4] = [
        np.float64(E_PLANCK[0]),
        np.float64(E_PLANCK[2]),
    ]
    return np.asarray(pts, dtype=np.float64)


def smoke_face_block_vs_fixed_omegab(
    *,
    device: Optional[torch.device] = None,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> None:
    """
    API smoke: at e1 = e1_Planck, (0,2) block of g_3D matches face util.

    Raises AssertionError / RuntimeError on mismatch. Not a §7.3 PASS gate.
    """
    if device is None:
        device = _default_device()

    bundle_3d: GSigma3D = load_g_sigma_3d_bundle(
        device=device, verify_dtype=True
    )
    bundle_2d = load_g_sigma_bundle(device=device, verify_dtype=False)

    uv_np: np.ndarray = _face_smoke_uv_points()
    n: int = int(uv_np.shape[0])
    e1_planck: float = float(E_PLANCK[1])
    e_np: np.ndarray = np.zeros((n, 3), dtype=np.float64)
    e_np[:, 0] = uv_np[:, 0]
    e_np[:, 1] = e1_planck
    e_np[:, 2] = uv_np[:, 1]

    uv: torch.Tensor = torch.tensor(uv_np, dtype=torch.float64, device=device)
    e: torch.Tensor = torch.tensor(e_np, dtype=torch.float64, device=device)

    g3: torch.Tensor = bundle_3d.g(e)
    g2: torch.Tensor = bundle_2d.g_sigma(uv)

    if tuple(g3.shape) != (n, 3, 3):
        raise RuntimeError(f"Expected g3 shape {(n, 3, 3)}, got {tuple(g3.shape)}")
    if tuple(g2.shape) != (n, 2, 2):
        raise RuntimeError(f"Expected g2 shape {(n, 2, 2)}, got {tuple(g2.shape)}")

    # (0,2) block of 3D g ↔ 2×2 face g_Σ
    block: torch.Tensor = torch.stack(
        [
            torch.stack([g3[:, 0, 0], g3[:, 0, 2]], dim=-1),
            torch.stack([g3[:, 2, 0], g3[:, 2, 2]], dim=-1),
        ],
        dim=-2,
    )
    max_abs: float = float((block - g2).abs().max().item())
    print(
        f"Face-block smoke: N={n}, e1=Planck={e1_planck:.12g}, "
        f"max|g3[0,2]-g_face|={max_abs:.3e}",
        flush=True,
    )
    if not bool(torch.allclose(block, g2, rtol=rtol, atol=atol)):
        raise AssertionError(
            f"Face-block mismatch: max abs diff {max_abs:.3e} "
            f"(rtol={rtol}, atol={atol})"
        )

    # Shape / index sanity on fused path
    g_f, gamma_f, dgamma_f = bundle_3d.g_gamma_dgamma(e[:2])
    if tuple(g_f.shape) != (2, 3, 3):
        raise RuntimeError(f"fused g shape {tuple(g_f.shape)}")
    if tuple(gamma_f.shape) != (2, 3, 3, 3):
        raise RuntimeError(f"fused Gamma shape {tuple(gamma_f.shape)}")
    if tuple(dgamma_f.shape) != (2, 3, 3, 3, 3):
        raise RuntimeError(f"fused dGamma shape {tuple(dgamma_f.shape)}")
    print(
        "Shapes OK: g (N,3,3), Gamma (N,3,3,3) [Γ^ρ_μν], "
        "dGamma (N,3,3,3,3)",
        flush=True,
    )
    print("Face-block smoke PASS", flush=True)


def main() -> None:
    """Runnable smoke with Tee under parent out_err_folder (task38_* stems)."""
    torch.set_default_dtype(torch.float64)
    start_time = datetime.now()
    tz_name = time.tzname[0] if time.tzname else "UTC"
    script_name = "task38_g_sigma_3d_smoke"

    out_name = (
        f'{script_name}_output_{start_time.strftime("%Y%m%d_%H%M%S")}_{tz_name}.txt'
    )
    err_name = (
        f'{script_name}_errors_{start_time.strftime("%Y%m%d_%H%M%S")}_{tz_name}.txt'
    )
    log_file_path = os.path.join(out_err_folder, out_name)
    error_file_path = os.path.join(out_err_folder, err_name)

    log_file_handle = open(log_file_path, "w", encoding="utf-8")
    error_file_handle = open(error_file_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = Tee(original_stdout, log_file_handle)
        sys.stderr = Tee(original_stderr, error_file_handle)

        print(f"🚀 Script: {script_name}")
        print(f"📅 Started: {start_time}")
        print("Task 38 Phase 1 — g_sigma_3d face-block smoke")

        device = _default_device()
        print(f"Device: {device}")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            _ = torch.zeros(1, dtype=torch.float64, device=device)
            torch.cuda.synchronize()

        smoke_face_block_vs_fixed_omegab(device=device)
        print("\n✅ PASS: task38_g_sigma_3d_smoke")

    except Exception as exc:
        print(f"\n💥 UNHANDLED EXCEPTION in main: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print("\n❌ FAIL: task38_g_sigma_3d_smoke")
        raise

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
        error_file_handle.close()
        print(f"✅ Logs saved to: {out_err_folder}")


if __name__ == "__main__":
    main()
