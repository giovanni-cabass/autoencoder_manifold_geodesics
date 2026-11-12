"""
Ricci scalar of the latent manifold by autodiff on the N = 48 cell-centre lattice.

The differentiable pullback, the Christoffel (the result is not detached), and
the covariant Riemann are the calculation kept in ``_metric_at``,
``_christoffel_at``, and ``_riemann_cov_at``. The scalar contracts that tensor
with the index order in ``_riemann_cov_at`` and the Carroll convention in
``.cursor/agents/diff-geometry.md``. Decoder weights are constants. One GPU.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacfwd, jacrev, vmap

torch.set_default_dtype(torch.float64)

from utils.logging_scripts import Tee
from utils.models import load_decoder
from utils.workspace_folders import geometry_outputs, main_folder, out_err_folder

N: int = 48
N_GRID: int = N * N * N
SUB_BATCH: int = 64

_DECODER_MODULE: nn.Module = nn.Module()


def _frobenius(violation: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of a symmetry-violation tensor at one node."""
    flat: torch.Tensor = violation.reshape(-1)
    return torch.sqrt(torch.sum(flat * flat))


def _metric_at(
    e3: torch.Tensor,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    g_metric: torch.Tensor,
) -> torch.Tensor:
    """Full 3x3 pullback metric g_ij = J_ai G_ab J_bj at latent e3."""

    def dec(z: torch.Tensor) -> torch.Tensor:
        return functional_call(
            _DECODER_MODULE, {**params, **buffers}, (z.unsqueeze(0),)
        ).squeeze(0)

    jac: torch.Tensor = jacrev(dec)(e3)                       # (16, 3)
    return torch.einsum("ai,ab,bj->ij", jac, g_metric, jac)  # (3, 3)


def _christoffel_at(
    e3: torch.Tensor,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    g_metric: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable Christoffel Gamma^k_{lm}(e3).  NO .detach() on the result, so
    jacfwd/jacrev of this function is valid (needed for the Riemann tensor).
    Uses the pullback identity Gamma^i_{kl} = g^{ij} <d_j F, d_k d_l F>_G.
    """

    def dec(z: torch.Tensor) -> torch.Tensor:
        return functional_call(
            _DECODER_MODULE, {**params, **buffers}, (z.unsqueeze(0),)
        ).squeeze(0)

    jac: torch.Tensor = jacrev(dec)(e3)               # (16, 3)  d_i F_a
    hess: torch.Tensor = jacfwd(jacrev(dec))(e3)      # (16, 3, 3)  d_k d_l F_a
    g: torch.Tensor = torch.einsum("ai,ab,bj->ij", jac, g_metric, jac)  # (3, 3)
    g_inv: torch.Tensor = torch.linalg.inv(g)
    ch_help: torch.Tensor = torch.einsum("aj,ab,bkl->jkl", jac, g_metric, hess)  # (3,3,3)
    return torch.einsum("ij,jkl->ikl", g_inv, ch_help)  # Gamma^i_{kl}


def _riemann_cov_at(
    e3: torch.Tensor,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    g_metric: torch.Tensor,
) -> torch.Tensor:
    """Fully covariant ambient Riemann R_{a sigma mu nu} (3,3,3,3) at e3."""

    def gamma_fn(z: torch.Tensor) -> torch.Tensor:
        return _christoffel_at(z, params, buffers, g_metric)

    gam: torch.Tensor = gamma_fn(e3)                 # (3,3,3)  gam[r,m,n] = Gamma^r_{mn}
    dgam: torch.Tensor = jacfwd(gamma_fn)(e3)        # (3,3,3,3) dgam[r,l,m,n] = d_n Gamma^r_{lm}

    # R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
    #                       + Gamma^rho_{mu lam} Gamma^lam_{nu sigma}
    #                       - Gamma^rho_{nu lam} Gamma^lam_{mu sigma}
    # dgam axes are (rho, l, m, n) = d_n Gamma^rho_{l m}; Gamma symmetric in its lower pair.
    # d_mu Gamma^rho_{nu sigma}: (l=nu, m=sigma, n=mu) -> want tensor [rho,sigma,mu,nu]
    A_term: torch.Tensor = torch.einsum("rlmn->rmnl", dgam)
    # d_nu Gamma^rho_{mu sigma}: (l=mu, m=sigma, n=nu) -> [rho,sigma,mu,nu]
    B_term: torch.Tensor = torch.einsum("rlmn->rmln", dgam)
    quad: torch.Tensor = (
        torch.einsum("rml,lns->rsmn", gam, gam)
        - torch.einsum("rnl,lms->rsmn", gam, gam)
    )
    Rt: torch.Tensor = A_term - B_term + quad          # R^rho_{sigma mu nu}
    g: torch.Tensor = _metric_at(e3, params, buffers, g_metric)
    return torch.einsum("ar,rsmn->asmn", g, Rt)        # R_{a sigma mu nu}


def _ricci_scalar_and_residuals_at(
    e3: torch.Tensor,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    g_metric: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ricci scalar and the three Riemann symmetry residuals at one node.

    ``_riemann_cov_at`` returns ``R_{a sigma mu nu}``, the lowering of
    ``R^rho_{sigma mu nu}``. Those four slots are Carroll's order
    (rho, sigma, mu, nu), as written in ``_riemann_cov_at``. The contraction
    is the Carroll convention in ``.cursor/agents/diff-geometry.md``: Ricci
    contracts the first and third slots, ``R_{sigma nu} = R^rho_{sigma rho nu}``,
    and the scalar is ``R = g^{sigma nu} R_{sigma nu}``. A round sphere has
    ``R > 0``.

    Each residual is the Frobenius norm, at this node, of the violation of
    antisymmetry in one index pair, or of the pair-swap symmetry, on that
    same tensor.
    """
    Rcov: torch.Tensor = _riemann_cov_at(e3, params, buffers, g_metric)
    g: torch.Tensor = _metric_at(e3, params, buffers, g_metric)
    g_inv: torch.Tensor = torch.linalg.inv(g)
    # Slots of R^rho_{sigma mu nu} are (rho, sigma, mu, nu).
    # Rcov[a, sigma, mu, nu] = g_{a rho} R^rho_{sigma mu nu}.
    # First-with-third: R_{sigma nu} = g^{rho a} R_{a sigma rho nu}.
    ricci: torch.Tensor = torch.einsum("ra,asrn->sn", g_inv, Rcov)
    scalar: torch.Tensor = torch.einsum("sn,sn->", g_inv, ricci)
    anti_first: torch.Tensor = Rcov + Rcov.permute(1, 0, 2, 3)
    anti_second: torch.Tensor = Rcov + Rcov.permute(0, 1, 3, 2)
    pair_swap: torch.Tensor = Rcov - Rcov.permute(2, 3, 0, 1)
    return (
        scalar,
        _frobenius(anti_first),
        _frobenius(anti_second),
        _frobenius(pair_swap),
    )


def _cell_centre_lattice() -> np.ndarray:
    """Full N^3 cell-centre lattice, flat index n = i N^2 + j N + k.

    ``e_i = -1 + (2 i + 1) / N`` on every axis. Every coordinate is in (-1, 1).
    """
    i_axis: np.ndarray = np.arange(N, dtype=np.float64)
    e_axis: np.ndarray = (
        np.float64(-1.0)
        + (np.float64(2.0) * i_axis + np.float64(1.0)) / np.float64(N)
    )
    n_index: np.ndarray = np.arange(N_GRID, dtype=np.int64)
    i_idx: np.ndarray = n_index // (N * N)
    j_idx: np.ndarray = (n_index // N) % N
    k_idx: np.ndarray = n_index % N
    pts: np.ndarray = np.ascontiguousarray(
        np.stack((e_axis[i_idx], e_axis[j_idx], e_axis[k_idx]), axis=1),
        dtype=np.float64,
    )
    inside: np.ndarray = (pts > np.float64(-1.0)) & (pts < np.float64(1.0))
    if not bool(np.all(inside)):
        raise RuntimeError("A lattice sample is not strictly inside (-1, 1).")
    if pts.shape != (N_GRID, 3):
        raise RuntimeError(f"Lattice shape {pts.shape} is not ({N_GRID}, 3).")
    return pts


def _print_finite_summary(name: str, values: np.ndarray) -> None:
    """Print min, max, mean, and median of the finite entries."""
    flat: np.ndarray = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    finite_mask: np.ndarray = np.isfinite(flat)
    n_all: int = int(flat.shape[0])
    n_bad: int = int(n_all - int(np.count_nonzero(finite_mask)))
    print(f"{name}: n={n_all} nonfinite={n_bad}", flush=True)
    if n_bad == n_all:
        print(f"{name}: no finite values", flush=True)
        return
    finite: np.ndarray = flat[finite_mask]
    print(
        f"{name}: min={float(np.min(finite)):.16e} "
        f"max={float(np.max(finite)):.16e} "
        f"mean={float(np.mean(finite)):.16e} "
        f"median={float(np.median(finite)):.16e}",
        flush=True,
    )


def _verify_decoder_float64(model: nn.Module) -> None:
    all_float64: bool = (
        all(p.dtype == torch.float64 for p in model.parameters())
        and all(b.dtype == torch.float64 for b in model.buffers())
    )
    print("\n--- Verifying Model Dtype ---", flush=True)
    for name, param in model.named_parameters():
        status = "V" if param.dtype == torch.float64 else "X"
        print(f"  [{status}] param  '{name}': {param.dtype}", flush=True)
    for name, buf in model.named_buffers():
        status = "V" if buf.dtype == torch.float64 else "X"
        print(f"  [{status}] buffer '{name}': {buf.dtype}", flush=True)
    if not all_float64:
        raise RuntimeError(
            "MODEL DTYPE MISMATCH: not all parameters/buffers are float64."
        )
    print("SUCCESS: All parameters and buffers verified as float64.\n", flush=True)


def main() -> None:
    start_time = datetime.now()
    tz_name = time.tzname[0] if time.tzname else "UTC"
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

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

        print(f"🚀 Script: {script_name}.py")
        print(f"📅 Started: {start_time}")

        device_count: int = int(torch.cuda.device_count())
        print(f"torch.cuda.device_count() = {device_count}")
        if device_count != 1:
            print(
                f"Stopping: torch.cuda.device_count() = {device_count}, expected 1.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(f"Device: {device}")

        global _DECODER_MODULE
        _DECODER_MODULE = load_decoder(main_folder, device=device)
        _DECODER_MODULE.eval()
        _verify_decoder_float64(_DECODER_MODULE)
        params: dict[str, torch.Tensor] = {
            k: v.detach() for k, v in _DECODER_MODULE.named_parameters()
        }
        buffers: dict[str, torch.Tensor] = {
            k: v.detach() for k, v in _DECODER_MODULE.named_buffers()
        }

        g_path: str = os.path.join(geometry_outputs, "induced_metric.npy")
        g_metric_np: np.ndarray = np.load(g_path).astype(np.float64)
        if g_metric_np.shape != (16, 16):
            raise ValueError(
                f"Expected ambient G shape (16, 16), got {g_metric_np.shape} from {g_path}"
            )
        g_metric: torch.Tensor = torch.tensor(
            g_metric_np, dtype=torch.float64, device=device
        )
        print(f"Ambient G: {g_path} shape={tuple(g_metric.shape)} dtype={g_metric.dtype}")

        pts: np.ndarray = _cell_centre_lattice()
        print(f"Lattice N={N}, nodes={N_GRID}, e_axis[0]={pts[0, 0]}")
        print(
            "Flat index n = i*N*N + j*N + k, "
            "e_i = -1 + (2*i+1)/N on each axis."
        )

        ricci_scalar: np.ndarray = np.zeros(N_GRID, dtype=np.float64)
        residual_first_pair: np.ndarray = np.zeros(N_GRID, dtype=np.float64)
        residual_second_pair: np.ndarray = np.zeros(N_GRID, dtype=np.float64)
        residual_pair_swap: np.ndarray = np.zeros(N_GRID, dtype=np.float64)

        def one_node(
            e: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return _ricci_scalar_and_residuals_at(e, params, buffers, g_metric)

        print(f"Computing Ricci scalar on all {N_GRID} nodes (sub_batch={SUB_BATCH})...")
        t0 = time.time()
        for start in range(0, N_GRID, SUB_BATCH):
            end: int = min(start + SUB_BATCH, N_GRID)
            batch: torch.Tensor = torch.tensor(
                pts[start:end], dtype=torch.float64, device=device
            ).contiguous()
            scalar_b, first_b, second_b, swap_b = vmap(one_node)(batch)
            ricci_scalar[start:end] = (
                scalar_b.detach().cpu().numpy().astype(np.float64)
            )
            residual_first_pair[start:end] = (
                first_b.detach().cpu().numpy().astype(np.float64)
            )
            residual_second_pair[start:end] = (
                second_b.detach().cpu().numpy().astype(np.float64)
            )
            residual_pair_swap[start:end] = (
                swap_b.detach().cpu().numpy().astype(np.float64)
            )
            del batch
        print(f"Done in {time.time() - t0:.1f}s")

        ricci_scalar = np.ascontiguousarray(ricci_scalar, dtype=np.float64)
        if ricci_scalar.shape != (N_GRID,):
            raise RuntimeError(
                f"Ricci scalar shape {ricci_scalar.shape} is not ({N_GRID},)."
            )
        if ricci_scalar.dtype != np.float64:
            raise RuntimeError(
                f"Ricci scalar dtype {ricci_scalar.dtype} is not float64."
            )

        ricci_dir: str = os.path.join(
            geometry_outputs,
            "compute_curvature_at_uniform_grid_outputs",
            "planck_best_fit",
        )
        os.makedirs(ricci_dir, exist_ok=True)
        ricci_path: str = os.path.join(ricci_dir, "ricci_scalar.uniform_grid.npy")
        np.save(ricci_path, ricci_scalar)
        print(f"Saved float64 R shape {ricci_scalar.shape} to {ricci_path}")

        radius_of_curvature: np.ndarray = (
            np.float64(1.0) / np.sqrt(np.abs(ricci_scalar))
        ).astype(np.float64)
        width_over_radius: np.ndarray = (
            np.float64(2.0) / radius_of_curvature
        ).astype(np.float64)
        _print_finite_summary("R", ricci_scalar)
        _print_finite_summary("mathcal_R = 1/sqrt(|R|)", radius_of_curvature)
        _print_finite_summary(
            "width_2 / mathcal_R", width_over_radius
        )
        _print_finite_summary(
            "Riemann antisymmetry, first pair", residual_first_pair
        )
        _print_finite_summary(
            "Riemann antisymmetry, second pair", residual_second_pair
        )
        _print_finite_summary(
            "Riemann pair-swap", residual_pair_swap
        )

        end_time = datetime.now()
        print(f"\nFinished: {end_time}  (elapsed: {end_time - start_time})")

    except Exception as exc:
        print(f"\n💥 UNHANDLED EXCEPTION in main: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
        error_file_handle.close()
        print(f"✅ Logs saved to: {out_err_folder}")


if __name__ == "__main__":
    main()
