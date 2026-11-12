"""
Task 38 Phase 9 — adapted discrete path energy + AdamW (campaign-local copy).

Copy-rename-edit of task34_path_energy_core.py for the 3D soft-6σ campaign
(design §0.9 / §5). Production path for Task 38 §5 — do NOT thin-wrap
energy_geodesic_scripts.minimize_pairwise_path_energy or the 2D core.

Locked math (design §5.1):
  Δe_k = e_{k+1} - e_k,  ē_k = ½(e_k + e_{k+1}),
  Ê = (K/2) Σ_k g_ab(ē_k) Δe_k^a Δe_k^b,
  D = √(2 Ê★),  Ê★ = per-pair best-of-trajectory (incl. §4 Dijkstra init).

  g is the unsliced 3×3 decoder pullback (utils.g_sigma_3d); nodes in R^3.

Adaptations vs 2D:
  - Chart dim 3; g (N,3,3) via GSigma3D.g — never slice to a 2×2 face
  - Soft 6σ box report-only (locked inlined bounds); hard inset project
  - weight_decay fixed 0; full n_iters; endpoints frozen
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from utils.g_sigma_3d import GSigma3D, load_g_sigma_3d_bundle

# Hard decoder inset (design §0.1 / §5.3): project free e into this closed box only.
EPS_DOM: np.float64 = np.float64(1.0e-6)
DECODER_LO: np.float64 = np.float64(-1.0) + EPS_DOM
DECODER_HI: np.float64 = np.float64(1.0) - EPS_DOM

# Soft 6σ Planck-centered box — report only; never project onto it (§0.1).
SOFT_6SIGMA_E0: Tuple[np.float64, np.float64] = (
    np.float64(-0.286947),
    np.float64(-0.057053),
)
SOFT_6SIGMA_E1: Tuple[np.float64, np.float64] = (
    np.float64(-0.368000),
    np.float64(0.304000),
)
SOFT_6SIGMA_E2: Tuple[np.float64, np.float64] = (
    np.float64(-0.301500),
    np.float64(-0.049500),
)

# AdamW weight_decay is locked at 0 — not a config input (design §0.6).
WEIGHT_DECAY_FIXED: float = 0.0

CHART_DIM: int = 3

ArrayLike = Union[np.ndarray, torch.Tensor]


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _as_e_batch(
    arr: ArrayLike,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        t: torch.Tensor = torch.tensor(arr, dtype=torch.float64, device=device)
    elif isinstance(arr, torch.Tensor):
        t = arr.to(dtype=torch.float64, device=device)
    else:
        raise TypeError(f"{name} must be np.ndarray or torch.Tensor")
    if t.ndim != 2 or t.shape[1] != CHART_DIM:
        raise ValueError(
            f"{name} expected shape (P, 3), got {tuple(t.shape)}"
        )
    return t


def _as_nodes_batch(
    arr: ArrayLike,
    device: torch.device,
    k: int,
    name: str,
) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        t: torch.Tensor = torch.tensor(arr, dtype=torch.float64, device=device)
    elif isinstance(arr, torch.Tensor):
        t = arr.to(dtype=torch.float64, device=device)
    else:
        raise TypeError(f"{name} must be np.ndarray or torch.Tensor")
    if t.ndim != 3 or t.shape[1] != (k - 1) or t.shape[2] != CHART_DIM:
        raise ValueError(
            f"{name} expected shape (P, K-1, 3)=(?, {k - 1}, 3), "
            f"got {tuple(t.shape)}"
        )
    return t


def discrete_path_energy(
    nodes: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    g_bundle: GSigma3D,
    K: int,
) -> torch.Tensor:
    """
    Per-pair discrete energy Ê; shape (P,).

    Live Torch through g_3D (no detach on g). Endpoints x,y frozen.
    """
    path: torch.Tensor = torch.cat(
        [x[:, None, :], nodes, y[:, None, :]], dim=1
    )  # (P, K+1, 3)
    mid: torch.Tensor = 0.5 * (path[:, 1:, :] + path[:, :-1, :])  # (P, K, 3)
    dseg: torch.Tensor = path[:, 1:, :] - path[:, :-1, :]  # (P, K, 3)
    p: int = int(mid.shape[0])
    mid_flat: torch.Tensor = mid.reshape(p * K, CHART_DIM)
    g_flat: torch.Tensor = g_bundle.g(mid_flat)  # (P*K, 3, 3)
    g_mid: torch.Tensor = g_flat.reshape(p, K, CHART_DIM, CHART_DIM)
    e_per_seg: torch.Tensor = torch.einsum(
        "pki,pkij,pkj->pk", dseg, g_mid, dseg
    )
    e_p: torch.Tensor = (0.5 * float(K)) * e_per_seg.sum(dim=1)
    return e_p


def _count_outside_6sigma(nodes: torch.Tensor) -> Tuple[int, int]:
    """Return (n_nodes_outside, n_free_nodes) for current free-node tensor."""
    e0: torch.Tensor = nodes[..., 0]
    e1: torch.Tensor = nodes[..., 1]
    e2: torch.Tensor = nodes[..., 2]
    outside: torch.Tensor = (
        (e0 < float(SOFT_6SIGMA_E0[0]))
        | (e0 > float(SOFT_6SIGMA_E0[1]))
        | (e1 < float(SOFT_6SIGMA_E1[0]))
        | (e1 > float(SOFT_6SIGMA_E1[1]))
        | (e2 < float(SOFT_6SIGMA_E2[0]))
        | (e2 > float(SOFT_6SIGMA_E2[1]))
    )
    n_out: int = int(outside.sum().item())
    n_tot: int = int(outside.numel())
    return n_out, n_tot


def _pair_has_node_outside_inset(nodes: torch.Tensor) -> torch.Tensor:
    """
    Per-pair bool: True if any free node is outside hard inset (pre-projection).

    nodes: (P, K-1, 3) → (P,) bool
    """
    lo: float = float(DECODER_LO)
    hi: float = float(DECODER_HI)
    outside: torch.Tensor = (
        (nodes[..., 0] < lo)
        | (nodes[..., 0] > hi)
        | (nodes[..., 1] < lo)
        | (nodes[..., 1] > hi)
        | (nodes[..., 2] < lo)
        | (nodes[..., 2] > hi)
    )
    return outside.any(dim=1)


def _pair_has_node_outside_full_cube(nodes: torch.Tensor) -> torch.Tensor:
    """Per-pair bool: any free node outside chart cube [-1,1]^3 (pre-projection)."""
    outside: torch.Tensor = (
        (nodes[..., 0] < -1.0)
        | (nodes[..., 0] > 1.0)
        | (nodes[..., 1] < -1.0)
        | (nodes[..., 1] > 1.0)
        | (nodes[..., 2] < -1.0)
        | (nodes[..., 2] > 1.0)
    )
    return outside.any(dim=1)


def _project_decoder_inset_(nodes: nn.Parameter) -> Tuple[int, torch.Tensor]:
    """
    In-place project free e into decoder inset.

    Returns (n_moved_nodes, pair_moved_mask) where pair_moved_mask is (P,) bool.
    """
    before: torch.Tensor = nodes.data.clone()
    nodes.data.clamp_(float(DECODER_LO), float(DECODER_HI))
    moved_per_node: torch.Tensor = (nodes.data - before).abs().amax(dim=-1) > 0.0
    n_moved: int = int(moved_per_node.sum().item())
    pair_moved: torch.Tensor = moved_per_node.any(dim=1)
    return n_moved, pair_moved


def _probe_nested_ad_path(
    nodes_init: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    g_bundle: GSigma3D,
    K: int,
) -> str:
    """Try Parameter+backward; fall back to autograd.grad. Returns path name."""
    probe: nn.Parameter = nn.Parameter(nodes_init.detach().clone())
    e_p: torch.Tensor = discrete_path_energy(probe, x, y, g_bundle, K)
    loss: torch.Tensor = e_p.sum()
    try:
        loss.backward()
        if probe.grad is None:
            raise RuntimeError("nodes.grad is None after backward()")
        if not bool(torch.isfinite(probe.grad).all()):
            raise RuntimeError("non-finite grad after backward()")
        return "backward"
    except Exception as exc:
        print(
            f"WARNING: primary nested-AD path (Parameter+backward) failed: {exc}",
            flush=True,
        )
        print("Falling back to torch.autograd.grad ...", flush=True)
        probe2: nn.Parameter = nn.Parameter(nodes_init.detach().clone())
        e_p2: torch.Tensor = discrete_path_energy(probe2, x, y, g_bundle, K)
        grads: torch.Tensor
        (grads,) = torch.autograd.grad(e_p2.sum(), probe2)
        if grads is None or not bool(torch.isfinite(grads).all()):
            raise RuntimeError(
                "Fallback autograd.grad also failed or produced non-finite grads"
            )
        return "grad"


def minimize_pairwise_path_energy_from_init(
    x: ArrayLike,
    y: ArrayLike,
    nodes_init: ArrayLike,
    *,
    K: int = 16,
    lr: float = 1e-3,
    n_iters: int = 100,
    device: Optional[torch.device] = None,
    g_bundle: Optional[GSigma3D] = None,
    verbose: bool = False,
    probe_nested_ad: bool = True,
    nested_ad_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Minimize discrete path energy from §4 Dijkstra-resampled free-node inits.

    Endpoints x,y stay frozen; only (K-1) free interiors are AdamW variables.
    weight_decay is hardcoded to WEIGHT_DECAY_FIXED (0.0) — not an argument.

    Returns
    -------
    D : (P,) float64, D = sqrt(2 Ê★)
    nodes_best : (P, K-1, 3) best-of-trajectory free nodes
    info : diagnostics (clamps, soft-6σ, comfort masks, timings, …)
    """
    if int(K) < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if float(lr) <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if int(n_iters) < 0:
        raise ValueError(f"n_iters must be >= 0, got {n_iters}")

    if device is None:
        device = _default_device()
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if g_bundle is None:
        g_bundle = load_g_sigma_3d_bundle(device=device, verify_dtype=True)
    else:
        if g_bundle.device != device:
            raise ValueError(
                f"g_bundle.device={g_bundle.device} != requested device={device}"
            )

    x_t: torch.Tensor = _as_e_batch(x, device, "x")
    y_t: torch.Tensor = _as_e_batch(y, device, "y")
    if x_t.shape[0] != y_t.shape[0]:
        raise ValueError(
            f"x and y batch size mismatch: {x_t.shape[0]} vs {y_t.shape[0]}"
        )
    nodes0: torch.Tensor = _as_nodes_batch(nodes_init, device, int(K), "nodes_init")
    if nodes0.shape[0] != x_t.shape[0]:
        raise ValueError(
            f"nodes_init batch {nodes0.shape[0]} != endpoints batch {x_t.shape[0]}"
        )

    with torch.no_grad():
        e_init: torch.Tensor = discrete_path_energy(
            nodes0, x_t, y_t, g_bundle, int(K)
        ).detach().clone()

    path_name: str
    if nested_ad_path is not None:
        path_name = str(nested_ad_path)
    elif probe_nested_ad:
        path_name = _probe_nested_ad_path(nodes0, x_t, y_t, g_bundle, int(K))
    else:
        path_name = "backward"

    if verbose:
        print(f"nested_ad_path={path_name}", flush=True)
        print(
            f"E_init (Dijkstra resample) mean={float(e_init.mean().item()):.6e}",
            flush=True,
        )

    nodes: nn.Parameter = nn.Parameter(nodes0.detach().clone())
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        [nodes], lr=float(lr), weight_decay=float(WEIGHT_DECAY_FIXED)
    )

    best_e: torch.Tensor = e_init.detach().clone()
    best_nodes: torch.Tensor = nodes0.detach().clone()

    n_clamp_to_decoder_box: int = 0
    n_steps_outside_6sigma: int = 0
    sum_frac_outside: float = 0.0
    n_soft_node_excursions: int = 0

    pair_exited_inset_pre_proj: torch.Tensor = torch.zeros(
        x_t.shape[0], dtype=torch.bool, device=device
    )
    pair_exited_full_cube: torch.Tensor = torch.zeros(
        x_t.shape[0], dtype=torch.bool, device=device
    )
    with torch.no_grad():
        pair_exited_inset_pre_proj |= _pair_has_node_outside_inset(nodes0)
        pair_exited_full_cube |= _pair_has_node_outside_full_cube(nodes0)

    for it in range(int(n_iters)):
        optimizer.zero_grad(set_to_none=True)
        e_p: torch.Tensor = discrete_path_energy(nodes, x_t, y_t, g_bundle, int(K))
        loss: torch.Tensor = e_p.sum()
        if path_name == "backward":
            loss.backward()
        else:
            grads: torch.Tensor
            (grads,) = torch.autograd.grad(loss, nodes)
            nodes.grad = grads

        if nodes.grad is None or not bool(torch.isfinite(nodes.grad).all()):
            raise RuntimeError(
                f"Non-finite or missing grad at AdamW step {it} "
                f"(nested_ad_path={path_name})"
            )

        optimizer.step()

        with torch.no_grad():
            pair_exited_inset_pre_proj |= _pair_has_node_outside_inset(nodes.data)
            pair_exited_full_cube |= _pair_has_node_outside_full_cube(nodes.data)

            n_out, n_tot = _count_outside_6sigma(nodes.data)
            n_soft_node_excursions += n_out
            frac_this: float = float(n_out) / float(n_tot) if n_tot > 0 else 0.0
            sum_frac_outside += frac_this
            if n_out > 0:
                n_steps_outside_6sigma += 1
                if verbose:
                    print(
                        f"WARNING: {n_out}/{n_tot} free node(s) outside soft 6σ "
                        f"box at step {it} (report only; not projected).",
                        flush=True,
                    )

            n_moved, _pair_moved = _project_decoder_inset_(nodes)
            if n_moved > 0:
                n_clamp_to_decoder_box += n_moved
                if verbose:
                    print(
                        f"WARNING: projected {n_moved} free node(s) into decoder "
                        f"inset at step {it}.",
                        flush=True,
                    )

            e_step: torch.Tensor = discrete_path_energy(
                nodes.data, x_t, y_t, g_bundle, int(K)
            )
            improved: torch.Tensor = e_step < best_e
            if bool(improved.any()):
                best_e = torch.where(improved, e_step, best_e)
                best_nodes = torch.where(
                    improved[:, None, None], nodes.data, best_nodes
                )

    with torch.no_grad():
        e_last: torch.Tensor = discrete_path_energy(
            nodes, x_t, y_t, g_bundle, int(K)
        )
        e_final: torch.Tensor = best_e.detach().clone()
        d_final: torch.Tensor = torch.sqrt(2.0 * e_final)
        n_pairs_best_at_init: int = int((~(best_e < e_init)).sum().item())

    frac_nodes_outside_6sigma: float = (
        sum_frac_outside / float(n_iters) if n_iters > 0 else 0.0
    )

    pair_comfortable: torch.Tensor = ~pair_exited_inset_pre_proj
    n_comfortable: int = int(pair_comfortable.sum().item())
    n_pairs: int = int(x_t.shape[0])
    comfort_fraction: float = (
        float(n_comfortable) / float(n_pairs) if n_pairs > 0 else 1.0
    )

    info: Dict[str, Any] = {
        "nested_ad_path": path_name,
        "n_steps_outside_6sigma": int(n_steps_outside_6sigma),
        "frac_nodes_outside_6sigma": float(frac_nodes_outside_6sigma),
        "n_soft_node_excursions": int(n_soft_node_excursions),
        "n_clamp_to_decoder_box": int(n_clamp_to_decoder_box),
        "K": int(K),
        "lr": float(lr),
        "n_iters": int(n_iters),
        "optimizer": "AdamW",
        "weight_decay": float(WEIGHT_DECAY_FIXED),
        "E_init": e_init.detach().clone(),
        "E": e_final.detach().clone(),
        "E_last": e_last.detach().clone(),
        "best_of_trajectory": True,
        "n_pairs_best_at_init": int(n_pairs_best_at_init),
        "pair_exited_inset_pre_proj": pair_exited_inset_pre_proj.detach().clone(),
        "pair_comfortable": pair_comfortable.detach().clone(),
        "pair_exited_full_cube": pair_exited_full_cube.detach().clone(),
        "n_comfortable": int(n_comfortable),
        "comfort_fraction": float(comfort_fraction),
        "n_pairs_exited_full_cube": int(pair_exited_full_cube.sum().item()),
        "device": str(device),
        "init_source": "dijkstra_resample_section_4_energy_set",
        "eps_dom": float(EPS_DOM),
        "decoder_lo": float(DECODER_LO),
        "decoder_hi": float(DECODER_HI),
        "chart_dim": int(CHART_DIM),
    }

    return d_final.detach().clone(), best_nodes.detach().clone(), info
