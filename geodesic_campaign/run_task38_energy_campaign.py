"""
Task 38 Phase 9 — energy-set discrete path-energy campaign
(AdamW from §4 Dijkstra inits; optional dual chord Pass B + min D).

mp.spawn multi-GPU non-DDP (§0.9 skill + §0.11 list-pad; override skill
N % world_size == 0). Zero inter-rank communication; each GPU loads g_3D +
its pad-slice of inits, runs adapted AdamW, writes per-rank D/γ; parent
gathers real slots only.

Work list = energy set only (P_energy). Never AdamW on all P at Nc=21.

Pass A: if P_energy==0 skip (no empty-list §0.11); else N=P_energy on
  Dijkstra-resampled energy inits → D_dij_arm + provisional win.
Pass B: if n_dual==0 skip; else N=n_dual on sparse chord row axis;
  pads append after real duals; gather real only; scatter via energy-axis
  indices from pair_index_dual ⊂ pair_index_energy (no global-id re-sort).
Combine: D=min; arm_choice int8 {0,1,2} on energy axis; tie → Dijkstra (1).

Multi-job Pass A/B (§0.9 / Task 39 §18.5):
  --energy-pass both (default) = today's one job (A then B then production write).
  --energy-pass a | b = one AdamW arm → intermediate arm NPZ (not production).
  --energy-pass merge = load arm NPZs → combine → production energy + catalog
    (CPU-only; no CUDA). Staging/rank: both keeps legacy untagged names;
    a/b append _passa/_passb so separate jobs do not clobber.

Catalog (length P, separate NPZ): D, endpoints (P,2,3), pair_source —
  no production interiors (P,K-1,3). Analytic slots from early oracle;
  energy slots scatter winning D/endpoints after Pass A/B.

Campaign energy logic: task38_path_energy_core.py (copy/adapt — not a thin
wrap of 2D / energy_geodesic_scripts).

Implement-time names (via task38_campaign_knobs):
  Core:      task38_path_energy_core.py
  Script:    run_task38_energy_campaign.py
  Submit:    submit.run_task38_energy_campaign.sh
  Staging:   phase5_staging_basename(..., t_exp=, n_iters=, dual knobs)
  Rank out:  phase5_rank_basename(...)
  Energy:    energy_basename / energy_output_basename (length P_energy)
  Catalog:   task38_catalog_D_endpoints_* (implement-time; same tags)

Smoke:
  Prefer reduced --n-iters (e.g. 5) while keeping knobs DEFAULT_N_ITERS=100
  for production. Optional --max-pairs N → *_smokeP{N}.npz.
  Example Nc=3 / 26nbr:
    --N_c 3 --stride 2 --K 16 --connectivity 26 \\
    --t-exp 1e-3 --t-fat 1.01 --t-nlo 1e-2 --dual-init on \\
    --n-iters 5 --p-batch 64

Run from workspace root (main_folder):
  /home/gcabass/.conda/envs/torch_cuda_pyro/bin/python -u \\
    python_scripts/run_task38_energy_campaign.py \\
    --N_c 3 --stride 2 --K 16 --connectivity 26 \\
    --t-exp 1e-3 --t-fat 1.01 --t-nlo 1e-2 --dual-init on \\
    --n-iters 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

torch.set_default_dtype(torch.float64)

from task38_campaign_knobs import (
    ALLOWED_CONN_TAGS,
    DEFAULT_CONNECTIVITY,
    DEFAULT_DUAL_INIT,
    DEFAULT_ENERGY_PASS,
    DEFAULT_K,
    DEFAULT_LR,
    DEFAULT_N_C,
    DEFAULT_N_ITERS,
    DEFAULT_P_BATCH,
    DEFAULT_STRIDE,
    artifact_path,
    campaign_tee_dir,
    catalog_basename,
    chord_inits_basename,
    dual_init_mask_basename,
    energy_basename,
    energy_output_basename,
    energy_pass_a_arm_basename,
    energy_pass_b_arm_basename,
    grids_basename,
    inits_basename,
    n_f_from,
    note02_oracle_basename,
    pair_split_basename,
    phase5_rank_basename,
    phase5_staging_basename,
    validate_connectivity,
    validate_dual_init,
    validate_dual_threshold_consistency,
    validate_energy_pass,
    validate_grid_knobs,
    validate_n_iters,
    validate_t_exp,
    validate_t_fat,
    validate_t_nlo,
)
from utils.g_sigma_3d import load_g_sigma_3d_bundle
from utils.logging_scripts import Tee
from utils.workspace_folders import main_folder

import task38_path_energy_core as pec

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_LR_F: float = float(DEFAULT_LR)
BASE_SEED: int = 38_009
CACHE_CLEAR_INTERVAL: int = 4
CHART_DIM: int = 3
COMFORT_TARGET: np.float64 = np.float64(0.997)

# arm_choice int8 codes (§5.2): 0=Dijkstra-only, 1=Dijkstra won or tie, 2=chord won
ARM_NON_DUAL: np.int8 = np.int8(0)
ARM_DIJKSTRA: np.int8 = np.int8(1)
ARM_CHORD: np.int8 = np.int8(2)

PAIR_SOURCE_ANALYTIC: str = "analytic"
PAIR_SOURCE_ENERGY: str = "energy"
PAIR_SOURCE_INVALID: str = "invalid_or_tiny"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task 38 Phase 9 energy campaign: Pass A Dijkstra energy-set + "
            "optional Pass B chord (mp.spawn + §0.11; catalog + energy NPZ)."
        )
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        default=None,
        # No choices=: validate_connectivity strip+lowers (ALL / " 26 " OK).
        help=(
            f"Connectivity: all (default {DEFAULT_CONNECTIVITY}) | 6 | 18 | 26. "
            "Env CONNECTIVITY if CLI omitted. Sequential when all."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR_F,
        help=f"AdamW learning rate (default {DEFAULT_LR_F})",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help=(
            f"AdamW iterations, no early stop (default {DEFAULT_N_ITERS}; "
            "env N_ITERS if CLI omitted). Smoke: prefer smaller, e.g. 5."
        ),
    )
    parser.add_argument(
        "--p-batch",
        type=int,
        default=DEFAULT_P_BATCH,
        help=f"Pairs per AdamW microbatch (default {DEFAULT_P_BATCH})",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help=(
            "If >0, truncate energy list to first N rows (smoke) and write "
            "*_smokeP{N}.npz — never production basename. 0 = full P_energy."
        ),
    )
    parser.add_argument(
        "--energy-pass",
        type=str,
        default=None,
        # No choices=: validate_energy_pass strip+lowers (BOTH/A/ merge OK).
        help=(
            f"Pass packaging (default {DEFAULT_ENERGY_PASS}; env ENERGY_PASS; "
            f"{{both,a,b,merge}}). "
            "both=A then B then production (legacy one-job; CONNECTIVITY=all OK); "
            "a|b|merge require CONNECTIVITY=6|18|26 (not all)."
        ),
    )
    parser.add_argument(
        "--init-6",
        type=str,
        default="",
        help="Optional §4.7 6-neighbor init basename (default: inits_basename).",
    )
    parser.add_argument(
        "--init-18",
        type=str,
        default="",
        help="Optional §4.7 18-neighbor init basename (default: inits_basename).",
    )
    parser.add_argument(
        "--init-26",
        type=str,
        default="",
        help="Optional §4.7 26-neighbor init basename (default: inits_basename).",
    )
    parser.add_argument(
        "--N_c",
        type=int,
        default=DEFAULT_N_C,
        help=f"Coarse grid size per axis (odd; default {DEFAULT_N_C})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Fine/coarse stride s (default {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        help=f"Campaign K (default {DEFAULT_K}; must match init artifact)",
    )
    parser.add_argument(
        "--t-exp",
        type=float,
        default=None,
        help=(
            "Expansion trust threshold t_exp > 0 (required; must match split). "
            "Env T_EXP if CLI omitted."
        ),
    )
    parser.add_argument(
        "--t-fat",
        type=float,
        default=None,
        help="Fat threshold t_fat > 1 (required). Env T_FAT if CLI omitted.",
    )
    parser.add_argument(
        "--t-nlo",
        type=float,
        default=None,
        help=(
            "NLO/LO threshold t_nlo > 0 (required; dual-on ⇒ t_exp < t_nlo). "
            "Env T_NLO if CLI omitted."
        ),
    )
    parser.add_argument(
        "--dual-init",
        type=str,
        default=None,
        choices=("on", "off"),
        help=(
            f"Dual-init flag (default {DEFAULT_DUAL_INIT}; "
            "env DUAL_INIT if CLI omitted). Must match mask sidecar."
        ),
    )
    return parser.parse_args()


def _env_or_default_float(env_name: str, default: Optional[np.float64]) -> Optional[np.float64]:
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return default
    return np.float64(raw)


def _env_or_default_str(env_name: str, default: str) -> str:
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return str(default)
    return str(raw).strip()


def _env_or_default_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(str(raw).strip())


def _resolve_knobs(
    args: argparse.Namespace,
) -> Tuple[np.float64, np.float64, np.float64, str, int, str, str]:
    """CLI overrides env; FAIL-closed validate. t_exp/t_fat/t_nlo required."""
    if args.t_exp is None:
        t_exp_raw = _env_or_default_float("T_EXP", None)
    else:
        t_exp_raw = np.float64(args.t_exp)
    if t_exp_raw is None:
        raise ValueError(
            "t_exp is required (--t-exp or env T_EXP); no production default "
            "until design §6.0"
        )

    if args.t_fat is None:
        t_fat_raw = _env_or_default_float("T_FAT", None)
    else:
        t_fat_raw = np.float64(args.t_fat)
    if t_fat_raw is None:
        raise ValueError(
            "t_fat is required (--t-fat or env T_FAT); no production default "
            "until design §6.0"
        )

    if args.t_nlo is None:
        t_nlo_raw = _env_or_default_float("T_NLO", None)
    else:
        t_nlo_raw = np.float64(args.t_nlo)
    if t_nlo_raw is None:
        raise ValueError(
            "t_nlo is required (--t-nlo or env T_NLO); no production default "
            "until design §6.0"
        )

    if args.dual_init is None:
        dual_raw: str = _env_or_default_str("DUAL_INIT", DEFAULT_DUAL_INIT)
    else:
        dual_raw = str(args.dual_init)

    if args.n_iters is None:
        n_iters_raw: int = _env_or_default_int("N_ITERS", DEFAULT_N_ITERS)
    else:
        n_iters_raw = int(args.n_iters)

    if args.connectivity is None:
        conn_raw: str = _env_or_default_str("CONNECTIVITY", DEFAULT_CONNECTIVITY)
    else:
        conn_raw = str(args.connectivity)

    if args.energy_pass is None:
        energy_pass_raw: str = _env_or_default_str(
            "ENERGY_PASS", DEFAULT_ENERGY_PASS
        )
    else:
        energy_pass_raw = str(args.energy_pass)

    t_exp: np.float64 = validate_t_exp(t_exp_raw)
    t_fat: np.float64 = validate_t_fat(t_fat_raw)
    t_nlo: np.float64 = validate_t_nlo(t_nlo_raw)
    dual_init: str = validate_dual_init(dual_raw)
    n_iters: int = validate_n_iters(n_iters_raw)
    connectivity: str = validate_connectivity(conn_raw)
    energy_pass: str = validate_energy_pass(energy_pass_raw)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    # Split modes write/merge one connectivity at a time. CONNECTIVITY=all would
    # loop 6→18→26 and can leave earlier catalogs on disk if a later conn fails.
    if energy_pass in ("a", "b", "merge") and connectivity == "all":
        raise ValueError(
            f"energy_pass={energy_pass!r} forbids CONNECTIVITY=all "
            "(use CONNECTIVITY=6|18|26 per job; default energy_pass=both "
            "still allows all for knob-scan)."
        )
    return t_exp, t_fat, t_nlo, dual_init, n_iters, connectivity, energy_pass


def _warn_both_connectivity_all(energy_pass: str, connectivity: str) -> None:
    """Knob-scan allows both+all; warn about half-finished 6→18→26 risk."""
    if energy_pass == "both" and connectivity == "all":
        print(
            "WARNING: energy_pass=both with CONNECTIVITY=all runs 6→18→26 in "
            "one job; if a later connectivity fails, earlier production "
            "energy/catalog NPZs may already be on disk. Prefer per-conn "
            "jobs for production (CONNECTIVITY=26|18|6).",
            flush=True,
        )


def _conn_list(connectivity: str) -> List[str]:
    if connectivity == "all":
        return list(ALLOWED_CONN_TAGS)
    return [str(connectivity)]


def build_section_011_pad(
    n_real: int,
    world_size: int,
) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
    """
    §0.11 list-pad: N_work = ceil(N/world_size)*world_size; assert n_pad < world_size.

    Returns (N_work, n_pad, slots_per_gpu, is_dummy[N_work], work_to_real[N_work]).
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1 (got {world_size})")
    if n_real < 0:
        raise ValueError(f"n_real must be >= 0 (got {n_real})")
    n_work: int = int(math.ceil(float(n_real) / float(world_size))) * world_size
    n_pad: int = n_work - n_real
    if n_pad >= world_size:
        raise AssertionError(
            f"§0.11 pad contract FAIL: n_pad={n_pad} >= world_size={world_size} "
            f"(N={n_real}, N_work={n_work}). Abort — never soft-drop duals."
        )
    slots: int = n_work // world_size
    is_dummy: np.ndarray = np.zeros(n_work, dtype=bool)
    for i in range(n_pad):
        rank: int = world_size - 1 - i
        gidx: int = rank * slots + (slots - 1)
        is_dummy[gidx] = True

    work_to_real: np.ndarray = np.full(n_work, -1, dtype=np.int64)
    real_cursor: int = 0
    for i in range(n_work):
        if not bool(is_dummy[i]):
            work_to_real[i] = np.int64(real_cursor)
            real_cursor += 1
    if real_cursor != n_real:
        raise AssertionError(
            f"Pad layout bug: filled {real_cursor} real slots, expected {n_real}"
        )
    n_dummy_check: int = int(np.sum(is_dummy.astype(np.int64)))
    if n_dummy_check != n_pad:
        raise AssertionError(
            f"Pad layout bug: is_dummy sum={n_dummy_check}, n_pad={n_pad}"
        )
    return n_work, n_pad, slots, is_dummy, work_to_real


def _cuda_warmup(device: torch.device) -> None:
    _z: torch.Tensor = torch.zeros(1, dtype=torch.float64, device=device)
    torch.cuda.synchronize(device)
    del _z
    a: torch.Tensor = torch.randn(
        2, 2, dtype=torch.float64, device=device, requires_grad=True
    )
    (a @ a.T).sum().backward()
    torch.cuda.synchronize(device)
    del a


def run_chain(
    rank: int,
    world_size: int,
    base_seed: int,
    lr: float,
    n_iters: int,
    p_batch: int,
    k: int,
    conn_tag: str,
    staging_basename: str,
    n_c: int,
    stride: int,
    t_exp: float,
    t_fat: float,
    t_nlo: float,
    dual_init: str,
) -> None:
    """
    Per-GPU worker: Tee → set_device → warm-up → load_g_sigma_3d_bundle →
    AdamW on real pad-slice pairs → save per-rank D/γ (real only).
    """
    start_time = datetime.now()
    tz_name = time.tzname[0] if time.tzname else "UTC"
    script_name = "run_task38_energy_campaign"
    tee_dir = campaign_tee_dir()

    out_name = (
        f"{script_name}_rank{rank}_{conn_tag}nbr_output_"
        f'{start_time.strftime("%Y%m%d_%H%M%S")}_{tz_name}.txt'
    )
    err_name = (
        f"{script_name}_rank{rank}_{conn_tag}nbr_errors_"
        f'{start_time.strftime("%Y%m%d_%H%M%S")}_{tz_name}.txt'
    )
    log_file_path = os.path.join(tee_dir, out_name)
    error_file_path = os.path.join(tee_dir, err_name)

    log_file_handle = open(log_file_path, "w", encoding="utf-8")
    error_file_handle = open(error_file_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = Tee(original_stdout, log_file_handle)
        sys.stderr = Tee(original_stderr, error_file_handle)

        print(f"🚀 Script: {script_name}.py (rank {rank})", flush=True)
        print(f"📅 Started: {start_time}", flush=True)
        print(f"[rank {rank}] connectivity={conn_tag}-neighbor", flush=True)
        print(f"[rank {rank}] world_size={world_size}", flush=True)
        print(
            f"[rank {rank}] lr={lr}, n_iters={n_iters}, p_batch={p_batch}, K={k}",
            flush=True,
        )
        print(
            f"[rank {rank}] weight_decay={pec.WEIGHT_DECAY_FIXED} (fixed; not config)",
            flush=True,
        )
        print(
            f"[rank {rank}] t_exp={t_exp:g}, t_fat={t_fat:g}, t_nlo={t_nlo:g}, "
            f"dual_init={dual_init!r}",
            flush=True,
        )
        print(f"[rank {rank}] seed={base_seed + rank}", flush=True)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        _cuda_warmup(device)

        torch.manual_seed(int(base_seed) + int(rank))
        np.random.seed(int(base_seed) + int(rank))

        staging_path = artifact_path(staging_basename)
        staging = np.load(staging_path, allow_pickle=False)
        n_work: int = int(np.asarray(staging["n_work"]).item())
        slots: int = int(np.asarray(staging["slots_per_gpu"]).item())
        n_real: int = int(np.asarray(staging["P"]).item())
        assert n_work == slots * world_size
        assert slots == n_work // world_size
        if "energy_pass" in staging.files:
            energy_pass_staging: str = validate_energy_pass(
                str(np.asarray(staging["energy_pass"]).item())
            )
        else:
            energy_pass_staging = DEFAULT_ENERGY_PASS
        print(
            f"[rank {rank}] energy_pass(staging)={energy_pass_staging!r}",
            flush=True,
        )

        lo: int = rank * slots
        hi: int = lo + slots
        is_dummy_slice: np.ndarray = np.asarray(
            staging["is_dummy"][lo:hi], dtype=bool
        ).copy()
        work_to_real_slice: np.ndarray = np.asarray(
            staging["work_to_real"][lo:hi], dtype=np.int64
        ).copy()
        interiors_pad: np.ndarray = np.asarray(
            staging["interiors_pad"][lo:hi], dtype=np.float64
        ).copy()
        endpoints_pad: np.ndarray = np.asarray(
            staging["endpoints_pad"][lo:hi], dtype=np.float64
        ).copy()

        real_local_pos: np.ndarray = np.flatnonzero(~is_dummy_slice)
        n_real_local: int = int(real_local_pos.shape[0])
        n_dummy_local: int = int(np.sum(is_dummy_slice.astype(np.int64)))
        print(
            f"[rank {rank}] slots={slots}, n_real_local={n_real_local}, "
            f"n_dummy_local={n_dummy_local}",
            flush=True,
        )

        g_bundle = load_g_sigma_3d_bundle(device=device, verify_dtype=True)

        nested_ad_path: str = "backward"
        if n_real_local > 0:
            probe_n: int = min(2, n_real_local)
            probe_idx: np.ndarray = real_local_pos[:probe_n]
            nested_ad_path = pec._probe_nested_ad_path(
                torch.tensor(
                    interiors_pad[probe_idx], dtype=torch.float64, device=device
                ),
                torch.tensor(
                    endpoints_pad[probe_idx, 0, :],
                    dtype=torch.float64,
                    device=device,
                ),
                torch.tensor(
                    endpoints_pad[probe_idx, 1, :],
                    dtype=torch.float64,
                    device=device,
                ),
                g_bundle,
                int(k),
            )
        print(f"[rank {rank}] nested_ad_path={nested_ad_path}", flush=True)

        D_local = np.full(n_real_local, np.nan, dtype=np.float64)
        interiors_best = np.full(
            (n_real_local, int(k) - 1, CHART_DIM), np.nan, dtype=np.float64
        )
        endpoints_out = np.full(
            (n_real_local, 2, CHART_DIM), np.nan, dtype=np.float64
        )
        real_pair_ids = np.full(n_real_local, -1, dtype=np.int64)
        E_init_local = np.full(n_real_local, np.nan, dtype=np.float64)
        E_best_local = np.full(n_real_local, np.nan, dtype=np.float64)
        E_last_local = np.full(n_real_local, np.nan, dtype=np.float64)
        pair_comfortable = np.ones(n_real_local, dtype=bool)
        pair_exited_inset = np.zeros(n_real_local, dtype=bool)
        pair_exited_full_cube = np.zeros(n_real_local, dtype=bool)
        best_at_init = np.zeros(n_real_local, dtype=bool)

        n_clamp_total: int = 0
        n_soft_excursions_total: int = 0
        n_steps_outside_6sigma_total: int = 0

        t_adamw0 = time.time()
        cursor: int = 0
        for start in range(0, n_real_local, int(p_batch)):
            end = min(start + int(p_batch), n_real_local)
            batch_pos: np.ndarray = real_local_pos[start:end]
            bsz: int = int(batch_pos.shape[0])
            if bsz == 0:
                continue

            x_np: np.ndarray = np.ascontiguousarray(
                endpoints_pad[batch_pos, 0, :], dtype=np.float64
            )
            y_np: np.ndarray = np.ascontiguousarray(
                endpoints_pad[batch_pos, 1, :], dtype=np.float64
            )
            nodes_np: np.ndarray = np.ascontiguousarray(
                interiors_pad[batch_pos], dtype=np.float64
            )

            d_t, nodes_best_t, info = pec.minimize_pairwise_path_energy_from_init(
                x_np,
                y_np,
                nodes_np,
                K=int(k),
                lr=float(lr),
                n_iters=int(n_iters),
                device=device,
                g_bundle=g_bundle,
                verbose=False,
                probe_nested_ad=False,
                nested_ad_path=nested_ad_path,
            )

            d_np: np.ndarray = d_t.detach().cpu().numpy().astype(np.float64)
            nodes_np_best: np.ndarray = (
                nodes_best_t.detach().cpu().numpy().astype(np.float64)
            )
            e_init_np: np.ndarray = (
                info["E_init"].detach().cpu().numpy().astype(np.float64)
            )
            e_best_np: np.ndarray = (
                info["E"].detach().cpu().numpy().astype(np.float64)
            )
            e_last_np: np.ndarray = (
                info["E_last"].detach().cpu().numpy().astype(np.float64)
            )
            comfortable_np: np.ndarray = (
                info["pair_comfortable"].detach().cpu().numpy().astype(bool)
            )
            exited_inset_np: np.ndarray = (
                info["pair_exited_inset_pre_proj"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            exited_cube_np: np.ndarray = (
                info["pair_exited_full_cube"].detach().cpu().numpy().astype(bool)
            )

            sl = slice(cursor, cursor + bsz)
            D_local[sl] = d_np
            interiors_best[sl] = nodes_np_best
            endpoints_out[sl, 0, :] = x_np
            endpoints_out[sl, 1, :] = y_np
            real_pair_ids[sl] = work_to_real_slice[batch_pos]
            E_init_local[sl] = e_init_np
            E_best_local[sl] = e_best_np
            E_last_local[sl] = e_last_np
            pair_comfortable[sl] = comfortable_np
            pair_exited_inset[sl] = exited_inset_np
            pair_exited_full_cube[sl] = exited_cube_np
            best_at_init[sl] = ~(e_best_np < e_init_np)

            n_clamp_total += int(info["n_clamp_to_decoder_box"])
            n_soft_excursions_total += int(info["n_soft_node_excursions"])
            n_steps_outside_6sigma_total += int(info["n_steps_outside_6sigma"])
            cursor += bsz

            if (start // int(p_batch)) % int(CACHE_CLEAR_INTERVAL) == 0:
                torch.cuda.empty_cache()

        adamw_seconds: float = float(time.time() - t_adamw0)
        print(
            f"[rank {rank}] AdamW done: n_real_local={n_real_local}, "
            f"seconds={adamw_seconds:.3f}, n_clamp={n_clamp_total}, "
            f"n_soft={n_soft_excursions_total}",
            flush=True,
        )

        rank_bn = phase5_rank_basename(
            rank,
            conn_tag,
            n_c,
            stride,
            int(k),
            n_iters=int(n_iters),
            t_exp=t_exp,
            t_fat=t_fat,
            t_nlo=t_nlo,
            dual_init=dual_init,
            energy_pass=energy_pass_staging,
        )
        rank_path = artifact_path(rank_bn)
        np.savez_compressed(
            rank_path,
            D=np.ascontiguousarray(D_local, dtype=np.float64),
            interiors=np.ascontiguousarray(interiors_best, dtype=np.float64),
            endpoints=np.ascontiguousarray(endpoints_out, dtype=np.float64),
            real_pair_ids=np.ascontiguousarray(real_pair_ids, dtype=np.int64),
            E_init=np.ascontiguousarray(E_init_local, dtype=np.float64),
            E_best=np.ascontiguousarray(E_best_local, dtype=np.float64),
            E_last=np.ascontiguousarray(E_last_local, dtype=np.float64),
            pair_comfortable=np.ascontiguousarray(pair_comfortable, dtype=bool),
            pair_exited_inset_pre_proj=np.ascontiguousarray(
                pair_exited_inset, dtype=bool
            ),
            pair_exited_full_cube=np.ascontiguousarray(
                pair_exited_full_cube, dtype=bool
            ),
            best_at_init=np.ascontiguousarray(best_at_init, dtype=bool),
            n_real_local=np.asarray(n_real_local, dtype=np.int64),
            n_clamp_to_decoder_box=np.asarray(n_clamp_total, dtype=np.int64),
            n_soft_node_excursions=np.asarray(
                n_soft_excursions_total, dtype=np.int64
            ),
            n_steps_outside_6sigma=np.asarray(
                n_steps_outside_6sigma_total, dtype=np.int64
            ),
            adamw_seconds=np.asarray(adamw_seconds, dtype=np.float64),
            nested_ad_path=np.asarray(nested_ad_path),
            rank=np.asarray(rank, dtype=np.int64),
            world_size=np.asarray(world_size, dtype=np.int64),
            N_work_axis=np.asarray(n_real, dtype=np.int64),
            chart_dim=np.asarray(CHART_DIM, dtype=np.int64),
        )
        print(f"[rank {rank}] wrote {rank_path}", flush=True)

    except Exception as exc:
        print(f"\n💥 UNHANDLED EXCEPTION rank {rank}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
        error_file_handle.close()


def _catalog_duals_to_energy_axis(
    pair_index_dual: np.ndarray,
    pair_index_energy: np.ndarray,
) -> np.ndarray:
    """Map catalog dual indices → energy-axis rows via searchsorted."""
    pid: np.ndarray = np.ascontiguousarray(pair_index_dual, dtype=np.int64)
    pe: np.ndarray = np.ascontiguousarray(pair_index_energy, dtype=np.int64)
    if pid.ndim != 1 or pe.ndim != 1:
        raise AssertionError("pair_index arrays must be 1-D")
    if not bool(np.all(pe[1:] >= pe[:-1])):
        raise AssertionError("pair_index_energy must be ascending")
    pos: np.ndarray = np.searchsorted(pe, pid)
    if bool(np.any(pos < 0)) or bool(np.any(pos >= pe.shape[0])):
        raise AssertionError("pair_index_dual out of energy-set range")
    if not bool(np.array_equal(pe[pos], pid)):
        raise AssertionError(
            "pair_index_dual is not a subset of pair_index_energy "
            "(dual must lie on energy set)"
        )
    return np.ascontiguousarray(pos, dtype=np.int64)


def _load_mask_and_chord(
    conn_tag: str,
    *,
    n_c: int,
    stride: int,
    k: int,
    p_catalog: int,
    p_energy: int,
    t_exp_cli: np.float64,
    t_fat_cli: np.float64,
    t_nlo_cli: np.float64,
    dual_init_cli: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str, str]:
    """Load dual-init mask + sparse chord. FAIL-closed knob/index asserts."""
    mask_bn: str = dual_init_mask_basename(
        conn_tag,
        n_c,
        stride,
        t_exp=t_exp_cli,
        t_fat=t_fat_cli,
        t_nlo=t_nlo_cli,
        dual_init=dual_init_cli,
    )
    chord_bn: str = chord_inits_basename(
        conn_tag,
        n_c,
        stride,
        k,
        t_exp=t_exp_cli,
        t_fat=t_fat_cli,
        t_nlo=t_nlo_cli,
        dual_init=dual_init_cli,
    )
    mask_path: str = artifact_path(mask_bn)
    chord_path: str = artifact_path(chord_bn)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Dual-init mask sidecar missing: {mask_path}")
    if not os.path.isfile(chord_path):
        raise FileNotFoundError(f"Sparse chord init artifact missing: {chord_path}")

    mz = np.load(mask_path, allow_pickle=False)
    cz = np.load(chord_path, allow_pickle=False)

    t_exp_side: np.float64 = np.float64(np.asarray(mz["t_exp"]).item())
    t_fat_side: np.float64 = np.float64(np.asarray(mz["t_fat"]).item())
    t_nlo_side: np.float64 = np.float64(np.asarray(mz["t_nlo"]).item())
    dual_side: str = str(np.asarray(mz["dual_init"]).item())
    if t_exp_side != np.float64(t_exp_cli):
        raise AssertionError(
            f"Knob assert FAIL: CLI/env t_exp={float(t_exp_cli):g} != "
            f"mask sidecar t_exp={float(t_exp_side):g} ({mask_bn}). "
            "No silent remask."
        )
    if t_fat_side != np.float64(t_fat_cli):
        raise AssertionError(
            f"Knob assert FAIL: CLI/env t_fat={float(t_fat_cli):g} != "
            f"mask sidecar t_fat={float(t_fat_side):g} ({mask_bn})."
        )
    if t_nlo_side != np.float64(t_nlo_cli):
        raise AssertionError(
            f"Knob assert FAIL: CLI/env t_nlo={float(t_nlo_cli):g} != "
            f"mask sidecar t_nlo={float(t_nlo_side):g} ({mask_bn})."
        )
    if dual_side != str(dual_init_cli):
        raise AssertionError(
            f"Knob assert FAIL: CLI/env dual-init={dual_init_cli!r} != "
            f"mask sidecar dual_init={dual_side!r} ({mask_bn})."
        )

    pair_index_dual: np.ndarray = np.ascontiguousarray(
        mz["pair_index_dual"], dtype=np.int64
    )
    pair_index_chord: np.ndarray = np.ascontiguousarray(
        cz["pair_index"], dtype=np.int64
    )
    if pair_index_chord.shape != pair_index_dual.shape or not bool(
        np.array_equal(pair_index_chord, pair_index_dual)
    ):
        raise AssertionError(
            "Chord↔mask index assert FAIL: chord pair_index != "
            f"mask pair_index_dual ({chord_bn} vs {mask_bn})."
        )

    n_dual: int = int(pair_index_chord.shape[0])
    n_dual_mask_meta: int = int(np.asarray(mz["n_dual"]).item())
    n_dual_chord_meta: int = int(np.asarray(cz["n_dual"]).item())
    if n_dual != n_dual_mask_meta or n_dual != n_dual_chord_meta:
        raise AssertionError(
            f"n_dual mismatch: pair_index.len={n_dual}, "
            f"mask n_dual={n_dual_mask_meta}, chord n_dual={n_dual_chord_meta}"
        )

    p_mask: int = int(np.asarray(mz["P"]).item())
    p_chord: int = int(np.asarray(cz["P"]).item())
    pe_mask: int = int(np.asarray(mz["P_energy"]).item())
    pe_chord: int = int(np.asarray(cz["P_energy"]).item())
    if p_mask != p_catalog or p_chord != p_catalog:
        raise AssertionError(
            f"P mismatch vs catalog: catalog P={p_catalog}, "
            f"mask P={p_mask}, chord P={p_chord}"
        )
    if pe_mask != p_energy or pe_chord != p_energy:
        raise AssertionError(
            f"P_energy mismatch: energy inits P_energy={p_energy}, "
            f"mask={pe_mask}, chord={pe_chord}"
        )

    k_chord: int = int(np.asarray(cz["K"]).item())
    if k_chord != k:
        raise AssertionError(f"Chord K={k_chord} != campaign K={k}")

    chord_interiors: np.ndarray = np.ascontiguousarray(
        cz["interiors"], dtype=np.float64
    )
    chord_endpoints: np.ndarray = np.ascontiguousarray(
        cz["endpoints"], dtype=np.float64
    )
    if chord_interiors.shape != (n_dual, k - 1, CHART_DIM):
        raise AssertionError(
            f"chord interiors shape {chord_interiors.shape} != "
            f"(n_dual,K-1,3)=({n_dual},{k - 1},{CHART_DIM})"
        )
    if chord_endpoints.shape != (n_dual, 2, CHART_DIM):
        raise AssertionError(
            f"chord endpoints shape {chord_endpoints.shape} != "
            f"(n_dual,2,3)=({n_dual},2,{CHART_DIM})"
        )

    return (
        pair_index_chord,
        chord_interiors,
        chord_endpoints,
        n_dual,
        mask_bn,
        chord_bn,
    )


def _filter_duals_for_energy_subset(
    pair_index: np.ndarray,
    chord_interiors: np.ndarray,
    chord_endpoints: np.ndarray,
    pair_index_energy_eff: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Smoke: keep duals whose catalog index is in truncated energy set."""
    pe: np.ndarray = np.ascontiguousarray(pair_index_energy_eff, dtype=np.int64)
    pid: np.ndarray = np.ascontiguousarray(pair_index, dtype=np.int64)
    keep: np.ndarray = np.isin(pid, pe)
    pair_f: np.ndarray = np.ascontiguousarray(pid[keep], dtype=np.int64)
    int_f: np.ndarray = np.ascontiguousarray(
        chord_interiors[keep], dtype=np.float64
    )
    end_f: np.ndarray = np.ascontiguousarray(
        chord_endpoints[keep], dtype=np.float64
    )
    return pair_f, int_f, end_f, int(pair_f.shape[0])


def _load_energy_init_artifact(
    basename: str,
    max_pairs: int,
    *,
    n_c_cli: int,
    stride_cli: int,
    k_cli: int,
    t_exp_cli: np.float64,
) -> Dict[str, np.ndarray]:
    path = artifact_path(basename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"§4.7 energy-init artifact missing: {path}")
    z = np.load(path, allow_pickle=False)
    interiors: np.ndarray = np.asarray(z["interiors"], dtype=np.float64)
    endpoints: np.ndarray = np.asarray(z["endpoints"], dtype=np.float64)
    pair_index_energy: np.ndarray = np.asarray(
        z["pair_index_energy"], dtype=np.int64
    )
    p_catalog: int = int(np.asarray(z["P"]).item())
    p_energy: int = int(np.asarray(z["P_energy"]).item())
    k: int = int(np.asarray(z["K"]).item())
    n_c_art: int = int(np.asarray(z["N_c"]).item())
    n_f_art: int = int(np.asarray(z["N_f"]).item())
    t_exp_art: np.float64 = np.float64(np.asarray(z["t_exp"]).item())

    if n_c_art != n_c_cli:
        raise AssertionError(f"Init N_c={n_c_art} != CLI --N_c={n_c_cli}")
    n_f_expected: int = n_f_from(n_c_cli, stride_cli)
    if n_f_art != n_f_expected:
        raise AssertionError(
            f"Init N_f={n_f_art} != s*(N_c-1)+1={n_f_expected}"
        )
    if k != k_cli:
        raise AssertionError(
            f"Init continuum K={k} != CLI --K={k_cli}"
        )
    if t_exp_art != np.float64(t_exp_cli):
        raise AssertionError(
            f"Init t_exp={float(t_exp_art):g} != CLI t_exp={float(t_exp_cli):g}"
        )
    if "stride" in z.files:
        stride_art: int = int(np.asarray(z["stride"]).item())
        if stride_art != stride_cli:
            raise AssertionError(
                f"Init stride={stride_art} != CLI --stride={stride_cli}"
            )
    if interiors.shape != (p_energy, k - 1, CHART_DIM):
        raise AssertionError(
            f"interiors shape {interiors.shape} != "
            f"(P_energy,K-1,3)=({p_energy},{k - 1},{CHART_DIM})"
        )
    if endpoints.shape != (p_energy, 2, CHART_DIM):
        raise AssertionError(
            f"endpoints shape {endpoints.shape} != "
            f"(P_energy,2,3)=({p_energy},2,{CHART_DIM})"
        )
    if pair_index_energy.shape != (p_energy,):
        raise AssertionError(
            f"pair_index_energy shape {pair_index_energy.shape} != ({p_energy},)"
        )
    chart_dim_art: int = int(np.asarray(z["chart_dim"]).item())
    if chart_dim_art != CHART_DIM:
        raise AssertionError(f"chart_dim={chart_dim_art} != {CHART_DIM}")

    interiors = np.ascontiguousarray(interiors, dtype=np.float64)
    endpoints = np.ascontiguousarray(endpoints, dtype=np.float64)
    pair_index_energy = np.ascontiguousarray(pair_index_energy, dtype=np.int64)

    if max_pairs > 0:
        if max_pairs > p_energy:
            raise ValueError(
                f"--max-pairs={max_pairs} > P_energy={p_energy}"
            )
        p_energy = int(max_pairs)
        interiors = np.ascontiguousarray(interiors[:p_energy], dtype=np.float64)
        endpoints = np.ascontiguousarray(endpoints[:p_energy], dtype=np.float64)
        pair_index_energy = np.ascontiguousarray(
            pair_index_energy[:p_energy], dtype=np.int64
        )

    out: Dict[str, np.ndarray] = {
        "interiors": interiors,
        "endpoints": endpoints,
        "pair_index_energy": pair_index_energy,
        "P": np.asarray(p_catalog, dtype=np.int64),
        "P_energy": np.asarray(p_energy, dtype=np.int64),
        "K": np.asarray(k, dtype=np.int64),
        "N_c": np.asarray(n_c_art, dtype=np.int64),
        "N_f": np.asarray(n_f_art, dtype=np.int64),
        "stride": np.asarray(stride_cli, dtype=np.int64),
        "S": np.asarray(z["S"], dtype=np.int64),
        "connectivity": np.asarray(z["connectivity"]),
        "pair_order": np.asarray(z["pair_order"]),
        "init_artifact": np.asarray(basename),
        "t_exp": np.asarray(float(t_exp_art), dtype=np.float64),
        "chart_dim": np.asarray(CHART_DIM, dtype=np.int64),
    }
    if "D_dij" in z.files:
        out["D_dij"] = np.ascontiguousarray(
            np.asarray(z["D_dij"][:p_energy], dtype=np.float64), dtype=np.float64
        )
    if "L_graph" in z.files:
        out["L_graph"] = np.ascontiguousarray(
            np.asarray(z["L_graph"][:p_energy], dtype=np.float64), dtype=np.float64
        )
    if "m_ij" in z.files:
        out["m_ij"] = np.ascontiguousarray(
            np.asarray(z["m_ij"][:p_energy], dtype=np.int64), dtype=np.int64
        )
    return out


def _write_staging(
    conn_tag: str,
    init: Dict[str, np.ndarray],
    world_size: int,
    lr: float,
    n_iters: int,
    p_batch: int,
    n_c: int,
    stride: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    energy_pass: str,
) -> Tuple[str, int, int, int, np.ndarray, np.ndarray]:
    # Work length for this pass is init["P"] (Pass A: P_energy; Pass B: n_dual).
    p: int = int(np.asarray(init["P"]).item())
    k: int = int(np.asarray(init["K"]).item())
    n_work, n_pad, slots, is_dummy, work_to_real = build_section_011_pad(
        p, world_size
    )

    interiors_pad = np.zeros((n_work, k - 1, CHART_DIM), dtype=np.float64)
    endpoints_pad = np.zeros((n_work, 2, CHART_DIM), dtype=np.float64)
    for w in range(n_work):
        if not bool(is_dummy[w]):
            ridx: int = int(work_to_real[w])
            interiors_pad[w] = init["interiors"][ridx]
            endpoints_pad[w] = init["endpoints"][ridx]

    staging_bn = phase5_staging_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=int(n_iters),
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        energy_pass=energy_pass,
    )
    staging_path = artifact_path(staging_bn)
    np.savez_compressed(
        staging_path,
        interiors_pad=np.ascontiguousarray(interiors_pad, dtype=np.float64),
        endpoints_pad=np.ascontiguousarray(endpoints_pad, dtype=np.float64),
        is_dummy=np.ascontiguousarray(is_dummy, dtype=bool),
        work_to_real=np.ascontiguousarray(work_to_real, dtype=np.int64),
        n_work=np.asarray(n_work, dtype=np.int64),
        n_pad=np.asarray(n_pad, dtype=np.int64),
        slots_per_gpu=np.asarray(slots, dtype=np.int64),
        world_size=np.asarray(world_size, dtype=np.int64),
        P=np.asarray(p, dtype=np.int64),
        K=np.asarray(k, dtype=np.int64),
        N_c=np.asarray(n_c, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        N_f=np.asarray(n_f_from(n_c, stride), dtype=np.int64),
        lr=np.asarray(float(lr), dtype=np.float64),
        n_iters=np.asarray(int(n_iters), dtype=np.int64),
        p_batch=np.asarray(int(p_batch), dtype=np.int64),
        weight_decay=np.asarray(pec.WEIGHT_DECAY_FIXED, dtype=np.float64),
        connectivity=np.asarray(conn_tag),
        t_exp=np.asarray(float(t_exp), dtype=np.float64),
        t_fat=np.asarray(float(t_fat), dtype=np.float64),
        t_nlo=np.asarray(float(t_nlo), dtype=np.float64),
        dual_init=np.asarray(str(dual_init)),
        energy_pass=np.asarray(str(energy_pass)),
        pad_contract=np.asarray("section_0_11"),
        chart_dim=np.asarray(CHART_DIM, dtype=np.int64),
        pair_order=init["pair_order"],
        init_artifact=init["init_artifact"],
        skill_override=np.asarray(
            "section_0_11_pad; NOT skill N%world_size==0"
        ),
    )
    return staging_bn, n_work, n_pad, slots, is_dummy, work_to_real


def _gather_rank_arrays(
    conn_tag: str,
    world_size: int,
    n_real: int,
    k: int,
    n_c: int,
    stride: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    n_iters: int,
    energy_pass: str,
) -> Dict[str, object]:
    """Gather length-n_real per-pair arrays from per-rank NPZs (real only)."""
    D = np.full(n_real, np.nan, dtype=np.float64)
    interiors = np.full((n_real, k - 1, CHART_DIM), np.nan, dtype=np.float64)
    endpoints = np.full((n_real, 2, CHART_DIM), np.nan, dtype=np.float64)
    E_init = np.full(n_real, np.nan, dtype=np.float64)
    E_best = np.full(n_real, np.nan, dtype=np.float64)
    E_last = np.full(n_real, np.nan, dtype=np.float64)
    pair_comfortable = np.zeros(n_real, dtype=bool)
    pair_exited_inset = np.zeros(n_real, dtype=bool)
    pair_exited_full_cube = np.zeros(n_real, dtype=bool)
    best_at_init = np.zeros(n_real, dtype=bool)

    n_clamp_sum: int = 0
    n_soft_sum: int = 0
    n_steps_soft_sum: int = 0
    filled: np.ndarray = np.zeros(n_real, dtype=bool)
    nested_paths: List[str] = []
    adamw_seconds_sum: float = 0.0

    for r in range(world_size):
        rank_name = phase5_rank_basename(
            r,
            conn_tag,
            n_c,
            stride,
            k,
            n_iters=int(n_iters),
            t_exp=t_exp,
            t_fat=t_fat,
            t_nlo=t_nlo,
            dual_init=dual_init,
            energy_pass=energy_pass,
        )
        rank_path = artifact_path(rank_name)
        if not os.path.isfile(rank_path):
            raise FileNotFoundError(f"Missing rank artifact: {rank_path}")
        rz = np.load(rank_path, allow_pickle=False)
        n_local: int = int(np.asarray(rz["n_real_local"]).item())
        ids: np.ndarray = np.asarray(rz["real_pair_ids"], dtype=np.int64)
        if ids.shape[0] != n_local:
            raise AssertionError(
                f"rank {r}: real_pair_ids len {ids.shape[0]} != n_real_local {n_local}"
            )
        # Task 40: vectorized scatter (was pure-Python for j in range(n_local)).
        if n_local > 0:
            if bool((ids < 0).any()) or bool((ids >= n_real).any()):
                raise AssertionError(f"rank {r}: real_pair_id out of range")
            if int(np.unique(ids).shape[0]) != int(ids.shape[0]):
                raise AssertionError(f"rank {r}: duplicate real_pair_ids within shard")
            if bool(filled[ids].any()):
                raise AssertionError(
                    f"rank {r}: overlaps previously gathered real_pair_ids"
                )
            d_src: np.ndarray = np.asarray(rz["D"], dtype=np.float64)
            interiors_src: np.ndarray = np.asarray(rz["interiors"], dtype=np.float64)
            endpoints_src: np.ndarray = np.asarray(rz["endpoints"], dtype=np.float64)
            e_init_src: np.ndarray = np.asarray(rz["E_init"], dtype=np.float64)
            e_best_src: np.ndarray = np.asarray(rz["E_best"], dtype=np.float64)
            e_last_src: np.ndarray = np.asarray(rz["E_last"], dtype=np.float64)
            comfortable_src: np.ndarray = np.asarray(
                rz["pair_comfortable"], dtype=bool
            )
            exited_inset_src: np.ndarray = np.asarray(
                rz["pair_exited_inset_pre_proj"], dtype=bool
            )
            exited_full_src: np.ndarray = np.asarray(
                rz["pair_exited_full_cube"], dtype=bool
            )
            best_at_init_src: np.ndarray = np.asarray(
                rz["best_at_init"], dtype=bool
            )
            # FAIL-closed vs NumPy length-1 fancy-index broadcast on corrupt shards.
            if d_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: D shape {d_src.shape} != ({n_local},)"
                )
            if interiors_src.shape != (n_local, k - 1, CHART_DIM):
                raise AssertionError(
                    f"rank {r}: interiors shape {interiors_src.shape} != "
                    f"({n_local}, {k - 1}, {CHART_DIM})"
                )
            if endpoints_src.shape != (n_local, 2, CHART_DIM):
                raise AssertionError(
                    f"rank {r}: endpoints shape {endpoints_src.shape} != "
                    f"({n_local}, 2, {CHART_DIM})"
                )
            if e_init_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: E_init shape {e_init_src.shape} != ({n_local},)"
                )
            if e_best_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: E_best shape {e_best_src.shape} != ({n_local},)"
                )
            if e_last_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: E_last shape {e_last_src.shape} != ({n_local},)"
                )
            if comfortable_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: pair_comfortable shape {comfortable_src.shape} "
                    f"!= ({n_local},)"
                )
            if exited_inset_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: pair_exited_inset_pre_proj shape "
                    f"{exited_inset_src.shape} != ({n_local},)"
                )
            if exited_full_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: pair_exited_full_cube shape "
                    f"{exited_full_src.shape} != ({n_local},)"
                )
            if best_at_init_src.shape != (n_local,):
                raise AssertionError(
                    f"rank {r}: best_at_init shape {best_at_init_src.shape} "
                    f"!= ({n_local},)"
                )
            D[ids] = d_src
            interiors[ids] = interiors_src
            endpoints[ids] = endpoints_src
            E_init[ids] = e_init_src
            E_best[ids] = e_best_src
            E_last[ids] = e_last_src
            pair_comfortable[ids] = comfortable_src
            pair_exited_inset[ids] = exited_inset_src
            pair_exited_full_cube[ids] = exited_full_src
            best_at_init[ids] = best_at_init_src
            filled[ids] = True
        n_clamp_sum += int(np.asarray(rz["n_clamp_to_decoder_box"]).item())
        n_soft_sum += int(np.asarray(rz["n_soft_node_excursions"]).item())
        n_steps_soft_sum += int(np.asarray(rz["n_steps_outside_6sigma"]).item())
        adamw_seconds_sum += float(np.asarray(rz["adamw_seconds"]).item())
        nested_paths.append(str(np.asarray(rz["nested_ad_path"]).item()))

    if not bool(filled.all()):
        missing = int(np.sum((~filled).astype(np.int64)))
        raise AssertionError(f"Gather incomplete: {missing} pairs unfilled")

    d_from_e: np.ndarray = np.sqrt(2.0 * E_best)
    max_abs_d_check: float = float(np.max(np.abs(D - d_from_e)))
    if max_abs_d_check > 1.0e-12:
        raise AssertionError(
            f"D vs sqrt(2 E★) mismatch after gather: max_abs={max_abs_d_check}"
        )

    return {
        "D": np.ascontiguousarray(D, dtype=np.float64),
        "interiors": np.ascontiguousarray(interiors, dtype=np.float64),
        "endpoints": np.ascontiguousarray(endpoints, dtype=np.float64),
        "E_init": np.ascontiguousarray(E_init, dtype=np.float64),
        "E_best": np.ascontiguousarray(E_best, dtype=np.float64),
        "E_last": np.ascontiguousarray(E_last, dtype=np.float64),
        "pair_comfortable": np.ascontiguousarray(pair_comfortable, dtype=bool),
        "pair_exited_inset": np.ascontiguousarray(pair_exited_inset, dtype=bool),
        "pair_exited_full_cube": np.ascontiguousarray(
            pair_exited_full_cube, dtype=bool
        ),
        "best_at_init": np.ascontiguousarray(best_at_init, dtype=bool),
        "n_clamp_to_decoder_box": int(n_clamp_sum),
        "n_soft_node_excursions": int(n_soft_sum),
        "n_steps_outside_6sigma": int(n_steps_soft_sum),
        "adamw_seconds_sum": float(adamw_seconds_sum),
        "nested_ad_paths": nested_paths,
        "max_abs_D_minus_sqrt_2E": float(max_abs_d_check),
    }


def _combine_min_d_arms(
    pass_a: Dict[str, object],
    pass_b: Optional[Dict[str, object]],
    energy_dual_idx: np.ndarray,
    n_dual: int,
) -> Dict[str, object]:
    """
    Winning D = min(D_dij_arm, D_chord_arm) on duals; tie → Dijkstra (arm=1).

    energy_dual_idx: energy-axis indices (0..P_energy-1), not catalog ids.
    """
    d_a: np.ndarray = np.asarray(pass_a["D"], dtype=np.float64)
    p: int = int(d_a.shape[0])

    D: np.ndarray = np.ascontiguousarray(d_a.copy(), dtype=np.float64)
    interiors: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["interiors"], dtype=np.float64).copy(), dtype=np.float64
    )
    endpoints: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["endpoints"], dtype=np.float64).copy(), dtype=np.float64
    )
    E_init: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["E_init"], dtype=np.float64).copy(), dtype=np.float64
    )
    E_best: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["E_best"], dtype=np.float64).copy(), dtype=np.float64
    )
    E_last: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["E_last"], dtype=np.float64).copy(), dtype=np.float64
    )
    pair_comfortable: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["pair_comfortable"], dtype=bool).copy(), dtype=bool
    )
    pair_exited_inset: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["pair_exited_inset"], dtype=bool).copy(), dtype=bool
    )
    pair_exited_full_cube: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["pair_exited_full_cube"], dtype=bool).copy(), dtype=bool
    )
    best_at_init: np.ndarray = np.ascontiguousarray(
        np.asarray(pass_a["best_at_init"], dtype=bool).copy(), dtype=bool
    )

    D_dij_arm: np.ndarray = np.ascontiguousarray(d_a, dtype=np.float64)
    D_chord_arm: np.ndarray = np.full(p, np.nan, dtype=np.float64)
    arm_choice: np.ndarray = np.full(p, ARM_NON_DUAL, dtype=np.int8)

    if pass_b is not None:
        if n_dual < 1:
            raise AssertionError(
                "pass_b provided with n_dual<1 — empty Pass B is forbidden"
            )
        d_b: np.ndarray = np.asarray(pass_b["D"], dtype=np.float64)
        if int(d_b.shape[0]) != n_dual:
            raise AssertionError(
                f"Pass B length {int(d_b.shape[0])} != n_dual={n_dual}"
            )
        eid: np.ndarray = np.ascontiguousarray(energy_dual_idx, dtype=np.int64)
        if int(eid.shape[0]) != n_dual:
            raise AssertionError(
                f"energy_dual_idx length {int(eid.shape[0])} != n_dual={n_dual}"
            )
        if bool(np.any(eid < 0)) or bool(np.any(eid >= np.int64(p))):
            raise AssertionError(
                f"energy_dual_idx out of range for P_energy={p}: "
                f"min={int(eid.min())}, max={int(eid.max())}"
            )

        D_chord_arm[eid] = d_b
        d_dij_dual: np.ndarray = D_dij_arm[eid]
        if not bool(np.isfinite(d_dij_dual).all()):
            raise AssertionError(
                "Non-finite D_dij_arm on dual rows after Pass A; "
                "cannot combine min-D arms"
            )
        if not bool(np.isfinite(d_b).all()):
            raise AssertionError(
                "Non-finite D_chord_arm (Pass B) on dual rows; "
                "cannot combine min-D arms"
            )
        # Strict < : tie (equality) → Dijkstra (arm=1).
        chord_wins: np.ndarray = np.ascontiguousarray(d_b < d_dij_dual, dtype=bool)

        arm_choice[eid] = ARM_DIJKSTRA
        if bool(chord_wins.any()):
            win_sparse: np.ndarray = np.flatnonzero(chord_wins)
            win_eid: np.ndarray = eid[win_sparse]
            arm_choice[win_eid] = ARM_CHORD
            D[win_eid] = d_b[win_sparse]
            interiors[win_eid] = np.asarray(pass_b["interiors"], dtype=np.float64)[
                win_sparse
            ]
            endpoints[win_eid] = np.asarray(pass_b["endpoints"], dtype=np.float64)[
                win_sparse
            ]
            E_init[win_eid] = np.asarray(pass_b["E_init"], dtype=np.float64)[
                win_sparse
            ]
            E_best[win_eid] = np.asarray(pass_b["E_best"], dtype=np.float64)[
                win_sparse
            ]
            E_last[win_eid] = np.asarray(pass_b["E_last"], dtype=np.float64)[
                win_sparse
            ]
            pair_comfortable[win_eid] = np.asarray(
                pass_b["pair_comfortable"], dtype=bool
            )[win_sparse]
            pair_exited_inset[win_eid] = np.asarray(
                pass_b["pair_exited_inset"], dtype=bool
            )[win_sparse]
            pair_exited_full_cube[win_eid] = np.asarray(
                pass_b["pair_exited_full_cube"], dtype=bool
            )[win_sparse]
            best_at_init[win_eid] = np.asarray(
                pass_b["best_at_init"], dtype=bool
            )[win_sparse]
    elif n_dual != 0:
        raise AssertionError(
            f"pass_b is None but n_dual={n_dual} — Pass B skip only when n_dual==0"
        )

    d_from_e: np.ndarray = np.sqrt(2.0 * E_best)
    max_abs_d_check: float = float(np.max(np.abs(D - d_from_e))) if p > 0 else 0.0
    if p > 0 and max_abs_d_check > 1.0e-12:
        raise AssertionError(
            f"Combined D vs sqrt(2 E★) mismatch: max_abs={max_abs_d_check}"
        )

    allowed: np.ndarray = np.array(
        [ARM_NON_DUAL, ARM_DIJKSTRA, ARM_CHORD], dtype=np.int8
    )
    if p > 0 and not bool(np.all(np.isin(arm_choice, allowed))):
        raise AssertionError("arm_choice contains values outside {0,1,2}")

    return {
        "D": np.ascontiguousarray(D, dtype=np.float64),
        "interiors": np.ascontiguousarray(interiors, dtype=np.float64),
        "endpoints": np.ascontiguousarray(endpoints, dtype=np.float64),
        "E_init": np.ascontiguousarray(E_init, dtype=np.float64),
        "E_best": np.ascontiguousarray(E_best, dtype=np.float64),
        "E_last": np.ascontiguousarray(E_last, dtype=np.float64),
        "pair_comfortable": np.ascontiguousarray(pair_comfortable, dtype=bool),
        "pair_exited_inset": np.ascontiguousarray(pair_exited_inset, dtype=bool),
        "pair_exited_full_cube": np.ascontiguousarray(
            pair_exited_full_cube, dtype=bool
        ),
        "best_at_init": np.ascontiguousarray(best_at_init, dtype=bool),
        "D_dij_arm": np.ascontiguousarray(D_dij_arm, dtype=np.float64),
        "D_chord_arm": np.ascontiguousarray(D_chord_arm, dtype=np.float64),
        "arm_choice": np.ascontiguousarray(arm_choice, dtype=np.int8),
        "max_abs_D_minus_sqrt_2E": float(max_abs_d_check),
    }


def _empty_pass_a(k: int) -> Dict[str, object]:
    """Empty Pass A result when P_energy==0 (skip spawn)."""
    return {
        "D": np.zeros(0, dtype=np.float64),
        "interiors": np.zeros((0, k - 1, CHART_DIM), dtype=np.float64),
        "endpoints": np.zeros((0, 2, CHART_DIM), dtype=np.float64),
        "E_init": np.zeros(0, dtype=np.float64),
        "E_best": np.zeros(0, dtype=np.float64),
        "E_last": np.zeros(0, dtype=np.float64),
        "pair_comfortable": np.zeros(0, dtype=bool),
        "pair_exited_inset": np.zeros(0, dtype=bool),
        "pair_exited_full_cube": np.zeros(0, dtype=bool),
        "best_at_init": np.zeros(0, dtype=bool),
        "n_clamp_to_decoder_box": 0,
        "n_soft_node_excursions": 0,
        "n_steps_outside_6sigma": 0,
        "adamw_seconds_sum": 0.0,
        "nested_ad_paths": [],
        "max_abs_D_minus_sqrt_2E": 0.0,
    }


def _empty_pass_b(k: int) -> Dict[str, object]:
    """Empty Pass B result when n_dual==0 (skip spawn)."""
    return _empty_pass_a(k)


def _arm_dict_from_npz(z: np.lib.npyio.NpzFile, k: int) -> Dict[str, object]:
    nested_raw = str(np.asarray(z["nested_ad_path"]).item())
    nested_paths: List[str] = (
        [p for p in nested_raw.split(",") if p] if nested_raw else []
    )
    d_arr = np.ascontiguousarray(z["D"], dtype=np.float64)
    n: int = int(d_arr.shape[0])
    interiors = np.ascontiguousarray(z["interiors"], dtype=np.float64)
    endpoints = np.ascontiguousarray(z["endpoints"], dtype=np.float64)
    if interiors.shape != (n, k - 1, CHART_DIM):
        raise AssertionError(
            f"arm interiors shape {interiors.shape} != ({n}, {k - 1}, {CHART_DIM})"
        )
    if endpoints.shape != (n, 2, CHART_DIM):
        raise AssertionError(
            f"arm endpoints shape {endpoints.shape} != ({n}, 2, {CHART_DIM})"
        )
    return {
        "D": d_arr,
        "interiors": interiors,
        "endpoints": endpoints,
        "E_init": np.ascontiguousarray(z["E_init"], dtype=np.float64),
        "E_best": np.ascontiguousarray(z["E_best"], dtype=np.float64),
        "E_last": np.ascontiguousarray(z["E_last"], dtype=np.float64),
        "pair_comfortable": np.ascontiguousarray(z["pair_comfortable"], dtype=bool),
        "pair_exited_inset": np.ascontiguousarray(z["pair_exited_inset"], dtype=bool),
        "pair_exited_full_cube": np.ascontiguousarray(
            z["pair_exited_full_cube"], dtype=bool
        ),
        "best_at_init": np.ascontiguousarray(z["best_at_init"], dtype=bool),
        "n_clamp_to_decoder_box": int(np.asarray(z["n_clamp_to_decoder_box"]).item()),
        "n_soft_node_excursions": int(np.asarray(z["n_soft_node_excursions"]).item()),
        "n_steps_outside_6sigma": int(np.asarray(z["n_steps_outside_6sigma"]).item()),
        "adamw_seconds_sum": float(np.asarray(z["adamw_seconds_sum"]).item()),
        "nested_ad_paths": nested_paths,
        "max_abs_D_minus_sqrt_2E": float(
            np.asarray(z["max_abs_D_minus_sqrt_2E"]).item()
        ),
    }


def _assert_arm_npz_knobs_match_cli(
    z: np.lib.npyio.NpzFile,
    *,
    label: str,
    conn_tag: str,
    n_c: int,
    stride: int,
    k: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
) -> None:
    """FAIL-closed: arm NPZ metadata must match merge CLI (not basename-only)."""
    mismatches: List[str] = []
    checks = (
        ("connectivity", str(np.asarray(z["connectivity"]).item()), str(conn_tag)),
        ("N_c", int(np.asarray(z["N_c"]).item()), int(n_c)),
        ("stride", int(np.asarray(z["stride"]).item()), int(stride)),
        ("K", int(np.asarray(z["K"]).item()), int(k)),
        ("n_iters", int(np.asarray(z["n_iters"]).item()), int(n_iters)),
        ("dual_init", str(np.asarray(z["dual_init"]).item()), str(dual_init)),
        ("max_pairs", int(np.asarray(z["max_pairs"]).item()), int(max_pairs)),
    )
    for name, got, expect in checks:
        if got != expect:
            mismatches.append(f"{name}: file={got!r} cli={expect!r}")
    t_exp_f = float(np.asarray(z["t_exp"]).item())
    t_fat_f = float(np.asarray(z["t_fat"]).item())
    t_nlo_f = float(np.asarray(z["t_nlo"]).item())
    if abs(t_exp_f - float(t_exp)) > 0.0:
        mismatches.append(f"t_exp: file={t_exp_f!r} cli={float(t_exp)!r}")
    if abs(t_fat_f - float(t_fat)) > 0.0:
        mismatches.append(f"t_fat: file={t_fat_f!r} cli={float(t_fat)!r}")
    if abs(t_nlo_f - float(t_nlo)) > 0.0:
        mismatches.append(f"t_nlo: file={t_nlo_f!r} cli={float(t_nlo)!r}")
    if mismatches:
        raise AssertionError(
            f"{label} arm NPZ knob mismatch vs merge CLI: "
            + "; ".join(mismatches)
        )


def _write_pass_a_arm_npz(
    conn_tag: str,
    pass_a: Dict[str, object],
    init: Dict[str, np.ndarray],
    *,
    n_c: int,
    stride: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
    world_size: int,
    lr: float,
    p_batch: int,
) -> str:
    p_energy: int = int(np.asarray(init["P_energy"]).item())
    k: int = int(np.asarray(init["K"]).item())
    bn = energy_pass_a_arm_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    path = artifact_path(bn)
    nested_paths: List[str] = list(pass_a["nested_ad_paths"])  # type: ignore[arg-type]
    np.savez_compressed(
        path,
        D=np.ascontiguousarray(pass_a["D"], dtype=np.float64),
        interiors=np.ascontiguousarray(pass_a["interiors"], dtype=np.float64),
        endpoints=np.ascontiguousarray(pass_a["endpoints"], dtype=np.float64),
        E_init=np.ascontiguousarray(pass_a["E_init"], dtype=np.float64),
        E_best=np.ascontiguousarray(pass_a["E_best"], dtype=np.float64),
        E_last=np.ascontiguousarray(pass_a["E_last"], dtype=np.float64),
        pair_comfortable=np.ascontiguousarray(
            pass_a["pair_comfortable"], dtype=bool
        ),
        pair_exited_inset=np.ascontiguousarray(
            pass_a["pair_exited_inset"], dtype=bool
        ),
        pair_exited_full_cube=np.ascontiguousarray(
            pass_a["pair_exited_full_cube"], dtype=bool
        ),
        best_at_init=np.ascontiguousarray(pass_a["best_at_init"], dtype=bool),
        n_clamp_to_decoder_box=np.asarray(
            int(pass_a["n_clamp_to_decoder_box"]), dtype=np.int64
        ),
        n_soft_node_excursions=np.asarray(
            int(pass_a["n_soft_node_excursions"]), dtype=np.int64
        ),
        n_steps_outside_6sigma=np.asarray(
            int(pass_a["n_steps_outside_6sigma"]), dtype=np.int64
        ),
        adamw_seconds_sum=np.asarray(
            float(pass_a["adamw_seconds_sum"]), dtype=np.float64
        ),
        nested_ad_path=np.asarray(",".join(sorted(set(nested_paths)))),
        max_abs_D_minus_sqrt_2E=np.asarray(
            float(pass_a["max_abs_D_minus_sqrt_2E"]), dtype=np.float64
        ),
        pair_index_energy=np.ascontiguousarray(
            init["pair_index_energy"], dtype=np.int64
        ),
        P_energy=np.asarray(p_energy, dtype=np.int64),
        K=np.asarray(k, dtype=np.int64),
        N_c=np.asarray(n_c, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        connectivity=np.asarray(conn_tag),
        t_exp=np.asarray(float(t_exp), dtype=np.float64),
        t_fat=np.asarray(float(t_fat), dtype=np.float64),
        t_nlo=np.asarray(float(t_nlo), dtype=np.float64),
        dual_init=np.asarray(str(dual_init)),
        n_iters=np.asarray(int(n_iters), dtype=np.int64),
        max_pairs=np.asarray(int(max_pairs), dtype=np.int64),
        world_size=np.asarray(int(world_size), dtype=np.int64),
        lr=np.asarray(float(lr), dtype=np.float64),
        p_batch=np.asarray(int(p_batch), dtype=np.int64),
        energy_pass=np.asarray("a"),
        chart_dim=np.asarray(CHART_DIM, dtype=np.int64),
    )
    print(f"[{conn_tag}nbr] wrote Pass A arm (intermediate): {path}", flush=True)
    return path


def _write_pass_b_arm_npz(
    conn_tag: str,
    pass_b: Dict[str, object],
    energy_dual_idx: np.ndarray,
    pair_index_dual: np.ndarray,
    *,
    n_c: int,
    stride: int,
    k: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
    n_dual: int,
    p_energy: int,
    world_size: int,
    lr: float,
    p_batch: int,
) -> str:
    bn = energy_pass_b_arm_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    path = artifact_path(bn)
    nested_paths: List[str] = list(pass_b["nested_ad_paths"])  # type: ignore[arg-type]
    np.savez_compressed(
        path,
        D=np.ascontiguousarray(pass_b["D"], dtype=np.float64),
        interiors=np.ascontiguousarray(pass_b["interiors"], dtype=np.float64),
        endpoints=np.ascontiguousarray(pass_b["endpoints"], dtype=np.float64),
        E_init=np.ascontiguousarray(pass_b["E_init"], dtype=np.float64),
        E_best=np.ascontiguousarray(pass_b["E_best"], dtype=np.float64),
        E_last=np.ascontiguousarray(pass_b["E_last"], dtype=np.float64),
        pair_comfortable=np.ascontiguousarray(
            pass_b["pair_comfortable"], dtype=bool
        ),
        pair_exited_inset=np.ascontiguousarray(
            pass_b["pair_exited_inset"], dtype=bool
        ),
        pair_exited_full_cube=np.ascontiguousarray(
            pass_b["pair_exited_full_cube"], dtype=bool
        ),
        best_at_init=np.ascontiguousarray(pass_b["best_at_init"], dtype=bool),
        n_clamp_to_decoder_box=np.asarray(
            int(pass_b["n_clamp_to_decoder_box"]), dtype=np.int64
        ),
        n_soft_node_excursions=np.asarray(
            int(pass_b["n_soft_node_excursions"]), dtype=np.int64
        ),
        n_steps_outside_6sigma=np.asarray(
            int(pass_b["n_steps_outside_6sigma"]), dtype=np.int64
        ),
        adamw_seconds_sum=np.asarray(
            float(pass_b["adamw_seconds_sum"]), dtype=np.float64
        ),
        nested_ad_path=np.asarray(",".join(sorted(set(nested_paths)))),
        max_abs_D_minus_sqrt_2E=np.asarray(
            float(pass_b["max_abs_D_minus_sqrt_2E"]), dtype=np.float64
        ),
        energy_dual_idx=np.ascontiguousarray(energy_dual_idx, dtype=np.int64),
        pair_index_dual=np.ascontiguousarray(pair_index_dual, dtype=np.int64),
        n_dual=np.asarray(int(n_dual), dtype=np.int64),
        P_energy=np.asarray(int(p_energy), dtype=np.int64),
        K=np.asarray(k, dtype=np.int64),
        N_c=np.asarray(n_c, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        connectivity=np.asarray(conn_tag),
        t_exp=np.asarray(float(t_exp), dtype=np.float64),
        t_fat=np.asarray(float(t_fat), dtype=np.float64),
        t_nlo=np.asarray(float(t_nlo), dtype=np.float64),
        dual_init=np.asarray(str(dual_init)),
        n_iters=np.asarray(int(n_iters), dtype=np.int64),
        max_pairs=np.asarray(int(max_pairs), dtype=np.int64),
        world_size=np.asarray(int(world_size), dtype=np.int64),
        lr=np.asarray(float(lr), dtype=np.float64),
        p_batch=np.asarray(int(p_batch), dtype=np.int64),
        energy_pass=np.asarray("b"),
        chart_dim=np.asarray(CHART_DIM, dtype=np.int64),
    )
    print(f"[{conn_tag}nbr] wrote Pass B arm (intermediate): {path}", flush=True)
    return path


def _adamw_meta_from_arm_npz(z: np.lib.npyio.NpzFile, *, label: str) -> Tuple[float, int, int]:
    """lr / p_batch / world_size from an arm NPZ (required for honest merge metadata)."""
    if "lr" not in z.files or "p_batch" not in z.files:
        raise AssertionError(
            f"{label} arm NPZ missing lr/p_batch — re-run the arm job with "
            "current run_task38_energy_campaign.py before merge"
        )
    if "world_size" not in z.files:
        raise AssertionError(f"{label} arm NPZ missing world_size")
    lr_v: float = float(np.asarray(z["lr"]).item())
    p_batch_v: int = int(np.asarray(z["p_batch"]).item())
    world_size_v: int = int(np.asarray(z["world_size"]).item())
    if not bool(np.isfinite(lr_v)) or lr_v <= 0.0:
        raise AssertionError(f"{label} arm lr must be finite and > 0 (got {lr_v})")
    if p_batch_v < 1:
        raise AssertionError(f"{label} arm p_batch must be >= 1 (got {p_batch_v})")
    if world_size_v < 1:
        raise AssertionError(
            f"{label} arm world_size must be >= 1 (got {world_size_v})"
        )
    return lr_v, p_batch_v, world_size_v


def _load_pass_a_arm_npz(
    conn_tag: str,
    *,
    n_c: int,
    stride: int,
    k: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
    p_energy: int,
    pair_index_energy: np.ndarray,
) -> Dict[str, object]:
    bn = energy_pass_a_arm_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    path = artifact_path(bn)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pass A arm missing for merge: {path}. Run --energy-pass a first."
        )
    z = np.load(path, allow_pickle=False)
    if str(np.asarray(z["energy_pass"]).item()) != "a":
        raise AssertionError(f"expected energy_pass=a in {bn}")
    _assert_arm_npz_knobs_match_cli(
        z,
        label="Pass A",
        conn_tag=conn_tag,
        n_c=n_c,
        stride=stride,
        k=k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    if int(np.asarray(z["P_energy"]).item()) != p_energy:
        raise AssertionError(
            f"Pass A arm P_energy={int(np.asarray(z['P_energy']).item())} "
            f"!= expected {p_energy}"
        )
    pe = np.ascontiguousarray(z["pair_index_energy"], dtype=np.int64)
    if not bool(np.array_equal(pe, pair_index_energy)):
        raise AssertionError("Pass A arm pair_index_energy mismatch vs init")
    out = _arm_dict_from_npz(z, k)
    lr_v, p_batch_v, world_size_v = _adamw_meta_from_arm_npz(z, label="Pass A")
    out["lr"] = lr_v
    out["p_batch"] = p_batch_v
    out["world_size"] = world_size_v
    if int(np.asarray(out["D"]).shape[0]) != p_energy:
        raise AssertionError(
            f"Pass A arm D length {int(np.asarray(out['D']).shape[0])} "
            f"!= P_energy={p_energy}"
        )
    print(f"[{conn_tag}nbr] loaded Pass A arm: {path}", flush=True)
    return out


def _load_pass_b_arm_npz(
    conn_tag: str,
    *,
    n_c: int,
    stride: int,
    k: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
    n_dual: int,
    p_energy: int,
    pair_index_dual_live: np.ndarray,
    pair_index_energy: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray]:
    bn = energy_pass_b_arm_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    path = artifact_path(bn)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pass B arm missing for merge: {path}. Run --energy-pass b first."
        )
    z = np.load(path, allow_pickle=False)
    if str(np.asarray(z["energy_pass"]).item()) != "b":
        raise AssertionError(f"expected energy_pass=b in {bn}")
    _assert_arm_npz_knobs_match_cli(
        z,
        label="Pass B",
        conn_tag=conn_tag,
        n_c=n_c,
        stride=stride,
        k=k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    n_dual_file: int = int(np.asarray(z["n_dual"]).item())
    if n_dual_file != n_dual:
        raise AssertionError(
            f"Pass B arm n_dual={n_dual_file} != expected {n_dual}"
        )
    if int(np.asarray(z["P_energy"]).item()) != p_energy:
        raise AssertionError("Pass B arm P_energy mismatch vs energy set")
    pid_file = np.ascontiguousarray(z["pair_index_dual"], dtype=np.int64)
    pid_live = np.ascontiguousarray(pair_index_dual_live, dtype=np.int64)
    if not bool(np.array_equal(pid_file, pid_live)):
        raise AssertionError(
            "Pass B arm pair_index_dual mismatch vs live dual mask "
            "(regenerated mask/split between arm jobs and merge?)"
        )
    energy_dual_idx_file = np.ascontiguousarray(
        z["energy_dual_idx"], dtype=np.int64
    )
    energy_dual_idx_live = _catalog_duals_to_energy_axis(
        pid_live, pair_index_energy
    )
    if not bool(np.array_equal(energy_dual_idx_file, energy_dual_idx_live)):
        raise AssertionError(
            "Pass B arm energy_dual_idx mismatch vs live "
            "_catalog_duals_to_energy_axis(mask, energy set)"
        )
    out = _arm_dict_from_npz(z, k)
    lr_v, p_batch_v, world_size_v = _adamw_meta_from_arm_npz(z, label="Pass B")
    out["lr"] = lr_v
    out["p_batch"] = p_batch_v
    out["world_size"] = world_size_v
    if int(np.asarray(out["D"]).shape[0]) != n_dual:
        raise AssertionError(
            f"Pass B arm D length {int(np.asarray(out['D']).shape[0])} "
            f"!= n_dual={n_dual}"
        )
    print(f"[{conn_tag}nbr] loaded Pass B arm: {path}", flush=True)
    return out, energy_dual_idx_live


def _pass_b_arm_path(
    conn_tag: str,
    *,
    n_c: int,
    stride: int,
    k: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
) -> str:
    bn = energy_pass_b_arm_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    return artifact_path(bn)


def _assert_no_stale_pass_b_when_n_dual_zero(
    conn_tag: str,
    *,
    n_c: int,
    stride: int,
    k: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
) -> None:
    """
    FAIL-closed: live n_dual=0 must not coexist with a non-empty Pass B arm.

    Empty B arm (n_dual=0 from ENERGY_PASS=b) is OK; leftover B with n_dual>0
    means the mask shrank after the B job — refuse merge.
    """
    path = _pass_b_arm_path(
        conn_tag,
        n_c=n_c,
        stride=stride,
        k=k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=max_pairs,
    )
    if not os.path.isfile(path):
        return
    z = np.load(path, allow_pickle=False)
    if "energy_pass" in z.files:
        ep_file: str = str(np.asarray(z["energy_pass"]).item())
        if ep_file != "b":
            raise AssertionError(
                f"merge live n_dual=0 but Pass B path has energy_pass={ep_file!r} "
                f"({path})"
            )
    if "n_dual" not in z.files:
        raise AssertionError(
            f"merge live n_dual=0 but Pass B arm missing n_dual ({path})"
        )
    n_dual_file: int = int(np.asarray(z["n_dual"]).item())
    if n_dual_file != 0:
        raise AssertionError(
            f"merge live n_dual=0 but Pass B arm has n_dual={n_dual_file} "
            f"({path}). Re-run ENERGY_PASS=b after the current dual mask, "
            "or remove the stale arm before merge."
        )
    # Corrupt empty-B: n_dual metadata 0 but payload nonempty.
    if "D" in z.files:
        n_d: int = int(np.asarray(z["D"]).shape[0])
        if n_d != 0:
            raise AssertionError(
                f"merge live n_dual=0 but Pass B arm D length={n_d} "
                f"(expected 0) ({path})"
            )
    if "pair_index_dual" in z.files:
        n_pid: int = int(np.asarray(z["pair_index_dual"]).shape[0])
        if n_pid != 0:
            raise AssertionError(
                f"merge live n_dual=0 but Pass B arm pair_index_dual "
                f"length={n_pid} (expected 0) ({path})"
            )
    if "energy_dual_idx" in z.files:
        n_edi: int = int(np.asarray(z["energy_dual_idx"]).shape[0])
        if n_edi != 0:
            raise AssertionError(
                f"merge live n_dual=0 but Pass B arm energy_dual_idx "
                f"length={n_edi} (expected 0) ({path})"
            )
    print(
        f"[{conn_tag}nbr] merge: live n_dual=0; found empty Pass B arm OK",
        flush=True,
    )


def _staging_energy_pass_for_spawn(energy_pass: str, arm: str) -> str:
    """
    Map job mode → staging tag.

    both → legacy untagged ('both'); a/b jobs → tagged 'a'/'b'.
    """
    ep: str = validate_energy_pass(energy_pass)
    if ep == "both":
        return "both"
    if ep == arm:
        return arm
    raise AssertionError(
        f"internal: cannot spawn arm={arm!r} under energy_pass={ep!r}"
    )


def _spawn_energy_pass(
    conn_tag: str,
    init: Dict[str, np.ndarray],
    world_size: int,
    lr: float,
    n_iters: int,
    p_batch: int,
    n_c: int,
    stride: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    pass_label: str,
    energy_pass: str,
) -> Dict[str, object]:
    """Write §0.11 staging → mp.spawn(run_chain) → gather length-N arrays."""
    n_real: int = int(np.asarray(init["P"]).item())
    k: int = int(np.asarray(init["K"]).item())
    if n_real < 1:
        raise ValueError(
            f"§0.11 empty-list forbidden ({pass_label}): N={n_real}. "
            "Skip Pass A when P_energy=0 / Pass B when n_dual=0."
        )

    staging_bn, n_work, n_pad, slots, is_dummy, _w2r = _write_staging(
        conn_tag,
        init,
        world_size,
        lr,
        n_iters,
        p_batch,
        n_c,
        stride,
        t_exp,
        t_fat,
        t_nlo,
        dual_init,
        energy_pass,
    )
    print(
        f"[{pass_label}] §0.11 pad: N={n_real}, N_work={n_work}, n_pad={n_pad}, "
        f"slots_per_gpu={slots} (assert n_pad < world_size OK; skill "
        f"N%world_size==0 OVERRIDDEN)",
        flush=True,
    )
    print(
        f"[{pass_label}] staging = {staging_bn} (energy_pass={energy_pass!r})",
        flush=True,
    )
    print(
        f"[{pass_label}] mp.spawn run_chain nprocs={world_size} "
        "(§0.11 pad; NOT skill N%world_size==0)",
        flush=True,
    )
    mp.spawn(
        run_chain,
        args=(
            world_size,
            BASE_SEED,
            float(lr),
            int(n_iters),
            int(p_batch),
            int(k),
            str(conn_tag),
            str(staging_bn),
            int(n_c),
            int(stride),
            float(t_exp),
            float(t_fat),
            float(t_nlo),
            str(dual_init),
        ),
        nprocs=world_size,
        join=True,
    )
    t_gather0: float = time.perf_counter()
    gathered = _gather_rank_arrays(
        conn_tag,
        world_size,
        n_real,
        k,
        n_c,
        stride,
        t_exp,
        t_fat,
        t_nlo,
        dual_init,
        n_iters,
        energy_pass,
    )
    gather_seconds: float = time.perf_counter() - t_gather0
    print(
        f"[{pass_label}] gather OK: N={n_real}, "
        f"gather_seconds={gather_seconds:.3f}, "
        f"max|D-sqrt(2E★)|={float(gathered['max_abs_D_minus_sqrt_2E']):.3e}",
        flush=True,
    )
    return gathered


def _write_energy_npz(
    conn_tag: str,
    init: Dict[str, np.ndarray],
    combined: Dict[str, object],
    pass_a: Dict[str, object],
    pass_b: Optional[Dict[str, object]],
    lr: float,
    n_iters: int,
    p_batch: int,
    phase_seconds: float,
    n_c: int,
    stride: int,
    grid_basename: str,
    world_size: int,
    n_dual: int,
    f_dual: float,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    mask_basename: str,
    chord_basename: str,
    max_pairs: int,
    p_energy_full: int,
) -> str:
    """Write energy-work NPZ (length P_energy)."""
    p_energy: int = int(np.asarray(init["P_energy"]).item())
    p_catalog: int = int(np.asarray(init["P"]).item())
    k: int = int(np.asarray(init["K"]).item())
    mp: int = int(max_pairs)
    if mp < 0:
        raise ValueError(f"max_pairs must be >= 0, got {mp}")
    if mp > 0 and p_energy != mp:
        raise AssertionError(
            f"smoke write: P_energy={p_energy} != max_pairs={mp}"
        )
    if mp == 0 and p_energy != p_energy_full:
        raise AssertionError(
            f"production write: P_energy={p_energy} != full={p_energy_full}"
        )

    D = np.asarray(combined["D"], dtype=np.float64)
    if int(D.shape[0]) != p_energy:
        raise AssertionError(
            f"combined D length {D.shape[0]} != P_energy={p_energy}"
        )
    pair_comfortable = np.asarray(combined["pair_comfortable"], dtype=bool)
    pair_exited_full_cube = np.asarray(combined["pair_exited_full_cube"], dtype=bool)
    best_at_init = np.asarray(combined["best_at_init"], dtype=bool)

    comfort_fraction: float = (
        float(np.sum(pair_comfortable.astype(np.int64))) / float(p_energy)
        if p_energy > 0
        else 1.0
    )
    best_at_init_frac: float = (
        float(np.sum(best_at_init.astype(np.int64))) / float(p_energy)
        if p_energy > 0
        else 0.0
    )
    n_full_cube: int = int(np.sum(pair_exited_full_cube.astype(np.int64)))

    n_clamp_sum: int = int(pass_a["n_clamp_to_decoder_box"])
    n_soft_sum: int = int(pass_a["n_soft_node_excursions"])
    n_steps_soft_sum: int = int(pass_a["n_steps_outside_6sigma"])
    adamw_seconds_sum: float = float(pass_a["adamw_seconds_sum"])
    nested_paths: List[str] = list(pass_a["nested_ad_paths"])  # type: ignore[arg-type]
    if pass_b is not None:
        n_clamp_sum += int(pass_b["n_clamp_to_decoder_box"])
        n_soft_sum += int(pass_b["n_soft_node_excursions"])
        n_steps_soft_sum += int(pass_b["n_steps_outside_6sigma"])
        adamw_seconds_sum += float(pass_b["adamw_seconds_sum"])
        nested_paths.extend(list(pass_b["nested_ad_paths"]))  # type: ignore[arg-type]

    max_abs_d_check: float = float(combined["max_abs_D_minus_sqrt_2E"])

    print(
        f"[{conn_tag}nbr] comfort_fraction={comfort_fraction:.6f} "
        f"(target report-only >= {float(COMFORT_TARGET)}; never automatic PASS)",
        flush=True,
    )
    if p_energy > 0 and comfort_fraction < float(COMFORT_TARGET):
        print(
            f"[{conn_tag}nbr] FLAG: comfort_fraction < {float(COMFORT_TARGET)} "
            f"— report-only; not a campaign FAIL here.",
            flush=True,
        )
    print(
        f"[{conn_tag}nbr] n_clamp_to_decoder_box={n_clamp_sum}, "
        f"n_soft_node_excursions={n_soft_sum}, "
        f"n_steps_outside_6sigma={n_steps_soft_sum}",
        flush=True,
    )
    if p_energy > 0:
        print(
            f"[{conn_tag}nbr] D: min={float(np.min(D)):.6e}, "
            f"median={float(np.median(D)):.6e}, max={float(np.max(D)):.6e}",
            flush=True,
        )
    arm_choice = np.asarray(combined["arm_choice"], dtype=np.int8)
    n_arm1: int = int(np.sum((arm_choice == ARM_DIJKSTRA).astype(np.int64)))
    n_arm2: int = int(np.sum((arm_choice == ARM_CHORD).astype(np.int64)))
    print(
        f"[{conn_tag}nbr] arm_choice: n_dual={n_dual}, "
        f"f_dual={f_dual:.6e}, n_arm1(dij/tie)={n_arm1}, n_arm2(chord)={n_arm2}",
        flush=True,
    )

    prod_name: str = energy_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=int(n_iters),
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
    )
    art_name: str = energy_output_basename(
        conn_tag,
        n_c,
        stride,
        k,
        mp,
        n_iters=int(n_iters),
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
    )
    art_path = artifact_path(art_name)
    if mp > 0:
        print(
            f"[{conn_tag}nbr] --max-pairs smoke write: production={prod_name}; "
            f"actual={art_name}",
            flush=True,
        )
        if art_name == prod_name:
            raise AssertionError(
                "smoke max_pairs>0 must not resolve to production energy_basename"
            )
    else:
        print(f"[{conn_tag}nbr] production energy write = {art_name}", flush=True)

    save_kw: Dict[str, object] = {
        "D": np.ascontiguousarray(D, dtype=np.float64),
        "interiors": np.ascontiguousarray(
            np.asarray(combined["interiors"], dtype=np.float64), dtype=np.float64
        ),
        "endpoints": np.ascontiguousarray(
            np.asarray(combined["endpoints"], dtype=np.float64), dtype=np.float64
        ),
        "E_init": np.ascontiguousarray(
            np.asarray(combined["E_init"], dtype=np.float64), dtype=np.float64
        ),
        "E_best": np.ascontiguousarray(
            np.asarray(combined["E_best"], dtype=np.float64), dtype=np.float64
        ),
        "E_last": np.ascontiguousarray(
            np.asarray(combined["E_last"], dtype=np.float64), dtype=np.float64
        ),
        "pair_comfortable": np.ascontiguousarray(pair_comfortable, dtype=bool),
        "pair_exited_inset_pre_proj": np.ascontiguousarray(
            np.asarray(combined["pair_exited_inset"], dtype=bool), dtype=bool
        ),
        "pair_exited_full_cube": np.ascontiguousarray(
            pair_exited_full_cube, dtype=bool
        ),
        "best_at_init": np.ascontiguousarray(best_at_init, dtype=bool),
        "D_dij_arm": np.ascontiguousarray(
            np.asarray(combined["D_dij_arm"], dtype=np.float64), dtype=np.float64
        ),
        "D_chord_arm": np.ascontiguousarray(
            np.asarray(combined["D_chord_arm"], dtype=np.float64), dtype=np.float64
        ),
        "arm_choice": np.ascontiguousarray(arm_choice, dtype=np.int8),
        "pair_index_energy": np.ascontiguousarray(
            init["pair_index_energy"], dtype=np.int64
        ),
        "P": np.asarray(p_catalog, dtype=np.int64),
        "P_energy": np.asarray(p_energy, dtype=np.int64),
        "P_energy_full": np.asarray(p_energy_full, dtype=np.int64),
        "max_pairs": np.asarray(mp, dtype=np.int64),
        "K": np.asarray(k, dtype=np.int64),
        "N_c": init["N_c"],
        "N_f": init["N_f"],
        "stride": np.asarray(stride, dtype=np.int64),
        "S": init["S"],
        "n_dual": np.asarray(n_dual, dtype=np.int64),
        "f_dual": np.asarray(float(f_dual), dtype=np.float64),
        "t_exp": np.asarray(float(t_exp), dtype=np.float64),
        "t_fat": np.asarray(float(t_fat), dtype=np.float64),
        "t_nlo": np.asarray(float(t_nlo), dtype=np.float64),
        "dual_init": np.asarray(str(dual_init)),
        "mask_artifact": np.asarray(mask_basename),
        "chord_artifact": np.asarray(chord_basename),
        "lr": np.asarray(float(lr), dtype=np.float64),
        "n_iters": np.asarray(int(n_iters), dtype=np.int64),
        "p_batch": np.asarray(int(p_batch), dtype=np.int64),
        "weight_decay": np.asarray(pec.WEIGHT_DECAY_FIXED, dtype=np.float64),
        "optimizer": np.asarray("AdamW"),
        "best_of_trajectory": np.asarray(True),
        "early_stop": np.asarray(False),
        "connectivity": np.asarray(conn_tag),
        "chart_dim": np.asarray(CHART_DIM, dtype=np.int64),
        "eps_dom": np.asarray(float(pec.EPS_DOM), dtype=np.float64),
        "decoder_lo": np.asarray(float(pec.DECODER_LO), dtype=np.float64),
        "decoder_hi": np.asarray(float(pec.DECODER_HI), dtype=np.float64),
        "soft_6sigma_e0_lo": np.asarray(
            float(pec.SOFT_6SIGMA_E0[0]), dtype=np.float64
        ),
        "soft_6sigma_e0_hi": np.asarray(
            float(pec.SOFT_6SIGMA_E0[1]), dtype=np.float64
        ),
        "soft_6sigma_e1_lo": np.asarray(
            float(pec.SOFT_6SIGMA_E1[0]), dtype=np.float64
        ),
        "soft_6sigma_e1_hi": np.asarray(
            float(pec.SOFT_6SIGMA_E1[1]), dtype=np.float64
        ),
        "soft_6sigma_e2_lo": np.asarray(
            float(pec.SOFT_6SIGMA_E2[0]), dtype=np.float64
        ),
        "soft_6sigma_e2_hi": np.asarray(
            float(pec.SOFT_6SIGMA_E2[1]), dtype=np.float64
        ),
        "comfort_fraction": np.asarray(comfort_fraction, dtype=np.float64),
        "comfort_target": np.asarray(float(COMFORT_TARGET), dtype=np.float64),
        "comfort_definition": np.asarray(
            "frac_real_energy_pairs_with_no_free_node_exiting_hard_inset_before_projection"
        ),
        "soft_6sigma_definition": np.asarray(
            "free_node_soft_6sigma_counts_before_hard_inset_projection_report_only"
        ),
        "comfort_is_pass_gate": np.asarray(False),
        "soft_6sigma_is_pass_gate": np.asarray(False),
        "n_clamp_to_decoder_box": np.asarray(n_clamp_sum, dtype=np.int64),
        "n_soft_node_excursions": np.asarray(n_soft_sum, dtype=np.int64),
        "n_steps_outside_6sigma": np.asarray(n_steps_soft_sum, dtype=np.int64),
        "n_pairs_exited_full_cube": np.asarray(n_full_cube, dtype=np.int64),
        "best_at_init_frac": np.asarray(best_at_init_frac, dtype=np.float64),
        "max_abs_D_minus_sqrt_2E": np.asarray(max_abs_d_check, dtype=np.float64),
        "nested_ad_path": np.asarray(",".join(sorted(set(nested_paths)))),
        "pair_order": init["pair_order"],
        "init_artifact": init["init_artifact"],
        "grid_artifact": np.asarray(grid_basename),
        "world_size": np.asarray(world_size, dtype=np.int64),
        "pad_contract": np.asarray("section_0_11"),
        "skill_override": np.asarray(
            "section_0_11_pad; NOT skill N%world_size==0"
        ),
        "phase9_seconds": np.asarray(float(phase_seconds), dtype=np.float64),
        "adamw_seconds_sum_ranks": np.asarray(
            float(adamw_seconds_sum), dtype=np.float64
        ),
        "endpoint_axis": np.asarray("axis1_eq_x_i_then_x_j"),
        "interiors_dtype": np.asarray("float64"),
        "endpoints_dtype": np.asarray("float64"),
        "D_formula": np.asarray("sqrt(2 * E_star_best_of_trajectory)"),
        "arm_choice_codes": np.asarray(
            "0=dijkstra_only_no_chord;1=dijkstra_won_or_tie;2=chord_won"
        ),
        "phase": np.asarray("9_energy_campaign_energy_set_only"),
        "implement_time_note": np.asarray(
            "energy_output_basename via task38_campaign_knobs "
            "(smoke → *_smokeP{N} when max_pairs>0); "
            "script=run_task38_energy_campaign.py; "
            "core=task38_path_energy_core.py; "
            "Pass A/B min-D dual init on energy axis only; "
            "catalog separate NPZ without interiors (P,K-1,3)"
        ),
    }
    if "D_dij" in init:
        save_kw["D_dij"] = np.ascontiguousarray(init["D_dij"], dtype=np.float64)
    if "L_graph" in init:
        save_kw["L_graph"] = np.ascontiguousarray(init["L_graph"], dtype=np.float64)
    if "m_ij" in init:
        save_kw["m_ij"] = np.ascontiguousarray(init["m_ij"], dtype=np.int64)

    np.savez_compressed(art_path, **save_kw)
    print(f"[{conn_tag}nbr] wrote energy NPZ {art_path}", flush=True)
    return art_path


def _write_catalog_npz(
    conn_tag: str,
    *,
    n_c: int,
    stride: int,
    k: int,
    n_iters: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    max_pairs: int,
    grid_bn: str,
    oracle_bn: str,
    split_bn: str,
    energy_art_path: str,
    init: Dict[str, np.ndarray],
    combined: Dict[str, object],
) -> str:
    """
    Catalog length-P: D, endpoints (P,2,3), pair_source — NO interiors.
    Analytic from early oracle; energy from combined scatter.
    """
    p_catalog: int = int(np.asarray(init["P"]).item())
    s: int = int(np.asarray(init["S"]).item())
    if p_catalog != s * (s - 1) // 2:
        raise AssertionError(
            f"P={p_catalog} != S(S-1)/2 with S={s}"
        )

    grid_path = artifact_path(grid_bn)
    oracle_path = artifact_path(oracle_bn)
    split_path = artifact_path(split_bn)
    gz = np.load(grid_path, allow_pickle=False)
    oz = np.load(oracle_path, allow_pickle=False)
    sz = np.load(split_path, allow_pickle=False)

    t_exp_split: np.float64 = np.float64(np.asarray(sz["t_exp"]).item())
    if t_exp_split != np.float64(t_exp):
        raise AssertionError(
            f"pair-split t_exp={float(t_exp_split):g} != CLI t_exp={float(t_exp):g}"
        )
    if int(np.asarray(sz["P"]).item()) != p_catalog:
        raise AssertionError("pair-split P != catalog P")
    if int(np.asarray(oz["P"]).item()) != p_catalog:
        raise AssertionError("oracle P != catalog P")

    coarse: np.ndarray = np.ascontiguousarray(
        gz["coarse_coords_flat"], dtype=np.float64
    )
    if coarse.shape != (s, CHART_DIM):
        raise AssertionError(
            f"coarse_coords_flat shape {coarse.shape} != (S,3)=({s},{CHART_DIM})"
        )

    ii, jj = np.triu_indices(s, k=1)
    if int(ii.shape[0]) != p_catalog:
        raise AssertionError("triu_indices length != P")

    endpoints_cat = np.empty((p_catalog, 2, CHART_DIM), dtype=np.float64)
    endpoints_cat[:, 0, :] = coarse[ii]
    endpoints_cat[:, 1, :] = coarse[jj]

    d_note02: np.ndarray = np.ascontiguousarray(
        oz["D_note02"], dtype=np.float64
    )
    if d_note02.shape != (p_catalog,):
        raise AssertionError("oracle D_note02 shape mismatch")

    analytic: np.ndarray = np.ascontiguousarray(sz["analytic"], dtype=bool)
    energy_mask: np.ndarray = np.ascontiguousarray(sz["energy"], dtype=bool)
    invalid: np.ndarray = np.ascontiguousarray(sz["invalid_or_tiny"], dtype=bool)

    D_cat = np.full(p_catalog, np.nan, dtype=np.float64)
    pair_source = np.full(p_catalog, PAIR_SOURCE_INVALID, dtype=object)

    # Analytic: D = D_note02; no AdamW.
    D_cat[analytic] = d_note02[analytic]
    pair_source[analytic] = PAIR_SOURCE_ANALYTIC

    # Energy: scatter winning D / endpoints from energy list.
    pe_idx: np.ndarray = np.ascontiguousarray(
        init["pair_index_energy"], dtype=np.int64
    )
    d_e: np.ndarray = np.asarray(combined["D"], dtype=np.float64)
    end_e: np.ndarray = np.asarray(combined["endpoints"], dtype=np.float64)
    p_energy: int = int(pe_idx.shape[0])
    if int(d_e.shape[0]) != p_energy:
        raise AssertionError("combined D length != pair_index_energy")
    if p_energy > 0:
        if not bool(np.all(energy_mask[pe_idx])):
            raise AssertionError(
                "pair_index_energy rows are not marked energy in split"
            )
        D_cat[pe_idx] = d_e
        endpoints_cat[pe_idx] = end_e
        pair_source[pe_idx] = PAIR_SOURCE_ENERGY

    # Sanity: invalid stays invalid (NaN D) unless somehow overlapping — fail.
    if bool(np.any(invalid & analytic)) or bool(np.any(invalid & energy_mask)):
        raise AssertionError("split masks overlap invalid with analytic/energy")

    pair_source_arr = np.asarray(pair_source, dtype=f"U{max(len(PAIR_SOURCE_INVALID), len(PAIR_SOURCE_ANALYTIC), len(PAIR_SOURCE_ENERGY))}")

    cat_bn = catalog_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=int(n_iters),
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=int(max_pairs),
    )
    cat_path = artifact_path(cat_bn)
    np.savez_compressed(
        cat_path,
        D=np.ascontiguousarray(D_cat, dtype=np.float64),
        endpoints=np.ascontiguousarray(endpoints_cat, dtype=np.float64),
        pair_source=np.ascontiguousarray(pair_source_arr),
        analytic=analytic,
        energy=energy_mask,
        invalid_or_tiny=invalid,
        pair_index_energy=pe_idx,
        P=np.asarray(p_catalog, dtype=np.int64),
        P_energy=np.asarray(p_energy, dtype=np.int64),
        P_analytic=np.asarray(int(np.sum(analytic.astype(np.int64))), dtype=np.int64),
        P_invalid=np.asarray(int(np.sum(invalid.astype(np.int64))), dtype=np.int64),
        K=np.asarray(k, dtype=np.int64),
        N_c=np.asarray(n_c, dtype=np.int64),
        N_f=np.asarray(n_f_from(n_c, stride), dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        S=np.asarray(s, dtype=np.int64),
        connectivity=np.asarray(conn_tag),
        chart_dim=np.asarray(CHART_DIM, dtype=np.int64),
        t_exp=np.asarray(float(t_exp), dtype=np.float64),
        t_fat=np.asarray(float(t_fat), dtype=np.float64),
        t_nlo=np.asarray(float(t_nlo), dtype=np.float64),
        dual_init=np.asarray(str(dual_init)),
        n_iters=np.asarray(int(n_iters), dtype=np.int64),
        grid_artifact=np.asarray(grid_bn),
        oracle_artifact=np.asarray(oracle_bn),
        pair_split_artifact=np.asarray(split_bn),
        energy_artifact=np.asarray(os.path.basename(energy_art_path)),
        pair_order=init["pair_order"],
        endpoint_axis=np.asarray("axis1_eq_x_i_then_x_j"),
        catalog_interiors_policy=np.asarray(
            "FORBIDDEN_as_production_(P,K-1,3)_section_5_6"
        ),
        pair_source_codes=np.asarray(
            "analytic | energy | invalid_or_tiny"
        ),
        phase=np.asarray("9_catalog_length_P_no_interiors"),
        implement_time_note=np.asarray(
            "catalog basename implement-time "
            "task38_catalog_D_endpoints_* with texp+niters+dual tags; "
            "analytic D=D_note02; energy D scattered from energy NPZ; "
            "no interiors (P,K-1,3)"
        ),
    )
    print(f"[{conn_tag}nbr] wrote catalog NPZ {cat_path}", flush=True)
    print(
        f"[{conn_tag}nbr] catalog: P={p_catalog}, "
        f"n_analytic={int(np.sum(analytic))}, "
        f"n_energy={p_energy}, n_invalid={int(np.sum(invalid))}",
        flush=True,
    )
    return cat_path


def run_one_connectivity(
    conn_tag: str,
    init_bn: str,
    world_size: int,
    lr: float,
    n_iters: int,
    p_batch: int,
    max_pairs: int,
    n_c: int,
    stride: int,
    k_cli: int,
    t_exp: np.float64,
    t_fat: np.float64,
    t_nlo: np.float64,
    dual_init: str,
    energy_pass: str,
) -> Tuple[str, str]:
    """
    energy_pass=both: Pass A → Pass B → combine → production energy+catalog
      (legacy one-job; untagged staging).
    energy_pass=a|b: one AdamW arm → intermediate arm NPZ (tagged staging).
    energy_pass=merge: load arms → combine → production (CPU; world_size unused).
    """
    ep: str = validate_energy_pass(energy_pass)
    print("=" * 80, flush=True)
    print(
        f"Task 38 Phase 9 energy campaign — {conn_tag}-neighbor "
        f"(energy_pass={ep!r})",
        flush=True,
    )
    print("=" * 80, flush=True)

    t0 = time.time()
    grid_bn: str = grids_basename(n_c, stride)
    oracle_bn: str = note02_oracle_basename(n_c, stride)
    split_bn: str = pair_split_basename(n_c, stride, t_exp=t_exp)

    init_full = _load_energy_init_artifact(
        init_bn,
        0,
        n_c_cli=n_c,
        stride_cli=stride,
        k_cli=k_cli,
        t_exp_cli=t_exp,
    )
    p_catalog: int = int(np.asarray(init_full["P"]).item())
    p_energy_full: int = int(np.asarray(init_full["P_energy"]).item())
    k: int = int(np.asarray(init_full["K"]).item())
    print(
        f"Loaded {init_bn}: P={p_catalog}, P_energy={p_energy_full}, K={k}, "
        f"N_c={n_c}, stride={stride}, chart_dim={CHART_DIM}",
        flush=True,
    )

    pair_index, chord_interiors, chord_endpoints, n_dual, mask_bn, chord_bn = (
        _load_mask_and_chord(
            conn_tag,
            n_c=n_c,
            stride=stride,
            k=k,
            p_catalog=p_catalog,
            p_energy=p_energy_full,
            t_exp_cli=t_exp,
            t_fat_cli=t_fat,
            t_nlo_cli=t_nlo,
            dual_init_cli=dual_init,
        )
    )
    print(
        f"Mask/chord OK: {mask_bn} / {chord_bn}; n_dual(full)={n_dual}",
        flush=True,
    )

    if max_pairs > 0:
        if max_pairs > p_energy_full:
            raise ValueError(
                f"--max-pairs={max_pairs} > P_energy={p_energy_full}"
            )
        init = _load_energy_init_artifact(
            init_bn,
            max_pairs,
            n_c_cli=n_c,
            stride_cli=stride,
            k_cli=k_cli,
            t_exp_cli=t_exp,
        )
        pair_index, chord_interiors, chord_endpoints, n_dual = (
            _filter_duals_for_energy_subset(
                pair_index,
                chord_interiors,
                chord_endpoints,
                init["pair_index_energy"],
            )
        )
        print(
            f"--max-pairs smoke: P_energy_eff={max_pairs}; "
            f"duals in truncated energy set → n_dual={n_dual}",
            flush=True,
        )
    else:
        init = init_full

    p_energy: int = int(np.asarray(init["P_energy"]).item())
    f_dual: float = (
        float(n_dual) / float(p_energy) if p_energy > 0 else 0.0
    )
    print(
        f"n_dual={n_dual}  (f_dual={f_dual:.6e} = n_dual/P_energy, "
        f"P_energy={p_energy})",
        flush=True,
    )

    # ---- merge: load intermediates, no AdamW ----
    if ep == "merge":
        pass_a = _load_pass_a_arm_npz(
            conn_tag,
            n_c=n_c,
            stride=stride,
            k=k,
            n_iters=int(n_iters),
            t_exp=t_exp,
            t_fat=t_fat,
            t_nlo=t_nlo,
            dual_init=dual_init,
            max_pairs=int(max_pairs),
            p_energy=p_energy,
            pair_index_energy=init["pair_index_energy"],
        )
        pass_b: Optional[Dict[str, object]] = None
        energy_dual_idx = np.zeros(0, dtype=np.int64)
        if n_dual == 0:
            print(
                f"[{conn_tag}nbr] merge: n_dual=0 → no Pass B arm required",
                flush=True,
            )
            _assert_no_stale_pass_b_when_n_dual_zero(
                conn_tag,
                n_c=n_c,
                stride=stride,
                k=k,
                n_iters=int(n_iters),
                t_exp=t_exp,
                t_fat=t_fat,
                t_nlo=t_nlo,
                dual_init=dual_init,
                max_pairs=int(max_pairs),
            )
        else:
            pass_b, energy_dual_idx = _load_pass_b_arm_npz(
                conn_tag,
                n_c=n_c,
                stride=stride,
                k=k,
                n_iters=int(n_iters),
                t_exp=t_exp,
                t_fat=t_fat,
                t_nlo=t_nlo,
                dual_init=dual_init,
                max_pairs=int(max_pairs),
                n_dual=n_dual,
                p_energy=p_energy,
                pair_index_dual_live=pair_index,
                pair_index_energy=init["pair_index_energy"],
            )
        if p_energy == 0:
            if n_dual != 0:
                raise AssertionError(
                    "merge: n_dual>0 with P_energy=0 is inconsistent "
                    "(duals ⊂ energy set)"
                )
            combined = _combine_min_d_arms(pass_a, None, energy_dual_idx, 0)
        else:
            combined = _combine_min_d_arms(
                pass_a, pass_b, energy_dual_idx, n_dual
            )
        # Honest AdamW metadata from arm job(s), not merge CLI defaults.
        lr_meta: float = float(pass_a["lr"])
        p_batch_meta: int = int(pass_a["p_batch"])
        ws_meta: int = int(pass_a["world_size"])
        if pass_b is not None:
            if float(pass_b["lr"]) != lr_meta:
                raise AssertionError(
                    f"Pass A/B arm lr mismatch: {lr_meta} vs {float(pass_b['lr'])}"
                )
            if int(pass_b["p_batch"]) != p_batch_meta:
                raise AssertionError(
                    f"Pass A/B arm p_batch mismatch: {p_batch_meta} vs "
                    f"{int(pass_b['p_batch'])}"
                )
            if int(pass_b["world_size"]) != ws_meta:
                raise AssertionError(
                    f"Pass A/B arm world_size mismatch: {ws_meta} vs "
                    f"{int(pass_b['world_size'])}"
                )
        print(
            f"[{conn_tag}nbr] merge AdamW meta from arms: "
            f"lr={lr_meta}, p_batch={p_batch_meta}, world_size={ws_meta}",
            flush=True,
        )
        t1 = time.time()
        energy_path = _write_energy_npz(
            conn_tag,
            init,
            combined,
            pass_a,
            pass_b,
            float(lr_meta),
            int(n_iters),
            int(p_batch_meta),
            float(t1 - t0),
            int(n_c),
            int(stride),
            grid_bn,
            int(ws_meta),
            n_dual,
            f_dual,
            t_exp,
            t_fat,
            t_nlo,
            dual_init,
            mask_bn,
            chord_bn,
            int(max_pairs),
            int(p_energy_full),
        )
        catalog_path = _write_catalog_npz(
            conn_tag,
            n_c=n_c,
            stride=stride,
            k=k,
            n_iters=int(n_iters),
            t_exp=t_exp,
            t_fat=t_fat,
            t_nlo=t_nlo,
            dual_init=dual_init,
            max_pairs=int(max_pairs),
            grid_bn=grid_bn,
            oracle_bn=oracle_bn,
            split_bn=split_bn,
            energy_art_path=energy_path,
            init=init,
            combined=combined,
        )
        return energy_path, catalog_path

    # ---- Pass A (modes a / both) ----
    pass_a = _empty_pass_a(k)
    if ep in ("a", "both"):
        if p_energy == 0:
            print(
                f"[{conn_tag}nbr] Pass A SKIPPED (P_energy=0): "
                "no empty-list §0.11",
                flush=True,
            )
            pass_a = _empty_pass_a(k)
        else:
            print(
                f"[{conn_tag}nbr] Pass A start: N=P_energy={p_energy}",
                flush=True,
            )
            pass_a_init: Dict[str, np.ndarray] = {
                "interiors": init["interiors"],
                "endpoints": init["endpoints"],
                "P": np.asarray(p_energy, dtype=np.int64),
                "K": init["K"],
                "pair_order": init["pair_order"],
                "init_artifact": init["init_artifact"],
            }
            pass_a = _spawn_energy_pass(
                conn_tag,
                pass_a_init,
                world_size,
                float(lr),
                int(n_iters),
                int(p_batch),
                int(n_c),
                int(stride),
                t_exp,
                t_fat,
                t_nlo,
                dual_init,
                pass_label=f"{conn_tag}nbr-PassA",
                energy_pass=_staging_energy_pass_for_spawn(ep, "a"),
            )
            print(
                f"[{conn_tag}nbr] Pass A finished; staging/rank free for Pass B",
                flush=True,
            )
        if ep == "a":
            arm_path = _write_pass_a_arm_npz(
                conn_tag,
                pass_a,
                init,
                n_c=n_c,
                stride=stride,
                n_iters=int(n_iters),
                t_exp=t_exp,
                t_fat=t_fat,
                t_nlo=t_nlo,
                dual_init=dual_init,
                max_pairs=int(max_pairs),
                world_size=int(world_size),
                lr=float(lr),
                p_batch=int(p_batch),
            )
            return arm_path, ""

    # ---- Pass B (modes b / both) ----
    pass_b = None
    energy_dual_idx = np.zeros(0, dtype=np.int64)
    if ep in ("b", "both"):
        if n_dual == 0:
            print(
                f"[{conn_tag}nbr] Pass B SKIPPED (n_dual=0): "
                "D_chord_arm=NaN, arm_choice=0; no empty-list §0.11",
                flush=True,
            )
            pass_b = None
            if ep == "b":
                arm_path_b = _write_pass_b_arm_npz(
                    conn_tag,
                    _empty_pass_b(k),
                    energy_dual_idx,
                    pair_index[:0],
                    n_c=n_c,
                    stride=stride,
                    k=k,
                    n_iters=int(n_iters),
                    t_exp=t_exp,
                    t_fat=t_fat,
                    t_nlo=t_nlo,
                    dual_init=dual_init,
                    max_pairs=int(max_pairs),
                    n_dual=0,
                    p_energy=p_energy,
                    world_size=int(world_size),
                    lr=float(lr),
                    p_batch=int(p_batch),
                )
                return arm_path_b, ""
        else:
            if p_energy == 0:
                raise AssertionError(
                    "n_dual>0 with P_energy=0 is inconsistent "
                    "(duals ⊂ energy set)"
                )
            energy_dual_idx = _catalog_duals_to_energy_axis(
                pair_index, init["pair_index_energy"]
            )
            print(
                f"[{conn_tag}nbr] Pass B start: N=n_dual={n_dual} "
                "(sparse row axis 0..n_dual-1; scatter via energy-axis indices; "
                "no global-id re-sort)",
                flush=True,
            )
            chord_init: Dict[str, np.ndarray] = {
                "interiors": np.ascontiguousarray(
                    chord_interiors, dtype=np.float64
                ),
                "endpoints": np.ascontiguousarray(
                    chord_endpoints, dtype=np.float64
                ),
                "P": np.asarray(n_dual, dtype=np.int64),
                "K": np.asarray(k, dtype=np.int64),
                "pair_order": init["pair_order"],
                "init_artifact": np.asarray(chord_bn),
            }
            pass_b = _spawn_energy_pass(
                conn_tag,
                chord_init,
                world_size,
                float(lr),
                int(n_iters),
                int(p_batch),
                int(n_c),
                int(stride),
                t_exp,
                t_fat,
                t_nlo,
                dual_init,
                pass_label=f"{conn_tag}nbr-PassB",
                energy_pass=_staging_energy_pass_for_spawn(ep, "b"),
            )
            print(f"[{conn_tag}nbr] Pass B finished", flush=True)
            if ep == "b":
                arm_path_b = _write_pass_b_arm_npz(
                    conn_tag,
                    pass_b,
                    energy_dual_idx,
                    pair_index,
                    n_c=n_c,
                    stride=stride,
                    k=k,
                    n_iters=int(n_iters),
                    t_exp=t_exp,
                    t_fat=t_fat,
                    t_nlo=t_nlo,
                    dual_init=dual_init,
                    max_pairs=int(max_pairs),
                    n_dual=n_dual,
                    p_energy=p_energy,
                    world_size=int(world_size),
                    lr=float(lr),
                    p_batch=int(p_batch),
                )
                return arm_path_b, ""

    # ---- both: combine + production write ----
    if ep != "both":
        raise AssertionError(f"unexpected fallthrough energy_pass={ep!r}")

    if p_energy == 0:
        combined = _combine_min_d_arms(pass_a, None, energy_dual_idx, 0)
    else:
        combined = _combine_min_d_arms(pass_a, pass_b, energy_dual_idx, n_dual)

    t1 = time.time()
    energy_path = _write_energy_npz(
        conn_tag,
        init,
        combined,
        pass_a,
        pass_b,
        float(lr),
        int(n_iters),
        int(p_batch),
        float(t1 - t0),
        int(n_c),
        int(stride),
        grid_bn,
        world_size,
        n_dual,
        f_dual,
        t_exp,
        t_fat,
        t_nlo,
        dual_init,
        mask_bn,
        chord_bn,
        int(max_pairs),
        int(p_energy_full),
    )
    catalog_path = _write_catalog_npz(
        conn_tag,
        n_c=n_c,
        stride=stride,
        k=k,
        n_iters=int(n_iters),
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
        max_pairs=int(max_pairs),
        grid_bn=grid_bn,
        oracle_bn=oracle_bn,
        split_bn=split_bn,
        energy_art_path=energy_path,
        init=init,
        combined=combined,
    )
    return energy_path, catalog_path


def main() -> None:
    args = _parse_args()
    start_time = datetime.now()
    tz_name = time.tzname[0] if time.tzname else "UTC"
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    tee_dir = campaign_tee_dir()

    out_name = (
        f'{script_name}_main_output_{start_time.strftime("%Y%m%d_%H%M%S")}_'
        f"{tz_name}.txt"
    )
    err_name = (
        f'{script_name}_main_errors_{start_time.strftime("%Y%m%d_%H%M%S")}_'
        f"{tz_name}.txt"
    )
    log_file_path = os.path.join(tee_dir, out_name)
    error_file_path = os.path.join(tee_dir, err_name)

    log_file_handle = open(log_file_path, "w", encoding="utf-8")
    error_file_handle = open(error_file_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = Tee(original_stdout, log_file_handle)
        sys.stderr = Tee(original_stderr, error_file_handle)

        print(f"🚀 Script: {script_name}.py")
        print(f"📅 Started: {start_time}")
        print(f"cwd / main_folder: {main_folder}")
        print(f"Tee dir (out_err_files/energy_campaign_3D_logs): {tee_dir}")
        print(
            "Task 38 Phase 9 — energy-set dual Pass A/B + min D + catalog "
            "(mp.spawn + §0.11)"
        )

        n_c: int = int(args.N_c)
        stride: int = int(args.stride)
        k_cli: int = int(args.K)
        n_f: int = validate_grid_knobs(n_c, stride)
        assert n_f == n_f_from(n_c, stride)
        if k_cli < 2:
            raise ValueError(f"--K must be >= 2 (got {k_cli})")
        if float(args.lr) <= 0.0:
            raise ValueError(f"--lr must be > 0 (got {args.lr})")
        if int(args.p_batch) < 1:
            raise ValueError(f"--p-batch must be >= 1 (got {args.p_batch})")
        if int(args.max_pairs) < 0:
            raise ValueError(f"--max-pairs must be >= 0 (got {args.max_pairs})")

        t_exp, t_fat, t_nlo, dual_init, n_iters, connectivity, energy_pass = (
            _resolve_knobs(args)
        )
        _warn_both_connectivity_all(energy_pass, connectivity)
        conns = _conn_list(connectivity)

        init_6: str = (
            str(args.init_6).strip()
            if str(args.init_6).strip()
            else inits_basename("6", n_c, stride, k_cli, t_exp=t_exp)
        )
        init_18: str = (
            str(args.init_18).strip()
            if str(args.init_18).strip()
            else inits_basename("18", n_c, stride, k_cli, t_exp=t_exp)
        )
        init_26: str = (
            str(args.init_26).strip()
            if str(args.init_26).strip()
            else inits_basename("26", n_c, stride, k_cli, t_exp=t_exp)
        )
        init_map = {"6": init_6, "18": init_18, "26": init_26}

        print(
            f"Config: connectivity={connectivity} → {conns}, "
            f"energy_pass={energy_pass!r}, n_iters={n_iters}, "
            f"max_pairs={args.max_pairs}, weight_decay={pec.WEIGHT_DECAY_FIXED} "
            f"(FIXED — not configurable)"
        )
        if energy_pass == "merge":
            print(
                "Config AdamW: lr/p_batch/world_size taken from arm NPZs "
                f"(merge CLI lr={args.lr}, p_batch={args.p_batch} ignored "
                "for production metadata)"
            )
        else:
            print(f"Config AdamW: lr={args.lr}, p_batch={args.p_batch}")
        print(f"Knobs: N_c={n_c}, stride={stride}, N_f={n_f}, K={k_cli}")
        print(
            f"Split/dual knobs: t_exp={float(t_exp):g}, "
            f"t_fat={float(t_fat):g}, t_nlo={float(t_nlo):g}, "
            f"dual_init={dual_init!r}"
        )
        print(
            "Smoke note: prefer --n-iters 5 for Nc=3 smoke; production default "
            f"in knobs remains DEFAULT_N_ITERS={DEFAULT_N_ITERS}."
        )
        print("Resolved basenames (task38_campaign_knobs):")
        print(f"  grid   = {grids_basename(n_c, stride)}")
        print(f"  oracle = {note02_oracle_basename(n_c, stride)}")
        print(f"  split  = {pair_split_basename(n_c, stride, t_exp=t_exp)}")
        print(f"  init6  = {init_6}")
        print(f"  init18 = {init_18}")
        print(f"  init26 = {init_26}")
        for c in conns:
            if energy_pass == "a":
                print(
                    f"  [{c}nbr] pass_a_arm (this job writes) = "
                    f"{energy_pass_a_arm_basename(c, n_c, stride, k_cli, n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init, max_pairs=int(args.max_pairs))}"
                )
                print(
                    f"  [{c}nbr] production energy/catalog: not written "
                    "(run ENERGY_PASS=b then merge)"
                )
            elif energy_pass == "b":
                print(
                    f"  [{c}nbr] pass_b_arm (this job writes) = "
                    f"{energy_pass_b_arm_basename(c, n_c, stride, k_cli, n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init, max_pairs=int(args.max_pairs))}"
                )
                print(
                    f"  [{c}nbr] production energy/catalog: not written "
                    "(run ENERGY_PASS=merge after a+b)"
                )
            elif energy_pass == "merge":
                print(
                    f"  [{c}nbr] pass_a_arm (load) = "
                    f"{energy_pass_a_arm_basename(c, n_c, stride, k_cli, n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init, max_pairs=int(args.max_pairs))}"
                )
                print(
                    f"  [{c}nbr] pass_b_arm (load if n_dual>0) = "
                    f"{energy_pass_b_arm_basename(c, n_c, stride, k_cli, n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init, max_pairs=int(args.max_pairs))}"
                )
                print(
                    f"  [{c}nbr] energy (this job writes) = "
                    f"{energy_output_basename(c, n_c, stride, k_cli, int(args.max_pairs), n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init)}"
                )
                print(
                    f"  [{c}nbr] catalog (this job writes) = "
                    f"{catalog_basename(c, n_c, stride, k_cli, n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init, max_pairs=int(args.max_pairs))}"
                )
            else:
                print(
                    f"  [{c}nbr] energy = "
                    f"{energy_output_basename(c, n_c, stride, k_cli, int(args.max_pairs), n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init)}"
                )
                print(
                    f"  [{c}nbr] catalog = "
                    f"{catalog_basename(c, n_c, stride, k_cli, n_iters=n_iters, t_exp=t_exp, t_fat=t_fat, t_nlo=t_nlo, dual_init=dual_init, max_pairs=int(args.max_pairs))}"
                )

        world_size: int
        if energy_pass == "merge":
            world_size = 0
            print(
                "energy_pass=merge: CPU-only combine (CUDA not required); "
                "production lr/p_batch/world_size come from arm NPZs",
                flush=True,
            )
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for Task 38 energy campaign")
            world_size = int(torch.cuda.device_count())
            if world_size < 1:
                raise RuntimeError("torch.cuda.device_count() < 1")
            print(f"world_size (CUDA devices): {world_size}")
            print(
                "§0.11 pad OVERRIDE: skill N%world_size==0 does NOT apply; "
                "N = P_energy (Pass A) or n_dual (Pass B)."
            )

        results: List[Tuple[str, str, str]] = []
        for c in conns:
            e_path, c_path = run_one_connectivity(
                c,
                init_map[c],
                world_size,
                float(args.lr),
                int(n_iters),
                int(args.p_batch),
                int(args.max_pairs),
                int(n_c),
                int(stride),
                int(k_cli),
                t_exp,
                t_fat,
                t_nlo,
                dual_init,
                energy_pass,
            )
            results.append((c, e_path, c_path))

        print("=" * 80)
        print(f"Phase 9 summary (energy_pass={energy_pass!r}):")
        for c, e_path, c_path in results:
            if energy_pass == "a":
                print(f"  [{c}nbr] pass_a_arm = {e_path}")
            elif energy_pass == "b":
                print(f"  [{c}nbr] pass_b_arm = {e_path}")
            else:
                print(f"  [{c}nbr] energy  = {e_path}")
                print(f"  [{c}nbr] catalog = {c_path}")
        if energy_pass in ("a", "b"):
            print(
                "Intermediate arm only — run the other arm then "
                "--energy-pass merge for production energy+catalog."
            )
        elif energy_pass == "merge":
            print("Ready for Phase 10 (late Note-02).")
        else:
            print(
                "Ready for Phase 10 (late Note-02) — not implemented in this wave."
            )
        print("=" * 80)

    except Exception as exc:
        print(f"\n💥 UNHANDLED EXCEPTION in main: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
        error_file_handle.close()
        print(f"✅ Logs saved to: {tee_dir}")


if __name__ == "__main__":
    main()
