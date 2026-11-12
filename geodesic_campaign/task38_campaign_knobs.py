"""
Task 38 campaign knobs + locked artifact basenames (§0.8–§0.11 / §0.10).

Copy-rename-edit of task34_campaign_knobs.py for the 3D soft-6σ campaign.
Do not edit the 2D knobs module in place.

Fine size (derived, not independent):
  N_f = s * (N_c - 1) + 1
  N_c must be odd (FAIL-closed if even). Not Task 34's N_f = N_c * s.

Defaults (production-oriented):
  N_c=21, s=2 → N_f=41; K=16; lr=1e-3; n_iters=100; P_batch=64;
  connectivity=all; dual-init=on; far-pair quantile=0.05.
  weight_decay is fixed 0 (not configurable).

t_exp / t_fat / t_nlo:
  TBD after design §6.0 — NO production numeric defaults here.
  Do NOT freeze Task 34's t_fat=1.05 or 2D t_nlo=1e-2 as 3D defaults.
  Expose as required config inputs; FAIL-closed validators only.
  CLI / env (when wired by later phases):
    --t-exp / T_EXP, --t-fat / T_FAT, --t-nlo / T_NLO,
    --dual-init / DUAL_INIT ({on,off}),
    --n-iters / N_ITERS, --lr / LR, --p-batch / P_BATCH,
    --far-pair-quantile / FAR_PAIR_QUANTILE,
    --connectivity / CONNECTIVITY ({6,18,26,all}),
    --energy-pass / ENERGY_PASS ({both,a,b,merge}; default both).

FAIL-closed:
  even N_c; t_exp<=0; t_fat<=1; t_nlo<=0; non-finite thresholds;
  dual-init=on with t_exp >= t_nlo (strict t_exp < t_nlo required);
  P_batch<=0; far-pair quantile outside (0, 1]; bad connectivity / dual-init.

Basename policy (§0.10):
  stem task38_; soft6sigma; {6|18|26}nbr; texp{slug}; dual_knob_tag =
  tfat…_tnlo…_dual{on|off}. Always tag even at defaults; no legacy untagged
  aliases.

  Untagged: grids, Stage A/B, early oracle.
  t_exp only: pair-split; energy-set Dijkstra resample/inits (TEXP-TAGGED —
    not Task 36's untagged inits rule).
  t_exp then dual_knob_tag (± niters where required): mask, chord, energy,
    late Note-02, report, pass_fail, phase5 staging/rank
    (historical "phase5" token = energy campaign, not scaffold Wave 5).

Float slug (_float_knob_slug): same encoding as Task 36
  (1.05 → 1p05, 1e-2 → 1em2).

Disk (§0.9):
  Artifacts → python_outputs/energy_campaign_3D_outputs/ (FLAT, task38_* stems)
    next to archive subdirs; NOT inside audit_and_smoke_outputs /
    early_task_38_outputs; NOT under energy_campaign_2D_outputs/.
  Tees → out_err_files/energy_campaign_3D_logs/ (FLAT, task38_* stems)
    NOT under energy_campaign_2D_logs/. Stage B worker logs use
    os.path.join(campaign_tee_dir(), "logs").

GPU pad reminder (§0.11):
  Stage A / Phase-1.5 oracle (N=P) / Pass A (N=P_energy) / Pass B (N=n_dual)
  use Task-34 §0.11 list-pad. The mp-spawn skill's N % world_size == 0
  assert does NOT apply.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from utils.workspace_folders import out_err_folder, python_outputs

DEFAULT_N_C: int = 21
DEFAULT_STRIDE: int = 2
DEFAULT_K: int = 16
DEFAULT_N_ITERS: int = 100
DEFAULT_LR: np.float64 = np.float64(1.0e-3)
DEFAULT_P_BATCH: int = 64
DEFAULT_CONNECTIVITY: str = "all"
DEFAULT_DUAL_INIT: str = "on"
DEFAULT_ENERGY_PASS: str = "both"
DEFAULT_FAR_PAIR_QUANTILE: np.float64 = np.float64(0.05)
ALLOWED_ENERGY_PASS: tuple[str, ...] = ("both", "a", "b", "merge")
# AdamW weight_decay is fixed 0 — not a campaign knob.
FIXED_WEIGHT_DECAY: np.float64 = np.float64(0.0)

# TBD after §6.0 — no frozen 3D production defaults (do not copy Task 34).
DEFAULT_T_EXP: Optional[np.float64] = None
DEFAULT_T_FAT: Optional[np.float64] = None
DEFAULT_T_NLO: Optional[np.float64] = None

ConnTag = Union[str, int]
ALLOWED_CONN_TAGS: tuple[str, ...] = ("6", "18", "26")
ALLOWED_CONNECTIVITY: tuple[str, ...] = ("6", "18", "26", "all")


def n_f_from(n_c: int, stride: int) -> int:
    """Fine lattice size: N_f = s * (N_c - 1) + 1 (Task 38 / §0.2.2)."""
    return int(stride) * (int(n_c) - 1) + 1


def validate_grid_knobs(n_c: int, stride: int) -> int:
    """
    Require positive ints, N_c odd, stride >= 1.
    Return N_f = s*(N_c-1)+1. Raises ValueError on invalid knobs.
    """
    if not isinstance(n_c, int) or isinstance(n_c, bool):
        raise ValueError(f"N_c must be a positive odd int (got {n_c!r})")
    if not isinstance(stride, int) or isinstance(stride, bool):
        raise ValueError(f"stride must be a positive int (got {stride!r})")
    if n_c < 1:
        raise ValueError(f"N_c must be >= 1 (got {n_c})")
    if n_c % 2 == 0:
        raise ValueError(f"N_c must be odd (got even {n_c})")
    if stride < 1:
        raise ValueError(f"stride must be >= 1 (got {stride})")
    return n_f_from(n_c, stride)


def validate_t_exp(t_exp: Union[float, np.floating]) -> np.float64:
    """
    FAIL-closed: t_exp must be finite and > 0.
    CLI --t-exp / env T_EXP (no production default until §6.0).
    """
    try:
        value: np.float64 = np.float64(t_exp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"t_exp must be a finite float > 0 (got {t_exp!r})") from exc
    if not bool(np.isfinite(value)):
        raise ValueError(f"t_exp must be finite (got {t_exp!r})")
    if not bool(value > np.float64(0.0)):
        raise ValueError(f"t_exp must be > 0 (got {value})")
    return value


def validate_t_fat(t_fat: Union[float, np.floating]) -> np.float64:
    """
    FAIL-closed: t_fat must be finite and > 1.
    CLI --t-fat / env T_FAT (no production default until §6.0).
    """
    try:
        value: np.float64 = np.float64(t_fat)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"t_fat must be a finite float > 1 (got {t_fat!r})") from exc
    if not bool(np.isfinite(value)):
        raise ValueError(f"t_fat must be finite (got {t_fat!r})")
    if not bool(value > np.float64(1.0)):
        raise ValueError(f"t_fat must be > 1 (got {value})")
    return value


def validate_t_nlo(t_nlo: Union[float, np.floating]) -> np.float64:
    """
    FAIL-closed: t_nlo must be finite and > 0.
    CLI --t-nlo / env T_NLO (no production default until §6.0).
    """
    try:
        value: np.float64 = np.float64(t_nlo)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"t_nlo must be a finite float > 0 (got {t_nlo!r})") from exc
    if not bool(np.isfinite(value)):
        raise ValueError(f"t_nlo must be finite (got {t_nlo!r})")
    if not bool(value > np.float64(0.0)):
        raise ValueError(f"t_nlo must be > 0 (got {value})")
    return value


def validate_dual_init(dual_init: str) -> str:
    """
    FAIL-closed: dual-init must be exactly 'on' or 'off'.
    CLI --dual-init / env DUAL_INIT (default DEFAULT_DUAL_INIT).
    """
    if not isinstance(dual_init, str):
        raise ValueError(
            f"dual-init must be 'on' or 'off' (got {dual_init!r})"
        )
    if dual_init not in ("on", "off"):
        raise ValueError(
            f"dual-init must be 'on' or 'off' (got {dual_init!r})"
        )
    return dual_init


def validate_dual_threshold_consistency(
    t_exp: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> None:
    """
    FAIL-closed: dual-init=on requires t_exp < t_nlo (strict).
    Equality or t_exp > t_nlo empties the dual window — abort.
    """
    t_exp_v: np.float64 = validate_t_exp(t_exp)
    t_nlo_v: np.float64 = validate_t_nlo(t_nlo)
    dual_v: str = validate_dual_init(dual_init)
    if dual_v == "on" and not bool(t_exp_v < t_nlo_v):
        raise ValueError(
            f"dual-init=on requires t_exp < t_nlo (strict); "
            f"got t_exp={t_exp_v}, t_nlo={t_nlo_v}"
        )


def validate_n_iters(n_iters: int) -> int:
    """
    FAIL-closed: n_iters must be a real int (not bool) and >= 1.
    CLI --n-iters / env N_ITERS (default DEFAULT_N_ITERS).
    """
    if not isinstance(n_iters, int) or isinstance(n_iters, bool):
        raise ValueError(f"n_iters must be a positive int (got {n_iters!r})")
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1 (got {n_iters})")
    return n_iters


def validate_lr(lr: Union[float, np.floating]) -> np.float64:
    """
    FAIL-closed: AdamW lr must be finite and > 0.
    CLI --lr / env LR (default DEFAULT_LR). weight_decay is fixed 0.
    """
    try:
        value: np.float64 = np.float64(lr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lr must be a finite float > 0 (got {lr!r})") from exc
    if not bool(np.isfinite(value)):
        raise ValueError(f"lr must be finite (got {lr!r})")
    if not bool(value > np.float64(0.0)):
        raise ValueError(f"lr must be > 0 (got {value})")
    return value


def validate_p_batch(p_batch: int) -> int:
    """
    FAIL-closed: P_batch must be a real int (not bool) and >= 1.
    CLI --p-batch / env P_BATCH (default DEFAULT_P_BATCH).
    """
    if not isinstance(p_batch, int) or isinstance(p_batch, bool):
        raise ValueError(f"P_batch must be a positive int (got {p_batch!r})")
    if p_batch < 1:
        raise ValueError(f"P_batch must be >= 1 (got {p_batch})")
    return p_batch


def validate_far_pair_quantile(
    q: Union[float, np.floating],
) -> np.float64:
    """
    FAIL-closed: far-pair quantile must be finite and in (0, 1].
    CLI --far-pair-quantile / env FAR_PAIR_QUANTILE
    (default DEFAULT_FAR_PAIR_QUANTILE = 0.05).
    """
    try:
        value: np.float64 = np.float64(q)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"far-pair quantile must be finite in (0, 1] (got {q!r})"
        ) from exc
    if not bool(np.isfinite(value)):
        raise ValueError(f"far-pair quantile must be finite (got {q!r})")
    if not bool(value > np.float64(0.0) and value <= np.float64(1.0)):
        raise ValueError(
            f"far-pair quantile must be in (0, 1] (got {value})"
        )
    return value


def validate_connectivity(connectivity: str) -> str:
    """
    FAIL-closed: connectivity in {6, 18, 26, all}.
    CLI --connectivity / env CONNECTIVITY (default DEFAULT_CONNECTIVITY).
    Strip + lower (same as validate_energy_pass).
    """
    if not isinstance(connectivity, str):
        raise ValueError(
            f"connectivity must be one of {ALLOWED_CONNECTIVITY} "
            f"(got {connectivity!r})"
        )
    c: str = str(connectivity).strip().lower()
    if c not in ALLOWED_CONNECTIVITY:
        raise ValueError(
            f"connectivity must be one of {ALLOWED_CONNECTIVITY} "
            f"(got {connectivity!r})"
        )
    return c


def validate_energy_pass(energy_pass: str) -> str:
    """
    FAIL-closed: energy_pass in {both, a, b, merge}.

    both = today's one-job A then B then write production (default).
    a / b = AdamW one arm only → intermediate arm NPZ (not production).
    merge = load arm NPZs → combine → production energy + catalog.
    CLI --energy-pass / env ENERGY_PASS.
    """
    if not isinstance(energy_pass, str):
        raise ValueError(
            f"energy_pass must be one of {ALLOWED_ENERGY_PASS} "
            f"(got {energy_pass!r})"
        )
    ep: str = str(energy_pass).strip().lower()
    if ep not in ALLOWED_ENERGY_PASS:
        raise ValueError(
            f"energy_pass must be one of {ALLOWED_ENERGY_PASS} (got {energy_pass!r})"
        )
    return ep


def _energy_pass_staging_suffix(energy_pass: str) -> str:
    """
    Staging/rank pass tag.

    both → '' (legacy untagged names — bit-compatible with pre-split jobs).
    a → '_passa'; b → '_passb'. merge never writes staging.
    """
    ep: str = validate_energy_pass(energy_pass)
    if ep == "both":
        return ""
    if ep == "a":
        return "_passa"
    if ep == "b":
        return "_passb"
    raise ValueError(
        f"energy_pass={ep!r} has no staging/rank basename (merge is CPU-only)"
    )


def _float_knob_slug(x: Union[float, np.floating]) -> str:
    """
    Deterministic ASCII slug for a campaign float knob (no dots).

    Same encoding as Task 36 dual_knob_tag / texp_tag.
    """
    v: np.float64 = np.float64(x)
    if not bool(np.isfinite(v)):
        raise ValueError(f"float knob slug requires finite value (got {x!r})")
    abs_v: float = float(np.abs(v))
    use_sci: bool = bool(abs_v > 0.0 and (abs_v < 0.1 or abs_v >= 100.0))
    if use_sci:
        sci: str = f"{float(v):.12e}"
        mant_str, exp_str = sci.split("e")
        mant_clean: str = mant_str.rstrip("0").rstrip(".")
        if mant_clean in ("", "+", "-", "+.", "-."):
            mant_clean = mant_str.split(".")[0]
        mant_slug: str = mant_clean.replace(".", "p").replace("-", "m")
        exp: int = int(exp_str)
        if exp < 0:
            return f"{mant_slug}em{abs(exp)}"
        if exp > 0:
            return f"{mant_slug}ep{exp}"
        return mant_slug
    dec: str = f"{float(v):.12f}".rstrip("0").rstrip(".")
    if dec in ("", "-", "+"):
        dec = f"{float(v):.0f}"
    return dec.replace(".", "p").replace("-", "m")


def texp_tag(t_exp: Union[float, np.floating]) -> str:
    """Basename segment 'texp{slug}' (validates t_exp)."""
    t_exp_v: np.float64 = validate_t_exp(t_exp)
    return f"texp{_float_knob_slug(t_exp_v)}"


def dual_knob_tag(
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """
    Basename segment for dual-dependent artifacts.

    Validates knobs via validate_t_fat / validate_t_nlo / validate_dual_init.
    Returns e.g. 'tfat1p05_tnlo1em2_dualon' (illustrative — not a 3D default).
    """
    t_fat_v: np.float64 = validate_t_fat(t_fat)
    t_nlo_v: np.float64 = validate_t_nlo(t_nlo)
    dual_v: str = validate_dual_init(dual_init)
    return (
        f"tfat{_float_knob_slug(t_fat_v)}"
        f"_tnlo{_float_knob_slug(t_nlo_v)}"
        f"_dual{dual_v}"
    )


def _conn_tag(conn_tag: ConnTag) -> str:
    tag: str = str(conn_tag)
    if tag not in ALLOWED_CONN_TAGS:
        raise ValueError(
            f"conn_tag must be one of {ALLOWED_CONN_TAGS} (got {conn_tag!r})"
        )
    return tag


def campaign_artifact_dir() -> str:
    """
    Flat 3D campaign artifact umbrella for task38_* stems.

    python_outputs/energy_campaign_3D_outputs/ — next to archive subdirs,
    not inside them, and not under energy_campaign_2D_outputs/.
    """
    path: str = os.path.join(python_outputs, "energy_campaign_3D_outputs")
    os.makedirs(path, exist_ok=True)
    return path


def campaign_tee_dir() -> str:
    """
    Flat 3D campaign Tee umbrella for task38_* stems.

    out_err_files/energy_campaign_3D_logs/ — not under
    energy_campaign_2D_logs/. Stage B workers use
    os.path.join(campaign_tee_dir(), "logs").
    """
    path: str = os.path.join(out_err_folder, "energy_campaign_3D_logs")
    os.makedirs(path, exist_ok=True)
    return path


def artifact_path(basename: str) -> str:
    """Join a locked §0.10 basename under the 3D campaign artifact umbrella."""
    return os.path.join(campaign_artifact_dir(), basename)


# ---------------------------------------------------------------------------
# Untagged basenames (no t_exp, no dual knobs)
# ---------------------------------------------------------------------------


def grids_basename(n_c: int, stride: int) -> str:
    return f"task38_chart_uniform_grids_soft6sigma_Nc{n_c}_s{stride}.npz"


def coo_basename(conn_tag: ConnTag, n_c: int, stride: int) -> str:
    tag: str = _conn_tag(conn_tag)
    return f"task38_stage_a_undirected_coo_{tag}nbr_Nc{n_c}_s{stride}.npz"


def dijkstra_basename(conn_tag: ConnTag, n_c: int, stride: int) -> str:
    tag: str = _conn_tag(conn_tag)
    return f"task38_stage_b_dijkstra_{tag}nbr_Nc{n_c}_s{stride}.npz"


def note02_oracle_basename(n_c: int, stride: int) -> str:
    """Early Phase-1.5 Note-02 oracle (no K, no connectivity; Nc/s only)."""
    return f"task38_note02_oracle_Nc{n_c}_s{stride}.npz"


def stage_a_staging_basename(n_c: int, stride: int) -> str:
    return f"task38_stage_a_work_staging_Nc{n_c}_s{stride}.npz"


def stage_a_rank_weights_basename(rank: int, n_c: int, stride: int) -> str:
    return f"task38_stage_a_rank{rank}_weights_Nc{n_c}_s{stride}.npz"


# ---------------------------------------------------------------------------
# Tagged with t_exp only
# ---------------------------------------------------------------------------


def pair_split_basename(
    n_c: int,
    stride: int,
    *,
    t_exp: Union[float, np.floating],
) -> str:
    """Oracle-first pair-split sidecar (Nc/s + texp)."""
    ttag: str = texp_tag(t_exp)
    return f"task38_pair_split_Nc{n_c}_s{stride}_{ttag}.npz"


def inits_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    t_exp: Union[float, np.floating],
) -> str:
    """
    Energy-set Dijkstra resample/inits — TEXP-TAGGED (§0.10).

    Unlike Task 36 untagged all-pairs inits, Task 38 energy-set inits always
    carry texp{slug}.
    """
    tag: str = _conn_tag(conn_tag)
    ttag: str = texp_tag(t_exp)
    return (
        f"task38_energy_inits_resampled_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}.npz"
    )


# ---------------------------------------------------------------------------
# Tagged with t_exp then dual_knob_tag (± niters)
# ---------------------------------------------------------------------------


def dual_init_mask_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    *,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """Dual-init mask sidecar (Nc/s + texp + dual_knob_tag; no niters)."""
    tag: str = _conn_tag(conn_tag)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_dual_init_mask_{tag}nbr_Nc{n_c}_s{stride}_{ttag}_{dtag}.npz"
    )


def chord_inits_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """Sparse chord inits (Nc/s/K + texp + dual_knob_tag; no niters)."""
    tag: str = _conn_tag(conn_tag)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_energy_inits_chord_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_{dtag}.npz"
    )


def energy_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """Production energy product (full P_energy). Late validation / report load this."""
    tag: str = _conn_tag(conn_tag)
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_energy_min_D_gamma_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}.npz"
    )


def energy_smoke_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    max_pairs: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """
    Smoke energy product when --max-pairs > 0.

    Distinct from energy_basename so a truncated run cannot overwrite production.
    Dual tag sits immediately before _smokeP{N}; texp then niters after K.
    """
    tag: str = _conn_tag(conn_tag)
    mp: int = int(max_pairs)
    if mp < 1:
        raise ValueError(
            f"energy_smoke_basename requires max_pairs >= 1, got {mp}"
        )
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_energy_min_D_gamma_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}_smokeP{mp}.npz"
    )


def energy_output_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    max_pairs: int = 0,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """Production basename if max_pairs==0; else smoke basename (FAIL-closed)."""
    mp: int = int(max_pairs)
    if mp > 0:
        return energy_smoke_basename(
            conn_tag,
            n_c,
            stride,
            k,
            mp,
            n_iters=n_iters,
            t_exp=t_exp,
            t_fat=t_fat,
            t_nlo=t_nlo,
            dual_init=dual_init,
        )
    if mp < 0:
        raise ValueError(f"max_pairs must be >= 0, got {mp}")
    return energy_basename(
        conn_tag,
        n_c,
        stride,
        k,
        n_iters=n_iters,
        t_exp=t_exp,
        t_fat=t_fat,
        t_nlo=t_nlo,
        dual_init=dual_init,
    )


def catalog_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
    max_pairs: int = 0,
) -> str:
    """
    Catalog D/endpoints NPZ (§5.6; same texp + niters + dual_knob_tag as energy).

    Production when max_pairs==0; smoke appends _smokeP{N} when max_pairs>0
    (Phase 9 truncated runs). Single helper for Phase 9 write + Phase 10 load.
    """
    tag: str = _conn_tag(conn_tag)
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    base: str = (
        f"task38_catalog_D_endpoints_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}"
    )
    mp: int = int(max_pairs)
    if mp > 0:
        return f"{base}_smokeP{mp}.npz"
    if mp < 0:
        raise ValueError(f"max_pairs must be >= 0, got {mp}")
    return f"{base}.npz"


def note02_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """Late Phase-6' validation NPZ (Nc/s/K + texp + niters + dual_knob_tag)."""
    tag: str = _conn_tag(conn_tag)
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_note02_validation_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}.npz"
    )


def note02_summary_basename(
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    """Late Phase-6' summary NPZ (Nc/s/K + texp + niters + dual_knob_tag)."""
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_note02_validation_summary_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}.npz"
    )


def report_txt_basename(
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_campaign_report_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}.txt"
    )


def pass_fail_basename(
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
) -> str:
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    return (
        f"task38_campaign_pass_fail_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}.npz"
    )


def phase5_staging_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
    energy_pass: str = DEFAULT_ENERGY_PASS,
) -> str:
    """
    Energy-campaign shared staging (historical phase5_* token from Task 34/36).

    Must match energy n_iters + texp + dual tagging. Not scaffold Wave 5 (Stage A).
    energy_pass=both → legacy untagged name; a/b append _passa/_passb.
    """
    tag: str = _conn_tag(conn_tag)
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    psuf: str = _energy_pass_staging_suffix(energy_pass)
    return (
        f"task38_phase5_work_staging_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}{psuf}.npz"
    )


def phase5_rank_basename(
    rank: int,
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
    energy_pass: str = DEFAULT_ENERGY_PASS,
) -> str:
    """Energy-campaign per-rank shard (historical phase5_* token)."""
    tag: str = _conn_tag(conn_tag)
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    psuf: str = _energy_pass_staging_suffix(energy_pass)
    return (
        f"task38_phase5_rank{rank}_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}{psuf}.npz"
    )


def energy_pass_a_arm_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
    max_pairs: int = 0,
) -> str:
    """
    Intermediate Pass A arm NPZ (Dijkstra AdamW on energy axis).

    Not a production energy product — merge (or both) writes that.
    """
    tag: str = _conn_tag(conn_tag)
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    base: str = (
        f"task38_energy_pass_a_arm_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}"
    )
    mp: int = int(max_pairs)
    if mp > 0:
        return f"{base}_smokeP{mp}.npz"
    if mp < 0:
        raise ValueError(f"max_pairs must be >= 0, got {mp}")
    return f"{base}.npz"


def energy_pass_b_arm_basename(
    conn_tag: ConnTag,
    n_c: int,
    stride: int,
    k: int,
    *,
    n_iters: int,
    t_exp: Union[float, np.floating],
    t_fat: Union[float, np.floating],
    t_nlo: Union[float, np.floating],
    dual_init: str,
    max_pairs: int = 0,
) -> str:
    """
    Intermediate Pass B arm NPZ (chord AdamW on sparse dual axis).

    Not a production energy product — merge (or both) writes that.
    """
    tag: str = _conn_tag(conn_tag)
    n_iters_v: int = validate_n_iters(n_iters)
    ttag: str = texp_tag(t_exp)
    dtag: str = dual_knob_tag(t_fat, t_nlo, dual_init)
    validate_dual_threshold_consistency(t_exp, t_nlo, dual_init)
    base: str = (
        f"task38_energy_pass_b_arm_{tag}nbr_Nc{n_c}_s{stride}_K{k}"
        f"_{ttag}_niters{n_iters_v}_{dtag}"
    )
    mp: int = int(max_pairs)
    if mp > 0:
        return f"{base}_smokeP{mp}.npz"
    if mp < 0:
        raise ValueError(f"max_pairs must be >= 0, got {mp}")
    return f"{base}.npz"
