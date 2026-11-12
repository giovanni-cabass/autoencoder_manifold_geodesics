"""
svd_definitions.py
==================
Centralised SVD pre-processing for the Plin_CAMB_LCDM_Wide dataset.

Module-level variables are computed once at import time with the default
number of retained components (N_COMP = 16, dims = 16).

For a different choice of ``dims`` call ``compute_svd_definitions(dims)``
directly; it returns a plain dict with all quantities.

Publicly exposed symbols (module level)
----------------------------------------
karr            : np.ndarray  (500,)                — wavenumber array [1/Mpc]
n_bins          : np.int64                          — number of wavenumber bins (== 500)
n_cosmos        : np.int64                          — number of training cosmologies
n_comp          : np.int64                          — number of SVD components fitted
dims            : np.int64                          — number of SVD components retained for downstream use
latent_dims     : np.int64                          — latent dimensionality of the flow/decoder (3)
train_spectra   : np.ndarray  (n_cosmos, 500)       — original training spectra (P_lin)
mean_plin       : np.ndarray  (500,)                — mean P_lin over training set
pnew_nml_mean   : np.ndarray  (500,)                — mean of the normalised spectra
vj_hat          : np.ndarray  (n_comp, 500)         — right singular vectors (SVD basis)
svs             : np.ndarray  (n_comp,)             — singular values
weights         : np.ndarray  (n_cosmos, n_comp)    — full projection weights
cosmoparams     : np.ndarray  (n_cosmos, 3)         — (Ω_c h², Ω_b h², n_s)
p_min_vals      : list[np.float64]                  — [ω_c_min, ω_b_min, n_s_min]
p_max_vals      : list[np.float64]                  — [ω_c_max, ω_b_max, n_s_max]
delta_omega_c   : np.float64
delta_omega_b   : np.float64
delta_ns        : np.float64
norm_factor     : torch.Tensor (scalar, float64)    — std of weights[:,0]
#### ALSO (not exported in final dictionary) ####
DIMS            : np.int64                          — number of SVD components retained (== N_COMP)
LATENT_DIMS     : np.int64                          — latent dimensionality of the flow/decoder (3)
"""

import os
import numpy as np  # type: ignore
import torch  # type: ignore
###################################################
torch.set_default_dtype(torch.float64)
###################################################
from sklearn.decomposition import TruncatedSVD  # type: ignore

from utils.workspace_folders import ML_data


# ---------------------------------------------------------------------------
# Fixed hyper-parameters for the SVD decomposition
# ---------------------------------------------------------------------------
N_COMP:       np.int64 = np.int64(16)   # total number of SVD components to fit
DIMS:         np.int64 = np.int64(16)   # number of components to retain for downstream use
LATENT_DIMS:  np.int64 = np.int64(3)    # latent dimensionality of the flow / decoder
N_ITER:       np.int64 = np.int64(10)
RANDOM_STATE: np.int64 = np.int64(45)

# Wavenumber grid (fixed by dataset convention)
karr: np.ndarray = np.logspace(np.log10(0.0008), np.log10(4), 500) ## wavenumbers at which the power spectra are computed
nbins: np.int64 = np.int64(len(karr))


# ---------------------------------------------------------------------------
# Core computation function
# ---------------------------------------------------------------------------

def compute_svd_definitions(dims: int = DIMS) -> dict:
    """
    Load the training dataset, run the TruncatedSVD and derive all
    downstream preprocessing quantities.

    Parameters
    ----------
    dims : int
        Number of SVD components to retain when computing ``norm_factor``.
        Must be <= N_COMP.

    Returns
    -------
    dict with keys:
        karr, n_cosmos, n_comp, mean_plin, pnew_nml_mean,
        vj_hat, svs, weights, cosmoparams,
        p_min_vals, p_max_vals,
        delta_omega_c, delta_omega_b, delta_ns,
        norm_factor
    """
    if dims != N_COMP:
        raise ValueError(
            f"dims={dims} must equal N_COMP={N_COMP}."
        )

    # ------------------------------------------------------------------
    # 1. Load training file
    # ------------------------------------------------------------------
    train_file: np.ndarray = np.loadtxt(
        os.path.join(ML_data, "Plin_CAMB_LCDM_Wide.dat")
    )

    # ------------------------------------------------------------------
    # 2. Normalise spectra and centre
    # ------------------------------------------------------------------
    pnew: np.ndarray = train_file[:, 3:]   # columns 0-2 are cosmo params
    n_cosmos: np.int64 = np.int64(pnew.shape[0])
    mean_plin: np.ndarray = np.mean(pnew, axis=0)

    pnew_nml: np.ndarray = pnew / mean_plin  # NOTE: divides by mean P_lin
    pnew_nml_mean: np.ndarray = np.mean(pnew_nml, axis=0)
    data_centered: np.ndarray = pnew_nml - pnew_nml_mean

    # ------------------------------------------------------------------
    # 3. TruncatedSVD
    # ------------------------------------------------------------------
    svd = TruncatedSVD(n_components=N_COMP, n_iter=N_ITER, random_state=RANDOM_STATE)
    svd.fit(data_centered)
    vj_hat: np.ndarray = svd.components_          # shape (N_COMP, 500)
    svs: np.ndarray = svd.singular_values_        # shape (N_COMP,)

    # Basis is already orthonormal → weights via direct projection
    weights: np.ndarray = data_centered @ vj_hat.T   # shape (n_cosmos, N_COMP)

    # ------------------------------------------------------------------
    # 4. Cosmological parameter bounds (used for flow training)
    # ------------------------------------------------------------------
    cosmoparams: np.ndarray = train_file[:, :3]   # (Ω_c h², Ω_b h², n_s)

    omega_c_min: np.float64 = np.min(cosmoparams[:, 0])
    omega_c_max: np.float64 = np.max(cosmoparams[:, 0])
    omega_b_min: np.float64 = np.min(cosmoparams[:, 1])
    omega_b_max: np.float64 = np.max(cosmoparams[:, 1])
    ns_min: np.float64      = np.min(cosmoparams[:, 2])
    ns_max: np.float64      = np.max(cosmoparams[:, 2])

    # Grid spacings (from test_flow.ipynb convention)
    delta_omega_c: np.float64 = (omega_c_max - omega_c_min) / np.float64(39)
    delta_omega_b: np.float64 = (omega_b_max - omega_b_min) / np.float64(19)
    delta_ns: np.float64      = (ns_max - ns_min) / np.float64(19)

    p_min_vals: list[np.float64] = [omega_c_min, omega_b_min, ns_min]
    p_max_vals: list[np.float64] = [omega_c_max, omega_b_max, ns_max]

    # ------------------------------------------------------------------
    # 5. Normalisation factor (std of first weight column)
    # ------------------------------------------------------------------
    data_torch: torch.Tensor = torch.tensor(
        weights[:, :dims], dtype=torch.float64
    )
    norm_factor: torch.Tensor = torch.std(data_torch[:, 0])

    # ------------------------------------------------------------------
    # 6. Sanity-check: norm_factor must match the reference file
    # ------------------------------------------------------------------
    ref_norm_factor: np.float64 = np.loadtxt(
        os.path.join(ML_data, "norm_factor_float64.txt")
    )
    assert torch.isclose(
        norm_factor,
        torch.tensor(ref_norm_factor, dtype=torch.float64),
        #rtol=1e-9,
        rtol=1e-14,
        atol=0.0,
    ), (
        f"norm_factor mismatch: computed {norm_factor.item():.16e} "
        f"!= reference {ref_norm_factor:.16e} "
        f"(from norm_factor_float64.txt)"
    )

    return {
        "karr":          karr,
        "nbins":         nbins,
        "n_cosmos":      n_cosmos,
        "n_comp":        N_COMP,
        "dims":          dims,
        "latent_dims":   LATENT_DIMS,
        "train_spectra": pnew,
        "mean_plin":     mean_plin,
        "pnew_nml_mean": pnew_nml_mean,
        "vj_hat":        vj_hat,
        "svs":           svs,
        "weights":       weights,
        "cosmoparams":   cosmoparams,
        "p_min_vals":    p_min_vals,
        "p_max_vals":    p_max_vals,
        "delta_omega_c": delta_omega_c,
        "delta_omega_b": delta_omega_b,
        "delta_ns":      delta_ns,
        "norm_factor":   norm_factor,
    }


# ---------------------------------------------------------------------------
# Module-level defaults (computed once at import with DIMS=16)
# ---------------------------------------------------------------------------
_defaults: dict = compute_svd_definitions(dims=DIMS)

karr:          np.ndarray     = _defaults["karr"]
nbins:         np.int64       = _defaults["nbins"]
n_cosmos:      np.int64       = _defaults["n_cosmos"]
n_comp:        np.int64       = _defaults["n_comp"]
dims:          np.int64       = _defaults["dims"]
latent_dims:   np.int64       = _defaults["latent_dims"]
train_spectra: np.ndarray     = _defaults["train_spectra"]
mean_plin:     np.ndarray     = _defaults["mean_plin"]
pnew_nml_mean: np.ndarray     = _defaults["pnew_nml_mean"]
vj_hat:        np.ndarray     = _defaults["vj_hat"]
svs:           np.ndarray     = _defaults["svs"]
weights:       np.ndarray     = _defaults["weights"]
cosmoparams:   np.ndarray     = _defaults["cosmoparams"]
p_min_vals:    list[np.float64] = _defaults["p_min_vals"]
p_max_vals:    list[np.float64] = _defaults["p_max_vals"]
delta_omega_c: np.float64     = _defaults["delta_omega_c"]
delta_omega_b: np.float64     = _defaults["delta_omega_b"]
delta_ns:      np.float64     = _defaults["delta_ns"]
norm_factor:   torch.Tensor   = _defaults["norm_factor"]
