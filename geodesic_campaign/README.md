# `geodesic_campaign/`

Geodesic distances between points in $\theta$-space, computed over the metric induced by
[`../cosmology_autoencoder/`](../cosmology_autoencoder/)'s forward decoder — plus a Ricci-scalar
curvature diagnostic on the same manifold. Everything here is already computed, shipped with its
logs; **nothing in this folder runs in CI** (see "Why no CI" below).

## Method

For two points $e_i,e_j$ in the latent box $e\in[-1,1]^3$ (the decoder's input, rescaled from
$\theta$ — see [`../cosmology_autoencoder/README.md`](../cosmology_autoencoder/README.md) for why
this $e$ is *not* that folder's autoencoder latent $e^\mu$), the decoder Jacobian induces a metric
$g$. A piecewise-linear path between $e_i$ and $e_j$ is optimized (AdamW) to minimize a
discretized path energy $\hat E$ built from $g$; the minimum defines the geodesic distance
$D(e_i,e_j)$ — but, as the next section explains, this energy-minimization step doesn't run for
most pairs.

(Filenames below carry a `task38` prefix — this project's own internal numbering for the script
family, not meaningful beyond that.)

- `task38_path_energy_core.py` ($\hat E$ + AdamW, far pairs only), `task38_note02_core.py` (the
  truncated chord ruler — the production distance for most pairs, see below).
- `task38_campaign_knobs.py`, `run_task38_energy_campaign.py` — the one pipeline script kept from
  the 10-stage campaign, run in three modes via `ENERGY_PASS=a|b|merge`.
- `submit.run_task38_energy_merge.sh` — the SLURM launcher that produced the three merge logs
  below (CPU-only merge pass).
- `compute_ricci_ad_planck_g.py` + `submit.compute_ricci_ad_planck_g.sh` — the Ricci-scalar
  diagnostic and its launcher (1 GPU, 30-minute limit).
- Logs: three energy-merge logs (26-, 18-, 6-neighbor connectivity), the Note-02 validation log,
  the campaign-report log, and the Ricci output log.
- `induced_metric.npy` — the $16\times16$ ambient metric $G$, Planck-best-fit-centered, built by
  the SVD procedure of [Bakx, Chisari & Vlah (2025), "COBRA: Optimal Factorization of
  Cosmological Observables," Phys. Rev. Lett. 134, 191002, arXiv:2407.04660] applied to a
  CAMB-generated training grid.
- `ricci_scalar.uniform_grid.npy` — the result array, $48^3$ grid.
- The 592-pair $K=32$ probe files (saved paths, already small).

## How distances are computed

$D_{\mathrm{note02}}$ (the subscript names the internal method note it's drawn from, not a
physical quantity) is not just a validation tool — it's the production distance for most
pairs, because running AdamW energy-minimization on all $\sim43$ million coarse-grid pairs would
never finish. `task38_note02_core.py` computes a truncated 4th-order expansion of the geodesic
distance along the Riemannian chord between two points,
$$D_{\mathrm{note02}} = L_c + \frac{Q_4^{\mathrm{mid}}}{2L_c},$$
dropping terms of order $\xi^5$ in the latent separation $\xi$ ($L_c$ = the straight chord's
length under $g$ at the midpoint). For **close pairs** — where this correction is small relative
to the leading term, $|\mathrm{NLO}/\mathrm{LO}|<10^{-3}$ — $D_{\mathrm{note02}}$ *is* the stored
distance; energy is never minimized for these at all. On the production catalog this is
$37{,}608{,}225$ of $42{,}878{,}430$ pairs ($87.7\%$).

Only the remaining $5{,}270{,}205$ pairs ($12.3\%$, the **far pairs**) get AdamW
energy-minimization — and only on this subset does $D_{\mathrm{note02}}$ play a genuine
cross-check role, compared against the independently-computed $D_{\mathrm{energy}}$. The frozen
pass gate is median relative difference $\le10^{-4}$, max $\le10^{-3}$; the 26-neighbor energy set
measured median $1.111\times10^{-5}$, worst pair $3.810\times10^{-4}$.

Separately, AdamW's own starting polyline for far-pair minimization is sometimes seeded from the
straight chord too (the "chord" arm of dual-init) — a cheap initial guess, kept only if it beats
a Dijkstra-informed start, not the same thing as the $D_{\mathrm{note02}}$ formula above — a
separate mechanism entirely.

**Not an independent ground truth.** Agreement between $D_{\mathrm{energy}}$ and
$D_{\mathrm{note02}}$ on the far-pair set means the optimizer landed near the truncated-chord
answer — it does not by itself prove either number equals the true geodesic distance.

## Results

**Energy campaign results ($N_c=21$):**

| Connectivity | Log | $D$ min | $D$ median | $D$ max |
|---|---|---|---|---|
| 26-neighbor | `task38_energy_merge_17252178.out` | $8.733390\times10^{-3}$ | $0.3388971$ | $0.6790661$ |
| 18-neighbor | `task38_energy_merge_17369549.out` | $8.733390\times10^{-3}$ | $0.3388978$ | $0.6790661$ |
| 6-neighbor | `task38_energy_merge_17369552.out` | same range to 5 figures | | |

**Ricci scalar, full 3D manifold, Planck-best-fit $G$, $48^3$ grid (110592 nodes), 167.8s on one
GPU:**
$$R\in[-649.2,\,445.2],\quad\text{mean}=-16.0,\quad\text{median}=-10.6.$$
Riemann antisymmetry/pair-swap residuals sit at the autodiff-noise level (median $\sim10^{-9}$,
max $\sim10^{-6}$) — a consistency check on the computation, not a physical result.

## Why no CI

Every result here is already computed, with its own out/err log as provenance. Re-running it
needs either a multi-hour CPU/GPU cluster job or (for the Ricci diagnostic) a CUDA device — not a
CI smoke test. The logs stand as the record of what ran.

## Pipeline context

The energy-merge stage is the last of a 10-stage pipeline (grid construction, the Note-02 oracle,
pair splitting, Stage A/B path initialization, dual-init masking, resampling); only this stage's
code and logs are shipped — the other nine only ever produce inputs to it.
