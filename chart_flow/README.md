# `chart_flow/`

A neural-ODE (RK4) experiment: can a learned coordinate change flatten an induced metric back to
Euclidean? Demonstrated on a synthetic flat-shear control — no decoder, no cosmology. The
cosmological version of this idea isn't finished yet, so it isn't in this repo; see "Why the
cosmological version isn't here" below.

## What's in this folder

(Filenames below carry a `task43` prefix — this project's own internal numbering for the script
family, not meaningful beyond that.)

- `run_task43_flat_chart_flow.py` — fixed learning rate ($10^{-3}$), hardcoded CPU, reduced here
  to `MAX_OPT_STEPS=20` (from 2000) as a smoke test. **The one script that runs in CI.** It
  exercises the same code path — the AdamW step, RK4 integration both directions, the
  induced-metric Jacobian, and the periodic closure check (fires at steps 10 and 20, since
  `HALVE_EVERY_STEPS=10`) — without trying to reproduce either historical result below.
- `run_task43_flat_chart_flow_plateau.py` — `ReduceLROnPlateau` variant, hardcoded to require
  CUDA, `MAX_OPT_STEPS=20000`. Shipped untouched, as historical, unrunnable provenance.

Only `run_task43_flat_chart_flow.py` — the live CI script — has `build_mlp` duplicated directly
into it, rather than imported from `utils.models`: that module imports `utils.svd_definitions` at
module level, which eagerly fits a `TruncatedSVD` on a large, excluded training grid the moment
it's imported, even though `build_mlp` itself doesn't need that grid. `run_task43_flat_chart_flow_plateau.py`
ships untouched — it still has the original `from utils.models import build_mlp` line exactly as
it stood on the cluster. That's harmless here: the plateau script requires CUDA and is never run
in this repo or in CI, so that import is never exercised.

## Historical results (no raw log survived for either run)

- Fixed-rate, step 2000 (100× more steps than what runs in CI):
  $\max\rho=3.3529351294675799\times10^{-3}$, stopped at the 2000-step cap.
- Plateau Run B, step 20000 (the better-converged of the two historical runs — quoted as the
  reference outcome, not something reproduced by anything shipped here): training-cloud mean
  Frobenius loss $2.6107056779516312\times10^{-4}$,
  $\max\rho=8.7768921331711134\times10^{-4}$; grid-average $\rho=1.507291625940259\times10^{-4}$;
  at the final closure check, $r=1.2644841412404203\times10^{-15}$,
  $R=6.1460329306719752\times10^{-1}$, $\max\rho=8.7759387847268953\times10^{-4}$.

$\rho$ is the residual measuring how far the learned coordinate change is from flattening the
metric; smaller is better.

## Why the cosmological version isn't here

An unfinished script, `run_cosmo_chart_flow.py`, tries a stronger version of this same idea on
the real decoder: find *one* coordinate change that flattens the induced metric to the identity
everywhere on the whole latent box $[-1,1]^3$, not just locally. That's not achievable once the
manifold has real curvature and the box is wide compared to the curvature scale. Riemann normal
coordinates can always flatten a metric exactly at a single point ($g=\delta$, $\partial g=0$
there), but a displacement $x$ away from that point, curvature comes back:
$$g_{ij}(x)=\delta_{ij}-\frac{1}{3}R_{ikjl}x^kx^l+\cdots,\qquad \lVert g-I\rVert\sim \frac{\lvert x\rvert^2}{\mathcal{R}^2},$$
with curvature radius $\mathcal{R}=1/\sqrt{|R|}$. At the box's centre, $\mathcal{R}\approx0.23$ —
far smaller than the box's own coordinate width of $2$. A single chart spanning the whole box
can't stay near-flat. Enforcing $g=I$ across all $129{,}024$ training points simultaneously also
made the run too slow to find out how close it could get: it stalled at step 2860
($\rho\approx0.81$), never finishing an epoch. Not a bug in the usual sense — an infeasible
target, too slow to even map out.

This is being reworked as an atlas instead of one global chart: a separate flow per centre
point, each only required to flatten a small neighborhood around its own centre, then walked
outward until the flattening residual exceeds a tolerance — that contour, mapped back into the
latent box, is one patch. Coverage is tested by checking whether the resulting patches' images
jointly tile the box. Not finished yet, which is why it isn't shipped here even as unrun
provenance.
