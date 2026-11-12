# autoencoder_manifold_geodesics

[![CI](https://github.com/giovanni-cabass/autoencoder_manifold_geodesics/actions/workflows/ci.yml/badge.svg)](https://github.com/giovanni-cabass/autoencoder_manifold_geodesics/actions/workflows/ci.yml)

Geodesic distances on the latent manifold induced by a trained decoder — one decoder, reused
three ways: first, as the teacher a separate autoencoder's decoder is distilled against; second,
as the map whose Jacobian induces the geodesic metric; third, as the target of a
metric-flattening experiment.

**Paper:** [arXiv:2511.10466](https://arxiv.org/abs/2511.10466), "Generating optimal
Gravitational-Wave template banks with metric-preserving autoencoders" (Cabass, Wadekar,
Zaldarriaga, Zhou) — companion repo
[`autoencoder_GWs`](https://github.com/giovanni-cabass/autoencoder_GWs).

## Development note

This repository's packaging (README, CI) was built with help from Claude Code. The research
behind it was built with help from Claude Code and Cursor.

## What this does (plain language)

A decoder network, $\mathrm{MLP}_{\mathrm{fwd}}:\theta\to d$, maps three cosmological parameters
$\theta=(\omega_c,\omega_b,n_s)$ to a compressed, 16-dimensional representation of the linear
matter power spectrum $P(k)$. Treating $\theta$-space as a 3D coordinate chart and the decoder as
an embedding, its Jacobian induces a Riemannian metric on that chart — distance between two
cosmologies isn't just Euclidean distance in $\theta$, it's distance measured through what the
decoder actually does to nearby points. This repo is three pieces of work around that one
decoder:

- **[`cosmology_autoencoder/`](cosmology_autoencoder/)** — where the decoder comes from: an
  autoencoder on $P(k)$'s SVD coefficients, trained with the decoder as a frozen distillation
  teacher (one checkpoint, two opposite distillation roles — see that folder's README), plus the
  MLP that maps the autoencoder's own latent back to $\theta$.
- **[`geodesic_campaign/`](geodesic_campaign/)** — using the decoder's induced metric to compute
  geodesic distances by minimizing a discretized path-energy functional, plus a Ricci-scalar
  curvature diagnostic on the full 3D manifold.
- **[`chart_flow/`](chart_flow/)** — a neural-ODE experiment asking whether a learned coordinate
  change can flatten an induced metric back to Euclidean, demonstrated on a synthetic flat-shear
  control (not yet the real cosmological decoder — see that folder's README for why).

Each folder has its own README with the numbers and logs, and what's live vs. historical.

## Running it

```bash
pip install -r requirements.txt

python cosmology_autoencoder/show_teacher_student_loss.py
python chart_flow/run_task43_flat_chart_flow.py
```

`geodesic_campaign/` ships only already-computed results and their logs — see its own README for
why nothing there is re-run.

## License

MIT, see `LICENSE`.
