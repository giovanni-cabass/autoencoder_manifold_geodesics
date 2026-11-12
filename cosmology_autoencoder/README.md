# `cosmology_autoencoder/`

An autoencoder on the linear matter power spectrum $P(k)$'s SVD coefficients, trained with a
frozen forward decoder as a distillation teacher — the same decoder reused in
[`../geodesic_campaign/`](../geodesic_campaign/) to define distances on $\theta$-space.

## Notation

- $c=(x-\bar x)V^{\mathsf T}$ — the raw `TruncatedSVD(n_components=16, random_state=45)`
  coefficients of a spectrum, relative to the SVD-fit grid's mean.
- $d=c/\mathcal{N}$, with $\mathcal{N}=9.055999494024467467$ the unbiased standard deviation of
  the first raw coefficient on the 16,000-row fitting grid.
- $\theta=(\omega_c,\omega_b,n_s)$ — the physical cosmological parameters. The forward decoder
  shipped here, $\mathrm{MLP}_{\mathrm{fwd}}:\theta\to d$, maps directly from $\theta$, no
  rescaling.
- $e^\mu$ — the reconstruction autoencoder's own learned latent ($d\to e^\mu\to\hat d$), related
  to $\theta$ only through a separately-trained MLP ($e^\mu\to\theta$) — neither network's
  training code ships in this folder, only the forward decoder and the validation numbers below.
  **Not the same $e$ used in `../geodesic_campaign/`** — that $e\in[-1,1]^3$ is $\theta$ itself
  under a fixed, invertible affine rescaling. Same letter, unrelated objects — don't conflate
  them.
- **One checkpoint, two distillation roles.** The forward decoder is named `student_...`
  because it is itself the finished output of an earlier, upstream five-teacher distillation
  (`teacher_1`…`teacher_5` → one student → 10 refinement passes — none of that ladder is
  retrained here, only the finished checkpoints ship). Once trained, that same student
  checkpoint is reused *here* as the teacher for a second, separate autoencoder's own internal
  decoder, via `CustomLoss_with_distillation`. Same file, opposite role each time.

## What's in this folder

- `show_teacher_student_loss.py` — loads the student decoder plus all five teacher checkpoints,
  runs inference, reports the known validation numbers. The one script that runs in CI.
- `student_decoder_float64.no_maf.small_arch.no_bn.refine.10.pth` — the forward decoder
  checkpoint, sitting beside the script.
- `teachers/teacher_1/` through `teacher_5/` — the five teacher checkpoints
  (`decoder_float64.no_maf.small_arch.no_bn.refine.{8,5,3,3,2}.pth`), same architecture and
  training lineage as the student.
- `data/` — the small, already-derived $d$/$\theta$ arrays `show_teacher_student_loss.py` needs.

The reconstruction autoencoder and the $e^\mu\to\theta$ MLP themselves (run 0 a baseline, runs
1–5 each continuing training from the last) aren't shipped — only their validation numbers,
below.

## Results

- $\theta$-recovery: 99.7th-percentile error $\sim1.6$–$3.1\times10^{-4}$ across
  $\omega_c,\omega_b,n_s$ (run 5).
- Encoder reproducibility residual: $3.5\times10^{-13}$.

## A note on paper version

This repository's cosmology-appendix content reflects an updated version of the paper's
Appendix F (revised three-network design, tighter parameter-recovery numbers) that is not yet on
arXiv; the version currently at arXiv:2511.10466 describes an earlier, preliminary flow-based
attempt at the same appendix.

## Design history

The paper's own earlier attempt at $\theta$-recovery (Appendix F as currently on arXiv) used an
invertible flow, $\theta^\mu=f^\mu(e)$ — chosen specifically so the same trained map could run
backward, $e=f^{-1}(\theta)$, without training a second network. The flow itself is 24 Masked
Autoregressive Transforms (MAF), each built from monotonic affine univariate components, trained
with a plain MSE term comparing $\theta^\mu$ to $f^\mu(e)$ (not a flow likelihood).

At the time, neural ODEs weren't yet a known option for this project. A natural upgrade: swap the MAF transform
for a neural ODE on that same $e\leftrightarrow\theta$ map, nothing else about the architecture
changed. Both $e$ and $\theta$ are 3-dimensional, so it's a clean, dimension-matched
substitution — the same two-way invertibility (integrate the learned vector field forward or
backward in time), but with a far more expressive vector field than MAF's restrictive,
layer-wise monotonic-affine construction.

The three-separate-network design this project settled on instead (autoencoder
$d\to e^\mu\to\hat d$ + a separate MLP $e^\mu\to\theta$) took a different route: it drops the
invertibility requirement altogether — the MLP only runs $e^\mu\to\theta$, not back.
[`../chart_flow/`](../chart_flow/) is this project's neural-ODE work, built later, on a
different (synthetic, metric-flattening) problem.
