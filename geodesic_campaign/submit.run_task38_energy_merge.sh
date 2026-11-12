#!/bin/bash -l
# Task 38 Phase 9 — CPU merge of Pass A/B arm NPZs → production energy+catalog.
# Implement-time name: submit.run_task38_energy_merge.sh
#
# Same Python entrypoint as the GPU energy submit, always --energy-pass merge.
# No CUDA. Prefer this over ENERGY_PASS=merge on submit.run_task38_energy_campaign.sh
# (that script still requests apollo 4 GPUs). AdamW arms stay on the GPU submit.
#
# Required: T_EXP T_FAT T_NLO; CONNECTIVITY in {6,18,26} (all forbidden).
# Optional: N_C K N_ITERS STRIDE DUAL_INIT LR P_BATCH MAX_PAIRS
#
# Production (after ENERGY_PASS=a and b SUCCESS on the same knobs/conn):
#   N_C=21 STRIDE=2 K=16 N_ITERS=100 T_EXP=1e-3 T_FAT=1.000000001 T_NLO=1e-2 \
#     DUAL_INIT=on CONNECTIVITY=26 \
#     sbatch python_scripts/submit.run_task38_energy_merge.sh
#
# Smoke:
#   N_C=3 STRIDE=2 K=16 N_ITERS=5 T_EXP=1e-3 T_FAT=1.01 T_NLO=1e-2 \
#     DUAL_INIT=on CONNECTIVITY=26 MAX_PAIRS=8 \
#     bash -l python_scripts/submit.run_task38_energy_merge.sh
#
#SBATCH --job-name=task38_energy_merge
#SBATCH --partition=typhon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=out_err_files/energy_campaign_3D_logs/task38_energy_merge_%j.out
#SBATCH --error=out_err_files/energy_campaign_3D_logs/task38_energy_merge_%j.err

set -eo pipefail

# Login shells (`bash -l`) source ~/.bashrc. A csh-ism there like `set filec`
# is interpreted by bash as setting $1=filec. Drop undashed strays, but keep
# values of --opt VAL.
_cli_args=()
_expect_val=0
for _a in "$@"; do
  if [[ ${_expect_val} -eq 1 ]]; then
    _cli_args+=("${_a}")
    _expect_val=0
    continue
  fi
  if [[ "${_a}" != -* ]]; then
    continue
  fi
  _cli_args+=("${_a}")
  case "${_a}" in
    --connectivity|--energy-pass|--max-pairs|--N_c|--stride|--K|--t-exp|--t-fat|--t-nlo|--dual-init|--lr|--n-iters|--p-batch|--init-6|--init-18|--init-26)
      _expect_val=1
      ;;
  esac
done
set -- "${_cli_args[@]}"
unset _cli_args _expect_val _a

# Under sbatch, SLURM runs a spool copy of this script — BASH_SOURCE is
# under /var/spool/slurmd/, not the repo. Prefer SLURM_SUBMIT_DIR (2D habit).
# Local bash: fall back to repo root via this file's location.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT}"
# Fail closed: relative python_scripts/... paths require the campaign workspace.
if [[ ! -d python_scripts || ! -f utils/workspace_folders.py ]]; then
  echo "ERROR: cwd is not the campaign workspace root." >&2
  echo "  cwd=${PWD}" >&2
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}" >&2
  echo "  BASH_SOURCE=${BASH_SOURCE[0]:-unset}" >&2
  echo "sbatch from the repo root (main_folder), e.g.:" >&2
  echo "  cd /path/to/curvature_calculation && sbatch python_scripts/<this_submit>.sh" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

# Env defaults (CLI args take precedence when forwarded as "$@")
N_C="${N_C:-21}"
STRIDE="${STRIDE:-2}"
K="${K:-16}"
N_ITERS="${N_ITERS:-100}"
LR="${LR:-1e-3}"
P_BATCH="${P_BATCH:-64}"
DUAL_INIT="${DUAL_INIT:-on}"
CONNECTIVITY="${CONNECTIVITY:-all}"
MAX_PAIRS="${MAX_PAIRS:-0}"
# This submit is merge-only; ignore ENERGY_PASS env (always merge).
ENERGY_PASS="merge"

if [[ -z "${T_EXP:-}" || -z "${T_FAT:-}" || -z "${T_NLO:-}" ]]; then
  echo "ERROR: T_EXP, T_FAT, and T_NLO must be set (no production defaults until §6.0)." >&2
  echo "Example smoke: T_EXP=1e-3 T_FAT=1.01 T_NLO=1e-2 N_C=3 CONNECTIVITY=26" >&2
  exit 1
fi

# CLI args override env for CONNECTIVITY (and detect forbidden --energy-pass).
if [[ $# -gt 0 ]]; then
  _args=("$@")
  _i=0
  while [[ ${_i} -lt ${#_args[@]} ]]; do
    case "${_args[_i]}" in
      --connectivity)
        _i=$((_i + 1))
        CONNECTIVITY="${_args[_i]}"
        ;;
      --connectivity=*)
        CONNECTIVITY="${_args[_i]#--connectivity=}"
        ;;
      --energy-pass)
        _i=$((_i + 1))
        _ep_cli="${_args[_i],,}"
        _ep_cli="${_ep_cli#"${_ep_cli%%[![:space:]]*}"}"
        _ep_cli="${_ep_cli%"${_ep_cli##*[![:space:]]}"}"
        if [[ "${_ep_cli}" != "merge" ]]; then
          echo "ERROR: this submit is merge-only (got --energy-pass ${_args[_i]})." >&2
          echo "Use submit.run_task38_energy_campaign.sh for ENERGY_PASS=a|b|both." >&2
          exit 1
        fi
        ;;
      --energy-pass=*)
        _ep_cli="${_args[_i]#--energy-pass=}"
        _ep_cli="${_ep_cli,,}"
        _ep_cli="${_ep_cli#"${_ep_cli%%[![:space:]]*}"}"
        _ep_cli="${_ep_cli%"${_ep_cli##*[![:space:]]}"}"
        if [[ "${_ep_cli}" != "merge" ]]; then
          echo "ERROR: this submit is merge-only (got ${_args[_i]})." >&2
          echo "Use submit.run_task38_energy_campaign.sh for ENERGY_PASS=a|b|both." >&2
          exit 1
        fi
        ;;
    esac
    _i=$((_i + 1))
  done
  unset _args _i _ep_cli
fi

CONNECTIVITY="${CONNECTIVITY,,}"
CONNECTIVITY="${CONNECTIVITY#"${CONNECTIVITY%%[![:space:]]*}"}"
CONNECTIVITY="${CONNECTIVITY%"${CONNECTIVITY##*[![:space:]]}"}"
if [[ "${CONNECTIVITY}" == "all" ]]; then
  echo "ERROR: ENERGY_PASS=merge forbids CONNECTIVITY=all." >&2
  echo "Use CONNECTIVITY=6|18|26 per job." >&2
  exit 1
fi
case "${CONNECTIVITY}" in
  6|18|26) ;;
  *)
    echo "ERROR: CONNECTIVITY must be 6|18|26 (got ${CONNECTIVITY})." >&2
    exit 1
    ;;
esac

echo "------------------------------------------------------------"
echo "Task 38 Phase 9 energy merge (CPU)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Date: $(date)"
echo "Nodes: 1  ntasks: 1  cpus-per-task: 1  (no GPUs)"
echo "Knobs: N_C=${N_C} STRIDE=${STRIDE} K=${K} N_ITERS=${N_ITERS}"
echo "       LR=${LR} P_BATCH=${P_BATCH} MAX_PAIRS=${MAX_PAIRS}"
echo "       T_EXP=${T_EXP} T_FAT=${T_FAT} T_NLO=${T_NLO} DUAL_INIT=${DUAL_INIT}"
echo "       CONNECTIVITY=${CONNECTIVITY} ENERGY_PASS=merge"
echo "       (production lr/p_batch/world_size come from arm NPZs)"
echo "CLI args: ${*:-<none; using env defaults>}"
echo "------------------------------------------------------------"

conda activate torch_cuda_pyro

if [[ $# -eq 0 ]]; then
  EXTRA=(
    --connectivity "${CONNECTIVITY}"
    --N_c "${N_C}"
    --stride "${STRIDE}"
    --K "${K}"
    --t-exp "${T_EXP}"
    --t-fat "${T_FAT}"
    --t-nlo "${T_NLO}"
    --dual-init "${DUAL_INIT}"
    --lr "${LR}"
    --n-iters "${N_ITERS}"
    --p-batch "${P_BATCH}"
    --energy-pass merge
  )
  if [[ "${MAX_PAIRS}" != "0" ]]; then
    EXTRA+=(--max-pairs "${MAX_PAIRS}")
  fi
  exec python -u \
    python_scripts/run_task38_energy_campaign.py \
    "${EXTRA[@]}"
else
  # Force merge even if CLI omitted --energy-pass.
  exec python -u \
    python_scripts/run_task38_energy_campaign.py \
    "$@" \
    --energy-pass merge
fi
