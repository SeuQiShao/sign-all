#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
NODE_COUNT="${NODE_COUNT:-1000}"
TIME_POINTS="${TIME_POINTS:-1000}"
DATA_ROOT="${DATA_ROOT:-output/trajectories}"
RESULT_ROOT="${RESULT_ROOT:-output/fnfp}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FN_RATES="${FN_RATES:-0.01 0.1 0.15 0.2}"
FP_RATES="${FP_RATES:-0.01 0.1 0.15 0.2}"
GPU="${GPU:-0}"
WORKER_ID="${WORKER_ID:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
PREPARE_TRAJECTORIES="${PREPARE_TRAJECTORIES:-1}"
DEVICE="${DEVICE:-cpu}"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "${DATA_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

# Each case is an independent worker; cap BLAS/OpenMP fan-out to avoid
# exhausting the host thread quota when several cases run concurrently.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

if ! [[ "${WORKER_ID}" =~ ^[0-9]+$ && "${WORKER_COUNT}" =~ ^[1-9][0-9]*$ && "${WORKER_ID}" -lt "${WORKER_COUNT}" ]]; then
  echo "Invalid worker configuration: WORKER_ID=${WORKER_ID} WORKER_COUNT=${WORKER_COUNT}" >&2
  exit 2
fi

rate_code() {
  case "$1" in
    0.01) echo 1;; 0.1) echo 10;; 0.15) echo 15;; 0.2) echo 20;;
    *) echo "Unsupported corruption rate: $1" >&2; return 1;;
  esac
}

task_id=0
for seed in ${SEEDS}; do
  for system in Kuramoto FHN Rossler; do
    trajectory_parent="${DATA_ROOT}/seed${seed}"
    if [ "${PREPARE_TRAJECTORIES}" = "1" ]; then
      "${PYTHON_BIN}" scripts/prepare_trajectory.py --system "${system}" --seed "${seed}" \
        --output-root "${trajectory_parent}" --node-count "${NODE_COUNT}" --time-points "${TIME_POINTS}" --device "${DEVICE}"
    fi
    network="power_law"; [ "${system}" = "Kuramoto" ] || network="small_world"
    trajectory_root="${trajectory_parent}/${system}_${NODE_COUNT}_${network}_0.01_${TIME_POINTS}"
    true_edge_path="${trajectory_root}/true_edge_index.pt"
    for fn in ${FN_RATES}; do
      for fp in ${FP_RATES}; do
        assigned_id="${task_id}"
        task_id=$((task_id + 1))
        if [ $((assigned_id % WORKER_COUNT)) -ne "${WORKER_ID}" ]; then
          continue
        fi
        fn_tag="${fn/./p}"; fp_tag="${fp/./p}"
        case_root="${RESULT_ROOT}/${system}/seed${seed}/fn${fn_tag}_fp${fp_tag}"
        if [ -f "${case_root}/case_manifest.json" ]; then
          echo "SKIP ${system} seed=${seed} fn=${fn} fp=${fp}"
          continue
        fi
        fn_code="$(rate_code "${fn}")"; fp_code="$(rate_code "${fp}")"
        corruption_seed=$((1000000 + seed * 10000 + fn_code * 100 + fp_code))
        log_file="${LOG_ROOT}/${system}_seed${seed}_fn${fn_tag}_fp${fp_tag}.log"
        dims=3; [ "${system}" = "Kuramoto" ] && dims=1; [ "${system}" = "FHN" ] && dims=2
        if CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/run_robust_case.py \
          --trajectory-root "${trajectory_root}" --true-edge-path "${true_edge_path}" --case-root "${case_root}" \
          --system "${system}" --seed "${seed}" --fn-rate "${fn}" --fp-rate "${fp}" \
          --corruption-seed "${corruption_seed}" --ode_model "${system}" --network "${network}" \
          --num_atoms "${NODE_COUNT}" --dims "${dims}" --e1_library L1 --e1_basis_variant trig_exp_v2 \
          --e1_intercept_mode with --e1_intercept_aic_margin 5.0 --e1_ard_threshold_lambda 1e4 \
          --e1_ard_max_iter 2000 --e1_lasso_alpha 0.005 --e1_lasso_max_iter 2000 \
          --e1_dbscan_coef_relative_threshold 0 --e1_final_coef_relative_threshold 0 --e1_max_selected_terms 0 \
          --lasso_node_num 50 --lasso_neighbor_num 200 --batch_size 1 --teacher 5 --epochs 300 --lr 0.005 \
          --e2_sparse_warmup_epochs 30 --e2_min_epochs 30 --e2_loss_patience 30 --e2_support_patience 40 \
          --e2_abs_loss_stop 1e-8 --e2_abs_loss_min_epochs 8 --GPU_to_use 0 --no_validate \
          --no-generate_data_if_missing > "${log_file}" 2>&1; then
          echo "SUCCESS ${system} seed=${seed} fn=${fn} fp=${fp}"
        else
          echo "FAILED ${system} seed=${seed} fn=${fn} fp=${fp}" | tee -a "${log_file}"
          exit 1
        fi
      done
    done
  done
done
if [ "${RUN_AGGREGATE}" = "1" ]; then
  "${PYTHON_BIN}" scripts/aggregate_robust.py --result-root "${RESULT_ROOT}" > "${RESULT_ROOT}/aggregate.stdout.log" 2>&1
fi
echo "ROBUST_MATRIX_COMPLETE worker=${WORKER_ID}/${WORKER_COUNT}"
