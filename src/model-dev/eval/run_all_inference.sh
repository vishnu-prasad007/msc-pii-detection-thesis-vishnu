#!/usr/bin/env bash
set -euo pipefail

# Run each (regime, checkpoint, dataset) in a fresh Python process.

BASE_MODEL="unsloth/Qwen2.5-VL-3B-Instruct"

# Regimes and their checkpoint roots
declare -A REGIME_ROOTS=(
  [regime_1]="/home/user/training/models_with_bbox"
  [regime_2]="/home/user/training/models_without_bbox"
)

# Datasets: name -> data_file|img_dir|output_prefix
declare -A DATASETS=(
  [val_ocr_cdip]="/home/user/training/ocr_idl-data-mixed-840px.json|/home/user/training/ocr_idl_images/images|ocr_cdip"
  [val_ad_buy]="/home/user/training/ad-buy-form-val-labels-consolidated.json|/home/user/training/ad_buy_images/images|ad_buy"
)

BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.0}
MIN_P=${MIN_P:-0.0}
SEED=${SEED:-64}
RESIZED_WIDTH=${RESIZED_WIDTH:-840}
RESIZED_HEIGHT=${RESIZED_HEIGHT:-840}

PYTHON=${PYTHON:-python3}

printf "run_all_inference.sh starting...\n"
printf "DRY_RUN=%s DEBUG=%s\n" "${DRY_RUN:-0}" "${DEBUG:-0}"

# Skip policy: if output dir already has results, skip the run
# - Set SKIP_IF_EXISTS=0 to disable skipping
# - Set MIN_OUTPUT_FILES to require a minimum number of JSON files to consider it done
# - Set FORCE=1 to force running even if outputs exist
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-1}
MIN_OUTPUT_FILES=${MIN_OUTPUT_FILES:-1}
FORCE=${FORCE:-0}

# Parallelism settings
# Number of GPUs/workers to use; override with NUM_GPUS env var
NUM_GPUS=${NUM_GPUS:-8}
# Dataset execution order: run OCR first, then AD-BUY
DATASET_EXEC_ORDER=(val_ocr_cdip val_ad_buy)

discover_checkpoints() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return
  fi
  # Find dirs named checkpoint-* that contain .bin or .safetensors
  while IFS= read -r -d '' d; do
    if compgen -G "$d/*.bin" > /dev/null || compgen -G "$d/*.safetensors" > /dev/null; then
      echo "$d"
    fi
  done < <(find "$root" -type d -name 'checkpoint-*' -print0 | sort -z)
}

# Compute expected output directory for a given (regime, checkpoint, dataset)
# Path shape: model_pred/<output_prefix>/<regime>/<experiment>/<checkpoint>
output_dir_for_ckpt() {
  local regime="$1"; shift
  local ckpt="$1"; shift
  local out_prefix="$1"; shift
  local experiment
  experiment="$(basename "$(dirname "$ckpt")")"
  local checkpoint_name
  checkpoint_name="$(basename "$ckpt")"
  printf "model_pred/%s/%s/%s/%s" "$out_prefix" "$regime" "$experiment" "$checkpoint_name"
}

# Build a list of jobs for a dataset (across all regimes and checkpoints)
build_jobs_for_dataset() {
  local dataset_name="$1"
  jobs=()
  if [[ -z "${DATASETS[$dataset_name]:-}" ]]; then
    echo "[WARN] Unknown dataset: $dataset_name"
    return
  fi
  for regime in "${!REGIME_ROOTS[@]}"; do
    local root="${REGIME_ROOTS[$regime]}"
    local checkpoints=()
    while IFS= read -r line; do
      [[ -n "$line" ]] && checkpoints+=("$line")
    done < <(discover_checkpoints "$root")
    printf "[INFO] %s: found %s checkpoints under %s\n" "$regime" "${#checkpoints[@]}" "$root"
    if (( ${#checkpoints[@]} == 0 )); then
      echo "[INFO] No checkpoints for $regime in $root; skipping."
      continue
    fi
    IFS='|' read -r data_file img_dir out_prefix <<< "${DATASETS[$dataset_name]}"
    for ckpt in "${checkpoints[@]}"; do
      jobs+=("$regime|$ckpt|$dataset_name|$data_file|$img_dir|$out_prefix")
    done
  done
}

# Build a combined list of jobs across datasets in priority order (spillover enabled)
build_jobs_for_all_datasets_in_order() {
  jobs=()
  for dataset_name in "${DATASET_EXEC_ORDER[@]}"; do
    if [[ -z "${DATASETS[$dataset_name]:-}" ]]; then
      echo "[WARN] Unknown dataset in order list: $dataset_name"
      continue
    fi
    for regime in "${!REGIME_ROOTS[@]}"; do
      local root="${REGIME_ROOTS[$regime]}"
      local checkpoints=()
      while IFS= read -r line; do
        [[ -n "$line" ]] && checkpoints+=("$line")
      done < <(discover_checkpoints "$root")
      printf "[INFO] %s: found %s checkpoints under %s (for %s)\n" "$regime" "${#checkpoints[@]}" "$root" "$dataset_name"
      if (( ${#checkpoints[@]} == 0 )); then
        echo "[INFO] No checkpoints for $regime in $root; skipping."
        continue
      fi
      IFS='|' read -r data_file img_dir out_prefix <<< "${DATASETS[$dataset_name]}"
      for ckpt in "${checkpoints[@]}"; do
        jobs+=("$regime|$ckpt|$dataset_name|$data_file|$img_dir|$out_prefix")
      done
    done
  done
}

# Run all jobs for a dataset in parallel across GPUs using strided workers
run_dataset_in_parallel() {
  local dataset_name="$1"
  printf "\n================ DATASET: %s (NUM_GPUS=%s) ================\n" "$dataset_name" "$NUM_GPUS"
  build_jobs_for_dataset "$dataset_name"
  local num_jobs=${#jobs[@]}
  printf "[INFO] Prepared %s jobs for %s\n" "$num_jobs" "$dataset_name"
  if (( num_jobs == 0 )); then
    echo "[INFO] No jobs to run for $dataset_name"
    return
  fi
  local tmpdir
  tmpdir=$(mktemp -d)
  local pids=()
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    (
      local completed=0 failed=0 skipped=0 total=0
      local idx
      for ((idx=gpu; idx<num_jobs; idx+=NUM_GPUS)); do
        IFS='|' read -r regime ckpt ds_name data_file img_dir out_prefix <<< "${jobs[$idx]}"
        ((++total))
        printf "\n================ RUN %s: %s | %s | %s (GPU %s) ================\n" "$total" "$regime" "$(basename "$ckpt")" "$ds_name" "$gpu"
        set +e
        out_dir="$(output_dir_for_ckpt "$regime" "$ckpt" "$out_prefix")"
        json_count=$(find "$out_dir" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l || true)
        if [[ "$SKIP_IF_EXISTS" == "1" && "$FORCE" != "1" && "$json_count" -ge "$MIN_OUTPUT_FILES" ]]; then
          printf "[SKIP] Existing outputs in %s (%s json files)\n" "$out_dir" "$json_count"
          rc=0
          set -e
          ((++skipped))
          ((++completed))
          continue
        fi
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
          printf "DRY_RUN: would run single_infer.py for %s | %s | %s\n" "$regime" "$(basename "$ckpt")" "$ds_name"
          rc=0
        else
          [[ "${DEBUG:-0}" == "1" ]] && set -x
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" /home/user/training/single_infer.py \
          --base_model "$BASE_MODEL" \
          --regime "$regime" \
          --checkpoint "$ckpt" \
          --dataset_name "$ds_name" \
          --data_file "$data_file" \
          --img_dir "$img_dir" \
          --output_prefix "$out_prefix" \
          --batch_size "$BATCH_SIZE" \
          --max_new_tokens "$MAX_NEW_TOKENS" \
          --temperature "$TEMPERATURE" \
          --min_p "$MIN_P" \
          --seed "$SEED" \
          --resized_width "$RESIZED_WIDTH" \
          --resized_height "$RESIZED_HEIGHT"
          rc=$?
          [[ "${DEBUG:-0}" == "1" ]] && set +x
        fi
        set -e
        if [[ $rc -eq 0 ]]; then
          ((++completed))
        else
          ((++failed))
          echo "[WARN] Run failed with code $rc"
        fi
      done
      echo "$completed $failed $skipped $total" > "$tmpdir/worker_${gpu}.stats"
    ) &
    pids+=("$!")
  done
  # Wait for all workers
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  # Aggregate results
  local c f s t
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    if [[ -f "$tmpdir/worker_${gpu}.stats" ]]; then
      read -r c f s t < "$tmpdir/worker_${gpu}.stats"
      completed_runs=$((completed_runs + c))
      failed_runs=$((failed_runs + f))
      skipped_runs=$((skipped_runs + s))
      total_runs=$((total_runs + t))
    fi
  done
  rm -rf "$tmpdir"
}

# Run all jobs (all datasets) in parallel across GPUs using strided workers with spillover
run_all_jobs_in_parallel() {
  printf "\n================ ALL DATASETS (spillover) NUM_GPUS=%s ================\n" "$NUM_GPUS"
  build_jobs_for_all_datasets_in_order
  local num_jobs=${#jobs[@]}
  printf "[INFO] Prepared %s total jobs across datasets: %s\n" "$num_jobs" "${DATASET_EXEC_ORDER[*]}"
  if (( num_jobs == 0 )); then
    echo "[INFO] No jobs to run"
    return
  fi
  local tmpdir
  tmpdir=$(mktemp -d)
  local pids=()
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    (
      local completed=0 failed=0 skipped=0 total=0
      local idx
      for ((idx=gpu; idx<num_jobs; idx+=NUM_GPUS)); do
        IFS='|' read -r regime ckpt ds_name data_file img_dir out_prefix <<< "${jobs[$idx]}"
        ((++total))
        printf "\n================ RUN %s: %s | %s | %s (GPU %s) ================\n" "$total" "$regime" "$(basename "$ckpt")" "$ds_name" "$gpu"
        set +e
        out_dir="$(output_dir_for_ckpt "$regime" "$ckpt" "$out_prefix")"
        json_count=$(find "$out_dir" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l || true)
        if [[ "$SKIP_IF_EXISTS" == "1" && "$FORCE" != "1" && "$json_count" -ge "$MIN_OUTPUT_FILES" ]]; then
          printf "[SKIP] Existing outputs in %s (%s json files)\n" "$out_dir" "$json_count"
          rc=0
          set -e
          ((++skipped))
          ((++completed))
          continue
        fi
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
          printf "DRY_RUN: would run single_infer.py for %s | %s | %s\n" "$regime" "$(basename "$ckpt")" "$ds_name"
          rc=0
        else
          [[ "${DEBUG:-0}" == "1" ]] && set -x
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" /home/user/training/single_infer.py \
          --base_model "$BASE_MODEL" \
          --regime "$regime" \
          --checkpoint "$ckpt" \
          --dataset_name "$ds_name" \
          --data_file "$data_file" \
          --img_dir "$img_dir" \
          --output_prefix "$out_prefix" \
          --batch_size "$BATCH_SIZE" \
          --max_new_tokens "$MAX_NEW_TOKENS" \
          --temperature "$TEMPERATURE" \
          --min_p "$MIN_P" \
          --seed "$SEED" \
          --resized_width "$RESIZED_WIDTH" \
          --resized_height "$RESIZED_HEIGHT"
          rc=$?
          [[ "${DEBUG:-0}" == "1" ]] && set +x
        fi
        set -e
        if [[ $rc -eq 0 ]]; then
          ((++completed))
        else
          ((++failed))
          echo "[WARN] Run failed with code $rc"
        fi
      done
      echo "$completed $failed $skipped $total" > "$tmpdir/worker_${gpu}.stats"
    ) &
    pids+=("$!")
  done
  # Wait for all workers
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  # Aggregate results
  local c f s t
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    if [[ -f "$tmpdir/worker_${gpu}.stats" ]]; then
      read -r c f s t < "$tmpdir/worker_${gpu}.stats"
      completed_runs=$((completed_runs + c))
      failed_runs=$((failed_runs + f))
      skipped_runs=$((skipped_runs + s))
      total_runs=$((total_runs + t))
    fi
  done
  rm -rf "$tmpdir"
}

total_runs=0
completed_runs=0
failed_runs=0
skipped_runs=0

# Execute all datasets together with spillover (priority order respected)
run_all_jobs_in_parallel

printf "\n================ SUMMARY ================\n"
printf "Total runs:     %s\n" "$total_runs"
printf "Completed runs: %s\n" "$completed_runs"
printf "Failed runs:    %s\n" "$failed_runs"
printf "Skipped runs:   %s\n" "$skipped_runs"


