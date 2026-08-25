#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
EVALUATION_INTERVAL_SECONDS="${EVALUATION_INTERVAL_SECONDS:-60}"

if [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/run_multi_models.sh" >&2
  exit 2
fi

CODE_DIR="code/Multi-Models"
RESULTS_ROOT="results/Multi-Models"
DB="data/raw/flavordb.db"
MFP_EVIDENCE="data/collected_evidences/collected_evidences_task1.pkl"
MPC_EVIDENCE="data/collected_evidences/collected_evidences_task2.pkl"

# DeepSeek 放在最后，降低与 Only-Deepseek MPC 重跑争用同一 API key 的概率。
PROVIDERS=(
  "aihubmix-coding-glm-4.7-free"
  "aihubmix-gpt-4.1-free"
  "aihubmix-xiaomi-mimo-v2.5-free"
  "deepseek"
)
MODELS=(
  "coding-glm-4.7-free"
  "gpt-4.1-free"
  "xiaomi-mimo-v2.5-free"
  "deepseek-v4-flash"
)

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

predictions_complete() {
  local task="$1"
  local test_path="$2"
  local prediction_path="$3"
  "${PYTHON_BIN}" - "${task}" "${test_path}" "${prediction_path}" <<'PY'
import json
import sys
from pathlib import Path

task, test_value, prediction_value = sys.argv[1:4]
test_path = Path(test_value)
prediction_path = Path(prediction_value)
if not prediction_path.is_file():
    raise SystemExit(1)

expected = {}
for line in test_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        expected[str(row["id"])] = row

latest = {}
for line in prediction_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        if isinstance(row, dict) and row.get("id") is not None:
            latest[str(row["id"])] = row

successful = set()
for row_id, row in latest.items():
    if row.get("error"):
        continue
    if task == "mfp" and str(row.get("predicted_food") or "").strip():
        successful.add(row_id)
    if task == "mpc" and isinstance(row.get("predicted_molecules"), list) and row["predicted_molecules"]:
        successful.add(row_id)

raise SystemExit(0 if expected and set(expected) <= successful else 1)
PY
}

wait_before_retry() {
  echo "Incomplete formal output; retrying in ${RETRY_SECONDS} seconds."
  sleep "${RETRY_SECONDS}"
}

run_zero_shot() {
  local task="$1" provider="$2" model="$3" test_path="$4" result_dir="$5"
  local output="${result_dir}/predictions.jsonl"
  while ! predictions_complete "${task}" "${test_path}" "${output}"; do
    if [[ -f "${output}" ]]; then
      "${PYTHON_BIN}" "${CODE_DIR}/zero_shot.py" \
        --task "${task}" --input "${test_path}" --output "${output}" \
        --llm-provider "${provider}" --llm-model "${model}" --use-llm --resume || true
      if ! predictions_complete "${task}" "${test_path}" "${output}"; then
        "${PYTHON_BIN}" "${CODE_DIR}/zero_shot.py" \
          --task "${task}" --input "${test_path}" --output "${output}" \
          --llm-provider "${provider}" --llm-model "${model}" \
          --use-llm --resume --retry-errors || true
      fi
    else
      "${PYTHON_BIN}" "${CODE_DIR}/zero_shot.py" \
        --task "${task}" --input "${test_path}" --output "${output}" \
        --llm-provider "${provider}" --llm-model "${model}" --use-llm || true
    fi
    predictions_complete "${task}" "${test_path}" "${output}" || wait_before_retry
  done
}

run_icl() {
  local task="$1" provider="$2" model="$3" train_path="$4" test_path="$5" result_dir="$6"
  local output="${result_dir}/predictions.jsonl"
  local retrieval="${result_dir}/retrieval_metadata.jsonl"
  while ! predictions_complete "${task}" "${test_path}" "${output}"; do
    local resume_args=()
    [[ -f "${output}" || -f "${retrieval}" ]] && resume_args=(--resume)
    "${PYTHON_BIN}" "${CODE_DIR}/bm25_icl.py" \
      --task "${task}" --train "${train_path}" --test "${test_path}" \
      --output "${output}" --retrieval-metadata "${retrieval}" \
      --llm-provider "${provider}" --llm-model "${model}" --use-llm \
      ${resume_args[@]+"${resume_args[@]}"} || true
    predictions_complete "${task}" "${test_path}" "${output}" || wait_before_retry
  done
}

run_agent() {
  local task="$1" provider="$2" model="$3" train_path="$4" test_path="$5" result_dir="$6"
  local output="${result_dir}/predictions.jsonl"
  local icl_retrieval="${RESULTS_ROOT}/icl/${task}/${model}/retrieval_metadata.jsonl"
  while ! predictions_complete "${task}" "${test_path}" "${output}"; do
    local resume_args=()
    [[ -f "${output}" ]] && resume_args=(--resume)
    local evidence_args=()
    if [[ "${task}" == "mfp" ]]; then
      evidence_args=(
        --official-evidence-pkl "${MFP_EVIDENCE}"
        --icl-retrieval-metadata "${icl_retrieval}"
      )
    else
      evidence_args=(--evidence "${MPC_EVIDENCE}")
    fi
    "${PYTHON_BIN}" "${CODE_DIR}/scientific_agent.py" \
      --task "${task}" --train "${train_path}" --test "${test_path}" \
      "${evidence_args[@]}" \
      --output "${output}" \
      --evidence-metadata "${result_dir}/evidence_metadata.jsonl" \
      --retrieval-metadata "${result_dir}/retrieval_metadata.jsonl" \
      --hypotheses-metadata "${result_dir}/hypotheses_metadata.jsonl" \
      --llm-provider "${provider}" --llm-model "${model}" --use-llm \
      ${resume_args[@]+"${resume_args[@]}"} || true
    predictions_complete "${task}" "${test_path}" "${output}" || wait_before_retry
  done
}

for required in \
  "${CODE_DIR}/zero_shot.py" \
  "${CODE_DIR}/bm25_icl.py" \
  "${CODE_DIR}/scientific_agent.py" \
  "${CODE_DIR}/evaluation.py" \
  "${DB}" \
  "${MFP_EVIDENCE}" \
  "${MPC_EVIDENCE}" \
  "results/splits/mfp/train.jsonl" \
  "results/splits/mfp/test.jsonl" \
  "results/splits/mpc/train.jsonl" \
  "results/splits/mpc/test.jsonl"; do
  require_file "${required}"
done

for index in "${!MODELS[@]}"; do
  provider="${PROVIDERS[$index]}"
  model="${MODELS[$index]}"
  for task in mfp mpc; do
    train_path="results/splits/${task}/train.jsonl"
    test_path="results/splits/${task}/test.jsonl"
    for method in zero-shot icl agent; do
      mkdir -p "${RESULTS_ROOT}/${method}/${task}/${model}"
    done
    run_zero_shot \
      "${task}" "${provider}" "${model}" "${test_path}" \
      "${RESULTS_ROOT}/zero-shot/${task}/${model}"
    run_icl \
      "${task}" "${provider}" "${model}" "${train_path}" "${test_path}" \
      "${RESULTS_ROOT}/icl/${task}/${model}"
    run_agent \
      "${task}" "${provider}" "${model}" "${train_path}" "${test_path}" \
      "${RESULTS_ROOT}/agent/${task}/${model}"
  done
done

# 评测严格串行，避免多个进程同时争用 GPT-4.1-free 的免费限额。
for index in "${!MODELS[@]}"; do
  model="${MODELS[$index]}"
  for task in mfp mpc; do
    test_path="results/splits/${task}/test.jsonl"
    for method in zero-shot icl agent; do
      result_dir="${RESULTS_ROOT}/${method}/${task}/${model}"
      "${PYTHON_BIN}" "${CODE_DIR}/evaluation.py" \
        --task "${task}" --gold "${test_path}" \
        --pred "${result_dir}/predictions.jsonl" --db "${DB}" --use-llm \
        --llm-provider aihubmix-gpt-4.1-free \
        --request-interval-seconds "${EVALUATION_INTERVAL_SECONDS}" \
        --save-details "${result_dir}/evaluation_details.jsonl" \
        --save-summary-json "${result_dir}/evaluation_summary.json"
    done
  done
done

echo "MULTI_MODELS_AUTOMATION: COMPLETE"
