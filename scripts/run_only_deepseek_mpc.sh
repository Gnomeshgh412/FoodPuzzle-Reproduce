#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
FRESH=0

if [[ "${1:-}" == "--fresh" ]]; then
  FRESH=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/run_only_deepseek_mpc.sh [--fresh]" >&2
  exit 2
fi

CODE_DIR="code/Only-Deepseek"
RESULTS_ROOT="results/Only-Deepseek"
TRAIN="results/splits/mpc/train.jsonl"
TEST="results/splits/mpc/test.jsonl"
DB="data/raw/flavordb.db"
EVIDENCE="data/collected_evidences/collected_evidences_task2.pkl"
PROVIDER="deepseek"
MODEL="deepseek-v4-flash"

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

for required in \
  "${CODE_DIR}/zero_shot.py" \
  "${CODE_DIR}/bm25_icl.py" \
  "${CODE_DIR}/scientific_agent.py" \
  "${CODE_DIR}/evaluation.py" \
  "${TRAIN}" \
  "${TEST}" \
  "${DB}" \
  "${EVIDENCE}"; do
  require_file "${required}"
done

for method in zero-shot icl agent; do
  mkdir -p "${RESULTS_ROOT}/${method}/mpc/${MODEL}"
done

if [[ "${FRESH}" -eq 1 ]]; then
  rm -f \
    "${RESULTS_ROOT}/zero-shot/mpc/${MODEL}/predictions.jsonl" \
    "${RESULTS_ROOT}/zero-shot/mpc/${MODEL}/evaluation_details.jsonl" \
    "${RESULTS_ROOT}/zero-shot/mpc/${MODEL}/evaluation_summary.json" \
    "${RESULTS_ROOT}/icl/mpc/${MODEL}/predictions.jsonl" \
    "${RESULTS_ROOT}/icl/mpc/${MODEL}/retrieval_metadata.jsonl" \
    "${RESULTS_ROOT}/icl/mpc/${MODEL}/evaluation_details.jsonl" \
    "${RESULTS_ROOT}/icl/mpc/${MODEL}/evaluation_summary.json" \
    "${RESULTS_ROOT}/agent/mpc/${MODEL}/predictions.jsonl" \
    "${RESULTS_ROOT}/agent/mpc/${MODEL}/evidence_metadata.jsonl" \
    "${RESULTS_ROOT}/agent/mpc/${MODEL}/retrieval_metadata.jsonl" \
    "${RESULTS_ROOT}/agent/mpc/${MODEL}/hypotheses_metadata.jsonl" \
    "${RESULTS_ROOT}/agent/mpc/${MODEL}/evaluation_details.jsonl" \
    "${RESULTS_ROOT}/agent/mpc/${MODEL}/evaluation_summary.json"
fi

ZERO_DIR="${RESULTS_ROOT}/zero-shot/mpc/${MODEL}"
while ! predictions_complete mpc "${TEST}" "${ZERO_DIR}/predictions.jsonl"; do
  if [[ -f "${ZERO_DIR}/predictions.jsonl" ]]; then
    "${PYTHON_BIN}" "${CODE_DIR}/zero_shot.py" \
      --task mpc --input "${TEST}" --output "${ZERO_DIR}/predictions.jsonl" \
      --llm-provider "${PROVIDER}" --llm-model "${MODEL}" --use-llm --resume || true
    if ! predictions_complete mpc "${TEST}" "${ZERO_DIR}/predictions.jsonl"; then
      "${PYTHON_BIN}" "${CODE_DIR}/zero_shot.py" \
        --task mpc --input "${TEST}" --output "${ZERO_DIR}/predictions.jsonl" \
        --llm-provider "${PROVIDER}" --llm-model "${MODEL}" \
        --use-llm --resume --retry-errors || true
    fi
  else
    "${PYTHON_BIN}" "${CODE_DIR}/zero_shot.py" \
      --task mpc --input "${TEST}" --output "${ZERO_DIR}/predictions.jsonl" \
      --llm-provider "${PROVIDER}" --llm-model "${MODEL}" --use-llm || true
  fi
  predictions_complete mpc "${TEST}" "${ZERO_DIR}/predictions.jsonl" || wait_before_retry
done

ICL_DIR="${RESULTS_ROOT}/icl/mpc/${MODEL}"
while ! predictions_complete mpc "${TEST}" "${ICL_DIR}/predictions.jsonl"; do
  ICL_RESUME=()
  [[ -f "${ICL_DIR}/predictions.jsonl" || -f "${ICL_DIR}/retrieval_metadata.jsonl" ]] && ICL_RESUME=(--resume)
  "${PYTHON_BIN}" "${CODE_DIR}/bm25_icl.py" \
    --task mpc --train "${TRAIN}" --test "${TEST}" \
    --output "${ICL_DIR}/predictions.jsonl" \
    --retrieval-metadata "${ICL_DIR}/retrieval_metadata.jsonl" \
    --llm-provider "${PROVIDER}" --llm-model "${MODEL}" --use-llm \
    ${ICL_RESUME[@]+"${ICL_RESUME[@]}"} || true
  predictions_complete mpc "${TEST}" "${ICL_DIR}/predictions.jsonl" || wait_before_retry
done

AGENT_DIR="${RESULTS_ROOT}/agent/mpc/${MODEL}"
while ! predictions_complete mpc "${TEST}" "${AGENT_DIR}/predictions.jsonl"; do
  AGENT_RESUME=()
  [[ -f "${AGENT_DIR}/predictions.jsonl" ]] && AGENT_RESUME=(--resume)
  "${PYTHON_BIN}" "${CODE_DIR}/scientific_agent.py" \
    --task mpc --train "${TRAIN}" --test "${TEST}" --evidence "${EVIDENCE}" \
    --output "${AGENT_DIR}/predictions.jsonl" \
    --evidence-metadata "${AGENT_DIR}/evidence_metadata.jsonl" \
    --retrieval-metadata "${AGENT_DIR}/retrieval_metadata.jsonl" \
    --hypotheses-metadata "${AGENT_DIR}/hypotheses_metadata.jsonl" \
    --llm-provider "${PROVIDER}" --llm-model "${MODEL}" --use-llm \
    ${AGENT_RESUME[@]+"${AGENT_RESUME[@]}"} || true
  predictions_complete mpc "${TEST}" "${AGENT_DIR}/predictions.jsonl" || wait_before_retry
done

for method in zero-shot icl agent; do
  RESULT_DIR="${RESULTS_ROOT}/${method}/mpc/${MODEL}"
  until "${PYTHON_BIN}" "${CODE_DIR}/evaluation.py" \
    --task mpc --gold "${TEST}" --pred "${RESULT_DIR}/predictions.jsonl" \
    --db "${DB}" --use-llm --llm-provider "${PROVIDER}" --llm-model "${MODEL}" \
    --save-details "${RESULT_DIR}/evaluation_details.jsonl" \
    --save-summary-json "${RESULT_DIR}/evaluation_summary.json"; do
    wait_before_retry
  done
done

echo "ONLY_DEEPSEEK_MPC_AUTOMATION: COMPLETE"
