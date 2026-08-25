#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3}"
MODEL="deepseek-v4-flash"
PROVIDER="deepseek"
METHOD_VERSION="heterogeneous_multi_agent_single_conformer_v1"
SCHEMA_VERSION="multi_agent_v1"
CODE_DIR="code/Only-Deepseek"
AGENT_CODE="${CODE_DIR}/multi_agent.py"
EVALUATION_CODE="${CODE_DIR}/evaluation.py"
RESULTS_ROOT="results/Only-Deepseek/multi-agent"
DB="data/raw/flavordb.db"
UNIMOL_EMBEDDINGS="data/structure/unimol/unimol_embeddings.npz"
MFP_TRAIN="results/splits/mfp/train.jsonl"
MFP_TEST="results/splits/mfp/test.jsonl"
MPC_TRAIN="results/splits/mpc/train.jsonl"
MPC_TEST="results/splits/mpc/test.jsonl"
MFP_EVIDENCE="data/collected_evidences/collected_evidences_task1.pkl"
MPC_EVIDENCE="data/collected_evidences/collected_evidences_task2.pkl"
FUNCTIONAL_GROUP_CACHE="results/Only-Deepseek/shared_cache/${MODEL}_functional_group_cache.json"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

if [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/run_multi_agent.sh" >&2
  exit 2
fi

for required in \
  "${AGENT_CODE}" \
  "${EVALUATION_CODE}" \
  "${DB}" \
  "${UNIMOL_EMBEDDINGS}" \
  "${MFP_TRAIN}" \
  "${MFP_TEST}" \
  "${MPC_TRAIN}" \
  "${MPC_TEST}" \
  "${MFP_EVIDENCE}" \
  "${MPC_EVIDENCE}"; do
  require_file "${required}"
done

initialize_result() {
  local task="$1"
  local train_path="$2"
  local test_path="$3"
  local evidence_path="$4"
  local result_dir="$5"

  mkdir -p "${result_dir}"
  "${PYTHON_BIN}" - \
    "${METHOD_VERSION}" \
    "${SCHEMA_VERSION}" \
    "${task}" \
    "${train_path}" \
    "${test_path}" \
    "${evidence_path}" \
    "${result_dir}" \
    "${AGENT_CODE}" \
    "${SCRIPT_DIR}/run_multi_agent.sh" \
    "${DB}" \
    "${UNIMOL_EMBEDDINGS}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

(
    method_version,
    schema_version,
    task,
    train_value,
    test_value,
    evidence_value,
    result_value,
    code_value,
    runner_value,
    db_value,
    unimol_value,
) = sys.argv[1:12]

result_dir = Path(result_value)
metadata_path = result_dir / "run_metadata.json"
formal_names = [
    "predictions.jsonl",
    "agent_metadata.jsonl",
    "retrieval_metadata.jsonl",
    "evidence_metadata.jsonl",
    "evaluation_details.jsonl",
    "evaluation_summary.json",
]

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

code_sha256 = digest(code_value)
runner_sha256 = digest(runner_value)
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    compatible = (
        metadata.get("method") == method_version
        and metadata.get("schema_version") == schema_version
        and metadata.get("task") == task
        and metadata.get("files", {}).get("code", {}).get("sha256") == code_sha256
        and metadata.get("files", {}).get("runner", {}).get("sha256") == runner_sha256
    )
    if not compatible:
        raise SystemExit(
            "Existing multi-agent result belongs to a different code/config "
            "snapshot; refusing to overwrite it."
        )
    print(f"MULTI_AGENT_{task.upper()}_WORKSPACE: RESUME")
    raise SystemExit(0)

unexpected = [
    name for name in formal_names if (result_dir / name).exists()
]
if unexpected:
    raise SystemExit(
        "Formal artifacts exist without compatible run_metadata; refusing to "
        f"overwrite: {unexpected}"
    )

metadata = {
    "method": method_version,
    "schema_version": schema_version,
    "task": task,
    "model": "deepseek-v4-flash",
    "provider": "deepseek",
    "status": "running",
    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "holdout_used": False,
    "variant": "single_conformer",
    "generation": (
        {
            "candidate_count": 7,
            "occurrence_weight": 0.55,
            "structure_weight": 0.30,
            "evidence_weight": 0.15,
            "max_generation_api_calls_per_sample": 2,
        }
        if task == "mfp"
        else {
            "primary_pool": 100,
            "rescue_pool": 300,
            "unary_weight": 1.0,
            "coverage_weight": 0.12,
            "redundancy_weight": 0.04,
            "evidence_weight": 0.03,
            "swap_margin": 0.025,
            "nnpu_epochs": 80,
            "nnpu_unlabeled_per_row": 48,
            "nnpu_learning_rate": 0.08,
            "nnpu_l2": 0.01,
            "max_generation_api_calls_per_sample": 1,
        }
    ),
    "files": {
        "code": {"path": code_value, "sha256": code_sha256},
        "runner": {"path": runner_value, "sha256": runner_sha256},
        "train": {"path": train_value, "sha256": digest(train_value)},
        "test": {"path": test_value, "sha256": digest(test_value)},
        "evidence": {"path": evidence_value, "sha256": digest(evidence_value)},
        "db": {"path": db_value, "sha256": digest(db_value)},
        "unimol": {"path": unimol_value, "sha256": digest(unimol_value)},
    },
    "isolation": {
        "optimized_agent_inputs_used": False,
        "evaluation_cache_available_to_prediction": False,
        "prediction_output_root": str(result_dir),
    },
    "notes": [
        "MFP and MPC are trained and inferred independently.",
        "This version uses exactly one frozen UniMol representation per molecule.",
        "Prediction code cannot access the official functional-group evaluation cache.",
        "Gold fields are removed before prediction-stage agents receive each query.",
    ],
}
temporary = metadata_path.with_name(metadata_path.name + ".tmp")
temporary.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, metadata_path)
print(f"MULTI_AGENT_{task.upper()}_WORKSPACE: INITIALIZED")
PY
}

generation_complete() {
  local task="$1"
  local test_path="$2"
  local result_dir="$3"
  "${PYTHON_BIN}" - \
    "${task}" \
    "${test_path}" \
    "${result_dir}/predictions.jsonl" \
    "${result_dir}/agent_metadata.jsonl" <<'PY'
import json
import sys
from pathlib import Path

task, test_value, prediction_value, agent_value = sys.argv[1:5]

def rows(path):
    if not Path(path).is_file():
        return []
    output = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                output.append(json.loads(line))
    return output

expected = [str(row["id"]) for row in rows(test_value)]
predictions = {}
for row in rows(prediction_value):
    row_id = str(row.get("id"))
    if row.get("error"):
        continue
    if task == "mfp":
        valid = bool(str(row.get("predicted_food") or "").strip())
    else:
        values = row.get("predicted_molecules")
        valid = (
            isinstance(values, list)
            and len(values) == int(row.get("n") or -1)
        )
    if valid:
        predictions[row_id] = row
agents = {
    str(row.get("id")): row
    for row in rows(agent_value)
    if row.get("schema_version") == "multi_agent_v1"
}
complete = all(row_id in predictions and row_id in agents for row_id in expected)
raise SystemExit(0 if complete and len(expected) > 0 else 1)
PY
}

run_generation() {
  local task="$1"
  local train_path="$2"
  local test_path="$3"
  local evidence_path="$4"
  local result_dir="$5"
  local resume_args=()

  if [[ -f "${result_dir}/predictions.jsonl" ]]; then
    resume_args=(--resume)
  fi
  "${PYTHON_BIN}" "${AGENT_CODE}" \
    --task "${task}" \
    --train "${train_path}" \
    --test "${test_path}" \
    --db "${DB}" \
    --evidence "${evidence_path}" \
    --unimol-embeddings "${UNIMOL_EMBEDDINGS}" \
    --output "${result_dir}/predictions.jsonl" \
    --agent-metadata "${result_dir}/agent_metadata.jsonl" \
    --retrieval-metadata "${result_dir}/retrieval_metadata.jsonl" \
    --evidence-metadata "${result_dir}/evidence_metadata.jsonl" \
    --llm-provider "${PROVIDER}" \
    --llm-model "${MODEL}" \
    --use-llm \
    ${resume_args[@]+"${resume_args[@]}"}

  if ! generation_complete "${task}" "${test_path}" "${result_dir}"; then
    echo "MULTI_AGENT_${task^^}: incomplete; rerun the same script to resume." >&2
    return 1
  fi
}

run_evaluation() {
  local task="$1"
  local test_path="$2"
  local result_dir="$3"
  local cache_args=()

  # The functional-group cache first enters the process here, after generation
  # has passed its completeness check.  multi_agent.py has no cache argument.
  if [[ "${task}" == "mpc" ]]; then
    cache_args=(--functional-group-cache "${FUNCTIONAL_GROUP_CACHE}")
  fi
  "${PYTHON_BIN}" "${EVALUATION_CODE}" \
    --task "${task}" \
    --gold "${test_path}" \
    --pred "${result_dir}/predictions.jsonl" \
    --db "${DB}" \
    --use-llm \
    --llm-provider "${PROVIDER}" \
    --llm-model "${MODEL}" \
    --save-details "${result_dir}/evaluation_details.jsonl" \
    --save-summary-json "${result_dir}/evaluation_summary.json" \
    ${cache_args[@]+"${cache_args[@]}"}
}

finalize_result() {
  local task="$1"
  local result_dir="$2"
  "${PYTHON_BIN}" - \
    "${task}" \
    "${result_dir}" \
    "${METHOD_VERSION}" \
    "${SCHEMA_VERSION}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

task, result_value, method_version, schema_version = sys.argv[1:5]
result_dir = Path(result_value)
metadata_path = result_dir / "run_metadata.json"
summary_path = result_dir / "evaluation_summary.json"
output_names = [
    "predictions.jsonl",
    "agent_metadata.jsonl",
    "retrieval_metadata.jsonl",
    "evidence_metadata.jsonl",
    "evaluation_details.jsonl",
    "evaluation_summary.json",
]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if (
    metadata.get("method") != method_version
    or metadata.get("schema_version") != schema_version
):
    raise SystemExit("run metadata changed during execution")
for name in output_names:
    if not (result_dir / name).is_file():
        raise SystemExit(f"missing formal output: {result_dir / name}")

api_calls = 0
prompt_tokens = 0
completion_tokens = 0
with (result_dir / "agent_metadata.jsonl").open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        for call in row.get("api_calls") or []:
            api_calls += 1
            usage = call.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)

evaluation = json.loads(summary_path.read_text(encoding="utf-8"))
evaluation_api_calls = (
    int(evaluation.get("matched_ids") or 0)
    - int(evaluation.get("gold_category_lookup_failures") or 0)
    if task == "mfp"
    else int(evaluation.get("llm_functional_group_prediction_count") or 0)
)
metadata.update(
    {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation": evaluation,
        "budget": {
            "generation_api_calls": api_calls,
            "evaluation_api_calls": evaluation_api_calls,
            "total_api_calls": api_calls + evaluation_api_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "token_counts_cover_generation_only": True,
        },
        "outputs": {
            name: digest(result_dir / name) for name in output_names
        },
        "constraints": {
            "mfp_allowed_categories": 21 if task == "mfp" else None,
            "mpc_exact_n": True if task == "mpc" else None,
            "structured_agent_schema": schema_version,
            "holdout_used": False,
        },
    }
)
temporary = metadata_path.with_name(metadata_path.name + ".tmp")
temporary.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, metadata_path)
print(f"MULTI_AGENT_{task.upper()}: COMPLETE")
PY
}

MFP_RESULT="${RESULTS_ROOT}/mfp/${MODEL}"
MPC_RESULT="${RESULTS_ROOT}/mpc/${MODEL}"

initialize_result \
  "mfp" "${MFP_TRAIN}" "${MFP_TEST}" "${MFP_EVIDENCE}" "${MFP_RESULT}"
run_generation \
  "mfp" "${MFP_TRAIN}" "${MFP_TEST}" "${MFP_EVIDENCE}" "${MFP_RESULT}"
run_evaluation "mfp" "${MFP_TEST}" "${MFP_RESULT}"
finalize_result "mfp" "${MFP_RESULT}"

initialize_result \
  "mpc" "${MPC_TRAIN}" "${MPC_TEST}" "${MPC_EVIDENCE}" "${MPC_RESULT}"
run_generation \
  "mpc" "${MPC_TRAIN}" "${MPC_TEST}" "${MPC_EVIDENCE}" "${MPC_RESULT}"
run_evaluation "mpc" "${MPC_TEST}" "${MPC_RESULT}"
finalize_result "mpc" "${MPC_RESULT}"
