#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
MODEL="deepseek-v4-flash"
METHOD_VERSION="optimized_agent_v15_mfp_concrete_protocol"
CODE_DIR="code/Only-Deepseek"
RESULTS_ROOT="results/Only-Deepseek/optimized-agent"
DB="data/raw/flavordb.db"
UNIMOL_EMBEDDINGS="data/structure/unimol/unimol_embeddings.npz"
MFP_TRAIN="results/splits/mfp/train.jsonl"
MFP_TEST="results/splits/mfp/test.jsonl"
MPC_TRAIN="results/splits/mpc/train.jsonl"
MPC_TEST="results/splits/mpc/test.jsonl"
MFP_EVIDENCE="data/collected_evidences/collected_evidences_task1.pkl"
MPC_EVIDENCE="data/collected_evidences/collected_evidences_task2.pkl"
MFP_ICL_RETRIEVAL="results/Only-Deepseek/icl/mfp/${MODEL}/retrieval_metadata.jsonl"
FUNCTIONAL_GROUP_CACHE="results/Only-Deepseek/shared_cache/${MODEL}_functional_group_cache.json"

TASK_FILTER="${1:-all}"
if [[ $# -gt 1 || "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "mfp" && "${TASK_FILTER}" != "mpc" ]]; then
  echo "Usage: bash scripts/run_optimized_agent.sh [all|mfp|mpc]" >&2
  exit 2
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

initialize_method_result() {
  local task="$1"
  local result_dir="$2"
  "${PYTHON_BIN}" - \
    "${METHOD_VERSION}" \
    "${task}" \
    "${result_dir}" \
    "${CODE_DIR}/optimized_agent.py" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

method_version, task, result_value, code_value = sys.argv[1:5]
result_dir = Path(result_value)
metadata_path = result_dir / "run_metadata.json"
code_path = Path(code_value)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

code_sha256 = sha256(code_path)
existing = {}
if metadata_path.is_file():
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}

existing_code_sha256 = (
    existing.get("code_sha256")
    or existing.get("files", {}).get("code", {}).get("optimized_agent.py")
)
same_run = (
    existing.get("method") == method_version
    and existing.get("task") == task
    and existing_code_sha256 == code_sha256
)
if same_run:
    print(f"OPTIMIZED_{task.upper()}_WORKSPACE: RESUME")
    raise SystemExit(0)

formal_artifacts = [
    "predictions.jsonl",
    "retrieval_metadata.jsonl",
    "evidence_metadata.jsonl",
    "hypotheses_metadata.jsonl",
    "evaluation_details.jsonl",
    "evaluation_summary.json",
]
for filename in formal_artifacts:
    path = result_dir / filename
    if path.is_file():
        path.unlink()

metadata = {
    "method": method_version,
    "task": task,
    "model": "deepseek-v4-flash",
    "provider": "deepseek",
    "ablation": "full",
    "status": "running",
    "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "code_sha256": code_sha256,
    "notes": [
        "The previous optimized-agent artifacts in this same directory were intentionally replaced.",
        "This file doubles as the formal breakpoint marker; do not remove it during resume.",
        "MFP emits concrete food names and leaves macro-category mapping exclusively to evaluation; MPC v15 uses no UniMol and keeps the v14 train-only exact-profile-clustered action bank.",
        "MPC v15 freezes H1 and Scientist Top-K, then separately verifies add necessity and remove safety with low-capacity stacked-OOF gates; an action executes only when both gates are admitted."
    ]
}
temporary = metadata_path.with_name(metadata_path.name + ".tmp")
temporary.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, metadata_path)
print(f"OPTIMIZED_{task.upper()}_WORKSPACE: INITIALIZED_{method_version}")
PY
}

finalize_run_metadata() {
  local task="$1"
  local train_path="$2"
  local test_path="$3"
  local evidence_path="$4"
  local result_dir="$5"
  "${PYTHON_BIN}" - \
    "${METHOD_VERSION}" \
    "${task}" \
    "${train_path}" \
    "${test_path}" \
    "${evidence_path}" \
    "${result_dir}" \
    "${MODEL}" \
    "${CODE_DIR}" \
    "${DB}" \
    "${UNIMOL_EMBEDDINGS}" \
    "${MFP_ICL_RETRIEVAL}" \
    "${FUNCTIONAL_GROUP_CACHE}" \
    "${SCRIPT_DIR}/run_optimized_agent.sh" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

(
    method_version,
    task,
    train_value,
    test_value,
    evidence_value,
    result_value,
    model,
    code_dir_value,
    db_value,
    unimol_value,
    mfp_retrieval_value,
    cache_value,
    runner_value,
) = sys.argv[1:14]

train_path = Path(train_value)
test_path = Path(test_value)
evidence_path = Path(evidence_value)
result_dir = Path(result_value)
code_dir = Path(code_dir_value)
db_path = Path(db_value)
unimol_path = Path(unimol_value)
metadata_path = result_dir / "run_metadata.json"
summary_path = result_dir / "evaluation_summary.json"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def file_record(path):
    return {"path": str(path), "sha256": sha256(path)}

running = json.loads(metadata_path.read_text(encoding="utf-8"))
evaluation = json.loads(summary_path.read_text(encoding="utf-8"))
code_files = {
    "optimized_agent.py": code_dir / "optimized_agent.py",
    "scientific_agent.py": code_dir / "scientific_agent.py",
    "evaluation.py": code_dir / "evaluation.py",
    "run_optimized_agent.sh": Path(runner_value),
}
output_names = [
    "predictions.jsonl",
    "retrieval_metadata.jsonl",
    "evidence_metadata.jsonl",
    "hypotheses_metadata.jsonl",
    "evaluation_details.jsonl",
    "evaluation_summary.json",
]

evaluation_protocol = {
    "judge_provider": "deepseek",
    "judge_model": model,
}
if task == "mfp":
    evaluation_protocol.update(
        {
            "mode": "official_style_llm_macro_category_mapping",
            "category_judge_cache": None,
        }
    )
else:
    cache_path = Path(cache_value)
    cache_metadata_path = cache_path.with_name(cache_path.name + ".metadata.json")
    evaluation_protocol.update(
        {
            "mode": "official_llm_functional_group_f1",
            "functional_group_cache": file_record(cache_path),
            "functional_group_cache_metadata": file_record(cache_metadata_path),
        }
    )

data = {
    "train": file_record(train_path),
    "test": file_record(test_path),
    "db": file_record(db_path),
    "evidence": file_record(evidence_path),
}
if task == "mfp":
    data["unimol_embeddings"] = file_record(unimol_path)
    data["icl_retrieval_metadata"] = file_record(Path(mfp_retrieval_value))

metadata = {
    "method": method_version,
    "task": task,
    "model": model,
    "provider": "deepseek",
    "ablation": "full",
    "agent_type": (
        "concrete_food_unimol_set_retrieval_scientist_reviewer_agent"
        if task == "mfp"
        else "metric_aligned_exact_n_set_agent_with_typed_evidence_auditor"
    ),
    "status": "complete",
    "started_at": running.get("started_at"),
    "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": None,
    "split": {
        "strategy": "reconstructed",
        "seed": 42,
        "is_official_split": False,
        "test_samples": 71,
        "train_test_id_overlap": 0,
    },
    "generation": {
        "uses_llm": True,
        "structure_top_k": 30,
        "bm25_top_k": 3,
        "evidence_molecule_limit": 8 if task == "mfp" else None,
        "mfp_max_snippets_per_molecule": 3 if task == "mfp" else None,
        "mpc_max_evidence_snippets": 10 if task == "mpc" else None,
        "max_structure_candidates": 300 if task == "mpc" else None,
        "mfp_output_protocol": (
            "concrete_food_name_then_evaluation_macro_mapping"
            if task == "mfp" else None
        ),
        "mfp_unimol_adapter": (
            "bidirectional_concrete_food_set_retrieval"
            if task == "mfp"
            else None
        ),
        "mfp_structured_response_fallback": (
            "fixed_concrete_candidates_without_invented_evidence"
            if task == "mfp" else None
        ),
        "mfp_reviewer_candidate_space": (
            "three_concrete_retrieved_foods" if task == "mfp" else None
        ),
        "mpc_ranker_training": (
            "task_shaped_pairwise_positive_unlabeled_completion"
            if task == "mpc"
            else None
        ),
        "mpc_primary_ranker": (
            "masked_query_pairwise_positive_unlabeled_occurrence"
            if task == "mpc"
            else None
        ),
        "mpc_global_rank_fusion": False if task == "mpc" else None,
        "mpc_residual_calibration": (
            "train_only_grouped_oof_with_strict_zero_budget_fallback"
            if task == "mpc"
            else None
        ),
        "mpc_residual_selection_metric": (
            "macro_functional_group_f1"
            if task == "mpc"
            else None
        ),
        "mpc_set_decoder": (
            "grouped_oof_metric_aligned_exact_n_functional_group_decoder_or_h1"
            if task == "mpc"
            else None
        ),
        "mpc_structural_hypothesis": (
            None
            if task == "mpc"
            else None
        ),
        "mpc_unimol_used": False if task == "mpc" else None,
        "mpc_candidate_catalog": (
            "flavordb_catalog_plus_training_profiles"
            if task == "mpc"
            else None
        ),
        "mpc_retrieval": (
            "idf_weighted_partial_and_food_profile_retrieval"
            if task == "mpc"
            else None
        ),
        "mpc_action_policy": (
            "typed_evidence_review_disabled_until_metric_aligned_incremental_admission"
            if task == "mpc"
            else None
        ),
        "mpc_evidence_changes_structure_score": False if task == "mpc" else None,
        "mpc_reviewer_margin_threshold": None,
        "mpc_reviewer_disagreement_threshold": None,
        "mpc_scientist_calls_per_reviewed_sample": 0 if task == "mpc" else None,
        "mpc_reviewer_calls_per_reviewed_sample": 0 if task == "mpc" else None,
        "mpc_fusion_calls_per_reviewed_sample": 0 if task == "mpc" else None,
        "mpc_method_reads_functional_group_evaluation_cache": (
            False if task == "mpc" else None
        ),
    },
    "evaluation": evaluation,
    "evaluation_protocol": evaluation_protocol,
    "files": {
        "code": {name: sha256(path) for name, path in code_files.items()},
        "data": data,
        "outputs": {
            name: sha256(result_dir / name)
            for name in output_names
        },
    },
    "notes": [
        "MFP and MPC are trained and inferred independently.",
        "MFP uses frozen single-conformer UniMol to retrieve concrete food candidates; MPC does not load or use UniMol.",
        "MFP generation never exposes the evaluator macro-category ontology; category mapping occurs only in evaluation.",
        "MPC molecule-local FlavorDB descriptors and functional groups never use entity_molecule_link or the LLM functional-group evaluation cache.",
        "MFP prediction authority comes only from three concrete foods ranked by bidirectional UniMol set retrieval and audited by the Scientist and Reviewer.",
        "Malformed MFP structured responses use the fixed concrete candidates without inventing evidence, preventing unbounded retries.",
        "MPC trains on the task-shaped empirical observation process with low-capacity pairwise positive-unlabeled comparisons.",
        "MPC retrieval actions use an independent FlavorDB-plus-training candidate catalog and IDF-weighted profile retrieval.",
        "MPC uses deterministic molecule-intrinsic FlavorDB functional groups and never loads UniMol.",
        "The exact-N set decoder is admitted only after positive train-only grouped-OOF macro functional-group F1 with a positive paired-bootstrap lower bound and non-negative cardinality-half diagnostics.",
        "The legacy exact-molecule action gate is disabled because its target is not aligned with the released functional-group metric.",
        "MPC evidence remains relation typed; occurrence and functional-replication claims must be supported by matching evidence relations.",
        "The Scientist and Reviewer remain available as typed-evidence auditors but receive no prediction-changing authority until a metric-aligned incremental gate is admitted.",
        "The final MPC executor is deterministic, exact-N, and makes no fusion language-model call.",
        "The v13 method selects decoder policy only from train-side molecule-intrinsic functional groups and never reads the released LLM evaluation cache.",
        "No Git commit identifies this run; SHA-256 values bind the formal artifacts."
    ],
}
temporary = metadata_path.with_name(metadata_path.name + ".tmp")
temporary.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, metadata_path)
print(f"OPTIMIZED_{task.upper()}_RUN_METADATA: COMPLETE")
PY
}

generation_complete() {
  local task="$1"
  local test_path="$2"
  local prediction_path="$3"
  local hypotheses_path="$4"
  "${PYTHON_BIN}" - \
    "${task}" \
    "${test_path}" \
    "${prediction_path}" \
    "${hypotheses_path}" <<'PY'
import json
import sys
from pathlib import Path

task, test_value, prediction_value, hypotheses_value = sys.argv[1:5]
test_path = Path(test_value)
prediction_path = Path(prediction_value)
hypotheses_path = Path(hypotheses_value)
if not prediction_path.is_file() or not hypotheses_path.is_file():
    raise SystemExit(1)

expected = {
    str(json.loads(line)["id"])
    for line in test_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
latest = {}
for line in prediction_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if isinstance(row, dict) and row.get("id") is not None:
        latest[str(row["id"])] = row

successful_predictions = set()
for row_id, row in latest.items():
    if row.get("error"):
        continue
    if task == "mfp" and str(row.get("predicted_food") or "").strip():
        successful_predictions.add(row_id)
    if task == "mpc":
        predicted = row.get("predicted_molecules")
        if isinstance(predicted, list) and len(predicted) == int(row.get("n") or 0):
            successful_predictions.add(row_id)

successful_hypotheses = set()
for line in hypotheses_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if not isinstance(row, dict) or row.get("id") is None or row.get("error"):
        continue
    hypotheses = row.get("hypotheses")
    reviewer = row.get("reviewer_output")
    valid_hypotheses = (
        isinstance(hypotheses, list)
        and (
            len(hypotheses) == 3
            if task == "mfp"
            else len(hypotheses) >= 1
        )
    )
    if valid_hypotheses and isinstance(reviewer, dict) and reviewer:
        successful_hypotheses.add(str(row["id"]))

complete = expected & successful_predictions & successful_hypotheses
raise SystemExit(0 if expected and expected <= complete else 1)
PY
}

evaluation_complete() {
  local task="$1"
  local summary_path="$2"
  "${PYTHON_BIN}" - "${task}" "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

task = sys.argv[1]
path = Path(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
try:
    summary = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

if task == "mfp":
    complete = (
        summary.get("total_gold") == 71
        and summary.get("total_predictions") == 71
        and summary.get("matched_ids") == 71
        and summary.get("missing_predictions") == 0
        and summary.get("extra_predictions") == 0
        and summary.get("parse_failures") == 0
        and summary.get("gold_category_lookup_failures") == 0
        and summary.get("llm_mapping_failures") == 0
    )
elif task == "mpc":
    complete = (
        summary.get("total_gold") == 71
        and summary.get("samples_evaluated") == 71
        and summary.get("matched_ids") == 71
        and summary.get("missing_predictions") == 0
        and summary.get("extra_predictions") == 0
        and summary.get("parse_failures") == 0
    )
else:
    complete = False

raise SystemExit(0 if complete else 1)
PY
}

compact_predictions() {
  local task="$1"
  local test_path="$2"
  local prediction_path="$3"
  "${PYTHON_BIN}" - "${task}" "${test_path}" "${prediction_path}" <<'PY'
import json
import os
import sys
from pathlib import Path

task, test_value, prediction_value = sys.argv[1:4]
test_path = Path(test_value)
prediction_path = Path(prediction_value)
expected_rows = [
    json.loads(line)
    for line in test_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
latest_success = {}
for line in prediction_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if not isinstance(row, dict) or row.get("id") is None or row.get("error"):
        continue
    row_id = str(row["id"])
    if task == "mfp" and str(row.get("predicted_food") or "").strip():
        latest_success[row_id] = row
    if task == "mpc":
        predicted = row.get("predicted_molecules")
        if isinstance(predicted, list) and len(predicted) == int(row.get("n") or 0):
            latest_success[row_id] = row

ordered = []
for expected in expected_rows:
    row_id = str(expected["id"])
    if row_id not in latest_success:
        raise SystemExit(f"cannot compact incomplete prediction id: {row_id}")
    ordered.append(latest_success[row_id])

temporary = prediction_path.with_name(prediction_path.name + ".compact.tmp")
with temporary.open("w", encoding="utf-8") as handle:
    for row in ordered:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(prediction_path)
PY
}

compact_generation_metadata() {
  local task="$1"
  local test_path="$2"
  local evidence_path="$3"
  local retrieval_path="$4"
  local hypotheses_path="$5"
  "${PYTHON_BIN}" - \
    "${task}" \
    "${test_path}" \
    "${evidence_path}" \
    "${retrieval_path}" \
    "${hypotheses_path}" <<'PY'
import json
import os
import sys
from pathlib import Path

task = sys.argv[1]
test_path = Path(sys.argv[2])
metadata_paths = {
    "evidence": Path(sys.argv[3]),
    "retrieval": Path(sys.argv[4]),
    "hypotheses": Path(sys.argv[5]),
}
expected_ids = [
    str(json.loads(line)["id"])
    for line in test_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict) and row.get("id") is not None:
                rows.append(row)
    return rows

selected = {}
for kind, path in metadata_paths.items():
    if not path.is_file():
        raise SystemExit(f"missing formal metadata: {path}")
    latest = {}
    for row in read_rows(path):
        row_id = str(row["id"])
        if kind == "hypotheses":
            hypotheses = row.get("hypotheses")
            successful = (
                not row.get("error")
                and isinstance(hypotheses, list)
                and (len(hypotheses) == 3 if task == "mfp" else len(hypotheses) >= 1)
                and isinstance(row.get("reviewer_output"), dict)
                and bool(row["reviewer_output"])
            )
            if successful:
                latest[row_id] = row
        else:
            latest[row_id] = row
    missing = [row_id for row_id in expected_ids if row_id not in latest]
    if missing:
        raise SystemExit(
            f"incomplete formal {kind} metadata; missing ids: {', '.join(missing[:10])}"
        )
    selected[kind] = [latest[row_id] for row_id in expected_ids]

for kind, path in metadata_paths.items():
    temporary = path.with_name(path.name + ".compact.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected[kind]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
PY
}

run_generation() {
  local task="$1"
  local train_path="$2"
  local test_path="$3"
  local evidence_path="$4"
  local result_dir="$5"
  local output="${result_dir}/predictions.jsonl"

  while ! generation_complete \
    "${task}" \
    "${test_path}" \
    "${output}" \
    "${result_dir}/hypotheses_metadata.jsonl"; do
    local resume_args=()
    if [[ -f "${output}" ]]; then
      resume_args=(--resume)
    fi
    local task_args=()
    if [[ "${task}" == "mfp" ]]; then
      task_args=(--icl-retrieval-metadata "${MFP_ICL_RETRIEVAL}")
    fi
    if ! "${PYTHON_BIN}" "${CODE_DIR}/optimized_agent.py" \
      --task "${task}" \
      --train "${train_path}" \
      --test "${test_path}" \
      --db "${DB}" \
      --evidence "${evidence_path}" \
      --unimol-embeddings "${UNIMOL_EMBEDDINGS}" \
      --ablation full \
      --output "${output}" \
      --evidence-metadata "${result_dir}/evidence_metadata.jsonl" \
      --retrieval-metadata "${result_dir}/retrieval_metadata.jsonl" \
      --hypotheses-metadata "${result_dir}/hypotheses_metadata.jsonl" \
      --llm-provider deepseek \
      --llm-model "${MODEL}" \
      --use-llm \
      ${task_args[@]+"${task_args[@]}"} \
      ${resume_args[@]+"${resume_args[@]}"}; then
      echo "OPTIMIZED_${task}: generation stopped after a fatal error." >&2
      return 1
    fi
    if ! generation_complete \
      "${task}" \
      "${test_path}" \
      "${output}" \
      "${result_dir}/hypotheses_metadata.jsonl"; then
      echo "OPTIMIZED_${task}: incomplete; retrying in ${RETRY_SECONDS} seconds."
      sleep "${RETRY_SECONDS}"
    fi
  done
  compact_predictions "${task}" "${test_path}" "${output}"
  compact_generation_metadata \
    "${task}" \
    "${test_path}" \
    "${result_dir}/evidence_metadata.jsonl" \
    "${result_dir}/retrieval_metadata.jsonl" \
    "${result_dir}/hypotheses_metadata.jsonl"
}

run_evaluation() {
  local task="$1"
  local test_path="$2"
  local result_dir="$3"
  local summary="${result_dir}/evaluation_summary.json"

  while ! evaluation_complete "${task}" "${summary}"; do
    if ! "${PYTHON_BIN}" "${CODE_DIR}/evaluation.py" \
      --task "${task}" \
      --gold "${test_path}" \
      --pred "${result_dir}/predictions.jsonl" \
      --db "${DB}" \
      --use-llm \
      --llm-provider deepseek \
      --llm-model "${MODEL}" \
      --save-details "${result_dir}/evaluation_details.jsonl" \
      --save-summary-json "${summary}"; then
      echo "OPTIMIZED_${task}_EVALUATION: stopped after a fatal error." >&2
      return 1
    fi
    if ! evaluation_complete "${task}" "${summary}"; then
      echo "OPTIMIZED_${task}_EVALUATION: incomplete; retrying in ${RETRY_SECONDS} seconds."
      sleep "${RETRY_SECONDS}"
    fi
  done
}

for required in \
  "${CODE_DIR}/optimized_agent.py" \
  "${CODE_DIR}/evaluation.py" \
  "${DB}" \
  "${UNIMOL_EMBEDDINGS}" \
  "${MFP_TRAIN}" \
  "${MFP_TEST}" \
  "${MPC_TRAIN}" \
  "${MPC_TEST}" \
  "${MFP_EVIDENCE}" \
  "${MPC_EVIDENCE}" \
  "${MFP_ICL_RETRIEVAL}"; do
  require_file "${required}"
done

MFP_RESULT="${RESULTS_ROOT}/mfp/${MODEL}"
MPC_RESULT="${RESULTS_ROOT}/mpc/${MODEL}"
mkdir -p "${MFP_RESULT}" "${MPC_RESULT}"

if [[ "${TASK_FILTER}" == "all" || "${TASK_FILTER}" == "mfp" ]]; then
  initialize_method_result "mfp" "${MFP_RESULT}"
  run_generation "mfp" "${MFP_TRAIN}" "${MFP_TEST}" "${MFP_EVIDENCE}" "${MFP_RESULT}"
  run_evaluation "mfp" "${MFP_TEST}" "${MFP_RESULT}"
  finalize_run_metadata \
    "mfp" "${MFP_TRAIN}" "${MFP_TEST}" "${MFP_EVIDENCE}" "${MFP_RESULT}"
  echo "OPTIMIZED_MFP: COMPLETE"
fi

if [[ "${TASK_FILTER}" == "all" || "${TASK_FILTER}" == "mpc" ]]; then
  initialize_method_result "mpc" "${MPC_RESULT}"
  run_generation "mpc" "${MPC_TRAIN}" "${MPC_TEST}" "${MPC_EVIDENCE}" "${MPC_RESULT}"
  run_evaluation "mpc" "${MPC_TEST}" "${MPC_RESULT}"
  finalize_run_metadata \
    "mpc" "${MPC_TRAIN}" "${MPC_TEST}" "${MPC_EVIDENCE}" "${MPC_RESULT}"
  echo "OPTIMIZED_MPC: COMPLETE"
fi
echo "OPTIMIZED_AGENT_AUTOMATION: COMPLETE"
