#!/usr/bin/env python3
"""Heterogeneous, task-isolated multi-agent route for FoodPuzzle.

The implementation deliberately keeps MFP and MPC independent.  Agents are
defined by private observations and constrained actions rather than by persona
prompts alone:

* retrieval agents build fixed candidate spaces;
* occurrence agents use training-split associations only;
* structure agents use frozen UniMol representations and molecule-local fields;
* evidence critics may audit official offline evidence but cannot invent items;
* deterministic arbiters and set planners enforce the final protocol.

The prediction process never accepts the official functional-group evaluation
cache.  That cache belongs exclusively to the later evaluation subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "multi_agent_v1"
METHOD_VERSION = "heterogeneous_multi_agent_single_conformer_v1"
MFP_CANDIDATE_COUNT = 7
MFP_OCCURRENCE_WEIGHT = 0.55
MFP_STRUCTURE_WEIGHT = 0.30
MFP_EVIDENCE_WEIGHT = 0.15
MPC_PRIMARY_POOL = 100
MPC_RESCUE_POOL = 300
MPC_UNARY_WEIGHT = 1.0
MPC_COVERAGE_WEIGHT = 0.12
MPC_REDUNDANCY_WEIGHT = 0.04
MPC_EVIDENCE_WEIGHT = 0.03
MPC_SWAP_MARGIN = 0.025
MPC_NNPU_EPOCHS = 80
MPC_NNPU_UNLABELED_PER_ROW = 48
MPC_NNPU_LEARNING_RATE = 0.08
MPC_NNPU_L2 = 0.01
MPC_MAX_AGENT_CALLS = 1
MFP_MAX_AGENT_CALLS = 2

MFP_MACRO_CATEGORIES = [
    "cereal",
    "fruit",
    "essentialoil",
    "plant",
    "bakery",
    "fungus",
    "seed",
    "dish",
    "spice",
    "flower",
    "nutseed",
    "beverage",
    "animalproduct",
    "vegetable",
    "plantderivative",
    "additive",
    "meat",
    "fishseafood",
    "cerealcrop",
    "dairy",
    "herb",
]

DIRECT_OCCURRENCE_CUES = (
    "found in",
    "present in",
    "contains",
    "contained in",
    "detected in",
    "identified in",
    "isolated from",
    "component of",
    "constituent of",
    "occurs in",
    "reported in",
)
SENSORY_CUES = (
    "odor",
    "odour",
    "aroma",
    "flavor",
    "flavour",
    "note",
    "smell",
    "reminiscent",
    "characterized by",
)
FUNCTIONAL_ROLE_CUES = (
    "contributes",
    "contribute",
    "responsible for",
    "gives",
    "imparts",
    "odorant",
    "flavoring",
    "flavouring",
    "key aroma",
)
CONTRADICTION_CUES = (
    "does not contain",
    "not present",
    "not detected",
    "no evidence",
    "unable to verify",
    "cannot verify",
)


class MultiAgentError(Exception):
    """Expected multi-agent pipeline failure."""


def normalize(value: Any) -> str:
    text = str(value or "").lower().strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^0-9a-z+\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def stable_unique(values: Iterable[Any], excluded: set[str] | None = None) -> list[str]:
    blocked = excluded or set()
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        key = normalize(item)
        if not item or not key or key in blocked or key in seen:
            continue
        output.append(item)
        seen.add(key)
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MultiAgentError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict) or row.get("id") is None:
                raise MultiAgentError(f"invalid JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_sibling_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MultiAgentError(f"cannot import sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_json_object(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    candidates = [text]
    fence = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        candidates.insert(0, fence.group(1).strip())
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def agent_envelope(
    task: str,
    sample_id: str,
    agent: str,
    status: str,
    input_payload: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if status not in {"ok", "abstain", "invalid"}:
        raise MultiAgentError(f"invalid agent status: {status}")
    encoded = json.dumps(
        input_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "sample_id": sample_id,
        "agent": agent,
        "status": status,
        "input_digest": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def validate_split(
    train_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    task: str,
) -> None:
    if not train_rows or not query_rows:
        raise MultiAgentError("train and query splits must be non-empty")
    train_ids = {str(row["id"]) for row in train_rows}
    query_ids = {str(row["id"]) for row in query_rows}
    if train_ids & query_ids:
        raise MultiAgentError("train and query IDs overlap")
    train_required = (
        {"molecules", "actual_food"}
        if task == "mfp"
        else {"target_food", "partial_molecules", "n", "missing_molecules"}
    )
    query_required = (
        {"molecules"}
        if task == "mfp"
        else {"target_food", "partial_molecules", "n"}
    )
    for source, rows, required in (
        ("train", train_rows, train_required),
        ("query", query_rows, query_required),
    ):
        for row in rows:
            missing = [field for field in required if field not in row]
            if missing:
                raise MultiAgentError(
                    f"{source} row {row['id']} missing fields: {missing}"
                )
    if task == "mpc":
        for row in train_rows:
            if not isinstance(row.get("missing_molecules"), list):
                raise MultiAgentError(
                    f"MPC train row {row['id']} lacks missing_molecules labels"
                )


def validate_prediction_paths(args: argparse.Namespace) -> None:
    paths = [
        Path(args.output),
        Path(args.agent_metadata),
        Path(args.retrieval_metadata),
        Path(args.evidence_metadata),
    ]
    forbidden_tokens = (
        "optimized-agent",
        "functional_group_cache",
        "evaluation_details",
        "evaluation_summary",
    )
    for path in paths:
        lowered = str(path).lower()
        if any(token in lowered for token in forbidden_tokens):
            raise MultiAgentError(f"forbidden prediction output path: {path}")
        expected = (
            Path("results")
            / "Only-Deepseek"
            / "multi-agent"
            / str(args.task)
            / "deepseek-v4-flash"
        )
        try:
            path.resolve().relative_to((Path.cwd() / expected).resolve())
        except ValueError as exc:
            raise MultiAgentError(
                f"formal output must stay under {expected}: {path}"
            ) from exc
        if not path.parent.is_dir():
            raise MultiAgentError(f"output parent does not exist: {path.parent}")
        if path.exists() and not args.resume:
            raise MultiAgentError(f"output exists; use --resume: {path}")


def successful_prediction_ids(path: Path, task: str) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        if row.get("error"):
            continue
        if task == "mfp" and str(row.get("predicted_food") or "").strip():
            completed.add(str(row["id"]))
        if task == "mpc" and isinstance(row.get("predicted_molecules"), list):
            if len(row["predicted_molecules"]) == int(row.get("n") or -1):
                completed.add(str(row["id"]))
    return completed


def existing_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in read_jsonl(path)} if path.is_file() else set()


class EmbeddingStore:
    """Frozen single-conformer UniMol representation store."""

    def __init__(self, path: Path):
        try:
            import numpy as np
        except Exception as exc:
            raise MultiAgentError("numpy is required for UniMol features") from exc
        if not path.is_file():
            raise MultiAgentError(f"UniMol embeddings not found: {path}")
        data = np.load(path, allow_pickle=False)
        if "names" not in data:
            raise MultiAgentError("UniMol NPZ is missing names")
        names = [str(value) for value in data["names"].tolist()]
        if "embeddings" not in data:
            raise MultiAgentError("single-conformer UniMol NPZ needs embeddings")
        matrix = np.asarray(data["embeddings"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(names):
            raise MultiAgentError("invalid single-conformer embeddings shape")
        conformers = matrix[:, None, :]
        norms = np.linalg.norm(conformers, axis=2, keepdims=True)
        self.conformers = conformers / np.maximum(norms, 1e-12)
        self.names = names
        self.index = {normalize(name): index for index, name in enumerate(names)}
        self.dimension = int(conformers.shape[2])
        self.conformer_count = 1
        self.np = np

    def available(self, values: Iterable[Any]) -> list[str]:
        return [str(value) for value in values if normalize(value) in self.index]

    def molecule_vector(self, value: Any) -> Any | None:
        index = self.index.get(normalize(value))
        if index is None:
            return None
        conformers = self.conformers[index]
        mean = conformers.mean(axis=0)
        norm = float(self.np.linalg.norm(mean))
        return mean / max(norm, 1e-12)

    def molecule_features(self, value: Any) -> Any | None:
        index = self.index.get(normalize(value))
        if index is None:
            return None
        conformers = self.conformers[index]
        return self.np.concatenate([conformers.mean(axis=0), conformers.std(axis=0)])

    def set_features(
        self,
        values: Iterable[Any],
        weights: dict[str, float] | None = None,
    ) -> Any:
        vectors: list[Any] = []
        vector_weights: list[float] = []
        for value in values:
            vector = self.molecule_vector(value)
            if vector is None:
                continue
            vectors.append(vector)
            vector_weights.append((weights or {}).get(normalize(value), 1.0))
        if not vectors:
            return self.np.zeros(self.dimension * 2, dtype=self.np.float32)
        matrix = self.np.asarray(vectors, dtype=self.np.float32)
        mean = self.np.average(
            matrix,
            axis=0,
            weights=self.np.asarray(vector_weights),
        )
        spread = matrix.std(axis=0)
        return self.np.concatenate([mean, spread])

    def conditional_similarity(
        self,
        candidate: Any,
        known_values: Iterable[Any],
    ) -> tuple[float, float, float]:
        candidate_vector = self.molecule_vector(candidate)
        known = [
            self.molecule_vector(value)
            for value in known_values
            if self.molecule_vector(value) is not None
        ]
        if candidate_vector is None or not known:
            return 0.0, 0.0, 0.0
        similarities = self.np.asarray(known) @ candidate_vector
        mean_value = float(similarities.mean())
        max_value = float(similarities.max())
        attention = self.np.exp(
            (similarities - similarities.max()) / 0.10
        )
        attention /= max(float(attention.sum()), 1e-12)
        attended = float((attention * similarities).sum())
        return mean_value, max_value, attended

    def cosine(self, left: Any, right: Any) -> float:
        left_vector = self.molecule_vector(left)
        right_vector = self.molecule_vector(right)
        if left_vector is None or right_vector is None:
            return 0.0
        return float(left_vector @ right_vector)


class BM25Index:
    def __init__(self, rows: list[dict[str, Any]], task: str):
        self.rows = rows
        self.task = task
        self.documents = [self._tokens(row) for row in rows]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        total = max(1, len(rows))
        self.idf = {
            token: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for token, count in document_frequency.items()
        }

    def _tokens(self, row: dict[str, Any]) -> list[str]:
        if self.task == "mfp":
            return [
                normalize(value)
                for value in row.get("molecules") or []
                if normalize(value)
            ]
        output = [
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        ]
        output.extend(normalize(row.get("target_food")).split())
        return output

    def retrieve(self, row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        query = Counter(self._tokens(row))
        scores: list[tuple[float, int]] = []
        for index, document in enumerate(self.documents):
            frequencies = Counter(document)
            score = 0.0
            for token, query_frequency in query.items():
                frequency = frequencies[token]
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * len(document) / max(self.average_length, 1e-12)
                )
                score += (
                    self.idf.get(token, 0.0)
                    * query_frequency
                    * frequency
                    * 2.5
                    / denominator
                )
            scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": str(self.rows[index]["id"]),
                "rank": rank,
                "score": round(score, 8),
                "row": self.rows[index],
            }
            for rank, (score, index) in enumerate(scores[:top_k], 1)
        ]


def load_food_categories(db_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT category, entity_alias_readable, entity_alias,
                   entity_alias_basket, entity_alias_synonyms
            FROM food_entities
            """
        ).fetchall()
    for category, readable, alias, basket, synonyms in rows:
        if not category:
            continue
        label = normalize(str(category).split("-")[0]).replace(" ", "")
        for value in (readable, alias, synonyms):
            key = normalize(value)
            if key:
                mapping[key] = label
        for value in str(basket or "").split(","):
            key = normalize(value)
            if key:
                mapping[key] = label
    return mapping


def molecule_idf(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    frequencies: Counter[str] = Counter()
    for row in rows:
        frequencies.update(
            {
                normalize(value)
                for value in row.get(field) or []
                if normalize(value)
            }
        )
    total = max(1, len(rows))
    return {
        value: math.log(1.0 + (total + 1.0) / (count + 1.0))
        for value, count in frequencies.items()
    }


def load_molecule_descriptors(
    db_path: Path,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    names: dict[str, str] = {}
    attributes: dict[str, set[str]] = {}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT common_name, functional_groups, fooddb_flavor_profile,
                   flavor_profile, fema_flavor_profile, odor, taste
            FROM molecules
            """
        ).fetchall()
    for common_name, groups, fooddb, flavor, fema, odor, taste in rows:
        key = normalize(common_name)
        if not key:
            continue
        names[key] = str(common_name)
        tokens: set[str] = set()
        for prefix, value in (
            ("group", groups),
            ("flavor", fooddb),
            ("flavor", flavor),
            ("flavor", fema),
            ("odor", odor),
            ("taste", taste),
        ):
            for item in re.split(r"[@,;/|]+", str(value or "")):
                cleaned = normalize(item)
                if cleaned and len(cleaned.split()) <= 6:
                    tokens.add(f"{prefix}:{cleaned}")
        attributes[key] = tokens
    return names, attributes


def load_evidence(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise MultiAgentError("official evidence must be a dictionary")
    output: dict[str, list[str]] = {}
    for key, value in obj.items():
        if isinstance(value, list):
            output[normalize(key)] = [
                json.dumps(item, ensure_ascii=False)
                if isinstance(item, dict)
                else str(item)
                for item in value
            ]
        elif isinstance(value, dict):
            output[normalize(key)] = [json.dumps(value, ensure_ascii=False)]
        elif value is not None:
            output[normalize(key)] = [str(value)]
    return output


def evidence_relation(snippet: str) -> str:
    text = normalize(snippet)
    if any(cue in text for cue in CONTRADICTION_CUES):
        return "contradiction"
    if any(cue in text for cue in DIRECT_OCCURRENCE_CUES):
        return "direct_occurrence"
    if any(cue in text for cue in FUNCTIONAL_ROLE_CUES):
        return "functional_role"
    if any(cue in text for cue in SENSORY_CUES):
        return "sensory_property"
    return "ambiguous"


def link_candidate_ids(
    snippet: str,
    candidates: list[dict[str, Any]],
) -> list[str]:
    text = f" {normalize(snippet)} "
    linked: list[str] = []
    for candidate in candidates:
        name = candidate.get("molecule") or candidate.get("category")
        key = normalize(name)
        if len(key) >= 3 and f" {key} " in text:
            linked.append(str(candidate["candidate_id"]))
    return linked


def call_structured_llm(
    messages: list[dict[str, str]],
    llm_config: dict[str, str],
    evaluation: Any,
) -> tuple[str, dict[str, int]]:
    """Call the existing DeepSeek provider and retain only safe usage counts."""
    use_response_format = True
    use_thinking = llm_config["provider"] == "deepseek"
    use_temperature = True
    while True:
        payload = evaluation.build_chat_payload(
            messages,
            llm_config,
            use_response_format=use_response_format,
            use_thinking=use_thinking,
            use_temperature=use_temperature,
        )
        try:
            body = evaluation.post_chat_payload(payload, llm_config)
            break
        except evaluation.ChatCompletionHTTPError as exc:
            if use_thinking and evaluation.is_thinking_compat_error(exc):
                use_thinking = False
                continue
            if (
                use_response_format
                and evaluation.is_response_format_compat_error(exc)
            ):
                use_response_format = False
                continue
            if use_temperature and evaluation.is_temperature_compat_error(exc):
                use_temperature = False
                continue
            raise
    try:
        content = str(body["choices"][0]["message"]["content"])
    except Exception as exc:
        raise MultiAgentError("provider response has unexpected shape") from exc
    usage_raw = body.get("usage") if isinstance(body, dict) else None
    usage = {
        key: int((usage_raw or {}).get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return content, usage


class MFPAgents:
    """MFP-only retrieval, occurrence, structure, evidence, and arbitration."""

    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        db_path: Path,
        embeddings: EmbeddingStore,
    ):
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            raise MultiAgentError("numpy and scikit-learn are required") from exc
        self.np = np
        self.train_rows = train_rows
        self.categories = load_food_categories(db_path)
        self.embeddings = embeddings
        self.idf = molecule_idf(train_rows, "molecules")
        self.bm25 = BM25Index(train_rows, "mfp")
        labels = [
            self.categories.get(normalize(row.get("actual_food")))
            for row in train_rows
        ]
        if any(label not in MFP_MACRO_CATEGORIES for label in labels):
            raise MultiAgentError("MFP train food-to-category mapping failed")
        self.labels = [str(label) for label in labels]
        documents = [
            [normalize(value) for value in row.get("molecules") or [] if normalize(value)]
            for row in train_rows
        ]
        self.vectorizer = TfidfVectorizer(
            analyzer=lambda values: values,
            lowercase=False,
            token_pattern=None,
            sublinear_tf=True,
            norm="l2",
        )
        sparse = self.vectorizer.fit_transform(documents)
        self.occurrence_classifier = LogisticRegression(
            C=0.35,
            class_weight="balanced",
            max_iter=2000,
            random_state=20260727,
        ).fit(sparse, self.labels)
        structure_features = np.asarray(
            [
                embeddings.set_features(row.get("molecules") or [], self.idf)
                for row in train_rows
            ],
            dtype=np.float32,
        )
        self.structure_classifier = LogisticRegression(
            C=0.05,
            class_weight="balanced",
            max_iter=2000,
            random_state=20260727,
        ).fit(structure_features, self.labels)

    @staticmethod
    def _aligned(
        classifier: Any,
        features: Any,
    ) -> dict[str, float]:
        probabilities = classifier.predict_proba(features)[0]
        return {
            str(label): float(probabilities[index])
            for index, label in enumerate(classifier.classes_)
        }

    def candidate_state(
        self,
        row: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        molecules = row.get("molecules") or []
        occurrence = self._aligned(
            self.occurrence_classifier,
            self.vectorizer.transform(
                [[normalize(value) for value in molecules if normalize(value)]]
            ),
        )
        structure = self._aligned(
            self.structure_classifier,
            self.np.asarray(
                [self.embeddings.set_features(molecules, self.idf)],
                dtype=self.np.float32,
            ),
        )
        demonstrations = self.bm25.retrieve(row, 5)
        demonstration_votes: Counter[str] = Counter()
        for item in demonstrations:
            category = self.categories.get(
                normalize(item["row"].get("actual_food"))
            )
            if category:
                demonstration_votes[category] += max(
                    0.0, float(item["score"])
                )
        vote_scale = max(demonstration_votes.values(), default=1.0)
        combined = {
            category: (
                0.65 * occurrence.get(category, 0.0)
                + 0.25 * structure.get(category, 0.0)
                + 0.10 * demonstration_votes[category] / vote_scale
            )
            for category in MFP_MACRO_CATEGORIES
        }
        ordered = sorted(
            MFP_MACRO_CATEGORIES,
            key=lambda category: (-combined[category], category),
        )[:MFP_CANDIDATE_COUNT]
        candidates = [
            {
                "candidate_id": f"C{index:02d}",
                "category": category,
                "occurrence_score": round(occurrence.get(category, 0.0), 8),
                "structure_score": round(structure.get(category, 0.0), 8),
                "retrieval_vote": round(
                    demonstration_votes[category] / vote_scale, 8
                ),
            }
            for index, category in enumerate(ordered, 1)
        ]
        rare_molecules = sorted(
            stable_unique(molecules),
            key=lambda value: (-self.idf.get(normalize(value), 0.0), normalize(value)),
        )[:12]
        diagnostics = {
            "mapped_unimol_count": len(self.embeddings.available(molecules)),
            "input_molecule_count": len(molecules),
            "conformer_count": self.embeddings.conformer_count,
            "rare_molecules": rare_molecules,
        }
        return candidates, diagnostics, demonstrations

    def occurrence_report(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        demonstrations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "candidate_assessments": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "verdict": "support",
                    "confidence": candidate["occurrence_score"],
                    "reason_codes": ["train_occurrence_model", "bm25_retrieval"],
                    "evidence_refs": [
                        f"D{item['rank']}"
                        for item in demonstrations
                        if self.categories.get(
                            normalize(item["row"].get("actual_food"))
                        )
                        == candidate["category"]
                    ],
                }
                for candidate in candidates
            ],
            "abstain_reason": None,
        }
        return agent_envelope(
            "mfp",
            str(row["id"]),
            "occurrence_scientist",
            "ok",
            {
                "molecules": row.get("molecules") or [],
                "candidate_ids": [item["candidate_id"] for item in candidates],
                "demo_ids": [item["id"] for item in demonstrations],
            },
            payload,
        )

    def structure_messages(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        diagnostics: dict[str, Any],
    ) -> list[dict[str, str]]:
        private_view = [
            {
                "candidate_id": item["candidate_id"],
                "category": item["category"],
                "structure_score": item["structure_score"],
            }
            for item in candidates
        ]
        prompt = (
            "FoodPuzzle MFP structure-only audit. Infer support for fixed macro-category IDs "
            "from a complete molecule set. You do not have occurrence frequencies, training "
            "labels, demonstrations, evidence snippets, or another agent's answer. Structural "
            "similarity is suggestive rather than proof of natural occurrence.\n"
            f"Input molecules: {json.dumps(row.get('molecules') or [], ensure_ascii=False)}\n"
            f"Rare structural anchors: {json.dumps(diagnostics['rare_molecules'], ensure_ascii=False)}\n"
            f"Fixed candidates: {json.dumps(private_view, ensure_ascii=False)}\n"
            "Return JSON only: "
            '{"candidate_assessments":[{"candidate_id":"C01",'
            '"verdict":"support|oppose|unknown","confidence":0.0,'
            '"reason_codes":["structure_pattern"],"evidence_refs":[]}],'
            '"abstain_reason":null}. Assess every candidate ID exactly once.'
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are the structure-and-sensory specialist. "
                    "Use only the private structural view and return valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def evidence_messages(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        evidence_rows: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        anonymous_candidates = [
            {"candidate_id": item["candidate_id"], "category": item["category"]}
            for item in candidates
        ]
        prompt = (
            "FoodPuzzle MFP evidence audit. Audit the supplied offline evidence; do not use "
            "outside knowledge and do not select a final answer. Odor resemblance is not proof "
            "that a molecule occurs in a food. Ambiguous evidence must remain ambiguous.\n"
            f"Fixed candidates: {json.dumps(anonymous_candidates, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence_rows, ensure_ascii=False)}\n"
            "Return JSON only: "
            '{"candidate_assessments":[{"candidate_id":"C01",'
            '"verdict":"support|oppose|unknown","confidence":0.0,'
            '"reason_codes":["direct_occurrence"],"evidence_refs":["E01"]}],'
            '"abstain_reason":null}. Assess every candidate ID exactly once.'
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are an evidence critic with no authority to invent "
                    "evidence or candidates. Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]


def validate_candidate_assessments(
    data: dict[str, Any] | None,
    candidate_ids: list[str],
) -> dict[str, Any]:
    if not data or not isinstance(data.get("candidate_assessments"), list):
        return {
            "candidate_assessments": [],
            "abstain_reason": "invalid_json",
        }
    by_id: dict[str, dict[str, Any]] = {}
    for item in data["candidate_assessments"]:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip().upper()
        verdict = str(item.get("verdict") or "unknown").strip().lower()
        if candidate_id not in candidate_ids or verdict not in {
            "support",
            "oppose",
            "unknown",
        }:
            continue
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "verdict": verdict,
            "confidence": confidence,
            "reason_codes": stable_unique(item.get("reason_codes") or [])[:8],
            "evidence_refs": stable_unique(item.get("evidence_refs") or [])[:12],
        }
    if set(by_id) != set(candidate_ids):
        return {
            "candidate_assessments": list(by_id.values()),
            "abstain_reason": "incomplete_candidate_assessment",
        }
    return {
        "candidate_assessments": [by_id[value] for value in candidate_ids],
        "abstain_reason": None,
    }


def assessment_scores(report: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in report.get("candidate_assessments") or []:
        sign = {
            "support": 1.0,
            "oppose": -1.0,
            "unknown": 0.0,
        }.get(str(item.get("verdict")), 0.0)
        scores[str(item.get("candidate_id"))] = sign * float(
            item.get("confidence") or 0.0
        )
    return scores


def finalize_mfp(
    candidates: list[dict[str, Any]],
    occurrence_report: dict[str, Any],
    structure_report: dict[str, Any],
    evidence_report: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    occurrence = {
        item["candidate_id"]: float(item["occurrence_score"])
        for item in candidates
    }
    structure_llm = (
        {}
        if structure_report.get("abstain_reason")
        else assessment_scores(structure_report)
    )
    evidence_llm = (
        {}
        if evidence_report.get("abstain_reason")
        else assessment_scores(evidence_report)
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        structure_score = float(candidate["structure_score"])
        if not structure_report.get("abstain_reason"):
            structure_score = (
                0.5 * structure_score
                + 0.5 * (structure_llm.get(candidate_id, 0.0) + 1.0) / 2.0
            )
        evidence_score = (evidence_llm.get(candidate_id, 0.0) + 1.0) / 2.0
        evidence_weight = (
            0.0
            if evidence_report.get("abstain_reason")
            else MFP_EVIDENCE_WEIGHT
        )
        total_weight = (
            MFP_OCCURRENCE_WEIGHT + MFP_STRUCTURE_WEIGHT + evidence_weight
        )
        score = (
            MFP_OCCURRENCE_WEIGHT * occurrence[candidate_id]
            + MFP_STRUCTURE_WEIGHT * structure_score
            + evidence_weight * evidence_score
        ) / max(total_weight, 1e-12)
        rows.append(
            {
                "candidate_id": candidate_id,
                "category": candidate["category"],
                "score": round(score, 8),
                "occurrence": round(occurrence[candidate_id], 8),
                "structure": round(structure_score, 8),
                "evidence": round(evidence_score, 8),
            }
        )
    rows.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    if not rows or rows[0]["category"] not in MFP_MACRO_CATEGORIES:
        raise MultiAgentError("MFP deterministic arbiter has no valid category")
    margin = (
        float(rows[0]["score"]) - float(rows[1]["score"])
        if len(rows) > 1
        else 1.0
    )
    return str(rows[0]["category"]), {
        "selection_method": "calibrated_deterministic_channel_stacking",
        "ranked_candidates": rows,
        "margin": round(margin, 8),
        "uncertain": margin < 0.05,
        "occurrence_agent_status": occurrence_report.get("status"),
        "structure_agent_abstained": bool(
            structure_report.get("abstain_reason")
        ),
        "evidence_agent_abstained": bool(
            evidence_report.get("abstain_reason")
        ),
    }


@dataclass
class LinearNNPU:
    weights: Any
    bias: float
    mean: Any
    scale: Any

    def predict(self, values: list[float]) -> float:
        vector = (self.weights * 0) + values
        normalized = (vector - self.mean) / self.scale
        return sigmoid(float(normalized @ self.weights + self.bias))


def fit_linear_nnpu(
    groups: list[tuple[Any, Any, float]],
    dimension: int,
    seed: int,
) -> LinearNNPU:
    import numpy as np

    all_values = np.concatenate(
        [array for positives, unlabeled, _ in groups for array in (positives, unlabeled)],
        axis=0,
    )
    mean = all_values.mean(axis=0)
    scale = np.maximum(all_values.std(axis=0), 1e-4)
    normalized_groups = [
        ((positives - mean) / scale, (unlabeled - mean) / scale, prior)
        for positives, unlabeled, prior in groups
    ]
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=dimension)
    bias = 0.0
    for epoch in range(MPC_NNPU_EPOCHS):
        order = rng.permutation(len(normalized_groups))
        step = MPC_NNPU_LEARNING_RATE / math.sqrt(epoch + 1.0)
        for group_index in order:
            positives, unlabeled, prior = normalized_groups[int(group_index)]
            positive_prob = 1.0 / (
                1.0 + np.exp(-np.clip(positives @ weights + bias, -30, 30))
            )
            unlabeled_prob = 1.0 / (
                1.0 + np.exp(-np.clip(unlabeled @ weights + bias, -30, 30))
            )
            positive_gradient = (
                ((positive_prob - 1.0)[:, None] * positives).mean(axis=0)
            )
            positive_bias = float((positive_prob - 1.0).mean())
            unlabeled_negative_gradient = (
                (unlabeled_prob[:, None] * unlabeled).mean(axis=0)
            )
            unlabeled_negative_bias = float(unlabeled_prob.mean())
            positive_as_negative_gradient = (
                (positive_prob[:, None] * positives).mean(axis=0)
            )
            positive_as_negative_bias = float(positive_prob.mean())
            unlabeled_negative_loss = float(
                (-np.log(np.maximum(1.0 - unlabeled_prob, 1e-8))).mean()
            )
            positive_as_negative_loss = float(
                (-np.log(np.maximum(1.0 - positive_prob, 1e-8))).mean()
            )
            negative_risk = (
                unlabeled_negative_loss - prior * positive_as_negative_loss
            )
            gradient = prior * positive_gradient
            bias_gradient = prior * positive_bias
            if negative_risk > 0.0:
                gradient += (
                    unlabeled_negative_gradient
                    - prior * positive_as_negative_gradient
                )
                bias_gradient += (
                    unlabeled_negative_bias
                    - prior * positive_as_negative_bias
                )
            gradient += MPC_NNPU_L2 * weights
            weights -= step * gradient
            bias -= step * bias_gradient
    return LinearNNPU(weights, bias, mean, scale)


class MPCAgents:
    """MPC-only PU rankers, evidence critic, set planner, and swap gate."""

    def __init__(
        self,
        train_rows: list[dict[str, Any]],
        db_path: Path,
        embeddings: EmbeddingStore,
    ):
        import numpy as np

        self.np = np
        self.train_rows = train_rows
        self.embeddings = embeddings
        self.names, self.attributes = load_molecule_descriptors(db_path)
        self.full_profiles: list[set[str]] = []
        self.frequency: Counter[str] = Counter()
        self.cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
        for row in train_rows:
            full = {
                normalize(value)
                for value in (
                    list(row.get("partial_molecules") or [])
                    + list(row.get("missing_molecules") or [])
                )
                if normalize(value)
            }
            self.full_profiles.append(full)
            self.frequency.update(full)
            for molecule in full:
                self.cooccurrence[molecule].update(full - {molecule})
        self.universe = sorted(self.names)
        self.bm25 = BM25Index(train_rows, "mpc")
        occurrence_groups: list[tuple[Any, Any, float]] = []
        structure_groups: list[tuple[Any, Any, float]] = []
        rng = random.Random(20260727)
        for row_index, row in enumerate(train_rows):
            positives = stable_unique(row.get("missing_molecules") or [])
            positive_keys = {
                normalize(value) for value in positives if normalize(value) in self.names
            }
            partial = {
                normalize(value)
                for value in row.get("partial_molecules") or []
                if normalize(value)
            }
            unlabeled = [
                value
                for value in self.universe
                if value not in partial and value not in positive_keys
            ]
            if not positive_keys or not unlabeled:
                continue
            rng.shuffle(unlabeled)
            unlabeled = unlabeled[:MPC_NNPU_UNLABELED_PER_ROW]
            retrieval_support = self._retrieval_support(
                row, exclude_index=row_index
            )
            occurrence_positive = np.asarray(
                [
                    self._occurrence_features(row, value, retrieval_support)
                    for value in sorted(positive_keys)
                ],
                dtype=np.float64,
            )
            occurrence_unlabeled = np.asarray(
                [
                    self._occurrence_features(row, value, retrieval_support)
                    for value in unlabeled
                ],
                dtype=np.float64,
            )
            structure_positive = np.asarray(
                [self._structure_features(row, value) for value in sorted(positive_keys)],
                dtype=np.float64,
            )
            structure_unlabeled = np.asarray(
                [self._structure_features(row, value) for value in unlabeled],
                dtype=np.float64,
            )
            candidate_count = max(1, len(self.universe) - len(partial))
            prior = min(0.5, max(1.0 / candidate_count, len(positive_keys) / candidate_count))
            occurrence_groups.append(
                (occurrence_positive, occurrence_unlabeled, prior)
            )
            structure_groups.append(
                (structure_positive, structure_unlabeled, prior)
            )
        if not occurrence_groups or not structure_groups:
            raise MultiAgentError("cannot construct MPC nnPU training groups")
        self.occurrence_ranker = fit_linear_nnpu(
            occurrence_groups, 3, 20260727
        )
        self.structure_ranker = fit_linear_nnpu(
            structure_groups, 4, 20260728
        )

    def _retrieval_support(
        self,
        row: dict[str, Any],
        exclude_index: int | None = None,
    ) -> dict[str, float]:
        partial = {
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        }
        food_tokens = set(normalize(row.get("target_food")).split())
        scored: list[tuple[float, int]] = []
        for index, (train_row, profile) in enumerate(
            zip(self.train_rows, self.full_profiles)
        ):
            if index == exclude_index:
                continue
            train_food = set(normalize(train_row.get("target_food")).split())
            food_overlap = len(food_tokens & train_food) / max(
                1, len(food_tokens | train_food)
            )
            molecule_overlap = len(partial & profile) / max(
                1, len(partial | profile)
            )
            scored.append((0.35 * food_overlap + 0.65 * molecule_overlap, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        support: dict[str, float] = {}
        for score, index in scored[:10]:
            for molecule in self.full_profiles[index] - partial:
                support[molecule] = max(support.get(molecule, 0.0), score)
        return support

    def _occurrence_features(
        self,
        row: dict[str, Any],
        candidate: str,
        retrieval_support: dict[str, float],
    ) -> list[float]:
        partial = {
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        }
        prior = self.frequency[candidate] / max(1, len(self.train_rows))
        cooccurrence = (
            sum(
                self.cooccurrence[candidate].get(existing, 0)
                / max(1, self.frequency[existing])
                for existing in partial
            )
            / len(partial)
            if partial
            else 0.0
        )
        return [prior, cooccurrence, retrieval_support.get(candidate, 0.0)]

    def _structure_features(
        self,
        row: dict[str, Any],
        candidate: str,
    ) -> list[float]:
        mean_value, max_value, attended = self.embeddings.conditional_similarity(
            candidate, row.get("partial_molecules") or []
        )
        partial_attributes: set[str] = set()
        for value in row.get("partial_molecules") or []:
            partial_attributes.update(self.attributes.get(normalize(value), set()))
        candidate_attributes = self.attributes.get(candidate, set())
        novelty = len(candidate_attributes - partial_attributes) / max(
            1, len(candidate_attributes)
        )
        return [mean_value, max_value, attended, novelty]

    def candidate_state(
        self,
        row: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        partial = {
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        }
        retrieval_support = self._retrieval_support(row)
        rows: list[dict[str, Any]] = []
        for candidate in self.universe:
            if candidate in partial:
                continue
            occurrence_features = self._occurrence_features(
                row, candidate, retrieval_support
            )
            structure_features = self._structure_features(row, candidate)
            occurrence_score = self.occurrence_ranker.predict(
                occurrence_features
            )
            structure_score = self.structure_ranker.predict(structure_features)
            rows.append(
                {
                    "molecule": self.names[candidate],
                    "key": candidate,
                    "occurrence_score": occurrence_score,
                    "structure_score": structure_score,
                    "retrieval_support": retrieval_support.get(candidate, 0.0),
                    "attributes": sorted(self.attributes.get(candidate, set())),
                }
            )
        occurrence_order = sorted(
            rows,
            key=lambda item: (-item["occurrence_score"], item["key"]),
        )
        structure_order = sorted(
            rows,
            key=lambda item: (-item["structure_score"], item["key"]),
        )
        union: dict[str, dict[str, Any]] = {}
        for item in occurrence_order[:MPC_RESCUE_POOL]:
            union[item["key"]] = item
        for item in structure_order[:MPC_RESCUE_POOL]:
            union[item["key"]] = item
        candidates = list(union.values())
        for item in candidates:
            item["unary_score"] = (
                0.62 * item["occurrence_score"]
                + 0.38 * item["structure_score"]
            )
        candidates.sort(
            key=lambda item: (-item["unary_score"], item["key"])
        )
        candidates = candidates[:MPC_RESCUE_POOL]
        for index, item in enumerate(candidates, 1):
            item["candidate_id"] = f"M{index:03d}"
            for field in (
                "occurrence_score",
                "structure_score",
                "retrieval_support",
                "unary_score",
            ):
                item[field] = round(float(item[field]), 8)
        diagnostics = {
            "candidate_universe_size": len(rows),
            "candidate_pool_size": len(candidates),
            "primary_pool_size": min(MPC_PRIMARY_POOL, len(candidates)),
            "conformer_count": self.embeddings.conformer_count,
            "nnpu": True,
            "known_molecule_count": len(partial),
        }
        demonstrations = self.bm25.retrieve(row, 5)
        return candidates, diagnostics, demonstrations

    def deterministic_evidence_report(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        snippets: list[str],
    ) -> dict[str, Any]:
        claims: list[dict[str, Any]] = []
        for index, snippet in enumerate(snippets[:12], 1):
            relation = evidence_relation(snippet)
            linked = link_candidate_ids(snippet, candidates)
            polarity = (
                "oppose"
                if relation == "contradiction"
                else "support"
                if relation in {
                    "direct_occurrence",
                    "sensory_property",
                    "functional_role",
                }
                else "neutral"
            )
            claims.append(
                {
                    "evidence_id": f"E{index:02d}",
                    "relation": relation,
                    "candidate_ids": linked,
                    "polarity": polarity,
                    "confidence": (
                        0.8
                        if relation == "direct_occurrence"
                        else 0.55
                        if relation in {"sensory_property", "functional_role"}
                        else 0.25
                    ),
                }
            )
        status = "ok" if claims else "abstain"
        return agent_envelope(
            "mpc",
            str(row["id"]),
            "evidence_critic",
            status,
            {
                "target_food": row.get("target_food"),
                "evidence_count": len(snippets),
                "candidate_ids": [
                    item["candidate_id"] for item in candidates
                ],
            },
            {
                "claims": claims,
                "invalid_evidence_ids": [],
                "abstain_reason": None if claims else "no_offline_evidence",
            },
        )

    def evidence_bonus(
        self,
        evidence_report: dict[str, Any],
    ) -> dict[str, float]:
        output: dict[str, float] = defaultdict(float)
        for claim in evidence_report.get("payload", {}).get("claims") or []:
            sign = -1.0 if claim.get("polarity") == "oppose" else 1.0
            if claim.get("polarity") == "neutral":
                sign = 0.0
            for candidate_id in claim.get("candidate_ids") or []:
                output[str(candidate_id)] += sign * float(
                    claim.get("confidence") or 0.0
                )
        return dict(output)

    def set_plan(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        evidence_report: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        requested_n = int(row.get("n") or 0)
        if requested_n <= 0:
            raise MultiAgentError("MPC requires positive n")
        excluded = {
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        }
        evidence = self.evidence_bonus(evidence_report)
        selected: list[dict[str, Any]] = []
        # Compute candidate-to-candidate UniMol similarities once.  The
        # previous equivalent implementation recomputed molecule vectors for
        # every greedy comparison, which becomes prohibitively slow for large
        # n.  Incremental maximum redundancy preserves the same objective and
        # deterministic tie-breaking in O(pool^2 + n*pool).
        candidate_vectors = self.np.zeros(
            (len(candidates), self.embeddings.dimension), dtype=self.np.float32
        )
        for index, item in enumerate(candidates):
            vector = self.embeddings.molecule_vector(item["molecule"])
            if vector is not None:
                candidate_vectors[index] = vector
        pairwise_similarity = self.np.maximum(
            0.0, candidate_vectors @ candidate_vectors.T
        )
        maximum_redundancy = self.np.zeros(
            len(candidates), dtype=self.np.float32
        )
        remaining_indices = list(range(len(candidates)))
        covered: set[str] = set()
        for value in row.get("partial_molecules") or []:
            covered.update(self.attributes.get(normalize(value), set()))
        while remaining_indices and len(selected) < requested_n:
            best_position = 0
            best_candidate_index = remaining_indices[0]
            best_key: tuple[float, float, str] | None = None
            for position, candidate_index in enumerate(remaining_indices):
                item = candidates[candidate_index]
                attributes = set(item["attributes"])
                coverage = len(attributes - covered) / max(1, len(attributes))
                redundancy = float(maximum_redundancy[candidate_index])
                score = (
                    MPC_UNARY_WEIGHT * float(item["unary_score"])
                    + MPC_COVERAGE_WEIGHT * coverage
                    - MPC_REDUNDANCY_WEIGHT * redundancy
                    + MPC_EVIDENCE_WEIGHT
                    * evidence.get(str(item["candidate_id"]), 0.0)
                )
                key = (score, float(item["unary_score"]), str(item["key"]))
                if best_key is None or (
                    key[0] > best_key[0] + 1e-12
                    or (
                        abs(key[0] - best_key[0]) <= 1e-12
                        and (
                            key[1] > best_key[1] + 1e-12
                            or (
                                abs(key[1] - best_key[1]) <= 1e-12
                                and key[2] < best_key[2]
                            )
                        )
                    )
                ):
                    best_key = key
                    best_position = position
                    best_candidate_index = candidate_index
            remaining_indices.pop(best_position)
            chosen = candidates[best_candidate_index]
            chosen["set_score"] = round(float(best_key[0]), 8)
            chosen["set_rank"] = len(selected) + 1
            selected.append(chosen)
            covered.update(chosen["attributes"])
            maximum_redundancy = self.np.maximum(
                maximum_redundancy,
                pairwise_similarity[:, best_candidate_index],
            )
        if len(selected) < requested_n:
            blocked = excluded | {normalize(item["molecule"]) for item in selected}
            for key in self.universe:
                if key in blocked:
                    continue
                selected.append(
                    {
                        "candidate_id": f"R{len(selected) + 1:03d}",
                        "molecule": self.names[key],
                        "key": key,
                        "unary_score": 0.0,
                        "structure_score": 0.0,
                        "occurrence_score": 0.0,
                        "set_score": 0.0,
                        "set_rank": len(selected) + 1,
                        "attributes": sorted(self.attributes.get(key, set())),
                    }
                )
                blocked.add(key)
                if len(selected) == requested_n:
                    break
        values = stable_unique(
            [item["molecule"] for item in selected],
            excluded,
        )[:requested_n]
        if len(values) != requested_n:
            raise MultiAgentError(
                f"MPC exact-n repair failed: {len(values)} != {requested_n}"
            )
        payload = {
            "selected_candidate_ids": [
                item["candidate_id"] for item in selected[:requested_n]
            ],
            "requested_n": requested_n,
            "objective": {
                "unary_weight": MPC_UNARY_WEIGHT,
                "coverage_weight": MPC_COVERAGE_WEIGHT,
                "redundancy_weight": MPC_REDUNDANCY_WEIGHT,
                "evidence_weight": MPC_EVIDENCE_WEIGHT,
            },
            "rescue_pool_size": len(candidates),
            "exact_n": len(values) == requested_n,
            "selected_rows": selected[:requested_n],
        }
        return values, payload

    def requires_swap_arbiter(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        plan: dict[str, Any],
    ) -> bool:
        requested_n = int(row.get("n") or 0)
        if requested_n <= 0 or requested_n >= len(candidates):
            return False
        selected_ids = set(plan.get("selected_candidate_ids") or [])
        selected = [
            item for item in candidates if item["candidate_id"] in selected_ids
        ]
        unselected = [
            item for item in candidates if item["candidate_id"] not in selected_ids
        ]
        if not selected or not unselected:
            return False
        cutoff_margin = min(
            float(item["unary_score"]) for item in selected
        ) - max(float(item["unary_score"]) for item in unselected)
        disagreements = sum(
            1
            for item in candidates[:MPC_PRIMARY_POOL]
            if abs(
                float(item["occurrence_score"])
                - float(item["structure_score"])
            )
            >= 0.30
        )
        if requested_n <= 5:
            return cutoff_margin <= MPC_SWAP_MARGIN and disagreements >= 2
        return cutoff_margin <= MPC_SWAP_MARGIN and disagreements >= 5

    def swap_messages(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        plan: dict[str, Any],
        evidence_report: dict[str, Any],
    ) -> list[dict[str, str]]:
        selected_ids = set(plan.get("selected_candidate_ids") or [])
        private_candidates = [
            {
                "candidate_id": item["candidate_id"],
                "selected": item["candidate_id"] in selected_ids,
                "occurrence_support": item["occurrence_score"],
                "structure_support": item["structure_score"],
                "retrieval_support": item["retrieval_support"],
            }
            for item in candidates[:MPC_PRIMARY_POOL]
        ]
        prompt = (
            "FoodPuzzle MPC one-for-one swap audit. You may propose swaps only among the "
            "provided IDs. Do not output molecule names, do not change cardinality, and do "
            "not infer a target from absent evidence. A swap should be proposed only when "
            "the incoming candidate has independent occurrence and structure support.\n"
            f"Target food: {row.get('target_food')}\n"
            f"Known partial molecule count: {len(row.get('partial_molecules') or [])}\n"
            f"n: {row.get('n')}\n"
            f"Candidates: {json.dumps(private_candidates, ensure_ascii=False)}\n"
            f"Evidence claims: {json.dumps(evidence_report.get('payload', {}).get('claims') or [], ensure_ascii=False)}\n"
            "Return JSON only: "
            '{"proposed_swaps":[{"out_id":"M001","in_id":"M087",'
            '"reason_codes":["two_channel_support"],"evidence_refs":[]}],'
            '"abstain":false}. Empty proposed_swaps is valid.'
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a bounded MPC swap arbiter. "
                    "You cannot create candidates or final predictions."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def apply_swaps(
        self,
        row: dict[str, Any],
        candidates: list[dict[str, Any]],
        values: list[str],
        plan: dict[str, Any],
        swap_data: dict[str, Any] | None,
    ) -> tuple[list[str], dict[str, Any]]:
        by_id = {str(item["candidate_id"]): item for item in candidates}
        selected_ids = list(plan.get("selected_candidate_ids") or [])
        accepted: list[dict[str, Any]] = []
        proposed = (
            swap_data.get("proposed_swaps")
            if isinstance(swap_data, dict)
            else None
        )
        if isinstance(proposed, list):
            for swap in proposed[:3]:
                if not isinstance(swap, dict):
                    continue
                out_id = str(swap.get("out_id") or "").upper()
                in_id = str(swap.get("in_id") or "").upper()
                if (
                    out_id not in selected_ids
                    or in_id in selected_ids
                    or out_id not in by_id
                    or in_id not in by_id
                ):
                    continue
                incoming = by_id[in_id]
                outgoing = by_id[out_id]
                two_channel = (
                    float(incoming["occurrence_score"])
                    >= float(outgoing["occurrence_score"])
                    and float(incoming["structure_score"])
                    >= float(outgoing["structure_score"])
                )
                unary_gain = (
                    float(incoming["unary_score"])
                    - float(outgoing["unary_score"])
                )
                if not two_channel or unary_gain < 0.0:
                    continue
                position = selected_ids.index(out_id)
                selected_ids[position] = in_id
                accepted.append(
                    {
                        "out_id": out_id,
                        "in_id": in_id,
                        "unary_gain": round(unary_gain, 8),
                    }
                )
        excluded = {
            normalize(value)
            for value in row.get("partial_molecules") or []
        }
        final = stable_unique(
            [
                by_id[candidate_id]["molecule"]
                for candidate_id in selected_ids
                if candidate_id in by_id
            ],
            excluded,
        )
        requested_n = int(row.get("n") or 0)
        if len(final) < requested_n:
            final.extend(
                value
                for value in values
                if normalize(value) not in {normalize(item) for item in final}
            )
        final = stable_unique(final, excluded)[:requested_n]
        if len(final) != requested_n:
            raise MultiAgentError("MPC swap constraint violated exact-n")
        return final, {
            "proposed_swap_count": len(proposed or []),
            "accepted_swaps": accepted,
            "exact_n": len(final) == requested_n,
            "constraint": "verified_one_for_one_two_channel_swap",
        }


def mfp_evidence_rows(
    row: dict[str, Any],
    evidence: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    molecules = stable_unique(row.get("molecules") or [])
    molecules.sort(
        key=lambda value: (
            -int(bool(evidence.get(normalize(value)))),
            normalize(value),
        )
    )
    for molecule in molecules[:20]:
        for snippet in evidence.get(normalize(molecule), [])[:2]:
            rows.append(
                {
                    "evidence_id": f"E{len(rows) + 1:02d}",
                    "molecule": molecule,
                    "relation_hint": evidence_relation(snippet),
                    "snippet": snippet,
                }
            )
            if len(rows) == 20:
                return rows
    return rows


def run_mfp_sample(
    row: dict[str, Any],
    agents: MFPAgents,
    evidence: dict[str, list[str]],
    evaluation: Any,
    llm_config: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates, diagnostics, demonstrations = agents.candidate_state(row)
    occurrence_envelope = agents.occurrence_report(
        row, candidates, demonstrations
    )
    candidate_ids = [item["candidate_id"] for item in candidates]
    calls: list[dict[str, Any]] = []
    structure_content, structure_usage = call_structured_llm(
        agents.structure_messages(row, candidates, diagnostics),
        llm_config,
        evaluation,
    )
    structure_report = validate_candidate_assessments(
        parse_json_object(structure_content),
        candidate_ids,
    )
    calls.append({"agent": "unimol_sensory_scientist", "usage": structure_usage})
    evidence_rows = mfp_evidence_rows(row, evidence)
    if evidence_rows:
        evidence_content, evidence_usage = call_structured_llm(
            agents.evidence_messages(row, candidates, evidence_rows),
            llm_config,
            evaluation,
        )
        evidence_report = validate_candidate_assessments(
            parse_json_object(evidence_content),
            candidate_ids,
        )
        calls.append({"agent": "evidence_critic", "usage": evidence_usage})
    else:
        evidence_report = {
            "candidate_assessments": [],
            "abstain_reason": "no_offline_evidence",
        }
    if len(calls) > MFP_MAX_AGENT_CALLS:
        raise MultiAgentError("MFP per-sample API budget exceeded")
    predicted, finalization = finalize_mfp(
        candidates,
        occurrence_envelope["payload"],
        structure_report,
        evidence_report,
    )
    prediction = {"id": str(row["id"]), "predicted_food": predicted}
    trace = {
        "id": str(row["id"]),
        "schema_version": SCHEMA_VERSION,
        "task": "mfp",
        "candidates": candidates,
        "occurrence_agent": occurrence_envelope,
        "structure_agent": agent_envelope(
            "mfp",
            str(row["id"]),
            "unimol_sensory_scientist",
            "abstain" if structure_report.get("abstain_reason") else "ok",
            {"candidate_ids": candidate_ids},
            structure_report,
        ),
        "evidence_agent": agent_envelope(
            "mfp",
            str(row["id"]),
            "evidence_critic",
            "abstain" if evidence_report.get("abstain_reason") else "ok",
            {"evidence_ids": [item["evidence_id"] for item in evidence_rows]},
            evidence_report,
        ),
        "final_arbiter": finalization,
        "api_calls": calls,
    }
    retrieval = {
        "id": str(row["id"]),
        "retrieved": [
            {
                "id": item["id"],
                "rank": item["rank"],
                "score": item["score"],
            }
            for item in demonstrations
        ],
    }
    evidence_metadata = {
        "id": str(row["id"]),
        "evidence_count": len(evidence_rows),
        "evidence_ids": [item["evidence_id"] for item in evidence_rows],
    }
    return prediction, trace, retrieval, evidence_metadata


def run_mpc_sample(
    row: dict[str, Any],
    agents: MPCAgents,
    evidence: dict[str, list[str]],
    evaluation: Any,
    llm_config: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates, diagnostics, demonstrations = agents.candidate_state(row)
    snippets = evidence.get(normalize(row.get("target_food")), [])[:12]
    evidence_envelope = agents.deterministic_evidence_report(
        row, candidates, snippets
    )
    values, plan = agents.set_plan(row, candidates, evidence_envelope)
    calls: list[dict[str, Any]] = []
    swap_report: dict[str, Any] = {
        "proposed_swaps": [],
        "abstain": True,
        "reason": "gate_not_triggered",
    }
    gate = agents.requires_swap_arbiter(row, candidates, plan)
    if gate:
        content, usage = call_structured_llm(
            agents.swap_messages(
                row, candidates, plan, evidence_envelope
            ),
            llm_config,
            evaluation,
        )
        parsed = parse_json_object(content)
        if isinstance(parsed, dict):
            swap_report = parsed
        calls.append({"agent": "swap_arbiter", "usage": usage})
    if len(calls) > MPC_MAX_AGENT_CALLS:
        raise MultiAgentError("MPC per-sample API budget exceeded")
    final, swap_metadata = agents.apply_swaps(
        row, candidates, values, plan, swap_report
    )
    requested_n = int(row.get("n") or 0)
    if len(final) != requested_n:
        raise MultiAgentError("MPC final prediction is not exact-n")
    prediction = {
        "id": str(row["id"]),
        "task": row.get("task", "MPC"),
        "target_food": row.get("target_food"),
        "partial_molecules": row.get("partial_molecules") or [],
        "n": requested_n,
        "predicted_molecules": final,
    }
    trace = {
        "id": str(row["id"]),
        "schema_version": SCHEMA_VERSION,
        "task": "mpc",
        "structure_diagnostics": diagnostics,
        "candidate_ledger": candidates,
        "occurrence_agent": {
            "schema_version": SCHEMA_VERSION,
            "agent": "nnpu_occurrence_scientist",
            "status": "ok",
        },
        "structure_agent": {
            "schema_version": SCHEMA_VERSION,
            "agent": "conditional_unimol_sensory_scientist",
            "status": "ok",
        },
        "evidence_agent": evidence_envelope,
        "set_planner": plan,
        "swap_arbiter": {
            "gate_triggered": gate,
            "report": swap_report,
            **swap_metadata,
        },
        "api_calls": calls,
    }
    retrieval = {
        "id": str(row["id"]),
        "retrieved": [
            {
                "id": item["id"],
                "rank": item["rank"],
                "score": item["score"],
            }
            for item in demonstrations
        ],
    }
    evidence_metadata = {
        "id": str(row["id"]),
        "evidence_count": len(snippets),
        "claim_count": len(
            evidence_envelope.get("payload", {}).get("claims") or []
        ),
    }
    return prediction, trace, retrieval, evidence_metadata


def _safe_query_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove task labels before any prediction-stage agent sees them."""
    safe = dict(row)
    safe.pop("actual_food", None)
    safe.pop("missing_molecules", None)
    return safe


def run_agent(args: argparse.Namespace) -> int:
    required = [
        (Path(args.train), "train split"),
        (Path(args.test), "query split"),
        (Path(args.db), "FlavorDB"),
        (Path(args.evidence), "official evidence"),
        (Path(args.unimol_embeddings), "UniMol embeddings"),
    ]
    for path, label in required:
        if not path.is_file():
            raise MultiAgentError(f"{label} not found: {path}")
    validate_prediction_paths(args)
    if not args.use_llm:
        raise MultiAgentError("--use-llm is required for formal prediction")
    evaluation = load_sibling_module(
        "evaluation.py", "foodpuzzle_multi_agent_evaluation"
    )
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(args)
    if llm_config["provider"] != "deepseek":
        raise MultiAgentError("Only-Deepseek route requires DeepSeek Platform")
    if llm_config["model"] != "deepseek-v4-flash":
        raise MultiAgentError("formal model must be deepseek-v4-flash")
    evaluation.require_api_key(llm_config)
    train_rows = read_jsonl(Path(args.train))
    test_rows = read_jsonl(Path(args.test))
    validate_split(train_rows, test_rows, args.task)
    embeddings = EmbeddingStore(Path(args.unimol_embeddings))
    evidence = load_evidence(Path(args.evidence))
    if args.task == "mfp":
        agents: Any = MFPAgents(train_rows, Path(args.db), embeddings)
    else:
        agents = MPCAgents(train_rows, Path(args.db), embeddings)
    completed = (
        successful_prediction_ids(Path(args.output), args.task)
        if args.resume
        else set()
    )
    agent_done = (
        existing_ids(Path(args.agent_metadata)) if args.resume else set()
    )
    retrieval_done = (
        existing_ids(Path(args.retrieval_metadata)) if args.resume else set()
    )
    evidence_done = (
        existing_ids(Path(args.evidence_metadata)) if args.resume else set()
    )
    success = 0
    skipped = 0
    for row in test_rows:
        row_id = str(row["id"])
        if row_id in completed and row_id in agent_done:
            skipped += 1
            continue
        # Gold fields are removed before any prediction-stage component sees
        # the sample.  They remain available only to the later evaluation
        # subprocess through the original test file.
        safe_row = _safe_query_row(row)
        try:
            if args.task == "mfp":
                prediction, trace, retrieval, evidence_metadata = run_mfp_sample(
                    safe_row, agents, evidence, evaluation, llm_config
                )
            else:
                prediction, trace, retrieval, evidence_metadata = run_mpc_sample(
                    safe_row, agents, evidence, evaluation, llm_config
                )
        except (
            evaluation.ChatCompletionHTTPError,
            evaluation.EvaluationError,
        ) as exc:
            raise MultiAgentError(
                "provider request failed; stop and resume without recording "
                f"a sample failure: {exc}"
            ) from exc
        if row_id not in retrieval_done:
            append_jsonl(Path(args.retrieval_metadata), retrieval)
        if row_id not in evidence_done:
            append_jsonl(Path(args.evidence_metadata), evidence_metadata)
        if row_id not in agent_done:
            append_jsonl(Path(args.agent_metadata), trace)
        append_jsonl(Path(args.output), prediction)
        success += 1
    print("MULTI_AGENT_STATUS: PASS")
    print(f"method_version: {METHOD_VERSION}")
    print(f"task: {args.task}")
    print(f"total: {len(test_rows)}")
    print(f"new_success: {success}")
    print(f"skipped: {skipped}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task-isolated heterogeneous FoodPuzzle multi-agent route"
    )
    parser.add_argument("--task", choices=["mfp", "mpc"])
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--db", default="data/raw/flavordb.db")
    parser.add_argument("--evidence")
    parser.add_argument(
        "--unimol-embeddings",
        default="data/structure/unimol/unimol_embeddings.npz",
    )
    parser.add_argument("--output")
    parser.add_argument("--agent-metadata")
    parser.add_argument("--retrieval-metadata")
    parser.add_argument("--evidence-metadata")
    parser.add_argument("--llm-provider", choices=["deepseek"], default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    required = {
        "--task": args.task,
        "--train": args.train,
        "--test": args.test,
        "--evidence": args.evidence,
        "--output": args.output,
        "--agent-metadata": args.agent_metadata,
        "--retrieval-metadata": args.retrieval_metadata,
        "--evidence-metadata": args.evidence_metadata,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise MultiAgentError(
            f"missing required arguments: {', '.join(missing)}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_cli(args)
        return run_agent(args)
    except MultiAgentError as exc:
        print("MULTI_AGENT_STATUS: FAIL")
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
