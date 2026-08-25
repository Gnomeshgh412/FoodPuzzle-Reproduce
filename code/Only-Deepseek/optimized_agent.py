#!/usr/bin/env python3
"""Task-specific scientific Agent for FoodPuzzle MFP and MPC.

MFP and MPC are deliberately trained and inferred independently.  MFP uses
the frozen UniMol encoder; MPC uses food-conditioned occurrence, retrieval,
and evidence without structural-model scoring.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ABLATIONS = {
    "full",
    "no_unimol",
    "raw_unimol_nn",
    "no_evidence",
    "flat_fusion",
    "no_reviewer",
    "no_ledger",
}
METHOD_VERSION = "optimized_agent_v15_mfp_concrete_protocol"


class OptimizedAgentError(Exception):
    """Expected optimized-agent failure."""


def load_sibling_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OptimizedAgentError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(value: Any) -> str:
    text = str(value or "").lower().strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^0-9a-z+\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_official_functional_groups(raw_value: Any) -> set[str]:
    """Mirror the released MPC gold-side FlavorDB parser.

    This reads molecule-intrinsic FlavorDB metadata only.  It deliberately
    does not read the LLM functional-group evaluation cache.
    """
    text = str(raw_value or "").strip()
    if not text:
        return set()
    groups: set[str] = set()
    for token in text.split(" "):
        token = token.strip()
        if not token:
            continue
        if "@" in token:
            parts = token.split("@")
            if "compound" in token and len(parts) > 1 and parts[1].strip():
                groups.add(parts[1].strip().lower())
            if "primary" in token and parts[0].strip():
                groups.add(parts[0].strip().lower())
        else:
            groups.add(token.lower())
    return groups


def stable_unique(values: list[Any], excluded: set[str] | None = None) -> list[str]:
    blocked = excluded or set()
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        key = normalize(item)
        if not item or not key or key in seen or key in blocked:
            continue
        output.append(item)
        seen.add(key)
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OptimizedAgentError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict) or row.get("id") is None:
                raise OptimizedAgentError(f"invalid row at {path}:{line_no}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def successful_ids(path: Path, task: str) -> set[str]:
    if not path.is_file():
        return set()
    successful: set[str] = set()
    for row in read_jsonl(path):
        if row.get("error"):
            continue
        if task == "mfp" and str(row.get("predicted_food") or "").strip():
            successful.add(str(row["id"]))
        if task == "mpc" and isinstance(row.get("predicted_molecules"), list):
            if row["predicted_molecules"]:
                successful.add(str(row["id"]))
    return successful


def existing_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in read_jsonl(path)} if path.is_file() else set()


def successful_hypothesis_ids(path: Path, task: str) -> set[str]:
    if not path.is_file():
        return set()
    successful: set[str] = set()
    for row in read_jsonl(path):
        if row.get("error"):
            continue
        hypotheses = row.get("hypotheses")
        has_hypotheses = (
            isinstance(hypotheses, list)
            and (len(hypotheses) == 3 if task == "mfp" else len(hypotheses) >= 1)
        )
        has_reviewer = isinstance(row.get("reviewer_output"), dict)
        if has_hypotheses and (has_reviewer or row.get("ablation") == "no_reviewer"):
            successful.add(str(row["id"]))
    return successful


def require_files(paths: list[tuple[str | None, str]]) -> None:
    for raw_path, label in paths:
        if not raw_path or not Path(raw_path).is_file():
            raise OptimizedAgentError(f"{label} not found: {raw_path}")


def validate_split(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> None:
    train_ids = {str(row["id"]) for row in train_rows}
    test_ids = {str(row["id"]) for row in test_rows}
    if train_ids & test_ids:
        raise OptimizedAgentError("train and test split overlap")
    if not train_rows or not test_rows:
        raise OptimizedAgentError("empty train or test split")


def validate_output_paths(args: argparse.Namespace) -> None:
    paths = [
        args.output,
        args.evidence_metadata,
        args.retrieval_metadata,
        args.hypotheses_metadata,
    ]
    for raw_path in paths:
        path = Path(raw_path)
        if not path.parent.is_dir():
            raise OptimizedAgentError(f"output parent does not exist: {path.parent}")
        if path.exists() and not args.resume:
            raise OptimizedAgentError(f"output exists; use --resume: {path}")


def prepare_unimol_embeddings(args: argparse.Namespace) -> int:
    input_path = Path(args.unimol_input_csv)
    output_path = Path(args.unimol_embeddings)
    require_files([(str(input_path), "UniMol input CSV")])
    if not output_path.parent.is_dir():
        raise OptimizedAgentError(f"embedding parent does not exist: {output_path.parent}")
    if output_path.exists():
        raise OptimizedAgentError(f"embedding file already exists: {output_path}")

    repo_root = Path(__file__).resolve().parents[2]
    tools_root = repo_root / "UniMol" / "unimol_tools"
    if not tools_root.is_dir():
        raise OptimizedAgentError(f"official UniMol tools directory not found: {tools_root}")
    sys.path.insert(0, str(tools_root))
    try:
        import numpy as np
        from unimol_tools import UniMolRepr
    except Exception as exc:
        raise OptimizedAgentError(
            "UniMol dependencies are unavailable; use an isolated environment "
            "with the official unimol_tools requirements"
        ) from exc

    records: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("molecule") or "").strip()
            smiles = str(row.get("smiles") or "").strip()
            if name and smiles:
                records.append({"name": name, "smiles": smiles})
    if not records:
        raise OptimizedAgentError("UniMol input CSV has no valid molecule/SMILES rows")

    encoder = UniMolRepr(data_type="molecule", remove_hs=False)
    all_vectors: list[Any] = []
    batch_size = max(1, args.unimol_batch_size)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        representation = encoder.get_repr([row["smiles"] for row in batch])
        vectors = representation.get("cls_repr")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise OptimizedAgentError(f"UniMol failed for batch starting at {start}")
        all_vectors.extend(vectors)

    matrix = np.asarray(all_vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise OptimizedAgentError(f"invalid UniMol matrix shape: {matrix.shape}")
    np.savez_compressed(
        output_path,
        names=np.asarray([row["name"] for row in records]),
        smiles=np.asarray([row["smiles"] for row in records]),
        embeddings=matrix,
    )
    print("UNIMOL_PREPARATION_STATUS: PASS")
    print(f"molecules: {len(records)}")
    print(f"embedding_dimension: {matrix.shape[1]}")
    print(f"output: {output_path}")
    return 0


class EmbeddingStore:
    def __init__(self, path: Path):
        try:
            import numpy as np
        except Exception as exc:
            raise OptimizedAgentError("numpy is required") from exc
        if not path.is_file():
            raise OptimizedAgentError(f"UniMol embeddings not found: {path}")
        data = np.load(path, allow_pickle=False)
        if "names" not in data or "embeddings" not in data:
            raise OptimizedAgentError("UniMol NPZ requires names and embeddings")
        names = [str(item) for item in data["names"].tolist()]
        matrix = np.asarray(data["embeddings"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(names):
            raise OptimizedAgentError("UniMol names/embeddings shape mismatch")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.maximum(norms, 1e-12)
        self.names = names
        self.index = {normalize(name): idx for idx, name in enumerate(names)}
        self.dimension = int(matrix.shape[1])
        self.np = np

    def available(self, names: list[Any]) -> list[str]:
        return [str(name) for name in names if normalize(name) in self.index]

    def vector(self, name: Any) -> Any | None:
        idx = self.index.get(normalize(name))
        return None if idx is None else self.matrix[idx]

    def profile(self, names: list[Any], weights: dict[str, float] | None = None) -> Any | None:
        vectors: list[Any] = []
        vector_weights: list[float] = []
        for name in names:
            vector = self.vector(name)
            if vector is None:
                continue
            vectors.append(vector)
            vector_weights.append((weights or {}).get(normalize(name), 1.0))
        if not vectors:
            return None
        matrix = self.np.asarray(vectors)
        profile = self.np.average(matrix, axis=0, weights=self.np.asarray(vector_weights))
        norm = float(self.np.linalg.norm(profile))
        return profile / max(norm, 1e-12)

    @staticmethod
    def cosine(left: Any | None, right: Any | None) -> float:
        if left is None or right is None:
            return 0.0
        return float(left @ right)

    def weighted_chamfer(
        self,
        left_names: list[Any],
        right_names: list[Any],
        left_weights: dict[str, float] | None = None,
    ) -> tuple[float, float, int, int]:
        """Return bidirectional molecule-set similarity without mean-pooling the set."""
        left = [
            (normalize(name), self.vector(name))
            for name in left_names
            if self.vector(name) is not None
        ]
        right = [
            (normalize(name), self.vector(name))
            for name in right_names
            if self.vector(name) is not None
        ]
        if not left or not right:
            return 0.0, 0.0, len(left), len(right)
        left_matrix = self.np.asarray([item[1] for item in left])
        right_matrix = self.np.asarray([item[1] for item in right])
        pairwise = left_matrix @ right_matrix.T
        weights = self.np.asarray(
            [(left_weights or {}).get(item[0], 1.0) for item in left],
            dtype=self.np.float32,
        )
        query_to_candidate = float(
            self.np.average(pairwise.max(axis=1), weights=weights)
        )
        candidate_to_query = float(pairwise.max(axis=0).mean())
        return query_to_candidate, candidate_to_query, len(left), len(right)


def molecule_idf(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for row in rows:
        values = row.get(field)
        if isinstance(values, list):
            document_frequency.update({normalize(value) for value in values if normalize(value)})
    total = max(1, len(rows))
    return {
        name: math.log(1.0 + (total + 1.0) / (count + 1.0))
        for name, count in document_frequency.items()
    }


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
        for alias_field in [readable, alias, synonyms]:
            key = normalize(alias_field)
            if key:
                mapping[key] = label
        if basket:
            for basket_alias in str(basket).split(","):
                key = normalize(basket_alias)
                if key:
                    mapping[key] = label
    return mapping


class MFPStructureModel:
    """MFP-only category adapter over a permutation-invariant molecule set.

    Nearest-neighbour retrieval is retained as an auditable evidence channel,
    but the category posterior is produced by two train-only supervised
    adapters: a sparse molecule-presence model and a frozen-UniMol profile
    model.  Their mixing weight is selected only from out-of-fold predictions
    on the training split.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        embeddings: EmbeddingStore | None,
        categories: dict[str, str],
        ablation: str,
    ):
        self.rows = rows
        self.embeddings = embeddings
        self.categories = categories
        self.ablation = ablation
        self.idf = molecule_idf(rows, "molecules")
        self.profile_vectors: list[Any | None] = []
        self.profile_sets: list[set[str]] = []
        self.profile_names: list[list[str]] = []
        for row in rows:
            molecules = row.get("molecules") if isinstance(row.get("molecules"), list) else []
            self.profile_names.append([str(item) for item in molecules])
            self.profile_sets.append({normalize(item) for item in molecules if normalize(item)})
            self.profile_vectors.append(
                embeddings.profile(molecules, self.idf) if embeddings is not None else None
            )
        category_vectors: dict[str, list[Any]] = defaultdict(list)
        for row, vector in zip(self.rows, self.profile_vectors):
            label = self.categories.get(normalize(row.get("actual_food")))
            if label and vector is not None:
                category_vectors[label].append(vector)
        self.category_centroids: dict[str, Any] = {}
        if embeddings is not None:
            for label, vectors in category_vectors.items():
                centroid = embeddings.np.asarray(vectors).mean(axis=0)
                norm = float(embeddings.np.linalg.norm(centroid))
                self.category_centroids[label] = centroid / max(norm, 1e-12)
        self.category_labels = sorted(
            {
                self.categories.get(normalize(row.get("actual_food")))
                for row in self.rows
                if self.categories.get(normalize(row.get("actual_food")))
            }
        )
        self.category_blend_unimol = 0.0
        self.category_blend_set = 0.0
        self.category_oof_accuracy = 0.0
        self.sparse_vectorizer: Any | None = None
        self.sparse_classifier: Any | None = None
        self.unimol_classifier: Any | None = None
        self.category_set_stats: dict[str, dict[str, Any]] = {}
        self._fit_category_adapters()

    def _documents(self, rows: list[dict[str, Any]]) -> list[list[str]]:
        return [
            [
                normalize(value)
                for value in row.get("molecules") or []
                if normalize(value)
            ]
            for row in rows
        ]

    def _unimol_features(self, rows: list[dict[str, Any]]) -> Any | None:
        if self.embeddings is None or self.ablation == "no_unimol":
            return None
        vectors: list[Any] = []
        for row in rows:
            molecules = row.get("molecules") or []
            available = [
                self.embeddings.vector(value)
                for value in molecules
                if self.embeddings.vector(value) is not None
            ]
            if not available:
                vectors.append(
                    self.embeddings.np.zeros(
                        self.embeddings.dimension * 2,
                        dtype=self.embeddings.np.float32,
                    )
                )
                continue
            matrix = self.embeddings.np.asarray(available)
            weights = self.embeddings.np.asarray(
                [self.idf.get(normalize(value), 1.0) for value in molecules
                 if self.embeddings.vector(value) is not None],
                dtype=self.embeddings.np.float32,
            )
            weighted_mean = self.embeddings.np.average(
                matrix, axis=0, weights=weights
            )
            # Standard deviation retains mixture heterogeneity that was lost
            # by the previous single-centroid representation.
            spread = matrix.std(axis=0)
            vectors.append(
                self.embeddings.np.concatenate([weighted_mean, spread])
            )
        return self.embeddings.np.asarray(vectors, dtype=self.embeddings.np.float32)

    def _fit_category_set_stats(
        self,
        indices: list[int],
        labels: list[str],
        documents: list[list[str]],
    ) -> dict[str, dict[str, Any]]:
        """Fit low-capacity category-conditioned statistics on food sets.

        The adapter deliberately keeps the UniMol backbone frozen.  It learns
        which molecules are diagnostic for each category and represents every
        category as a weighted set of frozen molecule embeddings.
        """
        category_rows: dict[str, list[int]] = defaultdict(list)
        molecule_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for idx in indices:
            label = str(labels[idx])
            category_rows[label].append(idx)
            molecule_counts[label].update(set(documents[idx]))

        stats: dict[str, dict[str, Any]] = {}
        total_rows = len(indices)
        for label in self.category_labels:
            member_indices = category_rows.get(label, [])
            member_count = len(member_indices)
            other_count = max(0, total_rows - member_count)
            log_odds: dict[str, float] = {}
            all_names = {
                name
                for idx in indices
                for name in documents[idx]
            }
            for name in all_names:
                inside = molecule_counts[label].get(name, 0)
                outside = sum(
                    molecule_counts[other].get(name, 0)
                    for other in self.category_labels
                    if other != label
                )
                inside_rate = (inside + 1.0) / (member_count + 2.0)
                outside_rate = (outside + 1.0) / (other_count + 2.0)
                log_odds[name] = math.log(inside_rate / outside_rate)

            prototype = None
            if self.embeddings is not None and member_indices:
                vectors: list[Any] = []
                weights: list[float] = []
                for idx in member_indices:
                    for name in documents[idx]:
                        vector = self.embeddings.vector(name)
                        if vector is None:
                            continue
                        diagnostic = max(0.0, math.tanh(log_odds.get(name, 0.0) / 3.0))
                        vectors.append(vector)
                        weights.append(
                            self.idf.get(name, 1.0) * (1.0 + diagnostic)
                        )
                if vectors:
                    matrix = self.embeddings.np.asarray(vectors)
                    prototype = self.embeddings.np.average(
                        matrix,
                        axis=0,
                        weights=self.embeddings.np.asarray(weights),
                    )
                    norm = float(self.embeddings.np.linalg.norm(prototype))
                    prototype = prototype / max(norm, 1e-12)
            stats[label] = {
                "member_count": member_count,
                "log_odds": log_odds,
                "prototype": prototype,
            }
        return stats

    def _category_set_probabilities(
        self,
        row: dict[str, Any],
        stats: dict[str, dict[str, Any]],
    ) -> Any:
        """Score a food as an unordered set of diagnostic molecules."""
        import numpy as np

        molecules = [
            normalize(value)
            for value in row.get("molecules") or []
            if normalize(value)
        ]
        profile = (
            self.embeddings.profile(molecules, self.idf)
            if self.embeddings is not None
            else None
        )
        raw_scores: list[float] = []
        for label in self.category_labels:
            category = stats.get(label, {})
            log_odds = category.get("log_odds") or {}
            prototype = category.get("prototype")
            molecule_scores: list[tuple[float, float]] = []
            for molecule in molecules:
                diagnostic = max(
                    0.0,
                    math.tanh(float(log_odds.get(molecule, 0.0)) / 3.0),
                )
                vector = (
                    self.embeddings.vector(molecule)
                    if self.embeddings is not None
                    else None
                )
                structural = max(
                    0.0,
                    EmbeddingStore.cosine(vector, prototype),
                )
                weight = self.idf.get(molecule, 1.0)
                molecule_scores.append(
                    (weight * (0.70 * diagnostic + 0.30 * structural), weight)
                )
            molecule_scores.sort(key=lambda item: -item[0])
            top_values = molecule_scores[: min(5, len(molecule_scores))]
            top_support = (
                sum(value for value, _ in top_values)
                / max(1e-12, sum(weight for _, weight in top_values))
                if top_values
                else 0.0
            )
            mean_support = (
                sum(value for value, _ in molecule_scores)
                / max(1e-12, sum(weight for _, weight in molecule_scores))
                if molecule_scores
                else 0.0
            )
            profile_support = max(
                0.0,
                EmbeddingStore.cosine(profile, prototype),
            )
            raw_scores.append(
                0.55 * top_support
                + 0.25 * mean_support
                + 0.20 * profile_support
            )
        raw = np.asarray(raw_scores, dtype=np.float64)
        raw -= raw.max(initial=0.0)
        probabilities = np.exp(raw / 0.20)
        return probabilities / max(float(probabilities.sum()), 1e-12)

    @staticmethod
    def _aligned_probabilities(
        classifier: Any,
        features: Any,
        labels: list[str],
    ) -> Any:
        import numpy as np

        raw = classifier.predict_proba(features)
        aligned = np.zeros((raw.shape[0], len(labels)), dtype=np.float64)
        label_index = {label: idx for idx, label in enumerate(labels)}
        for source_idx, label in enumerate(classifier.classes_):
            if str(label) in label_index:
                aligned[:, label_index[str(label)]] = raw[:, source_idx]
        denominator = aligned.sum(axis=1, keepdims=True)
        return aligned / np.maximum(denominator, 1e-12)

    def _fit_category_adapters(self) -> None:
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            raise OptimizedAgentError(
                "scikit-learn is required for the MFP category adapter"
            ) from exc

        labels = [
            self.categories.get(normalize(row.get("actual_food")))
            for row in self.rows
        ]
        if not self.category_labels or any(label is None for label in labels):
            raise OptimizedAgentError(
                "MFP training rows require FlavorDB macro categories"
            )
        documents = self._documents(self.rows)
        unimol_features = self._unimol_features(self.rows)

        def new_vectorizer() -> Any:
            return TfidfVectorizer(
                analyzer=lambda values: values,
                lowercase=False,
                token_pattern=None,
                sublinear_tf=True,
                norm="l2",
            )

        def new_classifier() -> Any:
            return LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            )

        # Five deterministic class-aware folds are used only to choose the
        # fusion weights.  Round-robin assignment preserves every category as
        # far as its sample count permits and remains honest for singleton
        # categories: their one validation fold contains no same-class training
        # example.  The test split never participates in this calibration.
        fold_buckets: list[list[int]] = [[] for _ in range(5)]
        indices_by_label: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            indices_by_label[str(label)].append(idx)
        for label in sorted(indices_by_label):
            category_indices = list(indices_by_label[label])
            random.Random(f"mfp-v6-fold:{label}").shuffle(category_indices)
            for offset, idx in enumerate(category_indices):
                fold_buckets[offset % 5].append(idx)
        sparse_oof = np.zeros(
            (len(self.rows), len(self.category_labels)), dtype=np.float64
        )
        unimol_oof = np.zeros_like(sparse_oof)
        set_oof = np.zeros_like(sparse_oof)
        all_indices = np.arange(len(self.rows))
        for fold_index in range(5):
            validation_indices = np.asarray(
                sorted(fold_buckets[fold_index]),
                dtype=np.int64,
            )
            validation_set = set(validation_indices.tolist())
            train_indices = np.asarray(
                [idx for idx in all_indices if int(idx) not in validation_set],
                dtype=np.int64,
            )
            fold_vectorizer = new_vectorizer()
            fold_sparse = fold_vectorizer.fit_transform(
                [documents[idx] for idx in train_indices]
            )
            fold_classifier = new_classifier()
            fold_classifier.fit(
                fold_sparse,
                [labels[idx] for idx in train_indices],
            )
            sparse_oof[validation_indices] = self._aligned_probabilities(
                fold_classifier,
                fold_vectorizer.transform(
                    [documents[idx] for idx in validation_indices]
                ),
                self.category_labels,
            )
            if unimol_features is not None:
                fold_unimol_classifier = new_classifier()
                fold_unimol_classifier.fit(
                    unimol_features[train_indices],
                    [labels[idx] for idx in train_indices],
                )
                unimol_oof[validation_indices] = self._aligned_probabilities(
                    fold_unimol_classifier,
                    unimol_features[validation_indices],
                    self.category_labels,
                )
                fold_set_stats = self._fit_category_set_stats(
                    train_indices.tolist(),
                    [str(label) for label in labels],
                    documents,
                )
                for validation_idx in validation_indices:
                    set_oof[validation_idx] = self._category_set_probabilities(
                        self.rows[int(validation_idx)],
                        fold_set_stats,
                    )

        label_indices = np.asarray(
            [self.category_labels.index(str(label)) for label in labels]
        )
        candidate_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_unimol_weight = 0.0
        best_set_weight = 0.0
        best_accuracy = -1.0
        for unimol_weight in candidate_weights:
            for set_weight in candidate_weights:
                if unimol_features is None and (
                    unimol_weight > 0.0 or set_weight > 0.0
                ):
                    continue
                if unimol_weight + set_weight > 1.0 + 1e-12:
                    continue
                sparse_weight = 1.0 - unimol_weight - set_weight
                probabilities = (
                    sparse_weight * sparse_oof
                    + unimol_weight * unimol_oof
                    + set_weight * set_oof
                )
                accuracy = float(
                    (probabilities.argmax(axis=1) == label_indices).mean()
                )
                key = (
                    accuracy,
                    -set_weight - unimol_weight,
                    set_weight,
                )
                best_key = (
                    best_accuracy,
                    -best_set_weight - best_unimol_weight,
                    best_set_weight,
                )
                if key > best_key:
                    best_accuracy = accuracy
                    best_unimol_weight = unimol_weight
                    best_set_weight = set_weight
        self.category_blend_unimol = best_unimol_weight
        self.category_blend_set = best_set_weight
        self.category_oof_accuracy = best_accuracy
        self.category_set_stats = self._fit_category_set_stats(
            all_indices.tolist(),
            [str(label) for label in labels],
            documents,
        )

        self.sparse_vectorizer = new_vectorizer()
        sparse_matrix = self.sparse_vectorizer.fit_transform(documents)
        self.sparse_classifier = new_classifier()
        self.sparse_classifier.fit(sparse_matrix, labels)
        if unimol_features is not None:
            self.unimol_classifier = new_classifier()
            self.unimol_classifier.fit(unimol_features, labels)

    def _category_probabilities(
        self,
        row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.sparse_vectorizer is None or self.sparse_classifier is None:
            return []
        sparse = self._aligned_probabilities(
            self.sparse_classifier,
            self.sparse_vectorizer.transform(self._documents([row])),
            self.category_labels,
        )[0]
        unimol = sparse
        features = self._unimol_features([row])
        if self.unimol_classifier is not None and features is not None:
            unimol = self._aligned_probabilities(
                self.unimol_classifier,
                features,
                self.category_labels,
            )[0]
        set_probabilities = (
            self._category_set_probabilities(row, self.category_set_stats)
            if self.category_set_stats
            else sparse
        )
        sparse_weight = (
            1.0 - self.category_blend_unimol - self.category_blend_set
        )
        blended = (
            sparse_weight * sparse
            + self.category_blend_unimol * unimol
            + self.category_blend_set * set_probabilities
        )
        order = sorted(
            range(len(self.category_labels)),
            key=lambda idx: (-float(blended[idx]), self.category_labels[idx]),
        )
        return [
            {
                "category": self.category_labels[idx],
                "probability": round(float(blended[idx]), 6),
                "sparse_probability": round(float(sparse[idx]), 6),
                "unimol_probability": round(float(unimol[idx]), 6),
                "unimol_set_probability": round(
                    float(set_probabilities[idx]), 6
                ),
            }
            for idx in order
        ]

    def _anchor_scores(self, molecules: list[Any]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for molecule in molecules:
            key = normalize(molecule)
            vector = self.embeddings.vector(molecule) if self.embeddings is not None else None
            category_similarities = sorted(
                (
                    EmbeddingStore.cosine(vector, centroid)
                    for centroid in self.category_centroids.values()
                ),
                reverse=True,
            )
            contrast = (
                max(0.0, category_similarities[0] - category_similarities[1])
                if len(category_similarities) >= 2
                else 0.0
            )
            # IDF identifies rare molecules; category contrast identifies molecules
            # whose 3D representation actually narrows the food search space.
            scores[key] = self.idf.get(key, 0.0) * (1.0 + 8.0 * contrast)
        return scores

    def rank(self, row: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        molecules = row.get("molecules") if isinstance(row.get("molecules"), list) else []
        query_set = {normalize(item) for item in molecules if normalize(item)}
        query_vector = (
            self.embeddings.profile(molecules, self.idf)
            if self.embeddings is not None and self.ablation != "no_unimol"
            else None
        )
        coarse_scores: list[tuple[float, int, float, float]] = []
        for idx, train_set in enumerate(self.profile_sets):
            union = query_set | train_set
            jaccard = len(query_set & train_set) / len(union) if union else 0.0
            cosine = EmbeddingStore.cosine(query_vector, self.profile_vectors[idx])
            score = jaccard if self.ablation == "no_unimol" else 0.8 * cosine + 0.2 * jaccard
            coarse_scores.append((score, idx, cosine, jaccard))
        coarse_scores.sort(key=lambda item: (-item[0], item[1]))

        shortlist_size = min(len(coarse_scores), max(100, top_k * 4))
        scores: list[tuple[float, int, float, float, float, float]] = []
        for _, idx, cosine, jaccard in coarse_scores[:shortlist_size]:
            if self.embeddings is None or self.ablation == "no_unimol":
                query_to_food = jaccard
                food_to_query = jaccard
                score = jaccard
            else:
                query_to_food, food_to_query, _, _ = self.embeddings.weighted_chamfer(
                    molecules,
                    self.profile_names[idx],
                    self.idf,
                )
                # Selected on the reconstructed dev split only.  Query-to-food
                # preserves diagnostic input molecules; the reverse direction
                # penalizes foods whose known profile is poorly covered.
                score = 0.6 * query_to_food + 0.4 * food_to_query
            scores.append(
                (score, idx, cosine, jaccard, query_to_food, food_to_query)
            )
        scores.sort(key=lambda item: (-item[0], item[1]))

        candidates: list[dict[str, Any]] = []
        for rank, (
            score, idx, cosine, jaccard, query_to_food, food_to_query
        ) in enumerate(scores[:top_k], 1):
            train_row = self.rows[idx]
            candidates.append(
                {
                    "rank": rank,
                    "food": train_row.get("actual_food"),
                    "category": self.categories.get(normalize(train_row.get("actual_food"))),
                    "score": round(score, 6),
                    "unimol_profile_cosine": round(cosine, 6),
                    "unimol_query_to_food": round(query_to_food, 6),
                    "unimol_food_to_query": round(food_to_query, 6),
                    "molecule_jaccard": round(jaccard, 6),
                    "source": (
                        "molecule_overlap"
                        if self.ablation == "no_unimol"
                        else "task_profile_retrieval"
                    ),
                }
            )
        category_scores = self._category_probabilities(row)
        anchor_scores = self._anchor_scores(molecules)
        diagnostics = {
            "query_molecule_count": len(molecules),
            "mapped_unimol_count": (
                len(self.embeddings.available(molecules)) if self.embeddings is not None else 0
            ),
            "category_adapter": True,
            "category_score_method": (
                "train_oof_calibrated_sparse_unimol_set_adapter"
            ),
            "category_blend_unimol": self.category_blend_unimol,
            "category_blend_unimol_set": self.category_blend_set,
            "category_oof_accuracy": round(self.category_oof_accuracy, 6),
            "category_scores": category_scores[:10],
            "anchor_scores": [
                {
                    "molecule": str(molecule),
                    "score": round(anchor_scores.get(normalize(molecule), 0.0), 6),
                }
                for molecule in sorted(
                    molecules,
                    key=lambda item: (
                        -anchor_scores.get(normalize(item), 0.0),
                        normalize(item),
                    ),
                )[:10]
            ],
        }
        return candidates, diagnostics


class MPCPerceptualAdapter:
    """Project frozen UniMol vectors into a molecule-local flavor space.

    The adapter uses only descriptor fields from the FlavorDB ``molecules``
    table.  It deliberately never reads ``entity_molecule_link``: food-to-
    molecule associations outside the training split would reveal MPC labels.
    """

    def __init__(self, embeddings: EmbeddingStore, db_path: Path):
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.linear_model import LogisticRegression, Ridge
        except Exception as exc:
            raise OptimizedAgentError(
                "scikit-learn is required for the MPC perceptual adapter"
            ) from exc

        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                """
                SELECT common_name,
                       functional_groups,
                       fooddb_flavor_profile,
                       flavor_profile,
                       fema_flavor_profile,
                       odor,
                       taste
                FROM molecules
                """
            ).fetchall()
        descriptor_sets_by_name: dict[str, set[str]] = {}
        functional_groups_by_name: dict[str, set[str]] = {}
        canonical_names: dict[str, str] = {}
        for common_name, groups, fooddb, flavor, fema, odor, taste in rows:
            key = normalize(common_name)
            if not key:
                continue
            canonical_names[key] = str(common_name)
            functional_groups_by_name[key] = (
                parse_official_functional_groups(groups)
            )
            descriptors: set[str] = set()
            descriptors.update(self._parse_descriptors(groups, "group"))
            descriptors.update(self._parse_descriptors(fooddb, "flavor"))
            descriptors.update(self._parse_descriptors(flavor, "flavor"))
            descriptors.update(self._parse_descriptors(fema, "flavor"))
            # Odor and taste strings are less standardized.  Keeping only
            # short normalized clauses prevents long prose fragments from
            # becoming one-off labels.
            descriptors.update(self._parse_descriptors(odor, "odor", max_words=5))
            descriptors.update(self._parse_descriptors(taste, "taste", max_words=5))
            descriptor_sets_by_name[key] = descriptors
        descriptor_sets = {
            name: descriptor_sets_by_name.get(normalize(name), set())
            for name in embeddings.names
        }
        functional_group_sets = {
            name: functional_groups_by_name.get(normalize(name), set())
            for name in embeddings.names
        }
        vocabulary = sorted(
            {
                descriptor
                for descriptors in descriptor_sets.values()
                for descriptor in descriptors
            }
        )
        if len(vocabulary) < 2:
            raise OptimizedAgentError(
                "FlavorDB does not contain enough flavor descriptors"
            )

        np = embeddings.np
        descriptor_index = {
            descriptor: idx for idx, descriptor in enumerate(vocabulary)
        }
        document_frequency: Counter[str] = Counter()
        matrix = np.zeros(
            (len(embeddings.names), len(vocabulary)), dtype=np.float32
        )
        for row_idx, name in enumerate(embeddings.names):
            descriptors = descriptor_sets[name]
            document_frequency.update(descriptors)
            for descriptor in descriptors:
                matrix[row_idx, descriptor_index[descriptor]] = 1.0
        idf = np.asarray(
            [
                math.log(
                    (1.0 + len(embeddings.names))
                    / (1.0 + document_frequency[descriptor])
                )
                + 1.0
                for descriptor in vocabulary
            ],
            dtype=np.float32,
        )
        matrix *= idf
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

        latent_dimension = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if latent_dimension < 1:
            raise OptimizedAgentError("invalid MPC perceptual latent dimension")
        latent_targets = TruncatedSVD(
            n_components=latent_dimension,
            random_state=0,
        ).fit_transform(matrix)
        projected = Ridge(alpha=10.0).fit(
            embeddings.matrix,
            latent_targets,
        ).predict(embeddings.matrix)
        projected = np.asarray(projected, dtype=np.float32)
        projected /= np.maximum(
            np.linalg.norm(projected, axis=1, keepdims=True), 1e-12
        )

        functional_group_vocabulary = sorted(
            {
                group
                for groups in functional_group_sets.values()
                for group in groups
            }
        )
        if not functional_group_vocabulary:
            raise OptimizedAgentError(
                "FlavorDB does not contain functional-group supervision"
            )
        group_index = {
            group: idx
            for idx, group in enumerate(functional_group_vocabulary)
        }
        group_targets = np.zeros(
            (
                len(embeddings.names),
                len(functional_group_vocabulary),
            ),
            dtype=np.float32,
        )
        for row_idx, name in enumerate(embeddings.names):
            for group in functional_group_sets[name]:
                group_targets[row_idx, group_index[group]] = 1.0

        # UniMol is a frozen structural encoder.  A low-capacity, strongly
        # regularized probe translates its representation into the same
        # molecule-intrinsic functional-group space used by the released MPC
        # gold-side evaluator.  No food-to-molecule links or evaluator cache
        # are consumed here.
        probe_dimension = min(
            32,
            embeddings.matrix.shape[0] - 1,
            embeddings.matrix.shape[1] - 1,
        )
        probe_features = TruncatedSVD(
            n_components=probe_dimension,
            random_state=0,
        ).fit_transform(embeddings.matrix)
        group_probabilities = np.zeros_like(group_targets)
        group_probe_models: dict[str, Any] = {}
        for group, column in group_index.items():
            labels = group_targets[:, column]
            positive_count = int(labels.sum())
            if positive_count < 3 or positive_count > len(labels) - 3:
                group_probabilities[:, column] = (
                    positive_count / max(1, len(labels))
                )
                continue
            classifier = LogisticRegression(
                C=0.05,
                max_iter=2000,
                random_state=0,
            )
            classifier.fit(probe_features, labels)
            group_probabilities[:, column] = classifier.predict_proba(
                probe_features
            )[:, 1]
            group_probe_models[group] = classifier

        self.matrix = projected
        self.index = embeddings.index
        self.attribute_sets = {
            normalize(name): set(descriptor_sets[name])
            for name in embeddings.names
        }
        self.canonical_names = canonical_names
        self.functional_group_sets = {
            normalize(name): set(functional_group_sets[name])
            for name in embeddings.names
        }
        self.functional_group_vocabulary = functional_group_vocabulary
        self.functional_group_index = group_index
        self.functional_group_probabilities_matrix = group_probabilities
        self.functional_group_probe_models = group_probe_models
        self.functional_group_probe_dimension = probe_dimension
        self.functional_group_probe_label_count = len(
            functional_group_vocabulary
        )
        self.functional_group_probe_trainable_label_count = len(
            group_probe_models
        )
        self.vocabulary = vocabulary
        self.attribute_idf = {
            descriptor: float(idf[idx])
            for idx, descriptor in enumerate(vocabulary)
        }
        self.dimension = int(projected.shape[1])
        self.descriptor_vocabulary_size = len(vocabulary)
        self.descriptor_covered_count = sum(
            bool(descriptor_sets[name]) for name in embeddings.names
        )
        self.np = np

    @staticmethod
    def _parse_descriptors(
        value: Any,
        prefix: str,
        max_words: int = 8,
    ) -> set[str]:
        output: set[str] = set()
        for item in re.split(r"[@,;/|]+", str(value or "")):
            normalized = normalize(item)
            if normalized and len(normalized.split()) <= max_words:
                output.add(f"{prefix}:{normalized}")
        return output

    def attributes(self, molecule: Any) -> set[str]:
        return self.attribute_sets.get(normalize(molecule), set())

    def functional_groups(self, molecule: Any) -> set[str]:
        return self.functional_group_sets.get(normalize(molecule), set())

    def functional_group_probabilities(
        self,
        molecule: Any,
    ) -> list[float]:
        idx = self.index.get(normalize(molecule))
        if idx is None:
            return [0.0] * len(self.functional_group_vocabulary)
        known_groups = self.functional_groups(molecule)
        if known_groups:
            return [
                float(group in known_groups)
                for group in self.functional_group_vocabulary
            ]
        return self.functional_group_probe_probabilities(molecule)

    def functional_group_probe_probabilities(
        self,
        molecule: Any,
    ) -> list[float]:
        idx = self.index.get(normalize(molecule))
        if idx is None:
            return [0.0] * len(self.functional_group_vocabulary)
        return [
            float(value)
            for value in self.functional_group_probabilities_matrix[idx]
        ]

    def vector(self, molecule: Any) -> Any | None:
        idx = self.index.get(normalize(molecule))
        return None if idx is None else self.matrix[idx]

    def canonical_name(self, molecule: Any) -> str:
        key = normalize(molecule)
        return self.canonical_names.get(key, str(molecule))

    def similarity(self, left: Any, right: Any) -> float:
        left_idx = self.index.get(normalize(left))
        right_idx = self.index.get(normalize(right))
        if left_idx is None or right_idx is None:
            return 0.0
        return float(self.matrix[left_idx] @ self.matrix[right_idx])

    def pairwise_summary(
        self,
        candidate: str,
        partial_molecules: list[Any],
    ) -> tuple[float, float]:
        candidate_idx = self.index.get(normalize(candidate))
        partial_indices = [
            self.index[normalize(value)]
            for value in partial_molecules
            if normalize(value) in self.index
        ]
        if candidate_idx is None or not partial_indices:
            return 0.0, 0.0
        similarities = self.matrix[candidate_idx] @ self.matrix[partial_indices].T
        return float(similarities.mean()), float(similarities.max())


class MPCStructureModel:
    """MPC food-conditioned candidate and local-action model."""

    FEATURE_NAMES = (
        "frequency_prior",
        "cooccurrence_mean",
        "cooccurrence_max",
        "retrieved_profile_support",
        "unimol_mean_similarity",
        "unimol_max_similarity",
        "unimol_min_similarity",
        "unimol_query_compatibility",
        "unimol_query_distance",
        "perceptual_mean_similarity",
        "perceptual_max_similarity",
        "perceptual_residual_support",
    )
    PRIMARY_FEATURE_INDICES = (
        0,   # occurrence frequency
        1,   # mean co-occurrence with the observed partial set
        2,   # strongest co-occurrence with the observed partial set
        3,   # retrieved-profile residual support
    )
    STRUCTURAL_FEATURE_INDICES = (
        4,   # frozen UniMol mean similarity
        5,   # frozen UniMol maximum similarity
        6,   # frozen UniMol minimum similarity
        7,   # frozen UniMol compatibility with the partial-set centroid
        8,   # frozen UniMol distance from the partial-set centroid
        9,   # molecule-local perceptual mean similarity
        10,  # molecule-local perceptual maximum similarity
        11,  # expected descriptor-set residual support
    )

    def __init__(
        self,
        rows: list[dict[str, Any]],
        embeddings: EmbeddingStore | None,
        ablation: str,
        db_path: Path,
        calibrate_residuals: bool = True,
    ):
        self.rows = rows
        self.embeddings = embeddings
        self.ablation = ablation
        self.db_path = db_path
        self.full_profiles: list[set[str]] = []
        self.display_names: dict[str, str] = {}
        self.frequency: Counter[str] = Counter()
        self.cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
        self.food_token_df: Counter[str] = Counter()
        for row in rows:
            values = list(row.get("partial_molecules") or []) + list(row.get("missing_molecules") or [])
            profile = {normalize(value) for value in values if normalize(value)}
            self.full_profiles.append(profile)
            for value in values:
                key = normalize(value)
                if key:
                    self.display_names.setdefault(key, str(value))
            self.frequency.update(profile)
            self.food_token_df.update(
                set(normalize(row.get("target_food")).split())
            )
            for molecule in profile:
                self.cooccurrence[molecule].update(profile - {molecule})
        self.training_universe = sorted(self.display_names)
        self.perceptual_adapter: MPCPerceptualAdapter | None = None
        if embeddings is not None and ablation == "full":
            self.perceptual_adapter = MPCPerceptualAdapter(embeddings, db_path)
            # Molecule-local FlavorDB attributes and frozen UniMol vectors are
            # valid open-world candidate information.  Food-to-molecule links
            # are intentionally excluded to prevent split leakage.
            for name in embeddings.names:
                self.display_names.setdefault(
                    normalize(name),
                    self.perceptual_adapter.canonical_name(name),
                )
        self.universe = sorted(self.display_names)
        self.reduced_unimol_matrix: Any | None = None
        self.reduced_unimol_index: dict[str, int] = {}
        if embeddings is not None and ablation == "full":
            try:
                import numpy as np
                from sklearn.decomposition import TruncatedSVD
            except Exception as exc:
                raise OptimizedAgentError(
                    "scikit-learn is required for the MPC UniMol adapter"
                ) from exc
            reduced_dimension = min(
                32,
                embeddings.matrix.shape[0] - 1,
                embeddings.matrix.shape[1] - 1,
            )
            reduced = TruncatedSVD(
                n_components=reduced_dimension,
                random_state=0,
            ).fit_transform(embeddings.matrix)
            reduced = np.asarray(reduced, dtype=np.float32)
            reduced /= np.maximum(
                np.linalg.norm(reduced, axis=1, keepdims=True),
                1e-12,
            )
            self.reduced_unimol_matrix = reduced
            self.reduced_unimol_index = dict(embeddings.index)
        self.ranker: Any | None = None
        self.structural_ranker: Any | None = None
        self.boundary_swap_ranker: Any | None = None
        self.set_ranker: Any | None = None
        self.ranker_training_pairs = 0
        self.ranker_training_queries = 0
        self.structural_ranker_training_pairs = 0
        self.boundary_swap_training_pairs = 0
        self.boundary_swap_training_queries = 0
        self.boundary_swap_positive_count = 0
        self.boundary_swap_negative_count = 0
        self.boundary_swap_neutral_count = 0
        self.set_ranker_training_pairs = 0
        self.set_ranker_training_queries = 0
        self.group_demand_models: dict[str, Any] = {}
        self.group_demand_prevalence: list[float] = []
        self.group_demand_training_rows = 0
        self.retrieval_action_ranker: Any | None = None
        self.retrieval_action_training_pairs = 0
        self.retrieval_action_training_queries = 0
        self.retrieval_action_positive_count = 0
        self.retrieval_action_policy: dict[str, Any] = {
            "budget": 0,
            "threshold": 1.0,
        }
        self.retrieval_action_calibration: dict[str, Any] = {
            "protocol": "disabled",
            "attempts": {},
        }
        self.residual_policy: dict[str, dict[str, int]] = {
            "retrieval": {"global": 0},
            "structural": {"global": 0},
            "complementarity": {"global": 0},
        }
        self.residual_calibration: dict[str, Any] = {
            "protocol": "disabled",
            "query_count": 0,
            "policies": {},
        }
        self.metric_group_policy: dict[str, Any] = {
            "budget": 0,
            "enabled": False,
            "minimum_expected_f1_gain": None,
        }
        self.metric_group_calibration: dict[str, Any] = {
            "protocol": "disabled",
            "selection_metric": "macro_functional_group_f1",
            "attempts": {},
        }
        self.add_necessity_verifier: Any | None = None
        self.remove_safety_verifier: Any | None = None
        self.dual_gate_policy: dict[str, Any] = {
            "enabled": False,
            "add_threshold": None,
            "remove_threshold": None,
        }
        self.functional_group_sets: dict[str, set[str]] = {}
        self.functional_group_vocabulary: list[str] = []
        self.functional_group_prevalence: dict[str, float] = {}
        # The legal MPC candidate catalog is independent of UniMol coverage.
        # This prevents a no-UniMol ablation from silently shrinking the
        # molecule universe.
        connection = sqlite3.connect(db_path)
        try:
            for table in ("molecules", "molecules_all"):
                for common_name, raw_groups in connection.execute(
                    f"SELECT common_name, functional_groups FROM {table} "
                    "WHERE common_name IS NOT NULL "
                    "AND TRIM(common_name) <> ''"
                ):
                    key = normalize(common_name)
                    if key:
                        self.display_names.setdefault(
                            key,
                            str(common_name),
                        )
                        groups = parse_official_functional_groups(
                            raw_groups
                        )
                        if groups:
                            self.functional_group_sets[key] = groups
        finally:
            connection.close()
        self.universe = sorted(self.display_names)
        self.catalog_size = len(self.universe)
        self.functional_group_vocabulary = sorted(
            {
                group
                for groups in self.functional_group_sets.values()
                for group in groups
            }
        )
        group_document_frequency: Counter[str] = Counter()
        for row in rows:
            row_groups = self._functional_group_set(
                {
                    normalize(value)
                    for value in row.get("missing_molecules") or []
                    if normalize(value)
                }
            )
            group_document_frequency.update(row_groups)
        self.functional_group_prevalence = {
            group: group_document_frequency[group] / max(1, len(rows))
            for group in self.functional_group_vocabulary
        }
        if ablation == "full":
            self._fit_group_demand_models()
            self._fit_rankers()
            # v15 does not train the legacy v12 exact-molecule action or
            # UniMol boundary rankers.  Their objectives are not aligned with
            # the functional-group metric and they dominated grouped-OOF
            # runtime without receiving prediction authority.
            if calibrate_residuals:
                self._calibrate_v15_dual_gate_policy()

    def _set_feature_vector(
        self,
        row: dict[str, Any],
        selected: list[str] | set[str],
        raw_features: dict[str, list[float]],
    ) -> list[float]:
        """Permutation-invariant features for an exact-cardinality set.

        All statistics are generic molecule/set quantities.  They do not use
        FoodPuzzle evaluation labels, functional-group caches, sample IDs, or
        cardinality buckets.
        """
        keys = stable_unique([normalize(value) for value in selected])
        feature_rows = [
            raw_features[key] for key in keys if key in raw_features
        ]
        feature_dimension = len(self.FEATURE_NAMES)
        if feature_rows:
            feature_mean = [
                sum(values[column] for values in feature_rows)
                / len(feature_rows)
                for column in range(feature_dimension)
            ]
            feature_min = [
                min(values[column] for values in feature_rows)
                for column in range(feature_dimension)
            ]
        else:
            feature_mean = [0.0] * feature_dimension
            feature_min = [0.0] * feature_dimension

        pairwise: list[float] = []
        # Pairwise statistics use a deterministic, evenly spaced sketch for
        # large sets.  This keeps both training and inference bounded while
        # preserving coverage across the canonicalized set order.
        pairwise_keys = keys
        if len(pairwise_keys) > 32:
            pairwise_keys = [
                pairwise_keys[
                    round(
                        index
                        * (len(pairwise_keys) - 1)
                        / 31
                    )
                ]
                for index in range(32)
            ]
        selected_vectors = [
            vector
            for key in pairwise_keys
            if self.embeddings is not None
            and (vector := self.embeddings.vector(key)) is not None
        ]
        for left_index, left in enumerate(selected_vectors):
            for right in selected_vectors[left_index + 1 :]:
                pairwise.append(EmbeddingStore.cosine(left, right))
        pairwise_mean = (
            sum(pairwise) / len(pairwise) if pairwise else 0.0
        )
        pairwise_min = min(pairwise) if pairwise else 0.0
        pairwise_max = max(pairwise) if pairwise else 0.0
        pairwise_variance = (
            sum((value - pairwise_mean) ** 2 for value in pairwise)
            / len(pairwise)
            if pairwise
            else 0.0
        )

        partial_vectors = [
            vector
            for value in row.get("partial_molecules") or []
            if self.embeddings is not None
            and (vector := self.embeddings.vector(value)) is not None
        ]
        centroid_compatibility = 0.0
        if selected_vectors and partial_vectors:
            np = self.embeddings.np
            selected_centroid = np.asarray(selected_vectors).mean(axis=0)
            partial_centroid = np.asarray(partial_vectors).mean(axis=0)
            centroid_compatibility = EmbeddingStore.cosine(
                selected_centroid,
                partial_centroid,
            )

        observed_count = len(row.get("partial_molecules") or [])
        selected_count = len(keys)
        return (
            feature_mean
            + feature_min
            + [
                pairwise_mean,
                pairwise_min,
                pairwise_max,
                math.sqrt(max(0.0, pairwise_variance)),
                centroid_compatibility,
                math.log1p(selected_count),
                math.log1p(observed_count),
                selected_count / max(1, selected_count + observed_count),
            ]
        )

    def _score_set(
        self,
        row: dict[str, Any],
        selected: list[str] | set[str],
        raw_features: dict[str, list[float]],
    ) -> float:
        features = self._set_feature_vector(
            row,
            selected,
            raw_features,
        )
        if self.set_ranker is not None:
            return float(
                self.set_ranker.decision_function([features])[0]
            )
        # A model-free fallback preserves a sensible conditional set signal
        # for ablations or very small training collections.
        feature_dimension = len(self.FEATURE_NAMES)
        mean_features = features[:feature_dimension]
        pairwise_offset = 2 * feature_dimension
        return (
            0.50 * mean_features[1]
            + 0.20 * mean_features[3]
            + 0.20 * mean_features[7]
            + 0.10 * features[pairwise_offset + 4]
        )

    def _build_query_context(
        self,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        partial_values = list(row.get("partial_molecules") or [])
        partial = {
            normalize(value) for value in partial_values if normalize(value)
        }
        embedding_vectors = (
            [
                vector
                for value in partial_values
                if (vector := self.embeddings.vector(value)) is not None
            ]
            if self.embeddings is not None
            else []
        )
        reduced_query_vector = None
        partial_indices = [
            self.reduced_unimol_index[normalize(value)]
            for value in partial_values
            if normalize(value) in self.reduced_unimol_index
        ]
        if self.reduced_unimol_matrix is not None and partial_indices:
            reduced_query_vector = self.reduced_unimol_matrix[
                partial_indices
            ].mean(axis=0)
            query_norm = float(
                self.embeddings.np.linalg.norm(reduced_query_vector)
            )
            reduced_query_vector = (
                reduced_query_vector / max(query_norm, 1e-12)
            )
        perceptual_partial_indices = (
            [
                self.perceptual_adapter.index[normalize(value)]
                for value in partial_values
                if normalize(value) in self.perceptual_adapter.index
            ]
            if self.perceptual_adapter is not None
            else []
        )
        return {
            "partial_values": partial_values,
            "partial": partial,
            "embedding_vectors": embedding_vectors,
            "reduced_query_vector": reduced_query_vector,
            "perceptual_partial_indices": perceptual_partial_indices,
        }

    def _query_features(
        self,
        row: dict[str, Any],
        candidate: str,
        retrieved_support: dict[str, float],
        excluded_profile: set[str] | None = None,
        expected_attributes: dict[str, float] | None = None,
        query_context: dict[str, Any] | None = None,
    ) -> list[float]:
        context = query_context or self._build_query_context(row)
        partial_values = context["partial_values"]
        partial = context["partial"]
        excluded = excluded_profile or set()
        candidate_vector = self.embeddings.vector(candidate) if self.embeddings is not None else None
        similarities = [
            EmbeddingStore.cosine(candidate_vector, vector)
            for vector in context["embedding_vectors"]
        ]
        mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0
        min_similarity = min(similarities) if similarities else 0.0
        query_compatibility = 0.0
        query_distance = 1.0
        candidate_idx = self.reduced_unimol_index.get(candidate)
        query_vector = context["reduced_query_vector"]
        if (
            self.reduced_unimol_matrix is not None
            and candidate_idx is not None
            and query_vector is not None
        ):
            candidate_reduced = self.reduced_unimol_matrix[candidate_idx]
            query_compatibility = float(candidate_reduced @ query_vector)
            query_distance = float(
                self.embeddings.np.abs(
                    candidate_reduced - query_vector
                ).mean()
            )
        perceptual_mean = 0.0
        perceptual_max = 0.0
        if (
            self.perceptual_adapter is not None
            and context["perceptual_partial_indices"]
        ):
            candidate_perceptual_idx = self.perceptual_adapter.index.get(
                candidate
            )
            if candidate_perceptual_idx is not None:
                perceptual_similarities = (
                    self.perceptual_adapter.matrix[
                        candidate_perceptual_idx
                    ]
                    @ self.perceptual_adapter.matrix[
                        context["perceptual_partial_indices"]
                    ].T
                )
                perceptual_mean = float(
                    perceptual_similarities.mean()
                )
                perceptual_max = float(
                    perceptual_similarities.max()
                )
        cooccurrence_values = [
            (
                self.cooccurrence[candidate].get(existing, 0)
                - int(candidate in excluded and existing in excluded)
            )
            / max(1, self.frequency[existing] - int(existing in excluded))
            for existing in partial
        ]
        cooccurrence_mean = (
            sum(cooccurrence_values) / len(cooccurrence_values) if cooccurrence_values else 0.0
        )
        cooccurrence_max = max(cooccurrence_values) if cooccurrence_values else 0.0
        adjusted_frequency = self.frequency[candidate] - int(candidate in excluded)
        prior = math.log1p(max(0, adjusted_frequency)) / max(
            math.log1p(max(1, len(self.rows) - int(bool(excluded)))),
            1e-12,
        )
        residual_support = 0.0
        if self.perceptual_adapter is not None and expected_attributes:
            residual_support = sum(
                expected_attributes.get(attribute, 0.0)
                for attribute in self.perceptual_adapter.attributes(candidate)
            )
            residual_support /= max(
                sum(expected_attributes.values()),
                1e-12,
            )
        return [
            prior,
            cooccurrence_mean,
            cooccurrence_max,
            retrieved_support.get(candidate, 0.0),
            mean_similarity,
            max_similarity,
            min_similarity,
            query_compatibility,
            query_distance,
            perceptual_mean,
            perceptual_max,
            residual_support,
        ]

    def _build_retrieved_support(
        self,
        row: dict[str, Any],
        exclude_train_index: int | None = None,
        top_k: int = 5,
    ) -> dict[str, float]:
        target_tokens = set(normalize(row.get("target_food")).split())
        partial = {normalize(value) for value in row.get("partial_molecules") or []}
        scored_profiles: list[tuple[float, int]] = []
        for idx, (train_row, profile) in enumerate(zip(self.rows, self.full_profiles)):
            if idx == exclude_train_index:
                continue
            train_tokens = set(normalize(train_row.get("target_food")).split())
            food_overlap = len(target_tokens & train_tokens) / max(1, len(target_tokens | train_tokens))
            partial_overlap = len(partial & profile) / max(1, len(partial | profile))
            retrieval_score = 0.35 * food_overlap + 0.65 * partial_overlap
            scored_profiles.append((retrieval_score, idx))
        scored_profiles.sort(key=lambda item: (-item[0], item[1]))
        support: dict[str, float] = {}
        for score, idx in scored_profiles[:top_k]:
            for molecule in self.full_profiles[idx]:
                support[molecule] = max(support.get(molecule, 0.0), score)
        return support

    def _retrieved_profiles(
        self,
        row: dict[str, Any],
        top_k: int = 10,
        exclude_train_index: int | None = None,
    ) -> list[tuple[float, int]]:
        target_tokens = set(normalize(row.get("target_food")).split())
        partial = {normalize(value) for value in row.get("partial_molecules") or []}
        scored: list[tuple[float, int]] = []
        for idx, (train_row, profile) in enumerate(
            zip(self.rows, self.full_profiles)
        ):
            if idx == exclude_train_index:
                continue
            train_tokens = set(normalize(train_row.get("target_food")).split())
            food_overlap = len(target_tokens & train_tokens) / max(
                1, len(target_tokens | train_tokens)
            )
            partial_overlap = len(partial & profile) / max(
                1, len(partial | profile)
            )
            score = 0.35 * food_overlap + 0.65 * partial_overlap
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[:top_k]

    def _build_idf_retrieved_support(
        self,
        row: dict[str, Any],
        exclude_train_index: int | None = None,
        top_k: int = 15,
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Aggregate complementary candidates across weighted neighbours.

        Unlike the legacy max-support retriever, this channel discounts common
        molecules, rewards agreement across independent food profiles, and is
        used only to expand the H1 boundary candidate pool.
        """
        partial = {
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        }
        target_tokens = set(normalize(row.get("target_food")).split())
        row_count = max(
            1,
            len(self.rows) - int(exclude_train_index is not None),
        )

        def molecule_idf(key: str) -> float:
            count = self.frequency[key]
            if (
                exclude_train_index is not None
                and key in self.full_profiles[exclude_train_index]
            ):
                count -= 1
            return math.log((row_count + 1) / (max(0, count) + 1)) + 1.0

        def token_idf(token: str) -> float:
            count = self.food_token_df[token]
            if (
                exclude_train_index is not None
                and token
                in set(
                    normalize(
                        self.rows[exclude_train_index].get("target_food")
                    ).split()
                )
            ):
                count -= 1
            return math.log((row_count + 1) / (max(0, count) + 1)) + 1.0

        partial_weight = sum(molecule_idf(key) for key in partial)
        token_weight = sum(token_idf(token) for token in target_tokens)
        neighbours: list[tuple[float, int]] = []
        for index, (train_row, profile) in enumerate(
            zip(self.rows, self.full_profiles)
        ):
            if index == exclude_train_index:
                continue
            profile_containment = sum(
                molecule_idf(key) for key in partial & profile
            ) / max(partial_weight, 1e-12)
            train_tokens = set(
                normalize(train_row.get("target_food")).split()
            )
            food_containment = (
                sum(
                    token_idf(token)
                    for token in target_tokens & train_tokens
                )
                / max(token_weight, 1e-12)
                if target_tokens
                else 0.0
            )
            score = (
                0.75 * profile_containment
                + 0.25 * food_containment
            )
            neighbours.append((score, index))
        neighbours.sort(key=lambda item: (-item[0], item[1]))
        support: defaultdict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for rank, (score, index) in enumerate(
            neighbours[:top_k],
            1,
        ):
            weight = max(0.0, score) / math.log2(rank + 1.0)
            for candidate in self.full_profiles[index] - partial:
                support[candidate] += weight
                counts[candidate] += 1
        return dict(support), dict(counts)

    def _group_query_features(
        self,
        row: dict[str, Any],
        exclude_train_index: int | None = None,
    ) -> list[float]:
        adapter = self.perceptual_adapter
        if adapter is None:
            return []
        group_count = len(adapter.functional_group_vocabulary)
        partial_probabilities = [
            adapter.functional_group_probabilities(value)
            for value in row.get("partial_molecules") or []
            if adapter.vector(value) is not None
        ]
        if partial_probabilities:
            partial_mean = [
                sum(values[column] for values in partial_probabilities)
                / len(partial_probabilities)
                for column in range(group_count)
            ]
            partial_max = [
                max(values[column] for values in partial_probabilities)
                for column in range(group_count)
            ]
        else:
            partial_mean = [0.0] * group_count
            partial_max = [0.0] * group_count

        retrieval_support = [0.0] * group_count
        retrieval_denominator = 0.0
        for retrieval_score, idx in self._retrieved_profiles(
            row,
            top_k=10,
            exclude_train_index=exclude_train_index,
        ):
            weight = max(1e-6, retrieval_score)
            retrieval_denominator += weight
            neighbor_groups: set[str] = set()
            for molecule in self.full_profiles[idx]:
                neighbor_groups.update(
                    adapter.functional_groups(molecule)
                )
            for group in neighbor_groups:
                column = adapter.functional_group_index.get(group)
                if column is not None:
                    retrieval_support[column] += weight
        if retrieval_denominator:
            retrieval_support = [
                value / retrieval_denominator
                for value in retrieval_support
            ]

        partial_unimol_features: list[float] = []
        partial_indices = [
            self.reduced_unimol_index[normalize(value)]
            for value in row.get("partial_molecules") or []
            if normalize(value) in self.reduced_unimol_index
        ]
        if self.reduced_unimol_matrix is not None:
            dimension = int(self.reduced_unimol_matrix.shape[1])
            if partial_indices:
                partial_matrix = self.reduced_unimol_matrix[
                    partial_indices
                ]
                partial_unimol_features = [
                    float(value) for value in partial_matrix.mean(axis=0)
                ] + [
                    float(value) for value in partial_matrix.max(axis=0)
                ]
            else:
                partial_unimol_features = [0.0] * (2 * dimension)

        n = max(0, int(row.get("n") or 0))
        return (
            partial_mean
            + partial_max
            + retrieval_support
            + partial_unimol_features
            + [
                math.log1p(n) / math.log1p(250),
                float(n <= 5),
                float(5 < n <= 20),
                float(n > 20),
            ]
        )

    def _fit_group_demand_models(self) -> None:
        """Record deterministic, train-only functional-group supervision.

        v13 deliberately has no UniMol group probe.  Group labels are parsed
        directly from molecule-local FlavorDB fields, while food-conditional
        demand is estimated from retrieved training profiles at inference.
        """
        self.group_demand_models = {}
        self.group_demand_training_rows = len(self.rows)
        self.group_demand_prevalence = [
            self.functional_group_prevalence.get(group, 0.0)
            for group in self.functional_group_vocabulary
        ]

    def _predict_group_demand(
        self,
        row: dict[str, Any],
    ) -> list[float]:
        if not self.functional_group_vocabulary:
            return []
        weighted_counts: Counter[str] = Counter()
        denominator = 0.0
        for rank, (retrieval_score, index) in enumerate(
            self._retrieved_profiles(row, top_k=15),
            1,
        ):
            weight = max(0.0, retrieval_score) / math.log2(rank + 1.0)
            if weight <= 0.0:
                continue
            denominator += weight
            missing_groups = self._functional_group_set(
                {
                    normalize(value)
                    for value in self.rows[index].get(
                        "missing_molecules"
                    )
                    or []
                    if normalize(value)
                }
            )
            for group in missing_groups:
                weighted_counts[group] += weight
        probabilities: list[float] = []
        for group in self.functional_group_vocabulary:
            retrieved_probability = (
                weighted_counts[group] / denominator
                if denominator
                else 0.0
            )
            prior = self.functional_group_prevalence.get(group, 0.0)
            probability = (
                0.85 * retrieved_probability + 0.15 * prior
                if denominator
                else prior
            )
            probabilities.append(max(0.0, min(1.0, probability)))
        return probabilities

    def _expected_attribute_weights(
        self,
        row: dict[str, Any],
        exclude_train_index: int | None = None,
    ) -> dict[str, float]:
        if self.perceptual_adapter is None:
            return {}
        partial = {
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        }
        weights: dict[str, float] = defaultdict(float)
        denominator = 0.0
        for retrieval_score, idx in self._retrieved_profiles(
            row,
            exclude_train_index=exclude_train_index,
        ):
            weight = max(1e-6, retrieval_score)
            denominator += weight
            for molecule in self.full_profiles[idx] - partial:
                for attribute in self.perceptual_adapter.attributes(molecule):
                    weights[attribute] += weight
        if denominator:
            for attribute in list(weights):
                weights[attribute] = (
                    weights[attribute] / denominator
                ) * self.perceptual_adapter.attribute_idf.get(attribute, 1.0)
        return dict(weights)

    def _fit_rankers(self) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            raise OptimizedAgentError("scikit-learn is required for the MPC adapter") from exc
        pair_features: list[list[float]] = []
        structural_pair_features: list[list[float]] = []
        set_pair_features: list[list[float]] = []
        set_pair_labels: list[int] = []
        set_pair_weights: list[float] = []
        labels: list[int] = []
        sample_weights: list[float] = []
        frequency_order = sorted(
            self.training_universe,
            key=lambda key: (-self.frequency[key], key),
        )
        training_queries = 0
        for idx, source_row in enumerate(self.rows):
            full_profile = self.full_profiles[idx]
            original_positives = {
                normalize(value)
                for value in source_row.get("missing_molecules") or []
                if normalize(value) in full_profile
            }
            query_specs: list[tuple[str, set[str], set[str]]] = []
            original_partial = {
                normalize(value)
                for value in source_row.get("partial_molecules") or []
                if normalize(value)
            }
            if original_partial and original_positives:
                query_specs.append(
                    ("task_shaped", original_partial, original_positives)
                )
            # v11 deliberately trains on the task-shaped observation process.
            # The reconstructed train split already supplies one partial/hidden
            # profile per food at the same high-missingness regime as test.
            # Adding several low-missingness masks would make those easier
            # surrogate queries dominate the empirical risk.

            for mask_kind, partial, positives in query_specs:
                masked_row = {
                    "id": f"{source_row.get('id')}:{mask_kind}",
                    "target_food": source_row.get("target_food"),
                    "partial_molecules": [
                        self.display_names.get(value, value)
                        for value in sorted(partial)
                    ],
                    "n": len(positives),
                }
                retrieved_support = self._build_retrieved_support(
                    masked_row,
                    exclude_train_index=idx,
                    top_k=10,
                )
                expected_attributes = self._expected_attribute_weights(
                    masked_row,
                    exclude_train_index=idx,
                )
                query_context = self._build_query_context(masked_row)
                cooccurrence_scores: Counter[str] = Counter()
                for observed in partial:
                    for candidate, count in self.cooccurrence.get(
                        observed,
                        {},
                    ).items():
                        if candidate not in full_profile:
                            cooccurrence_scores[candidate] += count
                hard_pool = stable_unique(
                    [
                        candidate
                        for candidate, _ in sorted(
                            retrieved_support.items(),
                            key=lambda item: (-item[1], item[0]),
                        )[:40]
                        if candidate not in full_profile
                    ]
                    + [
                        candidate
                        for candidate, _ in cooccurrence_scores.most_common(40)
                        if candidate not in full_profile
                    ]
                    + [
                        candidate
                        for candidate in frequency_order[:40]
                        if candidate not in full_profile
                    ]
                )
                random_pool = [
                    candidate
                    for candidate in self.training_universe
                    if candidate not in full_profile
                    and candidate not in set(hard_pool)
                ]
                random.Random(
                    f"mpc-v6-unlabeled:{source_row.get('id')}:{mask_kind}"
                ).shuffle(random_pool)
                unlabeled = (hard_pool[:20] + random_pool[:10])[:30]
                selected_positives = sorted(positives)
                random.Random(
                    f"mpc-v6-positive:{source_row.get('id')}:{mask_kind}"
                ).shuffle(selected_positives)
                selected_positives = selected_positives[:20]
                if not selected_positives or not unlabeled:
                    continue
                all_positive_features = {
                    candidate: self._query_features(
                        masked_row,
                        candidate,
                        retrieved_support,
                        full_profile,
                        expected_attributes,
                        query_context,
                    )
                    for candidate in sorted(positives)
                }
                unlabeled_features = {
                    candidate: self._query_features(
                        masked_row,
                        candidate,
                        retrieved_support,
                        full_profile,
                        expected_attributes,
                        query_context,
                    )
                    for candidate in unlabeled
                }
                positive_features = {
                    candidate: all_positive_features[candidate]
                    for candidate in selected_positives
                    if candidate in all_positive_features
                }
                pair_count = len(positive_features) * len(unlabeled_features)
                if not pair_count:
                    continue
                # The outside-profile side is unlabeled rather than a certain
                # real-world absence.  A reduced per-query weight prevents
                # large profiles and potentially false negatives from
                # dominating the pairwise objective.
                pair_weight = 0.35 / pair_count
                for positive in positive_features.values():
                    for unknown in unlabeled_features.values():
                        difference = [
                            left - right
                            for left, right in zip(positive, unknown)
                        ]
                        pair_features.append(
                            [
                                difference[index]
                                for index in self.PRIMARY_FEATURE_INDICES
                            ]
                        )
                        structural_pair_features.append(
                            [
                                difference[index]
                                for index in self.STRUCTURAL_FEATURE_INDICES
                            ]
                        )
                        labels.append(1)
                        sample_weights.append(pair_weight)
                        pair_features.append(
                            [
                                -difference[index]
                                for index in self.PRIMARY_FEATURE_INDICES
                            ]
                        )
                        structural_pair_features.append(
                            [
                                -difference[index]
                                for index in self.STRUCTURAL_FEATURE_INDICES
                            ]
                        )
                        labels.append(0)
                        sample_weights.append(pair_weight)

                training_queries += 1
        if not pair_features or len(set(labels)) < 2:
            return
        model = LogisticRegression(
            C=0.25,
            max_iter=2000,
            random_state=0,
        )
        model.fit(pair_features, labels, sample_weight=sample_weights)
        self.ranker = model
        self.ranker_training_pairs = len(pair_features) // 2
        self.ranker_training_queries = training_queries
        if self.perceptual_adapter is not None and structural_pair_features:
            structural_model = LogisticRegression(
                C=0.1,
                max_iter=2000,
                random_state=0,
            )
            structural_model.fit(
                structural_pair_features,
                labels,
                sample_weight=sample_weights,
            )
            self.structural_ranker = structural_model
            self.structural_ranker_training_pairs = (
                len(structural_pair_features) // 2
            )
        # v10's global exact-cardinality set energy is intentionally retired.
        # v11 admits only bounded residual exchanges selected by grouped OOF.

    @staticmethod
    def _retrieval_action_features(
        add_item: dict[str, Any],
        remove_item: dict[str, Any],
        n: int,
    ) -> list[float]:
        """Describe one food-conditioned add/remove action without UniMol."""
        scale = max(1.0, float(n))
        return [
            float(add_item.get("occurrence_score") or 0.0)
            - float(remove_item.get("occurrence_score") or 0.0),
            float(add_item.get("idf_retrieved_support") or 0.0)
            - float(remove_item.get("idf_retrieved_support") or 0.0),
            (
                float(
                    add_item.get("idf_retrieved_profile_count")
                    or 0.0
                )
                - float(
                    remove_item.get("idf_retrieved_profile_count")
                    or 0.0
                )
            )
            / 15.0,
            float(add_item.get("retrieved_profile_support") or 0.0)
            - float(
                remove_item.get("retrieved_profile_support") or 0.0
            ),
            (
                float(remove_item.get("occurrence_rank") or 0.0)
                - float(add_item.get("occurrence_rank") or 0.0)
            )
            / scale,
            float(add_item.get("idf_retrieved_support") or 0.0),
            float(
                add_item.get("idf_retrieved_profile_count") or 0.0
            )
            / 15.0,
        ]

    def _retrieval_action_pairs(
        self,
        items: list[dict[str, Any]],
        n: int,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if n <= 0 or len(items) <= n:
            return []
        item_by_key = {
            normalize(item.get("molecule")): item
            for item in items
            if normalize(item.get("molecule"))
        }
        occurrence = sorted(
            item_by_key,
            key=lambda key: (
                int(
                    item_by_key[key].get("occurrence_rank")
                    or 10**9
                ),
                key,
            ),
        )
        selected = occurrence[:n]
        selected_set = set(selected)
        removals = selected[max(0, n - min(10, n)) :]
        additions = sorted(
            (
                key
                for key in occurrence
                if key not in selected_set
                and float(
                    item_by_key[key].get(
                        "idf_retrieved_support"
                    )
                    or 0.0
                )
                > 0.0
            ),
            key=lambda key: (
                -float(
                    item_by_key[key].get(
                        "idf_retrieved_support"
                    )
                    or 0.0
                ),
                -int(
                    item_by_key[key].get(
                        "idf_retrieved_profile_count"
                    )
                    or 0
                ),
                int(
                    item_by_key[key].get("occurrence_rank")
                    or 10**9
                ),
                key,
            ),
        )[:30]
        return [
            (item_by_key[add_key], item_by_key[remove_key])
            for add_key in additions
            for remove_key in removals
        ]

    def _fit_retrieval_action_ranker(self) -> None:
        """Fit a low-capacity verifier-utility prior on local actions."""
        if self.ranker is None:
            return
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise OptimizedAgentError(
                "scikit-learn is required for retrieval action fitting"
            ) from exc
        features: list[list[float]] = []
        labels: list[int] = []
        weights: list[float] = []
        training_queries = 0
        for index, source_row in enumerate(self.rows):
            gold = {
                normalize(value)
                for value in source_row.get("missing_molecules") or []
                if normalize(value)
            }
            partial = {
                normalize(value)
                for value in source_row.get("partial_molecules") or []
                if normalize(value)
            }
            n = len(gold)
            if not gold or not partial:
                continue
            query = {
                "id": f"{source_row.get('id')}:v12-action-train",
                "target_food": source_row.get("target_food"),
                "partial_molecules": list(
                    source_row.get("partial_molecules") or []
                ),
                "n": n,
            }
            items = self._boundary_training_items(
                query,
                exclude_train_index=index,
                limit=n + 40,
            )
            pairs = self._retrieval_action_pairs(items, n)
            if not pairs:
                continue
            query_weight = 1.0 / len(pairs)
            for add_item, remove_item in pairs:
                add_key = normalize(add_item.get("molecule"))
                remove_key = normalize(remove_item.get("molecule"))
                beneficial = add_key in gold and remove_key not in gold
                features.append(
                    self._retrieval_action_features(
                        add_item,
                        remove_item,
                        n,
                    )
                )
                labels.append(int(beneficial))
                weights.append(query_weight)
            training_queries += 1
        if not features or len(set(labels)) < 2:
            return
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.05,
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            ),
        )
        model.fit(
            features,
            labels,
            logisticregression__sample_weight=weights,
        )
        self.retrieval_action_ranker = model
        self.retrieval_action_training_pairs = len(features)
        self.retrieval_action_training_queries = training_queries
        self.retrieval_action_positive_count = sum(labels)

    def _rank_retrieval_actions(
        self,
        items: list[dict[str, Any]],
        n: int,
    ) -> list[dict[str, Any]]:
        if self.retrieval_action_ranker is None:
            return []
        actions: list[dict[str, Any]] = []
        for add_item, remove_item in self._retrieval_action_pairs(
            items,
            n,
        ):
            features = self._retrieval_action_features(
                add_item,
                remove_item,
                n,
            )
            probability = float(
                self.retrieval_action_ranker.predict_proba(
                    [features]
                )[0, 1]
            )
            independent_support_count = sum(
                [
                    float(
                        add_item.get("idf_retrieved_support") or 0.0
                    )
                    > 0.0,
                    float(
                        add_item.get("retrieved_profile_support")
                        or 0.0
                    )
                    > 0.0,
                    float(add_item.get("cooccurrence") or 0.0)
                    > 0.0,
                ]
            )
            actions.append(
                {
                    "remove_key": normalize(
                        remove_item.get("molecule")
                    ),
                    "add_key": normalize(add_item.get("molecule")),
                    "utility_probability": probability,
                    "h1_margin_cost": (
                        float(
                            remove_item.get("occurrence_score")
                            or 0.0
                        )
                        - float(
                            add_item.get("occurrence_score")
                            or 0.0
                        )
                    ),
                    "idf_retrieved_support": float(
                        add_item.get("idf_retrieved_support") or 0.0
                    ),
                    "idf_retrieved_profile_count": int(
                        add_item.get(
                            "idf_retrieved_profile_count"
                        )
                        or 0
                    ),
                    "legacy_retrieved_support": float(
                        add_item.get("retrieved_profile_support")
                        or 0.0
                    ),
                    "independent_statistical_support_count": int(
                        independent_support_count
                    ),
                }
            )
        actions.sort(
            key=lambda item: (
                -float(item["utility_probability"]),
                -int(item["independent_statistical_support_count"]),
                -float(item["idf_retrieved_support"]),
                item["add_key"],
                item["remove_key"],
            )
        )
        return actions

    def _reduced_unimol_vector(self, molecule: Any) -> Any | None:
        if self.reduced_unimol_matrix is None:
            return None
        index = self.reduced_unimol_index.get(normalize(molecule))
        if index is None:
            return None
        return self.reduced_unimol_matrix[index]

    def _swap_feature_vector(
        self,
        row: dict[str, Any],
        item_by_key: dict[str, dict[str, Any]],
        selected_keys: list[str],
        add_key: str,
        remove_key: str,
        group_demand: list[float],
        expected_attributes: dict[str, float],
    ) -> list[float]:
        """Describe one H1-boundary exchange in its query/set context."""
        try:
            import numpy as np
        except Exception as exc:
            raise OptimizedAgentError(
                "numpy is required for the MPC boundary adapter"
            ) from exc
        add_item = item_by_key[add_key]
        remove_item = item_by_key[remove_key]
        add_vector = self._reduced_unimol_vector(add_key)
        remove_vector = self._reduced_unimol_vector(remove_key)
        dimension = (
            int(self.reduced_unimol_matrix.shape[1])
            if self.reduced_unimol_matrix is not None
            else 0
        )
        zero_vector = np.zeros(dimension, dtype=np.float32)
        add_vector = add_vector if add_vector is not None else zero_vector
        remove_vector = (
            remove_vector if remove_vector is not None else zero_vector
        )
        vector_difference = add_vector - remove_vector

        partial_vectors = [
            vector
            for value in row.get("partial_molecules") or []
            if (vector := self._reduced_unimol_vector(value)) is not None
        ]
        selected_vectors = [
            vector
            for key in selected_keys
            if (vector := self._reduced_unimol_vector(key)) is not None
        ]
        partial_centroid = (
            np.asarray(partial_vectors, dtype=np.float32).mean(axis=0)
            if partial_vectors
            else zero_vector
        )
        selected_centroid = (
            np.asarray(selected_vectors, dtype=np.float32).mean(axis=0)
            if selected_vectors
            else zero_vector
        )

        def attribute_demand(key: str) -> float:
            if self.perceptual_adapter is None:
                return 0.0
            return sum(
                expected_attributes.get(attribute, 0.0)
                for attribute in self.perceptual_adapter.attributes(key)
            )

        selected_items = [
            item_by_key[key]
            for key in selected_keys
            if key in item_by_key and key != remove_key
        ]
        group_gain = (
            self._soft_group_f1(
                selected_items + [add_item],
                group_demand,
            )
            - self._soft_group_f1(
                selected_items + [remove_item],
                group_demand,
            )
            if group_demand
            else 0.0
        )

        scalar_fields = (
            "occurrence_score",
            "retrieved_profile_support",
            "functional_group_demand_score",
            "perceptual_residual_support",
            "unimol_query_compatibility",
            "unimol_query_distance",
            "unimol_mean_similarity",
            "unimol_max_similarity",
        )
        scalar_differences = [
            float(add_item.get(field) or 0.0)
            - float(remove_item.get(field) or 0.0)
            for field in scalar_fields
        ]
        occurrence_rank_margin = (
            int(remove_item.get("occurrence_rank") or 10**9)
            - int(add_item.get("occurrence_rank") or 10**9)
        ) / max(1, len(item_by_key))
        mapping_features = [
            float(self._reduced_unimol_vector(add_key) is not None),
            float(self._reduced_unimol_vector(remove_key) is not None),
        ]
        return (
            scalar_differences
            + [
                occurrence_rank_margin,
                attribute_demand(add_key) - attribute_demand(remove_key),
                group_gain,
            ]
            + mapping_features
            + [float(value) for value in vector_difference]
            + [
                float(value)
                for value in vector_difference * partial_centroid
            ]
            + [
                float(value)
                for value in vector_difference * selected_centroid
            ]
        )

    def _fit_boundary_swap_adapter(self) -> None:
        """Fit a low-capacity UniMol adapter on beneficial H1 swaps only."""
        if (
            self.perceptual_adapter is None
            or self.reduced_unimol_matrix is None
            or self.ranker is None
        ):
            return
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise OptimizedAgentError(
                "scikit-learn is required for the MPC boundary adapter"
            ) from exc
        features: list[list[float]] = []
        labels: list[int] = []
        sample_weights: list[float] = []
        training_queries = 0
        for index, source_row in enumerate(self.rows):
            gold = {
                normalize(value)
                for value in source_row.get("missing_molecules") or []
                if normalize(value)
            }
            partial = {
                normalize(value)
                for value in source_row.get("partial_molecules") or []
                if normalize(value)
            }
            n = len(gold)
            if not partial or n <= 0:
                continue
            masked_row = {
                "id": f"{source_row.get('id')}:boundary-train",
                "target_food": source_row.get("target_food"),
                "partial_molecules": list(
                    source_row.get("partial_molecules") or []
                ),
                "n": n,
            }
            items = self._boundary_training_items(
                masked_row,
                exclude_train_index=index,
                limit=n + 30,
            )
            item_by_key = {
                normalize(item.get("molecule")): item for item in items
            }
            ordered = sorted(
                item_by_key,
                key=lambda key: (
                    int(
                        item_by_key[key].get("occurrence_rank") or 10**9
                    ),
                    key,
                ),
            )
            selected = ordered[:n]
            remove_candidates = [
                key
                for key in selected[max(0, n - 10) :]
                if self._reduced_unimol_vector(key) is not None
            ]
            add_candidates = [
                key
                for key in ordered[n : n + 20]
                if self._reduced_unimol_vector(key) is not None
            ]
            if not remove_candidates or not add_candidates:
                continue
            group_demand = self._predict_group_demand(masked_row)
            expected_attributes = self._expected_attribute_weights(
                masked_row,
                exclude_train_index=index,
            )
            query_examples = 0
            for remove_key in remove_candidates:
                for add_key in add_candidates:
                    beneficial = add_key in gold and remove_key not in gold
                    harmful = add_key not in gold and remove_key in gold
                    features.append(
                        self._swap_feature_vector(
                            masked_row,
                            item_by_key,
                            selected,
                            add_key,
                            remove_key,
                            group_demand,
                            expected_attributes,
                        )
                    )
                    labels.append(int(beneficial))
                    self.boundary_swap_neutral_count += int(
                        not beneficial and not harmful
                    )
                    query_examples += 1
            if query_examples:
                sample_weights.extend(
                    [1.0 / query_examples] * query_examples
                )
                training_queries += 1
        if not features or len(set(labels)) < 2:
            return
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.05,
                max_iter=2000,
                random_state=0,
            ),
        )
        model.fit(
            features,
            labels,
            logisticregression__sample_weight=sample_weights,
        )
        self.boundary_swap_ranker = model
        self.boundary_swap_training_pairs = len(features)
        self.boundary_swap_training_queries = training_queries
        self.boundary_swap_positive_count = sum(labels)
        self.boundary_swap_negative_count = len(labels) - sum(labels)

    def _boundary_training_items(
        self,
        row: dict[str, Any],
        exclude_train_index: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Build only the candidate fields required by swap adaptation.

        This is score-equivalent to H1 in ``rank`` but deliberately omits
        proposal construction, Reviewer diagnostics, and unrelated ledgers.
        """
        partial = {
            normalize(value)
            for value in row.get("partial_molecules") or []
            if normalize(value)
        }
        retrieved_support = self._build_retrieved_support(
            row,
            exclude_train_index=exclude_train_index,
            top_k=10,
        )
        idf_retrieved_support, idf_retrieved_counts = (
            self._build_idf_retrieved_support(
                row,
                exclude_train_index=exclude_train_index,
                top_k=15,
            )
        )
        expected_attributes = self._expected_attribute_weights(
            row,
            exclude_train_index=exclude_train_index,
        )
        query_context = self._build_query_context(row)
        group_demand = self._predict_group_demand(row)
        excluded_profile = (
            self.full_profiles[exclude_train_index]
            if exclude_train_index is not None
            else None
        )
        item_inputs: list[
            tuple[str, list[float], list[float]]
        ] = []
        for candidate in self.universe:
            if candidate in partial:
                continue
            values = self._query_features(
                row,
                candidate,
                retrieved_support,
                excluded_profile=excluded_profile,
                expected_attributes=expected_attributes,
                query_context=query_context,
            )
            primary_values = [
                values[index]
                for index in self.PRIMARY_FEATURE_INDICES
            ]
            item_inputs.append(
                (candidate, values, primary_values)
            )
        if self.ranker is not None and item_inputs:
            occurrence_scores = [
                float(value)
                for value in self.ranker.decision_function(
                    [item[2] for item in item_inputs]
                )
            ]
        else:
            occurrence_scores = [
                (
                    0.20 * values[0]
                    + 0.30 * values[1]
                    + 0.20 * values[2]
                    + 0.30 * values[3]
                )
                for _, values, _ in item_inputs
            ]
        items: list[dict[str, Any]] = []
        for (
            candidate,
            values,
            _primary_values,
        ), occurrence_score in zip(
            item_inputs,
            occurrence_scores,
        ):
            candidate_groups = self.functional_group_sets.get(
                candidate,
                set(),
            )
            group_probabilities = [
                float(group in candidate_groups)
                for group in self.functional_group_vocabulary
            ]
            group_score = (
                sum(
                    (2.0 * demand - 1.0) * probability
                    for demand, probability in zip(
                        group_demand,
                        group_probabilities,
                    )
                )
                / max(1, len(group_demand))
                if group_demand
                else 0.0
            )
            items.append(
                {
                    "molecule": self.display_names[candidate],
                    "occurrence_score": occurrence_score,
                    "frequency_prior": values[0],
                    "cooccurrence": values[1],
                    "cooccurrence_max": values[2],
                    "retrieved_profile_support": values[3],
                    "idf_retrieved_support": idf_retrieved_support.get(
                        candidate,
                        0.0,
                    ),
                    "idf_retrieved_profile_count": (
                        idf_retrieved_counts.get(candidate, 0)
                    ),
                    "functional_group_demand_score": group_score,
                    "perceptual_residual_support": values[11],
                    "unimol_query_compatibility": values[7],
                    "unimol_query_distance": values[8],
                    "unimol_mean_similarity": values[4],
                    "unimol_max_similarity": values[5],
                    "unimol_available": (
                        self._reduced_unimol_vector(candidate) is not None
                    ),
                    "functional_group_available": bool(
                        candidate_groups
                    ),
                    "_functional_group_probabilities": group_probabilities,
                }
            )
        items.sort(
            key=lambda item: (
                -float(item["occurrence_score"]),
                normalize(item["molecule"]),
            )
        )
        for rank, item in enumerate(items, 1):
            item["occurrence_rank"] = rank
        return items[: max(0, limit)]

    @staticmethod
    def _n_bucket(n: int) -> str:
        if n <= 5:
            return "small"
        if n <= 20:
            return "medium"
        return "large"

    @staticmethod
    def _set_f1(predicted: set[str], gold: set[str]) -> float:
        if not predicted and not gold:
            return 1.0
        if not predicted or not gold:
            return 0.0
        overlap = len(predicted & gold)
        precision = overlap / len(predicted)
        recall = overlap / len(gold)
        return (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    def _functional_group_set(
        self,
        molecules: list[str] | set[str],
    ) -> set[str]:
        groups: set[str] = set()
        for molecule in molecules:
            groups.update(
                self.functional_group_sets.get(
                    normalize(molecule),
                    set(),
                )
            )
        return groups

    @staticmethod
    def _soft_group_f1(
        selected: list[dict[str, Any]],
        demand: list[float],
    ) -> float:
        if not demand:
            return 0.0
        coverage = [0.0] * len(demand)
        for item in selected:
            probabilities = (
                item.get("_functional_group_probabilities") or []
            )
            for column in range(min(len(demand), len(probabilities))):
                probability = max(
                    0.0,
                    min(1.0, float(probabilities[column])),
                )
                coverage[column] = 1.0 - (
                    1.0 - coverage[column]
                ) * (1.0 - probability)
        expected_intersection = sum(
            probability * covered
            for probability, covered in zip(demand, coverage)
        )
        expected_predicted = sum(coverage)
        expected_gold = sum(demand)
        denominator = expected_predicted + expected_gold
        return (
            2.0 * expected_intersection / denominator
            if denominator
            else 0.0
        )

    def _apply_functional_budget(
        self,
        items: list[dict[str, Any]],
        n: int,
        budget: int,
        demand: list[float],
    ) -> list[str]:
        """Use deterministic functional-group coverage for boundary slots."""
        occurrence = sorted(
            items,
            key=lambda item: (
                int(item.get("occurrence_rank") or 10**9),
                normalize(item.get("molecule")),
            ),
        )
        baseline = occurrence[:n]
        if (
            n <= 0
            or budget <= 0
            or len(occurrence) <= n
            or not demand
        ):
            return [
                normalize(item.get("molecule"))
                for item in baseline
                if normalize(item.get("molecule"))
            ]
        budget = min(budget, n)
        protected_count = max(0, n - budget)
        selected = list(baseline[:protected_count])
        selected_keys = {
            normalize(item.get("molecule")) for item in selected
        }
        boundary_pool = [
            item
            for item in occurrence[
                protected_count : min(
                    len(occurrence),
                    n + max(25, 10 * budget),
                )
            ]
            if item.get("functional_group_available", False)
        ]
        for _ in range(budget):
            best_item = None
            best_score = -math.inf
            for item in boundary_pool:
                key = normalize(item.get("molecule"))
                if not key or key in selected_keys:
                    continue
                functional_gain = self._soft_group_f1(
                    selected + [item],
                    demand,
                )
                occurrence_guard = 1.0 - (
                    max(1, int(item.get("occurrence_rank") or 10**9))
                    / max(1, len(occurrence))
                )
                score = functional_gain + 0.02 * occurrence_guard
                if (
                    score > best_score
                    or (
                        abs(score - best_score) <= 1e-12
                        and (
                            int(item.get("occurrence_rank") or 10**9),
                            key,
                        )
                        < (
                            int(
                                (best_item or {}).get("occurrence_rank")
                                or 10**9
                            ),
                            normalize(
                                (best_item or {}).get("molecule")
                            ),
                        )
                    )
                ):
                    best_item = item
                    best_score = score
            if best_item is None:
                break
            selected.append(best_item)
            selected_keys.add(normalize(best_item.get("molecule")))
        for item in occurrence:
            key = normalize(item.get("molecule"))
            if key and key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
            if len(selected) >= n:
                break
        return [
            normalize(item.get("molecule"))
            for item in selected[:n]
            if normalize(item.get("molecule"))
        ]

    def _functional_proxy_score(
        self,
        items: list[dict[str, Any]],
        selected_molecules: list[str] | set[str],
        demand: list[float],
    ) -> float:
        selected_keys = {
            normalize(value) for value in selected_molecules
        }
        selected_items = [
            item
            for item in items
            if normalize(item.get("molecule")) in selected_keys
        ]
        return self._soft_group_f1(selected_items, demand)

    @staticmethod
    def _residual_order(
        items: list[dict[str, Any]],
        channel: str,
    ) -> list[dict[str, Any]]:
        if channel == "retrieval":
            return sorted(
                items,
                key=lambda item: (
                    -float(item.get("retrieved_profile_support") or 0.0),
                    int(item.get("occurrence_rank") or 10**9),
                    normalize(item.get("molecule")),
                ),
            )
        return sorted(
            items,
            key=lambda item: (
                (
                    int(
                        item.get("unimol_set_compatibility_rank")
                        or 10**9
                    )
                    - int(item.get("occurrence_rank") or 10**9)
                ),
                -float(
                    item.get("unimol_set_compatibility_score")
                    or 0.0
                ),
                int(item.get("occurrence_rank") or 10**9),
                normalize(item.get("molecule")),
            ),
        )

    def _apply_residual_budget(
        self,
        items: list[dict[str, Any]],
        n: int,
        channel: str,
        budget: int,
    ) -> list[str]:
        """Make bounded boundary exchanges while preserving the occurrence core."""
        occurrence = sorted(
            items,
            key=lambda item: (
                int(item.get("occurrence_rank") or 10**9),
                normalize(item.get("molecule")),
            ),
        )
        baseline = occurrence[:n]
        if n <= 0 or budget <= 0 or len(occurrence) <= n:
            return [
                normalize(item.get("molecule"))
                for item in baseline
                if normalize(item.get("molecule"))
            ]
        budget = min(budget, n)
        protected_count = max(0, n - budget)
        protected = baseline[:protected_count]
        boundary_pool = occurrence[
            protected_count : min(len(occurrence), n + max(10, 5 * budget))
        ]
        if channel == "retrieval":
            supported = [
                item
                for item in boundary_pool
                if float(item.get("retrieved_profile_support") or 0.0) > 0.0
            ]
        else:
            supported = [
                item
                for item in boundary_pool
                if (
                    item.get("unimol_available", False)
                    or float(item.get("perceptual_score") or 0.0) > 0.0
                )
            ]
        residual = self._residual_order(supported, channel)[:budget]
        selected = protected + residual
        selected_keys = {
            normalize(item.get("molecule"))
            for item in selected
        }
        for item in occurrence:
            key = normalize(item.get("molecule"))
            if key and key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
            if len(selected) >= n:
                break
        return [
            normalize(item.get("molecule"))
            for item in selected[:n]
            if normalize(item.get("molecule"))
        ]

    def _apply_contextual_structural_budget(
        self,
        row: dict[str, Any],
        items: list[dict[str, Any]],
        n: int,
        budget: int,
    ) -> list[str]:
        """Apply only positive, context-conditioned UniMol boundary swaps."""
        occurrence = sorted(
            items,
            key=lambda item: (
                int(item.get("occurrence_rank") or 10**9),
                normalize(item.get("molecule")),
            ),
        )
        baseline = [
            normalize(item.get("molecule"))
            for item in occurrence[:n]
            if normalize(item.get("molecule"))
        ]
        if (
            n <= 0
            or budget <= 0
            or len(occurrence) <= n
            or self.boundary_swap_ranker is None
        ):
            return baseline
        item_by_key = {
            normalize(item.get("molecule")): item
            for item in occurrence
            if normalize(item.get("molecule"))
        }
        selected = list(baseline)
        additions = [
            normalize(item.get("molecule"))
            for item in occurrence[n : n + max(25, 10 * budget)]
            if self._reduced_unimol_vector(item.get("molecule")) is not None
        ]
        group_demand = self._predict_group_demand(row)
        expected_attributes = self._expected_attribute_weights(row)
        for _ in range(min(budget, n)):
            removals = [
                key
                for key in selected[max(0, len(selected) - 10) :]
                if self._reduced_unimol_vector(key) is not None
            ]
            best: tuple[float, str, str] | None = None
            for remove_key in removals:
                for add_key in additions:
                    if add_key in selected:
                        continue
                    feature_vector = self._swap_feature_vector(
                        row,
                        item_by_key,
                        selected,
                        add_key,
                        remove_key,
                        group_demand,
                        expected_attributes,
                    )
                    score = float(
                        self.boundary_swap_ranker.decision_function(
                            [feature_vector]
                        )[0]
                    )
                    candidate = (score, remove_key, add_key)
                    if best is None or (
                        score,
                        -int(
                            item_by_key[add_key].get("occurrence_rank")
                            or 10**9
                        ),
                        add_key,
                    ) > (
                        best[0],
                        -int(
                            item_by_key[best[2]].get("occurrence_rank")
                            or 10**9
                        ),
                        best[2],
                    ):
                        best = candidate
            if best is None or best[0] <= 0.0:
                break
            _, remove_key, add_key = best
            selected[selected.index(remove_key)] = add_key
        return sorted(
            selected,
            key=lambda key: (
                int(item_by_key[key].get("occurrence_rank") or 10**9),
                key,
            ),
        )

    @staticmethod
    def _profile_signature(row: dict[str, Any]) -> str:
        values = {
            normalize(value)
            for field in ("partial_molecules", "missing_molecules")
            for value in row.get(field) or []
            if normalize(value)
        }
        return "\x1f".join(sorted(values))

    def _predict_group_cardinality_posterior(
        self,
        row: dict[str, Any],
    ) -> dict[str, dict[int, float]]:
        """Estimate q(group, gold-group-cardinality | query) from train only."""
        joint: dict[str, Counter[int]] = defaultdict(Counter)
        cardinality_mass: Counter[int] = Counter()
        denominator = 0.0
        retrieved = self._retrieved_profiles(row, top_k=20)
        for rank, (score, index) in enumerate(retrieved, 1):
            weight = max(0.0, float(score)) / math.log2(rank + 1.0)
            if weight <= 0.0:
                continue
            gold_groups = self._functional_group_set(
                {
                    normalize(value)
                    for value in self.rows[index].get("missing_molecules") or []
                    if normalize(value)
                }
            )
            cardinality = len(gold_groups)
            if cardinality <= 0:
                continue
            denominator += weight
            cardinality_mass[cardinality] += weight
            for group in gold_groups:
                joint[group][cardinality] += weight

        # A small empirical-prior component prevents a single nearest profile
        # from creating a degenerate posterior.  It contains no test labels.
        prior_weight = 0.15
        for train_row in self.rows:
            gold_groups = self._functional_group_set(
                {
                    normalize(value)
                    for value in train_row.get("missing_molecules") or []
                    if normalize(value)
                }
            )
            cardinality = len(gold_groups)
            if cardinality <= 0:
                continue
            weight = prior_weight / max(1, len(self.rows))
            denominator += weight
            cardinality_mass[cardinality] += weight
            for group in gold_groups:
                joint[group][cardinality] += weight
        if denominator <= 0.0:
            return {}
        return {
            group: {
                cardinality: mass / denominator
                for cardinality, mass in counts.items()
            }
            for group, counts in joint.items()
        }

    @staticmethod
    def _expected_group_f1(
        predicted_groups: set[str],
        posterior: dict[str, dict[int, float]],
    ) -> float:
        if not predicted_groups or not posterior:
            return 0.0
        predicted_size = len(predicted_groups)
        return sum(
            2.0 * probability / (predicted_size + gold_size)
            for group in predicted_groups
            for gold_size, probability in posterior.get(group, {}).items()
            if gold_size > 0
        )

    def _build_v14_action_bank(
        self,
        items: list[dict[str, Any]],
        n: int,
        posterior: dict[str, dict[int, float]],
        bank_size: int = 20,
        scientist_top_k: int = 5,
    ) -> dict[str, Any]:
        ordered = sorted(
            items,
            key=lambda item: (
                int(item.get("occurrence_rank") or 10**9),
                normalize(item.get("molecule")),
            ),
        )
        h1 = [
            normalize(item.get("molecule"))
            for item in ordered[:n]
            if normalize(item.get("molecule"))
        ]
        baseline_groups = self._functional_group_set(set(h1))
        baseline_expected = self._expected_group_f1(
            baseline_groups, posterior
        )
        if n <= 0 or len(ordered) <= n:
            return {
                "h1": h1,
                "h1_groups": sorted(baseline_groups),
                "h1_expected_f1": baseline_expected,
                "actions": [],
                "scientist_actions": [],
            }
        removals = h1[max(0, n - min(12, n)) :]
        additions = stable_unique(
            [
                normalize(item.get("molecule"))
                for item in ordered[n : min(len(ordered), n + 40)]
                if normalize(item.get("molecule"))
            ]
        )
        actions: list[dict[str, Any]] = []
        seen_group_sets: set[tuple[str, ...]] = set()
        for remove_key in removals:
            for add_key in additions:
                proposal = list(h1)
                proposal[proposal.index(remove_key)] = add_key
                proposal_groups = self._functional_group_set(set(proposal))
                signature = tuple(sorted(proposal_groups))
                if signature in seen_group_sets:
                    continue
                seen_group_sets.add(signature)
                expected = self._expected_group_f1(
                    proposal_groups, posterior
                )
                actions.append(
                    {
                        "remove_key": remove_key,
                        "add_key": add_key,
                        "predicted_expected_f1": expected,
                        "predicted_expected_f1_gain": (
                            expected - baseline_expected
                        ),
                        "removed_groups": sorted(
                            baseline_groups - proposal_groups
                        ),
                        "added_groups": sorted(
                            proposal_groups - baseline_groups
                        ),
                        "proposal_groups": sorted(proposal_groups),
                        "proposal": proposal,
                    }
                )
        actions.sort(
            key=lambda action: (
                -float(action["predicted_expected_f1_gain"]),
                action["remove_key"],
                action["add_key"],
            )
        )
        actions = actions[:bank_size]
        return {
            "h1": h1,
            "h1_groups": sorted(baseline_groups),
            "h1_expected_f1": baseline_expected,
            "actions": actions,
            "scientist_actions": actions[:scientist_top_k],
        }

    @staticmethod
    def _v16_group_jaccard_distance(
        left: set[str],
        right: set[str],
    ) -> float:
        union = left | right
        if not union:
            return 0.0
        return 1.0 - len(left & right) / len(union)

    def _v16_select_quality_diverse(
        self,
        actions: list[dict[str, Any]],
        limit: int,
        quality_quota: int,
        diversity_weight: float,
    ) -> list[dict[str, Any]]:
        """Select a fixed best-of-K slate without using query labels.

        Expected F1 remains the quality prior.  The remaining budget rewards
        marginally different added/removed functional-group ledgers so that a
        nominal K does not collapse into near-identical hypotheses.
        """
        if limit <= 0 or not actions:
            return []
        ordered = sorted(
            actions,
            key=lambda action: (
                -float(action["predicted_expected_f1_gain"]),
                int(action.get("depth") or 1),
                tuple(action.get("remove_keys") or []),
                tuple(action.get("add_keys") or []),
            ),
        )
        selected = ordered[: min(limit, max(0, quality_quota))]
        selected_ids = {id(action) for action in selected}
        remaining = [
            action for action in ordered if id(action) not in selected_ids
        ]
        denominator = max(1, len(ordered) - 1)
        rank_by_id = {
            id(action): rank for rank, action in enumerate(ordered)
        }
        while remaining and len(selected) < limit:
            best_action: dict[str, Any] | None = None
            best_value: tuple[float, float, float, tuple[str, ...]] | None = None
            for action in remaining:
                quality = 1.0 - rank_by_id[id(action)] / denominator
                action_groups = set(action.get("proposal_groups") or [])
                diversity = min(
                    self._v16_group_jaccard_distance(
                        action_groups,
                        set(chosen.get("proposal_groups") or []),
                    )
                    for chosen in selected
                ) if selected else 1.0
                ledger_novelty = min(
                    self._v16_group_jaccard_distance(
                        set(action.get("added_groups") or [])
                        | set(action.get("removed_groups") or []),
                        set(chosen.get("added_groups") or [])
                        | set(chosen.get("removed_groups") or []),
                    )
                    for chosen in selected
                ) if selected else 1.0
                value = (
                    quality
                    + diversity_weight
                    * (0.7 * diversity + 0.3 * ledger_novelty),
                    quality,
                    diversity,
                    tuple(action.get("proposal_groups") or []),
                )
                if best_value is None or value > best_value:
                    best_value = value
                    best_action = action
            if best_action is None:
                break
            selected.append(best_action)
            remaining.remove(best_action)
        return selected

    def _build_v16_scientist_bank(
        self,
        items: list[dict[str, Any]],
        n: int,
        posterior: dict[str, dict[int, float]],
        bank_size: int = 20,
        scientist_top_k: int = 5,
        addition_pool_size: int = 100,
        first_step_beam_size: int = 8,
    ) -> dict[str, Any]:
        """Build a label-free, depth-two, quality-diverse MPC slate.

        This method only expands Scientist proposals.  It never reads Gold,
        the formal functional-group cache, UniMol, or Reviewer verdicts, and
        it has no prediction authority until a separate OOF admission passes.

        The frozen 2026-08-01 OOF admission did not pass every preregistered
        gate (82.09% versus 85% expanded-oracle capture; 276 versus 280
        positive-query coverage).  Keep this implementation experimental and
        do not wire it into ``predict`` without a new, separately approved
        Scientist design and admission run.
        """
        ordered = sorted(
            items,
            key=lambda item: (
                int(item.get("occurrence_rank") or 10**9),
                normalize(item.get("molecule")),
            ),
        )
        h1 = stable_unique(
            normalize(item.get("molecule"))
            for item in ordered[:n]
            if normalize(item.get("molecule"))
        )
        baseline_groups = self._functional_group_set(set(h1))
        baseline_expected = self._expected_group_f1(
            baseline_groups, posterior
        )
        empty = {
            "h1": h1,
            "h1_groups": sorted(baseline_groups),
            "h1_expected_f1": baseline_expected,
            "actions": [],
            "scientist_actions": [],
            "one_step_candidate_count": 0,
            "two_step_candidate_count": 0,
            "addition_pool_size": 0,
        }
        if n <= 0 or len(h1) != n or len(ordered) <= n:
            return empty

        h1_set = set(h1)
        outside = [
            item
            for item in ordered
            if normalize(item.get("molecule")) not in h1_set
        ]

        def ranked_keys(field: str) -> list[str]:
            ranked = sorted(
                outside,
                key=lambda item: (
                    -float(item.get(field) or 0.0),
                    int(item.get("occurrence_rank") or 10**9),
                    normalize(item.get("molecule")),
                ),
            )
            return stable_unique(
                normalize(item.get("molecule")) for item in ranked
            )

        occurrence = stable_unique(
            normalize(item.get("molecule")) for item in outside
        )
        source_lists = [
            occurrence,
            ranked_keys("idf_retrieved_support"),
            ranked_keys("retrieved_profile_support"),
            ranked_keys("functional_group_demand_score"),
        ]
        # Preserve the strongest occurrence backbone while reserving a fixed
        # quarter of the pool for three independent training-side sources.
        additions = occurrence[: min(70, addition_pool_size)]
        for source in source_lists[1:]:
            for key in source[:10]:
                if key not in additions:
                    additions.append(key)
                if len(additions) >= addition_pool_size:
                    break
            if len(additions) >= addition_pool_size:
                break
        for key in occurrence:
            if key not in additions:
                additions.append(key)
            if len(additions) >= addition_pool_size:
                break
        additions = additions[:addition_pool_size]

        molecule_groups = {
            key: set(self.functional_group_sets.get(key, set()))
            for key in set(h1) | set(additions)
        }

        def group_counts(proposal: list[str]) -> Counter[str]:
            counts: Counter[str] = Counter()
            for key in proposal:
                counts.update(molecule_groups.get(key, set()))
            return counts

        def exchanged_group_signature(
            counts: Counter[str],
            remove_key: str,
            add_key: str,
        ) -> tuple[str, ...]:
            updated = counts.copy()
            updated.subtract(molecule_groups.get(remove_key, set()))
            updated.update(molecule_groups.get(add_key, set()))
            return tuple(
                sorted(group for group, count in updated.items() if count > 0)
            )

        def make_action(
            proposal: list[str],
            path: list[dict[str, str]],
            group_signature: tuple[str, ...],
        ) -> dict[str, Any]:
            proposal_groups = set(group_signature)
            expected = self._expected_group_f1(
                proposal_groups, posterior
            )
            return {
                "depth": len(path),
                "path": path,
                "remove_keys": [step["remove_key"] for step in path],
                "add_keys": [step["add_key"] for step in path],
                "remove_key": path[-1]["remove_key"],
                "add_key": path[-1]["add_key"],
                "predicted_expected_f1": expected,
                "predicted_expected_f1_gain": expected - baseline_expected,
                "removed_groups": sorted(
                    baseline_groups - proposal_groups
                ),
                "added_groups": sorted(
                    proposal_groups - baseline_groups
                ),
                "proposal_groups": sorted(proposal_groups),
                "proposal": proposal,
            }

        one_step_by_signature: dict[
            tuple[str, ...], dict[str, Any]
        ] = {}
        baseline_group_counts = group_counts(h1)
        for remove_key in h1:
            for add_key in additions:
                signature = exchanged_group_signature(
                    baseline_group_counts,
                    remove_key,
                    add_key,
                )
                previous = one_step_by_signature.get(signature)
                path_key = (remove_key, add_key)
                previous_key = (
                    previous["remove_keys"][0],
                    previous["add_keys"][0],
                ) if previous is not None else None
                if previous is None or path_key < previous_key:
                    proposal = list(h1)
                    proposal[proposal.index(remove_key)] = add_key
                    action = make_action(
                        proposal,
                        [{"remove_key": remove_key, "add_key": add_key}],
                        signature,
                    )
                    one_step_by_signature[signature] = action
        one_step = list(one_step_by_signature.values())
        first_beam = self._v16_select_quality_diverse(
            one_step,
            first_step_beam_size,
            quality_quota=min(4, first_step_beam_size),
            diversity_weight=0.25,
        )

        two_step_by_signature: dict[
            tuple[str, ...], dict[str, Any]
        ] = {}
        for first in first_beam:
            current = list(first["proposal"])
            current_group_counts = group_counts(current)
            first_remove = first["remove_keys"][0]
            first_add = first["add_keys"][0]
            for remove_key in current:
                # Removing the molecule just added creates a disguised
                # one-step path and wastes the depth-two budget.
                if remove_key == first_add:
                    continue
                for add_key in additions:
                    if add_key in current or add_key == first_remove:
                        continue
                    signature = exchanged_group_signature(
                        current_group_counts,
                        remove_key,
                        add_key,
                    )
                    previous = two_step_by_signature.get(signature)
                    path = list(first["path"]) + [
                        {"remove_key": remove_key, "add_key": add_key}
                    ]
                    path_key = (
                        tuple(step["remove_key"] for step in path),
                        tuple(step["add_key"] for step in path),
                    )
                    previous_key = (
                        tuple(previous["remove_keys"]),
                        tuple(previous["add_keys"]),
                    ) if previous is not None else None
                    if previous is None or path_key < previous_key:
                        proposal = list(current)
                        proposal[proposal.index(remove_key)] = add_key
                        action = make_action(
                            proposal,
                            path,
                            signature,
                        )
                        two_step_by_signature[signature] = action

        combined_by_signature = dict(one_step_by_signature)
        for signature, action in two_step_by_signature.items():
            previous = combined_by_signature.get(signature)
            if previous is None or float(
                action["predicted_expected_f1_gain"]
            ) > float(previous["predicted_expected_f1_gain"]):
                combined_by_signature[signature] = action
        bank = self._v16_select_quality_diverse(
            list(combined_by_signature.values()),
            bank_size,
            quality_quota=min(10, bank_size),
            diversity_weight=0.20,
        )
        slate = self._v16_select_quality_diverse(
            bank,
            scientist_top_k,
            quality_quota=min(2, scientist_top_k),
            diversity_weight=0.35,
        )
        return {
            "h1": h1,
            "h1_groups": sorted(baseline_groups),
            "h1_expected_f1": baseline_expected,
            "actions": bank,
            "scientist_actions": slate,
            "one_step_candidate_count": len(one_step_by_signature),
            "two_step_candidate_count": len(two_step_by_signature),
            "addition_pool_size": len(additions),
        }

    @staticmethod
    def _bootstrap_lower_bound(values: list[float], seed: str) -> float:
        if not values:
            return 0.0
        rng = random.Random(seed)
        means = sorted(
            sum(values[rng.randrange(len(values))] for _ in values)
            / len(values)
            for _ in range(1000)
        )
        return means[max(0, int(0.05 * len(means)) - 1)]

    def _calibrate_v14_action_policy(self) -> None:
        """Cross-fit the generic expected-F1 one-swap executor.

        Exact full-profile clusters are kept in one outer fold.  Decision
        thresholds are learned from the other outer folds, so neither action
        scoring nor risk gating sees the held fold's labels.
        """
        clusters: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            clusters[self._profile_signature(row)].append(index)
        fold_count = min(5, len(clusters))
        if fold_count < 2:
            return
        fold_sizes = [0] * fold_count
        fold_by_index: dict[int, int] = {}
        for signature, indices in sorted(
            clusters.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            fold = min(range(fold_count), key=lambda value: fold_sizes[value])
            fold_sizes[fold] += len(indices)
            for index in indices:
                fold_by_index[index] = fold

        records: list[dict[str, Any]] = []
        for fold in range(fold_count):
            fit_rows = [
                row for index, row in enumerate(self.rows)
                if fold_by_index[index] != fold
            ]
            held_rows = [
                row for index, row in enumerate(self.rows)
                if fold_by_index[index] == fold
            ]
            fold_model = MPCStructureModel(
                fit_rows,
                None,
                self.ablation,
                self.db_path,
                calibrate_residuals=False,
            )
            for source_row in held_rows:
                gold_molecules = {
                    normalize(value)
                    for value in source_row.get("missing_molecules") or []
                    if normalize(value)
                }
                n = len(gold_molecules)
                if n <= 0:
                    continue
                masked_row = {
                    "id": f"{source_row.get('id')}:v14-oof",
                    "target_food": source_row.get("target_food"),
                    "partial_molecules": list(
                        source_row.get("partial_molecules") or []
                    ),
                    "n": n,
                }
                items = fold_model._boundary_training_items(
                    masked_row, exclude_train_index=None, limit=n + 40
                )
                posterior = fold_model._predict_group_cardinality_posterior(
                    masked_row
                )
                bank = fold_model._build_v14_action_bank(
                    items, n, posterior
                )
                gold_groups = fold_model._functional_group_set(gold_molecules)
                baseline = self._set_f1(
                    set(bank["h1_groups"]), gold_groups
                )
                evaluated_actions: list[dict[str, Any]] = []
                for action in bank["actions"]:
                    actual_gain = self._set_f1(
                        set(action["proposal_groups"]), gold_groups
                    ) - baseline
                    evaluated_actions.append(
                        {**action, "actual_gain": actual_gain}
                    )
                records.append(
                    {
                        "fold": fold,
                        "n": n,
                        "baseline": baseline,
                        "actions": evaluated_actions,
                        "bank_oracle_gain": max(
                            [0.0]
                            + [a["actual_gain"] for a in evaluated_actions]
                        ),
                        "scientist_oracle_gain": max(
                            [0.0]
                            + [
                                a["actual_gain"]
                                for a in evaluated_actions[:5]
                            ]
                        ),
                    }
                )

        def choose_threshold(training_records: list[dict[str, Any]]) -> float:
            candidates = sorted(
                {
                    max(0.0, float(record["actions"][0]["predicted_expected_f1_gain"]))
                    for record in training_records
                    if record["actions"]
                }
            )
            if not candidates:
                return math.inf
            best: tuple[float, float] | None = None
            for threshold in candidates:
                gains = [
                    float(record["actions"][0]["actual_gain"])
                    if record["actions"]
                    and float(record["actions"][0]["predicted_expected_f1_gain"])
                    >= threshold
                    else 0.0
                    for record in training_records
                ]
                changed = sum(abs(gain) > 1e-12 for gain in gains)
                mean_gain = sum(gains) / max(1, len(gains))
                if changed >= 5 and mean_gain > 0.0:
                    candidate = (mean_gain, threshold)
                    if best is None or candidate > best:
                        best = candidate
            return best[1] if best is not None else math.inf

        crossfit_gains: list[float] = []
        fold_gains: dict[str, float] = {}
        fold_thresholds: dict[str, float | None] = {}
        for fold in range(fold_count):
            threshold = choose_threshold(
                [record for record in records if record["fold"] != fold]
            )
            held = [record for record in records if record["fold"] == fold]
            held_gains = [
                float(record["actions"][0]["actual_gain"])
                if record["actions"]
                and float(record["actions"][0]["predicted_expected_f1_gain"])
                >= threshold
                else 0.0
                for record in held
            ]
            crossfit_gains.extend(held_gains)
            fold_gains[str(fold)] = sum(held_gains) / max(1, len(held_gains))
            fold_thresholds[str(fold)] = (
                None if math.isinf(threshold) else round(threshold, 10)
            )

        bank_oracle = sum(r["bank_oracle_gain"] for r in records) / max(1, len(records))
        scientist_oracle = sum(r["scientist_oracle_gain"] for r in records) / max(1, len(records))
        final_gain = sum(crossfit_gains) / max(1, len(crossfit_gains))
        lower_bound = self._bootstrap_lower_bound(
            crossfit_gains, "mpc-v14-crossfit-bootstrap"
        )
        wins = sum(gain > 1e-12 for gain in crossfit_gains)
        losses = sum(gain < -1e-12 for gain in crossfit_gains)
        changed = wins + losses
        capture = scientist_oracle / bank_oracle if bank_oracle > 0.0 else 0.0
        admitted = bool(
            len(records) >= 25
            and bank_oracle > 0.0
            and scientist_oracle >= 0.5 * bank_oracle
            and changed >= 5
            and final_gain > 0.0
            and lower_bound > 0.0
            and wins >= 2 * max(1, losses)
            and all(gain >= -0.002 for gain in fold_gains.values())
            and capture > 0.2297
        )
        deployment_threshold = choose_threshold(records) if admitted else math.inf
        self.metric_group_policy = {
            "enabled": admitted,
            "budget": 1 if admitted else 0,
            "minimum_expected_f1_gain": (
                None if math.isinf(deployment_threshold)
                else deployment_threshold
            ),
            "scientist_top_k": 5,
            "action_bank_size": 20,
        }
        self.metric_group_calibration = {
            "protocol": "exact_profile_clustered_five_fold_crossfit_risk_gate",
            "selection_metric": "macro_functional_group_f1",
            "method_reads_llm_evaluation_cache": False,
            "uses_unimol": False,
            "query_count": len(records),
            "profile_cluster_count": len(clusters),
            "fold_sizes": fold_sizes,
            "bank_oracle_gain": round(bank_oracle, 8),
            "scientist_oracle_at_5_gain": round(scientist_oracle, 8),
            "scientist_oracle_capture_ratio": round(capture, 8),
            "crossfit_final_gain": round(final_gain, 8),
            "paired_bootstrap_95pct_lower_bound": round(lower_bound, 8),
            "wins": wins,
            "losses": losses,
            "changed_queries": changed,
            "fold_gains": {key: round(value, 8) for key, value in fold_gains.items()},
            "fold_thresholds": fold_thresholds,
            "deployment_threshold": (
                None if math.isinf(deployment_threshold)
                else round(deployment_threshold, 10)
            ),
            "admitted": admitted,
            "admission_rule": (
                "positive_bank_oracle; scientist_oracle_at_5_at_least_half_bank; "
                "positive_bootstrap_lower_bound; wins_at_least_twice_losses; "
                "no_fold_below_minus_0.002; capture_above_v13_22.97pct"
            ),
        }
        self.residual_calibration = {
            "protocol": "v14_phase1_metric_aligned_only",
            "query_count": len(records),
            "functional_group_decoder": self.metric_group_calibration,
        }

    def _v15_action_features(
        self,
        action: dict[str, Any],
        h1: list[str],
        items: list[dict[str, Any]],
        posterior: dict[str, dict[int, float]],
        n: int,
    ) -> tuple[list[float], list[float]]:
        item_by_key = {
            normalize(item.get("molecule")): item for item in items
        }
        add_item = item_by_key.get(action["add_key"], {})
        remove_item = item_by_key.get(action["remove_key"], {})
        added_groups = set(action.get("added_groups") or [])
        removed_groups = set(action.get("removed_groups") or [])
        add_all_groups = self.functional_group_sets.get(
            action["add_key"], set()
        )
        remove_all_groups = self.functional_group_sets.get(
            action["remove_key"], set()
        )

        def marginal(group: str) -> float:
            return sum(posterior.get(group, {}).values())

        added_marginals = [marginal(group) for group in added_groups]
        removed_marginals = [marginal(group) for group in removed_groups]
        cutoff_score = float(
            item_by_key.get(h1[-1], {}).get("occurrence_score") or 0.0
        ) if h1 else 0.0
        add_rank = int(add_item.get("occurrence_rank") or 10**9)
        remove_rank = int(remove_item.get("occurrence_rank") or 10**9)
        add_features = [
            float(action.get("predicted_expected_f1_gain") or 0.0),
            len(added_groups) / 10.0,
            sum(added_marginals) / max(1, len(added_marginals)),
            max(added_marginals, default=0.0),
            len(add_all_groups) / 20.0,
            float(add_item.get("retrieved_profile_support") or 0.0),
            float(add_item.get("idf_retrieved_support") or 0.0),
            1.0 / max(1, add_rank),
            float(add_item.get("occurrence_score") or 0.0) - cutoff_score,
            math.log1p(max(0, n)) / math.log1p(250),
        ]
        lost_fraction = len(removed_groups) / max(1, len(remove_all_groups))
        remove_features = [
            float(action.get("predicted_expected_f1_gain") or 0.0),
            len(removed_groups) / 10.0,
            sum(removed_marginals) / max(1, len(removed_marginals)),
            max(removed_marginals, default=0.0),
            lost_fraction,
            1.0 - lost_fraction,
            float(remove_item.get("retrieved_profile_support") or 0.0),
            float(remove_item.get("idf_retrieved_support") or 0.0),
            1.0 / max(1, remove_rank),
            float(remove_item.get("occurrence_score") or 0.0) - cutoff_score,
            math.log1p(max(0, n)) / math.log1p(250),
        ]
        return add_features, remove_features

    @staticmethod
    def _fit_v15_binary_verifier(
        features: list[list[float]],
        labels: list[int],
    ) -> Any | None:
        if not features or len(set(labels)) < 2:
            return None
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise OptimizedAgentError(
                "scikit-learn is required for the v15 dual-gate verifier"
            ) from exc
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            ),
        )
        model.fit(features, labels)
        return model

    @staticmethod
    def _choose_v15_action(
        actions: list[dict[str, Any]],
        add_threshold: float,
        remove_threshold: float,
    ) -> dict[str, Any] | None:
        admitted = [
            action
            for action in actions
            if float(action.get("add_necessity_probability") or 0.0)
            >= add_threshold
            and float(action.get("remove_safety_probability") or 0.0)
            >= remove_threshold
        ]
        return max(
            admitted,
            key=lambda action: (
                min(
                    float(action["add_necessity_probability"]),
                    float(action["remove_safety_probability"]),
                ),
                float(action.get("predicted_expected_f1_gain") or 0.0),
                action["remove_key"],
                action["add_key"],
            ),
            default=None,
        )

    def _calibrate_v15_dual_gate_policy(self) -> None:
        """Stacked OOF calibration for add-necessity/remove-safety gates."""
        clusters: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            clusters[self._profile_signature(row)].append(index)
        fold_count = min(5, len(clusters))
        if fold_count < 2:
            return
        fold_sizes = [0] * fold_count
        fold_by_index: dict[int, int] = {}
        for signature, indices in sorted(
            clusters.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            fold = min(
                range(fold_count), key=lambda value: fold_sizes[value]
            )
            fold_sizes[fold] += len(indices)
            for index in indices:
                fold_by_index[index] = fold

        records: list[dict[str, Any]] = []
        for fold in range(fold_count):
            fit_rows = [
                row for index, row in enumerate(self.rows)
                if fold_by_index[index] != fold
            ]
            held_rows = [
                row for index, row in enumerate(self.rows)
                if fold_by_index[index] == fold
            ]
            fold_model = MPCStructureModel(
                fit_rows,
                None,
                self.ablation,
                self.db_path,
                calibrate_residuals=False,
            )
            for source_row in held_rows:
                gold_molecules = {
                    normalize(value)
                    for value in source_row.get("missing_molecules") or []
                    if normalize(value)
                }
                n = len(gold_molecules)
                if n <= 0:
                    continue
                masked_row = {
                    "id": f"{source_row.get('id')}:v15-oof",
                    "target_food": source_row.get("target_food"),
                    "partial_molecules": list(
                        source_row.get("partial_molecules") or []
                    ),
                    "n": n,
                }
                items = fold_model._boundary_training_items(
                    masked_row, exclude_train_index=None, limit=n + 40
                )
                posterior = fold_model._predict_group_cardinality_posterior(
                    masked_row
                )
                bank = fold_model._build_v14_action_bank(
                    items, n, posterior
                )
                gold_groups = fold_model._functional_group_set(
                    gold_molecules
                )
                baseline_f1 = self._set_f1(
                    set(bank["h1_groups"]), gold_groups
                )
                actions: list[dict[str, Any]] = []
                for action in bank["scientist_actions"]:
                    add_features, remove_features = (
                        fold_model._v15_action_features(
                            action,
                            bank["h1"],
                            items,
                            posterior,
                            n,
                        )
                    )
                    added_groups = set(action["added_groups"])
                    removed_groups = set(action["removed_groups"])
                    actual_gain = self._set_f1(
                        set(action["proposal_groups"]), gold_groups
                    ) - baseline_f1
                    actions.append(
                        {
                            **action,
                            "add_features": add_features,
                            "remove_features": remove_features,
                            "add_label": int(bool(added_groups & gold_groups)),
                            "remove_label": int(
                                not bool(removed_groups & gold_groups)
                            ),
                            "actual_gain": actual_gain,
                        }
                    )
                records.append(
                    {"fold": fold, "n": n, "actions": actions}
                )

        for held_fold in range(fold_count):
            train_actions = [
                action
                for record in records
                if record["fold"] != held_fold
                for action in record["actions"]
            ]
            add_model = self._fit_v15_binary_verifier(
                [action["add_features"] for action in train_actions],
                [action["add_label"] for action in train_actions],
            )
            remove_model = self._fit_v15_binary_verifier(
                [action["remove_features"] for action in train_actions],
                [action["remove_label"] for action in train_actions],
            )
            if add_model is None or remove_model is None:
                continue
            held_actions = [
                action
                for record in records
                if record["fold"] == held_fold
                for action in record["actions"]
            ]
            add_probabilities = add_model.predict_proba(
                [action["add_features"] for action in held_actions]
            )[:, 1]
            remove_probabilities = remove_model.predict_proba(
                [action["remove_features"] for action in held_actions]
            )[:, 1]
            for action, add_probability, remove_probability in zip(
                held_actions, add_probabilities, remove_probabilities
            ):
                action["add_necessity_probability"] = float(add_probability)
                action["remove_safety_probability"] = float(
                    remove_probability
                )

        threshold_grid = (0.50, 0.60, 0.70, 0.80, 0.90)

        def choose_thresholds(
            training_records: list[dict[str, Any]],
        ) -> tuple[float, float] | None:
            best: tuple[float, int, float, float] | None = None
            for add_threshold in threshold_grid:
                for remove_threshold in threshold_grid:
                    gains: list[float] = []
                    for record in training_records:
                        selected = self._choose_v15_action(
                            record["actions"],
                            add_threshold,
                            remove_threshold,
                        )
                        gains.append(
                            float(selected["actual_gain"])
                            if selected is not None else 0.0
                        )
                    changed = sum(abs(gain) > 1e-12 for gain in gains)
                    mean_gain = sum(gains) / max(1, len(gains))
                    losses = sum(gain < -1e-12 for gain in gains)
                    if changed < 5 or mean_gain <= 0.0:
                        continue
                    candidate = (
                        mean_gain,
                        -losses,
                        add_threshold,
                        remove_threshold,
                    )
                    if best is None or candidate > best:
                        best = candidate
            return (best[2], best[3]) if best is not None else None

        # Reconstruct the v14 scalar gate on the same action records so the
        # paired comparison is exact and uses identical folds/candidates.
        def choose_v14_threshold(
            training_records: list[dict[str, Any]],
        ) -> float:
            candidates = sorted(
                {
                    max(
                        0.0,
                        float(
                            record["actions"][0][
                                "predicted_expected_f1_gain"
                            ]
                        ),
                    )
                    for record in training_records
                    if record["actions"]
                }
            )
            best: tuple[float, float] | None = None
            for threshold in candidates:
                gains = [
                    float(record["actions"][0]["actual_gain"])
                    if record["actions"]
                    and float(
                        record["actions"][0][
                            "predicted_expected_f1_gain"
                        ]
                    )
                    >= threshold
                    else 0.0
                    for record in training_records
                ]
                changed = sum(abs(gain) > 1e-12 for gain in gains)
                mean_gain = sum(gains) / max(1, len(gains))
                candidate = (mean_gain, threshold)
                if changed >= 5 and mean_gain > 0.0 and (
                    best is None or candidate > best
                ):
                    best = candidate
            return best[1] if best is not None else math.inf

        v15_gains: list[float] = []
        v14_gains: list[float] = []
        fold_gains: dict[str, float] = {}
        fold_thresholds: dict[str, Any] = {}
        for held_fold in range(fold_count):
            training_records = [
                record for record in records
                if record["fold"] != held_fold
            ]
            held_records = [
                record for record in records
                if record["fold"] == held_fold
            ]
            thresholds = choose_thresholds(training_records)
            v14_threshold = choose_v14_threshold(training_records)
            held_v15: list[float] = []
            for record in held_records:
                selected = (
                    self._choose_v15_action(
                        record["actions"], thresholds[0], thresholds[1]
                    )
                    if thresholds is not None else None
                )
                gain = (
                    float(selected["actual_gain"])
                    if selected is not None else 0.0
                )
                held_v15.append(gain)
                v15_gains.append(gain)
                v14_gains.append(
                    float(record["actions"][0]["actual_gain"])
                    if record["actions"]
                    and float(
                        record["actions"][0][
                            "predicted_expected_f1_gain"
                        ]
                    )
                    >= v14_threshold
                    else 0.0
                )
            fold_gains[str(held_fold)] = (
                sum(held_v15) / max(1, len(held_v15))
            )
            fold_thresholds[str(held_fold)] = (
                {
                    "add": thresholds[0],
                    "remove": thresholds[1],
                }
                if thresholds is not None else None
            )

        v15_mean = sum(v15_gains) / max(1, len(v15_gains))
        v14_mean = sum(v14_gains) / max(1, len(v14_gains))
        paired_deltas = [
            v15_gain - v14_gain
            for v15_gain, v14_gain in zip(v15_gains, v14_gains)
        ]
        lower_bound = self._bootstrap_lower_bound(
            v15_gains, "mpc-v15-dual-gate-bootstrap"
        )
        paired_lower_bound = self._bootstrap_lower_bound(
            paired_deltas, "mpc-v15-vs-v14-paired-bootstrap"
        )
        wins = sum(gain > 1e-12 for gain in v15_gains)
        losses = sum(gain < -1e-12 for gain in v15_gains)
        admitted = bool(
            len(records) >= 25
            and v15_mean > v14_mean
            and lower_bound > 0.0
            and paired_lower_bound > 0.0
            and losses <= 29
            and all(gain >= -0.002 for gain in fold_gains.values())
        )
        deployment_thresholds = (
            choose_thresholds(records) if admitted else None
        )

        all_actions = [
            action for record in records for action in record["actions"]
        ]
        self.add_necessity_verifier = self._fit_v15_binary_verifier(
            [action["add_features"] for action in all_actions],
            [action["add_label"] for action in all_actions],
        )
        self.remove_safety_verifier = self._fit_v15_binary_verifier(
            [action["remove_features"] for action in all_actions],
            [action["remove_label"] for action in all_actions],
        )
        self.dual_gate_policy = {
            "enabled": admitted,
            "add_threshold": (
                deployment_thresholds[0]
                if deployment_thresholds is not None else None
            ),
            "remove_threshold": (
                deployment_thresholds[1]
                if deployment_thresholds is not None else None
            ),
            "maximum_actions": 1,
        }
        self.metric_group_policy = {
            "enabled": admitted,
            "budget": 1 if admitted else 0,
            "minimum_expected_f1_gain": None,
            "scientist_top_k": 5,
            "action_bank_size": 20,
        }
        self.metric_group_calibration = {
            "protocol": (
                "exact_profile_clustered_stacked_oof_dual_counterfactual_gate"
            ),
            "selection_metric": "macro_functional_group_f1",
            "method_reads_llm_evaluation_cache": False,
            "uses_unimol": False,
            "query_count": len(records),
            "profile_cluster_count": len(clusters),
            "fold_sizes": fold_sizes,
            "v14_crossfit_gain": round(v14_mean, 8),
            "v15_crossfit_gain": round(v15_mean, 8),
            "increment_over_v14": round(v15_mean - v14_mean, 8),
            "paired_bootstrap_95pct_lower_bound_over_v14": round(
                paired_lower_bound, 8
            ),
            "v15_bootstrap_95pct_lower_bound": round(lower_bound, 8),
            "wins": wins,
            "losses": losses,
            "ties": len(v15_gains) - wins - losses,
            "fold_gains": {
                key: round(value, 8) for key, value in fold_gains.items()
            },
            "fold_thresholds": fold_thresholds,
            "deployment_thresholds": (
                {
                    "add": deployment_thresholds[0],
                    "remove": deployment_thresholds[1],
                }
                if deployment_thresholds is not None else None
            ),
            "admitted": admitted,
            "admission_rule": (
                "paired_bootstrap_lower_bound_over_v14_positive; "
                "v15_bootstrap_lower_bound_positive; losses_at_most_29; "
                "no_fold_below_minus_0.002"
            ),
        }
        self.residual_calibration = {
            "protocol": "v15_dual_gate_only",
            "query_count": len(records),
            "functional_group_decoder": self.metric_group_calibration,
        }

    def _calibrate_residual_policy(self) -> None:
        """Admit bounded residual experts using train-only grouped OOF.

        The selection target is exact hidden-molecule set F1.  No released
        functional-group cache, test identity, cardinality bucket, or API call
        participates in this calibration.
        """
        records: list[
            tuple[
                int,
                MPCStructureModel,
                dict[str, Any],
                list[dict[str, Any]],
                int,
                set[str],
                list[float],
            ]
        ] = []
        group_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            group = normalize(row.get("target_food")) or f"id:{row.get('id')}"
            group_to_indices[group].append(index)
        fold_count = min(5, len(group_to_indices))
        if fold_count < 2:
            return
        groups = sorted(group_to_indices)
        random.Random("mpc-v13-grouped-oof").shuffle(groups)
        fold_by_index: dict[int, int] = {}
        for group_position, group in enumerate(groups):
            for row_index in group_to_indices[group]:
                fold_by_index[row_index] = group_position % fold_count
        for fold in range(fold_count):
            fit_rows = [
                row
                for idx, row in enumerate(self.rows)
                if fold_by_index[idx] != fold
            ]
            held_rows = [
                row
                for idx, row in enumerate(self.rows)
                if fold_by_index[idx] == fold
            ]
            fold_model = MPCStructureModel(
                fit_rows,
                self.embeddings,
                self.ablation,
                self.db_path,
                calibrate_residuals=False,
            )
            for source_row in held_rows:
                partial = {
                    normalize(value)
                    for value in source_row.get("partial_molecules") or []
                    if normalize(value)
                }
                gold_molecules = {
                    normalize(value)
                    for value in source_row.get("missing_molecules") or []
                    if normalize(value)
                }
                if not partial or not gold_molecules:
                    continue
                masked_row = {
                    "id": f"{source_row.get('id')}:v13-oof",
                    "target_food": source_row.get("target_food"),
                    "partial_molecules": list(
                        source_row.get("partial_molecules") or []
                    ),
                    "n": len(gold_molecules),
                }
                items = fold_model._boundary_training_items(
                    masked_row,
                    exclude_train_index=None,
                    limit=len(gold_molecules) + 30,
                )
                group_demand = fold_model._predict_group_demand(masked_row)
                records.append(
                    (
                        fold,
                        fold_model,
                        masked_row,
                        items,
                        len(gold_molecules),
                        gold_molecules,
                        group_demand,
                    )
                )

        diagnostics: dict[str, Any] = {}
        for channel, budgets in {
            "retrieval": (1, 2),
        }.items():
            attempts: dict[str, Any] = {}
            best: tuple[float, int, dict[str, Any]] | None = None
            for budget in budgets:
                outcomes: list[tuple[int, float, bool]] = []
                for (
                    fold,
                    fold_model,
                    masked_row,
                    items,
                    n,
                    gold_molecules,
                    group_demand,
                ) in records:
                    baseline = set(
                        self._apply_residual_budget(
                            items, n, "retrieval", 0
                        )
                    )
                    if channel == "complementarity":
                        proposal = set(
                            self._apply_functional_budget(
                                items, n, budget, group_demand
                            )
                        )
                    elif channel == "structural":
                        proposal = set(
                            fold_model._apply_contextual_structural_budget(
                                masked_row,
                                items,
                                n,
                                budget,
                            )
                        )
                    else:
                        proposal = set(
                            self._apply_residual_budget(
                                items, n, channel, budget
                            )
                        )
                    gain = self._set_f1(
                        proposal, gold_molecules
                    ) - self._set_f1(baseline, gold_molecules)
                    outcomes.append((fold, gain, proposal != baseline))
                gains = [gain for _, gain, _ in outcomes]
                changed = sum(int(value) for _, _, value in outcomes)
                wins = sum(gain > 1e-12 for gain in gains)
                losses = sum(gain < -1e-12 for gain in gains)
                mean_gain = sum(gains) / len(gains) if gains else 0.0
                fold_gains = {
                    fold: (
                        sum(
                            gain
                            for record_fold, gain, _ in outcomes
                            if record_fold == fold
                        )
                        / max(
                            1,
                            sum(
                                record_fold == fold
                                for record_fold, _, _ in outcomes
                            ),
                        )
                    )
                    for fold in range(fold_count)
                }
                positive_folds = sum(
                    gain > 0.0 for gain in fold_gains.values()
                )
                bootstrap_means: list[float] = []
                if gains:
                    bootstrap_rng = random.Random(
                        f"mpc-v13-bootstrap:{channel}:{budget}"
                    )
                    for _ in range(500):
                        bootstrap_means.append(
                            sum(
                                gains[
                                    bootstrap_rng.randrange(len(gains))
                                ]
                                for _ in gains
                            )
                            / len(gains)
                        )
                bootstrap_means.sort()
                lower_bound = (
                    bootstrap_means[
                        max(0, int(0.05 * len(bootstrap_means)) - 1)
                    ]
                    if bootstrap_means
                    else 0.0
                )
                admitted = bool(
                    len(records) >= 25
                    and changed >= 5
                    and mean_gain > 0.0
                    and lower_bound > 0.0
                    and wins > losses
                    and positive_folds > fold_count / 2
                )
                attempt = {
                    "budget": budget,
                    "queries": len(records),
                    "changed_queries": changed,
                    "macro_hidden_molecule_f1_gain": round(
                        mean_gain, 8
                    ),
                    "paired_bootstrap_95pct_lower_bound": round(
                        lower_bound, 8
                    ),
                    "wins": wins,
                    "losses": losses,
                    "positive_folds": positive_folds,
                    "fold_gains": {
                        str(fold): round(gain, 8)
                        for fold, gain in fold_gains.items()
                    },
                    "admitted": admitted,
                }
                attempts[str(budget)] = attempt
                if admitted and (
                    best is None
                    or (mean_gain, -budget) > (best[0], -best[1])
                ):
                    best = (mean_gain, budget, attempt)
            selected_budget = best[1] if best is not None else 0
            self.residual_policy[channel]["global"] = selected_budget
            diagnostics[channel] = {
                "selected_budget": selected_budget,
                "selected": best[2] if best is not None else None,
                "attempts": attempts,
            }

        metric_attempts: dict[str, Any] = {}
        best_metric: (
            tuple[float, float, int, dict[str, Any]] | None
        ) = None
        record_cardinalities = sorted(
            n for _, _, _, _, n, _, _ in records
        )
        median_cardinality = (
            record_cardinalities[len(record_cardinalities) // 2]
            if record_cardinalities
            else 0
        )
        for budget in (1, 2, 3, 5):
            outcomes: list[tuple[int, int, float, bool]] = []
            for (
                fold,
                fold_model,
                _masked_row,
                items,
                n,
                gold_molecules,
                group_demand,
            ) in records:
                ordered_items = sorted(
                    items,
                    key=lambda item: (
                        int(
                            item.get("occurrence_rank")
                            or 10**9
                        ),
                        normalize(item.get("molecule")),
                    ),
                )
                baseline = [
                    normalize(item.get("molecule"))
                    for item in ordered_items[:n]
                    if normalize(item.get("molecule"))
                ]
                proposal = fold_model._apply_functional_budget(
                    items,
                    n,
                    budget,
                    group_demand,
                )
                gold_groups = fold_model._functional_group_set(
                    gold_molecules
                )
                baseline_groups = fold_model._functional_group_set(
                    set(baseline)
                )
                proposal_groups = fold_model._functional_group_set(
                    set(proposal)
                )
                gain = self._set_f1(
                    proposal_groups,
                    gold_groups,
                ) - self._set_f1(
                    baseline_groups,
                    gold_groups,
                )
                outcomes.append(
                    (
                        fold,
                        n,
                        gain,
                        proposal != baseline,
                    )
                )
            gains = [gain for _, _, gain, _ in outcomes]
            changed = sum(
                int(value) for _, _, _, value in outcomes
            )
            wins = sum(gain > 1e-12 for gain in gains)
            losses = sum(gain < -1e-12 for gain in gains)
            mean_gain = (
                sum(gains) / len(gains) if gains else 0.0
            )
            fold_gains = {
                fold: (
                    sum(
                        gain
                        for record_fold, _, gain, _ in outcomes
                        if record_fold == fold
                    )
                    / max(
                        1,
                        sum(
                            record_fold == fold
                            for record_fold, _, _, _ in outcomes
                        ),
                    )
                )
                for fold in range(fold_count)
            }
            positive_folds = sum(
                gain > 0.0 for gain in fold_gains.values()
            )
            lower_cardinality_gains = [
                gain
                for _, n, gain, _ in outcomes
                if n <= median_cardinality
            ]
            upper_cardinality_gains = [
                gain
                for _, n, gain, _ in outcomes
                if n > median_cardinality
            ]
            lower_cardinality_mean = (
                sum(lower_cardinality_gains)
                / len(lower_cardinality_gains)
                if lower_cardinality_gains
                else 0.0
            )
            upper_cardinality_mean = (
                sum(upper_cardinality_gains)
                / len(upper_cardinality_gains)
                if upper_cardinality_gains
                else 0.0
            )
            bootstrap_means: list[float] = []
            if gains:
                bootstrap_rng = random.Random(
                    f"mpc-v13-group-bootstrap:{budget}"
                )
                for _ in range(500):
                    bootstrap_means.append(
                        sum(
                            gains[
                                bootstrap_rng.randrange(len(gains))
                            ]
                            for _ in gains
                        )
                        / len(gains)
                    )
            bootstrap_means.sort()
            lower_bound = (
                bootstrap_means[
                    max(
                        0,
                        int(0.05 * len(bootstrap_means)) - 1,
                    )
                ]
                if bootstrap_means
                else 0.0
            )
            admitted = bool(
                budget == 1
                and len(records) >= 25
                and changed >= 5
                and mean_gain > 0.0
                and lower_bound > 0.0
                and wins > losses
                and positive_folds > fold_count / 2
                and lower_cardinality_mean >= 0.0
                and upper_cardinality_mean >= 0.0
            )
            attempt = {
                "budget": budget,
                "queries": len(records),
                "changed_queries": changed,
                "macro_functional_group_f1_gain": round(
                    mean_gain,
                    8,
                ),
                "paired_bootstrap_95pct_lower_bound": round(
                    lower_bound,
                    8,
                ),
                "wins": wins,
                "losses": losses,
                "positive_folds": positive_folds,
                "fold_gains": {
                    str(fold): round(gain, 8)
                    for fold, gain in fold_gains.items()
                },
                "cardinality_split": {
                    "median_n": median_cardinality,
                    "lower_half_mean_gain": round(
                        lower_cardinality_mean,
                        8,
                    ),
                    "upper_half_mean_gain": round(
                        upper_cardinality_mean,
                        8,
                    ),
                },
                "admitted": admitted,
            }
            metric_attempts[str(budget)] = attempt
            if admitted and (
                best_metric is None
                or (lower_bound, mean_gain, -budget)
                > (
                    best_metric[0],
                    best_metric[1],
                    -best_metric[2],
                )
            ):
                best_metric = (
                    lower_bound,
                    mean_gain,
                    budget,
                    attempt,
                )
        if best_metric is not None:
            self.metric_group_policy = {
                "budget": best_metric[2],
            }
        self.metric_group_calibration = {
            "protocol": (
                "train_only_grouped_five_fold_metric_aligned_set_decoding"
            ),
            "selection_metric": "macro_functional_group_f1",
            "group_source": (
                "molecule_intrinsic_flavordb_functional_groups"
            ),
            "method_reads_llm_evaluation_cache": False,
            "uses_unimol": False,
            "policy_selection_rule": (
                "predeclared_single_boundary_swap_only"
            ),
            "selected": (
                best_metric[3] if best_metric is not None else None
            ),
            "attempts": metric_attempts,
        }
        diagnostics["functional_group_decoder"] = {
            "selected_budget": int(
                self.metric_group_policy.get("budget", 0)
            ),
            "selected": (
                best_metric[3] if best_metric is not None else None
            ),
            "attempts": metric_attempts,
        }

        action_attempts: dict[str, Any] = {}
        best_action: (
            tuple[float, float, float, dict[str, Any]] | None
        ) = None
        for threshold in (0.55, 0.65, 0.75, 0.85, 0.90):
            outcomes: list[tuple[int, float, bool]] = []
            for (
                fold,
                fold_model,
                _masked_row,
                items,
                n,
                gold_molecules,
                _group_demand,
            ) in records:
                baseline_values = [
                    normalize(item.get("molecule"))
                    for item in sorted(
                        items,
                        key=lambda item: (
                            int(
                                item.get("occurrence_rank")
                                or 10**9
                            ),
                            normalize(item.get("molecule")),
                        ),
                    )[:n]
                ]
                proposal_values = list(baseline_values)
                ranked_actions = fold_model._rank_retrieval_actions(
                    items,
                    n,
                )
                accepted = next(
                    (
                        action
                        for action in ranked_actions
                        if float(
                            action["utility_probability"]
                        )
                        >= threshold
                        and int(
                            action[
                                "independent_statistical_support_count"
                            ]
                        )
                        >= 2
                    ),
                    None,
                )
                if (
                    accepted is not None
                    and accepted["remove_key"] in proposal_values
                    and accepted["add_key"] not in proposal_values
                ):
                    proposal_values[
                        proposal_values.index(
                            accepted["remove_key"]
                        )
                    ] = accepted["add_key"]
                gain = self._set_f1(
                    set(proposal_values),
                    gold_molecules,
                ) - self._set_f1(
                    set(baseline_values),
                    gold_molecules,
                )
                outcomes.append(
                    (
                        fold,
                        gain,
                        proposal_values != baseline_values,
                    )
                )
            gains = [gain for _, gain, _ in outcomes]
            changed = sum(
                int(value) for _, _, value in outcomes
            )
            wins = sum(gain > 1e-12 for gain in gains)
            losses = sum(gain < -1e-12 for gain in gains)
            mean_gain = (
                sum(gains) / len(gains) if gains else 0.0
            )
            fold_gains = {
                fold: (
                    sum(
                        gain
                        for record_fold, gain, _ in outcomes
                        if record_fold == fold
                    )
                    / max(
                        1,
                        sum(
                            record_fold == fold
                            for record_fold, _, _ in outcomes
                        ),
                    )
                )
                for fold in range(fold_count)
            }
            positive_folds = sum(
                gain > 0.0 for gain in fold_gains.values()
            )
            bootstrap_means: list[float] = []
            if gains:
                bootstrap_rng = random.Random(
                    f"mpc-v13-action-bootstrap:{threshold}"
                )
                for _ in range(500):
                    bootstrap_means.append(
                        sum(
                            gains[
                                bootstrap_rng.randrange(len(gains))
                            ]
                            for _ in gains
                        )
                        / len(gains)
                    )
            bootstrap_means.sort()
            lower_bound = (
                bootstrap_means[
                    max(
                        0,
                        int(0.05 * len(bootstrap_means)) - 1,
                    )
                ]
                if bootstrap_means
                else 0.0
            )
            admitted = bool(
                len(records) >= 25
                and changed >= 5
                and mean_gain > 0.0
                and lower_bound > 0.0
                and wins > losses
                and positive_folds > fold_count / 2
            )
            attempt = {
                "threshold": threshold,
                "queries": len(records),
                "changed_queries": changed,
                "macro_hidden_molecule_f1_gain": round(
                    mean_gain,
                    8,
                ),
                "paired_bootstrap_95pct_lower_bound": round(
                    lower_bound,
                    8,
                ),
                "wins": wins,
                "losses": losses,
                "positive_folds": positive_folds,
                "fold_gains": {
                    str(fold): round(gain, 8)
                    for fold, gain in fold_gains.items()
                },
                "admitted": admitted,
            }
            action_attempts[f"{threshold:.2f}"] = attempt
            if admitted and (
                best_action is None
                or (lower_bound, mean_gain, threshold)
                > (
                    best_action[0],
                    best_action[1],
                    best_action[2],
                )
            ):
                best_action = (
                    lower_bound,
                    mean_gain,
                    threshold,
                    attempt,
                )
        # v12's action gate optimized exact missing-molecule F1.  That target
        # is not admitted as a proxy for the released functional-group F1 in
        # v13, so the legacy Reviewer action path stays disabled even when its
        # historical diagnostic is positive.
        self.retrieval_action_policy = {
            "budget": 0,
            "threshold": 1.0,
        }
        self.retrieval_action_calibration = {
            "protocol": (
                "train_only_grouped_five_fold_action_utility"
            ),
            "selection_metric": "macro_hidden_molecule_set_f1",
            "policy_selection_rule": (
                "maximize_positive_paired_bootstrap_lower_bound_then_mean_gain"
            ),
            "selected": (
                best_action[3] if best_action is not None else None
            ),
            "admitted_for_v13": False,
            "disabled_reason": (
                "exact_molecule_objective_not_aligned_with_"
                "functional_group_f1"
            ),
            "attempts": action_attempts,
        }
        self.residual_calibration = {
            "protocol": (
                "train_only_grouped_five_fold_task_shaped_oof"
            ),
            "folds": fold_count,
            "query_count": len(records),
            "selection_metric": "macro_hidden_molecule_set_f1",
            "method_reads_llm_evaluation_cache": False,
            "minimum_changed_queries": 5,
            "requires_positive_paired_bootstrap_lower_bound": True,
            "requires_wins_greater_than_losses": True,
            "requires_majority_positive_folds": True,
            "uses_cardinality_buckets": False,
            "policies": diagnostics,
            "retrieval_action": self.retrieval_action_calibration,
            "functional_group_decoder": (
                self.metric_group_calibration
            ),
        }

    def _decode_exact_n_set(
        self,
        row: dict[str, Any],
        ranked_items: list[dict[str, Any]],
        n: int,
        seed_field: str,
    ) -> tuple[list[str], dict[str, Any]]:
        """Optimize a learned conditional set energy by exact-N exchanges."""
        if n <= 0:
            return [], {
                "initial_energy": 0.0,
                "final_energy": 0.0,
                "accepted_swaps": [],
            }
        ordered = sorted(
            ranked_items,
            key=lambda item: (
                -float(item.get(seed_field) or 0.0),
                int(item.get("occurrence_rank") or 10**9),
                normalize(item.get("molecule")),
            ),
        )
        selected = [
            normalize(item["molecule"]) for item in ordered[:n]
        ]
        candidate_pool_size = min(
            len(ranked_items),
            max(n + 30, 3 * n),
        )
        pool = stable_unique(
            [
                normalize(item["molecule"])
                for item in ranked_items[:candidate_pool_size]
            ]
            + [
                normalize(item["molecule"])
                for item in ordered[:candidate_pool_size]
            ]
        )
        raw_features = {
            normalize(item["molecule"]): list(
                item.get("_model_features") or []
            )
            for item in ranked_items
            if item.get("_model_features")
        }
        current_energy = self._score_set(
            row,
            selected,
            raw_features,
        )
        initial_energy = current_energy
        accepted_swaps: list[dict[str, Any]] = []
        max_rounds = min(5, n)
        for _ in range(max_rounds):
            selected_set = set(selected)
            additions = [
                candidate
                for candidate in pool
                if candidate not in selected_set
            ][:20]
            removable = sorted(
                selected,
                key=lambda key: (
                    float(
                        next(
                            (
                                item.get("occurrence_score") or 0.0
                                for item in ranked_items
                                if normalize(item["molecule"]) == key
                            ),
                            0.0,
                        )
                    ),
                    key,
                ),
            )[: min(8, len(selected))]
            best: tuple[float, str, str, list[str]] | None = None
            for remove_key in removable:
                for add_key in additions:
                    proposal = [
                        key for key in selected if key != remove_key
                    ] + [add_key]
                    proposal_energy = self._score_set(
                        row,
                        proposal,
                        raw_features,
                    )
                    gain = proposal_energy - current_energy
                    candidate_swap = (
                        gain,
                        remove_key,
                        add_key,
                        proposal,
                    )
                    if best is None or candidate_swap[:3] > best[:3]:
                        best = candidate_swap
            if best is None or best[0] <= 1e-9:
                break
            gain, remove_key, add_key, selected = best
            current_energy += gain
            accepted_swaps.append(
                {
                    "removed": self.display_names.get(
                        remove_key,
                        remove_key,
                    ),
                    "added": self.display_names.get(
                        add_key,
                        add_key,
                    ),
                    "energy_gain": round(gain, 8),
                }
            )
        return selected, {
            "seed_field": seed_field,
            "candidate_pool_size": candidate_pool_size,
            "initial_energy": round(initial_energy, 8),
            "final_energy": round(current_energy, 8),
            "accepted_swaps": accepted_swaps,
            "exact_n": len(selected) == n
            and len(set(selected)) == n,
        }

    def rank(
        self,
        row: dict[str, Any],
        evidence_molecules: set[str],
        top_k: int,
        exclude_train_index: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        partial = {normalize(value) for value in row.get("partial_molecules") or []}
        candidates = [candidate for candidate in self.universe if candidate not in partial]
        retrieved_support = self._build_retrieved_support(
            row,
            exclude_train_index=exclude_train_index,
            top_k=10,
        )
        idf_retrieved_support, idf_retrieved_counts = (
            self._build_idf_retrieved_support(
                row,
                exclude_train_index=exclude_train_index,
                top_k=15,
            )
        )
        expected_attributes = self._expected_attribute_weights(
            row,
            exclude_train_index=exclude_train_index,
        )
        query_context = self._build_query_context(row)
        group_demand = self._predict_group_demand(row)
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            features = self._query_features(
                row,
                candidate,
                retrieved_support,
                excluded_profile=(
                    self.full_profiles[exclude_train_index]
                    if exclude_train_index is not None
                    else None
                ),
                expected_attributes=expected_attributes,
                query_context=query_context,
            )
            if self.ablation == "raw_unimol_nn":
                occurrence_score = (
                    0.5 * features[4] + 0.5 * features[5]
                )
                source = "raw_unimol_nearest_neighbor"
            elif self.ranker is not None:
                occurrence_score = float(
                    self.ranker.decision_function(
                        [
                            [
                                features[index]
                                for index in self.PRIMARY_FEATURE_INDICES
                            ]
                        ]
                    )[0]
                )
                source = "task_shaped_food_relevance_adapter"
            else:
                occurrence_score = (
                    0.20 * features[0]
                    + 0.30 * features[1]
                    + 0.20 * features[2]
                    + 0.30 * features[3]
                )
                source = "train_statistics"
            unimol_score = (
                0.20 * features[4]
                + 0.20 * features[5]
                + 0.45 * features[7]
                - 0.15 * features[8]
            )
            perceptual_score = (
                0.20 * features[9]
                + 0.25 * features[10]
                + 0.55 * features[11]
            )
            structural_features = [
                features[index]
                for index in self.STRUCTURAL_FEATURE_INDICES
            ]
            candidate_functional_groups = (
                self.functional_group_sets.get(candidate, set())
            )
            functional_group_probabilities = [
                float(group in candidate_functional_groups)
                for group in self.functional_group_vocabulary
            ]
            functional_group_demand_score = (
                sum(
                    (2.0 * demand - 1.0) * probability
                    for demand, probability in zip(
                        group_demand,
                        functional_group_probabilities,
                    )
                )
                / max(1, len(group_demand))
                if group_demand
                else 0.0
            )
            if self.structural_ranker is not None:
                unimol_set_compatibility_score = float(
                    self.structural_ranker.decision_function(
                        [structural_features]
                    )[0]
                )
            else:
                # This guarded fallback is intentionally occurrence-aware:
                # raw molecular similarity alone was anti-discriminative in
                # the v7 audit and must not become a global MPC ranker.
                unimol_set_compatibility_score = (
                    0.30 * features[1]
                    + 0.20 * features[2]
                    + 0.15 * features[7]
                    - 0.05 * features[8]
                    + 0.15 * features[10]
                    + 0.15 * features[11]
                )
            scored.append(
                {
                    "molecule": self.display_names[candidate],
                    "score": round(occurrence_score, 6),
                    "base_score": round(occurrence_score, 6),
                    "occurrence_score": round(occurrence_score, 6),
                    "unimol_conditional_score": round(unimol_score, 6),
                    "unimol_set_compatibility_score": round(
                        unimol_set_compatibility_score,
                        6,
                    ),
                    "perceptual_score": round(perceptual_score, 6),
                    "source": source,
                    "frequency_prior": round(features[0], 6),
                    "cooccurrence": round(features[1], 6),
                    "cooccurrence_max": round(features[2], 6),
                    "retrieved_profile_support": round(features[3], 6),
                    "idf_retrieved_support": round(
                        idf_retrieved_support.get(candidate, 0.0),
                        6,
                    ),
                    "idf_retrieved_profile_count": int(
                        idf_retrieved_counts.get(candidate, 0)
                    ),
                    "unimol_mean_similarity": round(features[4], 6),
                    "unimol_max_similarity": round(features[5], 6),
                    "unimol_min_similarity": round(features[6], 6),
                    "unimol_query_compatibility": round(features[7], 6),
                    "unimol_query_distance": round(features[8], 6),
                    "perceptual_mean_similarity": round(features[9], 6),
                    "perceptual_max_similarity": round(features[10], 6),
                    "perceptual_residual_support": round(features[11], 6),
                    "functional_group_demand_score": round(
                        functional_group_demand_score,
                        6,
                    ),
                    "predicted_functional_groups": (
                        sorted(candidate_functional_groups)
                    ),
                    "unimol_available": (
                        self.embeddings is not None
                        and self.embeddings.vector(candidate) is not None
                    ),
                    "functional_group_available": bool(
                        candidate_functional_groups
                    ),
                    "typed_evidence_linked": candidate in evidence_molecules,
                    "boundary_adjusted": False,
                    "_intrinsic_attributes": sorted(
                        self.perceptual_adapter.attributes(candidate)
                        if self.perceptual_adapter is not None
                        else set()
                    ),
                    "_functional_group_probabilities": (
                        functional_group_probabilities
                    ),
                    "_model_features": features,
                }
            )

        requested_n = max(0, int(row.get("n") or 0))
        rows = sorted(
            scored,
            key=lambda item: (
                -float(item["occurrence_score"]),
                normalize(item["molecule"]),
            ),
        )
        for rank, item in enumerate(rows, 1):
            item["occurrence_rank"] = rank
            item["relevance_score"] = item["occurrence_score"]
            item["score"] = item["occurrence_score"]

        def active_rank(
            field: str,
            predicate: Any,
        ) -> dict[str, int | None]:
            active = [item for item in rows if predicate(item)]
            active.sort(
                key=lambda item: (
                    -float(item.get(field) or 0.0),
                    int(item["occurrence_rank"]),
                    normalize(item["molecule"]),
                )
            )
            output: dict[str, int | None] = {
                normalize(item["molecule"]): None for item in rows
            }
            for rank, item in enumerate(active, 1):
                output[normalize(item["molecule"])] = rank
            return output

        retrieval_ranks = active_rank(
            "retrieved_profile_support",
            lambda item: float(
                item.get("retrieved_profile_support") or 0.0
            ) > 0.0,
        )
        idf_retrieval_ranks = active_rank(
            "idf_retrieved_support",
            lambda item: float(
                item.get("idf_retrieved_support") or 0.0
            )
            > 0.0,
        )
        unimol_ranks = active_rank(
            "unimol_conditional_score",
            lambda item: bool(item.get("unimol_available")),
        )
        perceptual_ranks = active_rank(
            "perceptual_score",
            lambda item: float(item.get("perceptual_score") or 0.0) > 0.0,
        )
        structural_set_ranks = active_rank(
            "unimol_set_compatibility_score",
            lambda item: bool(item.get("unimol_available")),
        )
        for item in rows:
            key = normalize(item["molecule"])
            item["retrieval_rank"] = retrieval_ranks[key]
            item["idf_retrieval_rank"] = (
                idf_retrieval_ranks[key]
            )
            item["unimol_rank"] = unimol_ranks[key]
            item["perceptual_rank"] = perceptual_ranks[key]
            item["unimol_set_compatibility_rank"] = (
                structural_set_ranks[key]
            )
            item["unimol_set_rank_lift"] = (
                int(item["occurrence_rank"])
                - int(structural_set_ranks[key])
                if structural_set_ranks[key] is not None
                else None
            )
            item["candidate_sources"] = (
                ["occurrence"]
                + (["retrieval"] if retrieval_ranks[key] is not None else [])
                + (
                    ["idf_retrieval"]
                    if idf_retrieval_ranks[key] is not None
                    else []
                )
                + (["unimol"] if unimol_ranks[key] is not None else [])
                + (
                    ["perceptual"]
                    if perceptual_ranks[key] is not None
                    else []
                )
                + (["evidence"] if item["typed_evidence_linked"] else [])
            )

        occurrence_keys = [
            normalize(item["molecule"])
            for item in rows[:requested_n]
        ]
        group_cardinality_posterior = (
            self._predict_group_cardinality_posterior(row)
        )
        v14_action_bank = self._build_v14_action_bank(
            rows,
            requested_n,
            group_cardinality_posterior,
            bank_size=int(
                self.metric_group_policy.get("action_bank_size", 20)
            ),
            scientist_top_k=int(
                self.metric_group_policy.get("scientist_top_k", 5)
            ),
        )
        metric_group_budget = int(
            self.metric_group_policy.get("budget", 0)
        )
        for action in v14_action_bank["scientist_actions"]:
            add_features, remove_features = self._v15_action_features(
                action,
                v14_action_bank["h1"],
                rows,
                group_cardinality_posterior,
                requested_n,
            )
            action["add_necessity_probability"] = (
                float(
                    self.add_necessity_verifier.predict_proba(
                        [add_features]
                    )[0, 1]
                )
                if self.add_necessity_verifier is not None else 0.0
            )
            action["remove_safety_probability"] = (
                float(
                    self.remove_safety_verifier.predict_proba(
                        [remove_features]
                    )[0, 1]
                )
                if self.remove_safety_verifier is not None else 0.0
            )
        add_threshold = self.dual_gate_policy.get("add_threshold")
        remove_threshold = self.dual_gate_policy.get("remove_threshold")
        best_v15_action = (
            self._choose_v15_action(
                v14_action_bank["scientist_actions"],
                float(add_threshold),
                float(remove_threshold),
            )
            if metric_group_budget > 0
            and add_threshold is not None
            and remove_threshold is not None
            else None
        )
        metric_group_keys = (
            list(best_v15_action["proposal"])
            if best_v15_action is not None
            else list(occurrence_keys)
        )
        ranked_actions = self._rank_retrieval_actions(
            rows,
            requested_n,
        )
        action_threshold = float(
            self.retrieval_action_policy.get("threshold", 1.0)
        )
        action_budget = int(
            self.retrieval_action_policy.get("budget", 0)
        )
        admitted_actions = [
            action
            for action in ranked_actions
            if action_budget > 0
            and float(action["utility_probability"])
            >= action_threshold
            and int(
                action[
                    "independent_statistical_support_count"
                ]
            )
            >= 2
        ][: max(0, action_budget)]
        retrieval_budget = int(
            self.residual_policy["retrieval"].get("global", 0)
        )
        structural_budget = int(
            self.residual_policy["structural"].get("global", 0)
        )
        complementarity_budget = int(
            self.residual_policy["complementarity"].get("global", 0)
        )
        conditional_keys = self._apply_residual_budget(
            rows,
            requested_n,
            "retrieval",
            retrieval_budget,
        )
        structural_candidate = self._apply_contextual_structural_budget(
            row,
            rows,
            requested_n,
            structural_budget,
        )
        complementarity_candidate = self._apply_functional_budget(
            rows,
            requested_n,
            complementarity_budget,
            group_demand,
        )
        structural_selected = (
            self.residual_calibration.get("policies", {})
            .get("structural", {})
            .get("selected")
            or {}
        )
        complementarity_selected = (
            self.residual_calibration.get("policies", {})
            .get("complementarity", {})
            .get("selected")
            or {}
        )
        structural_gain = float(
            structural_selected.get(
                "macro_hidden_molecule_f1_gain", 0.0
            )
            or 0.0
        )
        complementarity_gain = float(
            complementarity_selected.get(
                "macro_hidden_molecule_f1_gain", 0.0
            )
            or 0.0
        )
        if (
            complementarity_budget > 0
            and complementarity_gain > structural_gain
        ):
            structural_keys = complementarity_candidate
            h3_source = "local_attribute_complementarity"
            h3_budget = complementarity_budget
        else:
            structural_keys = structural_candidate
            h3_source = "task_adapted_unimol_swap_residual"
            h3_budget = structural_budget
        proposal_sets = [
            set(occurrence_keys),
            set(conditional_keys),
            set(structural_keys),
        ]
        proposal_union = set().union(*proposal_sets)
        proposal_intersection = set.intersection(*proposal_sets)
        proposal_disagreement = (
            1.0 - len(proposal_intersection) / max(1, len(proposal_union))
        )
        h2_additions = set(conditional_keys) - set(occurrence_keys)
        h3_additions = set(structural_keys) - set(occurrence_keys)
        independently_supported_additions = (
            h2_additions & h3_additions
        )
        base_cutoff_margin = (
            float(rows[requested_n - 1]["occurrence_score"])
            - float(rows[requested_n]["occurrence_score"])
            if 0 < requested_n < len(rows)
            else 1.0
        )
        occurrence_values = [
            float(item["occurrence_score"])
            for item in rows[: min(len(rows), max(20, requested_n + 1))]
        ]
        occurrence_scale = (
            max(1e-12, max(occurrence_values) - min(occurrence_values))
            if occurrence_values
            else 1.0
        )
        normalized_cutoff_margin = base_cutoff_margin / occurrence_scale
        conditional_swaps = len(
            set(occurrence_keys) - set(conditional_keys)
        )
        structural_swaps = len(
            set(occurrence_keys) - set(structural_keys)
        )
        metric_group_swaps = len(
            set(occurrence_keys) - set(metric_group_keys)
        )
        boundary_width = max(
            conditional_swaps,
            structural_swaps,
            metric_group_swaps,
        )
        boundary_start = max(0, requested_n - boundary_width)
        boundary_end = min(
            len(rows),
            requested_n + max(10, 5 * boundary_width),
        )
        selected_key_union = (
            set(conditional_keys)
            | set(structural_keys)
            | set(metric_group_keys)
        )
        for item in rows:
            item["boundary_adjusted"] = (
                normalize(item["molecule"]) in selected_key_union
                and normalize(item["molecule"]) not in set(occurrence_keys)
            )
            item["evidence_score_bonus"] = 0.0
        diagnostics = {
            "requested_n": requested_n,
            "candidate_universe_size": len(candidates),
            "candidate_pool_size": len(rows),
            "candidate_universe_source": (
                "flavordb_catalog_plus_training_profiles"
            ),
            "candidate_catalog_independent_of_unimol": True,
            "task_adapter": self.ranker is not None,
            "ranker_training_queries": self.ranker_training_queries,
            "ranker_training_pairs": self.ranker_training_pairs,
            "set_ranker_training_queries": (
                self.set_ranker_training_queries
            ),
            "set_ranker_training_pairs": (
                self.set_ranker_training_pairs
            ),
            "structural_ranker_training_pairs": (
                self.structural_ranker_training_pairs
            ),
            "boundary_swap_ranker": (
                self.boundary_swap_ranker is not None
            ),
            "boundary_swap_training_queries": (
                self.boundary_swap_training_queries
            ),
            "boundary_swap_training_pairs": (
                self.boundary_swap_training_pairs
            ),
            "boundary_swap_positive_count": (
                self.boundary_swap_positive_count
            ),
            "boundary_swap_negative_count": (
                self.boundary_swap_negative_count
            ),
            "boundary_swap_neutral_count": (
                self.boundary_swap_neutral_count
            ),
            "perceptual_adapter": self.perceptual_adapter is not None,
            "perceptual_latent_dimension": (
                self.perceptual_adapter.dimension
                if self.perceptual_adapter is not None
                else 0
            ),
            "perceptual_descriptor_vocabulary_size": (
                self.perceptual_adapter.descriptor_vocabulary_size
                if self.perceptual_adapter is not None
                else 0
            ),
            "perceptual_descriptor_covered_count": (
                self.perceptual_adapter.descriptor_covered_count
                if self.perceptual_adapter is not None
                else 0
            ),
            "boundary_width": boundary_width,
            "boundary_start": boundary_start,
            "boundary_end": boundary_end,
            "global_rank_fusion": False,
            "primary_ranker": "masked_query_pairwise_occurrence",
            "structural_hypothesis_ranker": h3_source,
            "residual_policy": self.residual_policy,
            "residual_calibration": self.residual_calibration,
            "query_bucket": None,
            "retrieval_exchange_budget": retrieval_budget,
            "structural_exchange_budget": structural_budget,
            "functional_exchange_budget": h3_budget,
            "complementarity_exchange_budget": complementarity_budget,
            "functional_policy_budget": complementarity_budget,
            "functional_proxy_gain": None,
            "functional_proxy_gain_threshold": None,
            "functional_action_enabled": complementarity_budget > 0,
            "base_cutoff_margin": round(base_cutoff_margin, 8),
            "normalized_cutoff_margin": round(
                normalized_cutoff_margin, 8
            ),
            "boundary_swap_count": max(
                conditional_swaps,
                structural_swaps,
            ),
            "proposal_disagreement": round(proposal_disagreement, 8),
            "independently_supported_residual_count": len(
                independently_supported_additions
            ),
            "reviewer_gate_enabled": bool(
                admitted_actions
            ),
            "metric_group_policy": self.metric_group_policy,
            "metric_group_calibration": (
                self.metric_group_calibration
            ),
            "v15_action_bank": {
                "bank_size": len(v14_action_bank["actions"]),
                "scientist_top_k": len(
                    v14_action_bank["scientist_actions"]
                ),
                "h1_expected_f1": round(
                    float(v14_action_bank["h1_expected_f1"]), 8
                ),
                "selected_action": best_v15_action,
                "actions": v14_action_bank["scientist_actions"],
                "executor": (
                    "dual_gate_single_exact_n_swap"
                    if best_v15_action is not None
                    else "keep_h1"
                ),
                "dual_gate_policy": self.dual_gate_policy,
            },
            "retrieval_action_policy": self.retrieval_action_policy,
            "retrieval_action_calibration": (
                self.retrieval_action_calibration
            ),
            "retrieval_action_ranker": (
                self.retrieval_action_ranker is not None
            ),
            "retrieval_action_training_queries": (
                self.retrieval_action_training_queries
            ),
            "retrieval_action_training_pairs": (
                self.retrieval_action_training_pairs
            ),
            "retrieval_action_positive_count": (
                self.retrieval_action_positive_count
            ),
            "admitted_retrieval_actions": [
                {
                    **action,
                    "remove_molecule": self.display_names.get(
                        action["remove_key"],
                        action["remove_key"],
                    ),
                    "add_molecule": self.display_names.get(
                        action["add_key"],
                        action["add_key"],
                    ),
                }
                for action in admitted_actions
            ],
            "proposal_occurrence": [
                self.display_names.get(key, key) for key in occurrence_keys
            ],
            "proposal_metric_aligned": [
                self.display_names.get(key, key)
                for key in metric_group_keys
            ],
            "proposal_retrieval_residual": [
                self.display_names.get(key, key)
                for key in conditional_keys
            ],
            "proposal_functional_residual": [
                self.display_names.get(key, key)
                for key in structural_keys
            ],
            "conditional_set_decoder": {
                "method": "grouped_oof_retrieval_boundary_residual",
                "budget": retrieval_budget,
                "exact_n": len(conditional_keys) == requested_n,
            },
            "structural_seed_set_decoder": {
                "method": h3_source,
                "budget": h3_budget,
                "exact_n": len(structural_keys) == requested_n,
            },
            "functional_group_vocabulary_size": (
                len(self.functional_group_vocabulary)
            ),
            "functional_group_probe_dimension": (
                self.perceptual_adapter.functional_group_probe_dimension
                if self.perceptual_adapter is not None
                else 0
            ),
            "functional_group_probe_trainable_label_count": (
                self.perceptual_adapter.functional_group_probe_trainable_label_count
                if self.perceptual_adapter is not None
                else 0
            ),
            "group_demand_training_rows": self.group_demand_training_rows,
            "predicted_group_demand": (
                [
                    {
                        "group": group,
                        "probability": round(probability, 6),
                    }
                    for group, probability in sorted(
                        zip(
                            self.functional_group_vocabulary,
                            group_demand,
                        ),
                        key=lambda item: (-item[1], item[0]),
                    )[:15]
                ]
                if self.functional_group_vocabulary
                else []
            ),
            "expected_attribute_count": len(expected_attributes),
            "set_decoder": (
                "counterfactual_add_necessity_remove_safety_executor"
                if metric_group_budget > 0
                else "frozen_occurrence_h1"
            ),
            "pseudo_negative_policy": (
                "task_shaped_pairwise_positive_unlabeled"
                if self.ranker is not None
                else "none"
            ),
            "evidence_changes_structure_score": False,
            "evidence_linked_candidate_count": sum(
                1 for item in rows if item["typed_evidence_linked"]
            ),
        }
        for item in rows:
            item["intrinsic_attribute_count"] = len(
                item.get("_intrinsic_attributes") or []
            )
            item.pop("_intrinsic_attributes", None)
            item.pop("_model_features", None)
        required_keys = (
            set(occurrence_keys)
            | set(metric_group_keys)
            | set(conditional_keys)
            | set(structural_keys)
            | {
                key
                for action in admitted_actions
                for key in (
                    action["remove_key"],
                    action["add_key"],
                )
            }
        )
        ledger_rows = [
            item
            for item in rows
            if normalize(item["molecule"]) in required_keys
        ]
        ledger_keys = {
            normalize(item["molecule"]) for item in ledger_rows
        }
        ledger_rows.extend(
            item
            for item in rows
            if normalize(item["molecule"]) not in ledger_keys
        )
        return ledger_rows[: max(top_k, len(required_keys))], diagnostics


def load_mfp_evidence(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise OptimizedAgentError("MFP evidence must be a dictionary")
    return {
        normalize(key): [str(item) for item in value]
        for key, value in obj.items()
        if isinstance(value, list)
    }


def load_mpc_evidence(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise OptimizedAgentError("MPC evidence must be a dictionary")
    output: dict[str, list[str]] = {}
    for key, value in obj.items():
        if isinstance(value, list):
            output[normalize(key)] = [
                json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                for item in value
            ]
        elif isinstance(value, dict):
            output[normalize(key)] = [json.dumps(value, ensure_ascii=False)]
        elif value is not None:
            output[normalize(key)] = [str(value)]
    return output


def link_molecules(snippets: list[str], molecule_names: list[str]) -> set[str]:
    normalized_text = f" {normalize(' '.join(snippets))} "
    linked: set[str] = set()
    for name in molecule_names:
        key = normalize(name)
        if len(key) >= 3 and f" {key} " in normalized_text:
            linked.add(key)
    return linked


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
SENSORY_ONLY_CUES = (
    "odor",
    "odour",
    "aroma",
    "flavor",
    "flavour",
    "note",
    "smell",
    "reminiscent",
    "like character",
    "characterized by",
)
FUNCTIONAL_ROLE_CUES = (
    "contributes",
    "contribute",
    "responsible for",
    "gives",
    "imparts",
    "masking",
    "odorant",
    "flavoring",
    "flavouring",
    "aroma chemical",
    "key aroma",
)
CONTRADICTION_CUES = (
    "does not contain",
    "not present",
    "not detected",
    "no evidence",
    "unable to verify",
    "cannot verify",
    "don't have",
    "do not have",
)


def classify_evidence_snippet(snippet: str) -> str:
    text = normalize(snippet)
    if any(cue in text for cue in DIRECT_OCCURRENCE_CUES):
        return "direct_occurrence"
    if any(cue in text for cue in SENSORY_ONLY_CUES):
        return "sensory_or_property_only"
    return "ambiguous"


def classify_mpc_evidence_snippet(snippet: str) -> str:
    text = normalize(snippet)
    if any(cue in text for cue in CONTRADICTION_CUES):
        return "contradiction"
    if any(cue in text for cue in DIRECT_OCCURRENCE_CUES):
        return "occurrence_support"
    if any(cue in text for cue in FUNCTIONAL_ROLE_CUES):
        return "functional_role_support"
    if any(cue in text for cue in SENSORY_ONLY_CUES):
        return "sensory_replication_support"
    return "ambiguous"


def anchor_score_map(diagnostics: dict[str, Any]) -> dict[str, float]:
    return {
        normalize(item.get("molecule")): float(item.get("score") or 0.0)
        for item in diagnostics.get("anchor_scores") or []
        if isinstance(item, dict) and normalize(item.get("molecule"))
    }


def format_mfp_evidence(
    row: dict[str, Any],
    evidence: dict[str, list[str]],
    idf: dict[str, float],
    anchor_scores: dict[str, float],
    max_molecules: int,
    max_snippets: int,
) -> tuple[str, dict[str, Any]]:
    blocks: list[str] = []
    used: list[str] = []
    evidence_type_counts: Counter[str] = Counter()
    ranked_molecules = sorted(
        enumerate(row.get("molecules") or []),
        key=lambda item: (
            -int(bool(evidence.get(normalize(item[1])))),
            -anchor_scores.get(
                normalize(item[1]),
                idf.get(normalize(item[1]), 0.0),
            ),
            item[0],
        ),
    )
    for _, molecule in ranked_molecules[:max_molecules]:
        snippets = evidence.get(normalize(molecule), [])[:max_snippets]
        if not snippets:
            continue
        typed_snippets = []
        for snippet in snippets:
            evidence_type = classify_evidence_snippet(snippet)
            evidence_type_counts[evidence_type] += 1
            typed_snippets.append(f"- [{evidence_type}] {snippet}")
        blocks.append(
            f"Molecule: {molecule}\n" + "\n".join(typed_snippets)
        )
        used.append(str(molecule))
    return "\n\n".join(blocks) or "No linked official evidence.", {
        "evidence_molecules": used,
        "evidence_block_count": len(blocks),
        "evidence_type_counts": dict(evidence_type_counts),
        "starting_point_method": "unimol_category_contrast_times_train_idf",
        "uses_test_label": False,
    }


def format_mpc_evidence(
    row: dict[str, Any],
    evidence: dict[str, list[str]],
    max_snippets: int,
    molecule_names: list[str],
) -> tuple[str, dict[str, Any], set[str]]:
    snippets = evidence.get(normalize(row.get("target_food")), [])[:max_snippets]
    typed: list[tuple[str, str, list[str]]] = []
    relation_counts: Counter[str] = Counter()
    direct_occurrence_linked: set[str] = set()
    for snippet in snippets:
        relation = classify_mpc_evidence_snippet(snippet)
        linked = sorted(link_molecules([snippet], molecule_names))
        typed.append((relation, snippet, linked))
        relation_counts[relation] += 1
        if relation == "occurrence_support":
            direct_occurrence_linked.update(linked)
    text = (
        "\n".join(
            f"- [E{idx}][relation={relation}]"
            f"[molecules={','.join(linked) if linked else 'none'}] {snippet}"
            for idx, (relation, snippet, linked) in enumerate(typed, 1)
        )
        or "No linked official evidence."
    )
    return text, {
        "evidence_snippet_count": len(snippets),
        "evidence_relation_counts": dict(relation_counts),
        "linked_direct_occurrence_molecules": sorted(
            direct_occurrence_linked
        ),
        "evidence_ids": [f"E{idx}" for idx in range(1, len(typed) + 1)],
        "candidate_evidence": [
            {
                "evidence_id": f"E{idx}",
                "relation": relation,
                "linked_molecules": linked,
            }
            for idx, (relation, _snippet, linked) in enumerate(
                typed,
                1,
            )
        ],
        "task_semantics": (
            "only_occurrence_support_can_ground_molecule_presence"
        ),
    }, direct_occurrence_linked


def format_ledger(rows: list[dict[str, Any]], ablation: str) -> str:
    if not rows:
        return "No structured candidates."
    if ablation == "no_ledger":
        return "\n".join(
            f"{idx}. {row.get('food') or row.get('molecule')}"
            for idx, row in enumerate(rows, 1)
        )
    formatted: list[str] = []
    for idx, row in enumerate(rows, 1):
        if row.get("molecule"):
            formatted.append(
                f"{idx}. molecule={row['molecule']} | "
                f"controller_score={row.get('score')} | "
                f"typed_evidence_linked={row.get('typed_evidence_linked')}"
            )
        else:
            formatted.append(
                f"{idx}. food={row.get('food')} | category={row.get('category')} | "
                f"score={row.get('score')} | source={row.get('source')} | "
                f"unimol_query_to_food={row.get('unimol_query_to_food')} | "
                f"unimol_food_to_query={row.get('unimol_food_to_query')} | "
                f"molecule_overlap={row.get('molecule_jaccard')}"
            )
    return "\n".join(formatted)


def build_mfp_fixed_candidates(
    ledger: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select three auditable concrete foods from the retrieval ledger.

    Macro categories are deliberately excluded from this object.  MFP category
    mapping belongs exclusively to evaluation, not hypothesis generation.
    """
    candidates: list[dict[str, Any]] = []
    seen_foods: set[str] = set()
    for item in ledger:
        food = str(item.get("food") or "").strip()
        food_key = normalize(food)
        if not food_key or food_key in seen_foods:
            continue
        candidates.append(
            {
                "candidate_id": f"C{len(candidates) + 1}",
                "food": food,
                "retrieval_rank": item.get("rank"),
                "retrieval_score": item.get("score"),
                "unimol_query_to_food": item.get("unimol_query_to_food"),
                "unimol_food_to_query": item.get("unimol_food_to_query"),
                "molecule_jaccard": item.get("molecule_jaccard"),
            }
        )
        seen_foods.add(food_key)
        if len(candidates) == 3:
            break
    if len(candidates) != 3:
        raise OptimizedAgentError(
            f"MFP structural controller requires three concrete food candidates; "
            f"received {len(candidates)}"
        )
    return candidates


def build_mfp_messages(
    row: dict[str, Any],
    ledger: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    evidence_text: str,
    demonstrations: list[dict[str, Any]],
    ablation: str,
    reviewer: bool,
    hypotheses: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    fixed_candidates = build_mfp_fixed_candidates(ledger, diagnostics)
    # Neither the candidate ledger nor diagnostics shown to the agents may
    # expose the evaluator's macro-category ontology.
    public_ledger = [
        {key: value for key, value in item.items() if key != "category"}
        for item in ledger
    ]
    public_diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if not key.startswith("category_")
    }
    demo_text = "\n\n".join(
        f"Example {idx}: molecules={demo['row'].get('molecules')}; "
        f"food={demo['row'].get('actual_food')}"
        for idx, demo in enumerate(demonstrations, 1)
    ) or "No demonstrations."
    sections = {
        "structure/statistical channel": (
            f"{format_ledger(public_ledger, ablation)}\nDiagnostics: "
            f"{json.dumps(public_diagnostics, ensure_ascii=False)}"
        ),
        "semantic/evidence channel": evidence_text,
        "training demonstrations": demo_text,
    }
    if ablation == "flat_fusion":
        context = "\n\n".join(sections.values())
    else:
        context = "\n\n".join(f"[{name}]\n{text}" for name, text in sections.items())
    if not reviewer:
        instruction = (
            "Audit exactly the three controller-provided concrete foods below, one hypothesis per "
            "candidate_id. Do not replace, rename, reorder, or invent candidates. Give priority "
            "to discriminative molecules and bidirectional UniMol set support, not generic "
            "molecules. Treat only [direct_occurrence] snippets as occurrence evidence; sensory "
            "similarity never proves that a molecule occurs in a food. Preserve conflicts between "
            "channels and do not treat any score as ground truth. Return JSON only as "
            '{"hypotheses":[{"predicted_food":"specific food name","support":["..."],'
            '"conflicts":["..."],"direct_evidence":["..."],"confidence":0.0}]}\n'
            f"Fixed candidates: {json.dumps(fixed_candidates, ensure_ascii=False)}"
        )
    else:
        context += "\n\n[scientist hypotheses]\n" + json.dumps(
            hypotheses or [], ensure_ascii=False
        )
        instruction = (
            "Independently audit the three fixed concrete foods against both channels. A sensory or "
            "odor resemblance is not occurrence evidence, and an unsupported scholarly-source "
            "inference must be rejected. Select exactly one of the three proposed concrete foods; "
            "do not invent a fourth food or output a macro category. Return JSON "
            "only as "
            '{"predicted_food":"specific food name","selected_hypothesis_index":1,"support":["..."],'
            '"conflicts":["..."],"rejected_claims":["..."]}'
        )
    prompt = (
        "FoodPuzzle MFP: infer one specific food name from the complete molecule set.\n"
        f"Input molecules: {json.dumps(row.get('molecules') or [], ensure_ascii=False)}\n\n"
        f"{context}\n\n{instruction}"
    )
    return [
        {"role": "system", "content": "You are a rigorous flavor scientist. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]


MPC_REVIEWER_MARGIN_THRESHOLD = 0.12
MPC_REVIEWER_DISAGREEMENT_THRESHOLD = 0.15


def mpc_requires_verifier(
    diagnostics: dict[str, Any],
    ablation: str,
) -> bool:
    if ablation == "no_reviewer":
        return False
    return bool(
        diagnostics.get("reviewer_gate_enabled")
        and diagnostics.get("admitted_retrieval_actions")
    )


def build_mpc_hypothesis_state(
    row: dict[str, Any],
    ledger: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    n = int(row.get("n") or 0)
    proposal_specs = [
        (
            "H1",
            "metric_aligned_exact_n_set_decoder",
            diagnostics.get("proposal_metric_aligned")
            or diagnostics.get("proposal_occurrence")
            or [],
        ),
        (
            "H2",
            "grouped_oof_retrieval_boundary_residual",
            diagnostics.get("proposal_retrieval_residual") or [],
        ),
        (
            "H3",
            str(
                diagnostics.get("structural_hypothesis_ranker")
                or "task_adapted_structural_boundary_residual"
            ),
            diagnostics.get("proposal_functional_residual") or [],
        ),
    ]
    ledger_by_key = {
        normalize(item.get("molecule")): item
        for item in ledger
        if normalize(item.get("molecule"))
    }
    pool_keys = stable_unique(
        [
            normalize(value)
            for _, _, values in proposal_specs
            for value in values
            if normalize(value) in ledger_by_key
        ]
    )
    pool = [ledger_by_key[key] for key in pool_keys]
    id_by_key = {
        normalize(item.get("molecule")): f"M{idx:03d}"
        for idx, item in enumerate(pool, 1)
        if normalize(item.get("molecule"))
    }
    candidates = [
        {
            "candidate_id": id_by_key[normalize(item["molecule"])],
            "molecule": str(item["molecule"]),
            "structural_fallback_rank": rank,
            "occurrence_rank": item.get("occurrence_rank"),
            "retrieval_rank": item.get("retrieval_rank"),
            "unimol_rank": item.get("unimol_rank"),
            "unimol_set_compatibility_rank": item.get(
                "unimol_set_compatibility_rank"
            ),
            "unimol_set_rank_lift": item.get(
                "unimol_set_rank_lift"
            ),
            "perceptual_rank": item.get("perceptual_rank"),
            "candidate_sources": item.get("candidate_sources") or [],
            "occurrence_score": item.get("occurrence_score"),
            "unimol_conditional_score": item.get(
                "unimol_conditional_score"
            ),
            "unimol_set_compatibility_score": item.get(
                "unimol_set_compatibility_score"
            ),
            "perceptual_residual_support": item.get(
                "perceptual_residual_support"
            ),
            "functional_group_demand_score": item.get(
                "functional_group_demand_score"
            ),
            "predicted_functional_groups": item.get(
                "predicted_functional_groups"
            )
            or [],
            "typed_evidence_linked": item.get("typed_evidence_linked"),
        }
        for rank, item in enumerate(pool, 1)
        if normalize(item.get("molecule")) in id_by_key
    ]
    if n <= 0 or len(candidates) < n:
        raise OptimizedAgentError(
            "invalid MPC hypothesis state"
        )
    by_id = {str(item["candidate_id"]): item for item in candidates}
    proposals: list[dict[str, Any]] = []
    for hypothesis_id, strategy, values in proposal_specs:
        selected_ids = [
            id_by_key[normalize(value)]
            for value in values
            if normalize(value) in id_by_key
        ]
        selected_ids = stable_unique(selected_ids)
        if len(selected_ids) != n:
            raise OptimizedAgentError(
                f"{hypothesis_id} is not an exact-n MPC proposal"
            )
        proposals.append(
            {
                "hypothesis_id": hypothesis_id,
                "strategy": strategy,
                "selected_candidate_ids": selected_ids,
            }
        )
    for hypothesis in proposals:
        hypothesis["predicted_molecules"] = [
            str(by_id[candidate_id]["molecule"])
            for candidate_id in hypothesis["selected_candidate_ids"]
        ]
    proposal_signatures = {
        proposal["hypothesis_id"]: tuple(
            sorted(proposal["selected_candidate_ids"])
        )
        for proposal in proposals
    }
    unique_signatures = sorted(set(proposal_signatures.values()))
    membership_by_id: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for proposal in proposals:
        signature = proposal_signatures[proposal["hypothesis_id"]]
        for candidate_id in proposal["selected_candidate_ids"]:
            membership_by_id[candidate_id].add(signature)
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate["independent_hypothesis_support"] = len(
            membership_by_id.get(candidate_id, set())
        )
        candidate["hypothesis_membership"] = [
            proposal["hypothesis_id"]
            for proposal in proposals
            if candidate_id in proposal["selected_candidate_ids"]
        ]
    pairwise_similarity: dict[str, float] = {}
    for left_index, left in enumerate(proposals):
        left_ids = set(left["selected_candidate_ids"])
        for right in proposals[left_index + 1 :]:
            right_ids = set(right["selected_candidate_ids"])
            pairwise_similarity[
                f"{left['hypothesis_id']}:{right['hypothesis_id']}"
            ] = round(
                len(left_ids & right_ids)
                / max(1, len(left_ids | right_ids)),
                8,
            )
    eligible_hypothesis_ids: list[str] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for proposal in proposals:
        hypothesis_id = str(proposal["hypothesis_id"])
        signature = proposal_signatures[hypothesis_id]
        if signature in seen_signatures:
            continue
        eligible_hypothesis_ids.append(hypothesis_id)
        seen_signatures.add(signature)
    max_fusion_swaps = min(
        5,
        max(
            int(diagnostics.get("retrieval_exchange_budget") or 0),
            int(diagnostics.get("functional_exchange_budget") or 0),
        ),
    )
    fusion_candidate_ids = stable_unique(
        [
            candidate_id
            for proposal in proposals
            if proposal["hypothesis_id"] in eligible_hypothesis_ids
            for candidate_id in proposal["selected_candidate_ids"]
        ]
    )
    return {
        "locked_core": [],
        "slots": n,
        "candidates": candidates,
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "proposals": proposals,
        "boundary_start": diagnostics.get("boundary_start"),
        "boundary_end": diagnostics.get("boundary_end"),
        "eligible_hypothesis_ids": eligible_hypothesis_ids,
        "unique_hypothesis_count": len(unique_signatures),
        "hypothesis_pairwise_jaccard": pairwise_similarity,
        "consensus_candidate_ids": [
            candidate_id
            for candidate_id, signatures in membership_by_id.items()
            if len(signatures) == len(unique_signatures)
        ],
        "disagreement_candidate_ids": [
            candidate_id
            for candidate_id, signatures in membership_by_id.items()
            if len(signatures) < len(unique_signatures)
        ],
        "fusion_candidate_ids": fusion_candidate_ids,
        "max_fusion_swaps": max_fusion_swaps,
        "predicted_group_demand": diagnostics.get(
            "predicted_group_demand"
        )
        or [],
    }


def compact_mpc_hypothesis_view(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Expose only the decision boundary to the language-model agents."""
    disagreement = set(
        state.get("disagreement_candidate_ids") or []
    )
    candidates = [
        {
            key: item.get(key)
            for key in (
                "candidate_id",
                "molecule",
                "occurrence_rank",
                "retrieval_rank",
                "unimol_set_compatibility_rank",
                "candidate_sources",
                "occurrence_score",
                "unimol_set_compatibility_score",
                "typed_evidence_linked",
                "hypothesis_membership",
            )
        }
        for item in state["candidates"]
        if str(item["candidate_id"]) in disagreement
    ]
    proposals = [
        {
            "hypothesis_id": proposal["hypothesis_id"],
            "disagreement_candidate_ids": [
                candidate_id
                for candidate_id in proposal[
                    "selected_candidate_ids"
                ]
                if candidate_id in disagreement
            ],
        }
        for proposal in state["proposals"]
    ]
    return {
        "exact_cardinality": state["slots"],
        "consensus_candidate_count": len(
            state.get("consensus_candidate_ids") or []
        ),
        "disagreement_candidate_count": len(disagreement),
        "disagreement_candidates": candidates,
        "proposal_differences": proposals,
        "pairwise_jaccard": state.get(
            "hypothesis_pairwise_jaccard"
        )
        or {},
    }


def build_mpc_scientist_messages(
    row: dict[str, Any],
    state: dict[str, Any],
    evidence_text: str,
) -> list[dict[str, str]]:
    slots = int(state["slots"])
    prompt = (
        "MPC Scientist conditional-set audit.\n"
        "The task is to complete a partially observed molecular set with exactly N molecules. "
        "Audit each complete proposal as a set: candidate relevance to the observation, molecular "
        "compatibility, redundancy among selected molecules, and evidence support. Neither raw "
        "structural similarity nor occurrence frequency alone is sufficient.\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known partial molecules: {json.dumps(row.get('partial_molecules') or [], ensure_ascii=False)}\n"
        f"Protected occurrence core is embedded in each proposal.\n"
        f"Exact output cardinality: {slots}\n\n"
        "[optional molecule-intrinsic descriptor demand]\n"
        f"{json.dumps(state['predicted_group_demand'], ensure_ascii=False)}\n\n"
        "[typed evidence]\n"
        "occurrence_support means reported presence; sensory_replication_support means odor or "
        "flavor resemblance; functional_role_support means contribution to an aroma/flavor; "
        "contradiction opposes a claim; ambiguous evidence must not be promoted to support.\n"
        f"{evidence_text}\n\n"
        "[fixed consensus and decision-boundary view]\n"
        f"{json.dumps(compact_mpc_hypothesis_view(state), ensure_ascii=False)}\n\n"
        "The proposal identifiers are neutral controller labels and do not reveal which expert "
        "generated them. Audit exactly H1, H2, and H3. Do not invent candidates or alter their selections. "
        "Separate occurrence support from sensory/functional analogy. Return JSON only as "
        '{"hypotheses":[{"hypothesis_id":"H1","support":["..."],'
        '"conflicts":["..."],"rejected_claims":["..."],"confidence":0.0}]}.'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous flavor-science verifier. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_mpc_scientist_output(
    content: str,
    evaluation: Any,
) -> dict[str, Any]:
    data = parse_json_object(content, evaluation)
    hypotheses = data.get("hypotheses") if data else None
    if not isinstance(hypotheses, list):
        raise OptimizedAgentError(
            "MPC Scientist did not return a hypotheses list"
        )
    parsed: list[dict[str, Any]] = []
    expected = {"H1", "H2", "H3"}
    for item in hypotheses:
        if not isinstance(item, dict):
            continue
        hypothesis_id = str(item.get("hypothesis_id") or "").strip().upper()
        if hypothesis_id in expected and hypothesis_id not in {
            row["hypothesis_id"] for row in parsed
        }:
            parsed.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "support": [
                        str(value)
                        for value in item.get("support") or []
                        if str(value).strip()
                    ],
                    "conflicts": [
                        str(value)
                        for value in item.get("conflicts") or []
                        if str(value).strip()
                    ],
                    "rejected_claims": [
                        str(value)
                        for value in item.get("rejected_claims") or []
                        if str(value).strip()
                    ],
                    "confidence": max(
                        0.0,
                        min(1.0, float(item.get("confidence") or 0.0)),
                    ),
                }
            )
    if {item["hypothesis_id"] for item in parsed} != expected:
        raise OptimizedAgentError(
            "MPC Scientist must audit H1, H2, and H3 exactly once"
        )
    return {"hypotheses": parsed}


def build_mpc_reviewer_messages(
    row: dict[str, Any],
    state: dict[str, Any],
    evidence_text: str,
    scientist_output: dict[str, Any],
) -> list[dict[str, str]]:
    prompt = (
        "MPC Reviewer counterfactual audit and base-hypothesis selection.\n"
        "Review the three Scientist-audited exact-N hypotheses and choose the most defensible "
        "complete base. For every molecule on which the hypotheses disagree, assess the "
        "counterfactual effect of removing it or replacing it with a competing candidate. Balance "
        "occurrence grounding, typed evidence, UniMol-conditioned set compatibility, and "
        "within-set redundancy. Raw structural similarity alone is not sufficient.\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known partial molecules: {json.dumps(row.get('partial_molecules') or [], ensure_ascii=False)}\n"
        f"Exact output cardinality represented by every hypothesis: {state['slots']}.\n"
        f"Eligible base hypotheses after training-only calibration: "
        f"{json.dumps(state['eligible_hypothesis_ids'], ensure_ascii=False)}.\n"
        f"Pairwise hypothesis Jaccard similarities: "
        f"{json.dumps(state['hypothesis_pairwise_jaccard'], ensure_ascii=False)}.\n\n"
        "[fixed consensus and decision-boundary view]\n"
        f"{json.dumps(compact_mpc_hypothesis_view(state), ensure_ascii=False)}\n\n"
        "[Scientist audit]\n"
        f"{json.dumps(scientist_output, ensure_ascii=False)}\n\n"
        "[typed evidence]\n"
        f"{evidence_text}\n\n"
        "Select one eligible neutral proposal ID, or ABSTAIN when the supplied evidence cannot "
        "justify overriding the default controller. Do not synthesize a new set, perform new swaps, "
        "or invent molecules. Correlated or duplicate hypotheses are not independent votes. "
        "ABSTAIN is the preferred output under unresolved conflict. "
        "Return JSON only as "
        '{"selected_hypothesis_id":"H2|ABSTAIN","strengths":["..."],'
        '"conflicts":["..."],"rejected_claims":["..."],'
        '"counterfactual_assessments":[{"candidate_id":"M001",'
        '"effect":"keep|remove|uncertain","reason":"..."}],'
        '"confidence":0.0}.'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative FoodPuzzle Reviewer. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_mpc_reviewer_output(
    content: str,
    state: dict[str, Any],
    evaluation: Any,
) -> dict[str, Any]:
    data = parse_json_object(content, evaluation)
    if not data:
        raise OptimizedAgentError(
            "MPC Reviewer did not return a JSON object"
        )
    proposal_by_id = {
        str(item["hypothesis_id"]): item
        for item in state["proposals"]
    }
    selected_hypothesis_id = str(
        data.get("selected_hypothesis_id") or ""
    ).strip().upper()
    abstained = selected_hypothesis_id == "ABSTAIN"
    if abstained:
        selected_hypothesis_id = "H1"
    if (
        selected_hypothesis_id not in proposal_by_id
        or selected_hypothesis_id
        not in set(state.get("eligible_hypothesis_ids") or ["H1"])
    ):
        raise OptimizedAgentError(
            "MPC Reviewer must select one controller-eligible hypothesis"
        )
    return {
        "selected_hypothesis_id": selected_hypothesis_id,
        "selected_candidate_ids": list(
            proposal_by_id[selected_hypothesis_id][
                "selected_candidate_ids"
            ]
        ),
        "strengths": [
            str(value)
            for value in data.get("strengths") or []
            if str(value).strip()
        ],
        "conflicts": [
            str(value)
            for value in data.get("conflicts") or []
            if str(value).strip()
        ],
        "rejected_claims": [
            str(value)
            for value in data.get("rejected_claims") or []
            if str(value).strip()
        ],
        "counterfactual_assessments": [
            {
                "candidate_id": str(
                    value.get("candidate_id") or ""
                ).strip(),
                "effect": str(
                    value.get("effect") or "uncertain"
                ).strip().lower(),
                "reason": str(
                    value.get("reason") or ""
                ).strip(),
            }
            for value in data.get("counterfactual_assessments") or []
            if isinstance(value, dict)
            and str(value.get("candidate_id") or "").strip()
            in set(state.get("disagreement_candidate_ids") or [])
            and str(value.get("effect") or "uncertain")
            .strip()
            .lower()
            in {"keep", "remove", "uncertain"}
        ],
        "confidence": max(
            0.0,
            min(1.0, float(data.get("confidence") or 0.0)),
        ),
        "abstained": abstained,
    }


def finalize_mpc_reviewer(
    state: dict[str, Any],
    reviewer_output: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    by_id = {
        str(item["candidate_id"]): item
        for item in state["candidates"]
    }
    selected_ids = list(reviewer_output["selected_candidate_ids"])
    values = [
        str(by_id[candidate_id]["molecule"])
        for candidate_id in selected_ids
        if candidate_id in by_id
    ]
    return values, {
        "mode": "selective_reviewer_local_proposal_audit",
        "selected_hypothesis_id": reviewer_output[
            "selected_hypothesis_id"
        ],
        "selected_candidate_ids": selected_ids,
        "strengths": reviewer_output["strengths"],
        "conflicts": reviewer_output["conflicts"],
        "rejected_claims": reviewer_output["rejected_claims"],
        "counterfactual_assessments": reviewer_output[
            "counterfactual_assessments"
        ],
        "confidence": reviewer_output["confidence"],
        "abstained": reviewer_output.get("abstained", False),
    }


def build_mpc_action_state(
    row: dict[str, Any],
    ledger: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    evidence_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build candidate-level typed-evidence action dossiers."""
    n = int(row.get("n") or 0)
    h1_values = stable_unique(
        diagnostics.get("proposal_metric_aligned")
        or diagnostics.get("proposal_occurrence")
        or [],
        {
            normalize(value)
            for value in row.get("partial_molecules") or []
        },
    )
    if len(h1_values) != n:
        raise OptimizedAgentError(
            "v13 base proposal is not an exact-N MPC set"
        )
    ledger_by_key = {
        normalize(item.get("molecule")): item
        for item in ledger
        if normalize(item.get("molecule"))
    }
    evidence_by_molecule: defaultdict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)
    valid_evidence_ids = set(
        evidence_metadata.get("evidence_ids") or []
    )
    for record in evidence_metadata.get("candidate_evidence") or []:
        if not isinstance(record, dict):
            continue
        evidence_id = str(record.get("evidence_id") or "")
        relation = str(record.get("relation") or "")
        if evidence_id not in valid_evidence_ids:
            continue
        for molecule in record.get("linked_molecules") or []:
            key = normalize(molecule)
            if key:
                evidence_by_molecule[key].append(
                    {
                        "evidence_id": evidence_id,
                        "relation": relation,
                    }
                )
    actions: list[dict[str, Any]] = []
    for index, raw_action in enumerate(
        diagnostics.get("admitted_retrieval_actions") or [],
        1,
    ):
        remove_key = normalize(raw_action.get("remove_key"))
        add_key = normalize(raw_action.get("add_key"))
        if (
            remove_key not in ledger_by_key
            or add_key not in ledger_by_key
            or remove_key
            not in {normalize(value) for value in h1_values}
            or add_key
            in {normalize(value) for value in h1_values}
        ):
            continue
        add_evidence = evidence_by_molecule.get(add_key, [])
        remove_evidence = evidence_by_molecule.get(
            remove_key,
            [],
        )
        direct_ids = sorted(
            {
                str(item["evidence_id"])
                for item in add_evidence
                if item["relation"] == "occurrence_support"
            }
        )
        add_item = ledger_by_key[add_key]
        remove_item = ledger_by_key[remove_key]
        actions.append(
            {
                "action_id": f"A{index:03d}",
                "claim_type": "occurrence",
                "remove_candidate": {
                    "molecule": str(remove_item["molecule"]),
                    "occurrence_rank": remove_item.get(
                        "occurrence_rank"
                    ),
                    "occurrence_score": remove_item.get(
                        "occurrence_score"
                    ),
                    "evidence": remove_evidence,
                },
                "add_candidate": {
                    "molecule": str(add_item["molecule"]),
                    "occurrence_rank": add_item.get(
                        "occurrence_rank"
                    ),
                    "occurrence_score": add_item.get(
                        "occurrence_score"
                    ),
                    "idf_retrieval_rank": add_item.get(
                        "idf_retrieval_rank"
                    ),
                    "idf_retrieved_support": add_item.get(
                        "idf_retrieved_support"
                    ),
                    "idf_retrieved_profile_count": add_item.get(
                        "idf_retrieved_profile_count"
                    ),
                    "legacy_retrieved_support": add_item.get(
                        "retrieved_profile_support"
                    ),
                    "evidence": add_evidence,
                },
                "utility_probability": round(
                    float(
                        raw_action.get(
                            "utility_probability"
                        )
                        or 0.0
                    ),
                    8,
                ),
                "h1_margin_cost": round(
                    float(
                        raw_action.get("h1_margin_cost")
                        or 0.0
                    ),
                    8,
                ),
                "independent_statistical_support_count": int(
                    raw_action.get(
                        "independent_statistical_support_count"
                    )
                    or 0
                ),
                "direct_occurrence_evidence_ids": direct_ids,
                "valid_evidence_ids": sorted(
                    {
                        str(item["evidence_id"])
                        for item in add_evidence + remove_evidence
                    }
                ),
                "provenance": [
                    "metric_aligned_base_boundary",
                    "idf_profile_retrieval",
                    "legacy_exact_molecule_gate_diagnostic_only",
                ],
            }
        )
    remove_keys = {
        normalize(action["remove_candidate"]["molecule"])
        for action in actions
    }
    locked_core = [
        value
        for value in h1_values
        if normalize(value) not in remove_keys
    ]
    return {
        "slots": n,
        "h1_values": h1_values,
        "locked_core": locked_core,
        "actions": actions,
        "action_ids": [
            str(action["action_id"]) for action in actions
        ],
        "utility_policy": diagnostics.get(
            "retrieval_action_policy"
        )
        or {},
        "candidate_catalog_independent_of_unimol": bool(
            diagnostics.get(
                "candidate_catalog_independent_of_unimol"
            )
        ),
    }


def build_mpc_action_scientist_messages(
    row: dict[str, Any],
    state: dict[str, Any],
    evidence_text: str,
) -> list[dict[str, str]]:
    prompt = (
        "MPC Scientist candidate-action audit.\n"
        "The task is to complete a partial molecular profile with exactly N "
        "molecules. The controller has locked the high-confidence H1 core and "
        "proposed only train-OOF-admitted boundary exchanges. Audit every "
        "action independently. A sensory resemblance or functional role is "
        "not evidence that a molecule occurs in the target food; conversely, "
        "occurrence evidence does not by itself establish functional "
        "replication. Match every cited relation to the action claim_type. Do not "
        "invent molecules, actions, or evidence IDs.\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known partial molecules: "
        f"{json.dumps(row.get('partial_molecules') or [], ensure_ascii=False)}\n"
        f"Exact N: {state['slots']}\n"
        f"Locked core count: {len(state['locked_core'])}\n\n"
        "[action dossiers]\n"
        f"{json.dumps(state['actions'], ensure_ascii=False)}\n\n"
        "[typed evidence]\n"
        f"{evidence_text}\n\n"
        "Return JSON only as "
        '{"actions":[{"action_id":"A001",'
        '"add_verdict":"SUPPORT|REFUTE|INSUFFICIENT",'
        '"remove_verdict":"SUPPORT|REFUTE|INSUFFICIENT",'
        '"cited_evidence_ids":["E1"],'
        '"reason":["..."]}]}.'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous flavor-science hypothesis "
                "generator and evidence auditor. Return JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_mpc_action_scientist_output(
    content: str,
    state: dict[str, Any],
    evaluation: Any,
) -> dict[str, Any]:
    data = parse_json_object(content, evaluation) or {}
    raw_actions = data.get("actions")
    if not isinstance(raw_actions, list):
        raise OptimizedAgentError(
            "MPC Scientist did not return action audits"
        )
    state_by_id = {
        str(action["action_id"]): action
        for action in state["actions"]
    }
    parsed: list[dict[str, Any]] = []
    allowed_verdicts = {
        "SUPPORT",
        "REFUTE",
        "INSUFFICIENT",
    }
    seen: set[str] = set()
    for raw in raw_actions:
        if not isinstance(raw, dict):
            continue
        action_id = str(raw.get("action_id") or "").upper()
        if action_id not in state_by_id or action_id in seen:
            continue
        add_verdict = str(
            raw.get("add_verdict") or "INSUFFICIENT"
        ).upper()
        remove_verdict = str(
            raw.get("remove_verdict") or "INSUFFICIENT"
        ).upper()
        if (
            add_verdict not in allowed_verdicts
            or remove_verdict not in allowed_verdicts
        ):
            continue
        valid_ids = set(
            state_by_id[action_id]["valid_evidence_ids"]
        )
        cited = sorted(
            {
                str(value)
                for value in raw.get("cited_evidence_ids") or []
                if str(value) in valid_ids
            }
        )
        parsed.append(
            {
                "action_id": action_id,
                "add_verdict": add_verdict,
                "remove_verdict": remove_verdict,
                "cited_evidence_ids": cited,
                "reason": [
                    str(value)
                    for value in raw.get("reason") or []
                    if str(value).strip()
                ],
            }
        )
        seen.add(action_id)
    if seen != set(state_by_id):
        raise OptimizedAgentError(
            "MPC Scientist must audit every admitted action once"
        )
    return {"actions": parsed}


def build_mpc_action_reviewer_messages(
    row: dict[str, Any],
    state: dict[str, Any],
    evidence_text: str,
    scientist_output: dict[str, Any],
) -> list[dict[str, str]]:
    prompt = (
        "MPC Reviewer independent local-action verification.\n"
        "Verify whether exactly one controller action is sufficiently "
        "grounded to override H1, or ABSTAIN. First inspect the raw dossier "
        "and typed evidence, then compare with the Scientist audit. Do not "
        "treat sensory similarity or functional role as molecule occurrence. "
        "Require the evidence relation to match the action claim_type. "
        "Do not invent or combine actions. Statistical retrieval support may "
        "justify a hypothesis only when the dossier contains at least two "
        "independent statistical signals; direct textual occurrence support "
        "must cite its evidence ID. ABSTAIN is the safe default.\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known partial molecules: "
        f"{json.dumps(row.get('partial_molecules') or [], ensure_ascii=False)}\n"
        f"Exact N: {state['slots']}\n\n"
        "[action dossiers]\n"
        f"{json.dumps(state['actions'], ensure_ascii=False)}\n\n"
        "[typed evidence]\n"
        f"{evidence_text}\n\n"
        "[Scientist audits]\n"
        f"{json.dumps(scientist_output, ensure_ascii=False)}\n\n"
        "Return JSON only as "
        '{"selected_action_id":"A001|ABSTAIN",'
        '"cited_evidence_ids":["E1"],'
        '"occurrence_grounded":true,'
        '"reason":["..."]}.'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative scientific Reviewer. "
                "Return JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_mpc_action_reviewer_output(
    content: str,
    state: dict[str, Any],
    scientist_output: dict[str, Any],
    evaluation: Any,
) -> dict[str, Any]:
    data = parse_json_object(content, evaluation) or {}
    selected = str(
        data.get("selected_action_id") or "ABSTAIN"
    ).upper()
    action_by_id = {
        str(action["action_id"]): action
        for action in state["actions"]
    }
    scientist_by_id = {
        str(action["action_id"]): action
        for action in scientist_output["actions"]
    }
    fallback_reasons: list[str] = []
    if selected == "ABSTAIN":
        return {
            "selected_action_id": "ABSTAIN",
            "abstained": True,
            "cited_evidence_ids": [],
            "occurrence_grounded": False,
            "reason": [
                str(value)
                for value in data.get("reason") or []
                if str(value).strip()
            ],
            "fallback_reasons": [],
        }
    if selected not in action_by_id:
        fallback_reasons.append("action_outside_oof_gate")
    action = action_by_id.get(selected)
    scientist_audit = scientist_by_id.get(selected)
    valid_ids = set(
        action.get("valid_evidence_ids") or []
        if action
        else []
    )
    returned_ids = {
        str(value)
        for value in data.get("cited_evidence_ids") or []
    }
    if not returned_ids <= valid_ids:
        fallback_reasons.append("invalid_evidence_reference")
    cited_ids = sorted(returned_ids & valid_ids)
    direct_ids = set(
        action.get("direct_occurrence_evidence_ids") or []
        if action
        else []
    )
    statistical_support = int(
        action.get(
            "independent_statistical_support_count",
            0,
        )
        if action
        else 0
    )
    direct_grounded = bool(direct_ids & set(cited_ids))
    statistical_grounded = statistical_support >= 2
    occurrence_grounded = bool(
        data.get("occurrence_grounded")
    )
    if not occurrence_grounded:
        fallback_reasons.append("reviewer_not_occurrence_grounded")
    if not (direct_grounded or statistical_grounded):
        fallback_reasons.append("insufficient_independent_support")
    if (
        scientist_audit is None
        or scientist_audit.get("add_verdict") == "REFUTE"
    ):
        fallback_reasons.append("scientist_refuted_or_missing_action")
    if fallback_reasons:
        selected = "ABSTAIN"
    return {
        "selected_action_id": selected,
        "abstained": selected == "ABSTAIN",
        "cited_evidence_ids": cited_ids,
        "occurrence_grounded": occurrence_grounded,
        "reason": [
            str(value)
            for value in data.get("reason") or []
            if str(value).strip()
        ],
        "fallback_reasons": fallback_reasons,
    }


def finalize_mpc_action_review(
    state: dict[str, Any],
    reviewer_output: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    values = list(state["h1_values"])
    selected_id = str(
        reviewer_output.get("selected_action_id")
        or "ABSTAIN"
    )
    action = next(
        (
            item
            for item in state["actions"]
            if item["action_id"] == selected_id
        ),
        None,
    )
    swap_count = 0
    if action is not None:
        remove_value = str(
            action["remove_candidate"]["molecule"]
        )
        add_value = str(action["add_candidate"]["molecule"])
        remove_key = normalize(remove_value)
        if (
            remove_key in {normalize(value) for value in values}
            and normalize(add_value)
            not in {normalize(value) for value in values}
        ):
            index = next(
                idx
                for idx, value in enumerate(values)
                if normalize(value) == remove_key
            )
            values[index] = add_value
            swap_count = 1
    exact_n = (
        len(values) == int(state["slots"])
        and len({normalize(value) for value in values})
        == int(state["slots"])
    )
    if not exact_n:
        raise OptimizedAgentError(
            "v12 deterministic action executor violated exact-N"
        )
    return values, {
        "mode": "evidence_grounded_local_action_review",
        **reviewer_output,
        "swap_count": swap_count,
        "exact_n_validated": True,
    }


def build_mpc_controller_actions(
    state: dict[str, Any],
    reviewer_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enumerate exact-N, calibrated exchanges; the LLM cannot invent one."""
    proposal_by_id = {
        str(item["hypothesis_id"]): item
        for item in state["proposals"]
    }
    base_hypothesis_id = str(
        reviewer_output["selected_hypothesis_id"]
    )
    base_ids = list(
        proposal_by_id[base_hypothesis_id]["selected_candidate_ids"]
    )
    candidate_by_id = {
        str(item["candidate_id"]): item
        for item in state["candidates"]
    }
    assessment_by_id = {
        str(item.get("candidate_id")): str(
            item.get("effect") or "uncertain"
        )
        for item in reviewer_output.get(
            "counterfactual_assessments"
        )
        or []
    }
    effect_priority = {
        "keep": 0,
        "uncertain": 1,
        "remove": 2,
    }
    actions: list[dict[str, Any]] = []
    maximum_swaps = int(state.get("max_fusion_swaps") or 0)
    for source_id in state.get("eligible_hypothesis_ids") or ["H1"]:
        if source_id == base_hypothesis_id:
            continue
        source_ids = list(
            proposal_by_id[source_id]["selected_candidate_ids"]
        )
        added = sorted(
            set(source_ids) - set(base_ids),
            key=lambda candidate_id: (
                effect_priority.get(
                    assessment_by_id.get(
                        candidate_id,
                        "uncertain",
                    ),
                    1,
                ),
                int(
                    candidate_by_id.get(candidate_id, {}).get(
                        "unimol_set_compatibility_rank"
                    )
                    or 10**9
                ),
                candidate_id,
            ),
        )
        removed = sorted(
            set(base_ids) - set(source_ids),
            key=lambda candidate_id: (
                -effect_priority.get(
                    assessment_by_id.get(
                        candidate_id,
                        "uncertain",
                    ),
                    1,
                ),
                -int(
                    candidate_by_id.get(candidate_id, {}).get(
                        "occurrence_rank"
                    )
                    or 0
                ),
                candidate_id,
            ),
        )
        for swap_count in range(
            1,
            min(maximum_swaps, len(added), len(removed)) + 1,
        ):
            remove_ids = removed[:swap_count]
            add_ids = added[:swap_count]
            selected_ids = [
                candidate_id
                for candidate_id in base_ids
                if candidate_id not in set(remove_ids)
            ] + add_ids
            if (
                len(selected_ids) != int(state["slots"])
                or len(set(selected_ids)) != len(selected_ids)
            ):
                continue
            actions.append(
                {
                    "action_id": f"A{len(actions) + 1:03d}",
                    "base_hypothesis_id": base_hypothesis_id,
                    "source_hypothesis_id": source_id,
                    "swap_count": swap_count,
                    "remove_candidate_ids": remove_ids,
                    "add_candidate_ids": add_ids,
                    "selected_candidate_ids": selected_ids,
                    "added_candidate_signals": [
                        {
                            "candidate_id": candidate_id,
                            "molecule": candidate_by_id.get(
                                candidate_id,
                                {},
                            ).get("molecule"),
                            "functional_group_demand_score": (
                                candidate_by_id.get(
                                    candidate_id,
                                    {},
                                ).get(
                                    "functional_group_demand_score"
                                )
                            ),
                            "predicted_functional_groups": (
                                candidate_by_id.get(
                                    candidate_id,
                                    {},
                                ).get(
                                    "predicted_functional_groups"
                                )
                                or []
                            ),
                            "retrieval_rank": candidate_by_id.get(
                                candidate_id,
                                {},
                            ).get("retrieval_rank"),
                            "typed_evidence_linked": candidate_by_id.get(
                                candidate_id,
                                {},
                            ).get("typed_evidence_linked"),
                            "reviewer_counterfactual_effect": (
                                assessment_by_id.get(
                                    candidate_id,
                                    "uncertain",
                                )
                            ),
                        }
                        for candidate_id in add_ids
                    ],
                }
            )
    return actions


def build_mpc_fusion_messages(
    row: dict[str, Any],
    state: dict[str, Any],
    evidence_text: str,
    scientist_output: dict[str, Any],
    reviewer_output: dict[str, Any],
) -> list[dict[str, str]]:
    actions = build_mpc_controller_actions(state, reviewer_output)
    prompt = (
        "MPC exact-cardinality Fusion Agent.\n"
        "The Reviewer selected a complete exact-N base. A deterministic controller has enumerated "
        "the only exchanges permitted by the learned conditional-set proposals and the "
        "Reviewer's candidate-level counterfactual audit. Select one action ID or NO_CHANGE. "
        "You may not invent, edit, or combine actions. Prefer NO_CHANGE unless the typed evidence "
        "and Scientist/Reviewer audits support a net improvement of the complete molecular set.\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known partial molecules: {json.dumps(row.get('partial_molecules') or [], ensure_ascii=False)}\n"
        f"Reviewer-selected base: {reviewer_output['selected_hypothesis_id']}.\n"
        f"Reviewer counterfactual assessments: "
        f"{json.dumps(reviewer_output.get('counterfactual_assessments') or [], ensure_ascii=False)}.\n\n"
        "[controller actions]\n"
        f"{json.dumps(actions, ensure_ascii=False)}\n\n"
        "[Scientist audit]\n"
        f"{json.dumps(scientist_output, ensure_ascii=False)}\n\n"
        "[Reviewer audit]\n"
        f"{json.dumps(reviewer_output, ensure_ascii=False)}\n\n"
        "[typed evidence]\n"
        f"{evidence_text}\n\n"
        "Return JSON only as "
        '{"base_hypothesis_id":"H1","selected_action_id":"A001",'
        '"reason":["..."],"confidence":0.0}. '
        'Use selected_action_id="NO_CHANGE" when no controlled exchange is justified.'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative exact-cardinality FoodPuzzle Fusion Agent. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def parse_mpc_fusion_output(
    content: str,
    state: dict[str, Any],
    reviewer_output: dict[str, Any],
    evaluation: Any,
) -> dict[str, Any]:
    proposal_by_id = {
        str(item["hypothesis_id"]): item
        for item in state["proposals"]
    }
    base_hypothesis_id = str(
        reviewer_output["selected_hypothesis_id"]
    )
    base_ids = list(
        proposal_by_id[base_hypothesis_id]["selected_candidate_ids"]
    )
    actions = build_mpc_controller_actions(state, reviewer_output)
    actions_by_id = {
        str(action["action_id"]): action for action in actions
    }
    data = parse_json_object(content, evaluation) or {}
    returned_base = str(
        data.get("base_hypothesis_id") or ""
    ).strip().upper()
    selected_action_id = str(
        data.get("selected_action_id") or ""
    ).strip().upper()
    fallback_reasons: list[str] = []
    if returned_base != base_hypothesis_id:
        fallback_reasons.append("base_hypothesis_mismatch")
    selected_action = None
    if selected_action_id == "NO_CHANGE":
        selected_ids = base_ids
        swap_count = 0
    elif selected_action_id in actions_by_id:
        selected_action = actions_by_id[selected_action_id]
        selected_ids = list(
            selected_action["selected_candidate_ids"]
        )
        swap_count = int(selected_action["swap_count"])
    else:
        fallback_reasons.append(
            "action_outside_controller_proposals"
        )
        selected_ids = base_ids
        swap_count = 0
    if fallback_reasons:
        selected_action = None
        selected_action_id = "NO_CHANGE"
        selected_ids = base_ids
        swap_count = 0
    return {
        "base_hypothesis_id": base_hypothesis_id,
        "selected_action_id": selected_action_id,
        "selected_candidate_ids": selected_ids,
        "swap_count": swap_count,
        "accepted_controller_action": selected_action,
        "available_controller_action_count": len(actions),
        "reason": [
            str(value)
            for value in data.get("reason") or []
            if str(value).strip()
        ],
        "confidence": max(
            0.0,
            min(1.0, float(data.get("confidence") or 0.0)),
        ),
        "fallback_to_reviewer_base": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
    }


def finalize_mpc_fusion(
    state: dict[str, Any],
    fusion_output: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    by_id = {
        str(item["candidate_id"]): item
        for item in state["candidates"]
    }
    selected_ids = list(fusion_output["selected_candidate_ids"])
    values = [
        str(by_id[candidate_id]["molecule"])
        for candidate_id in selected_ids
        if candidate_id in by_id
    ]
    exact_n = (
        len(values) == int(state["slots"])
        and len({normalize(value) for value in values}) == len(values)
    )
    if not exact_n:
        raise OptimizedAgentError(
            "deterministic MPC Fusion validation failed exact-N"
        )
    return values, {
        "mode": "reviewer_conditioned_constrained_hypothesis_fusion",
        **fusion_output,
        "exact_n_validated": True,
    }


def parse_json_object(content: str, evaluation: Any = None) -> dict[str, Any] | None:
    del evaluation
    text = str(content or "").strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(1).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_mfp_hypotheses(content: str, evaluation: Any) -> list[dict[str, Any]]:
    data = parse_json_object(content, evaluation)
    if not data or not isinstance(data.get("hypotheses"), list):
        return []
    output: list[dict[str, Any]] = []
    for item in data["hypotheses"]:
        if isinstance(item, dict) and str(item.get("predicted_food") or "").strip():
            output.append(item)
    return output[:3]


def validate_mfp_hypotheses(
    hypotheses: list[dict[str, Any]],
    fixed_candidates: list[dict[str, Any]],
) -> None:
    expected = {
        normalize(item.get("food"))
        for item in fixed_candidates
        if normalize(item.get("food"))
    }
    received = {
        normalize(item.get("predicted_food"))
        for item in hypotheses
        if normalize(item.get("predicted_food"))
    }
    if len(hypotheses) != 3 or received != expected:
        raise OptimizedAgentError(
            "MFP Scientist must audit each fixed structural candidate exactly once"
        )


def validate_mfp_reviewer_choice(
    reviewer_output: dict[str, Any] | None,
    hypotheses: list[dict[str, Any]],
) -> str:
    predicted = str((reviewer_output or {}).get("predicted_food") or "").strip()
    allowed = {
        normalize(item.get("predicted_food")): str(item.get("predicted_food")).strip()
        for item in hypotheses
        if normalize(item.get("predicted_food"))
    }
    if normalize(predicted) not in allowed:
        raise OptimizedAgentError(
            "MFP Reviewer must select one Scientist hypothesis without synthesizing a new food"
        )
    return allowed[normalize(predicted)]


def finalize_mfp(
    fixed_candidates: list[dict[str, Any]],
    reviewer_choice: str | None,
) -> tuple[str, dict[str, Any]]:
    """Return the Reviewer's selected concrete food, with retrieval fallback."""
    winner = fixed_candidates[0]
    selected = winner
    normalized_reviewer = normalize(reviewer_choice)
    reviewer_candidate = next(
        (
            item
            for item in fixed_candidates
            if normalize(item.get("food")) == normalized_reviewer
        ),
        None,
    )
    reviewer_override = bool(
        reviewer_candidate is not None and reviewer_candidate is not winner
    )
    if reviewer_candidate is not None:
        selected = reviewer_candidate
    predicted_food = str(selected.get("food") or "").strip()
    if not predicted_food:
        raise OptimizedAgentError(
            "MFP concrete-food controller produced an empty food"
        )
    return predicted_food, {
        "selection_method": "concrete_food_retrieval_with_reviewer_choice",
        "selected_food": predicted_food,
        "selected_retrieval_rank": selected.get("retrieval_rank"),
        "selected_retrieval_score": selected.get("retrieval_score"),
        "reviewer_choice": reviewer_choice,
        "reviewer_override": reviewer_override,
        "reviewer_agrees_with_controller": (
            normalize(reviewer_choice) == normalize(winner.get("food"))
            if reviewer_choice
            else None
        ),
        "fixed_candidates": fixed_candidates,
    }


def finalize_mpc(
    values: list[Any],
    row: dict[str, Any],
    ledger: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    n = int(row.get("n") or 0)
    excluded = {normalize(value) for value in row.get("partial_molecules") or []}
    model_values = stable_unique(values, excluded)
    reviewer_rank = {normalize(value): rank for rank, value in enumerate(model_values)}
    hypothesis_votes: Counter[str] = Counter()
    hypothesis_rank_sum: Counter[str] = Counter()
    for hypothesis in hypotheses or []:
        predicted = (
            hypothesis.get("predicted_molecules")
            if isinstance(hypothesis, dict)
            else None
        )
        for rank, value in enumerate(
            stable_unique(predicted if isinstance(predicted, list) else [], excluded)
        ):
            key = normalize(value)
            hypothesis_votes[key] += 1
            hypothesis_rank_sum[key] += rank

    ledger_by_key: dict[str, dict[str, Any]] = {}
    ledger_rank: dict[str, int] = {}
    for item in ledger:
        key = normalize(item.get("molecule"))
        if not key or key in excluded:
            continue
        ledger_by_key[key] = item
        ledger_rank[key] = len(ledger_rank)

    # Preserve the Reviewer's scientific ordering whenever it stays inside the
    # controller-defined search space.  UniMol/statistical scores initialize
    # that space and are used only to repair a short response; they no longer
    # overwrite every Reviewer decision.
    selected_keys = [
        normalize(value)
        for value in model_values
        if normalize(value) in ledger_by_key
    ][:n]
    reviewer_selected_count = len(selected_keys)
    selected_set = set(selected_keys)
    fallback_keys = sorted(
        (key for key in ledger_by_key if key not in selected_set),
        key=lambda key: (
            -hypothesis_votes[key],
            (
                hypothesis_rank_sum[key] / hypothesis_votes[key]
                if hypothesis_votes[key]
                else math.inf
            ),
            ledger_rank[key],
            key,
        ),
    )
    selected_keys.extend(fallback_keys[: max(0, n - len(selected_keys))])
    final = [
        str(ledger_by_key[key].get("molecule") or "").strip()
        for key in selected_keys[:n]
    ]
    final = stable_unique(final, excluded)[:n]
    ledger_fill_count = max(0, len(final) - reviewer_selected_count)
    return final, {
        "requested_n": n,
        "model_unique_count": len(model_values),
        "hypothesis_supported_candidate_count": len(hypothesis_votes),
        "controller_candidate_count": len(ledger_by_key),
        "reviewer_in_ledger_count": reviewer_selected_count,
        "rejected_open_world_count": sum(
            1 for key in reviewer_rank if key not in ledger_by_key
        ),
        "ledger_fill_count": ledger_fill_count,
        "selection_method": (
            "metric_aligned_controller_order_with_ledger_fallback"
        ),
        "final_count": len(final),
        "exact_n": len(final) == n,
    }


def run_check(args: argparse.Namespace) -> int:
    require_files(
        [
            (args.train, "train split"),
            (args.test, "test split"),
            (args.db, "FlavorDB"),
            (args.evidence, "evidence"),
        ]
    )
    train_rows = read_jsonl(Path(args.train))
    test_rows = read_jsonl(Path(args.test))
    validate_split(train_rows, test_rows)
    embeddings = None
    if args.task == "mfp" and args.ablation != "no_unimol":
        embeddings = EmbeddingStore(Path(args.unimol_embeddings))
    expected_field = "molecules" if args.task == "mfp" else "partial_molecules"
    total_molecules = 0
    mapped_molecules = 0
    for row in train_rows + test_rows:
        values = row.get(expected_field) or []
        total_molecules += len(values)
        if embeddings is not None:
            mapped_molecules += len(embeddings.available(values))

    baseline = load_sibling_module("scientific_agent.py", "foodpuzzle_scientific_agent_check")
    maximum_prompt_characters = 0
    minimum_ledger_size: int | None = None
    maximum_ledger_size = 0
    exact_n_count = 0
    evidence_linked_count = 0
    verifier_sample_count = 0
    verifier_prompt_count = 0
    adapter_enabled = False
    if args.task == "mfp":
        model: Any = MFPStructureModel(
            train_rows,
            embeddings,
            load_food_categories(Path(args.db)),
            args.ablation,
        )
        evidence = load_mfp_evidence(Path(args.evidence))
        retrieval_metadata = (
            baseline.load_retrieval_metadata(Path(args.icl_retrieval_metadata))
            if args.icl_retrieval_metadata
            and Path(args.icl_retrieval_metadata).is_file()
            else {}
        )
        train_by_id = {str(row["id"]): row for row in train_rows}
        for row in test_rows:
            ledger, diagnostics = model.rank(row, args.structure_top_k)
            evidence_text, _ = format_mfp_evidence(
                row,
                evidence if args.ablation != "no_evidence" else {},
                model.idf,
                anchor_score_map(diagnostics),
                args.evidence_molecule_limit,
                args.mfp_max_snippets_per_molecule,
            )
            demos = baseline.resolve_demos(
                str(row["id"]), retrieval_metadata, train_by_id, args.bm25_top_k
            )
            messages = build_mfp_messages(
                row,
                ledger,
                diagnostics,
                evidence_text,
                demos,
                args.ablation,
                reviewer=False,
            )
            maximum_prompt_characters = max(
                maximum_prompt_characters, len(messages[-1]["content"])
            )
            minimum_ledger_size = (
                len(ledger)
                if minimum_ledger_size is None
                else min(minimum_ledger_size, len(ledger))
            )
            maximum_ledger_size = max(maximum_ledger_size, len(ledger))
        adapter_enabled = model.sparse_classifier is not None
    else:
        model = MPCStructureModel(
            train_rows,
            embeddings,
            args.ablation,
            Path(args.db),
        )
        evidence = load_mpc_evidence(Path(args.evidence))
        retrieval = baseline.MPCBM25Index(train_rows)
        for row in test_rows:
            evidence_text, evidence_metadata, linked = format_mpc_evidence(
                row,
                evidence if args.ablation != "no_evidence" else {},
                args.mpc_max_evidence_snippets,
                list(model.display_names.values()),
            )
            evidence_linked_count += len(linked)
            candidate_count = max(
                100 if int(row.get("n") or 0) <= 5 else args.structure_top_k,
                min(args.max_structure_candidates, int(row.get("n") or 0) + 50),
            )
            ledger, diagnostics = model.rank(row, linked, candidate_count)
            retrieval.retrieve(row, args.bm25_top_k)
            if mpc_requires_verifier(diagnostics, args.ablation):
                action_state = build_mpc_action_state(
                    row,
                    ledger,
                    diagnostics,
                    evidence_metadata,
                )
                messages = build_mpc_action_scientist_messages(
                    row,
                    action_state,
                    evidence_text,
                )
                synthetic_scientist_output = {
                    "actions": [
                        {
                            "action_id": action["action_id"],
                            "add_verdict": "INSUFFICIENT",
                            "remove_verdict": "INSUFFICIENT",
                            "cited_evidence_ids": [],
                            "reason": [],
                        }
                        for action in action_state["actions"]
                    ]
                }
                reviewer_messages = build_mpc_action_reviewer_messages(
                    row,
                    action_state,
                    evidence_text,
                    synthetic_scientist_output,
                )
                maximum_prompt_characters = max(
                    maximum_prompt_characters,
                    len(messages[-1]["content"]),
                    len(reviewer_messages[-1]["content"]),
                )
                verifier_sample_count += 1
                verifier_prompt_count += 2
            structural_values = list(
                diagnostics.get("proposal_metric_aligned")
                or diagnostics.get("proposal_occurrence")
                or []
            )
            _, finalization = finalize_mpc(
                structural_values,
                row,
                ledger,
                [{"predicted_molecules": structural_values}],
            )
            exact_n_count += int(finalization["exact_n"])
            minimum_ledger_size = (
                len(ledger)
                if minimum_ledger_size is None
                else min(minimum_ledger_size, len(ledger))
            )
            maximum_ledger_size = max(maximum_ledger_size, len(ledger))
        adapter_enabled = model.ranker is not None
    print("OPTIMIZED_AGENT_CHECK_STATUS: PASS")
    print(f"method_version: {METHOD_VERSION}")
    print(f"task: {args.task}")
    print(f"ablation: {args.ablation}")
    print(f"train_samples: {len(train_rows)}")
    print(f"test_samples: {len(test_rows)}")
    print(f"molecule_occurrences: {total_molecules}")
    print(f"unimol_mapped_occurrences: {mapped_molecules}")
    if embeddings is not None:
        print(f"unimol_molecules: {len(embeddings.names)}")
        print(f"unimol_dimension: {embeddings.dimension}")
    print(f"task_adapter_enabled: {adapter_enabled}")
    print(f"minimum_candidate_ledger_size: {minimum_ledger_size or 0}")
    print(f"maximum_candidate_ledger_size: {maximum_ledger_size}")
    print(f"maximum_scientist_prompt_characters: {maximum_prompt_characters}")
    if args.task == "mpc":
        print(f"deterministic_exact_n_samples: {exact_n_count}")
        print(f"evidence_linked_candidate_occurrences: {evidence_linked_count}")
        print(f"uncertainty_gated_verifier_samples: {verifier_sample_count}")
        print(f"planned_verifier_api_calls: {verifier_prompt_count}")
        print(
            "residual_policy: "
            + json.dumps(model.residual_policy, sort_keys=True)
        )
        print(
            "residual_calibration: "
            + json.dumps(model.residual_calibration, sort_keys=True)
        )
        print(
            "retrieval_action_policy: "
            + json.dumps(model.retrieval_action_policy, sort_keys=True)
        )
        print(
            "retrieval_action_calibration: "
            + json.dumps(
                model.retrieval_action_calibration,
                sort_keys=True,
            )
        )
        print(
            "metric_group_policy: "
            + json.dumps(
                model.metric_group_policy,
                sort_keys=True,
            )
        )
        print(
            "metric_group_calibration: "
            + json.dumps(
                model.metric_group_calibration,
                sort_keys=True,
            )
        )
        print(
            "dual_gate_policy: "
            + json.dumps(model.dual_gate_policy, sort_keys=True)
        )
    return 0


def run_agent(args: argparse.Namespace) -> int:
    require_files(
        [
            (args.train, "train split"),
            (args.test, "test split"),
            (args.db, "FlavorDB"),
            (args.evidence, "evidence"),
        ]
    )
    validate_output_paths(args)
    if not args.use_llm:
        raise OptimizedAgentError("--use-llm is required for formal generation")

    baseline = load_sibling_module("scientific_agent.py", "foodpuzzle_scientific_agent")
    evaluation = load_sibling_module("evaluation.py", "foodpuzzle_evaluation")
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(llm_config)

    train_rows = read_jsonl(Path(args.train))
    test_rows = read_jsonl(Path(args.test))
    validate_split(train_rows, test_rows)
    embeddings = (
        EmbeddingStore(Path(args.unimol_embeddings))
        if args.task == "mfp" and args.ablation != "no_unimol"
        else None
    )
    evidence_done = existing_ids(Path(args.evidence_metadata)) if args.resume else set()
    retrieval_done = existing_ids(Path(args.retrieval_metadata)) if args.resume else set()
    hypotheses_done = (
        successful_hypothesis_ids(Path(args.hypotheses_metadata), args.task)
        if args.resume
        else set()
    )
    prediction_done = successful_ids(Path(args.output), args.task) if args.resume else set()
    completed = prediction_done & hypotheses_done

    if args.task == "mfp":
        categories = load_food_categories(Path(args.db))
        structure_model: Any = MFPStructureModel(
            train_rows, embeddings, categories, args.ablation
        )
        evidence_map = load_mfp_evidence(Path(args.evidence))
        retrieval_source = baseline.load_retrieval_metadata(Path(args.icl_retrieval_metadata))
        train_by_id = {str(row["id"]): row for row in train_rows}
    else:
        structure_model = MPCStructureModel(
            train_rows,
            embeddings,
            args.ablation,
            Path(args.db),
        )
        evidence_map = load_mpc_evidence(Path(args.evidence))
        retrieval_source = baseline.MPCBM25Index(train_rows)
        train_by_id = {}

    success = 0
    failures = 0
    skipped = 0
    for row in test_rows:
        row_id = str(row["id"])
        if row_id in completed:
            skipped += 1
            continue
        error: str | None = None
        hypotheses: list[dict[str, Any]] = []
        reviewer_output: dict[str, Any] | None = None
        finalization: dict[str, Any] = {}
        evidence_metadata: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        ledger: list[dict[str, Any]] = []
        demos: list[dict[str, Any]] = []
        try:
            if args.task == "mfp":
                ledger, diagnostics = structure_model.rank(row, args.structure_top_k)
                fixed_candidates = build_mfp_fixed_candidates(ledger, diagnostics)
                evidence_text, evidence_metadata = format_mfp_evidence(
                    row,
                    evidence_map if args.ablation != "no_evidence" else {},
                    structure_model.idf,
                    anchor_score_map(diagnostics),
                    args.evidence_molecule_limit,
                    args.mfp_max_snippets_per_molecule,
                )
                demos = baseline.resolve_demos(
                    row_id,
                    retrieval_source,
                    train_by_id,
                    args.bm25_top_k,
                )
                scientist_content = evaluation.call_chat_completion(
                    build_mfp_messages(
                        row, ledger, diagnostics, evidence_text, demos,
                        args.ablation, reviewer=False
                    ),
                    llm_config,
                )
                hypotheses = parse_mfp_hypotheses(scientist_content, evaluation)
                scientist_fallback = False
                try:
                    validate_mfp_hypotheses(hypotheses, fixed_candidates)
                except OptimizedAgentError:
                    # A malformed response must not turn formal automation
                    # into an unbounded retry loop.  Preserve the controller's
                    # concrete candidate space without inventing evidence; the
                    # Reviewer still makes the food selection.
                    hypotheses = [
                        {
                            "predicted_food": str(candidate["food"]),
                            "support": [],
                            "conflicts": [
                                "Scientist response was not valid structured JSON."
                            ],
                            "direct_evidence": [],
                            "confidence": 0.0,
                            "controller_fallback": True,
                        }
                        for candidate in fixed_candidates
                    ]
                    scientist_fallback = True
                reviewer_choice: str | None = None
                reviewer_fallback = False
                if args.ablation == "no_reviewer":
                    reviewer_output = None
                else:
                    reviewer_content = evaluation.call_chat_completion(
                        build_mfp_messages(
                            row, ledger, diagnostics, evidence_text, demos,
                            args.ablation, reviewer=True, hypotheses=hypotheses
                        ),
                        llm_config,
                    )
                    reviewer_output = parse_json_object(reviewer_content, evaluation)
                    try:
                        reviewer_choice = validate_mfp_reviewer_choice(
                            reviewer_output, hypotheses
                        )
                    except OptimizedAgentError:
                        reviewer_choice = str(fixed_candidates[0]["food"])
                        reviewer_output = {
                            "predicted_food": reviewer_choice,
                            "selected_hypothesis_index": 1,
                            "support": [],
                            "conflicts": [
                                "Reviewer response was outside the fixed candidate space."
                            ],
                            "rejected_claims": [],
                            "controller_fallback": True,
                        }
                        reviewer_fallback = True
                predicted_food, finalization = finalize_mfp(
                    fixed_candidates, reviewer_choice
                )
                finalization.update(
                    {
                        "scientist_parse_fallback": scientist_fallback,
                        "reviewer_parse_fallback": reviewer_fallback,
                    }
                )
                if not predicted_food:
                    raise OptimizedAgentError("empty MFP prediction")
                prediction_row = {"id": row_id, "predicted_food": predicted_food}
            else:
                evidence_text, evidence_metadata, evidence_molecules = format_mpc_evidence(
                    row,
                    evidence_map if args.ablation != "no_evidence" else {},
                    args.mpc_max_evidence_snippets,
                    list(structure_model.display_names.values()),
                )
                candidate_count = max(
                    100 if int(row.get("n") or 0) <= 5 else args.structure_top_k,
                    min(args.max_structure_candidates, int(row.get("n") or 0) + 50),
                )
                ledger, diagnostics = structure_model.rank(
                    row, evidence_molecules, candidate_count
                )
                demos = retrieval_source.retrieve(row, args.bm25_top_k)
                verifier_required = mpc_requires_verifier(
                    diagnostics,
                    args.ablation,
                )
                if verifier_required:
                    action_state = build_mpc_action_state(
                        row,
                        ledger,
                        diagnostics,
                        evidence_metadata,
                    )
                    scientist_content = evaluation.call_chat_completion(
                        build_mpc_action_scientist_messages(
                            row,
                            action_state,
                            evidence_text,
                        ),
                        llm_config,
                    )
                    scientist_output = parse_mpc_action_scientist_output(
                        scientist_content,
                        action_state,
                        evaluation,
                    )
                    audit_by_id = {
                        str(item["action_id"]): item
                        for item in scientist_output["actions"]
                    }
                    hypotheses = [{
                        "hypothesis_id": "H1",
                        "method": "occurrence_top_n",
                        "predicted_molecules": list(
                            action_state["h1_values"]
                        ),
                    }] + [
                        {
                            "hypothesis_id": action["action_id"],
                            "method": "oof_admitted_local_exchange",
                            "action": action,
                            "scientist_audit": audit_by_id.get(
                                str(action["action_id"])
                            ),
                            "predicted_molecules": finalize_mpc_action_review(
                                action_state,
                                {
                                    "selected_action_id": action[
                                        "action_id"
                                    ]
                                },
                            )[0],
                        }
                        for action in action_state["actions"]
                    ]
                    reviewer_content = evaluation.call_chat_completion(
                        build_mpc_action_reviewer_messages(
                            row,
                            action_state,
                            evidence_text,
                            scientist_output,
                        ),
                        llm_config,
                    )
                    parsed_reviewer = parse_mpc_action_reviewer_output(
                        reviewer_content,
                        action_state,
                        scientist_output,
                        evaluation,
                    )
                    raw_values, reviewer_metadata = finalize_mpc_action_review(
                        action_state,
                        parsed_reviewer,
                    )
                    reviewer_output = {
                        "predicted_molecules": raw_values,
                        "scientist_output": scientist_output,
                        **reviewer_metadata,
                    }
                else:
                    raw_values = list(
                        diagnostics.get("proposal_metric_aligned")
                        or diagnostics.get("proposal_occurrence")
                        or []
                    )
                    hypotheses = [
                        {
                            "mode": "metric_aligned_safe_base",
                            "predicted_molecules": raw_values,
                        }
                    ]
                    reviewer_output = {
                        "mode": "typed_evidence_auditor_not_admitted",
                        "predicted_molecules": raw_values,
                    }
                predicted_molecules, finalization = finalize_mpc(
                    raw_values, row, ledger, hypotheses
                )
                finalization.update(
                    {
                        "verifier_required": verifier_required,
                        "verifier_passes": (
                            1 if verifier_required else 0
                        ),
                        "scientist_calls": 1 if verifier_required else 0,
                        "reviewer_calls": 1 if verifier_required else 0,
                        "fusion_calls": 0,
                        "base_cutoff_margin": diagnostics.get(
                            "base_cutoff_margin"
                        ),
                        "boundary_swap_count": diagnostics.get(
                            "boundary_swap_count"
                        ),
                    }
                )
                if not predicted_molecules:
                    raise OptimizedAgentError("empty MPC prediction")
                prediction_row = {
                    "id": row_id,
                    "task": row.get("task", "MPC"),
                    "target_food": row.get("target_food"),
                    "partial_molecules": row.get("partial_molecules") or [],
                    "n": row.get("n"),
                    "predicted_molecules": predicted_molecules,
                }
            success += 1
        except Exception as exc:
            message = str(exc)
            provider_http_error = isinstance(
                exc, evaluation.ChatCompletionHTTPError
            )
            provider_network_error = (
                isinstance(exc, evaluation.EvaluationError)
                and (
                    " API request failed:" in message
                    or " API request failed after retries" in message
                )
            )
            if provider_http_error or provider_network_error:
                raise OptimizedAgentError(
                    "provider request failed; stop without recording a sample "
                    f"failure and resume after the provider recovers: {message}"
                ) from exc
            error = f"{type(exc).__name__}: {message}"
            failures += 1
            prediction_row = {"id": row_id}
            if args.task == "mfp":
                prediction_row["predicted_food"] = ""
            else:
                prediction_row.update(
                    {
                        "task": row.get("task", "MPC"),
                        "target_food": row.get("target_food"),
                        "partial_molecules": row.get("partial_molecules") or [],
                        "n": row.get("n"),
                        "predicted_molecules": [],
                    }
                )
            prediction_row["error"] = error

        if row_id not in evidence_done:
            append_jsonl(
                Path(args.evidence_metadata),
                {"id": row_id, "ablation": args.ablation, **evidence_metadata},
            )
        if row_id not in retrieval_done:
            append_jsonl(
                Path(args.retrieval_metadata),
                {
                    "id": row_id,
                    "ablation": args.ablation,
                    "retrieved": [
                        {
                            "id": item.get("id"),
                            "rank": item.get("rank"),
                            "score": item.get("score"),
                            "food": item.get("actual_food") or item.get("target_food"),
                        }
                        for item in demos
                    ],
                },
            )
        if row_id not in hypotheses_done:
            append_jsonl(
                Path(args.hypotheses_metadata),
                {
                    "id": row_id,
                    "ablation": args.ablation,
                    "structure_diagnostics": diagnostics,
                    "candidate_ledger": ledger,
                    "hypotheses": hypotheses,
                    "reviewer_output": reviewer_output,
                    "finalization": finalization,
                    "error": error,
                },
            )
        append_jsonl(Path(args.output), prediction_row)

    print("OPTIMIZED_AGENT_STATUS: PASS")
    print(f"method_version: {METHOD_VERSION}")
    print(f"task: {args.task}")
    print(f"ablation: {args.ablation}")
    print(f"total: {len(test_rows)}")
    print(f"existing_predictions: {len(completed)}")
    print(f"new_success: {success}")
    print(f"failures: {failures}")
    print(f"skipped: {skipped}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task-specific FoodPuzzle Agent: UniMol-grounded MFP and "
            "metric-aligned exact-N MPC"
        )
    )
    parser.add_argument("--task", choices=["mfp", "mpc"])
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--db", default="data/raw/flavordb.db")
    parser.add_argument("--evidence")
    parser.add_argument("--icl-retrieval-metadata")
    parser.add_argument("--output")
    parser.add_argument("--evidence-metadata")
    parser.add_argument("--retrieval-metadata")
    parser.add_argument("--hypotheses-metadata")
    parser.add_argument(
        "--unimol-input-csv",
        default="data/structure/unimol/inputs/unimol_molecules.csv",
    )
    parser.add_argument(
        "--unimol-embeddings",
        default="data/structure/unimol/unimol_embeddings.npz",
    )
    parser.add_argument("--prepare-unimol", action="store_true")
    parser.add_argument("--unimol-batch-size", type=int, default=32)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--structure-top-k", type=int, default=30)
    parser.add_argument("--max-structure-candidates", type=int, default=300)
    parser.add_argument("--bm25-top-k", type=int, default=3)
    parser.add_argument("--evidence-molecule-limit", type=int, default=8)
    parser.add_argument("--mfp-max-snippets-per-molecule", type=int, default=3)
    parser.add_argument("--mpc-max-evidence-snippets", type=int, default=10)
    parser.add_argument("--llm-provider", choices=["deepseek"], default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.prepare_unimol:
        return
    required = {
        "--task": args.task,
        "--train": args.train,
        "--test": args.test,
        "--evidence": args.evidence,
    }
    if not args.check_only:
        required.update(
            {
                "--output": args.output,
                "--evidence-metadata": args.evidence_metadata,
                "--retrieval-metadata": args.retrieval_metadata,
                "--hypotheses-metadata": args.hypotheses_metadata,
            }
        )
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise OptimizedAgentError(f"missing required arguments: {', '.join(missing)}")
    if args.task == "mfp" and not args.icl_retrieval_metadata and not args.check_only:
        raise OptimizedAgentError("--task mfp requires --icl-retrieval-metadata")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_cli(args)
        if args.prepare_unimol:
            return prepare_unimol_embeddings(args)
        if args.check_only:
            return run_check(args)
        return run_agent(args)
    except OptimizedAgentError as exc:
        print("OPTIMIZED_AGENT_STATUS: FAIL")
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
