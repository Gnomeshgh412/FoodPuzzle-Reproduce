#!/usr/bin/env python3
"""Train-only MPC candidate-recall and oracle audit.

The script intentionally does not call an LLM, read the released MPC test
labels, use the functional-group evaluation cache, or write result artifacts.
It evaluates candidate generators with deterministic food-grouped OOF over the
reconstructed MPC training split and prints the audit to stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "results/splits/mpc/train.jsonl"
DEFAULT_DB = REPO_ROOT / "data/raw/flavordb.db"
DEFAULT_EVIDENCE = (
    REPO_ROOT / "data/collected_evidences/collected_evidences_task2.pkl"
)
DEFAULT_AGENT = REPO_ROOT / "code/Only-Deepseek/optimized_agent.py"
CHANNELS = (
    "h1",
    "current_retrieval",
    "idf_retrieval",
    "cooccurrence",
    "direct_evidence",
    "rrf_statistical",
    "rrf_all",
)
CUTOFFS = ("n", "n+10", "n+30", "2n")


def normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def stable_unique(values: Iterable[str], excluded: set[str] | None = None) -> list[str]:
    blocked = excluded or set()
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = normalize(value)
        if not key or key in blocked or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_agent_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "foodpuzzle_candidate_audit_agent",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load optimized agent from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_groups(value: Any) -> frozenset[str]:
    return frozenset(
        normalize(item)
        for item in str(value or "").split("@")
        if normalize(item)
    )


def load_catalog(
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, frozenset[str]]]:
    display: dict[str, str] = {}
    groups: dict[str, frozenset[str]] = {}
    connection = sqlite3.connect(path)
    try:
        for table in ("molecules", "molecules_all"):
            query = (
                f"SELECT common_name, functional_groups FROM {table} "
                "WHERE common_name IS NOT NULL AND TRIM(common_name) <> ''"
            )
            for name, functional_groups in connection.execute(query):
                key = normalize(name)
                if not key:
                    continue
                display.setdefault(key, str(name))
                parsed = parse_groups(functional_groups)
                if parsed:
                    groups[key] = frozenset(set(groups.get(key, ())) | set(parsed))
    finally:
        connection.close()
    for row in rows:
        for name in list(row.get("partial_molecules") or []) + list(
            row.get("missing_molecules") or []
        ):
            key = normalize(name)
            if key:
                display.setdefault(key, str(name))
    return display, groups


def load_evidence(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("MPC evidence must be a dictionary")
    output: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            output[normalize(key)] = [str(item) for item in value]
        elif value is not None:
            output[normalize(key)] = [str(value)]
    return output


def cutoff_value(label: str, n: int, maximum: int) -> int:
    if label == "n":
        value = n
    elif label == "n+10":
        value = n + 10
    elif label == "n+30":
        value = n + 30
    elif label == "2n":
        value = 2 * n
    else:
        raise ValueError(f"Unknown cutoff {label}")
    return min(maximum, max(0, value))


def f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[float, list[str]]],
    excluded: set[str],
    constant: int = 60,
) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for weight, values in ranked_lists:
        for rank, key in enumerate(stable_unique(values, excluded), 1):
            scores[key] += weight / (constant + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
    return sorted(
        scores,
        key=lambda key: (-scores[key], best_rank[key], key),
    )


def food_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize(value).replace("-", " ").split()
        if token
    }


def idf_retrieval(
    query: dict[str, Any],
    fit_rows: list[dict[str, Any]],
    fit_profiles: list[set[str]],
    top_k: int = 15,
) -> tuple[list[str], dict[str, float]]:
    partial = {
        normalize(value)
        for value in query.get("partial_molecules") or []
        if normalize(value)
    }
    query_food_tokens = food_tokens(query.get("target_food"))
    molecule_df: Counter[str] = Counter()
    token_df: Counter[str] = Counter()
    for row, profile in zip(fit_rows, fit_profiles):
        molecule_df.update(profile)
        token_df.update(food_tokens(row.get("target_food")))
    row_count = max(1, len(fit_rows))
    molecule_idf = {
        key: math.log((row_count + 1) / (count + 1)) + 1.0
        for key, count in molecule_df.items()
    }
    token_idf = {
        key: math.log((row_count + 1) / (count + 1)) + 1.0
        for key, count in token_df.items()
    }
    partial_weight = sum(molecule_idf.get(key, 1.0) for key in partial)
    query_token_weight = sum(
        token_idf.get(token, 1.0) for token in query_food_tokens
    )
    neighbours: list[tuple[float, int]] = []
    for index, (row, profile) in enumerate(zip(fit_rows, fit_profiles)):
        matched_profile_weight = sum(
            molecule_idf.get(key, 1.0) for key in partial & profile
        )
        profile_containment = (
            matched_profile_weight / max(partial_weight, 1e-12)
        )
        neighbour_tokens = food_tokens(row.get("target_food"))
        matched_token_weight = sum(
            token_idf.get(token, 1.0)
            for token in query_food_tokens & neighbour_tokens
        )
        food_containment = (
            matched_token_weight / max(query_token_weight, 1e-12)
            if query_food_tokens
            else 0.0
        )
        score = 0.75 * profile_containment + 0.25 * food_containment
        neighbours.append((score, index))
    neighbours.sort(key=lambda item: (-item[0], item[1]))

    support: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for rank, (score, index) in enumerate(neighbours[:top_k], 1):
        weight = max(0.0, score) / math.log2(rank + 1.0)
        for candidate in fit_profiles[index] - partial:
            support[candidate] += weight
            counts[candidate] += 1
    ordered = sorted(
        support,
        key=lambda key: (
            -support[key],
            -counts[key],
            -molecule_idf.get(key, 1.0),
            key,
        ),
    )
    return ordered, dict(support)


def cooccurrence_ranking(
    query: dict[str, Any],
    frequency: Counter[str],
    cooccurrence: dict[str, Counter[str]],
    universe: Iterable[str],
) -> list[str]:
    partial = {
        normalize(value)
        for value in query.get("partial_molecules") or []
        if normalize(value)
    }
    scored: list[tuple[float, float, str]] = []
    for candidate in universe:
        if candidate in partial:
            continue
        values = [
            cooccurrence.get(candidate, {}).get(observed, 0)
            / max(1, frequency[observed])
            for observed in partial
        ]
        mean_value = sum(values) / len(values) if values else 0.0
        max_value = max(values) if values else 0.0
        score = 0.65 * mean_value + 0.35 * max_value
        scored.append((score, math.log1p(frequency[candidate]), candidate))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [candidate for _, _, candidate in scored]


def direct_evidence_ranking(
    query: dict[str, Any],
    evidence: dict[str, list[str]],
    catalog_keys: list[str],
    occurrence_cues: tuple[str, ...],
) -> list[str]:
    snippets = evidence.get(normalize(query.get("target_food")), [])
    occurrence_snippets = [
        normalize(snippet)
        for snippet in snippets
        if any(cue in normalize(snippet) for cue in occurrence_cues)
    ]
    if not occurrence_snippets:
        return []
    joined = f" {' '.join(occurrence_snippets)} "
    linked = [
        key
        for key in catalog_keys
        if len(key) >= 3 and f" {key} " in joined
    ]
    return sorted(
        linked,
        key=lambda key: (
            min(
                (
                    text.find(f" {key} ")
                    for text in occurrence_snippets
                    if f" {key} " in f" {text} "
                ),
                default=10**9,
            ),
            key,
        ),
    )


def greedy_group_oracle(
    candidate_pool: list[str],
    n: int,
    gold_molecules: set[str],
    group_map: dict[str, frozenset[str]],
) -> tuple[float | None, float]:
    mapped_gold = [key for key in gold_molecules if key in group_map]
    mapping_coverage = len(mapped_gold) / max(1, len(gold_molecules))
    gold_groups = set().union(
        *(set(group_map[key]) for key in mapped_gold)
    ) if mapped_gold else set()
    if not gold_groups:
        return None, mapping_coverage
    selected: list[str] = []
    selected_groups: set[str] = set()
    remaining = stable_unique(candidate_pool)
    for _ in range(min(n, len(remaining))):
        best_key: str | None = None
        best_value: tuple[float, int, int, str] | None = None
        for key in remaining:
            groups = set(group_map.get(key, ()))
            proposed_groups = selected_groups | groups
            value = (
                f1(proposed_groups, gold_groups),
                len((groups - selected_groups) & gold_groups),
                -len((groups - selected_groups) - gold_groups),
                key,
            )
            if best_value is None or value > best_value:
                best_value = value
                best_key = key
        if best_key is None:
            break
        selected.append(best_key)
        selected_groups.update(group_map.get(best_key, ()))
        remaining.remove(best_key)
    return f1(selected_groups, gold_groups), mapping_coverage


@dataclass
class Metric:
    recalls: list[float] = field(default_factory=list)
    hits: int = 0
    gold: int = 0
    fold_recalls: defaultdict[int, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, fold: int, selected: set[str], gold: set[str]) -> None:
        hit_count = len(selected & gold)
        recall = hit_count / max(1, len(gold))
        self.recalls.append(recall)
        self.hits += hit_count
        self.gold += len(gold)
        self.fold_recalls[fold].append(recall)

    def summary(self) -> dict[str, Any]:
        fold_means = {
            str(fold): sum(values) / len(values)
            for fold, values in sorted(self.fold_recalls.items())
            if values
        }
        return {
            "macro_recall": sum(self.recalls) / max(1, len(self.recalls)),
            "micro_recall": self.hits / max(1, self.gold),
            "fold_macro_recall": fold_means,
            "positive_fold_count_vs_zero": sum(
                value > 0.0 for value in fold_means.values()
            ),
        }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = read_jsonl(args.train)
    catalog_display, group_map = load_catalog(args.db, train_rows)
    evidence = load_evidence(args.evidence)
    agent = load_agent_module(args.agent)

    groups = sorted(
        {
            normalize(row.get("target_food")) or f"id:{row.get('id')}"
            for row in train_rows
        }
    )
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    fold_by_group = {
        group: index % args.folds for index, group in enumerate(groups)
    }

    metrics: dict[str, dict[str, Metric]] = {
        channel: {cutoff: Metric() for cutoff in CUTOFFS}
        for channel in CHANNELS
    }
    union_metric = Metric()
    union_recalls: list[float] = []
    union_gains_over_h1_2n: list[float] = []
    union_gain_by_fold: defaultdict[int, list[float]] = defaultdict(list)
    union_sizes: list[int] = []
    unique_gold: defaultdict[str, int] = defaultdict(int)
    source_gold: defaultdict[str, int] = defaultdict(int)
    pairwise_jaccards: defaultdict[str, list[float]] = defaultdict(list)
    group_oracles: defaultdict[str, list[float]] = defaultdict(list)
    group_mapping_coverages: list[float] = []

    catalog_keys = sorted(catalog_display)
    source_names = (
        "h1",
        "idf_retrieval",
        "cooccurrence",
        "direct_evidence",
    )
    processed = 0
    for fold in range(args.folds):
        fit_rows = [
            row
            for row in train_rows
            if fold_by_group[
                normalize(row.get("target_food")) or f"id:{row.get('id')}"
            ]
            != fold
        ]
        held_rows = [
            row
            for row in train_rows
            if fold_by_group[
                normalize(row.get("target_food")) or f"id:{row.get('id')}"
            ]
            == fold
        ]
        print(
            f"[fold {fold + 1}/{args.folds}] fit={len(fit_rows)} held={len(held_rows)}",
            file=sys.stderr,
            flush=True,
        )
        model = agent.MPCStructureModel(
            fit_rows,
            embeddings=None,
            ablation="full",
            db_path=args.db,
            calibrate_residuals=False,
        )
        fit_profiles = [
            {
                normalize(value)
                for value in list(row.get("partial_molecules") or [])
                + list(row.get("missing_molecules") or [])
                if normalize(value)
            }
            for row in fit_rows
        ]
        for row in held_rows:
            partial = {
                normalize(value)
                for value in row.get("partial_molecules") or []
                if normalize(value)
            }
            gold = {
                normalize(value)
                for value in row.get("missing_molecules") or []
                if normalize(value)
            }
            n = len(gold)
            if not partial or not gold:
                continue
            query = {
                "id": f"{row.get('id')}:candidate-audit",
                "target_food": row.get("target_food"),
                "partial_molecules": list(row.get("partial_molecules") or []),
                "n": n,
            }
            h1_items = model._boundary_training_items(
                query,
                exclude_train_index=None,
                limit=len(model.universe),
            )
            h1 = [
                normalize(item.get("molecule"))
                for item in h1_items
                if normalize(item.get("molecule"))
            ]
            current_retrieval_support = model._build_retrieved_support(
                query,
                exclude_train_index=None,
                top_k=10,
            )
            current_retrieval = sorted(
                current_retrieval_support,
                key=lambda key: (
                    -current_retrieval_support[key],
                    -model.frequency[key],
                    key,
                ),
            )
            improved_retrieval, _ = idf_retrieval(
                query,
                fit_rows,
                fit_profiles,
            )
            cooccurrence = cooccurrence_ranking(
                query,
                model.frequency,
                model.cooccurrence,
                model.training_universe,
            )
            direct_evidence = direct_evidence_ranking(
                query,
                evidence,
                catalog_keys,
                tuple(agent.DIRECT_OCCURRENCE_CUES),
            )
            rrf_statistical = reciprocal_rank_fusion(
                [
                    (1.0, h1),
                    (1.0, improved_retrieval),
                    (1.0, cooccurrence),
                ],
                partial,
            )
            rrf_all = reciprocal_rank_fusion(
                [
                    (1.0, h1),
                    (1.0, improved_retrieval),
                    (1.0, cooccurrence),
                    (1.5, direct_evidence),
                ],
                partial,
            )
            rankings = {
                "h1": stable_unique(h1, partial),
                "current_retrieval": stable_unique(
                    current_retrieval, partial
                ),
                "idf_retrieval": stable_unique(
                    improved_retrieval, partial
                ),
                "cooccurrence": stable_unique(cooccurrence, partial),
                "direct_evidence": stable_unique(direct_evidence, partial),
                "rrf_statistical": rrf_statistical,
                "rrf_all": rrf_all,
            }
            maximum = max(len(catalog_keys), len(model.universe))
            for channel, ranking in rankings.items():
                for cutoff in CUTOFFS:
                    k = cutoff_value(cutoff, n, maximum)
                    metrics[channel][cutoff].add(
                        fold,
                        set(ranking[:k]),
                        gold,
                    )

            source_pools = {
                name: set(
                    rankings[name][
                        : cutoff_value("2n", n, maximum)
                    ]
                )
                for name in source_names
            }
            union_pool = set().union(*source_pools.values())
            union_recall = len(union_pool & gold) / n
            h1_2n_recall = len(source_pools["h1"] & gold) / n
            union_recalls.append(union_recall)
            union_metric.add(fold, union_pool, gold)
            union_gains_over_h1_2n.append(
                union_recall - h1_2n_recall
            )
            union_gain_by_fold[fold].append(
                union_recall - h1_2n_recall
            )
            union_sizes.append(len(union_pool))
            for name, pool in source_pools.items():
                source_gold[name] += len(pool & gold)
                other_pool = set().union(
                    *(
                        value
                        for other_name, value in source_pools.items()
                        if other_name != name
                    )
                )
                unique_gold[name] += len((pool & gold) - other_pool)
            for left_index, left in enumerate(source_names):
                for right in source_names[left_index + 1 :]:
                    left_pool = source_pools[left]
                    right_pool = source_pools[right]
                    pairwise_jaccards[f"{left}:{right}"].append(
                        len(left_pool & right_pool)
                        / max(1, len(left_pool | right_pool))
                    )

            h1_group_oracle, mapping_coverage = greedy_group_oracle(
                rankings["h1"][
                    : cutoff_value("2n", n, maximum)
                ],
                n,
                gold,
                group_map,
            )
            union_group_oracle, _ = greedy_group_oracle(
                sorted(
                    union_pool,
                    key=lambda key: (
                        rankings["rrf_all"].index(key)
                        if key in rankings["rrf_all"]
                        else 10**9,
                        key,
                    ),
                ),
                n,
                gold,
                group_map,
            )
            group_mapping_coverages.append(mapping_coverage)
            if h1_group_oracle is not None:
                group_oracles["h1_2n"].append(h1_group_oracle)
            if union_group_oracle is not None:
                group_oracles["source_union_2n"].append(
                    union_group_oracle
                )
            processed += 1

    return {
        "protocol": {
            "train_path": str(args.train),
            "test_labels_read": False,
            "api_calls": 0,
            "functional_group_evaluation_cache_read": False,
            "folds": args.folds,
            "seed": args.seed,
            "group": "normalized_target_food",
            "queries": processed,
            "candidate_catalog_source": (
                "FlavorDB molecules/molecules_all plus train profile names"
            ),
            "h1_note": (
                "Current v11 primary ranker refit inside each OOF fold "
                "without UniMol features."
            ),
            "functional_oracle_note": (
                "Greedy exact-N oracle over FlavorDB intrinsic functional "
                "groups; this is not the released LLM-cache evaluation."
            ),
        },
        "candidate_recall": {
            channel: {
                cutoff: metric.summary()
                for cutoff, metric in cutoff_metrics.items()
            }
            for channel, cutoff_metrics in metrics.items()
        },
        "source_union_2n": {
            "macro_recall": sum(union_recalls)
            / max(1, len(union_recalls)),
            "exact_n_molecule_oracle_f1": sum(union_recalls)
            / max(1, len(union_recalls)),
            "average_pool_size": sum(union_sizes)
            / max(1, len(union_sizes)),
            "fold_macro_recall": union_metric.summary()[
                "fold_macro_recall"
            ],
            "paired_gain_over_h1_2n": {
                "macro_gain": sum(union_gains_over_h1_2n)
                / max(1, len(union_gains_over_h1_2n)),
                "wins": sum(
                    gain > 1e-12 for gain in union_gains_over_h1_2n
                ),
                "losses": sum(
                    gain < -1e-12 for gain in union_gains_over_h1_2n
                ),
                "ties": sum(
                    abs(gain) <= 1e-12
                    for gain in union_gains_over_h1_2n
                ),
                "fold_macro_gain": {
                    str(fold): sum(values) / len(values)
                    for fold, values in sorted(
                        union_gain_by_fold.items()
                    )
                    if values
                },
                "positive_folds": sum(
                    (sum(values) / len(values)) > 0.0
                    for values in union_gain_by_fold.values()
                    if values
                ),
            },
            "source_gold_hits": dict(source_gold),
            "unique_gold_hits": dict(unique_gold),
            "pairwise_pool_jaccard": {
                key: sum(values) / len(values)
                for key, values in sorted(pairwise_jaccards.items())
                if values
            },
        },
        "intrinsic_functional_group_oracle": {
            "average_gold_molecule_mapping_coverage": (
                sum(group_mapping_coverages)
                / max(1, len(group_mapping_coverages))
            ),
            **{
                key: {
                    "macro_f1": sum(values) / len(values),
                    "evaluated_queries": len(values),
                }
                for key, values in group_oracles.items()
                if values
            },
        },
    }


def print_summary(result: dict[str, Any]) -> None:
    protocol = result["protocol"]
    print("MPC CANDIDATE UPPER-BOUND AUDIT")
    print(
        f"queries={protocol['queries']} folds={protocol['folds']} "
        f"seed={protocol['seed']} api_calls=0 test_labels_read=false"
    )
    print()
    print("macro candidate recall")
    print("channel".ljust(24) + "".join(label.rjust(12) for label in CUTOFFS))
    for channel in CHANNELS:
        values = result["candidate_recall"][channel]
        print(
            channel.ljust(24)
            + "".join(
                f"{values[label]['macro_recall']:.6f}".rjust(12)
                for label in CUTOFFS
            )
        )
    print()
    union = result["source_union_2n"]
    print(
        "source_union_2n "
        f"macro_recall={union['macro_recall']:.6f} "
        f"exact_n_oracle_f1={union['exact_n_molecule_oracle_f1']:.6f} "
        f"average_pool_size={union['average_pool_size']:.2f}"
    )
    paired = union["paired_gain_over_h1_2n"]
    print(
        "union_vs_h1_2n "
        f"macro_gain={paired['macro_gain']:.6f} "
        f"wins={paired['wins']} losses={paired['losses']} "
        f"ties={paired['ties']} positive_folds={paired['positive_folds']}"
    )
    print(
        "unique_gold_hits="
        + json.dumps(union["unique_gold_hits"], sort_keys=True)
    )
    print()
    oracle = result["intrinsic_functional_group_oracle"]
    print(
        "intrinsic_group_mapping_coverage="
        f"{oracle['average_gold_molecule_mapping_coverage']:.6f}"
    )
    for key in ("h1_2n", "source_union_2n"):
        if key in oracle:
            print(
                f"{key}_greedy_exact_n_macro_f1="
                f"{oracle[key]['macro_f1']:.6f} "
                f"queries={oracle[key]['evaluated_queries']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit MPC candidate recall and oracle upper bounds using "
            "train-only grouped OOF."
        )
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete audit as JSON after the compact summary.",
    )
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    for path in (args.train, args.db, args.evidence, args.agent):
        if not path.is_file():
            parser.error(f"required file not found: {path}")
    return args


def main() -> int:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "warning: set PYTHONHASHSEED=0 for fully reproducible "
            "ordering inside the imported v11 ranker",
            file=sys.stderr,
        )
    result = audit(args)
    print_summary(result)
    if args.json:
        print()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
