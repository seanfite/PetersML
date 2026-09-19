#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from recommender.model import (
    DEFAULT_PETERS_POSITIVE_WEIGHTS,
    DEFAULT_PETERS_RISK_WEIGHTS,
    PetersRecommender,
    multitask_loss,
    peters_score,
)

from features.definitions import (
    COMPATIBILITY_FIELDS,
    FEATURE_SCHEMA_VERSION,
    MULTI_CATEGORICAL_FIELDS,
    NUMERIC_INPUT_FIELDS,
    PROFILE_DESCRIPTOR_FIELDS,
    SINGLE_CATEGORICAL_FIELDS,
    TARGET_BOOLEAN_FIELDS,
    USER_AFFINITY_FIELDS,
)

DEFAULT_DATASET = "data/prepared_pairs.jsonl"
DEFAULT_VOCAB = "data/vocab.json"
DEFAULT_CHECKPOINT = "models/peters_recommender.pt"

DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_SEED = 42
DEFAULT_POS_WEIGHT_CAP = 25.0
RANKING_K = 20


class PetersDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    return rows


def load_vocab(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        vocab = json.load(file)

    if "fields" not in vocab:
        raise RuntimeError("vocab.json uses the old format. Run prepare_features.py first.")

    if vocab.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            "Vocabulary feature schema does not match the current model. "
            "Run prepare_features.py again."
        )

    return vocab


def vocab_sizes(vocab: dict[str, Any]) -> dict[str, int]:
    sizes: dict[str, int] = {}

    for field in SINGLE_CATEGORICAL_FIELDS + MULTI_CATEGORICAL_FIELDS:
        if field not in vocab["fields"]:
            raise RuntimeError(f"Vocabulary is missing field: {field}")

        sizes[field] = len(vocab["fields"][field])

    return sizes


def validate_prepared_row(row: dict[str, Any], row_number: int) -> None:
    required_sections = [
        "metadata",
        "categorical",
        "multi_categorical",
        "numeric",
        "numeric_missing",
        "compatibility",
        "compatibility_missing",
        "viewer_profile_descriptors",
        "candidate_profile_descriptors",
        "viewer_affinities",
        "candidate_affinities",
        "targets",
    ]

    missing = [name for name in required_sections if name not in row]

    if missing:
        raise RuntimeError(
            f"Prepared row {row_number} is missing sections: {', '.join(missing)}"
        )

    schema_version = row["metadata"].get("feature_schema_version")
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Prepared row {row_number} uses feature schema {schema_version!r}; "
            f"expected {FEATURE_SCHEMA_VERSION}."
        )

    checks = [
        ("categorical", SINGLE_CATEGORICAL_FIELDS),
        ("multi_categorical", MULTI_CATEGORICAL_FIELDS),
        ("numeric", NUMERIC_INPUT_FIELDS),
        ("numeric_missing", NUMERIC_INPUT_FIELDS),
        ("compatibility", COMPATIBILITY_FIELDS),
        ("compatibility_missing", COMPATIBILITY_FIELDS),
        ("viewer_profile_descriptors", PROFILE_DESCRIPTOR_FIELDS),
        ("candidate_profile_descriptors", PROFILE_DESCRIPTOR_FIELDS),
        ("viewer_affinities", USER_AFFINITY_FIELDS),
        ("candidate_affinities", USER_AFFINITY_FIELDS),
        ("targets", TARGET_BOOLEAN_FIELDS),
    ]

    for section, fields in checks:
        for field in fields:
            if field not in row[section]:
                raise RuntimeError(
                    f"Prepared row {row_number} is missing {section}.{field}"
                )


def parse_cutoff(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def random_split_rows(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 3:
        return rows, [], []

    indexes = list(range(len(rows)))
    random.Random(seed).shuffle(indexes)

    validation_count = (
        max(1, round(len(rows) * validation_fraction)) if validation_fraction > 0 else 0
    )
    test_count = max(1, round(len(rows) * test_fraction)) if test_fraction > 0 else 0

    while validation_count + test_count >= len(rows):
        if test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break

    test_indexes = set(indexes[:test_count])
    validation_indexes = set(indexes[test_count:test_count + validation_count])

    train = [
        row
        for index, row in enumerate(rows)
        if index not in test_indexes and index not in validation_indexes
    ]
    validation = [
        row for index, row in enumerate(rows) if index in validation_indexes
    ]
    test = [row for index, row in enumerate(rows) if index in test_indexes]

    return train, validation, test


def split_rows(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    groups: dict[datetime, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        cutoff = parse_cutoff(row["metadata"].get("cutoff_at"))

        if cutoff is None:
            train, validation, test = random_split_rows(
                rows, validation_fraction, test_fraction, seed
            )
            return train, validation, test, "random"

        groups[cutoff].append(row)

    cutoffs = sorted(groups)

    # Current V1 data normally has a single cutoff. Once build_dataset.py
    # produces historical snapshots, this automatically becomes chronological.
    if len(cutoffs) < 3:
        print("WARNING: fewer than 3 cutoff snapshots; using random split.")
        train, validation, test = random_split_rows(
            rows, validation_fraction, test_fraction, seed
        )
        return train, validation, test, "random"

    test_target = len(rows) * test_fraction
    validation_target = len(rows) * validation_fraction

    test_cutoffs: set[datetime] = set()
    test_rows_count = 0

    for cutoff in reversed(cutoffs):
        if test_fraction <= 0 or test_rows_count >= test_target:
            break

        test_cutoffs.add(cutoff)
        test_rows_count += len(groups[cutoff])

    remaining = [cutoff for cutoff in cutoffs if cutoff not in test_cutoffs]

    validation_cutoffs: set[datetime] = set()
    validation_rows_count = 0

    for cutoff in reversed(remaining):
        if validation_fraction <= 0 or validation_rows_count >= validation_target:
            break

        if len(remaining) - len(validation_cutoffs) <= 1:
            break

        validation_cutoffs.add(cutoff)
        validation_rows_count += len(groups[cutoff])

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for cutoff in cutoffs:
        if cutoff in test_cutoffs:
            test.extend(groups[cutoff])
        elif cutoff in validation_cutoffs:
            validation.extend(groups[cutoff])
        else:
            train.extend(groups[cutoff])

    if not train:
        raise RuntimeError("Chronological split produced no training rows.")

    return train, validation, test, "chronological"


def optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    return parsed if math.isfinite(parsed) else None


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2

    if len(values) % 2:
        return values[middle]

    return (values[middle - 1] + values[middle]) / 2.0


def calculate_numeric_stats(
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    impute_values: list[float] = []

    for field in NUMERIC_INPUT_FIELDS:
        values = [
            value
            for row in rows
            if (value := optional_float(row["numeric"].get(field))) is not None
        ]
        impute_values.append(median(values) if values else 0.0)

    impute = torch.tensor(impute_values, dtype=torch.float32)

    matrix = torch.tensor(
        [
            [
                optional_float(row["numeric"].get(field))
                if optional_float(row["numeric"].get(field)) is not None
                else impute[index].item()
                for index, field in enumerate(NUMERIC_INPUT_FIELDS)
            ]
            for row in rows
        ],
        dtype=torch.float32,
    )

    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    return impute, mean, std


def calculate_positive_weights(
    rows: list[dict[str, Any]],
    cap: float,
) -> dict[str, float]:
    weights: dict[str, float] = {}

    for target in TARGET_BOOLEAN_FIELDS:
        positives = sum(int(row["targets"][target]) for row in rows)
        negatives = len(rows) - positives

        if positives == 0 or negatives == 0:
            weights[target] = 1.0
        else:
            weights[target] = min(max(1.0, negatives / positives), cap)

    return weights


def collate_optional_matrix(
    rows: list[dict[str, Any]],
    section: str,
    fields: list[str],
) -> torch.Tensor | None:
    if not fields:
        return None

    return torch.tensor(
        [
            [float(row[section].get(field, 0.0) or 0.0) for field in fields]
            for row in rows
        ],
        dtype=torch.float32,
    )


def collate_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(rows)

    categorical = {
        field: torch.tensor(
            [row["categorical"][field] for row in rows],
            dtype=torch.long,
        )
        for field in SINGLE_CATEGORICAL_FIELDS
    }

    multi_categorical: dict[str, torch.Tensor] = {}

    for field in MULTI_CATEGORICAL_FIELDS:
        values = [row["multi_categorical"][field] for row in rows]
        max_values = max(1, max(len(item) for item in values))

        padded = torch.zeros((batch_size, max_values), dtype=torch.long)

        for index, item in enumerate(values):
            if item:
                padded[index, :len(item)] = torch.tensor(item, dtype=torch.long)

        multi_categorical[field] = padded

    numeric_rows: list[list[float]] = []
    numeric_missing_rows: list[list[float]] = []

    for row in rows:
        values: list[float] = []
        missing_values: list[float] = []

        for field in NUMERIC_INPUT_FIELDS:
            value = optional_float(row["numeric"].get(field))
            explicit_missing = float(row["numeric_missing"].get(field, 0.0)) >= 0.5
            missing = explicit_missing or value is None

            values.append(float("nan") if missing else float(value))
            missing_values.append(1.0 if missing else 0.0)

        numeric_rows.append(values)
        numeric_missing_rows.append(missing_values)

    compatibility_rows: list[list[float]] = []
    compatibility_missing_rows: list[list[float]] = []

    for row in rows:
        values: list[float] = []
        missing_values: list[float] = []

        for field in COMPATIBILITY_FIELDS:
            value = optional_float(row["compatibility"].get(field))
            explicit_missing = (
                float(row["compatibility_missing"].get(field, 0.0)) >= 0.5
            )
            missing = explicit_missing or value is None

            values.append(float("nan") if missing else float(value))
            missing_values.append(1.0 if missing else 0.0)

        compatibility_rows.append(values)
        compatibility_missing_rows.append(missing_values)

    targets = {
        field: torch.tensor(
            [float(row["targets"][field]) for row in rows],
            dtype=torch.float32,
        )
        for field in TARGET_BOOLEAN_FIELDS
    }

    return {
        "categorical": categorical,
        "multi_categorical": multi_categorical,
        "numeric": torch.tensor(numeric_rows, dtype=torch.float32),
        "numeric_missing": torch.tensor(numeric_missing_rows, dtype=torch.float32),
        "compatibility": torch.tensor(compatibility_rows, dtype=torch.float32),
        "compatibility_missing": torch.tensor(
            compatibility_missing_rows, dtype=torch.float32
        ),
        "viewer_profile_descriptors": collate_optional_matrix(
            rows, "viewer_profile_descriptors", PROFILE_DESCRIPTOR_FIELDS
        ),
        "candidate_profile_descriptors": collate_optional_matrix(
            rows, "candidate_profile_descriptors", PROFILE_DESCRIPTOR_FIELDS
        ),
        "viewer_affinities": collate_optional_matrix(
            rows, "viewer_affinities", USER_AFFINITY_FIELDS
        ),
        "candidate_affinities": collate_optional_matrix(
            rows, "candidate_affinities", USER_AFFINITY_FIELDS
        ),
        "targets": targets,
        "metadata": [row["metadata"] for row in rows],
    }


def move_optional(
    tensor: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    return tensor.to(device) if tensor is not None else None


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "categorical": {
            field: tensor.to(device)
            for field, tensor in batch["categorical"].items()
        },
        "multi_categorical": {
            field: tensor.to(device)
            for field, tensor in batch["multi_categorical"].items()
        },
        "numeric": batch["numeric"].to(device),
        "numeric_missing": batch["numeric_missing"].to(device),
        "compatibility": batch["compatibility"].to(device),
        "compatibility_missing": batch["compatibility_missing"].to(device),
        "viewer_profile_descriptors": move_optional(
            batch["viewer_profile_descriptors"], device
        ),
        "candidate_profile_descriptors": move_optional(
            batch["candidate_profile_descriptors"], device
        ),
        "viewer_affinities": move_optional(batch["viewer_affinities"], device),
        "candidate_affinities": move_optional(batch["candidate_affinities"], device),
        "targets": {
            field: tensor.to(device)
            for field, tensor in batch["targets"].items()
        },
        "metadata": batch["metadata"],
    }


def model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "categorical": batch["categorical"],
        "multi_categorical": batch["multi_categorical"],
        "numeric": batch["numeric"],
        "numeric_missing": batch["numeric_missing"],
        "compatibility": batch["compatibility"],
        "compatibility_missing": batch["compatibility_missing"],
        "viewer_profile_descriptors": batch["viewer_profile_descriptors"],
        "candidate_profile_descriptors": batch["candidate_profile_descriptors"],
        "viewer_affinities": batch["viewer_affinities"],
        "candidate_affinities": batch["candidate_affinities"],
    }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(
    model: PetersRecommender,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    positive_weights: dict[str, float],
) -> float:
    model.train()

    total_loss = 0.0
    total_rows = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(**model_inputs(batch))
        loss, _ = multitask_loss(
            logits=logits,
            targets=batch["targets"],
            positive_weights=positive_weights,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = batch["numeric"].shape[0]
        total_loss += loss.item() * batch_size
        total_rows += batch_size

    return total_loss / max(total_rows, 1)


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives

    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, labels), key=lambda item: item[0])

    rank_sum = 0.0
    index = 0

    while index < len(ordered):
        end = index + 1

        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1

        average_rank = (index + 1 + end) / 2.0
        rank_sum += sum(label for _, label in ordered[index:end]) * average_rank
        index = end

    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)

    if positives == 0:
        return None

    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)

    true_positives = 0
    precision_sum = 0.0

    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank

    return precision_sum / positives


def binary_metrics(labels: list[int], scores: list[float]) -> dict[str, Any]:
    if not labels:
        return {}

    predicted = [1 if score >= 0.5 else 0 for score in scores]

    true_positive = sum(
        expected == 1 and actual == 1
        for expected, actual in zip(labels, predicted)
    )
    false_positive = sum(
        expected == 0 and actual == 1
        for expected, actual in zip(labels, predicted)
    )
    false_negative = sum(
        expected == 1 and actual == 0
        for expected, actual in zip(labels, predicted)
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else None
    )

    return {
        "rows": len(labels),
        "positives": sum(labels),
        "positive_rate": sum(labels) / len(labels),
        "accuracy": sum(a == b for a, b in zip(labels, predicted)) / len(labels),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
    }


def dcg(labels: list[int]) -> float:
    return sum(
        1.0 / math.log2(index + 1)
        for index, label in enumerate(labels, start=1)
        if label
    )


def ranking_metrics(
    metadata: list[dict[str, Any]],
    scores: list[float],
    targets: dict[str, list[int]],
    k: int = RANKING_K,
) -> dict[str, Any]:
    by_viewer: dict[int, list[int]] = defaultdict(list)

    for index, item in enumerate(metadata):
        viewer_id = item.get("viewer_user_id")
        if viewer_id is not None:
            by_viewer[int(viewer_id)].append(index)

    relevance = targets["outcome_any_positive"]

    ndcg_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    tracked = [
        "outcome_favorite",
        "outcome_message_started",
        "outcome_reciprocal_message",
        "outcome_block",
        "outcome_report",
    ]

    top_counts = {target: 0 for target in tracked}
    top_rows = 0

    for indexes in by_viewer.values():
        ranked = sorted(indexes, key=lambda index: scores[index], reverse=True)
        top = ranked[:k]

        if not top:
            continue

        top_labels = [relevance[index] for index in top]
        all_positive = sum(relevance[index] for index in indexes)
        top_positive = sum(top_labels)

        precision_values.append(top_positive / len(top))

        if all_positive > 0:
            recall_values.append(top_positive / all_positive)

        ideal_count = min(all_positive, len(top))
        ideal_dcg = dcg([1] * ideal_count + [0] * (len(top) - ideal_count))

        if ideal_dcg > 0:
            ndcg_values.append(dcg(top_labels) / ideal_dcg)

        for target in tracked:
            top_counts[target] += sum(targets[target][index] for index in top)

        top_rows += len(top)

    result = {
        "k": k,
        "viewers": len(by_viewer),
        "ndcg_at_k": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
        "precision_at_k": (
            sum(precision_values) / len(precision_values) if precision_values else None
        ),
        "recall_at_k": sum(recall_values) / len(recall_values) if recall_values else None,
    }

    for target, count in top_counts.items():
        result[f"{target}_rate_at_k"] = count / top_rows if top_rows else None

    return result


@torch.no_grad()
def evaluate(
    model: PetersRecommender,
    loader: DataLoader,
    device: torch.device,
    positive_weights: dict[str, float],
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_rows = 0

    all_scores: list[float] = []
    all_metadata: list[dict[str, Any]] = []

    labels = {target: [] for target in TARGET_BOOLEAN_FIELDS}
    probabilities = {target: [] for target in TARGET_BOOLEAN_FIELDS}

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(**model_inputs(batch))
        loss, _ = multitask_loss(
            logits=logits,
            targets=batch["targets"],
            positive_weights=positive_weights,
        )

        batch_probabilities = {
            target: torch.sigmoid(value) for target, value in logits.items()
        }
        scores = peters_score(batch_probabilities)

        batch_size = batch["numeric"].shape[0]
        total_loss += loss.item() * batch_size
        total_rows += batch_size

        all_scores.extend(scores.detach().cpu().tolist())
        all_metadata.extend(batch["metadata"])

        for target in TARGET_BOOLEAN_FIELDS:
            labels[target].extend(
                batch["targets"][target].detach().cpu().ge(0.5).int().tolist()
            )
            probabilities[target].extend(
                batch_probabilities[target].detach().cpu().tolist()
            )

    return {
        "loss": total_loss / max(total_rows, 1),
        "rows": total_rows,
        "targets": {
            target: binary_metrics(labels[target], probabilities[target])
            for target in TARGET_BOOLEAN_FIELDS
        },
        "ranking": ranking_metrics(all_metadata, all_scores, labels),
    }


def print_target_distribution(name: str, rows: list[dict[str, Any]]) -> None:
    print()
    print(name)
    print("────────────────────────────────────────────────────")

    for target in TARGET_BOOLEAN_FIELDS:
        positives = sum(int(row["targets"][target]) for row in rows)
        rate = positives / len(rows) if rows else 0.0

        print(f"{target:<40} {positives:>6}/{len(rows):<6} {rate:>7.2%}")


def print_positive_weights(weights: dict[str, float]) -> None:
    print()
    print("POSITIVE CLASS WEIGHTS")
    print("────────────────────────────────────────────────────")

    for target in TARGET_BOOLEAN_FIELDS:
        print(f"{target:<40} {weights[target]:>8.3f}")


def format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def print_evaluation(name: str, metrics: dict[str, Any]) -> None:
    ranking = metrics["ranking"]

    print()
    print(name)
    print("────────────────────────────────────────────────────")
    print(f"Loss:         {metrics['loss']:.6f}")
    print(f"NDCG@{ranking['k']}:      {format_metric(ranking['ndcg_at_k'])}")
    print(f"Precision@{ranking['k']}: {format_metric(ranking['precision_at_k'])}")
    print(f"Recall@{ranking['k']}:    {format_metric(ranking['recall_at_k'])}")
    print()
    print(f"{'Target':<38} {'AP':>8} {'ROC-AUC':>8}")

    for target in TARGET_BOOLEAN_FIELDS:
        item = metrics["targets"][target]
        print(
            f"{target:<38} "
            f"{format_metric(item['average_precision']):>8} "
            f"{format_metric(item['roc_auc']):>8}"
        )


def save_checkpoint(
    path: Path,
    model: PetersRecommender,
    optimizer_state: dict[str, Any],
    vocab: dict[str, Any],
    epoch: int,
    train_loss: float,
    validation_metrics: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    positive_weights: dict[str, float],
    split_mode: str,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state,
        "model_config": model.config.__dict__,
        "feature_spec": model.feature_spec(),
        "feature_schema_version": model.feature_schema_version,
        "vocab": vocab,
        "vocab_sizes": {
            field: len(values) for field, values in vocab["fields"].items()
        },
        "single_categorical_fields": list(model.single_fields),
        "multi_categorical_fields": list(model.multi_fields),
        "numeric_fields": list(model.numeric_fields),
        "compatibility_fields": list(model.compatibility_fields),
        "profile_descriptor_fields": list(model.profile_descriptor_fields),
        "user_affinity_fields": list(model.user_affinity_fields),
        "target_fields": list(model.target_fields),
        "numeric_impute": model.numeric_impute.detach().cpu(),
        "numeric_mean": model.numeric_mean.detach().cpu(),
        "numeric_std": model.numeric_std.detach().cpu(),
        "positive_weights": positive_weights,
        "peters_positive_weights": dict(DEFAULT_PETERS_POSITIVE_WEIGHTS),
        "peters_risk_weights": dict(DEFAULT_PETERS_RISK_WEIGHTS),
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_metrics["loss"] if validation_metrics else None,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "split_mode": split_mode,
        "seed": seed,
    }

    torch.save(checkpoint, path)


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Peters production recommender."
    )

    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics-output", default=None)

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument(
        "--positive-weight-cap", type=float, default=DEFAULT_POS_WEIGHT_CAP
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    vocab_path = Path(args.vocab)
    checkpoint_path = Path(args.checkpoint)
    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else checkpoint_path.with_suffix(".metrics.json")
    )

    if not dataset_path.exists():
        raise SystemExit(f"Prepared dataset does not exist: {dataset_path}")

    if not vocab_path.exists():
        raise SystemExit(f"Vocabulary does not exist: {vocab_path}")

    if args.epochs <= 0:
        raise SystemExit("--epochs must be greater than 0")

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")

    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be greater than 0")

    if not 0 <= args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be >= 0 and < 1")

    if not 0 <= args.test_fraction < 1:
        raise SystemExit("--test-fraction must be >= 0 and < 1")

    if args.validation_fraction + args.test_fraction >= 1:
        raise SystemExit("validation + test fractions must be less than 1")

    if args.positive_weight_cap < 1:
        raise SystemExit("--positive-weight-cap must be >= 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(dataset_path)

    for row_number, row in enumerate(rows, start=1):
        validate_prepared_row(row, row_number)

    train_rows, validation_rows, test_rows, split_mode = split_rows(
        rows,
        args.validation_fraction,
        args.test_fraction,
        args.seed,
    )

    vocab = load_vocab(vocab_path)
    model = PetersRecommender(vocab_sizes=vocab_sizes(vocab))

    numeric_impute, numeric_mean, numeric_std = calculate_numeric_stats(train_rows)
    model.set_numeric_stats(numeric_impute, numeric_mean, numeric_std)

    positive_weights = calculate_positive_weights(
        train_rows,
        cap=args.positive_weight_cap,
    )

    device = choose_device()
    model = model.to(device)

    train_loader = DataLoader(
        PetersDataset(train_rows),
        batch_size=min(args.batch_size, len(train_rows)),
        shuffle=True,
        collate_fn=collate_batch,
    )

    validation_loader = (
        DataLoader(
            PetersDataset(validation_rows),
            batch_size=min(args.batch_size, len(validation_rows)),
            shuffle=False,
            collate_fn=collate_batch,
        )
        if validation_rows
        else None
    )

    test_loader = (
        DataLoader(
            PetersDataset(test_rows),
            batch_size=min(args.batch_size, len(test_rows)),
            shuffle=False,
            collate_fn=collate_batch,
        )
        if test_rows
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    print()
    print("PETERS RECOMMENDER TRAINING")
    print("────────────────────────────────────────")
    print(f"Feature schema:    {FEATURE_SCHEMA_VERSION}")
    print(f"Dataset:           {dataset_path}")
    print(f"Rows:              {len(rows):,}")
    print(f"Split mode:        {split_mode}")
    print(f"Training rows:     {len(train_rows):,}")
    print(f"Validation rows:   {len(validation_rows):,}")
    print(f"Test rows:         {len(test_rows):,}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Learning rate:     {args.learning_rate}")
    print(f"Device:            {device}")
    print(f"Checkpoint:        {checkpoint_path}")
    print(f"Metrics:           {metrics_path}")

    print_target_distribution("TRAIN TARGET DISTRIBUTION", train_rows)

    if validation_rows:
        print_target_distribution("VALIDATION TARGET DISTRIBUTION", validation_rows)

    if test_rows:
        print_target_distribution("TEST TARGET DISTRIBUTION", test_rows)

    print_positive_weights(positive_weights)

    best_validation_loss = math.inf
    best_epoch = 0
    best_train_loss = math.inf
    best_model_state = None
    best_optimizer_state = None

    print()
    print("TRAINING")
    print("────────────────────────────────────────")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            positive_weights,
        )

        validation_metrics = (
            evaluate(model, validation_loader, device, positive_weights)
            if validation_loader
            else None
        )

        selection_loss = (
            validation_metrics["loss"] if validation_metrics else train_loss
        )

        improved = selection_loss < best_validation_loss

        if improved:
            best_validation_loss = selection_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:>3}/{args.epochs} "
            f"train={train_loss:.6f} val={selection_loss:.6f}{marker}"
        )

    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError("Training completed without capturing a model state.")

    model.load_state_dict(best_model_state)

    validation_metrics = (
        evaluate(model, validation_loader, device, positive_weights)
        if validation_loader
        else None
    )

    test_metrics = (
        evaluate(model, test_loader, device, positive_weights)
        if test_loader
        else None
    )

    if validation_metrics:
        print_evaluation("BEST MODEL — VALIDATION", validation_metrics)

    if test_metrics:
        print_evaluation("BEST MODEL — TEST", test_metrics)

    save_checkpoint(
        checkpoint_path,
        model,
        best_optimizer_state,
        vocab,
        best_epoch,
        best_train_loss,
        validation_metrics,
        test_metrics,
        positive_weights,
        split_mode,
        args.seed,
    )

    metrics_document = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "split_mode": split_mode,
        "rows": {
            "total": len(rows),
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "positive_weights": positive_weights,
        "peters_score": {
            "positive_weights": dict(DEFAULT_PETERS_POSITIVE_WEIGHTS),
            "risk_weights": dict(DEFAULT_PETERS_RISK_WEIGHTS),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    write_metrics(metrics_path, metrics_document)

    print()
    print("TRAINING COMPLETE")
    print("────────────────────────────────────────")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best train loss:  {best_train_loss:.6f}")

    if validation_metrics:
        print(f"Best val loss:    {validation_metrics['loss']:.6f}")

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics:    {metrics_path}")
    print()

    if split_mode == "random":
        print(
            "NOTE: Historical cutoff snapshots are not available yet, "
            "so validation is temporarily using a random split."
        )
        print()


if __name__ == "__main__":
    main()
    