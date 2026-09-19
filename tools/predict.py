#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from model import ModelConfig, PetersRecommender
from prepare_features import (
    MULTI_CATEGORICAL_FIELDS,
    NUMERIC_INPUT_FIELDS,
    SINGLE_CATEGORICAL_FIELDS,
    TARGET_BOOLEAN_FIELDS,
)


DEFAULT_DATASET = "data/prepared_pairs.jsonl"
DEFAULT_CHECKPOINT = "models/peters_recommender.pt"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def load_rows(path: Path) -> list[dict[str, Any]]:
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
        raise RuntimeError(
            f"No rows found in {path}"
        )

    return rows


def select_row(
    rows: list[dict[str, Any]],
    row_index: int,
    viewer_user_id: int | None,
    candidate_user_id: int | None,
) -> tuple[int, dict[str, Any]]:
    if viewer_user_id is not None or candidate_user_id is not None:
        if viewer_user_id is None or candidate_user_id is None:
            raise SystemExit(
                "--viewer-user-id and --candidate-user-id "
                "must be supplied together."
            )

        for index, row in enumerate(rows):
            metadata = row["metadata"]

            if (
                metadata["viewer_user_id"] == viewer_user_id
                and metadata["candidate_user_id"] == candidate_user_id
            ):
                return index, row

        raise SystemExit(
            f"No prepared row found for viewer {viewer_user_id} "
            f"-> candidate {candidate_user_id}."
        )

    if row_index < 0 or row_index >= len(rows):
        raise SystemExit(
            f"--row-index must be between 0 and {len(rows) - 1}"
        )

    return row_index, rows[row_index]


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PetersRecommender, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    if "vocab_sizes" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain vocab_sizes."
        )

    config_data = checkpoint.get(
        "model_config",
        {},
    )

    config = ModelConfig(
        **config_data
    )

    model = PetersRecommender(
        vocab_sizes=checkpoint["vocab_sizes"],
        config=config,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def row_to_tensors(
    row: dict[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    categorical: dict[str, torch.Tensor] = {}

    for field in SINGLE_CATEGORICAL_FIELDS:
        value = row["categorical"][field]

        categorical[field] = torch.tensor(
            [value],
            dtype=torch.long,
            device=device,
        )

    multi_categorical: dict[str, torch.Tensor] = {}

    for field in MULTI_CATEGORICAL_FIELDS:
        values = row["multi_categorical"][field]

        if not values:
            values = [0]

        multi_categorical[field] = torch.tensor(
            [values],
            dtype=torch.long,
            device=device,
        )

    numeric_values = [
        float(row["numeric"][field])
        for field in NUMERIC_INPUT_FIELDS
    ]

    numeric = torch.tensor(
        [numeric_values],
        dtype=torch.float32,
        device=device,
    )

    return (
        categorical,
        multi_categorical,
        numeric,
    )


@torch.no_grad()
def predict_row(
    model: PetersRecommender,
    row: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    categorical, multi_categorical, numeric = row_to_tensors(
        row=row,
        device=device,
    )

    probabilities = model.predict_probabilities(
        categorical=categorical,
        multi_categorical=multi_categorical,
        numeric=numeric,
    )

    return {
        target: float(
            probabilities[target]
            .detach()
            .cpu()
            .item()
        )
        for target in TARGET_BOOLEAN_FIELDS
    }


def print_prediction(
    row_index: int,
    row: dict[str, Any],
    probabilities: dict[str, float],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> None:
    metadata = row["metadata"]
    actual_targets = row["targets"]

    print()
    print("PETERS RECOMMENDER PREDICTION")
    print("────────────────────────────────────────")
    print(f"Row:              {row_index}")
    print(
        "Viewer:           "
        f"{metadata['viewer_user_id']}"
    )
    print(
        "Candidate:        "
        f"{metadata['candidate_user_id']}"
    )
    print(f"Device:            {device}")
    print(
        "Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    validation_loss = checkpoint.get(
        "validation_loss"
    )

    if validation_loss is not None:
        print(
            "Checkpoint loss:  "
            f"{validation_loss:.6f}"
        )

    print()
    print("PREDICTIONS")
    print(
        "────────────────────────────────"
        "────────────────────"
    )
    print(
        f"{'Target':<38} "
        f"{'Predicted':>10} "
        f"{'Observed':>10}"
    )
    print(
        "────────────────────────────────"
        "────────────────────"
    )

    for target in TARGET_BOOLEAN_FIELDS:
        probability = probabilities[target]
        observed = int(
            actual_targets[target]
        )

        print(
            f"{target:<38} "
            f"{probability:>9.2%} "
            f"{observed:>10}"
        )

    print()

    print("PAIR CONTEXT")
    print("────────────────────────────────────────")

    numeric = row["numeric"]

    print(
        "Distance:          "
        f"{numeric['distance_km']:.2f} km"
    )

    print(
        "Viewer age:        "
        f"{numeric['viewer_age']:.0f}"
    )

    print(
        "Candidate age:     "
        f"{numeric['candidate_age']:.0f}"
    )

    print(
        "Prior pair views:  "
        f"{numeric['history_profile_view_count']:.0f}"
    )

    print(
        "Prior favorites:   "
        f"{numeric['history_favorite_count']:.0f}"
    )

    print(
        "Prior messages:    "
        f"{numeric['history_message_count']:.0f}"
    )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Peters recommender inference "
            "for a prepared viewer/candidate row."
        )
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Prepared dataset row to predict. Default: 0",
    )

    parser.add_argument(
        "--viewer-user-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--candidate-user-id",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(
        args.dataset
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset does not exist: {dataset_path}"
        )

    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    rows = load_rows(
        dataset_path
    )

    row_index, row = select_row(
        rows=rows,
        row_index=args.row_index,
        viewer_user_id=args.viewer_user_id,
        candidate_user_id=args.candidate_user_id,
    )

    device = choose_device()

    model, checkpoint = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    probabilities = predict_row(
        model=model,
        row=row,
        device=device,
    )

    print_prediction(
        row_index=row_index,
        row=row,
        probabilities=probabilities,
        checkpoint=checkpoint,
        device=device,
    )


if __name__ == "__main__":
    main()
    