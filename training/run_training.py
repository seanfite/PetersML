#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import torch
from psycopg.types.json import Jsonb

from features.definitions import (
    FEATURE_SCHEMA_VERSION,
    GRAPH_NUMERIC_FIELDS,
    USER_AFFINITY_FIELDS,
)


DEFAULT_MODEL_NAME = "peters_recommender"

DEFAULT_HISTORY_DAYS = 90
DEFAULT_OUTCOME_DAYS = 7
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 64
DEFAULT_GRAPH_BATCH_SIZE = 1000
DEFAULT_SNAPSHOT_COUNT = 12
DEFAULT_SNAPSHOT_STEP_DAYS = 7
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_POSITIVE_WEIGHT_CAP = 25.0

DEFAULT_GRAPH_URL = "http://127.0.0.1:8082"
DEFAULT_AFFINITY_MODEL_VERSION = "behavioral_affinity_v1"

RUNS_DIR = Path("runs")
MODELS_DIR = Path("models")
CANDIDATES_DIR = MODELS_DIR / "candidates"
LATEST_CANDIDATE = MODELS_DIR / "latest_candidate.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Peters production ML training pipeline."
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
    )

    parser.add_argument(
        "--outcome-days",
        type=int,
        default=DEFAULT_OUTCOME_DAYS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=DEFAULT_GRAPH_BATCH_SIZE,
        help=(
            "Maximum candidate pairs per historical PetersGraph request. "
            f"Default: {DEFAULT_GRAPH_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=DEFAULT_SNAPSHOT_COUNT,
        help=(
            "Number of historical cutoff snapshots to build. "
            f"Default: {DEFAULT_SNAPSHOT_COUNT}"
        ),
    )

    parser.add_argument(
        "--snapshot-step-days",
        type=int,
        default=DEFAULT_SNAPSHOT_STEP_DAYS,
        help=(
            "Days between historical cutoff snapshots. "
            f"Default: {DEFAULT_SNAPSHOT_STEP_DAYS}"
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
    )

    parser.add_argument(
        "--positive-weight-cap",
        type=float,
        default=DEFAULT_POSITIVE_WEIGHT_CAP,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--gcs-bucket",
        default=os.getenv("PETERS_ML_BUCKET"),
    )

    parser.add_argument(
        "--gcs-prefix",
        default=os.getenv(
            "PETERS_ML_PREFIX",
            "peters-ml/recommender",
        ),
    )

    parser.add_argument(
        "--affinity-model-version",
        default=os.getenv(
            "PETERS_AFFINITY_MODEL_VERSION",
            DEFAULT_AFFINITY_MODEL_VERSION,
        ),
        help=(
            "Live behavioral affinity version recorded with the model. "
            f"Default: {DEFAULT_AFFINITY_MODEL_VERSION}"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise SystemExit(f"{name} is not set")

    return value


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_run_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ")


def run_command(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    print()

    subprocess.run(
        command,
        check=True,
    )


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(
            1
            for line in file
            if line.strip()
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file.write("\n")

    temporary_path.replace(path)


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def inspect_raw_dataset(
    path: Path,
) -> dict[str, Any]:
    """
    Validate that build_dataset.py produced graph + behavioral-affinity rows.

    This checks the raw dataset before feature preparation so we never create
    or register a candidate from a stale pre-v5 training pipeline.
    """

    row_count = 0
    hard_excluded_pairs = 0
    viewer_rows_with_affinity_evidence = 0
    candidate_rows_with_affinity_evidence = 0
    viewer_rows_with_recent_affinity_evidence = 0
    candidate_rows_with_recent_affinity_evidence = 0
    cutoffs: set[str] = set()

    required_graph_fields = [
        *GRAPH_NUMERIC_FIELDS,
        "graph_is_hard_excluded",
    ]

    def validate_affinity_section(
        row: dict[str, Any],
        section: str,
        line_number: int,
    ) -> dict[str, Any]:
        values = row.get(section)

        if not isinstance(values, dict):
            raise RuntimeError(
                f"Raw training row {line_number} is missing "
                f"{section}."
            )

        missing = [
            field
            for field in USER_AFFINITY_FIELDS
            if field not in values
        ]

        unexpected = [
            field
            for field in values
            if field not in USER_AFFINITY_FIELDS
        ]

        if missing:
            raise RuntimeError(
                f"Raw training row {line_number} {section} is missing: "
                + ", ".join(missing)
            )

        if unexpected:
            raise RuntimeError(
                f"Raw training row {line_number} {section} has "
                "unexpected fields: "
                + ", ".join(unexpected)
            )

        for field in USER_AFFINITY_FIELDS:
            try:
                value = float(values[field])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Raw training row {line_number} "
                    f"{section}.{field} is invalid: "
                    f"{values[field]!r}"
                ) from error

            if not math.isfinite(value):
                raise RuntimeError(
                    f"Raw training row {line_number} "
                    f"{section}.{field} is not finite."
                )

            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"Raw training row {line_number} "
                    f"{section}.{field} must be between 0 and 1."
                )

        return values

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid raw dataset JSON on line "
                    f"{line_number}: {error}"
                ) from error

            row_count += 1

            missing_graph = [
                field
                for field in required_graph_fields
                if field not in row
            ]

            if missing_graph:
                raise RuntimeError(
                    f"Raw training row {line_number} is missing "
                    "PetersGraph fields: "
                    f"{', '.join(missing_graph)}"
                )

            viewer_affinities = validate_affinity_section(
                row,
                "viewer_affinities",
                line_number,
            )

            candidate_affinities = validate_affinity_section(
                row,
                "candidate_affinities",
                line_number,
            )

            if float(
                viewer_affinities["evidence_strength"]
            ) > 0.0:
                viewer_rows_with_affinity_evidence += 1

            if float(
                candidate_affinities["evidence_strength"]
            ) > 0.0:
                candidate_rows_with_affinity_evidence += 1

            if float(
                viewer_affinities["recent_evidence_strength"]
            ) > 0.0:
                viewer_rows_with_recent_affinity_evidence += 1

            if float(
                candidate_affinities["recent_evidence_strength"]
            ) > 0.0:
                candidate_rows_with_recent_affinity_evidence += 1

            cutoff_at = row.get("cutoff_at")

            if cutoff_at is None:
                raise RuntimeError(
                    f"Raw training row {line_number} is missing cutoff_at"
                )

            cutoffs.add(
                str(cutoff_at)
            )

            if bool(
                row.get("graph_is_hard_excluded")
            ):
                hard_excluded_pairs += 1

    if row_count == 0:
        raise RuntimeError(
            "Raw training dataset contains no rows."
        )

    return {
        "rows": row_count,
        "cutoffs": sorted(cutoffs),
        "cutoff_count": len(cutoffs),
        "hard_excluded_pairs": hard_excluded_pairs,
        "graph_numeric_fields": list(
            GRAPH_NUMERIC_FIELDS
        ),
        "user_affinity_fields": list(
            USER_AFFINITY_FIELDS
        ),
        "viewer_rows_with_affinity_evidence": (
            viewer_rows_with_affinity_evidence
        ),
        "candidate_rows_with_affinity_evidence": (
            candidate_rows_with_affinity_evidence
        ),
        "viewer_rows_with_recent_affinity_evidence": (
            viewer_rows_with_recent_affinity_evidence
        ),
        "candidate_rows_with_recent_affinity_evidence": (
            candidate_rows_with_recent_affinity_evidence
        ),
    }


# ---------------------------------------------------------------------------
# Checkpoint validation
# ---------------------------------------------------------------------------


def load_checkpoint_summary(
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    return {
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("train_loss"),
        "validation_loss": checkpoint.get(
            "validation_loss"
        ),
        "split_mode": checkpoint.get("split_mode"),
        "seed": checkpoint.get("seed"),
        "model_config": checkpoint.get("model_config"),
        "feature_schema_version": checkpoint.get(
            "feature_schema_version"
        ),
        "feature_spec": checkpoint.get("feature_spec"),
        "numeric_fields": checkpoint.get(
            "numeric_fields"
        ),
        "user_affinity_fields": checkpoint.get(
            "user_affinity_fields"
        ),
        "positive_weights": checkpoint.get(
            "positive_weights"
        ),
        "validation_metrics": checkpoint.get(
            "validation_metrics"
        ),
        "test_metrics": checkpoint.get(
            "test_metrics"
        ),
    }


def validate_personalized_checkpoint(
    checkpoint_summary: dict[str, Any],
) -> None:
    schema_version = checkpoint_summary.get(
        "feature_schema_version"
    )

    if schema_version != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            "Checkpoint feature schema mismatch: "
            f"got {schema_version!r}, "
            f"expected {FEATURE_SCHEMA_VERSION}"
        )

    numeric_fields = checkpoint_summary.get(
        "numeric_fields"
    )

    if not isinstance(numeric_fields, list):
        raise RuntimeError(
            "Checkpoint does not contain numeric_fields."
        )

    missing_graph_fields = [
        field
        for field in GRAPH_NUMERIC_FIELDS
        if field not in numeric_fields
    ]

    if missing_graph_fields:
        raise RuntimeError(
            "Checkpoint is missing PetersGraph neural inputs: "
            f"{', '.join(missing_graph_fields)}"
        )

    if "graph_is_hard_excluded" in numeric_fields:
        raise RuntimeError(
            "graph_is_hard_excluded must not be a neural input."
        )

    affinity_fields = checkpoint_summary.get(
        "user_affinity_fields"
    )

    if not isinstance(affinity_fields, list):
        raise RuntimeError(
            "Checkpoint does not contain user_affinity_fields."
        )

    expected_affinity_fields = list(
        USER_AFFINITY_FIELDS
    )

    if affinity_fields != expected_affinity_fields:
        actual_set = set(affinity_fields)
        expected_set = set(expected_affinity_fields)

        missing = [
            field
            for field in expected_affinity_fields
            if field not in actual_set
        ]

        unexpected = [
            field
            for field in affinity_fields
            if field not in expected_set
        ]

        if not missing and not unexpected:
            detail = "same fields but different ordering"
        else:
            parts: list[str] = []

            if missing:
                parts.append(
                    "missing=" + ", ".join(missing)
                )

            if unexpected:
                parts.append(
                    "unexpected=" + ", ".join(unexpected)
                )

            detail = "; ".join(parts)

        raise RuntimeError(
            "Checkpoint user affinity fields do not match "
            f"schema {FEATURE_SCHEMA_VERSION}: {detail}"
        )


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------


def gcs_uri(
    bucket: str,
    object_name: str,
) -> str:
    return (
        f"gs://{bucket}/"
        f"{object_name.lstrip('/')}"
    )


def upload_to_gcs(
    bucket_name: str,
    object_name: str,
    path: Path,
) -> str:
    try:
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError(
            "google-cloud-storage is required "
            "when --gcs-bucket is used."
        ) from error

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    blob.upload_from_filename(
        str(path)
    )

    return gcs_uri(
        bucket_name,
        object_name,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def register_candidate(
    dsn: str,
    model_name: str,
    version: str,
    run_id: str,
    artifact_uri: str,
    manifest_uri: str,
    metrics: dict[str, Any],
    trained_at: datetime,
) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ml_model_versions (
                    model_name,
                    version,
                    run_id,
                    artifact_uri,
                    manifest_uri,
                    status,
                    metrics,
                    trained_at
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    'candidate',
                    %s,
                    %s
                )
                """,
                (
                    model_name,
                    version,
                    run_id,
                    artifact_uri,
                    manifest_uri,
                    Jsonb(metrics),
                    trained_at,
                ),
            )

        conn.commit()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    dsn = require_env("APP_DSN")

    if args.history_days <= 0:
        raise SystemExit(
            "--history-days must be greater than 0"
        )

    if args.outcome_days <= 0:
        raise SystemExit(
            "--outcome-days must be greater than 0"
        )

    if args.epochs <= 0:
        raise SystemExit(
            "--epochs must be greater than 0"
        )

    if args.batch_size <= 0:
        raise SystemExit(
            "--batch-size must be greater than 0"
        )

    if args.graph_batch_size <= 0:
        raise SystemExit(
            "--graph-batch-size must be greater than 0"
        )

    if args.graph_batch_size > 1000:
        raise SystemExit(
            "--graph-batch-size cannot exceed 1000"
        )

    if args.snapshot_count <= 0:
        raise SystemExit(
            "--snapshot-count must be greater than 0"
        )

    if args.snapshot_step_days <= 0:
        raise SystemExit(
            "--snapshot-step-days must be greater than 0"
        )

    if args.snapshot_step_days < args.outcome_days:
        raise SystemExit(
            "--snapshot-step-days must be greater than or equal "
            "to --outcome-days so outcome windows do not overlap"
        )

    if args.learning_rate <= 0:
        raise SystemExit(
            "--learning-rate must be greater than 0"
        )

    if not 0 <= args.validation_fraction < 1:
        raise SystemExit(
            "--validation-fraction must be >= 0 and < 1"
        )

    if not 0 <= args.test_fraction < 1:
        raise SystemExit(
            "--test-fraction must be >= 0 and < 1"
        )

    if (
        args.validation_fraction
        + args.test_fraction
        >= 1
    ):
        raise SystemExit(
            "validation + test fractions "
            "must be less than 1"
        )

    if args.positive_weight_cap < 1:
        raise SystemExit(
            "--positive-weight-cap must be >= 1"
        )

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise SystemExit(
            "--limit must be greater than 0"
        )

    affinity_model_version = str(
        args.affinity_model_version
    ).strip()

    if not affinity_model_version:
        raise SystemExit(
            "--affinity-model-version cannot be empty"
        )

    graph_url = os.getenv(
        "PETERS_GRAPH_URL",
        DEFAULT_GRAPH_URL,
    ).strip().rstrip("/")

    started_at = utc_now()
    run_id = make_run_id(started_at)
    version = run_id

    run_dir = RUNS_DIR / run_id

    raw_dataset_path = (
        run_dir / "training_pairs.jsonl"
    )

    prepared_dataset_path = (
        run_dir / "prepared_pairs.jsonl"
    )

    vocab_path = (
        run_dir / "vocab.json"
    )

    feature_schema_path = (
        run_dir / "feature_schema.json"
    )

    checkpoint_path = (
        run_dir / "model.pt"
    )

    metrics_path = (
        run_dir / "metrics.json"
    )

    manifest_path = (
        run_dir / "manifest.json"
    )

    candidate_path = (
        CANDIDATES_DIR
        / f"{args.model_name}_{version}.pt"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("PETERS PRODUCTION TRAINING")
    print("────────────────────────────────────────")
    print(
        f"Model:            {args.model_name}"
    )
    print(
        f"Version:          {version}"
    )
    print(
        f"Run ID:           {run_id}"
    )
    print(
        f"Feature schema:   {FEATURE_SCHEMA_VERSION}"
    )
    print(
        f"History days:     {args.history_days}"
    )
    print(
        f"Outcome days:     {args.outcome_days}"
    )
    print(
        f"Epochs:           {args.epochs}"
    )
    print(
        f"Batch size:       {args.batch_size}"
    )
    print(
        f"Graph batch size: {args.graph_batch_size}"
    )
    print(
        f"Snapshot count:   {args.snapshot_count}"
    )
    print(
        f"Snapshot step:    {args.snapshot_step_days} days"
    )
    print(
        f"PetersGraph:      {graph_url}"
    )
    print(
        f"Affinity model:   {affinity_model_version}"
    )
    print(
        f"Run directory:    {run_dir}"
    )
    print()

    # ------------------------------------------------------------------
    # 1. Build cutoff-safe SQL + PetersGraph + affinity training rows
    # ------------------------------------------------------------------

    build_command = [
        sys.executable,
        "-m",
        "training.build_dataset",
        "--history-days",
        str(args.history_days),
        "--outcome-days",
        str(args.outcome_days),
        "--graph-batch-size",
        str(args.graph_batch_size),
        "--snapshot-count",
        str(args.snapshot_count),
        "--snapshot-step-days",
        str(args.snapshot_step_days),
        "--output",
        str(raw_dataset_path),
    ]

    if args.limit is not None:
        build_command.extend(
            [
                "--limit",
                str(args.limit),
            ]
        )

    run_command(
        build_command
    )

    dataset_summary = inspect_raw_dataset(
        raw_dataset_path
    )

    raw_rows = int(
        dataset_summary["rows"]
    )

    if raw_rows < 2:
        raise RuntimeError(
            "Training run produced fewer than "
            "2 dataset rows."
        )

    print()
    print("PERSONALIZED DATASET VALIDATED")
    print("────────────────────────────────────────")
    print(
        f"Rows:              {raw_rows:,}"
    )
    print(
        "Cutoff snapshots:  "
        f"{dataset_summary['cutoff_count']}"
    )
    print(
        "Graph inputs:      "
        f"{len(GRAPH_NUMERIC_FIELDS)}"
    )
    print(
        "Affinity inputs:   "
        f"{len(USER_AFFINITY_FIELDS)} per direction"
    )
    print(
        "Viewer evidence:   "
        f"{dataset_summary['viewer_rows_with_affinity_evidence']:,}/"
        f"{raw_rows:,} rows"
    )
    print(
        "Candidate evidence:"
        f" {dataset_summary['candidate_rows_with_affinity_evidence']:,}/"
        f"{raw_rows:,} rows"
    )
    print(
        "Hard exclusions:   "
        f"{dataset_summary['hard_excluded_pairs']:,}"
    )
    print()

    # ------------------------------------------------------------------
    # 2. Prepare schema-v5 personalized model features
    # ------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "training.prepare_features",
            "--input",
            str(raw_dataset_path),
            "--output",
            str(prepared_dataset_path),
            "--vocab",
            str(vocab_path),
            "--schema",
            str(feature_schema_path),
        ]
    )

    prepared_rows = count_jsonl_rows(
        prepared_dataset_path
    )

    if prepared_rows != raw_rows:
        raise RuntimeError(
            "Prepared dataset row count does not "
            "match raw dataset row count."
        )

    # ------------------------------------------------------------------
    # 3. Train personalized PetersRecommender
    # ------------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "training.train",
            "--dataset",
            str(prepared_dataset_path),
            "--vocab",
            str(vocab_path),
            "--checkpoint",
            str(checkpoint_path),
            "--metrics-output",
            str(metrics_path),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--validation-fraction",
            str(args.validation_fraction),
            "--test-fraction",
            str(args.test_fraction),
            "--positive-weight-cap",
            str(args.positive_weight_cap),
        ]
    )

    # ------------------------------------------------------------------
    # 4. Verify training artifacts
    # ------------------------------------------------------------------

    for required_path in [
        checkpoint_path,
        metrics_path,
        vocab_path,
        feature_schema_path,
    ]:
        if not required_path.exists():
            raise RuntimeError(
                "Training artifact was not created: "
                f"{required_path}"
            )

    checkpoint_summary = load_checkpoint_summary(
        checkpoint_path
    )

    validate_personalized_checkpoint(
        checkpoint_summary
    )

    metrics = read_json(
        metrics_path
    )

    print()
    print("PERSONALIZED CHECKPOINT VALIDATED")
    print("────────────────────────────────────────")
    print(
        "Feature schema:   "
        f"{checkpoint_summary['feature_schema_version']}"
    )
    print(
        "Numeric inputs:   "
        f"{len(checkpoint_summary['numeric_fields'])}"
    )
    print(
        "Graph inputs:     "
        f"{len(GRAPH_NUMERIC_FIELDS)}"
    )
    print(
        "Affinity inputs:  "
        f"{len(checkpoint_summary['user_affinity_fields'])} per direction"
    )
    print(
        "Hard exclusion:   control metadata only"
    )
    print()

    # ------------------------------------------------------------------
    # 5. Candidate artifact
    # ------------------------------------------------------------------

    shutil.copy2(
        checkpoint_path,
        candidate_path,
    )

    completed_at = utc_now()

    artifact_uri = str(
        candidate_path.resolve()
    )

    manifest_uri = str(
        manifest_path.resolve()
    )

    if args.gcs_bucket:
        prefix = args.gcs_prefix.strip("/")

        run_prefix = (
            f"{prefix}/"
            f"{args.model_name}/"
            f"{version}"
        )

        artifact_uri = gcs_uri(
            args.gcs_bucket,
            f"{run_prefix}/model.pt",
        )

        manifest_uri = gcs_uri(
            args.gcs_bucket,
            f"{run_prefix}/manifest.json",
        )

    # ------------------------------------------------------------------
    # 6. Manifest
    # ------------------------------------------------------------------

    manifest = {
        "model_name": args.model_name,
        "version": version,
        "run_id": run_id,
        "status": "candidate",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "artifact_uri": artifact_uri,
        "manifest_uri": manifest_uri,
        "training": {
            "history_days": args.history_days,
            "outcome_days": args.outcome_days,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "graph_batch_size": (
                args.graph_batch_size
            ),
            "snapshot_count": args.snapshot_count,
            "snapshot_step_days": (
                args.snapshot_step_days
            ),
            "populated_cutoff_count": (
                dataset_summary["cutoff_count"]
            ),
            "learning_rate": args.learning_rate,
            "validation_fraction": (
                args.validation_fraction
            ),
            "test_fraction": (
                args.test_fraction
            ),
            "positive_weight_cap": (
                args.positive_weight_cap
            ),
            "raw_rows": raw_rows,
            "prepared_rows": prepared_rows,
        },
        "graph": {
            "enabled": True,
            "service_url": graph_url,
            "historical_training": True,
            "cutoff_count": (
                dataset_summary["cutoff_count"]
            ),
            "cutoffs": dataset_summary["cutoffs"],
            "numeric_fields": list(
                GRAPH_NUMERIC_FIELDS
            ),
            "numeric_field_count": len(
                GRAPH_NUMERIC_FIELDS
            ),
            "hard_excluded_pairs": (
                dataset_summary[
                    "hard_excluded_pairs"
                ]
            ),
            "hard_exclusion_is_model_input": False,
        },
        "affinities": {
            "enabled": True,
            "model_version": affinity_model_version,
            "historical_training": True,
            "fields": list(
                USER_AFFINITY_FIELDS
            ),
            "field_count": len(
                USER_AFFINITY_FIELDS
            ),
            "directions": [
                "viewer_to_candidate",
                "candidate_to_viewer",
            ],
            "viewer_rows_with_evidence": (
                dataset_summary[
                    "viewer_rows_with_affinity_evidence"
                ]
            ),
            "candidate_rows_with_evidence": (
                dataset_summary[
                    "candidate_rows_with_affinity_evidence"
                ]
            ),
            "viewer_rows_with_recent_evidence": (
                dataset_summary[
                    "viewer_rows_with_recent_affinity_evidence"
                ]
            ),
            "candidate_rows_with_recent_evidence": (
                dataset_summary[
                    "candidate_rows_with_recent_affinity_evidence"
                ]
            ),
        },
        "checkpoint": checkpoint_summary,
        "metrics": metrics,
        "artifacts": {
            "raw_dataset": {
                "path": str(
                    raw_dataset_path
                ),
                "sha256": sha256_file(
                    raw_dataset_path
                ),
            },
            "prepared_dataset": {
                "path": str(
                    prepared_dataset_path
                ),
                "sha256": sha256_file(
                    prepared_dataset_path
                ),
            },
            "vocab": {
                "path": str(
                    vocab_path
                ),
                "sha256": sha256_file(
                    vocab_path
                ),
            },
            "feature_schema": {
                "path": str(
                    feature_schema_path
                ),
                "sha256": sha256_file(
                    feature_schema_path
                ),
            },
            "checkpoint": {
                "path": str(
                    checkpoint_path
                ),
                "sha256": sha256_file(
                    checkpoint_path
                ),
            },
            "candidate_model": {
                "path": str(
                    candidate_path
                ),
                "sha256": sha256_file(
                    candidate_path
                ),
            },
            "metrics": {
                "path": str(
                    metrics_path
                ),
                "sha256": sha256_file(
                    metrics_path
                ),
            },
        },
    }

    write_json(
        manifest_path,
        manifest,
    )

    # ------------------------------------------------------------------
    # 7. Optional durable GCS artifact upload
    # ------------------------------------------------------------------

    if args.gcs_bucket:
        prefix = args.gcs_prefix.strip("/")

        run_prefix = (
            f"{prefix}/"
            f"{args.model_name}/"
            f"{version}"
        )

        print()
        print("UPLOADING MODEL ARTIFACTS")
        print("────────────────────────────────────────")

        upload_to_gcs(
            args.gcs_bucket,
            f"{run_prefix}/model.pt",
            candidate_path,
        )

        upload_to_gcs(
            args.gcs_bucket,
            f"{run_prefix}/manifest.json",
            manifest_path,
        )

        upload_to_gcs(
            args.gcs_bucket,
            f"{run_prefix}/metrics.json",
            metrics_path,
        )

        upload_to_gcs(
            args.gcs_bucket,
            f"{run_prefix}/feature_schema.json",
            feature_schema_path,
        )

        upload_to_gcs(
            args.gcs_bucket,
            f"{run_prefix}/vocab.json",
            vocab_path,
        )

        print(
            f"Model:            {artifact_uri}"
        )
        print(
            f"Manifest:         {manifest_uri}"
        )

    # ------------------------------------------------------------------
    # 8. Candidate registry
    #
    # Only happens AFTER graph + affinity checkpoint validation above.
    # ------------------------------------------------------------------

    register_candidate(
        dsn=dsn,
        model_name=args.model_name,
        version=version,
        run_id=run_id,
        artifact_uri=artifact_uri,
        manifest_uri=manifest_uri,
        metrics=metrics,
        trained_at=completed_at,
    )

    latest_candidate = {
        "model_name": args.model_name,
        "version": version,
        "run_id": run_id,
        "status": "candidate",
        "model_uri": artifact_uri,
        "manifest_uri": manifest_uri,
        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),
        "graph_aware": True,
        "graph_numeric_field_count": len(
            GRAPH_NUMERIC_FIELDS
        ),
        "snapshot_count": args.snapshot_count,
        "snapshot_step_days": args.snapshot_step_days,
        "populated_cutoff_count": (
            dataset_summary["cutoff_count"]
        ),
        "behavioral_affinity_aware": True,
        "affinity_model_version": (
            affinity_model_version
        ),
        "user_affinity_field_count": len(
            USER_AFFINITY_FIELDS
        ),
        "created_at": completed_at.isoformat(),
    }

    write_json(
        LATEST_CANDIDATE,
        latest_candidate,
    )

    # ------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------

    print()
    print("TRAINING RUN COMPLETE")
    print("────────────────────────────────────────")
    print(
        f"Model:            {args.model_name}"
    )
    print(
        f"Version:          {version}"
    )
    print(
        f"Run ID:           {run_id}"
    )
    print(
        f"Feature schema:   {FEATURE_SCHEMA_VERSION}"
    )
    print(
        f"Rows:             {prepared_rows:,}"
    )
    print(
        f"Cutoff snapshots: {dataset_summary['cutoff_count']}"
    )
    print(
        f"Graph inputs:     {len(GRAPH_NUMERIC_FIELDS)}"
    )
    print(
        f"Affinity inputs:  {len(USER_AFFINITY_FIELDS)} per direction"
    )
    print(
        f"Affinity model:   {affinity_model_version}"
    )
    print(
        "Train loss:       "
        f"{checkpoint_summary.get('train_loss')}"
    )
    print(
        "Validation loss:  "
        f"{checkpoint_summary.get('validation_loss')}"
    )
    print(
        f"Artifact:         {artifact_uri}"
    )
    print(
        f"Manifest:         {manifest_uri}"
    )
    print(
        "Registry status:  candidate"
    )
    print()
    print(
        "Candidate was NOT automatically promoted "
        "to production."
    )
    print()


if __name__ == "__main__":
    main()
    