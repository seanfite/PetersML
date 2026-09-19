#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from features.definitions import (
    FEATURE_SCHEMA_VERSION,
    GRAPH_NUMERIC_FIELDS,
)
from inference.model_loader import load_candidate_model


DEFAULT_MODEL_NAME = "peters_recommender"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateValidation:
    model_name: str
    version: str
    feature_schema_version: int
    numeric_field_count: int
    graph_field_count: int
    target_field_count: int
    device: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a Peters ML candidate model to production."
    )

    parser.add_argument(
        "version",
        help="Candidate model version to promote.",
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise SystemExit(f"{name} is not set")

    return value


# ---------------------------------------------------------------------------
# Registry reads
# ---------------------------------------------------------------------------


def load_model(
    conn: psycopg.Connection,
    model_name: str,
    version: str,
    *,
    for_update: bool = False,
) -> dict[str, Any]:
    lock_clause = "FOR UPDATE" if for_update else ""

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            f"""
            SELECT
                id,
                model_name,
                version,
                status,
                artifact_uri,
                manifest_uri,
                metrics,
                trained_at,
                promoted_at
            FROM ml_model_versions
            WHERE model_name = %s
              AND version = %s
            {lock_clause}
            """,
            (
                model_name,
                version,
            ),
        )

        row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            f"Model not found: {model_name} version {version}"
        )

    return dict(row)


def current_production(
    conn: psycopg.Connection,
    model_name: str,
    *,
    for_update: bool = False,
) -> dict[str, Any] | None:
    lock_clause = "FOR UPDATE" if for_update else ""

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            f"""
            SELECT
                id,
                model_name,
                version,
                status,
                artifact_uri,
                manifest_uri,
                trained_at,
                promoted_at
            FROM ml_model_versions
            WHERE model_name = %s
              AND status = 'production'
            {lock_clause}
            """,
            (
                model_name,
            ),
        )

        row = cursor.fetchone()

    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Candidate artifact validation
# ---------------------------------------------------------------------------


def validate_candidate_artifact(
    model_name: str,
    version: str,
) -> CandidateValidation:
    """
    Load the exact registered candidate through the same checkpoint loader
    used by inference and verify the current Peters model contract.

    Nothing is promoted here.
    """

    candidate = load_candidate_model(
        version=version,
        model_name=model_name,
        device="cpu",
    )

    loaded = candidate.loaded

    if loaded.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            "Candidate feature schema is not current: "
            f"got {loaded.feature_schema_version}, "
            f"expected {FEATURE_SCHEMA_VERSION}"
        )

    numeric_fields = list(
        loaded.numeric_fields
    )

    graph_fields = [
        field
        for field in numeric_fields
        if field.startswith("graph_")
    ]

    missing_graph_fields = [
        field
        for field in GRAPH_NUMERIC_FIELDS
        if field not in numeric_fields
    ]

    if missing_graph_fields:
        raise RuntimeError(
            "Candidate checkpoint is missing PetersGraph inputs: "
            + ", ".join(missing_graph_fields)
        )

    unexpected_graph_fields = [
        field
        for field in graph_fields
        if field not in GRAPH_NUMERIC_FIELDS
    ]

    if unexpected_graph_fields:
        raise RuntimeError(
            "Candidate checkpoint contains unexpected graph inputs: "
            + ", ".join(unexpected_graph_fields)
        )

    if len(graph_fields) != len(GRAPH_NUMERIC_FIELDS):
        raise RuntimeError(
            "Candidate graph feature count mismatch: "
            f"got {len(graph_fields)}, "
            f"expected {len(GRAPH_NUMERIC_FIELDS)}"
        )

    if "graph_is_hard_excluded" in numeric_fields:
        raise RuntimeError(
            "graph_is_hard_excluded must not be a neural input."
        )

    if not loaded.target_fields:
        raise RuntimeError(
            "Candidate checkpoint contains no prediction targets."
        )

    return CandidateValidation(
        model_name=candidate.record.model_name,
        version=candidate.record.version,
        feature_schema_version=loaded.feature_schema_version,
        numeric_field_count=len(numeric_fields),
        graph_field_count=len(graph_fields),
        target_field_count=len(loaded.target_fields),
        device=str(loaded.device),
    )


# ---------------------------------------------------------------------------
# Registry promotion
# ---------------------------------------------------------------------------


def promote_model(
    conn: psycopg.Connection,
    model_name: str,
    version: str,
) -> tuple[str | None, str]:
    """
    Atomically archive the existing production model and promote the target.

    The candidate artifact has already passed validation before this function
    is called, but the registry status is checked again under row locks.
    """

    target = load_model(
        conn,
        model_name,
        version,
        for_update=True,
    )

    if target["status"] == "production":
        return version, version

    if target["status"] != "candidate":
        raise RuntimeError(
            f"Model {version} has status {target['status']!r}; "
            "only candidate models may be promoted."
        )

    artifact_uri = str(
        target["artifact_uri"] or ""
    ).strip()

    if not artifact_uri:
        raise RuntimeError(
            f"Candidate model {version} has no artifact_uri."
        )

    manifest_uri = str(
        target["manifest_uri"] or ""
    ).strip()

    if not manifest_uri:
        raise RuntimeError(
            f"Candidate model {version} has no manifest_uri."
        )

    current = current_production(
        conn,
        model_name,
        for_update=True,
    )

    previous_version = (
        str(current["version"])
        if current is not None
        else None
    )

    with conn.cursor() as cursor:
        if current is not None:
            cursor.execute(
                """
                UPDATE ml_model_versions
                SET status = 'archived'
                WHERE id = %s
                  AND status = 'production'
                """,
                (
                    current["id"],
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    "Failed to archive exactly one "
                    "current production model."
                )

        cursor.execute(
            """
            UPDATE ml_model_versions
            SET status = 'production',
                promoted_at = now()
            WHERE id = %s
              AND status = 'candidate'
            """,
            (
                target["id"],
            ),
        )

        if cursor.rowcount != 1:
            raise RuntimeError(
                "Promotion update did not modify "
                "exactly one candidate model."
            )

    verify_single_production(
        conn=conn,
        model_name=model_name,
        expected_version=version,
    )

    return previous_version, version


def verify_single_production(
    conn: psycopg.Connection,
    model_name: str,
    expected_version: str,
) -> None:
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT version
            FROM ml_model_versions
            WHERE model_name = %s
              AND status = 'production'
            """,
            (
                model_name,
            ),
        )

        rows = cursor.fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            "Registry validation failed: expected exactly "
            f"one production model, found {len(rows)}."
        )

    actual_version = str(
        rows[0]["version"]
    )

    if actual_version != expected_version:
        raise RuntimeError(
            "Registry validation failed after promotion: "
            f"expected {expected_version}, "
            f"found {actual_version}."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    model_name = args.model_name.strip()
    version = args.version.strip()

    if not model_name:
        raise SystemExit(
            "--model-name cannot be empty"
        )

    if not version:
        raise SystemExit(
            "version cannot be empty"
        )

    dsn = require_env(
        "APP_DSN"
    )

    # ------------------------------------------------------------------
    # Registry preflight
    # ------------------------------------------------------------------

    with psycopg.connect(
        dsn
    ) as conn:
        target = load_model(
            conn,
            model_name,
            version,
        )

        existing_production = (
            current_production(
                conn,
                model_name,
            )
        )

    if target["status"] == "production":
        print()
        print("PETERS MODEL PROMOTION")
        print("────────────────────────────────────────")
        print(f"Model:       {model_name}")
        print(f"Production:  {version}")
        print("Status:      already production")
        print()
        return

    if target["status"] != "candidate":
        raise RuntimeError(
            f"Model {version} has status {target['status']!r}; "
            "only candidate models may be promoted."
        )

    # ------------------------------------------------------------------
    # Artifact / schema preflight
    #
    # This happens BEFORE any production registry mutation.
    # ------------------------------------------------------------------

    print()
    print("PETERS MODEL PROMOTION PREFLIGHT")
    print("────────────────────────────────────────")
    print(f"Model:            {model_name}")
    print(f"Candidate:        {version}")
    print(
        "Current prod:     "
        + (
            str(existing_production["version"])
            if existing_production
            else "none"
        )
    )
    print()

    validation = validate_candidate_artifact(
        model_name=model_name,
        version=version,
    )

    print("PROMOTION CONTRACT VALIDATED")
    print("────────────────────────────────────────")
    print(
        f"Feature schema:   "
        f"{validation.feature_schema_version}"
    )
    print(
        f"Numeric inputs:   "
        f"{validation.numeric_field_count}"
    )
    print(
        f"Graph inputs:     "
        f"{validation.graph_field_count}"
    )
    print(
        f"Targets:          "
        f"{validation.target_field_count}"
    )
    print(
        "Hard exclusion:   control metadata only"
    )
    print(
        f"Validation device:{validation.device:>8}"
    )
    print()

    # ------------------------------------------------------------------
    # Atomic registry promotion
    # ------------------------------------------------------------------

    with psycopg.connect(
        dsn
    ) as conn:
        previous, promoted = promote_model(
            conn,
            model_name=model_name,
            version=version,
        )

        conn.commit()

    print()
    print("PETERS MODEL PROMOTION")
    print("────────────────────────────────────────")
    print(f"Model:       {model_name}")

    if previous == promoted:
        print(f"Production:  {promoted}")
        print("Status:      already production")
    else:
        print(
            f"Previous:    "
            f"{previous or 'none'}"
        )
        print(
            f"Production:  {promoted}"
        )
        print(
            "Status:      promoted"
        )

    print()


if __name__ == "__main__":
    main()
    