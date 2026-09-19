from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row

from recommender.checkpoint import (
    LoadedCheckpoint,
    load_recommender,
)


DEFAULT_MODEL_NAME = "peters_recommender"
RUNTIME_MODEL_DIR = Path("models/runtime")


# ---------------------------------------------------------------------------
# Registered model records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionModelRecord:
    model_name: str
    version: str
    artifact_uri: str
    manifest_uri: str | None


@dataclass(frozen=True)
class CandidateModelRecord:
    model_name: str
    version: str
    artifact_uri: str
    manifest_uri: str | None


@dataclass(frozen=True)
class ProductionModel:
    record: ProductionModelRecord
    loaded: LoadedCheckpoint


@dataclass(frozen=True)
class CandidateModel:
    record: CandidateModelRecord
    loaded: LoadedCheckpoint


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"{name} is not set"
        )

    return value


# ---------------------------------------------------------------------------
# Registry lookups
# ---------------------------------------------------------------------------


def get_production_model_record(
    model_name: str = DEFAULT_MODEL_NAME,
) -> ProductionModelRecord:
    dsn = require_env(
        "APP_DSN"
    )

    with psycopg.connect(
        dsn,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    model_name,
                    version,
                    artifact_uri,
                    manifest_uri
                FROM ml_model_versions
                WHERE model_name = %s
                  AND status = 'production'
                ORDER BY trained_at DESC
                LIMIT 1
                """,
                (
                    model_name,
                ),
            )

            row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            "No production model registered for "
            f"{model_name!r}."
        )

    artifact_uri = str(
        row["artifact_uri"]
    ).strip()

    if not artifact_uri:
        raise RuntimeError(
            "Production model "
            f"{row['version']} has no artifact_uri."
        )

    return ProductionModelRecord(
        model_name=str(
            row["model_name"]
        ),
        version=str(
            row["version"]
        ),
        artifact_uri=artifact_uri,
        manifest_uri=(
            str(
                row["manifest_uri"]
            )
            if row["manifest_uri"] is not None
            else None
        ),
    )


def get_candidate_model_record(
    version: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> CandidateModelRecord:
    dsn = require_env(
        "APP_DSN"
    )

    with psycopg.connect(
        dsn,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            if version is None:
                cursor.execute(
                    """
                    SELECT
                        model_name,
                        version,
                        artifact_uri,
                        manifest_uri
                    FROM ml_model_versions
                    WHERE model_name = %s
                      AND status = 'candidate'
                    ORDER BY trained_at DESC
                    LIMIT 1
                    """,
                    (
                        model_name,
                    ),
                )

            else:
                cursor.execute(
                    """
                    SELECT
                        model_name,
                        version,
                        artifact_uri,
                        manifest_uri
                    FROM ml_model_versions
                    WHERE model_name = %s
                      AND version = %s
                      AND status = 'candidate'
                    LIMIT 1
                    """,
                    (
                        model_name,
                        version,
                    ),
                )

            row = cursor.fetchone()

    if row is None:
        if version is None:
            raise RuntimeError(
                "No candidate model registered for "
                f"{model_name!r}."
            )

        raise RuntimeError(
            "Candidate model not found: "
            f"{model_name!r} version={version!r}"
        )

    artifact_uri = str(
        row["artifact_uri"]
    ).strip()

    if not artifact_uri:
        raise RuntimeError(
            "Candidate model "
            f"{row['version']} has no artifact_uri."
        )

    return CandidateModelRecord(
        model_name=str(
            row["model_name"]
        ),
        version=str(
            row["version"]
        ),
        artifact_uri=artifact_uri,
        manifest_uri=(
            str(
                row["manifest_uri"]
            )
            if row["manifest_uri"] is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Artifact resolution
# ---------------------------------------------------------------------------


def download_gcs_artifact(
    uri: str,
    model_name: str,
    version: str,
) -> Path:
    parsed = urlparse(
        uri
    )

    bucket_name = parsed.netloc
    object_name = parsed.path.lstrip(
        "/"
    )

    if not bucket_name or not object_name:
        raise RuntimeError(
            f"Invalid GCS model URI: {uri}"
        )

    destination_dir = (
        RUNTIME_MODEL_DIR
        / model_name
        / version
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        destination_dir
        / "model.pt"
    )

    if destination.exists():
        return destination

    try:
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError(
            "google-cloud-storage is required "
            "to load gs:// model artifacts."
        ) from error

    print(
        f"Downloading model: {uri}"
    )

    client = storage.Client()
    bucket = client.bucket(
        bucket_name
    )
    blob = bucket.blob(
        object_name
    )

    temporary = destination.with_suffix(
        ".pt.tmp"
    )

    blob.download_to_filename(
        str(temporary)
    )

    temporary.replace(
        destination
    )

    return destination


def resolve_artifact(
    record: ProductionModelRecord | CandidateModelRecord,
) -> Path:
    uri = record.artifact_uri

    if uri.startswith(
        "gs://"
    ):
        return download_gcs_artifact(
            uri=uri,
            model_name=record.model_name,
            version=record.version,
        )

    if uri.startswith(
        "file://"
    ):
        path = Path(
            urlparse(uri).path
        )
    else:
        path = Path(
            uri
        )

    if not path.exists():
        raise RuntimeError(
            "Model artifact does not exist: "
            f"{path}"
        )

    return path


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------


def load_candidate_model(
    version: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> CandidateModel:
    """
    Load a registered candidate model without changing production state.

    This is intended for pre-promotion validation.

    If version is omitted, the newest registered candidate is loaded.
    """

    record = get_candidate_model_record(
        version=version,
        model_name=model_name,
    )

    artifact_path = resolve_artifact(
        record
    )

    print()
    print("LOADING PETERS CANDIDATE MODEL")
    print("────────────────────────────────────────")
    print(
        f"Model:      {record.model_name}"
    )
    print(
        f"Version:    {record.version}"
    )
    print(
        f"Artifact:   {record.artifact_uri}"
    )
    print(
        f"Local path: {artifact_path}"
    )

    loaded = load_recommender(
        path=artifact_path,
        device=device,
        version=record.version,
    )

    candidate = CandidateModel(
        record=record,
        loaded=loaded,
    )

    print(
        f"Device:     {loaded.device}"
    )
    print(
        f"Schema:     {loaded.feature_schema_version}"
    )
    print(
        "Status:     candidate ready"
    )
    print()

    return candidate


# ---------------------------------------------------------------------------
# Production model manager
# ---------------------------------------------------------------------------


class ProductionModelManager:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device

        self._lock = threading.RLock()

        self._production: (
            ProductionModel | None
        ) = None

    @property
    def production(
        self,
    ) -> ProductionModel | None:
        with self._lock:
            return self._production

    @property
    def version(
        self,
    ) -> str | None:
        production = self.production

        return (
            production.record.version
            if production
            else None
        )

    def load(
        self,
        force: bool = False,
    ) -> ProductionModel:
        with self._lock:
            record = (
                get_production_model_record(
                    self.model_name
                )
            )

            if (
                not force
                and self._production is not None
                and (
                    self._production.record.version
                    == record.version
                )
            ):
                return self._production

            artifact_path = resolve_artifact(
                record
            )

            print()
            print(
                "LOADING PETERS PRODUCTION MODEL"
            )
            print(
                "────────────────────────────────────────"
            )
            print(
                f"Model:      {record.model_name}"
            )
            print(
                f"Version:    {record.version}"
            )
            print(
                f"Artifact:   {record.artifact_uri}"
            )
            print(
                f"Local path: {artifact_path}"
            )

            loaded = load_recommender(
                path=artifact_path,
                device=self.device,
                version=record.version,
            )

            production = ProductionModel(
                record=record,
                loaded=loaded,
            )

            self._production = production

            print(
                f"Device:     {loaded.device}"
            )
            print(
                "Schema:     "
                f"{loaded.feature_schema_version}"
            )
            print(
                "Status:     ready"
            )
            print()

            return production

    def refresh_if_changed(
        self,
    ) -> bool:
        with self._lock:
            record = (
                get_production_model_record(
                    self.model_name
                )
            )

            if (
                self._production is not None
                and (
                    self._production.record.version
                    == record.version
                )
            ):
                return False

            self.load(
                force=True
            )

            return True

    def get(
        self,
    ) -> ProductionModel:
        with self._lock:
            if self._production is None:
                return self.load()

            return self._production


production_model_manager = ProductionModelManager()
