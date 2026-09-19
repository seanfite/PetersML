from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from features.definitions import (
    COMPATIBILITY_FIELDS,
    FEATURE_SCHEMA_VERSION,
    GRAPH_NUMERIC_FIELDS,
    MULTI_CATEGORICAL_FIELDS,
    NUMERIC_INPUT_FIELDS,
    PROFILE_DESCRIPTOR_FIELDS,
    SINGLE_CATEGORICAL_FIELDS,
    TARGET_BOOLEAN_FIELDS,
    USER_AFFINITY_FIELDS,
)
from recommender.model import ModelConfig, PetersRecommender


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: PetersRecommender
    checkpoint: dict[str, Any]
    vocab: dict[str, Any]

    version: str | None
    feature_schema_version: int

    single_categorical_fields: tuple[str, ...]
    multi_categorical_fields: tuple[str, ...]
    numeric_fields: tuple[str, ...]
    compatibility_fields: tuple[str, ...]
    profile_descriptor_fields: tuple[str, ...]
    user_affinity_fields: tuple[str, ...]
    target_fields: tuple[str, ...]

    device: torch.device


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Raw checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint_file(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    required = [
        "model_state_dict",
        "vocab",
        "vocab_sizes",
    ]

    missing = [
        field
        for field in required
        if field not in checkpoint
    ]

    if missing:
        raise RuntimeError(
            "Checkpoint is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(
        checkpoint["model_state_dict"],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint model_state_dict is invalid."
        )

    if not isinstance(
        checkpoint["vocab"],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint vocab is invalid."
        )

    if not isinstance(
        checkpoint["vocab_sizes"],
        dict,
    ):
        raise RuntimeError(
            "Checkpoint vocab_sizes is invalid."
        )

    return checkpoint


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------


def checkpoint_feature_spec(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    spec = checkpoint.get(
        "feature_spec"
    )

    if isinstance(spec, dict):
        return spec

    # Compatibility fallback for checkpoints written before feature_spec
    # became a single saved object.
    return {
        "feature_schema_version": checkpoint.get(
            "feature_schema_version",
            FEATURE_SCHEMA_VERSION,
        ),
        "single_categorical_fields": checkpoint.get(
            "single_categorical_fields",
            list(SINGLE_CATEGORICAL_FIELDS),
        ),
        "multi_categorical_fields": checkpoint.get(
            "multi_categorical_fields",
            list(MULTI_CATEGORICAL_FIELDS),
        ),
        "numeric_fields": checkpoint.get(
            "numeric_fields",
            list(NUMERIC_INPUT_FIELDS),
        ),
        "compatibility_fields": checkpoint.get(
            "compatibility_fields",
            list(COMPATIBILITY_FIELDS),
        ),
        "profile_descriptor_fields": checkpoint.get(
            "profile_descriptor_fields",
            list(PROFILE_DESCRIPTOR_FIELDS),
        ),
        "user_affinity_fields": checkpoint.get(
            "user_affinity_fields",
            list(USER_AFFINITY_FIELDS),
        ),
        "target_fields": checkpoint.get(
            "target_fields",
            list(TARGET_BOOLEAN_FIELDS),
        ),
    }


def feature_list(
    feature_spec: dict[str, Any],
    name: str,
) -> list[str]:
    value = feature_spec.get(
        name
    )

    if not isinstance(value, list):
        raise RuntimeError(
            f"Checkpoint feature spec {name!r} must be a list."
        )

    if any(
        not isinstance(item, str)
        or not item.strip()
        for item in value
    ):
        raise RuntimeError(
            f"Checkpoint feature spec {name!r} "
            "contains an invalid field name."
        )

    if len(value) != len(set(value)):
        raise RuntimeError(
            f"Checkpoint feature spec {name!r} "
            "contains duplicate fields."
        )

    return list(value)


def checkpoint_schema_version(
    checkpoint: dict[str, Any],
    feature_spec: dict[str, Any],
) -> int:
    raw_version = feature_spec.get(
        "feature_schema_version",
        checkpoint.get(
            "feature_schema_version"
        ),
    )

    if raw_version is None:
        raise RuntimeError(
            "Checkpoint is missing feature_schema_version."
        )

    try:
        schema_version = int(
            raw_version
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Checkpoint feature_schema_version is invalid."
        ) from error

    top_level_version = checkpoint.get(
        "feature_schema_version"
    )

    if (
        top_level_version is not None
        and int(top_level_version) != schema_version
    ):
        raise RuntimeError(
            "Checkpoint feature schema versions disagree: "
            f"feature_spec={schema_version}, "
            f"checkpoint={top_level_version}"
        )

    return schema_version


# ---------------------------------------------------------------------------
# Current-schema contract
# ---------------------------------------------------------------------------


def require_exact_fields(
    name: str,
    actual: list[str],
    expected: list[str],
) -> None:
    if actual == expected:
        return

    actual_set = set(
        actual
    )
    expected_set = set(
        expected
    )

    missing = [
        field
        for field in expected
        if field not in actual_set
    ]

    unexpected = [
        field
        for field in actual
        if field not in expected_set
    ]

    if (
        not missing
        and not unexpected
    ):
        detail = (
            "same fields but different ordering"
        )
    else:
        parts: list[str] = []

        if missing:
            parts.append(
                "missing="
                + ", ".join(missing)
            )

        if unexpected:
            parts.append(
                "unexpected="
                + ", ".join(unexpected)
            )

        detail = "; ".join(
            parts
        )

    raise RuntimeError(
        f"Checkpoint {name} does not match "
        f"feature schema {FEATURE_SCHEMA_VERSION}: "
        f"{detail}"
    )


def validate_current_schema_contract(
    schema_version: int,
    single_fields: list[str],
    multi_fields: list[str],
    numeric_fields: list[str],
    compatibility_fields: list[str],
    profile_descriptor_fields: list[str],
    user_affinity_fields: list[str],
    target_fields: list[str],
) -> None:
    """
    Enforce the exact application feature contract for checkpoints that
    declare the current feature schema.

    Older production checkpoints remain loadable until they are retired.
    """

    if schema_version != FEATURE_SCHEMA_VERSION:
        return

    require_exact_fields(
        "single_categorical_fields",
        single_fields,
        list(SINGLE_CATEGORICAL_FIELDS),
    )

    require_exact_fields(
        "multi_categorical_fields",
        multi_fields,
        list(MULTI_CATEGORICAL_FIELDS),
    )

    require_exact_fields(
        "numeric_fields",
        numeric_fields,
        list(NUMERIC_INPUT_FIELDS),
    )

    require_exact_fields(
        "compatibility_fields",
        compatibility_fields,
        list(COMPATIBILITY_FIELDS),
    )

    require_exact_fields(
        "profile_descriptor_fields",
        profile_descriptor_fields,
        list(PROFILE_DESCRIPTOR_FIELDS),
    )

    require_exact_fields(
        "user_affinity_fields",
        user_affinity_fields,
        list(USER_AFFINITY_FIELDS),
    )

    require_exact_fields(
        "target_fields",
        target_fields,
        list(TARGET_BOOLEAN_FIELDS),
    )

    missing_graph_fields = [
        field
        for field in GRAPH_NUMERIC_FIELDS
        if field not in numeric_fields
    ]

    if missing_graph_fields:
        raise RuntimeError(
            "Current-schema checkpoint is missing "
            "PetersGraph features: "
            + ", ".join(missing_graph_fields)
        )

    if "graph_is_hard_excluded" in numeric_fields:
        raise RuntimeError(
            "graph_is_hard_excluded must remain control "
            "metadata and cannot be a neural input."
        )


# ---------------------------------------------------------------------------
# Vocabulary validation
# ---------------------------------------------------------------------------


def validate_vocab(
    vocab: dict[str, Any],
    vocab_sizes: dict[str, Any],
    single_fields: list[str],
    multi_fields: list[str],
    schema_version: int,
) -> None:
    fields = vocab.get(
        "fields"
    )

    if not isinstance(fields, dict):
        raise RuntimeError(
            "Checkpoint vocabulary is missing fields."
        )

    vocab_schema = vocab.get(
        "feature_schema_version"
    )

    if (
        vocab_schema is not None
        and int(vocab_schema) != schema_version
    ):
        raise RuntimeError(
            "Checkpoint vocabulary schema does not "
            "match checkpoint feature schema: "
            f"vocab={vocab_schema}, "
            f"checkpoint={schema_version}"
        )

    for field in (
        single_fields
        + multi_fields
    ):
        if field not in fields:
            raise RuntimeError(
                "Checkpoint vocabulary is missing field: "
                f"{field}"
            )

        values = fields[
            field
        ]

        if not isinstance(
            values,
            dict,
        ):
            raise RuntimeError(
                f"Vocabulary field {field!r} is invalid."
            )

        if values.get(
            "__PAD__"
        ) != 0:
            raise RuntimeError(
                f"Vocabulary field {field!r} must use "
                "__PAD__ = 0."
            )

        if "__UNKNOWN__" not in values:
            raise RuntimeError(
                f"Vocabulary field {field!r} is missing "
                "__UNKNOWN__."
            )

        saved_size = vocab_sizes.get(
            field
        )

        if saved_size is None:
            raise RuntimeError(
                "Checkpoint vocab_sizes is missing field: "
                f"{field}"
            )

        try:
            saved_size = int(
                saved_size
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Checkpoint vocab size for {field!r} "
                "is invalid."
            ) from error

        actual_size = len(
            values
        )

        if saved_size != actual_size:
            raise RuntimeError(
                f"Checkpoint vocab size mismatch for {field!r}: "
                f"saved={saved_size}, actual={actual_size}"
            )


# ---------------------------------------------------------------------------
# Numeric normalization validation
# ---------------------------------------------------------------------------


def validate_numeric_tensor(
    checkpoint: dict[str, Any],
    name: str,
    expected_size: int,
) -> None:
    value = checkpoint.get(
        name
    )

    if value is None:
        raise RuntimeError(
            f"Checkpoint is missing {name}."
        )

    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise RuntimeError(
            f"Checkpoint {name} must be a torch.Tensor."
        )

    if value.ndim != 1:
        raise RuntimeError(
            f"Checkpoint {name} must be one-dimensional."
        )

    if value.numel() != expected_size:
        raise RuntimeError(
            f"Checkpoint {name} has length "
            f"{value.numel()}, expected {expected_size}."
        )

    if not torch.isfinite(
        value
    ).all():
        raise RuntimeError(
            f"Checkpoint {name} contains non-finite values."
        )


def validate_numeric_state(
    checkpoint: dict[str, Any],
    numeric_fields: list[str],
) -> None:
    numeric_count = len(
        numeric_fields
    )

    validate_numeric_tensor(
        checkpoint,
        "numeric_impute",
        numeric_count,
    )

    validate_numeric_tensor(
        checkpoint,
        "numeric_mean",
        numeric_count,
    )

    validate_numeric_tensor(
        checkpoint,
        "numeric_std",
        numeric_count,
    )

    numeric_std = checkpoint[
        "numeric_std"
    ]

    if torch.any(
        numeric_std <= 0
    ):
        raise RuntimeError(
            "Checkpoint numeric_std must contain "
            "only positive values."
        )


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


def checkpoint_model_config(
    checkpoint: dict[str, Any],
) -> ModelConfig:
    raw_config = checkpoint.get(
        "model_config"
    )

    if raw_config is None:
        return ModelConfig()

    if not isinstance(
        raw_config,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint model_config must be a dictionary."
        )

    try:
        return ModelConfig(
            **raw_config
        )
    except TypeError as error:
        raise RuntimeError(
            "Checkpoint contains an incompatible "
            f"model_config: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


def validate_checkpoint_contract(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    feature_spec = checkpoint_feature_spec(
        checkpoint
    )

    schema_version = checkpoint_schema_version(
        checkpoint,
        feature_spec,
    )

    single_fields = feature_list(
        feature_spec,
        "single_categorical_fields",
    )

    multi_fields = feature_list(
        feature_spec,
        "multi_categorical_fields",
    )

    numeric_fields = feature_list(
        feature_spec,
        "numeric_fields",
    )

    compatibility_fields = feature_list(
        feature_spec,
        "compatibility_fields",
    )

    profile_descriptor_fields = feature_list(
        feature_spec,
        "profile_descriptor_fields",
    )

    user_affinity_fields = feature_list(
        feature_spec,
        "user_affinity_fields",
    )

    target_fields = feature_list(
        feature_spec,
        "target_fields",
    )

    if not numeric_fields:
        raise RuntimeError(
            "Checkpoint contains no numeric fields."
        )

    if not target_fields:
        raise RuntimeError(
            "Checkpoint contains no prediction targets."
        )

    validate_current_schema_contract(
        schema_version=schema_version,
        single_fields=single_fields,
        multi_fields=multi_fields,
        numeric_fields=numeric_fields,
        compatibility_fields=compatibility_fields,
        profile_descriptor_fields=(
            profile_descriptor_fields
        ),
        user_affinity_fields=user_affinity_fields,
        target_fields=target_fields,
    )

    vocab = checkpoint[
        "vocab"
    ]

    vocab_sizes = checkpoint[
        "vocab_sizes"
    ]

    validate_vocab(
        vocab=vocab,
        vocab_sizes=vocab_sizes,
        single_fields=single_fields,
        multi_fields=multi_fields,
        schema_version=schema_version,
    )

    validate_numeric_state(
        checkpoint=checkpoint,
        numeric_fields=numeric_fields,
    )

    return {
        "feature_spec": feature_spec,
        "feature_schema_version": schema_version,
        "single_categorical_fields": single_fields,
        "multi_categorical_fields": multi_fields,
        "numeric_fields": numeric_fields,
        "compatibility_fields": compatibility_fields,
        "profile_descriptor_fields": (
            profile_descriptor_fields
        ),
        "user_affinity_fields": user_affinity_fields,
        "target_fields": target_fields,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
) -> PetersRecommender:
    contract = validate_checkpoint_contract(
        checkpoint
    )

    single_fields = contract[
        "single_categorical_fields"
    ]

    multi_fields = contract[
        "multi_categorical_fields"
    ]

    numeric_fields = contract[
        "numeric_fields"
    ]

    compatibility_fields = contract[
        "compatibility_fields"
    ]

    profile_descriptor_fields = contract[
        "profile_descriptor_fields"
    ]

    user_affinity_fields = contract[
        "user_affinity_fields"
    ]

    target_fields = contract[
        "target_fields"
    ]

    feature_schema_version = contract[
        "feature_schema_version"
    ]

    vocab_sizes = {
        field: int(size)
        for field, size
        in checkpoint["vocab_sizes"].items()
    }

    model = PetersRecommender(
        vocab_sizes=vocab_sizes,
        config=checkpoint_model_config(
            checkpoint
        ),
        single_fields=single_fields,
        multi_fields=multi_fields,
        numeric_fields=numeric_fields,
        compatibility_fields=compatibility_fields,
        profile_descriptor_fields=(
            profile_descriptor_fields
        ),
        user_affinity_fields=(
            user_affinity_fields
        ),
        target_fields=target_fields,
        feature_schema_version=(
            feature_schema_version
        ),
    )

    try:
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint state does not match the "
            "saved model architecture. "
            f"{error}"
        ) from error

    return model


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_recommender(
    path: str | Path,
    device: torch.device | str | None = None,
    version: str | None = None,
) -> LoadedCheckpoint:
    checkpoint_path = Path(
        path
    )

    checkpoint = load_checkpoint_file(
        checkpoint_path
    )

    contract = validate_checkpoint_contract(
        checkpoint
    )

    if device is None:
        resolved_device = choose_device()
    else:
        resolved_device = torch.device(
            device
        )

    model = build_model_from_checkpoint(
        checkpoint
    )

    model = model.to(
        resolved_device
    )

    model.eval()

    return LoadedCheckpoint(
        model=model,
        checkpoint=checkpoint,
        vocab=checkpoint["vocab"],
        version=version,
        feature_schema_version=contract[
            "feature_schema_version"
        ],
        single_categorical_fields=tuple(
            contract[
                "single_categorical_fields"
            ]
        ),
        multi_categorical_fields=tuple(
            contract[
                "multi_categorical_fields"
            ]
        ),
        numeric_fields=tuple(
            contract[
                "numeric_fields"
            ]
        ),
        compatibility_fields=tuple(
            contract[
                "compatibility_fields"
            ]
        ),
        profile_descriptor_fields=tuple(
            contract[
                "profile_descriptor_fields"
            ]
        ),
        user_affinity_fields=tuple(
            contract[
                "user_affinity_fields"
            ]
        ),
        target_fields=tuple(
            contract[
                "target_fields"
            ]
        ),
        device=resolved_device,
    )
