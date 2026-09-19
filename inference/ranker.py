from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

import torch

from features.compatibility import (
    build_compatibility,
    optional_float,
)
from features.definitions import GRAPH_NUMERIC_FIELDS
from inference.feature_store import (
    FeatureStore,
    FeatureStoreResult,
)
from inference.model_loader import (
    CandidateModel,
    ProductionModel,
    ProductionModelManager,
    load_candidate_model,
)
from recommender.checkpoint import LoadedCheckpoint
from recommender.model import (
    DEFAULT_PETERS_POSITIVE_WEIGHTS,
    DEFAULT_PETERS_RISK_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Ranking result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate_user_id: int
    score: float
    predictions: dict[str, float]
    graph_features: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "rank": self.rank,
            "candidate_user_id": self.candidate_user_id,
            "score": self.score,
            "predictions": self.predictions,
        }

        if self.graph_features is not None:
            result["graph_features"] = self.graph_features

        return result


@dataclass(frozen=True)
class RankingResult:
    viewer_user_id: int

    model_name: str
    model_version: str
    model_status: str
    feature_schema_version: int

    graph_enriched: bool
    graph_version_id: int | None
    graph_version: str | None
    graph_schema_version: int | None
    graph_cutoff_at: str | None

    candidates: list[RankedCandidate]

    missing_candidate_ids: list[int]
    excluded_candidate_ids: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "viewer_user_id": self.viewer_user_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_status": self.model_status,
            "feature_schema_version": self.feature_schema_version,
            "graph_enriched": self.graph_enriched,
            "graph_version_id": self.graph_version_id,
            "graph_version": self.graph_version,
            "graph_schema_version": self.graph_schema_version,
            "graph_cutoff_at": self.graph_cutoff_at,
            "candidate_count": len(self.candidates),
            "missing_candidate_ids": self.missing_candidate_ids,
            "excluded_candidate_ids": self.excluded_candidate_ids,
            "candidates": [
                candidate.as_dict()
                for candidate in self.candidates
            ],
        }


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def encode_single(
    vocab: dict[str, Any],
    field: str,
    value: Any,
) -> int:
    if value is None:
        return 0

    text = str(value).strip()

    if not text:
        return 0

    field_vocab = vocab["fields"][field]

    return field_vocab.get(
        text,
        field_vocab["__UNKNOWN__"],
    )


def encode_multi(
    vocab: dict[str, Any],
    field: str,
    values: Any,
) -> list[int]:
    if not values:
        return []

    if not isinstance(values, list):
        values = [values]

    field_vocab = vocab["fields"][field]
    unknown_id = field_vocab["__UNKNOWN__"]

    encoded: list[int] = []

    for value in values:
        text = str(value).strip()

        if text:
            encoded.append(
                field_vocab.get(
                    text,
                    unknown_id,
                )
            )

    return sorted(set(encoded))


def build_categorical(
    rows: list[dict[str, Any]],
    fields: list[str],
    vocab: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return {
        field: torch.tensor(
            [
                encode_single(
                    vocab,
                    field,
                    row.get(field),
                )
                for row in rows
            ],
            dtype=torch.long,
        )
        for field in fields
    }


def build_multi_categorical(
    rows: list[dict[str, Any]],
    fields: list[str],
    vocab: dict[str, Any],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}

    for field in fields:
        encoded = [
            encode_multi(
                vocab,
                field,
                row.get(field),
            )
            for row in rows
        ]

        width = max(
            1,
            max(
                len(values)
                for values in encoded
            ),
        )

        padded = torch.zeros(
            (
                len(rows),
                width,
            ),
            dtype=torch.long,
        )

        for row_index, values in enumerate(encoded):
            if values:
                padded[
                    row_index,
                    :len(values),
                ] = torch.tensor(
                    values,
                    dtype=torch.long,
                )

        result[field] = padded

    return result


# ---------------------------------------------------------------------------
# Numeric encoding
# ---------------------------------------------------------------------------


def build_numeric(
    rows: list[dict[str, Any]],
    fields: list[str],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    values: list[list[float]] = []
    missing: list[list[float]] = []

    for row in rows:
        value_row: list[float] = []
        missing_row: list[float] = []

        for field in fields:
            value = optional_float(
                row.get(field)
            )

            if (
                value is None
                or not math.isfinite(value)
            ):
                value_row.append(
                    float("nan")
                )

                missing_row.append(
                    1.0
                )

            else:
                value_row.append(
                    value
                )

                missing_row.append(
                    0.0
                )

        values.append(
            value_row
        )

        missing.append(
            missing_row
        )

    return (
        torch.tensor(
            values,
            dtype=torch.float32,
        ),
        torch.tensor(
            missing,
            dtype=torch.float32,
        ),
    )


# ---------------------------------------------------------------------------
# Compatibility encoding
# ---------------------------------------------------------------------------


def build_compatibility_tensors(
    rows: list[dict[str, Any]],
    fields: list[str],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    values: list[list[float]] = []
    missing: list[list[float]] = []

    for row in rows:
        compatibility = build_compatibility(
            row
        )

        value_row: list[float] = []
        missing_row: list[float] = []

        for field in fields:
            if field not in compatibility:
                raise RuntimeError(
                    "Compatibility builder does not "
                    "provide model field: "
                    f"{field}"
                )

            value = compatibility[
                field
            ]

            if (
                value is None
                or not math.isfinite(
                    float(value)
                )
            ):
                value_row.append(
                    float("nan")
                )

                missing_row.append(
                    1.0
                )

            else:
                value_row.append(
                    float(value)
                )

                missing_row.append(
                    0.0
                )

        values.append(
            value_row
        )

        missing.append(
            missing_row
        )

    return (
        torch.tensor(
            values,
            dtype=torch.float32,
        ),
        torch.tensor(
            missing,
            dtype=torch.float32,
        ),
    )


# ---------------------------------------------------------------------------
# Optional feature sections
# ---------------------------------------------------------------------------


def build_optional_matrix(
    rows: list[dict[str, Any]],
    section: str,
    fields: list[str],
) -> torch.Tensor | None:
    if not fields:
        return None

    matrix: list[list[float]] = []

    for row in rows:
        values = row.get(
            section
        )

        if not isinstance(
            values,
            dict,
        ):
            raise RuntimeError(
                f"Model expects {section}, "
                "but FeatureStore did not provide it."
            )

        missing = [
            field
            for field in fields
            if field not in values
        ]

        if missing:
            raise RuntimeError(
                f"{section} is missing model fields: "
                + ", ".join(missing)
            )

        matrix.append(
            [
                float(
                    values[field]
                )
                for field in fields
            ]
        )

    return torch.tensor(
        matrix,
        dtype=torch.float32,
    )


def move_tensor(
    tensor: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    return (
        tensor.to(device)
        if tensor is not None
        else None
    )


# ---------------------------------------------------------------------------
# Model input assembly
# ---------------------------------------------------------------------------


def build_model_inputs(
    rows: list[dict[str, Any]],
    model: Any,
    vocab: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    categorical = build_categorical(
        rows,
        model.single_fields,
        vocab,
    )

    multi_categorical = (
        build_multi_categorical(
            rows,
            model.multi_fields,
            vocab,
        )
    )

    numeric, numeric_missing = (
        build_numeric(
            rows,
            model.numeric_fields,
        )
    )

    (
        compatibility,
        compatibility_missing,
    ) = build_compatibility_tensors(
        rows,
        model.compatibility_fields,
    )

    viewer_descriptors = (
        build_optional_matrix(
            rows,
            "viewer_profile_descriptors",
            model.profile_descriptor_fields,
        )
    )

    candidate_descriptors = (
        build_optional_matrix(
            rows,
            "candidate_profile_descriptors",
            model.profile_descriptor_fields,
        )
    )

    viewer_affinities = (
        build_optional_matrix(
            rows,
            "viewer_affinities",
            model.user_affinity_fields,
        )
    )

    candidate_affinities = (
        build_optional_matrix(
            rows,
            "candidate_affinities",
            model.user_affinity_fields,
        )
    )

    return {
        "categorical": {
            field: tensor.to(device)
            for field, tensor
            in categorical.items()
        },
        "multi_categorical": {
            field: tensor.to(device)
            for field, tensor
            in multi_categorical.items()
        },
        "numeric": numeric.to(
            device
        ),
        "numeric_missing": numeric_missing.to(
            device
        ),
        "compatibility": compatibility.to(
            device
        ),
        "compatibility_missing": (
            compatibility_missing.to(
                device
            )
        ),
        "viewer_profile_descriptors": (
            move_tensor(
                viewer_descriptors,
                device,
            )
        ),
        "candidate_profile_descriptors": (
            move_tensor(
                candidate_descriptors,
                device,
            )
        ),
        "viewer_affinities": (
            move_tensor(
                viewer_affinities,
                device,
            )
        ),
        "candidate_affinities": (
            move_tensor(
                candidate_affinities,
                device,
            )
        ),
    }


# ---------------------------------------------------------------------------
# PetersScore
# ---------------------------------------------------------------------------


def checkpoint_score_weights(
    checkpoint: dict[str, Any],
) -> tuple[
    dict[str, float],
    dict[str, float],
]:
    positive = checkpoint.get(
        "peters_positive_weights"
    )

    risk = checkpoint.get(
        "peters_risk_weights"
    )

    if not isinstance(
        positive,
        dict,
    ):
        positive = (
            DEFAULT_PETERS_POSITIVE_WEIGHTS
        )

    if not isinstance(
        risk,
        dict,
    ):
        risk = (
            DEFAULT_PETERS_RISK_WEIGHTS
        )

    return (
        {
            str(key): float(value)
            for key, value
            in positive.items()
        },
        {
            str(key): float(value)
            for key, value
            in risk.items()
        },
    )


def calculate_peters_score(
    probabilities: dict[str, torch.Tensor],
    positive_weights: dict[str, float],
    risk_weights: dict[str, float],
) -> torch.Tensor:
    if not probabilities:
        raise RuntimeError(
            "Model returned no prediction targets."
        )

    sample = next(
        iter(
            probabilities.values()
        )
    )

    positive_total = torch.zeros_like(
        sample
    )

    positive_weight_total = 0.0

    for target, weight in (
        positive_weights.items()
    ):
        if (
            target not in probabilities
            or weight <= 0
        ):
            continue

        positive_total += (
            probabilities[target]
            * weight
        )

        positive_weight_total += (
            weight
        )

    if positive_weight_total <= 0:
        raise RuntimeError(
            "No PetersScore positive targets "
            "exist in this model."
        )

    positive_score = (
        positive_total
        / positive_weight_total
    )

    risk_total = torch.zeros_like(
        sample
    )

    risk_weight_total = 0.0

    for target, weight in (
        risk_weights.items()
    ):
        if (
            target not in probabilities
            or weight <= 0
        ):
            continue

        risk_total += (
            probabilities[target]
            * weight
        )

        risk_weight_total += (
            weight
        )

    if risk_weight_total > 0:
        risk_score = (
            risk_total
            / risk_weight_total
        )
    else:
        risk_score = torch.zeros_like(
            positive_score
        )

    return torch.clamp(
        positive_score
        * (
            1.0
            - risk_score
        ),
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------------
# Graph requirements
# ---------------------------------------------------------------------------


def model_requires_graph(
    loaded: LoadedCheckpoint,
) -> bool:
    numeric_fields = set(
        loaded.numeric_fields
    )

    return any(
        field in numeric_fields
        for field in GRAPH_NUMERIC_FIELDS
    )


def validate_graph_features(
    features: FeatureStoreResult,
    loaded: LoadedCheckpoint,
) -> None:
    if not model_requires_graph(
        loaded
    ):
        return

    if not features.graph_enriched:
        raise RuntimeError(
            "Graph-aware model received a FeatureStore "
            "result without PetersGraph enrichment."
        )

    missing_fields: set[str] = set()

    required_fields = [
        field
        for field in loaded.numeric_fields
        if field.startswith("graph_")
    ]

    for row in features.rows:
        for field in required_fields:
            if field not in row:
                missing_fields.add(
                    field
                )

    if missing_fields:
        raise RuntimeError(
            "FeatureStore is missing required live "
            "PetersGraph fields: "
            + ", ".join(
                sorted(
                    missing_fields
                )
            )
        )


def graph_features_for_row(
    row: dict[str, Any],
    loaded: LoadedCheckpoint,
) -> dict[str, float] | None:
    if not model_requires_graph(
        loaded
    ):
        return None

    result: dict[str, float] = {}

    for field in loaded.numeric_fields:
        if not field.startswith(
            "graph_"
        ):
            continue

        value = optional_float(
            row.get(field)
        )

        if value is None:
            raise RuntimeError(
                "Graph-aware model received invalid "
                f"graph value for {field!r}."
            )

        result[field] = float(
            value
        )

    return result


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class Ranker:
    def __init__(
        self,
        model_manager: ProductionModelManager | None = None,
        feature_store: FeatureStore | None = None,
    ) -> None:
        self.model_manager = (
            model_manager
            or ProductionModelManager()
        )

        self.feature_store = (
            feature_store
            or FeatureStore()
        )

    # ------------------------------------------------------------------
    # Production ranking
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def rank(
        self,
        viewer_user_id: int,
        candidate_user_ids: list[int],
        top_k: int | None = None,
        refresh_model: bool = True,
    ) -> RankingResult:
        if (
            top_k is not None
            and top_k <= 0
        ):
            raise ValueError(
                "top_k must be greater than 0"
            )

        if refresh_model:
            production = (
                self.model_manager.load()
            )

            self.model_manager.refresh_if_changed()

            production = (
                self.model_manager.get()
            )
        else:
            production = (
                self.model_manager.get()
            )

        return self._rank_registered_model(
            registered_model=production,
            model_status="production",
            viewer_user_id=viewer_user_id,
            candidate_user_ids=(
                candidate_user_ids
            ),
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Candidate ranking
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def rank_candidate(
        self,
        viewer_user_id: int,
        candidate_user_ids: list[int],
        version: str | None = None,
        top_k: int | None = None,
        device: str | None = None,
    ) -> RankingResult:
        """
        Rank with a registered candidate model without changing production.

        Used for pre-promotion validation.
        """

        if (
            top_k is not None
            and top_k <= 0
        ):
            raise ValueError(
                "top_k must be greater than 0"
            )

        candidate = load_candidate_model(
            version=version,
            model_name=self.model_manager.model_name,
            device=device,
        )

        return self._rank_registered_model(
            registered_model=candidate,
            model_status="candidate",
            viewer_user_id=viewer_user_id,
            candidate_user_ids=(
                candidate_user_ids
            ),
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Shared ranking implementation
    # ------------------------------------------------------------------

    def _rank_registered_model(
        self,
        registered_model: ProductionModel | CandidateModel,
        model_status: str,
        viewer_user_id: int,
        candidate_user_ids: list[int],
        top_k: int | None,
    ) -> RankingResult:
        loaded = registered_model.loaded

        requires_graph = model_requires_graph(
            loaded
        )

        features = self.feature_store.fetch_pairs(
            viewer_user_id=viewer_user_id,
            candidate_user_ids=candidate_user_ids,
            include_graph_features=requires_graph,
        )

        validate_graph_features(
            features=features,
            loaded=loaded,
        )

        rows: list[dict[str, Any]] = []
        excluded_candidate_ids: list[int] = []

        for row in features.rows:
            candidate_user_id = int(
                row["candidate_user_id"]
            )

            if bool(
                row.get(
                    "graph_is_hard_excluded",
                    False,
                )
            ):
                excluded_candidate_ids.append(
                    candidate_user_id
                )
                continue

            rows.append(
                row
            )

        if not rows:
            return RankingResult(
                viewer_user_id=viewer_user_id,
                model_name=(
                    registered_model.record.model_name
                ),
                model_version=(
                    registered_model.record.version
                ),
                model_status=model_status,
                feature_schema_version=(
                    loaded.feature_schema_version
                ),
                graph_enriched=(
                    features.graph_enriched
                ),
                graph_version_id=(
                    features.graph_version_id
                ),
                graph_version=(
                    features.graph_version
                ),
                graph_schema_version=(
                    features.graph_schema_version
                ),
                graph_cutoff_at=(
                    str(features.graph_cutoff_at)
                    if features.graph_cutoff_at is not None
                    else None
                ),
                candidates=[],
                missing_candidate_ids=(
                    features.missing_candidate_ids
                ),
                excluded_candidate_ids=(
                    excluded_candidate_ids
                ),
            )

        model = loaded.model
        device = loaded.device

        inputs = build_model_inputs(
            rows=rows,
            model=model,
            vocab=loaded.vocab,
            device=device,
        )

        probabilities = (
            model.predict_probabilities(
                **inputs
            )
        )

        (
            positive_weights,
            risk_weights,
        ) = checkpoint_score_weights(
            loaded.checkpoint
        )

        scores = calculate_peters_score(
            probabilities=probabilities,
            positive_weights=positive_weights,
            risk_weights=risk_weights,
        )

        cpu_scores = (
            scores.detach()
            .cpu()
            .tolist()
        )

        cpu_probabilities = {
            target: (
                values.detach()
                .cpu()
                .tolist()
            )
            for target, values
            in probabilities.items()
        }

        candidates: list[
            RankedCandidate
        ] = []

        for index, row in enumerate(
            rows
        ):
            predictions = {
                target: float(
                    cpu_probabilities[
                        target
                    ][index]
                )
                for target in model.target_fields
            }

            graph_features = (
                graph_features_for_row(
                    row=row,
                    loaded=loaded,
                )
            )

            candidates.append(
                RankedCandidate(
                    rank=0,
                    candidate_user_id=int(
                        row[
                            "candidate_user_id"
                        ]
                    ),
                    score=float(
                        cpu_scores[index]
                    ),
                    predictions=predictions,
                    graph_features=(
                        graph_features
                    ),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.score
            ),
            reverse=True,
        )

        if top_k is not None:
            candidates = (
                candidates[:top_k]
            )

        ranked = [
            RankedCandidate(
                rank=index,
                candidate_user_id=(
                    candidate.candidate_user_id
                ),
                score=candidate.score,
                predictions=(
                    candidate.predictions
                ),
                graph_features=(
                    candidate.graph_features
                ),
            )
            for index, candidate
            in enumerate(
                candidates,
                start=1,
            )
        ]

        return RankingResult(
            viewer_user_id=viewer_user_id,
            model_name=(
                registered_model.record.model_name
            ),
            model_version=(
                registered_model.record.version
            ),
            model_status=model_status,
            feature_schema_version=(
                loaded.feature_schema_version
            ),
            graph_enriched=(
                features.graph_enriched
            ),
            graph_version_id=(
                features.graph_version_id
            ),
            graph_version=(
                features.graph_version
            ),
            graph_schema_version=(
                features.graph_schema_version
            ),
            graph_cutoff_at=(
                str(
                    features.graph_cutoff_at
                )
                if features.graph_cutoff_at
                is not None
                else None
            ),
            candidates=ranked,
            missing_candidate_ids=(
                features.missing_candidate_ids
            ),
            excluded_candidate_ids=(
                excluded_candidate_ids
            ),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank Peters candidates with a production "
            "or pre-promotion candidate recommender."
        )
    )

    parser.add_argument(
        "viewer_user_id",
        type=int,
    )

    parser.add_argument(
        "candidate_user_ids",
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-refresh",
        action="store_true",
    )

    parser.add_argument(
        "--candidate-version",
        default=None,
        help=(
            "Rank with this registered candidate version "
            "instead of the production model."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional candidate test device, for example "
            "cpu, mps, or cuda."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ranker = Ranker()

    if args.candidate_version:
        result = ranker.rank_candidate(
            viewer_user_id=(
                args.viewer_user_id
            ),
            candidate_user_ids=(
                args.candidate_user_ids
            ),
            version=(
                args.candidate_version
            ),
            top_k=args.top_k,
            device=args.device,
        )
    else:
        result = ranker.rank(
            viewer_user_id=(
                args.viewer_user_id
            ),
            candidate_user_ids=(
                args.candidate_user_ids
            ),
            top_k=args.top_k,
            refresh_model=(
                not args.no_refresh
            ),
        )

    print()
    print("PETERS RANKING")
    print("────────────────────────────────────────")
    print(
        f"Viewer:        "
        f"{result.viewer_user_id}"
    )
    print(
        f"Model:         "
        f"{result.model_name}"
    )
    print(
        f"Version:       "
        f"{result.model_version}"
    )
    print(
        f"Status:        "
        f"{result.model_status}"
    )
    print(
        f"Schema:        "
        f"{result.feature_schema_version}"
    )
    print(
        f"Graph:         "
        f"{'yes' if result.graph_enriched else 'no'}"
    )

    if result.graph_enriched:
        print(
            f"Graph version: "
            f"{result.graph_version}"
        )

    print(
        f"Candidates:    "
        f"{len(result.candidates)}"
    )

    if result.missing_candidate_ids:
        print(
            "Missing:       "
            + ", ".join(
                str(value)
                for value
                in result.missing_candidate_ids
            )
        )

    if result.excluded_candidate_ids:
        print(
            "Excluded:      "
            + ", ".join(
                str(value)
                for value
                in result.excluded_candidate_ids
            )
        )

    print()
    print(
        json.dumps(
            result.as_dict(),
            indent=2,
        )
    )
    print()


if __name__ == "__main__":
    main()
    