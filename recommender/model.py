from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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

DEFAULT_VOCAB = "data/vocab.json"


# ---------------------------------------------------------------------------
# Production Peters ranking score
#
# These are not training-loss weights.
#
# The neural network predicts each outcome independently. These weights turn
# those probabilities into one product ranking score for Suggestions/Search AI.
# ---------------------------------------------------------------------------

DEFAULT_PETERS_POSITIVE_WEIGHTS = {
    "outcome_favorite": 0.8,
    "outcome_poke": 0.3,
    "outcome_image_reaction": 0.6,
    "outcome_story_engagement": 0.4,
    "outcome_message_started": 1.5,
    "outcome_reciprocal_message": 2.5,
    "outcome_private_album_grant": 0.4,
    "outcome_partner_accepted": 1.5,
}

DEFAULT_PETERS_RISK_WEIGHTS = {
    "outcome_block": 1.0,
    "outcome_report": 2.0,
}


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 16

    numeric_hidden_dim: int = 64
    compatibility_hidden_dim: int = 48

    descriptor_hidden_dim: int = 32
    affinity_hidden_dim: int = 32

    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    hidden_dim_3: int = 64

    dropout: float = 0.20


# ---------------------------------------------------------------------------
# Multi-value categorical embedding
# ---------------------------------------------------------------------------

class MeanPooledEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

    def forward(
        self,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError(
                "Multi-categorical inputs must have shape "
                "[batch_size, max_values]."
            )

        embedded = self.embedding(ids)

        mask = ids.ne(0).unsqueeze(-1)

        summed = (
            embedded * mask
        ).sum(dim=1)

        counts = (
            mask.sum(dim=1)
            .clamp_min(1)
        )

        return summed / counts


# ---------------------------------------------------------------------------
# Peters recommender
# ---------------------------------------------------------------------------

class PetersRecommender(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        config: ModelConfig | None = None,
        *,
        feature_schema_version: int = FEATURE_SCHEMA_VERSION,
        single_fields: list[str] | None = None,
        multi_fields: list[str] | None = None,
        numeric_fields: list[str] | None = None,
        compatibility_fields: list[str] | None = None,
        profile_descriptor_fields: list[str] | None = None,
        user_affinity_fields: list[str] | None = None,
        target_fields: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config
            or ModelConfig()
        )

        self.feature_schema_version = (
            feature_schema_version
        )

        self.single_fields = list(
            SINGLE_CATEGORICAL_FIELDS
            if single_fields is None
            else single_fields
        )

        self.multi_fields = list(
            MULTI_CATEGORICAL_FIELDS
            if multi_fields is None
            else multi_fields
        )

        self.numeric_fields = list(
            NUMERIC_INPUT_FIELDS
            if numeric_fields is None
            else numeric_fields
        )

        self.compatibility_fields = list(
            COMPATIBILITY_FIELDS
            if compatibility_fields is None
            else compatibility_fields
        )

        self.profile_descriptor_fields = list(
            PROFILE_DESCRIPTOR_FIELDS
            if profile_descriptor_fields is None
            else profile_descriptor_fields
        )

        self.user_affinity_fields = list(
            USER_AFFINITY_FIELDS
            if user_affinity_fields is None
            else user_affinity_fields
        )

        self.target_fields = list(
            TARGET_BOOLEAN_FIELDS
            if target_fields is None
            else target_fields
        )

        self._validate_vocab_sizes(
            vocab_sizes
        )

        # ---------------------------------------------------------------
        # Single categorical embeddings
        # ---------------------------------------------------------------

        self.single_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(
                    num_embeddings=vocab_sizes[field],
                    embedding_dim=self.config.embedding_dim,
                    padding_idx=0,
                )
                for field in self.single_fields
            }
        )

        # ---------------------------------------------------------------
        # Multi-select categorical embeddings
        # ---------------------------------------------------------------

        self.multi_embeddings = nn.ModuleDict(
            {
                field: MeanPooledEmbedding(
                    vocab_size=vocab_sizes[field],
                    embedding_dim=self.config.embedding_dim,
                )
                for field in self.multi_fields
            }
        )

        # ---------------------------------------------------------------
        # Numeric features
        #
        # train.py installs:
        # - imputation values
        # - mean
        # - standard deviation
        #
        # Missingness is also passed as its own signal.
        # ---------------------------------------------------------------

        numeric_count = len(
            self.numeric_fields
        )

        self.register_buffer(
            "numeric_impute",
            torch.zeros(
                numeric_count,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "numeric_mean",
            torch.zeros(
                numeric_count,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "numeric_std",
            torch.ones(
                numeric_count,
                dtype=torch.float32,
            ),
        )

        self.numeric_encoder = nn.Sequential(
            nn.Linear(
                numeric_count * 2,
                self.config.numeric_hidden_dim,
            ),
            nn.LayerNorm(
                self.config.numeric_hidden_dim
            ),
            nn.GELU(),
        )

        # ---------------------------------------------------------------
        # Explicit viewer/candidate compatibility
        #
        # Input:
        #   compatibility values
        #   compatibility missing mask
        # ---------------------------------------------------------------

        compatibility_count = len(
            self.compatibility_fields
        )

        self.compatibility_encoder: nn.Module | None

        if compatibility_count > 0:
            self.compatibility_encoder = (
                nn.Sequential(
                    nn.Linear(
                        compatibility_count * 2,
                        self.config.compatibility_hidden_dim,
                    ),
                    nn.LayerNorm(
                        self.config.compatibility_hidden_dim
                    ),
                    nn.GELU(),
                )
            )
        else:
            self.compatibility_encoder = None

        # ---------------------------------------------------------------
        # Future image/profile descriptor representation
        #
        # Same encoder is shared by viewer and candidate because both
        # representations use the same descriptor feature space.
        # ---------------------------------------------------------------

        descriptor_count = len(
            self.profile_descriptor_fields
        )

        self.profile_descriptor_encoder: nn.Module | None

        if descriptor_count > 0:
            self.profile_descriptor_encoder = (
                nn.Sequential(
                    nn.Linear(
                        descriptor_count,
                        self.config.descriptor_hidden_dim,
                    ),
                    nn.LayerNorm(
                        self.config.descriptor_hidden_dim
                    ),
                    nn.GELU(),
                )
            )
        else:
            self.profile_descriptor_encoder = None

        # ---------------------------------------------------------------
        # Future learned taste / affinity representation
        #
        # Same feature space is used for both users in the pair.
        # ---------------------------------------------------------------

        affinity_count = len(
            self.user_affinity_fields
        )

        self.affinity_encoder: nn.Module | None

        if affinity_count > 0:
            self.affinity_encoder = (
                nn.Sequential(
                    nn.Linear(
                        affinity_count,
                        self.config.affinity_hidden_dim,
                    ),
                    nn.LayerNorm(
                        self.config.affinity_hidden_dim
                    ),
                    nn.GELU(),
                )
            )
        else:
            self.affinity_encoder = None

        # ---------------------------------------------------------------
        # Calculate combined representation size
        # ---------------------------------------------------------------

        categorical_feature_count = (
            len(self.single_fields)
            + len(self.multi_fields)
        )

        categorical_size = (
            categorical_feature_count
            * self.config.embedding_dim
        )

        combined_size = (
            categorical_size
            + self.config.numeric_hidden_dim
        )

        if self.compatibility_encoder is not None:
            combined_size += (
                self.config.compatibility_hidden_dim
            )

        if self.profile_descriptor_encoder is not None:
            # Viewer + candidate representations.
            combined_size += (
                self.config.descriptor_hidden_dim
                * 2
            )

        if self.affinity_encoder is not None:
            # Viewer + candidate learned tastes.
            combined_size += (
                self.config.affinity_hidden_dim
                * 2
            )

        # ---------------------------------------------------------------
        # Shared neural representation
        # ---------------------------------------------------------------

        self.trunk = nn.Sequential(
            nn.Linear(
                combined_size,
                self.config.hidden_dim_1,
            ),
            nn.LayerNorm(
                self.config.hidden_dim_1
            ),
            nn.GELU(),
            nn.Dropout(
                self.config.dropout
            ),

            nn.Linear(
                self.config.hidden_dim_1,
                self.config.hidden_dim_2,
            ),
            nn.LayerNorm(
                self.config.hidden_dim_2
            ),
            nn.GELU(),
            nn.Dropout(
                self.config.dropout
            ),

            nn.Linear(
                self.config.hidden_dim_2,
                self.config.hidden_dim_3,
            ),
            nn.LayerNorm(
                self.config.hidden_dim_3
            ),
            nn.GELU(),
        )

        # ---------------------------------------------------------------
        # Multi-task heads
        # ---------------------------------------------------------------

        self.heads = nn.ModuleDict(
            {
                target: nn.Linear(
                    self.config.hidden_dim_3,
                    1,
                )
                for target in self.target_fields
            }
        )

    # ------------------------------------------------------------------
    # Feature specification
    # ------------------------------------------------------------------

    def feature_spec(
        self,
    ) -> dict[str, Any]:
        return {
            "feature_schema_version":
                self.feature_schema_version,

            "single_categorical_fields":
                list(self.single_fields),

            "multi_categorical_fields":
                list(self.multi_fields),

            "numeric_fields":
                list(self.numeric_fields),

            "compatibility_fields":
                list(self.compatibility_fields),

            "profile_descriptor_fields":
                list(
                    self.profile_descriptor_fields
                ),

            "user_affinity_fields":
                list(self.user_affinity_fields),

            "target_fields":
                list(self.target_fields),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_vocab_sizes(
        self,
        vocab_sizes: dict[str, int],
    ) -> None:
        required_fields = (
            self.single_fields
            + self.multi_fields
        )

        missing = [
            field
            for field in required_fields
            if field not in vocab_sizes
        ]

        if missing:
            raise ValueError(
                "Missing vocabulary sizes for: "
                + ", ".join(missing)
            )

        invalid = [
            field
            for field in required_fields
            if vocab_sizes[field] < 2
        ]

        if invalid:
            raise ValueError(
                "Vocabulary sizes must include "
                "PAD and UNKNOWN for: "
                + ", ".join(invalid)
            )

    @staticmethod
    def _validate_matrix(
        name: str,
        tensor: torch.Tensor,
        batch_size: int,
        width: int,
    ) -> None:
        expected = (
            batch_size,
            width,
        )

        if tensor.shape != expected:
            raise ValueError(
                f"{name} must have shape "
                f"{expected}, got "
                f"{tuple(tensor.shape)}."
            )

    # ------------------------------------------------------------------
    # Numeric statistics
    # ------------------------------------------------------------------

    def set_numeric_stats(
        self,
        impute: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        expected = len(
            self.numeric_fields
        )

        expected_shape = (
            expected,
        )

        if impute.shape != expected_shape:
            raise ValueError(
                "Expected numeric impute shape "
                f"{expected_shape}, got "
                f"{tuple(impute.shape)}."
            )

        if mean.shape != expected_shape:
            raise ValueError(
                "Expected numeric mean shape "
                f"{expected_shape}, got "
                f"{tuple(mean.shape)}."
            )

        if std.shape != expected_shape:
            raise ValueError(
                "Expected numeric std shape "
                f"{expected_shape}, got "
                f"{tuple(std.shape)}."
            )

        safe_std = (
            std.float()
            .clamp_min(1e-6)
        )

        self.numeric_impute.copy_(
            impute.to(
                device=self.numeric_impute.device,
                dtype=torch.float32,
            )
        )

        self.numeric_mean.copy_(
            mean.to(
                device=self.numeric_mean.device,
                dtype=torch.float32,
            )
        )

        self.numeric_std.copy_(
            safe_std.to(
                device=self.numeric_std.device,
                dtype=torch.float32,
            )
        )

    # ------------------------------------------------------------------
    # Shared encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        categorical: dict[str, torch.Tensor],
        multi_categorical: dict[str, torch.Tensor],
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        compatibility: torch.Tensor,
        compatibility_missing: torch.Tensor,
        viewer_profile_descriptors: torch.Tensor | None = None,
        candidate_profile_descriptors: torch.Tensor | None = None,
        viewer_affinities: torch.Tensor | None = None,
        candidate_affinities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if numeric.ndim != 2:
            raise ValueError(
                "numeric must have shape "
                "[batch_size, numeric_feature_count]."
            )

        batch_size = (
            numeric.shape[0]
        )

        self._validate_matrix(
            "numeric",
            numeric,
            batch_size,
            len(self.numeric_fields),
        )

        self._validate_matrix(
            "numeric_missing",
            numeric_missing,
            batch_size,
            len(self.numeric_fields),
        )

        self._validate_matrix(
            "compatibility",
            compatibility,
            batch_size,
            len(self.compatibility_fields),
        )

        self._validate_matrix(
            "compatibility_missing",
            compatibility_missing,
            batch_size,
            len(self.compatibility_fields),
        )

        features: list[torch.Tensor] = []

        # ---------------------------------------------------------------
        # Single categorical fields
        # ---------------------------------------------------------------

        for field in self.single_fields:
            if field not in categorical:
                raise KeyError(
                    "Missing categorical field: "
                    f"{field}"
                )

            values = categorical[field]

            if values.shape != (
                batch_size,
            ):
                raise ValueError(
                    f"{field} must have shape "
                    f"({batch_size},), got "
                    f"{tuple(values.shape)}."
                )

            features.append(
                self.single_embeddings[field](
                    values
                )
            )

        # ---------------------------------------------------------------
        # Multi-select categorical fields
        # ---------------------------------------------------------------

        for field in self.multi_fields:
            if field not in multi_categorical:
                raise KeyError(
                    "Missing multi-categorical "
                    f"field: {field}"
                )

            values = (
                multi_categorical[field]
            )

            if values.ndim != 2:
                raise ValueError(
                    f"{field} must have shape "
                    "[batch_size, max_values]."
                )

            if (
                values.shape[0]
                != batch_size
            ):
                raise ValueError(
                    f"{field} has wrong "
                    "batch size."
                )

            features.append(
                self.multi_embeddings[field](
                    values
                )
            )

        # ---------------------------------------------------------------
        # Numeric features
        # ---------------------------------------------------------------

        numeric = numeric.float()
        numeric_missing = (
            numeric_missing.float()
            .clamp(0.0, 1.0)
        )

        numeric_missing_mask = (
            numeric_missing.ge(0.5)
        )

        numeric_nonfinite = (
            ~torch.isfinite(numeric)
        )

        unexpected_nonfinite = (
            numeric_nonfinite
            & ~numeric_missing_mask
        )

        if unexpected_nonfinite.any():
            raise ValueError(
                "numeric contains non-finite "
                "values that are not marked "
                "missing."
            )

        safe_numeric = torch.where(
            torch.isfinite(numeric),
            numeric,
            torch.zeros_like(numeric),
        )

        numeric_impute = (
            self.numeric_impute
            .unsqueeze(0)
            .expand_as(safe_numeric)
        )

        filled_numeric = torch.where(
            numeric_missing_mask,
            numeric_impute,
            safe_numeric,
        )

        normalized_numeric = (
            filled_numeric
            - self.numeric_mean
        ) / self.numeric_std

        numeric_input = torch.cat(
            [
                normalized_numeric,
                numeric_missing,
            ],
            dim=1,
        )

        features.append(
            self.numeric_encoder(
                numeric_input
            )
        )

        # ---------------------------------------------------------------
        # Compatibility features
        # ---------------------------------------------------------------

        if (
            self.compatibility_encoder
            is not None
        ):
            compatibility = (
                compatibility.float()
            )

            compatibility_missing = (
                compatibility_missing.float()
                .clamp(0.0, 1.0)
            )

            compatibility_missing_mask = (
                compatibility_missing.ge(0.5)
            )

            compatibility_nonfinite = (
                ~torch.isfinite(
                    compatibility
                )
            )

            unexpected_nonfinite = (
                compatibility_nonfinite
                & ~compatibility_missing_mask
            )

            if unexpected_nonfinite.any():
                raise ValueError(
                    "compatibility contains "
                    "non-finite values that are "
                    "not marked missing."
                )

            safe_compatibility = torch.where(
                torch.isfinite(
                    compatibility
                ),
                compatibility,
                torch.zeros_like(
                    compatibility
                ),
            )

            filled_compatibility = (
                torch.where(
                    compatibility_missing_mask,
                    torch.zeros_like(
                        safe_compatibility
                    ),
                    safe_compatibility,
                )
            )

            compatibility_input = (
                torch.cat(
                    [
                        filled_compatibility,
                        compatibility_missing,
                    ],
                    dim=1,
                )
            )

            features.append(
                self.compatibility_encoder(
                    compatibility_input
                )
            )

        # ---------------------------------------------------------------
        # Profile/image descriptor features
        # ---------------------------------------------------------------

        descriptor_count = len(
            self.profile_descriptor_fields
        )

        if (
            self.profile_descriptor_encoder
            is not None
        ):
            if (
                viewer_profile_descriptors
                is None
                or candidate_profile_descriptors
                is None
            ):
                raise ValueError(
                    "Profile descriptor tensors "
                    "are required by this model."
                )

            self._validate_matrix(
                "viewer_profile_descriptors",
                viewer_profile_descriptors,
                batch_size,
                descriptor_count,
            )

            self._validate_matrix(
                "candidate_profile_descriptors",
                candidate_profile_descriptors,
                batch_size,
                descriptor_count,
            )

            features.append(
                self.profile_descriptor_encoder(
                    viewer_profile_descriptors.float()
                )
            )

            features.append(
                self.profile_descriptor_encoder(
                    candidate_profile_descriptors.float()
                )
            )

        # ---------------------------------------------------------------
        # Learned user affinity/taste features
        # ---------------------------------------------------------------

        affinity_count = len(
            self.user_affinity_fields
        )

        if self.affinity_encoder is not None:
            if (
                viewer_affinities is None
                or candidate_affinities is None
            ):
                raise ValueError(
                    "Affinity tensors are "
                    "required by this model."
                )

            self._validate_matrix(
                "viewer_affinities",
                viewer_affinities,
                batch_size,
                affinity_count,
            )

            self._validate_matrix(
                "candidate_affinities",
                candidate_affinities,
                batch_size,
                affinity_count,
            )

            features.append(
                self.affinity_encoder(
                    viewer_affinities.float()
                )
            )

            features.append(
                self.affinity_encoder(
                    candidate_affinities.float()
                )
            )

        # ---------------------------------------------------------------
        # Shared representation
        # ---------------------------------------------------------------

        combined = torch.cat(
            features,
            dim=1,
        )

        return self.trunk(
            combined
        )

    # ------------------------------------------------------------------
    # Multi-task predictions
    # ------------------------------------------------------------------

    def forward(
        self,
        categorical: dict[str, torch.Tensor],
        multi_categorical: dict[str, torch.Tensor],
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        compatibility: torch.Tensor,
        compatibility_missing: torch.Tensor,
        viewer_profile_descriptors: torch.Tensor | None = None,
        candidate_profile_descriptors: torch.Tensor | None = None,
        viewer_affinities: torch.Tensor | None = None,
        candidate_affinities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        shared = self.encode(
            categorical=categorical,
            multi_categorical=multi_categorical,
            numeric=numeric,
            numeric_missing=numeric_missing,
            compatibility=compatibility,
            compatibility_missing=compatibility_missing,
            viewer_profile_descriptors=viewer_profile_descriptors,
            candidate_profile_descriptors=candidate_profile_descriptors,
            viewer_affinities=viewer_affinities,
            candidate_affinities=candidate_affinities,
        )

        return {
            target: (
                self.heads[target](
                    shared
                )
                .squeeze(-1)
            )
            for target in self.target_fields
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_probabilities(
        self,
        categorical: dict[str, torch.Tensor],
        multi_categorical: dict[str, torch.Tensor],
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        compatibility: torch.Tensor,
        compatibility_missing: torch.Tensor,
        viewer_profile_descriptors: torch.Tensor | None = None,
        candidate_profile_descriptors: torch.Tensor | None = None,
        viewer_affinities: torch.Tensor | None = None,
        candidate_affinities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.eval()

        logits = self.forward(
            categorical=categorical,
            multi_categorical=multi_categorical,
            numeric=numeric,
            numeric_missing=numeric_missing,
            compatibility=compatibility,
            compatibility_missing=compatibility_missing,
            viewer_profile_descriptors=viewer_profile_descriptors,
            candidate_profile_descriptors=candidate_profile_descriptors,
            viewer_affinities=viewer_affinities,
            candidate_affinities=candidate_affinities,
        )

        return {
            target: torch.sigmoid(value)
            for target, value in logits.items()
        }

    @torch.no_grad()
    def predict_with_score(
        self,
        categorical: dict[str, torch.Tensor],
        multi_categorical: dict[str, torch.Tensor],
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        compatibility: torch.Tensor,
        compatibility_missing: torch.Tensor,
        viewer_profile_descriptors: torch.Tensor | None = None,
        candidate_profile_descriptors: torch.Tensor | None = None,
        viewer_affinities: torch.Tensor | None = None,
        candidate_affinities: torch.Tensor | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        probabilities = (
            self.predict_probabilities(
                categorical=categorical,
                multi_categorical=multi_categorical,
                numeric=numeric,
                numeric_missing=numeric_missing,
                compatibility=compatibility,
                compatibility_missing=compatibility_missing,
                viewer_profile_descriptors=viewer_profile_descriptors,
                candidate_profile_descriptors=candidate_profile_descriptors,
                viewer_affinities=viewer_affinities,
                candidate_affinities=candidate_affinities,
            )
        )

        score = peters_score(
            probabilities
        )

        return (
            probabilities,
            score,
        )


# ---------------------------------------------------------------------------
# Production Peters score
# ---------------------------------------------------------------------------

def peters_score(
    probabilities: dict[str, torch.Tensor],
    positive_weights: dict[str, float] | None = None,
    risk_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    positive_weights = (
        positive_weights
        or DEFAULT_PETERS_POSITIVE_WEIGHTS
    )

    risk_weights = (
        risk_weights
        or DEFAULT_PETERS_RISK_WEIGHTS
    )

    positive_terms: list[torch.Tensor] = []
    positive_weight_total = 0.0

    for target, weight in positive_weights.items():
        if target not in probabilities:
            raise KeyError(
                "Missing PetersScore target: "
                f"{target}"
            )

        positive_terms.append(
            probabilities[target]
            * float(weight)
        )

        positive_weight_total += (
            float(weight)
        )

    if positive_weight_total <= 0.0:
        raise ValueError(
            "PetersScore positive weights "
            "must sum to more than zero."
        )

    positive_score = (
        torch.stack(
            positive_terms,
            dim=0,
        )
        .sum(dim=0)
        / positive_weight_total
    )

    risk_terms: list[torch.Tensor] = []
    risk_weight_total = 0.0

    for target, weight in risk_weights.items():
        if target not in probabilities:
            raise KeyError(
                "Missing PetersScore risk target: "
                f"{target}"
            )

        risk_terms.append(
            probabilities[target]
            * float(weight)
        )

        risk_weight_total += (
            float(weight)
        )

    if risk_weight_total > 0.0:
        risk_score = (
            torch.stack(
                risk_terms,
                dim=0,
            )
            .sum(dim=0)
            / risk_weight_total
        )
    else:
        risk_score = torch.zeros_like(
            positive_score
        )

    score = (
        positive_score
        * (
            1.0
            - risk_score.clamp(
                0.0,
                1.0,
            )
        )
    )

    return score.clamp(
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------------
# Multi-task training loss
# ---------------------------------------------------------------------------

def multitask_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    target_weights: dict[str, float] | None = None,
    positive_weights: dict[str, float] | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    if not logits:
        raise ValueError(
            "No logits supplied."
        )

    weights = (
        target_weights
        or {}
    )

    pos_weights = (
        positive_weights
        or {}
    )

    losses: dict[
        str,
        torch.Tensor,
    ] = {}

    weighted_losses: list[
        torch.Tensor
    ] = []

    for (
        target_name,
        prediction,
    ) in logits.items():
        if target_name not in targets:
            raise KeyError(
                "Missing target tensor: "
                f"{target_name}"
            )

        target = (
            targets[target_name]
            .float()
        )

        pos_weight_value = (
            pos_weights.get(
                target_name,
                1.0,
            )
        )

        pos_weight = torch.tensor(
            pos_weight_value,
            dtype=prediction.dtype,
            device=prediction.device,
        )

        loss = (
            F.binary_cross_entropy_with_logits(
                prediction,
                target,
                pos_weight=pos_weight,
            )
        )

        losses[target_name] = (
            loss
        )

        task_weight = weights.get(
            target_name,
            1.0,
        )

        weighted_losses.append(
            loss
            * task_weight
        )

    total_loss = torch.stack(
        weighted_losses
    ).sum()

    weight_total = sum(
        weights.get(
            name,
            1.0,
        )
        for name in logits
    )

    total_loss = (
        total_loss
        / weight_total
    )

    return (
        total_loss,
        losses,
    )


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def load_vocab_sizes(
    path: Path,
    single_fields: list[str] | None = None,
    multi_fields: list[str] | None = None,
) -> dict[str, int]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        vocab = json.load(
            file
        )

    if "fields" not in vocab:
        raise RuntimeError(
            "vocab.json uses the old format. "
            "Run the current "
            "prepare_features.py first."
        )

    single_fields = (
        SINGLE_CATEGORICAL_FIELDS
        if single_fields is None
        else single_fields
    )

    multi_fields = (
        MULTI_CATEGORICAL_FIELDS
        if multi_fields is None
        else multi_fields
    )

    sizes: dict[
        str,
        int,
    ] = {}

    for field in (
        list(single_fields)
        + list(multi_fields)
    ):
        if field not in vocab["fields"]:
            raise RuntimeError(
                "Vocabulary is missing "
                f"field: {field}"
            )

        sizes[field] = len(
            vocab["fields"][field]
        )

    return sizes


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )


def make_fake_batch(
    model: PetersRecommender,
    vocab_sizes: dict[str, int],
    batch_size: int,
    max_multi_values: int = 4,
) -> dict[str, Any]:
    categorical: dict[
        str,
        torch.Tensor,
    ] = {}

    for field in model.single_fields:
        categorical[field] = (
            torch.randint(
                low=0,
                high=vocab_sizes[field],
                size=(batch_size,),
                dtype=torch.long,
            )
        )

    multi_categorical: dict[
        str,
        torch.Tensor,
    ] = {}

    for field in model.multi_fields:
        values = torch.randint(
            low=0,
            high=vocab_sizes[field],
            size=(
                batch_size,
                max_multi_values,
            ),
            dtype=torch.long,
        )

        values[:, -1] = 0

        multi_categorical[
            field
        ] = values

    numeric = torch.rand(
        batch_size,
        len(model.numeric_fields),
        dtype=torch.float32,
    )

    numeric_missing = (
        torch.zeros_like(
            numeric
        )
    )

    if (
        batch_size > 0
        and len(model.numeric_fields) > 0
    ):
        numeric[0, 0] = float("nan")
        numeric_missing[0, 0] = 1.0

    compatibility = torch.rand(
        batch_size,
        len(model.compatibility_fields),
        dtype=torch.float32,
    )

    compatibility_missing = (
        torch.zeros_like(
            compatibility
        )
    )

    if (
        batch_size > 0
        and len(model.compatibility_fields) > 0
    ):
        compatibility[0, 0] = float("nan")
        compatibility_missing[
            0,
            0,
        ] = 1.0

    viewer_profile_descriptors = None
    candidate_profile_descriptors = None

    if model.profile_descriptor_fields:
        width = len(
            model.profile_descriptor_fields
        )

        viewer_profile_descriptors = (
            torch.rand(
                batch_size,
                width,
            )
        )

        candidate_profile_descriptors = (
            torch.rand(
                batch_size,
                width,
            )
        )

    viewer_affinities = None
    candidate_affinities = None

    if model.user_affinity_fields:
        width = len(
            model.user_affinity_fields
        )

        viewer_affinities = (
            torch.rand(
                batch_size,
                width,
            )
        )

        candidate_affinities = (
            torch.rand(
                batch_size,
                width,
            )
        )

    return {
        "categorical":
            categorical,

        "multi_categorical":
            multi_categorical,

        "numeric":
            numeric,

        "numeric_missing":
            numeric_missing,

        "compatibility":
            compatibility,

        "compatibility_missing":
            compatibility_missing,

        "viewer_profile_descriptors":
            viewer_profile_descriptors,

        "candidate_profile_descriptors":
            candidate_profile_descriptors,

        "viewer_affinities":
            viewer_affinities,

        "candidate_affinities":
            candidate_affinities,
    }


def run_smoke_test(
    vocab_path: Path,
    batch_size: int,
) -> None:
    torch.manual_seed(
        42
    )

    vocab_sizes = (
        load_vocab_sizes(
            vocab_path
        )
    )

    model = PetersRecommender(
        vocab_sizes=vocab_sizes
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    print()
    print("PETERS RECOMMENDER")
    print("────────────────────────────────────────")
    print(
        f"Feature schema:      "
        f"{model.feature_schema_version}"
    )
    print(
        f"Vocabulary:          "
        f"{vocab_path}"
    )
    print(
        f"Numeric features:    "
        f"{len(model.numeric_fields)}"
    )
    print(
        f"Compatibility:       "
        f"{len(model.compatibility_fields)}"
    )
    print(
        f"Profile descriptors: "
        f"{len(model.profile_descriptor_fields)}"
    )
    print(
        f"Affinity features:   "
        f"{len(model.user_affinity_fields)}"
    )
    print(
        f"Single categories:   "
        f"{len(model.single_fields)}"
    )
    print(
        f"Multi categories:    "
        f"{len(model.multi_fields)}"
    )
    print(
        f"Prediction targets:  "
        f"{len(model.target_fields)}"
    )
    print(
        f"Parameters:          "
        f"{total_params:,}"
    )
    print(
        f"Trainable:           "
        f"{trainable_params:,}"
    )
    print()

    batch = make_fake_batch(
        model=model,
        vocab_sizes=vocab_sizes,
        batch_size=batch_size,
    )

    model.eval()

    with torch.no_grad():
        probabilities, score = (
            model.predict_with_score(
                **batch
            )
        )

    print("SMOKE TEST")
    print("────────────────────────────────────────")

    for target in model.target_fields:
        probability = (
            probabilities[target]
        )

        print(
            f"{target:<38} "
            f"shape={str(tuple(probability.shape)):<12} "
            f"mean={probability.mean().item():.4f}"
        )

    print()
    print(
        f"{'peters_score':<38} "
        f"shape={str(tuple(score.shape)):<12} "
        f"mean={score.mean().item():.4f}"
    )

    print()
    print(
        "Model forward pass successful."
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and smoke-test the "
            "Peters production recommender."
        )
    )

    parser.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vocab_path = Path(
        args.vocab
    )

    if not vocab_path.exists():
        raise SystemExit(
            "Vocabulary does not exist: "
            f"{vocab_path}"
        )

    if args.batch_size <= 0:
        raise SystemExit(
            "--batch-size must be "
            "greater than 0"
        )

    run_smoke_test(
        vocab_path=vocab_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
    