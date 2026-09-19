from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from features.definitions import USER_AFFINITY_FIELDS


# ---------------------------------------------------------------------------
# Affinity contract
# ---------------------------------------------------------------------------


NEUTRAL_AFFINITY = 0.5

RECENT_AFFINITY_DAYS = 30

# Controls how quickly confidence approaches 1.0 as behavioral evidence grows.
EVIDENCE_SATURATION = 12.0

# Prior evidence centered at neutral. This prevents one interaction from
# immediately producing an extreme preference score.
NEUTRAL_PRIOR_WEIGHT = 2.0


# ---------------------------------------------------------------------------
# Behavioral evidence weights
#
# These weights do NOT directly become PetersML scores.
#
# They are used to construct a behavioral preference signal for one
# user -> historical candidate relationship. PetersML ultimately learns
# how useful the resulting affinity features are.
# ---------------------------------------------------------------------------


POSITIVE_EVENT_WEIGHTS = {
    "favorite": 1.50,
    "poke": 0.70,
    "image_reaction": 0.80,
    "story_reaction": 0.60,
    "story_reply": 1.00,
    "message": 1.20,
    "message_reaction": 0.50,
    "private_album_grant": 1.20,
    "partner_request": 1.50,
    "partner_accepted": 2.00,
}


NEGATIVE_EVENT_WEIGHTS: dict[str, float] = {}

# Passive profile/story views are currently neutral.
#
# Once ml_recommendation_impressions is sufficiently populated, explicit
# shown-but-not-engaged exposure can become clean negative preference evidence.
PASSIVE_PROFILE_VIEW_WEIGHT = 0.0
PASSIVE_STORY_VIEW_WEIGHT = 0.0
MAX_PASSIVE_PROFILE_VIEWS = 3
MAX_PASSIVE_STORY_VIEWS = 3


# ---------------------------------------------------------------------------
# Numeric preference buckets
#
# ml_user_affinities stores affinity_kind + affinity_key. Bucketing numeric
# values gives us a stable representation that can later be persisted there.
# ---------------------------------------------------------------------------


def age_bucket(value: float | None) -> str | None:
    if value is None:
        return None

    if value < 25:
        return "under_25"
    if value < 30:
        return "25_29"
    if value < 35:
        return "30_34"
    if value < 40:
        return "35_39"
    if value < 45:
        return "40_44"
    if value < 50:
        return "45_49"
    if value < 60:
        return "50_59"

    return "60_plus"


def height_bucket(value: float | None) -> str | None:
    if value is None:
        return None

    if value < 66:
        return "under_66"
    if value < 70:
        return "66_69"
    if value < 74:
        return "70_73"

    return "74_plus"


def weight_bucket(value: float | None) -> str | None:
    if value is None:
        return None

    if value < 150:
        return "under_150"
    if value < 175:
        return "150_174"
    if value < 200:
        return "175_199"
    if value < 225:
        return "200_224"
    if value < 250:
        return "225_249"

    return "250_plus"


def distance_bucket(value: float | None) -> str | None:
    if value is None:
        return None

    if value < 5:
        return "under_5"
    if value < 15:
        return "5_14"
    if value < 30:
        return "15_29"
    if value < 60:
        return "30_59"

    return "60_plus"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


def normalize_token(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    return text or None


def normalize_tokens(values: Any) -> tuple[str, ...]:
    if not values:
        return ()

    if isinstance(values, str):
        values = [values]

    normalized = {
        token
        for value in values
        if (token := normalize_token(value)) is not None
    }

    return tuple(sorted(normalized))


def nonnegative_count(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0

    return max(0, parsed)


# ---------------------------------------------------------------------------
# Profile representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffinityTarget:
    age: float | None = None
    height_inches: float | None = None
    weight: float | None = None
    distance_km: float | None = None

    position: str | None = None
    body_type: str | None = None
    body_hair: tuple[str, ...] = ()
    relationship_status: str | None = None
    types: tuple[str, ...] = ()
    fetishes: tuple[str, ...] = ()

    @classmethod
    def from_feature_row(
        cls,
        row: Mapping[str, Any],
        prefix: str,
    ) -> "AffinityTarget":
        if prefix not in {"viewer", "candidate"}:
            raise ValueError(
                "prefix must be 'viewer' or 'candidate'"
            )

        return cls(
            age=optional_float(
                row.get(f"{prefix}_age")
            ),
            height_inches=optional_float(
                row.get(f"{prefix}_height_inches")
            ),
            weight=optional_float(
                row.get(f"{prefix}_weight")
            ),
            distance_km=optional_float(
                row.get("distance_km")
            ),
            position=normalize_token(
                row.get(f"{prefix}_position")
            ),
            body_type=normalize_token(
                row.get(f"{prefix}_body_type")
            ),
            body_hair=normalize_tokens(
                row.get(f"{prefix}_body_hair")
            ),
            relationship_status=normalize_token(
                row.get(f"{prefix}_relationship_status")
            ),
            types=normalize_tokens(
                row.get(f"{prefix}_types")
            ),
            fetishes=normalize_tokens(
                row.get(f"{prefix}_fetishes")
            ),
        )


# ---------------------------------------------------------------------------
# Historical observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffinityObservation:
    """
    One user's behavioral evidence toward one historical candidate.

    profile describes the historical candidate.

    event_counts contains the user's actions toward that candidate during
    the requested history window.

    The caller decides which observations belong to long-term versus recent
    history. That keeps this module independent of SQL and makes historical
    cutoff handling explicit.
    """

    profile: AffinityTarget
    event_counts: Mapping[str, int]

    @classmethod
    def create(
        cls,
        profile: AffinityTarget,
        **event_counts: int,
    ) -> "AffinityObservation":
        return cls(
            profile=profile,
            event_counts={
                key: nonnegative_count(value)
                for key, value in event_counts.items()
            },
        )


# ---------------------------------------------------------------------------
# Learned affinity state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffinityEntry:
    score: float
    evidence_count: int
    evidence_weight: float


@dataclass(frozen=True)
class UserAffinityState:
    """
    Learned behavioral preference state for one user and one time window.

    values:
        affinity kind
            -> affinity key
                -> score/evidence

    Examples:

        values["body_type"]["stocky"]
        values["body_hair"]["hairy"]
        values["age"]["35_39"]
    """

    values: dict[str, dict[str, AffinityEntry]]
    observation_count: int
    evidence_weight: float

    @property
    def evidence_strength(self) -> float:
        if self.evidence_weight <= 0:
            return 0.0

        return clamp01(
            1.0
            - math.exp(
                -self.evidence_weight
                / EVIDENCE_SATURATION
            )
        )

    def lookup(
        self,
        kind: str,
        key: str | None,
    ) -> AffinityEntry | None:
        if key is None:
            return None

        return self.values.get(
            kind,
            {},
        ).get(
            key
        )


# ---------------------------------------------------------------------------
# Internal accumulation
# ---------------------------------------------------------------------------


@dataclass
class _Accumulator:
    weighted_score_sum: float = 0.0
    evidence_weight: float = 0.0
    evidence_count: int = 0

    def add(
        self,
        score: float,
        weight: float,
    ) -> None:
        if weight <= 0:
            return

        self.weighted_score_sum += (
            clamp01(score)
            * weight
        )

        self.evidence_weight += weight
        self.evidence_count += 1

    def finish(self) -> AffinityEntry:
        denominator = (
            NEUTRAL_PRIOR_WEIGHT
            + self.evidence_weight
        )

        numerator = (
            NEUTRAL_AFFINITY
            * NEUTRAL_PRIOR_WEIGHT
            + self.weighted_score_sum
        )

        return AffinityEntry(
            score=clamp01(
                numerator / denominator
            ),
            evidence_count=self.evidence_count,
            evidence_weight=self.evidence_weight,
        )


# ---------------------------------------------------------------------------
# Observation scoring
# ---------------------------------------------------------------------------


def event_count(
    observation: AffinityObservation,
    event_type: str,
) -> int:
    return nonnegative_count(
        observation.event_counts.get(
            event_type,
            0,
        )
    )


def observation_preference_signal(
    observation: AffinityObservation,
) -> tuple[float, float]:
    """
    Convert one historical user -> candidate relationship into:

        preference score in [0, 1]
        evidence weight >= 0

    0.5 means neutral / unknown.

    Strong positive interaction pushes toward 1.

    Passive profile/story views are currently neutral until recommendation
    impressions provide a clean shown-but-not-engaged signal.
    """

    positive_raw = sum(
        weight
        * event_count(
            observation,
            event_type,
        )
        for event_type, weight
        in POSITIVE_EVENT_WEIGHTS.items()
    )

    negative_raw = sum(
        weight
        * event_count(
            observation,
            event_type,
        )
        for event_type, weight
        in NEGATIVE_EVENT_WEIGHTS.items()
    )

    if (
        positive_raw <= 0
        and negative_raw <= 0
    ):
        profile_views = min(
            event_count(
                observation,
                "profile_view",
            ),
            MAX_PASSIVE_PROFILE_VIEWS,
        )

        story_views = min(
            event_count(
                observation,
                "story_view",
            ),
            MAX_PASSIVE_STORY_VIEWS,
        )

        negative_raw += (
            profile_views
            * PASSIVE_PROFILE_VIEW_WEIGHT
        )

        negative_raw += (
            story_views
            * PASSIVE_STORY_VIEW_WEIGHT
        )

    total_raw = (
        positive_raw
        + negative_raw
    )

    if total_raw <= 0:
        return (
            NEUTRAL_AFFINITY,
            0.0,
        )

    directional_strength = math.tanh(
        (
            positive_raw
            - negative_raw
        )
        / 2.0
    )

    score = (
        NEUTRAL_AFFINITY
        + 0.5
        * directional_strength
    )

    # Evidence grows sub-linearly so repeated actions toward one person
    # cannot completely dominate the user's learned representation.
    evidence_weight = min(
        4.0,
        math.log1p(
            total_raw
        )
        * 1.5,
    )

    return (
        clamp01(score),
        evidence_weight,
    )


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


def add_accumulator_value(
    accumulators: dict[
        str,
        dict[str, _Accumulator],
    ],
    kind: str,
    key: str | None,
    score: float,
    weight: float,
) -> None:
    if key is None:
        return

    kind_values = accumulators.setdefault(
        kind,
        {},
    )

    accumulator = kind_values.setdefault(
        key,
        _Accumulator(),
    )

    accumulator.add(
        score=score,
        weight=weight,
    )


def add_accumulator_values(
    accumulators: dict[
        str,
        dict[str, _Accumulator],
    ],
    kind: str,
    keys: Iterable[str],
    score: float,
    weight: float,
) -> None:
    for key in keys:
        add_accumulator_value(
            accumulators=accumulators,
            kind=kind,
            key=key,
            score=score,
            weight=weight,
        )


def build_user_affinity_state(
    observations: Iterable[AffinityObservation],
) -> UserAffinityState:
    accumulators: dict[
        str,
        dict[str, _Accumulator],
    ] = {}

    observation_count = 0
    total_evidence_weight = 0.0

    for observation in observations:
        score, evidence_weight = (
            observation_preference_signal(
                observation
            )
        )

        if evidence_weight <= 0:
            continue

        observation_count += 1
        total_evidence_weight += evidence_weight

        profile = observation.profile

        add_accumulator_value(
            accumulators,
            "age",
            age_bucket(profile.age),
            score,
            evidence_weight,
        )

        add_accumulator_value(
            accumulators,
            "height",
            height_bucket(
                profile.height_inches
            ),
            score,
            evidence_weight,
        )

        add_accumulator_value(
            accumulators,
            "weight",
            weight_bucket(
                profile.weight
            ),
            score,
            evidence_weight,
        )

        add_accumulator_value(
            accumulators,
            "distance",
            distance_bucket(
                profile.distance_km
            ),
            score,
            evidence_weight,
        )

        add_accumulator_value(
            accumulators,
            "position",
            profile.position,
            score,
            evidence_weight,
        )

        add_accumulator_value(
            accumulators,
            "body_type",
            profile.body_type,
            score,
            evidence_weight,
        )

        add_accumulator_values(
            accumulators,
            "body_hair",
            profile.body_hair,
            score,
            evidence_weight,
        )

        add_accumulator_value(
            accumulators,
            "relationship_status",
            profile.relationship_status,
            score,
            evidence_weight,
        )

        add_accumulator_values(
            accumulators,
            "type",
            profile.types,
            score,
            evidence_weight,
        )

        add_accumulator_values(
            accumulators,
            "fetish",
            profile.fetishes,
            score,
            evidence_weight,
        )

    values = {
        kind: {
            key: accumulator.finish()
            for key, accumulator
            in kind_values.items()
        }
        for kind, kind_values
        in accumulators.items()
    }

    return UserAffinityState(
        values=values,
        observation_count=observation_count,
        evidence_weight=total_evidence_weight,
    )


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


def score_single_affinity(
    state: UserAffinityState,
    kind: str,
    key: str | None,
) -> float:
    entry = state.lookup(
        kind,
        key,
    )

    if entry is None:
        return NEUTRAL_AFFINITY

    return clamp01(
        entry.score
    )


def score_multi_affinity(
    state: UserAffinityState,
    kind: str,
    keys: Iterable[str],
) -> float:
    scores: list[float] = []

    for key in keys:
        entry = state.lookup(
            kind,
            key,
        )

        if entry is not None:
            scores.append(
                clamp01(
                    entry.score
                )
            )

    if not scores:
        return NEUTRAL_AFFINITY

    return clamp01(
        sum(scores)
        / len(scores)
    )


def score_target_dimensions(
    state: UserAffinityState,
    target: AffinityTarget,
) -> dict[str, float]:
    return {
        "age_affinity": score_single_affinity(
            state,
            "age",
            age_bucket(
                target.age
            ),
        ),

        "height_affinity": score_single_affinity(
            state,
            "height",
            height_bucket(
                target.height_inches
            ),
        ),

        "weight_affinity": score_single_affinity(
            state,
            "weight",
            weight_bucket(
                target.weight
            ),
        ),

        "distance_affinity": score_single_affinity(
            state,
            "distance",
            distance_bucket(
                target.distance_km
            ),
        ),

        "position_affinity": score_single_affinity(
            state,
            "position",
            target.position,
        ),

        "body_type_affinity": score_single_affinity(
            state,
            "body_type",
            target.body_type,
        ),

        "body_hair_affinity": score_multi_affinity(
            state,
            "body_hair",
            target.body_hair,
        ),

        "relationship_status_affinity": score_single_affinity(
            state,
            "relationship_status",
            target.relationship_status,
        ),

        "type_affinity": score_multi_affinity(
            state,
            "type",
            target.types,
        ),

        "fetish_affinity": score_multi_affinity(
            state,
            "fetish",
            target.fetishes,
        ),
    }


def target_available_dimensions(
    target: AffinityTarget,
) -> set[str]:
    available: set[str] = set()

    if target.age is not None:
        available.add("age_affinity")

    if target.height_inches is not None:
        available.add("height_affinity")

    if target.weight is not None:
        available.add("weight_affinity")

    if target.distance_km is not None:
        available.add("distance_affinity")

    if target.position is not None:
        available.add("position_affinity")

    if target.body_type is not None:
        available.add("body_type_affinity")

    if target.body_hair:
        available.add("body_hair_affinity")

    if target.relationship_status is not None:
        available.add(
            "relationship_status_affinity"
        )

    if target.types:
        available.add("type_affinity")

    if target.fetishes:
        available.add("fetish_affinity")

    return available


def overall_affinity(
    dimension_scores: Mapping[str, float],
    target: AffinityTarget,
) -> float:
    available = target_available_dimensions(
        target
    )

    scores = [
        float(
            dimension_scores[field]
        )
        for field in available
        if field in dimension_scores
    ]

    if not scores:
        return NEUTRAL_AFFINITY

    return clamp01(
        sum(scores)
        / len(scores)
    )


# ---------------------------------------------------------------------------
# Final PetersML affinity feature construction
# ---------------------------------------------------------------------------


def build_directional_affinity_features(
    long_term_state: UserAffinityState,
    recent_state: UserAffinityState,
    target: AffinityTarget,
) -> dict[str, float]:
    """
    Create the exact USER_AFFINITY_FIELDS expected by PetersML.

    Example:

        viewer history -> candidate
            build_directional_affinity_features(
                viewer_state,
                viewer_recent_state,
                candidate_profile,
            )

        candidate history -> viewer
            build_directional_affinity_features(
                candidate_state,
                candidate_recent_state,
                viewer_profile,
            )
    """

    long_term = score_target_dimensions(
        state=long_term_state,
        target=target,
    )

    recent = score_target_dimensions(
        state=recent_state,
        target=target,
    )

    result = {
        **long_term,

        "recent_age_affinity":
            recent["age_affinity"],

        "recent_height_affinity":
            recent["height_affinity"],

        "recent_weight_affinity":
            recent["weight_affinity"],

        "recent_distance_affinity":
            recent["distance_affinity"],

        "recent_position_affinity":
            recent["position_affinity"],

        "recent_body_type_affinity":
            recent["body_type_affinity"],

        "recent_body_hair_affinity":
            recent["body_hair_affinity"],

        "recent_relationship_status_affinity":
            recent[
                "relationship_status_affinity"
            ],

        "recent_type_affinity":
            recent["type_affinity"],

        "recent_fetish_affinity":
            recent["fetish_affinity"],

        "overall_affinity": overall_affinity(
            dimension_scores=long_term,
            target=target,
        ),

        "recent_overall_affinity": overall_affinity(
            dimension_scores=recent,
            target=target,
        ),

        "evidence_strength":
            long_term_state.evidence_strength,

        "recent_evidence_strength":
            recent_state.evidence_strength,
    }

    validate_affinity_features(
        result
    )

    return result


def build_directional_affinity_from_observations(
    long_term_observations: Iterable[AffinityObservation],
    recent_observations: Iterable[AffinityObservation],
    target: AffinityTarget,
) -> dict[str, float]:
    long_term_state = build_user_affinity_state(
        long_term_observations
    )

    recent_state = build_user_affinity_state(
        recent_observations
    )

    return build_directional_affinity_features(
        long_term_state=long_term_state,
        recent_state=recent_state,
        target=target,
    )


# ---------------------------------------------------------------------------
# Neutral / cold-start state
# ---------------------------------------------------------------------------


def neutral_affinity_features() -> dict[str, float]:
    result = {
        field: (
            0.0
            if field in {
                "evidence_strength",
                "recent_evidence_strength",
            }
            else NEUTRAL_AFFINITY
        )
        for field in USER_AFFINITY_FIELDS
    }

    validate_affinity_features(
        result
    )

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def clamp01(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def validate_affinity_features(
    features: Mapping[str, Any],
) -> None:
    expected = set(
        USER_AFFINITY_FIELDS
    )

    actual = set(
        features
    )

    missing = expected - actual
    unexpected = actual - expected

    if missing:
        raise RuntimeError(
            "Affinity features are missing fields: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if unexpected:
        raise RuntimeError(
            "Affinity features contain unexpected fields: "
            + ", ".join(
                sorted(
                    unexpected
                )
            )
        )

    for field in USER_AFFINITY_FIELDS:
        value = optional_float(
            features[field]
        )

        if value is None:
            raise RuntimeError(
                "Affinity feature is not finite: "
                f"{field}={features[field]!r}"
            )

        if not 0.0 <= value <= 1.0:
            raise RuntimeError(
                "Affinity feature must be between 0 and 1: "
                f"{field}={value}"
            )
        