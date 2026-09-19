from __future__ import annotations
from typing import Any


def optional_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().casefold()
    return text if text else None


def normalize_values(value: Any) -> set[str]:
    if value is None:
        return set()

    values = value if isinstance(value, list) else [value]
    normalized: set[str] = set()

    for item in values:
        text = normalize_text(item)
        if text is not None:
            normalized.add(text)

    return normalized


def exact_match(preferred: Any, actual: Any) -> float | None:
    preferred_text = normalize_text(preferred)
    actual_text = normalize_text(actual)

    if preferred_text is None or actual_text is None:
        return None

    return float(preferred_text == actual_text)


def overlap_match(preferred: Any, actual: Any) -> tuple[float | None, float | None]:
    preferred_values = normalize_values(preferred)
    actual_values = normalize_values(actual)

    if not preferred_values or not actual_values:
        return None, None

    overlap = preferred_values & actual_values
    count = float(len(overlap))
    ratio = count / float(len(preferred_values))

    return count, ratio


def range_accepts(value: Any, minimum: Any, maximum: Any) -> float | None:
    parsed_value = optional_float(value)
    parsed_minimum = optional_float(minimum)
    parsed_maximum = optional_float(maximum)

    if parsed_value is None or parsed_minimum is None or parsed_maximum is None:
        return None

    return float(parsed_minimum <= parsed_value <= parsed_maximum)


def distance_accepts(distance: Any, maximum_distance: Any) -> float | None:
    parsed_distance = optional_float(distance)
    parsed_maximum = optional_float(maximum_distance)

    if parsed_distance is None or parsed_maximum is None:
        return None

    return float(parsed_distance <= parsed_maximum)


def combine_both(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None

    return float(first > 0.0 and second > 0.0)


def mean_available(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]

    if not available:
        return None

    return sum(available) / float(len(available))


def build_compatibility(row: dict[str, Any]) -> dict[str, float | None]:
    viewer_accepts_candidate_age = range_accepts(
        row.get("candidate_age"),
        row.get("viewer_min_age"),
        row.get("viewer_max_age"),
    )

    candidate_accepts_viewer_age = range_accepts(
        row.get("viewer_age"),
        row.get("candidate_min_age"),
        row.get("candidate_max_age"),
    )

    viewer_accepts_candidate_distance = distance_accepts(
        row.get("distance_km"),
        row.get("viewer_max_distance_km"),
    )

    candidate_accepts_viewer_distance = distance_accepts(
        row.get("distance_km"),
        row.get("candidate_max_distance_km"),
    )

    viewer_position_match = exact_match(
        row.get("viewer_into_position"),
        row.get("candidate_position"),
    )

    candidate_position_match = exact_match(
        row.get("candidate_into_position"),
        row.get("viewer_position"),
    )

    mutual_position_match = combine_both(
        viewer_position_match,
        candidate_position_match,
    )

    viewer_relationship_status_match = exact_match(
        row.get("viewer_into_relationship_status"),
        row.get("candidate_relationship_status"),
    )

    candidate_relationship_status_match = exact_match(
        row.get("candidate_into_relationship_status"),
        row.get("viewer_relationship_status"),
    )

    _, viewer_body_type_match = overlap_match(
        row.get("viewer_into_body_types"),
        row.get("candidate_body_type"),
    )

    _, candidate_body_type_match = overlap_match(
        row.get("candidate_into_body_types"),
        row.get("viewer_body_type"),
    )

    viewer_body_hair_match_count, viewer_body_hair_match_ratio = overlap_match(
        row.get("viewer_into_body_hair"),
        row.get("candidate_body_hair"),
    )

    candidate_body_hair_match_count, candidate_body_hair_match_ratio = overlap_match(
        row.get("candidate_into_body_hair"),
        row.get("viewer_body_hair"),
    )

    viewer_type_match_count, viewer_type_match_ratio = overlap_match(
        row.get("viewer_into_types"),
        row.get("candidate_types"),
    )

    candidate_type_match_count, candidate_type_match_ratio = overlap_match(
        row.get("candidate_into_types"),
        row.get("viewer_types"),
    )

    viewer_fetish_match_count, viewer_fetish_match_ratio = overlap_match(
        row.get("viewer_into_fetishes"),
        row.get("candidate_fetishes"),
    )

    candidate_fetish_match_count, candidate_fetish_match_ratio = overlap_match(
        row.get("candidate_into_fetishes"),
        row.get("viewer_fetishes"),
    )

    mutual_type_match = mean_available(
        [
            viewer_type_match_ratio,
            candidate_type_match_ratio,
        ]
    )

    mutual_preference_score = mean_available(
        [
            viewer_accepts_candidate_age,
            candidate_accepts_viewer_age,
            viewer_accepts_candidate_distance,
            candidate_accepts_viewer_distance,
            viewer_position_match,
            candidate_position_match,
            viewer_relationship_status_match,
            candidate_relationship_status_match,
            viewer_body_type_match,
            candidate_body_type_match,
            viewer_body_hair_match_ratio,
            candidate_body_hair_match_ratio,
            viewer_type_match_ratio,
            candidate_type_match_ratio,
            viewer_fetish_match_ratio,
            candidate_fetish_match_ratio,
        ]
    )

    return {
        "viewer_accepts_candidate_age": viewer_accepts_candidate_age,
        "candidate_accepts_viewer_age": candidate_accepts_viewer_age,
        "viewer_accepts_candidate_distance": viewer_accepts_candidate_distance,
        "candidate_accepts_viewer_distance": candidate_accepts_viewer_distance,
        "viewer_position_match": viewer_position_match,
        "candidate_position_match": candidate_position_match,
        "mutual_position_match": mutual_position_match,
        "viewer_relationship_status_match": viewer_relationship_status_match,
        "candidate_relationship_status_match": candidate_relationship_status_match,
        "viewer_body_type_match": viewer_body_type_match,
        "candidate_body_type_match": candidate_body_type_match,
        "viewer_body_hair_match_count": viewer_body_hair_match_count,
        "viewer_body_hair_match_ratio": viewer_body_hair_match_ratio,
        "candidate_body_hair_match_count": candidate_body_hair_match_count,
        "candidate_body_hair_match_ratio": candidate_body_hair_match_ratio,
        "viewer_type_match_count": viewer_type_match_count,
        "viewer_type_match_ratio": viewer_type_match_ratio,
        "candidate_type_match_count": candidate_type_match_count,
        "candidate_type_match_ratio": candidate_type_match_ratio,
        "viewer_fetish_match_count": viewer_fetish_match_count,
        "viewer_fetish_match_ratio": viewer_fetish_match_ratio,
        "candidate_fetish_match_count": candidate_fetish_match_count,
        "candidate_fetish_match_ratio": candidate_fetish_match_ratio,
        "mutual_type_match": mutual_type_match,
        "mutual_preference_score": mutual_preference_score,
    }
