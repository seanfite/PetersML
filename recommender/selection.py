from __future__ import annotations

import hashlib
import math
import random
from typing import Any


VALID_SELECTION_TYPES = {
    "model",
    "diversity",
    "exploration",
}


def _stable_seed(value: str | int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _candidate_id(candidate: dict[str, Any]) -> int:
    return int(candidate["candidate_user_id"])


def _candidate_rank(candidate: dict[str, Any]) -> int:
    return int(candidate["rank"])


def _candidate_score(candidate: dict[str, Any]) -> float:
    score = float(candidate["score"])

    if not math.isfinite(score):
        return 0.0

    return max(score, 0.0)


def _deduplicate_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[int] = set()
    result: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_user_id = _candidate_id(candidate)

        if candidate_user_id in seen:
            continue

        seen.add(candidate_user_id)
        result.append(candidate)

    result.sort(
        key=lambda candidate: (
            _candidate_rank(candidate),
            -_candidate_score(candidate),
            _candidate_id(candidate),
        )
    )

    return result


def _sampling_weight(
    candidate: dict[str, Any],
    window_start_rank: int,
) -> float:
    """
    Weighted randomness inside an already-personalized candidate pool.

    Higher model scores and shallower ranks remain somewhat more likely,
    but deeper candidates still have a real chance to be selected.
    """

    model_rank = _candidate_rank(candidate)
    model_score = max(_candidate_score(candidate), 0.001)

    relative_rank = max(
        1,
        model_rank - window_start_rank + 1,
    )

    rank_weight = 1.0 / math.sqrt(relative_rank)

    return model_score * rank_weight


def _weighted_random_index(
    candidates: list[dict[str, Any]],
    *,
    window_start_rank: int,
    rng: random.Random,
) -> int:
    if len(candidates) == 1:
        return 0

    weights = [
        _sampling_weight(
            candidate,
            window_start_rank,
        )
        for candidate in candidates
    ]

    total_weight = sum(weights)

    if total_weight <= 0:
        return rng.randrange(len(candidates))

    target = rng.random() * total_weight
    running = 0.0

    for index, weight in enumerate(weights):
        running += weight

        if running >= target:
            return index

    return len(candidates) - 1


def _sample_without_replacement(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    window_start_rank: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0 or not candidates:
        return []

    remaining = list(candidates)
    selected: list[dict[str, Any]] = []

    while remaining and len(selected) < count:
        index = _weighted_random_index(
            remaining,
            window_start_rank=window_start_rank,
            rng=rng,
        )

        selected.append(
            remaining.pop(index)
        )

    return selected


def _rank_window(
    candidates: list[dict[str, Any]],
    *,
    minimum_rank: int,
    maximum_rank: int | None,
    excluded_ids: set[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_user_id = _candidate_id(candidate)

        if candidate_user_id in excluded_ids:
            continue

        model_rank = _candidate_rank(candidate)

        if model_rank < minimum_rank:
            continue

        if maximum_rank is not None and model_rank > maximum_rank:
            continue

        result.append(candidate)

    return result


def select_candidates(
    ranked_candidates: list[dict[str, Any]],
    *,
    total_count: int,
    model_count: int,
    sampled_count: int,
    exploration_count: int,
    seed: str | int,
    sampled_max_rank: int = 250,
    exploration_start_rank: int = 251,
) -> list[dict[str, Any]]:
    """
    Select a final recommendation batch from a larger personalized ranking.

    model
        Take the highest-ranked candidates directly.

    diversity
        Weighted random sample from the stronger portion of the personalized
        ranking. The stored selection type remains "diversity" for database
        compatibility, but this means sampled exposure, not forced type
        diversity.

    exploration
        Weighted random sample from deeper in the same personalized ranking.

    No candidate is selected from outside the model-ranked candidate pool.

    The original neural-network rank is returned as "model_rank".
    The final displayed batch rank is returned as "rank".
    """

    if total_count <= 0:
        raise ValueError(
            "total_count must be greater than 0"
        )

    if model_count < 0:
        raise ValueError(
            "model_count cannot be negative"
        )

    if sampled_count < 0:
        raise ValueError(
            "sampled_count cannot be negative"
        )

    if exploration_count < 0:
        raise ValueError(
            "exploration_count cannot be negative"
        )

    if (
        model_count
        + sampled_count
        + exploration_count
        != total_count
    ):
        raise ValueError(
            "model_count + sampled_count + exploration_count "
            "must equal total_count"
        )

    if sampled_max_rank <= model_count:
        raise ValueError(
            "sampled_max_rank must be greater than model_count"
        )

    if exploration_start_rank <= sampled_max_rank:
        raise ValueError(
            "exploration_start_rank must be greater than sampled_max_rank"
        )

    candidates = _deduplicate_candidates(
        ranked_candidates
    )

    if not candidates:
        return []

    rng = random.Random(
        _stable_seed(seed)
    )

    selected: list[
        tuple[
            dict[str, Any],
            str,
        ]
    ] = []

    selected_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Model picks
    # ------------------------------------------------------------------

    for candidate in candidates:
        if len(selected) >= model_count:
            break

        candidate_user_id = _candidate_id(
            candidate
        )

        selected.append(
            (
                candidate,
                "model",
            )
        )

        selected_ids.add(
            candidate_user_id
        )

    # ------------------------------------------------------------------
    # Sampled picks
    #
    # Sample from the stronger part of the personalized ranking after the
    # direct model picks.
    # ------------------------------------------------------------------

    sampled_start_rank = model_count + 1

    sampled_pool = _rank_window(
        candidates,
        minimum_rank=sampled_start_rank,
        maximum_rank=sampled_max_rank,
        excluded_ids=selected_ids,
    )

    sampled = _sample_without_replacement(
        sampled_pool,
        count=sampled_count,
        window_start_rank=sampled_start_rank,
        rng=rng,
    )

    for candidate in sampled:
        candidate_user_id = _candidate_id(
            candidate
        )

        selected.append(
            (
                candidate,
                "diversity",
            )
        )

        selected_ids.add(
            candidate_user_id
        )

    # ------------------------------------------------------------------
    # Exploration picks
    #
    # These still come from PetersML's personalized ranking. They are simply
    # candidates deeper in that ranking that normally would not get exposure.
    # ------------------------------------------------------------------

    exploration_pool = _rank_window(
        candidates,
        minimum_rank=exploration_start_rank,
        maximum_rank=None,
        excluded_ids=selected_ids,
    )

    # If the ranked pool is too small to reach the configured exploration
    # window, use the deepest remaining portion instead.
    if exploration_count > 0 and not exploration_pool:
        remaining = [
            candidate
            for candidate in candidates
            if _candidate_id(candidate) not in selected_ids
        ]

        if remaining:
            tail_start = max(
                0,
                len(remaining) // 2,
            )

            exploration_pool = remaining[
                tail_start:
            ]

    exploration = _sample_without_replacement(
        exploration_pool,
        count=exploration_count,
        window_start_rank=exploration_start_rank,
        rng=rng,
    )

    for candidate in exploration:
        candidate_user_id = _candidate_id(
            candidate
        )

        selected.append(
            (
                candidate,
                "exploration",
            )
        )

        selected_ids.add(
            candidate_user_id
        )

    # ------------------------------------------------------------------
    # Backfill
    #
    # Small development datasets may not contain enough candidates in the
    # configured sampling windows. Fill any remaining slots with the
    # strongest unused candidates.
    # ------------------------------------------------------------------

    for candidate in candidates:
        if len(selected) >= total_count:
            break

        candidate_user_id = _candidate_id(
            candidate
        )

        if candidate_user_id in selected_ids:
            continue

        selected.append(
            (
                candidate,
                "model",
            )
        )

        selected_ids.add(
            candidate_user_id
        )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    output: list[dict[str, Any]] = []

    for final_rank, (
        candidate,
        selection_type,
    ) in enumerate(
        selected[:total_count],
        start=1,
    ):
        if selection_type not in VALID_SELECTION_TYPES:
            raise RuntimeError(
                f"Invalid selection type: {selection_type}"
            )

        result = dict(candidate)

        result["model_rank"] = _candidate_rank(
            candidate
        )

        result["rank"] = final_rank
        result["selection_type"] = selection_type

        output.append(result)

    return output
