from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from inference.ranker import Ranker
from jobs.generate_suggestions import (
    acquire_viewer_lock,
    get_default_ranker,
    release_viewer_lock,
    require_dsn,
    viewer_exists,
    viewer_has_plus,
)


SEARCH_FLIGHT_COUNT = 5
CANDIDATE_POOL_LIMIT = 1000
RANKED_PROFILE_LIMIT = 200
SEARCH_FLIGHT_BATCH_HOURS = 24

ALGORITHM_VERSION = "search_flight_filters_v2"

SEARCH_FLIGHT_LOCATION_KEYS = (
    "bellevue",
    "seattle",
    "renton",
    "tacoma",
    "everett",
)

LEARNED_FILTER_KINDS = (
    "type",
    "body_type",
    "body_hair",
    "age_range",
    "height_range",
    "weight_range",
)


@dataclass(frozen=True)
class SearchFilter:
    kind: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "value": self.value,
        }

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.value.lower()


@dataclass(frozen=True)
class SearchFlightRecipe:
    rank: int
    location_key: str
    filters: list[SearchFilter]
    estimated_result_count: int
    score: float
    selection_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "location_key": self.location_key,
            "filters": [
                item.as_dict()
                for item in self.filters
            ],
            "estimated_result_count":
                self.estimated_result_count,
            "score": self.score,
            "selection_type": self.selection_type,
        }


@dataclass(frozen=True)
class SearchFlightResult:
    batch_id: int
    viewer_user_id: int
    model_name: str
    model_version: str
    algorithm_version: str
    generated_at: datetime
    expires_at: datetime
    generated: bool
    candidate_profiles_considered: int
    ranked_profiles_considered: int
    flights: list[SearchFlightRecipe]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "viewer_user_id": self.viewer_user_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "algorithm_version": self.algorithm_version,
            "generated_at": self.generated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "generated": self.generated,
            "candidate_profiles_considered":
                self.candidate_profiles_considered,
            "ranked_profiles_considered":
                self.ranked_profiles_considered,
            "flight_count": len(self.flights),
            "flights": [
                flight.as_dict()
                for flight in self.flights
            ],
        }


@dataclass(frozen=True)
class CandidateTraits:
    user_id: int
    types: tuple[str, ...]
    body_type: str | None
    body_hair: tuple[str, ...]
    age: int | None
    height_inches: int | None
    weight: int | None

    def filters(self) -> list[SearchFilter]:
        values: list[SearchFilter] = []

        for value in self.types:
            normalized = normalize_value(value)

            if normalized is not None:
                values.append(
                    SearchFilter(
                        kind="type",
                        value=normalized,
                    )
                )

        body_type = normalize_value(
            self.body_type
        )

        if body_type is not None:
            values.append(
                SearchFilter(
                    kind="body_type",
                    value=body_type,
                )
            )

        for value in self.body_hair:
            normalized = normalize_value(value)

            if normalized is not None:
                values.append(
                    SearchFilter(
                        kind="body_hair",
                        value=normalized,
                    )
                )

        age_filter = age_range_filter(
            self.age
        )

        if age_filter is not None:
            values.append(age_filter)

        height_filter = height_range_filter(
            self.height_inches
        )

        if height_filter is not None:
            values.append(height_filter)

        weight_filter = weight_range_filter(
            self.weight
        )

        if weight_filter is not None:
            values.append(weight_filter)

        return values


@dataclass(frozen=True)
class RecipeCandidate:
    filters: tuple[SearchFilter, ...]
    score: float
    selection_type: str

    @property
    def key(
        self,
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                item.key
                for item in self.filters
            )
        )


def open_connection() -> psycopg.Connection:
    return psycopg.connect(
        require_dsn(),
        autocommit=True,
        row_factory=dict_row,
    )


def normalize_value(
    value: Any,
) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()

    if not normalized:
        return None

    if normalized.lower() in {
        "any",
        "none",
        "open to any",
        "prefer not to say",
        "prefer not to say.",
    }:
        return None

    return normalized


def normalize_array(
    value: Any,
) -> tuple[str, ...]:
    if value is None:
        return ()

    result: list[str] = []

    for item in value:
        normalized = normalize_value(item)

        if normalized is None:
            continue

        if normalized not in result:
            result.append(normalized)

    return tuple(result)


def parse_height_inches(
    value: Any,
) -> int | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if "'" in text:
        cleaned = text.replace('"', "")
        parts = cleaned.split("'", 1)

        if len(parts) != 2:
            return None

        feet_text = parts[0].strip()
        inches_text = parts[1].strip()

        try:
            feet = int(feet_text)
            inches = int(
                inches_text or "0"
            )
        except ValueError:
            return None

        if not 0 <= inches <= 11:
            return None

        total_inches = (
            feet * 12
            + inches
        )

        if 48 <= total_inches <= 90:
            return total_inches

        return None

    try:
        total_inches = int(
            float(text)
        )
    except ValueError:
        return None

    if 48 <= total_inches <= 90:
        return total_inches

    return None


def age_range_filter(
    age: int | None,
) -> SearchFilter | None:
    if age is None:
        return None

    ranges = (
        (18, 24),
        (25, 29),
        (30, 34),
        (35, 39),
        (40, 44),
        (45, 49),
        (50, 59),
        (60, 69),
        (70, 80),
    )

    for lower, upper in ranges:
        if lower <= age <= upper:
            return SearchFilter(
                kind="age_range",
                value=f"{lower}-{upper}",
            )

    return None


def height_range_filter(
    height_inches: int | None,
) -> SearchFilter | None:
    if height_inches is None:
        return None

    ranges = (
        (48, 64),
        (65, 67),
        (68, 70),
        (71, 73),
        (74, 76),
        (77, 90),
    )

    for lower, upper in ranges:
        if (
            lower
            <= height_inches
            <= upper
        ):
            return SearchFilter(
                kind="height_range",
                value=f"{lower}-{upper}",
            )

    return None


def weight_range_filter(
    weight: int | None,
) -> SearchFilter | None:
    if weight is None:
        return None

    ranges = (
        (100, 149),
        (150, 174),
        (175, 199),
        (200, 224),
        (225, 249),
        (250, 299),
        (300, 350),
    )

    for lower, upper in ranges:
        if lower <= weight <= upper:
            return SearchFilter(
                kind="weight_range",
                value=f"{lower}-{upper}",
            )

    return None


def load_current_batch(
    conn: psycopg.Connection,
    viewer_user_id: int,
) -> SearchFlightResult | None:
    """
    Return the newest ready, unexpired Search Flight batch.
    """

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                id,
                user_id,
                affinity_model_version,
                algorithm_version,
                generated_at,
                expires_at

            FROM ml_search_flight_batches

            WHERE user_id = %s
              AND status = 'ready'
              AND expires_at > NOW()

            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (
                viewer_user_id,
            ),
        )

        batch_row = cursor.fetchone()

        if batch_row is None:
            return None

        cursor.execute(
            """
            SELECT
                rank,
                location_key,
                filters,
                estimated_result_count,
                score,
                selection_type

            FROM ml_search_flight_items

            WHERE batch_id = %s

            ORDER BY rank ASC
            """,
            (
                batch_row["id"],
            ),
        )

        item_rows = cursor.fetchall()

    flights: list[
        SearchFlightRecipe
    ] = []

    for row in item_rows:
        raw_filters = (
            row["filters"]
            or []
        )

        parsed_filters: list[
            SearchFilter
        ] = []

        for item in raw_filters:
            if not isinstance(
                item,
                dict,
            ):
                continue

            kind = normalize_value(
                item.get("kind")
            )

            value = normalize_value(
                item.get("value")
            )

            if (
                kind is None
                or value is None
            ):
                continue

            parsed_filters.append(
                SearchFilter(
                    kind=kind,
                    value=value,
                )
            )

        flights.append(
            SearchFlightRecipe(
                rank=int(
                    row["rank"]
                ),
                location_key=str(
                    row["location_key"]
                ),
                filters=parsed_filters,
                estimated_result_count=int(
                    row[
                        "estimated_result_count"
                    ]
                ),
                score=float(
                    row["score"]
                ),
                selection_type=str(
                    row[
                        "selection_type"
                    ]
                ),
            )
        )

    return SearchFlightResult(
        batch_id=int(
            batch_row["id"]
        ),
        viewer_user_id=int(
            batch_row["user_id"]
        ),
        model_name="peters_recommender",
        model_version=str(
            batch_row[
                "affinity_model_version"
            ]
        ),
        algorithm_version=str(
            batch_row[
                "algorithm_version"
            ]
        ),
        generated_at=
            batch_row["generated_at"],
        expires_at=
            batch_row["expires_at"],
        generated=False,
        candidate_profiles_considered=0,
        ranked_profiles_considered=0,
        flights=flights,
    )


def fetch_candidate_ids(
    conn: psycopg.Connection,
    viewer_user_id: int,
    limit: int,
) -> list[int]:
    """
    Build a broad candidate pool for preference discovery.

    Location is intentionally NOT applied here.

    Search Flights asks:
        "What kind of profiles does this viewer seem interested in?"

    The final five flight locations are stable product slots and are assigned
    when the persisted batch is created.
    """

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT candidate.id
            FROM users candidate

            LEFT JOIN user_settings settings
              ON settings.user_id = candidate.id

            WHERE candidate.id <> %s
              AND candidate.is_bot = FALSE
              AND candidate.is_banned = FALSE
              AND COALESCE(
                    settings.visible,
                    TRUE
                  ) = TRUE

              AND NOT EXISTS (
                  SELECT 1
                  FROM user_blocks block
                  WHERE (
                      block.blocker_user_id = %s
                      AND block.blocked_user_id =
                          candidate.id
                  )
                  OR (
                      block.blocker_user_id =
                          candidate.id
                      AND block.blocked_user_id = %s
                  )
              )

            ORDER BY
                candidate.last_seen_at DESC NULLS LAST,
                candidate.id

            LIMIT %s
            """,
            (
                viewer_user_id,
                viewer_user_id,
                viewer_user_id,
                limit,
            ),
        )

        rows = cursor.fetchall()

    return [
        int(row["id"])
        for row in rows
    ]


def fetch_candidate_traits(
    conn: psycopg.Connection,
    candidate_user_ids: list[int],
) -> dict[int, CandidateTraits]:
    if not candidate_user_ids:
        return {}

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                id,
                my_types,
                body_type,
                body_hair,

                CASE
                    WHEN date_of_birth IS NULL
                        THEN NULL
                    ELSE EXTRACT(
                        YEAR FROM AGE(
                            CURRENT_DATE,
                            date_of_birth
                        )
                    )::int
                END AS age,

                height,
                weight

            FROM users
            WHERE id = ANY(%s)
            """,
            (
                candidate_user_ids,
            ),
        )

        rows = cursor.fetchall()

    result: dict[
        int,
        CandidateTraits,
    ] = {}

    for row in rows:
        user_id = int(
            row["id"]
        )

        result[user_id] = CandidateTraits(
            user_id=user_id,
            types=normalize_array(
                row["my_types"]
            ),
            body_type=normalize_value(
                row["body_type"]
            ),
            body_hair=normalize_array(
                row["body_hair"]
            ),
            age=(
                int(row["age"])
                if row["age"] is not None
                else None
            ),
            height_inches=parse_height_inches(
                row["height"]
            ),
            weight=(
                int(row["weight"])
                if row["weight"] is not None
                else None
            ),
        )

    return result


def ranked_profile_weight(
    rank: int,
    score: float,
) -> float:
    """
    Convert a model-ranked profile into evidence for Search Flight traits.

    PetersScore already represents personalized relevance. Rank decay keeps
    the top profiles more influential without allowing rank 1 to dominate
    the entire inferred preference distribution.
    """

    if rank <= 0:
        return 0.0

    finite_score = (
        score
        if math.isfinite(score)
        else 0.0
    )

    return (
        max(
            finite_score,
            0.0,
        )
        / math.sqrt(rank)
    )


def build_affinity_scores(
    ranked_candidates: list[
        dict[str, Any]
    ],
    traits_by_user: dict[
        int,
        CandidateTraits,
    ],
) -> tuple[
    dict[
        tuple[str, str],
        float,
    ],
    dict[
        tuple[
            tuple[str, str],
            tuple[str, str],
        ],
        float,
    ],
    dict[
        tuple[str, str],
        SearchFilter,
    ],
]:
    """
    Infer which visible Search filters describe profiles PetersML ranks highly.

    The neural ranker already receives:
    - profile traits
    - declared preferences
    - pair compatibility
    - interaction history
    - reverse interaction history
    - PetersGraph features

    This layer does not re-implement those preferences. It only translates
    the final ranked profiles into human-readable Search Flight filters.
    """

    single_scores: dict[
        tuple[str, str],
        float,
    ] = defaultdict(float)

    pair_scores: dict[
        tuple[
            tuple[str, str],
            tuple[str, str],
        ],
        float,
    ] = defaultdict(float)

    filter_lookup: dict[
        tuple[str, str],
        SearchFilter,
    ] = {}

    for candidate in ranked_candidates:
        candidate_user_id = int(
            candidate[
                "candidate_user_id"
            ]
        )

        traits = traits_by_user.get(
            candidate_user_id
        )

        if traits is None:
            continue

        weight = ranked_profile_weight(
            rank=int(
                candidate["rank"]
            ),
            score=float(
                candidate["score"]
            ),
        )

        if weight <= 0:
            continue

        unique_filters: dict[
            tuple[str, str],
            SearchFilter,
        ] = {}

        for search_filter in (
            traits.filters()
        ):
            if (
                search_filter.kind
                not in LEARNED_FILTER_KINDS
            ):
                continue

            unique_filters[
                search_filter.key
            ] = search_filter

        filter_list = list(
            unique_filters.values()
        )

        for search_filter in filter_list:
            key = search_filter.key

            filter_lookup[key] = (
                search_filter
            )

            single_scores[key] += weight

        for left_index in range(
            len(filter_list)
        ):
            left = (
                filter_list[
                    left_index
                ]
            )

            for right_index in range(
                left_index + 1,
                len(filter_list),
            ):
                right = (
                    filter_list[
                        right_index
                    ]
                )

                if (
                    left.kind
                    == right.kind
                ):
                    continue

                pair_key = tuple(
                    sorted(
                        (
                            left.key,
                            right.key,
                        )
                    )
                )

                pair_scores[
                    pair_key
                ] += weight

    return (
        dict(single_scores),
        dict(pair_scores),
        filter_lookup,
    )


def build_recipe_candidates(
    single_scores: dict[
        tuple[str, str],
        float,
    ],
    pair_scores: dict[
        tuple[
            tuple[str, str],
            tuple[str, str],
        ],
        float,
    ],
    filter_lookup: dict[
        tuple[str, str],
        SearchFilter,
    ],
) -> list[RecipeCandidate]:
    recipes: list[
        RecipeCandidate
    ] = []

    if not single_scores:
        return recipes

    max_single = max(
        single_scores.values()
    )

    if max_single <= 0:
        max_single = 1.0

    # ------------------------------------------------------------
    # Single-filter recipes
    # ------------------------------------------------------------

    for key, raw_score in (
        single_scores.items()
    ):
        search_filter = (
            filter_lookup.get(key)
        )

        if search_filter is None:
            continue

        recipes.append(
            RecipeCandidate(
                filters=(
                    search_filter,
                ),
                score=(
                    raw_score
                    / max_single
                ),
                selection_type="model",
            )
        )

    # ------------------------------------------------------------
    # Two-filter recipes
    # ------------------------------------------------------------

    if pair_scores:
        max_pair = max(
            pair_scores.values()
        )

        if max_pair <= 0:
            max_pair = 1.0

        for (
            pair_key,
            raw_pair_score,
        ) in pair_scores.items():
            left_key, right_key = (
                pair_key
            )

            left = filter_lookup.get(
                left_key
            )

            right = filter_lookup.get(
                right_key
            )

            if (
                left is None
                or right is None
            ):
                continue

            left_score = (
                single_scores.get(
                    left_key,
                    0.0,
                )
            )

            right_score = (
                single_scores.get(
                    right_key,
                    0.0,
                )
            )

            cooccurrence_score = (
                raw_pair_score
                / max_pair
            )

            individual_score = (
                (
                    left_score
                    + right_score
                )
                / 2.0
            ) / max_single

            combined_score = (
                0.65
                * cooccurrence_score
                + 0.35
                * individual_score
            )

            recipes.append(
                RecipeCandidate(
                    filters=(
                        left,
                        right,
                    ),
                    score=combined_score,
                    selection_type="model",
                )
            )

    recipes.sort(
        key=lambda recipe:
            recipe.score,
        reverse=True,
    )

    return recipes


def select_diverse_flights(
    recipes: list[
        RecipeCandidate
    ],
    count: int = SEARCH_FLIGHT_COUNT,
) -> list[SearchFlightRecipe]:
    """
    Pick useful but non-identical Search Flight recipes.

    Each selected recipe is assigned one of the five persisted Search Flight
    location slots by rank.
    """

    selected: list[
        RecipeCandidate
    ] = []

    used_recipe_keys: set[
        tuple[
            tuple[str, str],
            ...,
        ]
    ] = set()

    filter_use_count: dict[
        tuple[str, str],
        int,
    ] = defaultdict(int)

    kind_use_count: dict[
        str,
        int,
    ] = defaultdict(int)

    remaining = list(recipes)

    while (
        remaining
        and len(selected) < count
    ):
        best_index: int | None = None

        best_adjusted_score = (
            float("-inf")
        )

        for index, recipe in enumerate(
            remaining
        ):
            if (
                recipe.key
                in used_recipe_keys
            ):
                continue

            adjusted_score = (
                recipe.score
            )

            for search_filter in (
                recipe.filters
            ):
                adjusted_score -= (
                    0.12
                    * filter_use_count[
                        search_filter.key
                    ]
                )

                adjusted_score -= (
                    0.025
                    * kind_use_count[
                        search_filter.kind
                    ]
                )

            if (
                len(recipe.filters)
                == 2
            ):
                adjusted_score += 0.025

            if (
                adjusted_score
                > best_adjusted_score
            ):
                best_adjusted_score = (
                    adjusted_score
                )

                best_index = index

        if best_index is None:
            break

        chosen = remaining.pop(
            best_index
        )

        selected.append(chosen)

        used_recipe_keys.add(
            chosen.key
        )

        for search_filter in (
            chosen.filters
        ):
            filter_use_count[
                search_filter.key
            ] += 1

            kind_use_count[
                search_filter.kind
            ] += 1

    flights: list[
        SearchFlightRecipe
    ] = []

    for index, recipe in enumerate(
        selected
    ):
        if (
            index
            >= len(
                SEARCH_FLIGHT_LOCATION_KEYS
            )
        ):
            break

        flights.append(
            SearchFlightRecipe(
                rank=index + 1,
                location_key=
                    SEARCH_FLIGHT_LOCATION_KEYS[
                        index
                    ],
                filters=list(
                    recipe.filters
                ),
                estimated_result_count=0,
                score=float(
                    recipe.score
                ),
                selection_type=(
                    recipe.selection_type
                ),
            )
        )

    return flights


def save_search_flight_batch(
    conn: psycopg.Connection,
    viewer_user_id: int,
    model_name: str,
    model_version: str,
    algorithm_version: str,
    candidate_profiles_considered: int,
    ranked_profiles_considered: int,
    flights: list[
        SearchFlightRecipe
    ],
) -> SearchFlightResult:
    """
    Persist one complete 24-hour Search Flight batch atomically.
    """

    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ml_search_flight_batches (
                    user_id,
                    affinity_model_version,
                    algorithm_version,
                    status,
                    generated_at,
                    expires_at
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    'building',
                    NOW(),
                    NOW()
                        + (
                            %s
                            * INTERVAL '1 hour'
                        )
                )
                RETURNING
                    id,
                    generated_at,
                    expires_at
                """,
                (
                    viewer_user_id,
                    model_version,
                    algorithm_version,
                    SEARCH_FLIGHT_BATCH_HOURS,
                ),
            )

            batch_row = cursor.fetchone()

            batch_id = int(
                batch_row["id"]
            )

            if flights:
                cursor.executemany(
                    """
                    INSERT INTO ml_search_flight_items (
                        batch_id,
                        rank,
                        location_key,
                        filters,
                        estimated_result_count,
                        score,
                        selection_type
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    [
                        (
                            batch_id,
                            flight.rank,
                            flight.location_key,
                            Jsonb(
                                [
                                    item.as_dict()
                                    for item
                                    in flight.filters
                                ]
                            ),
                            flight.estimated_result_count,
                            flight.score,
                            flight.selection_type,
                        )
                        for flight
                        in flights
                    ],
                )

            cursor.execute(
                """
                UPDATE ml_search_flight_batches
                SET status = 'ready'
                WHERE id = %s
                """,
                (
                    batch_id,
                ),
            )

    return SearchFlightResult(
        batch_id=batch_id,
        viewer_user_id=viewer_user_id,
        model_name=model_name,
        model_version=model_version,
        algorithm_version=
            algorithm_version,
        generated_at=
            batch_row["generated_at"],
        expires_at=
            batch_row["expires_at"],
        generated=True,
        candidate_profiles_considered=
            candidate_profiles_considered,
        ranked_profiles_considered=
            ranked_profiles_considered,
        flights=flights,
    )


def generate_search_flights_for_user(
    viewer_user_id: int,
    *,
    flight_count: int = SEARCH_FLIGHT_COUNT,
    candidate_pool_limit: int = CANDIDATE_POOL_LIMIT,
    ranked_profile_limit: int = RANKED_PROFILE_LIMIT,
    force: bool = False,
    require_plus: bool = True,
    ranker_instance: Ranker | None = None,
) -> SearchFlightResult:
    """
    Return an existing ready/unexpired Search Flight batch or generate,
    persist and return a new one.

    PetersML determines the learned filter recipes. The resulting five
    ranked recipes are persisted into ml_search_flight_batches and
    ml_search_flight_items for PetersServer to consume.
    """

    if viewer_user_id <= 0:
        raise ValueError(
            "viewer_user_id must be greater than 0"
        )

    if not 1 <= flight_count <= 5:
        raise ValueError(
            "flight_count must be between 1 and 5"
        )

    if candidate_pool_limit <= 0:
        raise ValueError(
            "candidate_pool_limit must be greater than 0"
        )

    if ranked_profile_limit <= 0:
        raise ValueError(
            "ranked_profile_limit must be greater than 0"
        )

    conn = open_connection()

    try:
        acquire_viewer_lock(
            conn,
            viewer_user_id,
        )

        if not viewer_exists(
            conn,
            viewer_user_id,
        ):
            raise ValueError(
                f"Viewer user "
                f"{viewer_user_id} "
                "does not exist or is not eligible"
            )

        if (
            require_plus
            and not viewer_has_plus(
                conn,
                viewer_user_id,
            )
        ):
            raise PermissionError(
                f"Viewer user "
                f"{viewer_user_id} "
                "does not have active Plus access"
            )

        if not force:
            current = load_current_batch(
                conn,
                viewer_user_id,
            )

            if current is not None:
                return current

        candidate_user_ids = (
            fetch_candidate_ids(
                conn,
                viewer_user_id,
                limit=
                    candidate_pool_limit,
            )
        )

        active_ranker = (
            ranker_instance
            if ranker_instance
            is not None
            else get_default_ranker()
        )

        if not candidate_user_ids:
            production = (
                active_ranker
                .model_manager
                .load()
            )

            return save_search_flight_batch(
                conn=conn,
                viewer_user_id=
                    viewer_user_id,
                model_name=str(
                    production
                    .record
                    .model_name
                ),
                model_version=str(
                    production
                    .record
                    .version
                ),
                algorithm_version=
                    ALGORITHM_VERSION,
                candidate_profiles_considered=0,
                ranked_profiles_considered=0,
                flights=[],
            )

        top_k = min(
            ranked_profile_limit,
            len(candidate_user_ids),
        )

        # ------------------------------------------------------------
        # Full PetersML ranking
        # ------------------------------------------------------------

        ranking = active_ranker.rank(
            viewer_user_id=
                viewer_user_id,
            candidate_user_ids=
                candidate_user_ids,
            top_k=top_k,
            refresh_model=True,
        )

        payload = ranking.as_dict()

        ranked_candidates = list(
            payload["candidates"]
        )

        ranked_candidate_ids = [
            int(
                candidate[
                    "candidate_user_id"
                ]
            )
            for candidate
            in ranked_candidates
        ]

        # ------------------------------------------------------------
        # Decode ranked profiles into Search Flight filters.
        # ------------------------------------------------------------

        traits_by_user = (
            fetch_candidate_traits(
                conn,
                ranked_candidate_ids,
            )
        )

        (
            single_scores,
            pair_scores,
            filter_lookup,
        ) = build_affinity_scores(
            ranked_candidates=
                ranked_candidates,
            traits_by_user=
                traits_by_user,
        )

        recipe_candidates = (
            build_recipe_candidates(
                single_scores=
                    single_scores,
                pair_scores=
                    pair_scores,
                filter_lookup=
                    filter_lookup,
            )
        )

        flights = select_diverse_flights(
            recipe_candidates,
            count=flight_count,
        )

        return save_search_flight_batch(
            conn=conn,
            viewer_user_id=
                viewer_user_id,
            model_name=str(
                payload["model_name"]
            ),
            model_version=str(
                payload["model_version"]
            ),
            algorithm_version=
                ALGORITHM_VERSION,
            candidate_profiles_considered=
                len(
                    candidate_user_ids
                ),
            ranked_profiles_considered=
                len(
                    ranked_candidates
                ),
            flights=flights,
        )

    finally:
        try:
            release_viewer_lock(
                conn,
                viewer_user_id,
            )
        except Exception:
            pass

        conn.close()


def print_result(
    result: SearchFlightResult,
) -> None:
    print()
    print(
        "PETERS SEARCH FLIGHT FILTERS"
    )
    print(
        "────────────────────────────────────────"
    )
    print(
        f"Viewer:       "
        f"{result.viewer_user_id}"
    )
    print(
        f"Batch:        "
        f"{result.batch_id}"
    )
    print(
        f"Model:        "
        f"{result.model_version}"
    )
    print(
        f"Algorithm:    "
        f"{result.algorithm_version}"
    )
    print(
        f"Generated:    "
        f"{'yes' if result.generated else 'no - cached'}"
    )
    print(
        f"Expires:      "
        f"{result.expires_at.isoformat()}"
    )
    print(
        f"Candidates:   "
        f"{result.candidate_profiles_considered}"
    )
    print(
        f"Ranked:       "
        f"{result.ranked_profiles_considered}"
    )
    print(
        f"Flights:      "
        f"{len(result.flights)}"
    )
    print()

    print(
        json.dumps(
            result.as_dict(),
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and persist personalized "
            "Peters Search Flight recommendations."
        )
    )

    parser.add_argument(
        "--viewer-user-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--allow-non-paid",
        action="store_true",
        help=(
            "Allow local testing without "
            "active Plus access."
        ),
    )

    parser.add_argument(
        "--candidate-pool-limit",
        type=int,
        default=CANDIDATE_POOL_LIMIT,
    )

    parser.add_argument(
        "--ranked-profile-limit",
        type=int,
        default=RANKED_PROFILE_LIMIT,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore an existing ready/unexpired "
            "Search Flight batch."
        ),
    )

    args = parser.parse_args()

    result = (
        generate_search_flights_for_user(
            viewer_user_id=
                args.viewer_user_id,
            flight_count=
                SEARCH_FLIGHT_COUNT,
            candidate_pool_limit=
                args.candidate_pool_limit,
            ranked_profile_limit=
                args.ranked_profile_limit,
            force=
                args.force,
            require_plus=(
                not args.allow_non_paid
            ),
        )
    )

    print_result(result)


if __name__ == "__main__":
    main()
    