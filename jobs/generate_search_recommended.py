from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from inference.ranker import Ranker
from jobs.generate_suggestions import (
    CANDIDATE_POOL_LIMIT,
    acquire_viewer_lock,
    get_default_ranker,
    open_connection,
    release_viewer_lock,
    viewer_exists,
)
from recommender.selection import select_candidates

SEARCH_RECOMMENDED_LIMIT = 100
SEARCH_RECOMMENDED_BATCH_HOURS = 24

SEARCH_RECOMMENDED_MODEL_COUNT = 80
SEARCH_RECOMMENDED_SAMPLED_COUNT = 15
SEARCH_RECOMMENDED_EXPLORATION_COUNT = 5
SEARCH_RECOMMENDED_SAMPLED_MAX_RANK = 250
SEARCH_RECOMMENDED_EXPLORATION_START_RANK = 251


@dataclass(frozen=True)
class SearchRecommendedItem:
    rank: int
    model_rank: int
    candidate_user_id: int
    score: float
    selection_type: str
    predictions: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "model_rank": self.model_rank,
            "candidate_user_id": self.candidate_user_id,
            "score": self.score,
            "selection_type": self.selection_type,
            "predictions": self.predictions,
        }


@dataclass(frozen=True)
class SearchRecommendedBatch:
    batch_id: int
    viewer_user_id: int
    model_version: str
    generated_at: datetime
    expires_at: datetime
    generated: bool
    recommendations: list[SearchRecommendedItem]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "viewer_user_id": self.viewer_user_id,
            "model_version": self.model_version,
            "generated_at": self.generated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "generated": self.generated,
            "recommendation_count": len(self.recommendations),
            "recommendations": [
                recommendation.as_dict() for recommendation in self.recommendations
            ],
        }


def load_current_batch(
    conn: psycopg.Connection,
    viewer_user_id: int,
) -> SearchRecommendedBatch | None:
    """Return the newest ready, unexpired Search AI recommendation pool."""

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, viewer_user_id, model_version, generated_at, expires_at
            FROM ml_search_recommended_batches
            WHERE viewer_user_id = %s
              AND status = 'ready'
              AND expires_at > NOW()
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (viewer_user_id,),
        )

        batch_row = cursor.fetchone()

        if batch_row is None:
            return None

        cursor.execute(
            """
            SELECT
                rank,
                model_rank,
                candidate_user_id,
                score,
                selection_type,
                predictions
            FROM ml_search_recommended_items
            WHERE batch_id = %s
            ORDER BY rank
            """,
            (batch_row["id"],),
        )

        item_rows = cursor.fetchall()

    recommendations = [
        SearchRecommendedItem(
            rank=int(row["rank"]),
            model_rank=int(row["model_rank"]),
            candidate_user_id=int(row["candidate_user_id"]),
            score=float(row["score"]),
            selection_type=str(row["selection_type"]),
            predictions={
                str(key): float(value)
                for key, value in (row["predictions"] or {}).items()
            },
        )
        for row in item_rows
    ]

    return SearchRecommendedBatch(
        batch_id=int(batch_row["id"]),
        viewer_user_id=int(batch_row["viewer_user_id"]),
        model_version=str(batch_row["model_version"]),
        generated_at=batch_row["generated_at"],
        expires_at=batch_row["expires_at"],
        generated=False,
        recommendations=recommendations,
    )


def fetch_candidate_ids(
    conn: psycopg.Connection,
    viewer_user_id: int,
    limit: int,
) -> list[int]:
    """
    Build the broad Search AI candidate universe.

    Location is intentionally ignored here. PetersServer later intersects this
    pool with the user's live radius, map viewport, filters, and eligibility.
    """

    if limit <= 0:
        raise ValueError("candidate pool limit must be greater than 0")

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
              AND COALESCE(settings.visible, TRUE) = TRUE
              AND NOT EXISTS (
                  SELECT 1
                  FROM user_blocks block
                  WHERE (
                      block.blocker_user_id = %s
                      AND block.blocked_user_id = candidate.id
                  )
                  OR (
                      block.blocker_user_id = candidate.id
                      AND block.blocked_user_id = %s
                  )
              )
            ORDER BY candidate.last_seen_at DESC NULLS LAST, candidate.id
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

    return [int(row["id"]) for row in rows]


def save_batch(
    conn: psycopg.Connection,
    viewer_user_id: int,
    model_version: str,
    recommendations: list[SearchRecommendedItem],
) -> SearchRecommendedBatch:
    """Persist one complete Search AI recommendation pool atomically."""

    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ml_search_recommended_batches (
                    viewer_user_id,
                    model_version,
                    status,
                    generated_at,
                    expires_at
                )
                VALUES (
                    %s,
                    %s,
                    'building',
                    NOW(),
                    NOW() + (%s * INTERVAL '1 hour')
                )
                RETURNING id, generated_at, expires_at
                """,
                (
                    viewer_user_id,
                    model_version,
                    SEARCH_RECOMMENDED_BATCH_HOURS,
                ),
            )

            batch_row = cursor.fetchone()
            batch_id = int(batch_row["id"])

            if recommendations:
                cursor.executemany(
                    """
                    INSERT INTO ml_search_recommended_items (
                        batch_id,
                        candidate_user_id,
                        rank,
                        model_rank,
                        score,
                        selection_type,
                        predictions
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            batch_id,
                            recommendation.candidate_user_id,
                            recommendation.rank,
                            recommendation.model_rank,
                            recommendation.score,
                            recommendation.selection_type,
                            Jsonb(recommendation.predictions),
                        )
                        for recommendation in recommendations
                    ],
                )

            cursor.execute(
                """
                UPDATE ml_search_recommended_batches
                SET status = 'ready'
                WHERE id = %s
                """,
                (batch_id,),
            )

    return SearchRecommendedBatch(
        batch_id=batch_id,
        viewer_user_id=viewer_user_id,
        model_version=model_version,
        generated_at=batch_row["generated_at"],
        expires_at=batch_row["expires_at"],
        generated=True,
        recommendations=recommendations,
    )


def _selection_counts(limit: int) -> tuple[int, int, int]:
    if limit == SEARCH_RECOMMENDED_LIMIT:
        return (
            SEARCH_RECOMMENDED_MODEL_COUNT,
            SEARCH_RECOMMENDED_SAMPLED_COUNT,
            SEARCH_RECOMMENDED_EXPLORATION_COUNT,
        )

    exploration_count = round(limit * 0.05)
    sampled_count = round(limit * 0.15)

    if limit >= 10 and exploration_count == 0:
        exploration_count = 1

    if sampled_count + exploration_count >= limit:
        exploration_count = 0
        sampled_count = min(sampled_count, max(0, limit - 1))

    model_count = limit - sampled_count - exploration_count
    return model_count, sampled_count, exploration_count


def generate_search_recommended_for_user(
    viewer_user_id: int,
    *,
    limit: int = SEARCH_RECOMMENDED_LIMIT,
    candidate_pool_limit: int = CANDIDATE_POOL_LIMIT,
    force: bool = False,
    ranker_instance: Ranker | None = None,
) -> SearchRecommendedBatch:
    """
    Return an existing Search AI pool or generate a new one.

    Standard 100-item policy:
    - 80 direct model picks
    - 15 sampled personalized picks
    - 5 deeper personalized exploration picks

    The pool is deliberately location-independent. PetersServer later intersects
    it with Search radius, map viewport, normal filters, and current eligibility.
    """

    if viewer_user_id <= 0:
        raise ValueError("viewer_user_id must be greater than 0")

    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    if candidate_pool_limit <= 0:
        raise ValueError("candidate_pool_limit must be greater than 0")

    if limit > candidate_pool_limit:
        raise ValueError("limit cannot exceed candidate_pool_limit")

    conn = open_connection()

    try:
        acquire_viewer_lock(conn, viewer_user_id)

        if not viewer_exists(conn, viewer_user_id):
            raise ValueError(
                f"Viewer user {viewer_user_id} does not exist or is not eligible"
            )

        if not force:
            current = load_current_batch(conn, viewer_user_id)

            if current is not None:
                return current

        candidate_user_ids = fetch_candidate_ids(
            conn,
            viewer_user_id,
            limit=candidate_pool_limit,
        )

        active_ranker = (
            ranker_instance if ranker_instance is not None else get_default_ranker()
        )

        if not candidate_user_ids:
            production = active_ranker.model_manager.load()

            return save_batch(
                conn=conn,
                viewer_user_id=viewer_user_id,
                model_version=production.record.version,
                recommendations=[],
            )

        ranking = active_ranker.rank(
            viewer_user_id=viewer_user_id,
            candidate_user_ids=candidate_user_ids,
            top_k=len(candidate_user_ids),
            refresh_model=True,
        )

        payload = ranking.as_dict()
        model_count, sampled_count, exploration_count = _selection_counts(limit)

        selection_seed = (
            f"search-recommended:{viewer_user_id}:"
            f"{datetime.now(timezone.utc).date().isoformat()}:"
            f"{payload['model_version']}"
        )

        sampled_max_rank = max(
            SEARCH_RECOMMENDED_SAMPLED_MAX_RANK,
            model_count + sampled_count,
        )

        selected_candidates = select_candidates(
            payload["candidates"],
            total_count=limit,
            model_count=model_count,
            sampled_count=sampled_count,
            exploration_count=exploration_count,
            seed=selection_seed,
            sampled_max_rank=sampled_max_rank,
            exploration_start_rank=SEARCH_RECOMMENDED_EXPLORATION_START_RANK,
        )

        recommendations = [
            SearchRecommendedItem(
                rank=int(candidate["rank"]),
                model_rank=int(candidate["model_rank"]),
                candidate_user_id=int(candidate["candidate_user_id"]),
                score=float(candidate["score"]),
                selection_type=str(candidate["selection_type"]),
                predictions={
                    str(key): float(value)
                    for key, value in candidate["predictions"].items()
                },
            )
            for candidate in selected_candidates
        ]

        return save_batch(
            conn=conn,
            viewer_user_id=viewer_user_id,
            model_version=str(payload["model_version"]),
            recommendations=recommendations,
        )

    finally:
        try:
            release_viewer_lock(conn, viewer_user_id)
        except Exception:
            pass

        conn.close()


def print_batch(batch: SearchRecommendedBatch) -> None:
    print()
    print("PETERS SEARCH AI RECOMMENDED")
    print("────────────────────────────────────────")
    print(f"Viewer:          {batch.viewer_user_id}")
    print(f"Batch:           {batch.batch_id}")
    print(f"Model:           {batch.model_version}")
    print(f"Generated:       {'yes' if batch.generated else 'no - cached'}")
    print(f"Expires:         {batch.expires_at.isoformat()}")
    print(f"Recommendations: {len(batch.recommendations)}")
    print()
    print(json.dumps(batch.as_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Peters Search AI recommended candidate pools."
    )

    parser.add_argument("--viewer-user-id", type=int, required=True)
    parser.add_argument("--limit", type=int, default=SEARCH_RECOMMENDED_LIMIT)
    parser.add_argument(
        "--candidate-pool-limit",
        type=int,
        default=CANDIDATE_POOL_LIMIT,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore an existing unexpired Search AI pool. For local/debug use only.",
    )

    args = parser.parse_args()

    batch = generate_search_recommended_for_user(
        viewer_user_id=args.viewer_user_id,
        limit=args.limit,
        candidate_pool_limit=args.candidate_pool_limit,
        force=args.force,
    )

    print_batch(batch)


if __name__ == "__main__":
    main()
    