from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from inference.ranker import Ranker
from recommender.selection import select_candidates

SUGGESTION_LIMIT = 15
CANDIDATE_POOL_LIMIT = 2000
BATCH_LIFETIME_HOURS = 24
SUGGESTION_MODEL_COUNT = 12
SUGGESTION_SAMPLED_COUNT = 3
SUGGESTION_EXPLORATION_COUNT = 0
SUGGESTION_SAMPLED_MAX_RANK = 250

# We do not currently store a dedicated neighborhood field.
# Use a small geographic radius for "Same neighborhood".
NEIGHBORHOOD_RADIUS_MILES = 5.0
METERS_PER_MILE = 1609.344

_default_ranker: Ranker | None = None


@dataclass(frozen=True)
class SuggestionItem:
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
class SuggestionBatch:
    batch_id: int
    viewer_user_id: int
    model_version: str
    generated_at: datetime
    expires_at: datetime
    generated: bool
    suggestions: list[SuggestionItem]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "viewer_user_id": self.viewer_user_id,
            "model_version": self.model_version,
            "generated_at": self.generated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "generated": self.generated,
            "suggestion_count": len(self.suggestions),
            "suggestions": [suggestion.as_dict() for suggestion in self.suggestions],
        }


def get_default_ranker() -> Ranker:
    global _default_ranker
    if _default_ranker is None:
        _default_ranker = Ranker()
    return _default_ranker


def require_dsn() -> str:
    value = os.getenv("APP_DSN", "").strip()
    if not value:
        raise RuntimeError("APP_DSN is not set")
    return value


def open_connection() -> psycopg.Connection:
    return psycopg.connect(require_dsn(), autocommit=True, row_factory=dict_row)


def acquire_viewer_lock(conn: psycopg.Connection, viewer_user_id: int) -> None:
    """Prevent simultaneous suggestion generation for the same viewer."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_lock(%s)", (viewer_user_id,))


def release_viewer_lock(conn: psycopg.Connection, viewer_user_id: int) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_unlock(%s)", (viewer_user_id,))


def viewer_exists(conn: psycopg.Connection, viewer_user_id: int) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM users
                WHERE id = %s
                  AND is_bot = FALSE
                  AND is_banned = FALSE
            ) AS exists
            """,
            (viewer_user_id,),
        )
        row = cursor.fetchone()
    return bool(row["exists"])


def viewer_has_plus(conn: psycopg.Connection, viewer_user_id: int) -> bool:
    """Return whether the viewer has an active paid or complimentary Plus entitlement."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT (
                EXISTS (
                    SELECT 1
                    FROM subscriptions s
                    WHERE s.user_id = %s
                      AND LOWER(s.plan) = 'plus'
                      AND LOWER(s.status) IN ('active', 'trialing')
                      AND (s.expires_at IS NULL OR s.expires_at > NOW())
                )
                OR
                EXISTS (
                    SELECT 1
                    FROM plus_access_grants g
                    WHERE g.user_id = %s
                      AND g.revoked_at IS NULL
                      AND g.started_at <= NOW()
                      AND (g.expires_at IS NULL OR g.expires_at > NOW())
                )
            ) AS has_plus
            """,
            (viewer_user_id, viewer_user_id),
        )
        row = cursor.fetchone()
    return bool(row["has_plus"])


def fetch_paid_user_ids(conn: psycopg.Connection) -> list[int]:
    """Return all current non-bot, non-banned Plus users."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT u.id
            FROM users u
            WHERE u.is_bot = FALSE
              AND u.is_banned = FALSE
              AND (
                  EXISTS (
                      SELECT 1
                      FROM subscriptions s
                      WHERE s.user_id = u.id
                        AND LOWER(s.plan) = 'plus'
                        AND LOWER(s.status) IN ('active', 'trialing')
                        AND (s.expires_at IS NULL OR s.expires_at > NOW())
                  )
                  OR
                  EXISTS (
                      SELECT 1
                      FROM plus_access_grants g
                      WHERE g.user_id = u.id
                        AND g.revoked_at IS NULL
                        AND g.started_at <= NOW()
                        AND (g.expires_at IS NULL OR g.expires_at > NOW())
                  )
              )
            ORDER BY u.id
            """
        )
        rows = cursor.fetchall()
    return [int(row["id"]) for row in rows]


def load_current_batch(
    conn: psycopg.Connection, viewer_user_id: int
) -> SuggestionBatch | None:
    """Return the newest ready, unexpired suggestion batch for this viewer."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, viewer_user_id, model_version, generated_at, expires_at
            FROM ml_daily_suggestion_batches
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
            SELECT rank, model_rank, candidate_user_id, score, selection_type, predictions
            FROM ml_daily_suggestion_items
            WHERE batch_id = %s
            ORDER BY rank
            """,
            (batch_row["id"],),
        )
        item_rows = cursor.fetchall()

    suggestions = [
        SuggestionItem(
            rank=int(row["rank"]),
            model_rank=int(row["model_rank"]),
            candidate_user_id=int(row["candidate_user_id"]),
            score=float(row["score"]),
            selection_type=str(row["selection_type"]),
            predictions={
                str(key): float(value) for key, value in (row["predictions"] or {}).items()
            },
        )
        for row in item_rows
    ]

    return SuggestionBatch(
        batch_id=int(batch_row["id"]),
        viewer_user_id=int(batch_row["viewer_user_id"]),
        model_version=str(batch_row["model_version"]),
        generated_at=batch_row["generated_at"],
        expires_at=batch_row["expires_at"],
        generated=False,
        suggestions=suggestions,
    )


def get_viewer_proximity(
    conn: psycopg.Connection, viewer_user_id: int
) -> tuple[str, str | None, bool]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COALESCE(NULLIF(TRIM(into_proximity), ''), 'Any') AS into_proximity,
                NULLIF(TRIM(city), '') AS city,
                coord IS NOT NULL AS has_coord
            FROM users
            WHERE id = %s
            """,
            (viewer_user_id,),
        )
        row = cursor.fetchone()

    if row is None:
        raise ValueError(f"Viewer user {viewer_user_id} does not exist")

    return str(row["into_proximity"]), row["city"], bool(row["has_coord"])


def parse_distance_miles(proximity: str) -> float | None:
    match = re.fullmatch(
        r"Within\s+(\d+(?:\.\d+)?)\s+miles?",
        proximity.strip(),
        flags=re.IGNORECASE,
    )
    return None if match is None else float(match.group(1))


def fetch_eligible_candidate_ids(
    conn: psycopg.Connection,
    viewer_user_id: int,
    limit: int = CANDIDATE_POOL_LIMIT,
) -> list[int]:
    """
    Build the hard-eligible candidate pool.

    Hard rules: exclude self, bots, banned/invisible users, blocks in either direction,
    and honor the viewer's explicit proximity preference. Other compatibility and taste
    preferences remain ML features.
    """
    if limit <= 0:
        raise ValueError("candidate pool limit must be greater than 0")

    proximity, viewer_city, viewer_has_coord = get_viewer_proximity(conn, viewer_user_id)
    normalized = proximity.strip().lower()
    location_clause = ""
    location_params: list[Any] = []

    if normalized in {"", "any"}:
        pass
    elif normalized == "same city":
        if not viewer_city:
            return []

        location_clause = """
            AND c.city IS NOT NULL
            AND LOWER(TRIM(c.city)) = LOWER(TRIM(%s))
        """
        location_params.append(viewer_city)

    elif normalized == "same neighborhood":
        if not viewer_has_coord:
            return []

        radius_meters = NEIGHBORHOOD_RADIUS_MILES * METERS_PER_MILE
        location_clause = """
            AND viewer.coord IS NOT NULL
            AND c.coord IS NOT NULL
            AND ST_DWithin(viewer.coord, c.coord, %s)
        """
        location_params.append(radius_meters)

    else:
        distance_miles = parse_distance_miles(proximity)
        if distance_miles is not None:
            if not viewer_has_coord:
                return []

            location_clause = """
                AND viewer.coord IS NOT NULL
                AND c.coord IS NOT NULL
                AND ST_DWithin(viewer.coord, c.coord, %s)
            """
            location_params.append(distance_miles * METERS_PER_MILE)

    query = f"""
        SELECT c.id
        FROM users c
        JOIN users viewer ON viewer.id = %s
        LEFT JOIN user_settings candidate_settings ON candidate_settings.user_id = c.id
        WHERE c.id <> %s
          AND c.is_bot = FALSE
          AND c.is_banned = FALSE
          AND COALESCE(candidate_settings.visible, TRUE) = TRUE
          AND NOT EXISTS (
              SELECT 1
              FROM user_blocks b
              WHERE (b.blocker_user_id = %s AND b.blocked_user_id = c.id)
                 OR (b.blocker_user_id = c.id AND b.blocked_user_id = %s)
          )
          {location_clause}
        ORDER BY c.last_seen_at DESC NULLS LAST, c.id
        LIMIT %s
    """

    params: list[Any] = [
        viewer_user_id,
        viewer_user_id,
        viewer_user_id,
        viewer_user_id,
        *location_params,
        limit,
    ]

    with conn.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()

    return [int(row["id"]) for row in rows]


def save_suggestion_batch(
    conn: psycopg.Connection,
    viewer_user_id: int,
    model_version: str,
    suggestions: list[SuggestionItem],
) -> SuggestionBatch:
    """Save one complete 24-hour batch atomically."""
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ml_daily_suggestion_batches (
                    viewer_user_id, model_version, status, generated_at, expires_at
                )
                VALUES (%s, %s, 'building', NOW(), NOW() + (%s * INTERVAL '1 hour'))
                RETURNING id, generated_at, expires_at
                """,
                (viewer_user_id, model_version, BATCH_LIFETIME_HOURS),
            )

            batch_row = cursor.fetchone()
            batch_id = int(batch_row["id"])

            if suggestions:
                cursor.executemany(
                    """
                    INSERT INTO ml_daily_suggestion_items (
                        batch_id, candidate_user_id, rank, model_rank, score,
                        selection_type, predictions
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            batch_id,
                            suggestion.candidate_user_id,
                            suggestion.rank,
                            suggestion.model_rank,
                            suggestion.score,
                            suggestion.selection_type,
                            Jsonb(suggestion.predictions),
                        )
                        for suggestion in suggestions
                    ],
                )

            cursor.execute(
                "UPDATE ml_daily_suggestion_batches SET status = 'ready' WHERE id = %s",
                (batch_id,),
            )

    return SuggestionBatch(
        batch_id=batch_id,
        viewer_user_id=viewer_user_id,
        model_version=model_version,
        generated_at=batch_row["generated_at"],
        expires_at=batch_row["expires_at"],
        generated=True,
        suggestions=suggestions,
    )


def _selection_counts(limit: int) -> tuple[int, int, int]:
    if limit == SUGGESTION_LIMIT:
        return (
            SUGGESTION_MODEL_COUNT,
            SUGGESTION_SAMPLED_COUNT,
            SUGGESTION_EXPLORATION_COUNT,
        )

    sampled_count = min(3, round(limit * 0.20))
    return limit - sampled_count, sampled_count, 0


def generate_suggestions_for_user(
    viewer_user_id: int,
    *,
    limit: int = SUGGESTION_LIMIT,
    candidate_pool_limit: int = CANDIDATE_POOL_LIMIT,
    force: bool = False,
    require_plus: bool = True,
    ranker_instance: Ranker | None = None,
) -> SuggestionBatch:
    """
    Return an existing valid suggestion batch or generate a new one.

    Standard 15-item policy: rank the full eligible candidate pool, then select
    12 direct model picks + 3 weighted sampled picks and no deep exploration.
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

        if require_plus and not viewer_has_plus(conn, viewer_user_id):
            raise PermissionError(
                f"Viewer user {viewer_user_id} does not have active Plus access"
            )

        if not force:
            current = load_current_batch(conn, viewer_user_id)
            if current is not None:
                return current

        candidate_user_ids = fetch_eligible_candidate_ids(
            conn,
            viewer_user_id,
            limit=candidate_pool_limit,
        )

        active_ranker = (
            ranker_instance if ranker_instance is not None else get_default_ranker()
        )

        if not candidate_user_ids:
            production = active_ranker.model_manager.load()
            return save_suggestion_batch(
                conn=conn,
                viewer_user_id=viewer_user_id,
                model_version=production.record.version,
                suggestions=[],
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
            f"daily-suggestions:{viewer_user_id}:"
            f"{datetime.now(timezone.utc).date().isoformat()}:"
            f"{payload['model_version']}"
        )

        sampled_max_rank = max(
            SUGGESTION_SAMPLED_MAX_RANK,
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
        )

        suggestions = [
            SuggestionItem(
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

        return save_suggestion_batch(
            conn=conn,
            viewer_user_id=viewer_user_id,
            model_version=str(payload["model_version"]),
            suggestions=suggestions,
        )

    finally:
        try:
            release_viewer_lock(conn, viewer_user_id)
        except Exception:
            pass
        conn.close()


def generate_suggestions_for_all_paid_users(
    *,
    limit: int = SUGGESTION_LIMIT,
    candidate_pool_limit: int = CANDIDATE_POOL_LIMIT,
    ranker_instance: Ranker | None = None,
) -> list[SuggestionBatch]:
    """Daily scheduled-job entry point. Existing unexpired batches are returned from cache."""
    conn = open_connection()

    try:
        viewer_user_ids = fetch_paid_user_ids(conn)
    finally:
        conn.close()

    active_ranker = (
        ranker_instance if ranker_instance is not None else get_default_ranker()
    )

    results: list[SuggestionBatch] = []

    for viewer_user_id in viewer_user_ids:
        try:
            results.append(
                generate_suggestions_for_user(
                    viewer_user_id,
                    limit=limit,
                    candidate_pool_limit=candidate_pool_limit,
                    force=False,
                    require_plus=True,
                    ranker_instance=active_ranker,
                )
            )
        except Exception as error:
            print(
                f"Suggestion generation failed viewer_user_id={viewer_user_id}: {error}"
            )

    return results


def print_batch(batch: SuggestionBatch) -> None:
    print()
    print("PETERS DAILY SUGGESTIONS")
    print("────────────────────────────────────────")
    print(f"Viewer:        {batch.viewer_user_id}")
    print(f"Batch:         {batch.batch_id}")
    print(f"Model:         {batch.model_version}")
    print(f"Generated:     {'yes' if batch.generated else 'no - cached'}")
    print(f"Expires:       {batch.expires_at.isoformat()}")
    print(f"Suggestions:   {len(batch.suggestions)}")
    print()
    print(json.dumps(batch.as_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Peters daily personalized suggestions."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--viewer-user-id", type=int)
    mode.add_argument("--all-paid", action="store_true")

    parser.add_argument("--limit", type=int, default=SUGGESTION_LIMIT)
    parser.add_argument(
        "--candidate-pool-limit",
        type=int,
        default=CANDIDATE_POOL_LIMIT,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore an existing unexpired batch. For local/debug use only.",
    )
    parser.add_argument(
        "--allow-non-paid",
        action="store_true",
        help="Allow single-user local testing without active Plus access.",
    )

    args = parser.parse_args()

    if args.all_paid:
        if args.force:
            parser.error("--force is only supported with --viewer-user-id")
        if args.allow_non_paid:
            parser.error("--allow-non-paid is only supported with --viewer-user-id")

        results = generate_suggestions_for_all_paid_users(
            limit=args.limit,
            candidate_pool_limit=args.candidate_pool_limit,
        )

        generated_count = sum(1 for result in results if result.generated)
        cached_count = len(results) - generated_count

        print()
        print("PETERS DAILY SUGGESTIONS JOB")
        print("────────────────────────────────────────")
        print(f"Paid users:    {len(results)}")
        print(f"Generated:     {generated_count}")
        print(f"Cached:        {cached_count}")
        print()
        return

    batch = generate_suggestions_for_user(
        viewer_user_id=args.viewer_user_id,
        limit=args.limit,
        candidate_pool_limit=args.candidate_pool_limit,
        force=args.force,
        require_plus=not args.allow_non_paid,
    )

    print_batch(batch)


if __name__ == "__main__":
    main()
    