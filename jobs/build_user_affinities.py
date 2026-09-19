from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from features.user_affinity import (
    RECENT_AFFINITY_DAYS,
    AffinityObservation,
    AffinityTarget,
    UserAffinityState,
    build_user_affinity_state,
    normalize_token,
    normalize_tokens,
    optional_float,
)


DEFAULT_HISTORY_DAYS = 90
DEFAULT_BATCH_SIZE = 1000
DEFAULT_MODEL_VERSION = "behavioral_affinity_v1"

LONG_TERM_META_KIND = "__meta__"
RECENT_META_KIND = "__recent_meta__"
META_EVIDENCE_KEY = "evidence_strength"


# ---------------------------------------------------------------------------
# Behavioral source query
#
# Important:
# - Only behavior before cutoff_at is used.
# - Blocks/reports are intentionally NOT taste signals.
# - Profile/story views may exist here, but their treatment is controlled
#   centrally by features/user_affinity.py.
# ---------------------------------------------------------------------------


AFFINITY_SOURCE_QUERY = """
WITH events AS (
    -- ---------------------------------------------------------------
    -- Profile views
    -- ---------------------------------------------------------------

    SELECT
        viewer_user_id AS actor_user_id,
        owner_user_id AS target_user_id,
        'profile_view'::text AS event_type,
        created_at
    FROM user_profile_views
    WHERE created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Favorites
    -- ---------------------------------------------------------------

    SELECT
        from_user_id,
        to_user_id,
        'favorite'::text,
        created_at
    FROM user_favorites
    WHERE created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Pokes
    -- ---------------------------------------------------------------

    SELECT
        from_user_id,
        to_user_id,
        'poke'::text,
        created_at
    FROM user_pokes
    WHERE created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Image reactions
    -- ---------------------------------------------------------------

    SELECT
        reaction.from_user_id,
        image.user_id,
        'image_reaction'::text,
        reaction.created_at
    FROM image_reactions reaction
    JOIN user_images image
      ON image.id = reaction.image_id
    WHERE reaction.created_at >= %(history_start_at)s
      AND reaction.created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Story views
    -- ---------------------------------------------------------------

    SELECT
        story_view.viewer_user_id,
        story.user_id,
        'story_view'::text,
        story_view.viewed_at
    FROM story_views story_view
    JOIN user_stories story
      ON story.id = story_view.story_id
    WHERE story_view.viewed_at >= %(history_start_at)s
      AND story_view.viewed_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Story reactions / replies
    -- ---------------------------------------------------------------

    SELECT
        reaction.from_user_id,
        story.user_id,
        CASE
            WHEN reaction.kind = 'reply'
                THEN 'story_reply'::text
            ELSE 'story_reaction'::text
        END,
        reaction.created_at
    FROM story_reactions reaction
    JOIN user_stories story
      ON story.id = reaction.story_id
    WHERE reaction.created_at >= %(history_start_at)s
      AND reaction.created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Direct messages
    -- ---------------------------------------------------------------

    SELECT
        message.sender_user_id,
        participant.user_id,
        'message'::text,
        message.created_at
    FROM messages message
    JOIN conversations conversation
      ON conversation.id = message.conversation_id
     AND conversation.type = 'dm'
    JOIN conversation_participants participant
      ON participant.conversation_id = message.conversation_id
     AND participant.user_id <> message.sender_user_id
    WHERE message.sender_user_id IS NOT NULL
      AND message.deleted_at IS NULL
      AND message.created_at >= %(history_start_at)s
      AND message.created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Message reactions
    -- ---------------------------------------------------------------

    SELECT
        reaction.user_id,
        message.sender_user_id,
        'message_reaction'::text,
        reaction.created_at
    FROM message_reactions reaction
    JOIN messages message
      ON message.id = reaction.message_id
    WHERE message.sender_user_id IS NOT NULL
      AND reaction.user_id <> message.sender_user_id
      AND reaction.created_at >= %(history_start_at)s
      AND reaction.created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Private album grants
    -- ---------------------------------------------------------------

    SELECT
        owner_user_id,
        viewer_user_id,
        'private_album_grant'::text,
        granted_at
    FROM user_private_album_access
    WHERE granted_at >= %(history_start_at)s
      AND granted_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Partner requests
    -- ---------------------------------------------------------------

    SELECT
        requester_user_id,
        target_user_id,
        'partner_request'::text,
        created_at
    FROM partner_requests
    WHERE created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Accepted partner request: requester -> target
    -- ---------------------------------------------------------------

    SELECT
        requester_user_id,
        target_user_id,
        'partner_accepted'::text,
        responded_at
    FROM partner_requests
    WHERE status = 'accepted'
      AND responded_at IS NOT NULL
      AND responded_at >= %(history_start_at)s
      AND responded_at < %(cutoff_at)s

    UNION ALL

    -- ---------------------------------------------------------------
    -- Accepted partner request: target -> requester
    -- ---------------------------------------------------------------

    SELECT
        target_user_id,
        requester_user_id,
        'partner_accepted'::text,
        responded_at
    FROM partner_requests
    WHERE status = 'accepted'
      AND responded_at IS NOT NULL
      AND responded_at >= %(history_start_at)s
      AND responded_at < %(cutoff_at)s
),

valid_events AS (
    SELECT
        actor_user_id,
        target_user_id,
        event_type,
        created_at
    FROM events
    WHERE actor_user_id IS NOT NULL
      AND target_user_id IS NOT NULL
      AND actor_user_id <> target_user_id
      AND (
            %(user_id)s::bigint IS NULL
            OR actor_user_id = %(user_id)s
      )
),

event_counts AS (
    SELECT
        actor_user_id,
        target_user_id,
        event_type,

        COUNT(*)::int AS event_count,

        COUNT(*) FILTER (
            WHERE created_at >= %(recent_start_at)s
        )::int AS recent_event_count

    FROM valid_events
    GROUP BY
        actor_user_id,
        target_user_id,
        event_type
)

SELECT
    counts.actor_user_id,
    counts.target_user_id,
    counts.event_type,
    counts.event_count,
    counts.recent_event_count,

    CASE
        WHEN target.date_of_birth IS NULL THEN NULL
        ELSE EXTRACT(
            YEAR FROM age(
                %(cutoff_at)s::date,
                target.date_of_birth
            )
        )::int
    END AS target_age,

    target.height AS target_height,
    target.weight AS target_weight,
    target.relationship_status AS target_relationship_status,
    target.position AS target_position,
    target.body_type AS target_body_type,

    COALESCE(
        target.body_hair,
        ARRAY[]::text[]
    ) AS target_body_hair,

    COALESCE(
        target.my_types,
        ARRAY[]::text[]
    ) AS target_types,

    COALESCE(
        target.my_fetishes,
        ARRAY[]::text[]
    ) AS target_fetishes,

    CASE
        WHEN COALESCE(actor.coord, actor.home_coord) IS NULL
          OR COALESCE(target.coord, target.home_coord) IS NULL
        THEN NULL
        ELSE ST_Distance(
            COALESCE(actor.coord, actor.home_coord),
            COALESCE(target.coord, target.home_coord)
        ) / 1000.0
    END AS distance_km

FROM event_counts counts

JOIN users actor
  ON actor.id = counts.actor_user_id

JOIN users target
  ON target.id = counts.target_user_id

WHERE actor.is_bot = FALSE
  AND target.is_bot = FALSE
  AND actor.is_banned = FALSE
  AND target.is_banned = FALSE

ORDER BY
    counts.actor_user_id,
    counts.target_user_id,
    counts.event_type
"""


UPSERT_AFFINITY_QUERY = """
INSERT INTO ml_user_affinities (
    user_id,
    affinity_kind,
    affinity_key,
    score,
    evidence_count,
    model_version,
    window_start_at,
    window_end_at,
    updated_at
)
VALUES (
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    now()
)
ON CONFLICT (
    user_id,
    affinity_kind,
    affinity_key,
    model_version
)
DO UPDATE SET
    score = EXCLUDED.score,
    evidence_count = EXCLUDED.evidence_count,
    window_start_at = EXCLUDED.window_start_at,
    window_end_at = EXCLUDED.window_end_at,
    updated_at = now()
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build current Peters behavioral user affinities "
            "for live recommendation inference."
        )
    )

    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=(
            "Long-term behavioral history window. "
            f"Default: {DEFAULT_HISTORY_DAYS}"
        ),
    )

    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=(
            "Affinity algorithm/model version stored in "
            "ml_user_affinities."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Streaming SQL fetch batch size. "
            f"Default: {DEFAULT_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help=(
            "Optional single user rebuild. "
            "If omitted, rebuild all current user affinities."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and print diagnostics without modifying "
            "ml_user_affinities."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def require_env(name: str) -> str:
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise SystemExit(
            f"{name} is not set"
        )

    return value


# ---------------------------------------------------------------------------
# Profile normalization
# ---------------------------------------------------------------------------


def parse_height_inches(
    value: Any,
) -> float | None:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    match = re.fullmatch(
        r"\s*(\d+)\s*'\s*(\d+)\s*\"?\s*",
        text,
    )

    if match:
        feet = int(
            match.group(1)
        )

        inches = int(
            match.group(2)
        )

        if 0 <= inches <= 11:
            return float(
                feet * 12
                + inches
            )

    try:
        numeric = float(
            text
        )

        if 36 <= numeric <= 96:
            return numeric

    except ValueError:
        pass

    return None


def affinity_target_from_source_row(
    row: dict[str, Any],
) -> AffinityTarget:
    return AffinityTarget(
        age=optional_float(
            row.get(
                "target_age"
            )
        ),
        height_inches=parse_height_inches(
            row.get(
                "target_height"
            )
        ),
        weight=optional_float(
            row.get(
                "target_weight"
            )
        ),
        distance_km=optional_float(
            row.get(
                "distance_km"
            )
        ),
        position=normalize_token(
            row.get(
                "target_position"
            )
        ),
        body_type=normalize_token(
            row.get(
                "target_body_type"
            )
        ),
        body_hair=normalize_tokens(
            row.get(
                "target_body_hair"
            )
        ),
        relationship_status=normalize_token(
            row.get(
                "target_relationship_status"
            )
        ),
        types=normalize_tokens(
            row.get(
                "target_types"
            )
        ),
        fetishes=normalize_tokens(
            row.get(
                "target_fetishes"
            )
        ),
    )


# ---------------------------------------------------------------------------
# User grouping
# ---------------------------------------------------------------------------


def new_target_group(
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile": (
            affinity_target_from_source_row(
                row
            )
        ),
        "long_term_events": {},
        "recent_events": {},
    }


def add_source_row(
    targets: dict[
        int,
        dict[str, Any],
    ],
    row: dict[str, Any],
) -> None:
    target_user_id = int(
        row["target_user_id"]
    )

    group = targets.setdefault(
        target_user_id,
        new_target_group(
            row
        ),
    )

    event_type = str(
        row["event_type"]
    )

    event_count = max(
        0,
        int(
            row["event_count"]
        ),
    )

    recent_event_count = max(
        0,
        int(
            row["recent_event_count"]
        ),
    )

    if event_count > 0:
        group[
            "long_term_events"
        ][event_type] = event_count

    if recent_event_count > 0:
        group[
            "recent_events"
        ][event_type] = recent_event_count


def build_states_for_user(
    targets: dict[
        int,
        dict[str, Any],
    ],
) -> tuple[
    UserAffinityState,
    UserAffinityState,
]:
    long_term_observations: list[
        AffinityObservation
    ] = []

    recent_observations: list[
        AffinityObservation
    ] = []

    for item in targets.values():
        profile = item[
            "profile"
        ]

        long_term_events = item[
            "long_term_events"
        ]

        recent_events = item[
            "recent_events"
        ]

        if long_term_events:
            long_term_observations.append(
                AffinityObservation(
                    profile=profile,
                    event_counts=(
                        long_term_events
                    ),
                )
            )

        if recent_events:
            recent_observations.append(
                AffinityObservation(
                    profile=profile,
                    event_counts=(
                        recent_events
                    ),
                )
            )

    return (
        build_user_affinity_state(
            long_term_observations
        ),
        build_user_affinity_state(
            recent_observations
        ),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def affinity_rows_for_state(
    user_id: int,
    state: UserAffinityState,
    model_version: str,
    window_start_at: datetime,
    window_end_at: datetime,
    recent: bool,
) -> list[
    tuple[
        int,
        str,
        str,
        float,
        int,
        str,
        datetime,
        datetime,
    ]
]:
    result: list[
        tuple[
            int,
            str,
            str,
            float,
            int,
            str,
            datetime,
            datetime,
        ]
    ] = []

    for kind, values in state.values.items():
        stored_kind = (
            f"recent_{kind}"
            if recent
            else kind
        )

        for key, entry in values.items():
            result.append(
                (
                    user_id,
                    stored_kind,
                    key,
                    float(
                        entry.score
                    ),
                    int(
                        entry.evidence_count
                    ),
                    model_version,
                    window_start_at,
                    window_end_at,
                )
            )

    # Store state-level confidence separately.
    #
    # The existing table has no evidence_weight column, so this preserves
    # the exact [0, 1] evidence strength needed by live inference.
    result.append(
        (
            user_id,
            (
                RECENT_META_KIND
                if recent
                else LONG_TERM_META_KIND
            ),
            META_EVIDENCE_KEY,
            float(
                state.evidence_strength
            ),
            int(
                state.observation_count
            ),
            model_version,
            window_start_at,
            window_end_at,
        )
    )

    return result


def delete_existing_affinities(
    conn: psycopg.Connection,
    model_version: str,
    user_id: int | None,
) -> None:
    with conn.cursor() as cursor:
        if user_id is None:
            cursor.execute(
                """
                DELETE FROM ml_user_affinities
                WHERE model_version = %s
                """,
                (
                    model_version,
                ),
            )

        else:
            cursor.execute(
                """
                DELETE FROM ml_user_affinities
                WHERE model_version = %s
                  AND user_id = %s
                """,
                (
                    model_version,
                    user_id,
                ),
            )


def persist_user_states(
    conn: psycopg.Connection,
    user_id: int,
    long_term_state: UserAffinityState,
    recent_state: UserAffinityState,
    model_version: str,
    history_start_at: datetime,
    recent_start_at: datetime,
    cutoff_at: datetime,
) -> int:
    rows = affinity_rows_for_state(
        user_id=user_id,
        state=long_term_state,
        model_version=model_version,
        window_start_at=history_start_at,
        window_end_at=cutoff_at,
        recent=False,
    )

    rows.extend(
        affinity_rows_for_state(
            user_id=user_id,
            state=recent_state,
            model_version=model_version,
            window_start_at=recent_start_at,
            window_end_at=cutoff_at,
            recent=True,
        )
    )

    with conn.cursor() as cursor:
        cursor.executemany(
            UPSERT_AFFINITY_QUERY,
            rows,
        )

    return len(
        rows
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_user_affinities(
    dsn: str,
    history_days: int,
    model_version: str,
    batch_size: int,
    user_id: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    cutoff_at = datetime.now(
        timezone.utc
    )

    history_start_at = (
        cutoff_at
        - timedelta(
            days=history_days
        )
    )

    recent_start_at = (
        cutoff_at
        - timedelta(
            days=RECENT_AFFINITY_DAYS
        )
    )

    print()
    print("PETERS ML USER AFFINITIES")
    print("────────────────────────────────────────")
    print(
        f"Model version:   "
        f"{model_version}"
    )
    print(
        f"History window:  "
        f"{history_days} days"
    )
    print(
        f"Recent window:   "
        f"{RECENT_AFFINITY_DAYS} days"
    )
    print(
        f"Cutoff:          "
        f"{cutoff_at.isoformat()}"
    )
    print(
        f"Batch size:      "
        f"{batch_size:,}"
    )

    if user_id is not None:
        print(
            f"User:            "
            f"{user_id}"
        )
    else:
        print(
            "User:            all"
        )

    print(
        f"Dry run:         "
        f"{'yes' if dry_run else 'no'}"
    )
    print()

    params = {
        "history_start_at": (
            history_start_at
        ),
        "recent_start_at": (
            recent_start_at
        ),
        "cutoff_at": (
            cutoff_at
        ),
        "user_id": (
            user_id
        ),
    }

    users_processed = 0
    users_with_long_term_evidence = 0
    users_with_recent_evidence = 0
    affinity_rows_written = 0

    current_user_id: int | None = None

    current_targets: dict[
        int,
        dict[str, Any],
    ] = {}

    with psycopg.connect(
        dsn
    ) as read_conn:
        write_conn: (
            psycopg.Connection
            | None
        ) = None

        try:
            if not dry_run:
                write_conn = (
                    psycopg.connect(
                        dsn
                    )
                )

                delete_existing_affinities(
                    conn=write_conn,
                    model_version=model_version,
                    user_id=user_id,
                )

            def flush_current_user() -> None:
                nonlocal current_user_id
                nonlocal current_targets
                nonlocal users_processed
                nonlocal users_with_long_term_evidence
                nonlocal users_with_recent_evidence
                nonlocal affinity_rows_written

                if current_user_id is None:
                    return

                (
                    long_term_state,
                    recent_state,
                ) = build_states_for_user(
                    current_targets
                )

                users_processed += 1

                if (
                    long_term_state
                    .evidence_strength
                    > 0.0
                ):
                    users_with_long_term_evidence += 1

                if (
                    recent_state
                    .evidence_strength
                    > 0.0
                ):
                    users_with_recent_evidence += 1

                if (
                    not dry_run
                    and write_conn is not None
                ):
                    affinity_rows_written += (
                        persist_user_states(
                            conn=write_conn,
                            user_id=(
                                current_user_id
                            ),
                            long_term_state=(
                                long_term_state
                            ),
                            recent_state=(
                                recent_state
                            ),
                            model_version=(
                                model_version
                            ),
                            history_start_at=(
                                history_start_at
                            ),
                            recent_start_at=(
                                recent_start_at
                            ),
                            cutoff_at=(
                                cutoff_at
                            ),
                        )
                    )

                else:
                    affinity_rows_written += (
                        len(
                            affinity_rows_for_state(
                                user_id=(
                                    current_user_id
                                ),
                                state=(
                                    long_term_state
                                ),
                                model_version=(
                                    model_version
                                ),
                                window_start_at=(
                                    history_start_at
                                ),
                                window_end_at=(
                                    cutoff_at
                                ),
                                recent=False,
                            )
                        )
                        + len(
                            affinity_rows_for_state(
                                user_id=(
                                    current_user_id
                                ),
                                state=(
                                    recent_state
                                ),
                                model_version=(
                                    model_version
                                ),
                                window_start_at=(
                                    recent_start_at
                                ),
                                window_end_at=(
                                    cutoff_at
                                ),
                                recent=True,
                            )
                        )
                    )

                current_targets = {}

                print(
                    "\rUsers processed: "
                    f"{users_processed:,}",
                    end="",
                    flush=True,
                )

            with read_conn.cursor(
                name="peters_user_affinities",
                row_factory=dict_row,
            ) as cursor:
                cursor.execute(
                    AFFINITY_SOURCE_QUERY,
                    params,
                )

                while True:
                    raw_rows = (
                        cursor.fetchmany(
                            batch_size
                        )
                    )

                    if not raw_rows:
                        break

                    for raw_row in raw_rows:
                        row = dict(
                            raw_row
                        )

                        actor_user_id = int(
                            row[
                                "actor_user_id"
                            ]
                        )

                        if current_user_id is None:
                            current_user_id = (
                                actor_user_id
                            )

                        elif (
                            actor_user_id
                            != current_user_id
                        ):
                            flush_current_user()

                            current_user_id = (
                                actor_user_id
                            )

                        add_source_row(
                            targets=(
                                current_targets
                            ),
                            row=row,
                        )

            flush_current_user()

            if (
                not dry_run
                and write_conn is not None
            ):
                write_conn.commit()

        except Exception:
            if write_conn is not None:
                write_conn.rollback()

            raise

        finally:
            if write_conn is not None:
                write_conn.close()

    # If rebuilding one user with zero usable behavior, ensure old
    # affinities are still removed.
    if (
        user_id is not None
        and users_processed == 0
        and not dry_run
    ):
        with psycopg.connect(
            dsn
        ) as conn:
            delete_existing_affinities(
                conn=conn,
                model_version=model_version,
                user_id=user_id,
            )

            conn.commit()

    print()
    print()
    print("USER AFFINITY BUILD COMPLETE")
    print("────────────────────────────────────────")

    print(
        "Users processed:                 "
        f"{users_processed:,}"
    )

    print(
        "Users with long-term evidence:   "
        f"{users_with_long_term_evidence:,}"
    )

    print(
        "Users with recent evidence:      "
        f"{users_with_recent_evidence:,}"
    )

    print(
        "Affinity rows "
        f"{'calculated' if dry_run else 'written'}:          "
        f"{affinity_rows_written:,}"
    )

    print(
        "Affinity model version:          "
        f"{model_version}"
    )

    if dry_run:
        print()
        print(
            "Dry run complete. "
            "No database rows were changed."
        )

    print()

    return {
        "user_id": user_id,
        "model_version": model_version,
        "history_days": history_days,
        "recent_days": RECENT_AFFINITY_DAYS,
        "cutoff_at": cutoff_at.isoformat(),
        "users_processed": users_processed,
        "users_with_long_term_evidence": (
            users_with_long_term_evidence
        ),
        "users_with_recent_evidence": (
            users_with_recent_evidence
        ),
        "affinity_rows_written": affinity_rows_written,
        "dry_run": dry_run,
    }


def refresh_user_affinities(
    viewer_user_id: int,
    dsn: str | None = None,
    history_days: int = DEFAULT_HISTORY_DAYS,
    model_version: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """
    Refresh one user's current behavioral affinity state for live inference.

    This is the shared entry point used by the PetersML HTTP service.
    The CLI continues to use build_user_affinities directly.
    """

    if viewer_user_id <= 0:
        raise ValueError(
            "viewer_user_id must be greater than 0"
        )

    if history_days <= 0:
        raise ValueError(
            "history_days must be greater than 0"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0"
        )

    resolved_dsn = (
        dsn
        if dsn is not None
        else os.getenv(
            "APP_DSN",
            "",
        )
    ).strip()

    if not resolved_dsn:
        raise RuntimeError(
            "APP_DSN is not set"
        )

    resolved_model_version = (
        model_version
        if model_version is not None
        else os.getenv(
            "PETERS_AFFINITY_MODEL_VERSION",
            DEFAULT_MODEL_VERSION,
        )
    ).strip()

    if not resolved_model_version:
        raise ValueError(
            "affinity model version cannot be empty"
        )

    return build_user_affinities(
        dsn=resolved_dsn,
        history_days=history_days,
        model_version=resolved_model_version,
        batch_size=batch_size,
        user_id=viewer_user_id,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.history_days <= 0:
        raise SystemExit(
            "--history-days must be greater than 0"
        )

    if args.batch_size <= 0:
        raise SystemExit(
            "--batch-size must be greater than 0"
        )

    if (
        args.user_id is not None
        and args.user_id <= 0
    ):
        raise SystemExit(
            "--user-id must be greater than 0"
        )

    model_version = str(
        args.model_version
    ).strip()

    if not model_version:
        raise SystemExit(
            "--model-version cannot be empty"
        )

    build_user_affinities(
        dsn=require_env(
            "APP_DSN"
        ),
        history_days=(
            args.history_days
        ),
        model_version=(
            model_version
        ),
        batch_size=(
            args.batch_size
        ),
        user_id=args.user_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()