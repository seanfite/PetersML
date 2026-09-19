from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from inference.graph_client import (
    GraphPairFeatures,
    PetersGraphClient,
)
from features.user_affinity import (
    AffinityObservation,
    AffinityTarget,
    RECENT_AFFINITY_DAYS,
    UserAffinityState,
    build_directional_affinity_features,
    build_user_affinity_state,
)

DEFAULT_HISTORY_DAYS = 90
DEFAULT_OUTCOME_DAYS = 7
DEFAULT_OUTPUT = "data/training_pairs.jsonl"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_GRAPH_BATCH_SIZE = 1000
DEFAULT_SNAPSHOT_COUNT = 1
DEFAULT_SNAPSHOT_STEP_DAYS = 7


DATASET_QUERY = """
WITH params AS (
    SELECT
        %(cutoff_at)s::timestamptz AS cutoff_at,
        %(outcome_end_at)s::timestamptz AS outcome_end_at,
        %(cutoff_at)s::timestamptz
            - (%(history_days)s * interval '1 day')
            AS history_start_at
),

windows AS (
    SELECT
        cutoff_at,
        outcome_end_at,
        history_start_at,
        cutoff_at - interval '7 days' AS history_7d_start_at,
        cutoff_at - interval '30 days' AS history_30d_start_at,
        cutoff_at - interval '90 days' AS history_90d_start_at,

        LEAST(
            history_start_at,
            cutoff_at - interval '90 days'
        ) AS load_start_at
    FROM params
),

events AS (
    -- ---------------------------------------------------------------
    -- Profile views
    -- ---------------------------------------------------------------

    SELECT
        viewer_user_id AS actor_user_id,
        owner_user_id AS candidate_user_id,
        'profile_view'::text AS event_type,
        created_at
    FROM user_profile_views
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)

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
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)

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
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)

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
    WHERE reaction.created_at >= (SELECT load_start_at FROM windows)
      AND reaction.created_at < (SELECT outcome_end_at FROM windows)

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
    WHERE story_view.viewed_at >= (SELECT load_start_at FROM windows)
      AND story_view.viewed_at < (SELECT outcome_end_at FROM windows)

    UNION ALL

    -- ---------------------------------------------------------------
    -- Story reactions / replies
    -- ---------------------------------------------------------------

    SELECT
        reaction.from_user_id,
        story.user_id,
        CASE
            WHEN reaction.kind = 'reply' THEN 'story_reply'::text
            ELSE 'story_reaction'::text
        END,
        reaction.created_at
    FROM story_reactions reaction
    JOIN user_stories story
      ON story.id = reaction.story_id
    WHERE reaction.created_at >= (SELECT load_start_at FROM windows)
      AND reaction.created_at < (SELECT outcome_end_at FROM windows)

    UNION ALL

    -- ---------------------------------------------------------------
    -- Direct messages
    -- Sender -> other participant in DM
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
      AND message.created_at >= (SELECT load_start_at FROM windows)
      AND message.created_at < (SELECT outcome_end_at FROM windows)

    UNION ALL

    -- ---------------------------------------------------------------
    -- Message reactions
    -- Reactor -> message sender
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
      AND reaction.created_at >= (SELECT load_start_at FROM windows)
      AND reaction.created_at < (SELECT outcome_end_at FROM windows)

    UNION ALL

    -- ---------------------------------------------------------------
    -- Blocks
    -- ---------------------------------------------------------------

    SELECT
        blocker_user_id,
        blocked_user_id,
        'block'::text,
        created_at
    FROM user_blocks
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)

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
    WHERE granted_at >= (SELECT load_start_at FROM windows)
      AND granted_at < (SELECT outcome_end_at FROM windows)

    UNION ALL

    -- ---------------------------------------------------------------
    -- Partner request
    -- ---------------------------------------------------------------

    SELECT
        requester_user_id,
        target_user_id,
        'partner_request'::text,
        created_at
    FROM partner_requests
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)

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
      AND responded_at >= (SELECT load_start_at FROM windows)
      AND responded_at < (SELECT outcome_end_at FROM windows)

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
      AND responded_at >= (SELECT load_start_at FROM windows)
      AND responded_at < (SELECT outcome_end_at FROM windows)

    UNION ALL

    -- ---------------------------------------------------------------
    -- Reports
    -- ---------------------------------------------------------------

    SELECT
        reporter_user_id,
        reported_user_id,
        'report'::text,
        created_at
    FROM support_reports
    WHERE reported_user_id IS NOT NULL
      AND created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)
),

valid_events AS (
    SELECT
        actor_user_id,
        candidate_user_id,
        event_type,
        created_at
    FROM events
    WHERE actor_user_id IS NOT NULL
      AND candidate_user_id IS NOT NULL
      AND actor_user_id <> candidate_user_id
),

-- -------------------------------------------------------------------
-- Temporary exposure definition:
-- user must have viewed the candidate during the outcome period.
--
-- Later this should be recommendation impressions instead.
-- -------------------------------------------------------------------

pair_keys AS (
    SELECT DISTINCT
        actor_user_id,
        candidate_user_id
    FROM valid_events
    WHERE event_type = 'profile_view'
      AND created_at >= (SELECT cutoff_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)
),

-- -------------------------------------------------------------------
-- Viewer -> candidate HISTORY
-- -------------------------------------------------------------------

history_pair_stats AS (
    SELECT
        actor_user_id,
        candidate_user_id,

        MIN(created_at) AS first_history_interaction_at,
        MAX(created_at) AS last_history_interaction_at,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
        )::int AS history_profile_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
        )::int AS history_favorite_count,

        COUNT(*) FILTER (
            WHERE event_type = 'poke'
        )::int AS history_poke_count,

        COUNT(*) FILTER (
            WHERE event_type = 'image_reaction'
        )::int AS history_image_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_view'
        )::int AS history_story_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reaction'
        )::int AS history_story_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reply'
        )::int AS history_story_reply_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
        )::int AS history_message_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message_reaction'
        )::int AS history_message_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'private_album_grant'
        )::int AS history_private_album_grant_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_request'
        )::int AS history_partner_request_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_accepted'
        )::int AS history_partner_accepted_count,

        COUNT(*) FILTER (
            WHERE event_type = 'block'
        )::int AS history_block_count,

        COUNT(*) FILTER (
            WHERE event_type = 'report'
        )::int AS history_report_count,

        COUNT(*) FILTER (
            WHERE created_at >= (
                SELECT history_7d_start_at FROM windows
            )
        )::int AS history_all_events_7d,

        COUNT(*) FILTER (
            WHERE created_at >= (
                SELECT history_30d_start_at FROM windows
            )
        )::int AS history_all_events_30d,

        COUNT(*) FILTER (
            WHERE created_at >= (
                SELECT history_90d_start_at FROM windows
            )
        )::int AS history_all_events_90d,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
              AND created_at >= (
                  SELECT history_7d_start_at FROM windows
              )
        )::int AS history_profile_views_7d,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS history_profile_views_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (
                  SELECT history_7d_start_at FROM windows
              )
        )::int AS history_favorites_7d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS history_favorites_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (
                  SELECT history_7d_start_at FROM windows
              )
        )::int AS history_messages_7d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS history_messages_30d

    FROM valid_events
    WHERE created_at >= (SELECT history_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)
    GROUP BY actor_user_id, candidate_user_id
),

-- -------------------------------------------------------------------
-- Viewer overall HISTORY
-- -------------------------------------------------------------------

viewer_history_stats AS (
    SELECT
        actor_user_id AS user_id,

        COUNT(*)::int AS viewer_history_event_count,

        COUNT(DISTINCT candidate_user_id)::int
            AS viewer_history_unique_candidates,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
        )::int AS viewer_history_profile_views,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
        )::int AS viewer_history_favorites,

        COUNT(*) FILTER (
            WHERE event_type = 'poke'
        )::int AS viewer_history_pokes,

        COUNT(*) FILTER (
            WHERE event_type = 'image_reaction'
        )::int AS viewer_history_image_reactions,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
        )::int AS viewer_history_messages,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS viewer_history_profile_views_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS viewer_history_favorites_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS viewer_history_messages_30d,

        COUNT(DISTINCT candidate_user_id) FILTER (
            WHERE created_at >= (
                SELECT history_30d_start_at FROM windows
            )
        )::int AS viewer_history_unique_candidates_30d

    FROM valid_events
    WHERE created_at >= (SELECT history_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)
    GROUP BY actor_user_id
),

-- -------------------------------------------------------------------
-- Candidate popularity / received HISTORY
-- -------------------------------------------------------------------

candidate_history_stats AS (
    SELECT
        candidate_user_id AS user_id,

        COUNT(*)::int AS candidate_history_received_events,

        COUNT(DISTINCT actor_user_id)::int
            AS candidate_history_unique_viewers,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
        )::int AS candidate_history_received_profile_views,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
        )::int AS candidate_history_received_favorites,

        COUNT(*) FILTER (
            WHERE event_type = 'poke'
        )::int AS candidate_history_received_pokes,

        COUNT(*) FILTER (
            WHERE event_type = 'image_reaction'
        )::int AS candidate_history_received_image_reactions,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
        )::int AS candidate_history_received_messages,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS candidate_history_received_profile_views_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS candidate_history_received_favorites_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (
                  SELECT history_30d_start_at FROM windows
              )
        )::int AS candidate_history_received_messages_30d,

        COUNT(DISTINCT actor_user_id) FILTER (
            WHERE created_at >= (
                SELECT history_30d_start_at FROM windows
            )
        )::int AS candidate_history_unique_viewers_30d

    FROM valid_events
    WHERE created_at >= (SELECT history_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)
    GROUP BY candidate_user_id
),

-- -------------------------------------------------------------------
-- OUTCOME period
-- -------------------------------------------------------------------

outcome_pair_stats AS (
    SELECT
        actor_user_id,
        candidate_user_id,

        MIN(created_at) AS first_outcome_interaction_at,
        MAX(created_at) AS last_outcome_interaction_at,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
        )::int AS outcome_profile_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
        )::int AS outcome_favorite_count,

        COUNT(*) FILTER (
            WHERE event_type = 'poke'
        )::int AS outcome_poke_count,

        COUNT(*) FILTER (
            WHERE event_type = 'image_reaction'
        )::int AS outcome_image_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_view'
        )::int AS outcome_story_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reaction'
        )::int AS outcome_story_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reply'
        )::int AS outcome_story_reply_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
        )::int AS outcome_message_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message_reaction'
        )::int AS outcome_message_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'private_album_grant'
        )::int AS outcome_private_album_grant_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_request'
        )::int AS outcome_partner_request_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_accepted'
        )::int AS outcome_partner_accepted_count,

        COUNT(*) FILTER (
            WHERE event_type = 'block'
        )::int AS outcome_block_count,

        COUNT(*) FILTER (
            WHERE event_type = 'report'
        )::int AS outcome_report_count

    FROM valid_events
    WHERE created_at >= (SELECT cutoff_at FROM windows)
      AND created_at < (SELECT outcome_end_at FROM windows)
    GROUP BY actor_user_id, candidate_user_id
),

-- -------------------------------------------------------------------
-- Image state existing before cutoff
-- -------------------------------------------------------------------

image_stats AS (
    SELECT
        user_id,

        COUNT(*)::int AS image_count,

        COUNT(*) FILTER (
            WHERE is_public = TRUE
        )::int AS public_image_count,

        COUNT(*) FILTER (
            WHERE is_public = FALSE
        )::int AS private_image_count,

        COUNT(*) FILTER (
            WHERE is_avatar = TRUE
        )::int AS avatar_image_count,

        COUNT(*) FILTER (
            WHERE descriptor_processed_at IS NOT NULL
              AND descriptor_processed_at < (
                  SELECT cutoff_at FROM windows
              )
        )::int AS descriptor_processed_count,

        MAX(created_at) AS latest_image_at

    FROM user_images
    WHERE created_at < (SELECT cutoff_at FROM windows)
    GROUP BY user_id
)

SELECT
    -- ---------------------------------------------------------------
    -- Metadata
    -- ---------------------------------------------------------------

    (SELECT history_start_at FROM windows) AS history_start_at,
    (SELECT cutoff_at FROM windows) AS cutoff_at,
    (SELECT outcome_end_at FROM windows) AS outcome_end_at,

    key.actor_user_id AS viewer_user_id,
    key.candidate_user_id AS candidate_user_id,

    history.first_history_interaction_at,
    history.last_history_interaction_at,

    outcome.first_outcome_interaction_at,
    outcome.last_outcome_interaction_at,

    -- ---------------------------------------------------------------
    -- Pair history
    -- ---------------------------------------------------------------

    COALESCE(history.history_profile_view_count, 0)
        AS history_profile_view_count,

    COALESCE(history.history_favorite_count, 0)
        AS history_favorite_count,

    COALESCE(history.history_poke_count, 0)
        AS history_poke_count,

    COALESCE(history.history_image_reaction_count, 0)
        AS history_image_reaction_count,

    COALESCE(history.history_story_view_count, 0)
        AS history_story_view_count,

    COALESCE(history.history_story_reaction_count, 0)
        AS history_story_reaction_count,

    COALESCE(history.history_story_reply_count, 0)
        AS history_story_reply_count,

    COALESCE(history.history_message_count, 0)
        AS history_message_count,

    COALESCE(history.history_message_reaction_count, 0)
        AS history_message_reaction_count,

    COALESCE(history.history_private_album_grant_count, 0)
        AS history_private_album_grant_count,

    COALESCE(history.history_partner_request_count, 0)
        AS history_partner_request_count,

    COALESCE(history.history_partner_accepted_count, 0)
        AS history_partner_accepted_count,

    COALESCE(history.history_block_count, 0)
        AS history_block_count,

    COALESCE(history.history_report_count, 0)
        AS history_report_count,

    COALESCE(history.history_all_events_7d, 0)
        AS history_all_events_7d,

    COALESCE(history.history_all_events_30d, 0)
        AS history_all_events_30d,

    COALESCE(history.history_all_events_90d, 0)
        AS history_all_events_90d,

    COALESCE(history.history_profile_views_7d, 0)
        AS history_profile_views_7d,

    COALESCE(history.history_profile_views_30d, 0)
        AS history_profile_views_30d,

    COALESCE(history.history_favorites_7d, 0)
        AS history_favorites_7d,

    COALESCE(history.history_favorites_30d, 0)
        AS history_favorites_30d,

    COALESCE(history.history_messages_7d, 0)
        AS history_messages_7d,

    COALESCE(history.history_messages_30d, 0)
        AS history_messages_30d,

    -- ---------------------------------------------------------------
    -- Reverse pair history
    -- ---------------------------------------------------------------

    COALESCE(reverse_history.history_profile_view_count, 0)
        AS reverse_history_profile_view_count,

    COALESCE(reverse_history.history_favorite_count, 0)
        AS reverse_history_favorite_count,

    COALESCE(reverse_history.history_poke_count, 0)
        AS reverse_history_poke_count,

    COALESCE(reverse_history.history_image_reaction_count, 0)
        AS reverse_history_image_reaction_count,

    COALESCE(reverse_history.history_story_view_count, 0)
        AS reverse_history_story_view_count,

    COALESCE(reverse_history.history_story_reaction_count, 0)
        AS reverse_history_story_reaction_count,

    COALESCE(reverse_history.history_story_reply_count, 0)
        AS reverse_history_story_reply_count,

    COALESCE(reverse_history.history_message_count, 0)
        AS reverse_history_message_count,

    COALESCE(reverse_history.history_message_reaction_count, 0)
        AS reverse_history_message_reaction_count,

    COALESCE(reverse_history.history_private_album_grant_count, 0)
        AS reverse_history_private_album_grant_count,

    COALESCE(reverse_history.history_partner_request_count, 0)
        AS reverse_history_partner_request_count,

    COALESCE(reverse_history.history_partner_accepted_count, 0)
        AS reverse_history_partner_accepted_count,

    COALESCE(reverse_history.history_block_count, 0)
        AS reverse_history_block_count,

    COALESCE(reverse_history.history_report_count, 0)
        AS reverse_history_report_count,

    -- ---------------------------------------------------------------
    -- Viewer-wide history
    -- ---------------------------------------------------------------

    COALESCE(viewer_history.viewer_history_event_count, 0)
        AS viewer_history_event_count,

    COALESCE(viewer_history.viewer_history_unique_candidates, 0)
        AS viewer_history_unique_candidates,

    COALESCE(viewer_history.viewer_history_profile_views, 0)
        AS viewer_history_profile_views,

    COALESCE(viewer_history.viewer_history_favorites, 0)
        AS viewer_history_favorites,

    COALESCE(viewer_history.viewer_history_pokes, 0)
        AS viewer_history_pokes,

    COALESCE(viewer_history.viewer_history_image_reactions, 0)
        AS viewer_history_image_reactions,

    COALESCE(viewer_history.viewer_history_messages, 0)
        AS viewer_history_messages,

    COALESCE(viewer_history.viewer_history_profile_views_30d, 0)
        AS viewer_history_profile_views_30d,

    COALESCE(viewer_history.viewer_history_favorites_30d, 0)
        AS viewer_history_favorites_30d,

    COALESCE(viewer_history.viewer_history_messages_30d, 0)
        AS viewer_history_messages_30d,

    COALESCE(viewer_history.viewer_history_unique_candidates_30d, 0)
        AS viewer_history_unique_candidates_30d,

    -- ---------------------------------------------------------------
    -- Candidate history
    -- ---------------------------------------------------------------

    COALESCE(candidate_history.candidate_history_received_events, 0)
        AS candidate_history_received_events,

    COALESCE(candidate_history.candidate_history_unique_viewers, 0)
        AS candidate_history_unique_viewers,

    COALESCE(
        candidate_history.candidate_history_received_profile_views,
        0
    ) AS candidate_history_received_profile_views,

    COALESCE(
        candidate_history.candidate_history_received_favorites,
        0
    ) AS candidate_history_received_favorites,

    COALESCE(
        candidate_history.candidate_history_received_pokes,
        0
    ) AS candidate_history_received_pokes,

    COALESCE(
        candidate_history.candidate_history_received_image_reactions,
        0
    ) AS candidate_history_received_image_reactions,

    COALESCE(
        candidate_history.candidate_history_received_messages,
        0
    ) AS candidate_history_received_messages,

    COALESCE(
        candidate_history.candidate_history_received_profile_views_30d,
        0
    ) AS candidate_history_received_profile_views_30d,

    COALESCE(
        candidate_history.candidate_history_received_favorites_30d,
        0
    ) AS candidate_history_received_favorites_30d,

    COALESCE(
        candidate_history.candidate_history_received_messages_30d,
        0
    ) AS candidate_history_received_messages_30d,

    COALESCE(
        candidate_history.candidate_history_unique_viewers_30d,
        0
    ) AS candidate_history_unique_viewers_30d,

    -- ---------------------------------------------------------------
    -- Outcome counts
    -- ---------------------------------------------------------------

    COALESCE(outcome.outcome_profile_view_count, 0)
        AS outcome_profile_view_count,

    COALESCE(outcome.outcome_favorite_count, 0)
        AS outcome_favorite_count,

    COALESCE(outcome.outcome_poke_count, 0)
        AS outcome_poke_count,

    COALESCE(outcome.outcome_image_reaction_count, 0)
        AS outcome_image_reaction_count,

    COALESCE(outcome.outcome_story_view_count, 0)
        AS outcome_story_view_count,

    COALESCE(outcome.outcome_story_reaction_count, 0)
        AS outcome_story_reaction_count,

    COALESCE(outcome.outcome_story_reply_count, 0)
        AS outcome_story_reply_count,

    COALESCE(outcome.outcome_message_count, 0)
        AS outcome_message_count,

    COALESCE(outcome.outcome_message_reaction_count, 0)
        AS outcome_message_reaction_count,

    COALESCE(outcome.outcome_private_album_grant_count, 0)
        AS outcome_private_album_grant_count,

    COALESCE(outcome.outcome_partner_request_count, 0)
        AS outcome_partner_request_count,

    COALESCE(outcome.outcome_partner_accepted_count, 0)
        AS outcome_partner_accepted_count,

    COALESCE(outcome.outcome_block_count, 0)
        AS outcome_block_count,

    COALESCE(outcome.outcome_report_count, 0)
        AS outcome_report_count,

    -- Reverse outcomes
    COALESCE(reverse_outcome.outcome_message_count, 0)
        AS reverse_outcome_message_count,

    COALESCE(reverse_outcome.outcome_favorite_count, 0)
        AS reverse_outcome_favorite_count,

    COALESCE(reverse_outcome.outcome_poke_count, 0)
        AS reverse_outcome_poke_count,

    COALESCE(reverse_outcome.outcome_block_count, 0)
        AS reverse_outcome_block_count,

    COALESCE(reverse_outcome.outcome_report_count, 0)
        AS reverse_outcome_report_count,

    -- ---------------------------------------------------------------
    -- Boolean targets
    -- ---------------------------------------------------------------

    (
        COALESCE(outcome.outcome_favorite_count, 0) > 0
    ) AS outcome_favorite,

    (
        COALESCE(outcome.outcome_poke_count, 0) > 0
    ) AS outcome_poke,

    (
        COALESCE(outcome.outcome_image_reaction_count, 0) > 0
    ) AS outcome_image_reaction,

    (
        COALESCE(outcome.outcome_story_reaction_count, 0) > 0
        OR COALESCE(outcome.outcome_story_reply_count, 0) > 0
    ) AS outcome_story_engagement,

    (
        COALESCE(outcome.outcome_message_count, 0) > 0
    ) AS outcome_message_started,

    (
        COALESCE(outcome.outcome_message_count, 0) > 0
        AND COALESCE(reverse_outcome.outcome_message_count, 0) > 0
    ) AS outcome_reciprocal_message,

    (
        COALESCE(outcome.outcome_private_album_grant_count, 0) > 0
    ) AS outcome_private_album_grant,

    (
        COALESCE(outcome.outcome_partner_accepted_count, 0) > 0
    ) AS outcome_partner_accepted,

    (
        COALESCE(outcome.outcome_block_count, 0) > 0
    ) AS outcome_block,

    (
        COALESCE(outcome.outcome_report_count, 0) > 0
    ) AS outcome_report,

    (
        COALESCE(outcome.outcome_favorite_count, 0) > 0
        OR COALESCE(outcome.outcome_poke_count, 0) > 0
        OR COALESCE(outcome.outcome_image_reaction_count, 0) > 0
        OR COALESCE(outcome.outcome_story_reaction_count, 0) > 0
        OR COALESCE(outcome.outcome_story_reply_count, 0) > 0
        OR COALESCE(outcome.outcome_message_count, 0) > 0
        OR COALESCE(outcome.outcome_private_album_grant_count, 0) > 0
        OR COALESCE(outcome.outcome_partner_accepted_count, 0) > 0
    ) AS outcome_any_positive,

    -- ---------------------------------------------------------------
    -- Pair distance
    -- ---------------------------------------------------------------

    CASE
        WHEN COALESCE(viewer.coord, viewer.home_coord) IS NULL
          OR COALESCE(candidate.coord, candidate.home_coord) IS NULL
        THEN NULL
        ELSE ST_Distance(
            COALESCE(viewer.coord, viewer.home_coord),
            COALESCE(candidate.coord, candidate.home_coord)
        ) / 1000.0
    END AS distance_km,

    -- ---------------------------------------------------------------
    -- Viewer
    -- ---------------------------------------------------------------

    CASE
        WHEN viewer.date_of_birth IS NULL THEN NULL
        ELSE EXTRACT(
            YEAR FROM age(
                (SELECT cutoff_at FROM windows)::date,
                viewer.date_of_birth
            )
        )::int
    END AS viewer_age,

    viewer.height AS viewer_height,
    viewer.weight AS viewer_weight,
    viewer.relationship_status AS viewer_relationship_status,
    viewer.pronouns AS viewer_pronouns,
    viewer.position AS viewer_position,
    viewer.body_type AS viewer_body_type,

    COALESCE(viewer.body_hair, ARRAY[]::text[])
        AS viewer_body_hair,

    COALESCE(viewer.my_types, ARRAY[]::text[])
        AS viewer_types,

    COALESCE(viewer.my_fetishes, ARRAY[]::text[])
        AS viewer_fetishes,

    GREATEST(
        0,
        EXTRACT(
            EPOCH FROM (
                (SELECT cutoff_at FROM windows) - viewer.created_at
            )
        ) / 86400.0
    )::double precision AS viewer_account_age_days,

    -- Viewer preferences
    COALESCE(viewer.into_age_ranges, ARRAY[]::text[])
        AS viewer_into_age_ranges,

    COALESCE(viewer.into_height_ranges, ARRAY[]::text[])
        AS viewer_into_height_ranges,

    COALESCE(viewer.into_weight_ranges, ARRAY[]::text[])
        AS viewer_into_weight_ranges,

    viewer.into_position AS viewer_into_position,

    COALESCE(viewer.into_body_types, ARRAY[]::text[])
        AS viewer_into_body_types,

    COALESCE(viewer.into_body_hair, ARRAY[]::text[])
        AS viewer_into_body_hair,

    viewer.into_proximity AS viewer_into_proximity,

    viewer.into_relationship_status
        AS viewer_into_relationship_status,

    COALESCE(viewer.into_types, ARRAY[]::text[])
        AS viewer_into_types,

    COALESCE(viewer.into_fetishes, ARRAY[]::text[])
        AS viewer_into_fetishes,

    -- Viewer settings
    viewer_settings.max_distance_km AS viewer_max_distance_km,
    viewer_settings.min_age AS viewer_min_age,
    viewer_settings.max_age AS viewer_max_age,

    -- Viewer images
    COALESCE(viewer_images.image_count, 0)
        AS viewer_image_count,

    COALESCE(viewer_images.public_image_count, 0)
        AS viewer_public_image_count,

    COALESCE(viewer_images.private_image_count, 0)
        AS viewer_private_image_count,

    COALESCE(viewer_images.avatar_image_count, 0)
        AS viewer_avatar_image_count,

    COALESCE(viewer_images.descriptor_processed_count, 0)
        AS viewer_descriptor_processed_count,

    viewer_images.latest_image_at AS viewer_latest_image_at,

    -- ---------------------------------------------------------------
    -- Candidate
    -- ---------------------------------------------------------------

    CASE
        WHEN candidate.date_of_birth IS NULL THEN NULL
        ELSE EXTRACT(
            YEAR FROM age(
                (SELECT cutoff_at FROM windows)::date,
                candidate.date_of_birth
            )
        )::int
    END AS candidate_age,

    candidate.height AS candidate_height,
    candidate.weight AS candidate_weight,
    candidate.relationship_status AS candidate_relationship_status,
    candidate.pronouns AS candidate_pronouns,
    candidate.position AS candidate_position,
    candidate.body_type AS candidate_body_type,

    COALESCE(candidate.body_hair, ARRAY[]::text[])
        AS candidate_body_hair,

    COALESCE(candidate.my_types, ARRAY[]::text[])
        AS candidate_types,

    COALESCE(candidate.my_fetishes, ARRAY[]::text[])
        AS candidate_fetishes,

    GREATEST(
        0,
        EXTRACT(
            EPOCH FROM (
                (SELECT cutoff_at FROM windows) - candidate.created_at
            )
        ) / 86400.0
    )::double precision AS candidate_account_age_days,

    -- Candidate preferences
    COALESCE(candidate.into_age_ranges, ARRAY[]::text[])
        AS candidate_into_age_ranges,

    COALESCE(candidate.into_height_ranges, ARRAY[]::text[])
        AS candidate_into_height_ranges,

    COALESCE(candidate.into_weight_ranges, ARRAY[]::text[])
        AS candidate_into_weight_ranges,

    candidate.into_position AS candidate_into_position,

    COALESCE(candidate.into_body_types, ARRAY[]::text[])
        AS candidate_into_body_types,

    COALESCE(candidate.into_body_hair, ARRAY[]::text[])
        AS candidate_into_body_hair,

    candidate.into_proximity AS candidate_into_proximity,

    candidate.into_relationship_status
        AS candidate_into_relationship_status,

    COALESCE(candidate.into_types, ARRAY[]::text[])
        AS candidate_into_types,

    COALESCE(candidate.into_fetishes, ARRAY[]::text[])
        AS candidate_into_fetishes,

    -- Candidate settings
    candidate_settings.max_distance_km AS candidate_max_distance_km,
    candidate_settings.min_age AS candidate_min_age,
    candidate_settings.max_age AS candidate_max_age,

    -- Candidate images
    COALESCE(candidate_images.image_count, 0)
        AS candidate_image_count,

    COALESCE(candidate_images.public_image_count, 0)
        AS candidate_public_image_count,

    COALESCE(candidate_images.private_image_count, 0)
        AS candidate_private_image_count,

    COALESCE(candidate_images.avatar_image_count, 0)
        AS candidate_avatar_image_count,

    COALESCE(candidate_images.descriptor_processed_count, 0)
        AS candidate_descriptor_processed_count,

    candidate_images.latest_image_at AS candidate_latest_image_at

FROM pair_keys key

JOIN users viewer
  ON viewer.id = key.actor_user_id

JOIN users candidate
  ON candidate.id = key.candidate_user_id

LEFT JOIN history_pair_stats history
  ON history.actor_user_id = key.actor_user_id
 AND history.candidate_user_id = key.candidate_user_id

LEFT JOIN history_pair_stats reverse_history
  ON reverse_history.actor_user_id = key.candidate_user_id
 AND reverse_history.candidate_user_id = key.actor_user_id

LEFT JOIN viewer_history_stats viewer_history
  ON viewer_history.user_id = key.actor_user_id

LEFT JOIN candidate_history_stats candidate_history
  ON candidate_history.user_id = key.candidate_user_id

LEFT JOIN outcome_pair_stats outcome
  ON outcome.actor_user_id = key.actor_user_id
 AND outcome.candidate_user_id = key.candidate_user_id

LEFT JOIN outcome_pair_stats reverse_outcome
  ON reverse_outcome.actor_user_id = key.candidate_user_id
 AND reverse_outcome.candidate_user_id = key.actor_user_id

LEFT JOIN image_stats viewer_images
  ON viewer_images.user_id = viewer.id

LEFT JOIN image_stats candidate_images
  ON candidate_images.user_id = candidate.id

LEFT JOIN user_settings viewer_settings
  ON viewer_settings.user_id = viewer.id

LEFT JOIN user_settings candidate_settings
  ON candidate_settings.user_id = candidate.id

WHERE viewer.is_bot = FALSE
  AND candidate.is_bot = FALSE
  AND viewer.is_banned = FALSE
  AND candidate.is_banned = FALSE

ORDER BY key.actor_user_id, key.candidate_user_id
"""


# ---------------------------------------------------------------------------
# Historical user-affinity source
#
# This query intentionally uses the exact dataset cutoff supplied by the
# training rows. It never reads outcome-period behavior.
# ---------------------------------------------------------------------------


AFFINITY_HISTORY_QUERY = """
WITH events AS (
    SELECT
        viewer_user_id AS actor_user_id,
        owner_user_id AS candidate_user_id,
        'profile_view'::text AS event_type,
        created_at
    FROM user_profile_views
    WHERE viewer_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        from_user_id,
        to_user_id,
        'favorite'::text,
        created_at
    FROM user_favorites
    WHERE from_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        from_user_id,
        to_user_id,
        'poke'::text,
        created_at
    FROM user_pokes
    WHERE from_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        reaction.from_user_id,
        image.user_id,
        'image_reaction'::text,
        reaction.created_at
    FROM image_reactions reaction
    JOIN user_images image
      ON image.id = reaction.image_id
    WHERE reaction.from_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND reaction.created_at >= %(history_start_at)s
      AND reaction.created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        story_view.viewer_user_id,
        story.user_id,
        'story_view'::text,
        story_view.viewed_at
    FROM story_views story_view
    JOIN user_stories story
      ON story.id = story_view.story_id
    WHERE story_view.viewer_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND story_view.viewed_at >= %(history_start_at)s
      AND story_view.viewed_at < %(cutoff_at)s

    UNION ALL

    SELECT
        reaction.from_user_id,
        story.user_id,
        CASE
            WHEN reaction.kind = 'reply' THEN 'story_reply'::text
            ELSE 'story_reaction'::text
        END,
        reaction.created_at
    FROM story_reactions reaction
    JOIN user_stories story
      ON story.id = reaction.story_id
    WHERE reaction.from_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND reaction.created_at >= %(history_start_at)s
      AND reaction.created_at < %(cutoff_at)s

    UNION ALL

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
    WHERE message.sender_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND message.deleted_at IS NULL
      AND message.created_at >= %(history_start_at)s
      AND message.created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        reaction.user_id,
        message.sender_user_id,
        'message_reaction'::text,
        reaction.created_at
    FROM message_reactions reaction
    JOIN messages message
      ON message.id = reaction.message_id
    WHERE reaction.user_id = ANY(%(actor_user_ids)s::bigint[])
      AND message.sender_user_id IS NOT NULL
      AND reaction.user_id <> message.sender_user_id
      AND reaction.created_at >= %(history_start_at)s
      AND reaction.created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        blocker_user_id,
        blocked_user_id,
        'block'::text,
        created_at
    FROM user_blocks
    WHERE blocker_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        owner_user_id,
        viewer_user_id,
        'private_album_grant'::text,
        granted_at
    FROM user_private_album_access
    WHERE owner_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND granted_at >= %(history_start_at)s
      AND granted_at < %(cutoff_at)s

    UNION ALL

    SELECT
        requester_user_id,
        target_user_id,
        'partner_request'::text,
        created_at
    FROM partner_requests
    WHERE requester_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s

    UNION ALL

    SELECT
        requester_user_id,
        target_user_id,
        'partner_accepted'::text,
        responded_at
    FROM partner_requests
    WHERE requester_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND status = 'accepted'
      AND responded_at IS NOT NULL
      AND responded_at >= %(history_start_at)s
      AND responded_at < %(cutoff_at)s

    UNION ALL

    SELECT
        target_user_id,
        requester_user_id,
        'partner_accepted'::text,
        responded_at
    FROM partner_requests
    WHERE target_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND status = 'accepted'
      AND responded_at IS NOT NULL
      AND responded_at >= %(history_start_at)s
      AND responded_at < %(cutoff_at)s

    UNION ALL

    SELECT
        reporter_user_id,
        reported_user_id,
        'report'::text,
        created_at
    FROM support_reports
    WHERE reporter_user_id = ANY(%(actor_user_ids)s::bigint[])
      AND reported_user_id IS NOT NULL
      AND created_at >= %(history_start_at)s
      AND created_at < %(cutoff_at)s
),

valid_events AS (
    SELECT
        actor_user_id,
        candidate_user_id,
        event_type,
        created_at
    FROM events
    WHERE actor_user_id IS NOT NULL
      AND candidate_user_id IS NOT NULL
      AND actor_user_id <> candidate_user_id
),

pair_stats AS (
    SELECT
        actor_user_id,
        candidate_user_id,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
        )::int AS profile_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
        )::int AS favorite_count,

        COUNT(*) FILTER (
            WHERE event_type = 'poke'
        )::int AS poke_count,

        COUNT(*) FILTER (
            WHERE event_type = 'image_reaction'
        )::int AS image_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_view'
        )::int AS story_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reaction'
        )::int AS story_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reply'
        )::int AS story_reply_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
        )::int AS message_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message_reaction'
        )::int AS message_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'private_album_grant'
        )::int AS private_album_grant_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_request'
        )::int AS partner_request_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_accepted'
        )::int AS partner_accepted_count,

        COUNT(*) FILTER (
            WHERE event_type = 'block'
        )::int AS block_count,

        COUNT(*) FILTER (
            WHERE event_type = 'report'
        )::int AS report_count,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_profile_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_favorite_count,

        COUNT(*) FILTER (
            WHERE event_type = 'poke'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_poke_count,

        COUNT(*) FILTER (
            WHERE event_type = 'image_reaction'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_image_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_view'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_story_view_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reaction'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_story_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'story_reply'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_story_reply_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_message_count,

        COUNT(*) FILTER (
            WHERE event_type = 'message_reaction'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_message_reaction_count,

        COUNT(*) FILTER (
            WHERE event_type = 'private_album_grant'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_private_album_grant_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_request'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_partner_request_count,

        COUNT(*) FILTER (
            WHERE event_type = 'partner_accepted'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_partner_accepted_count,

        COUNT(*) FILTER (
            WHERE event_type = 'block'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_block_count,

        COUNT(*) FILTER (
            WHERE event_type = 'report'
              AND created_at >= %(recent_start_at)s
        )::int AS recent_report_count

    FROM valid_events
    GROUP BY actor_user_id, candidate_user_id
)

SELECT
    stats.actor_user_id,
    stats.candidate_user_id,

    stats.profile_view_count,
    stats.favorite_count,
    stats.poke_count,
    stats.image_reaction_count,
    stats.story_view_count,
    stats.story_reaction_count,
    stats.story_reply_count,
    stats.message_count,
    stats.message_reaction_count,
    stats.private_album_grant_count,
    stats.partner_request_count,
    stats.partner_accepted_count,
    stats.block_count,
    stats.report_count,

    stats.recent_profile_view_count,
    stats.recent_favorite_count,
    stats.recent_poke_count,
    stats.recent_image_reaction_count,
    stats.recent_story_view_count,
    stats.recent_story_reaction_count,
    stats.recent_story_reply_count,
    stats.recent_message_count,
    stats.recent_message_reaction_count,
    stats.recent_private_album_grant_count,
    stats.recent_partner_request_count,
    stats.recent_partner_accepted_count,
    stats.recent_block_count,
    stats.recent_report_count,

    CASE
        WHEN historical_candidate.date_of_birth IS NULL THEN NULL
        ELSE EXTRACT(
            YEAR FROM age(
                %(cutoff_at)s::date,
                historical_candidate.date_of_birth
            )
        )::int
    END AS candidate_age,

    historical_candidate.height AS candidate_height,
    historical_candidate.weight AS candidate_weight,
    historical_candidate.relationship_status
        AS candidate_relationship_status,
    historical_candidate.position AS candidate_position,
    historical_candidate.body_type AS candidate_body_type,

    COALESCE(
        historical_candidate.body_hair,
        ARRAY[]::text[]
    ) AS candidate_body_hair,

    COALESCE(
        historical_candidate.my_types,
        ARRAY[]::text[]
    ) AS candidate_types,

    COALESCE(
        historical_candidate.my_fetishes,
        ARRAY[]::text[]
    ) AS candidate_fetishes,

    CASE
        WHEN COALESCE(actor.coord, actor.home_coord) IS NULL
          OR COALESCE(
              historical_candidate.coord,
              historical_candidate.home_coord
          ) IS NULL
        THEN NULL
        ELSE ST_Distance(
            COALESCE(actor.coord, actor.home_coord),
            COALESCE(
                historical_candidate.coord,
                historical_candidate.home_coord
            )
        ) / 1000.0
    END AS distance_km

FROM pair_stats stats

JOIN users actor
  ON actor.id = stats.actor_user_id

JOIN users historical_candidate
  ON historical_candidate.id = stats.candidate_user_id

WHERE historical_candidate.is_bot = FALSE

ORDER BY stats.actor_user_id, stats.candidate_user_id
"""


AFFINITY_EVENT_TYPES = (
    "profile_view",
    "favorite",
    "poke",
    "image_reaction",
    "story_view",
    "story_reaction",
    "story_reply",
    "message",
    "message_reaction",
    "private_album_grant",
    "partner_request",
    "partner_accepted",
    "block",
    "report",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Peters ML history/outcome training rows."
    )

    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"History window. Default: {DEFAULT_HISTORY_DAYS}",
    )

    parser.add_argument(
        "--outcome-days",
        type=int,
        default=DEFAULT_OUTCOME_DAYS,
        help=f"Outcome window. Default: {DEFAULT_OUTCOME_DAYS}",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL. Default: {DEFAULT_OUTPUT}",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum row count.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Fetch batch size. Default: {DEFAULT_BATCH_SIZE}",
    )

    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=DEFAULT_GRAPH_BATCH_SIZE,
        help=(
            "Maximum candidates per PetersGraph historical feature request. "
            f"Default: {DEFAULT_GRAPH_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=DEFAULT_SNAPSHOT_COUNT,
        help=(
            "Number of historical cutoff snapshots to build. "
            f"Default: {DEFAULT_SNAPSHOT_COUNT}"
        ),
    )

    parser.add_argument(
        "--snapshot-step-days",
        type=int,
        default=DEFAULT_SNAPSHOT_STEP_DAYS,
        help=(
            "Days between historical cutoff snapshots. "
            f"Default: {DEFAULT_SNAPSHOT_STEP_DAYS}"
        ),
    )

    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise SystemExit(f"{name} is not set")

    return value


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, set):
        return sorted(value)

    return str(value)


def parse_height_inches(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    match = re.fullmatch(
        r"\s*(\d+)\s*'\s*(\d+)\s*\"?\s*",
        text,
    )

    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))

        if 0 <= inches <= 11:
            return float(feet * 12 + inches)

    try:
        numeric = float(text)

        if 36 <= numeric <= 96:
            return numeric

    except ValueError:
        pass

    return None


def prepare_output_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    row["viewer_height_inches"] = parse_height_inches(
        row.get("viewer_height")
    )

    row["candidate_height_inches"] = parse_height_inches(
        row.get("candidate_height")
    )

    return row


def affinity_target_from_history_row(
    row: dict[str, Any],
) -> AffinityTarget:
    feature_row = {
        "candidate_age": row.get(
            "candidate_age"
        ),
        "candidate_height_inches": parse_height_inches(
            row.get("candidate_height")
        ),
        "candidate_weight": row.get(
            "candidate_weight"
        ),
        "candidate_relationship_status": row.get(
            "candidate_relationship_status"
        ),
        "candidate_position": row.get(
            "candidate_position"
        ),
        "candidate_body_type": row.get(
            "candidate_body_type"
        ),
        "candidate_body_hair": row.get(
            "candidate_body_hair"
        ),
        "candidate_types": row.get(
            "candidate_types"
        ),
        "candidate_fetishes": row.get(
            "candidate_fetishes"
        ),
        "distance_km": row.get(
            "distance_km"
        ),
    }

    return AffinityTarget.from_feature_row(
        feature_row,
        "candidate",
    )


def affinity_event_counts(
    row: dict[str, Any],
    recent: bool,
) -> dict[str, int]:
    prefix = (
        "recent_"
        if recent
        else ""
    )

    return {
        event_type: int(
            row.get(
                f"{prefix}{event_type}_count",
                0,
            )
            or 0
        )
        for event_type in AFFINITY_EVENT_TYPES
    }


def build_affinity_observation(
    row: dict[str, Any],
    recent: bool,
) -> AffinityObservation:
    return AffinityObservation(
        profile=affinity_target_from_history_row(
            row
        ),
        event_counts=affinity_event_counts(
            row=row,
            recent=recent,
        ),
    )


def load_historical_affinity_states(
    conn: psycopg.Connection,
    user_ids: list[int],
    history_start_at: datetime,
    cutoff_at: datetime,
) -> dict[
    int,
    tuple[
        UserAffinityState,
        UserAffinityState,
    ],
]:
    """
    Build long-term and recent behavioral affinity states for the supplied
    users using only events before cutoff_at.
    """

    normalized_user_ids = sorted(
        {
            int(user_id)
            for user_id in user_ids
            if int(user_id) > 0
        }
    )

    if not normalized_user_ids:
        return {}

    recent_start_at = (
        cutoff_at
        - timedelta(
            days=RECENT_AFFINITY_DAYS
        )
    )

    params = {
        "actor_user_ids":
            normalized_user_ids,
        "history_start_at":
            history_start_at,
        "cutoff_at":
            cutoff_at,
        "recent_start_at":
            recent_start_at,
    }

    with conn.cursor(
        row_factory=dict_row
    ) as cursor:
        cursor.execute(
            AFFINITY_HISTORY_QUERY,
            params,
        )

        history_rows = [
            dict(row)
            for row in cursor.fetchall()
        ]

    long_term_by_user: dict[
        int,
        list[AffinityObservation],
    ] = {
        user_id: []
        for user_id in normalized_user_ids
    }

    recent_by_user: dict[
        int,
        list[AffinityObservation],
    ] = {
        user_id: []
        for user_id in normalized_user_ids
    }

    for row in history_rows:
        actor_user_id = int(
            row["actor_user_id"]
        )

        long_term_by_user.setdefault(
            actor_user_id,
            [],
        ).append(
            build_affinity_observation(
                row=row,
                recent=False,
            )
        )

        recent_by_user.setdefault(
            actor_user_id,
            [],
        ).append(
            build_affinity_observation(
                row=row,
                recent=True,
            )
        )

    return {
        user_id: (
            build_user_affinity_state(
                long_term_by_user.get(
                    user_id,
                    [],
                )
            ),
            build_user_affinity_state(
                recent_by_user.get(
                    user_id,
                    [],
                )
            ),
        )
        for user_id in normalized_user_ids
    }


def enrich_rows_with_historical_affinities(
    rows: list[dict[str, Any]],
    conn: psycopg.Connection,
    affinity_cache: dict[
        int,
        tuple[
            UserAffinityState,
            UserAffinityState,
        ],
    ],
) -> None:
    """
    Add cutoff-safe directional viewer/candidate affinity sections.

    viewer_affinities:
        How well the candidate matches the viewer's demonstrated behavior.

    candidate_affinities:
        How well the viewer matches the candidate's demonstrated behavior.
    """

    if not rows:
        return

    cutoff_at = rows[0].get(
        "cutoff_at"
    )

    history_start_at = rows[0].get(
        "history_start_at"
    )

    if not isinstance(
        cutoff_at,
        datetime,
    ):
        raise RuntimeError(
            "Training row cutoff_at is missing or invalid"
        )

    if not isinstance(
        history_start_at,
        datetime,
    ):
        raise RuntimeError(
            "Training row history_start_at is missing or invalid"
        )

    if cutoff_at.tzinfo is None:
        raise RuntimeError(
            "Training row cutoff_at must be timezone-aware"
        )

    if history_start_at.tzinfo is None:
        raise RuntimeError(
            "Training row history_start_at must be timezone-aware"
        )

    relevant_user_ids = sorted(
        {
            int(row["viewer_user_id"])
            for row in rows
        }
        | {
            int(row["candidate_user_id"])
            for row in rows
        }
    )

    missing_user_ids = [
        user_id
        for user_id in relevant_user_ids
        if user_id not in affinity_cache
    ]

    if missing_user_ids:
        loaded_states = (
            load_historical_affinity_states(
                conn=conn,
                user_ids=missing_user_ids,
                history_start_at=(
                    history_start_at
                ),
                cutoff_at=cutoff_at,
            )
        )

        affinity_cache.update(
            loaded_states
        )

    unresolved = [
        user_id
        for user_id in relevant_user_ids
        if user_id not in affinity_cache
    ]

    if unresolved:
        raise RuntimeError(
            "Historical affinity state missing for user IDs: "
            + ", ".join(
                str(user_id)
                for user_id in unresolved
            )
        )

    for row in rows:
        row_cutoff = row.get(
            "cutoff_at"
        )

        row_history_start = row.get(
            "history_start_at"
        )

        if row_cutoff != cutoff_at:
            raise RuntimeError(
                "Training batch contains multiple cutoff_at values"
            )

        if row_history_start != history_start_at:
            raise RuntimeError(
                "Training batch contains multiple history_start_at values"
            )

        viewer_user_id = int(
            row["viewer_user_id"]
        )

        candidate_user_id = int(
            row["candidate_user_id"]
        )

        (
            viewer_long_term,
            viewer_recent,
        ) = affinity_cache[
            viewer_user_id
        ]

        (
            candidate_long_term,
            candidate_recent,
        ) = affinity_cache[
            candidate_user_id
        ]

        viewer_target = (
            AffinityTarget.from_feature_row(
                row,
                "viewer",
            )
        )

        candidate_target = (
            AffinityTarget.from_feature_row(
                row,
                "candidate",
            )
        )

        row["viewer_affinities"] = (
            build_directional_affinity_features(
                long_term_state=(
                    viewer_long_term
                ),
                recent_state=(
                    viewer_recent
                ),
                target=(
                    candidate_target
                ),
            )
        )

        row["candidate_affinities"] = (
            build_directional_affinity_features(
                long_term_state=(
                    candidate_long_term
                ),
                recent_state=(
                    candidate_recent
                ),
                target=(
                    viewer_target
                ),
            )
        )


def merge_graph_pair_features(
    row: dict[str, Any],
    pair: GraphPairFeatures,
) -> None:
    """
    Attach cutoff-safe PetersGraph features to one training row.

    graph_is_hard_excluded is retained as metadata but is intentionally
    not part of the neural numeric input contract.
    """

    row.update(
        pair.as_feature_dict()
    )

    row["graph_two_hop_path_count"] = (
        pair.two_hop_path_count
    )

    row["graph_similar_user_support_count"] = (
        pair.similar_user_support_count
    )

    row["graph_is_hard_excluded"] = (
        pair.is_hard_excluded
    )


def enrich_rows_with_historical_graph(
    rows: list[dict[str, Any]],
    graph_client: PetersGraphClient,
    graph_batch_size: int,
) -> tuple[int, int, int]:
    """
    Add historical PetersGraph features to a SQL batch.

    All SQL rows in one enrichment batch share one cutoff_at.

    Rows are grouped by viewer so PetersGraph can calculate shared
    viewer-level graph evidence for multiple candidates together.

    Returns:
        graph request count,
        historical graph node count,
        historical graph edge count.
    """

    if not rows:
        return 0, 0, 0

    if graph_batch_size <= 0:
        raise ValueError(
            "graph_batch_size must be greater than 0"
        )

    cutoff_at = rows[0].get(
        "cutoff_at"
    )

    if not isinstance(
        cutoff_at,
        datetime,
    ):
        raise RuntimeError(
            "Training row cutoff_at is missing or invalid"
        )

    if cutoff_at.tzinfo is None:
        raise RuntimeError(
            "Training row cutoff_at must be timezone-aware"
        )

    rows_by_viewer: dict[
        int,
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        row_cutoff = row.get(
            "cutoff_at"
        )

        if row_cutoff != cutoff_at:
            raise RuntimeError(
                "Training batch contains multiple cutoff_at values"
            )

        viewer_user_id = int(
            row["viewer_user_id"]
        )

        rows_by_viewer.setdefault(
            viewer_user_id,
            [],
        ).append(
            row
        )

    graph_requests = 0
    graph_nodes: int | None = None
    graph_edges: int | None = None

    for (
        viewer_user_id,
        viewer_rows,
    ) in rows_by_viewer.items():
        candidate_ids = [
            int(row["candidate_user_id"])
            for row in viewer_rows
        ]

        pair_lookup: dict[
            int,
            GraphPairFeatures,
        ] = {}

        for start in range(
            0,
            len(candidate_ids),
            graph_batch_size,
        ):
            batch_candidate_ids = candidate_ids[
                start:start + graph_batch_size
            ]

            result = (
                graph_client.historical_pair_features(
                    cutoff_at=cutoff_at,
                    viewer_user_id=viewer_user_id,
                    candidate_user_ids=batch_candidate_ids,
                )
            )

            graph_requests += 1

            if graph_nodes is None:
                graph_nodes = (
                    result.graph_nodes
                )

                graph_edges = (
                    result.graph_edges
                )

            elif (
                result.graph_nodes != graph_nodes
                or result.graph_edges != graph_edges
            ):
                raise RuntimeError(
                    "Historical PetersGraph changed within "
                    "one snapshot batch: "
                    f"expected nodes={graph_nodes}, "
                    f"edges={graph_edges}; "
                    f"got nodes={result.graph_nodes}, "
                    f"edges={result.graph_edges}"
                )

            for pair in result.pairs:
                candidate_user_id = int(
                    pair.candidate_user_id
                )

                if candidate_user_id in pair_lookup:
                    raise RuntimeError(
                        "PetersGraph returned duplicate pair "
                        "features for "
                        f"viewer={viewer_user_id}, "
                        f"candidate={candidate_user_id}"
                    )

                pair_lookup[
                    candidate_user_id
                ] = pair

        missing_candidate_ids = [
            candidate_user_id
            for candidate_user_id in candidate_ids
            if candidate_user_id not in pair_lookup
        ]

        if missing_candidate_ids:
            raise RuntimeError(
                "PetersGraph did not return historical "
                "features for "
                f"viewer={viewer_user_id}, "
                f"candidate_ids={missing_candidate_ids}"
            )

        for row in viewer_rows:
            candidate_user_id = int(
                row["candidate_user_id"]
            )

            merge_graph_pair_features(
                row=row,
                pair=pair_lookup[
                    candidate_user_id
                ],
            )

    return (
        graph_requests,
        graph_nodes or 0,
        graph_edges or 0,
    )


def build_query(
    limit: int | None,
) -> str:
    query = DATASET_QUERY

    if limit is not None:
        query += "\nLIMIT %(row_limit)s"

    return query


def load_database_now(
    conn: psycopg.Connection,
) -> datetime:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT now()"
        )
        row = cursor.fetchone()

    if not row:
        raise RuntimeError(
            "Unable to read database time"
        )

    now_at = row[0]

    if not isinstance(now_at, datetime):
        raise RuntimeError(
            "Database now() did not return a datetime"
        )

    if now_at.tzinfo is None:
        raise RuntimeError(
            "Database now() must be timezone-aware"
        )

    return now_at


def build_snapshot_windows(
    now_at: datetime,
    outcome_days: int,
    snapshot_count: int,
    snapshot_step_days: int,
) -> list[tuple[datetime, datetime]]:
    latest_outcome_end_at = now_at
    latest_cutoff_at = (
        latest_outcome_end_at
        - timedelta(days=outcome_days)
    )

    return [
        (
            latest_cutoff_at
            - timedelta(
                days=snapshot_index
                * snapshot_step_days
            ),
            latest_outcome_end_at
            - timedelta(
                days=snapshot_index
                * snapshot_step_days
            ),
        )
        for snapshot_index in range(snapshot_count)
    ]


def update_totals(
    totals: dict[str, int],
    row: dict[str, Any],
) -> None:
    totals["rows"] += 1

    history_events = (
        row["history_profile_view_count"]
        + row["history_favorite_count"]
        + row["history_poke_count"]
        + row["history_image_reaction_count"]
        + row["history_story_view_count"]
        + row["history_story_reaction_count"]
        + row["history_story_reply_count"]
        + row["history_message_count"]
        + row["history_message_reaction_count"]
    )

    if history_events > 0:
        totals[
            "pairs_with_history"
        ] += 1

    totals[
        "outcome_profile_views"
    ] += row[
        "outcome_profile_view_count"
    ]

    totals[
        "outcome_favorites"
    ] += row[
        "outcome_favorite_count"
    ]

    totals[
        "outcome_pokes"
    ] += row[
        "outcome_poke_count"
    ]

    totals[
        "outcome_image_reactions"
    ] += row[
        "outcome_image_reaction_count"
    ]

    totals[
        "outcome_messages"
    ] += row[
        "outcome_message_count"
    ]

    totals[
        "outcome_blocks"
    ] += row[
        "outcome_block_count"
    ]

    totals[
        "outcome_reports"
    ] += row[
        "outcome_report_count"
    ]

    if row[
        "outcome_reciprocal_message"
    ]:
        totals[
            "reciprocal_message_pairs"
        ] += 1

    if row[
        "outcome_any_positive"
    ]:
        totals[
            "positive_pairs"
        ] += 1

    if row.get(
        "graph_is_hard_excluded"
    ):
        totals[
            "historical_hard_exclusions"
        ] += 1


def build_dataset(
    dsn: str,
    output_path: Path,
    history_days: int,
    outcome_days: int,
    limit: int | None,
    batch_size: int,
    graph_batch_size: int,
    snapshot_count: int,
    snapshot_step_days: int,
) -> int:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_output_path = (
        output_path.with_name(
            f"{output_path.name}.tmp"
        )
    )

    if temp_output_path.exists():
        temp_output_path.unlink()

    totals = {
        "rows": 0,
        "pairs_with_history": 0,
        "outcome_profile_views": 0,
        "outcome_favorites": 0,
        "outcome_pokes": 0,
        "outcome_image_reactions": 0,
        "outcome_messages": 0,
        "outcome_blocks": 0,
        "outcome_reports": 0,
        "reciprocal_message_pairs": 0,
        "positive_pairs": 0,
        "historical_hard_exclusions": 0,
        "graph_requests": 0,
    }

    snapshot_summaries: list[dict[str, Any]] = []

    print()
    print("PETERS ML DATASET")
    print("────────────────────────────────────────")
    print(
        f"History window:    "
        f"{history_days} days"
    )
    print(
        f"Outcome window:    "
        f"{outcome_days} days"
    )
    print(
        f"Snapshot count:    "
        f"{snapshot_count}"
    )
    print(
        f"Snapshot step:     "
        f"{snapshot_step_days} days"
    )
    print(
        f"SQL batch size:    "
        f"{batch_size:,}"
    )
    print(
        f"Graph batch size:  "
        f"{graph_batch_size:,}"
    )
    print(
        f"Output:            "
        f"{output_path}"
    )

    if limit is not None:
        print(
            f"Global row limit:  "
            f"{limit:,}"
        )

    print()

    with PetersGraphClient() as graph_client:
        print(
            "PetersGraph:      "
            f"{graph_client.base_url}"
        )

        health = graph_client.health()

        if health.get(
            "status"
        ) != "ok":
            raise RuntimeError(
                "PetersGraph service is not healthy"
            )

        print(
            "Graph service:    "
            f"{health.get('status')}"
        )
        print()

        with psycopg.connect(
            dsn
        ) as conn:
            now_at = load_database_now(
                conn
            )

            snapshot_windows = build_snapshot_windows(
                now_at=now_at,
                outcome_days=outcome_days,
                snapshot_count=snapshot_count,
                snapshot_step_days=(
                    snapshot_step_days
                ),
            )

            with temp_output_path.open(
                "w",
                encoding="utf-8",
            ) as output_file:
                for (
                    snapshot_index,
                    (cutoff_at, outcome_end_at),
                ) in enumerate(
                    snapshot_windows,
                    start=1,
                ):
                    if (
                        limit is not None
                        and totals["rows"] >= limit
                    ):
                        break

                    remaining_limit = (
                        None
                        if limit is None
                        else limit - totals["rows"]
                    )

                    query = build_query(
                        remaining_limit
                    )

                    params: dict[str, Any] = {
                        "cutoff_at": cutoff_at,
                        "outcome_end_at": (
                            outcome_end_at
                        ),
                        "history_days": history_days,
                    }

                    if remaining_limit is not None:
                        params["row_limit"] = (
                            remaining_limit
                        )

                    # Critical: affinity state is cutoff-specific.
                    # Never reuse another snapshot's user state.
                    affinity_cache: dict[
                        int,
                        tuple[
                            UserAffinityState,
                            UserAffinityState,
                        ],
                    ] = {}

                    snapshot_rows = 0
                    snapshot_graph_requests = 0
                    snapshot_graph_nodes: int | None = None
                    snapshot_graph_edges: int | None = None

                    print(
                        f"SNAPSHOT "
                        f"{snapshot_index}/{snapshot_count}"
                    )
                    print(
                        "────────────────────────────────────────"
                    )
                    print(
                        f"Cutoff:       {cutoff_at}"
                    )
                    print(
                        f"Outcome end:  {outcome_end_at}"
                    )

                    cursor_name = (
                        "peters_ml_dataset_"
                        f"{snapshot_index}"
                    )

                    with conn.cursor(
                        name=cursor_name,
                        row_factory=dict_row,
                    ) as cursor:
                        cursor.execute(
                            query,
                            params,
                        )

                        while True:
                            raw_rows = cursor.fetchmany(
                                batch_size
                            )

                            if not raw_rows:
                                break

                            rows = [
                                prepare_output_row(
                                    dict(raw_row)
                                )
                                for raw_row in raw_rows
                            ]

                            for row in rows:
                                row_cutoff = row.get(
                                    "cutoff_at"
                                )
                                row_outcome_end = row.get(
                                    "outcome_end_at"
                                )

                                if row_cutoff != cutoff_at:
                                    raise RuntimeError(
                                        "Dataset cutoff_at "
                                        "does not match "
                                        "requested snapshot"
                                    )

                                if (
                                    row_outcome_end
                                    != outcome_end_at
                                ):
                                    raise RuntimeError(
                                        "Dataset outcome_end_at "
                                        "does not match "
                                        "requested snapshot"
                                    )

                            enrich_rows_with_historical_affinities(
                                rows=rows,
                                conn=conn,
                                affinity_cache=(
                                    affinity_cache
                                ),
                            )

                            (
                                graph_requests,
                                batch_graph_nodes,
                                batch_graph_edges,
                            ) = (
                                enrich_rows_with_historical_graph(
                                    rows=rows,
                                    graph_client=(
                                        graph_client
                                    ),
                                    graph_batch_size=(
                                        graph_batch_size
                                    ),
                                )
                            )

                            snapshot_graph_requests += (
                                graph_requests
                            )
                            totals[
                                "graph_requests"
                            ] += graph_requests

                            if snapshot_graph_nodes is None:
                                snapshot_graph_nodes = (
                                    batch_graph_nodes
                                )
                                snapshot_graph_edges = (
                                    batch_graph_edges
                                )

                            elif (
                                snapshot_graph_nodes
                                != batch_graph_nodes
                                or snapshot_graph_edges
                                != batch_graph_edges
                            ):
                                raise RuntimeError(
                                    "Historical graph metadata "
                                    "changed within one "
                                    "snapshot build"
                                )

                            for row in rows:
                                output_file.write(
                                    json.dumps(
                                        row,
                                        default=json_default,
                                        separators=(
                                            ",",
                                            ":",
                                        ),
                                    )
                                )
                                output_file.write(
                                    "\n"
                                )

                                update_totals(
                                    totals=totals,
                                    row=row,
                                )

                                snapshot_rows += 1

                            print(
                                "\rRows written: "
                                f"{totals['rows']:,}",
                                end="",
                                flush=True,
                            )

                    affinity_users_with_history = sum(
                        long_term.evidence_weight > 0
                        for long_term, _
                        in affinity_cache.values()
                    )

                    affinity_users_with_recent = sum(
                        recent.evidence_weight > 0
                        for _, recent
                        in affinity_cache.values()
                    )

                    snapshot_summaries.append(
                        {
                            "index": snapshot_index,
                            "cutoff_at": cutoff_at,
                            "outcome_end_at": (
                                outcome_end_at
                            ),
                            "rows": snapshot_rows,
                            "affinity_users_loaded": (
                                len(affinity_cache)
                            ),
                            "affinity_users_with_history": (
                                affinity_users_with_history
                            ),
                            "affinity_users_with_recent": (
                                affinity_users_with_recent
                            ),
                            "graph_nodes": (
                                snapshot_graph_nodes or 0
                            ),
                            "graph_edges": (
                                snapshot_graph_edges or 0
                            ),
                            "graph_requests": (
                                snapshot_graph_requests
                            ),
                        }
                    )

                    print()
                    print(
                        f"Snapshot rows:     "
                        f"{snapshot_rows:,}"
                    )
                    print(
                        "Affinity users:    "
                        f"{len(affinity_cache):,}"
                    )
                    print(
                        "Affinity evidence: "
                        f"{affinity_users_with_history:,}"
                    )
                    print(
                        "Recent evidence:   "
                        f"{affinity_users_with_recent:,}"
                    )
                    print(
                        "Graph nodes:        "
                        f"{snapshot_graph_nodes or 0:,}"
                    )
                    print(
                        "Graph edges:        "
                        f"{snapshot_graph_edges or 0:,}"
                    )
                    print(
                        "Graph requests:     "
                        f"{snapshot_graph_requests:,}"
                    )
                    print()

    temp_output_path.replace(
        output_path
    )

    populated_snapshots = sum(
        summary["rows"] > 0
        for summary in snapshot_summaries
    )

    distinct_cutoffs = {
        summary["cutoff_at"]
        for summary in snapshot_summaries
        if summary["rows"] > 0
    }

    total_affinity_user_states = sum(
        int(summary["affinity_users_loaded"])
        for summary in snapshot_summaries
    )

    total_affinity_with_history = sum(
        int(
            summary[
                "affinity_users_with_history"
            ]
        )
        for summary in snapshot_summaries
    )

    total_affinity_with_recent = sum(
        int(
            summary[
                "affinity_users_with_recent"
            ]
        )
        for summary in snapshot_summaries
    )

    print()
    print("DATASET COMPLETE")
    print("────────────────────────────────────────")

    print(
        "Snapshots requested:            "
        f"{snapshot_count:,}"
    )
    print(
        "Snapshots processed:            "
        f"{len(snapshot_summaries):,}"
    )
    print(
        "Snapshots with rows:            "
        f"{populated_snapshots:,}"
    )
    print(
        "Distinct populated cutoffs:     "
        f"{len(distinct_cutoffs):,}"
    )
    print(
        "Exposed viewer/candidate pairs: "
        f"{totals['rows']:,}"
    )
    print(
        "Pairs with prior history:       "
        f"{totals['pairs_with_history']:,}"
    )

    print()
    print("HISTORICAL AFFINITIES")
    print("────────────────────────────────────────")
    print(
        "User-state loads across snapshots: "
        f"{total_affinity_user_states:,}"
    )
    print(
        "States with behavioral evidence:   "
        f"{total_affinity_with_history:,}"
    )
    print(
        "States with recent evidence:       "
        f"{total_affinity_with_recent:,}"
    )
    print(
        "Recent affinity window:            "
        f"{RECENT_AFFINITY_DAYS} days"
    )

    print()
    print("HISTORICAL GRAPH")
    print("────────────────────────────────────────")
    print(
        "Graph API requests:             "
        f"{totals['graph_requests']:,}"
    )
    print(
        "Hard-excluded historical pairs: "
        f"{totals['historical_hard_exclusions']:,}"
    )

    print()
    print("OUTCOMES")
    print("────────────────────────────────────────")
    print(
        "Outcome profile views:          "
        f"{totals['outcome_profile_views']:,}"
    )
    print(
        "Outcome favorites:              "
        f"{totals['outcome_favorites']:,}"
    )
    print(
        "Outcome pokes:                  "
        f"{totals['outcome_pokes']:,}"
    )
    print(
        "Outcome image reactions:        "
        f"{totals['outcome_image_reactions']:,}"
    )
    print(
        "Outcome messages:               "
        f"{totals['outcome_messages']:,}"
    )
    print(
        "Outcome blocks:                 "
        f"{totals['outcome_blocks']:,}"
    )
    print(
        "Outcome reports:                "
        f"{totals['outcome_reports']:,}"
    )
    print(
        "Reciprocal message pairs:       "
        f"{totals['reciprocal_message_pairs']:,}"
    )
    print(
        "Positive pairs:                 "
        f"{totals['positive_pairs']:,}"
    )

    print()
    print(
        f"Saved: {output_path}"
    )
    print()

    return totals["rows"]


def main() -> None:
    args = parse_args()

    if args.history_days <= 0:
        raise SystemExit(
            "--history-days must be greater than 0"
        )

    if args.outcome_days <= 0:
        raise SystemExit(
            "--outcome-days must be greater than 0"
        )

    if args.batch_size <= 0:
        raise SystemExit(
            "--batch-size must be greater than 0"
        )

    if args.graph_batch_size <= 0:
        raise SystemExit(
            "--graph-batch-size must be greater than 0"
        )

    if args.graph_batch_size > 1000:
        raise SystemExit(
            "--graph-batch-size cannot exceed 1000"
        )

    if args.snapshot_count <= 0:
        raise SystemExit(
            "--snapshot-count must be greater than 0"
        )

    if args.snapshot_step_days <= 0:
        raise SystemExit(
            "--snapshot-step-days must be greater than 0"
        )

    if args.snapshot_step_days < args.outcome_days:
        raise SystemExit(
            "--snapshot-step-days must be greater than or equal "
            "to --outcome-days so outcome windows do not overlap"
        )

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise SystemExit(
            "--limit must be greater than 0"
        )

    build_dataset(
        dsn=require_env(
            "APP_DSN"
        ),
        output_path=Path(
            args.output
        ),
        history_days=(
            args.history_days
        ),
        outcome_days=(
            args.outcome_days
        ),
        limit=args.limit,
        batch_size=(
            args.batch_size
        ),
        graph_batch_size=(
            args.graph_batch_size
        ),
        snapshot_count=(
            args.snapshot_count
        ),
        snapshot_step_days=(
            args.snapshot_step_days
        ),
    )


if __name__ == "__main__":
    main()
    