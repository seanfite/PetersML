from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from features.user_affinity import (
    EVIDENCE_SATURATION,
    AffinityEntry,
    AffinityTarget,
    UserAffinityState,
    build_directional_affinity_features,
)
from inference.graph_client import (
    GraphPairFeatureResult,
    PetersGraphClient,
)


DEFAULT_HISTORY_DAYS = 90
DEFAULT_MAX_CANDIDATES = 2000
DEFAULT_AFFINITY_MODEL_VERSION = "behavioral_affinity_v1"
GRAPH_PAIR_BATCH_SIZE = 1000

LONG_TERM_META_KIND = "__meta__"
RECENT_META_KIND = "__recent_meta__"
META_EVIDENCE_KEY = "evidence_strength"

AFFINITY_KINDS = {
    "age",
    "height",
    "weight",
    "distance",
    "position",
    "body_type",
    "body_hair",
    "relationship_status",
    "type",
    "fetish",
}


FEATURE_QUERY = """
WITH params AS (
    SELECT
        now() AS cutoff_at,
        now() - (%(history_days)s * interval '1 day') AS history_start_at
),

windows AS (
    SELECT
        cutoff_at,
        history_start_at,
        cutoff_at - interval '7 days' AS history_7d_start_at,
        cutoff_at - interval '30 days' AS history_30d_start_at,
        cutoff_at - interval '90 days' AS history_90d_start_at,
        LEAST(history_start_at, cutoff_at - interval '90 days') AS load_start_at
    FROM params
),

events AS (
    SELECT
        viewer_user_id AS actor_user_id,
        owner_user_id AS candidate_user_id,
        'profile_view'::text AS event_type,
        created_at
    FROM user_profile_views
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        from_user_id,
        to_user_id,
        'favorite'::text,
        created_at
    FROM user_favorites
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        from_user_id,
        to_user_id,
        'poke'::text,
        created_at
    FROM user_pokes
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        reaction.from_user_id,
        image.user_id,
        'image_reaction'::text,
        reaction.created_at
    FROM image_reactions reaction
    JOIN user_images image ON image.id = reaction.image_id
    WHERE reaction.created_at >= (SELECT load_start_at FROM windows)
      AND reaction.created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        story_view.viewer_user_id,
        story.user_id,
        'story_view'::text,
        story_view.viewed_at
    FROM story_views story_view
    JOIN user_stories story ON story.id = story_view.story_id
    WHERE story_view.viewed_at >= (SELECT load_start_at FROM windows)
      AND story_view.viewed_at < (SELECT cutoff_at FROM windows)

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
    JOIN user_stories story ON story.id = reaction.story_id
    WHERE reaction.created_at >= (SELECT load_start_at FROM windows)
      AND reaction.created_at < (SELECT cutoff_at FROM windows)

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
    WHERE message.sender_user_id IS NOT NULL
      AND message.deleted_at IS NULL
      AND message.created_at >= (SELECT load_start_at FROM windows)
      AND message.created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        reaction.user_id,
        message.sender_user_id,
        'message_reaction'::text,
        reaction.created_at
    FROM message_reactions reaction
    JOIN messages message ON message.id = reaction.message_id
    WHERE message.sender_user_id IS NOT NULL
      AND reaction.user_id <> message.sender_user_id
      AND reaction.created_at >= (SELECT load_start_at FROM windows)
      AND reaction.created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        blocker_user_id,
        blocked_user_id,
        'block'::text,
        created_at
    FROM user_blocks
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        owner_user_id,
        viewer_user_id,
        'private_album_grant'::text,
        granted_at
    FROM user_private_album_access
    WHERE granted_at >= (SELECT load_start_at FROM windows)
      AND granted_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        requester_user_id,
        target_user_id,
        'partner_request'::text,
        created_at
    FROM partner_requests
    WHERE created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        requester_user_id,
        target_user_id,
        'partner_accepted'::text,
        responded_at
    FROM partner_requests
    WHERE status = 'accepted'
      AND responded_at IS NOT NULL
      AND responded_at >= (SELECT load_start_at FROM windows)
      AND responded_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        target_user_id,
        requester_user_id,
        'partner_accepted'::text,
        responded_at
    FROM partner_requests
    WHERE status = 'accepted'
      AND responded_at IS NOT NULL
      AND responded_at >= (SELECT load_start_at FROM windows)
      AND responded_at < (SELECT cutoff_at FROM windows)

    UNION ALL

    SELECT
        reporter_user_id,
        reported_user_id,
        'report'::text,
        created_at
    FROM support_reports
    WHERE reported_user_id IS NOT NULL
      AND created_at >= (SELECT load_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)
),

valid_events AS (
    SELECT actor_user_id, candidate_user_id, event_type, created_at
    FROM events
    WHERE actor_user_id IS NOT NULL
      AND candidate_user_id IS NOT NULL
      AND actor_user_id <> candidate_user_id
      AND (
            actor_user_id = %(viewer_user_id)s
         OR candidate_user_id = ANY(%(candidate_user_ids)s::bigint[])
         OR (
                actor_user_id = ANY(%(candidate_user_ids)s::bigint[])
            AND candidate_user_id = %(viewer_user_id)s
         )
      )
),

pair_keys AS (
    SELECT
        %(viewer_user_id)s::bigint AS actor_user_id,
        candidate_user_id
    FROM unnest(%(candidate_user_ids)s::bigint[]) AS candidate_user_id
    WHERE candidate_user_id <> %(viewer_user_id)s
),

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
            WHERE created_at >= (SELECT history_7d_start_at FROM windows)
        )::int AS history_all_events_7d,

        COUNT(*) FILTER (
            WHERE created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS history_all_events_30d,

        COUNT(*) FILTER (
            WHERE created_at >= (SELECT history_90d_start_at FROM windows)
        )::int AS history_all_events_90d,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
              AND created_at >= (SELECT history_7d_start_at FROM windows)
        )::int AS history_profile_views_7d,

        COUNT(*) FILTER (
            WHERE event_type = 'profile_view'
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS history_profile_views_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (SELECT history_7d_start_at FROM windows)
        )::int AS history_favorites_7d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS history_favorites_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (SELECT history_7d_start_at FROM windows)
        )::int AS history_messages_7d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS history_messages_30d

    FROM valid_events
    WHERE created_at >= (SELECT history_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)
    GROUP BY actor_user_id, candidate_user_id
),

viewer_history_stats AS (
    SELECT
        actor_user_id AS user_id,

        COUNT(*)::int AS viewer_history_event_count,
        COUNT(DISTINCT candidate_user_id)::int AS viewer_history_unique_candidates,

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
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS viewer_history_profile_views_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS viewer_history_favorites_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS viewer_history_messages_30d,

        COUNT(DISTINCT candidate_user_id) FILTER (
            WHERE created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS viewer_history_unique_candidates_30d

    FROM valid_events
    WHERE actor_user_id = %(viewer_user_id)s
      AND created_at >= (SELECT history_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)
    GROUP BY actor_user_id
),

candidate_history_stats AS (
    SELECT
        candidate_user_id AS user_id,

        COUNT(*)::int AS candidate_history_received_events,
        COUNT(DISTINCT actor_user_id)::int AS candidate_history_unique_viewers,

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
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS candidate_history_received_profile_views_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'favorite'
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS candidate_history_received_favorites_30d,

        COUNT(*) FILTER (
            WHERE event_type = 'message'
              AND created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS candidate_history_received_messages_30d,

        COUNT(DISTINCT actor_user_id) FILTER (
            WHERE created_at >= (SELECT history_30d_start_at FROM windows)
        )::int AS candidate_history_unique_viewers_30d

    FROM valid_events
    WHERE candidate_user_id = ANY(%(candidate_user_ids)s::bigint[])
      AND created_at >= (SELECT history_start_at FROM windows)
      AND created_at < (SELECT cutoff_at FROM windows)
    GROUP BY candidate_user_id
),

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
              AND descriptor_processed_at < (SELECT cutoff_at FROM windows)
        )::int AS descriptor_processed_count

    FROM user_images
    WHERE created_at < (SELECT cutoff_at FROM windows)
      AND (
            user_id = %(viewer_user_id)s
         OR user_id = ANY(%(candidate_user_ids)s::bigint[])
      )
    GROUP BY user_id
)

SELECT
    (SELECT history_start_at FROM windows) AS history_start_at,
    (SELECT cutoff_at FROM windows) AS cutoff_at,

    key.actor_user_id AS viewer_user_id,
    key.candidate_user_id AS candidate_user_id,

    history.first_history_interaction_at,
    history.last_history_interaction_at,

    COALESCE(history.history_profile_view_count, 0) AS history_profile_view_count,
    COALESCE(history.history_favorite_count, 0) AS history_favorite_count,
    COALESCE(history.history_poke_count, 0) AS history_poke_count,
    COALESCE(history.history_image_reaction_count, 0) AS history_image_reaction_count,
    COALESCE(history.history_story_view_count, 0) AS history_story_view_count,
    COALESCE(history.history_story_reaction_count, 0) AS history_story_reaction_count,
    COALESCE(history.history_story_reply_count, 0) AS history_story_reply_count,
    COALESCE(history.history_message_count, 0) AS history_message_count,
    COALESCE(history.history_message_reaction_count, 0) AS history_message_reaction_count,
    COALESCE(history.history_private_album_grant_count, 0)
        AS history_private_album_grant_count,
    COALESCE(history.history_partner_request_count, 0)
        AS history_partner_request_count,
    COALESCE(history.history_partner_accepted_count, 0)
        AS history_partner_accepted_count,
    COALESCE(history.history_block_count, 0) AS history_block_count,
    COALESCE(history.history_report_count, 0) AS history_report_count,

    COALESCE(history.history_all_events_7d, 0) AS history_all_events_7d,
    COALESCE(history.history_all_events_30d, 0) AS history_all_events_30d,
    COALESCE(history.history_all_events_90d, 0) AS history_all_events_90d,

    COALESCE(history.history_profile_views_7d, 0) AS history_profile_views_7d,
    COALESCE(history.history_profile_views_30d, 0) AS history_profile_views_30d,
    COALESCE(history.history_favorites_7d, 0) AS history_favorites_7d,
    COALESCE(history.history_favorites_30d, 0) AS history_favorites_30d,
    COALESCE(history.history_messages_7d, 0) AS history_messages_7d,
    COALESCE(history.history_messages_30d, 0) AS history_messages_30d,

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

    COALESCE(candidate_history.candidate_history_received_events, 0)
        AS candidate_history_received_events,
    COALESCE(candidate_history.candidate_history_unique_viewers, 0)
        AS candidate_history_unique_viewers,
    COALESCE(candidate_history.candidate_history_received_profile_views, 0)
        AS candidate_history_received_profile_views,
    COALESCE(candidate_history.candidate_history_received_favorites, 0)
        AS candidate_history_received_favorites,
    COALESCE(candidate_history.candidate_history_received_pokes, 0)
        AS candidate_history_received_pokes,
    COALESCE(candidate_history.candidate_history_received_image_reactions, 0)
        AS candidate_history_received_image_reactions,
    COALESCE(candidate_history.candidate_history_received_messages, 0)
        AS candidate_history_received_messages,
    COALESCE(candidate_history.candidate_history_received_profile_views_30d, 0)
        AS candidate_history_received_profile_views_30d,
    COALESCE(candidate_history.candidate_history_received_favorites_30d, 0)
        AS candidate_history_received_favorites_30d,
    COALESCE(candidate_history.candidate_history_received_messages_30d, 0)
        AS candidate_history_received_messages_30d,
    COALESCE(candidate_history.candidate_history_unique_viewers_30d, 0)
        AS candidate_history_unique_viewers_30d,

    CASE
        WHEN COALESCE(viewer.coord, viewer.home_coord) IS NULL
          OR COALESCE(candidate.coord, candidate.home_coord) IS NULL
        THEN NULL
        ELSE ST_Distance(
            COALESCE(viewer.coord, viewer.home_coord),
            COALESCE(candidate.coord, candidate.home_coord)
        ) / 1000.0
    END AS distance_km,

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
    COALESCE(viewer.body_hair, ARRAY[]::text[]) AS viewer_body_hair,
    COALESCE(viewer.my_types, ARRAY[]::text[]) AS viewer_types,
    COALESCE(viewer.my_fetishes, ARRAY[]::text[]) AS viewer_fetishes,

    GREATEST(
        0,
        EXTRACT(
            EPOCH FROM ((SELECT cutoff_at FROM windows) - viewer.created_at)
        ) / 86400.0
    )::double precision AS viewer_account_age_days,

    COALESCE(viewer.into_age_ranges, ARRAY[]::text[]) AS viewer_into_age_ranges,
    COALESCE(viewer.into_height_ranges, ARRAY[]::text[])
        AS viewer_into_height_ranges,
    COALESCE(viewer.into_weight_ranges, ARRAY[]::text[])
        AS viewer_into_weight_ranges,
    viewer.into_position AS viewer_into_position,
    COALESCE(viewer.into_body_types, ARRAY[]::text[]) AS viewer_into_body_types,
    COALESCE(viewer.into_body_hair, ARRAY[]::text[]) AS viewer_into_body_hair,
    viewer.into_proximity AS viewer_into_proximity,
    viewer.into_relationship_status AS viewer_into_relationship_status,
    COALESCE(viewer.into_types, ARRAY[]::text[]) AS viewer_into_types,
    COALESCE(viewer.into_fetishes, ARRAY[]::text[]) AS viewer_into_fetishes,

    viewer_settings.max_distance_km AS viewer_max_distance_km,
    viewer_settings.min_age AS viewer_min_age,
    viewer_settings.max_age AS viewer_max_age,

    COALESCE(viewer_images.image_count, 0) AS viewer_image_count,
    COALESCE(viewer_images.public_image_count, 0) AS viewer_public_image_count,
    COALESCE(viewer_images.private_image_count, 0) AS viewer_private_image_count,
    COALESCE(viewer_images.avatar_image_count, 0) AS viewer_avatar_image_count,
    COALESCE(viewer_images.descriptor_processed_count, 0)
        AS viewer_descriptor_processed_count,

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
    COALESCE(candidate.body_hair, ARRAY[]::text[]) AS candidate_body_hair,
    COALESCE(candidate.my_types, ARRAY[]::text[]) AS candidate_types,
    COALESCE(candidate.my_fetishes, ARRAY[]::text[]) AS candidate_fetishes,

    GREATEST(
        0,
        EXTRACT(
            EPOCH FROM ((SELECT cutoff_at FROM windows) - candidate.created_at)
        ) / 86400.0
    )::double precision AS candidate_account_age_days,

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
    candidate.into_relationship_status AS candidate_into_relationship_status,
    COALESCE(candidate.into_types, ARRAY[]::text[]) AS candidate_into_types,
    COALESCE(candidate.into_fetishes, ARRAY[]::text[]) AS candidate_into_fetishes,

    candidate_settings.max_distance_km AS candidate_max_distance_km,
    candidate_settings.min_age AS candidate_min_age,
    candidate_settings.max_age AS candidate_max_age,

    COALESCE(candidate_images.image_count, 0) AS candidate_image_count,
    COALESCE(candidate_images.public_image_count, 0)
        AS candidate_public_image_count,
    COALESCE(candidate_images.private_image_count, 0)
        AS candidate_private_image_count,
    COALESCE(candidate_images.avatar_image_count, 0)
        AS candidate_avatar_image_count,
    COALESCE(candidate_images.descriptor_processed_count, 0)
        AS candidate_descriptor_processed_count

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

ORDER BY key.candidate_user_id
"""


AFFINITY_QUERY = """
SELECT
    user_id,
    affinity_kind,
    affinity_key,
    score,
    evidence_count
FROM ml_user_affinities
WHERE user_id = ANY(%(user_ids)s::bigint[])
  AND model_version = %(model_version)s
ORDER BY
    user_id,
    affinity_kind,
    affinity_key
"""


@dataclass(frozen=True)
class FeatureStoreResult:
    viewer_user_id: int
    requested_candidate_ids: list[int]
    rows: list[dict[str, Any]]
    missing_candidate_ids: list[int]
    cutoff_at: datetime | None

    graph_enriched: bool = False
    graph_version_id: int | None = None
    graph_version: str | None = None
    graph_schema_version: int | None = None
    graph_cutoff_at: str | None = None

    @property
    def candidate_ids(self) -> list[int]:
        return [
            int(row["candidate_user_id"])
            for row in self.rows
        ]


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(f"{name} is not set")

    return value


def parse_height_inches(
    value: Any,
) -> float | None:
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
            return float(
                feet * 12 + inches
            )

    try:
        numeric = float(text)

        if 36 <= numeric <= 96:
            return numeric
    except ValueError:
        pass

    return None


def normalize_candidate_ids(
    viewer_user_id: int,
    candidate_user_ids: list[int],
) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()

    for value in candidate_user_ids:
        candidate_id = int(value)

        if candidate_id <= 0:
            raise ValueError(
                f"Invalid candidate user ID: {candidate_id}"
            )

        if candidate_id == viewer_user_id:
            continue

        if candidate_id in seen:
            continue

        seen.add(candidate_id)
        normalized.append(candidate_id)

    return normalized


def prepare_feature_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    row["viewer_height_inches"] = (
        parse_height_inches(
            row.get("viewer_height")
        )
    )

    row["candidate_height_inches"] = (
        parse_height_inches(
            row.get("candidate_height")
        )
    )

    return row


def neutral_affinity_state() -> UserAffinityState:
    return UserAffinityState(
        values={},
        observation_count=0,
        evidence_weight=0.0,
    )


def evidence_weight_from_strength(
    strength: float,
) -> float:
    strength = max(
        0.0,
        min(
            1.0,
            float(strength),
        ),
    )

    if strength <= 0.0:
        return 0.0

    # UserAffinityState.evidence_strength is:
    #   1 - exp(-evidence_weight / EVIDENCE_SATURATION)
    #
    # Invert that transformation so live inference reconstructs the same
    # confidence value that was persisted by build_user_affinities.py.
    safe_strength = min(
        strength,
        1.0 - 1e-12,
    )

    return (
        -EVIDENCE_SATURATION
        * math.log(
            1.0 - safe_strength
        )
    )


def load_affinity_states(
    conn: psycopg.Connection,
    user_ids: list[int],
    model_version: str,
) -> tuple[
    dict[int, UserAffinityState],
    dict[int, UserAffinityState],
]:
    normalized_user_ids = sorted(
        {
            int(user_id)
            for user_id in user_ids
            if int(user_id) > 0
        }
    )

    if not normalized_user_ids:
        return {}, {}

    params = {
        "user_ids": normalized_user_ids,
        "model_version": model_version,
    }

    with conn.cursor(
        row_factory=dict_row,
    ) as cursor:
        cursor.execute(
            AFFINITY_QUERY,
            params,
        )

        raw_rows = [
            dict(row)
            for row in cursor.fetchall()
        ]

    long_values: dict[
        int,
        dict[
            str,
            dict[str, AffinityEntry],
        ],
    ] = {}

    recent_values: dict[
        int,
        dict[
            str,
            dict[str, AffinityEntry],
        ],
    ] = {}

    long_strength: dict[int, float] = {}
    recent_strength: dict[int, float] = {}

    long_observation_count: dict[int, int] = {}
    recent_observation_count: dict[int, int] = {}

    for row in raw_rows:
        user_id = int(
            row["user_id"]
        )

        affinity_kind = str(
            row["affinity_kind"]
        )

        affinity_key = str(
            row["affinity_key"]
        )

        score = float(
            row["score"]
        )

        evidence_count = max(
            0,
            int(
                row["evidence_count"]
            ),
        )

        if (
            affinity_kind == LONG_TERM_META_KIND
            and affinity_key == META_EVIDENCE_KEY
        ):
            long_strength[user_id] = score
            long_observation_count[user_id] = evidence_count
            continue

        if (
            affinity_kind == RECENT_META_KIND
            and affinity_key == META_EVIDENCE_KEY
        ):
            recent_strength[user_id] = score
            recent_observation_count[user_id] = evidence_count
            continue

        recent = affinity_kind.startswith(
            "recent_"
        )

        base_kind = (
            affinity_kind[len("recent_"):]
            if recent
            else affinity_kind
        )

        if base_kind not in AFFINITY_KINDS:
            continue

        target_values = (
            recent_values
            if recent
            else long_values
        )

        kind_values = (
            target_values
            .setdefault(
                user_id,
                {},
            )
            .setdefault(
                base_kind,
                {},
            )
        )

        kind_values[
            affinity_key
        ] = AffinityEntry(
            score=score,
            evidence_count=evidence_count,
            evidence_weight=0.0,
        )

    long_term_states: dict[
        int,
        UserAffinityState,
    ] = {}

    recent_states: dict[
        int,
        UserAffinityState,
    ] = {}

    for user_id in normalized_user_ids:
        long_term_states[user_id] = (
            UserAffinityState(
                values=long_values.get(
                    user_id,
                    {},
                ),
                observation_count=(
                    long_observation_count.get(
                        user_id,
                        0,
                    )
                ),
                evidence_weight=(
                    evidence_weight_from_strength(
                        long_strength.get(
                            user_id,
                            0.0,
                        )
                    )
                ),
            )
        )

        recent_states[user_id] = (
            UserAffinityState(
                values=recent_values.get(
                    user_id,
                    {},
                ),
                observation_count=(
                    recent_observation_count.get(
                        user_id,
                        0,
                    )
                ),
                evidence_weight=(
                    evidence_weight_from_strength(
                        recent_strength.get(
                            user_id,
                            0.0,
                        )
                    )
                ),
            )
        )

    return (
        long_term_states,
        recent_states,
    )


def enrich_rows_with_affinities(
    rows: list[dict[str, Any]],
    long_term_states: dict[int, UserAffinityState],
    recent_states: dict[int, UserAffinityState],
) -> None:
    if not rows:
        return

    neutral_state = neutral_affinity_state()

    for row in rows:
        viewer_user_id = int(
            row["viewer_user_id"]
        )

        candidate_user_id = int(
            row["candidate_user_id"]
        )

        viewer_long_term = (
            long_term_states.get(
                viewer_user_id,
                neutral_state,
            )
        )

        viewer_recent = (
            recent_states.get(
                viewer_user_id,
                neutral_state,
            )
        )

        candidate_long_term = (
            long_term_states.get(
                candidate_user_id,
                neutral_state,
            )
        )

        candidate_recent = (
            recent_states.get(
                candidate_user_id,
                neutral_state,
            )
        )

        candidate_target = (
            AffinityTarget.from_feature_row(
                row,
                "candidate",
            )
        )

        viewer_target = (
            AffinityTarget.from_feature_row(
                row,
                "viewer",
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
                target=candidate_target,
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
                target=viewer_target,
            )
        )


def merge_graph_features(
    rows: list[dict[str, Any]],
    graph_result: GraphPairFeatureResult,
) -> None:
    pair_lookup = graph_result.by_candidate_id()

    missing_graph_ids: list[int] = []

    for row in rows:
        candidate_user_id = int(
            row["candidate_user_id"]
        )

        pair = pair_lookup.get(
            candidate_user_id
        )

        if pair is None:
            missing_graph_ids.append(
                candidate_user_id
            )
            continue

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

    if missing_graph_ids:
        raise RuntimeError(
            "PetersGraph did not return pair features for "
            f"candidate IDs: {missing_graph_ids}"
        )


def fetch_graph_features_batched(
    graph_client: PetersGraphClient,
    viewer_user_id: int,
    candidate_user_ids: list[int],
) -> GraphPairFeatureResult:
    if not candidate_user_ids:
        raise ValueError(
            "candidate_user_ids cannot be empty"
        )

    combined_pairs = []

    graph_version_id: int | None = None
    graph_version: str | None = None
    graph_schema_version: int | None = None
    graph_cutoff_at: str | None = None

    similar_users_considered = 0
    two_hop_candidates_considered = 0
    collaborative_candidates_considered = 0

    for start in range(
        0,
        len(candidate_user_ids),
        GRAPH_PAIR_BATCH_SIZE,
    ):
        batch_ids = candidate_user_ids[
            start:start + GRAPH_PAIR_BATCH_SIZE
        ]

        result = graph_client.pair_features(
            viewer_user_id=viewer_user_id,
            candidate_user_ids=batch_ids,
        )

        if graph_version_id is None:
            graph_version_id = (
                result.graph_version_id
            )
            graph_version = (
                result.graph_version
            )
            graph_schema_version = (
                result.graph_schema_version
            )
            graph_cutoff_at = (
                result.graph_cutoff_at
            )

        elif (
            result.graph_version_id
            != graph_version_id
        ):
            raise RuntimeError(
                "PetersGraph production version changed "
                "during feature retrieval"
            )

        combined_pairs.extend(
            result.pairs
        )

        similar_users_considered = max(
            similar_users_considered,
            result.similar_users_considered,
        )

        two_hop_candidates_considered = max(
            two_hop_candidates_considered,
            result.two_hop_candidates_considered,
        )

        collaborative_candidates_considered = max(
            collaborative_candidates_considered,
            result.collaborative_candidates_considered,
        )

    if (
        graph_version_id is None
        or graph_version is None
        or graph_schema_version is None
        or graph_cutoff_at is None
    ):
        raise RuntimeError(
            "PetersGraph returned no graph version"
        )

    if len(combined_pairs) != len(candidate_user_ids):
        raise RuntimeError(
            "PetersGraph pair feature count mismatch: "
            f"requested={len(candidate_user_ids)}, "
            f"returned={len(combined_pairs)}"
        )

    return GraphPairFeatureResult(
        viewer_user_id=viewer_user_id,

        graph_version_id=graph_version_id,
        graph_version=graph_version,
        graph_schema_version=graph_schema_version,
        graph_cutoff_at=graph_cutoff_at,

        candidate_count=len(combined_pairs),

        similar_users_considered=(
            similar_users_considered
        ),
        two_hop_candidates_considered=(
            two_hop_candidates_considered
        ),
        collaborative_candidates_considered=(
            collaborative_candidates_considered
        ),

        pairs=tuple(combined_pairs),
    )


class FeatureStore:
    def __init__(
        self,
        dsn: str | None = None,
        history_days: int = DEFAULT_HISTORY_DAYS,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        affinity_model_version: str | None = None,
        graph_client: PetersGraphClient | None = None,
    ) -> None:
        self.dsn = (
            dsn
            or require_env("APP_DSN")
        )

        self.history_days = history_days
        self.max_candidates = max_candidates

        self.affinity_model_version = (
            affinity_model_version
            or os.getenv(
                "PETERS_AFFINITY_MODEL_VERSION",
                DEFAULT_AFFINITY_MODEL_VERSION,
            )
        ).strip()

        self._graph_client = graph_client

        if self.history_days <= 0:
            raise ValueError(
                "history_days must be greater than 0"
            )

        if self.max_candidates <= 0:
            raise ValueError(
                "max_candidates must be greater than 0"
            )

        if not self.affinity_model_version:
            raise ValueError(
                "affinity_model_version cannot be empty"
            )

    def _get_graph_client(
        self,
    ) -> PetersGraphClient:
        if self._graph_client is None:
            self._graph_client = PetersGraphClient()

        return self._graph_client

    def fetch_pairs(
        self,
        viewer_user_id: int,
        candidate_user_ids: list[int],
        include_graph_features: bool = False,
    ) -> FeatureStoreResult:
        viewer_user_id = int(
            viewer_user_id
        )

        if viewer_user_id <= 0:
            raise ValueError(
                "viewer_user_id must be greater than 0"
            )

        candidate_ids = normalize_candidate_ids(
            viewer_user_id,
            candidate_user_ids,
        )

        if not candidate_ids:
            return FeatureStoreResult(
                viewer_user_id=viewer_user_id,
                requested_candidate_ids=[],
                rows=[],
                missing_candidate_ids=[],
                cutoff_at=None,
                graph_enriched=False,
            )

        if len(candidate_ids) > self.max_candidates:
            raise ValueError(
                f"Requested {len(candidate_ids):,} candidates; "
                f"maximum is {self.max_candidates:,}."
            )

        params = {
            "viewer_user_id": viewer_user_id,
            "candidate_user_ids": candidate_ids,
            "history_days": self.history_days,
        }

        with psycopg.connect(
            self.dsn,
            row_factory=dict_row,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    FEATURE_QUERY,
                    params,
                )

                raw_rows = cursor.fetchall()

            rows = [
                prepare_feature_row(
                    dict(row)
                )
                for row in raw_rows
            ]

            affinity_user_ids = {
                viewer_user_id
            }

            affinity_user_ids.update(
                int(row["candidate_user_id"])
                for row in rows
            )

            (
                long_term_states,
                recent_states,
            ) = load_affinity_states(
                conn=conn,
                user_ids=sorted(
                    affinity_user_ids
                ),
                model_version=(
                    self.affinity_model_version
                ),
            )

        enrich_rows_with_affinities(
            rows=rows,
            long_term_states=long_term_states,
            recent_states=recent_states,
        )

        returned_ids = {
            int(row["candidate_user_id"])
            for row in rows
        }

        missing = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id not in returned_ids
        ]

        cutoff_at = (
            rows[0].get("cutoff_at")
            if rows
            else None
        )

        if (
            not include_graph_features
            or not rows
        ):
            return FeatureStoreResult(
                viewer_user_id=viewer_user_id,
                requested_candidate_ids=candidate_ids,
                rows=rows,
                missing_candidate_ids=missing,
                cutoff_at=cutoff_at,
                graph_enriched=False,
            )

        graph_candidate_ids = [
            int(row["candidate_user_id"])
            for row in rows
        ]

        graph_result = (
            fetch_graph_features_batched(
                graph_client=(
                    self._get_graph_client()
                ),
                viewer_user_id=viewer_user_id,
                candidate_user_ids=graph_candidate_ids,
            )
        )

        merge_graph_features(
            rows=rows,
            graph_result=graph_result,
        )

        return FeatureStoreResult(
            viewer_user_id=viewer_user_id,
            requested_candidate_ids=candidate_ids,
            rows=rows,
            missing_candidate_ids=missing,
            cutoff_at=cutoff_at,

            graph_enriched=True,
            graph_version_id=(
                graph_result.graph_version_id
            ),
            graph_version=(
                graph_result.graph_version
            ),
            graph_schema_version=(
                graph_result.graph_schema_version
            ),
            graph_cutoff_at=(
                graph_result.graph_cutoff_at
            ),
        )


feature_store = FeatureStore()

