from __future__ import annotations

from typing import Any


FEATURE_SCHEMA_VERSION = 5
VOCAB_VERSION = 4


# ---------------------------------------------------------------------------
# Categorical profile features
# ---------------------------------------------------------------------------


SINGLE_CATEGORICAL_FIELDS = [
    "viewer_relationship_status",
    "viewer_pronouns",
    "viewer_position",
    "viewer_body_type",
    "viewer_into_position",
    "viewer_into_proximity",
    "viewer_into_relationship_status",

    "candidate_relationship_status",
    "candidate_pronouns",
    "candidate_position",
    "candidate_body_type",
    "candidate_into_position",
    "candidate_into_proximity",
    "candidate_into_relationship_status",
]


MULTI_CATEGORICAL_FIELDS = [
    "viewer_body_hair",
    "viewer_types",
    "viewer_fetishes",

    "viewer_into_age_ranges",
    "viewer_into_height_ranges",
    "viewer_into_weight_ranges",
    "viewer_into_body_types",
    "viewer_into_body_hair",
    "viewer_into_types",
    "viewer_into_fetishes",

    "candidate_body_hair",
    "candidate_types",
    "candidate_fetishes",

    "candidate_into_age_ranges",
    "candidate_into_height_ranges",
    "candidate_into_weight_ranges",
    "candidate_into_body_types",
    "candidate_into_body_hair",
    "candidate_into_types",
    "candidate_into_fetishes",
]


# ---------------------------------------------------------------------------
# PetersGraph features
# ---------------------------------------------------------------------------


GRAPH_NUMERIC_FIELDS = [
    "graph_direct_strength",
    "graph_reverse_strength",

    "graph_two_hop_score",
    "graph_shared_neighbor_score",
    "graph_behavioral_similarity",
    "graph_collaborative_score",

    "graph_community_affinity",
    "graph_novelty_score",
    "graph_retrieval_score",

    "graph_two_hop_path_count",
    "graph_similar_user_support_count",
]


# ---------------------------------------------------------------------------
# Numeric features
# ---------------------------------------------------------------------------


NUMERIC_INPUT_FIELDS = [
    "distance_km",

    # Viewer profile
    "viewer_age",
    "viewer_height_inches",
    "viewer_weight",
    "viewer_account_age_days",
    "viewer_max_distance_km",
    "viewer_min_age",
    "viewer_max_age",
    "viewer_image_count",
    "viewer_public_image_count",
    "viewer_private_image_count",
    "viewer_avatar_image_count",
    "viewer_descriptor_processed_count",

    # Candidate profile
    "candidate_age",
    "candidate_height_inches",
    "candidate_weight",
    "candidate_account_age_days",
    "candidate_max_distance_km",
    "candidate_min_age",
    "candidate_max_age",
    "candidate_image_count",
    "candidate_public_image_count",
    "candidate_private_image_count",
    "candidate_avatar_image_count",
    "candidate_descriptor_processed_count",

    # Viewer -> candidate pair history
    "history_profile_view_count",
    "history_favorite_count",
    "history_poke_count",
    "history_image_reaction_count",
    "history_story_view_count",
    "history_story_reaction_count",
    "history_story_reply_count",
    "history_message_count",
    "history_message_reaction_count",
    "history_private_album_grant_count",
    "history_partner_request_count",
    "history_partner_accepted_count",
    "history_block_count",
    "history_report_count",

    "history_all_events_7d",
    "history_all_events_30d",
    "history_all_events_90d",

    "history_profile_views_7d",
    "history_profile_views_30d",

    "history_favorites_7d",
    "history_favorites_30d",

    "history_messages_7d",
    "history_messages_30d",

    # Candidate -> viewer pair history
    "reverse_history_profile_view_count",
    "reverse_history_favorite_count",
    "reverse_history_poke_count",
    "reverse_history_image_reaction_count",
    "reverse_history_story_view_count",
    "reverse_history_story_reaction_count",
    "reverse_history_story_reply_count",
    "reverse_history_message_count",
    "reverse_history_message_reaction_count",
    "reverse_history_private_album_grant_count",
    "reverse_history_partner_request_count",
    "reverse_history_partner_accepted_count",
    "reverse_history_block_count",
    "reverse_history_report_count",

    # Viewer aggregate behavior
    "viewer_history_event_count",
    "viewer_history_unique_candidates",
    "viewer_history_profile_views",
    "viewer_history_favorites",
    "viewer_history_pokes",
    "viewer_history_image_reactions",
    "viewer_history_messages",

    "viewer_history_profile_views_30d",
    "viewer_history_favorites_30d",
    "viewer_history_messages_30d",
    "viewer_history_unique_candidates_30d",

    # Candidate aggregate received behavior
    "candidate_history_received_events",
    "candidate_history_unique_viewers",
    "candidate_history_received_profile_views",
    "candidate_history_received_favorites",
    "candidate_history_received_pokes",
    "candidate_history_received_image_reactions",
    "candidate_history_received_messages",

    "candidate_history_received_profile_views_30d",
    "candidate_history_received_favorites_30d",
    "candidate_history_received_messages_30d",
    "candidate_history_unique_viewers_30d",

    # PetersGraph relationship/network intelligence
    *GRAPH_NUMERIC_FIELDS,
]


NUMERIC_MISSING_FIELDS = list(
    NUMERIC_INPUT_FIELDS
)


# ---------------------------------------------------------------------------
# Compatibility features
# ---------------------------------------------------------------------------


COMPATIBILITY_FIELDS = [
    "viewer_accepts_candidate_age",
    "candidate_accepts_viewer_age",

    "viewer_accepts_candidate_distance",
    "candidate_accepts_viewer_distance",

    "viewer_position_match",
    "candidate_position_match",
    "mutual_position_match",

    "viewer_relationship_status_match",
    "candidate_relationship_status_match",

    "viewer_body_type_match",
    "candidate_body_type_match",

    "viewer_body_hair_match_count",
    "viewer_body_hair_match_ratio",

    "candidate_body_hair_match_count",
    "candidate_body_hair_match_ratio",

    "viewer_type_match_count",
    "viewer_type_match_ratio",

    "candidate_type_match_count",
    "candidate_type_match_ratio",

    "viewer_fetish_match_count",
    "viewer_fetish_match_ratio",

    "candidate_fetish_match_count",
    "candidate_fetish_match_ratio",

    "mutual_type_match",
    "mutual_preference_score",
]


COMPATIBILITY_MISSING_FIELDS = list(
    COMPATIBILITY_FIELDS
)


# ---------------------------------------------------------------------------
# Learned feature families
# ---------------------------------------------------------------------------


PROFILE_DESCRIPTOR_FIELDS: list[str] = []


USER_AFFINITY_FIELDS = [
    # Long-term demonstrated preference match.
    #
    # Each value is directional:
    #   viewer_affinities:
    #       how well the candidate matches the viewer's learned behavior
    #
    #   candidate_affinities:
    #       how well the viewer matches the candidate's learned behavior
    #
    # Scores are bounded to [0, 1].
    # 0.5 represents neutral / insufficient evidence.
    "age_affinity",
    "height_affinity",
    "weight_affinity",
    "distance_affinity",

    "position_affinity",
    "body_type_affinity",
    "body_hair_affinity",
    "relationship_status_affinity",
    "type_affinity",
    "fetish_affinity",

    # Same preference calculations using recent behavior only.
    "recent_age_affinity",
    "recent_height_affinity",
    "recent_weight_affinity",
    "recent_distance_affinity",

    "recent_position_affinity",
    "recent_body_type_affinity",
    "recent_body_hair_affinity",
    "recent_relationship_status_affinity",
    "recent_type_affinity",
    "recent_fetish_affinity",

    # Combined directional preference scores.
    "overall_affinity",
    "recent_overall_affinity",

    # Confidence in the learned behavioral representation.
    #
    # These are also bounded to [0, 1], rather than raw event counts,
    # because the affinity encoder does not apply numeric normalization.
    "evidence_strength",
    "recent_evidence_strength",
]


# ---------------------------------------------------------------------------
# Supervised targets
# ---------------------------------------------------------------------------


TARGET_BOOLEAN_FIELDS = [
    "outcome_favorite",
    "outcome_poke",
    "outcome_image_reaction",
    "outcome_story_engagement",
    "outcome_message_started",
    "outcome_reciprocal_message",
    "outcome_private_album_grant",
    "outcome_partner_accepted",
    "outcome_block",
    "outcome_report",
    "outcome_any_positive",
]


TARGET_COUNT_FIELDS = [
    "outcome_profile_view_count",
    "outcome_favorite_count",
    "outcome_poke_count",
    "outcome_image_reaction_count",
    "outcome_story_view_count",
    "outcome_story_reaction_count",
    "outcome_story_reply_count",
    "outcome_message_count",
    "outcome_message_reaction_count",
    "outcome_private_album_grant_count",
    "outcome_partner_request_count",
    "outcome_partner_accepted_count",
    "outcome_block_count",
    "outcome_report_count",

    "reverse_outcome_message_count",
    "reverse_outcome_favorite_count",
    "reverse_outcome_poke_count",
    "reverse_outcome_block_count",
    "reverse_outcome_report_count",
]


# ---------------------------------------------------------------------------
# Metadata / passthrough
# ---------------------------------------------------------------------------


PASSTHROUGH_FIELDS = [
    "viewer_user_id",
    "candidate_user_id",

    "history_start_at",
    "cutoff_at",
    "outcome_end_at",

    "first_history_interaction_at",
    "last_history_interaction_at",

    "first_outcome_interaction_at",
    "last_outcome_interaction_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def feature_schema() -> dict[str, Any]:
    return {
        "version": FEATURE_SCHEMA_VERSION,

        "single_categorical_fields": list(
            SINGLE_CATEGORICAL_FIELDS
        ),

        "multi_categorical_fields": list(
            MULTI_CATEGORICAL_FIELDS
        ),

        "numeric_fields": list(
            NUMERIC_INPUT_FIELDS
        ),

        "numeric_missing_fields": list(
            NUMERIC_MISSING_FIELDS
        ),

        "graph_numeric_fields": list(
            GRAPH_NUMERIC_FIELDS
        ),

        "compatibility_fields": list(
            COMPATIBILITY_FIELDS
        ),

        "compatibility_missing_fields": list(
            COMPATIBILITY_MISSING_FIELDS
        ),

        "profile_descriptor_fields": list(
            PROFILE_DESCRIPTOR_FIELDS
        ),

        "user_affinity_fields": list(
            USER_AFFINITY_FIELDS
        ),

        "target_fields": list(
            TARGET_BOOLEAN_FIELDS
        ),
    }
