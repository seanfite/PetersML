from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


DEFAULT_GRAPH_URL = "http://127.0.0.1:8081"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_HISTORICAL_TIMEOUT_SECONDS = 300.0

GRAPH_URL_ENV = "PETERS_GRAPH_URL"


class PetersGraphError(RuntimeError):
    """Raised when PetersML cannot successfully communicate with PetersGraph."""


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphCandidate:
    candidate_user_id: int

    graph_retrieval_score: float

    direct_strength: float
    reverse_strength: float

    two_hop_score: float
    collaborative_score: float
    shared_neighbor_score: float
    behavioral_similarity: float

    novelty_score: float

    two_hop_path_count: int
    similar_user_support_count: int


@dataclass(frozen=True)
class GraphCandidateResult:
    viewer_user_id: int

    graph_version_id: int
    graph_version: str
    graph_schema_version: int
    graph_cutoff_at: str

    candidate_count: int

    similar_users_considered: int
    two_hop_candidates_considered: int
    collaborative_candidates_considered: int
    reverse_candidates_considered: int

    candidates: tuple[GraphCandidate, ...]


# ---------------------------------------------------------------------------
# Pair graph features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphPairFeatures:
    candidate_user_id: int

    graph_direct_strength: float
    graph_reverse_strength: float

    graph_two_hop_score: float
    graph_shared_neighbor_score: float
    graph_behavioral_similarity: float
    graph_collaborative_score: float

    graph_community_affinity: float
    graph_novelty_score: float
    graph_retrieval_score: float

    two_hop_path_count: int
    similar_user_support_count: int

    is_hard_excluded: bool

    def as_feature_dict(self) -> dict[str, float]:
        return {
            "graph_direct_strength": self.graph_direct_strength,
            "graph_reverse_strength": self.graph_reverse_strength,
            "graph_two_hop_score": self.graph_two_hop_score,
            "graph_shared_neighbor_score": self.graph_shared_neighbor_score,
            "graph_behavioral_similarity": self.graph_behavioral_similarity,
            "graph_collaborative_score": self.graph_collaborative_score,
            "graph_community_affinity": self.graph_community_affinity,
            "graph_novelty_score": self.graph_novelty_score,
            "graph_retrieval_score": self.graph_retrieval_score,
        }


@dataclass(frozen=True)
class GraphPairFeatureResult:
    viewer_user_id: int

    graph_version_id: int
    graph_version: str
    graph_schema_version: int
    graph_cutoff_at: str

    candidate_count: int

    similar_users_considered: int
    two_hop_candidates_considered: int
    collaborative_candidates_considered: int

    pairs: tuple[GraphPairFeatures, ...]

    def by_candidate_id(self) -> dict[int, GraphPairFeatures]:
        return {
            pair.candidate_user_id: pair
            for pair in self.pairs
        }


# ---------------------------------------------------------------------------
# Historical graph features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalGraphPairFeatureResult:
    cutoff_at: str
    graph_built_at: str

    graph_nodes: int
    graph_edges: int

    viewer_user_id: int
    candidate_count: int

    similar_users_considered: int
    two_hop_candidates_considered: int
    collaborative_candidates_considered: int

    pairs: tuple[GraphPairFeatures, ...]

    def by_candidate_id(self) -> dict[int, GraphPairFeatures]:
        return {
            pair.candidate_user_id: pair
            for pair in self.pairs
        }


# ---------------------------------------------------------------------------
# PetersGraph client
# ---------------------------------------------------------------------------


class PetersGraphClient:
    """
    HTTP client from PetersML to PetersGraph.

    Live inference:
        /candidates
        /pair-features

    Supervised training:
        /historical-pair-features

    PetersML should consume graph intelligence through this client rather
    than importing PetersGraph internals or reconstructing graph state from
    raw application tables.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        historical_timeout_seconds: float = (
            DEFAULT_HISTORICAL_TIMEOUT_SECONDS
        ),
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0"
            )

        if historical_timeout_seconds <= 0:
            raise ValueError(
                "historical_timeout_seconds must be greater than 0"
            )

        resolved_url = (
            base_url
            or os.getenv(
                GRAPH_URL_ENV,
                DEFAULT_GRAPH_URL,
            )
        )

        resolved_url = resolved_url.strip().rstrip("/")

        if not resolved_url:
            raise ValueError(
                "PetersGraph base URL cannot be empty"
            )

        self.base_url = resolved_url
        self.timeout_seconds = timeout_seconds
        self.historical_timeout_seconds = (
            historical_timeout_seconds
        )

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                self.timeout_seconds
            ),
            headers={
                "Accept": "application/json",
            },
        )

    def __enter__(
        self,
    ) -> PetersGraphClient:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Service status
    # ------------------------------------------------------------------

    def health(
        self,
    ) -> dict[str, Any]:
        return self._request_json(
            method="GET",
            path="/health",
        )

    def version(
        self,
    ) -> dict[str, Any]:
        return self._request_json(
            method="GET",
            path="/version",
        )

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def candidates(
        self,
        viewer_user_id: int,
        limit: int = 100,
    ) -> GraphCandidateResult:
        viewer_user_id = int(
            viewer_user_id
        )

        if viewer_user_id <= 0:
            raise ValueError(
                "viewer_user_id must be greater than 0"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0"
            )

        payload = self._request_json(
            method="POST",
            path="/candidates",
            json={
                "viewer_user_id": viewer_user_id,
                "limit": limit,
            },
        )

        candidates = tuple(
            self._parse_candidate(candidate)
            for candidate in payload["candidates"]
        )

        return GraphCandidateResult(
            viewer_user_id=int(
                payload["viewer_user_id"]
            ),

            graph_version_id=int(
                payload["graph_version_id"]
            ),
            graph_version=str(
                payload["graph_version"]
            ),
            graph_schema_version=int(
                payload["graph_schema_version"]
            ),
            graph_cutoff_at=str(
                payload["graph_cutoff_at"]
            ),

            candidate_count=int(
                payload["candidate_count"]
            ),

            similar_users_considered=int(
                payload["similar_users_considered"]
            ),
            two_hop_candidates_considered=int(
                payload[
                    "two_hop_candidates_considered"
                ]
            ),
            collaborative_candidates_considered=int(
                payload[
                    "collaborative_candidates_considered"
                ]
            ),
            reverse_candidates_considered=int(
                payload[
                    "reverse_candidates_considered"
                ]
            ),

            candidates=candidates,
        )

    # ------------------------------------------------------------------
    # Current production pair features
    # ------------------------------------------------------------------

    def pair_features(
        self,
        viewer_user_id: int,
        candidate_user_ids: list[int],
    ) -> GraphPairFeatureResult:
        candidate_ids = self._normalize_candidate_ids(
            viewer_user_id=viewer_user_id,
            candidate_user_ids=candidate_user_ids,
        )

        payload = self._request_json(
            method="POST",
            path="/pair-features",
            json={
                "viewer_user_id": int(
                    viewer_user_id
                ),
                "candidate_user_ids": candidate_ids,
            },
        )

        pairs = tuple(
            self._parse_pair_features(pair)
            for pair in payload["pairs"]
        )

        return GraphPairFeatureResult(
            viewer_user_id=int(
                payload["viewer_user_id"]
            ),

            graph_version_id=int(
                payload["graph_version_id"]
            ),
            graph_version=str(
                payload["graph_version"]
            ),
            graph_schema_version=int(
                payload["graph_schema_version"]
            ),
            graph_cutoff_at=str(
                payload["graph_cutoff_at"]
            ),

            candidate_count=int(
                payload["candidate_count"]
            ),

            similar_users_considered=int(
                payload["similar_users_considered"]
            ),
            two_hop_candidates_considered=int(
                payload[
                    "two_hop_candidates_considered"
                ]
            ),
            collaborative_candidates_considered=int(
                payload[
                    "collaborative_candidates_considered"
                ]
            ),

            pairs=pairs,
        )

    # ------------------------------------------------------------------
    # Historical training pair features
    # ------------------------------------------------------------------

    def historical_pair_features(
        self,
        cutoff_at: datetime | str,
        viewer_user_id: int,
        candidate_user_ids: list[int],
    ) -> HistoricalGraphPairFeatureResult:
        candidate_ids = self._normalize_candidate_ids(
            viewer_user_id=viewer_user_id,
            candidate_user_ids=candidate_user_ids,
        )

        cutoff_value = self._serialize_cutoff(
            cutoff_at
        )

        payload = self._request_json(
            method="POST",
            path="/historical-pair-features",
            json={
                "cutoff_at": cutoff_value,
                "viewer_user_id": int(
                    viewer_user_id
                ),
                "candidate_user_ids": candidate_ids,
            },
            timeout_seconds=(
                self.historical_timeout_seconds
            ),
        )

        pairs = tuple(
            self._parse_pair_features(pair)
            for pair in payload["pairs"]
        )

        return HistoricalGraphPairFeatureResult(
            cutoff_at=str(
                payload["cutoff_at"]
            ),
            graph_built_at=str(
                payload["graph_built_at"]
            ),

            graph_nodes=int(
                payload["graph_nodes"]
            ),
            graph_edges=int(
                payload["graph_edges"]
            ),

            viewer_user_id=int(
                payload["viewer_user_id"]
            ),
            candidate_count=int(
                payload["candidate_count"]
            ),

            similar_users_considered=int(
                payload["similar_users_considered"]
            ),
            two_hop_candidates_considered=int(
                payload[
                    "two_hop_candidates_considered"
                ]
            ),
            collaborative_candidates_considered=int(
                payload[
                    "collaborative_candidates_considered"
                ]
            ),

            pairs=pairs,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_candidate_ids(
        viewer_user_id: int,
        candidate_user_ids: list[int],
    ) -> list[int]:
        viewer_user_id = int(
            viewer_user_id
        )

        if viewer_user_id <= 0:
            raise ValueError(
                "viewer_user_id must be greater than 0"
            )

        if not candidate_user_ids:
            raise ValueError(
                "candidate_user_ids cannot be empty"
            )

        normalized: list[int] = []
        seen: set[int] = set()

        for value in candidate_user_ids:
            candidate_user_id = int(
                value
            )

            if candidate_user_id <= 0:
                raise ValueError(
                    "candidate_user_ids must contain "
                    "only positive IDs"
                )

            if candidate_user_id == viewer_user_id:
                raise ValueError(
                    "candidate_user_id cannot equal "
                    "viewer_user_id"
                )

            if candidate_user_id in seen:
                continue

            seen.add(
                candidate_user_id
            )

            normalized.append(
                candidate_user_id
            )

        return normalized

    @staticmethod
    def _serialize_cutoff(
        cutoff_at: datetime | str,
    ) -> str:
        if isinstance(
            cutoff_at,
            datetime,
        ):
            if cutoff_at.tzinfo is None:
                raise ValueError(
                    "cutoff_at must be timezone-aware"
                )

            return cutoff_at.isoformat()

        value = str(
            cutoff_at
        ).strip()

        if not value:
            raise ValueError(
                "cutoff_at cannot be empty"
            )

        return value

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_candidate(
        candidate: dict[str, Any],
    ) -> GraphCandidate:
        return GraphCandidate(
            candidate_user_id=int(
                candidate["candidate_user_id"]
            ),

            graph_retrieval_score=float(
                candidate["graph_retrieval_score"]
            ),

            direct_strength=float(
                candidate["direct_strength"]
            ),
            reverse_strength=float(
                candidate["reverse_strength"]
            ),

            two_hop_score=float(
                candidate["two_hop_score"]
            ),
            collaborative_score=float(
                candidate["collaborative_score"]
            ),
            shared_neighbor_score=float(
                candidate["shared_neighbor_score"]
            ),
            behavioral_similarity=float(
                candidate["behavioral_similarity"]
            ),

            novelty_score=float(
                candidate["novelty_score"]
            ),

            two_hop_path_count=int(
                candidate["two_hop_path_count"]
            ),
            similar_user_support_count=int(
                candidate[
                    "similar_user_support_count"
                ]
            ),
        )

    @staticmethod
    def _parse_pair_features(
        pair: dict[str, Any],
    ) -> GraphPairFeatures:
        return GraphPairFeatures(
            candidate_user_id=int(
                pair["candidate_user_id"]
            ),

            graph_direct_strength=float(
                pair["graph_direct_strength"]
            ),
            graph_reverse_strength=float(
                pair["graph_reverse_strength"]
            ),

            graph_two_hop_score=float(
                pair["graph_two_hop_score"]
            ),
            graph_shared_neighbor_score=float(
                pair[
                    "graph_shared_neighbor_score"
                ]
            ),
            graph_behavioral_similarity=float(
                pair[
                    "graph_behavioral_similarity"
                ]
            ),
            graph_collaborative_score=float(
                pair[
                    "graph_collaborative_score"
                ]
            ),

            graph_community_affinity=float(
                pair[
                    "graph_community_affinity"
                ]
            ),
            graph_novelty_score=float(
                pair[
                    "graph_novelty_score"
                ]
            ),
            graph_retrieval_score=float(
                pair[
                    "graph_retrieval_score"
                ]
            ),

            two_hop_path_count=int(
                pair[
                    "two_hop_path_count"
                ]
            ),
            similar_user_support_count=int(
                pair[
                    "similar_user_support_count"
                ]
            ),

            is_hard_excluded=bool(
                pair[
                    "is_hard_excluded"
                ]
            ),
        )

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _request_json(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(
                method=method,
                url=path,
                json=json,
                timeout=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else self.timeout_seconds
                ),
            )

        except httpx.TimeoutException as exc:
            raise PetersGraphError(
                "PetersGraph request timed out: "
                f"{method} {path}"
            ) from exc

        except httpx.RequestError as exc:
            raise PetersGraphError(
                "Unable to connect to PetersGraph at "
                f"{self.base_url}: {exc}"
            ) from exc

        if response.is_error:
            detail = response.text

            try:
                error_payload = response.json()

                if isinstance(
                    error_payload,
                    dict,
                ):
                    detail = str(
                        error_payload.get(
                            "detail",
                            detail,
                        )
                    )

            except ValueError:
                pass

            raise PetersGraphError(
                "PetersGraph request failed: "
                f"{response.status_code} {detail}"
            )

        try:
            payload = response.json()

        except ValueError as exc:
            raise PetersGraphError(
                "PetersGraph returned invalid JSON"
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise PetersGraphError(
                "PetersGraph returned an unexpected response"
            )

        return payload
    