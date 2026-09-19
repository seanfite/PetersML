from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import uvicorn
from fastapi import FastAPI, HTTPException
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, field_validator, model_validator

from inference.ranker import Ranker
from jobs.build_user_affinities import refresh_user_affinities
from jobs.generate_search_flights import generate_search_flights_for_user
from jobs.generate_search_recommended import generate_search_recommended_for_user
from jobs.generate_speed_dating import generate_speed_dating_for_user
from jobs.generate_suggestions import generate_suggestions_for_user


MAX_CANDIDATES_PER_REQUEST = 2000
MAX_IMPRESSIONS_PER_REQUEST = 200

ranker = Ranker()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(f"{name} is not set")

    return value


# ---------------------------------------------------------------------------
# Ranking request / response
# ---------------------------------------------------------------------------


class RankRequest(BaseModel):
    viewer_user_id: int = Field(gt=0)
    candidate_user_ids: list[int] = Field(
        min_length=1,
        max_length=MAX_CANDIDATES_PER_REQUEST,
    )
    top_k: int | None = Field(default=None, gt=0)

    @field_validator("candidate_user_ids")
    @classmethod
    def validate_candidate_ids(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError(
                "candidate_user_ids must all be greater than 0"
            )

        return values


class CandidatePrediction(BaseModel):
    rank: int
    candidate_user_id: int
    score: float
    predictions: dict[str, float]
    graph_features: dict[str, float] | None = None


class RankResponse(BaseModel):
    viewer_user_id: int

    model_name: str
    model_version: str
    model_status: str
    feature_schema_version: int

    graph_enriched: bool
    graph_version_id: int | None
    graph_version: str | None
    graph_schema_version: int | None
    graph_cutoff_at: str | None

    candidate_count: int
    missing_candidate_ids: list[int]
    excluded_candidate_ids: list[int]

    candidates: list[CandidatePrediction]


# ---------------------------------------------------------------------------
# User affinity refresh request / response
# ---------------------------------------------------------------------------


class AffinityRefreshRequest(BaseModel):
    viewer_user_id: int = Field(gt=0)


class AffinityRefreshResponse(BaseModel):
    viewer_user_id: int
    model_version: str

    history_days: int
    recent_days: int
    cutoff_at: str

    users_processed: int
    users_with_long_term_evidence: int
    users_with_recent_evidence: int
    affinity_rows_written: int


# ---------------------------------------------------------------------------
# Daily Suggestions request / response
# ---------------------------------------------------------------------------


class SuggestionGenerateRequest(BaseModel):
    viewer_user_id: int = Field(gt=0)


class SuggestionPrediction(BaseModel):
    rank: int
    model_rank: int
    candidate_user_id: int
    score: float
    selection_type: str
    predictions: dict[str, float]


class SuggestionGenerateResponse(BaseModel):
    batch_id: int
    viewer_user_id: int
    model_version: str
    generated_at: str
    expires_at: str
    generated: bool
    suggestion_count: int
    suggestions: list[SuggestionPrediction]


# ---------------------------------------------------------------------------
# Speed Dating request / response
# ---------------------------------------------------------------------------


class SpeedDatingGenerateRequest(BaseModel):
    viewer_user_id: int = Field(gt=0)


class SpeedDatingRecommendation(BaseModel):
    rank: int
    model_rank: int
    candidate_user_id: int
    score: float
    selection_type: str
    predictions: dict[str, float]
    decision: str | None
    decided_at: str | None


class SpeedDatingGenerateResponse(BaseModel):
    batch_id: int
    viewer_user_id: int
    model_version: str
    generated_at: str
    expires_at: str
    generated: bool
    recommendation_count: int
    recommendations: list[SpeedDatingRecommendation]


# ---------------------------------------------------------------------------
# Search Recommended request / response
# ---------------------------------------------------------------------------


class SearchRecommendedGenerateRequest(BaseModel):
    viewer_user_id: int = Field(gt=0)


class SearchRecommendedRecommendation(BaseModel):
    rank: int
    model_rank: int
    candidate_user_id: int
    score: float
    selection_type: str
    predictions: dict[str, float]


class SearchRecommendedGenerateResponse(BaseModel):
    batch_id: int
    viewer_user_id: int
    model_version: str
    generated_at: str
    expires_at: str
    generated: bool
    recommendation_count: int
    recommendations: list[SearchRecommendedRecommendation]


# ---------------------------------------------------------------------------
# Search Flights request / response
# ---------------------------------------------------------------------------


class SearchFlightsGenerateRequest(BaseModel):
    viewer_user_id: int = Field(gt=0)


class SearchFlightFilter(BaseModel):
    kind: str
    value: str


class SearchFlightRecipeResponse(BaseModel):
    rank: int
    location_key: str
    filters: list[SearchFlightFilter]
    estimated_result_count: int
    score: float
    selection_type: str


class SearchFlightsGenerateResponse(BaseModel):
    batch_id: int
    viewer_user_id: int

    model_name: str
    model_version: str
    algorithm_version: str

    generated_at: str
    expires_at: str
    generated: bool

    candidate_profiles_considered: int
    ranked_profiles_considered: int

    flight_count: int
    flights: list[SearchFlightRecipeResponse]


# ---------------------------------------------------------------------------
# Impression request / response
# ---------------------------------------------------------------------------


class ImpressionItem(BaseModel):
    candidate_user_id: int = Field(gt=0)
    rank: int = Field(gt=0)
    score: float
    predictions: dict[str, float]

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(
                "score must be finite"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "score must be between 0 and 1"
            )

        return value

    @field_validator("predictions")
    @classmethod
    def validate_predictions(
        cls,
        values: dict[str, float],
    ) -> dict[str, float]:
        for key, value in values.items():
            if not key.strip():
                raise ValueError(
                    "prediction names cannot be empty"
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"prediction {key!r} must be finite"
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"prediction {key!r} must be between 0 and 1"
                )

        return values


class ImpressionRequest(BaseModel):
    viewer_user_id: int = Field(gt=0)

    surface: str = Field(
        min_length=1,
        max_length=64,
    )

    context_key: str | None = Field(
        default=None,
        max_length=200,
    )

    model_name: str = Field(
        min_length=1,
        max_length=100,
    )

    model_version: str = Field(
        min_length=1,
        max_length=100,
    )

    items: list[ImpressionItem] = Field(
        min_length=1,
        max_length=MAX_IMPRESSIONS_PER_REQUEST,
    )

    @field_validator(
        "surface",
        "model_name",
        "model_version",
    )
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "value cannot be empty"
            )

        return value

    @field_validator("surface")
    @classmethod
    def validate_surface(cls, value: str) -> str:
        value = value.lower()

        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789_-."
        )

        if any(character not in allowed for character in value):
            raise ValueError(
                "surface may contain only lowercase letters, numbers, "
                "underscores, hyphens, and periods"
            )

        return value

    @field_validator("context_key")
    @classmethod
    def normalize_context_key(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_items(self) -> ImpressionRequest:
        candidate_ids: set[int] = set()
        ranks: set[int] = set()

        for item in self.items:
            if item.candidate_user_id == self.viewer_user_id:
                raise ValueError(
                    "viewer_user_id cannot equal candidate_user_id"
                )

            if item.candidate_user_id in candidate_ids:
                raise ValueError(
                    "candidate_user_id appears more than once "
                    "in the impression batch"
                )

            if item.rank in ranks:
                raise ValueError(
                    "rank appears more than once in the impression batch"
                )

            candidate_ids.add(
                item.candidate_user_id
            )
            ranks.add(
                item.rank
            )

        return self


class ImpressionResponse(BaseModel):
    viewer_user_id: int
    surface: str
    context_key: str | None
    model_name: str
    model_version: str
    impression_count: int


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model_name: str | None
    model_version: str | None
    device: str | None
    feature_schema_version: int | None
    graph_aware: bool


# ---------------------------------------------------------------------------
# Impression persistence
# ---------------------------------------------------------------------------


def record_impressions(
    request: ImpressionRequest,
) -> int:
    dsn = require_env(
        "APP_DSN"
    )

    rows = [
        (
            request.viewer_user_id,
            item.candidate_user_id,
            request.surface,
            request.context_key,
            item.rank,
            item.score,
            request.model_name,
            request.model_version,
            Jsonb(
                item.predictions
            ),
        )
        for item in request.items
    ]

    with psycopg.connect(
        dsn
    ) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO ml_recommendation_impressions (
                    viewer_user_id,
                    candidate_user_id,
                    surface,
                    context_key,
                    rank,
                    score,
                    model_name,
                    model_version,
                    predictions
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
                    %s
                )
                """,
                rows,
            )

        conn.commit()

    return len(
        rows
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    print()
    print("PETERS ML INFERENCE SERVICE")
    print("────────────────────────────────────────")
    print("Loading production model...")

    try:
        production = (
            ranker.model_manager.load()
        )

        graph_aware = any(
            field.startswith("graph_")
            for field in production.loaded.numeric_fields
        )

        print(
            f"Model:      "
            f"{production.record.model_name}"
        )
        print(
            f"Version:    "
            f"{production.record.version}"
        )
        print(
            f"Device:     "
            f"{production.loaded.device}"
        )
        print(
            f"Schema:     "
            f"{production.loaded.feature_schema_version}"
        )
        print(
            f"Graph:      "
            f"{'yes' if graph_aware else 'no'}"
        )
        print(
            "Status:     ready"
        )
        print()

    except Exception as error:
        print(
            f"Startup failed: {error}"
        )
        raise

    yield


app = FastAPI(
    title="Peters ML Ranking Service",
    version="1.4.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    production = (
        ranker.model_manager.production
    )

    if production is None:
        return HealthResponse(
            status="not_ready",
            model_name=None,
            model_version=None,
            device=None,
            feature_schema_version=None,
            graph_aware=False,
        )

    graph_aware = any(
        field.startswith("graph_")
        for field in production.loaded.numeric_fields
    )

    return HealthResponse(
        status="ready",
        model_name=(
            production.record.model_name
        ),
        model_version=(
            production.record.version
        ),
        device=str(
            production.loaded.device
        ),
        feature_schema_version=(
            production.loaded.feature_schema_version
        ),
        graph_aware=graph_aware,
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


@app.post(
    "/rank",
    response_model=RankResponse,
)
def rank(
    request: RankRequest,
) -> RankResponse:
    try:
        result = ranker.rank(
            viewer_user_id=(
                request.viewer_user_id
            ),
            candidate_user_ids=(
                request.candidate_user_ids
            ),
            top_k=(
                request.top_k
            ),
            refresh_model=True,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return RankResponse(
        **result.as_dict()
    )


# ---------------------------------------------------------------------------
# User affinity refresh
# ---------------------------------------------------------------------------


@app.post(
    "/affinities/refresh",
    response_model=AffinityRefreshResponse,
)
def refresh_affinities(
    request: AffinityRefreshRequest,
) -> AffinityRefreshResponse:
    """
    Rebuild one user's current behavioral affinity state.

    This runs before fresh recommendation inference so Suggestions,
    Speed Dating, Search Recommended, and Search Flights all consume
    the same latest persisted affinity state.
    """

    try:
        result = refresh_user_affinities(
            viewer_user_id=(
                request.viewer_user_id
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except psycopg.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to refresh user affinities",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return AffinityRefreshResponse(
        viewer_user_id=(
            request.viewer_user_id
        ),
        model_version=str(
            result["model_version"]
        ),
        history_days=int(
            result["history_days"]
        ),
        recent_days=int(
            result["recent_days"]
        ),
        cutoff_at=str(
            result["cutoff_at"]
        ),
        users_processed=int(
            result["users_processed"]
        ),
        users_with_long_term_evidence=int(
            result[
                "users_with_long_term_evidence"
            ]
        ),
        users_with_recent_evidence=int(
            result[
                "users_with_recent_evidence"
            ]
        ),
        affinity_rows_written=int(
            result[
                "affinity_rows_written"
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Daily Suggestions
# ---------------------------------------------------------------------------


@app.post(
    "/suggestions/generate",
    response_model=SuggestionGenerateResponse,
)
def generate_suggestions(
    request: SuggestionGenerateRequest,
) -> SuggestionGenerateResponse:
    """
    Return the viewer's existing valid Daily Suggestions batch or generate
    and persist a new batch.

    Production behavior:
    - requires active Plus access
    - up to 15 suggestions
    - 12 direct model picks + 3 sampled picks
    - no deep exploration
    - reuses a valid unexpired batch
    - uses the already-loaded production Ranker
    """

    try:
        batch = generate_suggestions_for_user(
            viewer_user_id=(
                request.viewer_user_id
            ),
            limit=15,
            force=False,
            require_plus=True,
            ranker_instance=ranker,
        )

    except PermissionError as error:
        raise HTTPException(
            status_code=403,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except psycopg.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate daily suggestions",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return SuggestionGenerateResponse(
        **batch.as_dict()
    )


# ---------------------------------------------------------------------------
# Speed Dating
# ---------------------------------------------------------------------------


@app.post(
    "/speed-dating/generate",
    response_model=SpeedDatingGenerateResponse,
)
def generate_speed_dating(
    request: SpeedDatingGenerateRequest,
) -> SpeedDatingGenerateResponse:
    """
    Return the viewer's existing valid Speed Dating batch or generate one.

    Production behavior:
    - up to 10 recommendations
    - 7 direct model picks + 2 sampled + 1 exploration
    - reuses a valid unexpired batch
    - Daily Suggestions overlap is allowed
    - previous Speed Dating appearances are allowed
    - uses the already-loaded production Ranker
    """

    try:
        batch = generate_speed_dating_for_user(
            viewer_user_id=(
                request.viewer_user_id
            ),
            limit=10,
            force=False,
            ranker_instance=ranker,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except psycopg.Error as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "Speed Dating recommendations"
            ),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return SpeedDatingGenerateResponse(
        **batch.as_dict()
    )


# ---------------------------------------------------------------------------
# Search AI Recommended
# ---------------------------------------------------------------------------


@app.post(
    "/search-recommended/generate",
    response_model=SearchRecommendedGenerateResponse,
)
def generate_search_recommended(
    request: SearchRecommendedGenerateRequest,
) -> SearchRecommendedGenerateResponse:
    """
    Return the viewer's existing Search AI recommendation pool or generate one.

    Production behavior:
    - up to 100 personalized candidates
    - 80 direct model picks + 15 sampled + 5 exploration
    - location-independent recommendation pool
    - reuses a valid unexpired 24-hour batch
    - PetersServer later applies live map/radius/search filters
    - uses the already-loaded production Ranker
    """

    try:
        batch = (
            generate_search_recommended_for_user(
                viewer_user_id=(
                    request.viewer_user_id
                ),
                limit=100,
                force=False,
                ranker_instance=ranker,
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except psycopg.Error as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "Search AI recommendations"
            ),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return SearchRecommendedGenerateResponse(
        **batch.as_dict()
    )


# ---------------------------------------------------------------------------
# Search Flights
# ---------------------------------------------------------------------------


@app.post(
    "/search-flights/generate",
    response_model=SearchFlightsGenerateResponse,
)
def generate_search_flights(
    request: SearchFlightsGenerateRequest,
) -> SearchFlightsGenerateResponse:
    """
    Return the viewer's existing valid Search Flight batch or generate one.

    Production behavior:
    - requires active Plus access
    - generates up to five personalized Search Flight filter recipes
    - assigns the fixed Search Flight location slots by rank
    - persists the batch and filters to Postgres
    - reuses a valid unexpired 24-hour batch
    - uses the already-loaded production Ranker
    """

    try:
        result = generate_search_flights_for_user(
            viewer_user_id=(
                request.viewer_user_id
            ),
            require_plus=True,
            force=False,
            ranker_instance=ranker,
        )

    except PermissionError as error:
        raise HTTPException(
            status_code=403,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except psycopg.Error as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "Search Flight recommendations"
            ),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return SearchFlightsGenerateResponse(
        **result.as_dict()
    )


# ---------------------------------------------------------------------------
# Impressions
# ---------------------------------------------------------------------------


@app.post(
    "/impressions",
    response_model=ImpressionResponse,
)
def impressions(
    request: ImpressionRequest,
) -> ImpressionResponse:
    try:
        count = record_impressions(
            request
        )

    except psycopg.Error as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to persist "
                "recommendation impressions"
            ),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    return ImpressionResponse(
        viewer_user_id=(
            request.viewer_user_id
        ),
        surface=request.surface,
        context_key=request.context_key,
        model_name=request.model_name,
        model_version=request.model_version,
        impression_count=count,
    )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict[str, Any]:
    production = (
        ranker.model_manager.production
    )

    return {
        "service": "peters-ml-ranking",
        "status": (
            "ready"
            if production is not None
            else "not_ready"
        ),
        "model_version": (
            production.record.version
            if production is not None
            else None
        ),
        "daily_suggestions": True,
        "speed_dating": True,
        "search_recommended": True,
        "search_flights": True,
        "user_affinity_refresh": True,
        "impression_logging": True,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    port = int(
        os.getenv(
            "PORT",
            "8080",
        )
    )

    uvicorn.run(
        "inference.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
    