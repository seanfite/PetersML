from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from features.compatibility import (
    build_compatibility,
    optional_float,
)
from features.definitions import (
    COMPATIBILITY_FIELDS,
    FEATURE_SCHEMA_VERSION,
    GRAPH_NUMERIC_FIELDS,
    MULTI_CATEGORICAL_FIELDS,
    NUMERIC_INPUT_FIELDS,
    PASSTHROUGH_FIELDS,
    SINGLE_CATEGORICAL_FIELDS,
    TARGET_BOOLEAN_FIELDS,
    TARGET_COUNT_FIELDS,
    USER_AFFINITY_FIELDS,
    VOCAB_VERSION,
    feature_schema,
)
from features.user_affinity import (
    validate_affinity_features,
)


DEFAULT_INPUT = "data/training_pairs.jsonl"
DEFAULT_OUTPUT = "data/prepared_pairs.jsonl"
DEFAULT_VOCAB = "data/vocab.json"


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------


def save_feature_schema(
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_schema(),
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def empty_vocab() -> dict[str, Any]:
    fields: dict[
        str,
        dict[str, int],
    ] = {}

    for field in (
        SINGLE_CATEGORICAL_FIELDS
        + MULTI_CATEGORICAL_FIELDS
    ):
        fields[field] = {
            "__PAD__": 0,
            "__UNKNOWN__": 1,
        }

    return {
        "version": VOCAB_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fields": fields,
    }


def load_vocab(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return empty_vocab()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    if "fields" not in loaded:
        print(
            "Existing vocab uses old format; "
            "rebuilding it."
        )

        return empty_vocab()

    for field in (
        SINGLE_CATEGORICAL_FIELDS
        + MULTI_CATEGORICAL_FIELDS
    ):
        if field not in loaded["fields"]:
            loaded["fields"][field] = {
                "__PAD__": 0,
                "__UNKNOWN__": 1,
            }

    loaded["version"] = VOCAB_VERSION
    loaded[
        "feature_schema_version"
    ] = FEATURE_SCHEMA_VERSION

    return loaded


def save_vocab(
    path: Path,
    vocab: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vocab,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")


def add_vocab_value(
    vocab: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    if value is None:
        return

    text = str(
        value
    ).strip()

    if not text:
        return

    field_vocab = vocab[
        "fields"
    ][field]

    if text not in field_vocab:
        field_vocab[text] = len(
            field_vocab
        )


def update_vocab_from_row(
    vocab: dict[str, Any],
    row: dict[str, Any],
) -> None:
    for field in SINGLE_CATEGORICAL_FIELDS:
        add_vocab_value(
            vocab,
            field,
            row.get(field),
        )

    for field in MULTI_CATEGORICAL_FIELDS:
        values = row.get(
            field
        )

        if not values:
            continue

        if not isinstance(
            values,
            list,
        ):
            values = [
                values
            ]

        for value in values:
            add_vocab_value(
                vocab,
                field,
                value,
            )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_single(
    vocab: dict[str, Any],
    field: str,
    value: Any,
) -> int:
    if value is None:
        return 0

    text = str(
        value
    ).strip()

    if not text:
        return 0

    field_vocab = vocab[
        "fields"
    ][field]

    return field_vocab.get(
        text,
        field_vocab[
            "__UNKNOWN__"
        ],
    )


def encode_multi(
    vocab: dict[str, Any],
    field: str,
    values: Any,
) -> list[int]:
    if not values:
        return []

    if not isinstance(
        values,
        list,
    ):
        values = [
            values
        ]

    field_vocab = vocab[
        "fields"
    ][field]

    unknown_id = field_vocab[
        "__UNKNOWN__"
    ]

    encoded: list[int] = []

    for value in values:
        text = str(
            value
        ).strip()

        if not text:
            continue

        encoded.append(
            field_vocab.get(
                text,
                unknown_id,
            )
        )

    return sorted(
        set(
            encoded
        )
    )


def count_float(
    value: Any,
) -> float:
    parsed = optional_float(
        value
    )

    return (
        parsed
        if parsed is not None
        else 0.0
    )


def encode_boolean(
    value: Any,
) -> int:
    return (
        1
        if value
        else 0
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_graph_features(
    row: dict[str, Any],
    line_number: int,
) -> None:
    missing = [
        field
        for field in GRAPH_NUMERIC_FIELDS
        if field not in row
    ]

    if missing:
        raise RuntimeError(
            f"Row {line_number} is missing PetersGraph features: "
            f"{', '.join(missing)}. "
            "Build the dataset with the current "
            "training/build_dataset.py first."
        )

    for field in GRAPH_NUMERIC_FIELDS:
        value = optional_float(
            row.get(field)
        )

        if value is None:
            raise RuntimeError(
                f"Row {line_number} has invalid PetersGraph "
                f"feature {field!r}: {row.get(field)!r}"
            )

    if "graph_is_hard_excluded" not in row:
        raise RuntimeError(
            f"Row {line_number} is missing "
            "graph_is_hard_excluded. "
            "Build the dataset with the current "
            "training/build_dataset.py first."
        )


def validate_affinity_section(
    row: dict[str, Any],
    section: str,
    line_number: int,
) -> None:
    values = row.get(
        section
    )

    if not isinstance(
        values,
        dict,
    ):
        raise RuntimeError(
            f"Row {line_number} is missing valid "
            f"{section} affinity data. "
            "Build the dataset with the current "
            "training/build_dataset.py first."
        )

    try:
        validate_affinity_features(
            values
        )

    except RuntimeError as error:
        raise RuntimeError(
            f"Row {line_number} has invalid "
            f"{section}: {error}"
        ) from error


def validate_row(
    row: dict[str, Any],
    line_number: int,
) -> None:
    required = [
        "viewer_user_id",
        "candidate_user_id",
        "cutoff_at",
        "outcome_favorite",
        "outcome_message_started",
        "history_profile_view_count",
        "viewer_age",
        "candidate_age",
    ]

    missing = [
        field
        for field in required
        if field not in row
    ]

    if missing:
        raise RuntimeError(
            f"Row {line_number} is missing expected fields: "
            f"{', '.join(missing)}. "
            "Make sure build_dataset.py uses the current "
            "history/outcome schema."
        )

    validate_graph_features(
        row=row,
        line_number=line_number,
    )

    validate_affinity_section(
        row=row,
        section="viewer_affinities",
        line_number=line_number,
    )

    validate_affinity_section(
        row=row,
        section="candidate_affinities",
        line_number=line_number,
    )


# ---------------------------------------------------------------------------
# Row preparation
# ---------------------------------------------------------------------------


def prepare_row(
    row: dict[str, Any],
    vocab: dict[str, Any],
) -> dict[str, Any]:
    prepared: dict[str, Any] = {
        "metadata": {
            "feature_schema_version": (
                FEATURE_SCHEMA_VERSION
            ),
        },
        "categorical": {},
        "multi_categorical": {},
        "numeric": {},
        "numeric_missing": {},
        "compatibility": {},
        "compatibility_missing": {},
        "viewer_profile_descriptors": {},
        "candidate_profile_descriptors": {},
        "viewer_affinities": {},
        "candidate_affinities": {},
        "targets": {},
        "target_counts": {},
    }

    for field in PASSTHROUGH_FIELDS:
        prepared[
            "metadata"
        ][field] = row.get(
            field
        )

    # This remains metadata/control information.
    # It is intentionally NOT part of NUMERIC_INPUT_FIELDS.
    prepared[
        "metadata"
    ][
        "graph_is_hard_excluded"
    ] = bool(
        row.get(
            "graph_is_hard_excluded",
            False,
        )
    )

    for field in SINGLE_CATEGORICAL_FIELDS:
        prepared[
            "categorical"
        ][field] = encode_single(
            vocab,
            field,
            row.get(field),
        )

    for field in MULTI_CATEGORICAL_FIELDS:
        prepared[
            "multi_categorical"
        ][field] = encode_multi(
            vocab,
            field,
            row.get(field),
        )

    for field in NUMERIC_INPUT_FIELDS:
        value = optional_float(
            row.get(field)
        )

        prepared[
            "numeric"
        ][field] = value

        prepared[
            "numeric_missing"
        ][field] = (
            1.0
            if value is None
            else 0.0
        )

    compatibility = build_compatibility(
        row
    )

    for field in COMPATIBILITY_FIELDS:
        value = compatibility[
            field
        ]

        prepared[
            "compatibility"
        ][field] = value

        prepared[
            "compatibility_missing"
        ][field] = (
            1.0
            if value is None
            else 0.0
        )

    viewer_affinities = row[
        "viewer_affinities"
    ]

    candidate_affinities = row[
        "candidate_affinities"
    ]

    for field in USER_AFFINITY_FIELDS:
        prepared[
            "viewer_affinities"
        ][field] = float(
            viewer_affinities[
                field
            ]
        )

        prepared[
            "candidate_affinities"
        ][field] = float(
            candidate_affinities[
                field
            ]
        )

    for field in TARGET_BOOLEAN_FIELDS:
        prepared[
            "targets"
        ][field] = encode_boolean(
            row.get(field)
        )

    for field in TARGET_COUNT_FIELDS:
        prepared[
            "target_counts"
        ][field] = count_float(
            row.get(field)
        )

    return prepared


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for (
            line_number,
            line,
        ) in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "Invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error

            validate_row(
                row,
                line_number,
            )

            rows.append(
                row
            )

    return rows


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def count_positive_targets(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {
        field: 0
        for field in TARGET_BOOLEAN_FIELDS
    }

    for row in rows:
        for field in TARGET_BOOLEAN_FIELDS:
            if row.get(
                field
            ):
                counts[
                    field
                ] += 1

    return counts


def print_target_summary(
    rows: list[dict[str, Any]],
) -> None:
    positives = count_positive_targets(
        rows
    )

    print()
    print("TARGETS")
    print(
        "────────────────────────────────────────────────────"
    )

    for field in TARGET_BOOLEAN_FIELDS:
        positive = positives[
            field
        ]

        rate = positive / len(
            rows
        )

        print(
            f"{field:<42} "
            f"{positive:>6}/{len(rows):<6} "
            f"{rate:>7.2%}"
        )


def print_numeric_presence(
    rows: list[dict[str, Any]],
) -> None:
    print()
    print("NUMERIC FEATURE PRESENCE")
    print(
        "────────────────────────────────────────────────────"
    )

    for field in NUMERIC_INPUT_FIELDS:
        present = sum(
            optional_float(
                row.get(field)
            ) is not None
            for row in rows
        )

        print(
            f"{field:<48} "
            f"{present:>6}/{len(rows):<6}"
        )


def print_graph_summary(
    rows: list[dict[str, Any]],
) -> None:
    print()
    print("PETERS GRAPH FEATURES")
    print(
        "────────────────────────────────────────────────────"
    )

    for field in GRAPH_NUMERIC_FIELDS:
        values = [
            optional_float(
                row.get(field)
            )
            for row in rows
        ]

        present_values = [
            value
            for value in values
            if value is not None
        ]

        nonzero = sum(
            value != 0.0
            for value in present_values
        )

        if present_values:
            minimum = min(
                present_values
            )

            maximum = max(
                present_values
            )

            mean = (
                sum(
                    present_values
                )
                / len(
                    present_values
                )
            )

        else:
            minimum = 0.0
            maximum = 0.0
            mean = 0.0

        print(
            f"{field:<42} "
            f"present={len(present_values):>5}/{len(rows):<5} "
            f"nonzero={nonzero:>5} "
            f"min={minimum:>8.4f} "
            f"mean={mean:>8.4f} "
            f"max={maximum:>8.4f}"
        )

    hard_excluded = sum(
        bool(
            row.get(
                "graph_is_hard_excluded"
            )
        )
        for row in rows
    )

    print()
    print(
        "graph_is_hard_excluded"
        f"{'':<20} "
        f"{hard_excluded:>5}/{len(rows):<5}"
    )


def print_compatibility_presence(
    rows: list[dict[str, Any]],
) -> None:
    print()
    print("COMPATIBILITY FEATURE PRESENCE")
    print(
        "────────────────────────────────────────────────────"
    )

    compatibility_rows = [
        build_compatibility(
            row
        )
        for row in rows
    ]

    for field in COMPATIBILITY_FIELDS:
        present = sum(
            item[
                field
            ] is not None
            for item in compatibility_rows
        )

        print(
            f"{field:<48} "
            f"{present:>6}/{len(rows):<6}"
        )


def print_affinity_summary(
    rows: list[dict[str, Any]],
) -> None:
    print()
    print("USER AFFINITY FEATURES")
    print(
        "────────────────────────────────────────────────────"
    )

    viewer_with_evidence = sum(
        float(
            row["viewer_affinities"][
                "evidence_strength"
            ]
        ) > 0.0
        for row in rows
    )

    candidate_with_evidence = sum(
        float(
            row["candidate_affinities"][
                "evidence_strength"
            ]
        ) > 0.0
        for row in rows
    )

    viewer_recent = sum(
        float(
            row["viewer_affinities"][
                "recent_evidence_strength"
            ]
        ) > 0.0
        for row in rows
    )

    candidate_recent = sum(
        float(
            row["candidate_affinities"][
                "recent_evidence_strength"
            ]
        ) > 0.0
        for row in rows
    )

    print(
        f"{'Viewer rows with evidence':<40} "
        f"{viewer_with_evidence:>6}/{len(rows):<6}"
    )

    print(
        f"{'Candidate rows with evidence':<40} "
        f"{candidate_with_evidence:>6}/{len(rows):<6}"
    )

    print(
        f"{'Viewer rows with recent evidence':<40} "
        f"{viewer_recent:>6}/{len(rows):<6}"
    )

    print(
        f"{'Candidate rows with recent evidence':<40} "
        f"{candidate_recent:>6}/{len(rows):<6}"
    )

    print()

    for field in USER_AFFINITY_FIELDS:
        viewer_values = [
            float(
                row[
                    "viewer_affinities"
                ][field]
            )
            for row in rows
        ]

        candidate_values = [
            float(
                row[
                    "candidate_affinities"
                ][field]
            )
            for row in rows
        ]

        viewer_mean = (
            sum(viewer_values)
            / len(viewer_values)
        )

        candidate_mean = (
            sum(candidate_values)
            / len(candidate_values)
        )

        print(
            f"{field:<34} "
            f"viewer={viewer_mean:>7.4f} "
            f"candidate={candidate_mean:>7.4f}"
        )


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    schema_path: Path,
) -> None:
    print()
    print("PETERS ML FEATURE PREPARATION")
    print("────────────────────────────────────────")
    print(
        f"Feature schema: {FEATURE_SCHEMA_VERSION}"
    )
    print(
        f"Input:          {input_path}"
    )
    print(
        f"Output:         {output_path}"
    )
    print(
        f"Vocab:          {vocab_path}"
    )
    print(
        f"Schema:         {schema_path}"
    )
    print()

    rows = read_jsonl(
        input_path
    )

    if not rows:
        raise RuntimeError(
            "Input dataset contains no rows."
        )

    print(
        f"Loaded rows:    {len(rows):,}"
    )

    vocab = load_vocab(
        vocab_path
    )

    for row in rows:
        update_vocab_from_row(
            vocab,
            row,
        )

    save_vocab(
        vocab_path,
        vocab,
    )

    save_feature_schema(
        schema_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for row in rows:
            prepared = prepare_row(
                row,
                vocab,
            )

            output_file.write(
                json.dumps(
                    prepared,
                    separators=(",", ":"),
                )
            )

            output_file.write(
                "\n"
            )

    print(
        f"Prepared rows:  {len(rows):,}"
    )

    print()
    print("VOCABULARIES")
    print(
        "────────────────────────────────────────────────────"
    )

    for field in (
        SINGLE_CATEGORICAL_FIELDS
        + MULTI_CATEGORICAL_FIELDS
    ):
        print(
            f"{field:<48} "
            f"{len(vocab['fields'][field]):>6}"
        )

    print_target_summary(
        rows
    )

    print_numeric_presence(
        rows
    )

    print_graph_summary(
        rows
    )

    print_compatibility_presence(
        rows
    )

    print_affinity_summary(
        rows
    )

    print()
    print("FEATURE PREPARATION COMPLETE")
    print("────────────────────────────────────────")
    print(
        f"Saved: {output_path}"
    )
    print(
        f"Saved: {vocab_path}"
    )
    print(
        f"Saved: {schema_path}"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Peters ML production features "
            "from history/outcome JSONL."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB,
    )

    parser.add_argument(
        "--schema",
        default=None,
        help=(
            "Optional feature-schema output. "
            "Defaults beside --output."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    vocab_path = Path(
        args.vocab
    )

    schema_path = (
        Path(
            args.schema
        )
        if args.schema
        else output_path.with_name(
            "feature_schema.json"
        )
    )

    if not input_path.exists():
        raise SystemExit(
            "Input dataset does not exist: "
            f"{input_path}"
        )

    prepare_dataset(
        input_path=input_path,
        output_path=output_path,
        vocab_path=vocab_path,
        schema_path=schema_path,
    )


if __name__ == "__main__":
    main()
    

