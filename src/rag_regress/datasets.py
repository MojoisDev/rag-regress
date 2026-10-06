"""Strict parsing and corpus validation for evaluation datasets."""

import json
from pathlib import Path

from pydantic import ValidationError

from rag_regress.errors import UserInputError
from rag_regress.models import CorpusSnapshot, EvaluationCase, EvaluationDataset


def load_evaluation_dataset(path: Path, corpus: CorpusSnapshot) -> EvaluationDataset:
    """Load a versioned dataset and verify every relevance target is in the corpus."""
    parsed = _parse_dataset_json(path)
    _validate_unique_case_ids(parsed.cases)
    _validate_expected_paths(parsed.cases, corpus)
    return parsed


def _parse_dataset_json(path: Path) -> EvaluationDataset:
    """Parse one JSON dataset file into its strict immutable model."""
    source = path.resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        return EvaluationDataset.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise UserInputError(f"Invalid evaluation dataset {source}: {exc}") from exc


def _validate_unique_case_ids(cases: tuple[EvaluationCase, ...]) -> None:
    """Reject case IDs that would make per-case results ambiguous."""
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        duplicate = next(case_id for case_id in ids if ids.count(case_id) > 1)
        raise UserInputError(f"Duplicate evaluation case ID: {duplicate}")


def _validate_expected_paths(
    cases: tuple[EvaluationCase, ...], corpus: CorpusSnapshot
) -> None:
    """Ensure relevance targets name documents from the evaluated corpus snapshot."""
    available_paths = {document.relative_path for document in corpus.documents}
    for case in cases:
        for expected_path in case.expected_documents:
            if expected_path not in available_paths:
                raise UserInputError(
                    f"Evaluation case {case.id!r} references missing corpus document: "
                    f"{expected_path}"
                )
