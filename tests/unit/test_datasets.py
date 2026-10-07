import json
from pathlib import Path

import pytest

from rag_regress.datasets import load_evaluation_dataset
from rag_regress.errors import UserInputError
from rag_regress.models import CorpusSnapshot, Document, RetrievalResult


@pytest.fixture
def corpus() -> CorpusSnapshot:
    return CorpusSnapshot(
        documents=(
            Document(
                id="guide-id",
                relative_path="guide.md",
                title="guide",
                content="Guide text",
                checksum="guide-sum",
            ),
            Document(
                id="nested-id",
                relative_path="nested/faq.md",
                title="faq",
                content="FAQ text",
                checksum="faq-sum",
            ),
        ),
        fingerprint="corpus-fingerprint",
    )


def test_dataset_rejects_missing_documents(tmp_path: Path, corpus: CorpusSnapshot) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "support",
                "cases": [
                    {
                        "id": "missing",
                        "question": "Where is it?",
                        "expected_documents": ["missing.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="missing.md"):
        load_evaluation_dataset(path, corpus)


def test_dataset_rejects_duplicate_case_ids(tmp_path: Path, corpus: CorpusSnapshot) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "support",
                "cases": [
                    {"id": "same", "question": "One?", "expected_documents": ["guide.md"]},
                    {"id": "same", "question": "Two?", "expected_documents": ["guide.md"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="Duplicate evaluation case ID"):
        load_evaluation_dataset(path, corpus)


def test_dataset_normalizes_trimmed_fields_and_document_paths(
    tmp_path: Path, corpus: CorpusSnapshot
) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "  support  ",
                "cases": [
                    {
                        "id": "  reset-password  ",
                        "question": "  How do I reset it?  ",
                        "expected_documents": ["nested\\faq.md"],
                        "expected_text": ["  reset link  "],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_evaluation_dataset(path, corpus)

    assert dataset.name == "support"
    assert dataset.cases[0].id == "reset-password"
    assert dataset.cases[0].question == "How do I reset it?"
    assert dataset.cases[0].expected_documents == ("nested/faq.md",)
    assert dataset.cases[0].expected_text == ("reset link",)


def test_dataset_fingerprint_uses_normalized_canonical_content(
    tmp_path: Path, corpus: CorpusSnapshot
) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "  support  ",
                "cases": [
                    {
                        "id": "  reset-password  ",
                        "question": "  How do I reset it?  ",
                        "expected_documents": ["nested\\faq.md"],
                        "expected_text": ["  reset link  "],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_evaluation_dataset(path, corpus)

    assert dataset.fingerprint == "afcb68d387ff0c90413004e2cd697059498cf3f3446801f835c21e69da053351"


@pytest.mark.parametrize(
    "expected_documents",
    [
        ["guide.md", "guide.md"],
        ["nested/faq.md", "nested\\faq.md"],
        ["../guide.md"],
        ["nested/../guide.md"],
        ["/guide.md"],
    ],
)
def test_dataset_rejects_duplicate_or_unsafe_expected_document_paths(
    tmp_path: Path, corpus: CorpusSnapshot, expected_documents: list[str]
) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "support",
                "cases": [
                    {
                        "id": "paths",
                        "question": "Where is it?",
                        "expected_documents": expected_documents,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError):
        load_evaluation_dataset(path, corpus)


def test_dataset_rejects_unknown_fields(
    tmp_path: Path, corpus: CorpusSnapshot
) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "support",
                "unknown": True,
                "cases": [
                    {
                        "id": "case",
                        "question": "Question?",
                        "expected_documents": ["guide.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="Invalid evaluation dataset"):
        load_evaluation_dataset(path, corpus)


def test_dataset_rejects_blank_required_text(tmp_path: Path, corpus: CorpusSnapshot) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "   ",
                "cases": [
                    {
                        "id": "case",
                        "question": "Question?",
                        "expected_documents": ["guide.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="Invalid evaluation dataset"):
        load_evaluation_dataset(path, corpus)


@pytest.mark.parametrize("rank", [0, -1])
def test_retrieval_result_rejects_non_one_based_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="document-1",
            document_path="guide.md",
            rank=rank,
            score=0.9,
            text="Guide text",
        )
