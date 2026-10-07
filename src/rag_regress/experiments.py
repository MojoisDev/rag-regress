"""Application services for reproducible local retrieval experiments."""

from __future__ import annotations

import platform
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rag_regress import __version__
from rag_regress.artifacts import atomic_write_json
from rag_regress.chunking import chunk_corpus
from rag_regress.config import PipelineConfig, RetrievalConfig, load_pipeline_config
from rag_regress.corpus import load_corpus
from rag_regress.datasets import load_evaluation_dataset
from rag_regress.embeddings import Embedder, EmbeddingMetadata, build_embedder
from rag_regress.errors import UserInputError
from rag_regress.index import LoadedIndexBundle, build_index_bundle, load_index_bundle
from rag_regress.metrics import (
    evidence_hit,
    hit_at_k,
    nearest_rank_percentile,
    recall_at_k,
    reciprocal_rank,
)
from rag_regress.models import (
    AggregateMetrics,
    CaseMetrics,
    CaseResult,
    CorpusSnapshot,
    EnvironmentInfo,
    EvaluationCase,
    EvaluationDataset,
    IndexBundleManifest,
    PipelineSnapshot,
    RetrievalResult,
    RunArtifact,
)


def ingest(config_path: Path, embedder: Embedder | None = None) -> IndexBundleManifest:
    """Build or reuse the deterministic index bundle for one pipeline configuration."""
    config = load_pipeline_config(config_path)
    corpus = load_corpus(config.corpus)
    chunks = chunk_corpus(corpus, config.chunking)
    active_embedder = embedder or build_embedder(config.embedding)
    return build_index_bundle(config, corpus, chunks, active_embedder)


def evaluate(
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    embedder: Embedder | None = None,
) -> RunArtifact:
    """Evaluate a prebuilt exact bundle and atomically publish its run artifact."""
    config = load_pipeline_config(config_path)
    corpus = load_corpus(config.corpus)
    dataset = load_evaluation_dataset(dataset_path, corpus)
    bundle = load_index_bundle(config, corpus.fingerprint)
    active_embedder = embedder or build_embedder(config.embedding)
    verify_embedding_compatibility(active_embedder.metadata, bundle.manifest.embedding)
    active_embedder.embed_query(dataset.cases[0].question)

    started_at = datetime.now(UTC)
    run_started = time.perf_counter()
    cases = tuple(
        evaluate_case(case, config.retrieval, bundle, active_embedder) for case in dataset.cases
    )
    artifact = create_run_artifact(
        config=config,
        corpus=corpus,
        dataset=dataset,
        embedding=active_embedder.metadata,
        cases=cases,
        started_at=started_at,
        duration_ms=(time.perf_counter() - run_started) * 1000,
    )
    atomic_write_json(output_path, artifact.model_dump(mode="json"))
    return artifact


def verify_embedding_compatibility(
    actual: EmbeddingMetadata, expected: EmbeddingMetadata
) -> None:
    """Reject a query embedder whose identity differs from the prebuilt bundle."""
    if actual != expected:
        raise UserInputError(
            "Embedding metadata does not match the prebuilt index bundle: "
            f"expected {expected.provider}/{expected.model}, "
            f"got {actual.provider}/{actual.model}"
        )


def evaluate_case(
    case: EvaluationCase,
    retrieval: RetrievalConfig,
    bundle: LoadedIndexBundle,
    embedder: Embedder,
) -> CaseResult:
    """Run and measure one query against an already validated index bundle."""
    started = time.perf_counter()
    query = embedder.embed_query(case.question)
    scores, positions = bundle.index.search(query, retrieval.top_k)
    results = tuple(
        RetrievalResult(
            chunk_id=bundle.chunks[int(position)].id,
            document_id=bundle.chunks[int(position)].document_id,
            document_path=bundle.chunks[int(position)].document_path,
            rank=rank,
            score=float(score),
            text=bundle.chunks[int(position)].text,
        )
        for rank, (score, position) in enumerate(zip(scores, positions, strict=True), start=1)
        if retrieval.relevance_threshold is None or float(score) >= retrieval.relevance_threshold
    )
    retrieval_ms = (time.perf_counter() - started) * 1000
    paths = [result.document_path for result in results]
    expected_paths = set(case.expected_documents)
    evidence = (
        evidence_hit([result.text for result in results], case.expected_text)
        if case.expected_text
        else None
    )
    return CaseResult(
        id=case.id,
        question=case.question,
        expected_documents=case.expected_documents,
        expected_text=case.expected_text,
        results=results,
        metrics=CaseMetrics(
            recall_at_k=recall_at_k(paths, expected_paths, retrieval.top_k),
            hit_at_k=hit_at_k(paths, expected_paths, retrieval.top_k),
            reciprocal_rank=reciprocal_rank(paths, expected_paths),
            evidence_hit=evidence,
        ),
        retrieval_ms=retrieval_ms,
    )


def create_run_artifact(
    *,
    config: PipelineConfig,
    corpus: CorpusSnapshot,
    dataset: EvaluationDataset,
    embedding: EmbeddingMetadata,
    cases: tuple[CaseResult, ...],
    started_at: datetime,
    duration_ms: float,
) -> RunArtifact:
    """Assemble the complete versioned artifact after every case has succeeded."""
    evidence_hits = [
        case.metrics.evidence_hit for case in cases if case.metrics.evidence_hit is not None
    ]
    latencies = [case.retrieval_ms for case in cases]
    recall_values, hit_values, reciprocal_ranks = _exact_case_quality(cases)
    return RunArtifact(
        schema_version=1,
        tool_version=__version__,
        metric_definition_version="1",
        dataset_name=dataset.name,
        corpus_fingerprint=corpus.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        pipeline=pipeline_snapshot(config),
        environment=environment_info(),
        embedding=embedding,
        started_at=started_at,
        duration_ms=duration_ms,
        warnings=(),
        errors=(),
        metrics=AggregateMetrics(
            recall_at_k=_fraction_mean(recall_values),
            hit_rate=_fraction_mean(hit_values),
            mrr=_fraction_mean(reciprocal_ranks),
            evidence_hit_rate=(
                _fraction_mean([Fraction(str(value)) for value in evidence_hits])
                if evidence_hits
                else None
            ),
            p50_retrieval_ms=nearest_rank_percentile(latencies, 0.5),
            p95_retrieval_ms=nearest_rank_percentile(latencies, 0.95),
        ),
        cases=cases,
    )


def pipeline_snapshot(config: PipelineConfig) -> PipelineSnapshot:
    """Capture retrieval settings without leaking corpus or local storage paths."""
    return PipelineSnapshot(
        corpus_include_patterns=tuple(config.corpus.include),
        chunking=config.chunking,
        embedding=config.embedding,
        retrieval=config.retrieval,
    )


def environment_info() -> EnvironmentInfo:
    """Record installed dependency versions without importing optional model packages."""
    return EnvironmentInfo(
        python_version=sys.version,
        platform=platform.platform(),
        numpy_version=_distribution_version("numpy"),
        faiss_version=_distribution_version("faiss-cpu"),
        sentence_transformers_version=_distribution_version("sentence-transformers"),
        rag_regress_version=_distribution_version("rag-regress"),
    )


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _exact_case_quality(
    cases: tuple[CaseResult, ...],
) -> tuple[list[Fraction], list[Fraction], list[Fraction]]:
    recalls: list[Fraction] = []
    hits: list[Fraction] = []
    reciprocal_ranks: list[Fraction] = []
    for case in cases:
        expected = set(case.expected_documents)
        retrieved = {result.document_path for result in case.results}
        relevant_count = len(retrieved & expected)
        recalls.append(Fraction(relevant_count, len(expected)))
        hits.append(Fraction(int(relevant_count > 0), 1))
        first_relevant_rank = next(
            (
                rank
                for rank, result in enumerate(case.results, start=1)
                if result.document_path in expected
            ),
            None,
        )
        reciprocal_ranks.append(
            Fraction(1, first_relevant_rank)
            if first_relevant_rank is not None
            else Fraction(0, 1)
        )
    return recalls, hits, reciprocal_ranks


def _fraction_mean(values: list[Fraction]) -> float:
    return float(sum(values, start=Fraction(0, 1)) / len(values))
