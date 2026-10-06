# RAG Regression Tool Design

**Date:** 2026-10-06
**Status:** Approved
**Working product name:** `rag-regress`

## 1. Summary

`rag-regress` is a local-first Python command-line tool for detecting retrieval-quality regressions in retrieval-augmented generation systems. It builds a deterministic retrieval pipeline from a document corpus, evaluates that pipeline against labelled questions, compares candidate results with a baseline, and enforces quality gates in local development or CI.

The first release intentionally evaluates retrieval rather than generated answers. It must work without a hosted model, API key, web application, or background service. A developer should be able to reproduce a run from its corpus, evaluation dataset, pipeline configuration, and recorded environment metadata.

The primary product claim is:

> Know when a change to chunking, embeddings, indexing, or retrieval makes a RAG system worse before shipping it.

## 2. Problem

RAG pipelines often change through small implementation choices: chunk size, overlap, embedding model, normalization, similarity metric, result count, and relevance thresholds. Teams commonly inspect a few successful answers manually, but lack a deterministic way to establish whether a candidate configuration improves the corpus as a whole or causes regressions on previously working questions.

General RAG frameworks focus on composing applications. Full RAG products focus on ingestion and chat. Observability platforms cover broad tracing and LLM evaluation. `rag-regress` instead focuses on a smaller workflow: reproducible retrieval experiments, run-to-run diffs, and CI quality gates.

## 3. Users and Use Cases

### Primary user

A developer or small startup team building a document-grounded application who needs to change retrieval configuration without silently reducing answer coverage.

### Primary use cases

1. Establish a baseline for a corpus and labelled question set.
2. Test a different chunking or embedding configuration.
3. Identify exactly which questions improved or regressed.
4. Reproduce one failed question with ranked chunks and scores.
5. Fail CI when quality or latency drops beyond an accepted budget.
6. Share a complete JSON run artifact for review or debugging.

## 4. Goals

- Provide deterministic document and evaluation-dataset fingerprinting.
- Support configurable word-based chunking with validated overlap.
- Generate normalized embeddings through a small provider interface.
- Build and query a FAISS inner-product index for cosine-compatible retrieval.
- Evaluate labelled questions with standard information-retrieval metrics.
- Preserve per-question ranked results and timing details.
- Compare compatible baseline and candidate runs.
- Enforce declarative quality gates with CI-friendly exit codes.
- Produce atomic, versioned, human-readable JSON artifacts.
- Run default tests without network access or model downloads.
- Offer clear CLI errors that identify the invalid file, field, or case.

## 5. Non-Goals for v0.1

- Answer generation or Ollama integration
- LLM-as-judge evaluation
- A web interface or hosted service
- User accounts, workspaces, billing, or multi-tenancy
- PDF, HTML, image, or OCR ingestion
- Remote vector databases
- Keyword, hybrid, graph, or reranked retrieval
- Automatic evaluation-question generation
- Production request tracing or online monitoring
- Compatibility adapters for LangChain, LlamaIndex, or Haystack

These capabilities may be added later only when they serve the regression-testing workflow.

## 6. User Experience

### Ingest and inspect a corpus

```bash
rag-regress ingest --config configs/baseline.yaml
```

The command validates the configuration, discovers supported files, reports ignored files, chunks and embeds the corpus, and persists an index bundle. Re-ingesting an unchanged corpus with the same pipeline configuration may reuse the bundle. Relative corpus and storage paths are resolved from the directory containing the configuration file.

### Evaluate a pipeline

```bash
rag-regress evaluate evals/questions.json \
  --config configs/baseline.yaml \
  --output runs/baseline.json
```

The command validates all inputs before embedding queries. It derives the required index-bundle identity from the corpus and pipeline configuration, then loads that exact bundle. If it does not exist, evaluation fails with an instruction to run `ingest`; evaluation never builds an index implicitly. It prints aggregate metrics and the location of the complete run artifact.

### Compare two runs

```bash
rag-regress compare runs/baseline.json runs/candidate.json
```

The command prints aggregate deltas and lists improved, unchanged, and regressed cases. It rejects incompatible runs unless the only changed values are pipeline parameters intended for comparison.

### Enforce quality gates

```bash
rag-regress check runs/candidate.json \
  --baseline runs/baseline.json \
  --gates quality-gates.yaml
```

The command exits with status `0` when all gates pass, `1` when a quality gate fails, and `2` for invalid input or an execution error.

## 7. Project Structure

```text
rag-from-scratch/
├── src/rag_regress/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── corpus.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── index.py
│   ├── retrieval.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── experiments.py
│   ├── comparison.py
│   ├── quality_gates.py
│   └── artifacts.py
├── tests/
│   ├── fixtures/
│   ├── unit/
│   └── integration/
├── examples/
│   ├── corpus/
│   ├── evals/
│   └── configs/
├── docs/superpowers/specs/
├── pyproject.toml
├── README.md
└── .gitignore
```

Modules expose typed domain operations. The CLI coordinates those operations but contains no retrieval or metric logic.

## 8. Domain Model

### Document

- `id`: stable hash derived from normalized relative path and content checksum
- `relative_path`: POSIX-style path relative to the corpus root
- `title`: filename stem in v0.1
- `content`: normalized UTF-8 text
- `checksum`: SHA-256 of normalized content

### Chunk

- `id`: stable hash of document ID, sequence number, boundaries, and chunking configuration
- `document_id`
- `document_path`
- `sequence`
- `text`
- `start_word`
- `end_word`: exclusive

### RetrievalResult

- `chunk_id`
- `document_id`
- `document_path`
- `rank`: one-based
- `score`: cosine-compatible inner-product score
- `text`

### EvaluationCase

- `id`: unique within the dataset
- `question`: non-empty string
- `expected_documents`: non-empty list of corpus-relative paths
- `expected_text`: optional list of substrings; matching is case-insensitive after whitespace normalization

### RunArtifact

- schema version and tool version
- corpus and dataset fingerprints
- full retrieval-relevant pipeline configuration, excluding absolute corpus and storage paths
- Python, platform, dependency, and embedding-model metadata
- start time, duration, warnings, and errors
- aggregate metrics
- per-case ranked results, metrics, and latency

## 9. Input Formats

### Pipeline configuration

```yaml
schema_version: 1
corpus:
  path: examples/corpus
  include:
    - "**/*.txt"
    - "**/*.md"
chunking:
  strategy: words
  size: 200
  overlap: 40
embedding:
  provider: sentence_transformers
  model: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
retrieval:
  metric: cosine
  top_k: 5
  relevance_threshold: null
storage:
  directory: .rag-regress
```

Rules:

- `schema_version` must equal `1`.
- `size` must be positive.
- `overlap` must be non-negative and smaller than `size`.
- v0.1 accepts only `words`, `sentence_transformers`, and `cosine` for the respective fields.
- `normalize` must be `true` when the metric is cosine.
- `top_k` must be positive.
- Unknown keys are rejected to catch misspellings.

### Evaluation dataset

```json
{
  "schema_version": 1,
  "name": "product-support-v1",
  "cases": [
    {
      "id": "reset-password",
      "question": "How can a user reset their password?",
      "expected_documents": ["account-guide.md"],
      "expected_text": ["reset link expires after 30 minutes"]
    }
  ]
}
```

Expected-document paths are relative to the corpus root, use `/` separators, and must be present in the evaluated corpus. Duplicate case IDs are invalid.

### Quality gates

```yaml
schema_version: 1
minimum:
  recall_at_k: 0.85
  mrr: 0.80
  hit_rate: 0.85
maximum:
  p95_retrieval_ms: 150
  regressed_cases: 2
allowed_drop:
  recall_at_k: 0.02
  mrr: 0.02
```

Absolute gates apply to the candidate. `allowed_drop` and `regressed_cases` require a compatible baseline.

## 10. Processing Flow

### Corpus ingestion

1. Resolve the corpus root and include patterns.
2. Reject paths outside the corpus root.
3. Read UTF-8 `.txt` and `.md` files in sorted relative-path order.
4. Normalize line endings and whitespace used for fingerprinting.
5. Fingerprint documents and the ordered corpus manifest.
6. Create deterministic chunks.
7. Embed chunks in deterministic order and L2-normalize vectors.
8. Create a FAISS `IndexFlatIP` index.
9. Persist vectors, metadata, and a manifest as one index bundle.

The index bundle is written to a temporary directory and renamed into place only after all files succeed. An existing valid bundle is never replaced by a partial build.

### Evaluation

1. Validate the dataset against the indexed corpus.
2. Embed each question and normalize its vector.
3. Search for `min(top_k, indexed_chunk_count)` results.
4. Hydrate ranked vectors with chunk metadata.
5. Apply the configured threshold, if present.
6. Calculate per-case document and optional expected-text matches.
7. Aggregate metrics and latency percentiles.
8. Write the complete run artifact atomically.

### Comparison

Runs are compatible when their artifact schema versions, corpus fingerprints, dataset fingerprints, case IDs, and metric definitions match. Pipeline configurations may differ because that is the subject of the experiment. Incompatible runs produce an error that lists every differing compatibility field.

Each case is classified by reciprocal-rank delta:

- improved: candidate reciprocal rank is greater than baseline
- regressed: candidate reciprocal rank is lower than baseline
- unchanged: values are equal

Aggregate metric deltas are reported in addition to case classifications.

## 11. Metrics

Let `R_q` be the set of expected corpus-relative document paths for question `q`, and let `A_q(k)` be the document paths represented by the first `k` retrieved chunks.

- **Recall@k per case:** `|R_q intersect A_q(k)| / |R_q|`
- **Hit@k per case:** `1` if the intersection is non-empty, otherwise `0`
- **Reciprocal rank per case:** reciprocal of the rank of the first chunk whose document is relevant, otherwise `0`
- **Aggregate Recall@k:** arithmetic mean of per-case Recall@k
- **Hit rate:** arithmetic mean of per-case Hit@k
- **MRR:** arithmetic mean of reciprocal ranks
- **Retrieval latency:** elapsed monotonic time around query embedding plus FAISS search and hydration; one unmeasured warm-up query runs before evaluating the dataset so model loading is excluded
- **p50/p95 latency:** nearest-rank percentiles over case latencies

Multiple chunks from one document count once for Recall@k, while reciprocal rank uses the first relevant chunk.

For a case with `expected_text`, its evidence hit is `1` only when every expected substring appears in the whitespace-normalized, case-insensitive concatenation of its top-k retrieved chunk texts; otherwise it is `0`. `evidence_hit_rate` is the mean over cases that define `expected_text`. It is `null` when no cases define expected text and does not alter document-level metrics.

## 12. Embedding and Index Interfaces

The embedding interface exposes model metadata, vector dimension, document embedding, and query embedding. v0.1 includes:

- `SentenceTransformerEmbedder` for real usage
- `DeterministicTestEmbedder` in test code for offline repeatability

The production adapter receives text and returns `float32` NumPy arrays. Dimension is discovered from produced vectors rather than hard-coded. Non-finite, empty, or inconsistent vectors are rejected before indexing.

The index interface supports build, search, save, and load. v0.1 uses `IndexFlatIP` and normalized vectors so scores are cosine similarity values. Index metadata records the embedding dimension and chunk ordering.

## 13. Error Handling

Expected user errors use concise messages and exit status `2`. They include invalid configuration, malformed datasets, missing files, unsupported encodings or extensions, duplicate IDs, missing expected documents, empty corpora, and incompatible artifacts.

Unexpected exceptions retain their causal chain in debug mode. The normal CLI prints an actionable summary without a full traceback and suggests rerunning with `--debug`.

Model-loading and embedding failures identify the configured provider and model. Network access is never attempted silently: model acquisition behavior follows SentenceTransformers/Hugging Face configuration, and offline errors explain how to pre-download or change the model.

A failed evaluation does not emit a completed run artifact. Optional diagnostic failure information may be written with a `.failed.json` suffix, but it is never accepted by `compare` or `check`.

## 14. Determinism and Reproducibility

- Files and cases are processed in stable sorted order.
- IDs and fingerprints use SHA-256 over canonical serialized data.
- Run artifacts use canonical field names and a versioned schema.
- Wall-clock timestamps are recorded but excluded from identity fingerprints.
- Latency is reported but excluded from deterministic result equality.
- Dependency versions include NumPy, FAISS, sentence-transformers, and the embedding model identifier.
- Tests compare semantic artifact content after removing timestamps and latency.

Exact embedding values may differ across hardware or dependency versions; recorded environment metadata makes those differences visible rather than claiming bit-for-bit portability.

## 15. Security and Data Boundaries

- The CLI reads only files matched beneath the configured corpus root.
- Symlinks resolving outside the corpus root are rejected.
- No corpus contents, questions, embeddings, or metrics are uploaded by the tool.
- v0.1 contains no analytics or telemetry.
- Artifacts may contain retrieved text and therefore must be treated as potentially sensitive project data.
- YAML is parsed with safe loading and validated into strict models.

## 16. Testing Strategy

### Unit tests

- Chunk creation, overlap, boundaries, empty input, and invalid parameters
- Stable document/chunk IDs and corpus fingerprints
- Dataset and configuration validation
- Vector normalization and dimension validation
- Recall@k, hit rate, MRR, evidence hit rate, and percentiles
- Comparison classifications and compatibility failures
- Every quality-gate operator and exit outcome
- Atomic artifact writes and invalid schema versions

### Integration tests

- Ingest fixture corpus, evaluate fixture questions, and produce a run
- Evaluate a candidate configuration and compare it with the baseline
- Pass and fail quality gates with asserted process exit codes
- Reload a persisted FAISS index and reproduce rankings
- Reject an evaluation dataset referencing a missing document

Default tests use the deterministic test embedder and require no network or Ollama process. A separately marked smoke test exercises `all-MiniLM-L6-v2` when the model is available locally.

## 17. Repository and Delivery Standards

- Python 3.11 and 3.12 are supported.
- Packaging and dependencies are declared in `pyproject.toml`.
- The console entry point is `rag-regress`.
- Formatting and linting use Ruff.
- Static type checking uses mypy.
- Tests use pytest.
- CI runs tests, linting, type checking, and package build.
- The local `.venv`, generated indexes, run artifacts, caches, and model files are ignored by Git.
- The README leads with the regression-testing problem, includes a five-minute example, and displays a sample comparison report.

## 18. v0.1 Acceptance Criteria

The release is complete when a clean checkout can:

1. Install the package from `pyproject.toml`.
2. Run all default checks without network access after dependencies are installed.
3. Ingest the included example text/Markdown corpus.
4. Evaluate the included labelled dataset using a locally available SentenceTransformer model.
5. Persist and reload a valid FAISS bundle.
6. Produce a versioned JSON run artifact with per-case results and aggregate metrics.
7. Compare baseline and candidate artifacts and identify regressed cases.
8. Pass and fail quality gates with documented exit codes.
9. Reject incompatible comparisons and invalid inputs with actionable errors.
10. Demonstrate the complete workflow in the README without requiring Ollama or a paid API.

## 19. Post-v0.1 Direction

Subsequent releases may add configuration matrices, GitHub Actions annotations, HTML comparison reports, additional chunkers, hybrid retrieval, reranking, framework adapters, and citation-integrity evaluation. Answer generation will use a provider-neutral interface with Ollama as the first adapter. These additions must preserve the deterministic retrieval baseline rather than turning the project into a general-purpose RAG framework.
