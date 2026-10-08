# RAG Regression Tool Implementation Plan

> **Status: historical implementation plan.** The v0.1 implementation described
> here is present in `src/`, `tests/`, `examples/`, and CI. The original unchecked
> steps below preserve the planning record; they are not a current backlog or
> evidence that each historical command or commit was executed. Consult the
> README and current source for supported configuration and development checks.
> Subsequent fixes require immutable embedding revisions, validate stored vectors
> in bounded batches, and validate/reuse a concurrently published index bundle.

**Goal:** Build a local-first `rag-regress` CLI that creates reproducible FAISS retrieval indexes, evaluates labelled questions, compares pipeline runs, and enforces CI quality gates.

**Architecture:** A small Python modular monolith separates corpus loading, deterministic chunking, embeddings, FAISS persistence, evaluation, comparison, and CLI presentation. Pydantic models define strict versioned inputs and artifacts; service functions remain independent from Typer so tests can use a deterministic offline embedder.

**Tech Stack:** Python 3.11/3.12, Pydantic 2, PyYAML, NumPy, FAISS CPU, Sentence Transformers, Typer, pytest, Ruff, mypy, `build`.

**Spec:** `docs/design.md`

## Global Constraints

- Support Python 3.11 and 3.12.
- The console entry point is exactly `rag-regress`.
- v0.1 reads UTF-8 `.txt` and `.md` files only.
- v0.1 supports word chunking, Sentence Transformers, normalized vectors, and FAISS `IndexFlatIP` only.
- Default tests must run without network access, Ollama, or a downloaded embedding model.
- Unknown YAML and JSON fields must be rejected.
- User input errors exit `2`; failed quality gates exit `1`; success exits `0`.
- No telemetry, corpus upload, hosted API, generation, web UI, PDF ingestion, or framework adapter belongs in v0.1.
- File edits use focused modules and typed interfaces; CLI code contains no retrieval or metric logic.
- Each task follows red-green-refactor and ends in its own commit.

## Review Focus

- A symlink inside the corpus that resolves outside the corpus root must be rejected; Task 2 adds the regression test.
- A zero, negative, or non-finite embedding vector must fail before FAISS receives it; Task 4 adds the regression tests.
- An interrupted index or run-artifact write must leave the previous valid output intact and no valid-looking partial output; Tasks 5 and 7 add the regression tests.
- Two runs with the same cases but different corpus or dataset fingerprints must be rejected with every mismatched compatibility field listed; Task 8 adds the regression test.
- A quality-gate file using `allowed_drop` without a baseline must fail as invalid input rather than silently skip the relative gate; Task 8 adds the regression test.

---

## File Map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, commands, test/lint/type configuration |
| `.gitignore` | Exclude virtual environments, caches, indexes, models, and generated runs |
| `src/rag_regress/__init__.py` | Package version |
| `src/rag_regress/errors.py` | Typed user-facing exception hierarchy |
| `src/rag_regress/hashing.py` | Canonical JSON serialization and SHA-256 identity helpers |
| `src/rag_regress/config.py` | Strict YAML models, path resolution, configuration fingerprints |
| `src/rag_regress/models.py` | Immutable domain and versioned artifact models |
| `src/rag_regress/corpus.py` | Safe file discovery, normalization, document IDs, corpus fingerprint |
| `src/rag_regress/chunking.py` | Deterministic word chunks and chunk IDs |
| `src/rag_regress/embeddings.py` | Embedder protocol, Sentence Transformer adapter, vector validation |
| `src/rag_regress/index.py` | FAISS wrapper and atomic index-bundle persistence |
| `src/rag_regress/datasets.py` | Evaluation-dataset parsing and corpus validation |
| `src/rag_regress/metrics.py` | Retrieval metrics, evidence matching, percentiles |
| `src/rag_regress/artifacts.py` | Canonical JSON, atomic file writes, artifact loading |
| `src/rag_regress/experiments.py` | Ingestion and evaluation application services |
| `src/rag_regress/comparison.py` | Run compatibility and case/aggregate deltas |
| `src/rag_regress/quality_gates.py` | Strict gate configuration and pass/fail evaluation |
| `src/rag_regress/cli.py` | Typer commands, output formatting, exit-code mapping |
| `tests/helpers.py` | Deterministic test embedder and reusable fixtures |
| `tests/unit/` | Focused component tests |
| `tests/integration/` | Full local workflow and CLI tests |
| `examples/` | Runnable sample corpus, dataset, configs, and preserved prototypes |
| `.github/workflows/ci.yml` | Python 3.11/3.12 checks and package build |

### Task 1: Establish the Package, Tooling, and Strict Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/rag_regress/__init__.py`
- Create: `src/rag_regress/errors.py`
- Create: `src/rag_regress/hashing.py`
- Create: `src/rag_regress/config.py`
- Create: `tests/unit/test_config.py`
- Move: `main.py` to `examples/prototype/naive_rag.py`
- Move: `faiss_test.py` to `examples/prototype/faiss_basics.py`
- Create: `examples/prototype/README.md`

**Interfaces:**
- Consumes: the approved YAML structure in the design spec.
- Produces: `hash_canonical(payload: object) -> str`, `PipelineConfig`, `load_pipeline_config(path: Path) -> PipelineConfig`, `index_config_fingerprint(config: PipelineConfig) -> str`, and `UserInputError`.

- [ ] **Step 1: Preserve the exploratory scripts and add package metadata**

Move the two existing scripts without changing their contents. Create `pyproject.toml` with these exact dependency groups and tool entry point:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-regress"
version = "0.1.0"
description = "Local-first retrieval regression testing for RAG pipelines"
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
  "faiss-cpu>=1.8,<2",
  "numpy>=1.26,<3",
  "pydantic>=2.8,<3",
  "PyYAML>=6,<7",
  "sentence-transformers>=3,<7",
  "typer>=0.12,<1",
]

[project.optional-dependencies]
dev = [
  "build>=1.2,<2",
  "mypy>=1.11,<2",
  "pytest>=8,<10",
  "ruff>=0.6,<1",
  "types-PyYAML>=6,<7",
]
prototype = ["ollama>=0.4,<1"]

[project.scripts]
rag-regress = "rag_regress.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["model: requires a locally available Sentence Transformer model"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["rag_regress"]
```

Create `.gitignore` with `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `dist/`, `build/`, `*.egg-info/`, `.rag-regress/`, and `runs/`. Create a minimal root `README.md` containing the project name, the product claim from the spec, and the sentence `Implementation in progress; see the approved design and implementation plan under docs/.` Document in `examples/prototype/README.md` that these scripts are the preserved learning prototype, are not part of the CLI, and that `naive_rag.py` additionally needs Ollama.

- [ ] **Step 2: Write strict configuration tests**

```python
from pathlib import Path

import pytest

from rag_regress.config import load_pipeline_config
from rag_regress.errors import UserInputError


def test_load_config_resolves_paths_from_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "project" / "configs"
    config_dir.mkdir(parents=True)
    path = config_dir / "baseline.yaml"
    path.write_text(
        """schema_version: 1
corpus:
  path: ../corpus
  include: ['**/*.md']
chunking: {strategy: words, size: 200, overlap: 40}
embedding:
  provider: sentence_transformers
  model: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
retrieval: {metric: cosine, top_k: 5, relevance_threshold: null}
storage: {directory: ../state}
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(path)

    assert config.corpus.path == (config_dir / "../corpus").resolve()
    assert config.storage.directory == (config_dir / "../state").resolve()


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("size: 0, overlap: 0", "size"),
        ("size: 10, overlap: 10", "overlap"),
        ("size: 10, overlap: -1", "overlap"),
    ],
)
def test_invalid_chunking_is_rejected(
    tmp_path: Path, fragment: str, message: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""schema_version: 1
corpus: {{path: corpus, include: ['**/*.md']}}
chunking: {{strategy: words, {fragment}}}
embedding: {{provider: sentence_transformers, model: test, normalize: true}}
retrieval: {{metric: cosine, top_k: 5, relevance_threshold: null}}
storage: {{directory: state}}
""",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match=message):
        load_pipeline_config(path)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")

    with pytest.raises(UserInputError, match="unknown"):
        load_pipeline_config(path)
```

- [ ] **Step 3: Run the focused tests and confirm red**

Run:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/unit/test_config.py -v
```

Expected: collection fails because `rag_regress.config` does not exist.

- [ ] **Step 4: Implement strict configuration loading**

Create frozen Pydantic models with `ConfigDict(extra="forbid")`. Add a model validator enforcing `0 <= overlap < size`, positive `top_k`, schema version `1`, `normalize=True`, and the supported literal values. Translate YAML, file, and Pydantic errors into `UserInputError` while retaining the source path in the message.

Use this public loader and fingerprint boundary:

```python
def load_pipeline_config(path: Path) -> PipelineConfig:
    source = path.resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        config = PipelineConfig.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise UserInputError(f"Invalid pipeline config {source}: {exc}") from exc

    base = source.parent
    return config.model_copy(
        update={
            "corpus": config.corpus.model_copy(
                update={"path": (base / config.corpus.path).resolve()}
            ),
            "storage": config.storage.model_copy(
                update={"directory": (base / config.storage.directory).resolve()}
            ),
        }
    )


def index_config_fingerprint(config: PipelineConfig) -> str:
    identity = {
        "schema_version": config.schema_version,
        "chunking": config.chunking.model_dump(mode="json"),
        "embedding": config.embedding.model_dump(mode="json"),
        "metric": config.retrieval.metric,
    }
    return hash_canonical(identity)
```

Implement `hash_canonical` as SHA-256 over UTF-8 JSON using `sort_keys=True`, `ensure_ascii=False`, and separators `(",", ":")`. Define `UserInputError(Exception)` and `ExecutionError(Exception)` in `errors.py`. Set `__version__ = "0.1.0"` in `__init__.py`.

- [ ] **Step 5: Run package checks**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_config.py -v
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
```

Expected: all commands pass.

- [ ] **Step 6: Commit the foundation**

```bash
git add pyproject.toml .gitignore README.md src tests/unit/test_config.py examples/prototype
git commit -m "build: establish rag-regress package"
```

### Task 2: Load, Normalize, and Fingerprint a Corpus Safely

**Files:**
- Create: `src/rag_regress/models.py`
- Create: `src/rag_regress/corpus.py`
- Create: `tests/unit/test_corpus.py`

**Interfaces:**
- Consumes: `PipelineConfig.corpus` from Task 1.
- Produces: `Document`, `CorpusSnapshot`, `normalize_text(text: str) -> str`, and `load_corpus(config: CorpusConfig) -> CorpusSnapshot`.

- [ ] **Step 1: Write corpus behavior and boundary tests**

```python
from pathlib import Path

import pytest

from rag_regress.config import CorpusConfig
from rag_regress.corpus import load_corpus, normalize_text
from rag_regress.errors import UserInputError


def test_corpus_is_sorted_normalized_and_stable(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("Beta  \r\ntext\r\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Alpha\n\n", encoding="utf-8")
    config = CorpusConfig(path=tmp_path, include=["**/*.txt", "**/*.md"])

    first = load_corpus(config)
    second = load_corpus(config)

    assert [doc.relative_path for doc in first.documents] == ["a.txt", "b.md"]
    assert first.documents[1].content == "Beta\ntext"
    assert first.fingerprint == second.fingerprint
    assert normalize_text("a  \r\n\r\n") == "a"


def test_empty_corpus_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UserInputError, match="No supported documents"):
        load_corpus(CorpusConfig(path=tmp_path, include=["**/*.md"]))


def test_symlink_escaping_corpus_is_rejected(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("private", encoding="utf-8")
    (corpus / "escape.md").symlink_to(outside)

    with pytest.raises(UserInputError, match="outside corpus root"):
        load_corpus(CorpusConfig(path=corpus, include=["**/*.md"]))
```

- [ ] **Step 2: Run the tests and confirm red**

Run: `.venv/bin/python -m pytest tests/unit/test_corpus.py -v`

Expected: FAIL because `rag_regress.corpus` and its models do not exist.

- [ ] **Step 3: Implement immutable document models and safe loading**

Define:

```python
class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    relative_path: str
    title: str
    content: str
    checksum: str


class CorpusSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    documents: tuple[Document, ...]
    fingerprint: str
```

Normalize CRLF/CR to LF, trim trailing horizontal whitespace per line, then trim leading/trailing blank space from the entire document. Discover glob matches in sorted corpus-relative order, deduplicate paths matched by multiple patterns, reject unsupported suffixes and paths resolving outside the root, and reject an empty result. Hash canonical JSON with SHA-256 for document IDs and the corpus manifest.

- [ ] **Step 4: Run focused and global checks**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_corpus.py -v
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit corpus loading**

```bash
git add src/rag_regress/models.py src/rag_regress/corpus.py tests/unit/test_corpus.py
git commit -m "feat: add deterministic corpus loading"
```

### Task 3: Create Deterministic Word Chunks

**Files:**
- Modify: `src/rag_regress/models.py`
- Create: `src/rag_regress/chunking.py`
- Create: `tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `CorpusSnapshot`, `ChunkingConfig`.
- Produces: `Chunk` and `chunk_corpus(corpus: CorpusSnapshot, config: ChunkingConfig) -> tuple[Chunk, ...]`.

- [ ] **Step 1: Write chunk-boundary and identity tests**

```python
from rag_regress.chunking import chunk_corpus
from rag_regress.config import ChunkingConfig
from rag_regress.models import CorpusSnapshot, Document


def _corpus(content: str) -> CorpusSnapshot:
    document = Document(
        id="doc-1",
        relative_path="guide.md",
        title="guide",
        content=content,
        checksum="sum-1",
    )
    return CorpusSnapshot(documents=(document,), fingerprint="corpus-1")


def test_word_chunks_overlap_and_preserve_boundaries() -> None:
    chunks = chunk_corpus(
        _corpus("one two three four five six seven"),
        ChunkingConfig(strategy="words", size=4, overlap=1),
    )

    assert [(chunk.text, chunk.start_word, chunk.end_word) for chunk in chunks] == [
        ("one two three four", 0, 4),
        ("four five six seven", 3, 7),
        ("seven", 6, 7),
    ]


def test_chunk_ids_are_stable_and_change_with_configuration() -> None:
    corpus = _corpus("one two three four five")
    first = chunk_corpus(corpus, ChunkingConfig(strategy="words", size=3, overlap=1))
    second = chunk_corpus(corpus, ChunkingConfig(strategy="words", size=3, overlap=1))
    changed = chunk_corpus(corpus, ChunkingConfig(strategy="words", size=4, overlap=1))

    assert [item.id for item in first] == [item.id for item in second]
    assert [item.id for item in first] != [item.id for item in changed]
```

- [ ] **Step 2: Run the test and confirm red**

Run: `.venv/bin/python -m pytest tests/unit/test_chunking.py -v`

Expected: FAIL because `Chunk` and `chunk_corpus` are missing.

- [ ] **Step 3: Implement the chunk model and function**

```python
class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    document_id: str
    document_path: str
    sequence: int
    text: str
    start_word: int
    end_word: int
```

Use `step = size - overlap`, slice `content.split()`, and hash document ID, sequence, boundaries, and `config.model_dump(mode="json")` as canonical JSON. Skip normalized documents containing no words and raise `UserInputError` if the complete corpus produces no chunks.

```python
def chunk_corpus(
    corpus: CorpusSnapshot, config: ChunkingConfig
) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    step = config.size - config.overlap
    for document in corpus.documents:
        words = document.content.split()
        for sequence, start in enumerate(range(0, len(words), step)):
            end = min(start + config.size, len(words))
            text = " ".join(words[start:end])
            if not text:
                continue
            chunks.append(make_chunk(document, sequence, start, end, text, config))
    if not chunks:
        raise UserInputError("Corpus produced no non-empty chunks")
    return tuple(chunks)
```

- [ ] **Step 4: Run checks**

Run: `.venv/bin/python -m pytest tests/unit/test_chunking.py -v && .venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit chunking**

```bash
git add src/rag_regress/models.py src/rag_regress/chunking.py tests/unit/test_chunking.py
git commit -m "feat: add deterministic word chunking"
```

### Task 4: Add the Embedder Boundary and Vector Validation

**Files:**
- Create: `src/rag_regress/embeddings.py`
- Create: `tests/helpers.py`
- Create: `tests/unit/test_embeddings.py`

**Interfaces:**
- Consumes: `EmbeddingConfig`, document or query strings.
- Produces: `Embedder` protocol, `EmbeddingMetadata`, `SentenceTransformerEmbedder`, `build_embedder(config: EmbeddingConfig) -> Embedder`, and `normalize_vectors(vectors: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Create a deterministic test embedder and failing tests**

```python
import numpy as np
import pytest

from rag_regress.embeddings import normalize_vectors
from rag_regress.errors import ExecutionError


def test_normalize_vectors_returns_float32_unit_rows() -> None:
    result = normalize_vectors(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64))

    assert result.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.ones(2))


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[float("nan"), 1.0]], dtype=np.float32),
        np.array([[float("inf"), 1.0]], dtype=np.float32),
        np.array([], dtype=np.float32),
    ],
)
def test_invalid_vectors_are_rejected(vectors: np.ndarray) -> None:
    with pytest.raises(ExecutionError, match="embedding"):
        normalize_vectors(vectors)
```

In `tests/helpers.py`, implement `DeterministicTestEmbedder` by hashing tokens into a fixed four-dimensional `float32` bag-of-words vector. It must expose `metadata`, `dimension`, `embed_documents`, and `embed_query`, and call the production `normalize_vectors` function.

- [ ] **Step 2: Run the tests and confirm red**

Run: `.venv/bin/python -m pytest tests/unit/test_embeddings.py -v`

Expected: FAIL because `rag_regress.embeddings` is missing.

- [ ] **Step 3: Implement the protocol and Sentence Transformer adapter**

```python
class Embedder(Protocol):
    @property
    def metadata(self) -> EmbeddingMetadata:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError
```

Implement protocol methods without using literal ellipsis in the concrete class. `SentenceTransformerEmbedder` loads the configured model once, converts output to two-dimensional `float32`, validates row counts and dimensions, and normalizes vectors. Wrap model-loading and encoding errors in `ExecutionError` that includes the model identifier.

- [ ] **Step 4: Add a locally optional model smoke test**

Mark the test with `@pytest.mark.model`, set `HF_HUB_OFFLINE=1`, and skip with the explicit message `model is not cached locally` when Sentence Transformers reports a missing local model. Assert one encoded query is finite and has unit norm when available.

- [ ] **Step 5: Run offline checks**

Run:

```bash
.venv/bin/python -m pytest -m "not model" -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
```

Expected: all default checks pass without network access.

- [ ] **Step 6: Commit embedding support**

```bash
git add src/rag_regress/embeddings.py tests/helpers.py tests/unit/test_embeddings.py
git commit -m "feat: add validated embedding interface"
```

### Task 5: Build and Persist Atomic FAISS Index Bundles

**Files:**
- Modify: `src/rag_regress/models.py`
- Create: `src/rag_regress/artifacts.py`
- Create: `src/rag_regress/index.py`
- Create: `tests/unit/test_artifacts.py`
- Create: `tests/integration/test_index_bundle.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `CorpusSnapshot`, `tuple[Chunk, ...]`, `Embedder`.
- Produces: `IndexBundleManifest`, `FaissIndex`, `bundle_path(config: PipelineConfig, corpus_fingerprint: str) -> Path`, `build_index_bundle(config: PipelineConfig, corpus: CorpusSnapshot, chunks: tuple[Chunk, ...], embedder: Embedder) -> IndexBundleManifest`, `load_index_bundle(config: PipelineConfig, corpus_fingerprint: str) -> LoadedIndexBundle`, `atomic_write_json(path: Path, payload: dict[str, object]) -> None`.

- [ ] **Step 1: Write atomic JSON and index round-trip tests**

```python
from pathlib import Path

import pytest

from rag_regress.artifacts import atomic_write_json


def test_atomic_json_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "run.json"
    target.write_text('{"state":"old"}', encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_json(target, {"state": "new"})

    assert target.read_text(encoding="utf-8") == '{"state":"old"}'
    assert list(tmp_path.glob("*.tmp")) == []
```

The integration test must build a two-document index with `DeterministicTestEmbedder`, save it, reload it, search the same query, and assert identical chunk IDs, ranks, and scores. Build the same content-addressed bundle twice and assert the second call validates and reuses it without rewriting files. For a different target bundle, monkeypatch the final directory rename to fail and assert that no final bundle directory or valid-looking partial output is left behind.

- [ ] **Step 2: Run the tests and confirm red**

Run: `.venv/bin/python -m pytest tests/unit/test_artifacts.py tests/integration/test_index_bundle.py -v`

Expected: FAIL because artifact and index APIs are missing.

- [ ] **Step 3: Implement canonical JSON and atomic writes**

Serialize with `json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)` plus one trailing newline. Write a sibling file with a random `.tmp` suffix, flush and `os.fsync`, then use `Path.replace`. In `finally`, remove only the temporary path created by the current call.

```python
def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
```

- [ ] **Step 4: Implement the FAISS wrapper and bundle identity**

Define `FaissIndex.build(vectors)`, `search(query, top_k)`, `save(path)`, and `load(path, expected_dimension)`. Reject non-two-dimensional arrays, inconsistent dimensions, non-finite values, and empty indexes. Search uses `min(top_k, ntotal)` and never exposes FAISS `-1` indices.

The bundle location is:

```python
config.storage.directory / "indexes" / corpus_fingerprint / index_config_fingerprint(config)
```

Write a temporary sibling directory containing `index.faiss`, `chunks.json`, and `manifest.json`; validate all three by loading them before renaming the temporary directory to its content-addressed destination. If a valid destination already exists, validate and reuse it without writing. If an invalid destination exists, raise `ExecutionError` with its path rather than overwriting it. The manifest records schema version, corpus fingerprint, index-config fingerprint, embedding metadata, dimension, ordered chunk IDs, and relative filenames.

```python
class FaissIndex:
    def __init__(self, raw: faiss.IndexFlatIP, dimension: int) -> None:
        self._raw = raw
        self.dimension = dimension

    @classmethod
    def build(cls, vectors: np.ndarray) -> "FaissIndex":
        validated = validate_index_vectors(vectors)
        raw = faiss.IndexFlatIP(validated.shape[1])
        raw.add(validated)
        return cls(raw, validated.shape[1])

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        validated = validate_query_vector(query, self.dimension)
        limit = min(top_k, self._raw.ntotal)
        scores, positions = self._raw.search(validated.reshape(1, -1), limit)
        return scores[0], positions[0]
```

- [ ] **Step 5: Run checks**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_artifacts.py tests/integration/test_index_bundle.py -v
.venv/bin/python -m pytest -m "not model" -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
```

Expected: all commands pass.

- [ ] **Step 6: Commit index persistence**

```bash
git add src/rag_regress/models.py src/rag_regress/artifacts.py src/rag_regress/index.py tests
git commit -m "feat: persist atomic FAISS index bundles"
```

### Task 6: Validate Evaluation Datasets and Calculate Retrieval Metrics

**Files:**
- Modify: `src/rag_regress/models.py`
- Create: `src/rag_regress/datasets.py`
- Create: `src/rag_regress/metrics.py`
- Create: `tests/unit/test_datasets.py`
- Create: `tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: dataset JSON, `CorpusSnapshot`, ranked `RetrievalResult` values.
- Produces: `EvaluationCase`, `EvaluationDataset`, `RetrievalResult`, `load_evaluation_dataset(path, corpus)`, `recall_at_k`, `hit_at_k`, `reciprocal_rank`, `evidence_hit`, and `nearest_rank_percentile`.

- [ ] **Step 1: Write strict dataset tests**

```python
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
    payload = {
        "schema_version": 1,
        "name": "support",
        "cases": [
            {"id": "same", "question": "One?", "expected_documents": ["guide.md"]},
            {"id": "same", "question": "Two?", "expected_documents": ["guide.md"]},
        ],
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="Duplicate evaluation case ID"):
        load_evaluation_dataset(path, corpus)
```

- [ ] **Step 2: Write exact metric tests**

```python
def test_document_metrics_deduplicate_multiple_chunks() -> None:
    paths = ["wrong.md", "right.md", "right.md", "other.md"]
    expected = {"right.md", "second.md"}

    assert recall_at_k(paths, expected, 4) == 0.5
    assert hit_at_k(paths, expected, 4) == 1.0
    assert reciprocal_rank(paths, expected) == 0.5


def test_evidence_hit_requires_every_expected_substring() -> None:
    chunks = ["Reset links expire", "after   30 minutes."]

    assert evidence_hit(chunks, ["reset links", "after 30 minutes"]) == 1.0
    assert evidence_hit(chunks, ["reset links", "administrator approval"]) == 0.0


def test_nearest_rank_percentile() -> None:
    assert nearest_rank_percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0
```

- [ ] **Step 3: Run the tests and confirm red**

Run: `.venv/bin/python -m pytest tests/unit/test_datasets.py tests/unit/test_metrics.py -v`

Expected: FAIL because dataset and metric APIs are missing.

- [ ] **Step 4: Implement strict versioned dataset models**

Use frozen Pydantic models with unknown keys forbidden, schema version fixed to `1`, non-empty trimmed fields, unique expected-document paths per case, unique case IDs, and `/`-normalized relative paths that cannot contain `..`. Validate all expected paths against `corpus.documents` before returning the dataset. Compute the dataset fingerprint from canonical model JSON.

```python
class EvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_documents: tuple[str, ...] = Field(min_length=1)
    expected_text: tuple[str, ...] = ()


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return hash_canonical(self.model_dump(mode="json"))


def load_evaluation_dataset(path: Path, corpus: CorpusSnapshot) -> EvaluationDataset:
    parsed = parse_dataset_json(path)
    validate_unique_case_ids(parsed.cases)
    validate_expected_paths(parsed.cases, corpus)
    return parsed
```

- [ ] **Step 5: Implement pure metric functions**

Metrics accept ranked document paths rather than chunk IDs. Recall deduplicates paths within top-k, reciprocal rank uses the first relevant path, and evidence matching normalizes whitespace and case across the concatenated top-k chunk text. Return `None` for aggregate evidence hit rate when no case defines expected text.

```python
def recall_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    retrieved = set(ranked[:k])
    return len(retrieved & expected) / len(expected)


def hit_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & expected))


def reciprocal_rank(ranked: Sequence[str], expected: set[str]) -> float:
    for rank, document_path in enumerate(ranked, start=1):
        if document_path in expected:
            return 1.0 / rank
    return 0.0


def evidence_hit(chunks: Sequence[str], expected_text: Sequence[str]) -> float:
    combined = normalize_match_text(" ".join(chunks))
    return float(all(normalize_match_text(item) in combined for item in expected_text))
```

- [ ] **Step 6: Run checks and commit**

Run: `.venv/bin/python -m pytest -m "not model" -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src`

Expected: all commands pass.

```bash
git add src/rag_regress/models.py src/rag_regress/datasets.py src/rag_regress/metrics.py tests/unit
git commit -m "feat: add retrieval evaluation metrics"
```

### Task 7: Execute Reproducible Ingestion and Evaluation Runs

**Files:**
- Modify: `src/rag_regress/models.py`
- Modify: `src/rag_regress/artifacts.py`
- Create: `src/rag_regress/experiments.py`
- Create: `tests/integration/test_experiments.py`

**Interfaces:**
- Consumes: config path, dataset path, output path, optional injected `Embedder`.
- Produces: `ingest(config_path: Path, embedder: Embedder | None = None) -> IndexBundleManifest` and `evaluate(config_path: Path, dataset_path: Path, output_path: Path, embedder: Embedder | None = None) -> RunArtifact`.

- [ ] **Step 1: Write the end-to-end service test**

Create a two-document fixture corpus and two evaluation cases in the test. Use `DeterministicTestEmbedder` and assert:

```python
manifest = ingest(config_path, embedder=test_embedder)
run = evaluate(config_path, dataset_path, output_path, embedder=test_embedder)

assert manifest.chunk_count > 0
assert run.schema_version == 1
assert run.corpus_fingerprint == manifest.corpus_fingerprint
assert run.dataset_fingerprint
assert len(run.cases) == 2
assert run.metrics.recall_at_k == 1.0
assert run.metrics.hit_rate == 1.0
assert output_path.exists()
assert RunArtifact.model_validate_json(output_path.read_text(encoding="utf-8")) == run
```

Also test that evaluation without a bundle raises `UserInputError` containing `rag-regress ingest`, and monkeypatch query embedding to fail on the second case; assert the requested output does not exist and a pre-existing valid output remains unchanged.

- [ ] **Step 2: Run the integration test and confirm red**

Run: `.venv/bin/python -m pytest tests/integration/test_experiments.py -v`

Expected: FAIL because experiment services and run models are missing.

- [ ] **Step 3: Add versioned run-artifact models**

Define frozen, strict `EnvironmentInfo`, `PipelineSnapshot`, `CaseMetrics`, `CaseResult`, `AggregateMetrics`, and `RunArtifact` models with these exact public fields:

```python
class RunArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    tool_version: str
    metric_definition_version: Literal["1"] = "1"
    dataset_name: str
    corpus_fingerprint: str
    dataset_fingerprint: str
    pipeline: PipelineSnapshot
    environment: EnvironmentInfo
    embedding: EmbeddingMetadata
    started_at: datetime
    duration_ms: float
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    metrics: AggregateMetrics
    cases: tuple[CaseResult, ...]
```

`PipelineSnapshot` includes corpus include patterns, chunking, embedding, and retrieval settings but no corpus or storage paths. `EnvironmentInfo` includes Python/platform plus NumPy, FAISS, sentence-transformers, and rag-regress versions. `CaseResult` includes case ID, question, expected paths/text, ranked `RetrievalResult` items, recall, hit, reciprocal rank, optional evidence hit, and `retrieval_ms`. `AggregateMetrics` includes recall-at-k, hit rate, MRR, optional evidence hit rate, p50 retrieval milliseconds, and p95 retrieval milliseconds.

- [ ] **Step 4: Implement ingestion and evaluation services**

`ingest` loads config and corpus, chunks, builds an embedder when one is not injected, and calls `build_index_bundle`.

`evaluate` reloads config and corpus, validates the dataset, calculates the exact bundle path, loads it, verifies embedding metadata, and runs one unmeasured warm-up embedding using the first question. Time each measured query with `time.perf_counter`. Hydrate FAISS positions from the ordered chunk table, apply `relevance_threshold`, calculate case metrics, aggregate arithmetic means and nearest-rank p50/p95, collect environment versions with `importlib.metadata.version`, then atomically write and return the artifact.

The start timestamp is timezone-aware UTC ISO 8601. Artifact identity and compatibility do not include timestamps or latency.

```python
def ingest(
    config_path: Path, embedder: Embedder | None = None
) -> IndexBundleManifest:
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
    config = load_pipeline_config(config_path)
    corpus = load_corpus(config.corpus)
    dataset = load_evaluation_dataset(dataset_path, corpus)
    bundle = load_index_bundle(config, corpus.fingerprint)
    active_embedder = embedder or build_embedder(config.embedding)
    verify_embedding_compatibility(active_embedder.metadata, bundle.manifest.embedding)
    active_embedder.embed_query(dataset.cases[0].question)

    started_at = datetime.now(timezone.utc)
    run_started = time.perf_counter()
    cases = tuple(
        evaluate_case(case, config.retrieval, bundle, active_embedder)
        for case in dataset.cases
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
```

Define the private helpers `verify_embedding_compatibility`, `evaluate_case`, `create_run_artifact`, `pipeline_snapshot`, and `environment_info` in `experiments.py`; unit behavior remains covered through the public services and pure metric functions.

- [ ] **Step 5: Run checks**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_experiments.py -v
.venv/bin/python -m pytest -m "not model" -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
```

Expected: all commands pass.

- [ ] **Step 6: Commit experiment execution**

```bash
git add src/rag_regress/models.py src/rag_regress/artifacts.py src/rag_regress/experiments.py tests/integration/test_experiments.py
git commit -m "feat: run reproducible retrieval evaluations"
```

### Task 8: Compare Runs and Enforce Quality Gates

**Files:**
- Modify: `src/rag_regress/models.py`
- Create: `src/rag_regress/comparison.py`
- Create: `src/rag_regress/quality_gates.py`
- Create: `tests/unit/test_comparison.py`
- Create: `tests/unit/test_quality_gates.py`

**Interfaces:**
- Consumes: baseline and candidate `RunArtifact`, quality-gate YAML.
- Produces: `ComparisonReport`, `compare_runs(baseline, candidate)`, `QualityGateConfig`, `load_quality_gates(path)`, and `check_quality_gates(candidate, gates, baseline=None) -> GateReport`.

- [ ] **Step 1: Write comparison tests, including all compatibility mismatches**

```python
def test_incompatible_runs_list_all_mismatches() -> None:
    baseline = make_run(corpus_fingerprint="corpus-a", dataset_fingerprint="data-a")
    candidate = make_run(corpus_fingerprint="corpus-b", dataset_fingerprint="data-b")

    with pytest.raises(UserInputError) as captured:
        compare_runs(baseline, candidate)

    message = str(captured.value)
    assert "corpus_fingerprint" in message
    assert "dataset_fingerprint" in message


def test_cases_are_classified_by_reciprocal_rank_delta() -> None:
    baseline = make_run_with_reciprocal_ranks({"better": 0.5, "worse": 1.0, "same": 0.0})
    candidate = make_run_with_reciprocal_ranks({"better": 1.0, "worse": 0.5, "same": 0.0})

    report = compare_runs(baseline, candidate)

    assert report.improved_case_ids == ("better",)
    assert report.regressed_case_ids == ("worse",)
    assert report.unchanged_case_ids == ("same",)
```

- [ ] **Step 2: Write gate tests**

```python
def test_absolute_and_relative_gates_report_every_failure() -> None:
    gates = QualityGateConfig.model_validate(
        {
            "schema_version": 1,
            "minimum": {"recall_at_k": 0.9, "mrr": 0.8},
            "maximum": {"p95_retrieval_ms": 100, "regressed_cases": 0},
            "allowed_drop": {"recall_at_k": 0.01, "mrr": 0.01},
        }
    )
    baseline = make_run(recall=0.95, mrr=0.9, p95=80)
    candidate = make_run(recall=0.80, mrr=0.70, p95=140)

    report = check_quality_gates(candidate, gates, baseline)

    assert report.passed is False
    assert {failure.gate for failure in report.failures} == {
        "minimum.recall_at_k",
        "minimum.mrr",
        "maximum.p95_retrieval_ms",
        "allowed_drop.recall_at_k",
        "allowed_drop.mrr",
    }


def test_relative_gate_without_baseline_is_invalid() -> None:
    gates = QualityGateConfig.model_validate(
        {"schema_version": 1, "allowed_drop": {"mrr": 0.01}}
    )

    with pytest.raises(UserInputError, match="baseline"):
        check_quality_gates(make_run(), gates, baseline=None)
```

- [ ] **Step 3: Run tests and confirm red**

Run: `.venv/bin/python -m pytest tests/unit/test_comparison.py tests/unit/test_quality_gates.py -v`

Expected: FAIL because comparison and gate APIs are missing.

- [ ] **Step 4: Implement compatibility, deltas, and strict gates**

Compatibility fields are artifact schema version, corpus fingerprint, dataset fingerprint, ordered case IDs, and metric-definition version. Gather all mismatches before raising. Compare cases by ID and sort each classification list by case ID for stable output. Aggregate deltas are `candidate - baseline`.

Gate models forbid unknown fields and reject values outside `[0, 1]` for quality metrics, negative latency/regression limits, and an otherwise empty gate file. Evaluate all applicable gates and return every failure with gate name, expected boundary, actual value, and optional baseline value.

```python
def compare_runs(baseline: RunArtifact, candidate: RunArtifact) -> ComparisonReport:
    mismatches = compatibility_mismatches(baseline, candidate)
    if mismatches:
        raise UserInputError("Incompatible runs: " + ", ".join(mismatches))
    baseline_cases = {case.id: case for case in baseline.cases}
    deltas = tuple(
        compare_case(baseline_cases[case.id], case)
        for case in sorted(candidate.cases, key=lambda item: item.id)
    )
    return build_comparison_report(baseline.metrics, candidate.metrics, deltas)


def check_quality_gates(
    candidate: RunArtifact,
    gates: QualityGateConfig,
    baseline: RunArtifact | None = None,
) -> GateReport:
    if gates.requires_baseline and baseline is None:
        raise UserInputError("A baseline run is required by relative quality gates")
    comparison = compare_runs(baseline, candidate) if baseline is not None else None
    failures = evaluate_absolute_gates(candidate, gates)
    failures.extend(evaluate_relative_gates(candidate, baseline, comparison, gates))
    return GateReport(passed=not failures, failures=tuple(failures))
```

Define `compatibility_mismatches`, `compare_case`, `build_comparison_report`, `evaluate_absolute_gates`, and `evaluate_relative_gates` as private pure helpers in their owning modules.

- [ ] **Step 5: Run checks and commit**

Run: `.venv/bin/python -m pytest -m "not model" -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src`

Expected: all commands pass.

```bash
git add src/rag_regress/models.py src/rag_regress/comparison.py src/rag_regress/quality_gates.py tests/unit
git commit -m "feat: compare runs and enforce quality gates"
```

### Task 9: Expose the Workflow Through a Stable CLI

**Files:**
- Create: `src/rag_regress/cli.py`
- Create: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: application services and artifact/gate loaders from Tasks 5–8.
- Produces: Typer `app` with `ingest`, `evaluate`, `compare`, and `check` commands.

- [ ] **Step 1: Write CLI success and exit-code tests**

Use `typer.testing.CliRunner` and monkeypatch service functions to avoid models. Assert these behaviors:

```python
def test_check_returns_one_when_gates_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_check", lambda *args, **kwargs: failed_gate_report())

    result = runner.invoke(
        cli.app,
        ["check", "candidate.json", "--baseline", "baseline.json", "--gates", "gates.yaml"],
    )

    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_user_input_error_returns_two(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise UserInputError("bad dataset")

    monkeypatch.setattr(cli, "run_evaluate", fail)
    result = runner.invoke(
        cli.app,
        ["evaluate", "eval.json", "--config", "config.yaml", "--output", "run.json"],
    )

    assert result.exit_code == 2
    assert "bad dataset" in result.stdout
    assert "Traceback" not in result.stdout
```

Add tests that `--debug` exposes a traceback for an unexpected exception, `compare` prints aggregate deltas and case IDs, and all command help text contains the arguments documented in the spec.

- [ ] **Step 2: Run the CLI tests and confirm red**

Run: `.venv/bin/python -m pytest tests/integration/test_cli.py -v`

Expected: FAIL because `rag_regress.cli` is missing.

- [ ] **Step 3: Implement commands and output formatting**

Use a Typer callback to store a global `--debug` boolean. Each command calls a thin `run_ingest`, `run_evaluate`, `run_compare`, or `run_check` function so tests can replace the boundary. Catch `UserInputError` and `ExecutionError`, print `Error: <message>` to stderr, and raise `typer.Exit(2)`. For unexpected errors, re-raise under `--debug`; otherwise print an actionable summary and exit `2`.

`check` prints each gate failure and exits `1` only after all failures are displayed. `compare` prints metric baseline, candidate, and delta values followed by improved/regressed/unchanged case IDs. Do not use ANSI color when output is not a terminal.

```python
app = typer.Typer(no_args_is_help=True)


@app.callback()
def main(ctx: typer.Context, debug: bool = typer.Option(False, "--debug")) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@app.command("ingest")
def ingest_command(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    execute(ctx, lambda: run_ingest(config))


def execute(ctx: typer.Context, operation: Callable[[], object]) -> object:
    try:
        return operation()
    except (UserInputError, ExecutionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:
        if bool(ctx.obj.get("debug")):
            raise
        typer.echo(f"Error: unexpected failure: {exc}", err=True)
        raise typer.Exit(2) from exc
```

Name the remaining Typer commands explicitly with `@app.command("evaluate")`, `@app.command("compare")`, and `@app.command("check")`; the Python function names must not determine the public command names.

- [ ] **Step 4: Run the installed console entry point**

Run:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/rag-regress --help
.venv/bin/rag-regress evaluate --help
```

Expected: help lists `ingest`, `evaluate`, `compare`, and `check`; evaluate help lists dataset, config, and output arguments.

- [ ] **Step 5: Run checks and commit**

Run: `.venv/bin/python -m pytest -m "not model" -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m mypy src`

Expected: all commands pass.

```bash
git add src/rag_regress/cli.py tests/integration/test_cli.py
git commit -m "feat: expose rag regression CLI"
```

### Task 10: Add the Demonstration, Documentation, CI, and Release Verification

**Files:**
- Create: `examples/corpus/account-guide.md`
- Create: `examples/corpus/billing-guide.md`
- Create: `examples/evals/product-support-v1.json`
- Create: `examples/configs/baseline.yaml`
- Create: `examples/configs/candidate.yaml`
- Create: `examples/quality-gates.yaml`
- Modify: `README.md`
- Create: `.github/workflows/ci.yml`
- Create: `tests/integration/test_example_files.py`
- Modify: `docs/design.md`

**Interfaces:**
- Consumes: the complete CLI.
- Produces: a five-minute public example, a reproducible CI workflow, and verified release artifacts.

- [ ] **Step 1: Write an example-file contract test**

```python
from pathlib import Path

from rag_regress.config import load_pipeline_config
from rag_regress.corpus import load_corpus
from rag_regress.datasets import load_evaluation_dataset
from rag_regress.quality_gates import load_quality_gates


def test_public_example_files_are_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_pipeline_config(root / "examples/configs/baseline.yaml")
    corpus = load_corpus(config.corpus)
    dataset = load_evaluation_dataset(
        root / "examples/evals/product-support-v1.json", corpus
    )
    gates = load_quality_gates(root / "examples/quality-gates.yaml")

    assert len(corpus.documents) == 2
    assert len(dataset.cases) >= 6
    assert gates.minimum is not None
```

- [ ] **Step 2: Run the test and confirm red**

Run: `.venv/bin/python -m pytest tests/integration/test_example_files.py -v`

Expected: FAIL because the public example files do not exist.

- [ ] **Step 3: Create a meaningful support corpus and labelled dataset**

Write account and billing guides containing distinct procedures, limitations, time windows, and escalation paths. Add at least six questions: two straightforward lookups, two paraphrases, one question with two expected documents, and one expected-text case. Baseline uses 80-word chunks with 20-word overlap; candidate uses 40-word chunks with 10-word overlap. Both use `sentence-transformers/all-MiniLM-L6-v2`, cosine, and top-k `5`.

Set example gates to minimum Recall@k `0.80`, MRR `0.70`, hit rate `0.80`, maximum p95 retrieval latency `500`, maximum regressed cases `1`, and allowed Recall@k/MRR drop `0.05`.

- [ ] **Step 4: Write the README around the regression problem**

The README must include:

- the one-sentence product claim from the spec;
- a comparison explaining that this is narrower than Ragas and Phoenix, without claiming invention;
- installation in a fresh virtual environment;
- the exact ingest, evaluate, compare, and check commands;
- a checked-in representative comparison-output block;
- metric definitions and dataset-labelling guidance;
- privacy and offline behavior;
- limitations of deterministic ground truth and latency comparisons;
- architecture and project-layout sections;
- development commands and contribution expectations.

Do not claim that `rag-regress` is the first, unique, production-ready, or superior RAG evaluation tool.

- [ ] **Step 5: Add CI for both supported Python versions**

Create `.github/workflows/ci.yml` triggered on pushes and pull requests. Use `actions/checkout`, `actions/setup-python` with a matrix of `3.11` and `3.12`, install `.[dev]`, then run:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  verify:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install ".[dev]"
      - run: ruff check src tests
      - run: mypy src
      - run: pytest -m "not model" -q
      - run: python -m build
      - uses: actions/upload-artifact@v4
        if: matrix.python-version == '3.12'
        with:
          name: dist
          path: dist/
```

The shell sequence in the workflow is:

```bash
ruff check src tests
mypy src
pytest -m "not model" -q
python -m build
```

Upload `dist/` as an artifact only on Python 3.12. Do not download a Sentence Transformer model in CI.

- [ ] **Step 6: Update the spec status and run full verification**

Change the design spec status from `Approved` to `Implemented` only after all checks below pass:

```bash
.venv/bin/python -m pytest -m "not model" -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
.venv/bin/python -m build
.venv/bin/rag-regress --help
git diff --check
git status --short
```

Expected: tests, lint, types, build, help, and whitespace checks pass. Before the final commit, `git status --short` lists only the intended Task 10 files.

- [ ] **Step 7: Commit the completed v0.1**

```bash
git add README.md .github examples tests/integration/test_example_files.py docs/design.md
git commit -m "docs: complete rag-regress v0.1 demonstration"
```

## Final Review Checklist

- [ ] Run every command in Task 10 Step 6 and capture its output.
- [ ] Inspect `git log --oneline` and confirm each task has one focused commit.
- [ ] Confirm `.venv/`, `.rag-regress/`, `runs/`, model files, and build output are untracked or ignored.
- [ ] Run the example with a locally cached model; if the model is absent, report the smoke test as skipped rather than attempting a network download without approval.
- [ ] Inspect one run artifact and verify it contains no absolute corpus paths or secrets.
- [ ] Confirm the README describes overlap with existing tools honestly and demonstrates the narrower regression-testing focus.
