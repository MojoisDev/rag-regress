# rag-regress

Know when a change to chunking, embeddings, indexing, or retrieval makes a RAG system worse before shipping it.

`rag-regress` is a local-first command-line tool for reproducible retrieval
experiments, run-to-run diffs, and CI quality gates. It evaluates retrieval,
not generated answers: configure a corpus and word chunking, label questions
with the documents that should be retrieved, and compare a candidate run with
a baseline.

## What it is—and is not

This project is deliberately narrower than Ragas and Arize Phoenix. Those
projects support broader LLM/RAG evaluation and observability workflows;
`rag-regress` concentrates on deterministic, local retrieval regression tests
for a labelled corpus. It is not a general RAG framework, a chat application,
an answer-generation system, or a replacement for broader tracing and
evaluation tooling.

## Five-minute example

The checked-in example uses two support guides, six labelled questions, and
two otherwise identical pipelines. It needs Python 3.11 or 3.12 and a locally
available `sentence-transformers/all-MiniLM-L6-v2` model. The commands do not
need an API key or a hosted application service. If the model is not already
cached, model resolution follows the configured Sentence Transformers and
Hugging Face settings; pre-download it according to your organization's
approved dependency/model process, or use their offline settings to prohibit
network access.

### Model revision pinning

Every pipeline requires `embedding.revision`, a full lowercase 40-character
Hugging Face commit SHA, passed to the
[Sentence Transformers loader](https://sbert.net/docs/package_reference/sentence_transformer/model.html).
The examples pin `all-MiniLM-L6-v2` to
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Model IDs resolve through the
Hugging Face cache as usual; local model directory paths are unsupported because
Sentence Transformers ignores revisions for those paths.

To migrate an existing config, add the SHA of the model version you intend to
use, then rerun `ingest` and `evaluate`. The revision is included in index
identity and embedding metadata, so changing it selects a separate bundle.
Existing bundles are preserved; artifacts without revision metadata must be
regenerated before comparison or quality checks.

### Run the example

Create a fresh environment and install the package:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ".[dev]"
```

Build each index, evaluate the same labelled dataset, compare the artifacts,
then enforce the candidate gates:

```bash
.venv/bin/rag-regress ingest --config examples/configs/baseline.yaml
.venv/bin/rag-regress evaluate examples/evals/product-support-v1.json \
  --config examples/configs/baseline.yaml --output runs/baseline.json
.venv/bin/rag-regress ingest --config examples/configs/candidate.yaml
.venv/bin/rag-regress evaluate examples/evals/product-support-v1.json \
  --config examples/configs/candidate.yaml --output runs/candidate.json
.venv/bin/rag-regress compare runs/baseline.json runs/candidate.json
.venv/bin/rag-regress check runs/candidate.json \
  --baseline runs/baseline.json --gates examples/quality-gates.yaml
```

The comparison prints aggregate baseline/candidate/delta values and the case
classification. This is a representative checked-in output shape; numerical
latency and rankings can vary with the local model/runtime:

```text
recall_at_k: baseline=1.0 candidate=1.0 delta=0.0
hit_rate: baseline=1.0 candidate=1.0 delta=0.0
mrr: baseline=1.0 candidate=1.0 delta=0.0
evidence_hit_rate: baseline=1.0 candidate=1.0 delta=0.0
p50_retrieval_ms: baseline=3.2 candidate=3.6 delta=0.4
p95_retrieval_ms: baseline=5.1 candidate=5.8 delta=0.7
Improved: (none)
Regressed: (none)
Unchanged: change-email-without-old-address, download-invoice, locked-user-needs-invoice, refund-arrival-time, repeat-lockout-escalation, reset-password
```

`check` exits `0` when every gate passes, `1` when a quality gate fails, and
`2` for invalid input or execution errors. The sample gates require Recall@k
of at least 0.80, MRR of at least 0.70, hit rate of at least 0.80, p95
retrieval latency no greater than 500 ms, at most one regressed case, and no
more than a 0.05 drop in Recall@k or MRR.

## Compare chunk sizes

Use `sweep` to build and evaluate several word chunk sizes on the same corpus
and labelled dataset:

```bash
.venv/bin/rag-regress sweep examples/evals/product-support-v1.json \
  --config examples/configs/baseline.yaml \
  --size 40 --size 80 --size 160 --output runs/chunk-sweep
```

The first size is the baseline. Provide at least two distinct positive sizes,
each greater than the config's overlap. Overlap, model revision, top-k, and score
threshold stay fixed; the model loads once and each variant gets a warm-up query.
The original config is unchanged, and validated index bundles are reused.

The command prints quality and retrieval latency for each size and creates:

- `size-40.json`, `size-80.json`, `size-160.json`: ordinary run artifacts usable
  with `compare` and `check`.
- `summary.json`: schema version, baseline size, corpus/dataset fingerprints,
  and a `runs` array containing size, chunk count, relative artifact filename,
  metrics, and the full comparison with the baseline (`null` for the baseline).

Choose a new output directory for each experiment. It is published only after
every variant succeeds; a failure removes temporary output while retaining any
completed index bundles. The sweep does not automatically select a winner:
review quality, evidence coverage, and latency together, and repeat measurements
before drawing performance conclusions.

This workflow is informed by the
[chunk-size experiments in RAG Techniques](https://github.com/NirDiamant/RAG_Techniques#advanced-techniques).

## Metrics and labels

- Recall@k is the fraction of a case's expected document paths represented in
  its first `k` retrieved chunks.
- Hit@k is 1 when at least one expected document appears in those results;
  aggregate hit rate is its mean across cases.
- MRR is the mean reciprocal rank of the first relevant retrieved chunk.
- Evidence hit rate applies only to cases with `expected_text`; all specified
  substrings must appear in the normalized retrieved text.

Label expected documents relative to the configured corpus root, use `/`
separators, and include every document needed to answer a multi-document
question. Keep labels independent of a particular chunk boundary or score,
cover both ordinary and high-risk support questions, and review labels when
the corpus meaning changes. `expected_text` is useful for a small number of
important snippets, not as a substitute for document relevance.

## Privacy, offline behavior, and limitations

The tool has no telemetry, uploads, hosted API, web UI, or answer generation.
It reads local `.txt` and `.md` corpus files and stores local FAISS indexes and
JSON run artifacts. Artifacts retain retrieved chunk text, so treat them as
potentially sensitive project data. The default test suite uses a deterministic
test embedder and runs without network access or a downloaded model; the
optional model smoke test runs only when the model is already available
locally.

Ground truth is deterministic but not complete: document labels can miss a
valid answer, become stale, or encode a team's judgement. Latency is a useful
local regression signal, not a portable performance benchmark; it varies with
hardware, cache state, package versions, and concurrent load. Retrieval
metrics do not prove generated answers are correct.

## Architecture

Typed modules keep input loading, corpus normalization, word chunking,
Sentence Transformer embeddings, FAISS `IndexFlatIP` retrieval, metrics,
comparison, gates, and JSON artifacts separate. Vectors are normalized so
inner-product search is cosine-compatible. The CLI coordinates those services
and presents their output; it does not contain retrieval or metric logic.

## Project layout

```text
src/rag_regress/       typed retrieval, evaluation, comparison, and CLI modules
tests/unit/            focused offline behavior tests
tests/integration/     bundle, workflow, CLI, and example-contract tests
examples/corpus/       small UTF-8 Markdown/text support corpus
examples/evals/        labelled evaluation datasets
examples/configs/      baseline and candidate pipeline configurations
examples/quality-gates.yaml
.github/workflows/     Python 3.11/3.12 offline verification
docs/                  public design and implementation plan
```

## Development and contributions

Run the offline checks before proposing a change:

```bash
.venv/bin/python -m pytest -m "not model" -q
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
.venv/bin/python -m build
.venv/bin/rag-regress --help
```

Contributions should preserve the v0.1 boundaries: UTF-8 `.txt`/`.md` input,
word chunking, normalized Sentence Transformer vectors, and FAISS
`IndexFlatIP`. Add tests for behavior changes, keep default tests offline,
reject unknown input fields, and document any new user-visible command or
artifact behavior. Do not add telemetry, uploads, hosted APIs, generation,
web UI, PDF ingestion, or framework adapters without an approved scope change.
