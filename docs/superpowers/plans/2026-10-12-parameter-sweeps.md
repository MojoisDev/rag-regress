# Broader Parameter Sweeps Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task, with a final independent code review. Steps use checkbox syntax for tracking.

**Goal:** Compare chunk size, overlap, and retrieval top_k in one reproducible experiment.

**Architecture:** Extend the existing sweep service and CLI. Generate an ordered Cartesian grid, group evaluations by size/overlap so each index is built and loaded once, and publish complete output through the existing temporary-directory pattern. Ordinary run artifacts and comparison/report/gate services remain the output contract.

**Tech Stack:** Python 3.11/3.12, existing Pydantic, Typer, Sentence Transformers, NumPy, FAISS, pytest, Ruff, mypy. No new dependencies.

**Design brief:** This document records the proposed bounded design for user review. The purpose is to measure retrieval quality and latency across parameter combinations on the same corpus, dataset, pinned model, device, and latency settings.

## Global Constraints

- Support both words and tokens; retain current tokenizer restrictions and model budgets.
- Preserve positional callers of `sweep(config_path, dataset_path, sizes, output_directory)` and its tuple-of-RunArtifact return value.
- Add keyword-only `overlaps: tuple[int, ...] | None = None` and `top_ks: tuple[int, ...] | None = None`.
- Omitted new axes use the base config values. Continue requiring CLI `--size`.
- A size-only sweep still requires two distinct sizes and retains current errors, filenames, and summary version 1.
- Extended sweeps allow one size when another axis produces multiple variants. Require at least two distinct variants overall.
- Reject duplicate axis values, empty explicit axes, nonpositive sizes/top_k, negative overlaps, and any overlap >= any selected size. Reject the entire invalid grid; do not skip combinations.
- Validate the full grid before loading the model or writing anything. Validate all token sizes against the loaded model before creating indexes or output directories.
- Model revision, strategy, device, score threshold, corpus/dataset, and latency settings stay fixed within one sweep.
- Deterministic order: input sizes, then input overlaps, then input top_ks. The first combination is the baseline.
- No automatic winner selection, new retrievers, model sweeps, parallel execution, or new config schema.

## Files and Interfaces

| File | Responsibility |
| --- | --- |
| `src/rag_regress/experiments.py` | Grid validation, grouped index reuse, evaluations, publication |
| `src/rag_regress/cli.py` | Repeatable flags, clear labels, run-count output |
| `tests/integration/test_sweep.py` | Grid, reuse, compatibility, failures, CLI |
| `tests/integration/test_token_experiments.py` | Token grid and model-limit preflight |
| `tests/integration/test_cli.py` | Plain and ANSI help |
| `README.md`, `docs/design.md` | Commands, units, ordering, output schema, runtime growth |

No changes to package exports or PipelineConfig fields are required. Reuse index_config_fingerprint, build_index_bundle, load_index_bundle, compare_runs, atomic_write_json, and existing test fixtures.

## Review Focus

1. Same size with different overlap/top_k must not overwrite another artifact.
2. An invalid combination late in the grid must fail before earlier combinations write indexes.
3. Top_k changes must not re-embed documents or change the index fingerprint.
4. Old size-only callers and summary consumers must keep their existing output contract.
5. Late evaluation/publication failure must preserve existing output and remove temporary files.

## Task 1: Validated Grid and Grouped Evaluation

- [ ] Add a failing integration test to `test_sweep.py` using the existing fixture writer and deterministic embedder:

```python
def test_grid_reuses_indexes(tmp_path, monkeypatch):
    config, dataset = _write_fixture_inputs(tmp_path)
    class CountingEmbedder(DeterministicTestEmbedder):
        document_calls = 0
        def embed_documents(self, texts):
            self.document_calls += 1
            return super().embed_documents(texts)
        def embed_query(self, text):
            return DeterministicTestEmbedder.embed_documents(self, [text])
    embedder = CountingEmbedder()
    loads = []
    def build(config):
        loads.append(config)
        return embedder
    monkeypatch.setattr(experiments, "build_embedder", build)
    runs = experiments.sweep(
        config, dataset, (12, 6), tmp_path / "grid",
        overlaps=(0, 1), top_ks=(2, 1),
    )
    assert [(r.pipeline.chunking.size, r.pipeline.chunking.overlap,
             r.pipeline.retrieval.top_k) for r in runs] == [
        (12, 0, 2), (12, 0, 1), (12, 1, 2), (12, 1, 1),
        (6, 0, 2), (6, 0, 1), (6, 1, 2), (6, 1, 1),
    ]
    assert len(loads) == 1
    assert embedder.document_calls == 4
```

- [ ] Run `.venv/bin/python -m pytest tests/integration/test_sweep.py -q`; new test must fail because new keyword arguments are unsupported.
- [ ] Extend the service signature with the two keyword-only axes. Validate values and grid length before loading inputs/model. Construct ChunkingConfig and RetrievalConfig variants through validation, not unchecked model_copy values.
- [ ] Use nested size/overlap/top_k loops. Chunk, build/reuse, and load once per size/overlap pair; only evaluate inside the top_k loop. Reuse the current warmup/repetition implementation for every evaluation.
- [ ] Add parametrized rejection tests for overlaps `()`, `(-1,)`, `(0, 0)`, `(6,)` with sizes `(12, 6)`; top_ks `()`, `(0,)`, `(-1,)`, `(2, 2)`; and a one-variant extended grid. Patch build_embedder to raise AssertionError if called and assert no state/output directory exists.
- [ ] Test a single-size/multiple-top_k sweep, repeated sweep cache reuse, and distinct configurations sharing the same index fingerprint only when top_k changes.
- [ ] In token tests, use the existing TokenEmbedder limit of 6 content tokens: an extended grid containing sizes `(4, 7)` must fail before embedding documents or creating index/output directories.
- [ ] Run focused word/token integration suites and all existing size-only sweep tests.

## Task 2: Output Contract and Failure Safety

Extended output uses summary schema version 2. Legacy size-only output stays version 1.

```json
{
  "schema_version": 2,
  "baseline": {"size": 120, "overlap": 20, "top_k": 3},
  "corpus_fingerprint": "recorded corpus fingerprint",
  "dataset_fingerprint": "recorded dataset fingerprint",
  "runs": [
    {
      "size": 120,
      "overlap": 20,
      "top_k": 3,
      "chunk_count": 4,
      "artifact": "size-120-overlap-20-top-k-3.json",
      "metrics": {},
      "comparison": null
    }
  ]
}
```

The schema illustration omits metric keys for readability; actual metrics are `run.metrics.model_dump(mode="json")`. Every non-baseline comparison is `compare_runs(runs[0], run).model_dump(mode="json")`.

- [ ] Before changing output logic, add failing assertions to the grid test for version 2, baseline tuple, eight distinct filenames, JSON roundtrips, and exact compare_runs results.
- [ ] Generate extended filenames with all three values. Preserve `size-N.json` and version-1 fields when both new axes are omitted.
- [ ] Preserve atomic temporary-directory publication and cleanup. Completed index bundles may remain after execution failures, as in current sweeps.
- [ ] Test a late query failure across top_k variants, existing output directory/symlink refusal, untouched input config bytes, and temporary-directory cleanup.
- [ ] Verify every extended run works with existing compare/report/check services; use permissive gates so this tests artifact compatibility rather than fixture performance.
- [ ] Run focused tests, then commit service/output changes with their tests.

## Task 3: CLI and Documentation

- [ ] Write failing CliRunner tests for repeatable `--overlap` and `--top-k`, the eight-run example, invalid-grid exit code 2, and labels containing all parameters.
- [ ] Add optional repeatable integer flags. Use explicit uppercase metavars `SIZE`, `OVERLAP`, `TOP_K`; normalize ANSI help text with the existing Rich test pattern.
- [ ] Print the planned variant count before evaluation and the parameter tuple plus existing quality/latency metrics for each result.
- [ ] Document the following command, baseline `(120,20,3)`, eight evaluations, and four distinct index configurations:

```bash
rag-regress sweep examples/evals/product-support-v1.json \
  --config examples/configs/tokens.yaml \
  --size 120 --size 200 \
  --overlap 20 --overlap 40 \
  --top-k 3 --top-k 5 \
  --output runs/parameter-grid
```

- [ ] Explain that larger top_k changes retrieval context and metrics, latency depends on warmup/repetitions and hardware, and combinations grow multiplicatively. Document omitted-axis defaults, invalid-grid rejection, and both summary versions.
- [ ] Run CLI tests in plain and forced-color environments, then commit CLI/docs changes.

## Task 4: Final Verification and Delivery

- [ ] Run full suites on Python 3.11 CPU, Python 3.12 CPU, and the isolated RTX 5060 CUDA environment; cached model tests must execute rather than silently skip.
- [ ] Run Ruff, mypy, dependency checks, and wheel/sdist builds. Build environments sequentially to avoid shared setuptools output races.
- [ ] Run real token CPU/CUDA grids through ingest/evaluation/sweep output and report/gates. Verify recorded devices, source/token boundaries, and one embedding build per size/overlap pair. Use separate new output directories per run.
- [ ] Request independent code review, fix actionable findings, and record meaningful mistakes in `/home/mujeeb_25/mistake.md`.
- [ ] Push a feature branch and open/attach a PR with exact verification results. Wait for both Python CI jobs. Merge only when the user requests it.

## Success Criteria and Risks

Eight requested combinations produce eight valid, separately named artifacts and a complete baseline-comparison summary. Four size/overlap pairs produce at most four new index builds, and changing only top_k produces no document re-embedding. Legacy commands and token limits retain their tested behavior.

Runtime scales with the product of axis lengths and the configured measurement repetitions. More top_k results may improve retrieval metrics at the cost of latency/context size; the summary reports that tradeoff without declaring an automatic winner.

**Recommended execution:** implement inline in this session, then use one independent reviewer. These tasks share one service/output contract, so parallel implementation would create unnecessary coordination.
