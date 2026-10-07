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


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_config_rejects_non_finite_relevance_threshold(
    tmp_path: Path, value: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""schema_version: 1
corpus: {{path: corpus, include: ['**/*.md']}}
chunking: {{strategy: words, size: 10, overlap: 0}}
embedding: {{provider: sentence_transformers, model: test, normalize: true}}
retrieval: {{metric: cosine, top_k: 5, relevance_threshold: {value}}}
storage: {{directory: state}}
""",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="relevance_threshold"):
        load_pipeline_config(path)
