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
