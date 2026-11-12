from collections import Counter
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Settings
from .ingest.index import Indexer
from .ingest.loaders import file_source
from .models import SearchHit, SearchMode, Verdict
from .retrieval.embedder import FastEmbedder, FastReranker
from .retrieval.hybrid import Retriever
from .store.sqlite_store import SQLiteStore
from .verify.claim import verify_claim


class PassageLabel(BaseModel):
    file: str
    contains: str = Field(min_length=1)

    def matches(self, hit: SearchHit, ids: dict[str, str]) -> bool:
        return (
            hit.location.doc_id == ids[self.file]
            and self.contains.casefold() in hit.text.casefold()
        )


class RetrievalCase(BaseModel):
    query: str = Field(min_length=1)
    relevant: list[PassageLabel] = Field(min_length=1)


class ClaimCase(BaseModel):
    claim: str = Field(min_length=1)
    verdict: Verdict
    file: str | None = None
    category: str = "uncategorized"


class Dataset(BaseModel):
    corpus: str
    files: list[str] = Field(min_length=1)
    retrieval: list[RetrievalCase] = Field(min_length=1)
    claims: list[ClaimCase] = Field(min_length=1)


def _score_retrieval(
    retriever: Retriever,
    cases: list[RetrievalCase],
    ids: dict[str, str],
    mode: SearchMode,
    rerank: bool,
) -> tuple[float, float] | None:
    recall = reciprocal_rank = 0.0
    for case in cases:
        result = retriever.search(case.query, 5, mode=mode, rerank=rerank)
        if result.mode != mode or (rerank and not result.reranked):
            return None
        ranks = [
            next(
                (i for i, hit in enumerate(result.hits, 1) if label.matches(hit, ids)),
                None,
            )
            for label in case.relevant
        ]
        found = [rank for rank in ranks if rank is not None]
        recall += len(found) / len(ranks)
        reciprocal_rank += 1 / min(found) if found else 0
    return recall / len(cases), reciprocal_rank / len(cases)


def run_evaluation(settings: Settings, dataset_path: str | None = None) -> None:
    manifest = Path(dataset_path or "eval/dataset.json").resolve()
    if not manifest.is_file():
        raise SystemExit(f"Evaluation manifest not found: {manifest}. Pass --dataset PATH.")
    raw = manifest.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    sources = {name: file_source(manifest.parent / dataset.corpus / name) for name in dataset.files}
    fingerprint = sha256(raw)
    for source in sources.values():
        fingerprint.update(source.name.encode())
        fingerprint.update(bytes.fromhex(source.digest))
    eval_dir = settings.data_dir / "eval" / fingerprint.hexdigest()[:16]
    store = SQLiteStore(eval_dir / "index.sqlite3")
    indexer = Indexer(store, settings)
    ids = {name: indexer.index(source).document.doc_id for name, source in sources.items()}
    for case in dataset.retrieval:
        for label in case.relevant:
            if not any(
                label.contains.casefold() in chunk.text.casefold()
                for chunk in store.chunks([ids[label.file]])
            ):
                raise SystemExit(f"Missing labeled passage in {label.file}: {label.contains!r}")
    embedder = FastEmbedder(settings) if settings.enable_dense else None
    retriever = Retriever(store, embedder, FastReranker(settings))

    print(f"Corpus: {len(ids)} documents, {store.stats().chunks} chunks")
    print(f"Retrieval: {len(dataset.retrieval)} queries")
    print(f"{'Mode':<18} {'Recall@5':>8} {'MRR@5':>8}")
    for mode, rerank in [("bm25", False), ("dense", False), ("hybrid", False), ("hybrid", True)]:
        label = "hybrid + rerank" if rerank else mode
        scores = _score_retrieval(retriever, dataset.retrieval, ids, mode, rerank)
        if scores is None:
            print(f"{label:<18} {'N/A':>8} {'N/A':>8}  (model unavailable)")
        else:
            print(f"{label:<18} {scores[0]:>8.3f} {scores[1]:>8.3f}")

    matrix = Counter()
    by_category: dict[str, list[int]] = {}
    for case in dataset.claims:
        doc_id = ids[case.file] if case.file else None
        predicted = verify_claim(retriever, settings, case.claim, doc_id).verdict
        matrix[case.verdict, predicted] += 1
        tally = by_category.setdefault(case.category, [0, 0])
        tally[0] += predicted == case.verdict
        tally[1] += 1
    correct = sum(matrix[verdict, verdict] for verdict in Verdict)
    print(f"Claim accuracy: {correct}/{len(dataset.claims)} ({correct / len(dataset.claims):.3f})")
    print(f"{'Category':<18} {'Correct':>8}")
    for category, (hits, total) in by_category.items():
        print(f"{category:<18} {f'{hits}/{total}':>8}")
    supported = Verdict.SUPPORTED
    unsupported_cases = sum(n for (exp, _), n in matrix.items() if exp != supported)
    false_support = sum(
        n for (exp, got), n in matrix.items() if exp != supported and got == supported
    )
    supported_cases = sum(n for (exp, _), n in matrix.items() if exp == supported)
    print(
        f"False SUPPORTED: {false_support}/{unsupported_cases} claims not labeled SUPPORTED. "
        f"SUPPORTED recall: {matrix[supported, supported]}/{supported_cases}"
    )
    print("Confusion matrix (rows = expected, columns = predicted)")
    print(f"{'':<22}" + "".join(f"{verdict.value:>22}" for verdict in Verdict))
    for expected in Verdict:
        print(
            f"{expected.value:<22}"
            + "".join(f"{matrix[expected, actual]:>22}" for actual in Verdict)
        )
