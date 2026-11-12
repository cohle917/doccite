"""Evaluate retrieval and claim verification on SciFact (Wadden et al., 2020).

Retrieval uses the BEIR release of SciFact: 5,183 paper abstracts and 300 test queries, scored
with nDCG@10, Recall@100 and MRR@10 like the BEIR benchmark. Claim verification uses the SciFact
dev claims, each paired with the abstracts it cites and labeled SUPPORT, CONTRADICT or NEI (not
enough information). The archives are downloaded on first use into ``<data_dir>/datasets`` and
are not part of this repository; see each dataset's license before redistributing them.
"""

import json
import sys
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import log2
from pathlib import Path
from time import perf_counter

import httpx

from .config import Settings
from .ingest.chunker import split_document
from .ingest.loaders import LoadedDocument, Section
from .models import SearchMode, Verdict
from .retrieval.embedder import FastEmbedder, FastReranker
from .retrieval.hybrid import Retriever
from .store.sqlite_store import SQLiteStore
from .verify.claim import verify_claim
from .verify.nli import OnnxNLI


@dataclass(frozen=True)
class Archive:
    url: str
    sha256: str
    root: str  # Directory inside the archive that holds the data files.


ARCHIVES = {
    "beir-scifact": Archive(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "scifact",
    ),
    "scifact": Archive(
        "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
        "data",
    ),
}
LABELS = ("SUPPORT", "CONTRADICT", "NEI")
_PREDICTED = {Verdict.SUPPORTED: "SUPPORT", Verdict.CONTRADICTED: "CONTRADICT"}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def fetch(settings: Settings, name: str) -> Path:
    """Download and unpack a dataset archive once, checking its SHA-256 digest."""
    archive = ARCHIVES[name]
    target = (settings.data_dir / "datasets" / name).resolve()
    data = target / archive.root
    if (target / ".complete").is_file():
        return data
    target.mkdir(parents=True, exist_ok=True)
    path = target / Path(archive.url).name
    _log(f"Downloading {archive.url}")
    digest = sha256()
    with httpx.stream("GET", archive.url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for part in response.iter_bytes():
                digest.update(part)
                file.write(part)
    if digest.hexdigest() != archive.sha256:
        path.unlink()
        raise SystemExit(f"Checksum mismatch for {archive.url}: {digest.hexdigest()}")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zipped:
            for member in zipped.infolist():
                if not (target / member.filename).resolve().is_relative_to(target):
                    raise SystemExit(f"Unsafe path in {path.name}: {member.filename}")
            zipped.extractall(target)
    else:
        with tarfile.open(path) as tarred:
            tarred.extractall(target, filter="data")
    path.unlink()
    (target / ".complete").touch()
    return data


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def index_corpus(settings: Settings, store: SQLiteStore, corpus: Path) -> dict[str, str]:
    """Index BEIR corpus rows as plain-text documents. Return corpus ID → DocCite doc_id."""
    ids = {}
    rows = _jsonl(corpus)
    for done, row in enumerate(rows, 1):
        title, text = row.get("title", "").strip(), row["text"].strip()
        body = f"{title}\n\n{text}" if title else text
        source = f"scifact:{row['_id']}"
        digest = sha256(body.encode()).hexdigest()
        revision = f"1:{settings.chunk_size}:{settings.chunk_overlap}:{digest}"
        info = store.unchanged(source, revision)
        if info is None:
            document = LoadedDocument(title or row["_id"], "txt", [Section(body)])
            full_text, passages = split_document(
                document, settings.chunk_size, settings.chunk_overlap
            )
            info, _ = store.save(source, revision, document.title, "txt", full_text, passages)
        ids[row["_id"]] = info.doc_id
        if done % 1000 == 0:
            _log(f"Indexed {done}/{len(rows)} abstracts")
    return ids


def _ndcg(ranking: list[str], relevant: dict[str, int], k: int = 10) -> float:
    gains = [relevant.get(doc, 0) for doc in ranking[:k]]
    dcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(rankings: list[list[str]], qrels: list[dict[str, int]]) -> dict[str, float]:
    ndcg = recall = mrr = 0.0
    for ranking, relevant in zip(rankings, qrels, strict=True):
        positive = {doc for doc, grade in relevant.items() if grade > 0}
        ndcg += _ndcg(ranking, relevant)
        recall += len(positive & set(ranking[:100])) / len(positive)
        mrr += next((1 / rank for rank, doc in enumerate(ranking[:10], 1) if doc in positive), 0)
    n = len(rankings)
    return {"nDCG@10": ndcg / n, "Recall@100": recall / n, "MRR@10": mrr / n}


def _run_retrieval(retriever: Retriever, data: Path, ids: dict[str, str], limit: int | None):
    queries = {row["_id"]: row["text"] for row in _jsonl(data / "queries.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    lines = (data / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        query_id, doc, grade = line.split("\t")
        qrels.setdefault(query_id, {})[doc] = int(grade)
    selected = sorted(qrels, key=int)[:limit]
    corpus_id = {doc_id: corpus for corpus, doc_id in ids.items()}
    print(f"Retrieval: BEIR SciFact test, {len(selected)} queries, {len(ids)} abstracts")
    print(f"{'Mode':<18} {'nDCG@10':>8} {'Recall@100':>10} {'MRR@10':>8} {'s/query':>8}")
    runs: list[tuple[str, SearchMode, bool, int]] = [
        ("bm25", "bm25", False, 400),
        ("dense", "dense", False, 400),
        ("hybrid", "hybrid", False, 400),
        # Reranking scores a 30-passage pool, so Recall@100 is not reported for it.
        ("hybrid + rerank", "hybrid", True, 30),
    ]
    results = {}
    for label, mode, rerank, depth in runs:
        rankings = []
        started = perf_counter()
        for done, query_id in enumerate(selected, 1):
            result = retriever.search(queries[query_id], depth, mode=mode, rerank=rerank)
            if result.mode != mode or (rerank and not result.reranked):
                rankings = None
                break
            # Passages → abstracts, keeping each abstract's best rank (MaxP).
            ranking = dict.fromkeys(corpus_id[hit.location.doc_id] for hit in result.hits)
            rankings.append(list(ranking))
            if done % 100 == 0:
                _log(f"  {label}: {done}/{len(selected)} queries")
        if rankings is None:
            print(f"{label:<18} {'N/A':>8} {'N/A':>10} {'N/A':>8}  (model unavailable)")
            continue
        seconds = (perf_counter() - started) / len(selected)
        metrics = retrieval_metrics(rankings, [qrels[q] for q in selected])
        recall = "-" if rerank else f"{metrics['Recall@100']:.3f}"
        print(
            f"{label:<18} {metrics['nDCG@10']:>8.3f} {recall:>10} "
            f"{metrics['MRR@10']:>8.3f} {seconds:>8.2f}"
        )
        results[label] = metrics
    return results


def claim_pairs(claims: Path) -> list[tuple[str, str, str]]:
    """Pair each SciFact claim with every abstract it cites: (claim, corpus ID, gold label)."""
    pairs = []
    for row in _jsonl(claims):
        for cited in row["cited_doc_ids"]:
            evidence = row["evidence"].get(str(cited))
            pairs.append((row["claim"], str(cited), evidence[0]["label"] if evidence else "NEI"))
    return pairs


def claim_metrics(gold: list[str], predicted: list[str]) -> dict:
    matrix = Counter(zip(gold, predicted, strict=True))
    f1 = {}
    for label in LABELS:
        tp = matrix[label, label]
        precision = tp / max(1, sum(matrix[g, label] for g in LABELS))
        recall = tp / max(1, sum(matrix[label, p] for p in LABELS))
        f1[label] = 2 * precision * recall / (precision + recall) if tp else 0.0
    not_support = sum(matrix[g, p] for g in LABELS for p in LABELS if g != "SUPPORT")
    false_support = sum(matrix[g, "SUPPORT"] for g in LABELS if g != "SUPPORT")
    return {
        "accuracy": sum(matrix[label, label] for label in LABELS) / len(gold),
        "macro_f1": sum(f1.values()) / len(LABELS),
        "f1": f1,
        "false_support": (false_support, not_support),
        "matrix": matrix,
    }


def _run_claims(
    retriever: Retriever,
    settings: Settings,
    data: Path,
    ids: dict[str, str],
    judges: list[str],
    limit: int | None,
):
    pairs = claim_pairs(data / "claims_dev.jsonl")[:limit]
    counts = Counter(label for _, _, label in pairs)
    print(
        f"Claims: SciFact dev, {len(pairs)} claim-abstract pairs "
        f"({', '.join(f'{label} {counts[label]}' for label in LABELS)})"
    )
    nli = OnnxNLI(settings)
    results = {}
    for judge in judges:
        predicted = []
        started = perf_counter()
        for done, (claim, cited, _) in enumerate(pairs, 1):
            result = verify_claim(
                retriever, settings, claim, ids[cited], nli=nli if judge == "nli" else None
            )
            if judge == "nli" and result.judged_by == "heuristic":
                raise SystemExit("NLI model unavailable; see the warning above.")
            predicted.append(_PREDICTED.get(result.verdict, "NEI"))
            if done % 100 == 0:
                _log(f"  {judge}: {done}/{len(pairs)} pairs")
        seconds = (perf_counter() - started) / len(pairs)
        metrics = claim_metrics([label for _, _, label in pairs], predicted)
        false_support, negatives = metrics["false_support"]
        print(
            f"\nJudge: {judge}  accuracy {metrics['accuracy']:.3f}  "
            f"macro-F1 {metrics['macro_f1']:.3f}  "
            f"false SUPPORT {false_support}/{negatives}  {seconds:.2f} s/pair"
        )
        print("  F1: " + "  ".join(f"{label} {metrics['f1'][label]:.3f}" for label in LABELS))
        print(f"  {'gold \\ predicted':<18}" + "".join(f"{label:>12}" for label in LABELS))
        for gold in LABELS:
            row = "".join(f"{metrics['matrix'][gold, p]:>12}" for p in LABELS)
            print(f"  {gold:<18}{row}")
        results[judge] = metrics
    return results


def run_public_evaluation(
    settings: Settings,
    *,
    retrieval: bool = True,
    claims: bool = True,
    judges: list[str] | None = None,
    limit: int | None = None,
) -> None:
    beir = fetch(settings, "beir-scifact")
    store = SQLiteStore(settings.data_dir / "public-eval" / "scifact" / "index.sqlite3")
    ids = index_corpus(settings, store, beir / "corpus.jsonl")
    embedder = FastEmbedder(settings) if settings.enable_dense else None
    retriever = Retriever(store, embedder, FastReranker(settings))
    print(
        f"Embedding model: {settings.embedding_model if embedder else '(disabled)'}  "
        f"Reranker: {settings.reranker_model}  NLI: {settings.nli_model}"
    )
    if retrieval:
        _run_retrieval(retriever, beir, ids, limit)
    if claims:
        if retrieval:
            print()
        scifact = fetch(settings, "scifact")
        _run_claims(retriever, settings, scifact, ids, judges or ["rules", "nli"], limit)
