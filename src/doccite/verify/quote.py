from difflib import SequenceMatcher

from rapidfuzz import fuzz

from ..models import Chunk, Location, QuoteEdit, QuoteResult
from ..store.sqlite_store import SQLiteStore, StoredDocument
from .normalize import normalize


def _result(
    store: SQLiteStore, doc: StoredDocument, quote: str, start: int, end: int, status, score: float
) -> QuoteResult:
    anchor = store.anchor(doc, start)
    text = doc.text[start:end]
    location = Location(
        doc_id=doc.info.doc_id,
        title=doc.info.title,
        page=anchor.page if anchor else None,
        section=anchor.section if anchor else None,
        chunk_id=anchor.chunk_id if anchor else None,
        start=start,
        end=end,
    )
    edits = [
        QuoteEdit(operation=op, quote=quote[a:b], source=text[c:d])
        for op, a, b, c, d in SequenceMatcher(None, quote, text, autojunk=False).get_opcodes()
        if op != "equal"
    ]
    return QuoteResult(
        status=status,
        similarity=score,
        match=Chunk(location=location, text=text),
        differences=edits,
    )


def verify_quote(
    store: SQLiteStore, quote: str, doc_id: str | None, threshold: float
) -> QuoteResult:
    query, _ = normalize(quote)
    if not query:
        raise ValueError("Quote cannot be empty")
    documents = store.documents(doc_id)
    for doc in documents:
        start = doc.text.find(quote)
        if start >= 0:
            return _result(store, doc, quote, start, start + len(quote), "verbatim", 1)

    normalized = [(doc, *normalize(doc.text)) for doc in documents]
    for doc, text, offsets in normalized:
        start = text.find(query)
        if start >= 0:
            end = start + len(query)
            return _result(
                store, doc, quote, offsets[start][0], offsets[end - 1][1], "normalized", 1
            )

    best = None
    score = 0.0
    stride = max(256, len(query))
    for doc, text, offsets in normalized:
        for start in range(0, len(text), stride):
            window = text[start : start + stride + len(query)]
            alignment = fuzz.partial_ratio_alignment(query, window, score_cutoff=threshold * 100)
            if alignment is None:
                continue
            left = start + alignment.dest_start
            right = start + alignment.dest_end
            similarity = fuzz.ratio(query, text[left:right]) / 100
            if similarity > score and similarity >= threshold:
                score = similarity
                best = (doc, offsets[left][0], offsets[right - 1][1])
    if best is None:
        return QuoteResult(status="not_found")
    doc, start, end = best
    return _result(store, doc, quote, start, end, "modified", score)
