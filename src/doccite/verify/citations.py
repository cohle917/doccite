import re
from collections import Counter
from collections.abc import Callable

from ..ingest.chunker import SENTENCE_CLOSERS, sentence_spans
from ..models import CitationCheck, CitationReport, CitedClaim, ClaimResult, Verdict
from ..store.sqlite_store import SQLiteStore

_ANCHOR = r"[\w-]+:p[1-9]\d*"
_MARKER = rf"\[(\^?\d+|{_ANCHOR})\]"
_AFTER_PUNCTUATION = re.compile(rf"([.!?。！？]{SENTENCE_CLOSERS})((?:[ \t]*{_MARKER})+)")
_LIST_PREFIX = re.compile(r"(?m)^[ \t]*(?:[-*+] |\d+[.)][ \t]+)")


def verify_citations(
    store: SQLiteStore,
    verify: Callable[[str, str | None, int | None], ClaimResult],
    text: str,
    sources: dict[str, str] | None = None,
) -> CitationReport:
    mapping = {key.strip("[]").lstrip("^"): value for key, value in (sources or {}).items()}
    known = {doc.doc_id: doc for doc in store.list_documents()}
    text = _LIST_PREFIX.sub("", text)
    text = _AFTER_PUNCTUATION.sub(lambda match: match[2] + match[1], text)
    claims = []
    summary = Counter(
        {
            "claims": 0,
            "uncited": 0,
            "citations": 0,
            "unresolved": 0,
            "missing_document": 0,
            "missing_page": 0,
            **{verdict.value: 0 for verdict in Verdict},
        }
    )
    for start, end in sentence_spans(text, line_breaks=True):
        sentence = text[start:end]
        markers = list(dict.fromkeys(re.findall(_MARKER, sentence)))
        claim = re.sub(r"\s+([.!?。！？,，])", r"\1", re.sub(_MARKER, "", sentence))
        claim = claim.strip()
        if not claim or not any(char.isalnum() for char in claim):
            continue
        item = CitedClaim(text=claim)
        summary["claims"] += 1
        if not markers:
            summary["uncited"] += 1
            result = verify(claim, None, None)
            if result.verdict != Verdict.NOT_FOUND:
                item.recommendations = result.evidence[:2]
        for marker in markers:
            summary["citations"] += 1
            ref = mapping.get(marker.lstrip("^")) if marker.lstrip("^").isdigit() else marker
            if not ref:
                item.citations.append(CitationCheck(marker=f"[{marker}]", status="unresolved"))
                summary["unresolved"] += 1
                continue
            match = re.fullmatch(r"([\w-]+):p([1-9]\d*)", ref)
            doc_id, page = (match[1], int(match[2])) if match else (ref, None)
            result = None
            if doc_id not in known:
                status = "missing_document"
            elif page is not None and page > (known[doc_id].page_count or 0):
                status = "missing_page"
            else:
                status = "checked"
                result = verify(claim, doc_id, page)
            summary[result.verdict.value if result else status] += 1
            item.citations.append(
                CitationCheck(
                    marker=f"[{marker}]", doc_id=doc_id, page=page, status=status, result=result
                )
            )
        claims.append(item)
    return CitationReport(claims=claims, summary=dict(summary))
