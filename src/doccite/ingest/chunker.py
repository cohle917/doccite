import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .loaders import LoadedDocument

SENTENCE_CLOSERS = r'[”’"」』）)\]]*'
_BOUNDARY = re.compile(
    rf"[。！？!?]+{SENTENCE_CLOSERS}|(?<!\.)\.(?<!\b[A-Z]\.){SENTENCE_CLOSERS}(?=\s|$)|\n+"
)


def sentence_spans(text: str, *, line_breaks: bool = False):
    """Yield nonempty sentence spans with offsets into the original string."""
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        if match[0] == "\n" and not line_breaks:
            continue
        left = start + len(text[start:end]) - len(text[start:end].lstrip())
        right = start + len(text[start:end].rstrip())
        if left < right:
            yield left, right
        start = end
    left = start + len(text[start:]) - len(text[start:].lstrip())
    right = len(text.rstrip())
    if left < right:
        yield left, right


@dataclass(frozen=True)
class Passage:
    start: int
    end: int
    page: int | None
    section: str | None


def _chunk_spans(text: str, size: int, overlap: int):
    spans = list(sentence_spans(text))
    boundaries = [end for _, end in spans]
    starts = [start for start, _ in spans]

    start = 0
    while start < len(text):
        limit = min(start + size, len(text))
        index = bisect_right(boundaries, limit) - 1
        end = boundaries[index] if index >= 0 else limit
        if limit == len(text) or end - start <= max(size // 2, overlap):
            end = limit
        left = start + len(text[start:end]) - len(text[start:end].lstrip())
        right = start + len(text[start:end].rstrip())
        if left < right:
            yield left, right
        if end == len(text):
            break
        next_start = end - overlap
        index = bisect_left(starts, next_start)
        if index < len(starts) and starts[index] <= end:
            next_start = starts[index]
        start = next_start


def split_document(doc: LoadedDocument, size: int, overlap: int) -> tuple[str, list[Passage]]:
    text = "\n\n".join(section.text for section in doc.sections)
    offset = 0
    passages = []
    for section in doc.sections:
        passages.extend(
            Passage(offset + start, offset + end, section.page, section.heading)
            for start, end in _chunk_spans(section.text, size, overlap)
        )
        offset += len(section.text) + 2
    return text, passages
