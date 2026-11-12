from pathlib import Path

import httpx

from ..config import Settings
from ..models import IngestItem
from ..store.sqlite_store import SQLiteStore
from .chunker import split_document
from .loaders import SUPPORTED_SUFFIXES, Source, file_source, load, url_source


def collect_paths(paths: list[str], recursive: bool) -> list[Path]:
    files = {}
    for item in paths:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            for candidate in sorted(candidates):
                if candidate.suffix.lower() in SUPPORTED_SUFFIXES and candidate.is_file():
                    resolved = candidate.resolve()
                    files[resolved] = None
        else:
            files[path] = None
    return list(files)


class Indexer:
    def __init__(self, store: SQLiteStore, settings: Settings):
        self.store = store
        self.settings = settings

    def index(self, source: Source) -> IngestItem:
        size, overlap = self.settings.chunk_size, self.settings.chunk_overlap
        revision = f"1:{size}:{overlap}:{source.digest}"
        current = self.store.unchanged(source.name, revision)
        if current:
            return IngestItem(source=source.name, status="unchanged", document=current)
        document = load(source)
        text, passages = split_document(document, size, overlap)
        info, changed = self.store.save(
            source.name,
            revision,
            document.title,
            document.format,
            text,
            passages,
            page_count=document.page_count,
        )
        return IngestItem(
            source=source.name, status="indexed" if changed else "unchanged", document=info
        )

    def file(self, path: Path) -> IngestItem:
        try:
            return self.index(file_source(path))
        except Exception as exc:
            # A failed extractor should not abort the rest of an ingest batch.
            return IngestItem(source=str(path), status="error", error=str(exc))

    def url(self, url: str) -> IngestItem:
        try:
            return self.index(url_source(url))
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return IngestItem(source=url, status="error", error=str(exc))
