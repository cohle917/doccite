import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ..ingest.chunker import Passage
from ..models import Chunk, ChunkContext, DocumentInfo, IndexStats, Location

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL UNIQUE,
    format TEXT NOT NULL,
    revision TEXT NOT NULL,
    text TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    page_count INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    page INTEGER,
    section TEXT,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE (doc_id, ordinal)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
-- Incremented whenever documents or chunks change; retrieval caches key on it.
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO meta VALUES ('generation', 0);
"""
_BUMP = "UPDATE meta SET value=value+1 WHERE key='generation'"
_INFO = "doc_id, title, source, format, chunk_count, page_count"
_CHUNKS = "SELECT c.*, d.title FROM chunks c JOIN documents d USING (doc_id)"


@dataclass(frozen=True)
class StoredDocument:
    info: DocumentInfo
    text: str
    revision: str


def _chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(location=Location.model_validate(dict(row)), text=row["text"])


class SQLiteStore:
    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: list[sqlite3.Connection] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @contextmanager
    def connect(self):
        """Run one transaction on this thread's connection, committing on success."""
        # Opening a connection costs tens of milliseconds on a large file (and more under
        # Windows antivirus), so each thread keeps one open for the life of the store.
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            # With WAL, NORMAL skips an fsync per commit but keeps the database consistent.
            db.execute("PRAGMA synchronous=NORMAL")
            self._local.db = db
            with self._lock:
                self._connections.append(db)
        with db:
            yield db

    def close(self) -> None:
        """Close every connection this store opened (for tests and shutdown)."""
        with self._lock:
            for db in self._connections:
                db.close()
            self._connections.clear()
        self._local = threading.local()

    def unchanged(self, source: str, revision: str) -> DocumentInfo | None:
        with self.connect() as db:
            row = db.execute(
                f"SELECT {_INFO} FROM documents WHERE source=? AND revision=?", (source, revision)
            ).fetchone()
        return DocumentInfo.model_validate(dict(row)) if row else None

    def save(
        self,
        source: str,
        revision: str,
        title: str,
        fmt: str,
        text: str,
        passages: list[Passage],
        *,
        page_count: int | None = None,
    ) -> tuple[DocumentInfo, bool]:
        doc_id = sha256(source.encode()).hexdigest()[:16]
        info = DocumentInfo(
            doc_id=doc_id,
            title=title,
            source=source,
            format=fmt,
            chunk_count=len(passages),
            page_count=page_count,
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT revision FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if current and current["revision"] == revision:
                return info, False
            db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, title, source, fmt, revision, text, len(passages), page_count),
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        sha256(f"{doc_id}:{revision}:{ordinal}".encode()).hexdigest()[:24],
                        doc_id,
                        ordinal,
                        passage.page,
                        passage.section,
                        passage.start,
                        passage.end,
                        text[passage.start : passage.end],
                    )
                    for ordinal, passage in enumerate(passages)
                ),
            )
            db.execute(_BUMP)
        return info, True

    def generation(self) -> int:
        with self.connect() as db:
            return db.execute("SELECT value FROM meta WHERE key='generation'").fetchone()[0]

    def list_documents(self) -> list[DocumentInfo]:
        with self.connect() as db:
            rows = db.execute(f"SELECT {_INFO} FROM documents ORDER BY source").fetchall()
        return [DocumentInfo.model_validate(dict(row)) for row in rows]

    def documents(self, doc_id: str | None = None) -> list[StoredDocument]:
        with self.connect() as db:
            rows = db.execute(
                f"SELECT {_INFO}, text, revision FROM documents "
                "WHERE (? IS NULL OR doc_id=?) ORDER BY source",
                (doc_id, doc_id),
            ).fetchall()
        return [
            StoredDocument(DocumentInfo.model_validate(dict(row)), row["text"], row["revision"])
            for row in rows
        ]

    def text_for_chunks(self, ids: list[str]) -> list[tuple[str, list[str]]]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT d.text, json_group_array(c.chunk_id) AS ids
                FROM documents d JOIN chunks c USING (doc_id)
                WHERE c.chunk_id IN (SELECT value FROM json_each(?)) GROUP BY d.doc_id
                """,
                (json.dumps(ids),),
            ).fetchall()
        return [(row["text"], json.loads(row["ids"])) for row in rows]

    def anchor(self, doc: StoredDocument, start: int) -> Location | None:
        with self.connect() as db:
            row = db.execute(
                _CHUNKS
                + """
                WHERE c.doc_id=? AND d.revision=? AND c.start<=? AND c.end>?
                ORDER BY c.end DESC LIMIT 1
                """,
                (doc.info.doc_id, doc.revision, start, start),
            ).fetchone()
        return Location.model_validate(dict(row)) if row else None

    def remove(self, doc_id: str) -> bool:
        with self.connect() as db:
            removed = db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,)).rowcount == 1
            if removed:
                db.execute(_BUMP)
            return removed

    def chunks(self, doc_ids: list[str] | None = None, page: int | None = None) -> list[Chunk]:
        query = (
            _CHUNKS
            + """
            WHERE (? IS NULL OR c.doc_id IN (SELECT value FROM json_each(?)))
            AND (? IS NULL OR c.page=?) ORDER BY c.doc_id, c.ordinal
        """
        )
        ids = json.dumps(doc_ids) if doc_ids is not None else None
        with self.connect() as db:
            return [_chunk(row) for row in db.execute(query, (ids, ids, page, page))]

    def chunk_context(self, chunk_id: str, context: int) -> ChunkContext:
        with self.connect() as db:
            rows = db.execute(
                _CHUNKS
                + """
                JOIN chunks target ON c.doc_id=target.doc_id
                WHERE target.chunk_id=? AND c.ordinal BETWEEN target.ordinal-? AND target.ordinal+?
                ORDER BY c.ordinal
                """,
                (chunk_id, context, context),
            ).fetchall()
        chunks = [_chunk(row) for row in rows]
        for index, chunk in enumerate(chunks):
            if chunk.location.chunk_id == chunk_id:
                return ChunkContext(chunk=chunk, before=chunks[:index], after=chunks[index + 1 :])
        raise ValueError(f"Unknown chunk: {chunk_id}")

    def stats(self) -> IndexStats:
        with self.connect() as db:
            row = db.execute("""
                SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks),
                       (SELECT count(*) FROM embeddings)
            """).fetchone()
        return IndexStats(documents=row[0], chunks=row[1], embeddings=row[2])

    def embeddings(self, model: str, ids: list[str]) -> dict[str, bytes]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT chunk_id, vector FROM embeddings WHERE model=? "
                "AND chunk_id IN (SELECT value FROM json_each(?))",
                (model, json.dumps(ids)),
            )
            return dict(rows)

    def save_embeddings(self, model: str, vectors: dict[str, bytes]) -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT OR IGNORE INTO embeddings SELECT chunk_id, ?, ? FROM chunks WHERE chunk_id=?",
                ((model, vector, chunk_id) for chunk_id, vector in vectors.items()),
            )
