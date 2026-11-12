from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from doccite.config import Settings
from doccite.ingest.index import Indexer
from doccite.store.sqlite_store import SQLiteStore


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "index", enable_dense=False, chunk_size=100, chunk_overlap=20
    )


@pytest.fixture
def store(settings):
    store = SQLiteStore(settings.data_dir / "index.sqlite3")
    yield store
    store.close()


@pytest.fixture
def indexer(store, settings):
    return Indexer(store, settings)


@pytest.fixture
def pdf_path(tmp_path) -> Path:
    path = tmp_path / "pilot.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.setTitle("Pilot Report")
    pdf.drawString(40, 700, "The pilot reduced operating costs by 12%.")
    pdf.showPage()
    pdf.showPage()
    pdf.drawString(40, 700, "Alice approved the final plan in 2025.")
    pdf.save()
    return path
