from docx import Document

from doccite.ingest.chunker import sentence_spans, split_document
from doccite.ingest.index import collect_paths
from doccite.ingest.loaders import LoadedDocument, Section, file_source, load


def test_pdf_preserves_offsets_and_original_page_numbers(store, indexer, pdf_path):
    item = indexer.file(pdf_path)
    assert item.status == "indexed"
    assert item.document.title == "Pilot Report"
    doc = store.documents(item.document.doc_id)[0]
    chunks = store.chunks([doc.info.doc_id])
    assert [chunk.location.page for chunk in chunks] == [1, 3]
    for chunk in chunks:
        loc = chunk.location
        assert doc.text[loc.start : loc.end] == chunk.text
    assert store.chunk_context(chunks[0].location.chunk_id, 1).after == chunks[1:]


def test_markdown_and_docx_heading_hierarchy(tmp_path):
    markdown = tmp_path / "测试.md"
    markdown.write_text(
        "# 报告\n\n正文。\n\n## 实验\n\n内容。\n```python\n# comment\n```\n",
        encoding="utf-8",
    )
    md = load(file_source(markdown))
    assert md.title == "报告"
    assert [section.heading for section in md.sections] == ["报告", "报告 / 实验"]
    assert "# comment" in md.sections[1].text
    path = tmp_path / "report.docx"
    doc = Document()
    doc.add_heading("Report", 0)
    doc.add_heading("Method", 1)
    doc.add_heading("Sample", 2)
    doc.add_paragraph("Sixty devices.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Batch"
    table.cell(0, 1).text = "A"
    doc.save(path)
    loaded = load(file_source(path))
    assert loaded.sections[-1].heading == "Report / Method / Sample"
    assert "Batch\tA" in loaded.sections[-1].text


def test_chinese_english_sentence_splitting_and_overlapping_chunks():
    text = "第一句。第二句有 12.5 元！“第三句？” He said “Ready.” Last sentence."
    sentences = [text[start:end] for start, end in sentence_spans(text)]
    assert sentences == [
        "第一句。",
        "第二句有 12.5 元！",
        "“第三句？”",
        "He said “Ready.”",
        "Last sentence.",
    ]
    text = "甲" * 320
    doc = LoadedDocument("测试", "txt", [Section(text)])
    extracted, passages = split_document(doc, 100, 25)
    assert extracted == text
    assert passages[1].start == 75
    assert all(p.end - p.start <= 100 for p in passages)
    assert passages[-1].end == len(text)


def test_reindex_and_failed_parse_are_atomic(tmp_path, indexer, store):
    path = tmp_path / "notes.txt"
    path.write_text("Initial text.", encoding="utf-8")
    first = indexer.file(path)
    assert indexer.file(path).status == "unchanged"
    chunks = store.chunks()
    old_id = chunks[0].location.chunk_id
    store.save_embeddings("test", {old_id: b"cached"})
    path.write_bytes(b"\xff")
    assert indexer.file(path).status == "error"
    assert store.chunks() == chunks
    path.write_text("Revised text.", encoding="utf-8")
    revised = indexer.file(path)
    assert revised.document.doc_id == first.document.doc_id
    assert store.chunks()[0].location.chunk_id != old_id
    store.save_embeddings("test", {old_id: b"stale"})
    assert store.stats().embeddings == 0
    assert store.remove(first.document.doc_id)
    assert store.stats().chunks == 0
    assert path.is_file()


def test_collect_paths_respects_recursion_and_deduplicates(tmp_path):
    root_file = tmp_path / "a.txt"
    root_file.touch()
    (tmp_path / "ignore.bin").touch()
    subdir = tmp_path / "nested"
    subdir.mkdir()
    nested = subdir / "b.md"
    nested.touch()
    assert collect_paths([str(tmp_path), str(root_file)], False) == [root_file]
    assert collect_paths([str(tmp_path)], True) == [root_file, nested]
