import re
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from docx import Document
from pypdf import PdfReader
from trafilatura import extract, extract_metadata

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx", ".html", ".htm"}
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class Section:
    text: str
    heading: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class LoadedDocument:
    title: str
    format: str
    sections: list[Section]
    page_count: int | None = None


@dataclass(frozen=True)
class Source:
    name: str
    format: str
    content: bytes
    url: str | None = None

    @property
    def digest(self) -> str:
        return sha256(self.content).hexdigest()


def file_source(path: Path) -> Source:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file format: {suffix or '(none)'}")
    return Source(str(path.resolve()), suffix[1:], path.read_bytes())


def url_source(url: str) -> Source:
    url = url.strip()
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Expected an HTTP or HTTPS URL")
    with httpx.Client(follow_redirects=True, timeout=20, max_redirects=8) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if media_type and media_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"Expected HTML, received {media_type}")
            body = bytearray()
            for part in response.iter_bytes():
                body.extend(part)
                if len(body) > 16 * 1024 * 1024:
                    raise ValueError("Web page exceeds 16 MiB")
            return Source(url, "html", bytes(body), str(response.url))


def _heading_path(stack: list[tuple[int, str]], level: int, title: str) -> str:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return " / ".join(name for _, name in stack)


def _markdown(text: str) -> tuple[str | None, list[Section]]:
    title = None
    heading = None
    stack: list[tuple[int, str]] = []
    lines: list[str] = []
    sections: list[Section] = []
    fence = None
    for line in text.splitlines():
        delimiter = FENCE.match(line)
        if delimiter:
            marks, suffix = delimiter.groups()
            if fence is None:
                fence = marks
            elif marks[0] == fence[0] and len(marks) >= len(fence) and not suffix.strip():
                fence = None
            lines.append(line)
            continue
        match = HEADING.match(line) if fence is None else None
        if match:
            if lines:
                sections.append(Section("\n".join(lines), heading))
            level, name = match.groups()
            name = re.sub(r"[ \t]+#+$", "", name)
            title = title or name
            heading = _heading_path(stack, len(level), name)
            lines = [name, ""]
        else:
            lines.append(line)
    if lines:
        sections.append(Section("\n".join(lines), heading))
    return title, sections


def _docx(content: bytes) -> tuple[str | None, list[Section]]:
    doc = Document(BytesIO(content))
    title = doc.core_properties.title.strip() or None
    heading = None
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    lines: list[str] = []
    for block in doc.iter_inner_content():
        if hasattr(block, "rows"):
            lines.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
            continue
        text = block.text.strip()
        if not text:
            continue
        style = block.style.name if block.style else ""
        match = re.fullmatch(r"Heading (\d+)", style)
        if match or style == "Title":
            if lines:
                sections.append(Section("\n\n".join(lines), heading))
            title = title or text
            heading = _heading_path(stack, int(match[1]) if match else 0, text)
            lines = []
        lines.append(text)
    if lines:
        sections.append(Section("\n\n".join(lines), heading))
    return title, sections


def load(source: Source) -> LoadedDocument:
    title = None
    page_count = None
    match source.format:
        case "md" | "markdown":
            title, sections = _markdown(source.content.decode("utf-8-sig"))
        case "txt":
            sections = [Section(source.content.decode("utf-8-sig"))]
        case "pdf":
            pdf = PdfReader(BytesIO(source.content))
            title = pdf.metadata.title if pdf.metadata else None
            page_count = len(pdf.pages)
            sections = [
                Section(page.extract_text() or "", page=index)
                for index, page in enumerate(pdf.pages, 1)
            ]
        case "docx":
            title, sections = _docx(source.content)
        case "html" | "htm":
            markdown = extract(
                source.content, url=source.url, output_format="markdown", include_comments=False
            )
            html_title, sections = _markdown(markdown or "")
            metadata = extract_metadata(source.content, default_url=source.url)
            title = (metadata.title if metadata else None) or html_title
        case _:
            raise ValueError(f"Unsupported format: {source.format}")
    sections = [
        Section(section.text.strip(), section.heading, section.page)
        for section in sections
        if section.text.strip()
    ]
    if not sections:
        raise ValueError("No extractable text")
    fallback_title = urlsplit(source.name).netloc if source.url else Path(source.name).stem
    return LoadedDocument(
        (title or "").strip() or fallback_title, source.format, sections, page_count
    )
