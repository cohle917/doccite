import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from doccite.server import create_server


async def call(client, tool, arguments=None, **kwargs):
    result = await client.call_tool(tool, arguments or {}, **kwargs)
    assert not result.isError, result.content
    assert result.structuredContent is not None, tool
    assert result.content and isinstance(result.content[0], types.TextContent)
    return result.structuredContent


class WebPage(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"""<html><head><title>Dispatch Study</title></head><body>
        <article><h1>Dispatch Study</h1><p>The study recorded 44 completed dispatches
        during the pilot. Operators recorded dispatch completion in a daily log.</p>
        <p>The study documented staffing changes after the pilot period.</p></article></body></html>
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def webpage():
    with ThreadingHTTPServer(("127.0.0.1", 0), WebPage) as http:
        thread = Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{http.server_port}/study"
        finally:
            http.shutdown()
            thread.join()


@pytest.mark.asyncio
async def test_end_to_end_tools_resources_prompts_and_progress(settings, pdf_path, webpage):
    progress = []

    async def report(done, total, message):
        progress.append((done, total, message))

    async with create_connected_server_and_client_session(create_server(settings)) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 9
        assert all(tool.outputSchema for tool in tools.tools)
        indexed = await call(
            client, "ingest_documents", {"paths": [str(pdf_path)]}, progress_callback=report
        )
        assert progress[0][:2] == (1, 1)
        doc_id = indexed["items"][0]["document"]["doc_id"]
        online = await call(client, "ingest_url", {"url": webpage})
        assert online["status"] == "indexed"
        assert online["document"]["title"] == "Dispatch Study"
        assert len((await call(client, "list_documents"))["documents"]) == 2
        hits = await call(client, "search", {"query": "operating costs", "doc_ids": [doc_id]})
        assert hits["mode"] == "bm25"
        anchor = hits["hits"][0]["location"]
        assert anchor["page"] == 1
        chunk_id = anchor["chunk_id"]
        chunk = await call(client, "read_chunk", {"chunk_id": chunk_id})
        assert chunk["after"][0]["location"]["page"] == 3
        quote = await call(client, "verify_quote", {"quote": "operating costs by 12%"})
        assert quote["status"] == "verbatim"
        claim = await call(
            client,
            "verify_claim",
            {
                "claim": "The pilot reduced operating costs by 21%.",
                "doc_id": doc_id,
                "use_llm_judge": True,
            },
        )
        assert claim["verdict"] == "PARTIALLY_SUPPORTED"
        assert any("does not support sampling" in issue for issue in claim["issues"])
        report = await call(
            client,
            "verify_citations",
            {
                "text": (
                    "The pilot reduced operating costs by 12%. [1] "
                    f"Alice approved the final plan in 2025 [{doc_id}:p3]. "
                    f"The pilot reduced operating costs by 12% [{doc_id}:p3]. "
                    "A fabricated outcome [fabricated:p1]. A mystery [9]. "
                    f"A blank page [{doc_id}:p2]. A bad page [{doc_id}:p4]. "
                    "Alice approved the final plan in 2025."
                ),
                "sources": {"[^1]": f"{doc_id}:p1"},
            },
        )
        assert report["summary"]["SUPPORTED"] == 2
        assert report["summary"]["missing_document"] == 1
        assert report["summary"]["missing_page"] == 1
        assert report["summary"]["unresolved"] == 1
        assert report["summary"]["NOT_FOUND"] == 2
        assert report["summary"]["uncited"] == 1
        assert report["claims"][-1]["recommendations"]
        resources = await client.list_resources()
        assert str(resources.resources[0].uri) == "doccite://stats"
        assert len((await client.list_resource_templates()).resourceTemplates) == 2
        stats = await client.read_resource("doccite://stats")
        assert json.loads(stats.contents[0].text)["documents"] == 2
        document = await client.read_resource(f"doccite://documents/{doc_id}")
        assert "12%" in document.contents[0].text
        data = await client.read_resource(f"doccite://chunks/{chunk_id}")
        assert json.loads(data.contents[0].text)["location"] == anchor
        assert {prompt.name for prompt in (await client.list_prompts()).prompts} == {
            "answer_with_citations",
            "fact_check",
        }
        for prompt, argument in [("answer_with_citations", "question"), ("fact_check", "text")]:
            assert (await client.get_prompt(prompt, {argument: "test"})).messages
        assert (await call(client, "remove_document", {"doc_id": doc_id}))["removed"]
        assert pdf_path.exists()


@pytest.mark.asyncio
async def test_sampling_verdict_uses_client_result(settings, indexer, pdf_path):
    doc = indexer.file(pdf_path).document

    async def sample(context, params):
        payload = json.loads(params.messages[0].content.text)
        assert "21%" in payload["claim"]
        assert any("12%" in sentence for sentence in payload["evidence"])
        return types.CreateMessageResult(
            role="assistant",
            model="fake-client",
            content=types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "verdict": "CONTRADICTED",
                        "confidence": 0.95,
                        "reason": "21% conflicts with 12%.",
                    }
                ),
            ),
        )

    server = create_server(settings)
    async with create_connected_server_and_client_session(
        server, sampling_callback=sample
    ) as client:
        checked = await call(
            client,
            "verify_claim",
            {
                "claim": "The pilot reduced operating costs by 21%.",
                "doc_id": doc.doc_id,
                "use_llm_judge": True,
            },
        )
        assert checked["verdict"] == "CONTRADICTED"
        assert checked["judged_by"] == "sampling"
