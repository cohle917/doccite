import argparse

from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doccite", description="Search and verify local citations."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the DocCite MCP server")
    serve.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    ingest = commands.add_parser("ingest", help="Index local files or directories")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    search = commands.add_parser("search", help="Find passages in the local index")
    search.add_argument("query")
    search.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--doc-id", action="append", dest="doc_ids")
    search.add_argument("--rerank", action="store_true")
    evaluate = commands.add_parser(
        "eval", help="Run the labeled retrieval and verification benchmark"
    )
    evaluate.add_argument("--dataset", help="Path to an evaluation JSON manifest")
    evaluate.add_argument(
        "--judge",
        action="append",
        choices=["rules", "nli"],
        dest="judges",
        help="Claim judge to evaluate; repeat for several (default: rules and nli)",
    )
    public = commands.add_parser(
        "eval-public", help="Benchmark retrieval and claim checks on SciFact (downloads ~6 MB)"
    )
    public.add_argument("--limit", type=int, help="Use only the first N queries and claim pairs")
    public.add_argument("--skip-retrieval", action="store_true")
    public.add_argument("--skip-claims", action="store_true")
    public.add_argument(
        "--judge",
        action="append",
        choices=["rules", "nli"],
        dest="judges",
        help="Claim judge to evaluate; repeat for several (default: rules and nli)",
    )
    args = parser.parse_args()
    settings = Settings()

    if args.command == "serve":
        from .server import create_server

        create_server(settings, host=args.host, port=args.port).run(transport=args.transport)
        return
    if args.command == "eval":
        from .evaluation import run_evaluation

        run_evaluation(settings, args.dataset, args.judges)
        return
    if args.command == "eval-public":
        from .public_eval import run_public_evaluation

        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        run_public_evaluation(
            settings,
            retrieval=not args.skip_retrieval,
            claims=not args.skip_claims,
            judges=args.judges,
            limit=args.limit,
        )
        return

    from .ingest.index import Indexer, collect_paths
    from .models import IngestResult
    from .retrieval.embedder import FastEmbedder, FastReranker
    from .retrieval.hybrid import Retriever
    from .store.sqlite_store import SQLiteStore

    store = SQLiteStore(settings.data_dir / "index.sqlite3")
    if args.command == "ingest":
        indexer = Indexer(store, settings)
        items = [indexer.file(path) for path in collect_paths(args.paths, args.recursive)]
        print(IngestResult(items=items).model_dump_json(indent=2))
        if any(item.status == "error" for item in items):
            raise SystemExit(1)
        return
    if not args.query.strip() or args.top_k < 1:
        parser.error("search requires a nonempty query and a positive --top-k")
    embedder = FastEmbedder(settings) if settings.enable_dense else None
    retriever = Retriever(store, embedder, FastReranker(settings))
    result = retriever.search(args.query, args.top_k, args.doc_ids, args.mode, args.rerank)
    print(result.model_dump_json(indent=2))
