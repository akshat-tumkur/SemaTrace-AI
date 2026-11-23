"""Native single-origin product server. External services require per-upload opt-in."""

import argparse, os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def configure(port=8010, env_file=None):
    load_dotenv(Path(env_file) if env_file else ROOT / ".env")
    from app.cloud import enabled as cloud_enabled

    if os.getenv("RENDER") == "true" and not cloud_enabled():
        raise RuntimeError(
            "Render requires persistent Supabase storage and authentication; local SQLite mode is private-only"
        )
    if cloud_enabled():
        from app.cloud.client import required, supabase_public_key, supabase_secret_key

        supabase_public_key()
        supabase_secret_key()

        for name in (
            "SUPABASE_URL",
            "SUPABASE_DB_URL",
            "PINECONE_API_KEY",
            "PINECONE_INDEX_HOST",
            "PINECONE_INDEX_NAME",
            "SEMATRACE_CORS_ORIGINS",
            "SEMATRACE_ALLOWED_HOSTS",
        ):
            required(name)
        if os.getenv("SEMATRACE_VECTOR_PROVIDER") != "pinecone":
            raise RuntimeError("Cloud mode requires SEMATRACE_VECTOR_PROVIDER=pinecone")
        if (
            "*" in os.environ["SEMATRACE_CORS_ORIGINS"]
            or "*" in os.environ["SEMATRACE_ALLOWED_HOSTS"]
        ):
            raise RuntimeError(
                "Cloud mode requires explicit frontend origins and API hosts"
            )
    os.environ.setdefault(
        "SEMATRACE_DB_PATH", str(ROOT / "backend/data/beta-demo/analyses.sqlite3")
    )
    os.environ.setdefault(
        "SEMATRACE_CORPUS_DB_PATH",
        str(ROOT / "backend/data/source-corpus/corpus.sqlite3"),
    )
    os.environ.update(
        SEMATRACE_IMPORT_LEGACY_CORPUS="false",
        SEMATRACE_EMBEDDING_PROVIDER="local-onnx",
    )
    os.environ.setdefault("SEMATRACE_SOURCE_UPLOAD_ENABLED", "true")
    origins = set(filter(None, os.getenv("SEMATRACE_CORS_ORIGINS", "").split(",")))
    if not cloud_enabled():
        origins.update(
            {
                f"http://localhost:{port}",
                f"http://127.0.0.1:{port}",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            }
        )
    os.environ["SEMATRACE_CORS_ORIGINS"] = ",".join(sorted(origins))
    os.environ.setdefault("SEMATRACE_WRITE_REQUESTS_PER_MINUTE", "60")
    private_model = ROOT / ".local/models/roberta-review-v1"
    if not cloud_enabled() and (private_model / "review-manifest.json").is_file():
        os.environ.setdefault("SEMATRACE_REVIEW_MODEL_DIR", str(private_model))
    from app.main import app
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    if (
        os.getenv(
            "SEMATRACE_SERVE_FRONTEND", "false" if cloud_enabled() else "true"
        ).lower()
        == "false"
    ):
        return app

    dist = Path(os.getenv("SEMATRACE_FRONTEND_DIST") or ROOT / "frontend/dist")
    if (dist / "index.html").is_file():
        app.mount(
            "/assets", StaticFiles(directory=dist / "assets"), name="frontend-assets"
        )

        @app.get("/", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def frontend():
            return FileResponse(
                dist / "index.html", headers={"Cache-Control": "no-store"}
            )
    else:
        raise RuntimeError(
            "Build the frontend first: npm.cmd run build in frontend, or use the root startup command."
        )
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8010")))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--host", choices=["127.0.0.1", "0.0.0.0", "::1"], default="127.0.0.1"
    )
    args = parser.parse_args()
    import uvicorn

    # Browsers retain several idle HTTP connections. Heavy analysis is bounded
    # separately by the shared one-worker semaphore, not this transport limit.
    uvicorn.run(
        configure(args.port, args.env_file),
        host=args.host,
        port=args.port,
        workers=1,
        limit_concurrency=64,
        timeout_keep_alive=5,
    )
