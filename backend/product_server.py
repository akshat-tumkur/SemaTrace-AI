"""Native single-origin product server. External services require per-upload opt-in."""

import argparse, os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def configure(port=8010):
    load_dotenv(ROOT / ".env")
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
        SEMATRACE_SOURCE_UPLOAD_ENABLED="true",
    )
    origins = set(filter(None, os.getenv("SEMATRACE_CORS_ORIGINS", "").split(",")))
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
    if (private_model / "review-manifest.json").is_file():
        os.environ.setdefault("SEMATRACE_REVIEW_MODEL_DIR", str(private_model))
    from app.main import app
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

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
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--host", choices=["127.0.0.1", "0.0.0.0", "::1"], default="127.0.0.1"
    )
    args = parser.parse_args()
    import uvicorn

    # Browsers retain several idle HTTP connections. Heavy analysis is bounded
    # separately by the shared one-worker semaphore, not this transport limit.
    uvicorn.run(
        configure(args.port),
        host=args.host,
        port=args.port,
        workers=1,
        limit_concurrency=64,
        timeout_keep_alive=5,
    )
