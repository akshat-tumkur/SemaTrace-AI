"""One-command native application: setup checks, frontend build, one local server."""

import argparse, json, os, shutil, socket, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    subprocess.run(
        [str(a) for a in args], check=True, cwd=kwargs.pop("cwd", ROOT), **kwargs
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    runtime = ROOT / ".local/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    address = f"http://127.0.0.1:{args.port}"
    try:
        with urllib.request.urlopen(
            address + "/api/product/status", timeout=2
        ) as response:
            if json.load(response).get("version") == "3.1":
                print("SemaTrace is already running: " + address)
                return
    except Exception:
        pass
    with socket.socket() as check:
        if check.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(
                f"Port {args.port} is already occupied. Stop the previous development backend or choose --port 8020."
            )
    python = ROOT / (
        "backend/.venv/Scripts/python.exe"
        if os.name == "nt"
        else "backend/.venv/bin/python"
    )
    if not python.exists():
        run([sys.executable, "-m", "venv", ROOT / "backend/.venv"])
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import fastapi,onnxruntime,tokenizers,reportlab,Bio,langgraph,httpx",
        ],
        capture_output=True,
    )
    if probe.returncode:
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "-r",
                "backend/requirements-product.txt",
            ]
        )
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise SystemExit(
            "Node.js/npm is required once to build the interface. Install Node22.12+ and retry."
        )
    if not (ROOT / "frontend/node_modules").is_dir():
        run([npm, "ci"], cwd=ROOT / "frontend")
    dist = ROOT / "frontend/dist/index.html"
    sources = list((ROOT / "frontend/src").rglob("*")) + [
        ROOT / "frontend/package-lock.json",
        ROOT / "frontend/vite.config.ts",
    ]
    if (
        args.rebuild
        or not dist.exists()
        or any(
            p.is_file() and p.stat().st_mtime > dist.stat().st_mtime for p in sources
        )
    ):
        environment = dict(os.environ, VITE_API_URL="")
        run([npm, "run", "build"], cwd=ROOT / "frontend", env=environment)
    model_check = subprocess.run(
        [
            str(python),
            "-c",
            "from dotenv import load_dotenv; load_dotenv('.env'); from app.retrieval.local_model import default_directory,verify_directory; verify_directory(default_directory())",
        ],
        cwd=ROOT,
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "backend")),
    )
    if model_check.returncode:
        run(
            [python, "-m", "app.retrieval.local_model_setup", "--download"],
            cwd=ROOT / "backend",
        )
    state_code = "from dotenv import load_dotenv; load_dotenv('.env'); from app.storage.source_corpus import SourceCorpusStore; from app.retrieval.generations import GenerationStore,IndexSpec; import json; s=SourceCorpusStore(); n=len(s.list_sources(limit=1)); c=s.snapshot() if n else None; print(json.dumps({'sources':n,'ready':bool(c and GenerationStore(c.retrieval_index_path).active(c,IndexSpec.local())[0])}))"
    state = subprocess.run(
        [str(python), "-c", state_code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "backend")),
    )
    try:
        corpus_state = json.loads(state.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise SystemExit(
            "Source storage failed validation. Inspect storage before starting; no substitute empty corpus was used."
        )
    if not corpus_state["sources"]:
        seed = runtime / "initial-wikipedia.json"
        if not seed.exists():
            run(
                [
                    python,
                    "-m",
                    "app.corpus.cli",
                    "resolve",
                    "--output",
                    seed,
                    "--max-documents",
                    "12",
                ],
                cwd=ROOT / "backend",
            )
        run(
            [
                python,
                "-m",
                "app.corpus.cli",
                "ingest",
                "--manifest",
                seed,
                "--max-documents",
                "12",
            ],
            cwd=ROOT / "backend",
        )
    if not corpus_state["ready"]:
        run(
            [
                python,
                "-m",
                "app.retrieval.index_cli",
                "build",
                "--provider",
                "local-onnx",
                "--batch-size",
                "32",
            ],
            cwd=ROOT / "backend",
        )
    out = (runtime / "server.stdout.log").open("ab")
    err = (runtime / "server.stderr.log").open("ab")
    process = subprocess.Popen(
        [
            str(python),
            str(ROOT / "backend/product_server.py"),
            "--port",
            str(args.port),
        ],
        cwd=ROOT,
        stdout=out,
        stderr=err,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    manifest = {
        "pid": process.pid,
        "port": args.port,
        "url": address,
        "command": str(ROOT / "backend/product_server.py"),
    }
    (runtime / "server.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for _ in range(60):
        if process.poll() is not None:
            raise SystemExit(
                "Server startup failed. Inspect .local/runtime/server.stderr.log."
            )
        try:
            with urllib.request.urlopen(
                address + "/api/product/status", timeout=2
            ) as r:
                if r.status == 200:
                    print("SemaTrace is ready: " + address)
                    return
        except Exception:
            time.sleep(0.5)
    raise SystemExit(
        "Startup is taking longer than expected. Inspect .local/runtime/server.stderr.log."
    )


if __name__ == "__main__":
    main()
