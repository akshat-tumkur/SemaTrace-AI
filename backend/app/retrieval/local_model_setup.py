"""Explicit bounded public model acquisition; never called by model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen
from uuid import uuid4

from app.retrieval.local_model import (
    default_directory,
    model_lock,
    verify_artifact,
    verify_directory,
)


def download(directory):
    lock = model_lock()
    if directory.exists():
        verify_directory(directory)
        return dict(status="CACHED", downloaded_bytes=0, revision=lock["revision"])
    start = time.perf_counter()
    total = 0
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / (".partial-" + uuid4().hex)
    staging.mkdir()
    for entry in lock["files"]:
        name = entry["rfilename"]
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or entry["size"] > 96 * 1024 * 1024
        ):
            raise ValueError("unbounded_model_artifact")
        url = (
            "https://huggingface.co/"
            + lock["model"]
            + "/resolve/"
            + lock["revision"]
            + "/"
            + name
        )
        with urlopen(url, timeout=60) as response:
            raw = response.read(entry["size"] + 1)
        total += len(raw)
        if total > 100 * 1024 * 1024:
            raise ValueError("model_download_budget")
        verify_artifact(raw, entry)
        path = staging / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)
    verify_directory(staging)
    result = dict(
        status="READY",
        model=lock["model"],
        revision=lock["revision"],
        downloaded_bytes=total,
        seconds=time.perf_counter() - start,
    )
    (staging / "download.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    staging.rename(directory)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if not args.download:
        p.error("Use --download to explicitly acquire the pinned 90.9 MB model subset.")
    print(json.dumps(download(default_directory())))


if __name__ == "__main__":
    main()
