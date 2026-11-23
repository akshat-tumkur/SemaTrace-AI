"""One-time local Docker corpus copy; never overwrite an initialized volume."""

import os
from pathlib import Path
import shutil


def main(source=Path("/seed"), volume=Path("/destination")):
    target = volume / "data"
    if target.exists():
        if not (target / "corpus.sqlite3").is_file():
            raise RuntimeError(
                "Existing Docker corpus is incomplete; preserve it before recovery."
            )
        print("Using the existing persistent Docker corpus.")
        return
    if not (source / "corpus.sqlite3").is_file():
        raise RuntimeError(
            "Provision the native source library first, then start Docker Compose."
        )
    staging = volume / ".initializing"
    # Resume only an unpublished copy. Publication is an atomic directory rename.
    shutil.copytree(source, staging, dirs_exist_ok=True)
    for directory, folders, files in os.walk(staging):
        os.chown(directory, 10001, 10001)
        for name in folders + files:
            os.chown(Path(directory) / name, 10001, 10001)
    os.chown(volume, 10001, 10001)
    staging.replace(target)
    print("Seeded the persistent Docker corpus; original source files are unchanged.")


if __name__ == "__main__":
    main()
