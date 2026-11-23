"""Public service readiness without exposing any account's documents."""

import time
from threading import Lock

from app.cloud.client import connect, CloudUnavailable
from app.cloud.pinecone import Pinecone
from app.cloud.libraries import GLOBAL_LIBRARY
from app.retrieval.local_model import default_directory, verify_directory

_last = 0.0
_lock = Lock()


def ready():
    global _last
    with _lock:
        if time.monotonic() - _last > 30:
            with connect() as c:
                row = c.execute(
                    "SELECT to_regclass('public.st_reports') AS reports,to_regclass('public.st_sources') AS sources,to_regclass('public.st_libraries') AS libraries,to_regclass('public.st_admins') AS admins"
                ).fetchone()
                if not all(row.values()):
                    raise CloudUnavailable("Apply cloud schema before starting")
                if not c.execute(
                    "SELECT 1 FROM st_sources WHERE owner_id=%s AND active_version IS NOT NULL LIMIT 1",
                    (GLOBAL_LIBRARY,),
                ).fetchone():
                    raise CloudUnavailable(
                        "Publish the shared global corpus before public deployment"
                    )
                proof = c.execute(
                    """SELECT o.occurrence_id,o.version_id,o.passage_id,v.source_id FROM st_occurrences o
                    JOIN st_sources s ON s.owner_id=o.owner_id AND s.active_version=o.version_id
                    JOIN st_versions v ON v.owner_id=o.owner_id AND v.version_id=o.version_id
                    WHERE o.owner_id=%s AND o.indexed ORDER BY o.occurrence_id LIMIT 1""",
                    (GLOBAL_LIBRARY,),
                ).fetchone()
                if (
                    not proof
                    or not c.execute("SELECT 1 FROM st_admins LIMIT 1").fetchone()
                ):
                    raise CloudUnavailable(
                        "Verify the global corpus and assign an administrator before deployment"
                    )
            pine = Pinecone(GLOBAL_LIBRARY)
            pine.validate()
            from app.cloud.sources import checked_copy_vector

            checked_copy_vector(
                pine.fetch([proof["occurrence_id"]]).get(proof["occurrence_id"]),
                dict(
                    space=pine.space,
                    version_id=proof["version_id"],
                    passage_id=proof["passage_id"],
                    source_id=proof["source_id"],
                ),
            )
            verify_directory(default_directory())
            _last = time.monotonic()
    return {
        "status": "ready",
        "storage": "supabase",
        "dense_provider": "pinecone",
        "embedding_provider": "local-onnx",
        "authentication_required": True,
    }
