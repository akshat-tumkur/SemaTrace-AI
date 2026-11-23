"""Library identities and database-controlled administrator authority."""

from fastapi import HTTPException
from app.cloud.client import connect, owner_id, CloudUnavailable

# A library identity, never an Auth user or the lifetime of an administrator.
GLOBAL_LIBRARY = "00000000-0000-0000-0000-000000000001"
GLOBAL_NAMESPACE = "global-v1"


def is_admin(actor):
    actor = owner_id(actor)
    with connect() as c:
        if not c.execute(
            "SELECT to_regclass('public.st_admins') AS role_table"
        ).fetchone()["role_table"]:
            raise CloudUnavailable(
                "Shared-library schema is not installed. Run scripts/cloud_admin.py migrate before importing global references."
            )
        return bool(
            c.execute("SELECT 1 FROM st_admins WHERE user_id=%s", (actor,)).fetchone()
        )


def require_admin(actor):
    if not is_admin(actor):
        raise HTTPException(
            403, "Administrator access is required to modify the global corpus."
        )


def ensure_private_library(actor):
    actor = owner_id(actor)
    with connect() as c:
        if not c.execute(
            "SELECT to_regclass('public.st_libraries') AS library_table"
        ).fetchone()["library_table"]:
            raise CloudUnavailable(
                "Shared-library schema is not installed. Run scripts/cloud_admin.py migrate first."
            )
        c.execute(
            "INSERT INTO st_libraries(library_id,kind,user_id) VALUES(%s,'private',%s) ON CONFLICT DO NOTHING",
            (actor, actor),
        )


def library_scope(key):
    return "global" if str(key) == GLOBAL_LIBRARY else "private"
