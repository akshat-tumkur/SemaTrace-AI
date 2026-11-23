"""Owner-scoped persistent reports, jobs and atomic daily usage reservations."""

import json
import os

from fastapi import HTTPException

from app.cloud.client import Blobs, connect, owner_id
from app.product.jobs import now

REPORT_MAX_BYTES = 24 * 1024 * 1024
DAILY_LIMITS = {"analysis": (10, 30), "reference": (3, 10), "ai": (0, 0), "web": (0, 0)}


def quota_limits(operation):
    user, total = DAILY_LIMITS[operation]
    return (
        int(os.getenv("SEMATRACE_" + operation.upper() + "_USER_DAILY_LIMIT", user)),
        int(os.getenv("SEMATRACE_" + operation.upper() + "_GLOBAL_DAILY_LIMIT", total)),
    )


def daily_usage(owner):
    """One read-only snapshot, using the same PostgreSQL day as reserve()."""
    owner = owner_id(owner)
    operations = ["analysis", "reference", "ai"]
    with connect() as c:
        row = c.execute(
            """WITH today AS (
              SELECT subject,operation,used FROM st_usage
              WHERE day=CURRENT_DATE AND subject=ANY(%s) AND operation=ANY(%s)
            ) SELECT CURRENT_DATE AS day,CURRENT_TIMESTAMP AS server_time,
              (CURRENT_DATE+1)::timestamptz AS reset_at,current_setting('TimeZone') AS timezone,
              coalesce((SELECT jsonb_object_agg(operation,used) FROM today WHERE subject=%s),'{}') AS personal,
              coalesce((SELECT jsonb_object_agg(operation,used) FROM today WHERE subject='__global__'),'{}') AS shared""",
            ([owner, "__global__"], operations, owner),
        ).fetchone()
    usage = {}
    for operation in operations:
        user_limit, global_limit = (max(0, n) for n in quota_limits(operation))
        used = max(0, row["personal"].get(operation, 0))
        remaining = max(0, user_limit - used)
        global_remaining = max(0, global_limit - row["shared"].get(operation, 0))
        enabled = user_limit > 0 and global_limit > 0
        if operation == "ai":
            enabled = enabled and bool(os.getenv("OPENAI_API_KEY"))
        if operation == "reference":
            enabled = (
                enabled
                and os.getenv("SEMATRACE_SOURCE_UPLOAD_ENABLED", "true").lower()
                == "true"
            )
        if not enabled:
            status, message = "disabled", "Disabled by this deployment."
        elif remaining == 0:
            status, message = "user_exhausted", "Your daily allowance is exhausted."
        elif global_remaining == 0:
            status, message = "global_exhausted", "Shared daily capacity is exhausted."
        else:
            status, message = "available", ""
        usage[operation] = dict(
            used=used,
            limit=user_limit,
            remaining=remaining,
            global_remaining=global_remaining,
            available=min(remaining, global_remaining) if enabled else 0,
            enabled=enabled,
            status=status,
            message=message,
        )
    return dict(
        day=row["day"],
        server_time=row["server_time"],
        reset_at=row["reset_at"],
        timezone=row["timezone"],
        usage=usage,
    )


def enforce_report_limit(value):
    # The default JSON encoder emits ASCII (including escaped Unicode), so its
    # character count is the exact UTF-8 byte count used by the JSONB adapter.
    # Stop at the bound without allocating a second full string and byte copy.
    size = 0
    for fragment in json.JSONEncoder().iterencode(value):
        size += len(fragment)
        if size > REPORT_MAX_BYTES:
            raise ValueError("Report exceeds cloud storage limit")


def jsonb(value):
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def reserve(owner, operation, user_limit, global_limit):
    """Both reservations commit together; failed work still consumes its budget."""
    owner = owner_id(owner)
    with connect() as c:
        for subject, limit in [("__global__", global_limit), (owner, user_limit)]:
            if limit <= 0:
                raise HTTPException(
                    403, "This external feature is disabled by the operator."
                )
            row = c.execute(
                """INSERT INTO st_usage(subject,operation,used) VALUES(%s,%s,1)
                ON CONFLICT(subject,day,operation) DO UPDATE SET used=st_usage.used+1
                WHERE st_usage.used < %s RETURNING used""",
                (subject, operation, limit),
            ).fetchone()
            if row is None:
                raise HTTPException(429, "Daily usage limit reached; retry tomorrow.")


def reserve_options(owner, options=None, operation="analysis"):
    def take(op):
        reserve(owner, op, *quota_limits(op))

    take(operation)
    if options:
        if options.get("cloud_review"):
            take("ai")
        if options.get("web_search"):
            take("web")


class CloudAnalysisStore:
    completes_jobs_atomically = True

    def __init__(self, owner):
        self.owner = owner_id(owner)

    def save(self, analysis_id, result):
        enforce_report_limit(result)
        with connect() as c:
            c.execute(
                "INSERT INTO st_reports(owner_id,analysis_id,payload,filename,status,created_at) VALUES(%s,%s,%s,%s,%s,coalesce(%s::timestamptz,now()))",
                (
                    self.owner,
                    analysis_id,
                    jsonb(result),
                    result.get("filename", ""),
                    result.get("status", "completed"),
                    result.get("created_at"),
                ),
            )
            job_id = result.get("processing", {}).get("job_id")
            if job_id:
                c.execute(
                    "UPDATE st_jobs SET payload=payload||%s WHERE owner_id=%s AND job_id=%s",
                    (
                        jsonb(
                            dict(
                                status="completed",
                                stage="Report ready",
                                progress=100,
                                analysis_id=analysis_id,
                                updated_at=now(),
                            )
                        ),
                        self.owner,
                        job_id,
                    ),
                )

    def get(self, analysis_id):
        with connect() as c:
            row = c.execute(
                "SELECT payload FROM st_reports WHERE owner_id=%s AND analysis_id=%s",
                (self.owner, analysis_id),
            ).fetchone()
        return row["payload"] if row else None

    def list_reports(self, offset=0, limit=20):
        with connect() as c:
            return c.execute(
                """SELECT analysis_id,filename,status,created_at
                FROM st_reports WHERE owner_id=%s ORDER BY created_at DESC,analysis_id LIMIT %s OFFSET %s""",
                (self.owner, limit, offset),
            ).fetchall()

    def delete(self, analysis_id):
        with connect() as c:
            row = c.execute(
                "DELETE FROM st_reports WHERE owner_id=%s AND analysis_id=%s RETURNING analysis_id",
                (self.owner, analysis_id),
            ).fetchone()
            c.execute(
                "DELETE FROM st_jobs WHERE owner_id=%s AND payload->>'analysis_id'=%s",
                (self.owner, analysis_id),
            )
        return bool(row)

    def _section(self, analysis_id, section, value):
        with connect() as c:
            row = c.execute(
                "UPDATE st_reports SET payload=jsonb_set(payload,%s,%s,true) WHERE owner_id=%s AND analysis_id=%s RETURNING analysis_id",
                ([section], jsonb(value), self.owner, analysis_id),
            ).fetchone()
        return bool(row)

    def update_similarity_report(self, analysis_id, report):
        return self._section(analysis_id, "similarity_report", report)

    def save_product_section(self, analysis_id, section, value):
        if section not in {"semantic_assessment", "web_discovery", "processing"}:
            raise ValueError("Invalid product section")
        return self._section(analysis_id, section, value)

    def save_semantic_review(self, analysis_id, kind, key, value):
        if kind not in {"rankings", "investigations"}:
            raise ValueError("Invalid review section")
        with connect() as c:
            row = c.execute(
                "SELECT payload FROM st_reports WHERE owner_id=%s AND analysis_id=%s FOR UPDATE",
                (self.owner, analysis_id),
            ).fetchone()
            if not row:
                return False
            payload = row["payload"]
            section = payload.setdefault("semantic_review", {}).setdefault(kind, {})
            if key not in section and len(section) >= 100:
                raise ValueError("Saved review limit reached")
            section[key] = value
            c.execute(
                "UPDATE st_reports SET payload=%s WHERE owner_id=%s AND analysis_id=%s",
                (jsonb(payload), self.owner, analysis_id),
            )
        return True


class CloudJobStore:
    def __init__(self, owner):
        self.owner = owner_id(owner)

    def create(self, key, filename, options):
        value = dict(
            job_id=key,
            filename=filename,
            status="running",
            stage="Preparing document",
            progress=2,
            analysis_id=None,
            error=None,
            created_at=now(),
            updated_at=now(),
            options=options,
        )
        with connect() as c:
            c.execute(
                "INSERT INTO st_jobs VALUES(%s,%s,%s)", (self.owner, key, jsonb(value))
            )
        return value

    def persist_upload(self, key, raw):
        fingerprint = Blobs(self.owner).put(raw)
        self.update(key, upload_sha256=fingerprint)

    def get(self, key):
        with connect() as c:
            row = c.execute(
                "SELECT payload FROM st_jobs WHERE owner_id=%s AND job_id=%s",
                (self.owner, key),
            ).fetchone()
        return row["payload"] if row else None

    def update(self, key, **values):
        values["updated_at"] = now()
        with connect() as c:
            c.execute(
                "UPDATE st_jobs SET payload=payload||%s WHERE owner_id=%s AND job_id=%s "
                "AND NOT (payload->>'status'='completed' AND %s='failed')",
                (jsonb(values), self.owner, key, values.get("status", "")),
            )

    def forget_analysis(self, analysis_id):
        with connect() as c:
            c.execute(
                "DELETE FROM st_jobs WHERE owner_id=%s AND payload->>'analysis_id'=%s",
                (self.owner, analysis_id),
            )

    @staticmethod
    def recover():
        # One Render instance/worker only. Preserve uploaded content and failed
        # job records; completed jobs and reports are never changed on restart.
        with connect() as c:
            c.execute(
                """UPDATE st_jobs SET payload=payload||%s WHERE payload->>'status'='running'""",
                (
                    jsonb(
                        dict(
                            status="failed",
                            stage="Interrupted",
                            error="The backend restarted. Your upload is retained; submit it again to run a new analysis.",
                            updated_at=now(),
                        )
                    ),
                ),
            )


def jobs_for(store):
    if isinstance(store, CloudAnalysisStore):
        return CloudJobStore(store.owner)
    from app.product.jobs import JobStore

    return JobStore(store.database_path.with_suffix(".jobs.sqlite3"))
