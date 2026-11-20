"""Durable progress records; uploads stay in one bounded worker, not a queue."""

import json, sqlite3
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path):
        self.path = path
        with self.connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS product_jobs (id TEXT PRIMARY KEY,payload TEXT NOT NULL)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def create(self, key, filename, options):
        value = {
            "job_id": key,
            "filename": filename,
            "status": "running",
            "stage": "Preparing document",
            "progress": 2,
            "analysis_id": None,
            "error": None,
            "created_at": now(),
            "updated_at": now(),
            "options": options,
        }
        with self.connect() as c:
            c.execute("INSERT INTO product_jobs VALUES(?,?)", (key, json.dumps(value)))
        return value

    def get(self, key):
        with self.connect() as c:
            row = c.execute(
                "SELECT payload FROM product_jobs WHERE id=?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, key, **values):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT payload FROM product_jobs WHERE id=?", (key,)
            ).fetchone()
            if not row:
                return
            value = json.loads(row[0])
            value.update(values, updated_at=now())
            c.execute(
                "UPDATE product_jobs SET payload=? WHERE id=?", (json.dumps(value), key)
            )

    def recover(self):
        with self.connect() as c:
            rows = c.execute("SELECT id,payload FROM product_jobs").fetchall()
            for key, payload in rows:
                value = json.loads(payload)
                if value["status"] == "running":
                    value.update(
                        status="failed",
                        stage="Interrupted",
                        error="The application restarted before this analysis completed. Upload it again.",
                        updated_at=now(),
                    )
                    c.execute(
                        "UPDATE product_jobs SET payload=? WHERE id=?",
                        (json.dumps(value), key),
                    )

    def forget_analysis(self, analysis_id):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            for key, payload in c.execute(
                "SELECT id,payload FROM product_jobs"
            ).fetchall():
                if json.loads(payload).get("analysis_id") == analysis_id:
                    c.execute("DELETE FROM product_jobs WHERE id=?", (key,))
