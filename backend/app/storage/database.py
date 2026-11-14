from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[3] / "data" / "app.db"
VERIFICATION_FIELDS = (
    "document",
    "report_synthesis",
    "similarity_report",
    "corpus",
    "created_at",
    "legacy_coverage_percent",
    "verified_coverage_percent",
    "risk_score_method",
    "partial_overlaps",
    "manual_review_count",
    "evidence_categories",
    "verification",
)
PRODUCT_FIELDS = (
    "report_version",
    "findings",
    "related_sources",
    "retrieval",
    "alignment_sources",
    "semantic_review",
    "semantic_assessment",
    "web_discovery",
    "processing",
)


def _sqlite_path_from_url() -> str | None:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith("sqlite:///"):
        return None
    configured_path = database_url.removeprefix("sqlite:///")
    if configured_path.startswith("./"):
        return str(Path.cwd() / configured_path[2:])
    return configured_path


class AnalysisStore:
    def __init__(self, database_path: str | Path | None = None) -> None:
        configured_path = (
            database_path
            or os.getenv("SEMATRACE_DB_PATH")
            or _sqlite_path_from_url()
            or DEFAULT_DATABASE_PATH
        )
        self.database_path = Path(configured_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    coverage_percent REAL NOT NULL,
                    strong_matches INTEGER NOT NULL,
                    possible_paraphrases INTEGER NOT NULL,
                    false_positives INTEGER NOT NULL,
                    candidate_count INTEGER NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    recommendations TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS units (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                    unit_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    unit_type TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(analyses)")
            }
            if "summary" not in columns:
                connection.execute(
                    "ALTER TABLE analyses ADD COLUMN summary TEXT NOT NULL DEFAULT ''"
                )
            if "recommendations" not in columns:
                connection.execute(
                    "ALTER TABLE analyses ADD COLUMN recommendations TEXT NOT NULL DEFAULT '[]'"
                )
            if "verification_metadata" not in columns:
                connection.execute(
                    "ALTER TABLE analyses ADD COLUMN verification_metadata TEXT NOT NULL DEFAULT '{}'"
                )
            unit_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(units)")
            }
            if "following_citation_spans" not in unit_columns:
                connection.execute(
                    "ALTER TABLE units ADD COLUMN following_citation_spans TEXT NOT NULL DEFAULT '[]'"
                )

    def save(self, analysis_id: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO analyses
                (id, filename, file_type, status, mode, risk_level, risk_score,
                 coverage_percent, strong_matches, possible_paraphrases, false_positives, candidate_count, summary, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    result["filename"],
                    result["file_type"],
                    result["status"],
                    result["mode"],
                    result["risk_level"],
                    result["risk_score"],
                    result["coverage_percent"],
                    result["strong_matches"],
                    result["possible_paraphrases"],
                    result["false_positives"],
                    result["candidate_count"],
                    result.get("summary", ""),
                    json.dumps(result.get("recommendations", [])),
                ),
            )
            connection.execute(
                "UPDATE analyses SET verification_metadata = ? WHERE id = ?",
                (
                    json.dumps(
                        {
                            key: result[key]
                            for key in VERIFICATION_FIELDS + PRODUCT_FIELDS
                            if key in result
                        }
                    ),
                    analysis_id,
                ),
            )
            connection.executemany(
                "INSERT INTO units (analysis_id, unit_key, text, start_char, end_char, unit_type, following_citation_spans) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        analysis_id,
                        unit["id"],
                        unit["text"],
                        unit["start_char"],
                        unit["end_char"],
                        unit["unit_type"],
                        json.dumps(unit.get("following_citation_spans", [])),
                    )
                    for unit in result.get("units", [])
                ],
            )
            connection.executemany(
                "INSERT INTO matches (analysis_id, payload) VALUES (?, ?)",
                [
                    (analysis_id, json.dumps(match))
                    for match in result.get("matches", [])
                ],
            )
            connection.executemany(
                "INSERT INTO audit_events (analysis_id, sequence, payload) VALUES (?, ?, ?)",
                [
                    (analysis_id, index, json.dumps(event))
                    for index, event in enumerate(result.get("audit_log", []))
                ],
            )

    def update_similarity_report(
        self, analysis_id: str, report: dict[str, Any]
    ) -> bool:
        """Atomically update only the additive metric, leaving legacy values intact."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT verification_metadata FROM analyses WHERE id = ?",
                (analysis_id,),
            ).fetchone()
            if row is None:
                return False
            metadata = json.loads(row["verification_metadata"])
            metadata["similarity_report"] = report
            connection.execute(
                "UPDATE analyses SET verification_metadata = ? WHERE id = ?",
                (json.dumps(metadata), analysis_id),
            )
        return True

    def list_reports(self, offset: int = 0, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,filename,status,verification_metadata FROM analyses ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "analysis_id": r["id"],
                "filename": r["filename"],
                "status": r["status"],
                "created_at": json.loads(r["verification_metadata"]).get("created_at"),
            }
            for r in rows
        ]

    def save_semantic_review(self, analysis_id, kind, key, value):
        if kind not in ("rankings", "investigations"):
            raise ValueError("Invalid review section")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT verification_metadata FROM analyses WHERE id=?", (analysis_id,)
            ).fetchone()
            if row is None:
                return False
            metadata = json.loads(row["verification_metadata"])
            review = metadata.setdefault("semantic_review", {})
            section = review.setdefault(kind, {})
            if key not in section and len(section) >= 100:
                raise ValueError("Saved review limit reached")
            section[key] = value
            connection.execute(
                "UPDATE analyses SET verification_metadata=? WHERE id=?",
                (json.dumps(metadata), analysis_id),
            )
        return True

    def save_product_section(self, analysis_id, key, value):
        if key not in ("semantic_assessment", "web_discovery"):
            raise ValueError("Invalid product section")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT verification_metadata FROM analyses WHERE id=?", (analysis_id,)
            ).fetchone()
            if row is None:
                return False
            metadata = json.loads(row["verification_metadata"])
            metadata[key] = value
            connection.execute(
                "UPDATE analyses SET verification_metadata=? WHERE id=?",
                (json.dumps(metadata), analysis_id),
            )
        return True

    def delete(self, analysis_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "DELETE FROM analyses WHERE id=?", (analysis_id,)
                ).rowcount
                == 1
            )

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            analysis = connection.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            if analysis is None:
                return None
            units = connection.execute(
                "SELECT unit_key, text, start_char, end_char, unit_type, following_citation_spans FROM units WHERE analysis_id = ? ORDER BY id",
                (analysis_id,),
            ).fetchall()
            matches = connection.execute(
                "SELECT payload FROM matches WHERE analysis_id = ? ORDER BY id",
                (analysis_id,),
            ).fetchall()
            events = connection.execute(
                "SELECT payload FROM audit_events WHERE analysis_id = ? ORDER BY sequence",
                (analysis_id,),
            ).fetchall()
        result = {
            "analysis_id": analysis_id,
            "filename": analysis["filename"],
            "file_type": analysis["file_type"],
            "status": analysis["status"],
            "mode": analysis["mode"],
            "risk_level": analysis["risk_level"],
            "risk_score": analysis["risk_score"],
            "coverage_percent": analysis["coverage_percent"],
            "strong_matches": analysis["strong_matches"],
            "possible_paraphrases": analysis["possible_paraphrases"],
            "false_positives": analysis["false_positives"],
            "candidate_count": analysis["candidate_count"],
            "summary": analysis["summary"],
            "recommendations": json.loads(analysis["recommendations"]),
            "units": [
                {
                    "id": unit["unit_key"],
                    "text": unit["text"],
                    "start_char": unit["start_char"],
                    "end_char": unit["end_char"],
                    "unit_type": unit["unit_type"],
                    "following_citation_spans": json.loads(
                        unit["following_citation_spans"]
                    ),
                }
                for unit in units
            ],
            "units_analyzed": len(units),
            "matches": [json.loads(match["payload"]) for match in matches],
            "audit_log": [json.loads(event["payload"]) for event in events],
            **json.loads(analysis["verification_metadata"]),
        }
        if result.get("report_version") == "3.0":
            return result
        from app.compat.evidence import refresh_report_support

        return refresh_report_support(result)
