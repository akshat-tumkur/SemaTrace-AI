from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    collection: str = Field(min_length=1, max_length=100)
    external_id: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=1000)
    url: str = Field(min_length=1, max_length=2000)
    revision: str = Field(min_length=1, max_length=200)
    extractor_version: str = Field(min_length=1, max_length=100)
    authors: tuple[str, ...] = ()
    published_at: datetime | None = None
    modified_at: datetime | None = None
    license_url: str | None = None
    license_name: str | None = None
    rights_status: Literal[
        "UNVERIFIED", "PRIVATE_UPLOAD", "WIKIMEDIA_DEFAULT_LICENSE"
    ] = "UNVERIFIED"
    attribution_url: str | None = None
    attribution_text: str | None = None
    version_url: str | None = None
    extraction_details: dict = Field(default_factory=dict)


@dataclass(frozen=True)
class SourceOccurrence:
    occurrence_id: str
    source_id: str
    source_document_id: str
    source_document_sha256: str
    source_start_char: int
    source_end_char: int
    title: str
    url: str
    collection: str
    revision: str
    authors: tuple[str, ...]
    license_url: str | None
    license_name: str | None
    rights_status: str
    attribution_url: str | None
    attribution_text: str | None


class CorpusUnavailable(RuntimeError):
    """No current validated snapshot; callers must not silently use an empty one."""
