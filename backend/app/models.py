from __future__ import annotations

from pydantic import BaseModel


class SentenceUnitResponse(BaseModel):
    id: str
    text: str
    start_char: int
    end_char: int
    unit_type: str
