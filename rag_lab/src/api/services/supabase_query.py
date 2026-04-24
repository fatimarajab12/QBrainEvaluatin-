from __future__ import annotations

from typing import Any, Callable, TypeVar
from uuid import UUID

from fastapi import HTTPException
from postgrest.exceptions import APIError  # type: ignore[import-not-found]

T = TypeVar("T")


def parse_uuid(value: str, *, field: str) -> str:
    try:
        return str(UUID(str(value)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field} (expected UUID)") from exc


def map_api_error(exc: APIError) -> HTTPException:
    payload = getattr(exc, "message", None) or str(exc)
    code = None
    if isinstance(exc.args, tuple) and exc.args and isinstance(exc.args[0], dict):
        code = exc.args[0].get("code")
        payload = exc.args[0].get("message") or payload
    if code == "22P02":
        return HTTPException(status_code=422, detail="Invalid UUID in query")
    if code == "PGRST205":
        return HTTPException(
            status_code=503,
            detail="Supabase schema is missing required tables. Run rag_lab/supabase_setup.sql in Supabase SQL editor.",
        )
    return HTTPException(status_code=502, detail=f"Supabase error: {payload}")


def safe_execute(label: str, fn: Callable[[], T]) -> T:
    try:
        return fn()
    except APIError as exc:
        raise map_api_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"{label}: {exc}") from exc
