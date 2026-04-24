from __future__ import annotations

from fastapi import APIRouter

from api.controllers.base_controller import BaseController

base_router = APIRouter(prefix="/api/v1", tags=["api_v1"])
base_controller = BaseController()


@base_router.get("/")
def welcome() -> dict[str, str]:
    return base_controller.app_info()


@base_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
