from __future__ import annotations

from config.settings import get_settings


class BaseController:
    def app_info(self) -> dict[str, str]:
        settings = get_settings()
        return {
            "app_name": "QBrain RAG API",
            "app_version": "1.0.0",
            "openai_model": settings.chat_model,
        }
