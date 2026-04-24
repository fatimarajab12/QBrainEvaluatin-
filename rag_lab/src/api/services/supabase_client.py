from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from config.settings import get_settings


@lru_cache
def get_supabase_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)
