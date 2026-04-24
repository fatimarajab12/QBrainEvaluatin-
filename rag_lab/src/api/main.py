from __future__ import annotations

from fastapi import FastAPI

from api.routes.base import base_router
from api.routes.chatbot import chatbot_router
from api.routes.features import features_router
from api.routes.projects import projects_router
from api.routes.rag import rag_router
from api.routes.test_cases import test_cases_router

app = FastAPI(title="QBrain RAG API", version="1.0.0")
app.include_router(base_router)
app.include_router(rag_router)
app.include_router(chatbot_router)
app.include_router(projects_router)
app.include_router(features_router)
app.include_router(test_cases_router)
