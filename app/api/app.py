"""
app/api/app.py
--------------
The FastAPI application: the Database lifespan and the composition root that wires
the real, OpenAI-backed service. This is the ONLY module that constructs the
concrete gateway and the path graph; everything below depends on interfaces.

`build_service` assembles the production service. `create_app` builds the app and
manages one shared Database connection across startup/shutdown via the lifespan.
`create_app` also accepts a `service_factory`, so tests mount the same routes with
a fake service and an in-memory database — no OpenAI key, no network.

Run locally:
    uvicorn app.api.app:app --reload
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.doctor_routes import router as doctor_router
from app.api.patient_routes import ConciergeRuntime, router as patient_router
from app.concierge import ConciergeSafety
from app.config import settings
from app.knowledge import (
    KnowledgeCuratorAgent,
    build_diagnosis_path,
    build_knowledge_agent as build_curator_agent,
    build_reasoning_facade,
)
from app.llm.factory import make_gateway
from app.orchestration.service import ConsultationService
from app.paths.lab_review import LabReviewPath
from app.paths.treatment import TreatmentPath
from app.persistence.database import Database
from app.persistence.store import ConsultationStore
from app.routing import LLMClassifier, Router
from app.safety.guard import SafetyGuard

logger = logging.getLogger(__name__)


def build_service(db: Database) -> ConsultationService:
    """Assemble the production service over a connected Database.

    The gateway is built here from `LLM_PROVIDER` (openai or ollama). OpenAI
    validates its API key in the constructor and Ollama resolves its local models,
    so a misconfigured provider fails loudly at startup. The router registers every
    clinical path (diagnosis, treatment, ...); each is a single `router.register(...)`
    line — that is the whole point of the router. The milestone runs the agentic RAG
    pipeline over PubMed as its evidence source.
    """
    gateway = make_gateway()
    store = ConsultationStore(db)
    guard = SafetyGuard(gateway)
    router_ = Router(LLMClassifier(gateway))
    router_.register(build_diagnosis_path(gateway))
    router_.register(TreatmentPath(gateway))
    router_.register(LabReviewPath(gateway))
    return ConsultationService(store=store, guard=guard, router=router_)


def build_knowledge_agent() -> KnowledgeCuratorAgent | None:
    """Compatibility seam retained for tests and external composition code."""
    return build_curator_agent()


def build_concierge_runtime(db: Database) -> ConciergeRuntime | None:
    """Assemble the remote-patient Concierge channel (its own conversation-owner agent over a
    shared gateway and the same database the doctor reviews). Returns None if no LLM provider is
    configured, so the rest of the API (and the offline tests) still start — the /patient routes
    then report 503 until a provider is set."""
    try:
        gateway = make_gateway()
    except Exception:  # noqa: BLE001 - no/invalid provider: degrade, don't crash the whole API
        logger.warning("Concierge runtime unavailable (no LLM provider configured).")
        return None
    safety = ConciergeSafety(SafetyGuard(gateway))
    return ConciergeRuntime(
        facade=build_reasoning_facade(gateway),
        gateway=gateway,
        store=ConsultationStore(db),
        safety_check=safety,
    )


def create_app(
    *,
    service_factory: Callable[[Database], ConsultationService] = build_service,
    concierge_factory: Callable[[Database], ConciergeRuntime | None] = build_concierge_runtime,
    db_path: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = db_path or settings.DB_PATH
        db = Database(path)
        await db.connect()
        app.state.db = db
        knowledge_agent = build_knowledge_agent()
        app.state.knowledge_agent = knowledge_agent
        app.state.service = service_factory(db)
        if knowledge_agent is not None and hasattr(app.state.service, "set_completion_observer"):
            app.state.service.set_completion_observer(knowledge_agent.submit_state)
        app.state.concierge = concierge_factory(db)
        if (
            knowledge_agent is not None
            and app.state.concierge is not None
            and hasattr(app.state.concierge, "set_knowledge_agent")
        ):
            app.state.concierge.set_knowledge_agent(knowledge_agent)
        logger.info("API started; database at %s", path)
        try:
            yield
        finally:
            if knowledge_agent is not None:
                await knowledge_agent.close()
            await db.close()
            logger.info("API stopped; database closed")

    application = FastAPI(
        title="Medical AI Clinical Decision-Support",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    application.include_router(doctor_router)
    application.include_router(patient_router)

    # Serve the two web UIs (patient chat, doctor console) at /ui/patient.html and
    # /ui/doctor.html. Guarded so a missing web/ folder never breaks app construction.
    web_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
    if os.path.isdir(web_dir):
        application.mount("/ui", StaticFiles(directory=web_dir, html=True), name="ui")
    return application


# The ASGI app uvicorn serves. Importing this module is key-free; the gateway is
# only constructed when the lifespan runs on startup.
app = create_app()
