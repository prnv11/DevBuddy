from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .. import schemas, models
from ..database import get_db
from ..utils import get_current_user
from app.rag.pipeline.router import build_router

router = APIRouter(prefix="/user", tags=["user"])
templates = Jinja2Templates(directory="app/templates")

# Instantiate the query classifier & RAG chain router once
rag_router = build_router()

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    current_user: Annotated[schemas.user.User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Renders the main user dashboard with tech stack context and history."""
    stmt = (
        select(models.conversation.Conversation)
        .where(models.conversation.Conversation.user_id == current_user.id)
        .order_by(models.conversation.Conversation.created_at.desc())
        .limit(10)
    )
    history = (await db.scalars(stmt)).all()

    return templates.TemplateResponse(
request, "dashboard.html", {
"request": request,
            "user": current_user,
            "history": history
        }
    )

@router.post("/chat")
async def query_mapping_engine(
    question: Annotated[str, Form()],
    current_user: Annotated[schemas.user.User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Receives questions from dashboard UI and returns RAG/Mapping answers with cited sources."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Retrieval (hybrid search or cross-framework concept alignment) + answer composition.
    #    Runs in a worker thread so the event loop is not blocked by scoring / LLM calls.
    result = await run_in_threadpool(rag_router.invoke, {
        "question": question,
        "primary": current_user.primary_tech_stack,
        "learning": current_user.learning_tech_stack,
    })
    answer = result["response"]

    # 2. Persist conversation entry
    chat_record = models.conversation.Conversation(
        user_id=current_user.id,
        question=question,
        response=answer
    )
    db.add(chat_record)
    await db.commit()

    return {"question": question, "response": answer, "mode": result["mode"],
            "route": result["route"], "sources": result["sources"]}


class RefineRequest(BaseModel):
    question: str
    relevant: list[str] = []
    non_relevant: list[str] = []


@router.post("/refine")
async def refine_answer(
    body: RefineRequest,
    current_user: Annotated[schemas.user.User, Depends(get_current_user)],
):
    """Rocchio relevance feedback: re-rank the sources using the user's thumbs up / down."""
    if not body.relevant and not body.non_relevant:
        raise HTTPException(status_code=400, detail="Mark at least one source as relevant or not relevant.")
    result = await run_in_threadpool(rag_router.refine, {
        "question": body.question, "relevant": body.relevant, "non_relevant": body.non_relevant,
        "primary": current_user.primary_tech_stack, "learning": current_user.learning_tech_stack,
    })
    return {"question": body.question, **result}