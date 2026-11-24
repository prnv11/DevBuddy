from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from app.api import auth, user

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import engine, Base  # Adjust import paths to your project
import app.models  # Ensure models are imported so Base knows about them!

# Tech stacks offered at onboarding. Only frameworks covered by the search index (data/)
# are listed, because answers for any other stack would have no documentation behind them.
SUPPORTED_STACKS = [
    {"name": "Flask", "category": "framework", "description": "Lightweight Python web framework"},
    {"name": "FastAPI", "category": "framework", "description": "Modern Python web framework"},
]


async def seed_tech_stacks():
    """Fill the techstack table on first start (a fresh database has no onboarding options)."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models.techstack import TechStack
    async with AsyncSession(engine) as session:
        existing = set((await session.scalars(select(TechStack.name))).all())
        for ts in SUPPORTED_STACKS:
            if ts["name"] not in existing:
                session.add(TechStack(**ts))
        await session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Recreate missing tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_tech_stacks()
    yield

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="app/templates")

# Mount authentication and user routers
app.include_router(auth.router)
app.include_router(user.router)

@app.get("/")
async def landing_page(request: Request):
    """Renders the main public landing page (index.html)."""
    return templates.TemplateResponse(request, "index.html", {"request": request})

