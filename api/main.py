# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from db.mongo import mongo
from db.postgres import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mongo.connect()
    await init_db()
    yield
    await mongo.disconnect()


app = FastAPI(title="Indian Legal AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import cases, documents, query
app.include_router(cases.router, prefix="/cases", tags=["cases"])
app.include_router(documents.router, prefix="/cases", tags=["documents"])
app.include_router(query.router, prefix="/cases", tags=["query"])


@app.get("/health")
async def health():
    return {"status": "ok"}
