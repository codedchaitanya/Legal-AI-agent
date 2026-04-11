# db/postgres.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import AsyncGenerator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, ForeignKey, Text, Float, JSON
from datetime import datetime, timezone
from typing import Optional
from core.config import settings


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    lawyer_id: Mapped[str] = mapped_column(String(100))
    domains: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    documents: Mapped[list["Document"]] = relationship(back_populates="case")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"))
    doc_type: Mapped[str] = mapped_column(String(50))
    filename: Mapped[str] = mapped_column(String(255))
    page_count: Mapped[int] = mapped_column(default=0)
    storage_path: Mapped[str] = mapped_column(String(500))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    case: Mapped["Case"] = relationship(back_populates="documents")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(50))
    session_id: Mapped[str] = mapped_column(String(100))
    query_type: Mapped[str] = mapped_column(String(30))
    query: Mapped[str] = mapped_column(Text)
    adapters_used: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    api_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


engine = create_async_engine(settings.POSTGRES_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
