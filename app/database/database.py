"""Engine, sessão e criação do banco.

SQLite por padrão (``DATABASE_URL``), mas a URL é genérica: trocar para
``postgresql+psycopg://...`` ativa o PostgreSQL sem alterar o código.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database.models import Base

log = logging.getLogger("app")

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _engine_kwargs(url: str) -> dict[str, Any]:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False, "timeout": 30}}
    return {"pool_pre_ping": True, "pool_size": 10, "max_overflow": 20}


def get_engine() -> Engine:
    """Cria a engine uma única vez."""
    global _engine
    if _engine is None:
        url = settings.web.database_url
        db_file = settings.db_path
        if db_file is not None:
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        kwargs = _engine_kwargs(url)
        if url.startswith("sqlite"):
            # check_same_thread=False permite uso em threads (workers de IA).
            _engine = create_engine(url, future=True, **kwargs)

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _rec: Any) -> None:  # pragma: no cover - infra
                cur = dbapi_conn.cursor()
                # WAL: leituras do dashboard não bloqueiam escritas do worker.
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        else:
            _engine = create_engine(url, future=True, **kwargs)
        log.info("Banco de dados conectado: %s", url.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Cria todas as tabelas (idempotente)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    log.info("Esquema do banco verificado/criado (%d tabelas)", len(Base.metadata.tables))


def reset_engine() -> None:
    """Descarta a engine (usado em testes e ao trocar DATABASE_URL)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Sessão transacional: commit no sucesso, rollback no erro."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """Dependência do FastAPI."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


__all__ = [
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_engine",
    "session_scope",
    "get_db",
]
