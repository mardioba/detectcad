"""Engine, sessão e criação do banco.

SQLite por padrão (``DATABASE_URL``), mas a URL é genérica: trocar para
``postgresql+psycopg://...`` ativa o PostgreSQL sem alterar o código.
"""

# ===========================================================================
# ARQUIVO / MAPA  -  database.py
#
# O que faz: a ponte entre a aplicação e o banco. É o único lugar que sabe
# criar a engine, abrir sessão e fechar transação. Nenhuma lógica de contagem
# mora aqui.
#
# Ordem de leitura:
#   1. _engine_kwargs       - opções que mudam por driver (SQLite x PostgreSQL)
#   2. get_engine           - engine única (singleton) + PRAGMAs do SQLite
#   3. get_session_factory  - fábrica de sessões (não grava sozinha)
#   4. init_db              - cria as tabelas que faltarem
#   5. session_scope        - contexto transacional: commit no ok, rollback no erro
#   6. get_db               - dependência do FastAPI (leitura, sem commit)
#   7. reset_engine         - joga a engine fora (testes / troca de DATABASE_URL)
#
# Por que este arquivo existe separado: a regra do projeto é gravar contagem
# SOMENTE quando o número estabilizado muda. A decisão é do InferenceService,
# mas a gravação precisa ser atômica (histórico + evento + snapshot juntos ou
# nada) - e isso é garantido pelo session_scope, não por cada repositório.
# ===========================================================================

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
    """Opções de conexão que dependem do driver.

    SQLite é arquivo local: precisa de ``check_same_thread=False`` porque o
    worker de IA grava em outra thread. PostgreSQL é servidor: precisa de pool
    e ``pool_pre_ping`` para não usar conexão morta.
    """
    if url.startswith("sqlite"):
        # timeout=30: quantos segundos esperar o lock de escrita do SQLite
        # antes de desistir. Sem isso, dashboard e worker travam um no outro.
        return {"connect_args": {"check_same_thread": False, "timeout": 30}}
    return {"pool_pre_ping": True, "pool_size": 10, "max_overflow": 20}


def get_engine() -> Engine:
    """Cria a engine uma única vez."""
    global _engine
    # Singleton: engine nova a cada chamada abriria um pool novo por frame.
    if _engine is None:
        url = settings.web.database_url
        db_file = settings.db_path
        if db_file is not None:
            # O arquivo do SQLite não nasce com a pasta; cria antes de conectar.
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        kwargs = _engine_kwargs(url)
        if url.startswith("sqlite"):
            # check_same_thread=False permite uso em threads (workers de IA).
            _engine = create_engine(url, future=True, **kwargs)

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _rec: Any) -> None:  # pragma: no cover - infra
                # PRAGMAs são por CONEXÃO, então precisam rodar em todo connect.
                cur = dbapi_conn.cursor()
                # WAL: leituras do dashboard não bloqueiam escritas do worker.
                cur.execute("PRAGMA journal_mode=WAL")
                # NORMAL: menos fsync por gravação; seguro o bastante porque
                # perder o último segundo de contagem não é prejuízo real.
                cur.execute("PRAGMA synchronous=NORMAL")
                # foreign_keys=ON: o SQLite ignora chaves estrangeiras por padrão.
                # Sem isso, apagar uma câmera deixaria pilhas e contagens órfãs.
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        else:
            _engine = create_engine(url, future=True, **kwargs)
        # split("@") esconde a senha do RTSP/banco que vem antes do @ na URL.
        log.info("Banco de dados conectado: %s", url.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Fábrica de sessões ligadas à engine.

    ``autoflush=False``: quem decide o flush é o repositório, não o SQLAlchemy
    surprise no meio de uma consulta. ``expire_on_commit=False``: o objeto
    devolvido continua utilizável depois do commit (a API lê e devolve JSON).
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Cria todas as tabelas (idempotente)."""
    engine = get_engine()
    # create_all só cria o que falta: em banco já existente a coluna nova não
    # aparece. Mudança de esquema aqui é assunto de migração, não de arranque.
    Base.metadata.create_all(engine)
    log.info("Esquema do banco verificado/criado (%d tabelas)", len(Base.metadata.tables))


def reset_engine() -> None:
    """Descarta a engine (usado em testes e ao trocar DATABASE_URL)."""
    global _engine, _SessionLocal
    if _engine is not None:
        # dispose fecha as conexões abertas: sem isso o arquivo de teste
        # ficaria preso e o próximo engine apontaria para o banco velho.
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
        # Falhou no meio (ex.: FK violada): desfaz TUDO do bloco, para não
        # deixar contagem gravada sem o evento que a explica.
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """Dependência do FastAPI."""
    # Rotas de leitura não dão commit: quem grava usa session_scope.
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
