"""Configuração de logging.

Quatro logs separados, como exigido em produção:

* ``logs/app.log``       - ciclo de vida da aplicação e da API
* ``logs/camera.log``    - conexão, reconexão e captura
* ``logs/ai.log``        - detecção, tracking e contagem
* ``logs/training.log``  - treino/validação/exportação de modelos
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(filename)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _level_from_name(name: str) -> int:
    return getattr(logging, str(name).upper(), logging.INFO)


def setup_logging(logs_dir: Path, level: str = "INFO", *, console: bool = True) -> None:
    """Configura os 4 arquivos de log + console. Idempotente."""
    global _configured
    if _configured:
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(_level_from_name(level))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
    console_fmt = logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)

    targets = ("app", "camera", "ai", "training")
    for name in targets:
        logger = logging.getLogger(name)
        logger.setLevel(_level_from_level_name(level))
        logger.propagate = False

        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / f"{name}.log", maxBytes=8 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if console:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(console_fmt)
            logger.addHandler(sh)

    # Bibliotecas de terceiros: só avisos, e sem duplicar no console.
    for noisy in ("uvicorn.access", "uvicorn.error", "httpx", "httpcore", "apscheduler"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.WARNING)

    # Ultralytics é muito verboso no stdout; deixa só erros.
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    _configured = True


def _level_from_level_name(name: str) -> int:
    return _level_from_name(name)


def get_logger(name: str) -> logging.Logger:
    """Atalho para obter um logger nomeado (app/camera/ai/training)."""
    return logging.getLogger(name)
