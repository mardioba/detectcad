"""Configuração de logging.

Quatro logs separados, como exigido em produção:

* ``logs/app.log``       - ciclo de vida da aplicação e da API
* ``logs/camera.log``    - conexão, reconexão e captura
* ``logs/ai.log``        - detecção, tracking e contagem
* ``logs/training.log``  - treino/validação/exportação de modelos
"""

# =============================================================================
# ARQUIVO / MAPA  -  app/logging_config.py
#
# O que faz: cria os 4 arquivos de log + a saída no console, com um único
#   logger pelo nome. Por nome e não por arquivo: o código chama
#   get_logger("camera") e o nível/rotação é resolvido aqui.
#
# Ordem de leitura:
#   1. _level_from_name ...... "INFO"/"DEBUG" -> constante do logging
#   2. setup_logging ........ idempotente, 1 handler por logger, sem duplicar
#   3. _level_from_level_name  apelido de compatibilidade
#   4. get_logger ........... atalho usado pelo resto do código
#
# Efeitos colaterais que importam saber:
#   - "idempotente": a 2ª chamada não toca em nada. O logo do uvicorn, que
#     reconfigura o logging, acaba depois e não apaga os handlers.
#   - "propagate = False": sem isso os mesmos registros cairiam também no
#     root e apareceriam duplicados no app.log.
#   - "RotatingFileHandler": 8 MB x 5 arquivos = no máximo ~40 MB por log.
#     Sem rotação, um stack trace por segundo enche o disco.
#   - Bibliotecas de terceiros são abaixadas para WARNING: log de cada requisição
#     HTTP enche o app.log e esconde o que interessa.
#   - Ultralytics vai para ERROR: ele imprime o treino inteiro em stdout, e
#     o dashboard mostra isso na página.
# =============================================================================

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
    # getattr com default: um LOG_LEVEL inválido no .env vira INFO em vez de
    # derrubar a aplicação na subida.
    return getattr(logging, str(name).upper(), logging.INFO)


def setup_logging(logs_dir: Path, level: str = "INFO", *, console: bool = True) -> None:
    """Configura os 4 arquivos de log + console. Idempotente."""
    global _configured
    if _configured:
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(_level_from_name(level))
    # Remove handlers que já existam (configuração do uvicorn, p.ex.); sem isso
    # quem chama setup_logging duas vezes passa a duplicar cada linha.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Arquivo tem arquivo:lineno, console não (ficaria largo demais).
    formatter = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
    console_fmt = logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)

    targets = ("app", "camera", "ai", "training")
    for name in targets:
        logger = logging.getLogger(name)
        logger.setLevel(_level_from_level_name(level))
        # Sem propagar, o registro não chega ao root - cada arquivo recebe só
        # o que é dele.
        logger.propagate = False

        # Rotação por tamanho: 8 MB por arquivo, guardando 5 = ~40 MB no máximo.
        # O contador zera sozinho quando a configuração é recriada, então os
        # logs não renomeiam a si mesmos a cada start.
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
    # Apelido: existia com esse nome no código antigo. Manter evita caçar
    # todos os call sites do projeto.
    return _level_from_name(name)


def get_logger(name: str) -> logging.Logger:
    """Atalho para obter um logger nomeado (app/camera/ai/training).

    Use sempre com um dos 4 nomes: qualquer outro não terá handler e a
    mensagem some silenciosamente (não vai nem para o console).
    """
    return logging.getLogger(name)
