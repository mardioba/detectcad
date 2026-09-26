"""Gravação e listagem de snapshots.

Um snapshot é uma imagem JPEG com o overlay das contagens, gravada quando:

* a contagem de uma pilha muda;
* uma pilha aparece ou desaparece;
* a confiança cai abaixo do limite;
* a câmera reconecta;
* o operador pede (botão no dashboard ou ``POST /api/snapshots``);
* um intervalo configurável decorre (``SNAPSHOT_INTERVAL_SEC``).

As imagens são a matéria-prima do dataset: o operador pode copiar as imagens
úteis para ``training/datasets/raw/`` e anotá-las.
"""

# ARQUIVO / MAPA
# Gravação de JPEG de auditoria. Duas saídas por snapshot: um arquivo em disco
# (que dá para abrir) e uma linha na tabela `snapshots` (que dá para filtrar).
#
# Ordem de leitura:
#   1. REASONS        -> dicionário de motivo -> texto legível
#   2. __init__/_filename -> nome do arquivo contém o motivo, sem colisão
#   3. save           -> o caminho feliz, com registro no banco
#   4. maybe_interval -> o único caminho com lock (evita rajada de snapshot)
#   5. resolve        -> segurança de path na leitura
#   6. purge_old      -> limpeza por data
#
# Quem CHAMA save: o InferenceService (por evento de contagem), o
# CountingService (correção manual) e a API (botão do operador).

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.database.database import session_scope
from app.database.repository import SnapshotRepository

log = logging.getLogger("app")

REASONS = {
    # Tradução de cada `reason` para texto de tela. É um dicionário fechado
    # (e não um campo livre no banco) para que o painel possa agrupar por
    # motivo sem depender do que cada chamada passou.
    "manual": "solicitado pelo operador",
    "count_change": "contagem mudou",
    "pile_appear": "nova pilha",
    "pile_disappear": "pilha desapareceu",
    "low_confidence": "confiança baixa",
    "camera_reconnect": "câmera reconectou",
    "interval": "intervalo periódico",
    "correction": "correção manual de contagem",
}


class SnapshotService:
    """Grava e gerencia snapshots no disco e no banco."""

    def __init__(self, camera_id: int = 1, camera_name: str = "camera1") -> None:
        self.camera_id = camera_id
        # Slug já no __init__: entra no nome do arquivo, então precisa estar
        # saneado uma vez só, não a cada gravação.
        self.camera_name = _slug(camera_name)
        # Marca do último snapshot de intervalo. Precisa ser atributo de
        # instância, não global: com duas câmeras, um intervalo não pode
        # desativar o snapshot da outra.
        self._last_interval = 0.0
        # Lock só do cronômetro de intervalo: protects maybe_interval sem
        # serializar as gravações por evento (que são raras).
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        d = settings.snapshots_dir
        # Criado no accessor: a pasta pode ser apagada entre duas gravações
        # (limpeza manual) e o próximo snapshot ainda precisa funcionar.
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _filename(self, when: datetime, reason: str) -> str:
        stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
        # Timestamp só com segundos: dois eventos na mesma segunda geram
        # arquivos com o mesmo nome e um sobrescreve o outro. Aceito porque a
        # janela de 1 s é curta e o conteúdo é praticamente o mesmo frame.
        return f"{stamp}_{self.camera_name}_{reason}.jpg"

    def save(
        self,
        frame: np.ndarray | None,
        *,
        reason: str = "manual",
        total: int | None = None,
        pile_count: int | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any] | None:
        """Salva um frame. Devolve ``None`` se não havia frame."""
        # size == 0 também conta como "sem frame": um array vazio no imwrite
        # levanta exceção e derrubaria o worker.
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        when = datetime.now()
        name = self._filename(when, reason)
        path = self.directory / name
        try:
            # Qualidade vem do .env (padrão 85): é a foto de auditoria, que
            # vai ser olhada de perto e serve de dataset. O stream do
            # dashboard usa 82 à parte.
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(settings.snapshots.jpeg_quality)]
            ok = cv2.imwrite(str(path), frame, params)
        except Exception as exc:
            # Gravar snapshot é acessório: falhar aqui não pode parar a
            # contagem. Log e segue.
            log.error("Falha ao gravar snapshot: %s", exc)
            return None
        if not ok:
            # imwrite devolve False (em vez de levantar) em disco cheio ou
            # caminho inválido. Precisa ser checado por fora.
            log.error("cv2.imwrite falhou para %s", path)
            return None

        # stat no arquivo gravado: o tamanho real em disco, que é o que
        # interessa para estimar o espaço ocupado.
        size = path.stat().st_size
        try:
            # Banco é o índice do arquivo. Se a gravação no banco falhar, o
            # JPEG continua existindo e aparece no log de erros.
            with session_scope() as session:
                SnapshotRepository(session).add(
                    filename=name,
                    reason=reason,
                    camera_id=self.camera_id,
                    # Contexto do momento da foto: o JPEG sozinho não diz
                    # quantas cadeiras havia.
                    total=total,
                    pile_count=pile_count,
                    confidence=confidence,
                    snapshot_size=size,
                )
        except Exception:
            log.exception("Não foi possível registrar o snapshot no banco")

        log.info("Snapshot salvo: %s (%s, %.1f KB)", name, REASONS.get(reason, reason), size / 1024)
        return {
            "filename": name,
            "path": str(path),
            "reason": reason,
            "reason_label": REASONS.get(reason, reason),
            "size_kb": round(size / 1024, 1),
            "timestamp": when.isoformat(timespec="seconds"),
            # URL relativa: a API serve o arquivo, o browser monta a absoluta.
            "url": f"/api/snapshots/file/{name}",
        }

    def maybe_interval(self, frame: np.ndarray | None, **kwargs: Any) -> dict[str, Any] | None:
        """Salva se ``SNAPSHOT_INTERVAL_SEC > 0`` e o intervalo passou."""
        interval = settings.snapshots.interval_sec
        if not settings.snapshots.save_snapshots or interval <= 0 or frame is None:
            return None
        now = time.time()
        with self._lock:
            # Verificação e atualização do carimbo dentro do lock: sem isso,
            # dois threads que chamassem no mesmo instante passariam as duas
            # e gravariam o mesmo snapshot duas vezes.
            if now - self._last_interval < interval:
                return None
            self._last_interval = now
        # O I/O fica FORA do lock: segurar a gravação inteira impediria o
        # snapshot de evento de passar.
        return self.save(frame, reason="interval", **kwargs)

    def resolve(self, filename: str) -> Path | None:
        """Resolve um nome de arquivo dentro do diretório de snapshots.

        Protege contra path traversal (``../``).
        """
        # .name descarta diretórios: "../../etc/passwd" vira "passwd" antes de
        # qualquer concatenação de caminho.
        name = Path(filename).name
        path = (self.directory / name).resolve()
        # Defense in depth: mesmo com o .name, confirma que o resultado está
        # dentro do diretório de snapshots. Um symlink apontando para fora
        # também cai aqui.
        try:
            path.relative_to(self.directory.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None

    def purge_old(self, keep_days: int | None = None) -> int:
        """Remove snapshots antigos. ``keep_days=0`` (padrão) desativa.

        Apaga SÓ os .jpg. As linhas da tabela `snapshots` ficam órfãs de
        propósito: o histórico de auditoria não deve sumir junto com a imagem.
        """
        days = settings.snapshots.keep_days if keep_days is None else keep_days
        if days <= 0:
            return 0
        # 86400 s = 1 dia.
        cutoff = time.time() - days * 86400
        removed = 0
        for path in self.directory.glob("*.jpg"):
            try:
                # mtime, não o nome do arquivo: o nome tem só o segundo de
                # precisão e o mtime é o que o sistema de arquivos sabe.
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                # Arquivo sumiu entre glob e stat (ou está bloqueado no
                # Windows): segue para o próximo.
                continue
        if removed:
            log.info("%d snapshots antigos removidos", removed)
        return removed


def _slug(text: str) -> str:
    """Nome de arquivo seguro para a câmera: só alfanumérico."""
    # Qualquer coisa fora de [a-z0-9] (incluindo acento, espaço e ponto) vira
    # "_". O "or camera1" cobre o caso de nome vazio ou só pontuação.
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower() or "camera1"


__all__ = ["SnapshotService", "REASONS"]
