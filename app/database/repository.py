"""Camada de acesso a dados.

Cada repositório é uma classe pequena e focada, com métodos de uso claro.
Nenhum SQLAlchemy direto vaza para os serviços.
"""

# ===========================================================================
# ARQUIVO / MAPA  -  repository.py
#
# O que faz: a única camada que fala com o banco. Os serviços conhecem estas
# classes; eles nunca escrevem SQL.
#
# Ordem de leitura:
#   1. CameraRepository     - quem é observado
#   2. PileRepository       - identidade da pilha entre frames
#   3. CountRepository      - histórico de contagens (grava SÓ em mudança)
#   4. EventRepository      - log de auditoria
#   5. ModelRepository      - troca/ativação de pesos do YOLO
#   6. CalibrationRepository- ROI e parâmetros por câmera
#   7. SnapshotRepository   - imagens salvas
#   8. CorrectionRepository - discordância do operador (dado de treino)
#
# Dois padrões valem para TODOS os repositórios:
#   * flush() e nunca commit(). O commit é do session_scope de quem chamou, o
#     que faz a gravação da contagem + evento + snapshot ser atômica.
#   * A sessão chega pronta no __init__; o repositório não abre nem fecha nada.
#
# REGRA CENTRAL (a mais importante do arquivo): este módulo GRAVA contagem
# sempre que chamado, mas quem chama decide QUANDO. A comparação
# "o número estabilizado mudou?" fica no InferenceService. Aqui ficam só as
# consultas que Transformam linhas em número (total_at, totals_series, stats).
# ===========================================================================

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import (
    Base,
    Calibration,
    Camera,
    CountCorrection,
    CountRecord,
    ModelVersion,
    Pile,
    Snapshot,
    SystemEvent,
    _utcnow,
)
from app.schemas import CountStatus, EventLevel

log = logging.getLogger("app")


# --------------------------------------------------------------------------- #
# Câmeras
# --------------------------------------------------------------------------- #
class CameraRepository:
    """Câmeras. A linha da câmera é a âncora de todas as outras tabelas."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def get(self, camera_id: int) -> Camera | None:
        return self.s.get(Camera, camera_id)

    def get_default(self) -> Camera | None:
        # is_default desc + id asc: mais de uma pode estar marcada, a mais
        # antiga vence. Sem esse ORDER BY o "padrão" seria aleatório.
        stmt = select(Camera).order_by(Camera.is_default.desc(), Camera.id.asc()).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def list(self, only_enabled: bool = False) -> list[Camera]:
        stmt = select(Camera).order_by(Camera.id.asc())
        if only_enabled:
            stmt = stmt.where(Camera.enabled.is_(True))
        return list(self.s.execute(stmt).scalars())

    def create(self, name: str, rtsp_url: str, enabled: bool = True, is_default: bool = False) -> Camera:
        cam = Camera(name=name, rtsp_url=rtsp_url, enabled=enabled, is_default=is_default)
        self.s.add(cam)
        # flush e não commit: garante o id (chave estrangeira) dentro da mesma
        # transação, sem decidir o destino da transação.
        self.s.flush()
        return cam

    def update(self, camera_id: int, **fields: Any) -> Camera | None:
        cam = self.get(camera_id)
        if cam is None:
            return None
        for key, value in fields.items():
            # Só sobrescreve o que veio preenchido: None significa "não mexeu",
            # senão um update parcial apagaria o campo. hasattr evita aceitar
            # nome de coluna inexistente.
            if value is not None and hasattr(cam, key):
                setattr(cam, key, value)
        self.s.flush()
        return cam

    def delete(self, camera_id: int) -> bool:
        cam = self.get(camera_id)
        if cam is None:
            return False
        # O ON DELETE CASCADE das outras tabelas apaga pilhas, contagens e
        # calibração junto - por isso o PRAGMA foreign_keys=ON no database.py.
        self.s.delete(cam)
        self.s.flush()
        return True

    def ensure_from_env(self, name: str, rtsp_url: str) -> Camera:
        """Garante que exista uma câmera padrão vinda do .env.

        Existe por causa da chave estrangeira: ``piles.camera_id`` e
        ``count_records.camera_id`` apontam para ``cameras.id``. Em banco novo,
        ou depois de a linha da câmera ser apagada, gravar a contagem falharia
        com FOREIGN KEY. Aqui o ciclo se fecha antes do primeiro insert.
        """
        cam = self.get_default()
        if cam is None:
            cam = self.create(name=name, rtsp_url=rtsp_url, is_default=True)
            log.info("Câmera padrão criada a partir do .env: %s", name)
        return cam


# --------------------------------------------------------------------------- #
# Pilhas
# --------------------------------------------------------------------------- #
class PileRepository:
    """Pilhas: mantém a identidade do tracker ligada a uma linha do banco."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def get_by_tracking(self, camera_id: int, tracking_id: int) -> Pile | None:
        # Par (câmera, tracking_id) é a identidade real: o id do banco é
        # interno e muda se a linha for recriada.
        stmt = select(Pile).where(Pile.camera_id == camera_id, Pile.tracking_id == tracking_id)
        return self.s.execute(stmt).scalars().first()

    def touch(self, pile: Pile, *, active: bool = True, stable_count: int | None = None) -> Pile:
        """Atualiza "visto agora" sem criar linha nova."""
        pile.last_seen = _utcnow()
        pile.active = active
        # stable_count=None preserva o valor: quem só quer "revivir" a pilha
        # não deve apagar o número que ela tinha.
        if stable_count is not None:
            pile.stable_count = stable_count
        return pile

    def get_or_create(self, camera_id: int, tracking_id: int) -> Pile:
        pile = self.get_by_tracking(camera_id, tracking_id)
        if pile is None:
            pile = Pile(camera_id=camera_id, tracking_id=tracking_id, first_seen=_utcnow(), last_seen=_utcnow())
            self.s.add(pile)
            self.s.flush()
        return pile

    def list_active(self, camera_id: int) -> list[Pile]:
        stmt = select(Pile).where(Pile.camera_id == camera_id, Pile.active.is_(True))
        return list(self.s.execute(stmt).scalars())

    def list(self, camera_id: int | None = None, limit: int = 200) -> list[Pile]:
        # last_seen desc: o que interessa é a pilha que está no frame agora.
        stmt = select(Pile).order_by(Pile.last_seen.desc()).limit(limit)
        if camera_id is not None:
            stmt = stmt.where(Pile.camera_id == camera_id)
        return list(self.s.execute(stmt).scalars())

    def deactivate_missing(self, camera_id: int, seen_tracking_ids: Sequence[int]) -> list[int]:
        """Sincroniza o estado ``active`` com o que foi visto neste frame.

        Pilhas não vistas são marcadas inativas; pilhas vistas voltam a
        ficar ativas (uma pilha pode sumir por oclusão e reaparecer com o
        mesmo ``tracking_id``). Retorna os tracking_ids que mudaram.
        """
        seen = set(seen_tracking_ids)
        changed: list[int] = []
        # Varre as pilhas da câmera, não as de todas: o "visto neste frame" só
        # tem sentido dentro de uma câmera.
        stmt = select(Pile).where(Pile.camera_id == camera_id)
        for pile in self.s.execute(stmt).scalars():
            # Sentido é nos DOIS lados: reativar é tão importante quanto
            # desativar. Sem o ramo de dentro, uma pilha que voltou de oclusão
            # ficaria inativa para sempre e sumiria do total.
            if pile.tracking_id in seen:
                if not pile.active:
                    pile.active = True
                    changed.append(pile.tracking_id)
            elif pile.active:
                pile.active = False
                changed.append(pile.tracking_id)
        return changed


# --------------------------------------------------------------------------- #
# Contagens
# --------------------------------------------------------------------------- #
class CountRepository:
    """Histórico de contagens.

    Este método :meth:`add` grava SEMPRE que é chamado. Ele não sabe se a
    contagem mudou - quem sabe é o InferenceService, que só chama aqui quando
    o valor estabilizado é diferente do último gravado. Separar as duas coisas é
    proposital: o banco fica com a verdade ("isto mudou") e o serviço com a
    decisão ("isto mudou mesmo?").
    """

    def __init__(self, session: Session) -> None:
        self.s = session

    def add(
        self,
        *,
        camera_id: int,
        pile_id: int | None,
        chair_count: int,
        confidence: float,
        detection_confidence: float = 0.0,
        stability_confidence: float = 0.0,
        status: CountStatus | str = CountStatus.STABLE,
        method: str = "auto",
        total_chairs: int = 0,
        pile_count: int = 1,
        previous_count: int | None = None,
        origin: str = "ai",
    ) -> CountRecord:
        # Aceita enum ou texto: quem chama é a IA (enum) ou um import/script
        # (string). A coluna é texto, então a conversão é feita aqui.
        status_value = status.value if isinstance(status, CountStatus) else str(status)
        rec = CountRecord(
            camera_id=camera_id,
            pile_id=pile_id,
            # int()/float() explícitos: numpy.int64 do contador não entra no
            # SQLite direto, e float32 perderia precisão na coluna Float.
            chair_count=int(chair_count),
            confidence=float(confidence),
            detection_confidence=float(detection_confidence),
            stability_confidence=float(stability_confidence),
            status=status_value,
            method=method,
            # total_chairs e pile_count são o retrato da cena inteira no
            # momento desta linha. Redundantemente, é o que permite ler o
            # gráfico do total sem juntar as pilhas.
            total_chairs=int(total_chairs),
            pile_count=int(pile_count),
            previous_count=previous_count,
            origin=origin,
        )
        self.s.add(rec)
        self.s.flush()
        return rec

    def last_for_pile(self, pile_id: int) -> CountRecord | None:
        # Filtra origin="ai": correção do operador é informação de auditoria,
        # não é o estado atual da contagem. Sem esse filtro o total da tela
        # passaria a depender de alguém ter clicado num botão.
        stmt = (
            select(CountRecord)
            .where(CountRecord.pile_id == pile_id, CountRecord.origin == "ai")
            .order_by(CountRecord.timestamp.desc())
            .limit(1)
        )
        return self.s.execute(stmt).scalar_one_or_none()

    def total_at(self, camera_id: int) -> int:
        """Total de cadeiras na leitura mais recente de cada pilha."""
        # Soma NÃO é só "soma tudo": cada pilha contribui com a ÚLTIMA linha
        # sua (max(id) por pile_id). Sem o group_by, o total cresceria a cada
        # mudança e o banco ficaria impossível de ler.
        sub = (
            select(CountRecord.pile_id, func.max(CountRecord.id).label("last_id"))
            .where(CountRecord.camera_id == camera_id, CountRecord.origin == "ai", CountRecord.pile_id.is_not(None))
            .group_by(CountRecord.pile_id)
            .subquery()
        )
        stmt = select(func.coalesce(func.sum(CountRecord.chair_count), 0)).where(
            CountRecord.id.in_(select(sub.c.last_id))
        )
        # coalesce evita None em câmera sem registro: 0 é o total correto.
        return int(self.s.execute(stmt).scalar_one() or 0)

    def history(
        self,
        camera_id: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        pile_id: int | None = None,
        limit: int = 500,
    ) -> list[CountRecord]:
        # Filtros são todos opcionais e empilhados: a mesma método serve o
        # histórico do gráfico, a janela de uma hora e a pilha de um alerta.
        stmt = select(CountRecord).order_by(CountRecord.timestamp.desc()).limit(limit)
        if camera_id is not None:
            stmt = stmt.where(CountRecord.camera_id == camera_id)
        if since is not None:
            stmt = stmt.where(CountRecord.timestamp >= since)
        if until is not None:
            stmt = stmt.where(CountRecord.timestamp <= until)
        if pile_id is not None:
            stmt = stmt.where(CountRecord.pile_id == pile_id)
        return list(self.s.execute(stmt).scalars())

    def totals_series(self, camera_id: int, since: datetime) -> list[tuple[datetime, int]]:
        """Série (timestamp, total) para o gráfico, reconstruindo o total
        histórico a partir dos deltas de cada pilha.

        ``count_records`` guarda o total no momento da mudança; para o gráfico
        basta interpolar o último total conhecido até o próximo evento.
        """
        # Limit grande de propósito: a série é montada em memória e o gráfico
        # não aceita 500 pontos. Quem chama escolhe a janela por `since`.
        recs = self.history(camera_id=camera_id, since=since, limit=100000)
        # history() devolve do mais novo para o mais antigo; o gráfico precisa
        # na ordem cronológica, daí o reverse.
        recs.reverse()
        series: list[tuple[datetime, int]] = []
        for rec in recs:
            # Descarta o total repetido: várias pilhas mudam no mesmo frame e
            # as linhas carregam o mesmo total. Sem este if, o gráfico teria
            # degraus verticais e mais pontos que mudanças reais.
            if rec.total_chairs and (not series or series[-1][1] != rec.total_chairs):
                series.append((rec.timestamp, int(rec.total_chairs)))
        return series


# --------------------------------------------------------------------------- #
# Eventos
# --------------------------------------------------------------------------- #
class EventRepository:
    """Log de auditoria. Nunca deve interromper o fluxo da contagem."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def log(
        self,
        level: EventLevel | str,
        event: str,
        message: str = "",
        camera_id: int | None = None,
    ) -> SystemEvent:
        # Enum ou texto, como no add() de contagem: normaliza tudo para string.
        lvl = level.value if isinstance(level, EventLevel) else str(level)
        # event[:120] corta no tamanho da coluna: message passes livre (Text),
        # mas "evento" é o índice que se consulta, e não pode estourar.
        row = SystemEvent(level=lvl, event=event[:120], message=message, camera_id=camera_id)
        self.s.add(row)
        self.s.flush()
        return row

    def list(self, limit: int = 300, level: str | None = None) -> list[SystemEvent]:
        stmt = select(SystemEvent).order_by(SystemEvent.timestamp.desc()).limit(limit)
        if level:
            stmt = stmt.where(SystemEvent.level == level)
        return list(self.s.execute(stmt).scalars())

    def purge(self, older_than_days: int = 30) -> int:
        """Apaga eventos antigos. Log cresce sem parar; precisa de faxina."""
        cutoff = _utcnow() - timedelta(days=older_than_days)
        # Carrega e apaga um a um em vez de DELETE em massa: o painel mostra
        # "N eventos removidos" e o log fica com a remoção registrada.
        rows = self.s.execute(select(SystemEvent).where(SystemEvent.timestamp < cutoff)).scalars().all()
        for row in rows:
            self.s.delete(row)
        return len(rows)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
class ModelRepository:
    """Versões de pesos. Regra: existe NO MÁXIMO um modelo ativo."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def list(self) -> list[ModelVersion]:
        stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
        return list(self.s.execute(stmt).scalars())

    def get(self, model_id: int) -> ModelVersion | None:
        return self.s.get(ModelVersion, model_id)

    def get_active(self) -> ModelVersion | None:
        # É a consulta do worker a cada carga/restart do modelo: por isso o
        # índice em ModelVersion.active.
        stmt = select(ModelVersion).where(ModelVersion.active.is_(True)).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def get_by_filename(self, filename: str) -> ModelVersion | None:
        stmt = select(ModelVersion).where(ModelVersion.filename == filename).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def upsert(
        self,
        *,
        filename: str,
        version: str,
        precision: float | None = None,
        recall: float | None = None,
        map50: float | None = None,
        map5095: float | None = None,
        source_path: str | None = None,
        notes: str | None = None,
        base_model: str | None = None,
        epochs: int | None = None,
        dataset_size: int | None = None,
        classes: str | None = None,
        active: bool = False,
    ) -> ModelVersion:
        """Cria ou atualiza a versão identificada pelo nome do arquivo.

        Retreinar o mesmo ``.pt`` não cria linha nova: atualiza as métricas
        no lugar, o que mantém o histórico de "qual modelo está ativo" coerente.
        """
        row = self.get_by_filename(filename)
        if row is None:
            row = ModelVersion(filename=filename, version=version, created_at=_utcnow())
            self.s.add(row)
        # Atribuição direta (e não um update filtrado como no CameraRepository):
        # aqui None significa "métrica desconhecida" e DEVE apagar o valor velho.
        row.version = version
        row.precision = precision
        row.recall = recall
        row.map50 = map50
        row.map5095 = map5095
        row.source_path = source_path
        row.notes = notes
        row.base_model = base_model
        row.epochs = epochs
        row.dataset_size = dataset_size
        row.classes = classes
        if active:
            # Desliga os outros ANTES de ligar este: o estado intermediário
            # (nenhum ativo) é o menos ruim, o contrário nunca acontece.
            self.deactivate_all()
            row.active = True
        self.s.flush()
        return row

    def deactivate_all(self) -> None:
        for row in self.s.execute(select(ModelVersion)).scalars():
            row.active = False

    def set_active(self, model_id: int) -> ModelVersion | None:
        row = self.get(model_id)
        if row is None:
            return None
        self.deactivate_all()
        row.active = True
        self.s.flush()
        return row


# --------------------------------------------------------------------------- #
# Calibração
# --------------------------------------------------------------------------- #
class CalibrationRepository:
    """Calibração: uma linha por câmera (unique em camera_id)."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def get(self, camera_id: int) -> Calibration | None:
        stmt = select(Calibration).where(Calibration.camera_id == camera_id).limit(1)
        return self.s.execute(stmt).scalar_one_or_none()

    def upsert(self, camera_id: int, **fields: Any) -> Calibration:
        """Cria a calibração se não existir, senão sobrescreve os campos vindos."""
        row = self.get(camera_id)
        if row is None:
            # Só camera_id: o resto entra com os default do model (ROI zerada,
            # janela de estabilidade 10, min_confidence 0.55).
            row = Calibration(camera_id=camera_id)
            self.s.add(row)
        for key, value in fields.items():
            # Diferente do update() de câmera: aqui o service manda o dict
            # completo da tela, então None é valor válido (desliga o ROI).
            if hasattr(row, key):
                setattr(row, key, value)
        # updated_at é manual porque só o repository sabe que a linha mudou
        # de verdade; a tela mostra "calibrado há X".
        row.updated_at = _utcnow()
        self.s.flush()
        return row

    def as_dict(self, camera_id: int) -> dict[str, Any]:
        """Calibração no formato que o serviço consome (e a API devolve)."""
        row = self.get(camera_id)
        if row is None:
            # Câmera nunca calibrada: dict vazio, e o serviço cai nos defaults
            # do .env. Não é erro, é o estado normal de uma câmera nova.
            return {}
        # ROI volta como objeto {"x","y","w","h"}: no banco são 4 colunas
        # para poder filtrar, na tela é um retângulo só.
        data: dict[str, Any] = {
            "enabled": row.enabled,
            "roi": {"x": row.roi_x, "y": row.roi_y, "w": row.roi_w, "h": row.roi_h},
            "chair_height_px": row.chair_height_px,
            "confidence_threshold": row.confidence_threshold,
            "stability_frames": row.stability_frames,
            "min_confidence": row.min_confidence,
            "counting_method": row.counting_method,
            "chair_weight": row.chair_weight,
            "pile_gap_px": row.pile_gap_px,
            # timespec="seconds": precisão de milissegundo não ajuda ninguém
            # numa tela, e o JSON fica menor.
            "updated_at": row.updated_at.isoformat(timespec="seconds") if row.updated_at else None,
        }
        if row.params_json:
            try:
                data["params"] = json.loads(row.params_json)
            except json.JSONDecodeError:
                # JSON corrompido não pode derrubar a leitura da calibração:
                # perde-se o extra, os parâmetros importantes continuam válidos.
                data["params"] = {}
        return data


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
class SnapshotRepository:
    """Metadados das imagens salvas. O arquivo em si fica no disco."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def add(
        self,
        *,
        filename: str,
        reason: str = "manual",
        camera_id: int | None = None,
        total: int | None = None,
        pile_count: int | None = None,
        confidence: float | None = None,
        snapshot_size: int | None = None,
    ) -> Snapshot:
        """Registra a foto já gravada em disco.

        ``reason`` é o que dá sentido à imagem: "count_change" é evidência de
        uma contagem que mudou, "low_confidence" é caso difícil para treino.
        """
        row = Snapshot(
            camera_id=camera_id,
            filename=filename,
            reason=reason,
            # total/pile_count/confidence são o rótulo do momento: sem eles a
            # imagem não serve nem para auditoria nem para dataset.
            total=total,
            pile_count=pile_count,
            confidence=confidence,
            snapshot_size=snapshot_size,
        )
        self.s.add(row)
        self.s.flush()
        return row

    def list(self, camera_id: int | None = None, limit: int = 100) -> list[Snapshot]:
        stmt = select(Snapshot).order_by(Snapshot.timestamp.desc()).limit(limit)
        if camera_id is not None:
            stmt = stmt.where(Snapshot.camera_id == camera_id)
        return list(self.s.execute(stmt).scalars())

    def get(self, snapshot_id: int) -> Snapshot | None:
        return self.s.get(Snapshot, snapshot_id)

    def delete(self, snapshot_id: int) -> bool:
        row = self.get(snapshot_id)
        if row is None:
            return False
        # Apaga só a linha. O JPEG em disco é removido pelo SnapshotService,
        # que conhece o caminho - o banco não guarda nada além do nome.
        self.s.delete(row)
        self.s.flush()
        return True


# --------------------------------------------------------------------------- #
# Correções manuais
# --------------------------------------------------------------------------- #
class CorrectionRepository:
    """Correções manuais: o dado mais valioso para melhorar o modelo."""

    def __init__(self, session: Session) -> None:
        self.s = session

    def add(
        self,
        *,
        camera_id: int,
        pile_id: int | None,
        ai_count: int,
        correct_count: int,
        ai_confidence: float = 0.0,
        author: str = "operador",
        note: str | None = None,
        image_path: str | None = None,
    ) -> CountCorrection:
        """Registra que a IA errou e qual era o número certo.

        A diferença (correct - ai) é o erro do modelo. Guardar os dois lados,
        e não só o correto, é o que permite detectar viés sistemático - contar
        sempre 1 a mais, por exemplo - e não apenas falhas isoladas.
        """
        row = CountCorrection(
            camera_id=camera_id,
            pile_id=pile_id,
            ai_count=ai_count,
            correct_count=correct_count,
            ai_confidence=ai_confidence,
            author=author,
            note=note,
            image_path=image_path,
        )
        self.s.add(row)
        self.s.flush()
        return row

    def list(self, limit: int = 200) -> list[CountCorrection]:
        stmt = select(CountCorrection).order_by(CountCorrection.timestamp.desc()).limit(limit)
        return list(self.s.execute(stmt).scalars())

    def stats(self) -> dict[str, Any]:
        """Métricas de erro da IA a partir das correções do operador."""
        rows = self.list(limit=100000)
        if not rows:
            # Nenhuma correção = nenhum dado, e não "modelo perfeito". O
            # chamador precisa ver os zeros explicitamente.
            return {"total": 0, "with_error": 0, "mean_abs_error": 0.0}
        # Erro absoluto: importar o sinal (contar demais vs. de menos) faz o
        # modelo parecer melhor do que é. O viés fica para o suggest_offset.
        errors = [abs(r.correct_count - r.ai_count) for r in rows]
        return {
            "total": len(rows),
            "with_error": sum(1 for e in errors if e > 0),
            "mean_abs_error": round(sum(errors) / len(errors), 3),
            # Fração de acertos exatos: é o número que o operador lê primeiro.
            "exact_match_rate": round(sum(1 for e in errors if e == 0) / len(errors), 4),
        }


__all__ = [
    "CameraRepository",
    "PileRepository",
    "CountRepository",
    "EventRepository",
    "ModelRepository",
    "CalibrationRepository",
    "SnapshotRepository",
    "CorrectionRepository",
]
