"""Tabelas do banco (SQLAlchemy 2.0, anotado por estilo declarativo).

Esquema:

* ``cameras``        - câmeras cadastradas (preparado para multicâmera)
* ``piles``          - pilhas detectadas, com identidade estável
* ``count_records``  - histórico de contagens (gravado só em mudanças reais)
* ``model_versions`` - versões de modelo YOLO e métricas reais de treino
* ``system_events``  - log de eventos do sistema
* ``calibration``    - ROI e parâmetros por câmera
* ``snapshots``      - imagens salvas para auditoria/dataset
* ``count_corrections`` - correções manuais do operador (IA vs. humano)
"""

# ===========================================================================
# ARQUIVO / MAPA  -  models.py
#
# O que faz: as 8 tabelas do sistema, em SQLAlchemy 2.0 anotado. Só descreve o
# formato; quem preenche é o repository.py.
#
# Ordem de leitura:
#   1. Base / _utcnow    - base comum e o relógio (UTC sem fuso) de todo mundo
#   2. Camera            - quem é observado; pai de pilhas, contagens, calibração
#   3. Pile              - a pilha e sua identidade entre frames (tracking_id)
#   4. CountRecord       - histórico de contagens (SÓ quando o número muda)
#   5. ModelVersion      - modelo YOLO em uso + métricas do treino
#   6. SystemEvent       - log de auditoria da aplicação
#   7. Calibration       - ROI e parâmetros de contagem por câmera
#   8. Snapshot          - imagens salvas (auditoria / dataset)
#   9. CountCorrection   - o que o operador discordou da IA
#
# Regras de ouro do esquema:
#   * ``Camera`` é a âncora. Deletar a câmera leva junto pilhas, contagens e
#     calibração (ondelete="CASCADE"). Por isso o InferenceService garante a
#     câmera via .env antes de gravar: sem a linha, a FK derruba a gravação.
#   * Datas são UTC "ingênuo" (sem tzinfo) porque o DateTime do SQLite não
#     guarda fuso; o _utcnow() converte uma vez só, na entrada.
#   * Índice quase sempre é (chave, tempo): o dashboard pergunta "o que mudou
#     nesta câmera depois de tal hora" o tempo todo.
# ===========================================================================

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base única: o metadata dela reúne as 8 tabelas em um create_all só."""

    pass


def _utcnow() -> datetime:
    """Agora em UTC, sem fuso (compatível com o DateTime do SQLite)."""
    # tzinfo é removido de propósito: o SQLite devolveria o mesmo valor com
    # fuso e a comparação com datetime "ingênuo" do Python quebraria.
    # Todo o sistema conta em UTC; a conversão para hora local é da tela.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Camera(Base):
    """Câmera IP/RTSP cadastrada."""

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    # String(1024): URL RTSP com usuário, senha e query string é longa.
    rtsp_url: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # is_default ordena na consulta da câmera padrão; mais de uma pode existir
    # marcada, a de menor id vence (ver CameraRepository.get_default).
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)

    piles: Mapped[list["Pile"]] = relationship(
        # delete-orphan: sem uma Pile ela não existe. passive_deletes diz ao
        # SQLAlchemy para deixar o ON DELETE do banco fazer o trabalho.
        back_populates="camera", cascade="all, delete-orphan", passive_deletes=True
    )


class Pile(Base):
    """Pilha identificada. ``tracking_id`` mantém a identidade entre frames."""

    __tablename__ = "piles"
    __table_args__ = (
        # (câmera, ativa): lista as pilhas do frame atual, a consulta mais frequente.
        Index("ix_piles_camera_active", "camera_id", "active"),
        # (câmera, tracking_id): o tracker reidentifica a pilha por este par.
        Index("ix_piles_camera_track", "camera_id", "tracking_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    # ID do tracker (ByteTrack/BoT-SORT). Vale para os dois lados: é o que
    # permite uma pilha sumir por oclusão e voltar sem virar pilha nova.
    tracking_id: Mapped[int] = mapped_column(Integer, nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    # active=False = pilha não vista neste frame. Não é exclusão: o histórico
    # dela continua válido e ela pode reativar com o mesmo tracking_id.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Último número ESTABILIZADO, não o do frame. Guarda a foto de "onde parou"
    # para a próxima pilha... e para reconciliar o banco com a memória.
    stable_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    camera: Mapped[Camera] = relationship(back_populates="piles")


class CountRecord(Base):
    """Registro de contagem.

    Só é gravado quando a contagem **muda de fato** (ver
    :class:`app.services.inference_service.InferenceService`), evitando
    milhares de linhas idênticas.
    """

    __tablename__ = "count_records"
    __table_args__ = (
        # Dois índices porque há dois jeitos de ler a tabela: gráfico do total
        # da câmera, e histórico de uma pilha. Ambos ordenam por tempo.
        Index("ix_count_records_camera_ts", "camera_id", "timestamp"),
        Index("ix_count_records_pile_ts", "pile_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    # NULL = leitura sem pilha associada (ex.: contagem total avulsa). Por isso
    # os índices de pilha são separados: não dá para indexar por ele sempre.
    pile_id: Mapped[int | None] = mapped_column(
        ForeignKey("piles.id", ondelete="CASCADE"), nullable=True, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    chair_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # confidence é o valor já fundido dos estimadores; os dois abaixo são as
    # parcelas, guardadas para poder auditar POR QUE a contagem foi aceita.
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stability_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Texto e não Enum do SQLAlchemy: o enum é o CountStatus do schemas.py e
    # mudar a lista de estados não pode exigir migração de banco.
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="STABLE")
    # Qual estimador venceu: "auto", "periodic", "extent", "detections", "size".
    method: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    # Snapshot do total e do nº de pilhas NO MOMENTO da mudança. É o que permite
    # desenhar o gráfico só com as linhas que existem (ver totals_series).
    total_chairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pile_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # NULL na primeira leitura da pilha; depois, o valor anterior. O par
    # (previous_count, chair_count) conta a história sem precisar de diff.
    previous_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    origin: Mapped[str] = mapped_column(String(16), nullable=False, default="ai")  # ai | manual
    # "ai" = saída do modelo. "manual" = operador. As consultas de "última
    # contagem" filtram por "ai" para não deixar o valor humano virar a média.


class ModelVersion(Base):
    """Versão de modelo YOLO com as métricas reais do treino."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Chave natural: retreinar o MESMO arquivo atualiza a linha em vez de
    # duplicar. unique=True é o que o upsert do repository usa para achar.
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    # Métricas do treino (YOLO valida com IoU 0.5 e 0.5:0.95). NULL quando o
    # modelo foi colocado na pasta na mão, sem treino registrado.
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    map5095: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Só um modelo fica ativo por vez. O repository desliga os outros antes
    # de ligar este, para o worker nunca ficar em dúvida de qual pesos usar.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # classes é texto (não tabela): são poucos rótulos e ninguém consulta por
    # eles - só exibir ao lado da versão.
    classes: Mapped[str | None] = mapped_column(String(255), nullable=True)


class SystemEvent(Base):
    """Evento do sistema (info/warn/error). Alimenta a página de Logs."""

    __tablename__ = "system_events"
    __table_args__ = (Index("ix_events_ts", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    # Nível no mesmo formato do logging do Python (DEBUG..CRITICAL). Sem FK
    # para câmera de propósito: um evento do sistema não pertence a uma câmera.
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    # event é o "tipo" curto e estável (ex.: "contagem_gravada"); message é o
    # detalhe livre. Filtrar por event é o que dá para construir relatório.
    event: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Integer "solto" (sem FK): mesmo opcional, a câmera pode ter sido removida
    # e o evento continua lá. Índice por câmera não é criado de propósito.
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Calibration(Base):
    """ROI e parâmetros de contagem por câmera (um registro por câmera)."""

    __tablename__ = "calibration"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # unique=True = uma calibração por câmera (1:1). O ON DELETE CASCADE faz
    # a calibração sumir junto com a câmera.
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), unique=True, index=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # ROI em pixels do frame original, guardada como 4 inteiros em vez de JSON
    # porque é consultada a cada frame e o banco não deve ser gargalo.
    roi_x: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    roi_y: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    roi_w: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    roi_h: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Altura média de uma cadeira em px: o "pitch" da pilha vem daqui e é a
    # escala que permite contar por altura em pilhas fora de alcance.
    chair_height_px: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Corte de detecção do YOLO (0.5 = padrão do YOLO, bem permissivo).
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    # 10 frames é a janela padrão do filtro de estabilidade: é daqui que vem a
    # regra de "um frame não muda o dashboard".
    stability_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    # Corte de aceitação da contagem estabilizada: abaixo disso o status vai
    # para LOW_CONFIDENCE e o valor não é tratado como verdade.
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.55)
    # "auto" deixa o StackCounter escolher entre periodic/extent/detections/size.
    counting_method: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    # Peso usado na fusão dos estimadores: método diferente = número diferente.
    chair_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    # Distância horizontal (px) que separa duas pilhas. É o mesmo critério do
    # x_overlap_ratio do schemas: pilhas estão lado a lado, nunca atrás.
    pile_gap_px: Mapped[int] = mapped_column(Integer, nullable=False, default=40)
    # Params extras do detector/ROI (o que ele sabe ajustar). Texto JSON: o
    # conjunto muda conforme o modelo, e não vale a pena uma tabela para isso.
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class Snapshot(Base):
    """Imagem salva (auditoria e material para o dataset)."""

    __tablename__ = "snapshots"
    __table_args__ = (Index("ix_snapshots_camera_ts", "camera_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Sem FK: snapshot é evidência, e evidência deve sobreviver à câmera.
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    # Só o caminho/nome do arquivo: o JPEG fica no disco, nunca no banco.
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    # Motivo do disparo: "count_change", "pile_appear", "pile_disappear",
    # "low_confidence", "interval" ou "manual". Diz por que a imagem existe.
    reason: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")
    # O estado do mundo quando a foto foi tirada. Sem isso a imagem não dá
    # para Ground Truth depois.
    total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pile_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Tamanho em bytes: alerta cedo de snapshot que encheu o disco.
    snapshot_size: Mapped[int | None] = mapped_column(Integer, nullable=True)


class CountCorrection(Base):
    """Correção manual do operador.

    Guarda ``ai_count`` e ``correct_count`` para virar dado de treino /
    análise de erro do modelo.
    """

    __tablename__ = "count_corrections"
    __table_args__ = (Index("ix_corrections_camera_ts", "camera_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Sem FK de propósito: a discordância do operador é um dado de treinamento
    # que faz sentido mesmo se a câmera ou a pilha saírem do sistema.
    camera_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pile_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    # O par (ai_count, correct_count) é o dado: a diferença é o erro do modelo.
    # suggest_offset do StackCounter usa esses pares para achar um deslocamento
    # sistemático (contar sempre 1 a mais, por exemplo).
    ai_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ai_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Quem corrigiu: sem login no sistema, é o rótulo que o operador digita.
    author: Mapped[str] = mapped_column(String(64), nullable=False, default="operador")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Caminho da foto da cena: é ela que permite retraining com o exemplo.
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)


__all__ = [
    "Base",
    "Camera",
    "Pile",
    "CountRecord",
    "ModelVersion",
    "SystemEvent",
    "Calibration",
    "Snapshot",
    "CountCorrection",
]
