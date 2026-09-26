"""Configuração central da aplicação.

Toda a configuração vem do arquivo ``.env`` (copie ``.env.example``).
Nenhuma credencial é hard-coded: a URL RTSP é sempre lida do ambiente e,
ao ser exposta para o frontend, é mascarada por :func:`mask_url`.
"""

# =============================================================================
# ARQUIVO / MAPA  -  app/config.py
#
# O que faz: junta TODA a configuração do .env em um único objeto `settings`,
#   organizado em seções (câmera, IA, contagem, calibração, alertas, snapshots,
#   web, treino). Cada seção é uma classe Pydantic com seus próprios campos.
#
# COMO O .ENV É LIDO
#   - arquivo: .env na raiz do projeto (ou o que estiver em CHAIR_COUNTER_ENV)
#   - variável no ambiente TEM PRECEDÊNCIA sobre o arquivo (padrão do
#     pydantic-settings): `DEVICE=cpu python -m app.main` ganha do .env
#   - os campos são case-insensitive: no .env escreve-se TRAINING_EPOCHS /
#     training_epochs, e os dois funcionam
#   - `extra="ignore"`: chave nova ou errada no .env NÃO quebra a aplicação
#   - validação de faixa (ge/le/gt) acontece na leitura: valor absurdo no .env
#     derruba o processo na subida, em vez de falhar no meio da operação
#
# REGRAS DE PRECEDÊNCIA
#   1. flags de linha de comando (main.py sobrescreve settings na hora)
#   2. variável de ambiente do shell
#   3. arquivo .env
#   4. default do Field(...) em config.py
#   E, no caso do MODELO, um acima de todos: o modelo ativo registrado no
#   BANCO tem prioridade sobre o YOLO_MODEL do .env (ver bootstrap em main.py).
#   Ver também settings.resolve_device() e settings.class_indices().
#
# Ordem de leitura:
#   1. BASE_DIR / _env_file . onde está o .env
#   2. seções (XxxSettings) . os campos de cada área do sistema
#   3. class Settings .... agrega as seções + caminhos + resolve_device()
#   4. mask_url .......... mascara a senha de URLs (vai para o dashboard)
#   5. get_settings ...... instância única (lru_cache) usada pelo resto do código
# =============================================================================

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto (pasta que contém app/, training/, data/)
BASE_DIR = Path(__file__).resolve().parent.parent


def _env_file() -> Path:
    """Caminho do arquivo .env.

    CHAIR_COUNTER_ENV permite rodar com outro arquivo (ex.: um .env de teste)
    sem editar nada. Ela é avaliada na Importação do módulo, e não a cada
    leitura, porque SettingsConfigDict congela esse valor.
    """
    env = os.environ.get("CHAIR_COUNTER_ENV", str(BASE_DIR / ".env"))
    return Path(env)


class CameraSettings(BaseSettings):
    """Conexão e captura da câmera."""

    # env_file calculado por função: pydantic-settings aceita um callable, o que
    # deixa o CHAIR_COUNTER_ENV ser resolvido uma vez, na criação da classe.
    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    # Todos os campos abaixo chegam do .env pelo NOME do atributo em MAIÚSCULAS
    # (CAMERA_RTSP_URL, PROCESS_FPS, ...). Os Field(...) abaixo definem o valor
    # padrão e a faixa aceita; o `description` é o que aparece no dashboard.
    camera_rtsp_url: str = Field(default="", description="URL RTSP completa da câmera IP")
    camera_name: str = Field(default="Camera 1", description="Nome exibido no dashboard")
    # FPS de PROCESSAMENTO, não da câmera: a câmera pode entregar 15 fps e a IA
    # só analisa 3. Mais FPS = mais CPU/GPU usada e pouca melhora na contagem.
    process_fps: float = Field(default=3.0, ge=0.2, le=60.0, description="FPS de processamento da IA")
    open_timeout_sec: float = Field(default=8.0, description="Timeout para abrir a câmera")
    read_timeout_sec: float = Field(default=10.0, description="Timeout para ler um frame")
    # 0 = reconexão infinita: em produção é o que se quer, senão a câmera cai
    # uma vez e o sistema desiste de vez.
    max_retries: int = Field(default=0, description="0 = reconexão infinita")
    retry_delay_sec: float = Field(default=3.0, description="Espera entre reconexões")
    # tcp = sem perder quadro, mas trava se a rede ficar ruim; udp = mais rápido
    # e "solto" (quadro perdido aparece como queda na contagem).
    transport: Literal["tcp", "udp", "any"] = Field(
        default="tcp", description="Transporte FFmpeg usado na captura RTSP"
    )
    # Buffer do writer de frames: 1 = somente o frame mais recente (evita atraso)
    frame_buffer_size: int = Field(default=2, ge=1, le=32)

    @field_validator("camera_rtsp_url")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        # .env com quebra de linha ou espaço sobrando quebraria o urlsplit
        # e a conexão; a limpeza acontece na leitura, uma vez só.
        return (v or "").strip()


class AISettings(BaseSettings):
    """Detecção, tracking e modelo."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    yolo_model: str = Field(
        default="data/models/chair_counter.pt",
        description="Caminho do modelo YOLO da empresa (relativo à raiz do projeto)",
    )
    # ATENÇÃO à precedência: este campo é só o PADRÃO. O modelo que realmente
    # roda é o que está marcado como ativo no BANCO, se existir (ver bootstrap
    # em main.py). Assim dá para trocar de modelo em produção sem editar .env
    # nem reiniciar o container.
    device: str = Field(
        default="auto",
        description="auto | cpu | cuda | cuda:0 | 0  (auto = CUDA se disponível, senão CPU)",
    )
    # confidence_threshold: corte de detecção. Alto = menos falso positivo, mas
    # cadeira pequena/embolada some. iou_threshold: quanto duas caixas podem
    # se sobrepor e ainda serem caixas diferentes (NMS). Alto = mais
    # duplicatas da mesma cadeira sobrando na imagem.
    confidence_threshold: float = Field(default=0.50, ge=0.01, le=0.99)
    iou_threshold: float = Field(default=0.45, ge=0.01, le=0.99)
    # imgsz precisa bater com o treino: modelo treinado em 640 e inferido em
    # 1280 perde precisão e ganha latência.
    imgsz: int = Field(default=640, ge=64, le=3840)
    half_precision: bool = Field(default=False, description="FP16 (só faz sentido em GPU)")
    # Teto de caixas por quadro. 300 cobre várias pilhas; estourar aqui
    # significa ruído (reflexo, sombra) virando detecção.
    max_det: int = Field(default=300, ge=1, le=3000)
    tracker: Literal["bytetrack", "botsort", "internal"] = Field(
        default="internal", description="bytetrack | botsort | internal (tracking próprio de pilhas)"
    )
    # Classes do modelo. Aceita índices ou nomes separados por vírgula.
    chair_classes: str = Field(default="chair", description="Classes que representam cadeira/pilha")
    pile_classes: str = Field(default="pile", description="Classes que representam a pilha inteira")
    chair_weight: float = Field(
        default=1.0, gt=0, description="Quantas cadeiras cada detecção de 'chair' representa"
    )
    use_segmentation: bool = Field(
        default=False, description="Usa máscaras de segmentação quando o modelo é de segmentação"
    )


class CountingSettings(BaseSettings):
    """Estabilidade temporal, confiança e algoritmo de contagem."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    # count_stability_frames x (1/PROCESS_FPS) = quantos segundos a contagem
    # precisa ficar igual antes de virar oficial. É o filtro que segura a
    # contagem quando alguém passa na frente da câmera.
    count_stability_frames: int = Field(default=10, ge=1, le=600, description="Janela de estabilidade")
    stability_mode: Literal["median", "mean", "mode"] = Field(
        default="mode", description="Como agregar a janela temporal"
    )
    change_min_frames: int = Field(
        default=3, ge=1, le=100, description="Frames consecutivos exigidos para aceitar nova contagem"
    )
    counting_method: Literal["auto", "fused", "detections", "periodicity", "size"] = Field(
        default="auto",
        description=(
            "auto = fusion de todos os estimadores; periodicity = só análise de padrão "
            "(lattice + extent); detections = só YOLO; size = só altura calibrada"
        ),
    )
    # Os quatro status possíveis (OK / LOW_CONFIDENCE / UNSTABLE / ...) vêm
    # destes limiares: o contador primeiro decide se a leitura é confiável,
    # e só depois o número é aceito.
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0, description="Abaixo disso: LOW_CONFIDENCE")
    stability_min_confidence: float = Field(
        default=0.60, ge=0.0, le=1.0, description="Abaixo disso: UNSTABLE"
    )
    min_chairs_for_valid: int = Field(default=1, ge=0, description="Quantidade mínima para uma pilha válida")
    max_chairs: int = Field(default=200, ge=1, le=5000, description="Teto de segurança da contagem")
    # Altura de uma cadeira em pixels na imagem da câmera. 0 = desconhecido.
    # Informe o valor real na página de Calibração para ativar os métodos
    # baseados em tamanho e período.
    chair_height_px: float = Field(default=0.0, ge=0.0, description="0 = ainda não calibrado")
    pile_overlap_min: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Overlap X mínimo para juntar duas detecções na mesma pilha"
    )
    pile_gap_px: int = Field(
        default=40, ge=0, description="Distância horizontal (px) que separa duas pilhas distintas"
    )
    partial_ratio: float = Field(
        default=0.12, ge=0.0, le=1.0, description="Fração da borda que indica pilha cortada pela imagem"
    )
    miss_grace_frames: int = Field(
        default=15, ge=0, description="Frames sem detecção antes de considerar a pilha desaparecida"
    )
    smoothing_alpha: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Fator do filtro exponencial sobre a contagem por pilha"
    )


class CalibrationSettings(BaseSettings):
    """ROI principal e recortes da área de trabalho."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    enabled: bool = Field(default=False, description="Aplica a ROI principal (ignora o resto da imagem)")
    # Formato: x,y,w,h em pixels da imagem original. Vazio = imagem inteira.
    roi: str = Field(default="", description="ROI principal em pixels: x,y,w,h")
    # Limite inferior da área onde a base das pilhas pode ficar (y máximo em %)
    floor_y_pct: float = Field(default=100.0, ge=1.0, le=100.0)
    # Ignora uma faixa vertical no topo (ex.: toldo, iluminação, teto)
    ceiling_y_pct: float = Field(default=0.0, ge=0.0, le=99.0)


class AlertSettings(BaseSettings):
    """Limites de alerta. Exibidos no dashboard; prontos para Telegram/e-mail."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    enabled: bool = Field(default=True)
    total_min: int = Field(default=0, ge=0, description="Alerta se o total ficar abaixo disso")
    total_max: int = Field(default=0, ge=0, description="0 = desativado; alerta se o total passar disso")
    pile_min: int = Field(default=0, ge=0, description="0 = desativado")
    low_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    camera_offline: bool = Field(default=True)


class SnapshotSettings(BaseSettings):
    """Gravação automática de imagens para auditoria e dataset."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    save_snapshots: bool = Field(default=True)
    interval_sec: int = Field(default=0, ge=0, description="0 = somente por evento")
    on_count_change: bool = Field(default=True)
    on_pile_appear: bool = Field(default=True)
    on_pile_disappear: bool = Field(default=True)
    on_low_confidence: bool = Field(default=True)
    on_reconnect: bool = Field(default=True)
    jpeg_quality: int = Field(default=85, ge=40, le=100)
    keep_days: int = Field(default=0, ge=0, description="0 = nunca apagar automaticamente")


class WebSettings(BaseSettings):
    """Servidor HTTP e estado global."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    database_url: str = Field(default="sqlite:///./data/chairs.db")
    log_level: str = Field(default="INFO")
    stream_fps: float = Field(default=8.0, ge=0.5, le=30.0, description="FPS do stream para o dashboard")
    websocket_broadcast_hz: float = Field(default=2.0, ge=0.2, le=20.0)
    cors_origins: str = Field(default="*", description="Lista separada por vírgula")
    start_banner: bool = Field(default=True)


class TrainingSettings(BaseSettings):
    """Parâmetros padrão de treino (overridáveis pela UI)."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    model: str = Field(default="yolo11n.pt", description="Modelo base")
    epochs: int = Field(default=100, ge=1, le=10000)
    # image_size = lado da imagem quadrada. Vai também fixado no ONNX exportado.
    image_size: int = Field(default=640, ge=64, le=3840)
    batch: int = Field(default=16, ge=-1, le=1024, description="-1 = auto (Ultralytics)")
    # Vazio = usa o DEVICE do app. Preenchido, ganha do auto-detect.
    device: str = Field(default="", description="Vazio = usa DEVICE do app")
    # Nº de processos do dataloader. 4 é bom para SSD/NVMe; em rede ou disco
    # mecânico, mais que 8 não ajuda e ainda compete com a GPU por CPU.
    workers: int = Field(default=4, ge=0, le=64)
    # Early stopping. 0 desliga: o treino roda todas as épocas, sem parar.
    patience: int = Field(default=50, ge=0)
    project: str = Field(default="training/runs", description="Diretório de saída")
    name: str = Field(default="chair_counter", description="Nome do run")
    # Precisa ser a MESMA seed usada na divisão do dataset: é ela que torna
    # dois treinamentos comparáveis.
    seed: int = Field(default=42)


class Settings:
    """Agregador das seções de configuração."""

    def __init__(self) -> None:
        self.camera = CameraSettings()
        self.ai = AISettings()
        self.counting = CountingSettings()
        self.calibration = CalibrationSettings()
        self.alerts = AlertSettings()
        self.snapshots = SnapshotSettings()
        self.web = WebSettings()
        self.training = TrainingSettings()

    # ------------------------------------------------------------------ paths
    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def models_dir(self) -> Path:
        return BASE_DIR / "data" / "models"

    @property
    def snapshots_dir(self) -> Path:
        return BASE_DIR / "data" / "snapshots"

    @property
    def results_dir(self) -> Path:
        return BASE_DIR / "data" / "results"

    @property
    def config_dir(self) -> Path:
        return BASE_DIR / "data" / "config"

    @property
    def logs_dir(self) -> Path:
        return BASE_DIR / "logs"

    @property
    def dataset_dir(self) -> Path:
        return BASE_DIR / "training" / "datasets"

    @property
    def source_images_dir(self) -> Path:
        """Imagens "cruas" capturadas do ambiente para anotação."""
        return BASE_DIR / "training" / "datasets" / "raw"

    @property
    def model_path(self) -> Path:
        # Caminho RELATIVO no .env é resolvido contra a raiz do projeto, e
        # não contra o cwd: assim o sistema acha o modelo de onde for rodado.
        p = Path(self.ai.yolo_model)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def db_path(self) -> Path | None:
        """Caminho do arquivo SQLite (None para outros bancos)."""
        url = self.web.database_url
        if url.startswith("sqlite"):
            raw = url.split("sqlite:///", 1)[-1] or "./data/chairs.db"
            p = Path(raw)
            return p if p.is_absolute() else BASE_DIR / p
        return None

    # ------------------------------------------------------------------ utils
    def resolve_device(self) -> str:
        """Resolve DEVICE=auto|cpu|cuda|cuda:0 para um valor aceito pelo Ultralytics.

        "auto" pergunta ao torch. Nota: TRAINING_DEVICE (vazio por padrão) tem
        precedência só no script de treino; aqui vale sempre o DEVICE do app.
        """
        requested = (self.ai.device or "auto").strip().lower()
        if requested in ("", "auto"):
            try:
                import torch

                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:  # torch ausente: modo preparação
                # Import em try/except porque o pacote de treino é opcional:
                # sem torch o sistema sobe assim mesmo, em CPU.
                return "cpu"
        if requested == "cuda":
            return "cuda:0"
        # "0", "cuda:1" etc. passam direto para o Ultralytics.
        return requested

    def ensure_dirs(self) -> None:
        # Cria na subida tudo que o sistema assume que existe (logs, snapshots,
        # splits do dataset). Sem isso, o primeiro erro apareceria só no meio
        # de uma contagem real.
        for d in (
            self.data_dir,
            self.models_dir,
            self.snapshots_dir,
            self.results_dir,
            self.config_dir,
            self.logs_dir,
            self.dataset_dir,
            self.source_images_dir,
            self.dataset_dir / "images" / "train",
            self.dataset_dir / "images" / "val",
            self.dataset_dir / "images" / "test",
            self.dataset_dir / "labels" / "train",
            self.dataset_dir / "labels" / "val",
            self.dataset_dir / "labels" / "test",
        ):
            d.mkdir(parents=True, exist_ok=True)

    def class_indices(self, names: list[str] | dict[int, str]) -> tuple[set[int], set[int]]:
        """Traduz os nomes de classe do .env para índices do modelo.

        Aceita ``names`` como lista ou como ``{id: nome}``. Um nome que não
        exista no modelo simply não gera índice (ex.: modelo só com ``chair``
        não terá índice para ``pile``).
        """
        if isinstance(names, dict):
            lut = {str(v).strip().lower(): int(k) for k, v in names.items()}
            order = sorted(lut.items(), key=lambda kv: kv[1])
        else:
            order = [(str(n).strip().lower(), i) for i, n in enumerate(names)]

        def _resolve(raw: str) -> set[int]:
            # Aceita "chair,pile", "chair; pile" e também índices puros ("0,1").
            # Aceitar os dois é o que permite usar o mesmo .env com modelos
            # diferentes sem editar a configuração.
            found: set[int] = set()
            for token in (raw or "").replace(";", ",").split(","):
                token = token.strip().lower()
                if not token:
                    continue
                if token.isdigit():
                    idx = int(token)
                    # Índice fora do modelo é ignorado de propósito: melhor
                    # perder uma classe do que derrubar a contagem.
                    if any(i == idx for _, i in order):
                        found.add(idx)
                    continue
                for name, idx in order:
                    if name == token:
                        found.add(idx)
            return found

        # Tupla: (índices de cadeira, índices de pilha). São filtros
        # independentes; um modelo só com "chair" devolve o segundo vazio.
        return _resolve(self.ai.chair_classes), _resolve(self.ai.pile_classes)


def mask_url(url: str) -> str:
    """Mascara a senha de uma URL para exibição no dashboard.

    ``rtsp://user:secret@host:554/path`` -> ``rtsp://user:****@host:554/path``

    Devolve a string **sem alteração** quando não há credencial a mascarar
    (URL sem ``@``, caminho de arquivo, string vazia). Isso é importante:
    mascarar tudo transformaria caminhos de imagem/vídeo em ``***`` e
    esconderia informação que não é sensível.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        # URL malformada não pode ser exibida nem reaproveitada como caminho.
        return "***"
    if parts.scheme and parts.netloc:
        netloc = parts.netloc
        if "@" in netloc:
            # rpartition (e não split) porque a senha pode conter "@".
            userinfo, _, host = netloc.rpartition("@")
            if ":" in userinfo:
                # Só a senha é trocada. O usuário fica visível de propósito:
                # ajuda a identificar a câmera sem expor a credencial.
                user = userinfo.split(":", 1)[0]
                return urlunsplit((parts.scheme, f"{user}:****@{host}", parts.path, parts.query, parts.fragment))
        return url
    # Sem esquema (ex.: "//user:pass@host/stream" ou caminho local)
    if "@" in url:
        userinfo, _, host = url.rpartition("@")
        if ":" in userinfo:
            user = userinfo.split(":", 1)[0]
            return f"{user}:****@{host}"
        return url
    return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # maxsize=1 = instância única. Isso importa porque main.py, a calibração e
    # a aba Treinamento mexem nos mesmos campos: com cópias diferentes, uma
    # sobrescrita não apareceria para o outro. Também garante que os diretórios
    # sejam criados uma única vez, na importação.
    s = Settings()
    s.ensure_dirs()
    return s


# Importado como `from app.config import settings` em todo o resto do código.
settings = get_settings()
