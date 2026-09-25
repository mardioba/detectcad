#!/usr/bin/env bash
# ============================================================================
#  CONTADOR DE CADEIRAS — script de inicialização
#
#  Verifica Python, cria/ativa o ambiente virtual, instala dependências,
#  cria as pastas necessárias e sobe o dashboard.
#
#  Uso:
#    ./run.sh                    instala o que faltar e sobe o dashboard
#    ./run.sh --no-install       não instala nada, só sobe
#    ./run.sh --image f.jpg      modo imagem (com dashboard)
#    ./run.sh --video f.mp4      modo vídeo (com dashboard)
#    ./run.sh --train            roda a validação do modelo
#    ./run.sh --test             roda a suíte de testes
#    ./run.sh --help             esta ajuda
# ============================================================================

set -uo pipefail

cd "$(dirname "$0")" || exit 1

# ------------------------------- cores --------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

VENV=".venv"
PY_MIN="3.10"
DO_INSTALL=1
RUN_ARGS=()

usage() {
    sed -n '3,17p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'
    exit 0
}

# ------------------------------ argumentos ----------------------------------
for arg in "$@"; do
    case "$arg" in
        --no-install) DO_INSTALL=0 ;;
        --help|-h) usage ;;
        --image|--video|--host|--port|--fps|--model|--device) RUN_ARGS+=("$arg") ;;
        --image=*|--video=*|--host=*|--port=*|--fps=*|--model=*|--device=*) RUN_ARGS+=("$arg") ;;
        --train|--test) RUN_ARGS+=("$arg") ;;
        *) RUN_ARGS+=("$arg") ;;
    esac
done

# ------------------------------- funções -----------------------------------
log()  { printf "${CYAN}==>${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}  ✓${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}  !${RESET} %s\n" "$*"; }
err()  { printf "${RED}  ✗${RESET} %s\n" "$*"; }

banner() {
    cat <<'EOF'

====================================
   CONTADOR DE CADEIRAS
====================================

EOF
}

# ------------------------------ 1. Python -----------------------------------
log "Verificando Python..."
PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        v=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0)
        major=${v%%.*}; minor=${v##*.}
        if [ "$major" -eq 3 ] && [ "$minor" -ge "${PY_MIN##*.}" ]; then
            PY="$cand"; break
        fi
    fi
done

if [ -z "$PY" ]; then
    err "Python ${PY_MIN}+ não encontrado."
    err "Instale com: sudo apt install python3-venv python3-pip"
    exit 1
fi
ok "Python: $($PY --version 2>&1) ($PY)"

# ------------------------- 2. ffmpeg (necessário p/ RTSP) ------------------
if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
else
    warn "ffmpeg não encontrado — recomendado: sudo apt install ffmpeg"
    warn "Sem ffmpeg, algumas câmeras RTSP podem não abrir."
fi

# ------------------------- 3. ambiente virtual ------------------------------
log "Preparando ambiente virtual..."
if [ ! -d "$VENV" ]; then
    if ! $PY -m venv "$VENV" 2>/dev/null; then
        err "Falha ao criar o ambiente virtual."
        err "No Ubuntu: sudo apt install python3-venv"
        exit 1
    fi
    ok "Ambiente virtual criado em $VENV"
else
    ok "Ambiente virtual já existe"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate" || { err "Falha ao ativar o venv"; exit 1; }
ok "venv ativado: $(python --version 2>&1)"

# ------------------------- 4. dependências ---------------------------------
need_install=0
if [ ! -f "$VENV/.deps_ok" ] || [ requirements.txt -nt "$VENV/.deps_ok" ]; then
    need_install=1
elif ! python -c "import fastapi, cv2, sqlalchemy" >/dev/null 2>&1; then
    need_install=1
fi

if [ "$DO_INSTALL" -eq 1 ] && [ "$need_install" -eq 1 ]; then
    log "Instalando dependências (pode levar alguns minutos)..."
    python -m pip install --upgrade pip --quiet --disable-pip-version-check
    # O Ultralytics exige opencv-python (com GUI). Em servidor headless isso
    # traz dependências de X11 desnecessárias, por isso instalamos com
    # --no-deps e usamos apenas o headless.
    if python -m pip install --quiet --disable-pip-version-check -r requirements.txt; then
        ok "Dependências instaladas"
    else
        warn "Instalação padrão falhou; tentando sem dependências do Ultralytics..."
        python -m pip install --quiet --disable-pip-version-check \
            --no-deps ultralytics || warn "Falha ao instalar o Ultralytics"
        python -m pip install --quiet --disable-pip-version-check -r requirements.txt
        ok "Dependências instaladas (modo headless)"
    fi
    touch "$VENV/.deps_ok"
else
    ok "Dependências já instaladas"
fi

if [ "$DO_INSTALL" -eq 1 ]; then
    log "Verificando dependências de treinamento..."
    if python -c "import pycocotools" >/dev/null 2>&1; then
        ok "pycocotools presente"
    else
        warn "pycocotools ausente — o treino vai instalar ou falhar."
        warn "Instale com: pip install -r requirements-train.txt"
    fi
fi

# ------------------------- 5. pastas e .env --------------------------------
log "Criando pastas de dados..."
mkdir -p data/models data/snapshots data/results data/config \
         logs training/runs training/datasets/raw \
         training/datasets/images/{train,val,test} \
         training/datasets/labels/{train,val,test}
ok "Pastas prontas"

if [ ! -f .env ]; then
    cp .env.example .env
    warn ".env criado a partir de .env.example — configure a câmera!"
    warn "  nano .env   (CAMERA_RTSP_URL, YOLO_MODEL, PORT...)"
else
    ok ".env encontrado"
fi

# ------------------------- 6. modo especial ---------------------------------
if [ "${RUN_ARGS[0]:-}" = "--test" ]; then
    banner; echo "Executando testes..."; echo
    python -m pytest tests/ -q 2>/dev/null || python -m pytest tests/ -q
    exit $?
fi
if [ "${RUN_ARGS[0]:-}" = "--train" ]; then
    banner; echo "Validando modelo..."; echo
    python -m training.scripts.validate "$@"
    exit $?
fi

# ------------------------- 7. diagnóstico do ambiente ---------------------
log "Verificando ambiente de IA..."
python - <<'PY' 2>/dev/null || warn "Não foi possível ler o ambiente de IA"
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"  \033[0;32m✓\033[0m GPU: {p.name}  VRAM {total/1024**3:.1f} GB (livre {free/1024**3:.1f} GB)")
    else:
        print("  \033[1;33m!\033[0m CUDA indisponível — rodando na CPU (funciona, mas mais devagar)")
except Exception:
    print("  \033[1;33m!\033[0m PyTorch não encontrado")
try:
    import ultralytics
    print(f"  \033[0;32m✓\033[0m Ultralytics {ultralytics.__version__}")
except Exception:
    print("  \033[1;33m!\033[0m Ultralytics ausente")
PY

# ------------------------- 8. sobe a aplicação -----------------------------
PORT=$(grep -E '^PORT=' .env 2>/dev/null | head -1 | cut -d= -f2 | tr -d ' ')
PORT=${PORT:-8000}
HOST=$(grep -E '^HOST=' .env 2>/dev/null | head -1 | cut -d= -f2 | tr -d ' ')
HOST=${HOST:-0.0.0.0}
SHOW_HOST=$HOST
[ "$HOST" = "0.0.0.0" ] && SHOW_HOST="localhost"

banner
cat <<EOF
Dashboard:  http://${SHOW_HOST}:${PORT}

  1. Abra esse endereço no navegador
  2. Confira o status da câmera na aba "Câmera"
  3. Se aparecer "Modelo específico ainda não treinado",
     treine o primeiro modelo (README, seção 11)

  Encerrar: Ctrl+C

====================================

EOF

exec python -m app.main "${RUN_ARGS[@]:-}"
