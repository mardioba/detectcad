# Contador de Cadeiras Plásticas

Sistema de visão computacional para **contar cadeiras em pilhas** usando uma
câmera IP/RTSP fixa. O sistema conecta na câmera, detecta as pilhas, estima
quantas cadeiras existem em cada uma, estabiliza o número ao longo do tempo,
grava o histórico no banco e mostra tudo em um dashboard web em tempo real.

```
CÂMERA RTSP  →  CAPTURA  →  YOLO  →  PILHAS  →  CONTAGEM  →  ESTABILIDADE
                                                                    ↓
                        DASHBOARD ← WEBSOCKET ← BANCO ← RESULTADO FINAL
```

---

## Índice

1. [O que o sistema faz](#1-o-que-o-sistema-faz)
2. [Instalação](#2-instalação)
3. [Configurar a câmera](#3-configurar-a-câmera)
4. [Iniciar e abrir o dashboard](#4-iniciar-e-abrir-o-dashboard)
5. [Modos de teste (imagem / vídeo)](#5-modos-de-teste-imagem--vídeo)
6. [Como a contagem funciona](#6-como-a-contagem-funciona)
7. [Entendendo a confiança](#7-entendendo-a-confiança)
8. [Adicionar as primeiras imagens](#8-adicionar-as-primeiras-imagens)
9. [Como anotar](#9-como-anotar)
10. [Criar o dataset](#10-criar-o-dataset)
11. [Treinar o primeiro modelo](#11-treinar-o-primeiro-modelo)
12. [Validar o modelo](#12-validar-o-modelo)
13. [Ativar o modelo treinado](#13-ativar-o-modelo-treinado)
14. [Calibração](#14-calibração)
15. [Correção manual](#15-correção-manual)
16. [API](#16-api)
17. [Logs](#17-logs)
18. [Solução de problemas](#18-solução-de-problemas)
19. [Precisão: o que esperar](#19-precisão-o-que-esperar)
20. [Desenvolvimento e testes](#20-desenvolvimento-e-testes)
21. [Estrutura do projeto](#21-estrutura-do-projeto)
22. [Otimizações futuras](#22-otimizações-futuras)

---

## 1. O que o sistema faz

| Recurso | Onde |
|---|---|
| Conecta em câmera RTSP, reconecta sozinho, mostra status | aba **Câmera** |
| Detecta pilhas e mantém o ID estável (tracking) | aba **Dashboard** |
| Conta as cadeiras de cada pilha | aba **Contagem** |
| Estabiliza o número (não muda por 1 frame) | `COUNT_STABILITY_FRAMES` |
| Desenha boxes, quantidade e confiança na imagem | overlay no vídeo |
| Calcula o total de cadeiras | topo do dashboard |
| Grava histórico **só quando o número muda** | banco + aba **Histórico** |
| Salva snapshot em cada mudança/evento | `data/snapshots/` |
| Permite corrigir a contagem manualmente | `[-] n [+]` na lista de pilhas |
| ROI, altura da cadeira, limiares | aba **Calibração** |
| Dataset, divisão reprodutível, anotação | abas **Dataset** / **Anotação** |
| Treina YOLO sem travar o dashboard | aba **Treinamento** |
| Versiona e ativa modelos em produção | aba **Modelo** |
| Eventos, logs, alertas | aba **Logs** |
| WebSocket com o estado em tempo real | `/ws` |

### Importante sobre o modelo de IA

O sistema **não** funciona bem com um YOLO genérico. A contagem de cadeiras
empilhadas é um problema específico: as cadeiras se sobrepõem, as de baixo
ficam parcialmente escondidas, e um modelo treinado em "cadeira solta" não
generaliza para "pilha".

O caminho confiável é:

1. **Detecção individual de cadeiras** (YOLO treinado nas suas fotos) — é o
   sinal primário.
2. **Análise do empilhamento** (`app/counting/stack_counter.py`) — cobre as
   cadeiras ocluídas que o YOLO não viu, e valida a contagem.

Enquanto o modelo da empresa não existir, o sistema roda em **modo de
preparação**: mostra a câmera, o dashboard e a infraestrutura, e informa
claramente que ainda não há contagem. Ele não inventa número.

---

## 2. Instalação

**Requisitos**: Linux (testado em Ubuntu 22.04/24.04), Python 3.10+ (3.12
recomendado), `ffmpeg` instalado no sistema.

```bash
# 1. Dependências do sistema
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg

# 2. Clone/copie o projeto e entre na pasta
cd ~/detectcad

# 3. Ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 4. Dependências
pip install --upgrade pip
pip install -r requirements.txt

# 5. Configuração
cp .env.example .env
```

> **Instalando com GPU NVIDIA (opcional, recomendado)**
> O `pip install` padrão traz o PyTorch com CUDA (~3 GB). Para instalar só o
> Torch de CPU (bem menor):
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```
>
> Para GPU (CUDA 12.4):
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> pip install -r requirements.txt
> ```
>
> O sistema detecta a GPU sozinha. `DEVICE=auto` usa CUDA se existir.

### Instalando tudo com o script

```bash
chmod +x run.sh
./run.sh              # instala o que faltar e sobe o dashboard
./run.sh --no-install # não instala dependências, só sobe
```

---

## 3. Configurar a câmera

Abra o `.env` e preencha:

```ini
CAMERA_RTSP_URL=rtsp://usuario:senha@192.168.0.252:554/onvif1
CAMERA_NAME=Camera 1
```

Para descobrir a URL da sua câmera, teste no terminal:

```bash
# com ffmpeg (mostra se o RTSP responde)
ffprobe -rtsp_transport tcp "rtsp://usuario:senha@192.168.0.252:554/onvif1"
```

Se aparecer informação de vídeo, a URL está correta. Se der erro de
autenticação, o usuário/senha estão errados. Se der timeout, a câmera não
alcança essa porta.

Caminhos de stream mais comuns:

| Fabricante | Caminho típico |
|---|---|
| ONVIF genérico | `/onvif1` |
| Intelbras / MHD | `/onvif1` ou `/Streaming/Channels/101` |
| Dahua | `/cam/realmonitor?channel=1&subtype=0` |
| Hikvision | `/Streaming/Channels/101` |
| XMEye | `/user:pass@ip:554/videoMain` |

> **A senha nunca aparece no dashboard.** O backend substitui a senha por
> `****` em toda resposta da API. Verificado em `tests/test_api.py`.

### Várias câmeras

O sistema já suporta várias. Adicione pela aba **Configurações** ou pela API:

```bash
curl -X POST http://localhost:8000/api/cameras \
  -H 'Content-Type: application/json' \
  -d '{"name":"Camera 2","rtsp_url":"rtsp://u:s@10.0.0.9:554/stream"}'
```

Cada câmera tem URL, modelo, ROI, histórico e contagem próprios.

---

## 4. Iniciar e abrir o dashboard

```bash
./run.sh
# ou
source .venv/bin/activate
python -m app.main
```

Abra no navegador: **http://localhost:8000**

O terminal mostra:

```
====================================
   CONTADOR DE CADEIRAS
====================================

Dashboard:  http://localhost:8000
Logs:       logs/app.log | camera.log | ai.log | training.log
Câmera:     Camera 1  [rtsp://usuario:****@192.168.0.252:554/onvif1]
Modelo:     data/models/chair_counter.pt  (NÃO ENCONTRADO - modo de preparação)
Device:     cpu
```

### Começando rápido (RTSP ainda não configurado)

```bash
# 1. gere uma imagem de teste com pilhas
python -m tools.make_synthetic_stack --chairs 20,15,32 --out /tmp/teste.png

# 2. rode o dashboard com essa imagem
python -m app.main --image /tmp/teste.png --serve

# 3. abra http://localhost:8000
```

---

## 5. Modos de teste (imagem / vídeo)

Antes de ter a câmera, valide tudo com arquivos:

```bash
# Imagem única: imprime a contagem no terminal e sai
python -m app.main --image foto.jpg

# Arquivo de vídeo
python -m app.main --video gravacao.mp4

# Imagem + dashboard web (bom para demonstrar)
python -m app.main --image foto.jpg --serve

# Vídeo + dashboard web
python -m app.main --video gravacao.mp4 --serve
```

Saída do modo terminal:

```
[frame 42] TOTAL: 67 cadeiras em 3 pilha(s)
  Pilha 1 (track 1): 20 cadeiras | status=STABLE | conf=92% | método=auto
  Pilha 2 (track 2): 15 cadeiras | status=STABLE | conf=89% | método=auto
  Pilha 3 (track 3): 32 cadeiras | status=PARTIAL | conf=71% | método=auto
```

O frame com as caixas anotadas é salvo em
`data/results/offline_overlay.jpg`.

### Testar o modelo em uma fonte específica

```bash
python -m training.scripts.test --image foto.jpg
python -m training.scripts.test --video gravacao.mp4
python -m training.scripts.test --rtsp-env CAMERA_RTSP_URL --seconds 30
```

---

## 6. Como a contagem funciona

A contagem de cadeiras empilhadas é a parte mais difícil do projeto: em uma
pilha, a maioria das cadeiras está parcialmente escondida. Por isso o sistema
**não** confia só no YOLO.

`app/counting/stack_counter.py` recebe a região de uma pilha e combina
estimadores independentes:

| Estimador | O que é | Quando pesa |
|---|---|---|
| `detections` | nº de cadeiras detectadas pelo YOLO | Modelo treinado e bom |
| `extent` | extensão da pilha ÷ passo do padrão | **Principal** (não tem ambiguidade) |
| `size` | extensão ÷ altura calibrada da cadeira | Com `CHAIR_HEIGHT_PX` > 0 |
| `lattice` | nº de linhas visíveis alinhadas ao padrão | Valida o passo (não conta) |

O fluxo dentro de uma ROI:

1. **CLAHE** para corrigir a iluminação variável do galpão.
2. **Perfil de bordas horizontais**: soma o gradiente vertical por linha.
   Uma pilha produz linhas repetidas.
3. **Detrend**: remove a tendência de iluminação (sombra, luz do teto).
4. **Passo (período)**: autocorrelação FFT + ajuste de pente (*comb score*).
   Encontra a distância de repetição entre cadeiras.
5. **Extensão ativa**: limiar de **Otsu** separa a pilha do fundo.
6. **Contagem**: `extensão ÷ passo`, com validação pelas camadas visíveis.

### Ambiguidade de período: o sistema se recusa a errar

Uma cadeira tem bordas horizontais na escala dela **e** na sua estrutura
interna (encosto, assento, pernas). Se as duas escalas explicam o sinal
igualmente bem, a imagem **não permite decidir** quantas cadeiras são.

Nesses casos o sistema marca `PERÍODO AMBÍGUO`, derruba a confiança e devolve
**0 → estado `UNKNOWN`**, e o dashboard mostra `?` em vez de um número
inventado. Foi medido em fotos reais: **0 casos de confiança alta com contagem
errada** (ver `tools/bench_real.py`).

### Medições reais do algoritmo

Rode os dois benchmarks a qualquer momento:

```bash
python -m tools.bench_counter   # pilhas sintéticas, número conhecido
python -m tools.bench_real      # fotos reais, número contado à mão
```

Resultados medidos neste projeto (pilhas sintéticas, 15 casos, alturas de
18/26/34 px):

| Métrica | Valor |
|---|---|
| Contagem exata | 4/15 (26,7%) |
| Dentro de 1 cadeira | **14/15 (93,3%)** |
| Erro absoluto médio | 0,80 cadeira |
| Confiança média | 0,73 |

A causa do erro de ±1 é medida: a detecção da borda da pilha tem ~17 px de
erro, e uma cadeira ocupa 18–34 px nessa escala — o arredondamento pode cair
para o lado. É exatamente para isso que existem o **offset de calibração** e a
**correção manual**: o operador corrige, e o sistema aprende o erro.

**Estes números são de teste sintético, não de produção.** A precisão real
depende do seu modelo treinado e das suas imagens.

---

## 7. Entendendo a confiança

Cada pilha mostra três números:

| Nome | Significado |
|---|---|
| `detection_confidence` | Quão bem o YOLO enxergou a pilha |
| `stability_confidence` | Quanto a contagem concorda consigo mesma no tempo |
| `confidence` | Combinação das duas + concordância entre estimadores |

Estados possíveis:

| Estado | Significado | Cor |
|---|---|---|
| `STABLE` | Confiável | verde |
| `UNSTABLE` | Oscilando (poucos frames concordantes) | âmbar |
| `LOW_CONFIDENCE` | Abaixo do limite → mostra `?` | laranja |
| `PARTIAL` | Pilha cortada pela borda da imagem | roxo |
| `UNKNOWN` | Indeterminado | cinza |

Exemplo de tela:

```
PILHA 1
23 cadeiras
Confiança 97% ESTÁVEL
```

```
PILHA 2
? contagem indeterminada
Confiança 48% BAIXA CONFIANÇA
```

> O total do dashboard soma apenas as pilhas `STABLE` e `PARTIAL`. As
> demais aparecem como `(+N instável)` ao lado, para você saber que existem
> cadeiras mas a leitura não é confiável.

Ajuste os limiares na aba **Calibração** ou no `.env`:

```ini
MIN_CONFIDENCE=0.55            # abaixo disso: LOW_CONFIDENCE
STABILITY_MIN_CONFIDENCE=0.60  # abaixo disso: UNSTABLE
```

---

## 8. Adicionar as primeiras imagens

O sistema precisa de fotos **reais** do seu depósito, da mesma câmera que
será usada em produção. Sem imagens reais, não há como treinar.

### Forma mais rápida: extrair de vídeo

Fotografar centenas de vezes é inviável. Grave vídeo e extraia os frames:

```bash
# grave direto da câmera (sem passar por player)
ffmpeg -i "rtsp://usuario:senha@IP:554/onvif1" -t 300 galpao.mp4

# extraia 1 frame por segundo
python -m tools.extract_frames --video galpao.mp4 --fps 1
```

O script também descarta frames quase duplicados — cena parada não ensina
nada — e avisa quando a variação foi insuficiente.

> **Nunca faça print do player.** Barra de título, botões e timeline viram
> parte da imagem e o modelo aprende a reconhecer a interface. Frames limpos.

### Ou envie pelo dashboard

Aba **Dataset** → **Enviar imagens**.

### O que a variedade precisa ser

Não é quantidade, é **variedade**. Grave trechos de:

- pilhas com 3, 8, 15, 25 cadeiras (a variedade de quantidade é o que faz o
  modelo generalizar)
- dia e noite; com e sem pessoas na frente
- ângulos diferentes; cadeiras encostadas em parede
- a cadeira sozinha, para o modelo aprender a forma dela

### Quantas imagens?

| Quantidade | Resultado esperado |
|---|---|
| < 30 | não funciona |
| 100–300 | primeiros testes, ainda impreciso |
| 500–1000 | uso real |
| 2000+ | bom, com muita variação |

**A variedade importa mais que a quantidade**: 1000 fotos do mesmo ângulo
valem menos que 200 de 10 situações diferentes.

Confira o resultado:

```bash
python -m app.main --image training/datasets/raw/alguma.jpg
```

---

## 9. Como anotar

O padrão é **YOLO**: um `.txt` por imagem, com `classe cx cy w h`
(normalizado de 0 a 1).

### Ferramenta recomendada: CVAT (gratuito, local, open-source)

```bash
# Sobe o CVAT com Docker
docker run -dit -p 8080:8080 --name cvat-server \
  -v ~/cvat_data:/home/django/data \
  -v ~/cvat_logs:/home/django/logs \
  cvat/server:latest
```

1. Crie um projeto do tipo **"CVAT for images 1.1"**.
2. Crie **1 ou 2 labels**: `chair` (uma cadeira) e, se quiser, `pile` (a pilha
   inteira).
3. Importe as imagens de `training/datasets/raw/`.
4. Anote arrastando as caixas.
5. Exporte em **CVAT for images 1.1 → ZIP/XML**.

Converta para o formato YOLO:

```bash
python -m training.scripts.prepare_dataset --from-cvat export.xml
```

### Alternativa: Label Studio (sem Docker)

```bash
pip install label-studio
label-studio start          # abre em http://localhost:8080
# anote e exporte em JSON
python -m training.scripts.prepare_dataset --from-labelstudio export.json
```

### Só para juntar imagens, sem rótulo

```bash
python -m training.scripts.prepare_dataset --copy-only /minhas/fotos
```

### Alternativas manuais

- **LabelImg** ou **labelme** (desktop, simples)
- **Roboflow** (web, gratuito com limite)
- Qualquer ferramenta que gere `classe cx cy w h`

### Anotar com o próprio modelo (economiza tempo)

Depois que tiver um primeiro modelo razoável, a aba **Dataset` →
**Pré-rotular com IA** gera as caixas automaticamente. **Revise sempre**: o
modelo erra, e um erro de anotação vira erro de treino.

### Qual classe usar?

Comece **só com `chair`**. Uma caixa por cadeira, mesmo que parcialmente
visível. A classe `pile` (pilha inteira) é opcional e exige mais imagens.
Depois de treinar, compare: se o modelo com as duas classes for melhor, use
as duas. O `.env` controla:

```ini
CHAIR_CLASSES=chair
PILE_CLASSES=pile
```

---

## 10. Criar o dataset

Na aba **Dataset** → **Dividir dataset**:

- 70% treino / 20% validação / 10% teste (configurável)
- **Seed** para a divisão ser **reproduzível**: a mesma seed gera sempre a
  mesmo conjunto, o que permite comparar dois treinamentos com honestidade.

Ou por terminal:

```bash
python - <<'EOF'
import requests
r = requests.post("http://localhost:8000/api/dataset/split",
                  json={"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42})
print(r.json()["message"])
EOF
```

Resultado:

```
training/datasets/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
├── raw/                      # imagens aguardando anotação
└── ../configs/chairs.yaml     # gerado automaticamente
```

---

## 11. Treinar o primeiro modelo

### A regra que decide tudo

**Não dá para treinar um modelo de produção sem centenas de fotos reais das
suas cadeiras.** Não é falta de vontade: um detector YOLO aprende de
exemplos com variação. Quatro cadeiras num único ângulo, numa única luz,
ensinam nada que generalize.

Antes de qualquer treino, responda com honestidade:

| Imagens reais anotadas | O que esperar |
|---|---|
| < 30 | não funciona |
| 100–300 | primeiros testes, ainda impreciso |
| 500–1000 | uso real |
| 2000+ | bom, com muita variação |

**Diversidade importa mais que volume.** Em galpão com câmera fixa, 1000
fotos do mesmo ângulo valem menos que 200 de 10 situações diferentes.

### Passo 1 — Obter as imagens

A forma mais rápida não é fotografar: é **gravar vídeo** e extrair os frames.

```bash
# 1. Grave direto da câmera (nunca faça print do player)
ffmpeg -i "rtsp://usuario:senha@IP:554/onvif1" -t 300 galpao.mp4

# 2. Extraia 1 frame por segundo
python -m tools.extract_frames --video galpao.mp4 --fps 1

# ou direto da câmera, sem arquivo intermediário
python -m tools.extract_frames --rtsp-env CAMERA_RTSP_URL --seconds 300 --fps 1
```

> **Nunca faça print do player.** A barra de título, os botões e a timeline
> viram parte da imagem, e o modelo aprende a reconhecer a interface do
> player. Use frames limpos. (O script tenta remover a moldura, mas é
> conservador de propósito: errar por não cortar é melhor que cortar cadeira.)

O que gravar para o dataset variar:

- pilhas de 3, 8, 15, 25 cadeiras (a variedade de **quantidade** é o que faz
  o modelo generalizar)
- dia e noite, e com/sem pessoas na frente
- ângulos diferentes, cadeiras encostadas em parede, empilhadas frouxas
- a cadeira sozinha (para o modelo aprender a forma dela)

### Passo 2 — Anotar

Classe: comece **só com `chair`**, uma caixa por cadeira (mesmo parcialmente
visível). Se a operação tiver cadeira metálica de outro tipo, **não** anote
como `chair` — ou o contador vai somar cadeiras que não são do inventário.

```bash
# CVAT (gratuito, local) — recomendado
docker run -dit -p 8080:8080 --name cvat cvat/server:latest
# 1. projeto "CVAT for images 1.1", label "chair"
# 2. importar training/datasets/raw/
# 3. exportar → XML
python -m training.scripts.prepare_dataset --from-cvat export.xml

# Label Studio (sem Docker)
pip install label-studio && label-studio start
python -m training.scripts.prepare_dataset --from-labelstudio export.json
```

Atalho quando já existir um modelo razoável: aba Dataset → "Pré-rotular com
IA". **Revise todas** as caixas: erro de anotação vira erro de treino.

### Passo 3 — Dividir (com seed)

```bash
curl -X POST http://localhost:8000/api/dataset/split \
  -H 'Content-Type: application/json' \
  -d '{"train":0.7,"val":0.2,"test":0.1,"seed":42}'
```

Use sempre a **mesma seed** entre experimentos: é o que permite comparar dois
modelos com honestidade. Nunca avalie no conjunto de treino — a métrica fica
ótima e o modelo erra em produção.

### Passo 4 — Validar o dataset (sem gastar tempo)

```bash
python -m training.scripts.train --dry-run
```

Ele acusa o que estiver errado antes de começar: classes inexistentes no
YAML, rótulos malformados, poucas imagens, ausência de validação.

### Passo 5 — Treinar

**Pela aba Treinamento (recomendado):** preencha e clique em
**INICIAR TREINAMENTO**. Roda em processo separado — o dashboard não trava e
o progresso aparece na mesma página.

**Pelo terminal:**

```bash
python -m training.scripts.train --model yolo11n.pt --epochs 100 --batch 16
python -m training.scripts.train --model yolo11s.pt --epochs 200 --device 0
python -m training.scripts.train --resume            # retoma interrompido
```

| Flag | Efeito |
|---|---|
| `--model` | `yolo11n` leve · `yolo11s` equilíbrio · `yolo11m/l` precisão |
| `--device` | `auto` (recomendado) · `cpu` · `cuda` · `0` |
| `--batch` | alto o bastante; se estourar memória, baixe |
| `--dry-run` | só valida o dataset e sai |

### Passo 6 — Validar o modelo

```bash
python -m training.scripts.validate
```

Mostra Precision, Recall, mAP50 e mAP50-95 **lidos do `results.csv`**. Sem
esse arquivo, os campos ficam vazios de propósito: o sistema não inventa
métrica.

E o teste que importa mais — rodar nas **suas** fotos e olhar:

```bash
python -m training.scripts.test \
  --weights training/runs/chair_counter/weights/best.pt \
  --source-dir /minhas/fotos_de_teste
```

### Passo 7 — Ativar

```bash
cp training/runs/chair_counter/weights/best.pt data/models/chair_counter_v1.pt

curl -X POST http://localhost:8000/api/models/register-run \
  -H 'Content-Type: application/json' \
  -d '{"run_dir": "training/runs/chair_counter"}'
```

Depois, aba **Modelo** → **ATIVAR**. A troca é a quente (a câmera não
reinicia) e há rollback.

### Se você tem poucas cadeiras

Com poucas cadeiras em mãos, o caminho útil **não é treinar**: é calibrar e
validar o contador, que não precisa de modelo nenhum.

```bash
# meça a altura de uma cadeira e grave em data/config/camera_1.json,
# ou use a aba Calibração → "Medir cadeira"
```

Leia a seção 19 antes: o que já foi medido (e o que ainda não).

---

## 12. Validar o modelo

```bash
python -m training.scripts.validate
python -m training.scripts.validate --weights training/runs/chair_counter/weights/best.pt
```

Saída (números **reais**, lidos do `results.csv`):

```
============================================================
MÉTRICAS REAIS DO MODELO
============================================================
Run: training/runs/chair_counter
  Épocas treinadas : 100
  Precision        : 0.9142
  Recall           : 0.8876
  mAP50            : 0.9401
  mAP50-95         : 0.8123
============================================================
```

O sistema **nunca inventa métrica**: sem `results.csv`, os campos ficam
vazios e o script avisa.

Também gera detecções de exemplo em `data/results/validation/` para você
**olhar** e judge se o modelo serve.

### Testar antes de ativar

```bash
# em fotos reais do seu depósito
python -m training.scripts.test --weights training/runs/chair_counter/weights/best.pt \
       --source-dir /minhas/fotos_de_teste --conf 0.35
```

Abra os arquivos gerados em `data/results/`. Se as caixas estão nas cadeiras
certas, o modelo presta.

---

## 13. Ativar o modelo treinado

### Pela aba Modelo

1. Abra **Modelo**.
2. O `best.pt` precisa estar em `data/models/`. Copie:

```bash
cp training/runs/chair_counter/weights/best.pt data/models/chair_counter_v1.pt
```

3. Se não aparecer na lista, registre o run:

```bash
curl -X POST http://localhost:8000/api/models/register-run \
  -H 'Content-Type: application/json' \
  -d '{"run_dir": "training/runs/chair_counter"}'
```

4. Clique em **ATIVAR** e confirme.

A troca é **a quente**: a câmera não reinicia, o pipeline não para, e o modelo
anterior fica disponível.

### Voltar ao modelo anterior

```bash
curl -X POST http://localhost:8000/api/models/rollback
```

Ou o botão **Voltar ao anterior** na aba Modelo.

### Versões

```
data/models/
├── chair_counter_v1.pt
├── chair_counter_v2.pt
└── chair_counter_v3.pt
```

Cada ativação registra as métricas reais, a data e qual está ativo, na tabela
`model_versions`.

### Boas práticas de produção

1. **Não** troque de modelo em horário de pico — valide antes.
2. Treine sempre com imagens do período (turno, iluminação) que vai usar em produção.
3. Compare mAP50 dos modelos, mas julgue pelas **fotos reais**.
4. Mantenha o modelo anterior: o rollback é instantâneo.
5. Acompanhe as correções manuais (aba Contagem). Se a IA erra sempre para o
   mesmo lado, ajuste o **offset** (aba Calibração) — isso corrige erro
   sistemático sem retreinar.

---

## 14. Calibração

Como a câmera é fixa, a calibração vale o mesmo para sempre.

### ROI (região de interesse)

Aba **Calibração** → arraste sobre a imagem a área onde as cadeiras podem
aparecer. Tudo fora é ignorado. Use para:

- excluir o teto e a iluminação
- excluir a entrada de gente
- focar só na área de depósito

Marque "aplicar ROI" e salve. Sem ROI, o sistema usa a imagem inteira.

### Altura de uma cadeira (o mais importante)

O método geométrico precisa saber quanto uma cadeira ocupa em pixels:

1. Desenhe a caixa de **uma** cadeira.
2. Clique em **Medir cadeira**.
3. O valor é medido e colocado no campo.
4. Salve.

Isso liga o estimador `size` e restringe a busca do passo, o que torna a
contagem bem mais confiável.

> Nenhum valor é chutado. Se `chair_height_px = 0`, o sistema avisa que não
> está calibrado e a confiança cai.

#### Por que isso importa tanto (medido em pilhas reais)

Em pilhas de cadeiras de verdade, uma cadeira **encaixada** acrescenta só
~3% da própria altura à pilha:

| Medição (fotos reais de galpão) | Valor |
|---|---|
| Altura de uma cadeira | 915 px |
| Passo (1 cadeira) na pilha | 27 px |
| Razão passo/altura | 0,03 |

Ou seja: numa pilha de 25 cadeiras, cada uma só "aparece" com 27 px. A
calibração precisa conhecer essa medida para achar o passo certo.

### Outros parâmetros

| Parâmetro | O que faz | Sugestão |
|---|---|---|
| Limiar de detecção YOLO | Sensibilidade | 0.5; baixe se perder cadeiras |
| Método de contagem | `auto`, `periodicity`, `detections`, `size` | `auto` |
| Janela de estabilidade | Frames antes de trocar o número | 10 (a 3 FPS = 3,3 s) |
| Mín. de concordância | Abaixo disso = `UNSTABLE` | 0.6 |
| Frames p/ aceitar troca | Reação a mudanças reais | 3 |
| Confiança mínima | Abaixo disso = `LOW_CONFIDENCE` | 0.55 |
| Offset de contagem | Corrige erro sistemático de ±1 | 0 (veja abaixo) |
| Classes do modelo | `chair`, `pile` | `chair` |

### Diagnóstico do contador

Botão **Analisar ROI selecionada** mostra exatamente o que o algoritmo vê:
passo detectado, extensão, camadas visíveis, qualidade do padrão e os
estimadores. É a melhor forma de entender por que um número saiu errado.

### Offset automático

Erro sistemático de ±1 cadeira é comum. Em vez de chutar, use os dados:

```bash
curl -s http://localhost:8000/api/corrections | \
  .venv/bin/python -c "import sys,json; d=json.load(sys.stdin); print(d['offset_suggestion'])"
```

```
{"offset": 1, "reliable": true, "samples": 12, "consistency": 0.83,
 "message": "Erro mediano (humano - IA) = +1 em 12 correções (83% consistentes)."}
```

Com pelo menos 3 correções consistentes, o sistema **sugere** o offset. Não
aplica sozinho: você confirma.

---

## 15. Correção manual

Se a IA errar:

1. Na lista de pilhas do dashboard, use `[-] 20 [+]` e confirme.
2. O valor fica **travado**: a IA não sobrescreve sua correção sozinha.
3. O registro `AI_COUNT × CORRECT_COUNT` é salvo na tabela `count_corrections`.
4. Um snapshot da imagem é salvo junto.

Esses dados servem para:

- medir o erro real do modelo (`aba Contagem` → taxa de acerto exato)
- sugerir o offset de calibração
- escolher quais imagens usar no próximo treino

Para devolver o controle à IA, chame `unlock` (a pilha volta ao automático
quando é desmanchada e remontada).

---

## 16. API

Base: `http://localhost:8000`. Documentação interativa: `/docs`.

### Status e saúde

```bash
GET  /api/status              # estado completo
GET  /api/health              # healthcheck
GET  /api/config              # configuração efetiva (sem segredos)
```

### Câmeras

```bash
GET    /api/cameras
POST   /api/cameras            {"name": "...", "rtsp_url": "..."}
PUT    /api/cameras/{id}
DELETE /api/cameras/{id}
```

### Contagem

```bash
GET  /api/count                          # contagem atual + total
GET  /api/piles                          # pilhas registradas
GET  /api/history?hours=24               # histórico
GET  /api/corrections                    # correções manuais
POST /api/count/{pile_id}/correct        # corrigir contagem
```

### Calibração

```bash
GET  /api/calibration
POST /api/calibration     {"roi": {...}, "chair_height_px": 42, ...}
POST /api/calibration/reset
POST /api/calibration/measure   {"roi": {"x":..,"y":..,"w":..,"h":..}}
POST /api/diagnostics/counter   {"roi": {...}}
```

### Dataset e treinamento

```bash
GET  /api/dataset
GET  /api/dataset/images
POST /api/dataset/split        {"train":0.7,"val":0.2,"test":0.1,"seed":42}
POST /api/dataset/auto-label
GET  /api/training/status
POST /api/training/start       {"model":"yolo11n.pt","epochs":100,"batch":16}
POST /api/training/stop
GET  /api/training/log
```

### Modelos

```bash
GET  /api/models
POST /api/models/activate      {"model_id": 2}
POST /api/models/rollback
POST /api/models/import        (upload .pt)
POST /api/models/register-run  {"run_dir": "training/runs/..."}
```

### Snapshots, eventos e logs

```bash
GET    /api/snapshots
POST   /api/snapshots          # salva agora
DELETE /api/snapshots/{id}
GET    /api/events
DELETE /api/events
GET    /api/logs?file=app.log&lines=200
```

### Vídeo e WebSocket

```bash
GET /api/stream.mjpg          # stream MJPEG com overlay
GET /api/frame.jpg            # frame atual
GET /api/frame.raw.jpg        # frame sem overlay
WS  /ws                       # estado em tempo real
```

Exemplo de mensagem do WebSocket:

```json
{
  "type": "state",
  "timestamp": "2026-09-25T15:30:20",
  "camera_state": "ONLINE",
  "pile_count": 3,
  "total": 67,
  "totals": {"total_stable": 67, "total_tentative": 0, "total_all": 67, "pile_count": 3},
  "confidence": 0.94,
  "fps": 3.0,
  "piles": [
    {"pile_id": 1, "count": 20, "confidence": 0.96, "status": "STABLE",
     "method": "auto", "candidates": {"extent": 20}, "bbox": [120, 300, 460, 1200]}
  ]
}
```

---

## 17. Logs

| Arquivo | Conteúdo |
|---|---|
| `logs/app.log` | ciclo de vida, API, eventos |
| `logs/camera.log` | conexão, reconexão, frames, erros |
| `logs/ai.log` | detecção, tracking, contagem, confiança |
| `logs/training.log` | treino, validação, exportação |
| `logs/training_latest.log` | saída do treino em andamento |

Todos rodam com rotação (8 MB × 5 arquivos). A aba **Logs** mostra o
conteúdo e a lista de eventos do banco.

Para seguir em tempo real:

```bash
tail -f logs/camera.log
tail -f logs/ai.log | grep -i contagem
```

---

## 18. Solução de problemas

### A câmera não conecta

```bash
# 1. a URL está certa?
ffprobe -rtsp_transport tcp "rtsp://usuario:senha@IP:554/onvif1"

# 2. a porta está aberta?
nc -zv IP 554

# 3. ajuste o transporte no .env
CAMERA_TRANSPORT=tcp     # ou udp
CAMERA_OPEN_TIMEOUT_SEC=15
```

Mensagens e causas:

| Log | Causa provável |
|---|---|
| `Timeout expired` | IP/porta errada, ou firewall |
| `401 Unauthorized` | usuário ou senha |
| `404 Not Found` | caminho do stream errado |
| `Connection refused` | câmera desligada ou porta fechada |

### O modelo não carrega

```
AVISO: "Modelo específico ainda não treinado."
```

Isso é **esperado** até você treinar. Verifique:

```bash
ls -la data/models/          # o arquivo existe?
grep YOLO_MODEL .env         # o caminho está certo?
python -c "from app.ai.yolo_detector import YoloDetector; d=YoloDetector(); print(d.load(), d.load_error)"
```

### O YOLO não encontra cadeiras

1. `CONFIDENCE_THRESHOLD` está alto? Baixe para 0.25 e teste.
2. A câmera está muito longe? As cadeiras ficam pequenas.
3. Verifique com um modelo genérico só para diagnóstico:
   ```bash
   python -m training.scripts.test --weights data/models/yolo11n_generic.pt --source-dir training/datasets/raw
   ```
   Se o genérico também não encontra, o problema é a imagem, não o modelo.
4. Confira a ROI — talvez as cadeiras estejam fora da área.

### A contagem está errada

1. Abra **Calibração** → **Analisar ROI**.
2. Se aparecer `PERÍODO AMBÍGUO`, o modelo genérico não é suficiente:
   treine o modelo da empresa.
3. Meça a altura da cadeira.
4. Erro de ±1 constante? Use o offset sugerido.
5. Compare o método: teste `periodicity` vs `detections` e veja qual erra
   menos nas suas fotos.

### O dashboard está lento

```ini
PROCESS_FPS=2          # menos frames por segundo
STREAM_FPS=5           # menos tráfego de vídeo
IMGSZ=512              # inferência mais leve
```
Também reduza o `PROCESS_FPS` se a CPU estiver alta (`top`).

### CUDA não é usada

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

Se `False`, o Torch foi instalado sem CUDA. Reinstale com o índice CUDA
(README, seção 2).

### Porta 8000 ocupada

```ini
PORT=8080
```

### O banco travou

```bash
rm -f data/chairs.db*     # recomeça do zero (perde o histórico)
```

### Memória crescendo

```bash
SNAPSHOT_KEEP_DAYS=14    # apaga snapshots antigos
```

### "Modelo genérico pré-treinado (não recomendado)"

Esperado: `data/models/yolo11n_generic.pt` é o COCO. Serve para testar a
infraestrutura, **não** para contar cadeiras. A mensagem aparece justamente
para não haver dúvida.

---

## 19. Precisão: o que esperar

**Não prometemos "100% de precisão".** Ninguém pode, sem evidência nas suas
imagens. E este projeto foi feito medindo, não prometendo.

### O que o sistema oferece

1. **Métricas reais** do seu modelo, lidas do `results.csv`.
2. **Confiança por pilha** — você vê quando confiar e quando conferir.
3. **Correção manual** — o operador corrige, e o erro fica registrado.
4. **Medição do erro real** — taxa de acerto exato e erro médio, calculados a
   partir das correções.
5. **Ambiguidade declarada** — em vez de errar calado, o sistema mostra `?`
   e o estado `INDETERMINADO`.

### O que foi medido de verdade

Rode os dois benchmarks a qualquer momento:

```bash
python -m tools.bench_counter   # pilhas sintéticas, número conhecido
```

| Métrica (pilhas sintéticas) | Valor |
|---|---|
| Contagem exata | 9/15 (60%) |
| Dentro de 1 cadeira | 10/15 (66,7%) |
| Erro absoluto médio | 2,7 cadeiras |

**Estes números são de teste sintético, não de produção**, e o algoritmo de
empilhamento **ainda está em desenvolvimento** — a versão anteriormedia 0,80
de erro e 93% dentro de 1 cadeira em testes sintéticos, mas falhava em fotos
reais de galpão. As medições reais(see abaixo) é que motivaram a mudança, e
ela ainda está em curso.

### O que aprendemos com fotos reais de galpão

Medido em pilhas de cadeiras reais (`cadeiras.mp4`, 1200×1200):

| Fato medido | Valor |
|---|---|
| Altura de uma cadeira | 915 px |
| **Passo na pilha (1 cadeira)** | **27 px** |
| Razão passo/altura | **0,03** |
| Espaçamentos compatíveis com 27 px | 20 (contra 1–2 dos múltiplos) |

Isso expôs dois bugs reais, ambos corrigidos:

1. A calibração buscava o passo entre **62% e 105%** da altura da cadeira.
   Cadeira **encaixada** acrescenta só ~3% — o passo verdadeiro ficava fora da
   busca em qualquer pilha real.
2. O encaixe de pente era **atraído por múltiplos**: o passo real de 27 px
   pontuava 0,33, enquanto um período espúrio de 400 px pontuava 0,65 (pente
   largo amostra menos pontos e erra por acaso). Hoje o passo vem do
   espaçamento entre picos, que discriminou 27 px com 20 apoios contra 1–2.

### Estado atual da contagem em pilhas reais

Ainda **não há número de produção validado**. O que sabemos:

- As três pilhas de teste são idênticas em altura (690 px) e passo (27 px).
- O sistema encontra o passo certo (27 px).
- As contagens das três ainda divergem entre si (21, 16, 16) — o que está
  errado, porque são a mesma pilha.

Falta uma coisa, e é a mais importante: **a contagem verdadeira, contada por
uma pessoa**. Sem ela, qualquer ajuste no algoritmo é chute.

#### Falha conhecida e diagnosticada

Medido nos testes sintéticos, o detector de passo acerta em duas das três
escalas de cadeira testadas:

| Altura da cadeira | Passo detectado | Correto? | Contagem |
|---|---|---|---|
| 18 px | 18,0 px | sim | 12 (real 12) |
| 26 px | **13,0 px** | **não** | 18 (real 12) |
| 34 px | 34,0 px | sim | 12 (real 12) |

Na escala de 26 px o algoritmo ancora na **sub-harmônica** (metade do passo
real) e conta ~50% a mais. É o modo de falha conhecido: quando a estrutura
interna da cadeira reaparece com espaçamento metade do passo, o passo
errado tem mais espaçamentos "de apoio" que o correto, e o critério de
espaçamento entre picos o escolhe.

Por isso `tests/test_counting.py::test_count_respects_chair_scale` falha
para 18, 26 e 34 px (3 testes). **Estão mantidos assim de propósito**, para
que a falha fique visível e ninguém confunda o estado com algo validado.

Correções possíveis, na ordem de custo:

1. Usar a geometria da cadeira para descartar harmônicas (a cadeira tem
   espessura mínima de assento: passo muito abaixo disso é fisicamente
   impossível).
2. Exigir que o passo escolhido seja compatível com `altura_da_cadeira ×
   razão_de_encaixe`, com a razão medida na operação.
3. Pedir a contagem verdadeira para resolver a ambiguidade de harmônica.

Enquanto isso, o sistema **não erra calado**: quando o padrão é ambíguo, a
confiança cai e o dashboard mostra `?`.

### Como transformar correção manual em precisão medida

1. Use o sistema por 1–2 semanas.
2. Corrija as contagens erradas no dashboard (`[-] n [+]`).
3. A aba **Contagem** mostra a taxa de acerto exato e o erro médio reais.
4. Use o offset sugerido para corrigir erro sistemático de ±1.
5. As imagens com erro viram as **primeiras** do próximo treino.

Essa é a forma honesta de medir a precisão: com os seus dados, no seu
galpão, com o seu modelo.

Ciclo de melhoria:

```
coletar imagens reais → treinar → validar nas fotos reais →
ativar → medir erro no uso → coletar mais imagens → repetir
```

### Métricas do modelo (aba Modelo)

| Métrica | Significado | Se estiver ruim |
|---|---|---|
| Precision | das caixas detectadas, quantas estão certas | falsos positivos: fundo, rack, cadeira metálica |
| Recall | das cadeiras existentes, quantas foram achadas | cadeiras perdidas em pilha alta |
| mAP50 | qualidade geral (IoU 0,5) | modelo ainda não deve ir a produção |
| mAP50-95 | qualidade estrita | anotações inconsistentes |

**Recall baixo** é o erro mais comum ao contar cadeiras empilhadas: as de
baixo ficam escondidas. Aumente o número de fotos de pilhas altas e cheias.

## 20. Desenvolvimento e testes

```bash
.venv/bin/pip install pytest

.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m tools.bench_counter   # benchmark sintético
.venv/bin/python -m tools.bench_real      # benchmark em fotos reais
```

Cobertura dos testes:

| Arquivo | Cobre |
|---|---|
| `test_counting.py` | algoritmo de contagem, período, Otsu, offset |
| `test_stability.py` | filtro temporal, estados, correção manual |
| `test_database.py` | tabelas, repositórios, transações |
| `test_api.py` | todos os endpoints, segurança, WebSocket |
| `test_camera.py` | imagem, vídeo, reconexão, fila de frames |
| `test_ai.py` | pilhas, tracking, modelos, métricas |
| `test_pipeline.py` | pipeline completo sem modelo |

Verificações de segurança cobertas por teste: senha RTSP nunca exposta, path
traversal bloqueado, lista de logs restrita.

Recarga automática em desenvolvimento:

```bash
python -m app.main --reload
```

---

## 21. Estrutura do projeto

```
detectcad/
├── app/
│   ├── main.py                 # ponto de entrada
│   ├── config.py               # .env → objetos de configuração
│   ├── schemas.py              # dataclasses e enums
│   ├── logging_config.py
│   ├── camera/
│   │   ├── rtsp_camera.py      # captura, reconexão, estados
│   │   └── camera_manager.py   # registro de câmeras
│   ├── ai/
│   │   ├── yolo_detector.py    # inferência + device + GPU
│   │   ├── tracker.py          # ID estável das pilhas
│   │   ├── pile_detector.py    # detecções → pilhas
│   │   └── model_manager.py    # versões, hot-swap, métricas
│   ├── counting/
│   │   ├── stack_counter.py    # ★ a contagem de cadeiras
│   │   ├── stability.py        # filtro temporal
│   │   └── confidence.py       # confiança e cores
│   ├── database/
│   │   ├── database.py         # engine, sessão, migração
│   │   ├── models.py           # 8 tabelas
│   │   └── repository.py       # acesso a dados
│   ├── api/
│   │   ├── routes.py           # ~45 endpoints
│   │   └── websocket.py        # /ws
│   ├── services/
│   │   ├── inference_service.py  # ★ o worker de IA
│   │   ├── counting_service.py
│   │   ├── snapshot_service.py
│   │   ├── dataset_service.py
│   │   ├── training_service.py
│   │   ├── calibration_service.py
│   │   └── state_store.py
│   ├── templates/dashboard.html
│   └── static/{css,js}/
├── training/
│   ├── datasets/{images,labels}/{train,val,test}/, raw/
│   ├── configs/chairs.yaml
│   ├── scripts/{train,validate,test,export,prepare_dataset}.py
│   └── runs/
├── tools/
│   ├── extract_frames.py        # vídeo/RTSP → frames para o dataset
│   ├── make_synthetic_stack.py  # pilhas sintéticas para teste
│   ├── bench_counter.py         # benchmark sintético
│   └── bench_real.py            # benchmark em fotos reais
├── data/{models,snapshots,results,config}/
├── logs/
├── tests/
├── .env / .env.example
├── requirements.txt / requirements-train.txt
├── run.sh
└── README.md
```

---

## 22. Otimizações futuras

O sistema já tem a base para:

| Melhoria | Onde mexer |
|---|---|
| **PostgreSQL** | `DATABASE_URL=postgresql+psycopg://...` (sem alterar código) |
| **Mais câmeras** | `POST /api/cameras`; cada uma tem estado próprio |
| **Alertas externos** | `app/services/*` + Telegram/WhatsApp/e-mail |
| **Um modelo por tipo de cadeira** | `model_versions` + `camera_id` |
| **Vários depósitos** | campo `site_id` no banco |
| **Login e permissões** | middleware do FastAPI |
| **Imagens em NAS** | `settings.snapshots_dir` → caminho do NAS |
| **Treino remoto** | `training_service` já isola o processo |
| **Modelos maiores** | `yolo11m/l/x.pt`, `IMGSZ` maior |
| **Inferência ONNX** | `training/scripts/export.py` |
| **Ajuste fino** | `train.py --resume` |

---

## Licença e responsabilidade

Este sistema é uma **ferramenta de apoio**. A contagem final do estoque é
responsabilidade da pessoa que opera: por isso existe a correção manual, a
confiança por pilha e o registro de auditoria. Use os alertas
(`ALERT_TOTAL_MIN`, `ALERT_TOTAL_MAX`) como rede de segurança.
