/* =========================================================================
   CONTADOR DE CADEIRAS — dashboard (sem frameworks)
   - WebSocket /ws  -> estado em tempo real
   - fetch /api/*   -> ações e páginas
   =========================================================================

   ARQUIVO / MAPA — static/js/dashboard.js
   Um arquivo só, sem build, sem framework. O navegador carrega, roda o
   boot() no DOMContentLoaded e pronto.

   Duas fontes de dado, com responsabilidades bem separadas:
     - WebSocket: TEMPO REAL. O backend empurra um snapshot de estado
       (pilhas, total, confiança, câmera, alertas) várias vezes por
       segundo. É o que mantém a aba Dashboard viva sem polling.
     - fetch: AÇÃO E PÁGINA. Só quando o usuário clica em algo ou troca de
       aba. Nenhuma aba faz polling de tabela, exceto Treinamento.
   A imagem nunca passa por aqui: é o <img src="/api/stream.mjpg"> do HTML.

   Ordem de leitura (por aba, seguindo o dashboard):
     0. Infraestrutura   estado S, helpers $, toast, modal, api(), WS,
                         applyState e as funções de pintura do cabeçalho
     1. Vídeo/Dashboard  updateHeader, updateTotals, updateAlerts,
                         renderPiles, saveCorrection
     2. Câmera           loadCamera + troca com/sem overlay
     3. Contagem         loadCounting (correções e offset sugerido)
     4. Histórico        loadHistory, exportação CSV
     5. Calibração       loadCalibration, ROI no canvas, diagnóstico
     6. Dataset          loadDataset, divisão, upload
     7. Anotação         loadAnnotation
     8. Treinamento      loadTraining, polling e log
     9. Modelo           loadModels, ativar, rollback, importar
     10. Histórico/Config loadSettings
     11. Logs            loadLogs, eventos ao vivo
     12. Roteamento      TITLES, LOADERS, route()
     13. Início          boot()

   Duas decisões que valem lembrar antes de mexer em qualquer função:
     - applyState() só redesenha as listas de pilhas quando a assinatura
       (id:contagem:status) muda. Sem isso, o campo de correção manual
       perderia o foco a cada frame e o digito sumiria.
     - Quase todo texto que vem da API entra por textContent, não innerHTML.
       Só listas montadas por concatenação usam innerHTML. Dados de log e de
       frame são os mais expostos a "<" e "&".
*/

'use strict';

// ------------------------------------------------------------------ estado
/*
  Objeto único com todo o estado mutável do front-end. Existe para evitar
  vazamento de global e para o roteador poder ler de um lugar só.
  - ws/wsRetry: conexão e contador de tentativas de reconexão
  - state: último snapshot do backend (piles fica junto, por conveniência)
  - page: aba atual. Vários trechos param de trabalhar se a aba não estiver
    visível — é o throttle mais barato que existe aqui (evita renderizar
    lista que ninguém está olhando).
  - roi: {x,y,w,h} em pixels DO FRAME (não da tela). Por isso a ROI continua
    valendo depois de redimensionar a janela.
*/
const S = {
  ws: null,
  wsRetry: 0,
  state: null,
  piles: [],
  page: 'dashboard',
  overlay: true,
  roi: null,              // {x,y,w,h} em pixels do frame
  dragging: null,
  lastHist: [],
  pollTimer: null,
  charts: {},
};

// Helpers mínimos. $, $$ e fmt existem para escrever menos e para ter UM
// lugar de normalizar dado incompleto: fmt(null), fmt(undefined) e fmt(NaN)
// devolvem "—" em vez de "NaN" na tela. Isso importa muito aqui, porque a
// API pode omitir campos e um "NaN" na tela parece erro do sistema.
// pct() é separado de fmt() de propósito: confiança chega como 0..1 e o
// valor mostrado é sempre inteiro em %.
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const fmt = (v, d = 0) => (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toFixed(d);
const pct = v => Math.round((v || 0) * 100);

// ------------------------------------------------------------------ toast
// Aviso efêmero. textContent (e não innerHTML) porque a mensagem costuma
// vir do backend e pode conter "<" ou "&": innerHTML interpretaria como
// marcação. O timer anterior é cancelado a cada chamada, então mensagens
// seguidas não se sobrepõem.
let toastTimer = null;
function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast ' + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 4200);
}

// ------------------------------------------------------------------ modal
// Confirmação vira Promise para o chamador ler como if/else, sem callback.
// Os handlers são limpos (onclick = null) ao resolver: sem isso, o botão
// guardaria a Promise antiga e um clique duplo dispararia duas vezes.
function confirmDialog(title, text) {
  return new Promise(resolve => {
    const m = $('#modal');
    $('#modal-title').textContent = title;
    $('#modal-text').textContent = text;
    m.classList.remove('hidden');
    const yes = $('#modal-yes'), no = $('#modal-no');
    const done = (v) => { m.classList.add('hidden'); yes.onclick = null; no.onclick = null; resolve(v); };
    yes.onclick = () => done(true);
    no.onclick = () => done(false);
  });
}

// ------------------------------------------------------------------ fetch
// Wrapper único de rede. Concentrar aqui significa que TODA chamada trata
// erro do mesmo jeito e que nenhuma tela precisa lidar com HTTP cru.
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  // Resposta sem JSON (ex.: 502 do proxy, página de erro) não pode quebrar
  // o chamador: cai no statusText. E o throw abaixo transforma erro HTTP em
  // exceção com mensagem em português vinda do backend, que o catch de cada
  // tela mostra no toast.
  let body = {};
  try { body = await res.json(); } catch (e) { body = { ok: false, message: res.statusText }; }
  if (!res.ok) throw new Error(body.message || ('HTTP ' + res.status));
  return body;
}

// ==================================================================== WS ===
/*
  Ponto único de tempo real. msg.type decide o destino:
    hello -> snapshot inicial; já vem renderizado por applyState depois,
             então aqui só é ignorado para não pintar duas vezes
    state -> o trabalho de verdade
    event -> linha nova na aba Logs

  A reconexão é exponencial e SEM "jitter" (500ms * 1.7^tentativa, teto 6).
  Serve para não martelar o servidor quando a rede cai, mas sofre com muitos
  dashboards subindo juntos (thundering herd) — se isso virar problema, o
  próximo passo é sortear um atraso aleatório nessa base.

  onerror chama close() de propósito: o caminho de reconexão é sempre o
  onclose, então existe um único lugar que sabe reconectar.
*/
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  S.ws = ws;

  ws.onopen = () => {
    S.wsRetry = 0;
    $('#ws-indicator').className = 'ws-on';
    $('#ws-text').textContent = 'ao vivo';
  };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    if (msg.type === 'hello') return;
    if (msg.type === 'event') { addEventRow(msg); return; }
    if (msg.type === 'state') { applyState(msg); }
  };
  ws.onclose = () => {
    $('#ws-indicator').className = 'ws-off';
    $('#ws-text').textContent = 'reconectando…';
    S.wsRetry = Math.min(S.wsRetry + 1, 6);
    setTimeout(connectWS, 500 * Math.pow(1.7, S.wsRetry));
  };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}

/*
  Coração do dashboard: roda a cada snapshot do backend (várias vezes por
  segundo). A ordem das chamadas importa.

  A ASSINATURA (sig) é a otimização central do arquivo. O estado chega
  completo a cada frame, mas as listas só são reconstruídas quando muda
  pilha:contagem:status. Sem isso, a cada 200ms o innerHTML da lista seria
  reescrito e:
    - o campo de correção manual perderia o foco e o valor digitado sumiria;
    - a lista piscaria, impossível de clicar numa pilha;
    - o custo de layout cresceria sem necessidade.
  O que muda o tempo todo (fps, uptime, latência) não entra na assinatura de
  propósito: é por isso que esses números podem mudar sem redesenhar nada.

  _sig fica na própria função como memória entre chamadas (evita colocar
  no estado global). Se applyState for chamada sem argumento isso quebra,
  então quem chama sempre passa o snapshot.
*/
function applyState(st) {
  S.state = st;
  const piles = st.piles || [];
  // Um modelo genérico pré-treinado (COCO) não foi feito para este ambiente:
  // ele enxerga a PILHA INTEIRA como se fosse uma cadeira. As contagens que saem
  // daí medem textura (ripas do encosto, nervuras das pernas), não cadeiras -
  // por isso o número varia e a confiança pode ficar alta por engano.
  //
  // O backend já manda `model_trained` no payload. Sem modelo treinado, nenhum
  // número é medição, independente do que a confiança dizer.
  const measurable = !!st.model_trained;
  // Só redesenha as listas se a pilha mudou (evita piscar)
  const sig = measurable + '|' + piles.map(p => `${p.pile_id}:${p.count}:${p.status}`).join('|');
  if (sig !== applyState._sig) { applyState._sig = sig; renderPiles(piles, measurable); renderCountTable(piles); }
  updateTotals(st);
  updateHeader(st);
  updateAlerts(st.alerts || []);
  // O gráfico só cresce na aba Dashboard: os outros canvases são recarregados
  // sob demanda pelo load*() da própria aba.
  if (S.page === 'dashboard') pushChartPoint(st);
  // Primeiro estado = primeiro frame recebido. O "aguardando…" some aqui.
  $('#video-empty')?.classList.add('hidden');
  $('#video2-empty')?.classList.add('hidden');
}

/*
  Cabeçalho e indicadores. Tudo por textContent: são valores vindos da API.
  Regra de ouro do painel aqui: valor ausente é "—", nunca 0. Zéro parece
  uma medição e levaria o operador a conclusão errada.
*/
function updateHeader(st) {
  const cam = (st.camera_state || 'OFFLINE').toUpperCase();
  const pill = $('#cam-state-pill');
  pill.textContent = cam;
  // Só três cores para o estado da câmera: conectada, em transição, fora.
  // Qualquer estado desconhecido cai em "offline" — melhor pessimista.
  pill.className = 'pill ' + (cam === 'ONLINE' ? 'pill-online' : cam === 'CONNECTING' || cam === 'RECONNECTING' ? 'pill-warn' : 'pill-offline');

  $('#stat-camera').textContent = cam === 'ONLINE' ? 'ONLINE' : cam;
  $('#stat-piles').textContent = st.pile_count ?? 0;
  // Confiança 0 vira "—", não "0%": zero aqui significa "não sei", e mostrar
  // 0% faria o operador desconfiar de uma contagem que talvez esteja certa.
  $('#stat-conf').textContent = (st.confidence > 0 ? pct(st.confidence) + '%' : '—');
  $('#meta-fps').textContent = 'FPS ' + fmt(st.fps, 1);
  $('#meta-res').textContent = st.mode ? 'modo: ' + st.mode : '';

  const badge = $('#mode-badge');
  const mode = st.mode || 'iniciando';
  badge.textContent = mode;
  // production = verde (pode confiar no número). preparation = vermelho
  // (câmera em ajuste, número não vale ainda). resto = âmbar.
  badge.className = 'badge ' + (mode.startsWith('production') ? 'badge-ok' : mode === 'preparation' ? 'badge-err' : 'badge-warn');
}

/*
  O número mais importante da tela. A animação de escala só dispara quando o
  texto realmente muda (guarda `el.textContent !== String(t)`), senão o total
  pulsaria a cada frame e o movimento passaria a ser ruído, não sinal.
*/
function updateTotals(st) {
  const t = st.total ?? 0;
  const el = $('#total-value');
  if (el.textContent !== String(t)) {
    el.textContent = t;
    el.style.color = '#fff';
    // Web Animations API: sem CSS extra, e some sozinha no fim.
    el.animate([{ transform: 'scale(1.10)', color: '#3ec93e' }, { transform: 'scale(1)', color: '#fff' }], { duration: 420 });
  }
  // "tentativo" é o total que teria se as pilhas instáveis adotassem a
  // estimativa mais alta. Fica separado do total firme para o operador
  // enxergar o tamanho da incerteza.
  const tt = st.totals || {};
  $('#total-tentative').textContent = tt.total_tentative ? `(+${tt.total_tentative} instável)` : 'todas estáveis';
  $('#total-piles').textContent = `${st.pile_count || 0} pilha(s)`;
}

/*
  Alertas. A barra é esvaziada e reescrita a cada estado: é uma lista
  instantânea do backend, não um histórico. O ícone vem do nível, e
  textContent evita que a mensagem do backend vire HTML.
*/
function updateAlerts(alerts) {
  const bar = $('#alerts-bar');
  bar.innerHTML = '';
  alerts.forEach(a => {
    const d = document.createElement('div');
    d.className = 'alert alert-' + (a.level || 'info');
    d.textContent = (a.level === 'error' ? '⛔ ' : a.level === 'warning' ? '⚠ ' : 'ℹ ') + a.message;
    bar.appendChild(d);
  });
}

// ------------------------------------------------------------- pilhas (cards)
/*
  Mapa estado -> classe de tag. Este é o MESMO dicionário de cores do
  overlay desenhado no frame (counting/confidence.py, STATUS_COLORS_HEX):
  se a caixa do vídeo é verde, a linha da lista também é. Uma cor, uma
  verdade, mesmo significado no vídeo e na tabela.
  Estado novo vindo do backend cai em tag-unk em vez de quebrar o layout.
*/
const STATUS_TAG = { STABLE: 'tag-ok', UNSTABLE: 'tag-warn', LOW_CONFIDENCE: 'tag-low', PARTIAL: 'tag-part', UNKNOWN: 'tag-unk' };

/*
  Cartões de pilha do Dashboard. Só é chamada quando a assinatura muda
  (ver applyState), então aqui não existe throttle: a proteção está na
  frente.

  O HTML é montado por template string com valores que vêm da API. Isso
  seria um problema de XSS se os valores fossem texto livre do usuário —
  aqui os campos numéricos e o status são controlados pelo backend, e o
  único texto livre (o nome da classe na estimativa) vem do próprio
  detector. Se um dia entrar nome de arquivo ou comentário do operador
  neste template, tem que escapar antes.

  O "detail" mostra os estimadores candidatos lado a lado (ex.: size=7
  periodicity=8). É o dado mais útil em campo: quando a IA erra, o motivo
  quase sempre é visível na divergência entre eles.
*/
function renderPiles(piles, measurable) {
  const box = $('#piles-list');
  if (!piles.length) { box.innerHTML = '<div class="empty">Nenhuma pilha detectada.</div>'; return; }
  box.innerHTML = '';
  piles.forEach(p => {
    const row = document.createElement('div');
    row.className = 'pile s-' + p.status;
    const est = Object.entries(p.candidates || {}).map(([k, v]) => `${k}=${Math.round(v)}`).join(' ');
    // Só mostra o número quando o backend diz que ele é confiável.
    //
    // O backend já calcula isso (confidence.combine -> `uncertain`; o
    // PileState chega como `status`), mas antes este arquivo renderizava
    // `p.count` sempre. Resultado: uma leitura que o próprio sistema
    // classificava como LOW_CONFIDENCE aparecia na tela com a mesma
    // prominence de uma medida, e era anotada no papel como se fosse real.
    //
    // Um número grande e errado é pior que "?" - o "?" diz ao operador que
    // ele precisa de modelo treinado ou de calibração, o número não diz nada.
    const trusted = measurable && (p.status === 'STABLE' || p.status === 'PARTIAL');
    // O valor bruto continua na linha de detalhe (e no input do stepper):
    // o operador precisa enxergar o que a IA pensou para poder corrigir, e
    // o stepper precisa de um número para funcionar.
    const shown = trusted ? p.count : '?';
    const tentative = trusted ? '' : ` · leitura ${p.count}`;
    row.innerHTML = `
      <div class="pile-id">PILHA ${p.pile_id}</div>
      <div class="pile-count" title="${trusted ? 'Contagem estabilizada' : 'Leitura não confiável: o número não foi validado'}">${shown}</div>
      <div class="pile-info">
        <div class="pile-status">${statusLabel(p.status)}</div>
        <div class="pile-detail">${est || 'sem estimadores'}${tentative}${p.manual ? ' · manual' : ''}</div>
      </div>
      <div class="pile-conf">${pct(p.confidence)}%</div>
      <div class="stepper">
        <button data-d="-1">−</button>
        <input type="number" value="${p.count}" data-id="${p.pile_id}">
        <button data-d="1">+</button>
      </div>`;
    box.appendChild(row);
  });
  // Handlers reaproveitados a cada redesenho. Como o innerHTML substitui tudo
  // de uma vez, não há vazamento de listener antigo: os nós antigos morrem
  // junto com seus handlers. Delegação seria mais barata aqui.
  box.querySelectorAll('.stepper button').forEach(b => {
    b.onclick = () => {
      const input = b.parentElement.querySelector('input');
      // Math.max(0, ...): não existe pilha negativa, e o input pode estar
      // vazio quando o usuário está digitando (parseInt devolve NaN).
      input.value = Math.max(0, (parseInt(input.value, 10) || 0) + parseInt(b.dataset.d, 10));
    };
  });
  // onchange (e não oninput): só salva quando o usuário sai do campo, o que
  // também evita chamar o modal de confirmação a cada tecla digitada.
  box.querySelectorAll('.stepper input').forEach(inp => {
    inp.onchange = () => saveCorrection(inp.dataset.id, parseInt(inp.value, 10));
  });
}

// Traduz o estado técnico para português. Fallback devolve o próprio valor
// de entrada: estado novo da API aparece cru em vez de "undefined" na tela.
function statusLabel(s) {
  return { STABLE: 'ESTÁVEL', UNSTABLE: 'INSTÁVEL', LOW_CONFIDENCE: 'BAIXA CONFIANÇA', PARTIAL: 'PARCIAL', UNKNOWN: 'INDETERMINADO' }[s] || s;
}

/*
  Correção manual: o dado mais valioso do sistema inteiro. Cada confirmação
  grava o par (o que a IA viu, o que era verdade) no banco, e é esse
  histórico que alimenta o offset sugerido e, depois, um retreinamento.

  Por isso a confirmação mostra os DOIS números e avisa do registro. Sem
  esse texto o operador corrigiria sem saber que está gerando dado de treino.
  Valores abaixo de zero ou não numéricos são descartados aqui, e não no
  backend, para nem gastar ida ao servidor.
*/
async function saveCorrection(pileId, value) {
  if (isNaN(value) || value < 0) return;
  const p = S.piles.find(x => x.pile_id == pileId);
  if (!p) return;
  const ok = await confirmDialog('Confirmar correção',
    `A IA informou ${p.count} cadeiras para a pilha ${pileId}.\n\nVocê está definindo ${value}.\n\nO registro AI_COUNT × CORRECT_COUNT será guardado para melhorar o modelo.`);
  if (!ok) return;
  try {
    const r = await api(`/api/count/${pileId}/correct`, { method: 'POST', body: JSON.stringify({ count: value }) });
    toast(r.message || 'Corrigido', 'ok');
  } catch (e) { toast(e.message, 'err'); }
}

// ------------------------------------------------------------- tabela
/*
  Tabela detalhada da aba Contagem. Também é alimentada pelo WebSocket
  (mesmo guardião de assinatura do Dashboard), então aparece preenchida
  mesmo sem abrir a aba. É a visão "por que a IA disse N" — cada coluna
  é uma peça do cálculo:

    method                qual estimador venceu a fusão
    detection_confidence  o YOLO viu a pilha com certeza
    stability_confidence  os últimos frames concordaram entre si
    pitch_px              distância entre cadeiras; sem padrão vertical
                          estável não há como contar por periodicidade
    manual                sinal de que alguém já corrigiu à mão
*/
function renderCountTable(piles) {
  const tb = $('#count-table tbody');
  if (!piles.length) { tb.innerHTML = '<tr><td colspan="9" class="empty">sem pilhas</td></tr>'; return; }
  tb.innerHTML = piles.map(p => `
    <tr>
      <td>Pilha ${p.pile_id}</td>
      <td class="num"><b>${p.count}</b></td>
      <td class="num">${pct(p.confidence)}%</td>
      <td><span class="tag ${STATUS_TAG[p.status] || 'tag-unk'}">${statusLabel(p.status)}</span></td>
      <td>${p.method}</td>
      <td class="num">${pct(p.detection_confidence)}%</td>
      <td class="num">${pct(p.stability_confidence)}%</td>
      <td class="num">${fmt(p.pitch_px, 1)}px</td>
      <td>${p.manual ? 'manual' : ''}</td>
    </tr>`).join('');
}

// ================================================================ CHARTS ===
/*
  Chart.js vem de CDN e é a ÚNICA dependência externa do sistema. O app
  precisa funcionar numa rede de fábrica, então o onerror não faz nada
  visível: apenas chama o callback, makeChart devolve null e cada
  atualização de gráfico já checa "se o chart existe". Resultado: o vídeo e
  as contagens continuam funcionando sem internet, só sem gráficos.
*/
function loadChartJs(cb) {
  if (window.Chart) return cb();
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';
  s.onload = cb;
  s.onerror = () => { console.warn('Chart.js indisponível offline; gráficos desativados.'); cb(); };
  document.head.appendChild(s);
}

function makeChart(id, color) {
  const cv = document.getElementById(id);
  if (!cv || !window.Chart) return null;
  const ctx = cv.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, cv.height);
  grad.addColorStop(0, color + '55'); grad.addColorStop(1, color + '00');
  // animation:false e update('none') lá embaixo: com dados chegando várias
  // vezes por segundo, animar cada ponto consome CPU e a linha parece
  // "escorregar". Um gráfico de operação deve ser estático e imediato.
  return new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Total de cadeiras', data: [], borderColor: color, backgroundColor: grad, fill: true, tension: .28, pointRadius: 0, borderWidth: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { ticks: { color: '#8b98a5', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#2a323d33' } },
        y: { ticks: { color: '#8b98a5', font: { size: 10 } }, grid: { color: '#2a323d66' }, beginAtZero: true, precision: 0 },
      },
      plugins: { legend: { labels: { color: '#8b98a5', font: { size: 11 } } } },
    },
  });
}

function initCharts() {
  S.charts.main = makeChart('chart-main', '#3b82f6');
  S.charts.count = makeChart('chart-count', '#3ec93e');
  S.charts.history = makeChart('chart-history', '#f0c828');
}

/*
  Gráfico ao vivo do Dashboard. Recebe um ponto por estado do WebSocket e
  mantém no máximo 180 em S.lastHist — esse teto é a única forma de o
  gráfico não crescer sem limite durante um turno de 8h.
  A janela desejada (60/600/3600) é lida do select a cada ponto, então
  trocar a faixa não redesenha nem recarrega nada.
*/
function pushChartPoint(st) {
  const now = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  S.lastHist.push({ t: now, v: st.total ?? 0 });
  if (S.lastHist.length > 180) S.lastHist.shift();
  const range = parseInt($('#chart-range')?.value || '600', 10);
  const ch = S.charts.main;
  if (!ch) return;
  ch.data.labels = S.lastHist.map(p => p.t);
  ch.data.datasets[0].data = S.lastHist.map(p => p.v);
  ch.update('none');
}

// Listener intencionalmente vazio: o select é lido no próximo ponto por
// pushChartPoint. Existe só para documentar que a troca de faixa é local.
$('#chart-range')?.addEventListener('change', () => { /* próximo ponto já respeita o range */ });

// ============================================================ PAGE: CÂMERA ===
/*
  Diagnóstico da conexão (GET /api/cameras + GET /api/status), só ao abrir a
  aba. Listar o que separa "câmera ruim" de "contagem ruim":
    FPS da fonte vs. FPS de processamento -> se o processing FPS é menor,
      o gargalo é a inferência, não a câmera
    frames perdidos / inválidos -> rede ruim ou codec
    reconexões -> instabilidade de rede ou servidor RTSP caindo
    latência de captura -> na verdade é o tempo de INFERÊNCIA medido; o
      rótulo é o mais próximo que o painel tem
*/
async function loadCamera() {
  try {
    const r = await api('/api/cameras');
    const st = await api('/api/status');
    const cam = (st.camera?.cameras || [])[0] || {};
    const rows = [
      ['Status', (st.camera?.state || '—').toUpperCase()],
      ['URL (mascarada)', st.config?.camera?.rtsp_url || cam.rtsp_url || '—'],
      ['Resolução', cam.width ? `${cam.width} × ${cam.height}` : '—'],
      ['FPS da fonte', fmt(cam.source_fps, 2)],
      ['FPS de processamento', fmt(st.system?.process_fps, 2)],
      ['Tempo conectado', cam.uptime_sec ? `${Math.floor(cam.uptime_sec / 60)} min ${Math.round(cam.uptime_sec % 60)} s` : '—'],
      ['Reconexões', cam.reconnects ?? 0],
      ['Frames capturados', cam.frames_captured ?? 0],
      ['Frames perdidos', cam.dropped_frames ?? 0],
      ['Frames inválidos', cam.invalid_frames ?? 0],
      ['Latência de captura', cam.last_error ? `erro: ${cam.last_error}` : (st.system?.inference_ms ? fmt(st.system.inference_ms, 1) + ' ms (inferência)' : '—')],
    ];
    $('#camera-table tbody').innerHTML = rows.map(([k, v]) => `<tr><td class="muted">${k}</td><td>${v}</td></tr>`).join('');
    $('#camera-note').textContent = 'A senha da URL nunca é enviada ao navegador: o backend mascara antes de responder.';
    $('#meta-url').textContent = st.config?.camera?.rtsp_url || '';
  } catch (e) { toast(e.message, 'err'); }
}

/*
  Alterna entre o stream com overlay e o frame JPEG sem overlay.
  Detalhes que não são óbvios:
    - o src é limpo antes de ser trocado. Sem isso, o navegador às vezes
      mantém a imagem anterior e o usuário acha que o botão não funcionou.
    - só o modo SEM overlay recebe ?t=Date.now(). O stream MJPEG não é
      cacheado assim, mas o frame único JPEG é: sem o parâmetro, o
      navegador mostraria sempre a mesma imagem antiga.
*/
$('#toggle-overlay')?.addEventListener('click', (e) => {
  S.overlay = !S.overlay;
  e.target.textContent = S.overlay ? 'Ver imagem original' : 'Ver com contagem';
  const src = S.overlay ? '/api/stream.mjpg' : '/api/frame.raw.jpg';
  $('#video2').src = '';
  $('#video2').src = src + (S.overlay ? '' : '?t=' + Date.now());
});

// ========================================================= PAGE: CONTAGEM ===
/*
  Estatísticas das correções manuais (GET /api/corrections?limit=50).
  É a aba que responde "a IA está errando sempre para o mesmo lado?".
  O erro médio absoluto mostra o tamanho do erro; a taxa de acerto exato
  mostra quantas contagens não precisaram de nada. Se o erro médio for
  sempre +1, o problema é o modelo de altura da cadeira (calibração), não
  o detector.
*/
async function loadCounting() {
  try {
    const r = await api('/api/corrections?limit=50');
    const st = r.stats || {};
    $('#correction-stats').innerHTML = `
      <div class="cs-item"><b>${st.total || 0}</b>correções</div>
      <div class="cs-item"><b>${st.with_error || 0}</b>com erro</div>
      <div class="cs-item"><b>${fmt(st.mean_abs_error, 2)}</b>erro absoluto médio</div>
      <div class="cs-item"><b>${st.exact_match_rate != null ? pct(st.exact_match_rate) + '%' : '—'}</b>acertos exatos</div>`;
    const tb = $('#correction-table tbody');
    tb.innerHTML = (r.corrections || []).length
      ? r.corrections.map(c => {
          const d = c.correct_count - c.ai_count;
          return `<tr><td>${c.timestamp.replace('T', ' ')}</td><td class="num">${c.ai_count}</td>
                  <td class="num"><b>${c.correct_count}</b></td>
                  <td class="num" style="color:${d === 0 ? 'var(--ok)' : 'var(--low)'}">${d > 0 ? '+' : ''}${d}</td></tr>`;
        }).join('')
      : '<tr><td colspan="4" class="empty">nenhuma correção registrada</td></tr>';

    // A sugestão de offset só aparece com amostras suficientes. Com poucas
    // correções, o backend marca confiável = não justamente para não
    // aplicar um número achado no chute na aba Calibração.
    const sug = r.offset_suggestion || {};
    if (sug.samples) {
      $('#offset-suggestion').textContent =
        `Offset sugerido: ${sug.offset} — ${sug.message} (${sug.samples} amostras, confiável: ${sug.reliable ? 'sim' : 'não'})`;
    }
  } catch (e) { toast(e.message, 'err'); }
}

$('#btn-snapshot')?.addEventListener('click', async () => {
  try { const r = await api('/api/snapshots', { method: 'POST' }); toast(r.message, 'ok'); }
  catch (e) { toast(e.message, 'err'); }
});

// ========================================================== PAGE: HISTÓRICO ===
/*
  Histórico vem do banco sob demanda (GET /api/history?hours=N), não do
  WebSocket. Motivo: histórico é imutável, então consultar a cada frame só
  gastaria CPU e latência sem trazer novidade.

  No modo "personalizado" o intervalo é calculado em horas a partir das duas
  datas. O .reverse() existe porque o banco devolve do mais novo para o mais
  antigo e a tabela quer ordem cronológica; o gráfico usa a série como veio.
*/
async function loadHistory() {
  const sel = $('#hist-range').value;
  let hours = parseInt(sel, 10);
  if (sel === 'custom') {
    const f = $('#hist-from').value, t = $('#hist-to').value;
    if (!f || !t) { toast('Escolha as datas.', 'err'); return; }
    hours = Math.max(1, Math.round((new Date(t) - new Date(f)) / 36e5));
  }
  try {
    const r = await api(`/api/history?hours=${hours}`);
    const recs = (r.records || []).slice().reverse();
    $('#hist-count').textContent = `${recs.length} registro(s) · total atual: ${r.current_total}`;

    if (S.charts.history) {
      const series = r.totals_series || [];
      S.charts.history.data.labels = series.map(s => s.timestamp.replace('T', ' ').slice(5, 16));
      S.charts.history.data.datasets[0].data = series.map(s => s.total);
      S.charts.history.update('none');
    }
    $('#history-table tbody').innerHTML = recs.length ? recs.map(x => `
      <tr>
        <td>${x.timestamp.replace('T', ' ')}</td>
        <td>${x.pile_id ?? '—'}</td>
        <td class="num"><b>${x.count}</b></td>
        <td class="num">${pct(x.confidence)}%</td>
        <td><span class="tag ${STATUS_TAG[x.status] || 'tag-unk'}">${statusLabel(x.status)}</span></td>
        <td>${x.method}</td>
        <td class="num">${x.previous_count ?? '—'}</td>
      </tr>`).join('') : '<tr><td colspan="7" class="empty">sem registros no período</td></tr>';
  } catch (e) { toast(e.message, 'err'); }
}

$('#hist-range')?.addEventListener('change', (e) => {
  // Os campos de data só aparecem no modo custom. Nesse modo o load é
  // adiado: espera o usuário escolher as duas datas.
  const custom = e.target.value === 'custom';
  $('#hist-from').classList.toggle('hidden', !custom);
  $('#hist-to').classList.toggle('hidden', !custom);
  if (custom) return;
  loadHistory();
});

/*
  Exportação CSV feita no navegador: monta a string, cria um Blob com
  text/csv e dispara um <a download> temporário. Evita uma rota nova no
  backend para um relatório de dados que já estão na tela.
  Separador ";" em vez de "," porque o Excel em pt-BR trata a vírgula como
  separador decimal e abriria o arquivo com todas as colunas grudadas.
*/
$('#hist-export')?.addEventListener('click', async () => {
  const hours = $('#hist-range').value === 'custom' ? 720 : parseInt($('#hist-range').value, 10);
  const r = await api(`/api/history?hours=${hours}`);
  const head = 'timestamp;pile_id;chair_count;confidence;status;method;previous_count;total\n';
  const body = (r.records || []).map(x =>
    [x.timestamp, x.pile_id, x.count, x.confidence, x.status, x.method, x.previous_count, x.total].join(';')
  ).join('\n');
  const blob = new Blob([head + body], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `historico_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
});

// ========================================================== PAGE: CALIBRAÇÃO ===
/*
  Hidrata o formulário a partir de GET /api/calibration iterando as chaves
  do objeto e casando com form.elements[nome]. Por isso o HTML é a fonte da
  verdade dos campos: campo sem name= nunca é lido nem salvo.
  Exceções explícitas (camera_id, updated_at, roi) são puladas porque são
  só de leitura; a flag "enabled" mora no checkbox roi_enabled e é
  copiada na mão.
*/
async function loadCalibration() {
  try {
    const r = await api('/api/calibration');
    const c = r.calibration;
    const form = $('#calib-form');
    Object.keys(c).forEach(k => {
      const input = form.elements[k];
      if (input && k !== 'camera_id' && k !== 'updated_at' && k !== 'roi') {
        if (input.type === 'checkbox') input.checked = !!c[k];
        else input.value = c[k];
      }
    });
    form.elements.roi_enabled.checked = !!c.enabled;
    if (c.roi && c.roi.w > 0) S.roi = { ...c.roi };
    drawRoi();
    $('#roi-values').textContent = S.roi ? `ROI: x=${S.roi.x} y=${S.roi.y} w=${S.roi.w} h=${S.roi.h}` : 'sem ROI';
    if (r.suggestion?.samples) {
      $('#offset-suggestion').textContent = `Offset sugerido: ${r.suggestion.offset} — ${r.suggestion.message}`;
    }
    setupRoiCanvas();
  } catch (e) { toast(e.message, 'err'); }
}

/*
  Salvar calibração. Diferente das outras abas, o payload é montado campo a
  campo à mão em vez de serializar o form: os valores usam "+" para virar
  número de verdade (um input vazio viraria NaN, que o JSON.stringify
  transformaria em null) e a ROI vem de S.roi, que não é um input do
  formulário. Send field-by-field também deixa explícito o contrato com a
  API em vez de confiar em "serialize tudo que tem name".
*/
$('#calib-form')?.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const payload = {
    enabled: form.elements.roi_enabled.checked,
    roi: S.roi || { x: 0, y: 0, w: 0, h: 0 },
    chair_height_px: +form.elements.chair_height_px.value,
    confidence_threshold: +form.elements.confidence_threshold.value,
    counting_method: form.elements.counting_method.value,
    stability_frames: +form.elements.stability_frames.value,
    stability_min_confidence: +form.elements.stability_min_confidence.value,
    change_min_frames: +form.elements.change_min_frames.value,
    min_confidence: +form.elements.min_confidence.value,
    count_offset: +form.elements.count_offset.value,
    chair_classes: form.elements.chair_classes.value,
    pile_classes: form.elements.pile_classes.value,
  };
  try { const r = await api('/api/calibration', { method: 'POST', body: JSON.stringify(payload) }); toast(r.message, 'ok'); }
  catch (e) { toast(e.message, 'err'); }
});

$('#calib-reset')?.addEventListener('click', async () => {
  const ok = await confirmDialog('Restaurar calibração', 'Todos os parâmetros voltam ao padrão do .env. Confirma?');
  if (!ok) return;
  await api('/api/calibration/reset', { method: 'POST' });
  loadCalibration(); toast('Calibração restaurada.', 'ok');
});

/*
  Medir a altura de UMA cadeira: o usuário desenha a ROI de uma cadeira e o
  backend devolve a altura em pixels. Esse número é a base do método "size"
  (altura da pilha dividida pela altura da cadeira) e por isso é a
  calibração mais valiosa do sistema. O envio de measure acontece agora, mas
  o valor só vira oficial no POST do formulário.
*/
$('#measure-chair')?.addEventListener('click', async () => {
  if (!S.roi) { toast('Desenhe a caixa de UMA cadeira primeiro.', 'err'); return; }
  try {
    const r = await api('/api/calibration/measure', { method: 'POST', body: JSON.stringify({ roi: S.roi }) });
    $('#calib-form').elements.chair_height_px.value = r.measured_height_px;
    toast(r.message, 'ok');
  } catch (e) { toast(e.message, 'err'); }
});

/*
  Diagnóstico: mostra o raciocínio do contador dentro da ROI, em grid.
  É o painel de maior densidade de informação do projeto: passo vertical
  detectado, camadas visíveis, razão extensão/passo (o quociente que dá a
  contagem), qualidade do padrão, picos brutos e — o mais útil de todos —
  a lista de reasons, que diz em português por que o contador desconfiou.
  Não afeta produção: é leitura da imagem.
*/
$('#diag-run')?.addEventListener('click', async () => {
  if (!S.roi) { toast('Desenhe uma ROI primeiro.', 'err'); return; }
  try {
    const r = await api('/api/diagnostics/counter', { method: 'POST', body: JSON.stringify({ roi: S.roi }) });
    const a = r.analysis, e = r.estimate;
    $('#diag-out').innerHTML = [
      ['Contagem estimada', e.count],
      ['Confiança', pct(e.confidence) + '%'],
      ['Passo detectado', fmt(a.pitch, 1) + ' px'],
      ['Camadas visíveis', a.n_lattice],
      ['Extensão / passo', a.n_extent],
      ['Qualidade do padrão', pct(a.periodicity_quality) + '%'],
      ['Picos brutos', a.n_peaks_raw],
      ['Altura da cadeira', fmt(e.chair_height_px, 0) + ' px'],
    ].map(([k, v]) => `<div class="diag-item"><b>${v}</b><span>${k}</span></div>`).join('')
      + (a.reasons?.length ? `<div class="note" style="grid-column:1/-1">${a.reasons.join('\n')}</div>` : '')
      + `<div class="note" style="grid-column:1/-1">Estimadores: ${JSON.stringify(e.candidates)}</div>`;
    $('#diag-summary').textContent = r.message;
  } catch (err) { toast(err.message, 'err'); }
});

$('#roi-clear')?.addEventListener('click', () => { S.roi = null; drawRoi(); $('#roi-values').textContent = 'sem ROI'; });
$('#roi-pile')?.addEventListener('click', async () => {
  try {
    const r = await api('/api/status');
    const piles = (r.piles || []);
    if (!piles.length) { toast('Nenhuma pilha detectada para marcar.', 'err'); return; }
    const p = piles[0];
    S.roi = { x: Math.round(p.bbox[0]), y: Math.round(p.bbox[1]), w: Math.round(p.bbox[2] - p.bbox[0]), h: Math.round(p.bbox[3] - p.bbox[1]) };
    drawRoi();
    $('#roi-values').textContent = `ROI: x=${S.roi.x} y=${S.roi.y} w=${S.roi.w} h=${S.roi.h}`;
  } catch (e) { toast(e.message, 'err'); }
});

// -------- canvas de ROI
/*
  Desenho da ROI sobre um frame estático.

  setupRoiCanvas é idempotente (cv._ready): loadCalibration pode rodar
  várias vezes e, sem essa trava, cada visita à aba criaria outro
  setInterval de 3s e outro conjunto de listeners. Uma imagem, um timer.

  O setInterval de 3s só busca frame quando a aba de calibração está
  visível. É um throttle por visibilidade: sem ele, o painel gastaria
  requisições de JPEG o turno inteiro para uma imagem que ninguém vê.
  Não há debounce na resposta: cada callback desenha por cima, e o
  flicker é imperceptível a 3s.
*/
function setupRoiCanvas() {
  const cv = $('#roi-canvas');
  if (!cv || cv._ready) return;
  cv._ready = true;
  let img = new Image();
  img.onload = () => { cv._img = img; drawRoi(); };
  img.onerror = () => {};
  img.src = '/api/frame.jpg?t=' + Date.now();
  setInterval(() => {
    if (S.page !== 'calibration') return;
    img.src = '/api/frame.jpg?t=' + Date.now();
  }, 3000);

  /*
    Converte coordenada de tela para coordenada do frame. São três
    correções encadeadas e todas necessárias:
      1. subtrair o offset do elemento (r.left/r.top)
      2. descontar a centralização do "contain" (o frame pode não ocupar a
         largura toda do canvas)
      3. dividir pela escala para voltar à resolução original
    Sem a etapa 2 a ROI sairia deslocada em telas largas, que é o caso mais
    comum (o vídeo é 16:9 dentro de um cartão mais alto).
  */
  const pt = (ev) => {
    const r = cv.getBoundingClientRect();
    const nat = cv._img || { width: cv.width, height: cv.height };
    const scale = Math.min(r.width / nat.width, r.height / nat.height);
    const ox = (r.width - nat.width * scale) / 2, oy = (r.height - nat.height * scale) / 2;
    const x = ((ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left - ox) / scale;
    const y = ((ev.touches ? ev.touches[0].clientY : ev.clientY) - r.top - oy) / scale;
    return { x: Math.max(0, x), y: Math.max(0, y), nat };
  };
  const down = (ev) => { ev.preventDefault(); const p = pt(ev); S.dragging = { x0: p.x, y0: p.y }; };
  const move = (ev) => {
    // Sem throttle neste mousemove, de propósito: cada pixel do arrasto é
    // um redesenho de canvas, que é barato (uma imagem e quatro retângulos)
    // e é o que faz a caixa "grudar" no cursor. O throttle aqui daria
    // sensação de Borrão. Só o estado é minimizado/desenhado, nunca atrasado.
    if (!S.dragging) return;
    ev.preventDefault();
    const p = pt(ev);
    S.roi = {
      x: Math.round(Math.min(S.dragging.x0, p.x)), y: Math.round(Math.min(S.dragging.y0, p.y)),
      w: Math.round(Math.abs(p.x - S.dragging.x0)), h: Math.round(Math.abs(p.y - S.dragging.y0)),
    };
    drawRoi();
  };
  const up = () => {
    if (S.dragging) {
      S.dragging = null;
      $('#roi-values').textContent = S.roi ? `ROI: x=${S.roi.x} y=${S.roi.y} w=${S.roi.w} h=${S.roi.h}` : 'sem ROI';
    }
  };
  // mouseup e touchend ficam na window, não no canvas: soltar o botão
  // FORA do canvas é o gesto mais comum e, se o listener morasse no canvas,
  // o arraste ficaria travado com S.dragging sem fim.
  cv.addEventListener('mousedown', down); cv.addEventListener('mousemove', move);
  window.addEventListener('mouseup', up);
  cv.addEventListener('touchstart', down, { passive: false });
  cv.addEventListener('touchmove', move, { passive: false });
  window.addEventListener('touchend', up);
}

/*
  Pintura da ROI. Escurece o que está FORA da caixa com quatro retângulos
  (em vez de recortar com save/clip), porque o recorte pediria redesenhar a
  área interna a cada mousemove, e apagaria o trecho por onde o usuário
  está arrastando.
  As dimensões do canvas são reatribuídas a partir da imagem: o canvas
  fica na resolução real do frame, e é o CSS que escala a exibição. Assim a
  ROI continua em pixels de frame, correta depois de qualquer resize.
*/
function drawRoi() {
  const cv = $('#roi-canvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const img = cv._img;
  const w = img ? img.width : 1280, h = img ? img.height : 720;
  cv.width = w; cv.height = h;
  if (img) ctx.drawImage(img, 0, 0, w, h); else { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h); }
  // escurece fora da ROI
  if (S.roi && S.roi.w > 0) {
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0, 0, w, S.roi.y);
    ctx.fillRect(0, S.roi.y + S.roi.h, w, h);
    ctx.fillRect(0, S.roi.y, S.roi.x, S.roi.h);
    ctx.fillRect(S.roi.x + S.roi.w, S.roi.y, w - (S.roi.x + S.roi.w), S.roi.h);
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 3;
    ctx.strokeRect(S.roi.x, S.roi.y, S.roi.w, S.roi.h);
    ctx.fillStyle = '#3b82f6'; ctx.font = '16px monospace';
    ctx.fillText(`${S.roi.w} × ${S.roi.h}`, S.roi.x + 4, Math.max(16, S.roi.y - 6));
  }
}

// ============================================================= PAGE: DATASET ===
/*
  Contagem de arquivos do dataset (GET /api/dataset). Nada aqui executa
  modelo: é inventário. O valor para o operador é enxergar o equilíbrio
  entre as três divisões e entre as classes — é isso que decide se o
  próximo treinamento vale a pena.
  A árvore de pastas é texto montado com contagens reais, com "?." para as
  divisões que ainda não existem: divisão sem pasta mostra 0, não erro.
*/
async function loadDataset() {
  try {
    const r = await api('/api/dataset');
    const s = r.stats;
    $('#ds-total').textContent = s.total_images;
    $('#ds-annotated').textContent = s.annotated;
    $('#ds-pending').textContent = s.pending;
    $('#ds-raw').textContent = s.raw_images;
    $('#ds-split-table tbody').innerHTML = Object.entries(s.per_split).map(([k, v]) =>
      `<tr><td>${k}</td><td class="num">${v.images}</td><td class="num">${v.annotated}</td><td class="num">${v.boxes}</td></tr>`).join('');
    $('#ds-class-table tbody').innerHTML = Object.entries(s.per_class || {}).length
      ? Object.entries(s.per_class).map(([k, v]) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join('')
      : '<tr><td colspan="2" class="empty">nenhuma caixa anotada</td></tr>';
    $('#ds-warnings').textContent = (s.warnings || []).join('\n');
    $('#ds-yaml').textContent = r.yaml;
    $('#ds-structure').textContent =
`training/datasets/
├── images/
│   ├── train/   ${s.per_split.train?.images || 0} imagens
│   ├── val/     ${s.per_split.val?.images || 0} imagens
│   └── test/    ${s.per_split.test?.images || 0} imagens
├── labels/
│   ├── train/   ${s.per_split.train?.annotated || 0} rótulos
│   ├── val/     ${s.per_split.val?.annotated || 0} rótulos
│   └── test/    ${s.per_split.test?.annotated || 0} rótulos
└── raw/         ${s.raw_images} imagens aguardando anotação

Classes: ${(s.classes || []).join(', ')}`;
  } catch (e) { toast(e.message, 'err'); }
}

$('#split-form')?.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  const payload = { train: +f.train.value, val: +f.val.value, test: +f.test.value, seed: +f.seed.value };
  try { const r = await api('/api/dataset/split', { method: 'POST', body: JSON.stringify(payload) }); $('#split-result').textContent = r.message; toast(r.message, 'ok'); loadDataset(); }
  catch (e) { toast(e.message, 'err'); }
});

$('#upload-btn')?.addEventListener('click', () => $('#upload-input').click());
/*
  Upload de várias imagens em sequência. Usa fetch cru (e não api()) porque
  o envio é multipart: cabeçalho Content-Type tem que ser definido pelo
  próprio navegador, com a fronteira. api() sempre forçaria JSON.
  Enviar em loop (e não em paralelo) evita estourar memória e banda com
  um lote grande de fotos de câmera. O erro individual é engolido de
  propósito: uma imagem ruim não pode impedir as outras de subirem.
*/
$('#upload-input')?.addEventListener('change', async (ev) => {
  for (const f of ev.target.files) {
    const fd = new FormData(); fd.append('file', f);
    try { await fetch('/api/dataset/upload', { method: 'POST', body: fd }); } catch (e) {}
  }
  toast('Imagens enviadas. Anote na aba Anotação.', 'ok');
  ev.target.value = ''; loadDataset();
});

$('#import-folder')?.addEventListener('click', async () => {
  const path = prompt('Caminho absoluto da pasta com imagens:', '/home/');
  if (!path) return;
  try { const r = await api('/api/dataset/import-dir', { method: 'POST', body: JSON.stringify({ path }) }); toast(r.message, 'ok'); loadDataset(); }
  catch (e) { toast(e.message, 'err'); }
});

// ========================================================== PAGE: ANOTAÇÃO ===
/*
  Pendências de anotação (GET /api/annotation). O front não desenha caixas:
  ele informa o que falta. Três situações e não duas — "sem objetos"
  (tag-info) é diferente de "pendente" (tag-warn), porque uma imagem
  negativa já está corretamente anotada: ela ensina ao modelo que aquela
  parte da cena é vazia.
*/
async function loadAnnotation() {
  try {
    const r = await api('/api/annotation');
    $('#ann-summary').textContent = `${r.total} imagens · ${r.pending} pendentes`;
    $('#ann-table tbody').innerHTML = (r.images || []).length ? r.images.map(i => `
      <tr>
        <td>${i.filename}</td><td>${i.split}</td>
        <td>${i.annotated ? (i.empty ? '<span class="tag tag-info">sem objetos</span>' : '<span class="tag tag-ok">anotada</span>')
                          : '<span class="tag tag-warn">pendente</span>'}</td>
        <td class="num">${i.boxes}</td>
        <td><a class="btn btn-ghost" href="${i.url}" target="_blank">abrir</a></td>
      </tr>`).join('') : '<tr><td colspan="5" class="empty">nenhuma imagem no dataset</td></tr>';
  } catch (e) { toast(e.message, 'err'); }
}

$('#ann-open')?.addEventListener('click', () => $('#ann-help').classList.toggle('hidden'));

// ======================================================= PAGE: TREINAMENTO ===
/*
  Estado do treino (GET /api/training/status). Três estados possíveis e
  mutuamente exclusivos na pill: rodando (âmbar), finalizado (verde),
  erro (vermelho). "Sem job nenhum" é um quarto caso e retorna cedo —
  sem job não há barra nem métricas para pintar.
  As métricas vêm do subprocesso de treino já parseadas pelo backend; aqui
  só se formata. 4 casas decimais porque loss e mAP mudam na terceira.
*/
async function loadTraining() {
  try {
    const r = await api('/api/training/status');
    const g = r.gpu || {};
    $('#gpu-report').textContent = g.text || '';
    if (!r.running && !r.job) { $('#train-state').textContent = 'parado'; $('#train-state').className = 'pill pill-offline'; return; }
    const j = r.job || {};
    const p = j.progress || {};
    $('#train-state').textContent = r.running ? 'treinando' : (j.error ? 'erro' : 'finalizado');
    $('#train-state').className = 'pill ' + (r.running ? 'pill-warn' : j.error ? 'pill-offline' : 'pill-online');
    $('#train-bar').style.width = (p.percent || 0) + '%';
    $('#train-epoch').textContent = `Epoch ${p.epoch || 0}/${p.total || j.epochs_total || '?'}  ·  ${Math.round(j.elapsed_sec || 0)}s`;
    $('#train-metrics tbody').innerHTML = [
      ['Loss (box)', fmt(p.box_loss, 4)], ['Loss (cls)', fmt(p.cls_loss, 4)],
      ['Precision', fmt(p.precision, 4)], ['Recall', fmt(p.recall, 4)],
      ['mAP50', fmt(p.map50, 4)], ['mAP50-95', fmt(p.map5095, 4)],
    ].map(([k, v]) => `<tr><td class="muted">${k}</td><td class="num">${v ?? '—'}</td></tr>`).join('');
  } catch (e) { toast(e.message, 'err'); }
}

$('#train-form')?.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = ev.target;
  const ok = await confirmDialog('Iniciar treinamento',
    `Modelo ${f.model.value}, ${f.epochs.value} épocas, batch ${f.batch.value}, device ${f.device.value || 'auto'}.\n\nO treinamento roda em processo separado e o dashboard continua funcionando.`);
  if (!ok) return;
  try {
    const r = await api('/api/training/start', {
      method: 'POST',
      body: JSON.stringify({
        model: f.model.value, epochs: +f.epochs.value, batch: +f.batch.value,
        image_size: +f.image_size.value, workers: +f.workers.value, device: f.device.value,
      }),
    });
    $('#train-msg').textContent = r.message;
    toast('Treinamento iniciado.', 'ok');
    startTrainingPoll();
  } catch (e) { toast(e.message, 'err'); }
});

$('#train-stop')?.addEventListener('click', async () => {
  const ok = await confirmDialog('Parar treinamento', 'Interrompe o processo de treino atual. Confirma?');
  if (!ok) return;
  try { await api('/api/training/stop', { method: 'POST' }); toast('Interrompido.', 'ok'); }
  catch (e) { toast(e.message, 'err'); }
});

$('#train-refresh-log')?.addEventListener('click', async () => {
  const r = await api('/api/training/log?lines=120');
  $('#train-log').textContent = r.log || 'sem saída ainda.';
});

/*
  Polling do progresso a cada 3s. O treinamento roda em processo separado, e
  o backend não emite WebSocket para isso; 3s é o compromisso entre
  responsividade da barra e peso no servidor durante um treino de horas.
  clearInterval no início garante um timer só, mesmo entrando e saindo da
  aba várias vezes.

  O fim do treino é detectado comparando o texto da pill antes e depois do
  load — truque barato, mas frágil: se alguém mudar esse texto, a detecção
  de conclusão quebra junto.
*/
function startTrainingPoll() {
  clearInterval(S.pollTimer);
  S.pollTimer = setInterval(async () => {
    if (S.page !== 'training' && !S.pollTimer) return;
    const wasRunning = $('#train-state').textContent === 'treinando';
    await loadTraining();
    if (wasRunning && $('#train-state').textContent !== 'treinando') {
      clearInterval(S.pollTimer); S.pollTimer = null;
      toast('Treinamento finalizado. Registre/ative o modelo na aba Modelo.', 'ok');
      const r = await api('/api/training/log?lines=160');
      $('#train-log').textContent = r.log || '';
    }
  }, 3000);
}

// ============================================================== PAGE: MODELO ===
/*
  Versões de modelo (GET /api/models). Três blocos:
    1. modelo ativo, com métricas lidas do results.csv do treino
    2. dispositivo: CUDA, VRAM, classes carregadas
    3. tabela de versões, com o botão ATIVAR em cada linha

  O texto de gpu-info tem uma consequência deliberada: se o modelo carregado
  for genérico pré-treinado, ele avisa que as contagens não são confiáveis.
  Um número bonito vindo de um modelo errado é pior que nenhum número.
  Coluna "sem arquivo" (tag-warn) marca versão cadastrada cujo .pt sumiu do
  disco — ativá-la falharia.
*/
async function loadModels() {
  try {
    const r = await api('/api/models');
    const a = r.active || {};
    $('#active-model tbody').innerHTML = [
      ['Arquivo', a.filename || '—'],
      ['Versão', a.version || '—'],
      ['Treinado em', a.created_at || '—'],
      ['Épocas', a.epochs ?? '—'],
      ['Tamanho do dataset', a.dataset_size ?? '—'],
      ['Precision', a.precision != null ? a.precision : '—'],
      ['Recall', a.recall != null ? a.recall : '—'],
      ['mAP50', a.map50 != null ? a.map50 : '—'],
      ['mAP50-95', a.map5095 != null ? a.map5095 : '—'],
    ].map(([k, v]) => `<tr><td class="muted">${k}</td><td>${v}</td></tr>`).join('');

    const L = r.loaded || {};
    $('#gpu-info').textContent =
`CUDA: ${L.gpu?.cuda_available ? 'disponível' : 'indisponível'}` +
(L.gpu?.gpus || []).map(g => `\nGPU: ${g.name}\nVRAM: ${g.vram_total_gb} GB (usada ${g.vram_used_gb} GB)`).join('') +
`\nDevice em uso: ${L.device}\nModelo carregado: ${L.filename}\nTarefa: ${L.task}\n` +
`Classes: ${Object.entries(L.classes || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '—'}` +
`\n${L.trained_model ? 'Este é um modelo treinado com dados da empresa.' : 'ATENÇÃO: modelo genérico pré-treinado — não conta cadeiras empilhadas com confiabilidade.'}` +
(L.load_error ? `\nErro: ${L.load_error}` : '');

    $('#models-table tbody').innerHTML = (r.models || []).map(m => `
      <tr>
        <td>${m.filename}</td><td>${m.version || '—'}</td><td>${m.created_at || '—'}</td>
        <td class="num">${m.precision ?? '—'}</td><td class="num">${m.recall ?? '—'}</td>
        <td class="num">${m.map50 ?? '—'}</td><td class="num">${m.map5095 ?? '—'}</td>
        <td>${m.active ? '<span class="tag tag-ok">ATIVO</span>' : (m.exists_on_disk ? '' : '<span class="tag tag-warn">sem arquivo</span>')}</td>
        <td>${m.active ? '' : `<button class="btn btn-ghost act" data-id="${m.id}" data-file="${m.filename}">ATIVAR</button>`}</td>
      </tr>`).join('');
    $$('#models-table .act').forEach(b => b.onclick = () => activateModel(b.dataset.id, b.dataset.file));
  } catch (e) { toast(e.message, 'err'); }
}

/*
  Ativar modelo: troca a quente. O aviso no modal existe porque o efeito é
  imediato e silencioso — o próximo frame já sai com o modelo novo, sem
  reiniciar a câmera e sem derrubar o WebSocket. Quem ativa precisa saber
  disso antes de clicar, e precisa saber que o rollback existe.
*/
async function activateModel(id, file) {
  const ok = await confirmDialog('Ativar modelo',
    `O modelo ${file} passará a ser usado na detecção AGORA.\n\nO modelo atual será trocado em produção, sem reiniciar a câmera. É possível voltar ao anterior.`);
  if (!ok) return;
  try { const r = await api('/api/models/activate', { method: 'POST', body: JSON.stringify({ model_id: id ? +id : null, filename: file }) }); toast(r.message, 'ok'); loadModels(); }
  catch (e) { toast(e.message, 'err'); }
}

$('#model-rollback')?.addEventListener('click', async () => {
  try { const r = await api('/api/models/rollback', { method: 'POST' }); toast(r.message, 'ok'); loadModels(); }
  catch (e) { toast(e.message, 'err'); }
});

$('#model-import')?.addEventListener('click', () => $('#model-file').click());
$('#model-file')?.addEventListener('change', async (ev) => {
  const f = ev.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append('file', f);
  const res = await fetch('/api/models/import', { method: 'POST', body: fd });
  const b = await res.json();
  toast(b.message || (b.ok ? 'importado' : 'falhou'), b.ok ? 'ok' : 'err');
  loadModels();
});

// ========================================================== PAGE: SETTINGS ===
/*
  Configuração efetiva (GET /api/config). Exibir a config ajuda a achar erro
  de digitação no .env sem precisar de acesso ao servidor — a URL RTSP vem
  mascarada, então mostrar isso é seguro.
  Aqui innerHTML é usado para as linhas da tabela porque os campos são
  identificadores e flags, não texto livre; o JSON cru vai por textContent
  para preservar a formatação e nunca ser interpretado como marcação.
*/
async function loadSettings() {
  try {
    const cfg = await api('/api/config');
    $('#cfg-out').textContent = JSON.stringify(cfg, null, 2);
    const r = await api('/api/cameras');
    $('#cams-table tbody').innerHTML = r.cameras.map(c => `
      <tr><td>${c.id}</td><td>${c.name}</td><td class="mono small">${c.rtsp_url}</td>
          <td>${c.enabled ? 'sim' : 'não'}</td><td>${c.is_default ? 'sim' : ''}</td>
          <td><button class="btn btn-ghost del" data-id="${c.id}">remover</button></td></tr>`).join('') ||
      '<tr><td colspan="6" class="empty">nenhuma câmera cadastrada</td></tr>';
    $$('#cams-table .del').forEach(b => b.onclick = async () => {
      const ok = await confirmDialog('Remover câmera', `Remover a câmera ${b.dataset.id}? O histórico dela também será apagado.`);
      if (!ok) return;
      try { await api('/api/cameras/' + b.dataset.id, { method: 'DELETE' }); loadSettings(); } catch (e) { toast(e.message, 'err'); }
    });
  } catch (e) { toast(e.message, 'err'); }
}

$('#cfg-refresh')?.addEventListener('click', loadSettings);

// ================================================================ PAGE: LOGS ===
/*
  Duas fontes: arquivo de log (texto puro, mantido como está para não
  distorcer alinhamento) e eventos do banco (tabela). Duas requisições
  sequenciais: pouco dado, e o arquivo é o que o operador quer ver primeiro.
  O nome do arquivo vem de um <select> com valores fixos, então o front
  nunca manda um caminho arbitrário para o backend ler.
*/
async function loadLogs() {
  try {
    const f = $('#log-file').value;
    const r = await api(`/api/logs?file=${f}&lines=200`);
    $('#log-out').textContent = r.log || r.message || 'vazio';
    const ev = await api('/api/events?limit=100');
    $('#events-table tbody').innerHTML = ev.events.map(e => `
      <tr><td>${e.timestamp.replace('T', ' ')}</td>
          <td><span class="tag ${e.level === 'ERROR' ? 'tag-low' : e.level === 'WARNING' ? 'tag-warn' : 'tag-info'}">${e.level}</span></td>
          <td>${e.event}</td><td>${e.message}</td></tr>`).join('') ||
      '<tr><td colspan="4" class="empty">sem eventos</td></tr>';
  } catch (e) { toast(e.message, 'err'); }
}

$('#log-refresh')?.addEventListener('click', loadLogs);
$('#log-file')?.addEventListener('change', loadLogs);
$('#events-clear')?.addEventListener('click', async () => {
  const ok = await confirmDialog('Limpar eventos', 'Apagar todos os eventos registrados no banco?');
  if (!ok) return;
  await api('/api/events', { method: 'DELETE' }); loadLogs(); toast('Eventos limpos.', 'ok');
});

/*
  Eventos chegando pelo WebSocket (msg.type === "event").
  Guarda dupla: só insere se a aba Logs estiver visível. Sem isso, o painel
  pagaria uma reconstrução de tabela para cada evento mesmo com a aba
  escondida. A linha entra no topo (prepend) porque o evento mais recente é
  o que interessa.
*/
function addEventRow(e) {
  const tb = $('#events-table tbody');
  if (!tb || S.page !== 'logs') return;
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${e.timestamp.replace('T', ' ')}</td>
    <td><span class="tag ${e.level === 'ERROR' ? 'tag-low' : e.level === 'WARNING' ? 'tag-warn' : 'tag-info'}">${e.level}</span></td>
    <td>${e.event}</td><td>${e.message}</td>`;
  tb.prepend(tr);
}

// ================================================================ ROTEAMENTO ===
/*
  SPA à mão, com a hash como fonte da verdade. Duas tabelas:
    TITLES  -> nome da aba no cabeçalho
    LOADERS -> o que buscar quando a aba abre
  Dashboard tem loader vazio de propósito: nada é buscado, tudo chega pelo
  WebSocket. Treinamento tem loader composto porque precisa do load E do
  polling.

  TITLES também serve de lista branca: hash desconhecida cai no dashboard em
  vez de mostrar uma seção vazia. Um erro de digitação na URL não pode
  deixar a tela em branco.
*/
const TITLES = {
  dashboard: 'Dashboard', camera: 'Câmera', counting: 'Contagem', history: 'Histórico',
  calibration: 'Calibração', dataset: 'Dataset', annotation: 'Anotação',
  training: 'Treinamento', model: 'Modelo', settings: 'Configurações', logs: 'Logs',
};
const LOADERS = {
  dashboard: () => {}, camera: loadCamera, counting: loadCounting, history: loadHistory,
  calibration: loadCalibration, dataset: loadDataset, annotation: loadAnnotation,
  training: () => { loadTraining(); startTrainingPoll(); }, model: loadModels,
  settings: loadSettings, logs: loadLogs,
};

/*
  Troca de aba. Tudo que muda aqui é visibilidade e dados: as seções já
  estavam no DOM, então o custo é pequeno. Note o que NÃO é redesenhado:
  o vídeo continua tocando e a lista de pilhas do Dashboard continua
  recebendo o WebSocket com a aba escondida. Interromper isso exigiria
  pausar o stream, e é uma decisão de arquitetura, não de detalhe.

  O try/catch no loader evita que uma falha de rede ao abrir uma aba
  derrube o roteamento inteiro (e deixe a página em branco sem aviso).
*/
function route() {
  const page = (location.hash || '#dashboard').slice(1) || 'dashboard';
  S.page = TITLES[page] ? page : 'dashboard';
  $$('.page').forEach(p => p.classList.add('hidden'));
  $('#page-' + S.page)?.classList.remove('hidden');
  $$('.menu a').forEach(a => a.classList.toggle('active', a.dataset.page === S.page));
  $('#page-title').textContent = TITLES[S.page];
  try { LOADERS[S.page]?.(); } catch (e) { console.error(e); }
}

window.addEventListener('hashchange', route);

// ================================================================ INÍCIO ===
// Relógio local. Independente do WS de propósito: mesmo com a câmera e o
// backend fora, o operador precisa de uma referência de hora para saber se
// o painel está congelado.
function tickClock() {
  $('#clock').textContent = new Date().toLocaleString('pt-BR');
}

/*
  Ordem de boot:
    route() primeiro, para a aba da URL já aparecer antes de qualquer rede
    connectWS() em seguida, para o estado começar a chegar o quanto antes
    Chart.js por último, porque é a única dependência externa e pode
      demorar ou nunca carregar sem quebrar o resto
*/
function boot() {
  route();
  connectWS();
  tickClock(); setInterval(tickClock, 1000);
  loadChartJs(initCharts);

  // botão de menu no mobile. Nasce por JavaScript em vez de estar no HTML
  // para não existir em telas largas, onde ficaria invisível por CSS.
  const toggle = document.createElement('button');
  toggle.className = 'menu-toggle';
  toggle.textContent = '☰';
  toggle.onclick = () => $('#sidebar').classList.toggle('open');
  $('.topbar').appendChild(toggle);

  // badges de estado inicial
  // Uma única chamada de status no boot, só para avisar sobre modelo
  // genérico. É redundante em propósito: o WebSocket ainda não mandou nada
  // e o operador precisa saber na hora se o número dele é confiável.
  api('/api/status').then(r => {
    if (r.model && !r.model.loaded) toast('Modelo específico ainda não treinado.', 'err');
  }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', boot);
