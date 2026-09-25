/* =========================================================================
   CONTADOR DE CADEIRAS — dashboard (sem frameworks)
   - WebSocket /ws  -> estado em tempo real
   - fetch /api/*   -> ações e páginas
   ========================================================================= */

'use strict';

// ------------------------------------------------------------------ estado
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

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const fmt = (v, d = 0) => (v === null || v === undefined || isNaN(v)) ? '—' : Number(v).toFixed(d);
const pct = v => Math.round((v || 0) * 100);

// ------------------------------------------------------------------ toast
let toastTimer = null;
function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast ' + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 4200);
}

// ------------------------------------------------------------------ modal
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
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  let body = {};
  try { body = await res.json(); } catch (e) { body = { ok: false, message: res.statusText }; }
  if (!res.ok) throw new Error(body.message || ('HTTP ' + res.status));
  return body;
}

// ==================================================================== WS ===
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

function applyState(st) {
  S.state = st;
  const piles = st.piles || [];
  // Só redesenha as listas se a pilha mudou (evita piscar)
  const sig = piles.map(p => `${p.pile_id}:${p.count}:${p.status}`).join('|');
  if (sig !== applyState._sig) { applyState._sig = sig; renderPiles(piles); renderCountTable(piles); }
  updateTotals(st);
  updateHeader(st);
  updateAlerts(st.alerts || []);
  if (S.page === 'dashboard') pushChartPoint(st);
  $('#video-empty')?.classList.add('hidden');
  $('#video2-empty')?.classList.add('hidden');
}

function updateHeader(st) {
  const cam = (st.camera_state || 'OFFLINE').toUpperCase();
  const pill = $('#cam-state-pill');
  pill.textContent = cam;
  pill.className = 'pill ' + (cam === 'ONLINE' ? 'pill-online' : cam === 'CONNECTING' || cam === 'RECONNECTING' ? 'pill-warn' : 'pill-offline');

  $('#stat-camera').textContent = cam === 'ONLINE' ? 'ONLINE' : cam;
  $('#stat-piles').textContent = st.pile_count ?? 0;
  $('#stat-conf').textContent = (st.confidence > 0 ? pct(st.confidence) + '%' : '—');
  $('#meta-fps').textContent = 'FPS ' + fmt(st.fps, 1);
  $('#meta-res').textContent = st.mode ? 'modo: ' + st.mode : '';

  const badge = $('#mode-badge');
  const mode = st.mode || 'iniciando';
  badge.textContent = mode;
  badge.className = 'badge ' + (mode.startsWith('production') ? 'badge-ok' : mode === 'preparation' ? 'badge-err' : 'badge-warn');
}

function updateTotals(st) {
  const t = st.total ?? 0;
  const el = $('#total-value');
  if (el.textContent !== String(t)) {
    el.textContent = t;
    el.style.color = '#fff';
    el.animate([{ transform: 'scale(1.10)', color: '#3ec93e' }, { transform: 'scale(1)', color: '#fff' }], { duration: 420 });
  }
  const tt = st.totals || {};
  $('#total-tentative').textContent = tt.total_tentative ? `(+${tt.total_tentative} instável)` : 'todas estáveis';
  $('#total-piles').textContent = `${st.pile_count || 0} pilha(s)`;
}

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
const STATUS_TAG = { STABLE: 'tag-ok', UNSTABLE: 'tag-warn', LOW_CONFIDENCE: 'tag-low', PARTIAL: 'tag-part', UNKNOWN: 'tag-unk' };

function renderPiles(piles) {
  const box = $('#piles-list');
  if (!piles.length) { box.innerHTML = '<div class="empty">Nenhuma pilha detectada.</div>'; return; }
  box.innerHTML = '';
  piles.forEach(p => {
    const row = document.createElement('div');
    row.className = 'pile s-' + p.status;
    const est = Object.entries(p.candidates || {}).map(([k, v]) => `${k}=${Math.round(v)}`).join(' ');
    row.innerHTML = `
      <div class="pile-id">PILHA ${p.pile_id}</div>
      <div class="pile-count">${p.count}</div>
      <div class="pile-info">
        <div class="pile-status">${statusLabel(p.status)}</div>
        <div class="pile-detail">${est || 'sem estimadores'}${p.manual ? ' · manual' : ''}</div>
      </div>
      <div class="pile-conf">${pct(p.confidence)}%</div>
      <div class="stepper">
        <button data-d="-1">−</button>
        <input type="number" value="${p.count}" data-id="${p.pile_id}">
        <button data-d="1">+</button>
      </div>`;
    box.appendChild(row);
  });
  box.querySelectorAll('.stepper button').forEach(b => {
    b.onclick = () => {
      const input = b.parentElement.querySelector('input');
      input.value = Math.max(0, (parseInt(input.value, 10) || 0) + parseInt(b.dataset.d, 10));
    };
  });
  box.querySelectorAll('.stepper input').forEach(inp => {
    inp.onchange = () => saveCorrection(inp.dataset.id, parseInt(inp.value, 10));
  });
}

function statusLabel(s) {
  return { STABLE: 'ESTÁVEL', UNSTABLE: 'INSTÁVEL', LOW_CONFIDENCE: 'BAIXA CONFIANÇA', PARTIAL: 'PARCIAL', UNKNOWN: 'INDETERMINADO' }[s] || s;
}

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

$('#chart-range')?.addEventListener('change', () => { /* próximo ponto já respeita o range */ });

// ============================================================ PAGE: CÂMERA ===
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

$('#toggle-overlay')?.addEventListener('click', (e) => {
  S.overlay = !S.overlay;
  e.target.textContent = S.overlay ? 'Ver imagem original' : 'Ver com contagem';
  const src = S.overlay ? '/api/stream.mjpg' : '/api/frame.raw.jpg';
  $('#video2').src = '';
  $('#video2').src = src + (S.overlay ? '' : '?t=' + Date.now());
});

// ========================================================= PAGE: CONTAGEM ===
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
  const custom = e.target.value === 'custom';
  $('#hist-from').classList.toggle('hidden', !custom);
  $('#hist-to').classList.toggle('hidden', !custom);
  if (custom) return;
  loadHistory();
});

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

$('#measure-chair')?.addEventListener('click', async () => {
  if (!S.roi) { toast('Desenhe a caixa de UMA cadeira primeiro.', 'err'); return; }
  try {
    const r = await api('/api/calibration/measure', { method: 'POST', body: JSON.stringify({ roi: S.roi }) });
    $('#calib-form').elements.chair_height_px.value = r.measured_height_px;
    toast(r.message, 'ok');
  } catch (e) { toast(e.message, 'err'); }
});

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
  cv.addEventListener('mousedown', down); cv.addEventListener('mousemove', move);
  window.addEventListener('mouseup', up);
  cv.addEventListener('touchstart', down, { passive: false });
  cv.addEventListener('touchmove', move, { passive: false });
  window.addEventListener('touchend', up);
}

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
function tickClock() {
  $('#clock').textContent = new Date().toLocaleString('pt-BR');
}

function boot() {
  route();
  connectWS();
  tickClock(); setInterval(tickClock, 1000);
  loadChartJs(initCharts);

  // botão de menu no mobile
  const toggle = document.createElement('button');
  toggle.className = 'menu-toggle';
  toggle.textContent = '☰';
  toggle.onclick = () => $('#sidebar').classList.toggle('open');
  $('.topbar').appendChild(toggle);

  // badges de estado inicial
  api('/api/status').then(r => {
    if (r.model && !r.model.loaded) toast('Modelo específico ainda não treinado.', 'err');
  }).catch(() => {});
}

document.addEventListener('DOMContentLoaded', boot);
