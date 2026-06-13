/* SignalTools - UI Frontend */

let currentSignal = null;
let currentSampleRate = 2000;

// ====== Tabs ======
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
  });
});

// ====== Demo signal generator ======
document.getElementById('btn-generate-demo').addEventListener('click', async () => {
  const length = +document.getElementById('demo-length').value;
  const freq1 = +document.getElementById('demo-freq1').value;
  const amp1 = +document.getElementById('demo-amp1').value;
  const freq2 = +document.getElementById('demo-freq2').value;
  const amp2 = +document.getElementById('demo-amp2').value;
  const noise = +document.getElementById('demo-noise').value;

  const t = Array.from({length}, (_, i) => i / currentSampleRate);
  const sig = t.map(x =>
    amp1 * Math.sin(2 * Math.PI * freq1 * x) +
    amp2 * Math.sin(2 * Math.PI * freq2 * x) +
    noise * (Math.random() - 0.5) * 2
  );
  setSignal(sig);
  document.getElementById('signal-input').value = sig.map(v => v.toFixed(4)).join(', ');
  document.getElementById('signal-status').textContent = `Señal demo generada (${sig.length} samples)`;
  document.getElementById('signal-status').className = 'status ready';
});

// ====== File upload ======
document.getElementById('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (file.name.endsWith('.json')) {
    const text = await file.text();
    let data;
    try { data = JSON.parse(text); } catch { setStatus('Error al parsear JSON', true); return; }
    const sig = Array.isArray(data) ? data : (data.signal || data.samples || data.values);
    if (!sig) { setStatus('JSON no contiene señal válida', true); return; }
    setSignal(sig);
    document.getElementById('signal-input').value = sig.join(', ');
    setStatus(`Cargado ${sig.length} samples desde JSON`, false);
  } else if (file.name.endsWith('.wav')) {
    setStatus('Procesando WAV...', false);
    const arrayBuf = await file.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await ctx.decodeAudioData(arrayBuf);
    const ch = audioBuf.getChannelData(0);
    const sig = Array.from(ch);
    currentSampleRate = audioBuf.sampleRate;
    document.getElementById('sample-rate').value = audioBuf.sampleRate;
    setSignal(sig);
    document.getElementById('signal-input').value = sig.map(v => v.toFixed(6)).join(', ');
    setStatus(`Cargado ${sig.length} samples desde WAV (${audioBuf.sampleRate} Hz)`, false);
  } else {
    const text = await file.text();
    const nums = text.split(/[\s,\n]+/).map(Number).filter(n => !isNaN(n));
    if (nums.length < 10) { setStatus('Muy pocos valores numéricos', true); return; }
    setSignal(nums);
    document.getElementById('signal-input').value = nums.join(', ');
    setStatus(`Cargado ${nums.length} valores desde archivo de texto`, false);
  }
});

// ====== Signal management ======
function setSignal(sig) {
  currentSignal = sig;
  document.getElementById('btn-analyze').disabled = false;
  document.getElementById('preview-section').classList.remove('hidden');
  document.getElementById('signal-info').textContent =
    `${sig.length} samples | min: ${Math.min(...sig).toFixed(4)} | max: ${Math.max(...sig).toFixed(4)}`;
  requestAnimationFrame(() => drawSignal(sig));
}

function setStatus(msg, isError) {
  const el = document.getElementById('signal-status');
  el.textContent = msg;
  el.className = 'status' + (isError ? ' error' : ' ready');
}

// ====== Manual input ======
document.getElementById('signal-input').addEventListener('input', () => {
  const val = document.getElementById('signal-input').value.trim();
  if (!val) { document.getElementById('btn-analyze').disabled = true; return; }
  const nums = val.split(/[\s,\n]+/).map(Number).filter(n => !isNaN(n));
  if (nums.length >= 8) {
    setSignal(nums);
    setStatus(`${nums.length} valores parseados`, false);
  } else {
    document.getElementById('btn-analyze').disabled = true;
    setStatus('Ingrese al menos 8 valores numéricos', false);
  }
});

// ====== Draw signal ======
function drawSignal(sig) {
  const canvas = document.getElementById('signal-canvas');
  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 200 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = 200;
  ctx.clearRect(0, 0, w, h);

  const maxVal = Math.max(...sig.map(Math.abs), 1e-6);
  const step = Math.max(1, Math.floor(sig.length / w));
  const data = [];
  for (let i = 0; i < w && i * step < sig.length; i++) data.push(sig[i * step]);

  const mid = h / 2;
  ctx.strokeStyle = '#6c8cff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  data.forEach((v, i) => {
    const y = mid - (v / maxVal) * (h * 0.4);
    i === 0 ? ctx.moveTo(i, y) : ctx.lineTo(i, y);
  });
  ctx.stroke();

  ctx.strokeStyle = 'rgba(108, 140, 255, 0.15)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  ctx.setLineDash([]);
}

// ====== Canvas draw helpers ======
function drawLine(canvasId, data, color = '#6c8cff') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = +canvas.clientHeight * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  if (!data || data.length < 2) return;
  const maxVal = Math.max(...data.map(Math.abs), 1e-10);
  const drawStep = Math.max(1, Math.floor(data.length / w));

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let first = true;
  for (let i = 0; i < w && i * drawStep < data.length; i++) {
    const idx = Math.min(i * drawStep, data.length - 1);
    const y = h / 2 - (data[idx] / maxVal) * (h * 0.4);
    if (first) { ctx.moveTo(i, y); first = false; } else { ctx.lineTo(i, y); }
  }
  ctx.stroke();

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  ctx.setLineDash([]);
}

function drawPSD(canvasId, freqs, values) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 200 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = 200;
  ctx.clearRect(0, 0, w, h);

  if (!freqs || !values || freqs.length < 2) return;
  const maxVal = Math.max(...values, 1e-10);
  const pad = 10;

  ctx.fillStyle = 'rgba(108, 140, 255, 0.15)';
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < freqs.length; i++) {
    const x = pad + (freqs[i] / freqs[freqs.length - 1]) * (w - pad * 2);
    const y = h - pad - (values[i] / maxVal) * (h - pad * 2);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = '#6c8cff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = pad + (freqs[i] / freqs[freqs.length - 1]) * (w - pad * 2);
    const y = h - pad - (values[i] / maxVal) * (h - pad * 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '11px sans-serif';
  ctx.fillText(`${freqs[0].toFixed(0)} Hz`, pad + 4, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText(`${freqs[freqs.length - 1].toFixed(0)} Hz`, w - pad - 4, h - 4);
  ctx.textAlign = 'left';
}

function drawSpectrogram(canvasId, spec) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 250 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = 250;
  ctx.clearRect(0, 0, w, h);

  if (!spec || spec.length < 2 || !spec[0] || spec[0].length < 2) return;

  const rows = spec.length;
  const cols = spec[0].length;
  let minV = Infinity, maxV = -Infinity;
  for (const row of spec) for (const v of row) { if (v < minV) minV = v; if (v > maxV) maxV = v; }
  const range = maxV - minV || 1;
  const colW = w / cols;
  const rowH = h / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const norm = (spec[r][c] - minV) / range;
      const val = Math.max(0, Math.min(1, norm));
      const hue = 240 - val * 200;
      ctx.fillStyle = `hsl(${hue}, 80%, ${20 + val * 40}%)`;
      ctx.fillRect(c * colW, r * rowH, Math.ceil(colW), Math.ceil(rowH) + 1);
    }
  }
}

function drawBars(canvasId, data, color = '#6c8cff') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 180 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!data || data.length < 2) return;
  const show = Math.min(data.length, 64);
  const maxVal = Math.max(...data.slice(0, show).map(Math.abs), 1e-10);
  const barW = (w - 20) / show;
  const pad = 10;

  for (let i = 0; i < show; i++) {
    const val = data[i] / maxVal;
    const barH = Math.abs(val) * (h - pad * 2);
    const x = pad + i * barW;
    const y = val >= 0 ? h - pad - barH : h / 2;
    ctx.fillStyle = i === 0 ? '#4ade80' : color;
    ctx.fillRect(x, y, Math.max(barW - 1, 1), Math.max(barH, 1));
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  ctx.setLineDash([]);
}

// ====== Metrics display ======
function renderMetrics(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';
  for (const [key, val] of Object.entries(data)) {
    const div = document.createElement('div');
    div.className = 'metric-item';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = key.replace(/_/g, ' ');
    const value = document.createElement('span');
    value.className = 'value';
    value.textContent = typeof val === 'number' ? (Math.abs(val) > 1000 ? val.toFixed(2) : val.toFixed(6)) : val;
    div.appendChild(label);
    div.appendChild(value);
    container.appendChild(div);
  }
}

// ====== Interpretation ======
function generateInterpretation(metrics, features, spectral) {
  const parts = [];

  // Tonal vs noise
  if (metrics && metrics.snr_db != null) {
    const snr = metrics.snr_db;
    if (snr > 30) parts.push(`<p><span class="tag tag-low">Tonal</span> SNR de <strong>${snr.toFixed(1)} dB</strong> indica una señal dominantemente tonal, con poco ruido de fondo. Apropiada para análisis de frecuencia preciso.</p>`);
    else if (snr > 15) parts.push(`<p><span class="tag tag-mid">Mixta</span> SNR de <strong>${snr.toFixed(1)} dB</strong> sugiere una mezcla de componentes tonales y ruido. La componente espectral aún es distinguible.</p>`);
    else parts.push(`<p><span class="tag tag-high">Ruidoso</span> SNR de <strong>${snr.toFixed(1)} dB</strong>. La señal tiene alta componente de ruido; el contenido espectral puede estar enmascarado.</p>`);

    const sf = spectral && spectral.flatness != null ? spectral.flatness : 0;
    if (sf < 0.3) parts.push(`<p><span class="tag tag-low">Tonal</span> La planitud espectral de <strong>${sf.toFixed(4)}</strong> confirma un espectro concentrado en pocas frecuencias (tonal).</p>`);
    else if (sf > 0.7) parts.push(`<p><span class="tag tag-high">Ruido</span> Alta planitud espectral (<strong>${sf.toFixed(4)}</strong>). El espectro es plano, similar a ruido blanco.</p>`);
    else parts.push(`<p><span class="tag tag-mid">Mixto</span> Planitud espectral de <strong>${sf.toFixed(4)}</strong>. Contenido armónico con algo de ruido de fondo.</p>`);
  }

  // Frequency content
  if (spectral) {
    const df = spectral.dominant_freq_hz;
    const bw = spectral.bandwidth_hz;
    if (df != null && bw != null) {
      parts.push(`<p><span class="tag tag-low">Frecuencia</span> Frecuencia dominante: <strong>${df} Hz</strong>. Ancho de banda espectral: <strong>${bw.toFixed(2)} Hz</strong>.`);
      if (bw / (df || 1) < 0.3) parts.push(`<p>El ancho de banda es angosto relativo a la frecuencia central, indicando una señal <strong>narrowband</strong>.</p>`);
      else parts.push(`<p>El ancho de banda es significativo, sugiriendo una señal <strong>broadband</strong> o con múltiples armónicos.</p>`);
    }
    if (spectral.centroid_hz != null) {
      parts.push(`<p>Centroide espectral: <strong>${spectral.centroid_hz.toFixed(1)} Hz</strong>. Describe el "brightness" percibido de la señal.</p>`);
    }
    if (spectral.pitch_hz != null && spectral.pitch_hz > 0) {
      parts.push(`<p>Frecuencia fundamental estimada (<em>pitch</em>): <strong>${spectral.pitch_hz.toFixed(1)} Hz</strong>.</p>`);
    }
    if (spectral.spectral_entropy != null) {
      const e = spectral.spectral_entropy;
      if (e < 1) parts.push(`<p><span class="tag tag-low">Baja entropía</span> Entropía espectral de <strong>${e.toFixed(3)}</strong>. Espectro concentrado (señal tonal/determinista).</p>`);
      else if (e > 3) parts.push(`<p><span class="tag tag-high">Alta entropía</span> Entropía espectral de <strong>${e.toFixed(3)}</strong>. Espectro disperso (señal estocástica/ruidosa).</p>`);
      else parts.push(`<p>Entropía espectral de <strong>${e.toFixed(3)}</strong>. Balance entre estructura tonal y ruido.</p>`);
    }
  }

  // Temporal metrics (from /api/features)
  if (features) {
    if (features.crest_factor != null) {
      const cf = features.crest_factor;
      if (cf > 3) parts.push(`<p><span class="tag tag-high">Picos</span> Factor de cresta de <strong>${cf.toFixed(2)}</strong>. La señal tiene picos pronunciados (transitorios o impulsos).</p>`);
      else parts.push(`<p>Factor de cresta de <strong>${cf.toFixed(2)}</strong>. Rango dinámico moderado, sin picos extremos.</p>`);
    }
    if (features.zero_crossing_rate != null) {
      const zcr = features.zero_crossing_rate;
      parts.push(`<p>Tasa de cruce por cero: <strong>${(zcr * 100).toFixed(1)}%</strong>. ${zcr > 0.1 ? 'Alta variación (contenido de alta frecuencia o ruido)' : 'Baja variación (señal de baja frecuencia o tonal)'}.</p>`);
    }
  }

  // PSD description
  if (spectral && spectral.psd_freqs && spectral.psd_values && spectral.psd_values.length > 0) {
    const peakIdx = spectral.psd_values.indexOf(Math.max(...spectral.psd_values));
    const peakFreq = spectral.psd_freqs[peakIdx] || 0;
    parts.push(`<p>El pico del PSD se encuentra en <strong>${peakFreq.toFixed(1)} Hz</strong>, consistente con la frecuencia dominante.</p>`);
  }

  if (parts.length === 0) parts.push('<p>No hay suficientes datos para generar interpretación.</p>');

  document.getElementById('interpretation').innerHTML = parts.join('\n');
}

// ====== Spectrum (magnitude with markers) ======
function drawSpectrum(canvasId, freqs, mag, peaks, centroidHz, rolloffHz) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 180 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!freqs || !mag || freqs.length < 2) return;

  const maxFreq = freqs[freqs.length - 1];
  const maxMag = Math.max(...mag, 1e-10);
  const pad = 10;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  // Fill under curve
  ctx.fillStyle = 'rgba(108, 140, 255, 0.08)';
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  for (let i = 0; i < freqs.length; i++) {
    const x = pad + (freqs[i] / maxFreq) * plotW;
    const y = h - pad - (mag[i] / maxMag) * plotH;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(pad + plotW, h - pad);
  ctx.closePath();
  ctx.fill();

  // Magnitude curve
  ctx.strokeStyle = '#6c8cff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = pad + (freqs[i] / maxFreq) * plotW;
    const y = h - pad - (mag[i] / maxMag) * plotH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Centroid line
  if (centroidHz) {
    const cx = pad + (centroidHz / maxFreq) * plotW;
    ctx.strokeStyle = '#f87171';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(cx, pad); ctx.lineTo(cx, h - pad); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f87171';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('C', cx, pad + 10);
  }

  // Rolloff line
  if (rolloffHz) {
    const rx = pad + (rolloffHz / maxFreq) * plotW;
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(rx, pad); ctx.lineTo(rx, h - pad); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#fbbf24';
    ctx.font = '10px sans-serif';
    ctx.fillText('R', rx, h - pad - 4);
  }

  // Peak markers
  if (peaks) {
    for (const p of peaks) {
      const x = pad + (p.freq / maxFreq) * plotW;
      const y = h - pad - (p.mag / maxMag) * plotH;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#4ade80';
      ctx.fill();
      ctx.strokeStyle = '#1a1c26';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('0 Hz', pad, h - 2);
  ctx.textAlign = 'right';
  ctx.fillText(`${maxFreq.toFixed(0)} Hz`, pad + plotW, h - 2);
}

// ====== Phase spectrum ======
function drawPhase(canvasId, freqs, phase) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 180 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!freqs || !phase || freqs.length < 2) return;

  const maxFreq = freqs[freqs.length - 1];
  const pad = 10;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;
  const midY = h / 2;

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(pad, midY); ctx.lineTo(pad + plotW, midY); ctx.stroke();
  ctx.setLineDash([]);

  // Phase curve
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = pad + (freqs[i] / maxFreq) * plotW;
    const y = midY - (phase[i] / Math.PI) * (plotH * 0.45);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('0 Hz', pad, h - 2);
  ctx.textAlign = 'right';
  ctx.fillText(`${maxFreq.toFixed(0)} Hz`, pad + plotW, h - 2);
  ctx.textAlign = 'left';
  ctx.fillText('π', pad + 2, pad + 8);
  ctx.fillText('-π', pad + 2, h - pad - 2);
}

// ====== Peaks table ======
function renderPeaks(containerId, peaks) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!peaks || peaks.length === 0) {
    container.innerHTML = '<div class="peak-row" style="color:var(--text-dim)">Sin picos detectados</div>';
    return;
  }
  container.innerHTML = peaks.map((p, i) => `
    <div class="peak-row">
      <span class="rank">#${i + 1}</span>
      <span class="freq">${p.freq} Hz</span>
      <span class="mag">${p.mag.toExponential(2)}</span>
    </div>
  `).join('');
}

// ====== Waveform temporal ======
function drawWaveform(canvasId, sig, sr) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 200 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.clientWidth, h = 200;
  ctx.clearRect(0, 0, w, h);

  if (!sig || sig.length < 2) return;

  const maxVal = Math.max(...sig.map(Math.abs), 1e-8);
  const mid = h / 2;
  const pad = { t: 8, b: 18, l: 10, r: 10 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const center = pad.t + plotH / 2;

  // Time axis
  const duration = sig.length / sr;
  const xScale = plotW / sig.length;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  const nGrid = 8;
  for (let i = 1; i < nGrid; i++) {
    const x = pad.l + (i / nGrid) * plotW;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
  }

  // Waveform fill
  ctx.fillStyle = 'rgba(108, 140, 255, 0.06)';
  ctx.beginPath();
  ctx.moveTo(pad.l, center);
  for (let i = 0; i < sig.length; i++) {
    const x = pad.l + i * xScale;
    const y = center - (sig[i] / maxVal) * (plotH * 0.45);
    ctx.lineTo(x, y);
  }
  for (let i = sig.length - 1; i >= 0; i--) {
    const x = pad.l + i * xScale;
    const y = center;
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();

  // Waveform line
  ctx.strokeStyle = '#6c8cff';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < sig.length; i++) {
    const x = pad.l + i * xScale;
    const y = center - (sig[i] / maxVal) * (plotH * 0.45);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(pad.l, center); ctx.lineTo(w - pad.r, center); ctx.stroke();
  ctx.setLineDash([]);

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('0', pad.l + 2, center - 4);
  ctx.fillText(`${maxVal.toFixed(2)}`, pad.l + 2, pad.t + 10);
  ctx.fillText(`${(-maxVal).toFixed(2)}`, pad.l + 2, h - pad.b - 2);
  ctx.textAlign = 'right';
  ctx.fillText(`${duration.toFixed(2)} s`, w - pad.r, h - pad.b - 2);
  ctx.textAlign = 'left';
  ctx.fillText('0 s', pad.l, h - 2);
}

// ====== Radar de métricas ======
function drawRadar(canvasId, metrics) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 250 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 250;
  ctx.clearRect(0, 0, w, h);

  const keys = Object.keys(metrics);
  if (keys.length < 2) return;

  // Normalize to 0-1
  const values = keys.map(k => {
    const v = metrics[k];
    const val = v.value != null ? v.value : 0;
    const mn = v.min != null ? v.min : 0;
    const mx = v.max != null ? v.max : Math.abs(val) * 2 || 1;
    return Math.max(0, Math.min(1, (val - mn) / (mx - mn)));
  });

  const n = keys.length;
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(cx, cy) - 30;
  const angleStep = (Math.PI * 2) / n;

  // Grid rings
  for (let ring = 1; ring <= 5; ring++) {
    const r = (radius / 5) * ring;
    ctx.strokeStyle = ring === 5 ? 'rgba(108, 140, 255, 0.2)' : 'rgba(255,255,255,0.05)';
    ctx.lineWidth = ring === 5 ? 1 : 0.5;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const angle = -Math.PI / 2 + i * angleStep;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Axis lines
  for (let i = 0; i < n; i++) {
    const angle = -Math.PI / 2 + i * angleStep;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
  }

  // Data polygon
  ctx.fillStyle = 'rgba(108, 140, 255, 0.12)';
  ctx.strokeStyle = '#6c8cff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const angle = -Math.PI / 2 + i * angleStep;
    const r = radius * values[i];
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Data points
  for (let i = 0; i < n; i++) {
    const angle = -Math.PI / 2 + i * angleStep;
    const r = radius * values[i];
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#6c8cff';
    ctx.fill();
    ctx.strokeStyle = '#1a1c26';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    const angle = -Math.PI / 2 + i * angleStep;
    const labelR = radius + 16;
    const x = cx + labelR * Math.cos(angle);
    const y = cy + labelR * Math.sin(angle);
    const a = ((angle + Math.PI / 2) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
    ctx.textAlign = a > Math.PI * 0.5 && a < Math.PI * 1.5 ? 'right' : 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(keys[i], x + (a > Math.PI * 0.5 && a < Math.PI * 1.5 ? -4 : 4), y);
  }

  // Legend values
  const legend = document.getElementById('radar-legend');
  if (legend) {
    legend.innerHTML = keys.map((k, i) => {
      const v = metrics[k];
      const raw = v.value != null ? v.value.toFixed(2) : '—';
      return `<span><span class="legend-dot" style="background:#6c8cff"></span> ${k}: ${raw}</span>`;
    }).join('');
  }
}

// ====== Envelope (Hilbert) ======
function drawEnvelope(canvasId, envelope, signal) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 180 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!envelope || envelope.length < 2) return;

  const maxVal = Math.max(...envelope.map(Math.abs), 1e-8);
  const pad = 10;
  const plotH = h - pad * 2;
  const plotW = w - pad * 2;
  const xScale = plotW / envelope.length;

  // Fill (envelope area)
  ctx.fillStyle = 'rgba(167, 139, 250, 0.1)';
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  for (let i = 0; i < envelope.length; i++) {
    const x = pad + i * xScale;
    const y = h - pad - (envelope[i] / maxVal) * plotH;
    ctx.lineTo(x, y);
  }
  for (let i = envelope.length - 1; i >= 0; i--) {
    const x = pad + i * xScale;
    ctx.lineTo(x, h - pad);
  }
  ctx.closePath();
  ctx.fill();

  // Envelope curve
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < envelope.length; i++) {
    const x = pad + i * xScale;
    const y = h - pad - (envelope[i] / maxVal) * plotH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Signal overlay (faded)
  if (signal && signal.length > 1) {
    const sigMax = Math.max(...signal.map(Math.abs), 1e-8);
    const sigScale = plotH / maxVal * 0.5;
    ctx.strokeStyle = 'rgba(108, 140, 255, 0.15)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(signal.length / envelope.length));
    for (let i = 0; i < envelope.length; i++) {
      const x = pad + i * xScale;
      const idx = Math.min(i * step, signal.length - 1);
      const y = h - pad - (signal[idx] / sigMax) * plotH * 0.5;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('0', pad, h - 2);
  ctx.textAlign = 'right';
  ctx.fillText(`${maxVal.toFixed(2)}`, pad + plotW, pad + 10);
}

// ====== Cepstrum ======
function drawCepstrum(canvasId, quefrency, amplitude) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 180 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!quefrency || !amplitude || quefrency.length < 2) return;

  const maxAmp = Math.max(...amplitude, 1e-10);
  const maxQ = quefrency[quefrency.length - 1];
  const pad = { l: 10, r: 10, t: 8, b: 20 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  // Bars
  const barW = Math.max(1, plotW / amplitude.length);
  for (let i = 0; i < amplitude.length; i++) {
    const x = pad.l + (quefrency[i] / maxQ) * plotW;
    const barH = (amplitude[i] / maxAmp) * plotH;
    ctx.fillStyle = i < 2 ? 'rgba(251, 191, 36, 0.4)' : 'rgba(108, 140, 255, 0.5)';
    ctx.fillRect(x, h - pad.b - barH, Math.max(barW - 0.5, 1), Math.max(barH, 1));
  }

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('0 s', pad.l, h - 2);
  ctx.textAlign = 'right';
  ctx.fillText(`${maxQ.toFixed(3)} s`, w - pad.r, h - 2);
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Quefrency`, pad.l, h - 5);
}

// ====== Wavelet Packet Energy ======
function drawWaveletEnergy(canvasId, wavelet) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 200 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 200;
  ctx.clearRect(0, 0, w, h);

  if (!wavelet || !wavelet.energies) return;

  const energies = wavelet.energies;
  const keys = Object.keys(energies);
  const maxE = Math.max(...Object.values(energies), 1e-10);
  const pad = { l: 30, r: 10, t: 10, b: 8 };
  const show = Math.min(keys.length, 31);
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const barW = plotW / show;

  const colors = ['#6c8cff', '#4ade80', '#fbbf24', '#f87171', '#a78bfa', '#60a5fa', '#34d399', '#f472b6'];
  for (let i = 0; i < show; i++) {
    const key = keys[i];
    const val = energies[key] / maxE;
    const x = pad.l + i * barW;
    const barH = val * plotH;
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(x, h - pad.b - barH, Math.max(barW - 1, 1), Math.max(barH, 1));
    // Node label (rotate for deep levels)
    if (barW > 15 && i % 2 === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.25)';
      ctx.font = '8px sans-serif';
      ctx.textAlign = 'center';
      ctx.save();
      ctx.translate(x + barW / 2, h - pad.b + 2);
      ctx.rotate(-Math.PI / 3);
      ctx.fillText(key, 0, 0);
      ctx.restore();
    }
  }

  // Meta info
  const meta = document.getElementById('wavelet-meta');
  if (meta && wavelet.meta) {
    meta.textContent = `Wavelet: ${wavelet.meta.wavelet} (${wavelet.meta.kind}) | Niveles: ${wavelet.level} | Nodos: ${keys.length}`;
  }
}

// ====== Mel Spectrogram ======
function drawMelSpectrogram(canvasId, values) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 180 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!values || values.length < 2) return;

  const nBands = values.length;
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const range = maxV - minV || 1;
  const bandH = h / nBands;

  for (let i = 0; i < nBands; i++) {
    const norm = Math.max(0, Math.min(1, (values[i] - minV) / range));
    const hue = 240 - norm * 200;
    ctx.fillStyle = `hsl(${hue}, 80%, ${20 + norm * 40}%)`;
    ctx.fillRect(0, i * bandH, w, Math.ceil(bandH) + 1);
  }

  // Band labels
  ctx.fillStyle = 'rgba(255,255,255,0.15)';
  ctx.font = '8px sans-serif';
  ctx.textAlign = 'right';
  if (nBands > 0) {
    ctx.fillText(`${nBands} bandas Mel`, w - 4, 10);
    ctx.textAlign = 'left';
    ctx.fillText('0', 4, h - 4);
  }
}

// ====== MFCC ======
function drawMFCC(canvasId, values) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 180 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!values || values.length < 2) return;

  const maxVal = Math.max(...values.map(Math.abs), 1e-10);
  const pad = { l: 8, r: 8, t: 8, b: 22 };
  const plotH = h - pad.t - pad.b;
  const n = values.length;
  const barW = (w - pad.l - pad.r) / n;

  for (let i = 0; i < n; i++) {
    const x = pad.l + i * barW;
    const norm = values[i] / maxVal;
    const barH = Math.abs(norm) * plotH;
    const y = norm >= 0 ? h - pad.b - barH : h - pad.b;
    ctx.fillStyle = norm >= 0 ? 'rgba(108, 140, 255, 0.7)' : 'rgba(248, 113, 113, 0.7)';
    ctx.fillRect(x, y, Math.max(barW - 1, 1), Math.max(barH, 1));
    // Index label
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${i + 1}`, x + barW / 2, h - 2);
  }

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = 'rgba(255,255,255,0.15)';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Coeficiente', pad.l, pad.t + 10);
}

// ====== Chromagram ======
function drawChromagram(canvasId, chromagram) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 180 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  if (!chromagram || !chromagram.notes || !chromagram.values) return;

  const notes = chromagram.notes;
  const values = chromagram.values;
  const n = notes.length;
  const maxVal = Math.max(...values, 1e-10);
  const pad = { l: 4, r: 4, t: 6, b: 24 };
  const plotH = h - pad.t - pad.b;
  const barW = (w - pad.l - pad.r) / n;

  const colors = ['#ef4444', '#f97316', '#fbbf24', '#84cc16', '#22c55e', '#14b8a6',
                  '#06b6d4', '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef'];

  for (let i = 0; i < n; i++) {
    const x = pad.l + i * barW;
    const barH = (values[i] / maxVal) * plotH;
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(x, h - pad.b - barH, Math.max(barW - 1, 1), Math.max(barH, 1));
    // Note label
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(notes[i], x + barW / 2, h - 2);
    // Value
    if (barH > 20) {
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '8px sans-serif';
      ctx.fillText((values[i] * 100).toFixed(0) + '%', x + barW / 2, h - pad.b - barH + 10);
    }
  }
}

// ====== FA Map (heatmap) ======
function drawFAMap(canvasId, faData, shape) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 220 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 220;
  ctx.clearRect(0, 0, w, h);

  if (!faData || !shape) return;
  const [Z, Y, X] = shape;
  const sliceW = w / X;
  const sliceH = h / Y;

  for (let y = 0; y < Y; y++) {
    for (let x = 0; x < X; x++) {
      const idx = y * X + x;
      const val = Math.min(faData[idx] || 0, 1);
      const hue = 240 - val * 240;
      ctx.fillStyle = `hsl(${hue}, 70%, ${15 + val * 45}%)`;
      ctx.fillRect(x * sliceW, y * sliceH, Math.ceil(sliceW) + 1, Math.ceil(sliceH) + 1);
    }
  }

  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '11px sans-serif';
  ctx.fillText('FA', 4, 14);
  ctx.textAlign = 'right';
  ctx.fillText('1.0', w - 4, 14);
  ctx.fillText('0.0', w - 4, h - 4);
}

// ====== Color FA ======
function drawColorFA(canvasId, colorData, shape) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 220 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 220;
  ctx.clearRect(0, 0, w, h);

  if (!colorData || !shape) return;
  const [Z, Y, X] = shape;
  const sliceW = w / X;
  const sliceH = h / Y;

  for (let y = 0; y < Y; y++) {
    for (let x = 0; x < X; x++) {
      const idx = y * X + x;
      const rgb = colorData[idx] || [0, 0, 0];
      const r = Math.min(Math.max(Math.round(rgb[0] * 255), 0), 255);
      const g = Math.min(Math.max(Math.round(rgb[1] * 255), 0), 255);
      const b = Math.min(Math.max(Math.round(rgb[2] * 255), 0), 255);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x * sliceW, y * sliceH, Math.ceil(sliceW) + 1, Math.ceil(sliceH) + 1);
    }
  }

  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('R=LR  G=AP  B=SI', 4, 14);
}

// ====== Zoom/Pan view state ======
const tractViews = {};

function initTractView(canvasId, streamlines, shape, getColor) {
  if (!streamlines || streamlines.length === 0 || !shape) return;
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const [Z, Y, X] = shape;
  const pad = 15;

  let minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const sl of streamlines) {
    for (const pt of sl) {
      if (pt[0] < minY) minY = pt[0];
      if (pt[0] > maxY) maxY = pt[0];
      if (pt[1] < minZ) minZ = pt[1];
      if (pt[1] > maxZ) maxZ = pt[1];
    }
  }
  if (!isFinite(minY)) return;

  const w = canvas.clientWidth;
  const h = canvas.clientHeight || 250;
  const baseZoom = Math.min((w - pad * 2) / (maxY - minY || 1), (h - pad * 2) / (maxZ - minZ || 1));

  tractViews[canvasId] = {
    minY, maxY, minZ, maxZ, baseZoom,
    panX: 0, panY: 0, zoom: 1, pad,
    streamlines, shape, getColor,
  };

  setupTractInteraction(canvasId);
}

function drawTractView(canvasId) {
  const v = tractViews[canvasId];
  if (!v) return;
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const ctx = canvas.getContext('2d');
  const ch = +(canvas.getAttribute('height') || 250);
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = ch * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = ch;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#080a12';
  ctx.fillRect(0, 0, w, h);

  const { minY, maxY, minZ, maxZ, baseZoom, panX, panY, zoom, pad, streamlines, getColor } = v;
  if (!streamlines || streamlines.length === 0) return;
  const effective = baseZoom * zoom;

  const toScreen = (wy, wz) => [
    pad + (wy - minY) * effective + panX,
    pad + (wz - minZ) * effective + panY,
  ];

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  const [vx, vy] = toScreen(minY, minZ);
  const [vx2, vy2] = toScreen(maxY, maxZ);
  ctx.strokeRect(vx, vy, vx2 - vx, vy2 - vy);

  const defaultColors = ['#a78bfa', '#6c8cff', '#4ade80', '#fbbf24', '#f87171', '#60a5fa', '#34d399', '#f472b6'];
  for (let i = 0; i < streamlines.length; i++) {
    const sl = streamlines[i];
    if (sl.length < 2) continue;
    ctx.strokeStyle = getColor ? getColor(i) : defaultColors[i % defaultColors.length];
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    for (let j = 0; j < sl.length; j++) {
      const [px, py] = toScreen(sl[j][0], sl[j][1]);
      j === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Streamlines: ${streamlines.length}  Zoom: ${zoom.toFixed(1)}x`, 4, 12);
  ctx.textAlign = 'right';
  ctx.fillText('Proyección axial (YZ)', w - 4, 12);
}

function setupTractInteraction(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  let dragging = false, lastX = 0, lastY = 0;

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const v = tractViews[canvasId];
    if (!v) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const delta = -e.deltaY * 0.001;
    const newZoom = Math.max(0.1, Math.min(50, v.zoom * (1 + delta)));
    const eff = v.baseZoom * v.zoom;
    const worldX = (mx - v.pad - v.panX) / eff + v.minY;
    const worldY = (my - v.pad - v.panY) / eff + v.minZ;
    v.zoom = newZoom;
    const newEff = v.baseZoom * v.zoom;
    v.panX = mx - (worldX - v.minY) * newEff - v.pad;
    v.panY = my - (worldY - v.minZ) * newEff - v.pad;
    drawTractView(canvasId);
  }, { passive: false });

  canvas.addEventListener('mousedown', (e) => {
    if (e.button === 0 || e.button === 1) {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.style.cursor = 'grabbing';
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const v = tractViews[canvasId];
    if (!v) return;
    v.panX += e.clientX - lastX; v.panY += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    drawTractView(canvasId);
  });

  window.addEventListener('mouseup', () => {
    if (dragging) { dragging = false; canvas.style.cursor = 'default'; }
  });

  canvas.addEventListener('dblclick', () => {
    const v = tractViews[canvasId];
    if (!v) return;
    v.zoom = 1; v.panX = 0; v.panY = 0;
    drawTractView(canvasId);
  });
}

// ====== Tractography (2D projection with zoom/pan) ======
function drawTractography(canvasId, streamlines, shape) {
  initTractView(canvasId, streamlines, shape, null);
  drawTractView(canvasId);
}

// ====== Shape indices ======
function drawShapeIndices(canvasId, slices, shape, faRange, faSlice) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 220 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 220;
  ctx.clearRect(0, 0, w, h);

  if (!slices || !shape) return;
  const [Z, Y, X] = shape;
  const sliceW = w / X;
  const sliceH = h / Y;

  const types = [
    { key: 'cl', color: '#f87171', label: 'CL (lineal)' },
    { key: 'cp', color: '#4ade80', label: 'CP (planar)' },
    { key: 'cs', color: '#6c8cff', label: 'CS (esférica)' },
  ];

  const third = Math.floor(X / 3);
  const offsets = [0, third, third * 2];

  for (let t = 0; t < 3; t++) {
    const data = slices[types[t].key];
    const ox = offsets[t];
    const w3 = third * sliceW;
    for (let y = 0; y < Y; y++) {
      for (let x = 0; x < third; x++) {
        const idx = y * X + (x + ox);
        const val = Math.min(Math.max((data ? data[idx] : 0) || 0, 0), 1);
        const r = Math.round(types[t].color === '#f87171' ? parseInt(types[t].color.slice(1,3),16) * val : 0);
        const g = Math.round(types[t].color === '#4ade80' ? parseInt(types[t].color.slice(1,3),16) * val : 0);
        const b = Math.round(types[t].color === '#6c8cff' ? parseInt(types[t].color.slice(1,3),16) * val : 0);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x * sliceW, y * sliceH, Math.ceil(sliceW) + 1, Math.ceil(sliceH) + 1);
      }
    }
    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(types[t].label, ox * sliceW + 2, 12);
  }
}

// ====== Fiber histogram ======
function drawFiberHistogram(canvasId, fiberStats) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  if (!fiberStats || !fiberStats.lengths) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 180 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 180;
  ctx.clearRect(0, 0, w, h);

  const lengths = fiberStats.lengths;
  const bins = Math.min(15, lengths.length);
  const minL = Math.min(...lengths);
  const maxL = Math.max(...lengths);
  const range = maxL - minL || 1;
  const binW = range / bins;
  const counts = new Array(bins).fill(0);
  for (const l of lengths) {
    const idx = Math.min(Math.floor((l - minL) / binW), bins - 1);
    counts[idx]++;
  }
  const maxCount = Math.max(...counts, 1);
  const pad = { l: 8, r: 8, t: 8, b: 22 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const barW = plotW / bins;

  for (let i = 0; i < bins; i++) {
    const x = pad.l + i * barW;
    const barH = (counts[i] / maxCount) * plotH;
    ctx.fillStyle = '#a78bfa';
    ctx.fillRect(x, h - pad.b - barH, Math.max(barW - 1, 1), Math.max(barH, 1));
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText((minL + i * binW).toFixed(1), x + barW / 2, h - 2);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Longitud (mm)', pad.l, pad.t + 8);
  ctx.textAlign = 'right';
  ctx.fillText(`${maxCount}`, w - pad.r, pad.t + 8);
}

// ====== Connectivity matrix (heatmap) ======
function drawConnectivityMatrix(canvasId, matrix, labels) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = canvas.clientWidth * dpr;
  const H = canvas.height = 220 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 220;
  ctx.clearRect(0, 0, w, h);

  if (!matrix || matrix.length < 2) return;
  const n = matrix.length;
  const size = Math.min(w, h - 20) / n;
  const ox = (w - size * n) / 2;
  const oy = 10;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = Math.min(Math.max(matrix[i][j] || 0, 0), 1);
      const r = Math.round(10 + val * 108);
      const g = Math.round(10 + val * 140);
      const b = Math.round(10 + val * 255);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(ox + j * size, oy + i * size, Math.ceil(size) + 1, Math.ceil(size) + 1);
    }
  }

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '7px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    if (labels && labels[i]) {
      ctx.fillText(labels[i].replace('Region_', 'R'), ox + i * size + size / 2, oy + n * size + 10);
      ctx.save();
      ctx.translate(ox - 4, oy + i * size + size / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(labels[i].replace('Region_', 'R'), 0, 0);
      ctx.restore();
    }
  }
}

// ====== Cluster tractography (with zoom/pan) ======
function drawClusterTracts(canvasId, streamlines, shape, clusters) {
  if (!streamlines || !clusters || streamlines.length === 0) return;
  const clusterColors = ['#f87171', '#4ade80', '#6c8cff', '#fbbf24', '#a78bfa', '#60a5fa', '#34d399', '#f472b6'];
  const getColor = (i) => {
    const cluster = clusters[i] !== undefined ? clusters[i] : 0;
    return clusterColors[cluster % clusterColors.length];
  };
  initTractView(canvasId, streamlines, shape, getColor);
  drawTractView(canvasId);
}

// ====== Main analysis ======
async function apiPost(endpoint, data) {
  const resp = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

document.getElementById('btn-analyze').addEventListener('click', async () => {
  if (!currentSignal) return;
  const sr = +document.getElementById('sample-rate').value;
  currentSampleRate = sr;
  document.getElementById('loading').classList.remove('hidden');

  try {
    const sig = currentSignal;

    // Build neuro params
    const neuroFa = +document.getElementById('neuro-fa').value;
    const neuroMd = +document.getElementById('neuro-md').value * 1e-4;
    const neuroDir = document.getElementById('neuro-dir').value.split(',').map(Number);

    // Run all analyses in parallel
    const [mData, sData, fData, spData, tData, nData] = await Promise.all([
      apiPost('/api/metrics', { signal: sig, sample_rate: sr }),
      apiPost('/api/spectral', { signal: sig, sample_rate: sr }),
      apiPost('/api/features', { signal: sig }),
      apiPost('/api/spectrum', { signal: sig, sample_rate: sr }),
      apiPost('/api/transforms', { signal: sig, sample_rate: sr }),
      apiPost('/api/neuro/analyze', {
        mode: 'demo',
        shape: [12, 10, 8],
        fa: neuroFa,
        md: neuroMd,
        fiber_dir: neuroDir,
      }),
    ]);

    document.getElementById('results-section').classList.remove('hidden');

    if (mData.success) {
      const { psd_freqs, psd_values, autocorr, ...rest } = mData.metrics;
      renderMetrics('metrics-display', fData.success ? fData.features : rest);
      drawPSD('psd-canvas', psd_freqs || [], psd_values || []);
      drawBars('ac-canvas', autocorr || []);
    }

    // Waveform temporal
    drawWaveform('waveform-canvas', sig, sr);

    if (sData.success) {
      const { psd_freqs, psd_values, autocorr, spectrogram, ...rest } = sData.spectral;
      renderMetrics('spectral-display', rest);
      if (!mData.success) {
        drawPSD('psd-canvas', psd_freqs || [], psd_values || []);
        drawBars('ac-canvas', autocorr || []);
      }
      drawSpectrogram('spec-canvas', spectrogram);
    }

    // Spectrum (magnitude + phase + peaks)
    if (spData.success) {
      const sp = spData.spectrum;
      drawSpectrum('spectrum-canvas', sp.freqs, sp.magnitude, sp.peaks, sp.centroid_hz, sp.rolloff_hz);
      drawPhase('phase-canvas', sp.freqs, sp.phase);
      renderPeaks('peaks-list', sp.peaks);
      const legend = document.getElementById('spectrum-legend');
      if (legend) {
        legend.innerHTML = `
          <span><span class="legend-line" style="background:#6c8cff"></span> Magnitud</span>
          <span><span class="legend-dot" style="background:#4ade80"></span> Picos</span>
          <span><span class="legend-line" style="background:transparent;height:2px;border-top:2px dashed #f87171"></span> Centroide ${sp.centroid_hz} Hz</span>
          <span><span class="legend-line" style="background:transparent;height:2px;border-top:2px dashed #fbbf24"></span> Rolloff ${sp.rolloff_hz} Hz</span>
        `;
      }
    }

    // Transforms
    if (tData.success) {
      const tr = tData.transforms;
      drawEnvelope('envelope-canvas', tr.envelope, sig);
      drawCepstrum('cepstrum-canvas', tr.cepstrum.quefrency, tr.cepstrum.amplitude);
      drawWaveletEnergy('wavelet-canvas', tr.wavelet);
      drawMelSpectrogram('mel-canvas', tr.mel_spectrogram);
      drawMFCC('mfcc-canvas', tr.mfcc);
      drawChromagram('chroma-canvas', tr.chromagram);
    }

    // Radar chart
    const radarMetrics = {};
    if (mData.success) {
      const m = mData.metrics;
      radarMetrics.SNR_dB = { value: m.snr_db, min: 0, max: 50 };
    }
    if (fData.success) {
      const f = fData.features;
      radarMetrics.Energía = { value: f.energy, min: 0 };
      radarMetrics.RMS = { value: f.rms, min: 0 };
    }
    if (sData.success) {
      const sp2 = sData.spectral;
      radarMetrics.Entropía = { value: sp2.spectral_entropy, min: 0, max: 5 };
      radarMetrics.Planitud = { value: sp2.flatness, min: 0, max: 1 };
    }
    if (spData.success) {
      const sp = spData.spectrum;
      radarMetrics.Centroide = { value: sp.centroid_hz / (sr / 2), min: 0, max: 1 };
    }
    drawRadar('radar-canvas', radarMetrics);

    // Waveform stats
    const wStats = document.getElementById('waveform-stats');
    if (wStats && currentSignal) {
      const s = currentSignal;
      wStats.innerHTML = `
        <span>Muestras: ${s.length}</span>
        <span>Min: ${Math.min(...s).toFixed(4)}</span>
        <span>Max: ${Math.max(...s).toFixed(4)}</span>
        <span>Media: ${(s.reduce((a, b) => a + b, 0) / s.length).toFixed(4)}</span>
      `;
    }

    // Neuro (DTI + tractography)
    if (nData.success) {
      const n = nData.neuro;
      drawFAMap('fa-canvas', n.fa_slice, n.shape);
      drawColorFA('color-fa-canvas', n.color_fa_slice, n.shape);
      drawShapeIndices('shape-canvas', {
        cl: n.cl_slice, cp: n.cp_slice, cs: n.cs_slice
      }, n.shape, n.fa_range, n.fa_slice);
      drawTractography('tract-canvas', n.streamlines, n.shape);
      drawFiberHistogram('fiber-hist-canvas', n.fiber_stats);
      drawConnectivityMatrix('connectivity-canvas', n.connectivity_matrix, n.connectivity_labels);
      drawClusterTracts('cluster-canvas', n.streamlines, n.shape, n.clusters);
      document.getElementById('fa-meta').textContent =
        `FA: ${n.fa_range[0].toFixed(3)} – ${n.fa_range[1].toFixed(3)}  |  MD: ${n.md_range[0]} – ${n.md_range[1]}`;
      document.getElementById('tract-meta').textContent =
        `Streamlines: ${n.n_streamlines} | Volumen: ${n.shape.join('×')} voxels`;
      if (n.fiber_stats) {
        const fs = n.fiber_stats;
        document.getElementById('fiber-stats-meta').textContent =
          `Longitud media: ${fs.mean_length} mm  σ=${fs.std_length}  [${fs.min_length}–${fs.max_length}]  FA media: ${fs.mean_fa}`;
      }
      // Show neuro cards
      document.querySelectorAll('.neuro-section').forEach(el => el.classList.remove('hidden'));
    } else {
      document.querySelectorAll('.neuro-section').forEach(el => el.classList.add('hidden'));
    }

    // Interpretation
    const metrics = mData.success ? mData.metrics : null;
    const features = fData.success ? fData.features : null;
    const spectral = sData.success ? sData.spectral : null;
    generateInterpretation(metrics, features, spectral);

    window.scrollTo({ top: document.getElementById('preview-section').offsetTop, behavior: 'smooth' });
  } catch (err) {
    console.error(err);
    setStatus('Error en el análisis: ' + err.message, true);
  } finally {
    document.getElementById('loading').classList.add('hidden');
  }
});

// ====== Resize canvases ======
window.addEventListener('resize', () => {
  if (currentSignal) drawSignal(currentSignal);
});
