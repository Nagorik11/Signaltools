/* SignalTools — WiFi Analyzer */

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

function setStatus(msg, isError) {
  const el = document.getElementById('scan-status');
  el.textContent = msg;
  el.className = 'status' + (isError ? ' error' : ' ready');
}

function renderChannelList(channels, showUtil) {
  const container = document.getElementById('channel-list');
  if (!channels || channels.length === 0) {
    container.innerHTML = '<div style="color:var(--text-dim)">Sin datos</div>';
    return;
  }
  container.innerHTML = channels.map(ch => {
    const rssiNorm = Math.max(0, Math.min(1, ((ch.rssi_dbm || -95) + 95) / 60));
    const utilNorm = (ch.utilization_pct || 0) / 100;
    const rssiColor = rssiNorm > 0.7 ? '#4ade80' : rssiNorm > 0.4 ? '#fbbf24' : '#f87171';
    const nets = ch.networks && ch.networks.length > 0
      ? ch.networks.join(', ') : '<span style="color:var(--text-dim)">—</span>';
    return `
      <div class="channel-row">
        <span class="ch-num">${ch.channel}</span>
        <span class="ch-freq">${ch.freq_mhz || '—'} MHz</span>
        <div class="ch-bar-wrap" title="RSSI: ${ch.rssi_dbm ?? '—'} dBm">
          <div class="ch-bar rssi" style="width:${rssiNorm * 100}%;background:${rssiColor}"></div>
        </div>
        <span class="ch-val">${ch.rssi_dbm ?? '—'} dBm</span>
        ${showUtil ? `
        <div class="ch-bar-wrap" title="Utilización: ${ch.utilization_pct}%">
          <div class="ch-bar util" style="width:${utilNorm * 100}%"></div>
        </div>
        <span class="ch-val">${ch.utilization_pct}%</span>` : `
        <span class="ch-val short">—</span>`}
        <span class="ch-networks">${nets}</span>
      </div>
    `;
  }).join('');
}

function renderNetworksTable(networks) {
  const container = document.getElementById('networks-table-container');
  if (!networks || networks.length === 0) {
    container.innerHTML = '';
    return;
  }
  const sorted = [...networks].sort((a, b) => (b.rssi_dbm || -100) - (a.rssi_dbm || -100));
  container.innerHTML = `
    <table class="networks-table">
      <tr><th>BSSID</th><th>SSID</th><th>Canal</th><th>Freq</th><th>RSSI</th><th>Ruido</th><th>SNR</th></tr>
      ${sorted.map(n => `
        <tr>
          <td style="color:var(--text-dim)">${n.bssid}</td>
          <td>${n.ssid}</td>
          <td>${n.channel}</td>
          <td>${n.freq_mhz} MHz</td>
          <td style="color:${n.rssi_dbm > -50 ? 'var(--green)' : n.rssi_dbm > -65 ? 'var(--orange)' : 'var(--red)'}">${n.rssi_dbm ?? '—'} dBm</td>
          <td style="color:var(--text-dim)">${n.noise_dbm ?? '—'} dBm</td>
          <td>${n.snr_db ?? '—'} dB</td>
        </tr>`).join('')}
    </table>`;
  document.getElementById('networks-table-card').classList.remove('hidden');
}

function renderMetrics(analysis) {
  const container = document.getElementById('wifi-metrics');
  if (!analysis) return;
  const metrics = [
    { label: 'RSSI medio', val: analysis.avg_rssi != null ? `${analysis.avg_rssi} dBm` : '—',
      cls: analysis.avg_rssi > -50 ? 'good' : analysis.avg_rssi > -65 ? 'warn' : 'bad' },
    { label: 'SNR medio', val: analysis.avg_snr != null ? `${analysis.avg_snr} dB` : '—',
      cls: analysis.avg_snr > 30 ? 'good' : analysis.avg_snr > 20 ? 'warn' : 'bad' },
    { label: 'Mejor canal', val: analysis.best_channel != null ? `${analysis.best_channel}` : '—', cls: 'good' },
    { label: 'SNR mejor canal', val: analysis.best_snr != null ? `${analysis.best_snr} dB` : '—', cls: 'good' },
    { label: 'Interferencia', val: analysis.interference_pct != null ? `${analysis.interference_pct}%` : '—',
      cls: analysis.interference_pct < 20 ? 'good' : analysis.interference_pct < 50 ? 'warn' : 'bad' },
    { label: 'Redes', val: analysis.networks_found != null ? `${analysis.networks_found}` : '—', cls: 'good' },
  ];
  container.innerHTML = metrics.map(m =>
    `<div class="metric-wifi ${m.cls}"><div class="val">${m.val}</div><div class="label">${m.label}</div></div>`
  ).join('');
}

function drawWiFiSpectrum(canvasId, freqs, mag) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = 200 * dpr;
  ctx.scale(dpr, dpr);
  const w = canvas.clientWidth, h = 200;
  ctx.clearRect(0, 0, w, h);

  if (!freqs || !mag || freqs.length < 2) {
    ctx.fillStyle = 'rgba(255,255,255,0.1)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Sin datos de espectro', w / 2, h / 2);
    return;
  }

  const maxFreq = freqs[freqs.length - 1];
  const maxMag = Math.max(...mag, 1e-10);
  const pad = 10;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  ctx.fillStyle = 'rgba(108, 140, 255, 0.06)';
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  for (let i = 0; i < freqs.length; i++) {
    ctx.lineTo(pad + (freqs[i] / maxFreq) * plotW, h - pad - (mag[i] / maxMag) * plotH);
  }
  ctx.lineTo(pad + plotW, h - pad);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = '#6c8cff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = pad + (freqs[i] / maxFreq) * plotW;
    const y = h - pad - (mag[i] / maxMag) * plotH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  const chFreqs = [2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 2452, 2457, 2462, 2467, 2472];
  const chLabels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
  for (let i = 0; i < chFreqs.length; i++) {
    const x = pad + (chFreqs[i] / maxFreq) * plotW;
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(chLabels[i], x, h - 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
  }

  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`${freqs[0].toFixed(0)} MHz`, pad, pad + 10);
  ctx.textAlign = 'right';
  ctx.fillText(`${maxFreq.toFixed(0)} MHz`, pad + plotW, pad + 10);

  const legend = document.getElementById('wifi-spectrum-legend');
  if (legend) {
    legend.innerHTML = `
      <span><span class="legend-line" style="background:#6c8cff"></span> Potencia recibida</span>
      <span><span class="legend-dot" style="background:transparent;border:1px solid rgba(255,255,255,0.15)"></span> Canales (1–13)</span>
    `;
  }
}

function generateRecommendation(analysis, channels) {
  const container = document.getElementById('wifi-recommendation');
  if (!analysis) return;
  const parts = [];

  if (analysis.best_channel != null) {
    const bestRssi = channels?.find(c => c.channel === analysis.best_channel)?.rssi_dbm;
    parts.push(`<p><span class="tag tag-low">Recomendado</span> Canal <strong>${analysis.best_channel}</strong> con SNR de <strong>${analysis.best_snr} dB</strong>${bestRssi != null ? ` y RSSI de ${bestRssi} dBm` : ''}.</p>`);
  }
  if (analysis.worst_channel != null) {
    parts.push(`<p><span class="tag tag-high">Evitar</span> Canal <strong>${analysis.worst_channel}</strong> (SNR ${analysis.worst_snr} dB).</p>`);
  }

  const interference = analysis.interference_pct ?? 0;
  if (interference > 50) {
    parts.push(`<p><span class="tag tag-high">Interferencia</span> ${interference}% de canales con múltiples redes. Considerar 5 GHz o canales 1, 6 u 11.</p>`);
  } else if (interference > 20) {
    parts.push(`<p><span class="tag tag-mid">Interferencia moderada</span> ${interference}% de canales solapados.</p>`);
  } else {
    parts.push(`<p><span class="tag tag-low">Baja interferencia</span> Solo ${interference}% con solapamiento.</p>`);
  }

  const snr = analysis.avg_snr;
  if (snr != null) {
    if (snr > 35) parts.push(`<p><span class="tag tag-low">SNR excelente</span> ${snr} dB promedio.</p>`);
    else if (snr > 25) parts.push(`<p><span class="tag tag-mid">SNR adecuada</span> ${snr} dB promedio.</p>`);
    else parts.push(`<p><span class="tag tag-high">SNR baja</span> ${snr} dB promedio.</p>`);
  }

  container.innerHTML = parts.join('\n');
}

function showResults(data) {
  document.getElementById('results-section').classList.remove('hidden');
  renderChannelList(data.scan, data.scan?.[0]?.utilization_pct != null);
  renderMetrics(data.analysis);
  renderNetworksTable(data.networks || []);
  if (data.spectrum) {
    drawWiFiSpectrum('wifi-spectrum-canvas', data.spectrum.freqs, data.spectrum.magnitude);
  }
  generateRecommendation(data.analysis, data.scan);
}

/* --- Tab switching --- */
document.querySelectorAll('.input-tabs .tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.input-tabs .tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

/* --- Synthetic scan --- */
document.getElementById('btn-scan').addEventListener('click', async () => {
  const band = document.getElementById('wifi-band').value;
  const numNetworks = +document.getElementById('wifi-networks').value;
  const noiseFloor = +document.getElementById('wifi-noise').value;

  document.getElementById('btn-scan').disabled = true;
  setStatus('Escaneando...', false);

  try {
    const data = await apiPost('/api/wifi/analyze', {
      band, num_networks: numNetworks, noise_floor: noiseFloor,
    });

    if (!data.success) { setStatus('Error: ' + data.error, true); return; }

    showResults(data);
    setStatus(`Simulación — ${data.analysis.channel_count} canales, ${data.scan.filter(c => c.networks.length > 0).length} con actividad`, false);
  } catch (err) {
    setStatus('Error: ' + err.message, true);
  } finally {
    document.getElementById('btn-scan').disabled = false;
  }
});

/* --- PCAP analysis --- */
document.getElementById('btn-analyze-pcap').addEventListener('click', async () => {
  const fileInput = document.getElementById('pcap-file');
  const file = fileInput.files[0];
  if (!file) { setStatus('Selecciona un archivo PCAP', true); return; }

  document.getElementById('btn-analyze-pcap').disabled = true;
  setStatus('Analizando PCAP...', false);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const resp = await fetch('/api/wifi/analyze-pcap', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!data.success) { setStatus('Error: ' + data.error, true); return; }

    showResults(data);
    setStatus(`${data.networks.length} redes encontradas en ${data.analysis.channel_count} canales`, false);
  } catch (err) {
    setStatus('Error: ' + err.message, true);
  } finally {
    document.getElementById('btn-analyze-pcap').disabled = false;
  }
});
