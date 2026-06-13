from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import signaltools as st

app = Flask(__name__, static_folder="static")


def _parse_signal(body: dict) -> list[float]:
    if "signal" in body:
        return body["signal"]
    raise ValueError("Missing 'signal' in request body")


def _generate_demo(length: int = 2000, sample_rate: int = 2000) -> list[float]:
    t = np.arange(length) / sample_rate
    sig = 0.7 * np.sin(2 * np.pi * 120 * t)
    sig += 0.25 * np.sin(2 * np.pi * 280 * t)
    sig += 0.08 * np.random.default_rng(42).normal(size=len(t))
    return sig.astype(np.float64).tolist()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/demo", methods=["GET"])
def api_demo():
    length = int(request.args.get("length", 2000))
    sr = int(request.args.get("sample_rate", 2000))
    return jsonify({"signal": _generate_demo(length, sr)})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(force=True)
    signal = _parse_signal(body)
    sr = int(body.get("sample_rate", 2000))
    frame_size = int(body.get("frame_size", 256))
    hop_size = int(body.get("hop_size", 128))

    try:
        result = st.analyze_signal_advanced(signal, sample_rate=sr, frame_size=frame_size, hop_size=hop_size)
        return jsonify({"success": True, "result": result.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/metrics", methods=["POST"])
def api_metrics():
    body = request.get_json(force=True)
    signal = _parse_signal(body)
    sr = int(body.get("sample_rate", 2000))

    try:
        pipeline = st.create_metrics_pipeline(sample_rate=sr)
        result = pipeline.execute(signal)
        metrics = result.context.get("metrics", {})
        return jsonify({"success": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/spectral", methods=["POST"])
def api_spectral():
    body = request.get_json(force=True)
    signal = _parse_signal(body)
    sr = int(body.get("sample_rate", 2000))

    try:
        spectral = {
            "centroid_hz": round(st.spectral_centroid(signal, sample_rate=sr), 6),
            "bandwidth_hz": round(st.spectral_bandwidth(signal, sample_rate=sr), 6),
            "rolloff_hz": round(st.spectral_rolloff(signal, sample_rate=sr), 6),
            "pitch_hz": round(st.estimate_pitch(signal, sample_rate=sr), 6),
            "dominant_freq_hz": round(st.dominant_frequency(signal, sample_rate=sr), 6),
            "spectral_entropy": round(st.spectral_entropy(signal), 6),
            "snr_db": st.estimate_snr(signal),
            "flatness": st.spectral_flatness(st.power_spectrum(signal)),
        }
        freqs, psd = st.power_spectral_density(signal, sample_rate=sr)
        spectral["psd_freqs"] = [round(f, 2) for f in freqs]
        spectral["psd_values"] = [round(p, 10) for p in psd]
        ac = st.autocorrelation(signal, normalize=True)
        spectral["autocorr"] = [round(v, 6) for v in ac[:64]]
        spec = st.spectrogram_matrix(signal, frame_size=128, hop_size=64)
        spectral["spectrogram"] = spec
        return jsonify({"success": True, "spectral": spectral})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/features", methods=["POST"])
def api_features():
    body = request.get_json(force=True)
    signal = _parse_signal(body)

    try:
        features = {
            "rms": round(st.rms(signal), 6),
            "energy": round(st.signal_energy(signal), 6),
            "power": round(st.signal_power(signal), 6),
            "variance": round(st.variance(signal), 6),
            "mean": round(st.mean(signal), 6),
            "stddev": round(st.stddev(signal), 6),
            "crest_factor": round(st.crest_factor(signal), 6),
            "zero_crossing_rate": round(st.zero_crossing_rate(signal), 6),
            "skewness": round(st.skewness(signal), 6),
            "kurtosis": round(st.kurtosis(signal), 6),
            "waveform_length": round(st.waveform_length(signal), 6),
            "dynamic_range_db": round(st.dynamic_range(signal), 6),
        }
        return jsonify({"success": True, "features": features})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/fingerprint", methods=["POST"])
def api_fingerprint():
    body = request.get_json(force=True)
    signal = _parse_signal(body)
    sr = int(body.get("sample_rate", 2000))

    try:
        fp = st.fingerprint_engine(signal, sample_rate=sr)
        return jsonify({"success": True, "fingerprint": fp.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


def _dct(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        X[k] = np.sum(x * np.cos(np.pi * k * (2 * np.arange(N) + 1) / (2 * N)))
    if norm == "ortho":
        X[0] = X[0] / np.sqrt(N)
        X[1:] = X[1:] / np.sqrt(N / 2)
    return X


def _mel_filterbank(n_fft: int, sr: int, n_mels: int = 26, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left = int(fft_bins[i - 1])
        center = int(fft_bins[i])
        right = int(fft_bins[i + 1])
        if center > left:
            fbank[i - 1, left:center] = np.linspace(0, 1, center - left)
        if right > center:
            fbank[i - 1, center:right] = np.linspace(1, 0, right - center)
    return fbank


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + hz / 700)


def _chroma_filterbank(n_fft: int, sr: int, n_chroma: int = 12) -> np.ndarray:
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    chroma_bins = np.zeros((n_chroma, len(freqs)))
    for i, f in enumerate(freqs):
        if f > 0:
            pitch_class = int(np.round(12 * np.log2(f / 440.0))) % 12
            chroma_bins[pitch_class, i] = 1
    return chroma_bins


@app.route("/api/transforms", methods=["POST"])
def api_transforms():
    body = request.get_json(force=True)
    signal = _parse_signal(body)
    sr = int(body.get("sample_rate", 2000))

    try:
        s = np.asarray(signal, dtype=np.float64)
        result = {}

        # 1. Envelope (Hilbert)
        env = st.envelope(signal)
        result["envelope"] = [round(float(v), 8) for v in env]

        # 2. Cepstrum
        fft_spec = np.fft.fft(s)
        log_mag = np.log(np.abs(fft_spec) + 1e-10)
        cepstrum = np.fft.ifft(log_mag).real
        half = len(cepstrum) // 2
        result["cepstrum"] = {
            "quefrency": [round(float(i / sr), 6) for i in range(half)],
            "amplitude": [round(float(v), 8) for v in cepstrum[:half]],
        }

        # 3. Wavelet Packet (level 4, db4)
        try:
            wp = st.wavelet_packet.wavelet_packet_decompose(signal, level=4, family="db4")
            wd = wp.to_dict()
            nodes = {}
            energies = {}
            for key, coeffs in wd["nodes"].items():
                c = np.asarray(coeffs, dtype=np.float64)
                nodes[key] = [round(float(v), 8) for v in c[:32]]
                energies[key] = round(float(np.sum(c ** 2)), 8)
            result["wavelet"] = {
                "nodes": nodes,
                "energies": energies,
                "level": wd["level"],
                "meta": wd["meta"],
            }
        except Exception:
            result["wavelet"] = None

        # 4. Mel-spectrogram
        n_fft = 256
        hop = 64
        fbank = _mel_filterbank(n_fft, sr, n_mels=26)
        _, psd = st.power_spectral_density(signal, sample_rate=sr)
        psd_arr = np.asarray(psd)
        mel_spec = np.dot(fbank, psd_arr[:fbank.shape[1]])
        mel_spec_db = 10 * np.log10(mel_spec + 1e-10)
        result["mel_spectrogram"] = [round(float(v), 6) for v in mel_spec_db.tolist()]

        # 5. MFCC (13 coeffs from mel)
        n_mfcc = 13
        mfcc_raw = _dct(mel_spec_db)[:n_mfcc]
        result["mfcc"] = [round(float(v), 6) for v in mfcc_raw.tolist()]

        # 6. Chromagram (12 pitch classes)
        chroma_fbank = _chroma_filterbank(n_fft, sr)
        chroma = np.dot(chroma_fbank, psd_arr[:chroma_fbank.shape[1]])
        chroma_norm = chroma / (np.sum(chroma) + 1e-10)
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        result["chromagram"] = {
            "notes": notes,
            "values": [round(float(v), 6) for v in chroma_norm.tolist()],
        }

        return jsonify({"success": True, "transforms": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/spectrum", methods=["POST"])
def api_spectrum():
    body = request.get_json(force=True)
    signal = _parse_signal(body)
    sr = int(body.get("sample_rate", 2000))

    try:
        s = np.asarray(signal, dtype=np.float64)
        spectrum = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(len(s), d=1.0 / sr)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Peaks (top 10)
        indices = np.argsort(mag)[-30:][::-1]
        peaks = [{"freq": round(float(freqs[i]), 2), "mag": round(float(mag[i]), 6)} for i in indices if mag[i] > 1e-10][:10]

        # Centroid
        centroid_num = np.sum(freqs * mag)
        centroid_den = np.sum(mag)
        centroid_hz = float(centroid_num / centroid_den) if centroid_den > 0 else 0.0

        # Rolloff (85%)
        power = mag ** 2
        total_power = np.sum(power)
        cumsum = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * total_power)) if total_power > 0 else 0
        rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        return jsonify({
            "success": True,
            "spectrum": {
                "freqs": [round(f, 2) for f in freqs.tolist()],
                "magnitude": [round(m, 8) for m in mag.tolist()],
                "phase": [round(p, 6) for p in phase.tolist()],
                "peaks": peaks,
                "centroid_hz": round(centroid_hz, 2),
                "rolloff_hz": round(rolloff_hz, 2),
                "n_freqs": len(freqs),
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/neuro/analyze", methods=["POST"])
def api_neuro_analyze():
    body = request.get_json(force=True)
    mode = body.get("mode", "demo")

    try:
        if mode == "demo":
            shape = tuple(body.get("shape", [12, 10, 8]))
            fa_val = float(body.get("fa", 0.7))
            md_val = float(body.get("md", 0.0007))
            fiber_dir = tuple(body.get("fiber_dir", [0, 1, 0]))
            tensors = st.generate_tensor_volume(shape, fiber_dir, fa_val, md_val)
        else:
            tensor_data = body.get("tensors")
            if tensor_data is None:
                return jsonify({"success": False, "error": "No tensor data provided"}), 400
            tensors = np.asarray(tensor_data, dtype=np.float64)

        Z, Y, X = tensors.shape[:3]
        metrics = st.tensor_metrics(tensors)

        fa_map = np.array(metrics["fa_map"])
        color_fa = np.array(metrics["color_fa"])
        md_map = np.array(metrics["md_map"])
        cl_map = np.array(metrics["cl_map"])
        cp_map = np.array(metrics["cp_map"])
        cs_map = np.array(metrics["cs_map"])
        mo_map = np.array(metrics["mo_map"])
        vr_map = np.array(metrics["vr_map"])

        # Center slices for 2D visualization
        cz, cy, cx = Z // 2, Y // 2, X // 2

        # Tractography
        evals = np.array(metrics["eigenvalues"])
        evecs = np.array(metrics["eigenvectors"])
        seed_mask = fa_map > 0.3
        tract_result = st.track_streamlines(
            evals, evecs, fa_map, seed_mask,
            step_size=0.5, min_fa=0.2, max_angle=45,
            max_steps=300, n_seeds=200, probabilistic=True,
        )

        # Project streamlines to 2D for canvas rendering (axial view)
        projected = []
        for sl in tract_result.streamlines:
            pts = sl.points
            proj = [[float(p[1]), float(p[2])] for p in pts]  # YZ projection
            projected.append(proj)

        return jsonify({
            "success": True,
            "neuro": {
                "fa_slice": [round(float(v), 6) for row in fa_map[cz] for v in row],
                "md_slice": [round(float(v), 8) for row in md_map[cz] for v in row],
                "cl_slice": [round(float(v), 6) for row in cl_map[cz] for v in row],
                "cp_slice": [round(float(v), 6) for row in cp_map[cz] for v in row],
                "cs_slice": [round(float(v), 6) for row in cs_map[cz] for v in row],
                "color_fa_slice": [
                    [round(float(v), 6) for v in row]
                    for row in color_fa[cz].reshape(-1, 3).tolist()
                ],
                "shape": [Z, Y, X],
                "streamlines": projected,
                "n_streamlines": len(projected),
                "fa_range": [round(float(fa_map.min()), 4), round(float(fa_map.max()), 4)],
                "md_range": [round(float(md_map.min()), 6), round(float(md_map.max()), 6)],
                "fiber_stats": tract_result.fiber_stats,
                "connectivity_matrix": tract_result.connectivity_matrix,
                "connectivity_labels": tract_result.connectivity_labels,
                "clusters": tract_result.clusters,
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ====== WiFi Analysis ======
WIFI_BANDS = {
    "2.4GHz": {"channels": {1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432, 6: 2437,
                            7: 2442, 8: 2447, 9: 2452, 10: 2457, 11: 2462, 12: 2467, 13: 2472},
                "width": 20, "range": (2400, 2485)},
    "5GHz": {"channels": {36: 5180, 40: 5200, 44: 5220, 48: 5240, 52: 5260, 56: 5280,
                          60: 5300, 64: 5320, 100: 5500, 104: 5520, 108: 5540, 112: 5560,
                          116: 5580, 120: 5600, 124: 5620, 128: 5640, 132: 5660, 136: 5680,
                          140: 5700, 149: 5745, 153: 5765, 157: 5785, 161: 5805, 165: 5825},
              "width": 20, "range": (5150, 5850)},
}


def _wifi_channel_scan(num_channels: int = 13, noise_floor: float = -95,
                       num_networks: int = 5) -> list:
    rng = np.random.default_rng(42)
    channels = list(range(1, num_channels + 1))
    result = []
    networks = []
    for i in range(num_networks):
        ch = int(rng.choice(channels))
        rssi = float(rng.uniform(-65, -35))
        bw = int(rng.choice([20, 40, 80]))
        networks.append({"channel": ch, "rssi": rssi, "bw": bw, "ssid": f"Red_{i + 1}"})

    for ch in channels:
        sig_power = noise_floor
        active_networks = []
        for net in networks:
            if abs(net["channel"] - ch) <= net["bw"] / 20:
                sig_power = max(sig_power, net["rssi"] - abs(net["channel"] - ch) * 1.5)
                active_networks.append(net["ssid"])
        rssi = float(rng.uniform(sig_power - 2, sig_power + 2))
        snr = rssi - noise_floor
        result.append({
            "channel": ch,
            "freq_mhz": 2412 + (ch - 1) * 5,
            "rssi_dbm": round(rssi, 1),
            "noise_floor_dbm": noise_floor,
            "snr_db": round(snr, 1),
            "utilization_pct": round(float(rng.uniform(0, 100)), 1),
            "networks": active_networks,
        })
    return result


def _wifi_spectrum(sample_rate: float = 20000, duration: float = 1.0,
                   networks: list | None = None) -> dict:
    rng = np.random.default_rng(42)
    t = np.arange(0, duration, 1.0 / sample_rate)
    signal = np.zeros_like(t)

    if networks is None:
        networks = [
            {"channel": 6, "rssi": -40, "bw": 20},
            {"channel": 1, "rssi": -55, "bw": 20},
            {"channel": 11, "rssi": -50, "bw": 40},
        ]

    for net in networks:
        center_hz = (2412 + (net["channel"] - 1) * 5) * 1e6
        # Normalize to simulation bandwidth
        norm_freq = 0.2 + (net["channel"] - 1) * 0.05
        bw_norm = net["bw"] / 200
        amp = 10 ** ((net["rssi"] + 95) / 20)
        for _ in range(3):
            carrier = norm_freq + rng.uniform(-bw_norm / 2, bw_norm / 2)
            sig_part = amp * np.sin(2 * np.pi * carrier * t * sample_rate)
            signal += sig_part

    signal += rng.normal(0, 0.01, size=len(t))

    # Compute spectrum
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    max_freq_idx = int(len(freqs) * 0.4)
    freqs = freqs[:max_freq_idx]
    spectrum = spectrum[:max_freq_idx]

    return {
        "freqs": [round(f * 100, 2) for f in freqs],
        "magnitude": [round(float(m), 6) for m in spectrum],
    }


@app.route("/wifi")
def wifi_page():
    return send_from_directory(app.static_folder, "wifi.html")


@app.route("/api/wifi/scan", methods=["POST"])
def api_wifi_scan():
    body = request.get_json(force=True)
    band = body.get("band", "2.4GHz")
    noise_floor = float(body.get("noise_floor", -95))
    num_networks = int(body.get("num_networks", 5))

    try:
        channels = _wifi_channel_scan(
            num_channels=len(WIFI_BANDS[band]["channels"]),
            noise_floor=noise_floor,
            num_networks=num_networks,
        )
        return jsonify({"success": True, "channels": channels, "band": band})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/wifi/spectrum", methods=["POST"])
def api_wifi_spectrum():
    body = request.get_json(force=True)
    networks = body.get("networks")

    try:
        spec = _wifi_spectrum(networks=networks)
        return jsonify({"success": True, "spectrum": spec})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/wifi/analyze", methods=["POST"])
def api_wifi_analyze():
    body = request.get_json(force=True)
    band = body.get("band", "2.4GHz")
    signal_data = body.get("signal")

    try:
        scan = _wifi_channel_scan(
            num_channels=len(WIFI_BANDS[band]["channels"]),
            noise_floor=float(body.get("noise_floor", -95)),
            num_networks=int(body.get("num_networks", 5)),
        )

        channel_count = len(scan)
        rssi_vals = [c["rssi_dbm"] for c in scan]
        snr_vals = [c["snr_db"] for c in scan]
        util_vals = [c["utilization_pct"] for c in scan]

        best_ch = max(scan, key=lambda c: c["snr_db"])
        worst_ch = min(scan, key=lambda c: c["snr_db"])

        interference = 0
        for c in scan:
            if len(c["networks"]) > 1:
                interference += 1
        interference_pct = round(interference / channel_count * 100, 1) if channel_count else 0

        spec = _wifi_spectrum(networks=[
            {"channel": int(body.get("channel", 6)), "rssi": -45, "bw": 20},
            {"channel": max(1, int(body.get("channel", 6)) - 5), "rssi": -60, "bw": 20},
        ])

        return jsonify({
            "success": True,
            "scan": scan,
            "spectrum": spec,
            "analysis": {
                "channel_count": channel_count,
                "avg_rssi": round(float(np.mean(rssi_vals)), 1),
                "avg_snr": round(float(np.mean(snr_vals)), 1),
                "max_snr": round(float(max(snr_vals)), 1),
                "min_snr": round(float(min(snr_vals)), 1),
                "best_channel": best_ch["channel"],
                "best_snr": best_ch["snr_db"],
                "worst_channel": worst_ch["channel"],
                "worst_snr": worst_ch["snr_db"],
                "interference_pct": interference_pct,
                "avg_utilization": round(float(np.mean(util_vals)), 1),
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ====== PCAP Analysis ======
def _parse_pcap_beacons(pcap_path: str) -> list:
    from scapy.all import rdpcap, RadioTap, Dot11, Dot11Beacon, Dot11Elt
    packets = rdpcap(pcap_path)
    seen = {}
    for pkt in packets:
        if not pkt.haslayer(Dot11Beacon):
            continue
        bssid = pkt[Dot11].addr3.lower()
        if bssid in seen:
            continue

        ssid = ""
        channel = 0
        elt = pkt.getlayer(Dot11Elt)
        while isinstance(elt, Dot11Elt):
            if elt.ID == 0:
                try:
                    ssid = elt.info.decode("utf-8", errors="replace")
                except Exception:
                    ssid = elt.info.hex()
            elif elt.ID == 3 and len(elt.info) > 0:
                channel = int(elt.info[0])
            elt = elt.payload

        rssi = None
        noise = None
        if pkt.haslayer(RadioTap):
            rt = pkt[RadioTap]
            rt_sig = getattr(rt, "dBm_AntSignal", None)
            rt_noise = getattr(rt, "dBm_AntNoise", None)
            if rt_sig is not None:
                rssi = int(rt_sig)
            if rt_noise is not None:
                noise = int(rt_noise)

        freq = 2412 + (channel - 1) * 5 if 1 <= channel <= 13 else 0
        if freq == 0 and channel > 13:
            # 5 GHz channels
            ch_map = {36: 5180, 40: 5200, 44: 5220, 48: 5240, 52: 5260, 56: 5280,
                      60: 5300, 64: 5320, 100: 5500, 104: 5520, 108: 5540, 112: 5560,
                      116: 5580, 120: 5600, 124: 5620, 128: 5640, 132: 5660, 136: 5680,
                      140: 5700, 149: 5745, 153: 5765, 157: 5785, 161: 5805, 165: 5825}
            freq = ch_map.get(channel, 0)

        seen[bssid] = {
            "bssid": bssid,
            "ssid": ssid if ssid else "(oculto)",
            "channel": channel,
            "freq_mhz": freq,
            "rssi_dbm": rssi,
            "noise_dbm": noise,
            "snr_db": round(rssi - noise, 1) if (rssi is not None and noise is not None) else None,
        }
    return list(seen.values())


def _pcap_to_scan_results(networks: list) -> dict:
    channels_map = {}
    for net in networks:
        ch = net["channel"]
        if ch not in channels_map:
            channels_map[ch] = {"channel": ch, "freq_mhz": net["freq_mhz"],
                                "rssi_dbm": net["rssi_dbm"],
                                "noise_floor_dbm": net["noise_dbm"],
                                "snr_db": net["snr_db"],
                                "networks": []}
        channels_map[ch]["networks"].append(net["ssid"])
        # Keep strongest RSSI
        if net["rssi_dbm"] is not None:
            cur = channels_map[ch]["rssi_dbm"]
            if cur is None or net["rssi_dbm"] > cur:
                channels_map[ch]["rssi_dbm"] = net["rssi_dbm"]
                channels_map[ch]["snr_db"] = net["snr_db"]

    scan = sorted(channels_map.values(), key=lambda c: c["channel"])
    rssi_vals = [c["rssi_dbm"] for c in scan if c["rssi_dbm"] is not None]
    snr_vals = [c["snr_db"] for c in scan if c["snr_db"] is not None]

    channel_count = len(scan)
    best = max(scan, key=lambda c: c["snr_db"] if c["snr_db"] is not None else (c["rssi_dbm"] or -999)) if scan else None
    worst = min(scan, key=lambda c: c["snr_db"] if c["snr_db"] is not None else (c["rssi_dbm"] or 999)) if scan else None
    interference = sum(1 for c in scan if len(c["networks"]) > 1)

    return {
        "scan": scan,
        "analysis": {
            "channel_count": channel_count,
            "avg_rssi": round(float(np.mean(rssi_vals)), 1) if rssi_vals else None,
            "avg_snr": round(float(np.mean(snr_vals)), 1) if snr_vals else None,
            "max_snr": round(float(max(snr_vals)), 1) if snr_vals else None,
            "min_snr": round(float(min(snr_vals)), 1) if snr_vals else None,
            "best_channel": best["channel"] if best else None,
            "best_snr": best["snr_db"] if best is not None and best["snr_db"] is not None else round(float(best["rssi_dbm"] or 0), 1) if best else None,
            "worst_channel": worst["channel"] if worst else None,
            "worst_snr": worst["snr_db"] if worst is not None and worst["snr_db"] is not None else round(float(worst["rssi_dbm"] or 0), 1) if worst else None,
            "interference_pct": round(interference / channel_count * 100, 1) if channel_count else 0,
            "networks_found": len(networks),
        },
    }


@app.route("/api/wifi/analyze-pcap", methods=["POST"])
def api_wifi_analyze_pcap():
    import tempfile, os

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No se envió ningún archivo"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "Archivo vacío"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
    try:
        f.save(tmp.name)
        tmp.close()
        networks = _parse_pcap_beacons(tmp.name)
        if not networks:
            return jsonify({"success": False,
                            "error": "No se encontraron tramas Beacon en el archivo. "
                                     "Asegúrate de capturar en modo monitor."}), 400
        result = _pcap_to_scan_results(networks)
        return jsonify({"success": True, **result, "networks": networks})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error al analizar pcap: {str(e)}"}), 400
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    import os, subprocess, signal, time, sys as _sys
    port = int(os.environ.get("PORT", 8080))

    # Kill any previous instance on this port
    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                pid = line.strip()
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"  Cerrando PID {pid}")
                        time.sleep(0.5)
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        try:
            result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                for pid in result.stdout.strip().split():
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"  Cerrando PID {pid}")
                        time.sleep(0.5)
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    print("=" * 55)
    print("  SignalTools API Server")
    print("  " + "-" * 35)
    print(f"  URL:    http://localhost:{port}")
    print(f"  API:    http://localhost:{port}/api/")
    print("=" * 55)

    try:
        import gunicorn
        print(f"  Server:  gunicorn ({gunicorn.__version__})")
        print("=" * 55)
        _sys.argv = [
            "gunicorn",
            "-w", "2",
            "-b", f"0.0.0.0:{port}",
            "--timeout", "60",
            "--keep-alive", "5",
            "web.server:app",
        ]
        from gunicorn.app.wsgiapp import WSGIApplication
        WSGIApplication().run()
    except ImportError:
        print("  [!] gunicorn no disponible, usando servidor Flask...")
        print("  [!] Si hay errores de conexión: pip install gunicorn")
        print("=" * 55)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
