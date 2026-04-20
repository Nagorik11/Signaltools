from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import struct
import subprocess
import wave
from typing import Iterator

import numpy as np


@dataclass
class SignalBuffer:
    path: str
    raw: bytes

    @property
    def size(self) -> int:
        return len(self.raw)

    def hex_preview(self, n: int = 64) -> str:
        return self.raw[:n].hex(" ")


def read_signal_file(path: str | Path) -> SignalBuffer:
    p = Path(path)
    return SignalBuffer(path=str(p), raw=p.read_bytes())


def iter_unpack(raw: bytes, fmt: str) -> Iterator[tuple]:
    size = struct.calcsize(fmt)
    usable = len(raw) - (len(raw) % size)
    for i in range(0, usable, size):
        yield struct.unpack(fmt, raw[i:i + size])


def decode_int16_le(raw: bytes) -> list[int]:
    return [x[0] for x in iter_unpack(raw, "<h")]


def decode_uint16_le(raw: bytes) -> list[int]:
    return [x[0] for x in iter_unpack(raw, "<H")]


def decode_float32_le(raw: bytes) -> list[float]:
    usable = raw[: len(raw) - (len(raw) % 4)]
    return [x[0] for x in iter_unpack(usable, "<f")]


def decode_uint8(raw: bytes) -> list[int]:
    return list(raw)


def guess_numeric_views(raw: bytes) -> dict[str, list[float] | list[int]]:
    return {
        "uint8": decode_uint8(raw),
        "int16_le": decode_int16_le(raw),
        "uint16_le": decode_uint16_le(raw),
        "float32_le": decode_float32_le(raw),
    }


def write_wav(path: str | Path, signal: list[float] | list[int], sample_rate: int = 44100) -> None:
    p = Path(path)
    s = [float(x) for x in signal]
    peak = max((abs(x) for x in s), default=1.0) or 1.0
    pcm = [int(max(min(x / peak * 32767.0, 32767.0), -32768.0)) for x in s]
    with wave.open(str(p), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


def read_wav(path: str | Path) -> list[float]:
    p = Path(path)
    with wave.open(str(p), "rb") as f:
        raw = f.readframes(f.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return (data / 32767.0).tolist()


def read_audio_file(path: str | Path, sample_rate: int = 44100, filters: str | None = None) -> SignalBuffer:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    cmd = ["ffmpeg", "-i", str(p)]
    if filters:
        cmd.extend(["-af", filters])
    cmd.extend(["-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-loglevel", "quiet", "pipe:1"])

    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return SignalBuffer(path=str(p), raw=result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to decode {p}: {e.stderr.decode(errors='ignore')}") from e
    except FileNotFoundError as e:
        raise RuntimeError("FFmpeg not found. Please install it to support compressed audio files.") from e


class Ingestor:
    @staticmethod
    def from_wav(file_path: str) -> np.ndarray:
        return np.array(read_wav(file_path), dtype=np.float32)

    @staticmethod
    def from_text(file_path: str, encoding: str = "utf-8") -> np.ndarray:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            text = f.read()
        if not text:
            raise ValueError("The text file is empty.")
        return np.array([ord(char) for char in text], dtype=np.float32)

    @staticmethod
    def from_json(file_path: str, key: str = "samples") -> np.ndarray:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return np.array(data, dtype=np.float32)
        if isinstance(data, dict):
            if key in data:
                return np.array(data[key], dtype=np.float32)
            for alt in ("signal", "data", "values", "samples"):
                if alt in data and isinstance(data[alt], list):
                    return np.array(data[alt], dtype=np.float32)
        raise ValueError(f"No signal list found in JSON. Expected key: {key!r} or common alternatives like 'signal'.")

    @staticmethod
    def from_pcap(file_path: str, feature: str = "size") -> np.ndarray:
        try:
            from scapy.all import rdpcap
        except ImportError as e:
            raise ImportError("scapy is required for PCAP ingestion. Install it with: pip install scapy") from e

        packets = rdpcap(file_path)
        if not packets:
            raise ValueError("The PCAP file contains no packets.")
        if feature == "size":
            signal = [len(p) for p in packets]
        elif feature == "time":
            timestamps = [float(p.time) for p in packets]
            signal = np.diff(timestamps).tolist()
        else:
            raise ValueError("Unsupported PCAP feature. Use 'size' or 'time'.")
        return np.array(signal, dtype=np.float32)

    @staticmethod
    def from_video(file_path: str) -> np.ndarray:
        try:
            import cv2
        except ImportError as e:
            raise ImportError("opencv-python is required for video ingestion. Install it with: pip install opencv-python") from e

        cap = cv2.VideoCapture(file_path)
        signal: list[float] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            signal.append(float(np.mean(gray)))
        cap.release()
        if not signal:
            raise ValueError("No frames could be extracted from the video.")
        return np.array(signal, dtype=np.float32)


__all__ = [
    "SignalBuffer",
    "Ingestor",
    "read_signal_file",
    "guess_numeric_views",
    "write_wav",
    "read_wav",
    "read_audio_file",
    "decode_int16_le",
    "decode_uint16_le",
    "decode_float32_le",
    "decode_uint8",
]
