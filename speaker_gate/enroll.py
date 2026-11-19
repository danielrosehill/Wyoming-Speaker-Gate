"""Enroll a speaker: average WeSpeaker embeddings of 1.5 s windows over the voiced parts of one or more recordings.

    python -m speaker_gate.enroll --voiceprints data/voiceprints.npz --name Daniel --audio daniel.wav [--audio more.wav] [--replace]

Record on the satellite's own microphone: a voiceprint from a different mic scores much lower (see README).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .embed import SR, Embedder, unit, voiced


def load16k(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voiceprints", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--audio", action="append", required=True)
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--step", type=float, default=0.75)
    ap.add_argument("--replace", action="store_true", help="drop this speaker's existing prints first")
    a = ap.parse_args()

    emb = Embedder()
    k, s = int(a.window * SR), int(a.step * SR)
    vecs = []
    for p in a.audio:
        y = voiced(load16k(p))
        vecs += [emb(y[i:i + k]) for i in range(0, len(y) - k + 1, s)]
    if len(vecs) < 10:
        raise SystemExit(f"only {len(vecs)} windows of speech; record at least 30 s")
    centroid = unit(np.mean(vecs, axis=0))

    names, cents = [], []
    if a.voiceprints.exists():
        d = np.load(a.voiceprints)
        for n, c in zip(d["names"], d["centroids"]):
            if a.replace and str(n) == a.name:
                continue
            names.append(str(n)); cents.append(c)
    names.append(a.name); cents.append(centroid)
    a.voiceprints.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.voiceprints, names=np.array(names), centroids=np.array(cents))
    print(f"enrolled {a.name}: {len(vecs)} windows -> {a.voiceprints} ({len(names)} prints)")


if __name__ == "__main__":
    main()
