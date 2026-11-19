"""Speaker embeddings (WeSpeaker ResNet34 via pyannote.audio) and voiceprint helpers."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SR = 16000
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM")


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def voiced(y: np.ndarray, top_db: float = 30.0, frame: int = 2048, hop: int = 512) -> np.ndarray:
    """Keep frames within top_db of the loudest frame (same rule as librosa.effects.split)."""
    if len(y) < frame:
        return y
    n = 1 + (len(y) - frame) // hop
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame] ** 2)) for i in range(n)])
    db = 20 * np.log10(rms + 1e-10)
    keep = db >= db.max() - top_db
    out = [y[i * hop:i * hop + hop] for i in range(n) if keep[i]]
    return np.concatenate(out) if out else y


class Embedder:
    def __init__(self) -> None:
        import torch
        from pyannote.audio import Inference, Model

        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
        self._torch = torch
        model = Model.from_pretrained(EMBEDDING_MODEL, token=os.environ.get("HF_TOKEN") or None)
        self._inf = Inference(model, window="whole")

    def __call__(self, y: np.ndarray) -> np.ndarray:
        wav = self._torch.from_numpy(np.asarray(y, dtype=np.float32)[None])
        return unit(self._inf({"waveform": wav, "sample_rate": SR}))


def load_voiceprints(path: Path) -> tuple[list[str], np.ndarray]:
    """npz with `names` (N,) and `centroids` (N, D). A name may appear more than once (several prints per person)."""
    d = np.load(path)
    names = [str(n) for n in d["names"]]
    cents = np.array([unit(c) for c in d["centroids"]])
    return names, cents
