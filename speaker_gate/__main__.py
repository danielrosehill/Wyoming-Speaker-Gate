"""Wyoming STT proxy that forwards audio to an upstream STT and returns its transcript only if the
speaker matches an enrolled voiceprint. Otherwise it returns an empty transcript, so the voice pipeline
ends without acting (Home Assistant: stt-no-text-recognized)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .embed import SR, Embedder, load_voiceprints, voiced

_LOG = logging.getLogger("speaker_gate")


class Gate:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.embedder = Embedder()
        self.names, self.cents = load_voiceprints(args.voiceprints)
        self.thresholds = {n: args.threshold for n in self.names}
        for spec in args.speaker_threshold or []:
            n, t = spec.split("=", 1)
            self.thresholds[n] = float(t)
        self.info: Info | None = None
        self.lock = asyncio.Lock()
        _LOG.info("voiceprints %s, thresholds %s, mode %s", self.names, self.thresholds, args.mode)

    def decide(self, audio: np.ndarray) -> dict:
        y = voiced(audio)
        dur = len(y) / SR
        if dur < self.args.min_speech:
            return {"speech_s": round(dur, 2), "speaker": None, "score": None, "scores": {}, "pass": self.args.allow_short}
        e = self.embedder(y)
        raw = self.cents @ e
        scores: dict[str, float] = {}
        for n, s in zip(self.names, raw):
            scores[n] = max(scores.get(n, -1.0), float(s))
        best = max(scores, key=lambda n: scores[n] - self.thresholds[n])
        ok = scores[best] >= self.thresholds[best]
        return {"speech_s": round(dur, 2), "speaker": best if ok else None, "best": best,
                "score": round(scores[best], 3), "scores": {k: round(v, 3) for k, v in scores.items()}, "pass": ok}


class Handler(AsyncEventHandler):
    def __init__(self, gate: Gate, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.gate = gate
        self.client: AsyncClient | None = None
        self.conv = AudioChunkConverter(rate=SR, width=2, channels=1)
        self.buf = bytearray()
        self.t0 = 0.0

    async def _upstream_info(self) -> Info:
        async with self.gate.lock:
            if self.gate.info is None:
                async with AsyncClient.from_uri(self.gate.args.upstream) as c:
                    await c.write_event(Describe().event())
                    while True:
                        ev = await c.read_event()
                        if ev is None:
                            raise ConnectionError("upstream closed during describe")
                        if Info.is_type(ev.type):
                            info = Info.from_event(ev)
                            break
                for prog in info.asr:
                    prog.name = f"{self.gate.args.name} ({prog.name})"
                    prog.description = f"Speaker-gated: {prog.description or prog.name}"
                self.gate.info = info
            return self.gate.info

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event((await self._upstream_info()).event())
            return True
        if Transcribe.is_type(event.type):
            self.client = AsyncClient.from_uri(self.gate.args.upstream)
            await self.client.connect()
            await self.client.write_event(event)
            return True
        if AudioStart.is_type(event.type):
            if self.client is None:  # no Transcribe sent first
                self.client = AsyncClient.from_uri(self.gate.args.upstream)
                await self.client.connect()
                await self.client.write_event(Transcribe().event())
            self.buf.clear(); self.t0 = time.monotonic()
            await self.client.write_event(event)
            return True
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.buf += self.conv.convert(chunk).audio
            if self.client:
                await self.client.write_event(event)
            return True
        if AudioStop.is_type(event.type):
            audio = np.frombuffer(bytes(self.buf), dtype=np.int16).astype(np.float32) / 32768.0
            loop = asyncio.get_running_loop()
            decision_f = loop.run_in_executor(None, self.gate.decide, audio)
            text = ""
            if self.client:
                await self.client.write_event(event)
                while True:
                    ev = await self.client.read_event()
                    if ev is None:
                        break
                    if Transcript.is_type(ev.type):
                        text = Transcript.from_event(ev).text
                        break
                await self.client.disconnect(); self.client = None
            d = await decision_f
            enforce = self.gate.args.mode == "enforce"
            out = text if (d["pass"] or not enforce) else ""
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "audio_s": round(len(audio) / SR, 2),
                   "peak_dbfs": round(20 * np.log10(peak + 1e-9), 1), "rms_dbfs": round(20 * np.log10(rms + 1e-9), 1), **d,
                   "mode": self.gate.args.mode, "latency_s": round(time.monotonic() - self.t0, 2)}
            if self.gate.args.log_text:
                rec["text"] = text
            _LOG.info(json.dumps(rec))
            if self.gate.args.log_file:
                with open(self.gate.args.log_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            await self.write_event(Transcript(text=out).event())
            return False
        return True

    async def disconnect(self) -> None:
        if self.client:
            await self.client.disconnect()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", default="tcp://0.0.0.0:10310")
    ap.add_argument("--upstream", required=True, help="upstream Wyoming STT, e.g. tcp://10.0.0.2:10300")
    ap.add_argument("--voiceprints", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.40, help="cosine needed to accept (default per speaker)")
    ap.add_argument("--speaker-threshold", action="append", metavar="NAME=T", help="per-speaker override")
    ap.add_argument("--min-speech", type=float, default=0.6, help="seconds of voiced audio needed to judge")
    ap.add_argument("--allow-short", action="store_true", help="pass utterances too short to judge")
    ap.add_argument("--mode", choices=["enforce", "shadow"], default="enforce", help="shadow = log only, never block")
    ap.add_argument("--name", default="speaker-gate")
    ap.add_argument("--log-file", type=Path)
    ap.add_argument("--log-text", action="store_true", help="include transcripts in the log")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gate = Gate(args)
    server = AsyncServer.from_uri(args.uri)
    _LOG.info("listening on %s, upstream %s", args.uri, args.upstream)
    await server.run(partial(Handler, gate))


if __name__ == "__main__":
    asyncio.run(main())
