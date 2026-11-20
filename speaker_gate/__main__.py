"""Wyoming STT proxy that forwards audio to an upstream STT and returns its transcript only if the
speaker matches an enrolled voiceprint. Otherwise it returns an empty transcript, so the voice pipeline
ends without acting (Home Assistant: stt-no-text-recognized)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import time
from datetime import date, timedelta
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
        key = os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter = None
        if args.stt_primary == "openrouter":
            if key:
                self.openrouter = OpenRouterSTT(key, args.openrouter_model,
                                                [p for p in args.openrouter_providers.split(",") if p.strip()],
                                                args.stt_language or None, args.stt_prompt_file)
            else:
                _LOG.warning("STT_PRIMARY=openrouter but OPENROUTER_API_KEY is empty: using the upstream only")
        self.recorder = (Recorder(args.record_dir, args.record_retention_days)
                         if args.record_dir and args.record else None)
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


def _env_on(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip().lower() not in ("0", "false", "no", "off")


class Recorder:
    """Keeps each command's audio as 16 kHz mono WAV plus a sidecar JSON (decision, scores, transcript) in
    <dir>/YYYY-MM-DD/<YYYYMMDDTHHMMSS-mmm>_<speaker|unknown>.wav/.json; day folders older than retention_days
    are deleted on startup and on each write."""

    def __init__(self, root: Path, retention_days: int) -> None:
        self.root, self.retention_days = root, retention_days
        self._pruned_on: date | None = None
        _LOG.info("recording commands to %s, retention %d days", root, retention_days)
        self.prune()

    def prune(self) -> None:
        today = date.today()
        if self._pruned_on == today:
            return
        cutoff = today - timedelta(days=self.retention_days)
        if self.root.is_dir():
            for d in self.root.iterdir():
                try:
                    day = date.fromisoformat(d.name)
                except ValueError:
                    continue  # not one of ours
                if d.is_dir() and day < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    _LOG.info("pruned recordings %s", d)
        self._pruned_on = today

    def path_for(self, when: float, speaker: str | None) -> Path:
        lt = time.localtime(when)
        stem = time.strftime("%Y%m%dT%H%M%S", lt) + f"-{int(when * 1000) % 1000:03d}"
        who = re.sub(r"[^A-Za-z0-9_-]+", "-", speaker) if speaker else "unknown"
        return self.root / time.strftime("%Y-%m-%d", lt) / f"{stem}_{who}.wav"

    def write(self, wav: Path, audio: np.ndarray, meta: dict) -> None:
        import soundfile as sf
        try:
            self.prune()
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, audio, SR, subtype="PCM_16")
            wav.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
        except Exception as e:  # noqa: BLE001 - recording must never break the pipeline
            _LOG.warning("recording failed: %r", e)


class OpenRouterSTT:
    """Whisper via OpenRouter's /audio/transcriptions (JSON body, base64 WAV). The vocabulary prompt is re-read from
    prompt_file on every call, so it can be edited without a restart."""

    URL = "https://openrouter.ai/api/v1/audio/transcriptions"

    def __init__(self, key: str, model: str, providers: list[str], language: str | None,
                 prompt_file: Path | None) -> None:
        self.key, self.model, self.providers, self.language, self.prompt_file = key, model, providers, language, prompt_file
        _LOG.info("primary STT: OpenRouter %s via %s, language %s, prompt file %s", model, providers or "any provider",
                  language, prompt_file)

    def prompt(self) -> str:
        try:
            if self.prompt_file and self.prompt_file.is_file():
                lines = [ln.strip() for ln in self.prompt_file.read_text().splitlines()]
                return " ".join(ln for ln in lines if ln and not ln.startswith("#"))
        except OSError as e:
            _LOG.warning("prompt file unreadable: %r", e)
        return ""

    def transcribe(self, audio: np.ndarray) -> str:
        import base64
        import io
        import urllib.request

        import soundfile as sf
        b = io.BytesIO()
        sf.write(b, audio, SR, subtype="PCM_16", format="WAV")
        body: dict = {"model": self.model, "input_audio": {"data": base64.b64encode(b.getvalue()).decode(), "format": "wav"}}
        if self.language:
            body["language"] = self.language
        if pr := self.prompt():
            body["prompt"] = pr
        if self.providers:
            body["provider"] = {"order": self.providers}  # fallbacks to other providers stay allowed
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return (json.load(r).get("text") or "").strip()


def post_speaker(url: str, payload: dict) -> None:
    """POST the decision as JSON; best effort, 1.5 s timeout, never blocks the transcript for long."""
    import urllib.request
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=1.5).close()
    except Exception as e:  # noqa: BLE001
        _LOG.warning("speaker webhook failed: %r", e)


class Handler(AsyncEventHandler):
    def __init__(self, gate: Gate, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.gate = gate
        self.client: AsyncClient | None = None
        self.conv = AudioChunkConverter(rate=SR, width=2, channels=1)
        self.buf = bytearray()
        self.t0 = 0.0
        self.wall0 = 0.0
        self.up_err: str | None = None

    async def _up(self, coro) -> bool:
        """Run one upstream step with the upstream timeout. On failure drop the upstream connection for the rest
        of this command (the command goes on without the upstream) and remember why."""
        try:
            await asyncio.wait_for(coro, self.gate.args.upstream_timeout)
            return True
        except Exception as e:  # noqa: BLE001
            self.up_err = "timeout" if isinstance(e, asyncio.TimeoutError) else repr(e)
            _LOG.warning("upstream STT failed: %s", self.up_err)
            await self._drop_upstream()
            return False

    async def _drop_upstream(self) -> None:
        c, self.client = self.client, None
        if c is not None:
            try:
                await asyncio.wait_for(c.disconnect(), 1.0)
            except Exception:  # noqa: BLE001
                pass

    async def _connect_upstream(self, first: Event) -> None:
        self.client = AsyncClient.from_uri(self.gate.args.upstream)
        if await self._up(self.client.connect()):
            await self._up(self.client.write_event(first))

    async def _upstream_info(self) -> Info:
        async with self.gate.lock:
            if self.gate.info is None:
                async def _describe() -> Info:
                    async with AsyncClient.from_uri(self.gate.args.upstream) as c:
                        await c.write_event(Describe().event())
                        while True:
                            ev = await c.read_event()
                            if ev is None:
                                raise ConnectionError("upstream closed during describe")
                            if Info.is_type(ev.type):
                                return Info.from_event(ev)
                info = await asyncio.wait_for(_describe(), self.gate.args.upstream_timeout)
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
            self.up_err = None
            await self._connect_upstream(event)
            return True
        if AudioStart.is_type(event.type):
            if self.client is None and self.up_err is None:  # no Transcribe sent first
                await self._connect_upstream(Transcribe().event())
            self.buf.clear(); self.t0 = time.monotonic(); self.wall0 = time.time()
            if self.client:
                await self._up(self.client.write_event(event))
            return True
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.buf += self.conv.convert(chunk).audio
            if self.client:
                await self._up(self.client.write_event(event))
            return True
        if AudioStop.is_type(event.type):
            audio = np.frombuffer(bytes(self.buf), dtype=np.int16).astype(np.float32) / 32768.0
            loop = asyncio.get_running_loop()
            decision_f = loop.run_in_executor(None, self.gate.decide, audio)
            stt: dict = {}
            or_task = None
            if self.gate.openrouter and len(audio):
                async def _or() -> str:
                    t = time.monotonic()
                    try:
                        return await asyncio.wait_for(loop.run_in_executor(None, self.gate.openrouter.transcribe, audio),
                                                      self.gate.args.stt_timeout)
                    finally:
                        stt["stt_ms_openrouter"] = round((time.monotonic() - t) * 1000)
                or_task = asyncio.ensure_future(_or())
            up_text = ""
            if self.client:
                t_up = time.monotonic()
                got: list[str] = []

                async def _finish() -> None:
                    await self.client.write_event(event)
                    while True:
                        ev = await self.client.read_event()
                        if ev is None:
                            raise ConnectionError("upstream closed without a transcript")
                        if Transcript.is_type(ev.type):
                            got.append(Transcript.from_event(ev).text)
                            return
                if await self._up(_finish()):
                    up_text = got[0]
                stt["stt_ms_upstream"] = round((time.monotonic() - t_up) * 1000)
                await self._drop_upstream()
            if self.up_err:
                stt["upstream_error"] = self.up_err
            self.up_err = None
            text, stt["stt_used"] = up_text, ("none" if "upstream_error" in stt else "upstream")
            if or_task is not None:
                stt["text_upstream"] = up_text
                try:
                    stt["text_openrouter"] = await or_task
                    text, stt["stt_used"] = stt["text_openrouter"], "openrouter"
                except Exception as e:  # noqa: BLE001 - timeout or HTTP error: fall back to the upstream
                    _LOG.warning("OpenRouter STT failed, using upstream: %r", e)
                    stt["openrouter_error"] = repr(e) if not isinstance(e, asyncio.TimeoutError) else "timeout"
                    if "upstream_error" in stt:
                        _LOG.warning("both STT engines failed: returning an empty transcript")
            stt["stt_primary"] = "openrouter" if self.gate.openrouter else "upstream"
            d = await decision_f
            enforce = self.gate.args.mode == "enforce"
            out = text if (d["pass"] or not enforce) else ""
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "audio_s": round(len(audio) / SR, 2),
                   "peak_dbfs": round(20 * np.log10(peak + 1e-9), 1), "rms_dbfs": round(20 * np.log10(rms + 1e-9), 1), **d,
                   "mode": self.gate.args.mode, "latency_s": round(time.monotonic() - self.t0, 2), **stt}
            if self.gate.args.log_text:
                rec["text"] = text
            else:
                rec.pop("text_upstream", None); rec.pop("text_openrouter", None)
            if self.gate.recorder:
                wav = self.gate.recorder.path_for(self.wall0 or time.time(), d.get("speaker"))
                rec["audio_file"] = str(wav.relative_to(self.gate.recorder.root))
                meta = {**rec, **stt, "text": text, "returned_text": out, "sample_rate": SR}
                await loop.run_in_executor(None, self.gate.recorder.write, wav, audio, meta)
            _LOG.info(json.dumps(rec))
            if self.gate.args.log_file:
                with open(self.gate.args.log_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if self.gate.args.speaker_webhook:
                # tell the consumer (e.g. a Home Assistant webhook automation) who spoke, before it gets the
                # transcript, so its conversation agent can address the speaker by name
                await loop.run_in_executor(None, post_speaker, self.gate.args.speaker_webhook,
                                           {k: d.get(k) for k in ("speaker", "best", "score", "pass")})
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
    ap.add_argument("--record-dir", "--save-audio", type=Path, default=os.environ.get("RECORD_DIR") or None,
                    help="keep every command as WAV + JSON sidecar under DIR/YYYY-MM-DD/ (env RECORD_DIR)")
    ap.add_argument("--no-record", dest="record", action="store_false", default=_env_on("RECORD", True),
                    help="disable recording even if --record-dir is set (env RECORD=0)")
    ap.add_argument("--record-retention-days", type=int, default=int(os.environ.get("RECORD_RETENTION_DAYS") or 7),
                    help="delete recording day folders older than this (env RECORD_RETENTION_DAYS, default 7)")
    ap.add_argument("--speaker-webhook", metavar="URL",
                    help="POST {speaker, best, score, pass} as JSON to this URL after each decision, "
                         "before the transcript is returned (e.g. a Home Assistant webhook)")
    ap.add_argument("--stt-primary", choices=["openrouter", "upstream"],
                    default=os.environ.get("STT_PRIMARY") or "openrouter",
                    help="openrouter = Whisper via OpenRouter (needs OPENROUTER_API_KEY), upstream transcribes in "
                         "parallel as shadow and fallback; upstream = upstream only (env STT_PRIMARY)")
    ap.add_argument("--openrouter-model", default=os.environ.get("OPENROUTER_STT_MODEL") or "openai/whisper-large-v3")
    ap.add_argument("--openrouter-providers", default=os.environ.get("OPENROUTER_STT_PROVIDERS", "Together,Groq"),
                    help="comma-separated provider order (env OPENROUTER_STT_PROVIDERS; empty = OpenRouter's choice)")
    ap.add_argument("--stt-language", default=os.environ.get("STT_LANGUAGE", "en"), help="ISO-639-1; empty = auto")
    ap.add_argument("--stt-prompt-file", type=Path, default=os.environ.get("STT_PROMPT_FILE") or None,
                    help="vocabulary prompt for Whisper, re-read each call (env STT_PROMPT_FILE)")
    ap.add_argument("--stt-timeout", type=float, default=float(os.environ.get("STT_TIMEOUT_S") or 4.0),
                    help="seconds to wait for OpenRouter before using the upstream transcript")
    ap.add_argument("--upstream-timeout", type=float, default=float(os.environ.get("UPSTREAM_TIMEOUT_S") or 3.0),
                    help="seconds per upstream step (connect, send, transcript after end of audio); on timeout the "
                         "command continues without the upstream (env UPSTREAM_TIMEOUT_S)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gate = Gate(args)
    server = AsyncServer.from_uri(args.uri)
    _LOG.info("listening on %s, upstream %s", args.uri, args.upstream)
    await server.run(partial(Handler, gate))


if __name__ == "__main__":
    asyncio.run(main())
