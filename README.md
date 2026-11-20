# Wyoming-Speaker-Gate

A [Wyoming](https://github.com/OHF-Voice/wyoming) speech-to-text proxy that only lets enrolled voices through.
Put it between Home Assistant's Assist pipeline and your real STT service: when the speaker matches a
voiceprint, the upstream transcript is returned unchanged; when not, the transcript is empty, so the pipeline
ends with "no text recognized" and nothing happens. Stops a TV, podcast or guest from commanding the house
after a false wake.

```
satellite (wake word) -> HA Assist pipeline -> speaker-gate -> your Wyoming STT (Whisper, Falafel, ...)
                                                    |
                                       WeSpeaker embedding vs voiceprints
```

- Speaker embeddings: [WeSpeaker ResNet34](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) via
  `pyannote.audio`, CPU only. The embedding runs while the upstream STT transcribes, so the added latency is small.
- Score: cosine similarity of the command (silence trimmed) to each enrolled voiceprint; per-speaker thresholds.
- Works for every satellite on that pipeline, because the gate sits in the pipeline, not on the device.
- It checks **who** is speaking, not whether the voice is live: a recording of an enrolled person passes.

## Run

```bash
cp .env.example .env    # set UPSTREAM, HF_TOKEN, thresholds
mkdir -p data
docker compose build
# enroll: ~60 s of normal speech per person, recorded ON THE SATELLITE'S MIC
docker compose run --rm --no-deps --entrypoint python speaker-gate \
  -m speaker_gate.enroll --voiceprints /data/voiceprints.npz --name Alice --audio /data/alice.wav --replace
docker compose up -d
```

Then in Home Assistant: add a **Wyoming Protocol** integration pointing at the gate's host and port (default 10310).
It appears as `stt.speaker_gate_<upstream name>`; pick it as the speech-to-text engine of your Assist pipeline.

Recording 60 s from a PipeWire/PulseAudio mic:
`parecord -d <source> --rate=16000 --channels=1 --format=s16le --file-format=wav alice.wav` (stop after a minute).

## Options

| Flag / env | Default | |
|---|---|---|
| `--upstream` / `UPSTREAM` | (required) | upstream Wyoming STT, `tcp://host:port` |
| `--threshold` / `THRESHOLD` | 0.40 | cosine needed to accept |
| `--speaker-threshold NAME=T` / `SPEAKER_THRESHOLDS` | | per-speaker override |
| `--mode` / `MODE` | enforce | `shadow` logs scores but never blocks (use it to calibrate) |
| `--min-speech` | 0.6 | seconds of voiced audio needed to judge; shorter is rejected unless `--allow-short` |
| `--log-file` | `/data/decisions.jsonl` in compose | one JSON line per utterance: scores, decision, latency |
| `--log-text` | off | add the transcript to the log |
| `--record-dir DIR` / `RECORD_DIR` | `/data/recordings` in compose, off otherwise | keep every command as WAV + JSON (see [Recordings](#recordings)); `--save-audio` is an alias |
| `--no-record` / `RECORD=0` | recording on | turn recording off without removing the directory setting |
| `--record-retention-days` / `RECORD_RETENTION_DAYS` | 7 | day folders older than this are deleted |
| `--stt-primary` / `STT_PRIMARY` | openrouter | `openrouter`: Whisper via OpenRouter, the upstream in parallel as shadow and fallback; `upstream`: upstream only (see [STT routing](#stt-routing-openrouter-whisper-with-upstream-fallback)) |
| `--openrouter-model` / `OPENROUTER_STT_MODEL` | `openai/whisper-large-v3` | needs `OPENROUTER_API_KEY` in `.env`; without it the gate uses the upstream only |
| `--openrouter-providers` / `OPENROUTER_STT_PROVIDERS` | `Together,Groq` | provider order; empty = OpenRouter's choice |
| `--stt-language` / `STT_LANGUAGE` | en | ISO-639-1; empty = auto-detect |
| `--stt-prompt-file` / `STT_PROMPT_FILE` | `/data/stt-prompt.txt` in compose | Whisper vocabulary prompt, re-read on every command |
| `--stt-timeout` / `STT_TIMEOUT_S` | 4 | seconds to wait for OpenRouter before returning the upstream transcript |
| `--speaker-webhook URL` | off | POST `{"speaker", "best", "score", "pass"}` as JSON after each decision, before the transcript is returned (1.5 s timeout, failures only logged). Lets the consumer know who is speaking |

### STT routing: OpenRouter Whisper with upstream fallback

With `STT_PRIMARY=openrouter` (default) and an `OPENROUTER_API_KEY`, each command goes to two engines at once:
the upstream Wyoming STT (streamed as before) and Whisper on OpenRouter (`POST /api/v1/audio/transcriptions`,
JSON body with the 16 kHz WAV base64-encoded, `language`, `prompt`, `provider.order`). The OpenRouter text is
returned; on an HTTP error or after `STT_TIMEOUT_S` the upstream text is returned instead. The upstream stays
the Wyoming backend Home Assistant sees, so the HA entity keeps its name (`stt.speaker_gate_<upstream name>`).

Logged per command (decisions.jsonl and the recording sidecar): `stt_primary` (configured), `stt_used`
(`openrouter` or `upstream`), `text_openrouter`, `text_upstream`, `stt_ms_openrouter`, `stt_ms_upstream`,
`openrouter_error`/`upstream_error` when one failed. `text` is the transcript used. In decisions.jsonl the
transcripts only appear with `--log-text`; the sidecar always has them.

Vocabulary prompt: a text file (`data/stt-prompt.txt` in compose, see `stt-prompt.example.txt`) of names and rare
words, comma-separated. Whisper uses it as preceding context, which biases spelling of those words. It is re-read on
every command, so edit it without restarting.

Measured 2026-11-20 from an Israeli home server, 3-4 s clips, `language=en`, with a prompt:

| model / provider | latency | prompt honoured? |
|---|---|---|
| whisper-large-v3 / Together | 0.8-1.2 s | yes ("Log a dose of Omeprazole and Vector Plus") |
| whisper-large-v3 / Groq | ~0.4 s | partly ("Omprazole and Vector Plus") |
| whisper-large-v3 / DeepInfra | 2.2-4.2 s | no ("Omprozole in VectorPlus") |
| whisper-large-v3-turbo / Groq, DeepInfra | 0.35-1.4 s | partly / no |
| whisper-1 (OpenAI) | 0.8-2.9 s | yes |

Hence the default: large-v3 with provider order Together, then Groq. Without a provider order OpenRouter picked
DeepInfra for some calls, which ignored the prompt and took several seconds. Price is per second of audio, about $0.03 to
$0.11 per hour depending on provider: a few cents a month for a household.

To go back to the upstream only: `STT_PRIMARY=upstream` in `.env`, then `docker compose up -d`.

### Recordings

On by default in the compose setup, so STT failures can be replayed and A/B tested against other STT engines.
For every command the gate receives (wake word to end of VAD, exactly what the upstream STT got, after conversion
to 16 kHz mono 16-bit) it writes:

```
data/recordings/YYYY-MM-DD/YYYYMMDDTHHMMSS-mmm_<Speaker|unknown>.wav
data/recordings/YYYY-MM-DD/YYYYMMDDTHHMMSS-mmm_<Speaker|unknown>.json
```

- Timestamp = local time (container `TZ`) the audio started; `<Speaker>` is the accepted voiceprint name, `unknown`
  when none passed its threshold (or the clip was too short to judge).
- The `.json` holds the decision-log record (`ts`, `audio_s`, levels, `speaker`, `best`, `score`, `scores`, `pass`,
  `mode`, `latency_s`) plus `text` (the upstream transcript, always), `returned_text` (what the gate passed on; empty
  if blocked) and `audio_file`. The same `audio_file` (relative to `data/recordings/`) is added to the
  `decisions.jsonl` line.
- Retention: day folders older than `RECORD_RETENTION_DAYS` (default 7) are deleted at startup and on the first
  write of each day. Only folders named `YYYY-MM-DD` are touched.
- Disable: `RECORD=0` in `.env`, then `docker compose up -d`. Existing files stay until you delete them.
- Writing failures are logged and never block the transcript. Roughly 32 KB per second of audio.

Replay a recording through any Wyoming STT (e.g. to compare engines):

```bash
uv run --with wyoming python - tcp://HOST:10300 data/recordings/2026-11-20/20261120T204307-123_Daniel.wav <<'PY'
import asyncio, sys, wave
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
async def main(uri, path):
    with wave.open(path) as w:
        rate, width, ch, pcm = w.getframerate(), w.getsampwidth(), w.getnchannels(), w.readframes(w.getnframes())
    async with AsyncClient.from_uri(uri) as c:
        await c.write_event(Transcribe().event())
        await c.write_event(AudioStart(rate=rate, width=width, channels=ch).event())
        for i in range(0, len(pcm), 2048):
            await c.write_event(AudioChunk(rate=rate, width=width, channels=ch, audio=pcm[i:i + 2048]).event())
        await c.write_event(AudioStop().event())
        while (ev := await c.read_event()) and not Transcript.is_type(ev.type):
            pass
        print(Transcript.from_event(ev).text if ev else "(no transcript)")
asyncio.run(main(*sys.argv[1:]))
PY
```

### Addressing the speaker by name (Home Assistant)

With `--speaker-webhook http://HA:8123/api/webhook/speaker_gate_decision`, a trigger-based template sensor stores the
latest speaker, and the conversation agent's prompt reads it:

```yaml
template:
  - triggers:
      - trigger: webhook
        webhook_id: speaker_gate_decision
        local_only: true
        allowed_methods: [POST]
    sensor:
      - name: Voice last speaker
        unique_id: voice_last_speaker
        state: "{{ trigger.json.speaker if trigger.json.speaker else 'unknown' }}"
        attributes:
          at: "{{ now().isoformat() }}"
```

In the LLM prompt (templates render on every turn), only trust it when fresh, since typed requests never pass the gate:

```jinja
{% set s = states.sensor.voice_last_speaker %}{% if s and s.state not in ['unknown', 'unavailable']
   and (now() - as_datetime(s.attributes.at)).total_seconds() < 30 %}{{ s.state }} is speaking.{% else %}Speaker unknown.{% endif %}
```

Several voiceprints per person are allowed (enroll again without `--replace`, e.g. one per room mic); the best one counts.

## Calibrating

The mic matters more than anything else. In the author's tests (WeSpeaker ResNet34, 2026-11):

- Voiceprints from a close mic scored only ~0.45 (man) and ~0.25 (woman) against the same people on a far-field
  speakerphone; re-enrolling on the speakerphone raised that to ~0.57 and ~0.43 on 1.5 s commands and ~0.67 / ~0.49 on 3 s.
- Other voices (an AI-voiced podcast, LibriSpeech speakers) peaked at 0.39 against the speakerphone prints.
- So: enroll on the device, start at 0.40, run `MODE=shadow` for a day if unsure, then read `decisions.jsonl`
  and set each person's threshold between their low scores and the strangers' high ones.

Short commands (about 1 s) are the hard case: there is less voice to judge.

## Licence

MIT.
