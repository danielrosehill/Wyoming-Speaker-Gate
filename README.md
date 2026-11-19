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
