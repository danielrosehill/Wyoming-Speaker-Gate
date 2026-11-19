FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch torchaudio "wyoming>=1.5" numpy soundfile librosa "pyannote.audio>=3.3"
COPY speaker_gate ./speaker_gate
ENV HF_HOME=/data/hf PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "speaker_gate"]
