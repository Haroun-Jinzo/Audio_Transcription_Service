"""Central configuration for the whisper-engine service."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Whisper / faster-whisper
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"

# Supported languages (auto-detect when language is omitted)
SUPPORTED_LANGUAGES = {"fr", "en"}

# Pyannote speaker diarization
HF_TOKEN = os.getenv("HF_TOKEN", "")
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "true").lower() == "true"

# Domain vocabulary and post-processing
DOMAIN_PROMPT_PATH = DATA_DIR / "domain_prompt.txt"
CORRECTIONS_PATH = DATA_DIR / "corrections.json"

# Training data collection (Phase 2)
TRAINING_SAMPLES_DIR = DATA_DIR / "training_samples"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MIN_TRAINING_HOURS = float(os.getenv("MIN_TRAINING_HOURS", "5"))
TRAIN_VAL_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", "0.9"))
TRAIN_SPLIT_SEED = int(os.getenv("TRAIN_SPLIT_SEED", "42"))

# Fine-tuning (World 2 — offline, local GPU)
WHISPER_BASE_MODEL = os.getenv("WHISPER_BASE_MODEL", "openai/whisper-large-v3-turbo")
LORA_OUTPUT_DIR = BASE_DIR / "models" / "lora"
MERGED_MODEL_DIR = BASE_DIR / "models" / "merged"
CT2_MODEL_DIR = BASE_DIR / "models" / "ct2"
FINETUNE_LANGUAGE = os.getenv("FINETUNE_LANGUAGE", "")  # empty = per-sample language

# Deploy gate thresholds
MAX_WER_REGRESSION = float(os.getenv("MAX_WER_REGRESSION", "0.02"))
MIN_DOMAIN_TERM_GAIN = float(os.getenv("MIN_DOMAIN_TERM_GAIN", "0.0"))

# Evaluation harness
TEST_SET_DIR = DATA_DIR / "test_set"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
