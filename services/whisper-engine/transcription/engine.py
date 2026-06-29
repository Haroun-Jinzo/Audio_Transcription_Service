"""faster-whisper transcription engine with domain vocabulary biasing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from faster_whisper import WhisperModel

import config


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    speaker: str = "Speaker 0"


@dataclass
class TranscriptionResult:
    transcript: str
    segments: list[TranscriptSegment]
    duration: float
    language: str
    language_probability: float


class TranscriptionEngine:
    def __init__(self) -> None:
        print(
            f"Loading WhisperModel {config.WHISPER_MODEL_SIZE} "
            f"on {config.WHISPER_DEVICE}..."
        )
        self._model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        self._domain_prompt = self._load_domain_prompt()
        print("Whisper model loaded.")

    @staticmethod
    def _load_domain_prompt() -> str | None:
        if not config.DOMAIN_PROMPT_PATH.exists():
            return None
        text = config.DOMAIN_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return text or None

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio. Omit language to auto-detect (fr/en meetings)."""
        kwargs: dict = {
            "beam_size": config.WHISPER_BEAM_SIZE,
            "vad_filter": config.WHISPER_VAD_FILTER,
        }

        if self._domain_prompt:
            kwargs["initial_prompt"] = self._domain_prompt

        if language:
            kwargs["language"] = language

        segments_iter, info = self._model.transcribe(audio_path, **kwargs)

        segments = list(self._collect_segments(segments_iter))
        transcript = " ".join(s.text for s in segments).strip()

        lang_prob = getattr(info, "language_probability", None)
        return TranscriptionResult(
            transcript=transcript,
            segments=segments,
            duration=info.duration,
            language=info.language,
            language_probability=float(lang_prob) if lang_prob is not None else 0.0,
        )

    @staticmethod
    def _collect_segments(
        segments: Iterator,
    ) -> Iterator[TranscriptSegment]:
        for segment in segments:
            yield TranscriptSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
