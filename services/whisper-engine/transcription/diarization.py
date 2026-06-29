"""Pyannote speaker diarization and alignment with Whisper segments."""

from __future__ import annotations

from dataclasses import dataclass

import config
from transcription.engine import TranscriptSegment


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class DiarizationEngine:
    def __init__(self) -> None:
        if not config.HF_TOKEN:
            raise ValueError(
                "HF_TOKEN is required for pyannote diarization. "
                "Set it in your environment and accept the model license on Hugging Face."
            )

        from pyannote.audio import Pipeline

        print(f"Loading pyannote pipeline {config.PYANNOTE_MODEL}...")
        try:
            self._pipeline = Pipeline.from_pretrained(
                config.PYANNOTE_MODEL,
                token=config.HF_TOKEN,
            )
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(
                config.PYANNOTE_MODEL,
                use_auth_token=config.HF_TOKEN,
            )
        print("Pyannote pipeline loaded.")

    def diarize(self, audio_path: str) -> list[SpeakerTurn]:
        diarization = self._pipeline(audio_path)
        turns: list[SpeakerTurn] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=turn.start,
                    end=turn.end,
                    speaker=self._format_speaker(speaker),
                )
            )
        return turns

    @staticmethod
    def _format_speaker(label: str) -> str:
        if label.startswith("SPEAKER_"):
            num = label.replace("SPEAKER_", "")
            try:
                return f"Speaker {int(num)}"
            except ValueError:
                pass
        return label

    @staticmethod
    def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        return max(0.0, min(end_a, end_b) - max(start_a, start_b))

    def assign_speakers(
        self,
        segments: list[TranscriptSegment],
        speaker_turns: list[SpeakerTurn],
    ) -> list[TranscriptSegment]:
        if not speaker_turns:
            return segments

        labeled: list[TranscriptSegment] = []
        for segment in segments:
            best_speaker = segment.speaker
            best_overlap = 0.0

            for turn in speaker_turns:
                overlap = self._overlap(
                    segment.start, segment.end, turn.start, turn.end
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker

            labeled.append(
                TranscriptSegment(
                    id=segment.id,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=best_speaker,
                )
            )
        return labeled


_diarization_engine: DiarizationEngine | None = None


def get_diarization_engine() -> DiarizationEngine:
    global _diarization_engine
    if _diarization_engine is None:
        _diarization_engine = DiarizationEngine()
    return _diarization_engine
