"""Post-transcription text corrections for domain-specific terms."""

from __future__ import annotations

import json
import re

import config
from transcription.engine import TranscriptSegment


class CorrectionEngine:
    def __init__(self, corrections_path: str | None = None) -> None:
        path = corrections_path or str(config.CORRECTIONS_PATH)
        self._corrections = self._load_corrections(path)

    @staticmethod
    def _load_corrections(path: str) -> dict[str, str]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {k.lower(): v for k, v in data.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def apply(self, text: str) -> str:
        if not self._corrections:
            return text

        result = text
        # Longer patterns first to avoid partial replacements
        for wrong, right in sorted(
            self._corrections.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            result = pattern.sub(right, result)
        return result

    def apply_to_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                id=s.id,
                start=s.start,
                end=s.end,
                text=self.apply(s.text),
                speaker=s.speaker,
            )
            for s in segments
        ]

    def apply_to_transcript(
        self, transcript: str, segments: list[TranscriptSegment]
    ) -> tuple[str, list[TranscriptSegment]]:
        corrected_segments = self.apply_to_segments(segments)
        corrected_transcript = " ".join(s.text for s in corrected_segments).strip()
        return corrected_transcript, corrected_segments
