"""Background training sample collection (Phase 2)."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config
from transcription.engine import TranscriptSegment


class TrainingCollector:
    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or config.TRAINING_SAMPLES_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_sample(
        self,
        audio_path: str,
        segments: list[TranscriptSegment],
        transcript: str,
        meeting_id: str | None = None,
        corrected_segments: list[TranscriptSegment] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Save an audio + transcript pair for future fine-tuning."""
        sample_id = meeting_id or str(uuid.uuid4())
        sample_dir = self._output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        audio_dest = sample_dir / Path(audio_path).name
        if Path(audio_path).resolve() != audio_dest.resolve():
            shutil.copy2(audio_path, audio_dest)

        final_segments = corrected_segments or segments
        sample_data = {
            "id": sample_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transcript": transcript,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker,
                }
                for s in final_segments
            ],
            "audio_file": audio_dest.name,
            "corrected": corrected_segments is not None,
            "metadata": metadata or {},
        }

        manifest_path = sample_dir / "sample.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        return {
            "sample_id": sample_id,
            "path": str(sample_dir),
            "corrected": sample_data["corrected"],
        }

    def list_samples(self) -> list[dict]:
        samples = []
        for sample_dir in sorted(self._output_dir.iterdir()):
            manifest = sample_dir / "sample.json"
            if manifest.exists():
                with open(manifest, encoding="utf-8") as f:
                    samples.append(json.load(f))
        return samples

    def total_audio_hours(self) -> float:
        total_seconds = 0.0
        for sample in self.list_samples():
            for seg in sample.get("segments", []):
                total_seconds += seg.get("end", 0) - seg.get("start", 0)
        return total_seconds / 3600.0

    def is_ready_for_finetuning(self) -> bool:
        return self.total_audio_hours() >= config.MIN_TRAINING_HOURS

    def status(self) -> dict:
        samples = self.list_samples()
        corrected = sum(1 for s in samples if s.get("corrected"))
        hours = self.total_audio_hours()
        return {
            "total_samples": len(samples),
            "corrected_samples": corrected,
            "total_audio_hours": round(hours, 2),
            "min_hours_required": config.MIN_TRAINING_HOURS,
            "ready_for_finetuning": hours >= config.MIN_TRAINING_HOURS,
        }
