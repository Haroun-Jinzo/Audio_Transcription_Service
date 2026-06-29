"""Build train/val splits from collected and external meeting data."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import config


@dataclass
class TrainingRow:
    audio: str
    text: str
    language: str = "fr"
    start: float | None = None
    end: float | None = None
    source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


class TrainingDataset:
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

    def __init__(
        self,
        samples_dir: Path | None = None,
        external_dir: Path | None = None,
    ) -> None:
        self._samples_dir = samples_dir or config.TRAINING_SAMPLES_DIR
        self._external_dir = external_dir or config.EXTERNAL_DATA_DIR

    def load_all_rows(
        self,
        include_uncorrected: bool = True,
        use_segments: bool = False,
    ) -> list[TrainingRow]:
        """Load rows from RapidoMeet samples, external folders, and manifest JSONL."""
        rows: list[TrainingRow] = []
        rows.extend(self._load_sample_rows(include_uncorrected, use_segments))
        rows.extend(self._load_external_folder_rows(use_segments))
        rows.extend(self._load_manifest_jsonl())
        return [r for r in rows if r.text.strip() and Path(r.audio).exists()]

    def _load_sample_rows(
        self, include_uncorrected: bool, use_segments: bool
    ) -> list[TrainingRow]:
        rows: list[TrainingRow] = []
        if not self._samples_dir.exists():
            return rows

        for sample_dir in sorted(self._samples_dir.iterdir()):
            manifest = sample_dir / "sample.json"
            if not manifest.exists():
                continue

            with open(manifest, encoding="utf-8") as f:
                data = json.load(f)

            if data.get("corrected"):
                segments = data.get("segments", [])
            elif include_uncorrected:
                segments = data.get("segments", [])
            else:
                continue

            audio_path = sample_dir / data["audio_file"]
            language = (data.get("metadata") or {}).get("language", "fr") or "fr"

            if use_segments:
                for seg in segments:
                    rows.append(
                        TrainingRow(
                            audio=str(audio_path),
                            text=seg["text"],
                            language=language,
                            start=seg.get("start"),
                            end=seg.get("end"),
                            source=f"samples:{sample_dir.name}",
                        )
                    )
            else:
                transcript = data.get("transcript") or " ".join(
                    s["text"] for s in segments
                )
                rows.append(
                    TrainingRow(
                        audio=str(audio_path),
                        text=transcript,
                        language=language,
                        source=f"samples:{sample_dir.name}",
                    )
                )
        return rows

    def _load_external_folder_rows(self, use_segments: bool) -> list[TrainingRow]:
        rows: list[TrainingRow] = []
        if not self._external_dir.exists():
            return rows

        for case_dir in sorted(self._external_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            audio_path = self._find_audio(case_dir)
            if audio_path is None:
                continue

            language = self._read_optional_text(case_dir / "language.txt") or "fr"
            transcript_path = case_dir / "transcript.txt"
            manifest_path = case_dir / "sample.json"
            manifest_path = case_dir / "manifest.json" if not manifest_path.exists() else manifest_path

            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    data = json.load(f)
                language = data.get("language", language)
                if use_segments and data.get("segments"):
                    for seg in data["segments"]:
                        rows.append(
                            TrainingRow(
                                audio=str(audio_path),
                                text=seg["text"],
                                language=language,
                                start=seg.get("start"),
                                end=seg.get("end"),
                                source=f"external:{case_dir.name}",
                            )
                        )
                else:
                    text = data.get("transcript") or data.get("text", "")
                    rows.append(
                        TrainingRow(
                            audio=str(audio_path),
                            text=text,
                            language=language,
                            source=f"external:{case_dir.name}",
                        )
                    )
            elif transcript_path.exists():
                rows.append(
                    TrainingRow(
                        audio=str(audio_path),
                        text=transcript_path.read_text(encoding="utf-8").strip(),
                        language=language,
                        source=f"external:{case_dir.name}",
                    )
                )
        return rows

    def _load_manifest_jsonl(self) -> list[TrainingRow]:
        rows: list[TrainingRow] = []
        manifest = self._external_dir / "manifest.jsonl"
        if not manifest.exists():
            return rows

        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rows.append(
                    TrainingRow(
                        audio=data["audio"],
                        text=data["text"],
                        language=data.get("language", "fr"),
                        start=data.get("start"),
                        end=data.get("end"),
                        source=data.get("source", "manifest"),
                    )
                )
        return rows

    @staticmethod
    def _find_audio(directory: Path) -> Path | None:
        for name in ("audio.wav", "audio.mp3", "audio.m4a", "audio.flac"):
            candidate = directory / name
            if candidate.exists():
                return candidate
        for path in directory.iterdir():
            if path.suffix.lower() in TrainingDataset.AUDIO_EXTENSIONS:
                return path
        return None

    @staticmethod
    def _read_optional_text(path: Path) -> str | None:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return None

    def build_splits(
        self,
        output_dir: Path | None = None,
        include_uncorrected: bool = True,
        use_segments: bool = False,
    ) -> dict:
        """90/10 train/val split. Test set in data/test_set/ is never included."""
        rows = self.load_all_rows(include_uncorrected, use_segments)
        if not rows:
            raise ValueError("No training rows found. Add data to data/external/ or training_samples/")

        rng = random.Random(config.TRAIN_SPLIT_SEED)
        rng.shuffle(rows)

        split_idx = int(len(rows) * config.TRAIN_VAL_SPLIT)
        train_rows = rows[:split_idx]
        val_rows = rows[split_idx:]

        if not val_rows:
            val_rows = train_rows[-1:]
            train_rows = train_rows[:-1]

        out = output_dir or config.PROCESSED_DATA_DIR
        out.mkdir(parents=True, exist_ok=True)

        train_path = out / "train.jsonl"
        val_path = out / "val.jsonl"

        self._write_jsonl(train_path, train_rows)
        self._write_jsonl(val_path, val_rows)

        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "total_count": len(rows),
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: list[TrainingRow]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    def to_whisper_jsonl(self, output_path: str | Path) -> int:
        """Legacy export — all corrected samples, no split."""
        rows = self.load_all_rows(include_uncorrected=False, use_segments=True)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(output, rows)
        return len(rows)

    @staticmethod
    def read_jsonl(path: str | Path) -> list[TrainingRow]:
        rows: list[TrainingRow] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(TrainingRow(**json.loads(line)))
        return rows
