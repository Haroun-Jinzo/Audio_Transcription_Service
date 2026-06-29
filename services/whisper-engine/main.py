"""RapidoMeet whisper-engine — Phase 1 production service with evaluation and data collection."""

from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import config
from evaluation.metrics import aggregate_metrics, evaluate_transcript
from training.collector import TrainingCollector
from transcription.corrections import CorrectionEngine
from transcription.diarization import get_diarization_engine
from transcription.engine import TranscriptionEngine, TranscriptSegment

# Lazy-loaded singletons
_engine: TranscriptionEngine | None = None
_corrections: CorrectionEngine | None = None
_collector: TrainingCollector | None = None


def get_engine() -> TranscriptionEngine:
    global _engine
    if _engine is None:
        _engine = TranscriptionEngine()
    return _engine


def get_corrections() -> CorrectionEngine:
    global _corrections
    if _corrections is None:
        _corrections = CorrectionEngine()
    return _corrections


def get_collector() -> TrainingCollector:
    global _collector
    if _collector is None:
        _collector = TrainingCollector()
    return _collector


def _segments_to_dict(segments: list[TranscriptSegment]) -> list[dict]:
    return [
        {
            "id": s.id,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "speaker": s.speaker,
        }
        for s in segments
    ]


def _parse_corrected_segments(raw: str | None) -> list[TranscriptSegment] | None:
    if not raw:
        return None
    data = json.loads(raw)
    return [
        TranscriptSegment(
            id=item.get("id", i),
            start=item["start"],
            end=item["end"],
            text=item["text"],
            speaker=item.get("speaker", "Speaker 0"),
        )
        for i, item in enumerate(data)
    ]


def _load_domain_terms() -> list[str]:
    if not config.DOMAIN_PROMPT_PATH.exists():
        return []
    text = config.DOMAIN_PROMPT_PATH.read_text(encoding="utf-8")
    terms = []
    for token in text.replace("\n", " ").split(","):
        token = token.strip()
        if token:
            terms.append(token)
    return terms


def run_pipeline(
    audio_path: str,
    language: str | None = None,
    enable_diarization: bool = True,
    apply_corrections: bool = True,
) -> dict:
    """Full Phase 1 pipeline: transcribe → diarize → correct."""
    engine = get_engine()
    result = engine.transcribe(audio_path, language=language or None)

    segments = result.segments
    if enable_diarization and config.DIARIZATION_ENABLED:
        try:
            diarizer = get_diarization_engine()
            speaker_turns = diarizer.diarize(audio_path)
            segments = diarizer.assign_speakers(segments, speaker_turns)
        except Exception as exc:
            print(f"Diarization failed, continuing without speakers: {exc}")

    transcript = " ".join(s.text for s in segments).strip()

    if apply_corrections:
        corrections = get_corrections()
        transcript, segments = corrections.apply_to_transcript(transcript, segments)

    return {
        "transcript": transcript,
        "segments": _segments_to_dict(segments),
        "duration": result.duration,
        "language": result.language,
        "language_probability": result.language_probability,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load whisper model at startup
    get_engine()
    yield


app = FastAPI(
    title="RapidoMeet Whisper Engine",
    description="French/English meeting transcription with speaker diarization",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    collector_status = get_collector().status()
    return {
        "status": "ok",
        "model": config.WHISPER_MODEL_SIZE,
        "diarization_enabled": config.DIARIZATION_ENABLED,
        "training": collector_status,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(""),
    collect: bool = Form(False),
    meeting_id: str = Form(""),
):
    """
    Transcribe a meeting recording with speaker labels and domain corrections.

    - language: 'fr', 'en', or empty for auto-detect
    - collect: if true, also save as a Phase 2 training sample
    """
    start_time = time.time()
    suffix = os.path.splitext(audio.filename or ".wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        lang = language.strip().lower() if language.strip() else None
        if lang and lang not in config.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{lang}'. Use 'fr', 'en', or omit for auto-detect.",
            )

        result = run_pipeline(tmp_path, language=lang)

        if collect:
            engine_segments = [
                TranscriptSegment(**s) for s in result["segments"]
            ]
            get_collector().save_sample(
                audio_path=tmp_path,
                segments=engine_segments,
                transcript=result["transcript"],
                meeting_id=meeting_id or None,
                metadata={"language": result["language"]},
            )

        result["processing_time_seconds"] = round(time.time() - start_time, 2)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        os.remove(tmp_path)


@app.post("/collect")
async def collect(
    audio: UploadFile = File(...),
    corrected_segments: str = Form(""),
    meeting_id: str = Form(""),
    language: str = Form(""),
):
    """
    Save a corrected transcript as a Phase 2 training sample.

    Upload the meeting audio plus user-corrected segments JSON.
    If no corrections provided, saves the auto-transcript for later correction.
    """
    suffix = os.path.splitext(audio.filename or ".wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        lang = language.strip().lower() if language.strip() else None
        auto_result = run_pipeline(tmp_path, language=lang)

        corrected = _parse_corrected_segments(corrected_segments)
        segments = [
            TranscriptSegment(**s) for s in auto_result["segments"]
        ]
        transcript = auto_result["transcript"]

        if corrected:
            transcript = " ".join(s.text for s in corrected).strip()

        saved = get_collector().save_sample(
            audio_path=tmp_path,
            segments=segments,
            transcript=transcript,
            meeting_id=meeting_id or None,
            corrected_segments=corrected,
            metadata={"language": auto_result["language"]},
        )

        return {
            "saved": saved,
            "training_status": get_collector().status(),
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid corrected_segments JSON")
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        os.remove(tmp_path)


@app.post("/evaluate")
async def evaluate(
    audio: UploadFile | None = File(default=None),
    reference: str = Form(""),
    language: str = Form(""),
    run_test_set: bool = Form(False),
):
    """
    Evaluate transcription quality against a reference transcript.

    - Single sample: provide audio + reference text
    - Batch: set run_test_set=true (expects data/test_set/<name>/audio.* + reference.txt)
    """
    domain_terms = _load_domain_terms()

    if run_test_set:
        return _evaluate_test_set(domain_terms)

    if not audio or not reference.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide audio + reference, or set run_test_set=true",
        )

    suffix = os.path.splitext(audio.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        lang = language.strip().lower() if language.strip() else None
        result = run_pipeline(tmp_path, language=lang)
        metrics = evaluate_transcript(reference, result["transcript"], domain_terms)

        return {
            "hypothesis": result["transcript"],
            "reference": reference,
            "metrics": {
                "wer": metrics.wer,
                "cer": metrics.cer,
                "domain_term_accuracy": metrics.domain_term_accuracy,
                "domain_terms_found": metrics.domain_terms_found,
                "domain_terms_total": metrics.domain_terms_total,
            },
            "model": config.WHISPER_MODEL_SIZE,
        }
    finally:
        os.remove(tmp_path)


def _evaluate_test_set(domain_terms: list[str]) -> dict:
    test_dir = config.TEST_SET_DIR
    if not test_dir.exists():
        raise HTTPException(status_code=404, detail="No test set found in data/test_set/")

    results = []
    samples = []

    for case_dir in sorted(test_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        reference_path = case_dir / "reference.txt"
        if not reference_path.exists():
            continue

        audio_path = None
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            candidate = case_dir / f"audio{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        if audio_path is None:
            for f in case_dir.iterdir():
                if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
                    audio_path = f
                    break
        if audio_path is None:
            continue

        reference = reference_path.read_text(encoding="utf-8").strip()
        hypothesis_result = run_pipeline(str(audio_path))
        metrics = evaluate_transcript(
            reference, hypothesis_result["transcript"], domain_terms
        )

        sample = {
            "case": case_dir.name,
            "wer": metrics.wer,
            "cer": metrics.cer,
            "domain_term_accuracy": metrics.domain_term_accuracy,
        }
        results.append(metrics)
        samples.append(sample)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Test set folders need audio + reference.txt",
        )

    return {
        "model": config.WHISPER_MODEL_SIZE,
        "aggregate": aggregate_metrics(results),
        "samples": samples,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
