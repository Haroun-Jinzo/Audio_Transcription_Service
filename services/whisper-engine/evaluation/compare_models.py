"""
Deploy gate — compare baseline vs candidate model on the fixed test set.

Runs the same held-out meetings through both models and decides whether the
fine-tuned model is safe to deploy. This is the final check AFTER LoRA training
AND CTranslate2 conversion (conversion can introduce small regressions).

Usage:
  # Compare base model vs fine-tuned CT2 model
  python -m evaluation.compare_models --candidate models/ct2

  # Custom baseline (default: large-v3-turbo)
  python -m evaluation.compare_models \\
    --baseline large-v3-turbo \\
    --candidate models/ct2 \\
    --max-wer-regression 0.02

Exit code 0 = safe to deploy, 1 = reject candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
from evaluation.metrics import aggregate_metrics, evaluate_transcript


def load_domain_terms() -> list[str]:
    if not config.DOMAIN_PROMPT_PATH.exists():
        return []
    text = config.DOMAIN_PROMPT_PATH.read_text(encoding="utf-8")
    return [t.strip() for t in text.replace("\n", " ").split(",") if t.strip()]


def find_test_cases(test_dir: Path) -> list[dict]:
    cases = []
    if not test_dir.exists():
        return cases

    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

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
                if f.suffix.lower() in audio_extensions:
                    audio_path = f
                    break
        if audio_path is None:
            continue

        language_path = case_dir / "language.txt"
        language = None
        if language_path.exists():
            language = language_path.read_text(encoding="utf-8").strip() or None

        cases.append(
            {
                "name": case_dir.name,
                "audio": str(audio_path),
                "reference": reference_path.read_text(encoding="utf-8").strip(),
                "language": language,
            }
        )
    return cases


def transcribe_file(model_path: str, audio_path: str, language: str | None) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_path,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    kwargs: dict = {"beam_size": config.WHISPER_BEAM_SIZE}
    if language:
        kwargs["language"] = language

    if config.DOMAIN_PROMPT_PATH.exists():
        prompt = config.DOMAIN_PROMPT_PATH.read_text(encoding="utf-8").strip()
        if prompt:
            kwargs["initial_prompt"] = prompt

    segments, _ = model.transcribe(audio_path, **kwargs)
    return " ".join(s.text.strip() for s in segments).strip()


def evaluate_model(
    model_path: str,
    cases: list[dict],
    domain_terms: list[str],
) -> dict:
    results = []
    per_case = []

    for case in cases:
        hypothesis = transcribe_file(model_path, case["audio"], case.get("language"))
        metrics = evaluate_transcript(case["reference"], hypothesis, domain_terms)
        results.append(metrics)
        per_case.append(
            {
                "case": case["name"],
                "wer": metrics.wer,
                "cer": metrics.cer,
                "domain_term_accuracy": metrics.domain_term_accuracy,
                "hypothesis_preview": hypothesis[:200],
            }
        )

    return {
        "model": model_path,
        "aggregate": aggregate_metrics(results),
        "cases": per_case,
    }


def compare(baseline: dict, candidate: dict) -> dict:
    b = baseline["aggregate"]
    c = candidate["aggregate"]

    wer_delta = c["wer_mean"] - b["wer_mean"]
    cer_delta = c["cer_mean"] - b["cer_mean"]
    domain_delta = c["domain_term_accuracy_mean"] - b["domain_term_accuracy_mean"]

    return {
        "wer_delta": round(wer_delta, 4),
        "cer_delta": round(cer_delta, 4),
        "domain_term_accuracy_delta": round(domain_delta, 4),
        "baseline_wer": b["wer_mean"],
        "candidate_wer": c["wer_mean"],
        "baseline_domain_term_accuracy": b["domain_term_accuracy_mean"],
        "candidate_domain_term_accuracy": c["domain_term_accuracy_mean"],
    }


def passes_gate(
    comparison: dict,
    max_wer_regression: float,
    min_domain_term_gain: float,
) -> tuple[bool, list[str]]:
    reasons = []

    if comparison["wer_delta"] > max_wer_regression:
        reasons.append(
            f"WER regressed by {comparison['wer_delta']:+.4f} "
            f"(max allowed: +{max_wer_regression})"
        )

    if comparison["domain_term_accuracy_delta"] < min_domain_term_gain:
        reasons.append(
            f"Domain term accuracy changed by {comparison['domain_term_accuracy_delta']:+.4f} "
            f"(min required gain: +{min_domain_term_gain})"
        )

    if comparison["candidate_wer"] >= comparison["baseline_wer"] and comparison["wer_delta"] > 0:
        reasons.append("Candidate WER is worse than baseline")

    return len(reasons) == 0, reasons


def print_report(baseline: dict, candidate: dict, comparison: dict, passed: bool, reasons: list[str]) -> None:
    print("=" * 60)
    print("DEPLOY GATE — before vs after")
    print("=" * 60)
    print(f"Baseline:  {baseline['model']}")
    print(f"Candidate: {candidate['model']}")
    print()
    print(f"{'Metric':<30} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}")
    print("-" * 60)
    print(
        f"{'WER (lower is better)':<30} "
        f"{comparison['baseline_wer']:>10.4f} "
        f"{comparison['candidate_wer']:>10.4f} "
        f"{comparison['wer_delta']:>+10.4f}"
    )
    print(
        f"{'Domain term accuracy':<30} "
        f"{comparison['baseline_domain_term_accuracy']:>10.4f} "
        f"{comparison['candidate_domain_term_accuracy']:>10.4f} "
        f"{comparison['domain_term_accuracy_delta']:>+10.4f}"
    )
    print()

    for case_b, case_c in zip(baseline["cases"], candidate["cases"]):
        print(f"  {case_b['case']}: WER {case_b['wer']:.4f} -> {case_c['wer']:.4f}")

    print()
    if passed:
        print("RESULT: PASS — safe to deploy")
        print(f"  Set WHISPER_MODEL_SIZE={candidate['model']}")
    else:
        print("RESULT: FAIL — do not deploy")
        for reason in reasons:
            print(f"  - {reason}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy gate: baseline vs candidate model")
    parser.add_argument("--baseline", default=config.WHISPER_MODEL_SIZE, help="Baseline model path or faster-whisper name")
    parser.add_argument("--candidate", required=True, help="Candidate CT2 model directory")
    parser.add_argument("--test-dir", type=Path, default=config.TEST_SET_DIR)
    parser.add_argument("--max-wer-regression", type=float, default=config.MAX_WER_REGRESSION)
    parser.add_argument("--min-domain-term-gain", type=float, default=config.MIN_DOMAIN_TERM_GAIN)
    parser.add_argument("--output", type=Path, help="Save JSON report to file")
    args = parser.parse_args()

    cases = find_test_cases(args.test_dir)
    if not cases:
        print(f"ERROR: No test cases in {args.test_dir}", file=sys.stderr)
        print("Add folders with audio + reference.txt (see data/test_set/README.txt)", file=sys.stderr)
        return 1

    domain_terms = load_domain_terms()
    print(f"Evaluating {len(cases)} test cases...")

    print(f"\n[1/2] Running baseline: {args.baseline}")
    baseline = evaluate_model(args.baseline, cases, domain_terms)

    print(f"\n[2/2] Running candidate: {args.candidate}")
    candidate = evaluate_model(args.candidate, cases, domain_terms)

    comparison = compare(baseline, candidate)
    passed, reasons = passes_gate(comparison, args.max_wer_regression, args.min_domain_term_gain)

    report = {
        "passed": passed,
        "comparison": comparison,
        "baseline": baseline,
        "candidate": candidate,
        "reasons": reasons,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output}")

    print_report(baseline, candidate, comparison, passed, reasons)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
