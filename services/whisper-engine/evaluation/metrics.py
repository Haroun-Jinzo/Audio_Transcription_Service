"""WER/CER and domain-term accuracy metrics for model evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import jiwer


@dataclass
class EvaluationMetrics:
    wer: float
    cer: float
    domain_term_accuracy: float
    domain_terms_found: int
    domain_terms_total: int
    details: dict = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace for fair comparison."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


def compute_domain_term_accuracy(
    reference: str,
    hypothesis: str,
    domain_terms: list[str],
) -> tuple[float, int, int]:
    """Fraction of domain terms present in reference that also appear in hypothesis."""
    if not domain_terms:
        return 1.0, 0, 0

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    found = 0
    total = 0
    for term in domain_terms:
        term_norm = normalize_text(term)
        if not term_norm:
            continue
        if term_norm in ref_norm:
            total += 1
            if term_norm in hyp_norm:
                found += 1

    accuracy = found / total if total > 0 else 1.0
    return accuracy, found, total


def evaluate_transcript(
    reference: str,
    hypothesis: str,
    domain_terms: list[str] | None = None,
) -> EvaluationMetrics:
    terms = domain_terms or []
    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)
    term_acc, found, total = compute_domain_term_accuracy(
        reference, hypothesis, terms
    )

    return EvaluationMetrics(
        wer=round(wer, 4),
        cer=round(cer, 4),
        domain_term_accuracy=round(term_acc, 4),
        domain_terms_found=found,
        domain_terms_total=total,
        details={
            "reference_length": len(normalize_text(reference).split()),
            "hypothesis_length": len(normalize_text(hypothesis).split()),
        },
    )


def aggregate_metrics(results: list[EvaluationMetrics]) -> dict:
    if not results:
        return {"count": 0}

    n = len(results)
    return {
        "count": n,
        "wer_mean": round(sum(r.wer for r in results) / n, 4),
        "cer_mean": round(sum(r.cer for r in results) / n, 4),
        "domain_term_accuracy_mean": round(
            sum(r.domain_term_accuracy for r in results) / n, 4
        ),
        "per_sample": [
            {
                "wer": r.wer,
                "cer": r.cer,
                "domain_term_accuracy": r.domain_term_accuracy,
            }
            for r in results
        ],
    }
