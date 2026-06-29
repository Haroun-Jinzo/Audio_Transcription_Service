"""
LoRA fine-tuning for Whisper (World 2 — offline, local GPU).

Trains only ~1-2% of model weights (LoRA adapters) on your meeting data.
The base model stays frozen; adapters learn domain vocabulary and accents.

Usage:
  # 1. Put internet meetings in data/external/ (see data/external/README.txt)
  # 2. Build train/val splits
  python -m training.finetune --prepare-only

  # 3. Fine-tune locally (needs NVIDIA GPU, ~4-8h on T4-class GPU)
  python -m training.finetune --epochs 3 --batch-size 4

Output: models/lora/ (adapter weights only, not deployable alone)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from training.dataset import TrainingDataset


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: object

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def load_audio(path: str, start: float | None, end: float | None, sr: int = 16000) -> np.ndarray:
    import librosa

    if start is not None and end is not None:
        duration = max(end - start, 0.01)
        audio, _ = librosa.load(path, sr=sr, offset=start, duration=duration, mono=True)
    else:
        audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def prepare_dataset(rows: list, processor, sampling_rate: int = 16000):
    from datasets import Dataset

    def _prepare(batch):
        audio = load_audio(batch["audio"], batch.get("start"), batch.get("end"), sr=sampling_rate)
        batch["input_features"] = processor(
            audio, sampling_rate=sampling_rate
        ).input_features[0]

        language = batch.get("language") or config.FINETUNE_LANGUAGE or "fr"
        processor.tokenizer.set_prefix_tokens(language=language)
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    ds = Dataset.from_list([r.to_dict() for r in rows])
    return ds.map(_prepare, remove_columns=ds.column_names)


def prepare_splits(use_segments: bool, include_uncorrected: bool) -> dict:
    builder = TrainingDataset()
    return builder.build_splits(
        include_uncorrected=include_uncorrected,
        use_segments=use_segments,
    )


def run_finetune(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Fine-tuning will be extremely slow on CPU.")

    processor = WhisperProcessor.from_pretrained(
        config.WHISPER_BASE_MODEL, language="fr", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.WHISPER_BASE_MODEL)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = TrainingDataset.read_jsonl(train_path)
    val_rows = TrainingDataset.read_jsonl(val_path)

    train_ds = prepare_dataset(train_rows, processor)
    val_ds = prepare_dataset(val_rows, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=100,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2Seq(processor),
        tokenizer=processor,
    )

    print(f"Training on {len(train_rows)} samples, validating on {len(val_rows)}...")
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"LoRA adapter saved to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Whisper on meeting data")
    parser.add_argument("--prepare-only", action="store_true", help="Only build train/val JSONL splits")
    parser.add_argument("--train-path", type=Path, default=config.PROCESSED_DATA_DIR / "train.jsonl")
    parser.add_argument("--val-path", type=Path, default=config.PROCESSED_DATA_DIR / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=config.LORA_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--use-segments", action="store_true", help="Train on segment clips instead of full meetings")
    parser.add_argument("--include-uncorrected", action="store_true", help="Include auto-transcripts from RapidoMeet samples")
    args = parser.parse_args()

    if args.prepare_only or not args.train_path.exists():
        print("Building train/val splits...")
        info = prepare_splits(args.use_segments, args.include_uncorrected)
        print(
            f"  train: {info['train_count']} rows -> {info['train_path']}\n"
            f"  val:   {info['val_count']} rows -> {info['val_path']}"
        )
        if args.prepare_only:
            return 0

    if not args.train_path.exists():
        print("ERROR: train.jsonl not found. Run with --prepare-only first.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_finetune(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
