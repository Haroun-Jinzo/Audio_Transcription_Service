"""
Merge LoRA adapters into base Whisper weights and convert to CTranslate2.

faster-whisper cannot load LoRA adapters directly — it needs CTranslate2 format.
This script:
  1. Merges LoRA weights into the base HuggingFace model
  2. Converts the merged model to CTranslate2 (float16)
  3. Saves to models/ct2/ ready for WHISPER_MODEL_SIZE

Usage:
  python -m training.convert
  python -m training.convert --lora-dir models/lora --output-dir models/ct2

After conversion, run the deploy gate before swapping the production model:
  python -m evaluation.compare_models --candidate models/ct2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config


def _find_ct2_converter() -> list[str]:
    scripts_dir = Path(sys.executable).parent
    for name in ("ct2-transformers-converter", "ct2-transformers-converter.exe"):
        path = scripts_dir / name
        if path.exists():
            return [str(path)]
    return ["ct2-transformers-converter"]


def merge_lora(
    base_model: str,
    lora_dir: Path,
    merged_dir: Path,
) -> Path:
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"Merging LoRA adapter from {lora_dir} into {base_model}...")
    merged_dir.mkdir(parents=True, exist_ok=True)

    base = WhisperForConditionalGeneration.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, str(lora_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))

    processor = WhisperProcessor.from_pretrained(str(lora_dir))
    processor.save_pretrained(str(merged_dir))

    print(f"Merged model saved to {merged_dir}")
    return merged_dir


def convert_to_ct2(
    merged_dir: Path,
    ct2_dir: Path,
    quantization: str = "float16",
) -> Path:
    ct2_dir.mkdir(parents=True, exist_ok=True)

    copy_files = [
        "tokenizer.json",
        "preprocessor_config.json",
        "vocab.json",
        "merges.txt",
        "normalizer.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ]
    existing_copies = [f for f in copy_files if (merged_dir / f).exists()]

    converter_cmd = _find_ct2_converter()
    cmd = [
        *converter_cmd,
        "--model",
        str(merged_dir),
        "--output_dir",
        str(ct2_dir),
        "--quantization",
        quantization,
    ]
    if existing_copies:
        cmd.extend(["--copy_files", *existing_copies])

    print("Running CTranslate2 converter...")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            "CTranslate2 conversion failed. Install with: pip install 'ctranslate2>=4.0.0'"
        )

    print(f"CTranslate2 model saved to {ct2_dir}")
    return ct2_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to CTranslate2")
    parser.add_argument("--base-model", default=config.WHISPER_BASE_MODEL)
    parser.add_argument("--lora-dir", type=Path, default=config.LORA_OUTPUT_DIR)
    parser.add_argument("--merged-dir", type=Path, default=config.MERGED_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.CT2_MODEL_DIR)
    parser.add_argument("--quantization", default="float16", choices=["float16", "int8", "int8_float16"])
    parser.add_argument("--skip-merge", action="store_true", help="Convert an already-merged HF model")
    args = parser.parse_args()

    if not args.skip_merge:
        if not args.lora_dir.exists():
            print(f"ERROR: LoRA directory not found: {args.lora_dir}", file=sys.stderr)
            print("Run fine-tuning first: python -m training.finetune", file=sys.stderr)
            return 1
        merge_lora(args.base_model, args.lora_dir, args.merged_dir)
        source_dir = args.merged_dir
    else:
        source_dir = args.merged_dir
        if not source_dir.exists():
            print(f"ERROR: Merged model not found: {source_dir}", file=sys.stderr)
            return 1

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    convert_to_ct2(source_dir, args.output_dir, args.quantization)

    print()
    print("Next step — run the deploy gate:")
    print(f"  python -m evaluation.compare_models --candidate {args.output_dir}")
    print()
    print("If it passes, set in .env:")
    print(f"  WHISPER_MODEL_SIZE={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
