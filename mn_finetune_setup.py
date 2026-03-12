#!/usr/bin/env python
"""
Prepare Mongolian ASR fine-tuning assets for NeMo from:
  - labels.csv (filename + transcript columns)
  - audios/   (wav files)

Outputs (under --out-dir):
  manifests/train_manifest.jsonl
  manifests/val_manifest.jsonl
  manifests/test_manifest.jsonl
  tokenizer/train_corpus.txt
  tokenizer/tokenizer.model + tokenizer.vocab (unless --skip-tokenizer)
  starter_finetune_config.yaml
  prep_report.json

Example:
  .\\.speech_venv\\Scripts\\python.exe mn_finetune_setup.py ^
      --labels labels.csv --audio-dir audios --out-dir mn_finetune_assets
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import soundfile as sf


@dataclass
class PrepStats:
    total_rows: int = 0
    kept_rows: int = 0
    missing_audio: int = 0
    bad_audio: int = 0
    empty_file: int = 0
    empty_text: int = 0
    duration_hours: float = 0.0
    duration_min_sec: float = 0.0
    duration_max_sec: float = 0.0
    duration_avg_sec: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Mongolian NeMo fine-tuning assets.")
    p.add_argument("--labels", type=Path, default=Path("labels.csv"))
    p.add_argument("--audio-dir", type=Path, default=Path("audios"))
    p.add_argument("--out-dir", type=Path, default=Path("mn_finetune_assets"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.90)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--tokenizer-vocab-size", type=int, default=1024)
    p.add_argument("--tokenizer-model-type", type=str, default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--tokenizer-char-coverage", type=float, default=1.0)
    p.add_argument("--skip-tokenizer", action="store_true")
    p.add_argument(
        "--base-model",
        type=str,
        default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
        help="Pretrained NeMo model to adapt. Tokenizer is replaced by this script's tokenizer.",
    )
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--precision", type=str, default="16")
    return p.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def pick_columns(fieldnames: Iterable[str]) -> Tuple[str, str]:
    fields = [f for f in fieldnames if f is not None]
    lower_to_original = {f.strip().lower(): f for f in fields}

    file_candidates = ["file", "filename", "wav", "audio", "path", "audio_filepath"]
    text_candidates = ["sentence", "text", "transcript", "label", "target"]

    file_col = None
    text_col = None

    for c in file_candidates:
        if c in lower_to_original:
            file_col = lower_to_original[c]
            break
    for c in text_candidates:
        if c in lower_to_original:
            text_col = lower_to_original[c]
            break

    if file_col is None and fields:
        file_col = fields[0]
    if text_col is None and len(fields) > 1:
        text_col = fields[-1]

    if not file_col or not text_col:
        raise ValueError(f"Could not determine file/text columns from CSV header: {fields}")
    return file_col, text_col


def load_samples(labels_csv: Path, audio_dir: Path) -> Tuple[List[Dict], PrepStats, str, str]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels CSV not found: {labels_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio directory not found: {audio_dir}")

    samples: List[Dict] = []
    stats = PrepStats()
    durations: List[float] = []

    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV header is empty.")
        file_col, text_col = pick_columns(reader.fieldnames)

        for row in reader:
            stats.total_rows += 1

            raw_file = (row.get(file_col) or "").strip()
            raw_text = normalize_text(row.get(text_col) or "")

            if not raw_file:
                stats.empty_file += 1
                continue
            if not raw_text:
                stats.empty_text += 1
                continue

            if not raw_file.lower().endswith(".wav"):
                raw_file = f"{raw_file}.wav"

            audio_path = (audio_dir / raw_file).resolve()
            if not audio_path.exists():
                stats.missing_audio += 1
                continue

            try:
                info = sf.info(str(audio_path))
                duration = float(info.duration)
            except Exception:
                stats.bad_audio += 1
                continue

            if duration <= 0:
                stats.bad_audio += 1
                continue

            samples.append(
                {
                    "audio_filepath": str(audio_path),
                    "duration": round(duration, 6),
                    "text": raw_text,
                }
            )
            durations.append(duration)

    stats.kept_rows = len(samples)
    if durations:
        total = sum(durations)
        stats.duration_hours = total / 3600.0
        stats.duration_min_sec = min(durations)
        stats.duration_max_sec = max(durations)
        stats.duration_avg_sec = total / len(durations)

    return samples, stats, file_col, text_col


def split_samples(samples: List[Dict], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if not samples:
        return [], [], []
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1.")

    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Keep non-empty splits when data size allows it.
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    elif n == 2:
        n_train = 1
        n_val = 0
    else:
        n_train = 1
        n_val = 0

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_manifest(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in items:
            # Keep manifests ASCII-safe for Windows default codepages.
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_corpus(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in items:
            f.write(row["text"] + "\n")


def train_tokenizer(
    corpus_path: Path,
    tokenizer_dir: Path,
    vocab_size: int,
    model_type: str,
    char_coverage: float,
) -> bool:
    try:
        import sentencepiece as spm
    except Exception:
        return False

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = tokenizer_dir / "tokenizer"

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=char_coverage,
        max_sentence_length=4096,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )

    # NeMo change_vocabulary() for BPE expects vocab.txt to exist.
    spm_vocab = tokenizer_dir / "tokenizer.vocab"
    nemo_vocab = tokenizer_dir / "vocab.txt"
    if spm_vocab.exists():
        with spm_vocab.open("r", encoding="utf-8") as src, nemo_vocab.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                piece = line.split("\t", 1)[0]
                dst.write(piece + "\n")

    return True


def write_starter_config(
    path: Path,
    args: argparse.Namespace,
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    tokenizer_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_dir_str = tokenizer_dir.resolve().as_posix()
    train_manifest_str = train_manifest.resolve().as_posix()
    val_manifest_str = val_manifest.resolve().as_posix()
    test_manifest_str = test_manifest.resolve().as_posix()
    yaml_text = f"""# Starter config for Mongolian ASR fine-tuning (NeMo)
# Generated by mn_finetune_setup.py

base_model: '{args.base_model}'
tokenizer:
  dir: '{tokenizer_dir_str}'
  type: 'bpe'

data:
  sample_rate: {args.sample_rate}
  train_manifest: '{train_manifest_str}'
  val_manifest: '{val_manifest_str}'
  test_manifest: '{test_manifest_str}'
  train_batch_size: {args.train_batch_size}
  val_batch_size: {args.val_batch_size}

training:
  max_epochs: {args.max_epochs}
  learning_rate: {args.learning_rate}
  precision: '{args.precision}'
  seed: {args.seed}

# Suggested NeMo fine-tune flow (Python API):
# 1) model = nemo_asr.models.ASRModel.from_pretrained(base_model)
# 2) model.change_vocabulary(new_tokenizer_dir=tokenizer.dir, new_tokenizer_type=tokenizer.type)
# 3) model.setup_training_data(...)
# 4) model.setup_validation_data(...)
# 5) Trainer(...).fit(model)
"""
    path.write_text(yaml_text, encoding="utf-8")


def write_report(
    path: Path,
    stats: PrepStats,
    file_col: str,
    text_col: str,
    n_train: int,
    n_val: int,
    n_test: int,
    tokenizer_trained: bool,
) -> None:
    payload = {
        "csv_file_column": file_col,
        "csv_text_column": text_col,
        "total_rows": stats.total_rows,
        "kept_rows": stats.kept_rows,
        "missing_audio": stats.missing_audio,
        "bad_audio": stats.bad_audio,
        "empty_file": stats.empty_file,
        "empty_text": stats.empty_text,
        "duration_hours": round(stats.duration_hours, 3),
        "duration_min_sec": round(stats.duration_min_sec, 3),
        "duration_max_sec": round(stats.duration_max_sec, 3),
        "duration_avg_sec": round(stats.duration_avg_sec, 3),
        "split_train": n_train,
        "split_val": n_val,
        "split_test": n_test,
        "tokenizer_trained": tokenizer_trained,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    samples, stats, file_col, text_col = load_samples(args.labels, args.audio_dir)
    if not samples:
        raise RuntimeError("No valid samples found after filtering.")

    train_items, val_items, test_items = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    manifests_dir = args.out_dir / "manifests"
    tokenizer_dir = args.out_dir / "tokenizer"

    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    test_manifest = manifests_dir / "test_manifest.jsonl"
    corpus_path = tokenizer_dir / "train_corpus.txt"
    config_path = args.out_dir / "starter_finetune_config.yaml"
    report_path = args.out_dir / "prep_report.json"

    write_manifest(train_manifest, train_items)
    write_manifest(val_manifest, val_items)
    write_manifest(test_manifest, test_items)
    write_corpus(corpus_path, train_items)

    tokenizer_trained = False
    if not args.skip_tokenizer:
        tokenizer_trained = train_tokenizer(
            corpus_path=corpus_path,
            tokenizer_dir=tokenizer_dir,
            vocab_size=args.tokenizer_vocab_size,
            model_type=args.tokenizer_model_type,
            char_coverage=args.tokenizer_char_coverage,
        )

    write_starter_config(
        path=config_path,
        args=args,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        tokenizer_dir=tokenizer_dir,
    )
    write_report(
        path=report_path,
        stats=stats,
        file_col=file_col,
        text_col=text_col,
        n_train=len(train_items),
        n_val=len(val_items),
        n_test=len(test_items),
        tokenizer_trained=tokenizer_trained,
    )

    print("Preparation complete.")
    print(f"- Output dir: {args.out_dir.resolve()}")
    print(f"- Train/Val/Test: {len(train_items)}/{len(val_items)}/{len(test_items)}")
    print(f"- Hours retained: {stats.duration_hours:.2f}")
    print(f"- Missing audio rows: {stats.missing_audio}")
    print(f"- Tokenizer trained: {tokenizer_trained}")
    if not tokenizer_trained and not args.skip_tokenizer:
        print("  (Install sentencepiece in your venv, then rerun without --skip-tokenizer.)")
    print(f"- Starter config: {config_path.resolve()}")
    print(f"- Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
