from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import sentencepiece as spm


def read_texts_from_manifest(manifest_path: Path, text_key: str) -> List[str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    texts: List[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_key)
            if text is None:
                raise ValueError(f"Line {i} in {manifest_path} missing '{text_key}'")
            text = str(text).strip()
            if text:
                texts.append(text)
    if not texts:
        raise RuntimeError(f"No non-empty text found in {manifest_path}")
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for NeMo ASR fine-tuning.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/train_manifest.jsonl"),
        help="Training manifest path.",
    )
    parser.add_argument("--text-key", type=str, default="text", help="Text field in manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer"),
        help="Directory for tokenizer.model/tokenizer.vocab.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="SentencePiece vocab size. For small Mongolian ASR sets, 256-1024 is usually practical.",
    )
    parser.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"])
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--input-sentence-size", type=int, default=0, help="0 means use all lines.")
    parser.add_argument("--shuffle-input-sentence", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    texts = read_texts_from_manifest(args.manifest, args.text_key)
    corpus_path = args.output_dir / "train_texts.txt"
    corpus_path.write_text("\n".join(texts), encoding="utf-8")

    model_prefix = args.output_dir / "tokenizer"

    trainer_args = {
        "input": str(corpus_path),
        "model_prefix": str(model_prefix),
        "vocab_size": int(args.vocab_size),
        "model_type": args.model_type,
        "character_coverage": float(args.character_coverage),
        "unk_id": 0,
        "bos_id": -1,
        "eos_id": -1,
        "pad_id": -1,
        "normalization_rule_name": "nfkc",
        "input_sentence_size": int(args.input_sentence_size),
        "shuffle_input_sentence": bool(args.shuffle_input_sentence),
        "num_threads": 4,
    }

    # TODO: If you want language-specific numeric/text normalization beyond NFKC,
    # add it upstream before tokenizer training (prepare_manifest + normalize_text).
    spm.SentencePieceTrainer.train(**trainer_args)

    config_path = args.output_dir / "tokenizer_config.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "text_key": args.text_key,
                "num_sentences": len(texts),
                "vocab_size": args.vocab_size,
                "model_type": args.model_type,
                "character_coverage": args.character_coverage,
                "tokenizer_model": str(args.output_dir / "tokenizer.model"),
                "tokenizer_vocab": str(args.output_dir / "tokenizer.vocab"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Tokenizer trained: {args.output_dir / 'tokenizer.model'}")
    print(f"Tokenizer vocab:   {args.output_dir / 'tokenizer.vocab'}")
    print(f"Tokenizer config:  {config_path}")


if __name__ == "__main__":
    main()
