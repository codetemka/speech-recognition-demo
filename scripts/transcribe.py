from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import lightning.pytorch as pl
import torch
from nemo.collections.asr.models import ASRModel


def collect_audio_files(input_path: Path, glob_pattern: str) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        return [input_path.resolve()]

    files = sorted(input_path.rglob(glob_pattern))
    if not files:
        raise RuntimeError(f"No files matching '{glob_pattern}' found under {input_path}")
    return [f.resolve() for f in files]


def extract_text(pred: Any) -> str:
    if isinstance(pred, str):
        return pred
    if hasattr(pred, "text") and pred.text is not None:
        return str(pred.text)
    return str(pred)


def maybe_set_decoder_type(model: ASRModel, decoder_type: str | None) -> None:
    if decoder_type is None:
        return

    try:
        if decoder_type == "rnnt":
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

            dec_cfg = RNNTDecodingConfig(fused_batch_size=-1)
        else:
            from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

            dec_cfg = CTCDecodingConfig()

        if hasattr(model, "cur_decoder"):
            model.change_decoding_strategy(dec_cfg, decoder_type=decoder_type)
        else:
            model.change_decoding_strategy(dec_cfg)
    except Exception as exc:
        # TODO: Decoder override can vary by NeMo release; validate locally if this warning appears.
        print(f"Warning: failed to set decoder_type='{decoder_type}': {exc}")


def load_model(model_path: Path | None, pretrained_name: str | None) -> ASRModel:
    if model_path and pretrained_name:
        raise ValueError("Pass only one of --model-path or --pretrained-name")
    if not model_path and not pretrained_name:
        raise ValueError("One of --model-path or --pretrained-name is required")

    if model_path:
        return ASRModel.restore_from(restore_path=str(model_path), map_location="cpu")
    return ASRModel.from_pretrained(model_name=str(pretrained_name))


def transcribe_batch(model: ASRModel, paths: List[str], batch_size: int, num_workers: int) -> List[str]:
    try:
        override_cfg = model.get_transcribe_config()
        override_cfg.batch_size = int(batch_size)
        override_cfg.num_workers = int(num_workers)
        override_cfg.return_hypotheses = False
        preds = model.transcribe(audio=paths, override_config=override_cfg)
    except Exception:
        preds = model.transcribe(paths2audio_files=paths, batch_size=int(batch_size))

    if isinstance(preds, tuple) and len(preds) > 0:
        preds = preds[0]

    return [extract_text(p) for p in preds]


def chunked(items: List[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ASR transcription for one WAV or a WAV directory.")
    parser.add_argument("--input", type=Path, required=True, help="Single audio file or directory.")
    parser.add_argument("--glob", type=str, default="*.wav", help="Glob pattern if --input is a directory.")

    parser.add_argument("--model-path", type=Path, default=None, help="Path to .nemo model.")
    parser.add_argument("--pretrained-name", type=str, default=None, help="Pretrained NeMo model name.")

    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional tokenizer dir for model.change_vocabulary() when needed.",
    )
    parser.add_argument("--tokenizer-type", type=str, default="bpe", choices=["bpe", "wpe"])

    parser.add_argument("--decoder-type", type=str, default="rnnt", choices=["ctc", "rnnt"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)

    parser.add_argument("--output-json", type=Path, default=Path("outputs/transcriptions.json"))
    parser.add_argument("--output-txt", type=Path, default=Path("outputs/transcriptions.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_files = collect_audio_files(args.input, args.glob)
    audio_paths = [str(p) for p in audio_files]

    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    model = load_model(args.model_path, args.pretrained_name)
    model.set_trainer(trainer)
    model = model.eval()

    if args.tokenizer_dir is not None:
        if not args.tokenizer_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {args.tokenizer_dir}")
        model.change_vocabulary(new_tokenizer_dir=str(args.tokenizer_dir), new_tokenizer_type=args.tokenizer_type)

    maybe_set_decoder_type(model, args.decoder_type)

    if torch.cuda.is_available():
        model = model.to("cuda")

    all_preds: List[str] = []
    with torch.no_grad():
        for batch in chunked(audio_paths, max(args.chunk_size, 1)):
            preds = transcribe_batch(model, batch, args.batch_size, args.num_workers)
            if len(preds) != len(batch):
                raise RuntimeError(
                    f"Transcribe output length mismatch: got {len(preds)} for {len(batch)} input files"
                )
            all_preds.extend(preds)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "audio_filepath": str(path),
            "text": pred,
        }
        for path, pred in zip(audio_files, all_preds)
    ]

    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_txt.write_text("\n".join(all_preds), encoding="utf-8")

    print(f"Transcribed {len(all_preds)} files")
    print(f"JSON output: {args.output_json}")
    print(f"Text output: {args.output_txt}")

    if len(all_preds) == 1:
        print(f"\nPrediction: {all_preds[0]}")


if __name__ == "__main__":
    main()
