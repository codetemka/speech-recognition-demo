from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jiwer
import lightning.pytorch as pl
import torch
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel


def read_manifest(path: Path, text_key: str) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "audio_filepath" not in obj:
                raise ValueError(f"Line {i} missing 'audio_filepath' in {path}")
            if text_key not in obj:
                raise ValueError(f"Line {i} missing '{text_key}' in {path}")
            samples.append(
                {
                    "audio_filepath": str(obj["audio_filepath"]),
                    "reference": str(obj[text_key]),
                    "duration": obj.get("duration"),
                }
            )
    if not samples:
        raise RuntimeError(f"No samples found in {path}")
    return samples


def load_model(model_path: Path | None, pretrained_name: str | None) -> ASRModel:
    if model_path and pretrained_name:
        raise ValueError("Pass either --model-path or --pretrained-name, not both")
    if not model_path and not pretrained_name:
        raise ValueError("One of --model-path or --pretrained-name is required")

    if model_path:
        return ASRModel.restore_from(restore_path=str(model_path), map_location="cpu")
    return ASRModel.from_pretrained(model_name=str(pretrained_name))


def maybe_set_decoder_type(model: ASRModel, decoder_type: str | None) -> None:
    if decoder_type is None:
        return
    decoder_type = decoder_type.lower()
    if decoder_type not in {"ctc", "rnnt"}:
        raise ValueError("decoder_type must be ctc or rnnt")

    if not hasattr(model, "change_decoding_strategy"):
        print("Warning: model does not support change_decoding_strategy; decoder_type override ignored.")
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
        # TODO: Decoder override APIs can differ across NeMo releases; confirm on your installed version if needed.
        print(f"Warning: failed to apply decoder_type='{decoder_type}': {exc}")


def extract_text(pred: Any) -> str:
    if isinstance(pred, str):
        return pred
    if hasattr(pred, "text") and pred.text is not None:
        return str(pred.text)
    return str(pred)


def transcribe_batch(model: ASRModel, audio_paths: List[str], batch_size: int, num_workers: int) -> List[str]:
    try:
        override_cfg = model.get_transcribe_config()
        override_cfg.batch_size = int(batch_size)
        override_cfg.num_workers = int(num_workers)
        override_cfg.return_hypotheses = False
        preds = model.transcribe(audio=audio_paths, override_config=override_cfg)
    except Exception:
        preds = model.transcribe(paths2audio_files=audio_paths, batch_size=int(batch_size))

    if isinstance(preds, tuple) and len(preds) > 0:
        preds = preds[0]
    return [extract_text(p) for p in preds]


def chunked(items: List[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def compute_error_analysis(refs: List[str], hyps: List[str]) -> Dict[str, Any]:
    word_out = jiwer.process_words(refs, hyps)
    char_out = jiwer.process_characters(refs, hyps)
    return {
        "word_level": {
            "substitutions": int(word_out.substitutions),
            "deletions": int(word_out.deletions),
            "insertions": int(word_out.insertions),
            "hits": int(word_out.hits),
        },
        "char_level": {
            "substitutions": int(char_out.substitutions),
            "deletions": int(char_out.deletions),
            "insertions": int(char_out.insertions),
            "hits": int(char_out.hits),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR model (WER/CER) on a NeMo manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSONL path.")
    parser.add_argument("--text-key", type=str, default="text", help="Reference text key in manifest.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to .nemo checkpoint.")
    parser.add_argument("--pretrained-name", type=str, default=None, help="NeMo pretrained model name.")
    parser.add_argument("--decoder-type", type=str, default="rnnt", choices=["ctc", "rnnt"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of files submitted per transcribe call.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all samples.")
    parser.add_argument("--sample-print", type=int, default=10, help="How many decoded examples to print.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_manifest(args.manifest, args.text_key)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

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
    maybe_set_decoder_type(model, args.decoder_type)

    if torch.cuda.is_available():
        model = model.to("cuda")

    all_preds: List[str] = []
    all_refs: List[str] = [x["reference"] for x in samples]
    all_audio: List[str] = [x["audio_filepath"] for x in samples]

    with torch.no_grad():
        for batch_audio in chunked(all_audio, max(args.chunk_size, 1)):
            preds = transcribe_batch(model, batch_audio, args.batch_size, args.num_workers)
            if len(preds) != len(batch_audio):
                raise RuntimeError(
                    f"Transcribe output length mismatch: got {len(preds)} preds for {len(batch_audio)} files"
                )
            all_preds.extend(preds)

    wer = word_error_rate(hypotheses=all_preds, references=all_refs, use_cer=False)
    cer = word_error_rate(hypotheses=all_preds, references=all_refs, use_cer=True)
    analysis = compute_error_analysis(all_refs, all_preds)

    pred_path = args.output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for sample, pred in zip(samples, all_preds):
            out = {
                "audio_filepath": sample["audio_filepath"],
                "reference": sample["reference"],
                "prediction": pred,
                "duration": sample.get("duration"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    txt_path = args.output_dir / "predictions.txt"
    txt_path.write_text("\n".join(all_preds), encoding="utf-8")

    summary = {
        "manifest": str(args.manifest),
        "num_samples": len(samples),
        "wer": float(wer),
        "cer": float(cer),
        "decoder_type": args.decoder_type,
        "error_analysis": analysis,
        "predictions_jsonl": str(pred_path),
        "predictions_txt": str(txt_path),
    }

    summary_path = args.output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    print(
        "Word-level edits: "
        f"S={analysis['word_level']['substitutions']} "
        f"D={analysis['word_level']['deletions']} "
        f"I={analysis['word_level']['insertions']}"
    )

    n = min(max(args.sample_print, 0), len(samples))
    if n > 0:
        print("\nSample decodes:")
        for i in range(n):
            print(f"[{i}] audio={samples[i]['audio_filepath']}")
            print(f"    ref: {samples[i]['reference']}")
            print(f"    hyp: {all_preds[i]}")

    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved metrics:     {summary_path}")


if __name__ == "__main__":
    main()
