from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from inspect_dataset import inspect_dataset
from normalize_text import NormalizationConfig, normalize_text


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def build_normalization_config(cfg: Dict[str, Any]) -> NormalizationConfig:
    norm = cfg.get("normalization", {}) or {}
    number = norm.get("number_normalization", {}) or {}
    return NormalizationConfig(
        whitespace_cleanup=bool(norm.get("whitespace_cleanup", True)),
        normalize_quotes=bool(norm.get("normalize_quotes", True)),
        normalize_dashes=bool(norm.get("normalize_dashes", True)),
        lowercase=bool(norm.get("lowercase", False)),
        remove_punctuation=bool(norm.get("remove_punctuation", False)),
        number_normalization_enabled=bool(number.get("enabled", False)),
        number_normalization_mode=str(number.get("mode", "conservative")),
    )


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {
                "audio_filepath": str(row["audio_filepath"]),
                "duration": float(row["duration"]),
                "text": str(row["text"]),
            }
            if "text_raw" in row and isinstance(row["text_raw"], str):
                obj["text_raw"] = row["text_raw"]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NeMo manifests from labels.csv + audios/.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Path to data config YAML.")
    parser.add_argument("--audio-column", type=str, default=None, help="Override CSV audio column name.")
    parser.add_argument("--text-column", type=str, default=None, help="Override CSV transcript column name.")
    parser.add_argument("--split-column", type=str, default=None, help="Override optional split column name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    csv_cfg = cfg.setdefault("csv", {})
    if args.audio_column:
        csv_cfg["audio_column"] = args.audio_column
    if args.text_column:
        csv_cfg["transcript_column"] = args.text_column
    if args.split_column is not None:
        csv_cfg["split_column"] = args.split_column

    summary, valid_df, invalid_df = inspect_dataset(cfg)
    if valid_df.empty:
        raise RuntimeError("No valid samples found after validation. Check reports/invalid_rows.csv for details.")

    norm_cfg_dict = cfg.get("normalization", {}) or {}
    normalization_enabled = bool(norm_cfg_dict.get("enabled", True))

    if normalization_enabled:
        normalizer_cfg = build_normalization_config(cfg)
        preserve_raw = bool(norm_cfg_dict.get("preserve_raw_text", True))

        if preserve_raw:
            valid_df["text_raw"] = valid_df["text"].astype(str)
        valid_df["text"] = valid_df["text"].apply(lambda x: normalize_text(x, normalizer_cfg))

        post_norm_empty = valid_df[valid_df["text"].astype(str).str.strip() == ""]
        if not post_norm_empty.empty:
            drop_invalid = bool((cfg.get("validation", {}) or {}).get("drop_invalid_rows", True))
            reason = "empty_after_normalization"
            post_norm_empty = post_norm_empty.copy()
            post_norm_empty["error_reasons"] = reason
            invalid_df = pd.concat([invalid_df, post_norm_empty], ignore_index=True)
            if drop_invalid:
                valid_df = valid_df[valid_df["text"].astype(str).str.strip() != ""].copy()

    if valid_df.empty:
        raise RuntimeError("All samples became invalid after normalization. Disable aggressive normalization settings.")

    manifests_dir = Path((cfg.get("paths", {}) or {}).get("manifests_dir", "manifests"))
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    test_manifest = manifests_dir / "test_manifest.jsonl"

    train_df = valid_df[valid_df["split"] == "train"].copy()
    val_df = valid_df[valid_df["split"] == "val"].copy()
    test_df = valid_df[valid_df["split"] == "test"].copy()

    if train_df.empty:
        raise RuntimeError("Train split is empty. Adjust split ratios or provide a split column.")
    if val_df.empty:
        print("Warning: validation split is empty.")
    if test_df.empty:
        print("Warning: test split is empty.")

    write_manifest(train_df, train_manifest)
    write_manifest(val_df, val_manifest)
    write_manifest(test_df, test_manifest)

    reports_cfg = cfg.get("reports", {}) or {}
    processed_csv_path = Path(reports_cfg.get("processed_csv_path", "reports/processed_labels.csv"))
    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(processed_csv_path, index=False, encoding="utf-8")

    if bool(reports_cfg.get("save_invalid_rows", True)) and not invalid_df.empty:
        invalid_path = Path(reports_cfg.get("invalid_rows_path", "reports/invalid_rows.csv"))
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_df.to_csv(invalid_path, index=False, encoding="utf-8")

    summary["outputs"] = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "test_manifest": str(test_manifest),
        "processed_csv": str(processed_csv_path),
    }
    summary["counts"] = {
        "num_files": int(len(valid_df)),
        "total_duration_sec": float(valid_df["duration"].sum()),
        "total_duration_hours": float(valid_df["duration"].sum() / 3600.0),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
    }

    summary_path = Path(reports_cfg.get("summary_json_path", "reports/dataset_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Prepared manifests:")
    print(f"  train: {train_manifest} ({len(train_df)} samples)")
    print(f"  val:   {val_manifest} ({len(val_df)} samples)")
    print(f"  test:  {test_manifest} ({len(test_df)} samples)")
    print(f"Processed CSV: {processed_csv_path}")
    print(f"Summary JSON:  {summary_path}")


if __name__ == "__main__":
    main()
