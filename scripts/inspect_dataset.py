from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

SPLIT_ALIASES = {
    "train": "train",
    "tr": "train",
    "trn": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "tst": "test",
    "eval": "test",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_audio_path(audio_dir: Path, raw_name: Any, add_wav_if_missing: bool) -> Optional[Path]:
    if raw_name is None or (isinstance(raw_name, float) and np.isnan(raw_name)):
        return None

    name = str(raw_name).strip()
    if not name:
        return None

    if add_wav_if_missing and Path(name).suffix == "":
        name = f"{name}.wav"

    path = Path(name)
    if not path.is_absolute():
        path = (audio_dir / path).resolve()
    return path


def normalize_split(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return SPLIT_ALIASES.get(key)


def safe_duration(path: Path) -> Optional[float]:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def auto_assign_splits(num_items: int, ratios: Dict[str, float], seed: int) -> List[str]:
    keys = ["train", "val", "test"]
    ratio_values = [float(ratios.get(k, 0.0)) for k in keys]
    total = sum(ratio_values)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    ratio_values = [r / total for r in ratio_values]

    n_train = int(num_items * ratio_values[0])
    n_val = int(num_items * ratio_values[1])
    n_test = num_items - n_train - n_val

    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng = random.Random(seed)
    rng.shuffle(labels)
    return labels


def inspect_dataset(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    paths_cfg = cfg.get("paths", {}) or {}
    csv_cfg = cfg.get("csv", {}) or {}
    split_cfg = cfg.get("splits", {}) or {}
    val_cfg = cfg.get("validation", {}) or {}

    audio_dir = Path(paths_cfg.get("audio_dir", "audios")).resolve()
    labels_csv = Path(paths_cfg.get("labels_csv", "labels.csv")).resolve()
    encoding = str(csv_cfg.get("encoding", "utf-8"))

    audio_col = str(csv_cfg.get("audio_column", "audio"))
    text_col = str(csv_cfg.get("transcript_column", "text"))
    split_col = csv_cfg.get("split_column")
    add_wav = bool(csv_cfg.get("add_wav_extension_if_missing", True))

    min_dur = float(val_cfg.get("min_duration_sec", 0.0))
    max_dur = float(val_cfg.get("max_duration_sec", 1e9))

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    df = pd.read_csv(labels_csv, encoding=encoding)
    missing_cols = [c for c in [audio_col, text_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}. Available: {list(df.columns)}")
    if split_col and split_col not in df.columns:
        raise ValueError(f"Configured split column '{split_col}' not found. Available: {list(df.columns)}")

    records: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        audio_path = resolve_audio_path(audio_dir, row[audio_col], add_wav)
        transcript = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()

        split_value = None
        if split_col:
            split_value = normalize_split(row[split_col])

        reasons: List[str] = []
        if audio_path is None:
            reasons.append("missing_audio_filename")
        elif not audio_path.exists():
            reasons.append("audio_file_not_found")

        if pd.isna(row[text_col]):
            reasons.append("missing_transcript")
        elif transcript == "":
            reasons.append("empty_transcript")

        duration = None
        if audio_path is not None and audio_path.exists():
            duration = safe_duration(audio_path)
            if duration is None:
                reasons.append("unreadable_audio")
            else:
                if duration < min_dur:
                    reasons.append("duration_too_short")
                if duration > max_dur:
                    reasons.append("duration_too_long")

        if split_col and split_value is None:
            reasons.append("invalid_split_value")

        rec = {
            "row_index": int(idx),
            "audio_filepath": "" if audio_path is None else str(audio_path),
            "audio_exists": bool(audio_path is not None and audio_path.exists()),
            "duration": duration,
            "text": transcript,
            "split": split_value,
        }

        if reasons:
            bad = rec.copy()
            bad["error_reasons"] = ";".join(reasons)
            invalid_rows.append(bad)
        else:
            records.append(rec)

    valid_df = pd.DataFrame(records)
    invalid_df = pd.DataFrame(invalid_rows)

    if not split_col and not valid_df.empty:
        assigned = auto_assign_splits(len(valid_df), split_cfg, int(cfg.get("seed", 42)))
        valid_df["split"] = assigned

    split_counts = valid_df["split"].value_counts().to_dict() if not valid_df.empty else {}
    durations = valid_df["duration"].astype(float).tolist() if not valid_df.empty else []
    text_lengths = valid_df["text"].astype(str).map(len).tolist() if not valid_df.empty else []

    summary = {
        "input": {
            "labels_csv": str(labels_csv),
            "audio_dir": str(audio_dir),
            "audio_column": audio_col,
            "transcript_column": text_col,
            "split_column": split_col,
            "num_rows_in_csv": int(len(df)),
        },
        "validation": {
            "num_valid_rows": int(len(valid_df)),
            "num_invalid_rows": int(len(invalid_df)),
            "drop_invalid_rows": bool(val_cfg.get("drop_invalid_rows", True)),
        },
        "counts": {
            "num_files": int(len(valid_df)),
            "total_duration_sec": float(np.sum(durations) if durations else 0.0),
            "total_duration_hours": float((np.sum(durations) / 3600.0) if durations else 0.0),
            "train_count": int(split_counts.get("train", 0)),
            "val_count": int(split_counts.get("val", 0)),
            "test_count": int(split_counts.get("test", 0)),
        },
        "duration_distribution_sec": summarize_distribution(durations),
        "transcript_length_distribution_chars": summarize_distribution([float(x) for x in text_lengths]),
    }

    return summary, valid_df, invalid_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Mongolian ASR dataset before manifest creation.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Path to data config YAML.")
    parser.add_argument("--audio-column", type=str, default=None, help="Override audio filename column.")
    parser.add_argument("--text-column", type=str, default=None, help="Override transcript column.")
    parser.add_argument("--split-column", type=str, default=None, help="Override split column (optional).")
    parser.add_argument("--print-invalid-sample", type=int, default=10, help="Number of invalid rows to print.")
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

    reports_cfg = cfg.get("reports", {}) or {}
    summary_path = Path(reports_cfg.get("summary_json_path", "reports/dataset_summary.json"))
    invalid_path = Path(reports_cfg.get("invalid_rows_path", "reports/invalid_rows.csv"))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if bool(reports_cfg.get("save_invalid_rows", True)) and not invalid_df.empty:
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_df.to_csv(invalid_path, index=False, encoding="utf-8")

    print("=== Dataset Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not invalid_df.empty:
        print(f"\nInvalid rows: {len(invalid_df)}")
        sample_n = min(len(invalid_df), max(args.print_invalid_sample, 0))
        if sample_n > 0:
            print(invalid_df.head(sample_n).to_string(index=False))

    print(f"\nSaved summary JSON: {summary_path}")
    if bool(reports_cfg.get("save_invalid_rows", True)) and not invalid_df.empty:
        print(f"Saved invalid rows CSV: {invalid_path}")


if __name__ == "__main__":
    main()
