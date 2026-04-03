from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u2032": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u2033": '"',
        "\u00AB": '"',
        "\u00BB": '"',
    }
)

DASH_MAP = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)

WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class NormalizationConfig:
    whitespace_cleanup: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    lowercase: bool = False
    remove_punctuation: bool = False
    number_normalization_enabled: bool = False
    number_normalization_mode: str = "conservative"


def _strip_punctuation(text: str) -> str:
    keep = {"'", '"', "-"}
    chars = []
    for ch in text:
        if ch in keep:
            chars.append(ch)
            continue
        if unicodedata.category(ch).startswith("P"):
            chars.append(" ")
            continue
        chars.append(ch)
    return "".join(chars)


def _normalize_numbers_conservative(text: str) -> str:
    # Decimal commas inside numbers -> dot
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    # Remove group separators between digits (space, underscore, NBSP, thin space)
    text = re.sub(r"(?<=\d)[ _\u00A0\u202F](?=\d{3}\b)", "", text)
    return text


def normalize_text(text: Any, cfg: NormalizationConfig) -> str:
    if text is None:
        return ""
    out = str(text)
    out = unicodedata.normalize("NFKC", out)

    if cfg.normalize_quotes:
        out = out.translate(QUOTE_MAP)
    if cfg.normalize_dashes:
        out = out.translate(DASH_MAP)
    if cfg.number_normalization_enabled and cfg.number_normalization_mode == "conservative":
        out = _normalize_numbers_conservative(out)
    if cfg.remove_punctuation:
        out = _strip_punctuation(out)
    if cfg.lowercase:
        out = out.lower()
    if cfg.whitespace_cleanup:
        out = WHITESPACE_RE.sub(" ", out).strip()
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _normalization_config_from_dict(raw: Dict[str, Any]) -> NormalizationConfig:
    num_cfg = raw.get("number_normalization", {}) or {}
    return NormalizationConfig(
        whitespace_cleanup=bool(raw.get("whitespace_cleanup", True)),
        normalize_quotes=bool(raw.get("normalize_quotes", True)),
        normalize_dashes=bool(raw.get("normalize_dashes", True)),
        lowercase=bool(raw.get("lowercase", False)),
        remove_punctuation=bool(raw.get("remove_punctuation", False)),
        number_normalization_enabled=bool(num_cfg.get("enabled", False)),
        number_normalization_mode=str(num_cfg.get("mode", "conservative")),
    )


def normalize_csv(
    input_csv: Path,
    output_csv: Path,
    text_column: str,
    encoding: str,
    cfg: NormalizationConfig,
    raw_text_column: str,
    normalized_text_column: str,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding=encoding)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {input_csv}. Available: {list(df.columns)}")

    df[raw_text_column] = df[text_column].astype(str)
    df[normalized_text_column] = df[text_column].apply(lambda x: normalize_text(x, cfg))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")


def normalize_manifest(
    input_manifest: Path,
    output_manifest: Path,
    text_key: str,
    cfg: NormalizationConfig,
    preserve_raw_key: str,
) -> None:
    if not input_manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {input_manifest}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with input_manifest.open("r", encoding="utf-8") as fin, output_manifest.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_key not in obj:
                raise ValueError(f"Line {i} missing '{text_key}' in {input_manifest}")
            raw = str(obj[text_key])
            obj[preserve_raw_key] = raw
            obj[text_key] = normalize_text(raw, cfg)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Mongolian transcripts for ASR.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Path to data YAML config.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", type=Path, help="CSV file containing transcript text.")
    input_group.add_argument("--input-manifest", type=Path, help="NeMo JSONL manifest to normalize.")

    parser.add_argument("--output", type=Path, required=True, help="Output CSV or JSONL path.")
    parser.add_argument("--text-column", type=str, default=None, help="CSV text column or manifest text key.")
    parser.add_argument("--encoding", type=str, default=None, help="CSV encoding override.")
    parser.add_argument("--raw-text-column", type=str, default="text_raw", help="Raw text column/key name.")
    parser.add_argument(
        "--normalized-text-column",
        type=str,
        default="text_normalized",
        help="Only for CSV mode: normalized text column name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    csv_cfg = cfg.get("csv", {}) or {}
    norm_cfg = _normalization_config_from_dict((cfg.get("normalization", {}) or {}))

    text_column = args.text_column
    if text_column is None:
        text_column = str(csv_cfg.get("transcript_column", "text"))

    if args.input_csv is not None:
        encoding = args.encoding or str(csv_cfg.get("encoding", "utf-8"))
        normalize_csv(
            input_csv=args.input_csv,
            output_csv=args.output,
            text_column=text_column,
            encoding=encoding,
            cfg=norm_cfg,
            raw_text_column=args.raw_text_column,
            normalized_text_column=args.normalized_text_column,
        )
        print(f"Wrote normalized CSV: {args.output}")
    else:
        normalize_manifest(
            input_manifest=args.input_manifest,
            output_manifest=args.output,
            text_key=text_column,
            cfg=norm_cfg,
            preserve_raw_key=args.raw_text_column,
        )
        print(f"Wrote normalized manifest: {args.output}")


if __name__ == "__main__":
    main()
