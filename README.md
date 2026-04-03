# Mongolian NeMo ASR Fine-Tuning Pipeline

Practical transfer-learning pipeline for fine-tuning:

- Base model: `nvidia/stt_kk_ru_fastconformer_hybrid_large`
- Dataset layout: `audios/` + `labels.csv`
- Frameworks: NeMo + PyTorch Lightning + SentencePiece

## Project Structure

```text
speech-recognition-demo/
  audios/
  labels.csv
  requirements.txt
  configs/
    data.yaml
    train.yaml
  scripts/
    inspect_dataset.py
    prepare_manifest.py
    normalize_text.py
    train_tokenizer.py
    finetune_asr.py
    evaluate_asr.py
    transcribe.py
  manifests/
  reports/
  artifacts/
  outputs/
```

## 1) Environment Setup

Install a CUDA-matched PyTorch build first, then install project deps.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
# Install torch first from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## 2) Expected Dataset Layout

```text
audios/
  clip_0001.wav
  clip_0002.wav
  ...
labels.csv
```

`labels.csv` must contain:

- audio filename column (for example `file`)
- transcript text column (for example `sentence`)
- optional split column (`train` / `val` / `test`, or aliases like `dev`)

Default column mapping in `configs/data.yaml`:

```yaml
csv:
  audio_column: file
  transcript_column: sentence
  split_column: null
```

You can either edit YAML or override in commands:

```bash
python scripts/inspect_dataset.py --audio-column file --text-column sentence
```

## 3) Inspect Dataset

Validates:

- missing audio files
- missing/empty transcripts
- unreadable audio
- invalid durations
- invalid split labels (if split column is supplied)

Also computes summary stats and writes reports.

```bash
python scripts/inspect_dataset.py --config configs/data.yaml
```

Outputs:

- `reports/dataset_summary.json`
- `reports/invalid_rows.csv` (if any invalid rows)

## 4) Prepare NeMo Manifests

Creates:

- `manifests/train_manifest.jsonl`
- `manifests/val_manifest.jsonl`
- `manifests/test_manifest.jsonl`

If no split column exists, train/val/test are auto-generated from ratios in `configs/data.yaml`.

```bash
python scripts/prepare_manifest.py --config configs/data.yaml
```

Each manifest line includes at least:

- `audio_filepath`
- `text`
- `duration`

and includes `text_raw` when normalization is enabled and raw text is preserved.

## 5) Transcript Normalization (Standalone)

If you want to normalize text directly (outside manifest prep):

CSV mode:

```bash
python scripts/normalize_text.py \
  --config configs/data.yaml \
  --input-csv labels.csv \
  --output reports/normalized_labels.csv \
  --text-column sentence
```

Manifest mode:

```bash
python scripts/normalize_text.py \
  --config configs/data.yaml \
  --input-manifest manifests/train_manifest.jsonl \
  --output manifests/train_manifest.normalized.jsonl \
  --text-column text
```

Normalization controls are in `configs/data.yaml`:

- whitespace cleanup
- quote normalization
- dash normalization
- optional lowercase
- optional punctuation removal
- optional conservative number normalization

## 6) Train SentencePiece Tokenizer

Recommended small-data starting range: vocab size `512` (try `256` if dataset is very small).

```bash
python scripts/train_tokenizer.py \
  --manifest manifests/train_manifest.jsonl \
  --output-dir artifacts/tokenizer \
  --vocab-size 512 \
  --model-type unigram
```

Outputs:

- `artifacts/tokenizer/tokenizer.model`
- `artifacts/tokenizer/tokenizer.vocab`
- `artifacts/tokenizer/tokenizer_config.json`

## 7) Fine-Tune ASR

Main training command:

```bash
python scripts/finetune_asr.py --config configs/train.yaml
```

What the script does:

- loads `nvidia/stt_kk_ru_fastconformer_hybrid_large` (or `model.init_from_nemo`)
- optionally replaces tokenizer (`model.change_vocabulary(...)`)
- sets train/val/test dataloaders from manifests
- configures optimizer and LR scheduler
- supports mixed precision and gradient accumulation
- uses checkpointing + validation-based model selection + early stopping
- supports freeze strategies:
  - `full_finetune`
  - `freeze_lower`
  - `gradual_unfreeze`

Useful overrides:

```bash
python scripts/finetune_asr.py \
  --config configs/train.yaml \
  --freeze-strategy freeze_lower
```

Training artifacts:

- checkpoints: `artifacts/checkpoints/`
- summary: `artifacts/checkpoints/training_summary.json`
- exported `.nemo` models (best/last) per `configs/train.yaml`

## 8) Evaluate (WER + CER + Error Analysis)

Validation set:

```bash
python scripts/evaluate_asr.py \
  --manifest manifests/val_manifest.jsonl \
  --model-path artifacts/checkpoints/best_model.nemo \
  --decoder-type rnnt \
  --output-dir outputs/eval_val
```

Test set:

```bash
python scripts/evaluate_asr.py \
  --manifest manifests/test_manifest.jsonl \
  --model-path artifacts/checkpoints/best_model.nemo \
  --decoder-type rnnt \
  --output-dir outputs/eval_test
```

Outputs:

- `predictions.jsonl`
- `predictions.txt`
- `metrics.json`

`metrics.json` includes:

- WER
- CER
- word-level substitutions/deletions/insertions
- char-level substitutions/deletions/insertions

## 9) Batch Transcription / Inference

Single WAV file:

```bash
python scripts/transcribe.py \
  --input audios/example.wav \
  --model-path artifacts/checkpoints/best_model.nemo \
  --decoder-type rnnt \
  --output-json outputs/single.json \
  --output-txt outputs/single.txt
```

Directory of WAVs:

```bash
python scripts/transcribe.py \
  --input audios \
  --glob *.wav \
  --model-path artifacts/checkpoints/best_model.nemo \
  --decoder-type rnnt \
  --output-json outputs/batch.json \
  --output-txt outputs/batch.txt
```

You can also load by pretrained model name instead of `.nemo`:

```bash
python scripts/transcribe.py --input audios --pretrained-name nvidia/stt_kk_ru_fastconformer_hybrid_large
```

## Notes for Small-Data Mongolian ASR

- Why `stt_kk_ru_fastconformer_hybrid_large`:
  - It is already pretrained for relevant regional languages (Kazakh/Russian) with a strong FastConformer hybrid RNNT+CTC architecture, making transfer to Mongolian practical.
- Why conservative fine-tuning:
  - Small supervised datasets overfit quickly. Lower LR, moderate regularization, selective freezing, and early stopping are usually more reliable than aggressive full updates.
- Why tokenizer choice matters:
  - Token granularity controls how well Mongolian morphology and OOV words are represented. A bad tokenizer can bottleneck decoding quality even with a strong encoder.
- How to detect overfitting:
  - Training loss keeps dropping while val WER/CER stops improving or gets worse.
  - Predictions become unstable on rare words/noisy samples.
  - Best checkpoint appears early, and later epochs regress.

## Recommended First Experiment Plan

1. Prepare manifests with mild normalization (`lowercase: false`, `remove_punctuation: false`).
2. Train tokenizer with `vocab_size=512` and `model_type=unigram`.
3. Fine-tune with `freeze.strategy=freeze_lower`, `lower_encoder_fraction=0.7`, `precision=16-mixed`, `accumulate_grad_batches=4`, LR `5e-5`.
4. Evaluate on val each epoch (default), then run final test evaluation.

## Second Experiment Plan (If First Overfits)

1. Switch to `freeze.strategy=gradual_unfreeze`.
2. Increase regularization conservatively:
   - slightly stronger SpecAug (for example `time_masks` 5 -> 8)
   - earlier stopping (`patience` 8 -> 5)
3. Reduce update aggressiveness:
   - lower LR (`5e-5` -> `2e-5`)
   - keep tokenizer fixed if already stable
4. Re-run val/test and compare substitutions/deletions/insertions patterns.

## Expected Best Fine-Tuning Strategy for Small Data

For a relatively small Mongolian supervised set, `freeze_lower` is usually the most stable first choice.

- `full_finetune` often overfits early.
- `freeze_lower` usually gives a better bias-variance tradeoff quickly.
- `gradual_unfreeze` can outperform `freeze_lower` if carefully tuned, but it needs more iteration.
