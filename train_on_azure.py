#!/usr/bin/env python
"""
Submit Mongolian NeMo fine-tuning to Azure ML.

This launcher runs two steps inside one Azure ML command job:
1) Build manifests/tokenizer/config via mn_finetune_setup.py
2) Fine-tune via mn_finetune_train.py

Example:
  python train_on_azure.py ^
    --stream

Expected .env keys:
  AZURE_SUBSCRIPTION_ID=<...>
  AZURE_RESOURCE_GROUP=<...>
  AZUREML_WORKSPACE_NAME=<...>
  AZUREML_COMPUTE=<...>
Optional:
  AZUREML_ENVIRONMENT=azureml:acpt-pytorch-2.2-cuda12.1@latest
  AZUREML_EXPERIMENT_NAME=mn-nemo-finetune
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import tempfile
from typing import Iterable, List, Sequence

try:
    from azure.ai.ml import Input, MLClient, Output, command
    from azure.ai.ml.constants import AssetTypes
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import AzureCliCredential, DefaultAzureCredential
except ImportError as exc:
    raise SystemExit(
        "Missing Azure ML SDK dependencies. Install with:\n"
        "  pip install azure-ai-ml azure-identity"
    ) from exc


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file that stores Azure/job defaults.",
    )
    bootstrap_args, remaining = bootstrap.parse_known_args()
    load_dotenv_file(bootstrap_args.env_file)

    p = argparse.ArgumentParser(
        description="Submit NeMo ASR fine-tuning job to Azure ML.",
        parents=[bootstrap],
    )

    p.add_argument("--subscription-id", type=str, default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    p.add_argument("--resource-group", type=str, default=os.getenv("AZURE_RESOURCE_GROUP"))
    p.add_argument("--workspace-name", type=str, default=os.getenv("AZUREML_WORKSPACE_NAME"))
    p.add_argument("--compute", type=str, default=os.getenv("AZUREML_COMPUTE"), help="Azure ML compute cluster name.")
    p.add_argument("--environment", type=str,vdefault=os.getenv("AZUREML_ENVIRONMENT"))
    p.add_argument("--experiment-name", type=str, default=os.getenv("AZUREML_EXPERIMENT_NAME", "mn-nemo-finetune"))
    p.add_argument("--display-name", type=str, default=None)
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--stream", action="store_true", help="Stream logs after submission.")
    p.add_argument("--labels-path", type=Path, default=Path("labels.csv"))
    p.add_argument("--audio-dir", type=Path, default=Path("audios"))
    p.add_argument("--input-mode", type=str, default="ro_mount", choices=["ro_mount", "download"])

    p.add_argument(
        "--prepared-assets-output-uri",
        type=str,
        default=None,
        help="Optional datastore URI for prepared assets output.",
    )
    p.add_argument(
        "--trained-model-output-uri",
        type=str,
        default=None,
        help="Optional datastore URI for trained model output.",
    )

    p.add_argument("--setup-script", type=Path, default=Path("mn_finetune_setup.py"))
    p.add_argument("--train-script", type=Path, default=Path("mn_finetune_train.py"))

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.90)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--tokenizer-vocab-size", type=int, default=1024)
    p.add_argument("--tokenizer-model-type", type=str, default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--tokenizer-char-coverage", type=float, default=1.0)
    p.add_argument(
        "--base-model",
        type=str,
        default="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
    )

    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--precision", type=str, default="16")

    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--max-time",
        type=str,
        default=None,
        help="Maximum wall-clock training time for Lightning Trainer, e.g. '01:08:00:00' (DD:HH:MM:SS).",
    )
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--test-after-fit", action="store_true")
    p.add_argument(
        "--nemo-log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    p.add_argument(
        "--skip-runtime-pip-install",
        action="store_true",
        help="Skip installing python packages at job runtime.",
    )
    return p.parse_args(remaining, namespace=bootstrap_args)


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value



def load_dotenv_file(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_quotes(value.strip())
        if not key:
            continue

        os.environ.setdefault(key, value)


def ensure_required_workspace_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.subscription_id:
        missing.append("AZURE_SUBSCRIPTION_ID")
    if not args.resource_group:
        missing.append("AZURE_RESOURCE_GROUP")
    if not args.workspace_name:
        missing.append("AZUREML_WORKSPACE_NAME")
    if not args.compute:
        missing.append("AZUREML_COMPUTE")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required values in {args.env_file}: {joined}"
        )


def normalize_environment_ref(env_ref: str) -> str:
    ref = str(env_ref).strip()
    if not ref:
        return ref

    lower = ref.lower()
    if lower.startswith("azureml:") or lower.startswith("azureml://"):
        return ref

    # Likely a docker image (for example mcr.microsoft.com/...:tag)
    if "/" in ref and ":" in ref:
        return ref

    # Treat plain AzureML env names as curated/workspace references.
    if "@" in ref or ":" in ref:
        return f"azureml:{ref}"

    return ref


def must_exist(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def append_if_not_none(parts: List[str], flag: str, value: object) -> None:
    if value is None:
        return
    parts.extend([flag, str(value)])


def stage_job_code(scripts: Iterable[Path]) -> Path:
    stage_dir = Path(tempfile.mkdtemp(prefix="azureml_nemo_job_"))
    for src in scripts:
        dst = stage_dir / src.name
        shutil.copy2(src, dst)
    return stage_dir


def get_ml_client(args: argparse.Namespace) -> MLClient:
    scope = "https://management.azure.com/.default"
    try:
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        credential.get_token(scope)
    except Exception:
        credential = AzureCliCredential()
        credential.get_token(scope)

    return MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )


def build_command_text(args: argparse.Namespace, setup_script_name: str, train_script_name: str) -> str:
    steps: List[str] = []

    if not args.skip_runtime_pip_install:
        steps.append("python -m pip install --upgrade pip")
        steps.append(
            shell_join(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "nemo_toolkit[asr]",
                    "lightning",
                    "omegaconf",
                    "soundfile",
                    "sentencepiece",
                ]
            )
        )

    setup_cmd: List[str] = [
        "python",
        setup_script_name,
        "--labels",
        "${{inputs.labels}}",
        "--audio-dir",
        "${{inputs.audio_dir}}",
        "--out-dir",
        "${{outputs.prepared_assets}}",
        "--seed",
        str(args.seed),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--sample-rate",
        str(args.sample_rate),
        "--tokenizer-vocab-size",
        str(args.tokenizer_vocab_size),
        "--tokenizer-model-type",
        args.tokenizer_model_type,
        "--tokenizer-char-coverage",
        str(args.tokenizer_char_coverage),
        "--base-model",
        args.base_model,
        "--train-batch-size",
        str(args.train_batch_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--precision",
        args.precision,
    ]
    steps.append(shell_join(setup_cmd))

    train_cmd: List[str] = [
        "python",
        train_script_name,
        "--config",
        "${{outputs.prepared_assets}}/starter_finetune_config.yaml",
        "--exp-dir",
        "${{outputs.trained_model}}",
        "--devices",
        str(args.devices),
        "--num-workers",
        str(args.num_workers),
        "--nemo-log-level",
        args.nemo_log_level,
        "--save-nemo-path",
        "${{outputs.trained_model}}/final_mn_finetuned.nemo",
    ]

    append_if_not_none(train_cmd, "--max-steps", args.max_steps)
    append_if_not_none(train_cmd, "--max-time", args.max_time)
    if args.freeze_encoder:
        train_cmd.append("--freeze-encoder")
    if args.dry_run:
        train_cmd.append("--dry-run")
    if args.test_after_fit:
        train_cmd.append("--test-after-fit")

    steps.append(shell_join(train_cmd))
    return " && \\\n".join(steps)


def main() -> None:
    args = parse_args()
    ensure_required_workspace_args(args)

    labels_path = must_exist(args.labels_path, "Labels CSV")
    audio_dir = must_exist(args.audio_dir, "Audio directory")
    setup_script = must_exist(args.setup_script, "Setup script")
    train_script = must_exist(args.train_script, "Train script")

    if not audio_dir.is_dir():
        raise SystemExit(f"--audio-dir must point to a directory: {audio_dir}")

    ml_client = get_ml_client(args)

    staged_code_dir = stage_job_code([setup_script, train_script])
    command_text = build_command_text(args, setup_script.name, train_script.name)
    environment_ref = normalize_environment_ref(args.environment)

    prepared_output = Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")
    trained_output = Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")
    if args.prepared_assets_output_uri:
        prepared_output.path = args.prepared_assets_output_uri
    if args.trained_model_output_uri:
        trained_output.path = args.trained_model_output_uri

    job = command(
        code=str(staged_code_dir),
        command=command_text,
        environment=environment_ref,
        compute=args.compute,
        experiment_name=args.experiment_name,
        display_name=args.display_name,
        instance_count=args.instance_count,
        inputs={
            "labels": Input(type=AssetTypes.URI_FILE, path=str(labels_path), mode=args.input_mode),
            "audio_dir": Input(type=AssetTypes.URI_FOLDER, path=str(audio_dir), mode=args.input_mode),
        },
        outputs={
            "prepared_assets": prepared_output,
            "trained_model": trained_output,
        },
    )

    try:
        created_job = ml_client.jobs.create_or_update(job)
    except ResourceNotFoundError as exc:
        msg = str(exc)
        if "No environment exists for name" in msg:
            raise SystemExit(
                "Azure ML environment not found. "
                f"Resolved environment value: {environment_ref}\n"
                "Use a valid environment such as:\n"
                "  AZUREML_ENVIRONMENT=azureml:acpt-pytorch-2.2-cuda12.1@latest"
            ) from exc
        raise
    print(f"Submitted job: {created_job.name}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Workspace: {args.workspace_name}")
    if getattr(created_job, "studio_url", None):
        print(f"Studio URL: {created_job.studio_url}")

    if args.stream:
        print("Streaming logs...")
        ml_client.jobs.stream(created_job.name)


if __name__ == "__main__":
    main()
