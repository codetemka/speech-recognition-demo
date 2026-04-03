from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.utils import model_utils


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Train config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def build_logger(cfg: Dict[str, Any]):
    log_cfg = cfg.get("logging", {}) or {}
    log_type = str(log_cfg.get("type", "tensorboard")).lower()
    save_dir = str(log_cfg.get("save_dir", "experiments"))
    project = str(log_cfg.get("project", "mn-asr"))
    run_name = str(log_cfg.get("run_name", "asr-finetune"))

    if log_type == "none":
        return False
    if log_type == "tensorboard":
        return TensorBoardLogger(save_dir=save_dir, name=run_name)
    if log_type == "wandb":
        return WandbLogger(project=project, name=run_name, save_dir=save_dir)
    raise ValueError(f"Unsupported logging.type: {log_type}. Use tensorboard, wandb, or none.")


def get_module_by_path(model: torch.nn.Module, module_path: str) -> Optional[torch.nn.Module]:
    current = model
    for part in module_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    if isinstance(current, torch.nn.Module):
        return current
    return None


def set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def find_encoder_layers(model: ASRModel) -> Optional[List[torch.nn.Module]]:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None

    candidates: List[torch.nn.Module] = [encoder]
    nested = getattr(encoder, "encoder", None)
    if isinstance(nested, torch.nn.Module):
        candidates.append(nested)

    attrs = ["layers", "conformer_layers", "encoder_layers", "blocks", "block_modules"]
    for obj in candidates:
        for attr in attrs:
            if not hasattr(obj, attr):
                continue
            maybe = getattr(obj, attr)
            if isinstance(maybe, (torch.nn.ModuleList, list, tuple)) and len(maybe) > 0:
                modules = [m for m in maybe if isinstance(m, torch.nn.Module)]
                if modules:
                    return modules
    return None


@dataclass
class FreezeState:
    strategy: str
    initial_frozen_layers: int
    current_frozen_layers: int


class GradualUnfreezeCallback(Callback):
    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        initial_frozen_layers: int,
        start_epoch: int,
        end_epoch: int,
        target_modules: Sequence[str],
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.initial_frozen_layers = int(initial_frozen_layers)
        self.current_frozen_layers = int(initial_frozen_layers)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.target_modules = list(target_modules)

        if self.end_epoch < self.start_epoch:
            raise ValueError("gradual_unfreeze.end_epoch must be >= start_epoch")

    def _ensure_targets_unfrozen(self, pl_module: ASRModel) -> None:
        for name in self.target_modules:
            mod = get_module_by_path(pl_module, name)
            if mod is not None:
                set_module_trainable(mod, True)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: ASRModel) -> None:
        epoch = int(trainer.current_epoch)
        if epoch < self.start_epoch:
            return

        span = max(1, self.end_epoch - self.start_epoch + 1)
        progress = min(1.0, (epoch - self.start_epoch + 1) / span)
        target_frozen = int(round(self.initial_frozen_layers * (1.0 - progress)))

        if target_frozen >= self.current_frozen_layers:
            self._ensure_targets_unfrozen(pl_module)
            return

        # Unfreeze newly released lower encoder layers.
        for idx in range(target_frozen, self.current_frozen_layers):
            set_module_trainable(self.layers[idx], True)

        self.current_frozen_layers = target_frozen
        self._ensure_targets_unfrozen(pl_module)

        print(
            f"[GradualUnfreeze] epoch={epoch} unfroze layers; "
            f"frozen_lower_layers_now={self.current_frozen_layers}/{len(self.layers)}"
        )


def apply_freezing(model: ASRModel, cfg: Dict[str, Any]) -> tuple[Optional[FreezeState], Optional[GradualUnfreezeCallback]]:
    freeze_cfg = cfg.get("freeze", {}) or {}
    strategy = str(freeze_cfg.get("strategy", "full_finetune")).lower()

    for p in model.parameters():
        p.requires_grad = True

    if strategy == "full_finetune":
        return FreezeState(strategy=strategy, initial_frozen_layers=0, current_frozen_layers=0), None

    if strategy not in {"freeze_lower", "gradual_unfreeze"}:
        raise ValueError(f"Unknown freeze.strategy: {strategy}")

    layers = find_encoder_layers(model)
    if not layers:
        print("Warning: Could not find encoder layer list; skipping lower-layer freezing.")
        return FreezeState(strategy=strategy, initial_frozen_layers=0, current_frozen_layers=0), None

    lower_fraction = float(freeze_cfg.get("lower_encoder_fraction", 0.7))
    min_layers = int(freeze_cfg.get("min_layers_to_freeze", 0))
    num_layers = len(layers)

    freeze_count = max(min_layers, int(round(num_layers * lower_fraction)))
    freeze_count = max(0, min(freeze_count, num_layers))

    # Freeze full encoder first; then unfreeze upper encoder layers.
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        set_module_trainable(encoder, False)

    for idx in range(freeze_count, num_layers):
        set_module_trainable(layers[idx], True)

    target_modules = freeze_cfg.get("target_modules_unfrozen", ["decoder", "joint", "aux_ctc"])
    for name in target_modules:
        mod = get_module_by_path(model, str(name))
        if mod is not None:
            set_module_trainable(mod, True)

    freeze_state = FreezeState(strategy=strategy, initial_frozen_layers=freeze_count, current_frozen_layers=freeze_count)

    if strategy == "freeze_lower":
        print(f"Applied freeze_lower: froze lower {freeze_count}/{num_layers} encoder layers.")
        return freeze_state, None

    gradual_cfg = freeze_cfg.get("gradual_unfreeze", {}) or {}
    callback = GradualUnfreezeCallback(
        layers=layers,
        initial_frozen_layers=freeze_count,
        start_epoch=int(gradual_cfg.get("start_epoch", 1)),
        end_epoch=int(gradual_cfg.get("end_epoch", 8)),
        target_modules=[str(x) for x in target_modules],
    )
    print(
        "Applied gradual_unfreeze: "
        f"initially froze lower {freeze_count}/{num_layers} encoder layers; "
        f"schedule epochs {callback.start_epoch}->{callback.end_epoch}."
    )
    return freeze_state, callback


def count_trainable_params(model: ASRModel) -> Dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


def resolve_pretrained_model_class(pretrained_name: str) -> Optional[type[ASRModel]]:
    infos = model_utils.resolve_subclass_pretrained_model_info(ASRModel)
    for info in infos:
        name = getattr(info, "pretrained_model_name", None)
        if name != pretrained_name:
            continue

        cls = getattr(info, "class_", None)
        if cls is None:
            class_path = getattr(info, "class_path", None)
            if class_path:
                try:
                    cls = model_utils.import_class_by_path(class_path)
                except Exception:
                    cls = None

        if isinstance(cls, type) and issubclass(cls, ASRModel):
            return cls
    return None


def load_base_model(cfg: Dict[str, Any], trainer: pl.Trainer) -> ASRModel:
    model_cfg = cfg.get("model", {}) or {}
    pretrained_name = model_cfg.get("pretrained_name")
    init_from_nemo = model_cfg.get("init_from_nemo")

    if pretrained_name and init_from_nemo:
        raise ValueError("Set only one of model.pretrained_name or model.init_from_nemo")
    if not pretrained_name and not init_from_nemo:
        raise ValueError("One of model.pretrained_name or model.init_from_nemo must be set")

    if init_from_nemo:
        model = ASRModel.restore_from(restore_path=str(init_from_nemo), map_location="cpu")
    else:
        model_name = str(pretrained_name)
        try:
            model = ASRModel.from_pretrained(model_name=model_name)
        except TypeError as exc:
            if "abstract class ASRModel" not in str(exc):
                raise

            cls = resolve_pretrained_model_class(model_name)
            if cls is None:
                raise RuntimeError(
                    f"Failed to resolve concrete ASR model class for '{model_name}'. "
                    "Please set model.init_from_nemo to a concrete .nemo checkpoint "
                    "or use a supported pretrained model name."
                ) from exc

            print(
                f"ASRModel.from_pretrained failed for '{model_name}' due to abstract base class; "
                f"retrying with concrete class {cls.__name__}."
            )
            model = cls.from_pretrained(model_name=model_name)

    model.set_trainer(trainer)
    return model


def update_tokenizer_if_requested(model: ASRModel, cfg: Dict[str, Any]) -> None:
    tok_cfg = cfg.get("tokenizer", {}) or {}
    if not bool(tok_cfg.get("update_tokenizer", False)):
        return

    tok_dir = tok_cfg.get("dir")
    tok_type = str(tok_cfg.get("type", "bpe"))
    keep_decoder_if_same = bool(tok_cfg.get("keep_decoder_if_same_vocab", True))

    if tok_dir is None:
        raise ValueError("tokenizer.update_tokenizer is true but tokenizer.dir is missing")
    tok_dir = Path(tok_dir)
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tok_dir}")

    prev_vocab = getattr(getattr(model, "tokenizer", None), "vocab_size", None)
    decoder_state = model.decoder.state_dict() if hasattr(model, "decoder") else None
    joint_state = model.joint.state_dict() if hasattr(model, "joint") else None

    model.change_vocabulary(new_tokenizer_dir=str(tok_dir), new_tokenizer_type=tok_type)

    new_vocab = getattr(getattr(model, "tokenizer", None), "vocab_size", None)
    if keep_decoder_if_same and prev_vocab is not None and new_vocab == prev_vocab:
        if decoder_state is not None:
            model.decoder.load_state_dict(decoder_state)
        if joint_state is not None:
            model.joint.load_state_dict(joint_state)
        print("Tokenizer replaced with same vocab size; decoder/joint weights restored.")
    else:
        print("Tokenizer replaced and decoder/joint heads remain reinitialized for new vocabulary.")


def override_rnnt_loss_if_requested(model: ASRModel, cfg: Dict[str, Any]) -> None:
    """
    Optionally override RNNT loss backend from train config.

    Example config:
      rnnt_loss:
        name: tdt_pytorch
        kwargs: null
        copy_existing_tdt_kwargs_for_tdt_pytorch: true
    """
    override_cfg = cfg.get("rnnt_loss", {}) or {}
    loss_name_raw = override_cfg.get("name")
    if not loss_name_raw:
        return

    if not hasattr(model, "extract_rnnt_loss_cfg") or not hasattr(model, "joint"):
        print("RNNT loss override requested, but loaded model does not expose RNNT loss hooks; skipping override.")
        return

    loss_name = str(loss_name_raw)
    loss_kwargs = override_cfg.get("kwargs")

    # For TDT models, keep duration-related kwargs when switching to tdt_pytorch unless explicitly provided.
    if (
        loss_kwargs is None
        and loss_name == "tdt_pytorch"
        and bool(override_cfg.get("copy_existing_tdt_kwargs_for_tdt_pytorch", True))
    ):
        existing_loss_cfg = model.cfg.get("loss", {}) if hasattr(model, "cfg") else {}
        loss_kwargs = existing_loss_cfg.get("tdt_kwargs", None)

    if not isinstance(loss_kwargs, (dict, type(None))):
        raise ValueError("rnnt_loss.kwargs must be an object/dict or null.")

    with open_dict(model.cfg):
        if model.cfg.get("loss", None) is None:
            model.cfg.loss = OmegaConf.create({})
        model.cfg.loss.loss_name = loss_name
        if loss_kwargs is not None:
            model.cfg.loss[f"{loss_name}_kwargs"] = loss_kwargs

    resolved_loss_name, resolved_loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))

    num_classes = model.joint.num_classes_with_blank - 1
    if resolved_loss_name in {"tdt", "tdt_pytorch"}:
        num_classes = num_classes - model.joint.num_extra_outputs

    model.loss = RNNTLoss(
        num_classes=num_classes,
        loss_name=resolved_loss_name,
        loss_kwargs=resolved_loss_kwargs,
        reduction=model.cfg.get("rnnt_reduction", "mean_batch"),
    )

    # If fused joint-loss path is active, wire the newly created loss object.
    if bool(getattr(model.joint, "fuse_loss_wer", False)):
        model.joint.set_loss(model.loss)

    # Keep decoding config aligned with selected loss (e.g., TDT durations).
    if hasattr(model, "set_decoding_type_according_to_loss"):
        model.cfg.decoding = model.set_decoding_type_according_to_loss(model.cfg.decoding)

    print(f"RNNT loss override applied: {resolved_loss_name} (kwargs={resolved_loss_kwargs})")


def setup_datasets(model: ASRModel, cfg: Dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {}) or {}

    train_manifest = Path(data_cfg.get("train_manifest", ""))
    val_manifest = Path(data_cfg.get("val_manifest", ""))
    test_manifest_raw = data_cfg.get("test_manifest")

    if not train_manifest.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Val manifest not found: {val_manifest}")

    common = {
        "sample_rate": int(data_cfg.get("sample_rate", 16000)),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "min_duration": float(data_cfg.get("min_duration", 0.1)),
        "max_duration": float(data_cfg.get("max_duration", 30.0)),
    }

    train_ds = {
        **common,
        "manifest_filepath": str(train_manifest),
        "batch_size": int(data_cfg.get("train_batch_size", 8)),
        "shuffle": True,
    }
    val_ds = {
        **common,
        "manifest_filepath": str(val_manifest),
        "batch_size": int(data_cfg.get("val_batch_size", 8)),
        "shuffle": False,
    }

    model.setup_training_data(OmegaConf.create(train_ds))
    if hasattr(model, "setup_multiple_validation_data"):
        model.setup_multiple_validation_data(OmegaConf.create(val_ds))
    else:
        model.setup_validation_data(OmegaConf.create(val_ds))

    if test_manifest_raw:
        test_manifest = Path(test_manifest_raw)
        if test_manifest.exists():
            test_ds = {
                **common,
                "manifest_filepath": str(test_manifest),
                "batch_size": int(data_cfg.get("test_batch_size", 8)),
                "shuffle": False,
            }
            if hasattr(model, "setup_multiple_test_data"):
                model.setup_multiple_test_data(OmegaConf.create(test_ds))
            else:
                model.setup_test_data(OmegaConf.create(test_ds))


def setup_spec_augment(model: ASRModel, cfg: Dict[str, Any]) -> None:
    spec_cfg = cfg.get("spec_augment", {}) or {}
    if not bool(spec_cfg.get("enabled", False)):
        return

    config = spec_cfg.get("config")
    if not config:
        raise ValueError("spec_augment.enabled is true but spec_augment.config is empty")
    model.spec_augment = ASRModel.from_config_dict(OmegaConf.create(config))


def build_callbacks(cfg: Dict[str, Any], has_logger: bool) -> tuple[List[Callback], ModelCheckpoint]:
    callbacks: List[Callback] = []

    ckpt_cfg = cfg.get("checkpointing", {}) or {}
    ckpt_dir = Path(str(ckpt_cfg.get("dirpath", "artifacts/checkpoints")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=str(ckpt_cfg.get("filename", "asr-{epoch:02d}-{val_wer:.4f}")),
        monitor=str(ckpt_cfg.get("monitor", "val_wer")),
        mode=str(ckpt_cfg.get("mode", "min")),
        save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
        save_last=bool(ckpt_cfg.get("save_last", True)),
    )
    callbacks.append(checkpoint_callback)

    es_cfg = cfg.get("early_stopping", {}) or {}
    if bool(es_cfg.get("enabled", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(es_cfg.get("monitor", "val_wer")),
                mode=str(es_cfg.get("mode", "min")),
                patience=int(es_cfg.get("patience", 8)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
            )
        )

    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks, checkpoint_callback


def export_nemo_artifacts(model: ASRModel, cfg: Dict[str, Any], checkpoint_callback: ModelCheckpoint) -> Dict[str, Any]:
    artifacts_cfg = cfg.get("artifacts", {}) or {}
    out: Dict[str, Any] = {
        "best_checkpoint": checkpoint_callback.best_model_path,
        "last_checkpoint": checkpoint_callback.last_model_path,
    }

    last_nemo = artifacts_cfg.get("save_last_nemo")
    if last_nemo:
        last_nemo_path = Path(str(last_nemo))
        last_nemo_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_to(str(last_nemo_path))
        out["last_nemo"] = str(last_nemo_path)

    best_nemo = artifacts_cfg.get("save_best_nemo")
    if best_nemo and checkpoint_callback.best_model_path:
        best_nemo_path = Path(str(best_nemo))
        best_nemo_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            best_model = model.__class__.load_from_checkpoint(checkpoint_callback.best_model_path, map_location="cpu")
            best_model.save_to(str(best_nemo_path))
            out["best_nemo"] = str(best_nemo_path)
        except Exception as exc:
            # TODO: Some NeMo versions may require a model-specific restore path for ckpt -> .nemo export.
            out["best_nemo_export_error"] = str(exc)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune NeMo ASR model for Mongolian speech.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Path to train YAML config.")
    parser.add_argument("--train-manifest", type=Path, default=None, help="Override train manifest path.")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Override val manifest path.")
    parser.add_argument("--test-manifest", type=Path, default=None, help="Override test manifest path.")
    parser.add_argument(
        "--freeze-strategy",
        type=str,
        default=None,
        choices=["full_finetune", "freeze_lower", "gradual_unfreeze"],
        help="Override freeze strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.train_manifest is not None:
        cfg.setdefault("data", {})["train_manifest"] = str(args.train_manifest)
    if args.val_manifest is not None:
        cfg.setdefault("data", {})["val_manifest"] = str(args.val_manifest)
    if args.test_manifest is not None:
        cfg.setdefault("data", {})["test_manifest"] = str(args.test_manifest)
    if args.freeze_strategy is not None:
        cfg.setdefault("freeze", {})["strategy"] = args.freeze_strategy

    seed_all(int(cfg.get("seed", 42)))

    logger = build_logger(cfg)
    callbacks, checkpoint_callback = build_callbacks(cfg, has_logger=bool(logger))

    trainer_cfg = dict(cfg.get("trainer", {}) or {})
    trainer_cfg["callbacks"] = callbacks
    trainer_cfg["logger"] = logger

    trainer = pl.Trainer(**trainer_cfg)

    model = load_base_model(cfg, trainer)
    update_tokenizer_if_requested(model, cfg)
    override_rnnt_loss_if_requested(model, cfg)
    setup_datasets(model, cfg)
    model.setup_optimization(OmegaConf.create(cfg.get("optim", {})))
    setup_spec_augment(model, cfg)

    freeze_state, gradual_callback = apply_freezing(model, cfg)
    if gradual_callback is not None:
        trainer.callbacks.append(gradual_callback)

    counts = count_trainable_params(model)
    print(
        f"Trainable parameters: {counts['trainable']:,} / {counts['total']:,} "
        f"({counts['trainable'] / counts['total']:.2%})"
    )

    trainer.fit(model)

    test_manifest = (cfg.get("data", {}) or {}).get("test_manifest")
    if test_manifest and Path(test_manifest).exists():
        try:
            trainer.test(model)
        except Exception as exc:
            print(f"Warning: trainer.test failed: {exc}")

    artifact_summary = export_nemo_artifacts(model, cfg, checkpoint_callback)

    summary = {
        "freeze_state": None if freeze_state is None else freeze_state.__dict__,
        "checkpoints": {
            "best": checkpoint_callback.best_model_path,
            "last": checkpoint_callback.last_model_path,
            "best_score": float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None,
        },
        "artifacts": artifact_summary,
        "trainable_params": counts,
    }

    summary_path = Path("artifacts/checkpoints/training_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training finished.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    if "best_nemo" in artifact_summary:
        print(f"Best .nemo:      {artifact_summary['best_nemo']}")
    if "last_nemo" in artifact_summary:
        print(f"Last .nemo:      {artifact_summary['last_nemo']}")
    if "best_nemo_export_error" in artifact_summary:
        print(f"Best .nemo export warning: {artifact_summary['best_nemo_export_error']}")
    print(f"Summary JSON:    {summary_path}")


if __name__ == "__main__":
    main()
