"""Train the RGB baseline detector."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from augmentations import (
    ROBUSTNESS_TRANSFORMS,
    build_eval_transform,
    build_paired_train_transform,
    build_train_transform,
)
from data_pipeline import (
    ManifestImageDataset,
    StreamingSIDDataset,
    create_balanced_sampler,
)
from losses import binary_classification_loss, difficulty_aware_consistency_loss
from metrics import collect_predictions, compute_robustness_metrics
from model import build_model, count_parameters
from utils import load_config, resolve_project_path, select_device, set_seed, worker_seed


def build_loader(
    dataset: ManifestImageDataset | StreamingSIDDataset,
    *,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    balanced: bool,
    sampling_columns: tuple[str, ...] = ("label",),
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    iterable = isinstance(dataset, IterableDataset)
    sampler = (
        create_balanced_sampler(dataset, columns=sampling_columns, seed=seed)
        if balanced and not iterable
        else None
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None and not iterable,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_seed,
        generator=generator,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    gradient_clip: float,
    amp_enabled: bool,
    max_batches: int | None,
    paired_views: bool,
    consistency_weight: float,
    consistency_alpha: float,
) -> dict[str, float]:
    model.train()
    running_total_loss = 0.0
    running_classification_loss = 0.0
    running_consistency_loss = 0.0
    processed_samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if paired_views:
                clean_images = batch["clean_image"].to(device, non_blocking=True)
                degraded_images = batch["degraded_image"].to(
                    device, non_blocking=True
                )
                combined_images = torch.cat([clean_images, degraded_images], dim=0)
                combined_logits = model(combined_images)
                clean_logits, degraded_logits = combined_logits.chunk(2, dim=0)
                classification_loss = 0.5 * (
                    binary_classification_loss(clean_logits, labels)
                    + binary_classification_loss(degraded_logits, labels)
                )
                consistency_loss = difficulty_aware_consistency_loss(
                    clean_logits,
                    degraded_logits,
                    alpha=consistency_alpha,
                )
                loss = classification_loss + consistency_weight * consistency_loss
                batch_size = clean_images.shape[0]
            else:
                images = batch["image"].to(device, non_blocking=True)
                logits = model(images)
                classification_loss = binary_classification_loss(logits, labels)
                consistency_loss = torch.zeros((), device=device)
                loss = classification_loss
                batch_size = images.shape[0]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        running_total_loss += float(loss.detach()) * batch_size
        running_classification_loss += (
            float(classification_loss.detach()) * batch_size
        )
        running_consistency_loss += float(consistency_loss.detach()) * batch_size
        processed_samples += batch_size
    if processed_samples == 0:
        raise ValueError("No training batches were produced.")
    return {
        "total": running_total_loss / processed_samples,
        "classification": running_classification_loss / processed_samples,
        "consistency": running_consistency_loss / processed_samples,
        "processed_samples": float(processed_samples),
    }


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rgb_baseline.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume model, optimizer, scheduler, and scaler state from a checkpoint.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        help="Initialize model weights only and begin a new training run at epoch 1.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Avoid downloading/loading backbone weights (useful for smoke tests).",
    )
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive.")

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = select_device(args.device)
    data_config = config["data"]
    training_config = config["training"]
    image_size = int(data_config["image_size"])
    epochs = args.epochs or int(training_config["epochs"])

    paired_views = bool(training_config.get("paired_views", False))
    train_transform = (
        build_paired_train_transform(image_size)
        if paired_views
        else build_train_transform(image_size)
    )
    train_limit = (
        args.limit_train_samples
        if args.limit_train_samples is not None
        else data_config.get("train_limit")
    )
    validation_limit = (
        args.limit_val_samples
        if args.limit_val_samples is not None
        else data_config.get("val_limit")
    )
    data_source = str(data_config.get("source", "manifest"))
    if data_source == "sid_streaming":
        train_samples = (
            int(train_limit)
            if train_limit is not None
            else int(data_config["train_samples_per_epoch"])
        )
        validation_samples = (
            int(validation_limit)
            if validation_limit is not None
            else int(
                data_config.get(
                    "val_samples_per_transform",
                    data_config["val_samples_per_epoch"],
                )
            )
        )
        dataset_id = str(data_config.get("dataset_id", "saberzl/SID_Set"))
        train_dataset = StreamingSIDDataset(
            train_transform,
            split="train",
            samples_per_epoch=train_samples,
            dataset_id=dataset_id,
            shuffle_buffer=int(data_config.get("shuffle_buffer", 512)),
            seed=seed,
        )
        validation_dataset_factory = lambda transform_name: StreamingSIDDataset(
            build_eval_transform(image_size, transform_name),
            split="validation",
            samples_per_epoch=validation_samples,
            dataset_id=dataset_id,
            shuffle_buffer=1,
            seed=seed,
        )
    elif data_source == "manifest":
        train_dataset = ManifestImageDataset(
            data_config["train_manifest"],
            train_transform,
            limit=int(train_limit) if train_limit is not None else None,
        )
        validation_dataset_factory = lambda transform_name: ManifestImageDataset(
            data_config["val_manifest"],
            build_eval_transform(image_size, transform_name),
            limit=int(validation_limit) if validation_limit is not None else None,
        )
    else:
        raise ValueError(f"Unsupported data.source: {data_source}")
    loader_arguments = {
        "batch_size": (
            args.batch_size
            if args.batch_size is not None
            else int(training_config["batch_size"])
        ),
        "num_workers": int(data_config.get("num_workers", 0)),
        "persistent_workers": bool(data_config.get("persistent_workers", True)),
        "prefetch_factor": int(data_config.get("prefetch_factor", 2)),
        "device": device,
        "seed": seed,
    }
    train_loader = build_loader(
        train_dataset,
        balanced=bool(data_config.get("balanced_sampling", True)),
        sampling_columns=tuple(data_config.get("sampling_columns", ["label"])),
        shuffle=True,
        **loader_arguments,
    )
    validation_transform_names = tuple(
        training_config.get("validation_transforms", ["clean"])
    )
    if not validation_transform_names:
        raise ValueError("training.validation_transforms cannot be empty.")
    unknown_validation_transforms = set(validation_transform_names) - set(
        ROBUSTNESS_TRANSFORMS
    )
    if unknown_validation_transforms:
        raise ValueError(
            "Unknown validation transforms: "
            f"{sorted(unknown_validation_transforms)}"
        )
    if len(set(validation_transform_names)) != len(validation_transform_names):
        raise ValueError("training.validation_transforms contains duplicates.")
    validation_datasets = {
        transform_name: validation_dataset_factory(transform_name)
        for transform_name in validation_transform_names
    }
    validation_loaders = {
        transform_name: build_loader(
            validation_dataset,
            balanced=False,
            sampling_columns=("label",),
            shuffle=False,
            **{
                **loader_arguments,
                "num_workers": int(
                    data_config.get(
                        "validation_num_workers",
                        data_config.get("num_workers", 0),
                    )
                ),
                "persistent_workers": bool(
                    data_config.get("validation_persistent_workers", False)
                ),
                "prefetch_factor": int(
                    data_config.get(
                        "validation_prefetch_factor",
                        data_config.get("prefetch_factor", 2),
                    )
                ),
            },
        )
        for transform_name, validation_dataset in validation_datasets.items()
    }

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(
            training_config.get("cudnn_benchmark", True)
        )
        if bool(training_config.get("allow_tf32", True)):
            torch.set_float32_matmul_precision("high")

    # A checkpoint contains the complete backbone state, so fetching pretrained
    # timm weights first only adds a redundant network request and startup cost.
    loading_checkpoint = bool(args.init_checkpoint or args.resume)
    pretrained_override = (
        False if args.no_pretrained or loading_checkpoint else None
    )
    model = build_model(config, pretrained=pretrained_override).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            backbone_lr=float(training_config["backbone_lr"]),
            head_lr=float(training_config["head_lr"]),
        ),
        weight_decay=float(training_config["weight_decay"]),
        fused=bool(training_config.get("fused_optimizer", True))
        and device.type == "cuda",
    )
    warmup_epochs = int(training_config.get("warmup_epochs", 1))

    def learning_rate_factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=float(training_config.get("amp_init_scale", 65536.0)),
    )
    checkpoint_directory = resolve_project_path(config["output"]["checkpoint_dir"])
    log_directory = resolve_project_path(config["output"]["log_dir"])
    writer = SummaryWriter(log_dir=log_directory)

    best_validation_score = float("-inf")
    best_worst_auc = float("-inf")
    epochs_without_improvement = 0
    patience = int(training_config.get("early_stopping_patience", 3))
    start_epoch = 0
    if args.init_checkpoint:
        initialization_path = resolve_project_path(args.init_checkpoint)
        initialization = torch.load(
            initialization_path, map_location=device, weights_only=False
        )
        model.load_state_dict(initialization["model_state"])
        print(f"Initialized model weights from {initialization_path}")
    if args.resume:
        resume_path = resolve_project_path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_score = float(
            checkpoint.get(
                "best_validation_score",
                checkpoint.get(
                    "best_validation_auc",
                    checkpoint.get("validation_metrics", {}).get(
                        "auroc", float("-inf")
                    ),
                ),
            )
        )
        best_worst_auc = float(
            checkpoint.get("best_validation_worst_auc", float("-inf"))
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(
            f"Resumed {resume_path} at epoch {start_epoch + 1} "
            f"with best validation score {best_validation_score:.4f}"
        )
    print(
        f"Training on {device} | train={len(train_dataset):,} | "
        f"val_per_transform={len(next(iter(validation_datasets.values()))):,} | "
        f"mode={model.training_mode} | "
        f"parameters={count_parameters(model):,} | "
        f"trainable={count_parameters(model, trainable_only=True):,} | "
        f"validation_transforms={','.join(validation_transform_names)}"
    )
    if model.training_mode == "lora":
        print(f"LoRA layers: {len(model.lora_layers)}")

    if start_epoch >= epochs:
        raise ValueError(
            f"Checkpoint has already completed {start_epoch} epochs, "
            f"but the run is configured for {epochs}. Increase --epochs."
        )

    for epoch in range(start_epoch, epochs):
        if isinstance(train_dataset, StreamingSIDDataset):
            train_dataset.set_epoch(epoch)
        start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        consistency_warmup = int(
            training_config.get("consistency_warmup_epochs", 0)
        )
        target_consistency_weight = float(
            training_config.get("consistency_weight", 0.0)
        )
        if consistency_warmup > 0:
            consistency_weight = target_consistency_weight * min(
                max(epoch / consistency_warmup, 0.0), 1.0
            )
        else:
            consistency_weight = target_consistency_weight
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            gradient_clip=float(training_config.get("gradient_clip", 1.0)),
            amp_enabled=amp_enabled,
            max_batches=args.max_train_batches,
            paired_views=paired_views,
            consistency_weight=consistency_weight,
            consistency_alpha=float(training_config.get("consistency_alpha", 1.0)),
        )
        training_elapsed = time.perf_counter() - start_time
        predictions_by_transform = {}
        for transform_name, validation_loader in validation_loaders.items():
            labels, probabilities, _ = collect_predictions(
                model,
                validation_loader,
                device,
                max_batches=args.max_val_batches,
            )
            predictions_by_transform[transform_name] = (labels, probabilities)
        validation_robustness = compute_robustness_metrics(
            predictions_by_transform
        )
        threshold = float(validation_robustness["threshold"])
        validation_metrics = validation_robustness["aggregate"]
        validation_score = float(validation_robustness["selection_score"])
        validation_worst_auc = float(validation_robustness["worst_auroc"])
        scheduler.step()

        writer.add_scalar("loss/train_total", train_losses["total"], epoch)
        writer.add_scalar(
            "loss/train_classification", train_losses["classification"], epoch
        )
        writer.add_scalar(
            "loss/train_consistency", train_losses["consistency"], epoch
        )
        writer.add_scalar(
            "loss/consistency_weight", consistency_weight, epoch
        )
        writer.add_scalar("metrics/val_auroc", validation_metrics["auroc"], epoch)
        writer.add_scalar("metrics/val_f1", validation_metrics["f1"], epoch)
        writer.add_scalar(
            "metrics/val_selection_score", validation_score, epoch
        )
        writer.add_scalar(
            "metrics/val_worst_auroc", validation_worst_auc, epoch
        )
        for transform_name, transform_metrics in validation_robustness[
            "transforms"
        ].items():
            writer.add_scalar(
                f"metrics/val_auroc_{transform_name}",
                transform_metrics["auroc"],
                epoch,
            )
        for parameter_group in optimizer.param_groups:
            writer.add_scalar(
                f"learning_rate/{parameter_group.get('name', 'parameters')}",
                parameter_group["lr"],
                epoch,
            )

        is_best = validation_score > best_validation_score or (
            math.isclose(validation_score, best_validation_score)
            and validation_worst_auc > best_worst_auc
        )
        if is_best:
            best_validation_score = validation_score
            best_worst_auc = validation_worst_auc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": config,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
            "validation_robustness": validation_robustness,
            "best_validation_score": best_validation_score,
            "best_validation_worst_auc": best_worst_auc,
            # Retained for compatibility with earlier checkpoint readers.
            "best_validation_auc": best_validation_score,
            "epochs_without_improvement": epochs_without_improvement,
            "training_mode": model.training_mode,
            "total_parameters": count_parameters(model),
            "trainable_parameters": count_parameters(model, trainable_only=True),
        }
        save_checkpoint(checkpoint, checkpoint_directory / "last.pt")
        if is_best:
            checkpoint_name = config["output"].get(
                "checkpoint_name", "rgb_baseline_best.pt"
            )
            save_checkpoint(checkpoint, checkpoint_directory / checkpoint_name)

        elapsed = time.perf_counter() - start_time
        throughput = train_losses["processed_samples"] / training_elapsed
        memory_summary = ""
        if device.type == "cuda":
            peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
            memory_summary = (
                f" peak_allocated_gib={peak_allocated_gib:.2f}"
                f" peak_reserved_gib={peak_reserved_gib:.2f}"
            )
        print(
            f"epoch={epoch + 1}/{epochs} loss={train_losses['total']:.4f} "
            f"cls={train_losses['classification']:.4f} "
            f"cons={train_losses['consistency']:.4f} "
            f"val_score={validation_score:.4f} "
            f"val_worst_auc={validation_worst_auc:.4f} "
            f"val_f1={validation_metrics['f1']:.4f} "
            f"threshold={threshold:.3f} seconds={elapsed:.1f} "
            f"train_seconds={training_elapsed:.1f} "
            f"train_samples_per_second={throughput:.1f}{memory_summary}"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    writer.close()
    print(f"Best validation score: {best_validation_score:.4f}")


if __name__ == "__main__":
    main()
