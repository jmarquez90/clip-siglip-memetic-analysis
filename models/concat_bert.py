"""BERT + ResNet-152 concatenation baseline for hateful-meme classification.

This module implements a literature baseline as a standalone model family. It
does not depend on CLIP or SigLIP embeddings and should not be presented as an
original architecture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet152_Weights, resnet152
from transformers import AutoModel, AutoTokenizer

TEXT_MODEL_ID = "google-bert/bert-base-uncased"
IMAGE_MODEL_ID = "torchvision/resnet152@IMAGENET1K_V2"


@dataclass(frozen=True)
class ConcatBertConfig:
    """Training and architecture settings for the Concat BERT baseline."""

    text_model_id: str = TEXT_MODEL_ID
    image_model_id: str = IMAGE_MODEL_ID
    text_feature_dim: int = 300
    image_feature_dim: int = 300
    fusion_output_dim: int = 256
    dropout: float = 0.1
    max_length: int = 128
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    max_epochs: int = 10
    patience: int = 3


@dataclass
class ConcatBertRun:
    """Outputs from one fitted Concat BERT model."""

    model: torch.nn.Module
    test_probabilities: np.ndarray
    metrics: dict[str, Any]
    history: list[dict[str, float | int]]
    artifact_path: Path


class HatefulMemesConcatDataset(Dataset):
    """Load paired text and image examples from an in-memory split frame."""

    def __init__(self, frame: Any, dataset_dir: Path) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.dataset_dir / Path(row["img"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return {
            "id": int(row["id"]),
            "image": image,
            "text": str(row["text"]),
            "label": int(row["label"]),
        }


class ConcatBertClassifier(torch.nn.Module):
    """Fine-tune BERT and ResNet-152 before mid-level concatenation."""

    def __init__(self, config: ConcatBertConfig) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config.text_model_id)
        text_hidden_size = self.text_encoder.config.hidden_size

        self.image_encoder = resnet152(
            weights=ResNet152_Weights.IMAGENET1K_V2
        )
        image_hidden_size = self.image_encoder.fc.in_features
        self.image_encoder.fc = torch.nn.Identity()

        self.text_projection = torch.nn.Linear(
            text_hidden_size, config.text_feature_dim
        )
        self.image_projection = torch.nn.Linear(
            image_hidden_size, config.image_feature_dim
        )
        combined_dim = config.text_feature_dim + config.image_feature_dim
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, config.fusion_output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.fusion_output_dim, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = torch.relu(
            self.text_projection(text_output.last_hidden_state[:, 0])
        )
        image_features = torch.relu(
            self.image_projection(self.image_encoder(pixel_values))
        )
        combined_features = torch.cat(
            [text_features, image_features], dim=1
        )
        return self.classifier(combined_features)


def _build_collate(config: ConcatBertConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_id)
    image_transform = ResNet152_Weights.IMAGENET1K_V2.transforms()

    def collate(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        text_inputs = tokenizer(
            [example["text"] for example in examples],
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        return {
            "ids": torch.as_tensor(
                [example["id"] for example in examples], dtype=torch.long
            ),
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": torch.stack(
                [image_transform(example["image"]) for example in examples]
            ),
            "labels": torch.as_tensor(
                [example["label"] for example in examples], dtype=torch.long
            ),
        }

    return collate


def _loader(
    frame: Any,
    dataset_dir: Path,
    config: ConcatBertConfig,
    collate,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        HatefulMemesConcatDataset(frame, dataset_dir),
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(
            device, non_blocking=True
        ),
        pixel_values=batch["pixel_values"].to(device, non_blocking=True),
    )


@torch.inference_mode()
def _probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probability_batches = []
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = _forward_batch(model, batch, device)
        probability_batches.append(
            torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        )
    return np.concatenate(probability_batches)


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def _step_optimizer_if_due(
    *,
    batch_index: int,
    batch_count: int,
    accumulation_steps: int,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
) -> None:
    step_is_due = (
        batch_index % accumulation_steps == 0
        or batch_index == batch_count
    )
    if not step_is_due:
        return
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_concat_bert_baseline(
    *,
    train_frame: Any,
    dev_frame: Any,
    test_frame: Any,
    dataset_dir: str | Path,
    artifact_dir: str | Path,
    device: torch.device,
    seed: int,
    config: ConcatBertConfig | None = None,
) -> ConcatBertRun:
    """Train, select on development ROC-AUC, evaluate, and save the baseline."""

    config = config or ConcatBertConfig()
    dataset_dir = Path(dataset_dir)
    artifact_dir = Path(artifact_dir)
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    collate = _build_collate(config)
    train_loader = _loader(
        train_frame, dataset_dir, config, collate, True, seed
    )
    dev_loader = _loader(
        dev_frame, dataset_dir, config, collate, False, seed
    )
    test_loader = _loader(
        test_frame, dataset_dir, config, collate, False, seed
    )
    model = ConcatBertClassifier(config).to(device)

    train_labels = train_frame["label"].to_numpy()
    positive_count = np.count_nonzero(train_labels == 1)
    negative_count = np.count_nonzero(train_labels == 0)
    class_weights = torch.tensor(
        [1.0, negative_count / positive_count],
        dtype=torch.float32,
        device=device,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )

    best_dev_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        example_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = _forward_batch(model, batch, device)
                batch_loss = loss_function(logits, labels)
                group_start = (
                    (batch_index - 1) // config.gradient_accumulation_steps
                ) * config.gradient_accumulation_steps
                group_size = min(
                    config.gradient_accumulation_steps,
                    len(train_loader) - group_start,
                )
                loss = batch_loss / group_size
            scaler.scale(loss).backward()
            _step_optimizer_if_due(
                batch_index=batch_index,
                batch_count=len(train_loader),
                accumulation_steps=config.gradient_accumulation_steps,
                scaler=scaler,
                optimizer=optimizer,
                model=model,
            )
            running_loss += batch_loss.item() * len(labels)
            example_count += len(labels)

        dev_probabilities = _probabilities(model, dev_loader, device)
        dev_auc = float(
            roc_auc_score(dev_frame["label"], dev_probabilities)
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(running_loss / example_count),
            "dev_roc_auc": dev_auc,
        }
        history.append(epoch_record)
        print(
            f"Concat BERT epoch {epoch}: "
            f"loss={epoch_record['train_loss']:.4f}, "
            f"dev ROC-AUC={dev_auc:.4f}"
        )
        if dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Concat BERT training did not produce a checkpoint")

    model.load_state_dict(best_state)
    test_probabilities = _probabilities(model, test_loader, device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    artifact_path = model_dir / "concat-bert--resnet152.pt"
    artifact_config = {
        "model": "Concat BERT",
        "family": "Literature baseline",
        "fusion": "concat",
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "dev_roc_auc": float(best_dev_auc),
        "trainable_parameters": int(trainable_parameters),
        **asdict(config),
    }
    torch.save(
        {
            "state_dict": best_state,
            "config": artifact_config,
            "history": history,
        },
        artifact_path,
    )
    artifact_path.with_suffix(".json").write_text(
        json.dumps(artifact_config, indent=2), encoding="utf-8"
    )

    metrics: dict[str, Any] = _classification_metrics(
        test_frame["label"].to_numpy(), test_probabilities
    )
    metrics.update(
        {
            "model": "Concat BERT",
            "family": "Literature baseline",
            "fusion": "concat",
            "dev_roc_auc": float(best_dev_auc),
            "best_epoch": int(best_epoch),
            "trainable_parameters": int(trainable_parameters),
            "artifact": artifact_path.name,
        }
    )
    return ConcatBertRun(
        model=model.cpu(),
        test_probabilities=test_probabilities,
        metrics=metrics,
        history=history,
        artifact_path=artifact_path,
    )
