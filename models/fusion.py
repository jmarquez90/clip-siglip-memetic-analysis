"""Fusion-specific linear probes and direct-classifier ablations.

Each fusion strategy is trained as an independent downstream classifier and
saved as its own checkpoint. The primary models use linear readouts over fixed
fusion maps; projected nonlinear heads provide matched no-LPE ablations. The
CLIP and SigLIP encoders remain frozen throughout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

FUSION_TYPES = ("concat", "align", "cross")


@dataclass(frozen=True)
class FusionConfig:
    """Training and architecture settings shared by all fusion heads."""

    projection_dim: int = 64
    hidden_dim: int = 128
    mapping_dropout: float = 0.1
    fusion_dropout: float = 0.4
    classifier_dropout: float = 0.2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 6


@dataclass(frozen=True)
class LinearProbeConfig:
    """Optimization settings for fusion-specific linear probes."""

    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 6


@dataclass
class FusionRun:
    """Outputs from one independently fitted fusion head."""

    model: torch.nn.Module
    test_probabilities: np.ndarray
    metrics: dict[str, Any]
    history: list[dict[str, float | int]]
    artifact_path: Path


class ProjectedFusionClassifier(torch.nn.Module):
    """Project image and text embeddings, fuse them, and classify the pair."""

    def __init__(
        self,
        image_input_dim: int,
        text_input_dim: int,
        fusion: str,
        config: FusionConfig,
    ) -> None:
        super().__init__()
        if fusion not in FUSION_TYPES:
            raise ValueError(f"Unsupported fusion method: {fusion}")

        self.fusion = fusion
        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(image_input_dim, config.projection_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.mapping_dropout),
        )
        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(text_input_dim, config.projection_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.mapping_dropout),
        )

        fused_dim = {
            "concat": config.projection_dim * 2,
            "align": config.projection_dim,
            "cross": config.projection_dim**2,
        }[fusion]
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.fusion_dropout),
            torch.nn.Linear(fused_dim, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.classifier_dropout),
            torch.nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one binary-classification logit per image-text pair."""

        image_projection = functional.normalize(
            self.image_projection(image_features), p=2, dim=1
        )
        text_projection = functional.normalize(
            self.text_projection(text_features), p=2, dim=1
        )
        if self.fusion == "concat":
            fused = torch.cat([image_projection, text_projection], dim=1)
        elif self.fusion == "align":
            fused = image_projection * text_projection
        else:
            fused = torch.bmm(
                image_projection.unsqueeze(2),
                text_projection.unsqueeze(1),
            ).flatten(start_dim=1)
        return self.classifier(fused).squeeze(1)


class FusionLinearProbe(torch.nn.Module):
    """Linear classifier over a fixed multimodal fusion feature map."""

    def __init__(
        self,
        image_input_dim: int,
        text_input_dim: int,
        fusion: str,
    ) -> None:
        super().__init__()
        if fusion not in FUSION_TYPES:
            raise ValueError(f"Unsupported fusion method: {fusion}")
        if fusion == "align" and image_input_dim != text_input_dim:
            raise ValueError(
                "Align fusion requires equal image and text embedding dimensions."
            )

        self.fusion = fusion
        if fusion == "concat":
            self.classifier = torch.nn.Linear(
                image_input_dim + text_input_dim,
                1,
            )
        elif fusion == "align":
            self.classifier = torch.nn.Linear(image_input_dim, 1)
        else:
            self.classifier = torch.nn.Bilinear(
                image_input_dim,
                text_input_dim,
                1,
            )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits from a linear readout of the selected fusion map."""

        if self.fusion == "concat":
            fused = torch.cat([image_features, text_features], dim=1)
            return self.classifier(fused).squeeze(1)
        if self.fusion == "align":
            fused = image_features * text_features
            return self.classifier(fused).squeeze(1)
        return self.classifier(image_features, text_features).squeeze(1)


def _embedding_loader(
    bundle: Mapping[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(bundle["image"], dtype=torch.float32),
        torch.as_tensor(bundle["text"], dtype=torch.float32),
        torch.as_tensor(bundle["labels"], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.inference_mode()
def _predict_probabilities(
    model: torch.nn.Module,
    bundle: Mapping[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    model.eval()
    loader = _embedding_loader(bundle, batch_size, shuffle=False, seed=seed)
    probability_batches = []
    for image_batch, text_batch, _ in loader:
        logits = model(
            image_batch.to(device),
            text_batch.to(device),
        )
        probability_batches.append(torch.sigmoid(logits).cpu().numpy())
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


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    example_count = 0
    for image_batch, text_batch, label_batch in loader:
        image_batch = image_batch.to(device)
        text_batch = text_batch.to(device)
        label_batch = label_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_batch, text_batch)
        loss = loss_function(logits, label_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(label_batch)
        example_count += len(label_batch)
    return running_loss / example_count


def _fit_with_early_stopping(
    model: torch.nn.Module,
    train_bundle: Mapping[str, np.ndarray],
    dev_bundle: Mapping[str, np.ndarray],
    config: FusionConfig | LinearProbeConfig,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]], int, float]:
    train_labels = np.asarray(train_bundle["labels"])
    positive_count = np.count_nonzero(train_labels == 1)
    negative_count = np.count_nonzero(train_labels == 0)
    positive_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float32,
        device=device,
    )
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _embedding_loader(
        train_bundle,
        config.batch_size,
        shuffle=True,
        seed=seed,
    )

    best_dev_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        dev_probability = _predict_probabilities(
            model,
            dev_bundle,
            config.batch_size,
            device,
            seed,
        )
        dev_auc = roc_auc_score(dev_bundle["labels"], dev_probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_roc_auc": float(dev_auc),
            }
        )
        if dev_auc > best_dev_auc:
            best_dev_auc = float(dev_auc)
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
    return best_state, history, best_epoch, best_dev_auc


def _train_and_persist(
    model: torch.nn.Module,
    model_name: str,
    encoder_id: str,
    fusion: str,
    family: str,
    artifact_suffix: str,
    model_splits: Mapping[str, Mapping[str, np.ndarray]],
    artifact_dir: Path,
    device: torch.device,
    seed: int,
    config: FusionConfig | LinearProbeConfig,
) -> FusionRun:
    train_bundle = model_splits["train"]
    dev_bundle = model_splits["dev"]
    test_bundle = model_splits["test"]
    best_state, history, best_epoch, best_dev_auc = _fit_with_early_stopping(
        model,
        train_bundle,
        dev_bundle,
        config,
        device,
        seed,
    )
    model.load_state_dict(best_state)
    test_probability = _predict_probabilities(
        model,
        test_bundle,
        config.batch_size,
        device,
        seed,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    artifact_dir = Path(artifact_dir)
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{model_name.lower()}--{fusion}-{artifact_suffix}.pt"
    artifact_config = {
        "model": model_name,
        "encoder_id": encoder_id,
        "family": family,
        "fusion": fusion,
        "image_input_dim": int(train_bundle["image"].shape[1]),
        "text_input_dim": int(train_bundle["text"].shape[1]),
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
        json.dumps(artifact_config, indent=2),
        encoding="utf-8",
    )

    metrics: dict[str, Any] = _classification_metrics(
        test_bundle["labels"],
        test_probability,
    )
    metrics.update(
        {
            "model": model_name,
            "family": family,
            "fusion": fusion,
            "dev_roc_auc": float(best_dev_auc),
            "best_epoch": int(best_epoch),
            "trainable_parameters": int(trainable_parameters),
            "artifact": artifact_path.name,
        }
    )
    return FusionRun(
        model=model.cpu(),
        test_probabilities=test_probability,
        metrics=metrics,
        history=history,
        artifact_path=artifact_path,
    )


def train_projected_fusion(
    model_name: str,
    encoder_id: str,
    fusion: str,
    model_splits: Mapping[str, Mapping[str, np.ndarray]],
    artifact_dir: Path,
    device: torch.device,
    seed: int,
    config: FusionConfig | None = None,
) -> FusionRun:
    """Train a nonlinear direct-fusion classifier for ablation analysis."""

    resolved_config = config or FusionConfig()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_bundle = model_splits["train"]
    model = ProjectedFusionClassifier(
        image_input_dim=train_bundle["image"].shape[1],
        text_input_dim=train_bundle["text"].shape[1],
        fusion=fusion,
        config=resolved_config,
    ).to(device)
    return _train_and_persist(
        model=model,
        model_name=model_name,
        encoder_id=encoder_id,
        fusion=fusion,
        family="Direct fusion ablation",
        artifact_suffix="direct-fusion",
        model_splits=model_splits,
        artifact_dir=artifact_dir,
        device=device,
        seed=seed,
        config=resolved_config,
    )


def train_fusion_linear_probe(
    model_name: str,
    encoder_id: str,
    fusion: str,
    model_splits: Mapping[str, Mapping[str, np.ndarray]],
    artifact_dir: Path,
    device: torch.device,
    seed: int,
    config: LinearProbeConfig | None = None,
) -> FusionRun:
    """Train a linear probe over concat, align, or cross fusion features."""

    resolved_config = config or LinearProbeConfig()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_bundle = model_splits["train"]
    model = FusionLinearProbe(
        image_input_dim=train_bundle["image"].shape[1],
        text_input_dim=train_bundle["text"].shape[1],
        fusion=fusion,
    ).to(device)
    return _train_and_persist(
        model=model,
        model_name=model_name,
        encoder_id=encoder_id,
        fusion=fusion,
        family="Fusion + LPE",
        artifact_suffix="linear-probe",
        model_splits=model_splits,
        artifact_dir=artifact_dir,
        device=device,
        seed=seed,
        config=resolved_config,
    )
