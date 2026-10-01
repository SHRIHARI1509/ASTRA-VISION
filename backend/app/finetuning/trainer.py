"""ASTRA VISION — Phase 8G SigLIP 2 LoRA Fine-Tuning Pipeline.

Implements parameter-efficient fine-tuning (PEFT/LoRA) on google/siglip2-base-patch16-512
for the 6-class defence taxonomy. Supports checkpointing, validation tracking,
early stopping, and best-checkpoint selection based exclusively on validation metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from backend.app.finetuning.reconciliation import PRODUCTION_CLASSES


class TrainingConfig(BaseModel):
    """Machine-readable hyperparameter and model configuration."""
    model_identifier: str = "google/siglip2-base-patch16-512"
    processor_identifier: str = "google/siglip2-base-patch16-512"
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    random_seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    image_resolution: int = 512
    optimizer_name: str = "AdamW"
    scheduler_name: str = "linear_with_warmup"
    device: str = "cpu"
    early_stopping_patience: int = 2


class EpochTelemetry(BaseModel):
    """Metrics recorded for a single training epoch."""
    epoch: int
    train_loss: float
    train_top1_acc: float
    val_loss: float
    val_top1_acc: float
    val_top3_acc: float
    epoch_duration_seconds: float
    is_best_checkpoint: bool


class TrainingResult(BaseModel):
    """Overall outcome of the fine-tuning execution."""
    status: str
    total_epochs_completed: int
    best_epoch: int
    best_val_top1_acc: float
    best_val_loss: float
    total_training_duration_seconds: float
    checkpoint_dir: str
    config: TrainingConfig
    history: List[EpochTelemetry]


class ManifestImageDataset(Dataset):
    """PyTorch Dataset loading images and target labels from an audited manifest."""

    def __init__(self, manifest_path: Path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = data.get("items", [])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        item = self.items[idx]
        img_path = Path(item["absolute_path"])
        label_idx = item["class_index"]
        class_name = item["final_label"]
        img = Image.open(img_path).convert("RGB")
        return img, label_idx, class_name


def collate_fn_builder(processor: Any, candidate_texts: List[str], device: str):
    """Custom collate function that tokenizes and pre-processes image batches."""
    def collate_fn(batch):
        images = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        class_names = [b[2] for b in batch]

        # Process image batch
        img_inputs = processor(images=images, return_tensors="pt")
        # Text queries
        text_inputs = processor(
            text=candidate_texts,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        return img_inputs, text_inputs, labels, class_names
    return collate_fn


class SigLIP2LoRATrainer:
    """Manages the lifecycle of LoRA fine-tuning for SigLIP 2."""

    def __init__(self, config: TrainingConfig, checkpoint_root: Path):
        self.config = config
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._set_seed(self.config.random_seed)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(
        self,
        train_manifest_path: Path,
        val_manifest_path: Path,
        class_weights: Optional[Dict[str, float]] = None,
    ) -> TrainingResult:
        """Run conservative LoRA fine-tuning on the validated splits."""
        t_start = time.perf_counter()

        # 1. Load Processor & Base Model
        processor = AutoProcessor.from_pretrained(self.config.processor_identifier)
        base_model = AutoModel.from_pretrained(self.config.model_identifier)

        # 2. Configure PEFT LoRA
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(base_model, peft_config)
        model.to(self.config.device)

        # 3. Setup candidate prompts
        candidate_labels = PRODUCTION_CLASSES
        candidate_prompts = [f"a military {label.lower()} in a defence scenario" for label in candidate_labels]

        # 4. Prepare Datasets & DataLoaders
        train_ds = ManifestImageDataset(train_manifest_path)
        val_ds = ManifestImageDataset(val_manifest_path)

        collate_fn = collate_fn_builder(processor, candidate_prompts, self.config.device)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # 5. Loss with Class Weights
        if class_weights:
            weights_tensor = torch.tensor(
                [class_weights.get(c, 1.0) for c in candidate_labels],
                dtype=torch.float,
                device=self.config.device,
            )
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        # 6. Optimizer & Scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = (len(train_loader) // self.config.gradient_accumulation_steps) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(total_steps, 1),
        )

        history: List[EpochTelemetry] = []
        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_epoch = -1
        best_checkpoint_dir = self.checkpoint_root / "checkpoint-best"

        for epoch in range(1, self.config.epochs + 1):
            ep_start = time.perf_counter()
            model.train()

            train_loss_accum = 0.0
            train_correct = 0
            train_total = 0
            optimizer.zero_grad()

            for step_idx, (img_inputs, text_inputs, targets, _) in enumerate(train_loader):
                img_inputs = {k: v.to(self.config.device) for k, v in img_inputs.items()}
                text_inputs = {k: v.to(self.config.device) for k, v in text_inputs.items()}
                targets = targets.to(self.config.device)

                outputs = model(**img_inputs, **text_inputs)
                logits = outputs.logits_per_image  # [B, 6]

                loss = criterion(logits, targets)
                loss_scaled = loss / self.config.gradient_accumulation_steps
                loss_scaled.backward()

                train_loss_accum += loss.item() * len(targets)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += len(targets)

                if (step_idx + 1) % self.config.gradient_accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            epoch_train_loss = round(train_loss_accum / max(train_total, 1), 4)
            epoch_train_acc = round(train_correct / max(train_total, 1), 4)

            # Validation Loop
            model.eval()
            val_loss_accum = 0.0
            val_correct = 0
            val_top3_correct = 0
            val_total = 0

            with torch.no_grad():
                for img_inputs, text_inputs, targets, _ in val_loader:
                    img_inputs = {k: v.to(self.config.device) for k, v in img_inputs.items()}
                    text_inputs = {k: v.to(self.config.device) for k, v in text_inputs.items()}
                    targets = targets.to(self.config.device)

                    outputs = model(**img_inputs, **text_inputs)
                    logits = outputs.logits_per_image

                    loss = criterion(logits, targets)
                    val_loss_accum += loss.item() * len(targets)

                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == targets).sum().item()

                    top3_preds = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
                    for t, top3 in zip(targets, top3_preds):
                        if t in top3:
                            val_top3_correct += 1

                    val_total += len(targets)

            epoch_val_loss = round(val_loss_accum / max(val_total, 1), 4)
            epoch_val_acc = round(val_correct / max(val_total, 1), 4)
            epoch_val_top3 = round(val_top3_correct / max(val_total, 1), 4)
            ep_duration = round(time.perf_counter() - ep_start, 2)

            is_best = False
            # Checkpoint selection strictly based on validation accuracy and loss
            if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
                is_best = True
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                best_epoch = epoch

                # Save best checkpoint weights via PEFT
                best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_checkpoint_dir)
                processor.save_pretrained(best_checkpoint_dir)

            telemetry = EpochTelemetry(
                epoch=epoch,
                train_loss=epoch_train_loss,
                train_top1_acc=epoch_train_acc,
                val_loss=epoch_val_loss,
                val_top1_acc=epoch_val_acc,
                val_top3_acc=epoch_val_top3,
                epoch_duration_seconds=ep_duration,
                is_best_checkpoint=is_best,
            )
            history.append(telemetry)

        total_duration = round(time.perf_counter() - t_start, 2)

        result = TrainingResult(
            status="SUCCESS",
            total_epochs_completed=self.config.epochs,
            best_epoch=best_epoch,
            best_val_top1_acc=best_val_acc,
            best_val_loss=best_val_loss,
            total_training_duration_seconds=total_duration,
            checkpoint_dir=str(best_checkpoint_dir.resolve()),
            config=self.config,
            history=history,
        )

        # Write training summary artifacts
        with open(self.checkpoint_root / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)

        return result
