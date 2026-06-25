"""
HCSI-Net SBI main training script
=================================

This script trains the assembled HCSI-Net model on Self-Blended Images (SBI).

Use this after the CNN and transformer backbones have already been pretrained
separately and loaded into the HCSI-Net model. 

"""

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from SelfBlendedImages.src.utils.sbi import SBI_Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_soft_targets(
    labels: torch.Tensor,
    num_classes: int = 2,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = F.one_hot(labels.long(), num_classes=num_classes).float()

    if label_smoothing > 0.0:
        targets = (1.0 - label_smoothing) * targets + label_smoothing / float(num_classes)

    return targets


def build_linear_decay_scheduler(
    optimizer: optim.Optimizer,
    n_epoch: int,
) -> optim.lr_scheduler.LambdaLR:
    n_epoch = max(1, int(n_epoch))

    def lr_lambda(epoch: int) -> float:
        return max(0.0, 1.0 - float(epoch) / float(n_epoch))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def apply_artifact_augmentations(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    device = x.device

    if torch.rand((), device=device) < 0.40:
        scale = torch.empty((), device=device).uniform_(0.45, 0.90)
        new_h = max(8, int(h * float(scale)))
        new_w = max(8, int(w * float(scale)))

        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    if torch.rand((), device=device) < 0.30:
        kernel = torch.tensor(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            dtype=x.dtype,
            device=device,
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        x = F.conv2d(x, kernel, padding=1, groups=c)

    if torch.rand((), device=device) < 0.35:
        sigma = torch.empty((), device=device).uniform_(0.003, 0.020)
        x = x + sigma * torch.randn_like(x)

    if torch.rand((), device=device) < 0.25:
        gain = torch.empty((b, 1, 1, 1), device=device).uniform_(0.90, 1.10)
        bias = torch.empty((b, 1, 1, 1), device=device).uniform_(-0.03, 0.03)
        x = x * gain + bias

    return x


def random_erasing(
    x: torch.Tensor,
    erase_prob: float,
) -> torch.Tensor:
    if torch.rand((), device=x.device) > erase_prob:
        return x

    b, _, h, w = x.shape
    erased = x.clone()

    erase_area = torch.rand(b, device=x.device) * 0.25 + 0.05
    erase_w = (torch.sqrt(erase_area) * w).long().clamp_min(1)
    erase_h = (torch.sqrt(erase_area) * h).long().clamp_min(1)

    cx = torch.randint(0, w, (b,), device=x.device)
    cy = torch.randint(0, h, (b,), device=x.device)

    for i in range(b):
        x1 = max(0, int(cx[i] - erase_w[i] // 2))
        x2 = min(w, int(cx[i] + erase_w[i] // 2))
        y1 = max(0, int(cy[i] - erase_h[i] // 2))
        y2 = min(h, int(cy[i] + erase_h[i] // 2))

        erased[i, :, y1:y2, x1:x2] = 0.0

    return erased


def mixup(
    x: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)

    x = lam * x + (1.0 - lam) * x[perm]
    targets = lam * targets + (1.0 - lam) * targets[perm]

    return x, targets


def rand_bbox(
    size,
    lam: float,
    device: torch.device,
):
    b, _, h, w = size

    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = torch.randint(w, (b,), device=device)
    cy = torch.randint(h, (b,), device=device)

    x1 = (cx - cut_w // 2).clamp(0, w - 1).long()
    y1 = (cy - cut_h // 2).clamp(0, h - 1).long()
    x2 = (cx + cut_w // 2).clamp(0, w - 1).long()
    y2 = (cy + cut_h // 2).clamp(0, h - 1).long()

    return x1, y1, x2, y2


def cutmix(
    x: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)

    x1, y1, x2, y2 = rand_bbox(x.size(), lam, device=x.device)

    mixed_x = x.clone()
    for i in range(x.size(0)):
        mixed_x[i, :, y1[i]:y2[i], x1[i]:x2[i]] = x[
            perm[i],
            :,
            y1[i]:y2[i],
            x1[i]:x2[i],
        ]

    patch_area = (x2 - x1).float() * (y2 - y1).float()
    lam_adjusted = 1.0 - patch_area / float(x.size(2) * x.size(3))
    lam_adjusted = lam_adjusted.view(-1, 1).to(x.device)

    targets = lam_adjusted * targets + (1.0 - lam_adjusted) * targets[perm]

    return mixed_x, targets


def apply_mixing_and_erasing(
    x: torch.Tensor,
    targets: torch.Tensor,
    mix_prob: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    erase_prob: float,
):
    if torch.rand((), device=x.device) < mix_prob:
        if torch.rand((), device=x.device) < 0.50:
            x, targets = mixup(x, targets, mixup_alpha)
        else:
            x, targets = cutmix(x, targets, cutmix_alpha)

    x = random_erasing(x, erase_prob=erase_prob)

    return x, targets


def plot_training_curves(
    train_losses,
    val_losses,
    val_aucs,
    output_path: Path,
) -> None:
    if len(train_losses) == 0:
        return

    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    fig, (ax_loss, ax_auc) = plt.subplots(2, 1, figsize=(8, 10))

    ax_loss.plot(epochs, train_losses, label="train loss", marker="o")
    ax_loss.plot(epochs, val_losses, label="validation loss", marker="o")
    ax_loss.set_title("Training and validation loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True)
    ax_loss.legend()

    ax_auc.plot(epochs, val_aucs, label="validation AUC", marker="o")
    ax_auc.set_title("Validation AUC")
    ax_auc.set_xlabel("epoch")
    ax_auc.set_ylabel("AUC")
    ax_auc.grid(True)
    ax_auc.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_best_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    score: float,
    best_checkpoints: dict,
    n_weight: int,
) -> dict:
    should_save = len(best_checkpoints) < n_weight

    if not should_save and not math.isnan(score):
        worst_checkpoint = min(best_checkpoints, key=best_checkpoints.get)
        should_save = score > best_checkpoints[worst_checkpoint]

    if not should_save:
        return best_checkpoints

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    best_checkpoints[str(checkpoint_path)] = score

    while len(best_checkpoints) > n_weight:
        worst_checkpoint = min(best_checkpoints, key=best_checkpoints.get)
        if os.path.exists(worst_checkpoint):
            os.remove(worst_checkpoint)
        del best_checkpoints[worst_checkpoint]

    return best_checkpoints


def train_hcsi(
    model: nn.Module,
    n_epoch: int,
    batch_size: int,
    weight_decay: float,
    learning_rate: float,
    experiment_name: str,
    n_weight: int,
    image_size: int,
    seed: int = 250425,
    num_workers: int = 4,
    label_smoothing: float = 0.05,
    mix_prob: float = 0.60,
    mixup_alpha: float = 0.30,
    cutmix_alpha: float = 0.60,
    erase_prob: float = 0.25,
):
    """
    Train the full HCSI-Net model on SBI.

    Args:
        model:
            Full HCSI-Net model returning B x 2 logits.
        n_epoch:
            Number of epochs.
        batch_size:
            Effective number of images after SBI collate_fn.
        weight_decay:
            AdamW weight decay.
        learning_rate:
            AdamW learning rate.
        experiment_name:
            Folder name used for checkpoints and visualisations.
        n_weight:
            Number of best validation-AUC checkpoints to keep.
        image_size:
            Well, it's image size:)
        seed:
            Random seed.
        num_workers:
            DataLoader workers.
        label_smoothing:
            Label smoothing used for training targets.
        mix_prob:
            Probability of applying MixUp or CutMix.
        mixup_alpha:
            MixUp beta-distribution alpha.
        cutmix_alpha:
            CutMix beta-distribution alpha.
        erase_prob:
            Random-erasing probability.
    """
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_dir = Path("./weights") / "training" / experiment_name
    visualisation_dir = Path("./visualisations") / experiment_name

    train_dataset = SBI_Dataset(phase="train", image_size=image_size)
    val_dataset = SBI_Dataset(phase="val", image_size=image_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=val_dataset.worker_init_fn,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = build_linear_decay_scheduler(optimizer, n_epoch)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_aucs = []
    best_checkpoints = {}

    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)

        model.train()

        train_loss_sum = 0.0
        train_acc_sum = 0.0

        train_iter = tqdm(train_loader, desc=f"{experiment_name} train {epoch + 1}/{n_epoch}")

        for data in train_iter:
            img = data["img"].to(device, non_blocking=True).float()
            labels = data["label"].to(device, non_blocking=True).long()

            targets = make_soft_targets(
                labels,
                num_classes=2,
                label_smoothing=label_smoothing,
            )

            img = apply_artifact_augmentations(img)
            img, targets = apply_mixing_and_erasing(
                img,
                targets,
                mix_prob=mix_prob,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                erase_prob=erase_prob,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(img)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())

            predictions = logits.argmax(dim=1)
            train_acc_sum += float((predictions == labels).float().mean().item())

        scheduler.step()

        train_loss = train_loss_sum / max(1, len(train_loader))
        train_acc = train_acc_sum / max(1, len(train_loader))

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()

        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_scores = []
        val_labels = []

        val_iter = tqdm(val_loader, desc=f"{experiment_name} val {epoch + 1}/{n_epoch}")

        with torch.no_grad():
            for data in val_iter:
                img = data["img"].to(device, non_blocking=True).float()
                labels = data["label"].to(device, non_blocking=True).long()

                targets = make_soft_targets(
                    labels,
                    num_classes=2,
                    label_smoothing=0.0,
                )

                logits = model(img)
                loss = criterion(logits, targets)

                val_loss_sum += float(loss.item())

                predictions = logits.argmax(dim=1)
                val_acc_sum += float((predictions == labels).float().mean().item())

                val_scores.extend(logits.softmax(dim=1)[:, 1].detach().cpu().numpy().tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = val_acc_sum / max(1, len(val_loader))

        try:
            val_auc = float(roc_auc_score(val_labels, val_scores))
        except ValueError:
            val_auc = float("nan")

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{n_epoch} | "
            f"lr: {current_lr:.6g} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, val AUC: {val_auc:.4f}"
        )

        checkpoint_path = checkpoint_dir / (
            f"epoch_{epoch + 1:03d}_val_auc_{val_auc:.5f}_val_loss_{val_loss:.5f}.pth"
        )

        best_checkpoints = save_best_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            score=val_auc,
            best_checkpoints=best_checkpoints,
            n_weight=n_weight,
        )

        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            val_aucs=val_aucs,
            output_path=visualisation_dir / "training_curves.png",
        )

    final_checkpoint = checkpoint_dir / "final.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_checkpoint)

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_aucs": val_aucs,
        "best_checkpoints": best_checkpoints,
        "final_checkpoint": str(final_checkpoint),
    }
