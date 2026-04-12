"""
Step 7: Training Strategy
- Epochs 1-5   : no GRL (lambda=0) – warm up the encoder
- Epochs 6+    : enable GRL with increasing lambda
- Total epochs : 40 (extended for better convergence)

UPDATED:
✔ Save ALL checkpoints (for later selection)
✔ EMA (Exponential Moving Average) for stable evaluation
✔ Reduced GRL strength (max_lambda=0.1)
✔ Weight decay increased (better generalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
import os


# ---------------------------------------------------------------------------
# Lambda scheduling for GRL
# ---------------------------------------------------------------------------

def compute_lambda(epoch, total_epochs, warmup_epochs=5, max_lambda=0.1):
    """
    lambda = 0 during warmup, then gradually ramps up

    UPDATED:
    ✔ Reduced max_lambda → prevents feature destruction
    """
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return max_lambda * (2.0 / (1.0 + np.exp(-5 * progress)) - 1.0)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """
    Maintains moving average of model weights

    WHY:
    ✔ Smooths noisy updates
    ✔ Improves generalization
    ✔ More stable evaluation
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ---------------------------------------------------------------------------
# FOCAL LOSS
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, lambda_, ema, use_domain_loss=True):
    model.train()

    class_weights = torch.tensor([1.0, 2.0]).to(device)

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    focal_criterion = FocalLoss()
    domain_criterion = nn.CrossEntropyLoss()

    alpha = 0.1

    total_loss = 0.0
    all_labels, all_preds = [], []

    for imgs, labels, domain_ids in loader:
        imgs       = imgs.to(device)
        labels     = labels.to(device)
        domain_ids = domain_ids.to(device)

        optimizer.zero_grad()

        task_out, domain_out, _ = model(imgs, lambda_=lambda_)

        ce_loss    = ce_criterion(task_out, labels)
        focal_loss = focal_criterion(task_out, labels)
        task_loss  = ce_loss + focal_loss

        # -------------------------------------------------------------------
        # SAFE DOMAIN LOSS
        # -------------------------------------------------------------------
        if use_domain_loss and lambda_ > 0 and domain_out is not None:
            domain_loss = domain_criterion(domain_out, domain_ids)
            loss = task_loss + alpha * domain_loss
        else:
            loss = task_loss

        loss.backward()
        optimizer.step()

        # ---------------------------------------------------------------
        # UPDATED: EMA update after each step
        # ---------------------------------------------------------------
        ema.update(model)

        total_loss += loss.item()

        probs = torch.softmax(task_out, dim=1)[:, 1]
        preds = (probs > 0.5).long().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        logits, _, _ = model(imgs, lambda_=0.0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.5).long().cpu().numpy()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")

    return {"accuracy": acc, "f1": f1, "auroc": auroc}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, test_loader, config):

    device   = config["device"]
    epochs   = config.get("epochs", 40)
    warmup   = config.get("warmup_epochs", 5)
    save_dir = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 5e-4),
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ---------------------------------------------------------------
    # UPDATED: Initialize EMA
    # ---------------------------------------------------------------
    ema = EMA(model, decay=0.999)

    history = []

    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs | Warmup: {warmup} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):

        # -----------------------------------------------------------
        # Freeze backbone initially
        # -----------------------------------------------------------
        if hasattr(model, "encoder"):
            if epoch <= 5:
                for p in model.encoder.parameters():
                    p.requires_grad = False
            else:
                for p in model.encoder.parameters():
                    p.requires_grad = True

        lambda_ = compute_lambda(epoch, epochs, warmup)
        use_domain = lambda_ > 0

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_=lambda_, ema=ema, use_domain_loss=use_domain
        )

        # -----------------------------------------------------------
        # UPDATED: Evaluate EMA weights (not raw model)
        # -----------------------------------------------------------
        ema.apply_shadow(model)
        metrics = evaluate(model, test_loader, device)
        ema.restore(model)

        scheduler.step()

        history.append({
            "epoch": epoch,
            "lambda": lambda_,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **metrics,
        })

        print(
            f"[{epoch:02d}/{epochs}] λ={lambda_:.3f} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"test acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auroc={metrics['auroc']:.3f}"
        )

        # -------------------------------------------------------------------
        # UPDATED: Save ALL checkpoints in structured directory
        # Save EMA weights instead of raw model
        # -------------------------------------------------------------------
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pth")

        ema.apply_shadow(model)
        torch.save(model.state_dict(), ckpt_path)
        ema.restore(model)
        
    print("\nTraining complete. All checkpoints saved.")
    return history