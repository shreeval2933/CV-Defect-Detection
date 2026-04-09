"""
Step 7: Training Strategy
- Epochs 1-5   : no GRL (lambda=0) – warm up the encoder
- Epochs 6+    : enable GRL with increasing lambda
- Total epochs : 20 (practical for 4-5 day deadline)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
import os


# ---------------------------------------------------------------------------
# Lambda scheduling for GRL
# ---------------------------------------------------------------------------

def compute_lambda(epoch, total_epochs, warmup_epochs=5, max_lambda=1.0):
    """
    lambda = 0 during warmup, then gradually ramps up using the
    schedule from the DANN paper.
    """
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return max_lambda * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, lambda_, use_domain_loss=True):
    model.train()
    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_labels, all_preds = [], []

    for imgs, labels, domain_ids in loader:
        imgs      = imgs.to(device)
        labels    = labels.to(device)
        domain_ids = domain_ids.to(device)

        optimizer.zero_grad()

        task_out, domain_out, _ = model(imgs, lambda_=lambda_)

        # Task loss (defect detection)
        task_loss = task_criterion(task_out, labels)

        # Domain adversarial loss
        if use_domain_loss and lambda_ > 0:
            domain_loss = domain_criterion(domain_out, domain_ids)
            loss = task_loss + domain_loss
        else:
            loss = task_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = task_out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device)
        logits, _, _ = model(imgs, lambda_=0.0)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # prob of defect class
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc   = accuracy_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")

    return {"accuracy": acc, "f1": f1, "auroc": auroc}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, test_loader, config):
    """
    config keys:
        epochs, warmup_epochs, lr, weight_decay, save_dir, device
    """
    device       = config["device"]
    epochs       = config.get("epochs", 20)
    warmup       = config.get("warmup_epochs", 5)
    save_dir     = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auroc = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs | Warmup: {warmup} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        lambda_ = compute_lambda(epoch, epochs, warmup)
        use_domain = lambda_ > 0

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_=lambda_, use_domain_loss=use_domain
        )

        metrics = evaluate(model, test_loader, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "lambda": lambda_,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **metrics,
        })

        tag = "GRL ON " if use_domain else "warmup"
        print(
            f"[{epoch:02d}/{epochs}] [{tag}] λ={lambda_:.3f} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"test acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auroc={metrics['auroc']:.3f}"
        )

        # Save best model
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            ckpt_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ✅ Saved best model (AUROC={best_auroc:.4f})")

    print(f"\nTraining complete. Best AUROC: {best_auroc:.4f}")
    return history
