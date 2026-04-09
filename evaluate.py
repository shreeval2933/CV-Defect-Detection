"""
Step 8: Evaluation
- Cross-domain evaluation (train on metal+plastic, test on fabric)
- Metrics: Accuracy, AUROC, F1
- Plots: training curves, ROC curve, uncertainty distribution
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from model import mc_dropout_predict
import os


# ---------------------------------------------------------------------------
# Full evaluation with uncertainty
# ---------------------------------------------------------------------------

def full_evaluate(model, loader, device, T=30):
    """
    Returns a dict with all evaluation metrics + uncertainty stats.
    Uses MC Dropout for uncertainty.
    """
    all_labels     = []
    all_mean_probs = []
    all_variance   = []
    all_entropy    = []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        mean_probs, variance, entropy = mc_dropout_predict(model, imgs, T=T)

        all_mean_probs.extend(mean_probs.cpu().numpy())
        all_variance.extend(variance.cpu().numpy())
        all_entropy.extend(entropy.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_labels     = np.array(all_labels)
    all_mean_probs = np.array(all_mean_probs)  # (N, C)
    all_variance   = np.array(all_variance)
    all_entropy    = np.array(all_entropy)

    defect_probs = all_mean_probs[:, 1]
    preds        = all_mean_probs.argmax(axis=1)

    acc   = accuracy_score(all_labels, preds)
    f1    = f1_score(all_labels, preds, zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, defect_probs)
    except ValueError:
        auroc = float("nan")

    print("\n" + "="*50)
    print("CROSS-DOMAIN EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUROC    : {auroc:.4f}")
    print(f"\nMean Variance (uncertainty): {all_variance.mean():.4f}")
    print(f"Mean Entropy               : {all_entropy.mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, preds,
                                target_names=["normal", "defect"],
                                zero_division=0))

    return {
        "accuracy": acc, "f1": f1, "auroc": auroc,
        "labels": all_labels, "preds": preds,
        "defect_probs": defect_probs,
        "variance": all_variance, "entropy": all_entropy,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], "b-o", ms=3)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [h["train_acc"] for h in history],   "b-o", ms=3, label="Train")
    axes[1].plot(epochs, [h["accuracy"] for h in history],    "r-o", ms=3, label="Test (fabric)")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # AUROC
    axes[2].plot(epochs, [h["auroc"] for h in history], "g-o", ms=3)
    axes[2].set_title("AUROC (Test)"); axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Training Curves (Cross-Domain: train=metal+plastic → test=fabric)")
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_roc_curve(results, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(results["labels"], results["defect_probs"])
    auroc = results["auroc"]

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, "b-", lw=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Fabric (unseen domain)")
    plt.legend(); plt.grid(alpha=0.3)
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_uncertainty(results, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    labels   = results["labels"]
    variance = results["variance"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Variance by class
    axes[0].hist(variance[labels == 0], bins=30, alpha=0.6, label="Normal",  color="green")
    axes[0].hist(variance[labels == 1], bins=30, alpha=0.6, label="Defect",  color="red")
    axes[0].set_xlabel("Predictive Variance (MC Dropout)")
    axes[0].set_title("Uncertainty Distribution")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Entropy by class
    entropy = results["entropy"]
    axes[1].hist(entropy[labels == 0], bins=30, alpha=0.6, label="Normal", color="green")
    axes[1].hist(entropy[labels == 1], bins=30, alpha=0.6, label="Defect", color="red")
    axes[1].set_xlabel("Predictive Entropy")
    axes[1].set_title("Entropy Distribution")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("MC Dropout Uncertainty Estimation (T=30)")
    plt.tight_layout()
    path = os.path.join(save_dir, "uncertainty.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(results, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(results["labels"], results["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Defect"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix – Fabric (unseen domain)")
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def run_all_plots(history, results, save_dir="plots"):
    plot_training_curves(history, save_dir)
    plot_roc_curve(results, save_dir)
    plot_uncertainty(results, save_dir)
    plot_confusion_matrix(results, save_dir)
    print(f"\nAll plots saved to: {save_dir}/")
