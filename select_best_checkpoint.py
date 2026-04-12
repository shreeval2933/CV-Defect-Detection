"""
Checkpoint Evaluation + Best Model Selection

UPDATED FINAL VERSION:
✔ Saves JSON + LOG + BEST SUMMARY
✔ Stores classification report
✔ Fix torch warning
✔ Clean logging
✔ Supports Grad-CAM output directory
"""

import os
import torch
import json
import argparse
from datetime import datetime

from dataset import build_dataloaders
from model import BaselineResNet, ArchitectureA, DualBranchArchitectureC
from evaluate import full_evaluate
from gradcam import batch_visualize
from evaluate import run_all_plots


# ---------------------------------------------------------------------------
# ARG PARSER
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="a", choices=["baseline", "a", "c"])
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--mc_T", type=int, default=30)
    parser.add_argument("--gradcam", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------------------------
def build_model(arch, n_domains=3):
    if arch == "baseline":
        return BaselineResNet(num_classes=2)
    elif arch == "a":
        return ArchitectureA(num_domains=n_domains, num_classes=2)
    elif arch == "c":
        return DualBranchArchitectureC(num_domains=n_domains, num_classes=2)


# ---------------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, path):
        self.file = open(path, "w")

    def log(self, msg):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()


def clean_results(results):
    """
    Convert results into JSON-safe + compact format
    (avoid storing large numpy arrays)
    """
    return {
        "accuracy": float(results["accuracy"]),
        "f1": float(results["f1"]),
        "auroc": float(results["auroc"]),
        "threshold": float(results["threshold"]),
        "mean_variance": float(results["variance"].mean()),
        "mean_entropy": float(results["entropy"].mean()),
    }
    

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CKPT_DIR = f"/DATA2/shrusti/cv_checkpoints/{args.arch}/run_{args.run_id}"

    RESULT_BASE = f"results/{args.arch}/run_{args.run_id}"
    EVAL_DIR    = os.path.join(RESULT_BASE, "evaluate")
    PLOT_DIR    = f"plots/{args.arch}/run_{args.run_id}"

    LOG_PATH = os.path.join(EVAL_DIR, "eval.log")

    # Create directories
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    logger = Logger(LOG_PATH)

    logger.log(f"Evaluating ARCH={args.arch}, RUN={args.run_id}")
    logger.log(f"Time: {datetime.now()}\n")

    # ---- Data ----
    train_loader, test_loader, n_domains = build_dataloaders(
        mvtec_root="data/mvtec",
        aitex_root="data/aitex",
        batch_size=args.batch,
        num_workers=4,
    )

    model = build_model(args.arch, n_domains).to(device)

    results_all = []
    best_score = -1
    best_epoch = -1
    best_ckpt_path = None
    best_results = None

    # -----------------------------------------------------------------------
    # LOOP
    # -----------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch}.pth")
        if not os.path.exists(ckpt_path):
            continue

        logger.log(f"\nEvaluating: epoch_{epoch}")

        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )

        results = full_evaluate(model, test_loader, device, T=args.mc_T)

        score = results["f1"] + results["auroc"]

        logger.log(
            f"Epoch {epoch} → F1={results['f1']:.4f}, "
            f"AUROC={results['auroc']:.4f}, Score={score:.4f}"
        )

        results_all.append({
            "epoch": epoch,
            "f1": results["f1"],
            "auroc": results["auroc"],
            "accuracy": results["accuracy"],
            "score": score
        })

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_ckpt_path = ckpt_path
            best_results = results

    # -----------------------------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------------------------
    json_path = os.path.join(EVAL_DIR, "checkpoint_eval.json")
    with open(json_path, "w") as f:
        json.dump(results_all, f, indent=2)

    # -----------------------------------------------------------------------
    # SAVE BEST SUMMARY
    # -----------------------------------------------------------------------
    summary_path = os.path.join(EVAL_DIR, "best.txt")
 
    clean_best = clean_results(best_results)

    with open(summary_path, "w") as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Score: {best_score:.4f}\n\n")
        f.write(json.dumps(clean_best, indent=2))

    logger.log("\n" + "="*60)
    logger.log("BEST CHECKPOINT")
    logger.log("="*60)
    logger.log(f"Epoch: {best_epoch}")
    logger.log(f"Score: {best_score:.4f}")

    # -----------------------------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------------------------
    best_model_path = os.path.join(CKPT_DIR, "best_model.pth")
    torch.save(torch.load(best_ckpt_path, weights_only=True), best_model_path)

    logger.log(f"\nSaved best model: {best_model_path}")

    # -----------------------------------------------------------------------
    # GRADCAM
    # -----------------------------------------------------------------------
    if args.gradcam:
        logger.log("\nRunning Grad-CAM...")

        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )

        batch_visualize(
            model,
            test_loader,
            device,
            n_samples=6,
            save_dir=os.path.join(PLOT_DIR, "gradcam")
        )

    # -----------------------------------------------------------------------
    # GENERATE FINAL PLOTS FROM BEST MODEL
    # -----------------------------------------------------------------------
    logger.log("\nGenerating plots using BEST model...")

    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

    best_results_full = full_evaluate(model, test_loader, device, T=args.mc_T)

    # -----------------------------------------------------------------------
    # LOAD TRAINING HISTORY (FIX EMPTY PLOT ISSUE)
    # -----------------------------------------------------------------------
    history_path = f"results/{args.arch}/run_{args.run_id}/train/history.json"

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = []
        logger.log("WARNING: history.json not found → plots may be empty")

    run_all_plots(
        history=history,
        results=best_results_full,
        save_dir=PLOT_DIR
    )

    # CLOSE LOGGER AFTER EVERYTHING
    logger.log("\nDone.")
    logger.file.close()


if __name__ == "__main__":
    main()