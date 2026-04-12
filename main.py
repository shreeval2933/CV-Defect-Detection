"""
UPDATED MAIN:
✔ Run-based directory management
✔ Logging to file
✔ Separate checkpoint storage (DATA2)
✔ Separate plots per run
✔ No overwrite issues
"""

import argparse
import torch
import json
import os
import sys
from datetime import datetime

from dataset  import build_dataloaders
from model    import BaselineResNet, ArchitectureA, DualBranchArchitectureC
from train    import train
from evaluate import full_evaluate, run_all_plots
from gradcam  import batch_visualize

import numpy as np
import random


# ---------------------------------------------------------------------------
# FIX RANDOMNESS
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


# ---------------------------------------------------------------------------
# LOGGING CLASS
# ---------------------------------------------------------------------------
class Logger:
    """
    Redirects print statements to both terminal and file
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# ARG PARSER
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="a", choices=["baseline", "a", "c"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--mvtec", default="data/mvtec")
    parser.add_argument("--aitex", default="data/aitex")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run_id", type=int, default=1)   
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--mc_T", type=int, default=30)
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
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # PATH SETUP (UPDATED STRUCTURE)
    # -----------------------------------------------------------------------
    ckpt_base = f"/DATA2/shrusti/cv_checkpoints/{args.arch}/run_{args.run_id}"

    result_base = f"results/{args.arch}/run_{args.run_id}"
    train_dir   = os.path.join(result_base, "train")
    eval_dir    = os.path.join(result_base, "evaluate")

    plot_dir    = f"plots/{args.arch}/run_{args.run_id}"

    os.makedirs(ckpt_base, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # LOGGER SETUP
    # -----------------------------------------------------------------------
    log_file = os.path.join(train_dir, "train.log")
    sys.stdout = Logger(log_file)

    print(f"Run ID: {args.run_id}")
    print(f"Architecture: {args.arch}")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------------
    train_loader, test_loader, n_domains = build_dataloaders(
        mvtec_root=args.mvtec,
        aitex_root=args.aitex,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    # -----------------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------------
    model = build_model(args.arch, n_domains)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    # -----------------------------------------------------------------------
    # TRAIN
    # -----------------------------------------------------------------------
    config = {
        "epochs": args.epochs,
        "warmup_epochs": args.warmup,
        "lr": args.lr,
        "weight_decay": 5e-4,
        "save_dir": ckpt_base,   
        "device": device,
    }

    history = train(model, train_loader, test_loader, config)

    # -----------------------------------------------------------------------
    # LOAD LAST MODEL
    # -----------------------------------------------------------------------
    last_ckpt = os.path.join(ckpt_base, f"epoch_{args.epochs}.pth")
    model.load_state_dict(torch.load(last_ckpt, map_location=device, weights_only=True))

    # -----------------------------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------------------------
    results = full_evaluate(model, test_loader, device, T=args.mc_T)

    # -----------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------
    run_all_plots(history, results, save_dir=plot_dir)

    # -----------------------------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------------------------
    with open(os.path.join(train_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # -----------------------------------------------------------------------
    # GRADCAM
    # -----------------------------------------------------------------------
    if args.gradcam:
        batch_visualize(model, test_loader, device, n_samples=6)

    print("\nRun complete. Logs saved.")


if __name__ == "__main__":
    main()