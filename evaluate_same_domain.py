"""
Same-Domain Evaluation Script

✔ Uses best_model.pth
✔ Evaluates on MVTec (same domain)
✔ Supports multiple runs
✔ Saves results properly
"""

import os
import torch
import json
import argparse

from dataset import build_same_domain_test_loader
from model import BaselineResNet, ArchitectureA, DualBranchArchitectureC
from evaluate import full_evaluate


# ---------------------------------------------------------------------------
# ARG PARSER
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="a", choices=["baseline", "a", "c"])
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--batch", type=int, default=32)
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

    # Paths
    CKPT_PATH = f"/DATA2/shrusti/cv_checkpoints/{args.arch}/run_{args.run_id}/best_model.pth"
    SAVE_DIR  = f"results/{args.arch}/run_{args.run_id}/same_domain"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nEvaluating SAME-DOMAIN → ARCH={args.arch}, RUN={args.run_id}")

    # -----------------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------------
    test_loader = build_same_domain_test_loader(
        mvtec_root="data/mvtec",
        batch_size=args.batch
    )

    # -----------------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------------
    model = build_model(args.arch).to(device)

    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=device, weights_only=True)
    )

    print(f"Loaded model from: {CKPT_PATH}")

    # -----------------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------------
    results = full_evaluate(model, test_loader, device, T=args.mc_T)

    # -----------------------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------------------
    save_path = os.path.join(SAVE_DIR, "same_domain_results.json")

    clean_results = {
        "accuracy": float(results["accuracy"]),
        "f1": float(results["f1"]),
        "auroc": float(results["auroc"]),
        "threshold": float(results["threshold"]),
        "mean_variance": float(results["variance"].mean()),
        "mean_entropy": float(results["entropy"].mean()),
    }

    with open(save_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    print("\nSaved results at:", save_path)
    print("\nDone.")


if __name__ == "__main__":
    main()