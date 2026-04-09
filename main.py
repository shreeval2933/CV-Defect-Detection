"""
main.py – Run the full CV defect detection pipeline.

Usage:
    python main.py --arch a --epochs 20 --mvtec /path/to/mvtec --aitex /path/to/aitex

Architecture choices:
    baseline  – ResNet50 only (Step 2)
    a         – Architecture A: single encoder + DANN GRL (Step 3)  ← START HERE
    c         – Architecture C: dual branch + DANN GRL (Step 4)
"""

import argparse
import torch
import json
import os

from dataset  import build_dataloaders
from model    import BaselineResNet, ArchitectureA, DualBranchArchitectureC
from train    import train
from evaluate import full_evaluate, run_all_plots
from gradcam  import batch_visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",    default="a",
                        choices=["baseline", "a", "c"],
                        help="Model architecture")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--warmup",  type=int, default=5,
                        help="Epochs before GRL is enabled")
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--mvtec",   default="data/mvtec",
                        help="Root path to MVTec dataset")
    parser.add_argument("--aitex",   default="data/aitex",
                        help="Root path to AITEX dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save",    default="checkpoints")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualizations after training")
    parser.add_argument("--mc_T",    type=int, default=30,
                        help="MC Dropout forward passes")
    return parser.parse_args()


def build_model(arch, n_domains=3):
    if arch == "baseline":
        return BaselineResNet(num_classes=2)
    elif arch == "a":
        return ArchitectureA(num_domains=n_domains, num_classes=2)
    elif arch == "c":
        return DualBranchArchitectureC(num_domains=n_domains, num_classes=2)
    raise ValueError(f"Unknown arch: {arch}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {args.arch}")

    # ---- Data ----
    train_loader, test_loader, n_domains = build_dataloaders(
        mvtec_root=args.mvtec,
        aitex_root=args.aitex,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    # ---- Model ----
    model = build_model(args.arch, n_domains)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    # ---- Training ----
    if args.arch == "baseline":
        # Baseline has no domain head – wrap for compatibility
        from train import evaluate
        import torch.optim as optim
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            from torch.nn import CrossEntropyLoss
            criterion = CrossEntropyLoss()
            total_loss = 0
            for imgs, labels, _ in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            metrics = evaluate(model, test_loader, device)
            # Patch model.forward to return triple for evaluate compatibility
            print(f"[{epoch:02d}/{args.epochs}] loss={total_loss/len(train_loader):.4f} | "
                  f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auroc={metrics['auroc']:.3f}")
            history.append({"epoch": epoch, **metrics, "train_loss": total_loss/len(train_loader),
                             "train_acc": metrics["accuracy"]})
    else:
        config = {
            "epochs": args.epochs,
            "warmup_epochs": args.warmup,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "save_dir": args.save,
            "device": device,
        }
        history = train(model, train_loader, test_loader, config)

    # ---- Load best checkpoint ----
    best_ckpt = os.path.join(args.save, "best_model.pth")
    if os.path.exists(best_ckpt) and args.arch != "baseline":
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"\nLoaded best checkpoint: {best_ckpt}")

    # ---- Evaluation ----
    results = full_evaluate(model, test_loader, device, T=args.mc_T)
    run_all_plots(history, results, save_dir="plots")

    # Save history
    hist_path = os.path.join(args.save, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ---- Optional Grad-CAM ----
    if args.gradcam:
        print("\nGenerating Grad-CAM visualizations...")
        batch_visualize(model, test_loader, device, n_samples=6)


if __name__ == "__main__":
    main()
