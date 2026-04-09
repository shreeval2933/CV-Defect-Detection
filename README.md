# CV Defect Detection – Domain Adversarial Training

**Multi‑domain industrial defect detection** featuring:
- **Domain Adversarial Training (DANN / GRL)** – learns domain‑invariant features
- **Monte‑Carlo Dropout** – predictive uncertainty estimation
- **Grad‑CAM (captum)** – visual defect localisation
- **Cross‑domain evaluation** – train on metal + plastic, test on unseen fabric

---

## Project Structure

```
CV project/
├── dataset.py          # MVTec + AITEX loaders, class‑balanced sampling
├── model.py            # Baseline, DANN (arch A), Dual‑branch (arch C), MC‑Dropout
├── train.py            # Training loop with GRL λ‑scheduling
├── evaluate.py         # Metrics, MC‑Dropout uncertainty, plotting utilities
├── gradcam.py          # Grad‑CAM visualisation via Captum
├── main.py             # CLI entry point – selects architecture & options
├── requirements.txt    # Python dependencies
└── plots/              # Generated figures (saved at runtime)
```

---

## Setup

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Layout

### Option 1: Bundled Dataset (Recommended)
You can download the pre-packaged dataset (including both MVTec categories and AITEX) directly from the project release:
*   **[Download data.zip](https://github.com/shreeval2933/CV-Defect-Detection/releases/download/v1.0.0/data.zip)**

### Option 2: Manual Collection
**MVTec** (download from https://www.mvtec.com/company/research/datasets/mvtec-ad):
```
data/mvtec/
├── metal_nut/
│   ├── train/good/
│   └── test/{good, bent, color, ...}/
└── bottle/
    ├── train/good/
    └── test/{good, broken_large, ...}/
```

**AITEX** (download from https://www.aitex.es/afid/):
```
data/aitex/
├── NODefect_images/   # label 0 (normal)
└── Defect_images/     # label 1 (defect)
```

The `dataset.py` loader automatically creates a **ConcatDataset** of the two MVTec categories for training and uses the AITEX dataset for the unseen test domain.

---

## Quick Start

All experiments are launched via `main.py`. The most common commands are:

```bash
# Baseline (no domain adaptation)
python main.py --arch baseline --epochs 20

# Architecture A – DANN with GRL (warm‑up λ=0 for first 5 epochs)
python main.py --arch a --epochs 20 --warmup 5

# Architecture C – Dual‑branch + DANN
python main.py --arch c --epochs 20 --warmup 5

# Enable Grad‑CAM visualisation (requires `--gradcam` flag)
python main.py --arch a --epochs 20 --gradcam
```

The script will:
1. Build balanced training loaders with per‑sample weights.
2. Train the selected architecture, printing λ‑schedule, loss, accuracy, F1 and AUROC each epoch.
3. Save the best model (by AUROC) to `checkpoints/best_model.pth`.
4. After training, `evaluate.py` is invoked to compute metrics on the unseen fabric domain and generate plots under `plots/`.

---

## Key Concepts

| Component | Description |
|---|---|
| **GRL (Gradient Reversal Layer)** | Reverses gradients for the domain classifier, encouraging domain‑invariant features. |
| **λ‑scheduling** | λ = 0 for the first `warmup` epochs, then follows the DANN exponential schedule. |
| **MC Dropout** | Performs `T=30` stochastic forward passes at inference to obtain predictive variance & entropy. |
| **Cross‑domain eval** | Trains on metal + plastic (MVTec) and evaluates on fabric (AITEX) to measure generalisation. |

---

## Expected Results (approximate)

| Setting | AUROC |
|---|---|
| Baseline (no DAT) | ~0.70 |
| Architecture A (DANN) | ~0.78 |
| Architecture C (dual‑branch) | ~0.82 |

These numbers are indicative; actual performance depends on hardware, random seed and training duration.

---

## Priority Order (from the original plan)

1. ✅ Dataset loading & baseline training
2. ✅ Domain Adversarial Training (Architecture A)
3. ✅ Cross‑domain evaluation (fabric test set)
4. ✅ Uncertainty estimation via MC‑Dropout
5. ⬜ Dual‑branch Architecture C (if time permits)
6. ⬜ Grad‑CAM visualisation (if time permits)

---

*Feel free to adjust hyper‑parameters (learning rate, batch size, epochs, warm‑up) directly in `main.py` or via the CLI flags.*
