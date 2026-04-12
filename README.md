# Cross-Domain Defect Detection using Domain Adaptation

**Industrial defect detection under domain shift** using:
- Domain Adversarial Training (GRL / DANN)
- Exponential Moving Average (EMA) for stable training
- Monte Carlo Dropout for uncertainty estimation
- Grad-CAM for visual explainability

---

## 🧩 Problem Statement

In real-world industrial settings, models trained on one domain often fail on unseen domains.

### Setup:
- **Train Domain:** Metal + Plastic (MVTec)
- **Test Domain:** Fabric (AITEX) → unseen domain

👉 This creates a **domain shift problem**, where:
- Data distributions differ
- Model generalization degrades

---

## ⚙️ Methods

### 1. Baseline Model
- ResNet50 classifier
- Trained only on source domain
- No domain adaptation

---

### 2. Architecture A (Proposed)
- Domain Adversarial Neural Network (DANN)
- Gradient Reversal Layer (GRL)
- Learns **domain-invariant features**

---

### 3. EMA (Exponential Moving Average)
- Maintains moving average of weights
- Used **during evaluation + checkpoint saving**
- Improves:
  - stability
  - generalization
  - consistency across runs

---

### 4. Additional Components
- **MC Dropout (T=30)** → predictive uncertainty
- **Grad-CAM (Captum)** → defect localization
- **Threshold tuning** → optimal classification

---

## 📊 Final Results (Mean ± Std over 3 runs)

| Model | Accuracy | F1 Score | AUROC |
|------|---------|---------|-------|
| Baseline | 0.569 ± 0.028 | 0.473 ± 0.028 | 0.495 ± 0.034 |
| **Arch A (EMA)** | **0.628 ± 0.016** | **0.646 ± 0.015** | **0.596 ± 0.007** |

---

## 🔥 Key Observations

### 🚀 1. Strong Improvement over Baseline
- F1 Score:
  - Baseline → **0.47**
  - Arch A → **0.65**
- 👉 ~**37% relative improvement**

---

### 📈 2. Better Class Separation
- AUROC improved from ~0.50 → ~0.60
- Model moves from:
  - ❌ near-random
  - ✅ meaningful discrimination

---

### 🎯 3. Higher Stability (Very Important)

| Metric | Baseline Std | Arch A Std |
|-------|-------------|------------|
| Accuracy | 0.028 | 0.016 |
| F1 | 0.028 | 0.015 |
| AUROC | 0.034 | 0.007 |

👉 EMA significantly reduces variance

---

### 🧠 4. Domain Adaptation Works
- Baseline fails on unseen domain
- Arch A consistently improves performance

---

### ⚠️ 5. Limitations
- AUROC still < 0.65 → moderate performance
- Entropy ~0.68 → model uncertainty remains

---

## 📸 Visual Results

### Included in `plots/`:
- Confusion Matrix
- ROC Curve
- Training Curves
- Uncertainty Distribution
- Grad-CAM Visualizations

👉 Grad-CAM shows model focusing on defect regions

---

## ▶️ How to Run

### Train model
```bash
python main.py --arch a --run_id 1
```

### Evaluate best checkpoint
```bash
python select_best_checkpoint.py --arch a --run_id 1 --gradcam
python main.py --arch baseline --run_id 1
```

## 📁 Project Structure
```bash
CV-Defect-Detection/
│
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── gradcam.py
├── main.py
├── select_best_checkpoint.py
│
├── results/
│   └── a/run_1/
│       ├── train/
│       │   ├── history.json
│       │   └── train.log
│       └── evaluate/
│           ├── eval.log
│           ├── best.txt
│           └── checkpoint_eval.json
│
├── plots/
│   └── a/run_1/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── training_curves.png
│       ├── uncertainty.png
│       └── gradcam/
│
├── checkpoints/
│   └── a/run_1/
│       └── best_model.pth
```

## 🧠 Key Concepts
- GRL: Encourages domain-invariant features
- EMA: Stabilizes training & evaluation
- MC Dropout: Estimates prediction uncertainty
- Cross-domain Eval	Tests: Generalization

## 🧾 Conclusion
Domain adaptation with GRL and EMA significantly improves cross-domain defect detection performance.
Higher F1 score
Better AUROC
Lower variance across runs
👉 Demonstrates effectiveness of learning domain-invariant representations.

## ⚠️ Notes
Dataset paths are configurable in main.py
Large checkpoints are excluded from repository
Only best models are stored for reproducibility

## 📌 Future Work
Improve AUROC (>0.65)
Try stronger backbones (EfficientNet, ViT)
Better domain alignment techniques
Semi-supervised adaptation