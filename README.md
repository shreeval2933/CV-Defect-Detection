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

## 📊 Results & Analysis (Mean ± Std over 3 runs)

To evaluate robustness, we conduct experiments in two settings:

- **Cross-domain:** Train on MVTec → Test on AITEX (unseen domain)
- **Same-domain:** Train on MVTec → Test on held-out MVTec split

---

## 🌍 Cross-Domain Performance (MVTec → AITEX)

| Model | Accuracy | F1 Score | AUROC |
|------|---------|---------|-------|
| Baseline | 0.569 ± 0.028 | 0.473 ± 0.028 | 0.495 ± 0.034 |
| **Arch A (EMA)** | **0.628 ± 0.016** | **0.646 ± 0.015** | **0.596 ± 0.007** |

### 🔍 Observations

- **Significant improvement with domain adaptation**
  - F1 Score improves from **0.47 → 0.65** (~37% relative gain)
- **Better class separability**
  - AUROC increases from ~0.50 (near random) → ~0.60
- **Higher stability**
  - Lower variance across all metrics using EMA

---

## 🏭 Same-Domain Performance (MVTec → MVTec)

| Model | Accuracy | F1 Score | AUROC |
|------|---------|---------|-------|
| Baseline | 0.816 ± 0.032 | **0.625 ± 0.121** | **0.861 ± 0.060** |
| **Arch A (EMA)** | **0.837 ± 0.004** | 0.537 ± 0.006 | 0.843 ± 0.005 |

### 🔍 Observations

- **Overall performance is significantly higher than cross-domain**
  - AUROC improves to ~0.84–0.86
- **Baseline achieves higher peak performance**
  - But shows **very high variance (unstable)**
- **Arch A is highly consistent**
  - Very low standard deviation across runs

---

## ⚖️ Cross-Domain vs Same-Domain Trade-off

| Model | Same-Domain AUROC | Cross-Domain AUROC | Behavior |
|------|------------------|--------------------|----------|
| Baseline | **0.861** | 0.495 | Overfits to source domain |
| Arch A | 0.843 | **0.596** | Generalizes better |

---

## 🎯 Key Takeaways

### 🚀 1. Domain Adaptation is Effective
- Arch A significantly improves performance on unseen domain
- Learns **domain-invariant features**

---

### 📉 2. Baseline Overfits
- Strong same-domain performance
- Fails under domain shift

---

### 📈 3. Stability Matters (EMA Impact)

| Metric | Baseline Std | Arch A Std |
|-------|-------------|------------|
| Accuracy | 0.028 | 0.016 |
| F1 | 0.028 | 0.015 |
| AUROC | 0.034 | 0.007 |

👉 EMA greatly reduces variance and stabilizes training

---

### ⚖️ 4. Trade-off Between Accuracy & Generalization
- Baseline:
  - Higher same-domain peak
  - Poor generalization
- Arch A:
  - Slight drop in same-domain F1
  - Strong cross-domain improvement

---

## 🧠 Final Conclusion

- **Baseline model memorizes source domain**
- **Architecture A learns transferable representations**

👉 Result:
- Improved robustness across domains  
- More reliable and consistent performance  
- Better suited for real-world deployment  

---

## ⚠️ Limitations

- Cross-domain AUROC still < 0.65 → moderate performance
- Model uncertainty remains relatively high (entropy ~0.68)
- Domain gap is not fully eliminated

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
- Domain adaptation with GRL and EMA significantly improves cross-domain defect detection performance.
- Higher F1 score
- Better AUROC
- Lower variance across runs
- Demonstrates effectiveness of learning domain-invariant representations.

## ⚠️ Notes
- Dataset paths are configurable in main.py
- Large checkpoints are excluded from repository
- Only best models are stored for reproducibility

## 📌 Future Work
- Improve AUROC (>0.65)
- Try stronger backbones (EfficientNet, ViT)
- Better domain alignment techniques
- Semi-supervised adaptation