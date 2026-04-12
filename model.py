"""
Step 2-5: Model Architecture
- Step 2: ResNet50 Baseline
- Step 3: Domain Adversarial Training with GRL (Architecture A)
- Step 4: Dual-Branch Architecture C (EfficientNet + WideResNet fusion)
- Step 5: MC Dropout for uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import timm


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """
    Forward pass: identity.
    Backward pass: multiply gradient by -lambda_ (reversal).
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


# ---------------------------------------------------------------------------
# Step 2: Baseline Model (ResNet50 only)
# ---------------------------------------------------------------------------

class BaselineResNet(nn.Module):
    """Simple ResNet50 → FC → defect/normal (no domain adaptation)."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=0  # strip classifier
        )
        feat_dim = self.encoder.num_features  # 2048

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lambda_=None):
        z = self.encoder(x)           # (B, 2048)
        logits = self.classifier(z)

        # return same format as other architectures
        return logits, None, z


# ---------------------------------------------------------------------------
# Step 3: Architecture A – Minimal DANN (single encoder + GRL)
# UPDATED: Reduced model capacity + reduced dropout for stability
# ---------------------------------------------------------------------------

class ArchitectureA(nn.Module):
    """
    Image → Encoder → Feature z
        z → Task Head      → defect prediction
        z → GRL → Domain Classifier → domain prediction

    UPDATED:
    ✔ ResNet50 → ResNet18 (reduces overfitting)
    ✔ Dropout 0.5 → 0.3 (improves stability)
    """

    def __init__(self, num_domains=3, num_classes=2, pretrained=True):
        super().__init__()

        # -------------------------------------------------------------------
        # UPDATED: Smaller backbone (reduces overfitting)
        # -------------------------------------------------------------------
        self.encoder = timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.encoder.num_features  # 512 (instead of 2048)

        # -------------------------------------------------------------------
        # Task head (reduced dropout)
        # -------------------------------------------------------------------
        self.task_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),   # UPDATED (was 0.5)
            nn.Linear(256, num_classes),
        )

        # -------------------------------------------------------------------
        # Domain classifier
        # -------------------------------------------------------------------
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_domains),
        )

    def forward(self, x, lambda_=1.0):
        z = self.encoder(x)

        task_out = self.task_head(z)

        z_rev = grad_reverse(z, lambda_)
        domain_out = self.domain_classifier(z_rev)

        return task_out, domain_out, z


# ---------------------------------------------------------------------------
# Step 4: Architecture C – Dual-Branch (EfficientNet + ResNet18 + fusion)
# UPDATED: Reduced model size + reduced dropout for better generalization
# ---------------------------------------------------------------------------

class DualBranchArchitectureC(nn.Module):
    """
    UPDATED ARCHITECTURE (LIGHTWEIGHT + STABLE)

    Branch 1 (Global): EfficientNet-B0 → g   (reduced from B3)
    Branch 2 (Local) : ResNet18       → l   (replaced WideResNet50)
    Fusion           : concat(g, l) → FC → z

    Improvements:
    ✔ Reduced parameters (~79M → ~15–20M)
    ✔ Less overfitting
    ✔ Faster training
    ✔ Better cross-domain generalization
    """

    def __init__(self, num_domains=3, num_classes=2, pretrained=True):
        super().__init__()

        # -------------------------------------------------------------------
        # Branch 1 – EfficientNet-B0 (lighter global features)
        # -------------------------------------------------------------------
        self.global_branch = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        g_dim = self.global_branch.num_features  # ~1280

        # -------------------------------------------------------------------
        # Branch 2 – ResNet18 (lighter local features)
        # -------------------------------------------------------------------
        self.local_branch = timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=0
        )
        l_dim = self.local_branch.num_features  # ~512

        fusion_dim = 256   # reduced from 512 (helps regularization)

        # -------------------------------------------------------------------
        # Feature fusion
        # Reduced dropout (0.5 → 0.3)
        # -------------------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(g_dim + l_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),   # reduced dropout
        )

        # -------------------------------------------------------------------
        # Task head
        # -------------------------------------------------------------------
        self.task_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),   # reduced dropout
            nn.Linear(128, num_classes),
        )

        # -------------------------------------------------------------------
        # Domain classifier (unchanged but stable now)
        # -------------------------------------------------------------------
        self.domain_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains),
        )

    def forward(self, x, lambda_=1.0):
        # Extract features
        g = self.global_branch(x)
        l = self.local_branch(x)

        # Fuse features
        z = self.fusion(torch.cat([g, l], dim=1))

        # Task prediction
        task_out = self.task_head(z)

        # Domain prediction (GRL)
        z_rev = grad_reverse(z, lambda_)
        domain_out = self.domain_classifier(z_rev)

        return task_out, domain_out, z
        

# ---------------------------------------------------------------------------
# Step 5: Uncertainty Estimation via MC Dropout
# ---------------------------------------------------------------------------

@torch.no_grad()
def mc_dropout_predict(model, x, T=30, num_classes=2):
    """
    Run model T times in train mode (activates dropout stochastically).
    Returns:
        mean_probs  : (B, num_classes)
        variance    : (B,)             <- predictive uncertainty
        entropy     : (B,)             <- entropy of mean distribution
    """
    model.train()   # IMPORTANT: keep dropout active

    preds = []
    for _ in range(T):
        logits, _, _ = model(x, lambda_=0.0)  # no GRL during inference
        probs = F.softmax(logits, dim=-1)      # (B, C)
        preds.append(probs.unsqueeze(0))       # (1, B, C)

    preds = torch.cat(preds, dim=0)            # (T, B, C)

    mean_probs = preds.mean(dim=0)             # (B, C)
    variance   = preds.var(dim=0).mean(dim=1)  # (B,)  mean variance over classes
    entropy    = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=1)  # (B,)

    return mean_probs, variance, entropy
