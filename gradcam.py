"""
Step 6: Grad-CAM Defect Localization
Uses captum library to generate heatmaps over defect regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

# captum import (install: pip install captum)
try:
    from captum.attr import LayerGradCam, LayerAttribution
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("WARNING: captum not installed. Run: pip install captum")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def denormalize(tensor):
    """Convert normalized tensor back to displayable numpy image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def get_target_layer(model):
    """
    Return the last conv layer of the encoder for Grad-CAM.
    Works for ArchitectureA (ResNet50) and ArchitectureC (EfficientNet branch).
    """
    # For dual-branch model, use global_branch (EfficientNet)
    if hasattr(model, "global_branch"):
        # timm EfficientNet: last block is model.global_branch.blocks[-1]
        return model.global_branch.blocks[-1]
    # For single encoder (ResNet50)
    if hasattr(model, "encoder"):
        return model.encoder.layer4[-1]
    raise ValueError("Cannot find target layer in model")


def generate_gradcam(model, image_tensor, device, target_class=1):
    """
    Generate Grad-CAM heatmap for a single image.

    Args:
        model         : trained model (ArchitectureA or C)
        image_tensor  : (1, 3, 224, 224) preprocessed tensor
        device        : torch device
        target_class  : 1 = defect class

    Returns:
        heatmap : (H, W) numpy array, values in [0, 1]
    """
    if not CAPTUM_AVAILABLE:
        raise RuntimeError("Install captum: pip install captum")

    model.eval()
    image_tensor = image_tensor.to(device)

    target_layer = get_target_layer(model)

    # Wrapper: GradCAM only needs task logits
    def forward_func(x):
        logits, _, _ = model(x, lambda_=0.0)
        return logits

    gc = LayerGradCam(forward_func, target_layer)
    attribution = gc.attribute(image_tensor, target=target_class)  # (1, C, h, w)

    # Upsample to input size
    upsampled = LayerAttribution.interpolate(attribution, (224, 224))  # (1, C, 224, 224)
    heatmap   = upsampled[0].mean(dim=0).detach().cpu().numpy()        # (224, 224)

    # Normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def visualize_gradcam(model, image_tensor, device, save_path=None, title="Grad-CAM"):
    """
    Generate and display (or save) a side-by-side visualization:
    original image | Grad-CAM overlay
    """
    heatmap = generate_gradcam(model, image_tensor, device)

    orig_img = denormalize(image_tensor.squeeze(0))  # (224, 224, 3)

    # Colorize heatmap
    colored   = cm.jet(heatmap)[..., :3]             # (224, 224, 3)
    overlay   = 0.5 * orig_img + 0.5 * colored
    overlay   = overlay.clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_img);    axes[0].set_title("Original");     axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title("Overlay");       axes[2].axis("off")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()
    return heatmap


def batch_visualize(model, loader, device, n_samples=6, save_dir="gradcam_outputs"):
    """Run Grad-CAM on the first n_samples from a DataLoader."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    preprocess = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    count = 0

    for imgs, labels, _ in loader:
        for i in range(imgs.size(0)):
            if count >= n_samples:
                return
            img_t = imgs[i:i+1]
            label = labels[i].item()
            label_str = "defect" if label == 1 else "normal"

            visualize_gradcam(
                model, img_t, device,
                save_path=os.path.join(save_dir, f"sample_{count}_{label_str}.png"),
                title=f"Sample {count} | GT: {label_str}",
            )
            count += 1
