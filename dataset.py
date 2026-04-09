"""
Step 1: Dataset Preparation
Handles MVTec (metal, plastic) and AITEX (fabric) datasets.
Each sample: (image, label, domain_id)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np


# Domain IDs
DOMAIN_MAP = {
    "metal": 0,
    "plastic": 1,
    "fabric": 2,
}

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
    """
    MVTec AD dataset loader.
    Expected structure:
        mvtec/<category>/train/good/*.png
        mvtec/<category>/test/good/*.png
        mvtec/<category>/test/<defect_type>/*.png

    domain_name: 'metal' | 'plastic'
    category:    e.g. 'metal_nut', 'bottle', 'tile'
    split:       'train' | 'test'
    """

    def __init__(self, root, category, domain_name, split="train", transform=None):
        self.transform = transform
        self.domain_id = DOMAIN_MAP[domain_name]
        self.samples = []  # list of (path, label)

        # MVTec's default 'train' folder only has normal ('good') images.
        # To get defect samples for supervised training, we MUST load from 'test' too.
        all_normal = []
        all_defect = []

        for sub_split in ["train", "test"]:
            category_dir = os.path.join(root, category, sub_split)
            if not os.path.exists(category_dir):
                continue

            for cls_name in os.listdir(category_dir):
                cls_dir = os.path.join(category_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        path = os.path.join(cls_dir, fname)
                        if cls_name == "good":
                            all_normal.append(path)
                        else:
                            all_defect.append(path)

        # Simple split logic in case we want to evaluate on MVTec directly later
        train_ratio = 0.8
        n_normal_train = int(len(all_normal) * train_ratio)
        n_defect_train = int(len(all_defect) * train_ratio)

        if split == "train":
            normal_set = all_normal[:n_normal_train]
            defect_set = all_defect[:n_defect_train]
        elif split == "test":
            normal_set = all_normal[n_normal_train:]
            defect_set = all_defect[n_defect_train:]
        else: # "all"
            normal_set = all_normal
            defect_set = all_defect

        # Populate samples list
        self.samples.extend([(f, 0) for f in normal_set])
        self.samples.extend([(f, 1) for f in defect_set])

        # After building self.samples, compute class distribution and weights
        labels = [lbl for _, lbl in self.samples]
        counts = np.bincount(labels, minlength=2)
        total = counts.sum()
        # Inverse frequency weighting (avoid division by zero)
        self.class_weights = (total / (2 * counts + 1e-8)).astype(np.float32)
        # Store counts for possible debugging
        self.class_counts = counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, self.domain_id


class AITEXDataset(Dataset):
    """
    AITEX fabric defect dataset loader.
    Expected structure:
        aitex/NODefect_images/*.png   -> label 0
        aitex/Defect_images/*.png     -> label 1

    split: 'train' | 'test'  (80/20 split applied here)
    """

    def __init__(self, root, split="train", transform=None, train_ratio=0.8):
        self.transform = transform
        self.domain_id = DOMAIN_MAP["fabric"]
        self.samples = []

        for subdir, label in [("NODefect_images", 0), ("Defect_images", 1)]:
            folder = os.path.join(root, subdir)
            if not os.path.exists(folder):
                raise FileNotFoundError(f"AITEX folder not found: {folder}")
            files = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".bmp"))
            ])
            n_train = int(len(files) * train_ratio)
            files = files[:n_train] if split == "train" else files[n_train:]
            # Populate samples list
            self.samples.extend([(f, label) for f in files])

        # After building self.samples, compute class distribution and weights
        labels = [lbl for _, lbl in self.samples]
        counts = np.bincount(labels, minlength=2)
        total = counts.sum()
        self.class_weights = (total / (2 * counts + 1e-8)).astype(np.float32)
        self.class_counts = counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, self.domain_id


def build_dataloaders(mvtec_root, aitex_root, batch_size=32, num_workers=4):
    """
    Train on: metal + plastic (MVTec)
    Test  on: fabric (AITEX)   <- cross-domain evaluation
    Returns: train_loader, test_loader, n_domains
    """
    train_tf = get_train_transforms()
    test_tf  = get_test_transforms()

    # ---- TRAINING domains ----
    metal_train   = MVTecDataset(mvtec_root, "metal_nut", "metal",   "all", train_tf)
    plastic_train = MVTecDataset(mvtec_root, "bottle",    "plastic", "all", train_tf)
    train_dataset = ConcatDataset([metal_train, plastic_train])
    # No need to attach class_weights to ConcatDataset; we'll compute per-sample weights directly.

    # ---- TEST domain (unseen) ----
    fabric_test = AITEXDataset(aitex_root, "test", test_tf)

    # Compute per-sample weights for balanced sampling across both sub‑datasets
    sample_weights = []
    for ds in train_dataset.datasets:
        # ds is either metal_train or plastic_train
        sample_weights.extend([ds.class_weights[label] for _, label in ds.samples])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    # Note: shuffle is omitted because sampler provides randomness
    test_loader  = DataLoader(fabric_test, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    n_domains = len(DOMAIN_MAP)
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(fabric_test)}")
    return train_loader, test_loader, n_domains
