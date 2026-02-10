#!/usr/bin/env python
"""
CoOp Training Script for CUB-200-2011 Dataset
Birds species classification with 200 classes.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml

# Import VL2Lite's CUB dataset
from src.data.components.kd_dataloader import CUB200Dataset

from src.models.coop_module import CoOpModule


def load_cub_classnames(config_path: str):
    """Load CUB-200-2011 classnames from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract classnames in order
    classnames = []
    for i in range(1, config['class_num'] + 1):
        classnames.append(config['classes'][i])
    
    return classnames, config


def main():
    # ============ Configuration ============
    # Load CUB-200-2011 configuration
    CONFIG_PATH = "/home/lvta/Project/VL2Lite/configs/data/attributes/0_CUB_200_2011.yaml"
    DATA_ROOT = "/data/lvta/datasets/vl2lite_datasets/0_CUB_200_2011"
    
    # Only print on rank 0 (main process) in distributed training
    import os
    is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    if is_main_process:
        print(f"Loading CUB-200-2011 configuration from: {CONFIG_PATH}")
    classnames, config = load_cub_classnames(CONFIG_PATH)
    if is_main_process:
        print(f"Loaded {len(classnames)} bird species classes")
        print(f"First 5 classes: {classnames[:5]}")
    
    # Model settings
    CLIP_MODEL = "convnext_xxlarge"  # or "ViT-B-32" for faster testing
    PRETRAINED = "laion2b_s34b_b82k_augreg_soup"
    N_CTX = 16  # Number of learnable context tokens
    CTX_INIT = None  # Random initialization, or use "a photo of a bird" for guided init
    CSC = False  # Class-specific context: True=each class has own context, False=shared context
    
    # Training settings
    BATCH_SIZE = 32  # Reduce to 16 if OOM
    NUM_EPOCHS = 200  # CoOp paper setting for few-shot
    LEARNING_RATE = 0.002  # SGD learning rate
    NUM_WORKERS = 4
    
    # Few-shot settings (optional, set to -1 for full dataset)
    # NUM_SHOTS = 16  # 16-shot per class (CoOp paper setting)
    NUM_SHOTS = -1  # Use full training set (FULL-SHOT)
    
    # GPU settings
    USE_GPU = torch.cuda.is_available()
    NUM_GPUS = 4  # Use 4 GPUs (per-device batch: 32, global batch: 128)
    DEVICE_COUNT = torch.cuda.device_count() if USE_GPU else 0
    
    if is_main_process:
        print("\n" + "="*60)
        print("CoOp Training Configuration")
        print("="*60)
        print(f"Dataset: CUB-200-2011 ({len(classnames)} classes)")
        print(f"Data root: {DATA_ROOT}")
        print(f"CLIP Model: {CLIP_MODEL}")
        print(f"Pretrained: {PRETRAINED}")
        print(f"Context tokens (n_ctx): {N_CTX}")
        print(f"Context init: {CTX_INIT}")
        print(f"CSC (class-specific context): {CSC}")
        print(f"Per-device batch size: {BATCH_SIZE}")
        print(f"Global batch size: {BATCH_SIZE * NUM_GPUS}")
        print(f"Number of GPUs: {NUM_GPUS}")
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Few-shot: {NUM_SHOTS} shots per class" if NUM_SHOTS > 0 else "Full dataset")
        print(f"GPU available: {USE_GPU} ({DEVICE_COUNT} devices)")
        print("="*60 + "\n")
    
    # ============ Initialize Model ============
    if is_main_process:
        print("Initializing CoOp model...")
    model = CoOpModule(
        clip_model_name=CLIP_MODEL,
        pretrained=PRETRAINED,
        classnames=classnames,
        n_ctx=N_CTX,
        ctx_init=CTX_INIT,
        csc=CSC,
        learning_rate=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0,
        max_epochs=NUM_EPOCHS
    )
    
    if is_main_process:
        print(f"✓ Model initialized")
        print(f"  Tokenizer: {type(model.model.clip_model.tokenizer).__name__}")
        print(f"  Context length: {model.model.clip_model.context_length}")
    
    # ============ Prepare Data ============
    if is_main_process:
        print(f"\nPreparing CUB-200-2011 dataset from {DATA_ROOT}...")
    
    # CRITICAL: Use model's preprocess transforms (from CLIP)!
    train_transform = model.preprocess_train
    val_transform = model.preprocess_val
    
    # Use VL2Lite's CUB200Dataset with CLIP transforms
    train_dataset = CUB200Dataset(
        root_dir=DATA_ROOT,
        split='train',
        transform=train_transform
    )
    
    val_dataset = CUB200Dataset(
        root_dir=DATA_ROOT,
        split='test',
        transform=val_transform
    )
    
    # Few-shot sampling (if enabled)
    if NUM_SHOTS > 0:
        print(f"\nApplying {NUM_SHOTS}-shot sampling...")
        from torch.utils.data import Subset
        import numpy as np
        
        # Group by class
        class_to_indices = {i: [] for i in range(len(classnames))}
        for idx in range(len(train_dataset)):
            label = train_dataset.labels[idx] - 1  # CUB200Dataset stores 1-indexed labels
            class_to_indices[label].append(idx)
        
        # Sample NUM_SHOTS examples per class
        indices = []
        for class_idx in range(len(classnames)):
            class_indices = class_to_indices[class_idx]
            if len(class_indices) >= NUM_SHOTS:
                selected = np.random.choice(class_indices, NUM_SHOTS, replace=False)
                indices.extend(selected)
            else:
                print(f"  Warning: Class {class_idx} has only {len(class_indices)} samples")
                indices.extend(class_indices)
        
        train_dataset = Subset(train_dataset, indices)
        print(f"  Selected {len(indices)} training samples (~{NUM_SHOTS} per class)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    if is_main_process:
        print(f"✓ Datasets prepared")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    
    # ============ Setup Callbacks ============
    checkpoint_callback = ModelCheckpoint(
        dirpath="/data/lvta/logs/vl2lite/train/runs/coop_cub/checkpoints",
        filename="coop_cub-{epoch:03d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/acc",
        mode="max",
        patience=50,
        verbose=True,
        min_delta=0.001
    )
    
    # ============ Setup Logger ============
    logger = CSVLogger(
        save_dir="/data/lvta/logs/vl2lite/train/runs/coop",
        name="cub_200_2011",
        version=f"{NUM_SHOTS}shot" if NUM_SHOTS > 0 else "fullshot"
    )
    
    # ============ Setup Trainer ============
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu" if USE_GPU else "cpu",
        devices=NUM_GPUS,  # Use 4 GPUs with DDP
        strategy="ddp",  # Distributed Data Parallel
        precision="16-mixed" if USE_GPU else "32",  # Mixed precision for speed
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=5,
        gradient_clip_val=None,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # ============ Train ============
    if is_main_process:
        print("\n" + "="*60)
        print("Starting CoOp Training")
        print("="*60)
        print(f"Checkpoint dir: outputs/coop_cub/checkpoints")
        print(f"CSV logs: logs/coop/cub_200_2011/fullshot/metrics.csv")
        print("="*60 + "\n")
    
    try:
        trainer.fit(model, train_loader, val_loader)
    except KeyboardInterrupt:
        if is_main_process:
            print("\n\nTraining interrupted by user!")
    except Exception as e:
        if is_main_process:
            print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ Training Summary ============
    if is_main_process:
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
    
    if checkpoint_callback.best_model_path and is_main_process:
        print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"✓ Best val/acc: {checkpoint_callback.best_model_score:.4f}")
        
        # ============ Feature Extraction Instructions ============
        print("\n" + "-"*60)
        print("Next Step: Extract Text Features")
        print("-"*60)
        print("Run the following command to extract learned text features:")
        print(f"\npython scripts/extract_coop_features.py \\")
        print(f"  --checkpoint {checkpoint_callback.best_model_path} \\")
        print(f"  --output data/coop_cub_200_2011_{NUM_SHOTS}shot.pt \\")
        print(f"  --clip_model_name {CLIP_MODEL} \\")
        print(f"  --pretrained {PRETRAINED} \\")
        print(f"  --device cuda")
        print("\nThis will save text features for all 200 bird classes.")
        print("These features can then be used in VL2Lite Stage B training.")
    else:
        print("⚠️  No checkpoint saved (training may have failed early)")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    main()
