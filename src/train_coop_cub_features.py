#!/usr/bin/env python
"""
CoOp Training Script for CUB-200-2011 with Precomputed Features
Uses precomputed CLIP image features to reduce memory usage from 15GB to ~3-4GB per GPU.
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
import yaml

from src.data.components.feature_dataset import FeatureDataset
from src.models.coop_module import CoOpModule


def load_cub_classnames(config_path: str):
    """Load classnames from config file and apply prompt template if specified."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract classnames in order
    # Some datasets use 0-indexed classes, others use 1-indexed
    classnames = []
    classes_dict = config['classes']
    
    # Check if 0-indexed or 1-indexed
    if 0 in classes_dict:
        # 0-indexed (e.g., Aircraft)
        for i in range(config['class_num']):
            classnames.append(classes_dict[i])
    else:
        # 1-indexed (e.g., StanfordCars)
        for i in range(1, config['class_num'] + 1):
            classnames.append(classes_dict[i])
    
    # Apply prompt template if specified (for domain-specific datasets)
    # This ensures CoOp uses the right context (e.g., "which is the model name of car")
    if 'prompt_tmpl' in config and '{}' in config['prompt_tmpl']:
        template = config['prompt_tmpl']
        classnames = [template.format(name) for name in classnames]
        print(f"Applied prompt template: '{template}'")
        print(f"Example classname: '{classnames[0]}'")
    
    return classnames, config


def main():
    # ============ Configuration ============
    # 从环境变量读取配置路径
    import os
    CONFIG_PATH = os.environ.get('CONFIG_PATH', '/home/lvta/Project/VL2Lite/configs/data/attributes/1_FGVC_AIRCRAFT.yaml')
    FEATURES_PATH = os.environ.get('FEATURES_PATH', '/data/lvta/logs/vl2lite/train/runs/coop_fgvc_aircraft/features/fgvc_aircraft_clip_features.pt')
    CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '/data/lvta/logs/vl2lite/train/runs/coop_fgvc_aircraft/checkpoints')
    
    # Only print on rank 0 (main process) in distributed training
    import os
    is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    if is_main_process:
        print(f"Loading configuration from: {CONFIG_PATH}")
    classnames, config = load_cub_classnames(CONFIG_PATH)
    if is_main_process:
        print(f"Loaded {len(classnames)} classes")
        print(f"First 5 classes: {classnames[:5]}")
    
    # Model settings (must match the precomputed features)
    # Read from environment variables to match extract_image_features.py
    CLIP_MODEL = os.environ.get('CLIP_MODEL', 'convnext_xxlarge')
    PRETRAINED = os.environ.get('PRETRAINED', 'laion2b_s34b_b82k_augreg_soup')
    N_CTX = 16
    # Use None for random initialization - model will learn from scratch
    # Note: Random init causes initial feature collapse (high similarity ~0.97)
    # but training should gradually differentiate features
    CTX_INIT = None
    CSC = False
    
    # Training settings
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.002
    NUM_WORKERS = 4
    
    # GPU settings
    USE_GPU = torch.cuda.is_available()
    NUM_GPUS = 4
    DEVICE_COUNT = torch.cuda.device_count() if USE_GPU else 0
    
    if is_main_process:
        print("\n" + "="*60)
        print("CoOp Training with Precomputed Features")
        print("="*60)
        print(f"Dataset: FGVC Aircraft ({len(classnames)} classes)")
        print(f"Features: {FEATURES_PATH}")
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
        print(f"GPU available: {USE_GPU} ({DEVICE_COUNT} devices)")
        print("="*60 + "\n")
    
    # ============ Load Precomputed Features ============
    if is_main_process:
        print(f"Loading precomputed features from {FEATURES_PATH}...")
    
    features_data = torch.load(FEATURES_PATH)
    
    # Verify feature metadata matches config
    assert features_data['clip_model_name'] == CLIP_MODEL, \
        f"Feature model mismatch: {features_data['clip_model_name']} != {CLIP_MODEL}"
    assert features_data['pretrained'] == PRETRAINED, \
        f"Pretrained mismatch: {features_data['pretrained']} != {PRETRAINED}"
    
    if is_main_process:
        print(f"✓ Features loaded successfully")
        print(f"  Feature dim: {features_data['feature_dim']}")
        print(f"  Train samples: {len(features_data['train_features'])}")
        print(f"  Val samples: {len(features_data['val_features'])}")
    
    # ============ Initialize Model ============
    if is_main_process:
        print("\nInitializing CoOp model with precomputed features mode...")
    
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
        max_epochs=NUM_EPOCHS,
        use_precomputed_features=True  # Enable feature mode
    )
    
    if is_main_process:
        print(f"✓ Model initialized in feature mode")
        print(f"  Context length: {model.model.clip_model.context_length}")
    
    # ============ Prepare Feature Datasets ============
    if is_main_process:
        print(f"\nPreparing feature datasets...")
    
    train_dataset = FeatureDataset(
        features=features_data['train_features'],
        labels=features_data['train_labels']
    )
    
    val_dataset = FeatureDataset(
        features=features_data['val_features'],
        labels=features_data['val_labels']
    )
    
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
    
    # 从CONFIG_PATH提取数据集名称
    dataset_name = os.path.basename(CONFIG_PATH).replace('.yaml', '').replace('_', '').lower()
    
    # ============ Setup Callbacks ============
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f"coop_{dataset_name}-epoch={{epoch:03d}}-valacc={{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False
    )
    
    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=50,
        mode="max",
        verbose=True
    )
    
    # ============ Setup Logger ============
    logger_dir = os.path.dirname(os.path.dirname(CHECKPOINT_DIR))  # .../coop_xxx -> .../runs
    logger = CSVLogger(
        # Put metrics under /.../train/runs/coop_features/<dataset>/<version>
        save_dir=os.path.join(logger_dir, 'coop_features'),
        name=dataset_name,
        version="fullshot_features"
    )
    
    # ============ Setup Trainer ============
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu' if USE_GPU else 'cpu',
        devices=NUM_GPUS if USE_GPU else 1,
        strategy='ddp' if NUM_GPUS > 1 else 'auto',
        precision='16-mixed' if USE_GPU else 32,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True
    )
    
    # ============ Start Training ============
    if is_main_process:
        print("\n" + "="*60)
        print("Starting CoOp training with precomputed features...")
        print("Expected memory usage: ~3-4GB per GPU (vs 15GB with online encoding)")
        print("="*60 + "\n")
    
    ckpt_path = os.environ.get("CKPT_PATH") or None
    if is_main_process:
        if ckpt_path:
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print("No CKPT_PATH provided, training from scratch")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    
    # ============ Final Report ============
    if is_main_process:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print(f"Best val/acc: {checkpoint_callback.best_model_score:.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
