#!/usr/bin/env python
"""
Pre-extract CLIP image features for CoOp training
Saves GPU memory and training time
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Import all available datasets
from src.data.components.kd_dataloader import (
    AircraftDataset,
    OxfordIIPetDataset,
    CUB200Dataset as CUBDataset,
    StanfordCarsDataset,
    Caltech256Dataset,
    StanfordDogsDataset,
)


def extract_features(
    clip_model_name: str,
    pretrained: str,
    data_root: str,
    output_path: str,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Extract and save CLIP features for all images
    
    Args:
        clip_model_name: CLIP model name
        pretrained: Pretrained weights
        data_root: Dataset root directory
        output_path: Output file path (.pt)
        batch_size: Batch size (for extraction, can be set larger)
        num_workers: Number of data loading threads
    """
    print("="*60)
    print("CLIP Image Feature Extraction")
    print("="*60)
    print(f"Model: {clip_model_name}")
    print(f"Pretrained: {pretrained}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print("="*60 + "\n")
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    print("✓ Model loaded\n")
    
    # Dynamically detect dataset type
    print("Detecting dataset type from data_root...")
    if 'FGVC_AIRCRAFT' in data_root or 'aircraft' in data_root.lower():
        DatasetClass = AircraftDataset
        dataset_name = "FGVC Aircraft"
        val_split = 'val'  # Aircraft uses 'val'
    elif 'OxfordIIITPet' in data_root or 'pet' in data_root.lower():
        DatasetClass = OxfordIIPetDataset
        dataset_name = "Oxford-IIIT Pet"
        val_split = 'test'  # OxfordPets uses 'test'
    elif 'CUB' in data_root or 'cub' in data_root.lower():
        DatasetClass = CUBDataset
        dataset_name = "CUB-200-2011"
        val_split = 'test'  # CUB uses 'test'
    elif 'StanfordCars' in data_root or 'stanford_cars' in data_root.lower() or 'cars' in data_root.lower():
        DatasetClass = StanfordCarsDataset
        dataset_name = "Stanford Cars"
        val_split = 'test'  # StanfordCars uses 'test'
    elif 'StanfordDogs' in data_root or 'stanford_dogs' in data_root.lower() or 'dogs' in data_root.lower():
        DatasetClass = StanfordDogsDataset
        dataset_name = "Stanford Dogs"
        val_split = 'test'  # StanfordDogs uses 'test'
    elif 'CALTECH256' in data_root or 'caltech256' in data_root.lower() or 'ct256' in data_root.lower():
        DatasetClass = Caltech256Dataset
        dataset_name = "Caltech-256"
        val_split = 'test'  # Caltech256 uses 'test'
    else:
        raise ValueError(f"Cannot detect dataset type from data_root: {data_root}")
    
    print(f"✓ Detected dataset: {dataset_name}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = DatasetClass(data_root, split='train', transform=preprocess)
    val_dataset = DatasetClass(data_root, split=val_split, transform=preprocess)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}\n")
    
    # Extract training set features
    print("Extracting training features...")
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Train"):
            images = images.to(device)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            
            train_features.append(features.cpu())
            train_labels.append(labels)
    
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    print(f"✓ Train features shape: {train_features.shape}")
    
    # Extract validation set features
    print("\nExtracting validation features...")
    val_features = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val"):
            images = images.to(device)
            features = clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            
            val_features.append(features.cpu())
            val_labels.append(labels)
    
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    
    print(f"✓ Val features shape: {val_features.shape}")
    
    # Save features
    print(f"\nSaving features to {output_path}...")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'clip_model_name': clip_model_name,
        'pretrained': pretrained,
        'feature_dim': train_features.shape[1],
    }, output_path)
    
    print("✓ Features saved successfully!")
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Train: {len(train_features)} samples × {train_features.shape[1]} dims")
    print(f"  Val:   {len(val_features)} samples × {val_features.shape[1]} dims")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    print("="*60)


if __name__ == "__main__":
    import os
    
    # Read configuration from environment variables, use defaults if not set
    CLIP_MODEL = os.environ.get('CLIP_MODEL', 'convnext_xxlarge')
    PRETRAINED = os.environ.get('PRETRAINED', 'laion2b_s34b_b82k_augreg_soup')
    DATA_ROOT = os.environ.get('DATA_ROOT', '/data/lvta/datasets/vl2lite_datasets/1_FGVC_AIRCRAFT')
    OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/data/lvta/logs/vl2lite/train/runs/coop_fgvc_aircraft/features/fgvc_aircraft_clip_features.pt')
    
    extract_features(
        clip_model_name=CLIP_MODEL,
        pretrained=PRETRAINED,
        data_root=DATA_ROOT,
        output_path=OUTPUT_PATH,
        batch_size=64,  # Reduced from 256 - training is using ~15GB GPU memory
        num_workers=4,
    )
