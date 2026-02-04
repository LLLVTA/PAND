"""Feature Dataset for precomputed CLIP image features."""

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Dataset wrapper for precomputed CLIP image features.
    
    Args:
        features: Tensor of shape [N, D] containing image features
        labels: Tensor of shape [N] containing class labels
    """
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert len(features) == len(labels), "Features and labels must have same length"
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
