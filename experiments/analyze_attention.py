"""
Analyze model attention using Grad-CAM to understand if misclassifications
are due to background vs foreground focus.

Usage:
    python experiments/analyze_attention.py \
        --checkpoint_path logs/train/runs/xxx/checkpoints/last.ckpt \
        --error_csv /data/lvta/fault_analysis/0_CUB_200_2011/distilled/error_cases_distilled.csv \
        --output_dir /data/lvta/attention_analysis \
        --num_samples 50
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from tqdm import tqdm
import cv2

from src.models.kd_module import KDModule
from torchvision import transforms
from omegaconf import OmegaConf


class GradCAM:
    """Grad-CAM implementation for ResNet18"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            x: Input image tensor [1, 3, H, W]
            class_idx: Target class index. If None, use predicted class
        
        Returns:
            cam: Heatmap [H, W]
            pred_class: Predicted class index
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        # Get prediction - handle different output formats
        if isinstance(output, tuple):
            if len(output) == 6:
                # Distilled model: (hidden_features, logits, clip_img, frozen_nlp, aligned_img, aligned_nlp)
                logits = output[1]
            elif len(output) == 2:
                # StudentNet only: (hidden_features, logits)
                logits = output[1]
            else:
                logits = output[0]
        else:
            logits = output
        
        pred_class = torch.argmax(logits, dim=1).item()
        
        if class_idx is None:
            class_idx = pred_class
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)  # [H, W]
        cam = F.relu(cam)  # ReLU to keep positive values
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), pred_class


def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image
    
    Args:
        img: Original image (PIL Image or numpy array)
        heatmap: Heatmap [H, W] in range [0, 1]
        alpha: Transparency of heatmap
    
    Returns:
        overlay: Overlayed image as numpy array
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = (1 - alpha) * img + alpha * heatmap_colored
    overlay = overlay.astype(np.uint8)
    
    return overlay, heatmap_resized


def calculate_attention_metrics(heatmap, threshold=0.5):
    """
    Calculate attention distribution metrics
    
    Args:
        heatmap: Attention heatmap [H, W] in range [0, 1]
        threshold: Threshold to consider as "attended"
    
    Returns:
        metrics: Dictionary of metrics
    """
    # Binarize attention map
    attention_mask = heatmap > threshold
    
    # Calculate attention area ratio
    attention_ratio = attention_mask.sum() / (heatmap.shape[0] * heatmap.shape[1])
    
    # Calculate attention center
    if attention_mask.sum() > 0:
        y_coords, x_coords = np.where(attention_mask)
        center_y = y_coords.mean() / heatmap.shape[0]
        center_x = x_coords.mean() / heatmap.shape[1]
    else:
        center_y = center_x = 0.5
    
    # Calculate attention dispersion (std of coordinates)
    if attention_mask.sum() > 1:
        dispersion_y = y_coords.std() / heatmap.shape[0]
        dispersion_x = x_coords.std() / heatmap.shape[1]
        dispersion = np.sqrt(dispersion_y**2 + dispersion_x**2)
    else:
        dispersion = 0
    
    # Calculate edge attention (attention near borders)
    # Using 5% instead of 10% for more strict edge definition
    border_width = int(min(heatmap.shape) * 0.05)
    edge_mask = np.zeros_like(heatmap, dtype=bool)
    edge_mask[:border_width, :] = True
    edge_mask[-border_width:, :] = True
    edge_mask[:, :border_width] = True
    edge_mask[:, -border_width:] = True
    
    edge_attention = (attention_mask & edge_mask).sum() / attention_mask.sum() if attention_mask.sum() > 0 else 0
    
    # Calculate maximum attention value
    max_attention = heatmap.max()
    
    # Calculate attention entropy (measure of dispersion)
    flat_heatmap = heatmap.flatten()
    flat_heatmap = flat_heatmap[flat_heatmap > 0.01]  # Filter very low values
    if len(flat_heatmap) > 0:
        flat_heatmap = flat_heatmap / flat_heatmap.sum()
        entropy = -np.sum(flat_heatmap * np.log(flat_heatmap + 1e-10))
    else:
        entropy = 0
    
    return {
        'attention_ratio': attention_ratio,
        'center_x': center_x,
        'center_y': center_y,
        'dispersion': dispersion,
        'edge_attention': edge_attention,
        'max_attention': max_attention,
        'entropy': entropy
    }


def visualize_attention(img_path, heatmap, true_label, pred_label, confidence, 
                       metrics, output_path, class_names):
    """Create comprehensive visualization with multiple views"""
    
    # Load original image
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Sample: {Path(img_path).stem}\n'
                 f'True: {class_names[true_label]} | Pred: {class_names[pred_label]} | Conf: {confidence:.3f}',
                 fontsize=12, fontweight='bold')
    
    # 1. Original image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Heatmap only
    axes[0, 1].imshow(heatmap, cmap='jet')
    axes[0, 1].set_title('Attention Heatmap')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
    
    # 3. Overlay
    overlay, _ = overlay_heatmap(img, heatmap, alpha=0.5)
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Overlay (50% alpha)')
    axes[0, 2].axis('off')
    
    # 4. Thresholded attention (top 30%)
    threshold_30 = np.percentile(heatmap, 70)
    mask_30 = heatmap > threshold_30
    axes[1, 0].imshow(img_array)
    axes[1, 0].imshow(mask_30, cmap='Reds', alpha=0.5)
    axes[1, 0].set_title(f'Top 30% Attention Regions')
    axes[1, 0].axis('off')
    
    # 5. Attention center and dispersion
    axes[1, 1].imshow(img_array)
    axes[1, 1].plot(metrics['center_x'] * img_array.shape[1], 
                    metrics['center_y'] * img_array.shape[0], 
                    'r+', markersize=20, markeredgewidth=3)
    circle = plt.Circle((metrics['center_x'] * img_array.shape[1],
                         metrics['center_y'] * img_array.shape[0]),
                        metrics['dispersion'] * img_array.shape[1],
                        color='red', fill=False, linewidth=2)
    axes[1, 1].add_patch(circle)
    axes[1, 1].set_title('Attention Center & Dispersion')
    axes[1, 1].axis('off')
    
    # 6. Metrics summary
    axes[1, 2].axis('off')
    metrics_text = (
        f"Attention Metrics:\n\n"
        f"Attention Ratio: {metrics['attention_ratio']:.3f}\n"
        f"Center X: {metrics['center_x']:.3f}\n"
        f"Center Y: {metrics['center_y']:.3f}\n"
        f"Dispersion: {metrics['dispersion']:.3f}\n"
        f"Edge Attention: {metrics['edge_attention']:.3f}\n"
        f"Max Attention: {metrics['max_attention']:.3f}\n"
        f"Entropy: {metrics['entropy']:.3f}\n\n"
        f"Interpretation:\n"
        f"- Edge Attention > 0.3 suggests\n  background focus\n"
        f"- Dispersion > 0.3 suggests\n  scattered attention\n"
        f"- Center near 0.5 suggests\n  centered object focus"
    )
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze model attention with Grad-CAM')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--error_csv', type=str, required=True,
                       help='Path to error cases CSV file (from fault_analysis.py)')
    parser.add_argument('--config_path', type=str, 
                       default='configs/data/attributes/0_CUB_200_2011.yaml',
                       help='Path to dataset config with class names')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of error samples to analyze')
    parser.add_argument('--target_classes', nargs='+', default=None,
                       help='Specific classes to analyze (e.g., "Fish_Crow" "American_Crow")')
    parser.add_argument('--data_root', type=str, 
                       default='/data/lvta/datasets/vl2lite_datasets/0_CUB_200_2011',
                       help='Root directory of dataset (for finding original images)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    config = OmegaConf.load(args.config_path)
    class_names_raw = config.classes
    
    if isinstance(class_names_raw, dict):
        class_names = [class_names_raw[i] for i in range(1, len(class_names_raw) + 1)]
    else:
        class_names = list(class_names_raw)
    
    print(f"Loaded {len(class_names)} class names")
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}")
    model = KDModule.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    model = model.cuda()
    
    # Get target layer (last conv layer of ResNet18)
    # Navigate through TeacherStudent -> StudentNet -> ModifiedResNet -> resnet -> layer4
    if hasattr(model.net, 'student'):
        # Distilled model with teacher
        resnet = model.net.student.model.resnet
    elif hasattr(model.net, 'module'):
        # DDP wrapped
        resnet = model.net.module
    else:
        # Baseline model
        resnet = model.net
    
    target_layer = resnet.layer4[-1].conv2
    print(f"Target layer: {target_layer}")
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model.net, target_layer)
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load error cases from CSV
    print(f"Loading error cases from {args.error_csv}")
    error_df = pd.read_csv(args.error_csv)
    print(f"Loaded {len(error_df)} error cases")
    
    # Load image paths mapping from images.txt
    images_txt_path = Path(args.data_root) / 'images.txt'
    image_paths = {}
    with open(images_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                img_id = int(parts[0]) - 1  # Convert to 0-indexed
                img_path = parts[1]
                image_paths[img_id] = img_path
    print(f"Loaded {len(image_paths)} image paths")
    
    # Filter by target classes if specified
    if args.target_classes:
        print(f"Filtering for classes: {args.target_classes}")
        # Convert underscores to spaces for matching
        target_classes_with_spaces = [cls.replace('_', ' ') for cls in args.target_classes]
        mask = error_df['true_class_name'].isin(target_classes_with_spaces)
        error_df = error_df[mask]
        print(f"Found {len(error_df)} samples matching target classes")
    
    # Sample
    if len(error_df) > args.num_samples:
        error_df = error_df.sample(n=args.num_samples, random_state=42)
    
    print(f"Analyzing {len(error_df)} error cases")
    
    # Collect metrics for analysis
    all_metrics = []
    
    # Process each sample
    for idx, row in tqdm(error_df.iterrows(), total=len(error_df)):
        sample_id = row['sample_id']
        true_label = row['true_class_id']
        pred_label = row['pred_class_id']
        
        # Get image path from mapping
        if sample_id not in image_paths:
            print(f"Warning: Image path not found for sample {sample_id}")
            continue
        
        rel_path = image_paths[sample_id]
        full_image_path = str(Path(args.data_root) / 'images' / rel_path)
        
        # Load and transform image
        img = Image.open(full_image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()
        img_tensor.requires_grad = True
        
        # Generate Grad-CAM (no torch.no_grad here, we need gradients!)
        heatmap, pred_class = grad_cam(img_tensor, class_idx=pred_label)
        
        # Calculate metrics
        metrics = calculate_attention_metrics(heatmap, threshold=0.5)
        metrics['sample_id'] = sample_id
        metrics['true_label'] = true_label
        metrics['pred_label'] = pred_label
        metrics['true_class_name'] = class_names[true_label]
        metrics['pred_class_name'] = class_names[pred_label]
        
        # Get confidence
        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, tuple):
                if len(output) >= 2:
                    logits = output[1]
                else:
                    logits = output[0]
            else:
                logits = output
            probs = F.softmax(logits, dim=1)
            confidence = probs[0, pred_label].item()
        
        metrics['confidence'] = confidence
        all_metrics.append(metrics)
        
        # Visualize
        output_path = output_dir / f"{sample_id:05d}_{class_names[true_label]}_to_{class_names[pred_label]}_gradcam.jpg"
        visualize_attention(full_image_path, heatmap, true_label, pred_label, 
                          confidence, metrics, output_path, class_names)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = output_dir / 'attention_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved attention metrics to {metrics_csv_path}")
    
    # Generate summary statistics only if we have data
    if len(metrics_df) == 0:
        print("\nNo samples analyzed. Please check:")
        print("  1. Target class names match CSV (use spaces, not underscores)")
        print("  2. Error CSV file contains the expected classes")
        return
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*60)
    
    summary_stats = metrics_df[['attention_ratio', 'dispersion', 'edge_attention', 
                                 'max_attention', 'entropy']].describe()
    print("\nOverall Statistics:")
    print(summary_stats)
    
    # Identify background-dominated cases
    background_dominated = metrics_df[
        (metrics_df['edge_attention'] > 0.3) | 
        (metrics_df['dispersion'] > 0.3)
    ]
    
    print(f"\nPotential Background-Dominated Cases: {len(background_dominated)} / {len(metrics_df)} "
          f"({len(background_dominated)/len(metrics_df)*100:.1f}%)")
    
    if len(background_dominated) > 0:
        print("\nTop 10 Most Background-Dominated:")
        bg_sorted = background_dominated.sort_values('edge_attention', ascending=False)
        print(bg_sorted[['sample_id', 'true_class_name', 'pred_class_name', 
                         'edge_attention', 'dispersion']].head(10).to_string(index=False))
    
    # Focused attention cases
    focused = metrics_df[
        (metrics_df['edge_attention'] < 0.2) & 
        (metrics_df['dispersion'] < 0.2)
    ]
    
    print(f"\nFocused Attention Cases: {len(focused)} / {len(metrics_df)} "
          f"({len(focused)/len(metrics_df)*100:.1f}%)")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_csv_path}")


if __name__ == '__main__':
    main()
