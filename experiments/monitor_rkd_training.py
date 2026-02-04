"""
Monitor RKD (L_vis) behavior during real training.
This script helps diagnose and optimize RKD loss in actual training scenarios.
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.components.criterion import KDCriterion
from src.data.kd_datamodule import KDDataModule
import hydra
from omegaconf import DictConfig


def analyze_feature_distributions(student_feats, teacher_feats):
    """Analyze feature distribution statistics."""
    print("\n" + "="*60)
    print("Feature Distribution Analysis")
    print("="*60)
    
    with torch.no_grad():
        # Student features
        s_mean = student_feats.mean().item()
        s_std = student_feats.std().item()
        s_min = student_feats.min().item()
        s_max = student_feats.max().item()
        s_norm = torch.norm(student_feats, dim=-1).mean().item()
        
        # Teacher features
        t_mean = teacher_feats.mean().item()
        t_std = teacher_feats.std().item()
        t_min = teacher_feats.min().item()
        t_max = teacher_feats.max().item()
        t_norm = torch.norm(teacher_feats, dim=-1).mean().item()
        
        print(f"\nStudent Features (shape: {student_feats.shape}):")
        print(f"  Mean: {s_mean:.4f}, Std: {s_std:.4f}")
        print(f"  Range: [{s_min:.4f}, {s_max:.4f}]")
        print(f"  L2 Norm (avg): {s_norm:.4f}")
        
        print(f"\nTeacher Features (shape: {teacher_feats.shape}):")
        print(f"  Mean: {t_mean:.4f}, Std: {t_std:.4f}")
        print(f"  Range: [{t_min:.4f}, {t_max:.4f}]")
        print(f"  L2 Norm (avg): {t_norm:.4f}")
        
        # Feature scale ratio
        scale_ratio = s_norm / (t_norm + 1e-8)
        print(f"\nFeature Scale Ratio (Student/Teacher): {scale_ratio:.4f}")
        
        if scale_ratio < 0.1 or scale_ratio > 10:
            print("⚠️  WARNING: Large feature scale mismatch detected!")
            print("   Consider: 1) Check condensation layer initialization")
            print("            2) Add feature normalization before RKD")


def analyze_distance_matrices(student_feats, teacher_feats):
    """Analyze pairwise distance matrix properties."""
    print("\n" + "="*60)
    print("Distance Matrix Analysis")
    print("="*60)
    
    with torch.no_grad():
        # Compute pairwise distances
        def compute_distances(feats):
            feats_flat = feats.view(feats.shape[0], -1)
            pred_square = feats_flat.pow(2).sum(dim=-1)
            prod = torch.mm(feats_flat, feats_flat.t())
            distance = (pred_square.unsqueeze(1) + pred_square.unsqueeze(0) - 2*prod).clamp(min=1e-12)
            distance = distance.sqrt()
            distance[range(len(feats_flat)), range(len(feats_flat))] = 0
            return distance
        
        d_student = compute_distances(student_feats)
        d_teacher = compute_distances(teacher_feats)
        
        # Mask out diagonal
        mask = ~torch.eye(d_student.shape[0], dtype=bool, device=d_student.device)
        
        # Statistics
        d_s_mean = d_student[mask].mean().item()
        d_s_std = d_student[mask].std().item()
        d_t_mean = d_teacher[mask].mean().item()
        d_t_std = d_teacher[mask].std().item()
        
        print(f"\nStudent Distance Matrix:")
        print(f"  Mean: {d_s_mean:.4f}, Std: {d_s_std:.4f}")
        print(f"  Min: {d_student[mask].min().item():.4f}, Max: {d_student[mask].max().item():.4f}")
        
        print(f"\nTeacher Distance Matrix:")
        print(f"  Mean: {d_t_mean:.4f}, Std: {d_t_std:.4f}")
        print(f"  Min: {d_teacher[mask].min().item():.4f}, Max: {d_teacher[mask].max().item():.4f}")
        
        # Distance scale ratio
        dist_ratio = d_s_mean / (d_t_mean + 1e-8)
        print(f"\nDistance Scale Ratio (Student/Teacher): {dist_ratio:.4f}")
        
        # Normalize distances
        d_s_norm = d_student / (d_student[mask].mean() + 1e-8)
        d_t_norm = d_teacher / (d_teacher[mask].mean() + 1e-8)
        
        print(f"\nAfter Normalization:")
        print(f"  Student mean: {d_s_norm[mask].mean().item():.4f}")
        print(f"  Teacher mean: {d_t_norm[mask].mean().item():.4f}")
        
        # Distance correlation
        d_s_flat = d_student[mask].flatten()
        d_t_flat = d_teacher[mask].flatten()
        correlation = torch.corrcoef(torch.stack([d_s_flat, d_t_flat]))[0, 1].item()
        print(f"\nDistance Correlation: {correlation:.4f}")
        
        if abs(correlation) < 0.3:
            print("⚠️  WARNING: Low distance correlation!")
            print("   Student and teacher distance structures are very different.")
            print("   This might indicate:")
            print("   1) Student features not learning properly")
            print("   2) Condensation layer needs better initialization")
            print("   3) Need more training epochs")


def analyze_rkd_loss_components(criterion, student_feats, teacher_feats, aligned_student, aligned_teacher):
    """Analyze RKD loss and its components."""
    print("\n" + "="*60)
    print("RKD Loss Component Analysis")
    print("="*60)
    
    with torch.no_grad():
        # Original features (before condensation)
        print(f"\nOriginal feature dimensions:")
        print(f"  Student: {student_feats.shape}")
        print(f"  Teacher: {teacher_feats.shape}")
        
        # Aligned features (after condensation)
        print(f"\nAligned feature dimensions:")
        print(f"  Student: {aligned_student.shape}")
        print(f"  Teacher: {aligned_teacher.shape}")
        
    # Compute RKD loss with gradient tracking
    # Create a copy with requires_grad to capture gradients
    aligned_student_copy = aligned_student.clone().detach().requires_grad_(True)
    rkd_loss = criterion.criterion_aligned_img_kd(aligned_student_copy, aligned_teacher)
    
    print(f"\nRKD Loss (L_vis): {rkd_loss.item():.6f}")
    
    # Check if loss is in reasonable range
    if rkd_loss.item() > 10.0:
        print("⚠️  WARNING: RKD loss is very high!")
        print("   Possible issues:")
        print("   1) Feature scale mismatch")
        print("   2) Condensation layer not initialized properly")
        print("   3) Distance normalization not working")
    elif rkd_loss.item() < 0.001:
        print("⚠️  WARNING: RKD loss is very low!")
        print("   Possible issues:")
        print("   1) Features collapsed to same values")
        print("   2) Gradient vanishing")
        print("   3) Student already converged (if late in training)")
    else:
        print("✓ RKD loss is in reasonable range")
    
    # Check gradient flow
    rkd_loss.backward()
    if aligned_student_copy.grad is not None:
        grad_norm = torch.norm(aligned_student_copy.grad).item()
        grad_mean = aligned_student_copy.grad.mean().item()
        grad_std = aligned_student_copy.grad.std().item()
        
        print(f"\nGradient Analysis:")
        print(f"  Norm: {grad_norm:.6f}")
        print(f"  Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")
        
        if grad_norm < 1e-6:
            print("⚠️  WARNING: Gradient is too small (vanishing)!")
        elif grad_norm > 100:
            print("⚠️  WARNING: Gradient is too large (exploding)!")
        else:
            print("✓ Gradient magnitude is healthy")
    else:
        print("⚠️  WARNING: No gradient computed!")
    
    return rkd_loss.item()


def test_with_real_data():
    """Test RKD with real training data."""
    print("\n" + "="*70)
    print("  REAL TRAINING DATA TEST - RKD Loss (L_vis) Analysis")
    print("="*70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("\n⚠️  GPU not available, using CPU")
    
    # Load configuration
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize(version_base="1.3", config_path="../configs")
    cfg = compose(config_name="train.yaml", overrides=[
        "model=kda",
        "data=kd_data",
        "data/attributes=0_CUB_200_2011",
        "trainer.devices=1",
        "data.batch_size=16",  # Smaller batch for single GPU test
    ])
    
    print("\n[1/5] Loading data...")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    
    # Get one batch
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    
    print(f"✓ Loaded batch: {images.shape}, Labels: {labels.shape}")
    
    print("\n[2/5] Loading model...")
    # Create model
    net = hydra.utils.instantiate(cfg.model.net)
    criterion = hydra.utils.instantiate(cfg.model.kd_criterion)
    
    print(f"✓ Model loaded: {type(net).__name__}")
    print(f"✓ Criterion: {type(criterion).__name__}")
    print(f"✓ Visual distillation: {type(criterion.criterion_aligned_img_kd).__name__}")
    
    # Move to GPU
    net = net.to(device)
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"✓ Model and data moved to: {device}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  GPU memory allocated: {mem_allocated:.2f} GB")
    
    print("\n[3/5] Forward pass...")
    with torch.no_grad():
        net.eval()
        outputs = net(images)
        
        # outputs format from campus.py forward():
        # (hidden_features, out, clip_img_features, frozen_nlp_features, aligned_img, aligned_nlp)
        student_img_feat = outputs[0]     # [16, 512] - 学生图像特征
        student_logits = outputs[1]       # [16, 200] - 学生分类输出
        teacher_img_feat = outputs[2]     # [16, 1024] - 教师图像特征
        frozen_nlp_feat = outputs[3]      # [200, 1024] - CLIP文本特征
        aligned_img = outputs[4]          # [16, 512] - 降维后的图像特征（学生+教师都在这）
        aligned_nlp = outputs[5]          # [200, 512] - 降维后的文本特征
        
        print(f"✓ Student image features: {student_img_feat.shape}")
        print(f"✓ Teacher image features: {teacher_img_feat.shape}")
        print(f"✓ Aligned image features: {aligned_img.shape}")
        print(f"✓ Aligned text features: {aligned_nlp.shape}")
    
    print("\n[4/5] Analyzing features and distances...")
    # RKD compares: student_img_feat vs aligned_img (teacher after condensation)
    analyze_feature_distributions(student_img_feat, aligned_img)
    analyze_distance_matrices(student_img_feat, aligned_img)
    
    print("\n[5/5] Computing RKD loss...")
    net.train()
    outputs_train = net(images)
    student_img_feat_train = outputs_train[0]  # 学生特征
    aligned_img_train = outputs_train[4]       # 教师降维后的特征
    
    rkd_loss = analyze_rkd_loss_components(
        criterion, 
        student_img_feat_train,  # 学生原始特征
        outputs_train[2],        # 教师原始特征
        student_img_feat_train,  # 学生特征（用于RKD）
        aligned_img_train        # 教师降维特征（用于RKD）
    )
    
    # Compute other losses for comparison
    with torch.no_grad():
        cls_loss = torch.nn.functional.cross_entropy(outputs_train[1], labels)
        
        # Prepare inputs for criterion (matches campus.py forward output)
        hidden_features = outputs_train[0]     # 学生特征
        out = outputs_train[1]                 # 学生logits
        clip_img_features = outputs_train[2]   # 教师图像特征
        clip_nlp_features = outputs_train[3]   # CLIP文本特征
        aligned_img = outputs_train[4]         # 降维后的图像特征
        aligned_nlp = outputs_train[5]         # 降维后的文本特征
        
        inputs = (hidden_features, out, clip_img_features, clip_nlp_features, aligned_img, aligned_nlp)
        img_loss, kd_loss = criterion(inputs)
        
        print(f"\n" + "="*60)
        print("Loss Comparison")
        print("="*60)
        print(f"  Classification Loss (L_cls): {cls_loss.item():.6f}")
        print(f"  Visual KD Loss (L_vis/RKD):  {img_loss.item():.6f}")
        print(f"  Text KD Loss (L_txt):        {kd_loss.item():.6f}")
        
        # Weight ratio (assuming mid-training)
        epoch_ratio = 0.5  # Simulate mid-training
        cls_weight = epoch_ratio / 8
        kd_weight = 1 - cls_weight
        
        total_loss = cls_weight * cls_loss + kd_weight * (img_loss + kd_loss) / 2
        print(f"\nWeighted Total Loss (mid-training simulation):")
        print(f"  cls_weight={cls_weight:.4f}, kd_weight={kd_weight:.4f}")
        print(f"  Total: {total_loss.item():.6f}")
    
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    # Provide recommendations based on analysis
    recommendations = []
    
    if img_loss.item() > 5.0:
        recommendations.append(
            "1. RKD loss is high - Consider:\n"
            "   - Increase warmup epochs for condensation layer\n"
            "   - Add learning rate warmup\n"
            "   - Check if teacher features are frozen correctly"
        )
    
    if img_loss.item() < 0.01:
        recommendations.append(
            "2. RKD loss is very low - Consider:\n"
            "   - Increase λ_vis weight\n"
            "   - Check for feature collapse\n"
            "   - Verify student is not just copying teacher"
        )
    
    recommendations.append(
        "3. Monitor during training:\n"
        "   - Watch train/img_loss in logs (should decrease steadily)\n"
        "   - Check distance correlation (should increase over time)\n"
        "   - Verify gradients don't vanish or explode"
    )
    
    recommendations.append(
        "4. Potential improvements:\n"
        "   - Try with_l2_norm=True for feature normalization\n"
        "   - Experiment with different distance normalization\n"
        "   - Consider adding angle-wise RKD component"
    )
    
    for rec in recommendations:
        print(f"\n{rec}")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        del net, images, labels, outputs, outputs_train
        torch.cuda.empty_cache()
        print(f"\n✓ GPU memory cleaned")
    
    print("\n" + "="*70)
    print("Test completed! Use this information to optimize training.")
    print("="*70)


if __name__ == "__main__":
    test_with_real_data()