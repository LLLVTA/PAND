"""
Quick test for RKD loss on real data with minimal GPU memory usage.
This script uses a tiny batch to diagnose RKD behavior.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.components.criterion import KDCriterion
from src.data.kd_datamodule import KDDataModule
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def quick_rkd_test():
    """Quick RKD test with minimal memory footprint."""
    print("\n" + "="*70)
    print("  QUICK RKD TEST - Minimal GPU Memory Usage")
    print("="*70)
    
    # Initialize Hydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize(version_base="1.3", config_path="../configs")
    cfg = compose(config_name="train.yaml", overrides=[
        "model=kda",
        "data=kd_data",
        "data/attributes=0_CUB_200_2011",
        "trainer.devices=1",
        "data.batch_size=8",  # Very small batch
        "data.num_workers=0",  # Avoid multiprocessing issues
    ])
    
    print("\n[Step 1] Loading data with batch_size=8...")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    
    # Get one tiny batch
    batch = next(iter(train_loader))
    images, labels = batch
    # Use only 4 samples to minimize memory
    images = images[:4]
    labels = labels[:4]
    
    print(f"✓ Using {images.shape[0]} samples: {images.shape}")
    
    print("\n[Step 2] Loading model...")
    net = hydra.utils.instantiate(cfg.model.net)
    criterion = hydra.utils.instantiate(cfg.model.kd_criterion)
    
    print(f"✓ Model: {type(net).__name__}")
    print(f"✓ Visual KD: {type(criterion.criterion_aligned_img_kd).__name__}")
    
    # Use GPU 3 (should have ~1.3GB free)
    device = torch.device("cuda:3")
    print(f"✓ Using device: {device}")
    
    try:
        net = net.to(device)
        images = images.to(device)
        labels = labels.to(device)
        print("✓ Model loaded to GPU successfully")
    except RuntimeError as e:
        print(f"✗ GPU memory error: {e}")
        print("  Trying CPU instead...")
        device = torch.device("cpu")
        net = net.to(device)
        images = images.to(device)
        labels = labels.to(device)
    
    print("\n[Step 3] Forward pass...")
    net.eval()
    with torch.no_grad():
        outputs = net(images)
        
        # Unpack outputs
        student_img_feat = outputs[0]
        student_logits = outputs[1]
        student_txt_feat = outputs[2]
        teacher_img_feat = outputs[3]
        teacher_txt_feat = outputs[4]
        aligned_student_img = outputs[5]
        aligned_teacher_img = outputs[6]
        aligned_student_txt = outputs[7]
        aligned_teacher_txt = outputs[8]
        
        print(f"✓ Student image feat: {student_img_feat.shape}")
        print(f"✓ Teacher image feat: {teacher_img_feat.shape}")
        print(f"✓ Aligned student: {aligned_student_img.shape}")
        print(f"✓ Aligned teacher: {aligned_teacher_img.shape}")
    
    print("\n[Step 4] Analyzing RKD Loss (L_vis)...")
    
    # Feature statistics
    with torch.no_grad():
        s_norm = torch.norm(aligned_student_img, dim=-1).mean().item()
        t_norm = torch.norm(aligned_teacher_img, dim=-1).mean().item()
        
        print(f"\nFeature L2 Norms:")
        print(f"  Student: {s_norm:.4f}")
        print(f"  Teacher: {t_norm:.4f}")
        print(f"  Ratio: {s_norm/t_norm:.4f}")
        
        # Compute distances
        def compute_dist(feats):
            feats_flat = feats.view(feats.shape[0], -1)
            pred_square = feats_flat.pow(2).sum(dim=-1)
            prod = torch.mm(feats_flat, feats_flat.t())
            distance = (pred_square.unsqueeze(1) + pred_square.unsqueeze(0) - 2*prod).clamp(min=1e-12)
            distance = distance.sqrt()
            return distance
        
        d_s = compute_dist(aligned_student_img)
        d_t = compute_dist(aligned_teacher_img)
        
        # Remove diagonal
        mask = ~torch.eye(d_s.shape[0], dtype=bool, device=d_s.device)
        
        print(f"\nPairwise Distances:")
        print(f"  Student mean: {d_s[mask].mean().item():.4f}")
        print(f"  Teacher mean: {d_t[mask].mean().item():.4f}")
        
        # Normalized distances
        d_s_norm = d_s / (d_s[mask].mean() + 1e-8)
        d_t_norm = d_t / (d_t[mask].mean() + 1e-8)
        
        print(f"\nNormalized Distances:")
        print(f"  Student mean: {d_s_norm[mask].mean().item():.4f}")
        print(f"  Teacher mean: {d_t_norm[mask].mean().item():.4f}")
    
    # Compute RKD loss with gradients
    print("\n[Step 5] Computing RKD Loss...")
    net.train()
    outputs_train = net(images)
    aligned_student_train = outputs_train[5]
    aligned_teacher_train = outputs_train[6]
    
    aligned_student_train.requires_grad_(True)
    
    rkd_loss = criterion.criterion_aligned_img_kd(aligned_student_train, aligned_teacher_train)
    
    print(f"\n{'='*60}")
    print(f"RKD Loss (L_vis): {rkd_loss.item():.6f}")
    print(f"{'='*60}")
    
    # Gradient check
    rkd_loss.backward()
    grad_norm = torch.norm(aligned_student_train.grad).item()
    grad_mean = aligned_student_train.grad.mean().item()
    
    print(f"\nGradient Statistics:")
    print(f"  Norm: {grad_norm:.6f}")
    print(f"  Mean: {grad_mean:.6f}")
    
    # Compare with other losses
    with torch.no_grad():
        cls_loss = torch.nn.functional.cross_entropy(outputs_train[1], labels)
        kd_loss = criterion.criterion_kd(outputs_train)
        
        print(f"\n{'='*60}")
        print("All Loss Components:")
        print(f"{'='*60}")
        print(f"  Classification (L_cls): {cls_loss.item():.6f}")
        print(f"  Visual KD (L_vis/RKD): {rkd_loss.item():.6f}")
        print(f"  Text KD (L_txt):       {kd_loss.item():.6f}")
        
        # Simulate mid-training weights
        epoch_ratio = 0.5
        cls_weight = epoch_ratio / 8  # ~0.0625
        kd_weight = 1 - cls_weight    # ~0.9375
        
        total_loss = cls_weight * cls_loss + kd_weight * (rkd_loss.item() + kd_loss) / 2
        
        print(f"\nWeighted Loss (epoch={int(150*epoch_ratio)}/300):")
        print(f"  cls_weight={cls_weight:.4f}, kd_weight={kd_weight:.4f}")
        print(f"  Total: {total_loss.item():.6f}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("="*70)
    
    # Diagnosis
    if rkd_loss.item() > 10.0:
        print("\n⚠️  RKD loss is HIGH (>10)")
        print("   Issues:")
        print("   - Feature scale mismatch between student and teacher")
        print("   - Condensation layer may need better initialization")
        print("\n   Solutions:")
        print("   1. Add feature normalization: set with_l2_norm=True")
        print("   2. Increase warmup epochs for condensation layer")
        print("   3. Lower learning rate for condensation layer")
        
    elif rkd_loss.item() < 0.001:
        print("\n⚠️  RKD loss is VERY LOW (<0.001)")
        print("   Issues:")
        print("   - Features may have collapsed")
        print("   - Student copying teacher too closely")
        print("\n   Solutions:")
        print("   1. Increase λ_vis weight")
        print("   2. Check for gradient vanishing")
        print("   3. Verify teacher features are diverse")
        
    elif 0.1 < rkd_loss.item() < 5.0:
        print("\n✓ RKD loss is in REASONABLE range (0.1-5.0)")
        print("   This is a healthy loss value for RKD.")
        
    else:
        print(f"\n→ RKD loss = {rkd_loss.item():.4f}")
        print("   Loss value is acceptable, monitor during training.")
    
    if grad_norm < 1e-6:
        print("\n⚠️  Gradient VANISHING")
        print("   Solutions:")
        print("   1. Increase loss weight for L_vis")
        print("   2. Check condensation layer learning rate")
        
    elif grad_norm > 100:
        print("\n⚠️  Gradient EXPLODING")
        print("   Solutions:")
        print("   1. Enable gradient clipping")
        print("   2. Lower learning rate")
        print("   3. Check distance normalization")
        
    else:
        print(f"\n✓ Gradient norm is healthy: {grad_norm:.4f}")
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR TRAINING")
    print("="*70)
    print("""
1. Monitor these metrics during training:
   - train/img_loss should decrease gradually
   - train/cls_loss should increase (due to weight scheduling)
   - train/acc should improve steadily

2. Check logs every 10 epochs:
   - If img_loss stays >5: feature alignment issue
   - If img_loss <0.01: may need higher weight
   - If acc not improving: balance cls_weight and kd_weight

3. For 4-GPU training with limited memory:
   - Use batch_size=4 per GPU (total=16)
   - Enable gradient_accumulation_steps=2 (effective batch=32)
   - Monitor GPU memory with nvidia-smi
    """)
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)


if __name__ == "__main__":
    quick_rkd_test()
