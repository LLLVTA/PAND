#!/usr/bin/env python
"""
错误案例分析脚本 (Fault Analysis Script)
==========================================

功能：
1. 加载指定的 checkpoint（baseline 或蒸馏后的模型）
2. 在测试集上进行推理，找出所有预测错误的样本
3. 保存错误样本的图片到分类目录（按 真实类别_to_预测类别 组织）
4. 生成详细的错误分析报告（JSON + TXT）
5. 统计错误类型分布和常见混淆对

使用方法:
    # 分析使用教师模型的distilled模型错误 (保存图片)
    python experiments/fault_analysis.py \\
        --ckpt_path /path/to/distilled_checkpoint.ckpt \\
        --dataset 1_FGVC_AIRCRAFT \\
        --model_type distilled \\
        --save_images \\
        --output_dir /data/lvta/fault_analysis
    
    # 分析baseline模型错误 (不保存图片，仅生成报告)
    python experiments/fault_analysis.py \\
        --ckpt_path /path/to/baseline_checkpoint.ckpt \\
        --dataset 0_CUB_200_2011 \\
        --model_type baseline \\
        --output_dir /data/lvta/fault_analysis

参数说明:
    --ckpt_path: 模型checkpoint文件路径 (必需)
    --dataset: 数据集名称，如 0_CUB_200_2011, 1_FGVC_AIRCRAFT 等 (默认: 0_CUB_200_2011)
    --model_type: 模型类型 - baseline (无教师) 或 distilled (使用教师) (默认: distilled)
    --save_images: 是否保存错误图片 (默认: False，添加此参数则保存)
    --output_dir: 输出目录 (默认: /data/lvta/fault_analysis)

输出结构:
    output_dir/
    ├── CUB_200_2011/
    │   ├── distilled/
    │   │   ├── error_images/
    │   │   │   ├── Black_footed_Albatross_to_Laysan_Albatross/
    │   │   │   │   ├── sample_00123_prob_0.856.jpg
    │   │   │   │   └── sample_00456_prob_0.923.jpg
    │   │   │   └── Blue_winged_Warbler_to_Tennessee_Warbler/
    │   │   │       └── ...
    │   │   ├── error_cases_CUB_200_2011_distilled.json  # 详细JSON数据
    │   │   └── error_report_CUB_200_2011_distilled.txt  # 人类可读报告
    │   └── baseline/
    │       └── ... (相同结构)
    └── FGVC_AIRCRAFT/
        └── ...

代码说明:
    - analyze_errors(): 主函数，加载checkpoint并执行错误分析
    - 使用Hydra加载配置，自动恢复数据集和模型设置
    - 支持GPU加速，自动检测CUDA可用性
    - 图片保存使用ImageNet标准反归一化
    - 生成的报告包含Top-5预测、置信度、混淆矩阵等统计信息
"""

import os
import sys

# 将项目根目录添加到Python路径，以便导入src模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
import shutil
from tqdm import tqdm
import json
import numpy as np

def analyze_errors(ckpt_path: str, output_dir: str, dataset_name: str = "0_CUB_200_2011", 
                   model_type: str = "baseline", save_images: bool = True):
    """
    分析模型预测错误的案例
    
    Args:
        ckpt_path: checkpoint 文件路径
        output_dir: 错误案例保存目录（会在此下创建数据集子目录）
        dataset_name: 数据集名称（如 0_CUB_200_2011, 1_FGVC_AIRCRAFT）
        model_type: 模型类型描述（如 "baseline", "with_teacher", "distilled" 等）
        save_images: 是否保存错误图片
    """
    
    # 为当前数据集和模型类型创建专属目录
    output_dir = Path(output_dir) / dataset_name / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建错误图片保存目录（不再创建子文件夹）
    if save_images:
        images_dir = output_dir / "error_images"
        images_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"错误案例分析")
    print(f"{'='*80}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Dataset:     {dataset_name}")
    print(f"Model Type:  {model_type}")
    print(f"Output Dir:  {output_dir}")
    print(f"Save Images: {save_images}")
    print(f"{'='*80}\n")
    
    # Initialize Hydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize(version_base="1.3", config_path="../configs")
    cfg = compose(config_name="eval.yaml", overrides=[
        "data=kd_data",
        f"data/attributes={dataset_name}",
        "model=kda",
        "trainer=gpu",
        "trainer.devices=1",
        "data.batch_size=1",  # 每次处理一张图
        "data.num_workers=4",
    ])
    
    print("Loading model and data...")
    
    # Load datamodule
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    test_dataset = test_loader.dataset
    
    # Load model from checkpoint
    from src.models.kd_module import KDModule
    model = KDModule.load_from_checkpoint(ckpt_path)
    model.eval()
    
    # 检测是否使用 teacher（蒸馏模型）
    use_teacher = model.use_teacher
    model_description = "蒸馏模型 (with teacher)" if use_teacher else "基线模型 (baseline)"
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Model: {model_description}")
    
    # Get class names (convert OmegaConf to list for integer indexing)
    from omegaconf import OmegaConf, ListConfig
    class_names_raw = cfg.data.attributes.classes
    
    # 使用 OmegaConf.to_container 统一转换为 Python 原生类型
    class_names = OmegaConf.to_container(class_names_raw, resolve=True)
    
    # 处理不同的类名存储格式
    if isinstance(class_names, dict):
        # 检查字典的键类型
        first_key = list(class_names.keys())[0]
        if isinstance(first_key, int) or (isinstance(first_key, str) and first_key.isdigit()):
            # 键是数字（可能从1开始），按键排序并构建列表
            sorted_keys = sorted([int(k) for k in class_names.keys()])
            # 如果从1开始，需要调整索引（模型预测是从0开始）
            if sorted_keys[0] == 1:
                class_names = [class_names[i] for i in range(1, len(class_names) + 1)]
            else:
                class_names = [class_names[i] for i in sorted_keys]
        else:
            # 键是字符串，直接转为列表
            class_names = list(class_names.values())
    # 如果是列表，保持不变
    elif not isinstance(class_names, list):
        class_names = list(class_names)
    
    print(f"Total test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    # 分析错误
    error_cases = []
    correct_count = 0
    total_count = 0
    
    print("\nAnalyzing predictions...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="分析中")):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if model.use_teacher:
                outputs = model(images)
                logits = outputs[1]  # student logits
            else:
                logits = model(images)
            
            preds = torch.argmax(logits, dim=1)
            
            # Check if prediction is correct
            is_correct = (preds == labels).item()
            total_count += 1
            
            if is_correct:
                correct_count += 1
            else:
                # 保存错误案例信息
                pred_class = preds.item()
                true_class = labels.item()
                
                # 获取预测概率
                probs = torch.softmax(logits, dim=1)
                pred_prob = probs[0, pred_class].item()
                true_prob = probs[0, true_class].item()
                
                # Top-5 predictions
                top5_probs, top5_indices = torch.topk(probs[0], k=min(5, len(class_names)))
                top5_info = [
                    {
                        "class_id": idx.item(),
                        "class_name": class_names[idx.item()],
                        "probability": prob.item()
                    }
                    for prob, idx in zip(top5_probs, top5_indices)
                ]
                
                # 保存错误图片（直接保存到error_images目录，不创建子文件夹）
                image_filename = None
                if save_images:
                    # 文件名包含：样本ID_真实类别_预测类别_概率
                    true_name_clean = class_names[true_class].replace('/', '_').replace(' ', '_')
                    pred_name_clean = class_names[pred_class].replace('/', '_').replace(' ', '_')
                    image_filename = f"{batch_idx:05d}_{true_name_clean}_to_{pred_name_clean}_prob{pred_prob:.3f}.jpg"
                    image_path = images_dir / image_filename
                    
                    # 将 tensor 转换回 PIL Image 并保存
                    img_tensor = images[0].cpu()
                    # 反归一化（假设使用 ImageNet 标准归一化）
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    
                    # 转换为 PIL Image
                    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                    img_pil.save(image_path)
                
                error_info = {
                    "sample_id": batch_idx,
                    "image_filename": str(image_filename) if image_filename else None,
                    "true_class_id": true_class,
                    "true_class_name": class_names[true_class],
                    "true_probability": true_prob,
                    "pred_class_id": pred_class,
                    "pred_class_name": class_names[pred_class],
                    "pred_probability": pred_prob,
                    "confidence_gap": pred_prob - true_prob,
                    "top5_predictions": top5_info,
                }
                
                error_cases.append(error_info)
    
    # 计算准确率
    accuracy = correct_count / total_count
    error_rate = (total_count - correct_count) / total_count
    
    # 统计错误类型
    true_class_errors = {}
    pred_class_errors = {}
    confusion_pairs = {}
    
    for error in error_cases:
        true_class = error['true_class_name']
        pred_class = error['pred_class_name']
        
        true_class_errors[true_class] = true_class_errors.get(true_class, 0) + 1
        pred_class_errors[pred_class] = pred_class_errors.get(pred_class, 0) + 1
        
        pair = f"{true_class} → {pred_class}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # 保存CSV格式的错误报告
    import csv
    from datetime import datetime
    
    csv_report = output_dir / f"error_cases_{dataset_name.split('_', 1)[1]}_{model_type}.csv"
    with open(csv_report, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'sample_id', 'image_filename', 
            'true_class_id', 'true_class_name', 'true_probability',
            'pred_class_id', 'pred_class_name', 'pred_probability',
            'confidence_gap',
            'top1_class', 'top1_prob',
            'top2_class', 'top2_prob',
            'top3_class', 'top3_prob',
            'top4_class', 'top4_prob',
            'top5_class', 'top5_prob'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for error in error_cases:
            row = {
                'sample_id': error['sample_id'],
                'image_filename': error.get('image_filename', ''),
                'true_class_id': error['true_class_id'],
                'true_class_name': error['true_class_name'],
                'true_probability': f"{error['true_probability']:.4f}",
                'pred_class_id': error['pred_class_id'],
                'pred_class_name': error['pred_class_name'],
                'pred_probability': f"{error['pred_probability']:.4f}",
                'confidence_gap': f"{error['confidence_gap']:.4f}",
            }
            # 添加Top-5预测
            for i, pred in enumerate(error['top5_predictions'][:5], 1):
                row[f'top{i}_class'] = pred['class_name']
                row[f'top{i}_prob'] = f"{pred['probability']:.4f}"
            
            writer.writerow(row)
    
    # 生成简短的TXT摘要报告
    txt_report = output_dir / f"error_summary_{dataset_name.split('_', 1)[1]}_{model_type}.txt"
    with open(txt_report, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("错误案例分析摘要\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检查点路径: {ckpt_path}\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"模型类型: {model_description}\n")
        f.write(f"设备: {device}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("整体性能\n")
        f.write("-" * 80 + "\n")
        f.write(f"总样本数: {total_count}\n")
        f.write(f"正确分类: {correct_count}\n")
        f.write(f"错误分类: {len(error_cases)}\n")
        f.write(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"错误率: {error_rate:.4f} ({error_rate*100:.2f}%)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("被误分类最多的真实类别 (Top 10)\n")
        f.write("-" * 80 + "\n")
        for cls, count in sorted(true_class_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{cls}: {count} 次\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("被错误预测最多的类别 (Top 10)\n")
        f.write("-" * 80 + "\n")
        for cls, count in sorted(pred_class_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{cls}: {count} 次\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("最常见的混淆对 (Top 20)\n")
        f.write("-" * 80 + "\n")
        for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"{pair}: {count} 次\n")
    
    # 打印终端总结
    print(f"\n{'='*80}")
    print(f"分析完成！")
    print(f"{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"模型类型: {model_description}")
    print(f"总样本数: {total_count}")
    print(f"正确分类: {correct_count} ({accuracy*100:.2f}%)")
    print(f"错误分类: {len(error_cases)} ({error_rate*100:.2f}%)")
    print(f"\n结果已保存:")
    print(f"  - CSV:  {csv_report}")
    print(f"  - TXT:  {txt_report}")
    if save_images and len(error_cases) > 0:
        print(f"  - 图片: {images_dir} ({len(error_cases)} 张)")
    print(f"{'='*80}\n")
    
    # 打印Top 5混淆对
    print("最常见的混淆对 (Top 5):")
    for i, (pair, count) in enumerate(sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        print(f"  {i}. {pair}: {count} 次")
    
    GlobalHydra.instance().clear()
    
    print(f"\n{'='*80}")
    print("分析完成!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="错误案例分析 - 保存错误图片和详细报告")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Checkpoint 文件路径")
    parser.add_argument("--output_dir", type=str, 
                       default="/data/lvta/fault_analysis",
                       help="输出目录")
    parser.add_argument("--dataset", type=str, 
                       default="0_CUB_200_2011",
                       help="数据集名称 (例如: 0_CUB_200_2011, 1_FGVC_AIRCRAFT)")
    parser.add_argument("--model_type", type=str,
                       choices=["baseline", "distilled"],
                       default="distilled",
                       help="模型类型: baseline (无教师) 或 distilled (使用教师)")
    parser.add_argument("--save_images", action="store_true",
                       help="是否保存错误图片")
    
    args = parser.parse_args()
    
    analyze_errors(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        model_type=args.model_type,
        save_images=args.save_images
    )
