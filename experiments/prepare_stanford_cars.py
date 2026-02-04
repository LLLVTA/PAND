#!/usr/bin/env python3
"""
准备 Stanford Cars 数据集，使其符合代码预期的格式
"""
import os
import pandas as pd
from pathlib import Path
import shutil

def prepare_stanford_cars(data_root):
    """
    准备 Stanford Cars 数据集
    
    输入格式:
    - anno_train.csv: filename,x1,y1,x2,y2,class_id (6列)
    - train/类别名/图片.jpg
    - test/类别名/图片.jpg
    
    输出格式:
    - train.csv: label,filename (2列，label从0开始)
    - test.csv: label,filename (2列，label从0开始)
    - car_ims/图片.jpg (所有图片在一个目录)
    """
    data_root = Path(data_root)
    
    # 创建 car_ims 目录
    car_ims_dir = data_root / "car_ims"
    car_ims_dir.mkdir(exist_ok=True)
    
    print(f"处理 Stanford Cars 数据集: {data_root}")
    
    # 处理 train 和 test
    for split in ['train', 'test']:
        print(f"\n处理 {split} 数据...")
        
        # 读取原始 CSV (格式: filename,x1,y1,x2,y2,class_id)
        anno_csv = data_root / f"anno_{split}.csv"
        df = pd.read_csv(anno_csv, header=None, names=['filename', 'x1', 'y1', 'x2', 'y2', 'class_id'])
        
        # 转换为目标格式 (label,filename)，class_id 从1开始，需要转为0开始
        new_df = pd.DataFrame({
            'label': df['class_id'] - 1,  # 转为 0-indexed
            'filename': df['filename']
        })
        
        # 保存新的 CSV
        output_csv = data_root / f"{split}.csv"
        new_df.to_csv(output_csv, index=False, header=False)
        print(f"  已生成 {output_csv}")
        print(f"  共 {len(new_df)} 条记录")
        
        # 链接图片文件到 car_ims 目录
        split_dir = data_root / split
        img_count = 0
        
        if split_dir.exists():
            # 遍历所有类别子目录
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    # 遍历该类别下的所有图片
                    for img_file in class_dir.glob("*.jpg"):
                        target_link = car_ims_dir / img_file.name
                        
                        # 如果已存在同名文件/链接，跳过
                        if target_link.exists():
                            continue
                        
                        # 创建符号链接
                        try:
                            target_link.symlink_to(img_file.absolute())
                            img_count += 1
                        except Exception as e:
                            print(f"  警告: 无法创建链接 {img_file.name}: {e}")
        
        print(f"  已链接 {img_count} 张图片到 car_ims/")
    
    # 检查结果
    print(f"\n完成！检查结果:")
    print(f"  train.csv: {(data_root / 'train.csv').exists()}")
    print(f"  test.csv: {(data_root / 'test.csv').exists()}")
    print(f"  car_ims/: {len(list(car_ims_dir.glob('*.jpg')))} 张图片")

if __name__ == "__main__":
    data_root = "/data/lvta/datasets/vl2lite_datasets/6_StanfordCars"
    prepare_stanford_cars(data_root)
