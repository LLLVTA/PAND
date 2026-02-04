#!/usr/bin/env python3
"""
将 cub_feature.json 转换为 class_prompts 格式并更新配置文件
"""
import json
import yaml
from pathlib import Path

def main():
    # 文件路径
    feature_file = Path('configs/data/attributes/cub_feature.json')
    config_file = Path('configs/data/attributes/0_CUB_200_2011.yaml')
    
    print(f"📖 Loading {feature_file}")
    with open(feature_file, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
    
    print(f"📖 Loading {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("🔨 Converting to class_prompts format...")
    class_prompts = {}
    matched = 0
    
    for class_id, class_name in config['classes'].items():
        if class_name in features_data:
            class_prompts[class_id] = features_data[class_name]
            matched += 1
        else:
            print(f"⚠️  No prompts for: {class_name} (using defaults)")
            class_prompts[class_id] = [
                f"a photo of {class_name}, a type of bird",
                f"a picture of {class_name}",
                f"{class_name} bird"
            ]
    
    # 删除旧的 prompt_templates，添加新的 class_prompts
    if 'prompt_templates' in config:
        del config['prompt_templates']
    config['class_prompts'] = class_prompts
    
    print(f"💾 Saving updated config to {config_file}")
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total classes: {len(config['classes'])}")
    print(f"   Matched with features: {matched}")
    print(f"   Using default prompts: {len(config['classes']) - matched}")
    
    # 显示示例
    print(f"\n📝 Example (Class 1):")
    for i, prompt in enumerate(class_prompts[1], 1):
        print(f"   {i}. {prompt}")

if __name__ == "__main__":
    main()
