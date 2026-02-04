# 错误案例分析工具 (Fault Analysis Tool)

## 功能说明

这个工具用于分析训练好的模型在测试集上的分类错误，支持以下功能：

1. **错误检测**: 识别所有被模型误分类的样本
2. **图片保存**: 将错误样本的图片保存到分类目录（按 `真实类别_to_预测类别` 组织）
3. **详细报告**: 生成JSON和TXT两种格式的详细分析报告
4. **统计分析**: 
   - 整体准确率和错误率
   - 被误分类最多的真实类别
   - 被错误预测最多的类别
   - 最常见的混淆对（confusion pairs）
   - 每个错误样本的Top-5预测和置信度

5. **模型对比**: 支持区分baseline模型和使用教师模型的distilled模型

## 使用方法

### 1. 基本用法

```bash
# 分析distilled模型的错误（保存图片）
python experiments/fault_analysis.py \
    --ckpt_path /path/to/distilled_checkpoint.ckpt \
    --dataset 1_FGVC_AIRCRAFT \
    --model_type distilled \
    --save_images \
    --output_dir /data/lvta/fault_analysis

# 分析baseline模型的错误（不保存图片）
python experiments/fault_analysis.py \
    --ckpt_path /path/to/baseline_checkpoint.ckpt \
    --dataset 0_CUB_200_2011 \
    --model_type baseline \
    --output_dir /data/lvta/fault_analysis
```

### 2. 使用Shell脚本

```bash
# 方法1: 分析distilled模型并保存图片
bash experiments/run_fault_analysis.sh \
    /path/to/distilled.ckpt \
    1_FGVC_AIRCRAFT \
    distilled \
    --save_images

# 方法2: 分析baseline模型并保存图片
bash experiments/run_fault_analysis.sh \
    /path/to/baseline.ckpt \
    0_CUB_200_2011 \
    baseline \
    --save_images

# 方法3: 仅生成报告，不保存图片
bash experiments/run_fault_analysis.sh \
    /path/to/checkpoint.ckpt \
    1_FGVC_AIRCRAFT \
    distilled
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt_path` | str | 必需 | 模型checkpoint文件路径 |
| `--dataset` | str | `0_CUB_200_2011` | 数据集名称（如 `1_FGVC_AIRCRAFT`, `4_OxfordIIITPet`） |
| `--model_type` | str | `distilled` | 模型类型：`baseline` 或 `distilled` |
| `--save_images` | flag | False | 是否保存错误图片（添加此参数则保存） |
| `--output_dir` | str | `/data/lvta/fault_analysis` | 输出根目录 |

## 输出结构

```
output_dir/
├── CUB_200_2011/                           # 数据集目录
│   ├── distilled/                          # 模型类型目录
│   │   ├── error_images/                   # 错误图片目录
│   │   │   ├── Black_footed_Albatross_to_Laysan_Albatross/
│   │   │   │   ├── sample_00123_prob_0.856.jpg  # 样本ID_预测概率
│   │   │   │   └── sample_00456_prob_0.923.jpg
│   │   │   ├── Blue_winged_Warbler_to_Tennessee_Warbler/
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── error_cases_CUB_200_2011_distilled.json  # JSON详细数据
│   │   └── error_report_CUB_200_2011_distilled.txt  # 人类可读报告
│   └── baseline/                           # baseline模型结果
│       └── ... (相同结构)
└── FGVC_AIRCRAFT/
    └── ...
```

## 报告内容

### JSON报告 (`error_cases_*.json`)

包含结构化的错误信息，适合程序处理：

```json
{
  "checkpoint": "/path/to/checkpoint.ckpt",
  "dataset": "1_FGVC_AIRCRAFT",
  "model_type": "使用教师模型 (Teacher: convnext_xxlarge)",
  "total_samples": 3333,
  "correct_predictions": 2856,
  "error_predictions": 477,
  "accuracy": 0.8568,
  "error_rate": 0.1432,
  "error_cases": [
    {
      "sample_id": 123,
      "image_filename": "sample_00123_prob_0.856.jpg",
      "true_class_id": 5,
      "true_class_name": "Boeing 737-200",
      "true_probability": 0.0234,
      "pred_class_id": 12,
      "pred_class_name": "Boeing 737-300",
      "pred_probability": 0.8562,
      "confidence_gap": 0.8328,
      "top5_predictions": [...]
    },
    ...
  ]
}
```

### TXT报告 (`error_report_*.txt`)

人类可读的详细分析报告：

```
================================================================================
错误案例分析报告
================================================================================

生成时间: 2025-11-27 15:30:45
检查点路径: /path/to/checkpoint.ckpt
数据集: 1_FGVC_AIRCRAFT
模型类型: 使用教师模型 (Teacher: convnext_xxlarge)
设备: cuda

--------------------------------------------------------------------------------
整体性能
--------------------------------------------------------------------------------
总样本数: 3333
正确分类: 2856
错误分类: 477
准确率: 0.8568 (85.68%)
错误率: 0.1432 (14.32%)

--------------------------------------------------------------------------------
被误分类最多的真实类别 (Top 10)
--------------------------------------------------------------------------------
Boeing 737-300: 25 次
Airbus A320: 18 次
...

--------------------------------------------------------------------------------
被错误预测最多的类别 (Top 10)
--------------------------------------------------------------------------------
Boeing 737-200: 22 次
Airbus A319: 16 次
...

--------------------------------------------------------------------------------
最常见的混淆对 (Top 15)
--------------------------------------------------------------------------------
Boeing 737-300 → Boeing 737-200: 12 次
Airbus A320 → Airbus A319: 8 次
...

--------------------------------------------------------------------------------
所有错误案例详情
--------------------------------------------------------------------------------

[错误 #1]
  样本ID: 123
  图片文件: sample_00123_prob_0.856.jpg
  真实类别: Boeing 737-200 (ID: 5)
  预测类别: Boeing 737-300 (ID: 12)
  真实类别概率: 0.0234
  预测类别概率: 0.8562
  置信度差距: 0.8328
  Top-5 预测:
    1. Boeing 737-300: 0.8562
    2. Boeing 737-400: 0.0987
    3. Boeing 737-200: 0.0234
    4. Airbus A320: 0.0156
    5. Airbus A319: 0.0045

[错误 #2]
...
```

## 使用场景

1. **模型对比分析**: 比较baseline和distilled模型的错误类型差异
2. **错误模式识别**: 找出模型最容易混淆的类别对
3. **数据质量检查**: 通过查看错误图片发现标注问题
4. **模型改进**: 针对性地改进模型在困难样本上的表现

## 示例：完整分析流程

```bash
# 1. 分析baseline模型
bash experiments/run_fault_analysis.sh \
    /data/lvta/logs/baseline/checkpoints/epoch_300.ckpt \
    1_FGVC_AIRCRAFT \
    baseline \
    --save_images

# 2. 分析distilled模型
bash experiments/run_fault_analysis.sh \
    /data/lvta/logs/distilled/checkpoints/epoch_300.ckpt \
    1_FGVC_AIRCRAFT \
    distilled \
    --save_images

# 3. 对比两个模型的错误报告
diff /data/lvta/fault_analysis/FGVC_AIRCRAFT/baseline/error_report_FGVC_AIRCRAFT_baseline.txt \
     /data/lvta/fault_analysis/FGVC_AIRCRAFT/distilled/error_report_FGVC_AIRCRAFT_distilled.txt
```

## 注意事项

1. **磁盘空间**: 保存错误图片可能占用较大空间，建议先不加 `--save_images` 查看错误数量
2. **GPU内存**: 默认使用单GPU (batch_size=1)，适合大部分场景
3. **数据集名称**: 必须与 `configs/data/attributes/` 中的配置文件名称一致
4. **Checkpoint格式**: 支持PyTorch Lightning的 `.ckpt` 格式

## 技术细节

- **图片反归一化**: 使用ImageNet标准均值和标准差
- **配置管理**: 通过Hydra自动加载数据集和模型配置
- **设备检测**: 自动检测CUDA可用性，回退到CPU
- **模型类型识别**: 自动从checkpoint中提取 `use_teacher` 标志
