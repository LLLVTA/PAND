#!/bin/bash
# 错误案例分析脚本 - 分析模型预测错误并保存错误图片
# 
# 使用方法:
#   1. 分析distilled模型 (默认保存图片):
#      bash experiments/run_fault_analysis.sh /path/to/distilled.ckpt 1_FGVC_AIRCRAFT distilled --save_images
#   
#   2. 分析baseline模型 (默认保存图片):
#      bash experiments/run_fault_analysis.sh /path/to/baseline.ckpt 0_CUB_200_2011 baseline --save_images
#   
#   3. 仅生成报告不保存图片:
#      bash experiments/run_fault_analysis.sh /path/to/checkpoint.ckpt 1_FGVC_AIRCRAFT distilled

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vl2lite_env

# 参数解析
CKPT_PATH="${1:-/data/lvta/logs/vl2lite/train/runs/2025-11-26_00-20-45/checkpoints/last.ckpt}"
DATASET="${2:-1_FGVC_AIRCRAFT}"
MODEL_TYPE="${3:-distilled}"  # baseline 或 distilled
SAVE_IMAGES="${4}"  # --save_images 或留空

# 输出目录（按数据集和模型类型组织）
OUTPUT_DIR="/data/lvta/fault_analysis"

echo "========================================"
echo "错误案例分析"
echo "========================================"
echo "Checkpoint:  $CKPT_PATH"
echo "Dataset:     $DATASET"
echo "Model Type:  $MODEL_TYPE"
echo "Save Images: ${SAVE_IMAGES:-No}"
echo "Output Dir:  $OUTPUT_DIR"
echo "========================================"

# 构建命令
CMD="python experiments/fault_analysis.py \
    --ckpt_path \"$CKPT_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --dataset \"$DATASET\" \
    --model_type \"$MODEL_TYPE\""

# 如果指定保存图片，添加参数
if [ "$SAVE_IMAGES" == "--save_images" ]; then
    CMD="$CMD --save_images"
fi

# 执行分析
eval $CMD

echo ""
echo "分析完成！结果保存在: $OUTPUT_DIR/$DATASET/$MODEL_TYPE/"
echo "  - JSON: error_cases_${DATASET#*_}_${MODEL_TYPE}.json"
echo "  - TXT:  error_report_${DATASET#*_}_${MODEL_TYPE}.txt"
if [ "$SAVE_IMAGES" == "--save_images" ]; then
    echo "  - 图片: error_images/"
fi
