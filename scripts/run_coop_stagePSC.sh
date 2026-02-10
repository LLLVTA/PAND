#!/bin/bash
# CoOp Stage A: 训练可学习文本 prompts
# 使用预计算的 CLIP 图像特征，节省显存（从 15GB 降到 3-4GB）

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
DATASET="5_StanfordDogs"  # 数据集名称
DATASET_NAME="stanford_dogs"  # 简短名称
DATA_ROOT="/data/lvta/datasets/vl2lite_datasets/${DATASET}"
CONFIG_PATH="/home/lvta/Project/VL2Lite/configs/data/attributes/${DATASET}.yaml"

# CLIP 模型配置
CLIP_MODEL="convnext_xxlarge"
PRETRAINED="laion2b_s34b_b82k_augreg_soup"

# 输出路径
OUTPUT_DIR="/data/lvta/logs/vl2lite/train/runs/coop_${DATASET_NAME}"
FEATURES_PATH="${OUTPUT_DIR}/features/${DATASET_NAME}_clip_features.pt"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
TEXT_FEATURES_PATH="${OUTPUT_DIR}/learned_text_features.pt"

# GPU 配置
GPUS="0,1,2,3"  # 使用的 GPU 编号

# 可选：Stage A 完成后自动启动 Stage B（KD），默认启动
RUN_STAGEB="false"  # 设为 "false" 可禁用自动 Stage B

echo "============================================================"
echo "CoOp Stage A Training Pipeline"
echo "============================================================"
echo "Dataset: ${DATASET}"
echo "CLIP Model: ${CLIP_MODEL}"
echo "Pretrained: ${PRETRAINED}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "GPUs: ${GPUS}"
echo "============================================================"
echo ""

# ==================== Step 1: 提取 CLIP 图像特征 ====================
echo "[Step 1/3] Extracting CLIP image features..."
echo "This will take ~5-10 minutes depending on dataset size."
echo ""

if [ -f "${FEATURES_PATH}" ]; then
    echo "✓ Features file already exists: ${FEATURES_PATH}"
    read -p "Do you want to re-extract features? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping feature extraction..."
    else
    echo "Re-extracting features..."
        CLIP_MODEL="${CLIP_MODEL}" \
        PRETRAINED="${PRETRAINED}" \
        DATA_ROOT="${DATA_ROOT}" \
        OUTPUT_PATH="${FEATURES_PATH}" \
        python src/extract_image_features.py
    fi
else
    echo "Extracting features to: ${FEATURES_PATH}"
    CLIP_MODEL="${CLIP_MODEL}" \
    PRETRAINED="${PRETRAINED}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUTPUT_PATH="${FEATURES_PATH}" \
    python src/extract_image_features.py
fi

echo ""
echo "✓ Step 1 completed!"
echo ""

# ==================== Step 2: 训练 CoOp (学习文本 prompts) ====================
echo "[Step 2/3] Training CoOp with precomputed features..."
echo "This will train learnable context tokens for text prompts."
echo ""

# 检查特征文件是否存在
if [ ! -f "${FEATURES_PATH}" ]; then
    echo "Error: Features file not found: ${FEATURES_PATH}"
    echo "Please run Step 1 first."
    exit 1
fi

echo "Starting CoOp training..."
echo "If you want to resume, set CKPT_PATH (e.g., export CKPT_PATH=.../last.ckpt)"
CUDA_VISIBLE_DEVICES=${GPUS} \
    CLIP_MODEL="${CLIP_MODEL}" \
    PRETRAINED="${PRETRAINED}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    FEATURES_PATH="${FEATURES_PATH}" \
    CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    CKPT_PATH="${CKPT_PATH:-}" \
    python src/train_coop_cub_features.py

echo ""
echo "✓ Step 2 completed!"
echo ""

# ==================== Step 3: 提取学习到的文本特征 ====================
echo "[Step 3/3] Extracting learned text features from best checkpoint..."
echo ""

# 查找最佳 checkpoint
# 格式: coop_<dataset>-epoch=122-valacc=0.7050.ckpt
# 按准确率数值排序，取最高的；若为空且存在last.ckpt，则回退到last.ckpt
BEST_CKPT=$(find ${CHECKPOINT_DIR} -name "coop_*-epoch=*-valacc=*.ckpt" -type f | grep -v "last" | while read f; do
    acc=$(basename "$f" | grep -oP 'valacc=\K[0-9.]+')
    echo "$acc $f"
done | sort -rn | head -1 | awk '{print $2}')

if [ -z "${BEST_CKPT}" ] && [ -f "${CHECKPOINT_DIR}/last.ckpt" ]; then
    echo "No best checkpoint found, fallback to last.ckpt"
    BEST_CKPT="${CHECKPOINT_DIR}/last.ckpt"
fi

if [ -z "${BEST_CKPT}" ]; then
    echo "Error: No checkpoint found in ${CHECKPOINT_DIR}"
    echo "Please check if Step 2 training completed successfully."
    exit 1
fi

echo "Found best checkpoint: ${BEST_CKPT}"
echo "Extracting text features to: ${TEXT_FEATURES_PATH}"
echo ""

# 提取文本特征
python scripts/extract_coop_features.py \
    --checkpoint "${BEST_CKPT}" \
    --output "${TEXT_FEATURES_PATH}" \
    --config "${CONFIG_PATH}" \
    --clip_model "${CLIP_MODEL}" \
    --pretrained "${PRETRAINED}"

echo ""
echo "✓ Step 3 completed!"
echo ""

# ==================== 完成总结 ====================
echo "============================================================"
echo "CoOp Stage A Training Completed Successfully!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  1. CLIP Features:    ${FEATURES_PATH}"
echo "  2. CoOp Checkpoint:  ${BEST_CKPT}"
echo "  3. Text Features:    ${TEXT_FEATURES_PATH}"
echo ""
echo "Next steps:"
echo "  Run Stage B (Knowledge Distillation) with:"
echo "  bash scripts/run_coop_stageB.sh"
echo ""
echo "Or directly run Stage B (manual command):"
echo "  TMPDIR=/data/lvta/tmp nohup python src/train.py \\" 
echo "    data/attributes=${DATASET} \\" 
echo "    trainer=ddp trainer.devices=4 trainer.max_epochs=300 \\" 
echo "    data.batch_size=128 \\" 
echo "    model=coop_kd \\" 
echo "    model.kd_criterion.use_nlrd=true \\" 
echo "    model.kd_criterion.nlrd_k=3 \\" 
echo "    model.kd_criterion.nlrd_weight=1.0 \\" 
echo "    logger=csv \\" 
echo "    'tags=[\"${DATASET_NAME}\",\"vl2lite\",\"nlrd\",\"k3\",\"coop_stageB\",\"nlrdw1\"]' \\" 
echo "    > experiments/logs/train_${DATASET_NAME}_coop_stageB_$(date +\"%Y%m%d_%H%M%S\").log 2>&1 &"
echo ""
echo "============================================================"

# ==================== Auto Stage B Execution ====================
if [ "${RUN_STAGEB}" = "true" ]; then
    echo ""
    echo "============================================================"
    echo "Auto-starting Stage B (Knowledge Distillation)..."
    echo "============================================================"
    echo ""
    
    # 更新 coop_kd.yaml 中的文本特征路径
    COOP_KD_CONFIG="configs/model/coop_kd.yaml"
    echo "Updating ${COOP_KD_CONFIG} with learned text features path..."
    
    # 使用 sed 替换 coop_text_features 路径
    sed -i "s|coop_text_features:.*|coop_text_features: ${TEXT_FEATURES_PATH}|" "${COOP_KD_CONFIG}"
    
    echo "✓ Config updated: coop_text_features -> ${TEXT_FEATURES_PATH}"
    echo ""
    
    # 启动 Stage B 训练
    STAGEB_LOG="experiments/logs/train_${DATASET_NAME}_coop_stageB_$(date +"%Y%m%d_%H%M%S").log"
    echo "Starting Stage B training (log: ${STAGEB_LOG})..."
    
    TMPDIR=/data/lvta/tmp nohup python src/train.py \
        data/attributes=${DATASET} \
        trainer=ddp \
        trainer.devices=4 \
        trainer.max_epochs=300 \
        data.batch_size=128 \
        model=coop_kd \
        model.kd_criterion.use_nlrd=true \
        model.kd_criterion.nlrd_k=3 \
        model.kd_criterion.nlrd_weight=1.0 \
        logger=csv \
        "tags=[\"${DATASET_NAME}\",\"vl2lite\",\"nlrd\",\"k3\",\"coop_stageB\",\"nlrdw1\"]" \
        > "${STAGEB_LOG}" 2>&1 &
    
    STAGEB_PID=$!
    echo "✓ Stage B started (PID: ${STAGEB_PID})"
    echo "✓ Monitor progress: tail -f ${STAGEB_LOG}"
    echo ""
else
    echo ""
    echo "Stage B auto-run disabled (RUN_STAGEB=${RUN_STAGEB})"
    echo "To enable: export RUN_STAGEB=true before running this script"
    echo ""
fi
