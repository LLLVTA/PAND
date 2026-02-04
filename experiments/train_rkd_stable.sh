#!/bin/bash

# 稳健的 RKD 训练脚本 - 解决 NCCL 超时问题
# 使用更保守的设置确保训练稳定

# ========== NCCL 环境变量配置 ==========
export NCCL_TIMEOUT=7200000              # 超时时间: 2小时
export NCCL_DEBUG=INFO                   # 调试级别
export NCCL_IB_DISABLE=1                 # 禁用 InfiniBand (系统未安装)
export NCCL_P2P_DISABLE=0                # 启用 P2P 通信
export NCCL_SOCKET_IFNAME=eno1,eno2      # 使用物理网卡,排除虚拟网卡
export NCCL_ASYNC_ERROR_HANDLING=1       # 异步错误处理

# ========== PyTorch 分布式训练配置 ==========
export TORCH_DISTRIBUTED_DEBUG=INFO      # 分布式调试信息
export TORCH_NCCL_BLOCKING_WAIT=1        # 阻塞等待
export OMP_NUM_THREADS=4                 # OpenMP 线程数

# ========== 训练配置 ==========
TASK_NAME="rkd_300ep_stable"
BATCH_SIZE=64                            # 减少 batch size 降低通信量
DEVICES=4
MAX_EPOCHS=300
ACCUMULATE_GRAD=2                        # gradient accumulation 补偿小 batch

# ========== 查找 checkpoint ==========
CKPT_DIR="/data/lvta/logs/vl2lite/rkd_full_300epochs/runs/2025-11-18_11-32-47"
LAST_CKPT="$CKPT_DIR/last.ckpt"

echo "========================================"
echo "稳健 RKD 训练脚本"
echo "========================================"
echo "任务名称: $TASK_NAME"
echo "Batch size: $BATCH_SIZE (per GPU)"
echo "有效 batch size: $((BATCH_SIZE * DEVICES * ACCUMULATE_GRAD))"
echo "Gradient accumulation: $ACCUMULATE_GRAD"
echo "GPUs: $DEVICES"
echo "Max epochs: $MAX_EPOCHS"
echo ""

# 检查 checkpoint
if [ -f "$LAST_CKPT" ]; then
    echo "✓ 找到 checkpoint: $LAST_CKPT"
    echo "  将从 epoch 6 继续训练"
    CKPT_ARG="ckpt_path=$LAST_CKPT"
else
    echo "✗ 未找到 checkpoint,将从头开始"
    CKPT_ARG=""
fi

echo ""
echo "开始训练..."
echo "========================================"

LOG_FILE="experiments/logs/train_rkd_stable_$(date +%Y%m%d_%H%M%S).log"

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python src/train.py \
    data/attributes=0_CUB_200_2011 \
    trainer=ddp \
    trainer.devices=$DEVICES \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.sync_batchnorm=false \
    trainer.accumulate_grad_batches=$ACCUMULATE_GRAD \
    data.batch_size=$BATCH_SIZE \
    data.num_workers=8 \
    model.use_teacher=true \
    logger=csv \
    task_name="$TASK_NAME" \
    $CKPT_ARG \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo ""
echo "✓ 训练已启动!"
echo "  PID: $PID"
echo "  日志: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  实时日志: tail -f $LOG_FILE"
echo "  错误检查: grep -i 'error\|timeout' $LOG_FILE"
echo "  停止训练: kill $PID"
echo ""
echo "查看进度:"
echo "  watch -n 30 'tail -30 $LOG_FILE'"
