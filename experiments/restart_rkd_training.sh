#!/bin/bash

# 重启 RKD 300-epoch 训练(从checkpoint恢复)
# 解决 NCCL 通信超时问题

# 设置环境变量以解决 NCCL 超时问题
export NCCL_TIMEOUT=7200000  # 增加超时时间到 2小时
export NCCL_DEBUG=INFO       # 启用调试信息
export NCCL_IB_DISABLE=0     # 启用 InfiniBand (如果可用)
export NCCL_P2P_DISABLE=0    # 启用 P2P 通信
export NCCL_SOCKET_IFNAME=^docker0,lo  # 排除虚拟网卡

# PyTorch 分布式训练环境变量
export TORCH_DISTRIBUTED_DEBUG=INFO  # 调试信息
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理
export TORCH_NCCL_BLOCKING_WAIT=1  # 阻塞等待,更容易调试

# 查找最新的 checkpoint
CKPT_DIR="/data/lvta/logs/vl2lite/rkd_full_300epochs/runs"
LATEST_RUN=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "2025-11-18_*" | sort | tail -1)
LAST_CKPT=$(find "$LATEST_RUN" -name "last.ckpt" 2>/dev/null | head -1)

echo "================================"
echo "重启 RKD 训练 (300 epochs)"
echo "================================"
echo "Latest run directory: $LATEST_RUN"
echo "Last checkpoint: $LAST_CKPT"
echo ""

# 检查是否有checkpoint可以恢复
if [ -f "$LAST_CKPT" ]; then
    echo "✓ 找到 checkpoint,将从 epoch $(basename $LATEST_RUN | cut -d'_' -f3) 继续训练"
    CKPT_ARG="ckpt_path=$LAST_CKPT"
else
    echo "✗ 未找到 checkpoint,将从头开始训练"
    CKPT_ARG=""
fi

# 启动训练
LOG_FILE="experiments/logs/train_rkd_restart_$(date +%Y%m%d_%H%M%S).log"

echo "日志文件: $LOG_FILE"
echo "开始训练..."
echo ""

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python src/train.py \
    data/attributes=0_CUB_200_2011 \
    trainer=ddp \
    trainer.devices=4 \
    trainer.max_epochs=300 \
    data.batch_size=128 \
    model.use_teacher=true \
    logger=csv \
    task_name="rkd_full_300epochs_restart" \
    $CKPT_ARG \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "训练已启动,PID: $PID"
echo "查看日志: tail -f $LOG_FILE"
echo "监控训练: watch -n 30 'tail -30 $LOG_FILE'"
echo "停止训练: kill $PID"
