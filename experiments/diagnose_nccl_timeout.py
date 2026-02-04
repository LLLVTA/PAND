#!/usr/bin/env python3
"""
诊断 NCCL 通信超时问题
检查系统资源和配置
"""

import subprocess
import os
from pathlib import Path

def run_command(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_gpu_status():
    """检查 GPU 状态"""
    print("\n" + "="*60)
    print("GPU 状态检查")
    print("="*60)
    
    # GPU 使用情况
    gpu_info = run_command("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
    print("\nGPU 状态:")
    for line in gpu_info.split('\n'):
        print(f"  {line}")
    
    # 检查是否有卡住的 GPU 进程
    processes = run_command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    print("\nGPU 进程:")
    if processes:
        for line in processes.split('\n'):
            print(f"  {line}")
    else:
        print("  无 GPU 进程运行")

def check_network():
    """检查网络接口"""
    print("\n" + "="*60)
    print("网络接口检查")
    print("="*60)
    
    # 列出所有网络接口
    interfaces = run_command("ip link show | grep -E '^[0-9]+:' | awk '{print $2}' | sed 's/:$//'")
    print("\n可用网络接口:")
    for interface in interfaces.split('\n'):
        if interface:
            print(f"  - {interface}")
    
    # 检查 InfiniBand
    ib_status = run_command("which ibstat")
    if "ibstat" in ib_status:
        ib_info = run_command("ibstat | grep -E 'State|Rate'")
        print("\nInfiniBand 状态:")
        print(f"  {ib_info if ib_info else '未检测到 IB 设备'}")
    else:
        print("\nInfiniBand: 未安装")

def check_nccl_tests():
    """检查 NCCL 测试工具"""
    print("\n" + "="*60)
    print("NCCL 测试工具")
    print("="*60)
    
    nccl_test = run_command("which nccl-tests || which all_reduce_perf")
    if nccl_test:
        print(f"\nNCCL 测试工具路径: {nccl_test}")
        print("建议运行: mpirun -np 4 nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1")
    else:
        print("\n未安装 NCCL 测试工具")
        print("建议安装: git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests && make")

def check_checkpoint():
    """检查最新的 checkpoint"""
    print("\n" + "="*60)
    print("Checkpoint 状态")
    print("="*60)
    
    ckpt_dir = Path("/data/lvta/logs/vl2lite/rkd_full_300epochs/runs")
    
    if ckpt_dir.exists():
        # 查找最新的运行目录
        run_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("2025-")])
        
        if run_dirs:
            latest_run = run_dirs[-1]
            print(f"\n最新运行目录: {latest_run.name}")
            
            # 查找 checkpoint 文件
            ckpt_files = list(latest_run.rglob("*.ckpt"))
            if ckpt_files:
                print("\nCheckpoint 文件:")
                for ckpt in sorted(ckpt_files):
                    size_mb = ckpt.stat().st_size / (1024*1024)
                    print(f"  - {ckpt.name}: {size_mb:.1f} MB")
                
                # 读取 CSV metrics
                csv_file = latest_run / "csv" / "version_0" / "metrics.csv"
                if csv_file.exists():
                    print("\n最后训练的 epoch:")
                    last_epochs = run_command(f"tail -5 {csv_file}")
                    for line in last_epochs.split('\n')[-3:]:
                        print(f"  {line}")
            else:
                print("\n未找到 checkpoint 文件")
        else:
            print("\n未找到运行目录")
    else:
        print("\nCheckpoint 目录不存在")

def suggest_solutions():
    """提供解决方案建议"""
    print("\n" + "="*60)
    print("解决 NCCL 超时问题的建议")
    print("="*60)
    
    solutions = [
        "1. 增加 NCCL 超时时间:",
        "   export NCCL_TIMEOUT=7200000  # 2小时",
        "",
        "2. 减少 batch size (降低通信量):",
        "   data.batch_size=64  # 从 128 降到 64",
        "",
        "3. 使用 gradient accumulation (减少通信频率):",
        "   trainer.accumulate_grad_batches=2",
        "",
        "4. 禁用 sync_batchnorm (减少通信):",
        "   trainer.sync_batchnorm=false",
        "",
        "5. 改用单 GPU 训练 (避免分布式通信):",
        "   trainer.devices=1 data.batch_size=64",
        "",
        "6. 使用 fp16 混合精度 (减少通信数据量):",
        "   trainer.precision=16",
        "",
        "7. 检查网络和硬件:",
        "   - 运行 nvidia-smi 检查 GPU 错误",
        "   - 检查 dmesg 查看系统日志",
        "   - 测试 GPU 间 P2P 通信",
    ]
    
    for line in solutions:
        print(f"  {line}")

def main():
    print("\n" + "="*80)
    print("NCCL 通信超时诊断报告")
    print("="*80)
    
    check_gpu_status()
    check_network()
    check_nccl_tests()
    check_checkpoint()
    suggest_solutions()
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)

if __name__ == "__main__":
    main()
