#!/usr/bin/env python
"""
对比 Baseline 和 Distilled 模型的错误案例
分析四种情况：
1. 两者都对 (Both Correct)
2. 两者都错 (Both Wrong)
3. Baseline对，Distilled错 (Baseline Better)
4. Baseline错，Distilled对 (Distilled Better)
"""

import os
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_error_cases(csv_path):
    """加载错误案例CSV，返回错误样本的ID集合"""
    if not os.path.exists(csv_path):
        print(f"警告: 文件不存在 - {csv_path}")
        return set(), pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    error_ids = set(df['sample_id'].values)
    return error_ids, df


def compare_models(dataset_folder, dataset_name, total_samples):
    """
    对比 baseline 和 distilled 模型
    
    Args:
        dataset_folder: 数据集文件夹路径 (如 /data/lvta/fault_analysis/0_CUB_200_2011)
        dataset_name: 数据集名称 (如 CUB_200_2011)
        total_samples: 测试集总样本数
    """
    
    baseline_csv = Path(dataset_folder) / 'baseline' / f'error_cases_{dataset_name}_baseline.csv'
    distilled_csv = Path(dataset_folder) / 'distilled' / f'error_cases_{dataset_name}_distilled.csv'
    
    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")
    print(f"Baseline CSV:  {baseline_csv}")
    print(f"Distilled CSV: {distilled_csv}")
    
    # 加载错误案例
    baseline_errors, baseline_df = load_error_cases(baseline_csv)
    distilled_errors, distilled_df = load_error_cases(distilled_csv)
    
    # 计算四种情况
    both_wrong = baseline_errors & distilled_errors  # 交集：两者都错
    baseline_only = baseline_errors - distilled_errors  # Baseline错，Distilled对
    distilled_only = distilled_errors - baseline_errors  # Baseline对，Distilled错
    both_correct = total_samples - len(baseline_errors | distilled_errors)  # 并集的补集：两者都对
    
    # 打印统计结果
    print(f"\n{'-'*80}")
    print(f"统计结果")
    print(f"{'-'*80}")
    print(f"测试集总样本数:        {total_samples}")
    print(f"\nBaseline 错误数:       {len(baseline_errors)} ({len(baseline_errors)/total_samples*100:.2f}%)")
    print(f"Distilled 错误数:      {len(distilled_errors)} ({len(distilled_errors)/total_samples*100:.2f}%)")
    print(f"\n四种情况分布:")
    print(f"  1. 两者都对:          {both_correct} ({both_correct/total_samples*100:.2f}%)")
    print(f"  2. 两者都错:          {len(both_wrong)} ({len(both_wrong)/total_samples*100:.2f}%)")
    print(f"  3. Baseline对，Distilled错:  {len(distilled_only)} ({len(distilled_only)/total_samples*100:.2f}%) ❌ 退化")
    print(f"  4. Baseline错，Distilled对:  {len(baseline_only)} ({len(baseline_only)/total_samples*100:.2f}%) ✅ 改进")
    
    # 计算改进指标
    net_improvement = len(baseline_only) - len(distilled_only)
    improvement_rate = (len(baseline_only) / len(baseline_errors) * 100) if len(baseline_errors) > 0 else 0
    degradation_rate = (len(distilled_only) / (total_samples - len(baseline_errors)) * 100) if (total_samples - len(baseline_errors)) > 0 else 0
    
    print(f"\n{'-'*80}")
    print(f"改进分析")
    print(f"{'-'*80}")
    print(f"净改进样本数:         {net_improvement}")
    print(f"修复率 (修复/baseline错误): {improvement_rate:.2f}%")
    print(f"退化率 (退化/baseline正确): {degradation_rate:.2f}%")
    
    # 详细分析：退化案例（Baseline对，Distilled错）
    if len(distilled_only) > 0:
        print(f"\n{'-'*80}")
        print(f"退化案例分析 (Baseline对但Distilled错的 {len(distilled_only)} 个样本)")
        print(f"{'-'*80}")
        
        # 获取这些样本的详细信息
        degradation_df = distilled_df[distilled_df['sample_id'].isin(distilled_only)]
        
        # 按真实类别统计
        degradation_by_class = degradation_df['true_class_name'].value_counts().head(10)
        print(f"\n退化最多的真实类别 (Top 10):")
        for cls, count in degradation_by_class.items():
            print(f"  {cls}: {count} 次")
        
        # 按预测类别统计
        degradation_by_pred = degradation_df['pred_class_name'].value_counts().head(10)
        print(f"\n退化后被错误预测为的类别 (Top 10):")
        for cls, count in degradation_by_pred.items():
            print(f"  {cls}: {count} 次")
        
        # 平均置信度
        avg_conf = degradation_df['pred_probability'].astype(float).mean()
        print(f"\n退化案例的平均预测置信度: {avg_conf:.4f}")
    
    # 详细分析：改进案例（Baseline错，Distilled对）
    if len(baseline_only) > 0:
        print(f"\n{'-'*80}")
        print(f"改进案例分析 (Baseline错但Distilled对的 {len(baseline_only)} 个样本)")
        print(f"{'-'*80}")
        
        # 获取这些样本的详细信息（从baseline的错误记录中）
        improvement_df = baseline_df[baseline_df['sample_id'].isin(baseline_only)]
        
        # 按真实类别统计
        improvement_by_class = improvement_df['true_class_name'].value_counts().head(10)
        print(f"\n改进最多的真实类别 (Top 10):")
        for cls, count in improvement_by_class.items():
            print(f"  {cls}: {count} 次")
        
        # Baseline原来错误预测为哪些类别
        improvement_by_pred = improvement_df['pred_class_name'].value_counts().head(10)
        print(f"\nBaseline原来错误预测为的类别 (Top 10):")
        for cls, count in improvement_by_pred.items():
            print(f"  {cls}: {count} 次")
        
        # 平均置信度
        avg_conf = improvement_df['pred_probability'].astype(float).mean()
        print(f"\nBaseline错误时的平均置信度: {avg_conf:.4f}")
    
    # 保存对比结果到CSV
    output_dir = Path(dataset_folder) / 'comparison'
    output_dir.mkdir(exist_ok=True)
    
    comparison_summary = {
        'Dataset': [dataset_name],
        'Total_Samples': [total_samples],
        'Baseline_Errors': [len(baseline_errors)],
        'Distilled_Errors': [len(distilled_errors)],
        'Both_Correct': [both_correct],
        'Both_Wrong': [len(both_wrong)],
        'Baseline_Better': [len(distilled_only)],
        'Distilled_Better': [len(baseline_only)],
        'Net_Improvement': [net_improvement],
        'Improvement_Rate': [improvement_rate],
        'Degradation_Rate': [degradation_rate]
    }
    
    summary_df = pd.DataFrame(comparison_summary)
    summary_csv = output_dir / f'{dataset_name}_comparison_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n对比摘要已保存到: {summary_csv}")
    
    # 保存详细的样本ID列表
    detail_csv = output_dir / f'{dataset_name}_sample_categories.csv'
    
    max_len = max(both_correct, len(both_wrong), len(baseline_only), len(distilled_only))
    
    detail_data = {
        'Both_Correct_IDs': list(range(total_samples - len(baseline_errors | distilled_errors)))[:max_len] + [''] * (max_len - both_correct),
        'Both_Wrong_IDs': list(both_wrong) + [''] * (max_len - len(both_wrong)),
        'Baseline_Better_IDs': list(distilled_only) + [''] * (max_len - len(distilled_only)),
        'Distilled_Better_IDs': list(baseline_only) + [''] * (max_len - len(baseline_only))
    }
    
    detail_df = pd.DataFrame(detail_data)
    detail_df.to_csv(detail_csv, index=False)
    print(f"详细样本分类已保存到: {detail_csv}")
    
    # 保存退化案例的详细信息（Baseline对但Distilled错）
    if len(distilled_only) > 0:
        degradation_detail = distilled_df[distilled_df['sample_id'].isin(distilled_only)].copy()
        degradation_detail = degradation_detail.sort_values('sample_id')
        
        degradation_csv = output_dir / f'{dataset_name}_degradation_details.csv'
        degradation_detail.to_csv(degradation_csv, index=False)
        print(f"退化案例详情已保存到: {degradation_csv}")
        print(f"  包含 {len(degradation_detail)} 个样本的完整信息（sample_id, 图片文件名, 真实类别, 预测类别, 置信度等）")
    
    return {
        'dataset': dataset_name,
        'total': total_samples,
        'both_correct': both_correct,
        'both_wrong': len(both_wrong),
        'baseline_better': len(distilled_only),
        'distilled_better': len(baseline_only),
        'net_improvement': net_improvement
    }


def main():
    """主函数"""
    
    # 定义数据集配置 (文件夹名, 显示名, 测试集样本数)
    datasets = [
        ('/data/lvta/fault_analysis/0_CUB_200_2011', 'CUB_200_2011', 5794),
        ('/data/lvta/fault_analysis/1_FGVC_AIRCRAFT', 'FGVC_AIRCRAFT', 3333),
        ('/data/lvta/fault_analysis/4_OxfordIIITPet', 'OxfordIIITPet', 3669),
    ]
    
    all_results = []
    
    for dataset_folder, dataset_name, total_samples in datasets:
        result = compare_models(dataset_folder, dataset_name, total_samples)
        all_results.append(result)
    
    # 汇总所有数据集的结果
    print(f"\n{'='*80}")
    print(f"所有数据集汇总")
    print(f"{'='*80}\n")
    
    summary_table = pd.DataFrame(all_results)
    summary_table.columns = [
        'Dataset', 'Total', 'Both Correct', 'Both Wrong', 
        'Baseline Better', 'Distilled Better', 'Net Improvement'
    ]
    
    print(summary_table.to_string(index=False))
    
    # 保存总汇总
    output_path = '/data/lvta/fault_analysis/overall_comparison.csv'
    summary_table.to_csv(output_path, index=False)
    print(f"\n总汇总已保存到: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
