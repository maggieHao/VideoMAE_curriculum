import pandas as pd
import numpy as np

def sample_by_group(input_file, output_file, sample_ratio=0.2):
    """
    从CSV文件中按组随机抽样数据并保存
    
    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    sample_ratio (float): 采样比例，默认0.2（20%）
    """
    # 读取CSV文件，无表头，指定列名
    df = pd.read_csv(input_file, header=None, names=['path', 'label'], sep=' ')
    
    # 按label分组并应用随机采样
    sampled_dfs = []
    for label, group in df.groupby('label'):
        # 计算需要采样的数量
        sample_size = max(int(len(group) * sample_ratio), 1)
        # 随机采样
        sampled_group = group.sample(n=sample_size, random_state=42)
        sampled_dfs.append(sampled_group)
    
    # 合并所有采样后的数据
    result_df = pd.concat(sampled_dfs)
    
    # 按原格式保存（无表头，空格分隔）
    result_df.to_csv(output_file, header=False, index=False, sep=' ')
    
    # 打印采样统计信息
    print(f"原始数据总数: {len(df)}")
    print(f"采样后数据总数: {len(result_df)}")
    print("\n按标签统计:")
    
    original_counts = df.groupby('label').size()
    sampled_counts = result_df.groupby('label').size()
    
    for label in original_counts.index:
        orig = original_counts[label]
        samp = sampled_counts[label]
        print(f"标签 {label}:")
        print(f"  - 原始数量: {orig}")
        print(f"  - 采样数量: {samp}")
        print(f"  - 采样比例: {(samp/orig*100):.2f}%")

if __name__ == "__main__":
    # 使用示例
    input_file = "/home/maggie/kinetics-dataset/k400/test.csv"
    output_file = "/home/maggie/VideoMAE_curriculum/labels/k400/test.csv"
    
    # 执行采样
    sample_by_group(input_file, output_file)