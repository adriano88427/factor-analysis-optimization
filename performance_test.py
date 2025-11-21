"""
性能测试脚本
用于比较原始IC计算方法和优化方法的性能差异
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from optimized_ic_calculation import OptimizedICCalculator, benchmark_ic_methods


def generate_test_data(n_samples: int = 10000, missing_ratio: float = 0.05) -> pd.DataFrame:
    """
    生成测试数据
    
    参数:
        n_samples: 样本数量
        missing_ratio: 缺失值比例
        
    返回:
        包含因子和收益数据的DataFrame
    """
    np.random.seed(42)
    
    # 生成因子数据（带有一些趋势）
    trend = np.linspace(0, 1, n_samples)
    factor_data = trend + np.random.randn(n_samples) * 0.5
    
    # 生成收益数据（与因子有一定相关性）
    correlation = 0.3
    noise = np.random.randn(n_samples) * np.sqrt(1 - correlation**2)
    return_data = correlation * factor_data + noise
    
    # 创建DataFrame
    df = pd.DataFrame({
        'factor': factor_data,
        'return': return_data
    })
    
    # 添加缺失值
    if missing_ratio > 0:
        n_missing = int(n_samples * missing_ratio)
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        df.loc[missing_indices, 'factor'] = np.nan
    
    return df


def test_different_data_sizes():
    """测试不同数据量下的性能"""
    print("测试不同数据量下的性能:")
    print("=" * 50)
    
    data_sizes = [1000, 5000, 10000, 20000, 50000]
    results = {}
    
    for size in data_sizes:
        print(f"\n数据量: {size}")
        test_data = generate_test_data(size)
        
        # 运行性能测试
        benchmark_results = benchmark_ic_methods(test_data, 'factor', 'return', use_pearson=True)
        results[size] = benchmark_results
        
        # 打印结果
        print(f"原始方法: {benchmark_results['original']['time']:.4f}秒")
        print(f"pandas rolling方法: {benchmark_results['pandas_rolling']['time']:.4f}秒 (提升: {benchmark_results['pandas_rolling']['speedup']:.2f}x)")
        print(f"numpy方法: {benchmark_results['numpy']['time']:.4f}秒 (提升: {benchmark_results['numpy']['speedup']:.2f}x)")
        print(f"numba方法: {benchmark_results['numba']['time']:.4f}秒 (提升: {benchmark_results['numba']['speedup']:.2f}x)")
    
    return results


def test_different_missing_ratios():
    """测试不同缺失值比例下的性能"""
    print("\n\n测试不同缺失值比例下的性能:")
    print("=" * 50)
    
    missing_ratios = [0.0, 0.05, 0.1, 0.2, 0.3]
    results = {}
    
    for ratio in missing_ratios:
        print(f"\n缺失值比例: {ratio:.1%}")
        test_data = generate_test_data(10000, ratio)
        
        # 运行性能测试
        benchmark_results = benchmark_ic_methods(test_data, 'factor', 'return', use_pearson=True)
        results[ratio] = benchmark_results
        
        # 打印结果
        print(f"原始方法: {benchmark_results['original']['time']:.4f}秒")
        print(f"pandas rolling方法: {benchmark_results['pandas_rolling']['time']:.4f}秒 (提升: {benchmark_results['pandas_rolling']['speedup']:.2f}x)")
        print(f"numpy方法: {benchmark_results['numpy']['time']:.4f}秒 (提升: {benchmark_results['numpy']['speedup']:.2f}x)")
        print(f"numba方法: {benchmark_results['numba']['time']:.4f}秒 (提升: {benchmark_results['numba']['speedup']:.2f}x)")
    
    return results


def test_correlation_methods():
    """测试Pearson和Spearman相关系数的性能差异"""
    print("\n\n测试不同相关系数方法的性能:")
    print("=" * 50)
    
    test_data = generate_test_data(10000, 0.05)
    correlation_methods = [('Pearson', True), ('Spearman', False)]
    results = {}
    
    for method_name, use_pearson in correlation_methods:
        print(f"\n相关系数方法: {method_name}")
        
        # 运行性能测试
        benchmark_results = benchmark_ic_methods(test_data, 'factor', 'return', use_pearson=use_pearson)
        results[method_name] = benchmark_results
        
        # 打印结果
        print(f"原始方法: {benchmark_results['original']['time']:.4f}秒")
        print(f"pandas rolling方法: {benchmark_results['pandas_rolling']['time']:.4f}秒 (提升: {benchmark_results['pandas_rolling']['speedup']:.2f}x)")
        print(f"numpy方法: {benchmark_results['numpy']['time']:.4f}秒 (提升: {benchmark_results['numpy']['speedup']:.2f}x)")
        print(f"numba方法: {benchmark_results['numba']['time']:.4f}秒 (提升: {benchmark_results['numba']['speedup']:.2f}x)")
    
    return results


def plot_performance_comparison(results_dict: dict, title: str, save_path: str = None):
    """
    绘制性能比较图
    
    参数:
        results_dict: 结果字典
        title: 图表标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 8))
    
    # 提取数据
    x_values = list(results_dict.keys())
    original_times = [results_dict[x]['original']['time'] for x in x_values]
    pandas_times = [results_dict[x]['pandas_rolling']['time'] for x in x_values]
    numpy_times = [results_dict[x]['numpy']['time'] for x in x_values]
    numba_times = [results_dict[x]['numba']['time'] for x in x_values]
    
    # 绘制图表
    plt.plot(x_values, original_times, 'o-', label='原始方法')
    plt.plot(x_values, pandas_times, 's-', label='pandas rolling方法')
    plt.plot(x_values, numpy_times, '^-', label='numpy方法')
    plt.plot(x_values, numba_times, 'd-', label='numba方法')
    
    plt.xlabel('数据量')
    plt.ylabel('执行时间 (秒)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_speedup_comparison(results_dict: dict, title: str, save_path: str = None):
    """
    绘制性能提升倍数比较图
    
    参数:
        results_dict: 结果字典
        title: 图表标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 8))
    
    # 提取数据
    x_values = list(results_dict.keys())
    pandas_speedups = [results_dict[x]['pandas_rolling']['speedup'] for x in x_values]
    numpy_speedups = [results_dict[x]['numpy']['speedup'] for x in x_values]
    numba_speedups = [results_dict[x]['numba']['speedup'] for x in x_values]
    
    # 绘制图表
    plt.plot(x_values, pandas_speedups, 's-', label='pandas rolling方法')
    plt.plot(x_values, numpy_speedups, '^-', label='numpy方法')
    plt.plot(x_values, numba_speedups, 'd-', label='numba方法')
    
    plt.xlabel('数据量')
    plt.ylabel('性能提升倍数')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def run_comprehensive_test():
    """运行综合性能测试"""
    print("开始综合性能测试...")
    print("=" * 60)
    
    # 测试不同数据量
    size_results = test_different_data_sizes()
    
    # 测试不同缺失值比例
    missing_results = test_different_missing_ratios()
    
    # 测试不同相关系数方法
    corr_results = test_correlation_methods()
    
    # 绘制图表
    try:
        plot_performance_comparison(size_results, "不同数据量下的性能比较")
        plot_speedup_comparison(size_results, "不同数据量下的性能提升倍数")
    except Exception as e:
        print(f"绘图时出错: {e}")
    
    return size_results, missing_results, corr_results


def analyze_results(size_results: dict, missing_results: dict, corr_results: dict):
    """分析测试结果"""
    print("\n\n测试结果分析:")
    print("=" * 60)
    
    # 分析数据量对性能的影响
    print("\n1. 数据量对性能的影响:")
    sizes = list(size_results.keys())
    pandas_speedups = [size_results[size]['pandas_rolling']['speedup'] for size in sizes]
    numpy_speedups = [size_results[size]['numpy']['speedup'] for size in sizes]
    numba_speedups = [size_results[size]['numba']['speedup'] for size in sizes]
    
    print(f"   pandas rolling方法平均性能提升: {np.mean(pandas_speedups):.2f}x")
    print(f"   numpy方法平均性能提升: {np.mean(numpy_speedups):.2f}x")
    print(f"   numba方法平均性能提升: {np.mean(numba_speedups):.2f}x")
    
    # 分析缺失值对性能的影响
    print("\n2. 缺失值对性能的影响:")
    ratios = list(missing_results.keys())
    pandas_speedups = [missing_results[ratio]['pandas_rolling']['speedup'] for ratio in ratios]
    numpy_speedups = [missing_results[ratio]['numpy']['speedup'] for ratio in ratios]
    numba_speedups = [missing_results[ratio]['numba']['speedup'] for ratio in ratios]
    
    print(f"   pandas rolling方法平均性能提升: {np.mean(pandas_speedups):.2f}x")
    print(f"   numpy方法平均性能提升: {np.mean(numpy_speedups):.2f}x")
    print(f"   numba方法平均性能提升: {np.mean(numba_speedups):.2f}x")
    
    # 分析相关系数方法对性能的影响
    print("\n3. 相关系数方法对性能的影响:")
    pearson_speedups = [corr_results['Pearson']['numpy']['speedup'], corr_results['Pearson']['numba']['speedup']]
    spearman_speedups = [corr_results['Spearman']['numpy']['speedup'], corr_results['Spearman']['numba']['speedup']]
    
    print(f"   Pearson相关系数: numpy {corr_results['Pearson']['numpy']['speedup']:.2f}x, numba {corr_results['Pearson']['numba']['speedup']:.2f}x")
    print(f"   Spearman相关系数: numpy {corr_results['Spearman']['numpy']['speedup']:.2f}x, numba {corr_results['Spearman']['numba']['speedup']:.2f}x")
    
    # 给出建议
    print("\n4. 优化建议:")
    best_method = None
    best_speedup = 0
    
    for size in sizes:
        for method in ['pandas_rolling', 'numpy', 'numba']:
            speedup = size_results[size][method]['speedup']
            if speedup > best_speedup:
                best_speedup = speedup
                best_method = method
    
    print(f"   最佳优化方法: {best_method} (最高性能提升: {best_speedup:.2f}x)")
    
    if best_method == 'numpy':
        print("   建议: 对于大规模数据，使用numpy向量化操作可获得最佳性能")
    elif best_method == 'numba':
        print("   建议: 对于计算密集型任务，使用Numba JIT编译可获得最佳性能")
    elif best_method == 'pandas_rolling':
        print("   建议: 对于中等规模数据，pandas rolling方法提供了良好的性能和易用性平衡")


if __name__ == "__main__":
    # 运行综合测试
    size_results, missing_results, corr_results = run_comprehensive_test()
    
    # 分析结果
    analyze_results(size_results, missing_results, corr_results)