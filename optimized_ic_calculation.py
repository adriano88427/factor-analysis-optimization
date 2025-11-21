"""
优化的IC计算实现
基于yinzifenxi1119.py中的calculate_ic方法进行性能优化
"""

import numpy as np
import pandas as pd
from numba import jit
from typing import Tuple, Optional


class OptimizedICCalculator:
    """优化的IC计算器类"""
    
    def __init__(self, data: pd.DataFrame, factor_col: str, return_col: str):
        """
        初始化IC计算器
        
        参数:
            data: 包含因子和收益数据的DataFrame
            factor_col: 因子列名
            return_col: 收益列名
        """
        self.data = data
        self.factor_col = factor_col
        self.return_col = return_col
        self.processed_data = None
        
    def set_processed_data(self, processed_data: pd.DataFrame):
        """设置预处理后的数据"""
        self.processed_data = processed_data
        
    def _get_clean_data(self) -> pd.DataFrame:
        """获取清理后的数据"""
        if self.processed_data is not None:
            return self.processed_data.dropna(subset=[self.factor_col, self.return_col])
        else:
            return self.data.dropna(subset=[self.factor_col, self.return_col])
    
    def calculate_ic_original(self, use_pearson: bool = False) -> Tuple[float, float, float, float]:
        """
        原始IC计算方法（来自yinzifenxi1119.py）
        
        参数:
            use_pearson: 是否使用Pearson相关系数，False则使用Spearman
            
        返回:
            (ic_mean, ic_std, t_stat, p_value)
        """
        df_clean = self._get_clean_data()
        
        if len(df_clean) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        try:
            # 计算相关系数
            if use_pearson:
                ic = df_clean[self.factor_col].corr(df_clean[self.return_col])
            else:
                ic = df_clean[self.factor_col].corr(df_clean[self.return_col], method='spearman')
            
            # 计算IC的均值和标准差（使用滚动窗口）
            window_size = min(30, len(df_clean) // 3)
            if window_size < 5:
                return ic, np.nan, np.nan, np.nan
            
            rolling_ic = []
            for i in range(window_size, len(df_clean)):
                subset = df_clean.iloc[i-window_size:i]
                if use_pearson:
                    corr = subset[self.factor_col].corr(subset[self.return_col])
                else:
                    corr = subset[self.factor_col].corr(subset[self.return_col], method='spearman')
                if not np.isnan(corr):
                    rolling_ic.append(corr)
            
            if len(rolling_ic) < 2:
                return ic, np.nan, np.nan, np.nan
            
            ic_mean = np.mean(rolling_ic)
            ic_std = np.std(rolling_ic)
            
            # 计算t统计量和p值
            if ic_std > 0:
                t_stat = ic_mean / (ic_std / np.sqrt(len(rolling_ic)))
                try:
                    from scipy.stats import t
                    p_value = 2 * (1 - t.cdf(abs(t_stat), len(rolling_ic) - 1))
                except:
                    p_value = np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
            
            return ic_mean, ic_std, t_stat, p_value
        
        except Exception as e:
            print(f"计算IC时出错: {e}")
            return np.nan, np.nan, np.nan, np.nan
    
    def calculate_ic_pandas_rolling(self, use_pearson: bool = False) -> Tuple[float, float, float, float]:
        """
        使用pandas rolling().corr()优化的IC计算方法
        
        参数:
            use_pearson: 是否使用Pearson相关系数，False则使用Spearman
            
        返回:
            (ic_mean, ic_std, t_stat, p_value)
        """
        df_clean = self._get_clean_data()
        
        if len(df_clean) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        try:
            # 计算整体相关系数
            if use_pearson:
                ic = df_clean[self.factor_col].corr(df_clean[self.return_col])
            else:
                ic = df_clean[self.factor_col].corr(df_clean[self.return_col], method='spearman')
            
            # 计算滚动IC（使用向量化操作）
            window_size = min(30, len(df_clean) // 3)
            if window_size < 5:
                return ic, np.nan, np.nan, np.nan
            
            # 使用pandas的rolling().corr()进行向量化计算
            if use_pearson:
                rolling_ic = df_clean[self.factor_col].rolling(window=window_size).corr(df_clean[self.return_col])
            else:
                # 对于Spearman相关系数，需要自定义函数
                rolling_ic = df_clean[self.factor_col].rolling(window=window_size).apply(
                    lambda x: x.corr(df_clean[self.return_col].loc[x.index], method='spearman')
                )
            
            # 过滤NaN值
            rolling_ic = rolling_ic.dropna()
            
            if len(rolling_ic) < 2:
                return ic, np.nan, np.nan, np.nan
            
            ic_mean = rolling_ic.mean()
            ic_std = rolling_ic.std()
            
            # 计算t统计量和p值
            if ic_std > 0:
                t_stat = ic_mean / (ic_std / np.sqrt(len(rolling_ic)))
                try:
                    from scipy.stats import t
                    p_value = 2 * (1 - t.cdf(abs(t_stat), len(rolling_ic) - 1))
                except:
                    p_value = np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
            
            return ic_mean, ic_std, t_stat, p_value
        
        except Exception as e:
            print(f"计算IC时出错: {e}")
            return np.nan, np.nan, np.nan, np.nan
    
    def calculate_ic_numpy(self, use_pearson: bool = False) -> Tuple[float, float, float, float]:
        """
        使用numpy向量化操作优化的IC计算方法
        
        参数:
            use_pearson: 是否使用Pearson相关系数，False则使用Spearman
            
        返回:
            (ic_mean, ic_std, t_stat, p_value)
        """
        df_clean = self._get_clean_data()
        
        if len(df_clean) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        try:
            # 转换为numpy数组
            factor_values = df_clean[self.factor_col].values
            return_values = df_clean[self.return_col].values
            
            # 计算整体相关系数
            if use_pearson:
                ic = np.corrcoef(factor_values, return_values)[0, 1]
            else:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_values, return_values)
            
            # 计算滚动IC
            window_size = min(30, len(df_clean) // 3)
            if window_size < 5:
                return ic, np.nan, np.nan, np.nan
            
            # 使用numpy的stride_tricks进行高效滚动窗口计算
            from numpy.lib.stride_tricks import sliding_window_view
            
            # 创建滑动窗口视图
            factor_windows = sliding_window_view(factor_values, window_shape=window_size)
            return_windows = sliding_window_view(return_values, window_shape=window_size)
            
            # 向量化计算每个窗口的相关系数
            if use_pearson:
                # 计算Pearson相关系数
                factor_means = np.mean(factor_windows, axis=1)
                return_means = np.mean(return_windows, axis=1)
                
                factor_centered = factor_windows - factor_means[:, np.newaxis]
                return_centered = return_windows - return_means[:, np.newaxis]
                
                numerator = np.sum(factor_centered * return_centered, axis=1)
                factor_std = np.sqrt(np.sum(factor_centered**2, axis=1))
                return_std = np.sqrt(np.sum(return_centered**2, axis=1))
                
                rolling_ic = numerator / (factor_std * return_std)
            else:
                # 对于Spearman相关系数，使用循环（因为向量化较复杂）
                from scipy.stats import spearmanr
                rolling_ic = np.array([
                    spearmanr(factor_windows[i], return_windows[i])[0]
                    for i in range(len(factor_windows))
                ])
            
            # 过滤NaN值
            valid_ic = rolling_ic[~np.isnan(rolling_ic)]
            
            if len(valid_ic) < 2:
                return ic, np.nan, np.nan, np.nan
            
            ic_mean = np.mean(valid_ic)
            ic_std = np.std(valid_ic)
            
            # 计算t统计量和p值
            if ic_std > 0:
                t_stat = ic_mean / (ic_std / np.sqrt(len(valid_ic)))
                try:
                    from scipy.stats import t
                    p_value = 2 * (1 - t.cdf(abs(t_stat), len(valid_ic) - 1))
                except:
                    p_value = np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
            
            return ic_mean, ic_std, t_stat, p_value
        
        except Exception as e:
            print(f"计算IC时出错: {e}")
            return np.nan, np.nan, np.nan, np.nan
    
    def calculate_ic_numba(self, use_pearson: bool = False) -> Tuple[float, float, float, float]:
        """
        使用Numba JIT编译优化的IC计算方法
        
        参数:
            use_pearson: 是否使用Pearson相关系数，False则使用Spearman
            
        返回:
            (ic_mean, ic_std, t_stat, p_value)
        """
        df_clean = self._get_clean_data()
        
        if len(df_clean) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        try:
            # 转换为numpy数组
            factor_values = df_clean[self.factor_col].values
            return_values = df_clean[self.return_col].values
            
            # 计算整体相关系数
            if use_pearson:
                ic = np.corrcoef(factor_values, return_values)[0, 1]
            else:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(factor_values, return_values)
            
            # 计算滚动IC
            window_size = min(30, len(df_clean) // 3)
            if window_size < 5:
                return ic, np.nan, np.nan, np.nan
            
            # 使用Numba优化的函数
            if use_pearson:
                rolling_ic = _calculate_rolling_correlation_numba(factor_values, return_values, window_size)
            else:
                # 对于Spearman相关系数，仍使用原始方法
                rolling_ic = []
                for i in range(window_size, len(df_clean)):
                    subset = df_clean.iloc[i-window_size:i]
                    corr = subset[self.factor_col].corr(subset[self.return_col], method='spearman')
                    if not np.isnan(corr):
                        rolling_ic.append(corr)
                rolling_ic = np.array(rolling_ic)
            
            # 过滤NaN值
            valid_ic = rolling_ic[~np.isnan(rolling_ic)]
            
            if len(valid_ic) < 2:
                return ic, np.nan, np.nan, np.nan
            
            ic_mean = np.mean(valid_ic)
            ic_std = np.std(valid_ic)
            
            # 计算t统计量和p值
            if ic_std > 0:
                t_stat = ic_mean / (ic_std / np.sqrt(len(valid_ic)))
                try:
                    from scipy.stats import t
                    p_value = 2 * (1 - t.cdf(abs(t_stat), len(valid_ic) - 1))
                except:
                    p_value = np.nan
            else:
                t_stat = np.nan
                p_value = np.nan
            
            return ic_mean, ic_std, t_stat, p_value
        
        except Exception as e:
            print(f"计算IC时出错: {e}")
            return np.nan, np.nan, np.nan, np.nan


@jit(nopython=True)
def _calculate_rolling_correlation_numba(factor_values: np.ndarray, return_values: np.ndarray, 
                                        window_size: int) -> np.ndarray:
    """
    使用Numba优化的滚动相关系数计算
    
    参数:
        factor_values: 因子值数组
        return_values: 收益值数组
        window_size: 滚动窗口大小
        
    返回:
        滚动相关系数数组
    """
    n = len(factor_values)
    rolling_ic = np.empty(n - window_size + 1)
    
    for i in range(window_size - 1, n):
        start_idx = i - window_size + 1
        end_idx = i + 1
        
        # 提取窗口数据
        factor_window = factor_values[start_idx:end_idx]
        return_window = return_values[start_idx:end_idx]
        
        # 计算均值
        factor_mean = np.mean(factor_window)
        return_mean = np.mean(return_window)
        
        # 计算协方差和方差
        covariance = np.sum((factor_window - factor_mean) * (return_window - return_mean))
        factor_var = np.sum((factor_window - factor_mean) ** 2)
        return_var = np.sum((return_window - return_mean) ** 2)
        
        # 计算相关系数
        if factor_var > 0 and return_var > 0:
            correlation = covariance / np.sqrt(factor_var * return_var)
            rolling_ic[start_idx] = correlation
        else:
            rolling_ic[start_idx] = np.nan
    
    return rolling_ic


def benchmark_ic_methods(data: pd.DataFrame, factor_col: str, return_col: str, 
                        use_pearson: bool = False) -> dict:
    """
    测试不同IC计算方法的性能
    
    参数:
        data: 包含因子和收益数据的DataFrame
        factor_col: 因子列名
        return_col: 收益列名
        use_pearson: 是否使用Pearson相关系数，False则使用Spearman
        
    返回:
        包含各方法性能和结果的字典
    """
    import time
    
    # 创建IC计算器实例
    calculator = OptimizedICCalculator(data, factor_col, return_col)
    
    results = {}
    
    # 测试原始方法
    start_time = time.time()
    result_original = calculator.calculate_ic_original(use_pearson)
    original_time = time.time() - start_time
    results['original'] = {'result': result_original, 'time': original_time}
    
    # 测试pandas rolling方法
    start_time = time.time()
    result_pandas = calculator.calculate_ic_pandas_rolling(use_pearson)
    pandas_time = time.time() - start_time
    results['pandas_rolling'] = {'result': result_pandas, 'time': pandas_time}
    
    # 测试numpy方法
    start_time = time.time()
    result_numpy = calculator.calculate_ic_numpy(use_pearson)
    numpy_time = time.time() - start_time
    results['numpy'] = {'result': result_numpy, 'time': numpy_time}
    
    # 测试numba方法
    start_time = time.time()
    result_numba = calculator.calculate_ic_numba(use_pearson)
    numba_time = time.time() - start_time
    results['numba'] = {'result': result_numba, 'time': numba_time}
    
    # 计算性能提升
    baseline_time = original_time
    for method in ['pandas_rolling', 'numpy', 'numba']:
        if results[method]['time'] > 0:
            results[method]['speedup'] = baseline_time / results[method]['time']
        else:
            results[method]['speedup'] = float('inf')
    
    return results


def print_benchmark_results(results: dict):
    """打印性能测试结果"""
    print("性能测试结果:")
    print(f"原始方法: {results['original']['time']:.4f}秒")
    print(f"pandas rolling方法: {results['pandas_rolling']['time']:.4f}秒 (提升: {results['pandas_rolling']['speedup']:.2f}x)")
    print(f"numpy方法: {results['numpy']['time']:.4f}秒 (提升: {results['numpy']['speedup']:.2f}x)")
    print(f"numba方法: {results['numba']['time']:.4f}秒 (提升: {results['numba']['speedup']:.2f}x)")
    
    print("\n结果一致性检查:")
    print(f"原始方法: {results['original']['result']}")
    print(f"pandas rolling方法: {results['pandas_rolling']['result']}")
    print(f"numpy方法: {results['numpy']['result']}")
    print(f"numba方法: {results['numba']['result']}")


if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    n_samples = 10000
    factor_data = np.random.randn(n_samples)
    return_data = np.random.randn(n_samples)
    
    df = pd.DataFrame({
        'factor': factor_data,
        'return': return_data
    })
    
    # 运行性能测试
    results = benchmark_ic_methods(df, 'factor', 'return', use_pearson=True)
    print_benchmark_results(results)