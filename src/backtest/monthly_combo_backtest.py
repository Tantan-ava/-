# -*- coding: utf-8 -*-
"""
月度组合策略回测引擎

策略配置：
- EP价值因子权重 0.4（或优化后的权重）
- K=6反转因子权重 0.6
- 月度再平衡（每月第1个交易日）
- 等权配置选中的股票

与hybrid_backtester的区别：
- 纯月度调仓，无日度调仓
- 无情绪因子参与
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MonthlyComboBacktester:
    """
    月度组合策略回测引擎
    
    策略逻辑：
    - 每月第1个交易日执行选股
    - 价值因子(EP) + 反转因子(K=6) 合成得分
    - 等权持有Top K只股票一个月
    - 月度收益计算
    """
    
    def __init__(self,
                 monthly_returns_path='data/processed/TRD_Mnth.xlsx',
                 start_date='2000-01-01',
                 end_date='2025-12-31',
                 ep_weight=0.4,
                 reversal_weight=0.6,
                 top_k=100,
                 winsorize=True):
        """
        初始化回测引擎
        
        Parameters:
        -----------
        monthly_returns_path : str
            月度收益率数据路径
        start_date : str
            回测开始日期
        end_date : str
            回测结束日期
        ep_weight : float
            EP价值因子权重
        reversal_weight : float
            反转因子权重
        top_k : int
            选股数量
        winsorize : bool
            是否使用Winsorization标准化
        """
        self.ep_weight = ep_weight
        self.reversal_weight = reversal_weight
        self.top_k = top_k
        self.winsorize = winsorize
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        
        # 加载数据
        self._load_data(monthly_returns_path)
        
        # 初始化持仓
        self.current_portfolio = None
        self.records = []
    
    def _load_data(self, filepath):
        """加载月度收益率数据"""
        print(f"加载月度收益率数据: {filepath}")
        
        try:
            # 尝试加载Excel文件
            df = pd.read_excel(filepath, index_col=0, engine='openpyxl')
            df.index = pd.to_datetime(df.index)
            
            # 过滤日期范围
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            self.monthly_returns = df
            self.trading_months = df.index
            
            print(f"  加载完成: {len(df)} 个月, {len(df.columns)} 只股票")
            print(f"  日期范围: {df.index.min()} 至 {df.index.max()}")
            
        except Exception as e:
            print(f"  加载失败: {e}")
            # 使用CSV格式尝试
            try:
                csv_path = 'data/raw/TRD-daily.csv'
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                # 转换为月度收益
                df_monthly = df.resample('M').apply(lambda x: (1 + x).prod() - 1)
                self.monthly_returns = df_monthly
                self.trading_months = df_monthly.index
                print(f"  从日度数据转换: {len(df_monthly)} 个月")
            except Exception as e2:
                print(f"  CSV加载也失败: {e2}")
                raise
    
    def _calculate_reversal_signal(self, date_idx, k=6):
        """
        计算K期反转信号
        
        Signal = -Return_{t-k:t-1} (过去k个月累积收益的负值)
        """
        if date_idx < k:
            # 数据不足，返回零信号
            return pd.Series(0, index=self.monthly_returns.columns)
        
        # 计算k期累积收益
        period_returns = self.monthly_returns.iloc[date_idx-k:date_idx]
        cumulative_return = (1 + period_returns).prod() - 1
        
        # 反转信号 = -累积收益
        reversal_signal = -cumulative_return
        
        return reversal_signal.dropna()
    
    def _calculate_ep_signal(self, date_idx):
        """
        计算价值因子信号

        从价值因子数据文件加载
        """
        if date_idx < 12:
            return pd.Series(0, index=self.monthly_returns.columns)

        # 加载价值因子数据
        vf_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'value_factor.parquet')
        if os.path.exists(vf_path):
            vf = pd.read_parquet(vf_path)
            vf.index = pd.to_datetime(vf.index)
            current_date = self.monthly_returns.index[date_idx]
            if current_date in vf.index:
                return vf.loc[current_date].dropna()
            # 找最近的日期
            valid_dates = vf.index[vf.index <= current_date]
            if len(valid_dates) > 0:
                return vf.loc[valid_dates[-1]].dropna()

        # 回退：使用过去12个月收益均值
        period_returns = self.monthly_returns.iloc[date_idx-12:date_idx]
        return period_returns.mean().dropna()
    
    def _standardize(self, signal):
        """Winsorization + Z-score标准化"""
        if self.winsorize:
            lower = signal.quantile(0.05)
            upper = signal.quantile(0.95)
            signal = signal.clip(lower, upper)
        
        mean = signal.mean()
        std = signal.std()
        if std > 0:
            return (signal - mean) / std
        return signal - mean
    
    def _select_stocks(self, date_idx):
        """
        选股逻辑
        
        合成得分 = EP权重 * EP信号 + 反转权重 * 反转信号
        """
        # 计算信号
        reversal_signal = self._calculate_reversal_signal(date_idx, k=6)
        ep_signal = self._calculate_ep_signal(date_idx)
        
        # 对齐股票
        common_stocks = reversal_signal.index.intersection(ep_signal.index)
        
        if len(common_stocks) == 0:
            print(f"  警告: 无共同股票，仅使用反转因子")
            reversal_signal = reversal_signal.dropna()
            if len(reversal_signal) == 0:
                return pd.Series()
            selected = reversal_signal.nlargest(self.top_k)
        else:
            reversal_signal = reversal_signal.loc[common_stocks]
            ep_signal = ep_signal.loc[common_stocks]
            
            # 标准化
            reversal_z = self._standardize(reversal_signal)
            ep_z = self._standardize(ep_signal)
            
            # 合成得分
            composite_score = (
                self.ep_weight * ep_z +
                self.reversal_weight * reversal_z
            )
            
            # 选Top K
            selected = composite_score.nlargest(self.top_k)
        
        # 等权重
        weights = pd.Series(1.0 / len(selected), index=selected.index)
        
        return weights
    
    def run(self):
        """
        执行月度回测
        
        Returns:
        --------
        pd.DataFrame
            月度回测结果
        """
        print("\n" + "="*60)
        print("月度组合策略回测")
        print("="*60)
        print(f"配置: EP={self.ep_weight:.0%}, 反转={self.reversal_weight:.0%}, TopK={self.top_k}")
        print(f"期间: {self.start_date.date()} 至 {self.end_date.date()}")
        
        portfolio_value = 1.0  # 初始净值
        current_weights = None
        
        # 从第7个月开始（需要6个月历史数据计算反转）
        start_idx = 6
        
        for i in range(start_idx, len(self.trading_months)):
            date = self.trading_months[i]
            
            # 每月选股
            current_weights = self._select_stocks(i)
            
            if len(current_weights) == 0:
                print(f"  警告: {date} 未选出股票")
                monthly_return = 0
            else:
                # 计算当月收益
                month_return = self.monthly_returns.iloc[i]
                
                # 对齐股票
                available_stocks = month_return.index.intersection(current_weights.index)
                if len(available_stocks) == 0:
                    monthly_return = 0
                else:
                    weights = current_weights.loc[available_stocks]
                    weights = weights / weights.sum()  # 归一化
                    returns = month_return.loc[available_stocks]
                    
                    # 计算组合收益
                    monthly_return = (weights * returns).sum()
            
            # 更新净值
            portfolio_value *= (1 + monthly_return)
            
            # 记录
            self.records.append({
                'date': date,
                'monthly_return': monthly_return,
                'nav': portfolio_value,
                'num_stocks': len(current_weights) if current_weights is not None else 0
            })
            
            # 打印进度
            if i % 50 == 0 or i == len(self.trading_months) - 1:
                print(f"  [{date.strftime('%Y-%m')}] NAV: {portfolio_value:.4f} ({monthly_return*100:+.2f}%)")
        
        # 生成结果
        results = pd.DataFrame(self.records)
        results['date'] = pd.to_datetime(results['date'])
        results['year'] = results['date'].dt.year
        
        self.results = results
        self._print_summary()
        
        return results
    
    def _print_summary(self):
        """打印回测摘要"""
        if len(self.results) == 0:
            return
        
        ret = self.results['monthly_return']
        
        # 年化收益
        n_months = len(ret)
        total_return = (1 + ret).prod() - 1
        ann_ret = (1 + total_return) ** (12 / n_months) - 1
        
        # 年化波动
        ann_vol = ret.std() * np.sqrt(12)
        
        # 夏普比率
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        cum_ret = self.results['nav']
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        # 卡玛比率
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        print("\n" + "="*60)
        print("回测结果摘要")
        print("="*60)
        print(f"回测月数: {n_months}")
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益率: {ann_ret*100:.2f}%")
        print(f"年化波动率: {ann_vol*100:.2f}%")
        print(f"夏普比率: {sharpe:.3f}")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"卡玛比率: {calmar:.3f}")
        print("="*60)
    
    def save_results(self, filepath='results/tables/monthly_combo_returns.csv'):
        """保存回测结果"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 添加不同成本假设下的收益
        self.results['return_0cost'] = self.results['monthly_return']
        self.results['return_001'] = self.results['monthly_return'] - 0.001
        self.results['return_002'] = self.results['monthly_return'] - 0.002
        
        self.results.to_csv(filepath, index=False)
        print(f"\n结果已保存: {filepath}")


def main():
    """主函数"""
    print("="*60)
    print("月度组合策略 (价值+反转)")
    print("="*60)
    
    # 运行回测
    backtester = MonthlyComboBacktester(
        monthly_returns_path='data/processed/TRD_Mnth.xlsx',
        start_date='2000-01-01',
        end_date='2025-12-31',
        ep_weight=0.4,
        reversal_weight=0.6,
        top_k=100
    )
    
    results = backtester.run()
    backtester.save_results()
    
    print("\n回测完成!")


if __name__ == '__main__':
    main()
