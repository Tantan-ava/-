# -*- coding: utf-8 -*-
"""
政策冲击策略
基于Truth Social情绪指数生成交易信号
与现有strategy_returns.csv格式兼容
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.sentiment_analyzer import PolicySentimentAnalyzer, generate_sentiment_index


class PolicyShockStrategy:
    """
    政策冲击策略
    
    基于情绪指数生成交易信号：
    - 严重负面冲击(-2): 做空信号
    - 中度负面冲击(-1): 减仓信号
    - 正常(0): 持仓不变
    - 中度正面冲击(1): 加仓信号
    - 严重正面冲击(2): 满仓信号
    """
    
    def __init__(self, 
                 sentiment_data_path: str = None,
                 shock_threshold: float = -1.5,
                 holding_period: int = 5,
                 position_size: float = 0.5):
        """
        初始化策略
        
        Parameters
        ----------
        sentiment_data_path : str, optional
            情绪指数CSV路径，如果为None则重新生成
        shock_threshold : float
            冲击信号阈值，低于此值触发交易
        holding_period : int
            持仓周期（天数）
        position_size : float
            单次交易仓位大小 (0-1)
        """
        self.sentiment_data_path = sentiment_data_path
        self.shock_threshold = shock_threshold
        self.holding_period = holding_period
        self.position_size = position_size
        
        self.sentiment_df = None
        self.signals_df = None
        
    def load_sentiment_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        加载情绪数据
        
        Parameters
        ----------
        start_date : str, optional
            开始日期
        end_date : str, optional
            结束日期
        
        Returns
        -------
        pd.DataFrame
            情绪指数数据框
        """
        if self.sentiment_data_path and Path(self.sentiment_data_path).exists():
            print(f"从文件加载情绪数据: {self.sentiment_data_path}")
            df = pd.read_csv(self.sentiment_data_path)
            df['date'] = pd.to_datetime(df['date'])
        else:
            print("重新生成情绪指数...")
            df = generate_sentiment_index(
                start_date=start_date or '2022-02-01',
                end_date=end_date,
                use_mock=True
            )
            # 保存生成的数据
            output_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed' / 'daily_sentiment_index.csv'
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"情绪数据已保存至: {output_path}")
        
        # 日期过滤
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        self.sentiment_df = df.reset_index(drop=True)
        return self.sentiment_df
    
    def generate_signals(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        生成交易信号
        
        输出格式与现有strategy_returns.csv完全一致：
        date, signal(0/1), hedge_ratio, shock_intensity
        
        Parameters
        ----------
        start_date : str, optional
            开始日期 (YYYY-MM-DD)
        end_date : str, optional
            结束日期 (YYYY-MM-DD)
        
        Returns
        -------
        pd.DataFrame
            交易信号数据框
        """
        # 加载情绪数据
        if self.sentiment_df is None:
            self.load_sentiment_data(start_date, end_date)
        
        df = self.sentiment_df.copy()
        
        # 生成信号
        signals = []
        
        for idx, row in df.iterrows():
            date = row['date']
            shock_signal = row['shock_signal']
            sentiment_score = row['sentiment_score']
            vix_proxy = row['vix_proxy']
            
            # 计算冲击强度
            shock_intensity = self._calculate_shock_intensity(sentiment_score, vix_proxy)
            
            # 生成交易信号
            # signal: 0 = 无信号, 1 = 有信号
            # hedge_ratio: 对冲比例 (0-1)
            if shock_signal <= -2:  # 严重负面
                signal = 1
                hedge_ratio = self.position_size  # 建立对冲仓位
            elif shock_signal == -1:  # 中度负面
                signal = 1
                hedge_ratio = self.position_size * 0.5  # 部分对冲
            elif shock_signal >= 1:  # 正面冲击
                signal = 1
                hedge_ratio = 0  # 不 hedge，保持多头
            else:
                signal = 0
                hedge_ratio = 0
            
            signals.append({
                'date': date,
                'signal': signal,
                'hedge_ratio': hedge_ratio,
                'shock_intensity': shock_intensity,
                'sentiment_score': sentiment_score,
                'vix_proxy': vix_proxy,
                'shock_signal': shock_signal
            })
        
        self.signals_df = pd.DataFrame(signals)
        return self.signals_df
    
    def _calculate_shock_intensity(self, sentiment_score: float, vix_proxy: float) -> float:
        """
        计算冲击强度
        
        Parameters
        ----------
        sentiment_score : float
            情绪分数
        vix_proxy : float
            VIX指标（波动率）
        
        Returns
        -------
        float
            冲击强度 (-1 到 1)
        """
        # 结合情绪分数和波动率
        # 负面情绪 + 高波动 = 高冲击强度
        intensity = -sentiment_score * (1 + vix_proxy)
        
        # 限制在 -1 到 1 之间
        intensity = np.clip(intensity, -1, 1)
        
        return round(intensity, 4)
    
    def calculate_strategy_returns(self, 
                                   market_returns: pd.DataFrame = None,
                                   start_date: str = None,
                                   end_date: str = None) -> pd.DataFrame:
        """
        计算策略收益
        
        生成与strategy_returns.csv兼容的格式
        
        Parameters
        ----------
        market_returns : pd.DataFrame, optional
            市场收益数据，包含'date'和'market_return'列
        start_date : str, optional
            开始日期
        end_date : str, optional
            结束日期
        
        Returns
        -------
        pd.DataFrame
            策略收益数据框，列：date, strategy_return, signal, hedge_ratio
        """
        # 生成信号
        if self.signals_df is None:
            self.generate_signals(start_date, end_date)
        
        signals = self.signals_df.copy()
        
        # 如果没有市场收益数据，使用模拟数据
        if market_returns is None:
            market_returns = self._generate_mock_market_returns(signals['date'])
        
        # 合并信号和市场收益
        merged = pd.merge(signals, market_returns, on='date', how='outer')
        merged = merged.sort_values('date').fillna(method='ffill').fillna(0)
        
        # 计算策略收益
        # 策略逻辑：当检测到负面冲击时，对冲部分市场暴露
        # strategy_return = market_return * (1 - hedge_ratio) - market_return * hedge_ratio (做空)
        # 简化：strategy_return = market_return * (1 - 2 * hedge_ratio)
        
        merged['strategy_return'] = merged['market_return'] * (1 - 2 * merged['hedge_ratio'])
        
        # 选择输出列（与现有strategy_returns.csv兼容）
        output = merged[['date', 'strategy_return', 'signal', 'hedge_ratio']].copy()
        
        return output
    
    def _generate_mock_market_returns(self, dates: pd.Series) -> pd.DataFrame:
        """生成模拟市场收益数据"""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, len(dates))  # 日均收益0.05%，波动1.5%
        
        return pd.DataFrame({
            'date': dates,
            'market_return': returns
        })
    
    def save_signals(self, output_path: str = None):
        """
        保存信号到CSV
        
        Parameters
        ----------
        output_path : str, optional
            输出路径，默认为results/tables/policy_shock_signals.csv
        """
        if self.signals_df is None:
            raise ValueError("请先调用generate_signals()生成信号")
        
        if output_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_path = project_root / 'results' / 'tables' / 'policy_shock_signals.csv'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.signals_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"交易信号已保存至: {output_path}")
    
    def get_summary(self) -> Dict:
        """
        获取策略摘要统计
        
        Returns
        -------
        Dict
            策略统计信息
        """
        if self.signals_df is None:
            return {}
        
        df = self.signals_df
        
        return {
            'total_days': len(df),
            'signal_days': (df['signal'] == 1).sum(),
            'signal_ratio': (df['signal'] == 1).mean(),
            'avg_hedge_ratio': df[df['signal'] == 1]['hedge_ratio'].mean(),
            'severe_negative_shocks': (df['shock_signal'] == -2).sum(),
            'moderate_negative_shocks': (df['shock_signal'] == -1).sum(),
            'positive_shocks': (df['shock_signal'] >= 1).sum(),
            'avg_sentiment': df['sentiment_score'].mean(),
            'sentiment_std': df['sentiment_score'].std(),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        }


# 便捷函数
def run_policy_shock_strategy(start_date: str = '2022-02-01',
                              end_date: str = None,
                              save_results: bool = True) -> pd.DataFrame:
    """
    运行政策冲击策略的便捷函数
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str, optional
        结束日期
    save_results : bool
        是否保存结果
    
    Returns
    -------
    pd.DataFrame
        策略信号数据框
    """
    strategy = PolicyShockStrategy()
    signals = strategy.generate_signals(start_date, end_date)
    
    if save_results:
        strategy.save_signals()
    
    # 打印摘要
    summary = strategy.get_summary()
    print("\n策略摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return signals


if __name__ == '__main__':
    # 测试运行
    signals = run_policy_shock_strategy(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    print(f"\n生成的信号预览:")
    print(signals.head(10))
