# -*- coding: utf-8 -*-
"""
日度调仓模块：情绪驱动仓位调节

基于每日情绪指数调整持仓比例
每日开盘前执行，根据隔夜情绪调整仓位
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


class DailyTactician:
    """
    日度调仓：情绪驱动仓位调节
    
    基于daily_sentiment_index.csv中的情绪数据：
    - sentiment_score: 情绪分数
    - shock_signal: 冲击信号 (-2, 0, 2)
    
    每日开盘前执行，根据情绪调整持仓比例
    """
    
    def __init__(self, sentiment_data, use_shock_signal=True):
        """
        初始化日度调仓器
        
        Parameters:
        -----------
        sentiment_data : pd.DataFrame
            情绪数据，index=date, columns包含['sentiment_score', 'shock_signal']
        use_shock_signal : bool
            是否使用冲击信号作为辅助判断，默认True
        """
        self.sentiment = sentiment_data.copy()
        self.use_shock_signal = use_shock_signal
        
        # 计算情绪分数的Z-score（使用扩展窗口+shift(1)严格避免前视偏差）
        score = self.sentiment['sentiment_score']
        self.sentiment['sentiment_zscore'] = (
            (score - score.expanding().mean().shift(1)) / score.expanding().std().shift(1)
        )
        
        # 阈值校准（基于情绪历史分布，使用扩展窗口+shift(1)严格避免前视偏差）
        self.thresholds = self._calibrate_thresholds()
        
    def _calibrate_thresholds(self):
        """
        基于情绪历史分布自动校准阈值
        
        优化说明：
        - 原阈值基于标准正态分布理论值（±1, ±2）
        - 实际情绪数据分布不同，需要根据实际数据调整
        - 使用扩展窗口百分位数+shift(1)严格避免前视偏差
        
        Returns:
        --------
        pd.DataFrame: 阈值配置，index=date, columns=阈值名称
        """
        z = self.sentiment['sentiment_zscore']
        
        # 使用扩展窗口百分位数+shift(1)，确保每个时点只使用历史数据（不含当日）
        thresholds = pd.DataFrame(index=z.index)
        thresholds['extreme_panic'] = z.expanding().quantile(0.05).shift(1)
        thresholds['panic'] = z.expanding().quantile(0.20).shift(1)
        thresholds['neutral_low'] = z.expanding().quantile(0.40).shift(1)
        thresholds['neutral_high'] = z.expanding().quantile(0.60).shift(1)
        thresholds['optimistic'] = z.expanding().quantile(0.80).shift(1)
        thresholds['extreme_greed'] = z.expanding().quantile(0.95).shift(1)
        return thresholds
    
    def get_position_scalar(self, date):
        """
        获取指定日期的仓位系数
        
        Parameters:
        -----------
        date : str or pd.Timestamp
            交易日期
            
        Returns:
        --------
        scalar : float
            仓位系数 (0.5, 0.8, 1.0, 1.2, 1.5)
        signal_type : str
            决策原因
        sentiment_score : float
            当日情绪分数
        """
        # 转换日期格式
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        # 获取当日情绪数据
        try:
            row = self.sentiment.loc[date]
            sentiment_z = row['sentiment_zscore']
            sentiment_score = row['sentiment_score']
            shock = row.get('shock_signal', 0)
        except KeyError:
            # 无情绪数据（周末/节假日），返回中性
            return 1.0, 'no_data', 0.0
        
        # 使用当日对应的扩展窗口阈值（避免前视偏差）
        t = self.thresholds.loc[date]
        
        # 结合冲击信号调整
        shock_adjustment = 0
        if self.use_shock_signal and shock != 0:
            # 冲击信号放大情绪影响
            shock_adjustment = shock * 0.3  # -0.6, 0, +0.6
        
        adjusted_z = sentiment_z + shock_adjustment
        
        # 如果阈值尚未计算完成（历史数据不足），返回中性持仓
        if pd.isna(t['extreme_panic']) or pd.isna(t['extreme_greed']):
            return 1.0, 'insufficient_history', sentiment_score
        
        # 仓位决策逻辑
        if adjusted_z < t['extreme_panic']:
            # 极度恐慌：超配150%（杠杆做多）
            scalar = 1.5
            signal_type = 'extreme_panic_leverage'
            
        elif adjusted_z < t['panic']:
            # 恐慌：超配120%
            scalar = 1.2
            signal_type = 'panic_overweight'
            
        elif adjusted_z < t['neutral_low']:
            # 轻度担忧：低配80%
            scalar = 0.8
            signal_type = 'concern_underweight'
            
        elif t['neutral_low'] <= adjusted_z <= t['neutral_high']:
            # 中性：标配100%
            scalar = 1.0
            signal_type = 'neutral_hold'
            
        elif adjusted_z <= t['optimistic']:
            # 轻度乐观：标配100%
            scalar = 1.0
            signal_type = 'optimistic_hold'
            
        elif adjusted_z <= t['extreme_greed']:
            # 乐观：减仓80%
            scalar = 0.8
            signal_type = 'optimistic_reduce'
            
        else:
            # 极度乐观：大幅减仓50%
            scalar = 0.5
            signal_type = 'extreme_greed_reduce'
        
        return scalar, signal_type, sentiment_score
    
    def adjust_weights(self, base_weights, date):
        """
        应用仓位系数到基础权重
        
        Parameters:
        -----------
        base_weights : pd.Series
            基础权重（来自月度选股）
        date : str or pd.Timestamp
            交易日期
            
        Returns:
        --------
        target_weights : pd.Series
            调整后权重
        scalar : float
            仓位系数
        signal_type : str
            信号类型
        sentiment_score : float
            情绪分数
        """
        scalar, signal_type, sentiment_score = self.get_position_scalar(date)
        
        # 应用系数到基础权重
        target_weights = base_weights * scalar
        
        # 处理现金仓位
        if scalar < 1.0:
            # 减仓：剩余资金转现金
            cash_weight = 1.0 - target_weights.sum()
            target_weights = target_weights.copy()
            target_weights['cash'] = max(0, cash_weight)
        elif scalar > 1.0:
            # 超配：归一化到100%，记录杠杆意图
            total_weight = target_weights.sum()
            if total_weight > 0:
                target_weights = target_weights / total_weight
            target_weights = target_weights.copy()
            target_weights['cash'] = 0.0
            target_weights['leverage_intent'] = scalar - 1.0  # 记录杠杆比例
        else:
            # 标配：无现金
            target_weights = target_weights.copy()
            target_weights['cash'] = 0.0
        
        return target_weights, scalar, signal_type, sentiment_score
    
    def generate_daily_signals(self, start_date=None, end_date=None):
        """
        生成一段时间内的日度信号序列
        
        Parameters:
        -----------
        start_date : str, optional
            开始日期
        end_date : str, optional
            结束日期
            
        Returns:
        --------
        pd.DataFrame
            日度信号数据
        """
        df = self.sentiment.copy()
        
        # 过滤日期范围
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        signals = []
        for date in df.index:
            scalar, signal_type, sentiment_score = self.get_position_scalar(date)
            signals.append({
                'date': date,
                'sentiment_score': sentiment_score,
                'sentiment_zscore': df.loc[date, 'sentiment_zscore'],
                'shock_signal': df.loc[date].get('shock_signal', 0),
                'position_scalar': scalar,
                'signal_type': signal_type
            })
        
        return pd.DataFrame(signals).set_index('date')
    
    def get_signal_statistics(self):
        """
        获取信号统计信息
        
        Returns:
        --------
        dict
            信号统计
        """
        signals = self.generate_daily_signals()
        
        stats = {
            'total_days': len(signals),
            'extreme_panic_days': len(signals[signals['signal_type'] == 'extreme_panic_leverage']),
            'panic_days': len(signals[signals['signal_type'] == 'panic_overweight']),
            'neutral_days': len(signals[signals['signal_type'] == 'neutral_hold']),
            'optimistic_days': len(signals[signals['signal_type'] == 'optimistic_reduce']),
            'extreme_greed_days': len(signals[signals['signal_type'] == 'extreme_greed_reduce']),
            'avg_position_scalar': signals['position_scalar'].mean(),
            'sentiment_mean': signals['sentiment_score'].mean(),
            'sentiment_std': signals['sentiment_score'].std(),
        }
        
        return stats


def load_sentiment_data(filepath='data/processed/daily_sentiment_index.csv'):
    """
    加载日度情绪指数数据
    
    Parameters:
    -----------
    filepath : str
        情绪数据文件路径
        
    Returns:
    --------
    pd.DataFrame
        情绪数据，index=date
    """
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"成功加载情绪数据: {len(df)} 天")
        print(f"数据范围: {df.index[0].date()} 至 {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"加载情绪数据失败: {e}")
        return None


# 使用示例
if __name__ == '__main__':
    # 加载情绪数据
    sentiment_data = load_sentiment_data()
    
    if sentiment_data is not None:
        # 初始化日度调仓器
        tactician = DailyTactician(sentiment_data, use_shock_signal=True)
        
        # 生成日度信号统计
        stats = tactician.get_signal_statistics()
        print("\n信号统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 示例：获取某日的仓位系数
        # scalar, signal_type, score = tactician.get_position_scalar('2024-01-15')
        # print(f"\n示例: 2024-01-15 仓位系数={scalar}, 信号={signal_type}")
    
    print("\n日度调仓模块已加载")
