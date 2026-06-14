# -*- coding: utf-8 -*-
"""
政策情绪分析器
基于VADER的情绪分析，支持自定义词典和点赞数加权
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config_loader import get_config_loader
from .truth_social_fetcher import TruthSocialFetcher


class PolicySentimentAnalyzer:
    """
    政策情绪分析器
    
    功能：
    1. VADER情绪分析（支持自定义词典）
    2. 点赞数加权
    3. 时间衰减
    4. 生成日度情绪指数
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化分析器
        
        Parameters
        ----------
        config_path : str, optional
            配置文件路径
        """
        self.config_loader = get_config_loader(config_path)
        
        # 初始化VADER分析器
        self.analyzer = SentimentIntensityAnalyzer()
        
        # 加载自定义词典
        self._load_custom_lexicon()
        
        # 加载参数
        self.sentiment_params = self.config_loader.get_sentiment_params()
        self.favourite_weight_factor = self.sentiment_params.get('favourite_weight_factor', 0.1)
        self.max_favourite_weight = self.sentiment_params.get('max_favourite_weight', 5.0)
        self.time_decay_half_life = self.sentiment_params.get('time_decay_half_life', 7)
        
        # 阈值
        self.vader_positive_threshold = self.sentiment_params.get('vader_positive_threshold', 0.05)
        self.vader_negative_threshold = self.sentiment_params.get('vader_negative_threshold', -0.05)
        
        # 冲击阈值
        self.shock_thresholds = self.config_loader.get_shock_thresholds()
    
    def _load_custom_lexicon(self):
        """加载自定义VADER词典"""
        custom_lexicon = self.config_loader.get_vader_lexicon()
        
        # 更新VADER词典
        for word, score in custom_lexicon.items():
            self.analyzer.lexicon[word.lower()] = score
    
    def analyze_post(self, content: str) -> Dict[str, float]:
        """
        分析单条帖子的情绪
        
        Parameters
        ----------
        content : str
            帖子内容
        
        Returns
        -------
        Dict[str, float]
            情绪分数字典
        """
        if not content or not str(content).strip():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 1.0
            }
        
        scores = self.analyzer.polarity_scores(str(content))
        return scores
    
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分析数据框中所有帖子的情绪
        
        Parameters
        ----------
        df : pd.DataFrame
            帖子数据框，需包含'content'列
        
        Returns
        -------
        pd.DataFrame
            添加情绪分数列的数据框
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 分析每条帖子的情绪
        sentiment_scores = df['content'].apply(self.analyze_post)
        
        # 展开情绪分数
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
        df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
        df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
        
        # 添加情绪标签
        df['sentiment_label'] = df['sentiment_compound'].apply(self._compound_to_label)
        
        return df
    
    def _compound_to_label(self, compound: float) -> str:
        """
        将compound分数转换为标签
        
        Parameters
        ----------
        compound : float
            VADER compound分数
        
        Returns
        -------
        str
            情绪标签
        """
        if compound >= self.vader_positive_threshold:
            return 'positive'
        elif compound <= self.vader_negative_threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def calculate_weighted_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算点赞数加权的情绪分数
        
        Parameters
        ----------
        df : pd.DataFrame
            包含情绪分数和点赞数的数据框
        
        Returns
        -------
        pd.DataFrame
            添加加权情绪分数的数据框
        """
        if df.empty or 'favourites_count' not in df.columns:
            return df
        
        df = df.copy()
        
        # 计算点赞数权重 (对数缩放)
        df['favourite_weight'] = df['favourites_count'].apply(
            lambda x: min(1 + self.favourite_weight_factor * np.log1p(x), self.max_favourite_weight)
        )
        
        # 加权情绪分数
        df['weighted_sentiment'] = df['sentiment_compound'] * df['favourite_weight']
        
        return df
    
    def generate_daily_index(self, df: pd.DataFrame, 
                            start_date: str = None, 
                            end_date: str = None) -> pd.DataFrame:
        """
        生成日度情绪指数
        
        Parameters
        ----------
        df : pd.DataFrame
            分析后的帖子数据框
        start_date : str, optional
            开始日期 (YYYY-MM-DD)
        end_date : str, optional
            结束日期 (YYYY-MM-DD)
        
        Returns
        -------
        pd.DataFrame
            日度情绪指数，列：date, sentiment_score, post_count, avg_favourites, vix_proxy, shock_signal
        """
        if df.empty:
            return self._create_empty_daily_index(start_date, end_date)
        
        # 确保有日期列
        if 'created_at' not in df.columns:
            raise ValueError("数据框必须包含'created_at'列")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        
        # 按日期聚合
        daily_stats = []
        
        for date, group in df.groupby('date'):
            # 基础统计
            post_count = len(group)
            avg_favourites = group['favourites_count'].mean() if 'favourites_count' in group.columns else 0
            
            # 情绪分数 (使用加权分数如果存在)
            if 'weighted_sentiment' in group.columns:
                sentiment_score = group['weighted_sentiment'].mean()
            else:
                sentiment_score = group['sentiment_compound'].mean()
            
            # VIX指标：情绪波动的标准差
            vix_proxy = group['sentiment_compound'].std() if len(group) > 1 else 0
            
            # 冲击信号
            shock_signal = self._calculate_shock_signal(sentiment_score, vix_proxy)
            
            daily_stats.append({
                'date': date,
                'sentiment_score': sentiment_score,
                'post_count': post_count,
                'avg_favourites': avg_favourites,
                'vix_proxy': vix_proxy,
                'shock_signal': shock_signal
            })
        
        daily_df = pd.DataFrame(daily_stats)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # 填充缺失日期
        daily_df = self._fill_missing_dates(daily_df, start_date, end_date)
        
        return daily_df.sort_values('date').reset_index(drop=True)
    
    def _calculate_shock_signal(self, sentiment_score: float, vix_proxy: float) -> int:
        """
        计算冲击信号
        
        Returns
        -------
        int
            -2: 严重负面冲击
            -1: 中度负面冲击
             0: 正常
             1: 中度正面冲击
             2: 严重正面冲击
        """
        # 结合情绪分数和波动率
        shock_score = sentiment_score - vix_proxy * 0.5
        
        if shock_score <= self.shock_thresholds.get('severe_negative', -0.5):
            return -2
        elif shock_score <= self.shock_thresholds.get('moderate_negative', -0.3):
            return -1
        elif shock_score >= self.shock_thresholds.get('severe_positive', 0.5):
            return 2
        elif shock_score >= self.shock_thresholds.get('moderate_positive', 0.3):
            return 1
        else:
            return 0
    
    def _fill_missing_dates(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """填充缺失日期"""
        if df.empty:
            return self._create_empty_daily_index(start_date, end_date)
        
        # 确定日期范围
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = df['date'].min()
        
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = df['date'].max()
        
        # 创建完整的日期范围
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # 重新索引
        df = df.set_index('date').reindex(date_range)
        df.index.name = 'date'
        
        # 填充NaN
        df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill').fillna(0)
        df['post_count'] = df['post_count'].fillna(0)
        df['avg_favourites'] = df['avg_favourites'].fillna(0)
        df['vix_proxy'] = df['vix_proxy'].fillna(0)
        df['shock_signal'] = df['shock_signal'].fillna(0)
        
        return df.reset_index()
    
    def _create_empty_daily_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """创建空的日度指数"""
        if start_date and end_date:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            date_range = pd.date_range(start='2022-02-01', periods=1, freq='D')
        
        return pd.DataFrame({
            'date': date_range,
            'sentiment_score': 0.0,
            'post_count': 0,
            'avg_favourites': 0.0,
            'vix_proxy': 0.0,
            'shock_signal': 0
        })
    
    def run_pipeline(self, start_date: str = '2022-02-01', 
                    end_date: str = None,
                    use_mock: bool = False) -> pd.DataFrame:
        """
        运行完整的数据采集和分析流程
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str, optional
            结束日期
        use_mock : bool
            是否使用模拟数据
        
        Returns
        -------
        pd.DataFrame
            日度情绪指数
        """
        # 数据采集
        fetcher = TruthSocialFetcher()
        
        if use_mock:
            print("使用模拟数据...")
            posts_df = fetcher.get_mock_data(start_date=start_date, end_date=end_date, n_samples=500)
        else:
            print("从RSS源采集数据...")
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None
            posts_df = fetcher.fetch_all_posts(start_date=start_dt, end_date=end_dt)
        
        if posts_df.empty:
            print("警告: 未获取到数据，返回空指数")
            return self._create_empty_daily_index(start_date, end_date)
        
        # 情绪分析
        print("进行情绪分析...")
        analyzed_df = self.analyze_dataframe(posts_df)
        
        # 加权计算
        weighted_df = self.calculate_weighted_sentiment(analyzed_df)
        
        # 生成日度指数
        print("生成日度情绪指数...")
        daily_index = self.generate_daily_index(weighted_df, start_date, end_date)
        
        print(f"完成！共生成 {len(daily_index)} 天的情绪指数")
        return daily_index
    
    def save_daily_index(self, df: pd.DataFrame, output_path: str = None):
        """
        保存日度情绪指数到CSV
        
        Parameters
        ----------
        df : pd.DataFrame
            日度情绪指数
        output_path : str, optional
            输出路径，默认为data/processed/daily_sentiment_index.csv
        """
        if output_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_path = project_root / 'data' / 'processed' / 'daily_sentiment_index.csv'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"情绪指数已保存至: {output_path}")


# 便捷函数
def generate_sentiment_index(start_date: str = '2022-02-01',
                            end_date: str = None,
                            use_mock: bool = True,
                            output_path: str = None) -> pd.DataFrame:
    """
    生成情绪指数的便捷函数
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str, optional
        结束日期
    use_mock : bool
        是否使用模拟数据
    output_path : str, optional
        输出路径
    
    Returns
    -------
    pd.DataFrame
        日度情绪指数
    """
    analyzer = PolicySentimentAnalyzer()
    daily_index = analyzer.run_pipeline(start_date, end_date, use_mock)
    
    if output_path:
        analyzer.save_daily_index(daily_index, output_path)
    
    return daily_index
