# -*- coding: utf-8 -*-
"""
Week 7: 六大优化方向综合实验
1. 波动率目标 (Vol Targeting)
2. 风险平价权重 (1/σ替代等权)
3. TopK自适应 (牛市150/震荡100/熊市50)
4. 换手率惩罚/持仓延续
5. 行业中性化 (单行业≤15%)
6. 机器学习因子组合 (XGBoost/GBDT)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')


def load_value_factor(df_ret=None):
    """加载价值因子数据"""
    vf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'value_factor.parquet')
    if os.path.exists(vf_path):
        vf = pd.read_parquet(vf_path)
        vf.index = pd.to_datetime(vf.index)
        if df_ret is not None:
            common_cols = df_ret.columns.intersection(vf.columns)
            vf = vf.reindex(index=df_ret.index, columns=common_cols)
        return vf
    else:
        return df_ret.rolling(window=12).sum().shift(1)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 基础函数
# ============================================================================

def calculate_metrics(returns, rf=0.0):
    if len(returns) == 0 or returns.std() == 0:
        return 0, 0, 0, 0, pd.Series(), 0
    cum_ret = (1 + returns).prod() - 1
    n = len(returns)
    ann_ret = (1 + cum_ret) ** (12 / n) - 1 if n > 0 else 0
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ((returns.mean() - rf / 12) / returns.std()) * np.sqrt(12) if returns.std() > 0 else 0
    cum_wealth = (1 + returns).cumprod()
    drawdown = (cum_wealth - cum_wealth.cummax()) / cum_wealth.cummax()
    max_dd = drawdown.min()
    return ann_ret, sharpe, max_dd, ann_vol, cum_wealth, 0


def apply_standardization(signal_df):
    lower = signal_df.rolling(24, min_periods=6).quantile(0.01)
    upper = signal_df.rolling(24, min_periods=6).quantile(0.99)
    return signal_df.clip(lower=lower, upper=upper, axis=0)


def apply_market_filter(returns, market_returns=None, ma_window=12,
                        bear_scalar=0.5, bear_trend_scalar=0.3):
    if market_returns is None:
        market_returns = returns
    cum_market = (1 + market_returns).cumprod()
    ma = cum_market.rolling(ma_window).mean().shift(1).shift(1)
    ma_slope = ma.diff(3)
    position_scalar = pd.Series(1.0, index=returns.index)
    for date in returns.index:
        if pd.isna(ma.loc[date]):
            continue
        if cum_market.loc[date] < ma.loc[date]:
            if pd.notna(ma_slope.loc[date]) and ma_slope.loc[date] < 0:
                position_scalar.loc[date] = bear_trend_scalar
            else:
                position_scalar.loc[date] = bear_scalar
    return returns * position_scalar, position_scalar


def run_backtest_equal_weight(df_ret, signal_df, topk=100, K=6):
    """标准等权回测"""
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= topk).astype(float)
        selected = selected.where(sig.notna(), other=0.0)
        count = selected.sum()
        if count > 0:
            weights.loc[date] = selected / count
        if len(turnover_list) > 0:
            prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
            turnover_list.append((weights.loc[date] - weights.loc[prev_date]).abs().sum() / 2)
        else:
            turnover_list.append(1.0)

    port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
    if len(turnover_list) < len(port_ret):
        turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
    elif len(turnover_list) > len(port_ret):
        turnover_list = turnover_list[-len(port_ret):]
    turnover_series = pd.Series(turnover_list, index=port_ret.index)

    return port_ret, turnover_series, weights


def get_industry_map(stk_codes):
    """基于股票代码前缀的简单行业分类 (CSRC行业大类)"""
    industry_map = {}
    for code in stk_codes:
        code_str = str(code).zfill(6)
        prefix = code_str[:2]
        # 简化分类: 按代码前2位分行业
        if prefix in ('60', '68'):
            industry_map[code] = '沪市主板'
        elif prefix == '00':
            industry_map[code] = '深市主板'
        elif prefix == '30':
            industry_map[code] = '创业板'
        elif prefix == '92':
            industry_map[code] = '北交所'
        else:
            industry_map[code] = '其他'
    return industry_map


# ============================================================================
# 实验1: 波动率目标 (Vol Targeting)
# ============================================================================

def experiment_vol_targeting(df_ret, signal_df, K=6, topk=100):
    """波动率目标: 根据近期波动率动态调整仓位"""
    print("\n" + "=" * 70)
    print("实验1: 波动率目标 (Vol Targeting)")
    print("=" * 70)

    # 先跑基准等权回测
    port_ret, tover, weights = run_backtest_equal_weight(df_ret, signal_df, topk=topk, K=K)

    # 波动率目标
    vol_targets = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    lookback_windows = [6, 12, 24]

    results = {'基准(无Vol Target)': {
        'ann_ret': 0, 'sharpe': 0, 'max_dd': 0, 'ann_vol': 0, 'turnover': 0
    }}

    # 计算基准指标
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
    results['基准(无Vol Target)'] = {
        'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
        'turnover': tover.mean() * 12
    }
    print(f"  基准: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol:.2%}")

    for lookback in lookback_windows:
        for vol_target in vol_targets:
            # 计算滚动波动率
            rolling_vol = port_ret.rolling(lookback).std().shift(1) * np.sqrt(12)
            # 波动率目标仓位调整
            vol_scalar = vol_target / rolling_vol
            vol_scalar = vol_scalar.clip(0.3, 2.0)  # 仓位限制在30%~200%
            vol_scalar = vol_scalar.fillna(1.0)

            # 调整后收益
            adj_ret = port_ret * vol_scalar
            ann, sharpe, dd, vol_adj, cum, _ = calculate_metrics(adj_ret)

            name = f'VolTarget={vol_target:.0%}_L={lookback}'
            results[name] = {
                'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol_adj,
                'turnover': tover.mean() * 12
            }
            print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol_adj:.2%}")

    # 找最优
    best_name = max(results, key=lambda x: results[x]['sharpe'])
    best = results[best_name]
    print(f"\n  最优: {best_name}, 夏普={best['sharpe']:.2f}, 年化={best['ann_ret']:.2%}, 回撤={best['max_dd']:.2%}")

    return results


# ============================================================================
# 实验2: 风险平价权重
# ============================================================================

def experiment_risk_parity(df_ret, signal_df, K=6, topk=100):
    """风险平价: 用1/σ_i替代等权"""
    print("\n" + "=" * 70)
    print("实验2: 风险平价权重 (1/σ替代等权)")
    print("=" * 70)

    results = {}
    vol_lookback = 12  # 用过去12个月估算个股波动率

    # 基准等权
    port_ret_eq, tover_eq, _ = run_backtest_equal_weight(df_ret, signal_df, topk=topk, K=K)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_eq)
    results['等权(基准)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
                           'turnover': tover_eq.mean() * 12}
    print(f"  等权基准: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # 风险平价: weight_i ∝ 1/σ_i
    weights_rp = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= topk).astype(float)
        selected = selected.where(sig.notna(), other=0.0)

        if selected.sum() > 0:
            # 计算选中股票的波动率
            date_idx = df_ret.index.get_loc(date)
            start_idx = max(0, date_idx - vol_lookback)
            hist_ret = df_ret.iloc[start_idx:date_idx]

            stock_vols = hist_ret.std()
            inv_vol = 1.0 / stock_vols.replace(0, np.nan)
            inv_vol = inv_vol.fillna(0)

            # 仅对选中股票分配权重
            w = selected * inv_vol
            w_sum = w.sum()
            if w_sum > 0:
                weights_rp.loc[date] = w / w_sum

        if len(turnover_list) > 0:
            prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
            turnover_list.append((weights_rp.loc[date] - weights_rp.loc[prev_date]).abs().sum() / 2)
        else:
            turnover_list.append(1.0)

    port_ret_rp = (weights_rp * df_ret).sum(axis=1).iloc[K:]
    if len(turnover_list) < len(port_ret_rp):
        turnover_list = [1.0] * (len(port_ret_rp) - len(turnover_list)) + turnover_list
    elif len(turnover_list) > len(port_ret_rp):
        turnover_list = turnover_list[-len(port_ret_rp):]

    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_rp)
    results['风险平价(1/σ)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
                               'turnover': pd.Series(turnover_list, index=port_ret_rp.index).mean() * 12}
    print(f"  风险平价(1/σ): 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol:.2%}")

    # 信号强度加权: weight_i ∝ signal_rank_i
    weights_sig = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list2 = []

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= topk).astype(float)
        selected = selected.where(sig.notna(), other=0.0)

        if selected.sum() > 0:
            # 信号强度加权: 排名越靠前权重越大
            sig_strength = topk - ranks + 1  # 排名1→topk, 排名topk→1
            sig_strength = sig_strength.clip(lower=0)
            w = selected * sig_strength
            w_sum = w.sum()
            if w_sum > 0:
                weights_sig.loc[date] = w / w_sum

        if len(turnover_list2) > 0:
            prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
            turnover_list2.append((weights_sig.loc[date] - weights_sig.loc[prev_date]).abs().sum() / 2)
        else:
            turnover_list2.append(1.0)

    port_ret_sig = (weights_sig * df_ret).sum(axis=1).iloc[K:]
    if len(turnover_list2) < len(port_ret_sig):
        turnover_list2 = [1.0] * (len(port_ret_sig) - len(turnover_list2)) + turnover_list2
    elif len(turnover_list2) > len(port_ret_sig):
        turnover_list2 = turnover_list2[-len(port_ret_sig):]

    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_sig)
    results['信号强度加权'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
                             'turnover': pd.Series(turnover_list2, index=port_ret_sig.index).mean() * 12}
    print(f"  信号强度加权: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol:.2%}")

    # 最小方差组合 (简化版: 用对角协方差矩阵)
    weights_mv = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list3 = []

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= topk).astype(float)
        selected = selected.where(sig.notna(), other=0.0)

        if selected.sum() > 0:
            date_idx = df_ret.index.get_loc(date)
            start_idx = max(0, date_idx - vol_lookback)
            hist_ret = df_ret.iloc[start_idx:date_idx]

            stock_vols = hist_ret.std()
            inv_var = 1.0 / (stock_vols ** 2).replace(0, np.nan)
            inv_var = inv_var.fillna(0)

            w = selected * inv_var
            w_sum = w.sum()
            if w_sum > 0:
                weights_mv.loc[date] = w / w_sum

        if len(turnover_list3) > 0:
            prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
            turnover_list3.append((weights_mv.loc[date] - weights_mv.loc[prev_date]).abs().sum() / 2)
        else:
            turnover_list3.append(1.0)

    port_ret_mv = (weights_mv * df_ret).sum(axis=1).iloc[K:]
    if len(turnover_list3) < len(port_ret_mv):
        turnover_list3 = [1.0] * (len(port_ret_mv) - len(turnover_list3)) + turnover_list3
    elif len(turnover_list3) > len(port_ret_mv):
        turnover_list3 = turnover_list3[-len(port_ret_mv):]

    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_mv)
    results['最小方差(1/σ²)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
                                 'turnover': pd.Series(turnover_list3, index=port_ret_mv.index).mean() * 12}
    print(f"  最小方差(1/σ²): 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol:.2%}")

    return results


# ============================================================================
# 实验3: TopK自适应
# ============================================================================

def experiment_topk_adaptive(df_ret, signal_df, K=6):
    """TopK自适应: 牛市150/震荡100/熊市50"""
    print("\n" + "=" * 70)
    print("实验3: TopK自适应")
    print("=" * 70)

    market_ret = df_ret.mean(axis=1)
    cum_market = (1 + market_ret).cumprod()
    ma_market = cum_market.rolling(12).mean().shift(1)  # 避免前视偏差

    results = {}

    # 基准: 固定TopK=100
    for topk in [50, 80, 100, 120, 150]:
        port_ret, tover, _ = run_backtest_equal_weight(df_ret, signal_df, topk=topk, K=K)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
        results[f'固定TopK={topk}'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                        'ann_vol': vol, 'turnover': tover.mean() * 12}
        print(f"  固定TopK={topk}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # 自适应TopK方案
    configs = [
        ('自适应(50/100/150)', 50, 100, 150),
        ('自适应(80/100/120)', 80, 100, 120),
        ('自适应(30/100/200)', 30, 100, 200),
        ('自适应(50/80/120)', 50, 80, 120),
    ]

    for name, bear_k, mid_k, bull_k in configs:
        weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
        turnover_list = []

        for date in df_ret.index[K:]:
            sig = signal_df.loc[date]
            ranks = sig.rank(ascending=False, method='first')

            # 判断市场状态
            if pd.notna(ma_market.loc[date]):
                if cum_market.loc[date] > ma_market.loc[date] * 1.02:
                    current_topk = bull_k
                elif cum_market.loc[date] < ma_market.loc[date] * 0.98:
                    current_topk = bear_k
                else:
                    current_topk = mid_k
            else:
                current_topk = mid_k

            selected = (ranks <= current_topk).astype(float)
            selected = selected.where(sig.notna(), other=0.0)
            count = selected.sum()
            if count > 0:
                weights.loc[date] = selected / count

            if len(turnover_list) > 0:
                prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
                turnover_list.append((weights.loc[date] - weights.loc[prev_date]).abs().sum() / 2)
            else:
                turnover_list.append(1.0)

        port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
        if len(turnover_list) < len(port_ret):
            turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
        elif len(turnover_list) > len(port_ret):
            turnover_list = turnover_list[-len(port_ret):]

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
        results[name] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
                         'turnover': pd.Series(turnover_list, index=port_ret.index).mean() * 12}
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    return results


# ============================================================================
# 实验4: 换手率惩罚/持仓延续
# ============================================================================

def experiment_turnover_control(df_ret, signal_df, K=6, topk=100):
    """换手率控制: 持仓延续 + 换手惩罚"""
    print("\n" + "=" * 70)
    print("实验4: 换手率惩罚/持仓延续")
    print("=" * 70)

    results = {}

    # 基准
    port_ret, tover, _ = run_backtest_equal_weight(df_ret, signal_df, topk=topk, K=K)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
    results['基准(无控制)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                              'ann_vol': vol, 'turnover': tover.mean() * 12}
    print(f"  基准: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover.mean()*12:.0%}")

    # 方案A: 持仓延续 — 上月TopK中仍在TopK+buffer的保留不动
    for buffer in [20, 50, 100]:
        weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
        prev_selected = None
        turnover_list = []

        for date in df_ret.index[K:]:
            sig = signal_df.loc[date]
            ranks = sig.rank(ascending=False, method='first')

            if prev_selected is not None:
                # 上月持仓中仍在Top(K+buffer)的保留
                keep_mask = (ranks <= topk + buffer) & prev_selected
                # 新进入TopK的
                new_mask = (ranks <= topk) & ~keep_mask
                selected = keep_mask | new_mask
            else:
                selected = (ranks <= topk)

            selected = selected.astype(float)
            selected = selected.where(sig.notna(), other=0.0)
            count = selected.sum()
            if count > 0:
                weights.loc[date] = selected / count

            prev_selected = selected > 0

            if len(turnover_list) > 0:
                prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
                turnover_list.append((weights.loc[date] - weights.loc[prev_date]).abs().sum() / 2)
            else:
                turnover_list.append(1.0)

        port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
        if len(turnover_list) < len(port_ret):
            turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
        elif len(turnover_list) > len(port_ret):
            turnover_list = turnover_list[-len(port_ret):]

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
        tover_ann = pd.Series(turnover_list, index=port_ret.index).mean() * 12
        results[f'持仓延续(buffer={buffer})'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                                   'ann_vol': vol, 'turnover': tover_ann}
        print(f"  持仓延续(buffer={buffer}): 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")

    # 方案B: 季度调仓 (每3个月调仓一次)
    for rebal_months in [2, 3, 4, 6]:
        weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
        last_rebal_idx = -rebal_months
        turnover_list = []

        for i, date in enumerate(df_ret.index[K:]):
            if i - last_rebal_idx >= rebal_months:
                # 调仓
                sig = signal_df.loc[date]
                ranks = sig.rank(ascending=False, method='first')
                selected = (ranks <= topk).astype(float)
                selected = selected.where(sig.notna(), other=0.0)
                count = selected.sum()
                if count > 0:
                    weights.loc[date] = selected / count
                last_rebal_idx = i
            else:
                # 不调仓，沿用上期权重
                prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
                weights.loc[date] = weights.loc[prev_date]

            if len(turnover_list) > 0:
                prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
                turnover_list.append((weights.loc[date] - weights.loc[prev_date]).abs().sum() / 2)
            else:
                turnover_list.append(1.0)

        port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
        if len(turnover_list) < len(port_ret):
            turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
        elif len(turnover_list) > len(port_ret):
            turnover_list = turnover_list[-len(port_ret):]

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
        tover_ann = pd.Series(turnover_list, index=port_ret.index).mean() * 12
        results[f'季度调仓({rebal_months}月)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                                 'ann_vol': vol, 'turnover': tover_ann}
        print(f"  季度调仓({rebal_months}月): 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")

    return results


# ============================================================================
# 实验5: 行业中性化
# ============================================================================

def experiment_industry_neutral(df_ret, signal_df, K=6, topk=100):
    """行业中性化: 单行业权重上限"""
    print("\n" + "=" * 70)
    print("实验5: 行业中性化")
    print("=" * 70)

    results = {}

    # 行业映射
    industry_map = get_industry_map(df_ret.columns)
    industries = pd.Series(industry_map)

    # 基准
    port_ret, tover, _ = run_backtest_equal_weight(df_ret, signal_df, topk=topk, K=K)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
    results['基准(无行业约束)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                  'ann_vol': vol, 'turnover': tover.mean() * 12}
    print(f"  基准: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # 行业中性化: 信号行业内排名 → 行业内选股
    for max_industry_weight in [0.15, 0.20, 0.25, 0.30]:
        weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
        turnover_list = []

        for date in df_ret.index[K:]:
            sig = signal_df.loc[date]

            # 行业内排名选股
            selected = pd.Series(0.0, index=df_ret.columns)
            for ind_name in industries.unique():
                ind_stocks = industries[industries == ind_name].index
                ind_stocks = [s for s in ind_stocks if s in sig.index]
                if len(ind_stocks) == 0:
                    continue
                ind_sig = sig[ind_stocks]
                # 每个行业按比例选股
                n_select = max(1, int(topk * len(ind_stocks) / len(df_ret.columns)))
                ind_ranks = ind_sig.rank(ascending=False, method='first')
                ind_selected = (ind_ranks <= n_select).astype(float)
                ind_selected = ind_selected.where(ind_sig.notna(), other=0.0)
                selected[ind_stocks] = ind_selected.values

            # 应用行业权重上限
            count = selected.sum()
            if count > 0:
                w = selected / count
                # 检查行业权重
                for ind_name in industries.unique():
                    ind_stocks = industries[industries == ind_name].index
                    ind_stocks = [s for s in ind_stocks if s in w.index]
                    ind_weight = w[ind_stocks].sum()
                    if ind_weight > max_industry_weight:
                        # 缩减该行业权重
                        scale = max_industry_weight / ind_weight
                        w[ind_stocks] = w[ind_stocks] * scale
                # 重新归一化
                w = w / w.sum()
                weights.loc[date] = w

            if len(turnover_list) > 0:
                prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
                turnover_list.append((weights.loc[date] - weights.loc[prev_date]).abs().sum() / 2)
            else:
                turnover_list.append(1.0)

        port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
        if len(turnover_list) < len(port_ret):
            turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
        elif len(turnover_list) > len(port_ret):
            turnover_list = turnover_list[-len(port_ret):]

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
        tover_ann = pd.Series(turnover_list, index=port_ret.index).mean() * 12
        results[f'行业中性(上限{max_industry_weight:.0%})'] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol, 'turnover': tover_ann}
        print(f"  行业中性(上限{max_industry_weight:.0%}): 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    return results


# ============================================================================
# 实验6: 机器学习因子组合 (XGBoost/GBDT)
# ============================================================================

def experiment_ml_factor(df_ret, signal_df, K=6, topk=100):
    """机器学习因子组合: 用GBDT预测下月收益"""
    print("\n" + "=" * 70)
    print("实验6: 机器学习因子组合 (GBDT)")
    print("=" * 70)

    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge, Lasso
    except ImportError:
        print("  sklearn未安装，跳过ML实验")
        return {}

    results = {}

    # 基准
    port_ret, tover, _ = run_backtest_equal_weight(df_ret, signal_df, topk=topk, K=K)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
    results['基准(线性组合40V+60R)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                       'ann_vol': vol, 'turnover': tover.mean() * 12}
    print(f"  基准: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # 构造特征
    print("  构造特征...")
    # 特征1: 反转信号 (不同K)
    features = {}
    for k in [1, 2, 3, 6, 9, 12]:
        feat = -df_ret.rolling(window=k).sum().shift(1)
        features[f'reversal_K{k}'] = apply_standardization(feat)

    # 特征2: 价值信号 (12月累计收益)
    features['value_12m'] = apply_standardization(load_value_factor(df_ret))

    # 特征3: 波动率
    features['volatility_6m'] = df_ret.rolling(window=6).std().shift(1)
    features['volatility_12m'] = df_ret.rolling(window=12).std().shift(1)

    # 特征4: 动量 (6月-12月差异)
    mom_6 = df_ret.rolling(window=6).sum().shift(1)
    mom_12 = df_ret.rolling(window=12).sum().shift(1)
    features['momentum_accel'] = apply_standardization(mom_6 - mom_12)

    # 特征5: 偏度
    features['skewness_6m'] = df_ret.rolling(window=6).skew().shift(1)

    # 目标变量: 下月收益
    target = df_ret.shift(-1)

    # Rank标准化 (GKX方法)
    print("  Rank标准化特征...")
    for feat_name in features:
        feat_df = features[feat_name]
        # 截面排名映射到[-1,1]
        features[feat_name] = feat_df.rank(pct=True, axis=1) * 2 - 1

    # 滚动训练 + 预测
    print("  滚动训练GBDT模型...")
    train_window = 60  # 5年训练
    predict_start = K + train_window

    feature_names = list(features.keys())

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=50,
                                     random_state=42, n_jobs=-1),
        'GBDT': GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                           learning_rate=0.05, min_samples_leaf=50,
                                           subsample=0.8, random_state=42),
    }

    for model_name, base_model in models.items():
        print(f"\n  训练 {model_name}...")

        # 存储预测信号
        ml_signal = pd.DataFrame(np.nan, index=df_ret.index, columns=df_ret.columns, dtype=float)

        # 滚动预测 (每12个月重新训练)
        dates = df_ret.index
        last_train_idx = -1

        for i in range(predict_start, len(dates)):
            if i - last_train_idx < 12 and last_train_idx > 0:
                # 不重新训练，用上次模型
                pass
            else:
                # 重新训练
                train_end_idx = i
                train_start_idx = max(K, i - train_window)

                # 构建训练数据 (面板展开)
                X_train_list = []
                y_train_list = []

                for t_idx in range(train_start_idx, train_end_idx):
                    date_t = dates[t_idx]
                    # 获取该日期的特征和目标
                    feat_vals = []
                    for fn in feature_names:
                        if date_t in features[fn].index:
                            feat_vals.append(features[fn].loc[date_t].values)
                        else:
                            feat_vals.append(np.full(len(df_ret.columns), np.nan))

                    X_t = np.column_stack(feat_vals)
                    y_t = target.loc[date_t].values if date_t in target.index else np.full(len(df_ret.columns), np.nan)

                    # 过滤NaN
                    valid_mask = ~np.isnan(y_t) & ~np.isnan(X_t).any(axis=1)
                    if valid_mask.sum() > 100:
                        X_train_list.append(X_t[valid_mask])
                        y_train_list.append(y_t[valid_mask])

                if len(X_train_list) == 0:
                    continue

                X_train = np.vstack(X_train_list)
                y_train = np.concatenate(y_train_list)

                if len(X_train) < 500:
                    continue

                # 训练模型
                model = type(base_model)(**base_model.get_params())
                model.fit(X_train, y_train)
                last_train_idx = i

            # 预测
            date_pred = dates[i]
            feat_vals = []
            for fn in feature_names:
                if date_pred in features[fn].index:
                    feat_vals.append(features[fn].loc[date_pred].values)
                else:
                    feat_vals.append(np.full(len(df_ret.columns), np.nan))

            X_pred = np.column_stack(feat_vals)
            valid_mask = ~np.isnan(X_pred).any(axis=1)

            if valid_mask.sum() > 0 and last_train_idx > 0:
                preds = np.full(len(df_ret.columns), np.nan)
                preds[valid_mask] = model.predict(X_pred[valid_mask])
                ml_signal.loc[date_pred] = preds

        # 用ML信号选股 (ml_signal预测下月收益，需shift(1)对齐持仓周期)
        ml_signal = ml_signal.fillna(0).shift(1)
        port_ret_ml, tover_ml, _ = run_backtest_equal_weight(df_ret, ml_signal, topk=topk, K=predict_start)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_ml)
        results[f'{model_name}选股'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                        'ann_vol': vol, 'turnover': tover_ml.mean() * 12}
        print(f'  {model_name}选股: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}')

        # ML信号 + 原始信号组合
        combined_signal = 0.5 * signal_df + 0.5 * ml_signal
        port_ret_comb, tover_comb, _ = run_backtest_equal_weight(df_ret, combined_signal, topk=topk, K=predict_start)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_comb)
        results[f'{model_name}+原始(50/50)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                                  'ann_vol': vol, 'turnover': tover_comb.mean() * 12}
        print(f"  {model_name}+原始(50/50): 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("Week 7: 六大优化方向综合实验")
    print("=" * 80)

    # 加载数据
    print("\n[1] 加载数据...")
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)

    print(f"  数据: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")

    # 信号
    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal)
    combo_signal = 0.4 * value_signal + 0.6 * reversal_signal

    market_ret = df_ret.mean(axis=1)

    # ========================================================================
    # 运行所有实验
    # ========================================================================
    all_results = {}

    # 实验1: 波动率目标
    res1 = experiment_vol_targeting(df_ret, combo_signal, K=K)
    all_results['1_VolTarget'] = res1

    # 实验2: 风险平价
    res2 = experiment_risk_parity(df_ret, combo_signal, K=K)
    all_results['2_RiskParity'] = res2

    # 实验3: TopK自适应
    res3 = experiment_topk_adaptive(df_ret, combo_signal, K=K)
    all_results['3_TopK'] = res3

    # 实验4: 换手率控制
    res4 = experiment_turnover_control(df_ret, combo_signal, K=K)
    all_results['4_TurnoverControl'] = res4

    # 实验5: 行业中性化
    res5 = experiment_industry_neutral(df_ret, combo_signal, K=K)
    all_results['5_IndustryNeutral'] = res5

    # 实验6: ML因子组合
    res6 = experiment_ml_factor(df_ret, combo_signal, K=K)
    all_results['6_ML'] = res6

    # ========================================================================
    # 综合排名
    # ========================================================================
    print("\n" + "=" * 80)
    print("综合排名 (所有实验)")
    print("=" * 80)

    flat_results = []
    for exp_name, res in all_results.items():
        for cfg_name, metrics in res.items():
            flat_results.append({
                '实验': exp_name,
                '配置': cfg_name,
                '年化收益': metrics['ann_ret'],
                '夏普': metrics['sharpe'],
                '最大回撤': metrics['max_dd'],
                '波动率': metrics['ann_vol'],
                '换手率': metrics['turnover']
            })

    df_results = pd.DataFrame(flat_results)
    df_results = df_results.sort_values('夏普', ascending=False)

    print(f"\n{'排名':>3} {'实验':<20} {'配置':<30} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'波动率':>8} {'换手率':>6}")
    print("-" * 100)
    for i, row in df_results.head(20).iterrows():
        rank = df_results.index.get_loc(i) + 1
        print(f"{rank:>3} {row['实验']:<20} {row['配置']:<30} {row['年化收益']:>7.2%} "
              f"{row['夏普']:>6.2f} {row['最大回撤']:>7.2%} {row['波动率']:>7.2%} {row['换手率']:>5.0%}")

    # 每个实验的最优
    print("\n" + "=" * 80)
    print("各实验最优配置")
    print("=" * 80)

    for exp_name, res in all_results.items():
        best_name = max(res, key=lambda x: res[x]['sharpe'])
        best = res[best_name]
        print(f"  {exp_name}: {best_name}")
        print(f"    年化={best['ann_ret']:.2%}, 夏普={best['sharpe']:.2f}, "
              f"回撤={best['max_dd']:.2%}, 波动={best['ann_vol']:.2%}, 换手={best['turnover']:.0%}")

    # 保存结果
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    df_results.to_csv(output_dir / 'week7_all_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: {output_dir / 'week7_all_results.csv'}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for idx, (exp_name, res) in enumerate(all_results.items()):
        ax = axes[idx // 3, idx % 3]
        names = list(res.keys())
        sharpes = [res[n]['sharpe'] for n in names]
        turnovers = [res[n]['turnover'] for n in names]

        # 简化名称
        short_names = [n[:20] for n in names]

        ax.scatter(turnovers, sharpes, s=60, alpha=0.7)
        for i, name in enumerate(short_names):
            ax.annotate(name, (turnovers[i], sharpes[i]), fontsize=6, alpha=0.8)
        ax.set_xlabel('换手率', fontsize=9)
        ax.set_ylabel('夏普比率', fontsize=9)
        ax.set_title(exp_name, fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Week 7: 六大优化方向实验', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'week7_optimization.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_dir / 'week7_optimization.png'}")
    plt.close()

    return all_results


if __name__ == '__main__':
    all_results = main()
