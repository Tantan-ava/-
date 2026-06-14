# -*- coding: utf-8 -*-
"""
换手率优化实验
数据源: CSMAR LIQ_TOVER_M 月度换手率

优化方向:
1. 流动性过滤 — 排除低换手率股票，提升可交易性
2. 换手率因子 — 将换手率变化作为辅助因子融入选股
3. 换手率市场情绪 — 全市场换手率判断牛熊，优化市场过滤器
4. 换手率加权 — 用换手率调整持仓权重
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


def calculate_metrics(returns, rf=0.0):
    """计算策略指标"""
    if len(returns) == 0 or returns.std() == 0:
        return 0, 0, 0, 0, pd.Series()
    cum_ret = (1 + returns).prod() - 1
    ann_ret = (1 + cum_ret) ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ((returns.mean() - rf / 12) / returns.std()) * np.sqrt(12) if returns.std() > 0 else 0
    cum_wealth = (1 + returns).cumprod()
    drawdown = (cum_wealth - cum_wealth.cummax()) / cum_wealth.cummax()
    max_dd = drawdown.min()
    return ann_ret, sharpe, max_dd, ann_vol, cum_wealth


def apply_standardization(signal_df, method='Winsorization'):
    """信号标准化"""
    if method == 'Raw':
        return signal_df
    elif method == 'Winsorization':
        lower = signal_df.rolling(24, min_periods=6).quantile(0.01)
        upper = signal_df.rolling(24, min_periods=6).quantile(0.99)
        return signal_df.clip(lower=lower, upper=upper, axis=0)
    elif method == 'Z-Score':
        return (signal_df - signal_df.mean()) / signal_df.std()
    return signal_df


def apply_market_filter(returns, market_returns=None, ma_window=12,
                        bear_scalar=0.5, bear_trend_scalar=0.3):
    """市场状态过滤器"""
    if market_returns is None:
        market_returns = returns
    cum_market = (1 + market_returns).cumprod()
    ma = cum_market.rolling(ma_window).mean().shift(1).shift(1)
    ma_slope = ma.diff(3)
    position_scalar = pd.Series(1.0, index=returns.index)
    for i, date in enumerate(returns.index):
        if pd.isna(ma.loc[date]):
            continue
        if cum_market.loc[date] < ma.loc[date]:
            if pd.notna(ma_slope.loc[date]) and ma_slope.loc[date] < 0:
                position_scalar.loc[date] = bear_trend_scalar
            else:
                position_scalar.loc[date] = bear_scalar
    adjusted_returns = returns * position_scalar
    return adjusted_returns, position_scalar


def apply_drawdown_control(returns, soft_dd_threshold=-0.25, hard_dd_threshold=-0.35,
                           fuse_dd_threshold=-0.50, soft_position=0.80,
                           hard_position=0.40, fuse_position=0.0):
    """回撤止损（极宽阈值）"""
    cum_wealth = (1 + returns).cumprod()
    drawdown = (cum_wealth - cum_wealth.cummax()) / cum_wealth.cummax()
    position_levels = pd.Series(1.0, index=returns.index)
    current_position = 1.0
    stop_counter = 0
    for i, date in enumerate(returns.index):
        dd = drawdown.iloc[i]
        if dd <= fuse_dd_threshold:
            current_position = fuse_position
            stop_counter = 3
        elif dd <= hard_dd_threshold:
            current_position = hard_position
            stop_counter = 2
        elif dd <= soft_dd_threshold:
            current_position = soft_position
            stop_counter = 1
        if stop_counter > 0:
            stop_counter -= 1
            if stop_counter == 0 and dd > soft_dd_threshold / 2:
                current_position = 1.0
        position_levels.iloc[i] = current_position
    adjusted_returns = returns * position_levels
    return adjusted_returns, position_levels


def apply_realistic_costs(returns, turnover_series, commission=0.00025,
                          stamp_tax=0.001, transfer_fee=0.00002, impact_cost=0.001):
    """真实交易成本"""
    buy_cost = commission + transfer_fee + impact_cost
    sell_cost = commission + stamp_tax + transfer_fee + impact_cost
    round_trip_cost = buy_cost + sell_cost
    monthly_cost = turnover_series * round_trip_cost
    net_returns = returns - monthly_cost
    return net_returns, monthly_cost, round_trip_cost


def run_backtest(df_ret, K=6, TopK=100, standardization='Winsorization',
                 rebalance_freq='M', weight_scheme='equal',
                 liquidity_filter=None, turnover_weights=None):
    """
    运行回测，支持流动性过滤和换手率加权

    liquidity_filter: pd.DataFrame, 与df_ret同index/columns, 布尔值(True=可交易)
    turnover_weights: pd.DataFrame, 与df_ret同index/columns, 换手率权重
    """
    # 反转信号
    signal = -df_ret.rolling(window=K).sum().shift(1)
    signal = apply_standardization(signal, standardization)

    # 价值因子
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal, standardization)

    # 综合信号
    final_signal = 0.4 * value_signal + 0.6 * signal

    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []

    for date in df_ret.index[K:]:
        sig = final_signal.loc[date]

        # 流动性过滤
        if liquidity_filter is not None and date in liquidity_filter.index:
            tradable = liquidity_filter.loc[date]
            sig = sig.where(tradable, other=np.nan)

        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= TopK).astype(float)

        # 排除NaN
        selected = selected.where(sig.notna(), other=0.0)

        count = selected.sum()
        if count == 0:
            continue

        if weight_scheme == 'equal':
            weights.loc[date] = selected / count
        elif weight_scheme == 'turnover_weighted' and turnover_weights is not None:
            tw = turnover_weights.loc[date] * selected
            tw_sum = tw.sum()
            if tw_sum > 0:
                weights.loc[date] = tw / tw_sum
            else:
                weights.loc[date] = selected / count
        else:
            weights.loc[date] = selected / count

        # 换手率计算
        if len(turnover_list) > 0:
            old_w = weights.loc[df_ret.index[df_ret.index.get_loc(date) - 1]]
            new_w = weights.loc[date]
            turnover_list.append((new_w - old_w).abs().sum() / 2)
        else:
            turnover_list.append(1.0)

    port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
    # 对齐换手率序列长度
    if len(turnover_list) < len(port_ret):
        turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
    elif len(turnover_list) > len(port_ret):
        turnover_list = turnover_list[-len(port_ret):]
    turnover_series = pd.Series(turnover_list, index=port_ret.index)

    return port_ret, turnover_series, weights


def main():
    print("=" * 80)
    print("换手率优化实验")
    print("数据源: CSMAR LIQ_TOVER_M 月度换手率")
    print("=" * 80)

    # ========================================================================
    # 1. 数据加载
    # ========================================================================
    print("\n[1] 加载数据...")

    # 收益率数据
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    print(f"  收益率数据: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")

    # 换手率数据
    df_tover = pd.read_excel('data/raw/LIQ_TOVER_M.xlsx', engine='openpyxl', skiprows=[1, 2])
    df_tover.columns = ['Stkcd', 'Trdmnt', 'ToverOsM']
    df_tover['ToverOsM'] = pd.to_numeric(df_tover['ToverOsM'], errors='coerce')
    df_tover['Trdmnt'] = pd.to_datetime(df_tover['Trdmnt'], format='%Y-%m')
    print(f"  换手率数据: {df_tover.shape}, {df_tover['Trdmnt'].min()} ~ {df_tover['Trdmnt'].max()}")

    # ========================================================================
    # 2. 数据对齐: 将换手率转为与df_ret相同的宽表格式
    # ========================================================================
    print("\n[2] 数据对齐...")

    # 换手率宽表: index=月份, columns=股票代码
    df_tover['Stkcd'] = df_tover['Stkcd'].astype(str).str.zfill(6)
    tover_pivot = df_tover.pivot(index='Trdmnt', columns='Stkcd', values='ToverOsM')
    tover_pivot.index = pd.to_datetime(tover_pivot.index)
    print(f"  换手率宽表: {tover_pivot.shape}")

    # 对齐列名
    ret_cols = set(df_ret.columns.astype(str))
    tover_cols = set(tover_pivot.columns.astype(str))
    common_cols = ret_cols & tover_cols
    print(f"  收益率列数: {len(ret_cols)}, 换手率列数: {len(tover_cols)}, 交集: {len(common_cols)}")

    # 统一列名格式
    df_ret_aligned = df_ret.copy()
    df_ret_aligned.columns = df_ret_aligned.columns.astype(str)
    tover_aligned = tover_pivot[sorted(common_cols)].copy()
    df_ret_aligned = df_ret_aligned[sorted(common_cols)].copy()

    # 对齐时间范围
    common_idx = df_ret_aligned.index.intersection(tover_aligned.index)
    df_ret_aligned = df_ret_aligned.loc[common_idx]
    tover_aligned = tover_aligned.loc[common_idx]
    print(f"  对齐后: {df_ret_aligned.shape}, {common_idx[0]} ~ {common_idx[-1]}")

    # 换手率统计
    print(f"\n  换手率统计:")
    print(f"    均值: {tover_aligned.mean().mean():.2f}%")
    print(f"    中位数: {tover_aligned.median().median():.2f}%")
    print(f"    25分位: {tover_aligned.quantile(0.25).mean():.2f}%")
    print(f"    75分位: {tover_aligned.quantile(0.75).mean():.2f}%")

    all_results = {}

    # ========================================================================
    # 基准策略 (无换手率优化)
    # ========================================================================
    print("\n" + "=" * 80)
    print("基准策略 (K=6, TopK=100, 无换手率优化)")
    print("=" * 80)

    base_ret, base_turnover, base_weights = run_backtest(
        df_ret_aligned, K=6, TopK=100, standardization='Winsorization')
    base_ann, base_sharpe, base_dd, base_vol, base_cum = calculate_metrics(base_ret)
    base_turnover_ann = base_turnover.mean() * 12

    print(f"  年化收益: {base_ann:.2%}")
    print(f"  夏普比率: {base_sharpe:.2f}")
    print(f"  最大回撤: {base_dd:.2%}")
    print(f"  年化换手率: {base_turnover_ann:.2%}")

    all_results['基准'] = {
        'ann_ret': base_ann, 'sharpe': base_sharpe, 'max_dd': base_dd,
        'ann_vol': base_vol, 'turnover': base_turnover_ann,
        'returns': base_ret, 'cum_wealth': base_cum
    }

    # ========================================================================
    # 实验1: 流动性过滤 — 排除低换手率股票
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验1: 流动性过滤 — 排除月换手率低于阈值的股票")
    print("=" * 80)

    liquidity_thresholds = [5, 10, 15, 20, 30]  # 月换手率%阈值

    for threshold in liquidity_thresholds:
        # 构建流动性过滤矩阵: True=可交易
        liquidity_filter = tover_aligned >= threshold

        ret, turnover, weights = run_backtest(
            df_ret_aligned, K=6, TopK=100, standardization='Winsorization',
            liquidity_filter=liquidity_filter)

        ann, sharpe, dd, vol, cum = calculate_metrics(ret)
        turnover_ann = turnover.mean() * 12

        # 统计每月平均被过滤的股票数
        filter_rate = (1 - liquidity_filter.mean(axis=1).mean())
        avg_stocks = (weights > 0).sum(axis=1).mean()

        name = f'流动性过滤(>{threshold}%)'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, "
              f"回撤={dd:.2%}, 换手={turnover_ann:.2%}, "
              f"均选股={avg_stocks:.0f}, 过滤率={filter_rate:.1%}")

        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': turnover_ann,
            'filter_rate': filter_rate, 'avg_stocks': avg_stocks,
            'returns': ret, 'cum_wealth': cum
        }

    # ========================================================================
    # 实验2: 换手率因子 — 将换手率变化融入选股信号
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验2: 换手率因子 — 换手率变化作为辅助因子")
    print("=" * 80)

    # 换手率变化率 (月度环比)
    tover_change = tover_aligned.pct_change(periods=1)
    tover_change = tover_change.replace([np.inf, -np.inf], np.nan)

    # 换手率动量 (3个月累计变化)
    tover_mom = tover_aligned.pct_change(periods=3)
    tover_mom = tover_mom.replace([np.inf, -np.inf], np.nan)

    factor_configs = [
        # (名称, 换手率因子, 权重)
        ('换手率变化+0.1', tover_change, 0.1),
        ('换手率变化+0.2', tover_change, 0.2),
        ('换手率动量+0.1', tover_mom, 0.1),
        ('换手率动量+0.2', tover_mom, 0.2),
    ]

    # 基础信号
    reversal_signal = -df_ret_aligned.rolling(window=6).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal, 'Winsorization')
    value_signal = load_value_factor(df_ret_aligned)
    value_signal = apply_standardization(value_signal, 'Winsorization')
    base_signal = 0.4 * value_signal + 0.6 * reversal_signal

    for factor_name, tover_factor, weight in factor_configs:
        # 标准化换手率因子
        tover_factor_std = apply_standardization(tover_factor, 'Winsorization')

        # 融合信号: Final_Score = base_signal * (1 - w) + tover_factor * w
        combined_signal = base_signal * (1 - weight) + tover_factor_std * weight

        weights_df = pd.DataFrame(0.0, index=df_ret_aligned.index,
                                  columns=df_ret_aligned.columns, dtype=float)
        turnover_list = []

        for date in df_ret_aligned.index[6:]:
            sig = combined_signal.loc[date]
            ranks = sig.rank(ascending=False, method='first')
            selected = (ranks <= 100).astype(float)
            selected = selected.where(sig.notna(), other=0.0)
            count = selected.sum()
            if count > 0:
                weights_df.loc[date] = selected / count
            if len(turnover_list) > 0:
                old_w = weights_df.loc[df_ret_aligned.index[df_ret_aligned.index.get_loc(date) - 1]]
                new_w = weights_df.loc[date]
                turnover_list.append((new_w - old_w).abs().sum() / 2)
            else:
                turnover_list.append(1.0)

        port_ret = (weights_df * df_ret_aligned).sum(axis=1).iloc[6:]
        turnover_series = pd.Series(turnover_list, index=port_ret.index)
        ann, sharpe, dd, vol, cum = calculate_metrics(port_ret)
        turnover_ann = turnover_series.mean() * 12

        print(f"  {factor_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, "
              f"回撤={dd:.2%}, 换手={turnover_ann:.2%}")

        all_results[f'因子:{factor_name}'] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': turnover_ann,
            'returns': port_ret, 'cum_wealth': cum
        }

    # ========================================================================
    # 实验3: 换手率市场情绪 — 全市场换手率判断牛熊
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验3: 换手率市场情绪 — 全市场换手率优化市场过滤器")
    print("=" * 80)

    # 全市场月度平均换手率
    market_tover = tover_aligned.mean(axis=1)
    market_tover_ma12 = market_tover.rolling(12).mean().shift(1)  # 避免前视偏差

    # 方案A: 换手率低于12月均值 → 降仓 (低换手=熊市/低迷)
    tover_filter_a = pd.Series(1.0, index=base_ret.index)
    for date in base_ret.index:
        if pd.isna(market_tover_ma12.loc[date]):
            continue
        if market_tover.loc[date] < market_tover_ma12.loc[date] * 0.8:
            tover_filter_a.loc[date] = 0.5
        elif market_tover.loc[date] < market_tover_ma12.loc[date]:
            tover_filter_a.loc[date] = 0.7

    ret_a = base_ret * tover_filter_a
    ann_a, sharpe_a, dd_a, vol_a, cum_a = calculate_metrics(ret_a)
    print(f"  方案A(低换手降仓): 年化={ann_a:.2%}, 夏普={sharpe_a:.2f}, "
          f"回撤={dd_a:.2%}, 降仓={(tover_filter_a < 1).sum()}次")

    all_results['换手率情绪-A低换手降仓'] = {
        'ann_ret': ann_a, 'sharpe': sharpe_a, 'max_dd': dd_a,
        'ann_vol': vol_a, 'returns': ret_a, 'cum_wealth': cum_a
    }

    # 方案B: 换手率异常高 → 降仓 (高换手=投机过热/见顶信号)
    tover_filter_b = pd.Series(1.0, index=base_ret.index)
    for date in base_ret.index:
        if pd.isna(market_tover_ma12.loc[date]):
            continue
        if market_tover.loc[date] > market_tover_ma12.loc[date] * 2.0:
            tover_filter_b.loc[date] = 0.3
        elif market_tover.loc[date] > market_tover_ma12.loc[date] * 1.5:
            tover_filter_b.loc[date] = 0.6

    ret_b = base_ret * tover_filter_b
    ann_b, sharpe_b, dd_b, vol_b, cum_b = calculate_metrics(ret_b)
    print(f"  方案B(高换手降仓): 年化={ann_b:.2%}, 夏普={sharpe_b:.2f}, "
          f"回撤={dd_b:.2%}, 降仓={(tover_filter_b < 1).sum()}次")

    all_results['换手率情绪-B高换手降仓'] = {
        'ann_ret': ann_b, 'sharpe': sharpe_b, 'max_dd': dd_b,
        'ann_vol': vol_b, 'returns': ret_b, 'cum_wealth': cum_b
    }

    # 方案C: 双向过滤 (低换手+高换手都降仓)
    tover_filter_c = pd.Series(1.0, index=base_ret.index)
    for date in base_ret.index:
        if pd.isna(market_tover_ma12.loc[date]):
            continue
        ratio = market_tover.loc[date] / market_tover_ma12.loc[date]
        if ratio > 2.0:
            tover_filter_c.loc[date] = 0.3
        elif ratio > 1.5:
            tover_filter_c.loc[date] = 0.6
        elif ratio < 0.8:
            tover_filter_c.loc[date] = 0.5
        elif ratio < 1.0:
            tover_filter_c.loc[date] = 0.7

    ret_c = base_ret * tover_filter_c
    ann_c, sharpe_c, dd_c, vol_c, cum_c = calculate_metrics(ret_c)
    print(f"  方案C(双向过滤): 年化={ann_c:.2%}, 夏普={sharpe_c:.2f}, "
          f"回撤={dd_c:.2%}, 降仓={(tover_filter_c < 1).sum()}次")

    all_results['换手率情绪-C双向过滤'] = {
        'ann_ret': ann_c, 'sharpe': sharpe_c, 'max_dd': dd_c,
        'ann_vol': vol_c, 'returns': ret_c, 'cum_wealth': cum_c
    }

    # 方案D: 换手率情绪 + MA市场过滤器 组合
    print("\n  方案D: 换手率情绪 + MA市场过滤器 组合")
    market_ret = df_ret_aligned.mean(axis=1)
    mf_ret, mf_scalar = apply_market_filter(base_ret, market_returns=market_ret)

    # 取方案C的换手率过滤
    combined_scalar = mf_scalar * tover_filter_c
    combined_scalar = combined_scalar.clip(lower=0.1)
    ret_d = base_ret * combined_scalar
    ann_d, sharpe_d, dd_d, vol_d, cum_d = calculate_metrics(ret_d)
    print(f"  方案D(MA+换手率组合): 年化={ann_d:.2%}, 夏普={sharpe_d:.2f}, "
          f"回撤={dd_d:.2%}, 降仓={(combined_scalar < 1).sum()}次")

    all_results['换手率情绪-D MA+换手率组合'] = {
        'ann_ret': ann_d, 'sharpe': sharpe_d, 'max_dd': dd_d,
        'ann_vol': vol_d, 'returns': ret_d, 'cum_wealth': cum_d
    }

    # ========================================================================
    # 实验4: 换手率加权 — 高换手率股票降低权重
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验4: 换手率加权 — 高换手率股票降低权重(交易成本更高)")
    print("=" * 80)

    # 换手率倒数加权: 高换手率→低权重
    tover_inv = 1.0 / tover_aligned.replace(0, np.nan)
    tover_inv = tover_inv.clip(lower=tover_inv.quantile(0.05),
                               upper=tover_inv.quantile(0.95), axis=1)

    ret_tw, turnover_tw, weights_tw = run_backtest(
        df_ret_aligned, K=6, TopK=100, standardization='Winsorization',
        weight_scheme='turnover_weighted', turnover_weights=tover_inv)

    ann_tw, sharpe_tw, dd_tw, vol_tw, cum_tw = calculate_metrics(ret_tw)
    turnover_tw_ann = turnover_tw.mean() * 12

    print(f"  换手率倒数加权: 年化={ann_tw:.2%}, 夏普={sharpe_tw:.2f}, "
          f"回撤={dd_tw:.2%}, 换手={turnover_tw_ann:.2%}")

    all_results['换手率倒数加权'] = {
        'ann_ret': ann_tw, 'sharpe': sharpe_tw, 'max_dd': dd_tw,
        'ann_vol': vol_tw, 'turnover': turnover_tw_ann,
        'returns': ret_tw, 'cum_wealth': cum_tw
    }

    # ========================================================================
    # 实验5: 最优组合 — 流动性过滤 + MA过滤器 + 极宽止损 + 真实成本
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验5: 最优组合 — 流动性过滤 + MA过滤器 + 极宽止损 + 真实成本")
    print("=" * 80)

    # 找最优流动性过滤阈值
    best_liq_name = None
    best_liq_sharpe = -999
    for key in all_results:
        if key.startswith('流动性过滤'):
            if all_results[key]['sharpe'] > best_liq_sharpe:
                best_liq_sharpe = all_results[key]['sharpe']
                best_liq_name = key

    print(f"  最优流动性过滤: {best_liq_name} (夏普={best_liq_sharpe:.2f})")

    # 用最优流动性过滤重跑
    best_threshold = int(best_liq_name.split('>')[1].split('%')[0])
    best_liq_filter = tover_aligned >= best_threshold

    combo_ret, combo_turnover, combo_weights = run_backtest(
        df_ret_aligned, K=6, TopK=100, standardization='Winsorization',
        liquidity_filter=best_liq_filter)

    # 叠加MA市场过滤器
    market_ret = df_ret_aligned.mean(axis=1)
    combo_mf_ret, combo_mf_scalar = apply_market_filter(combo_ret, market_returns=market_ret)

    # 叠加极宽止损
    combo_dd_ret, combo_dd_pos = apply_drawdown_control(combo_mf_ret)

    # 叠加真实成本
    combo_turnover_est = pd.Series(combo_turnover.mean() * 12 / 12, index=combo_dd_ret.index)
    combo_net, _, _ = apply_realistic_costs(combo_dd_ret, combo_turnover_est)

    combo_ann, combo_sharpe, combo_dd, combo_vol, combo_cum = calculate_metrics(combo_net)

    print(f"  最优组合: 年化={combo_ann:.2%}, 夏普={combo_sharpe:.2f}, "
          f"回撤={combo_dd:.2%}, 波动={combo_vol:.2%}")

    all_results['最优组合'] = {
        'ann_ret': combo_ann, 'sharpe': combo_sharpe, 'max_dd': combo_dd,
        'ann_vol': combo_vol, 'returns': combo_net, 'cum_wealth': combo_cum
    }

    # ========================================================================
    # 结果汇总
    # ========================================================================
    print("\n" + "=" * 80)
    print("换手率优化实验结果汇总")
    print("=" * 80)

    summary_rows = []
    for name, r in all_results.items():
        if isinstance(r, dict) and 'ann_ret' in r:
            summary_rows.append({
                '策略': name,
                '年化收益': f"{r['ann_ret']:.2%}",
                '夏普比率': f"{r['sharpe']:.2f}",
                '最大回撤': f"{r['max_dd']:.2%}",
                '年化波动率': f"{r.get('ann_vol', 0):.2%}",
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    # ========================================================================
    # 可视化
    # ========================================================================
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    # 图1: 流动性过滤对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    for name in ['基准'] + [f'流动性过滤(>{t}%)' for t in liquidity_thresholds]:
        if name in all_results and 'cum_wealth' in all_results[name]:
            all_results[name]['cum_wealth'].plot(ax=ax1, label=name, linewidth=1.5)
    ax1.set_title('流动性过滤: 累计净值对比', fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 图2: 换手率情绪对比
    ax2 = axes[0, 1]
    for name in ['基准', '换手率情绪-A低换手降仓', '换手率情绪-B高换手降仓',
                 '换手率情绪-C双向过滤', '换手率情绪-D MA+换手率组合']:
        if name in all_results and 'cum_wealth' in all_results[name]:
            all_results[name]['cum_wealth'].plot(ax=ax2, label=name, linewidth=1.5)
    ax2.set_title('换手率情绪: 累计净值对比', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 图3: 换手率因子对比
    ax3 = axes[1, 0]
    for name in ['基准'] + [f'因子:{n}' for n, _, _ in factor_configs]:
        if name in all_results and 'cum_wealth' in all_results[name]:
            all_results[name]['cum_wealth'].plot(ax=ax3, label=name, linewidth=1.5)
    ax3.set_title('换手率因子: 累计净值对比', fontsize=13)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 图4: 最优组合 vs 基准
    ax4 = axes[1, 1]
    for name in ['基准', '最优组合']:
        if name in all_results and 'cum_wealth' in all_results[name]:
            all_results[name]['cum_wealth'].plot(ax=ax4, label=name, linewidth=1.5)
    ax4.set_title('最优组合 vs 基准', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'turnover_optimization.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'turnover_optimization.png'}")
    plt.close()

    # 保存汇总
    summary_df.to_excel(output_dir / 'turnover_optimization_summary.xlsx', index=False)
    print(f"✓ 汇总表已保存: {output_dir / 'turnover_optimization_summary.xlsx'}")

    return all_results, summary_df


if __name__ == '__main__':
    results, summary = main()

    print("\n" + "=" * 80)
    print("换手率优化实验完成！")
    print("=" * 80)
