# -*- coding: utf-8 -*-
"""
Week 10: 全面优化实验
1. 信号层：波动率调整反转(特质反转) + 信号平滑 + 信号加权
2. 组合层：风险平价 + TopK动态化(信号阈值)
3. 风控层：双均线 + 指数MA + 波动率状态 + 时间止损 + 相关性止损
4. 换手优化：信号平滑 + 调仓阈值 + 月半调季全调
5. 实战层：可交易性过滤 + 冲击成本分段 + 参数网格
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


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
        return df_ret.rolling(12, min_periods=1).sum().shift(1)


# ============================================================
# 基础工具函数
# ============================================================

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
    turnover_ann = 0
    return ann_ret, sharpe, max_dd, ann_vol, cum_wealth, turnover_ann


def apply_standardization(signal_df):
    lower = signal_df.rolling(24, min_periods=6).quantile(0.01)
    upper = signal_df.rolling(24, min_periods=6).quantile(0.99)
    return signal_df.clip(lower=lower, upper=upper, axis=0)


def apply_market_filter(returns, market_returns=None, ma_window=12,
                        bear_scalar=0.5, bear_trend_scalar=0.3):
    if market_returns is None:
        market_returns = returns
    cum_market = (1 + market_returns).cumprod()
    ma = cum_market.rolling(ma_window).mean().shift(1)
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


def apply_vol_target(returns, target_vol=0.12, lookback=24, min_scalar=0.3, max_scalar=2.0):
    rolling_vol = returns.rolling(lookback).std().shift(1) * np.sqrt(12)
    vol_scalar = (target_vol / rolling_vol).clip(min_scalar, max_scalar)
    vol_scalar = vol_scalar.fillna(1.0)
    return returns * vol_scalar, vol_scalar


def run_backtest(df_ret, signal_df, topk=100, K=6, weight_method='inv_var',
                 vol_lookback=12, hold_buffer=0, stock_filter=None,
                 signal_threshold=None, cost_per_turn=0.0,
                 rebalance_mode='monthly'):
    """
    扩展回测引擎:
    - signal_threshold: 信号阈值，如0.8表示只选信号>80%分位数的股票
    - cost_per_turn: 每次换手的双边成本(bps)
    - rebalance_mode: 'monthly', 'half_monthly', 'monthly_quarterly'
    - weight_method: 'equal', 'inv_vol', 'inv_var', 'risk_parity', 'signal_weighted'
    """
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    prev_selected = None

    for date in df_ret.index[K:]:
        # 调仓模式控制
        date_idx = df_ret.index.get_loc(date)
        if rebalance_mode == 'quarterly' and date_idx % 3 != 0 and prev_selected is not None:
            # 季度全调：非季月保持持仓
            weights.loc[date] = weights.iloc[date_idx - 1] if date_idx > 0 else 0
            turnover_list.append(0.0)
            continue
        elif rebalance_mode == 'half_monthly' and date_idx % 2 != 0 and prev_selected is not None:
            # 双月调仓：非调仓月保持
            weights.loc[date] = weights.iloc[date_idx - 1] if date_idx > 0 else 0
            turnover_list.append(0.0)
            continue
        elif rebalance_mode == 'monthly_quarterly':
            # 月度调一半，季度全部重置
            if date_idx % 3 == 0:
                pass  # 季度月：全量调仓
            elif prev_selected is not None:
                # 非季月：只调换出信号变化最大的20%
                pass  # 简化处理：仍做正常调仓但buffer更大
                pass

        sig = signal_df.loc[date].copy()
        ranks = sig.rank(ascending=False, method='first')

        # 股票过滤
        if stock_filter is not None and date in stock_filter.index:
            valid_stocks = stock_filter.loc[date]
            sig = sig.where(valid_stocks, other=np.nan)
            ranks = sig.rank(ascending=False, method='first')

        # 选股：阈值 vs 固定TopK
        if signal_threshold is not None:
            pct_rank = sig.rank(pct=True, method='first')
            selected = (pct_rank >= signal_threshold).astype(float)
        else:
            if hold_buffer > 0 and prev_selected is not None:
                keep_mask = (ranks <= topk + hold_buffer) & prev_selected
                new_mask = (ranks <= topk) & ~keep_mask
                selected = (keep_mask | new_mask).astype(float)
            else:
                selected = (ranks <= topk).astype(float)

        selected = selected.where(sig.notna(), other=0.0)
        count = selected.sum()

        if count > 0:
            if weight_method == 'equal':
                weights.loc[date] = selected / count
            elif weight_method in ('inv_vol', 'inv_var'):
                start_idx = max(0, date_idx - vol_lookback)
                hist_ret = df_ret.iloc[start_idx:date_idx]
                stock_vols = hist_ret.std()
                if weight_method == 'inv_vol':
                    w_factor = 1.0 / stock_vols.replace(0, np.nan).fillna(1)
                else:
                    w_factor = 1.0 / (stock_vols ** 2).replace(0, np.nan).fillna(1)
                w = selected * w_factor
                w_sum = w.sum()
                if w_sum > 0:
                    weights.loc[date] = w / w_sum
                else:
                    weights.loc[date] = selected / count
            elif weight_method == 'risk_parity':
                # 风险平价：每只股票对组合的风险贡献相等
                start_idx = max(0, date_idx - vol_lookback)
                hist_ret = df_ret.iloc[start_idx:date_idx]
                stock_vols = hist_ret.std().replace(0, np.nan).fillna(1)
                # 简化风险平价：w_i ∝ 1/σ_i（与inv_vol相同但更稳健）
                w_factor = 1.0 / stock_vols
                w = selected * w_factor
                w_sum = w.sum()
                if w_sum > 0:
                    weights.loc[date] = w / w_sum
                else:
                    weights.loc[date] = selected / count
            elif weight_method == 'signal_weighted':
                # 信号加权：w_i ∝ signal_i / σ_i²
                start_idx = max(0, date_idx - vol_lookback)
                hist_ret = df_ret.iloc[start_idx:date_idx]
                stock_vols = hist_ret.std().replace(0, np.nan).fillna(1)
                sig_abs = sig.abs().replace(0, np.nan).fillna(sig.abs().mean())
                w_factor = sig_abs / (stock_vols ** 2)
                w = selected * w_factor
                w_sum = w.sum()
                if w_sum > 0:
                    weights.loc[date] = w / w_sum
                else:
                    weights.loc[date] = selected / count

        prev_selected = selected > 0

        if len(turnover_list) > 0 and date_idx > 0:
            prev_date = df_ret.index[date_idx - 1]
            turnover_list.append((weights.loc[date] - weights.loc[prev_date]).abs().sum() / 2)
        else:
            turnover_list.append(1.0)

    port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
    if len(turnover_list) < len(port_ret):
        turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
    elif len(turnover_list) > len(port_ret):
        turnover_list = turnover_list[-len(port_ret):]
    turnover_series = pd.Series(turnover_list, index=port_ret.index)

    # 扣交易成本
    if cost_per_turn > 0:
        cost_drag = turnover_series * cost_per_turn / 10000
        port_ret = port_ret - cost_drag

    return port_ret, turnover_series, weights


# ============================================================
# 实验1：信号层优化
# ============================================================

def experiment_signal(df_ret, K=6):
    print("\n" + "="*80)
    print("实验1：信号层优化")
    print("="*80)

    market_ret = df_ret.mean(axis=1)

    # --- 1A: 波动率调整反转(特质反转) ---
    print("\n--- 1A: 波动率调整反转(特质反转) ---")
    reversal_raw = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    value_signal = load_value_factor(df_ret)

    # 计算个股波动率
    stock_vol = df_ret.rolling(12, min_periods=6).std().shift(1)

    # 特质反转 = 反转 / 波动率
    reversal_vol_adj = reversal_raw / stock_vol.replace(0, np.nan)

    results = []

    # 原始反转
    sig_raw = 0.6 * apply_standardization(reversal_raw) + 0.4 * apply_standardization(value_signal)
    ret_raw, tov_raw, _ = run_backtest(df_ret, sig_raw, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=80)
    ret_raw, _ = apply_market_filter(ret_raw, market_ret)
    ret_raw, _ = apply_vol_target(ret_raw, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_raw)
    results.append(('原始反转(60R+40V)', ann, sharpe, dd, vol, tov_raw.mean()*12))

    # 特质反转(波动率调整)
    sig_vol_adj = 0.6 * apply_standardization(reversal_vol_adj.fillna(0)) + 0.4 * apply_standardization(value_signal)
    ret_vol, tov_vol, _ = run_backtest(df_ret, sig_vol_adj, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=80)
    ret_vol, _ = apply_market_filter(ret_vol, market_ret)
    ret_vol, _ = apply_vol_target(ret_vol, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_vol)
    results.append(('特质反转(波动率调整)', ann, sharpe, dd, vol, tov_vol.mean()*12))

    # 纯特质反转(无价值因子)
    sig_pure_vol = apply_standardization(reversal_vol_adj.fillna(0))
    ret_pure, tov_pure, _ = run_backtest(df_ret, sig_pure_vol, topk=100, K=K,
                                          weight_method='inv_var', hold_buffer=80)
    ret_pure, _ = apply_market_filter(ret_pure, market_ret)
    ret_pure, _ = apply_vol_target(ret_pure, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_pure)
    results.append(('纯特质反转(无价值)', ann, sharpe, dd, vol, tov_pure.mean()*12))

    # 特质反转(80R+20V)
    sig_vol_80 = 0.8 * apply_standardization(reversal_vol_adj.fillna(0)) + 0.2 * apply_standardization(value_signal)
    ret_vol80, tov_vol80, _ = run_backtest(df_ret, sig_vol_80, topk=100, K=K,
                                            weight_method='inv_var', hold_buffer=80)
    ret_vol80, _ = apply_market_filter(ret_vol80, market_ret)
    ret_vol80, _ = apply_vol_target(ret_vol80, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_vol80)
    results.append(('特质反转(80R+20V)', ann, sharpe, dd, vol, tov_vol80.mean()*12))

    print(f"\n{'配置':<25} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8} {'换手':>6}")
    print("-" * 65)
    for name, ann, sharpe, dd, vol, tov in results:
        print(f"{name:<25} {ann:>7.2f}% {sharpe:>6.2f} {dd:>7.2f}% {vol:>7.2f}% {tov:>5.0f}%")

    # --- 1B: 信号平滑 ---
    print("\n--- 1B: 信号平滑 ---")
    sig_base = 0.6 * apply_standardization(reversal_raw) + 0.4 * apply_standardization(value_signal)

    smooth_results = []
    for w0, w1, w2 in [(1.0, 0, 0), (0.7, 0.2, 0.1), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)]:
        if w1 == 0:
            sig_smooth = sig_base
            label = f"无平滑(1.0/0/0)"
        else:
            sig_smooth = w0 * sig_base + w1 * sig_base.shift(1).fillna(sig_base) + w2 * sig_base.shift(2).fillna(sig_base)
            label = f"平滑({w0}/{w1}/{w2})"

        ret_s, tov_s, _ = run_backtest(df_ret, sig_smooth, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=80)
        ret_s, _ = apply_market_filter(ret_s, market_ret)
        ret_s, _ = apply_vol_target(ret_s, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_s)
        smooth_results.append((label, ann, sharpe, dd, vol, tov_s.mean()*12))

    print(f"\n{'配置':<25} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8} {'换手':>6}")
    print("-" * 65)
    for name, ann, sharpe, dd, vol, tov in smooth_results:
        print(f"{name:<25} {ann:>7.2f}% {sharpe:>6.2f} {dd:>7.2f}% {vol:>7.2f}% {tov:>5.0f}%")

    return results, smooth_results


# ============================================================
# 实验2：组合层优化
# ============================================================

def experiment_portfolio(df_ret, K=6):
    print("\n" + "="*80)
    print("实验2：组合层优化")
    print("="*80)

    market_ret = df_ret.mean(axis=1)
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    value_signal = load_value_factor(df_ret)
    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    # --- 2A: 权重方案 ---
    print("\n--- 2A: 权重方案对比 ---")
    weight_results = []

    for wm, label in [('equal', '等权'), ('inv_vol', '1/σ(风险平价)'),
                       ('inv_var', '1/σ²(最小方差)'), ('signal_weighted', '信号加权(α/σ²)')]:
        ret_w, tov_w, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                        weight_method=wm, hold_buffer=80)
        ret_w, _ = apply_market_filter(ret_w, market_ret)
        ret_w, _ = apply_vol_target(ret_w, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_w)
        weight_results.append((label, ann, sharpe, dd, vol, tov_w.mean()*12))

    print(f"\n{'权重方案':<25} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8} {'换手':>6}")
    print("-" * 65)
    for name, ann, sharpe, dd, vol, tov in weight_results:
        print(f"{name:<25} {ann:>7.2f}% {sharpe:>6.2f} {dd:>7.2f}% {vol:>7.2f}% {tov:>5.0f}%")

    # --- 2B: TopK动态化(信号阈值) ---
    print("\n--- 2B: TopK动态化(信号阈值 vs 固定数量) ---")
    topk_results = []

    # 固定TopK
    for topk in [80, 100, 120]:
        ret_k, tov_k, _ = run_backtest(df_ret, combined_signal, topk=topk, K=K,
                                        weight_method='inv_var', hold_buffer=80)
        ret_k, _ = apply_market_filter(ret_k, market_ret)
        ret_k, _ = apply_vol_target(ret_k, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_k)
        topk_results.append((f'固定TopK={topk}', ann, sharpe, dd, vol, tov_k.mean()*12))

    # 信号阈值
    for threshold in [0.90, 0.85, 0.80]:
        ret_t, tov_t, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=80,
                                        signal_threshold=threshold)
        ret_t, _ = apply_market_filter(ret_t, market_ret)
        ret_t, _ = apply_vol_target(ret_t, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_t)
        topk_results.append((f'信号阈值>{threshold:.0%}', ann, sharpe, dd, vol, tov_t.mean()*12))

    print(f"\n{'配置':<25} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8} {'换手':>6}")
    print("-" * 65)
    for name, ann, sharpe, dd, vol, tov in topk_results:
        print(f"{name:<25} {ann:>7.2f}% {sharpe:>6.2f} {dd:>7.2f}% {vol:>7.2f}% {tov:>5.0f}%")

    return weight_results, topk_results


# ============================================================
# 实验3：风控层优化
# ============================================================

def experiment_risk_control(df_ret, K=6):
    print("\n" + "="*80)
    print("实验3：风控层优化")
    print("="*80)

    market_ret = df_ret.mean(axis=1)
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    value_signal = load_value_factor(df_ret)
    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    # 基准回测（无风控）
    ret_base, tov_base, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                          weight_method='inv_var', hold_buffer=80)

    risk_results = []

    # --- 3A: MA过滤变体 ---
    print("\n--- 3A: MA过滤变体 ---")

    # 原始单均线
    ret_ma1, _ = apply_market_filter(ret_base, market_ret, ma_window=12)
    ret_ma1, _ = apply_vol_target(ret_ma1, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_ma1)
    risk_results.append(('单均线(MA12)', ann, sharpe, dd, vol))

    # 双均线交叉
    cum_market = (1 + market_ret).cumprod()
    ma_short = cum_market.rolling(6).mean().shift(1)
    ma_long = cum_market.rolling(24).mean().shift(1)
    position_dual = pd.Series(1.0, index=ret_base.index)
    for date in ret_base.index:
        if pd.isna(ma_short.loc[date]) or pd.isna(ma_long.loc[date]):
            continue
        if ma_short.loc[date] < ma_long.loc[date]:
            # 短均线在长均线下方
            slope = ma_long.diff(3).loc[date] if pd.notna(ma_long.diff(3).loc[date]) else 0
            if slope < 0:
                position_dual.loc[date] = 0.3
            else:
                position_dual.loc[date] = 0.5
    ret_dual = ret_base * position_dual
    ret_dual, _ = apply_vol_target(ret_dual, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_dual)
    risk_results.append(('双均线(MA6/MA24)', ann, sharpe, dd, vol))

    # 用市场收益做MA（非组合净值）
    ret_mkt_ma, _ = apply_market_filter(ret_base, market_ret, ma_window=12)
    ret_mkt_ma, _ = apply_vol_target(ret_mkt_ma, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_mkt_ma)
    risk_results.append(('市场收益MA(已实现)', ann, sharpe, dd, vol))

    # --- 3B: 波动率状态判断 ---
    print("\n--- 3B: 波动率状态判断 ---")

    # 市场波动率飙升时降仓
    mkt_vol = market_ret.rolling(12).std().shift(1) * np.sqrt(12)
    mkt_vol_median = mkt_vol.expanding().median()  # 扩展窗口中位数避免前视偏差
    vol_position = pd.Series(1.0, index=ret_base.index)
    for date in ret_base.index:
        if pd.isna(mkt_vol.loc[date]) or pd.isna(mkt_vol_median.loc[date]):
            continue
        if mkt_vol.loc[date] > mkt_vol_median.loc[date] * 1.5:
            vol_position.loc[date] = 0.5  # 波动率飙升，降仓50%
        elif mkt_vol.loc[date] > mkt_vol_median.loc[date] * 2.0:
            vol_position.loc[date] = 0.3

    ret_vol_state = ret_base * vol_position
    ret_vol_state, _ = apply_vol_target(ret_vol_state, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_vol_state)
    risk_results.append(('波动率状态降仓', ann, sharpe, dd, vol))

    # MA + 波动率状态组合
    ret_combo, pos_ma = apply_market_filter(ret_base, market_ret, ma_window=12)
    ret_combo = ret_combo * vol_position
    ret_combo, _ = apply_vol_target(ret_combo, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_combo)
    risk_results.append(('MA+波动率状态', ann, sharpe, dd, vol))

    # --- 3C: 时间止损 ---
    print("\n--- 3C: 时间止损(连续跑输基准) ---")

    # 连续3个月跑输市场 → 降仓50%
    market_ret_aligned = market_ret.reindex(ret_base.index).fillna(0)
    underperform = ret_base < market_ret_aligned
    consec_under = underperform.rolling(3).sum().shift(1)  # shift(1)避免使用前视偏差
    time_stop_pos = pd.Series(1.0, index=ret_base.index)
    for date in ret_base.index:
        if pd.isna(consec_under.loc[date]):
            continue
        if consec_under.loc[date] >= 3:
            time_stop_pos.loc[date] = 0.5

    ret_time_stop = ret_base * time_stop_pos
    ret_time_stop, _ = apply_market_filter(ret_time_stop, market_ret)
    ret_time_stop, _ = apply_vol_target(ret_time_stop, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_time_stop)
    risk_results.append(('时间止损(3月跑输)', ann, sharpe, dd, vol))

    # 连续6个月跑输 → 降仓30%
    consec_under6 = underperform.rolling(6).sum().shift(1)  # shift(1)避免使用前视偏差
    time_stop_pos6 = pd.Series(1.0, index=ret_base.index)
    for date in ret_base.index:
        if pd.isna(consec_under6.loc[date]):
            continue
        if consec_under6.loc[date] >= 4:
            time_stop_pos6.loc[date] = 0.3
        elif consec_under6.loc[date] >= 3:
            time_stop_pos6.loc[date] = 0.5

    ret_time_stop6 = ret_base * time_stop_pos6
    ret_time_stop6, _ = apply_market_filter(ret_time_stop6, market_ret)
    ret_time_stop6, _ = apply_vol_target(ret_time_stop6, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_time_stop6)
    risk_results.append(('时间止损(渐进)', ann, sharpe, dd, vol))

    # --- 3D: 相关性止损 ---
    print("\n--- 3D: 相关性止损 ---")

    # 计算持仓股票间平均相关性
    # 简化：用全市场平均相关性
    corr_window = 12
    avg_corr = pd.Series(index=ret_base.index, dtype=float)
    for date in ret_base.index:
        date_idx = ret_base.index.get_loc(date)
        if date_idx < corr_window:
            avg_corr.loc[date] = 0.3
            continue
        window_ret = df_ret.iloc[date_idx-corr_window:date_idx]
        # 随机抽样100只股票计算相关性
        sample_cols = np.random.choice(df_ret.columns, min(100, len(df_ret.columns)), replace=False)
        corr_mat = window_ret[sample_cols].corr()
        # 取上三角均值
        mask = np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
        avg_corr.loc[date] = corr_mat.values[mask].mean()

    # 相关性飙升时降仓
    corr_median = avg_corr.rolling(24, min_periods=6).median().shift(1)
    corr_position = pd.Series(1.0, index=ret_base.index)
    for date in ret_base.index:
        if pd.isna(avg_corr.loc[date]) or pd.isna(corr_median.loc[date]):
            continue
        if avg_corr.loc[date] > corr_median.loc[date] * 1.5:
            corr_position.loc[date] = 0.5

    ret_corr_stop = ret_base * corr_position
    ret_corr_stop, _ = apply_market_filter(ret_corr_stop, market_ret)
    ret_corr_stop, _ = apply_vol_target(ret_corr_stop, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_corr_stop)
    risk_results.append(('相关性止损', ann, sharpe, dd, vol))

    print(f"\n{'风控方案':<25} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8}")
    print("-" * 55)
    for name, ann, sharpe, dd, vol in risk_results:
        print(f"{name:<25} {ann:>7.2f}% {sharpe:>6.2f} {dd:>7.2f}% {vol:>7.2f}%")

    return risk_results


# ============================================================
# 实验4：换手率优化
# ============================================================

def experiment_turnover(df_ret, K=6):
    print("\n" + "="*80)
    print("实验4：换手率优化")
    print("="*80)

    market_ret = df_ret.mean(axis=1)
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    value_signal = load_value_factor(df_ret)
    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    turnover_results = []

    # 基准
    ret_base, tov_base, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                          weight_method='inv_var', hold_buffer=80,
                                          cost_per_turn=20)
    ret_base, _ = apply_market_filter(ret_base, market_ret)
    ret_base, _ = apply_vol_target(ret_base, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_base)
    turnover_results.append(('基准(buffer=80)', ann, sharpe, dd, vol, tov_base.mean()*12))

    # --- 4A: 信号平滑 ---
    print("\n--- 4A: 信号平滑(换手优化) ---")
    for w0, w1, w2 in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2)]:
        sig_smooth = w0 * combined_signal + w1 * combined_signal.shift(1).fillna(combined_signal) + w2 * combined_signal.shift(2).fillna(combined_signal)
        ret_s, tov_s, _ = run_backtest(df_ret, sig_smooth, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=80,
                                        cost_per_turn=20)
        ret_s, _ = apply_market_filter(ret_s, market_ret)
        ret_s, _ = apply_vol_target(ret_s, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_s)
        turnover_results.append((f'信号平滑({w0}/{w1}/{w2})', ann, sharpe, dd, vol, tov_s.mean()*12))

    # --- 4B: 调仓阈值 ---
    print("\n--- 4B: 调仓阈值(信号变化>阈值才调) ---")
    for threshold_pct in [0.1, 0.2, 0.3]:
        # 简化实现：增大buffer
        buffer_val = int(80 + threshold_pct * 100)
        ret_t, tov_t, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=buffer_val,
                                        cost_per_turn=20)
        ret_t, _ = apply_market_filter(ret_t, market_ret)
        ret_t, _ = apply_vol_target(ret_t, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_t)
        turnover_results.append((f'Buffer={buffer_val}(阈值{threshold_pct:.0%})', ann, sharpe, dd, vol, tov_t.mean()*12))

    # --- 4C: 月半调+季度全调 ---
    print("\n--- 4C: 月半调+季度全调 ---")
    # 月度调一半（buffer更大）+ 季度全调（buffer=0）
    # 简化：月度用大buffer，季度用小buffer
    ret_mq, tov_mq, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                      weight_method='inv_var', hold_buffer=120,
                                      cost_per_turn=20)
    ret_mq, _ = apply_market_filter(ret_mq, market_ret)
    ret_mq, _ = apply_vol_target(ret_mq, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_mq)
    turnover_results.append(('月半调(buffer=120)', ann, sharpe, dd, vol, tov_mq.mean()*12))

    print(f"\n{'配置':<30} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8} {'换手':>6}")
    print("-" * 70)
    for name, ann, sharpe, dd, vol, tov in turnover_results:
        print(f"{name:<30} {ann:>7.2f}% {sharpe:>6.2f} {dd:>7.2f}% {vol:>7.2f}% {tov:>5.0f}%")

    return turnover_results


# ============================================================
# 实验5：实战层优化
# ============================================================

def experiment_practical(df_ret, K=6):
    print("\n" + "="*80)
    print("实验5：实战层优化")
    print("="*80)

    market_ret = df_ret.mean(axis=1)
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    value_signal = load_value_factor(df_ret)
    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    practical_results = []

    # --- 5A: 可交易性过滤 ---
    print("\n--- 5A: 可交易性过滤 ---")

    # 次新股过滤：上市不满12个月的股票
    # 用数据覆盖率：过去12个月有数据的股票
    coverage = df_ret.rolling(12, min_periods=10).count().shift(1)
    is_not_new = coverage >= 10  # 至少10个月有数据

    # 流动性过滤：月度收益非零（非停牌）
    is_tradable = df_ret.shift(1).abs() > 0.001  # 上月有交易

    # 组合过滤
    tradeable_filter = is_not_new & is_tradable

    ret_tf, tov_tf, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                      weight_method='inv_var', hold_buffer=80,
                                      stock_filter=tradeable_filter)
    ret_tf, _ = apply_market_filter(ret_tf, market_ret)
    ret_tf, _ = apply_vol_target(ret_tf, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_tf)
    practical_results.append(('可交易性过滤(非次新+非停牌)', ann, sharpe, dd, vol, tov_tf.mean()*12))

    # 基准对比
    ret_base, tov_base, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                          weight_method='inv_var', hold_buffer=80)
    ret_base, _ = apply_market_filter(ret_base, market_ret)
    ret_base, _ = apply_vol_target(ret_base, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_base)
    practical_results.append(('基准(无过滤)', ann, sharpe, dd, vol, tov_base.mean()*12))

    # --- 5B: 冲击成本分段建模 ---
    print("\n--- 5B: 冲击成本分段建模 ---")

    # 市值因子：高波动≈小市值
    stock_vol = df_ret.rolling(12, min_periods=6).std().shift(1)
    vol_ranks = stock_vol.rank(pct=True, axis=1)

    # 分段成本：低波动(Top30%)=10bps, 中波动=25bps, 高波动(Bottom30%)=50bps
    # 简化：用平均加权成本
    for cost_level in [15, 20, 30, 40]:
        ret_c, tov_c, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=80,
                                        cost_per_turn=cost_level)
        ret_c, _ = apply_market_filter(ret_c, market_ret)
        ret_c, _ = apply_vol_target(ret_c, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_c)
        practical_results.append((f'固定成本{cost_level}bps', ann, sharpe, dd, vol, tov_c.mean()*12))

    # 分段成本（更精细）
    # 每只股票根据波动率分位给不同成本
    ret_sc, tov_sc, weights_sc = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                                weight_method='inv_var', hold_buffer=80)
    # 计算每期加权平均成本
    cost_series = pd.Series(index=ret_sc.index, dtype=float)
    for date in ret_sc.index:
        if date not in vol_ranks.index:
            cost_series.loc[date] = 0
            continue
        w = weights_sc.loc[date]
        vr = vol_ranks.loc[date]
        # 低波动(Top30%)=10bps, 中=25bps, 高(Bottom30%)=50bps
        stock_cost = pd.Series(25, index=w.index)  # 默认25bps
        stock_cost[vr >= 0.70] = 10  # 低波动=大市值
        stock_cost[vr <= 0.30] = 50  # 高波动=小市值
        weighted_cost = (w * stock_cost / 10000).sum()
        # 用换手率调整
        if date in tov_sc.index:
            cost_series.loc[date] = weighted_cost * tov_sc.loc[date] * 2
        else:
            cost_series.loc[date] = 0

    ret_sc_adj = ret_sc - cost_series
    ret_sc_adj, _ = apply_market_filter(ret_sc_adj, market_ret)
    ret_sc_adj, _ = apply_vol_target(ret_sc_adj, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_sc_adj)
    practical_results.append(('分段成本(10/25/50bps)', ann, sharpe, dd, vol, tov_sc.mean()*12))

    # --- 5C: 参数网格 ---
    print("\n--- 5C: 参数网格(多策略稳健性) ---")

    grid_results = []
    param_combos = [
        (6, 100, 12, 0.12),
        (6, 80, 12, 0.12),
        (6, 120, 12, 0.12),
        (3, 100, 12, 0.12),
        (9, 100, 12, 0.12),
        (6, 100, 6, 0.12),
        (6, 100, 24, 0.12),
        (6, 100, 12, 0.10),
        (6, 100, 12, 0.15),
    ]

    for rev_k, topk, ma_w, vt_target in param_combos:
        rev_sig = -df_ret.rolling(rev_k, min_periods=1).sum().shift(1)
        val_sig = load_value_factor(df_ret)
        sig = 0.6 * apply_standardization(rev_sig) + 0.4 * apply_standardization(val_sig)

        ret_g, tov_g, _ = run_backtest(df_ret, sig, topk=topk, K=rev_k,
                                        weight_method='inv_var', hold_buffer=80)
        ret_g, _ = apply_market_filter(ret_g, market_ret, ma_window=ma_w)
        ret_g, _ = apply_vol_target(ret_g, target_vol=vt_target)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_g)
        grid_results.append({
            'K': rev_k, 'TopK': topk, 'MA': ma_w, 'VT': vt_target,
            'ann': ann, 'sharpe': sharpe, 'dd': dd
        })

    print(f"\n{'K':>3} {'TopK':>5} {'MA':>4} {'VT':>5} {'年化':>8} {'夏普':>6} {'回撤':>8}")
    print("-" * 45)
    for g in grid_results:
        print(f"{g['K']:>3} {g['TopK']:>5} {g['MA']:>4} {g['VT']:>5.2f} "
              f"{g['ann']:>7.2f}% {g['sharpe']:>6.2f} {g['dd']:>7.2f}%")

    # 参数网格统计
    sharpes = [g['sharpe'] for g in grid_results]
    anns = [g['ann'] for g in grid_results]
    print(f"\n  参数网格统计: 夏普均值={np.mean(sharpes):.2f}, "
          f"夏普范围=[{np.min(sharpes):.2f}, {np.max(sharpes):.2f}], "
          f"年化均值={np.mean(anns):.2f}%")

    print(f"\n{'配置':<30} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8} {'换手':>6}")
    print("-" * 70)
    for name, ann, sharpe, dd, vol, tov in practical_results:
        print(f"{name:<30} {ann:>7.2f}% {sharpe:>6.2f} {dd:>7.2f}% {vol:>7.2f}% {tov:>5.0f}%")

    return practical_results, grid_results


# ============================================================
# 可视化
# ============================================================

def create_visualization(signal_results, smooth_results, weight_results,
                        risk_results, turnover_results, grid_results):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Week 10: 全面优化实验', fontsize=16, fontweight='bold')

    # 图1: 特质反转 vs 原始反转
    ax = axes[0, 0]
    names = [r[0][:12] for r in signal_results]
    sharpes = [r[2] for r in signal_results]
    colors = ['#3498db' if '原始' in r[0] else '#e74c3c' for r in signal_results]
    ax.barh(names, sharpes, color=colors)
    ax.set_xlabel('夏普比率')
    ax.set_title('信号层: 特质反转 vs 原始反转')
    ax.grid(True, alpha=0.3)

    # 图2: 权重方案
    ax = axes[0, 1]
    names = [r[0] for r in weight_results]
    sharpes = [r[2] for r in weight_results]
    ax.barh(names, sharpes, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax.set_xlabel('夏普比率')
    ax.set_title('组合层: 权重方案对比')
    ax.grid(True, alpha=0.3)

    # 图3: 风控方案
    ax = axes[0, 2]
    names = [r[0][:14] for r in risk_results]
    sharpes = [r[2] for r in risk_results]
    ax.barh(names, sharpes, color='#f39c12')
    ax.set_xlabel('夏普比率')
    ax.set_title('风控层: 各方案对比')
    ax.grid(True, alpha=0.3)

    # 图4: 信号平滑
    ax = axes[1, 0]
    names = [r[0][:15] for r in smooth_results]
    sharpes = [r[2] for r in smooth_results]
    tovs = [r[5] for r in smooth_results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, sharpes, width, label='夏普', color='#2ecc71')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, tovs, width, label='换手率', color='#e74c3c', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('夏普比率', color='#2ecc71')
    ax2.set_ylabel('年化换手率', color='#e74c3c')
    ax.set_title('信号平滑: 夏普 vs 换手率')
    ax.grid(True, alpha=0.3)

    # 图5: 换手优化
    ax = axes[1, 1]
    names = [r[0][:18] for r in turnover_results]
    sharpes = [r[2] for r in turnover_results]
    tovs = [r[5] for r in turnover_results]
    ax.scatter(tovs, sharpes, s=100, c='#9b59b6', zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (tovs[i], sharpes[i]), fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('年化换手率(%)')
    ax.set_ylabel('夏普比率')
    ax.set_title('换手优化: 夏普-换手前沿')
    ax.grid(True, alpha=0.3)

    # 图6: 参数网格
    ax = axes[1, 2]
    for g in grid_results:
        ax.scatter(g['ann'], g['sharpe'], s=80, c='#3498db', alpha=0.7)
    ax.set_xlabel('年化收益(%)')
    ax.set_ylabel('夏普比率')
    ax.set_title(f'参数网格: {len(grid_results)}组参数')
    best = max(grid_results, key=lambda x: x['sharpe'])
    ax.annotate(f"最优K={best['K']},TopK={best['TopK']},MA={best['MA']}",
                (best['ann'], best['sharpe']), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='red'),
                xytext=(best['ann']-0.02, best['sharpe']+0.05))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'week10_comprehensive_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: results/week10_comprehensive_optimization.png")


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*80)
    print("Week 10: 全面优化实验")
    print("="*80)

    # 加载收益数据
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', engine='openpyxl')
    if 'Stkcd' in df_ret.columns:
        df_ret = df_ret.pivot(index='Trdmnt', columns='Stkcd', values='Mretwd')
    else:
        df_ret = df_ret.set_index('Trdmnt')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret = df_ret.sort_index()
    valid_stocks = df_ret.notna().sum() > len(df_ret) * 0.5
    df_ret = df_ret.loc[:, valid_stocks]
    df_ret = df_ret.fillna(0)
    print(f"收益数据: {df_ret.shape}, {df_ret.index[0]}~{df_ret.index[-1]}")

    K = 6

    # 实验1：信号层
    signal_results, smooth_results = experiment_signal(df_ret, K=K)

    # 实验2：组合层
    weight_results, topk_results = experiment_portfolio(df_ret, K=K)

    # 实验3：风控层
    risk_results = experiment_risk_control(df_ret, K=K)

    # 实验4：换手优化
    turnover_results = experiment_turnover(df_ret, K=K)

    # 实验5：实战层
    practical_results, grid_results = experiment_practical(df_ret, K=K)

    # 可视化
    create_visualization(signal_results, smooth_results, weight_results,
                        risk_results, turnover_results, grid_results)

    # 汇总
    print("\n" + "="*80)
    print("Week 10 实验汇总")
    print("="*80)

    print("\n--- 信号层 ---")
    print("  特质反转 vs 原始反转: 波动率标准化后的反转信号效果")

    print("\n--- 组合层 ---")
    print("  风险平价 vs 最小方差 vs 信号加权: 不同权重方案对比")
    print("  信号阈值 vs 固定TopK: 动态选股数量")

    print("\n--- 风控层 ---")
    print("  双均线/波动率状态/时间止损/相关性止损: 各风控方案对比")

    print("\n--- 换手优化 ---")
    print("  信号平滑/调仓阈值/月半调: 换手率-夏普前沿")

    print("\n--- 实战层 ---")
    print("  可交易性过滤/分段成本/参数网格: 实盘可行性")


if __name__ == '__main__':
    results = main()
