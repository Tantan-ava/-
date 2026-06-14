# -*- coding: utf-8 -*-
"""
Week 10B: 四大优化集成验证
1. MA6替代MA12
2. 时间止损(3月跑输渐进版)
3. 信号阈值>85%
4. 可交易性过滤(非次新+非停牌)
测试：单独效果 + 逐步叠加 + 全部集成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


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


def apply_time_stop(returns, market_returns, consec_months=3, scalar_gradual=True):
    """时间止损：连续跑输市场后降仓"""
    market_ret_aligned = market_returns.reindex(returns.index).fillna(0)
    underperform = returns < market_ret_aligned
    consec_under = underperform.rolling(consec_months).sum().shift(1)
    time_stop_pos = pd.Series(1.0, index=returns.index)
    for date in returns.index:
        if pd.isna(consec_under.loc[date]):
            continue
        if scalar_gradual:
            if consec_under.loc[date] >= 4:
                time_stop_pos.loc[date] = 0.3
            elif consec_under.loc[date] >= 3:
                time_stop_pos.loc[date] = 0.5
        else:
            if consec_under.loc[date] >= 3:
                time_stop_pos.loc[date] = 0.5
    return returns * time_stop_pos, time_stop_pos


def run_backtest(df_ret, signal_df, topk=100, K=6, weight_method='inv_var',
                 vol_lookback=12, hold_buffer=80, stock_filter=None,
                 signal_threshold=None, cost_per_turn=0.0):
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    prev_selected = None
    stock_counts = []

    for date in df_ret.index[K:]:
        date_idx = df_ret.index.get_loc(date)

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
        stock_counts.append(count)

        if count > 0:
            if weight_method == 'equal':
                weights.loc[date] = selected / count
            elif weight_method in ('inv_vol', 'inv_var', 'signal_weighted'):
                start_idx = max(0, date_idx - vol_lookback)
                hist_ret = df_ret.iloc[start_idx:date_idx]
                stock_vols = hist_ret.std()
                if weight_method == 'inv_vol':
                    w_factor = 1.0 / stock_vols.replace(0, np.nan).fillna(1)
                elif weight_method == 'inv_var':
                    w_factor = 1.0 / (stock_vols ** 2).replace(0, np.nan).fillna(1)
                elif weight_method == 'signal_weighted':
                    sig_abs = sig.abs().replace(0, np.nan).fillna(sig.abs().mean())
                    w_factor = sig_abs / (stock_vols ** 2).replace(0, np.nan).fillna(1)
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
    turnover_list = turnover_list[-len(port_ret):]
    turnover_series = pd.Series(turnover_list, index=port_ret.index)

    if cost_per_turn > 0:
        cost_drag = turnover_series * cost_per_turn / 10000
        port_ret = port_ret - cost_drag

    avg_stocks = np.mean(stock_counts) if stock_counts else topk
    return port_ret, turnover_series, weights, avg_stocks


def build_tradeable_filter(df_ret, min_history=10):
    """可交易性过滤：剔除次新股和停牌股"""
    coverage = df_ret.rolling(12, min_periods=min_history).count().shift(1)
    is_not_new = coverage >= min_history
    is_tradable = df_ret.shift(1).abs() > 0.001
    return is_not_new & is_tradable


# ============================================================
# 主实验
# ============================================================

def main():
    print("="*80)
    print("Week 10B: 四大优化集成验证")
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
    market_ret = df_ret.mean(axis=1)

    # 构建信号
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    value_signal = load_value_factor(df_ret)
    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    # 可交易性过滤
    tradeable_filter = build_tradeable_filter(df_ret, min_history=10)

    # ============================================================
    # 实验矩阵：逐步叠加四大优化
    # ============================================================
    print("\n" + "="*80)
    print("实验矩阵：单独效果 + 逐步叠加 + 全部集成")
    print("="*80)

    results = []

    # --- 基准配置 ---
    print("\n[0] 基准配置: 反转60%+价值40%, TopK=100, MA12, VT12%, buffer=80")
    ret0, tov0, _, avg0 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80)
    ret0_ma, _ = apply_market_filter(ret0, market_ret, ma_window=12)
    ret0_final, _ = apply_vol_target(ret0_ma, target_vol=0.12)
    ann0, sharpe0, dd0, vol0, cum0, _ = calculate_metrics(ret0_final)
    results.append(('基准(MA12+TopK100+无过滤+无止损)', ann0, sharpe0, dd0, vol0, tov0.mean()*12, avg0))

    # --- 优化1: MA6 ---
    print("[1] +MA6替代MA12")
    ret1, tov1, _, avg1 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80)
    ret1_ma, _ = apply_market_filter(ret1, market_ret, ma_window=6)
    ret1_final, _ = apply_vol_target(ret1_ma, target_vol=0.12)
    ann1, sharpe1, dd1, vol1, cum1, _ = calculate_metrics(ret1_final)
    results.append(('基准+MA6', ann1, sharpe1, dd1, vol1, tov1.mean()*12, avg1))

    # --- 优化2: 信号阈值>85% ---
    print("[2] +信号阈值>85%")
    ret2, tov2, _, avg2 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80,
                                         signal_threshold=0.85)
    ret2_ma, _ = apply_market_filter(ret2, market_ret, ma_window=12)
    ret2_final, _ = apply_vol_target(ret2_ma, target_vol=0.12)
    ann2, sharpe2, dd2, vol2, cum2, _ = calculate_metrics(ret2_final)
    results.append(('基准+信号阈值85%', ann2, sharpe2, dd2, vol2, tov2.mean()*12, avg2))

    # --- 优化3: 可交易性过滤 ---
    print("[3] +可交易性过滤")
    ret3, tov3, _, avg3 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80,
                                         stock_filter=tradeable_filter)
    ret3_ma, _ = apply_market_filter(ret3, market_ret, ma_window=12)
    ret3_final, _ = apply_vol_target(ret3_ma, target_vol=0.12)
    ann3, sharpe3, dd3, vol3, cum3, _ = calculate_metrics(ret3_final)
    results.append(('基准+可交易性过滤', ann3, sharpe3, dd3, vol3, tov3.mean()*12, avg3))

    # --- 优化4: 时间止损(渐进) ---
    print("[4] +时间止损(渐进)")
    ret4, tov4, _, avg4 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80)
    ret4_ma, _ = apply_market_filter(ret4, market_ret, ma_window=12)
    ret4_ts, _ = apply_time_stop(ret4_ma, market_ret, consec_months=3, scalar_gradual=True)
    ret4_final, _ = apply_vol_target(ret4_ts, target_vol=0.12)
    ann4, sharpe4, dd4, vol4, cum4, _ = calculate_metrics(ret4_final)
    results.append(('基准+时间止损(渐进)', ann4, sharpe4, dd4, vol4, tov4.mean()*12, avg4))

    # --- 逐步叠加 ---
    print("\n--- 逐步叠加 ---")

    # MA6 + 信号阈值85%
    print("[5] MA6 + 信号阈值85%")
    ret5, tov5, _, avg5 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80,
                                         signal_threshold=0.85)
    ret5_ma, _ = apply_market_filter(ret5, market_ret, ma_window=6)
    ret5_final, _ = apply_vol_target(ret5_ma, target_vol=0.12)
    ann5, sharpe5, dd5, vol5, cum5, _ = calculate_metrics(ret5_final)
    results.append(('MA6+信号阈值85%', ann5, sharpe5, dd5, vol5, tov5.mean()*12, avg5))

    # MA6 + 信号阈值85% + 可交易性过滤
    print("[6] MA6 + 信号阈值85% + 可交易性过滤")
    ret6, tov6, _, avg6 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80,
                                         signal_threshold=0.85,
                                         stock_filter=tradeable_filter)
    ret6_ma, _ = apply_market_filter(ret6, market_ret, ma_window=6)
    ret6_final, _ = apply_vol_target(ret6_ma, target_vol=0.12)
    ann6, sharpe6, dd6, vol6, cum6, _ = calculate_metrics(ret6_final)
    results.append(('MA6+阈值85%+可交易性', ann6, sharpe6, dd6, vol6, tov6.mean()*12, avg6))

    # MA6 + 信号阈值85% + 可交易性过滤 + 时间止损
    print("[7] 全部集成: MA6 + 信号阈值85% + 可交易性过滤 + 时间止损")
    ret7, tov7, _, avg7 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                         weight_method='inv_var', hold_buffer=80,
                                         signal_threshold=0.85,
                                         stock_filter=tradeable_filter)
    ret7_ma, _ = apply_market_filter(ret7, market_ret, ma_window=6)
    ret7_ts, _ = apply_time_stop(ret7_ma, market_ret, consec_months=3, scalar_gradual=True)
    ret7_final, _ = apply_vol_target(ret7_ts, target_vol=0.12)
    ann7, sharpe7, dd7, vol7, cum7, _ = calculate_metrics(ret7_final)
    results.append(('全部集成', ann7, sharpe7, dd7, vol7, tov7.mean()*12, avg7))

    # --- 不同止损参数 ---
    print("\n--- 止损参数敏感性 ---")

    # 全部集成 + 3月简单止损
    ret7b, tov7b, _, avg7b = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                            weight_method='inv_var', hold_buffer=80,
                                            signal_threshold=0.85,
                                            stock_filter=tradeable_filter)
    ret7b_ma, _ = apply_market_filter(ret7b, market_ret, ma_window=6)
    ret7b_ts, _ = apply_time_stop(ret7b_ma, market_ret, consec_months=3, scalar_gradual=False)
    ret7b_final, _ = apply_vol_target(ret7b_ts, target_vol=0.12)
    ann7b, sharpe7b, dd7b, vol7b, cum7b, _ = calculate_metrics(ret7b_final)
    results.append(('全部集成(简单止损)', ann7b, sharpe7b, dd7b, vol7b, tov7b.mean()*12, avg7b))

    # 全部集成 + 2月止损
    ret7c, tov7c, _, avg7c = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                            weight_method='inv_var', hold_buffer=80,
                                            signal_threshold=0.85,
                                            stock_filter=tradeable_filter)
    ret7c_ma, _ = apply_market_filter(ret7c, market_ret, ma_window=6)
    ret7c_ts, _ = apply_time_stop(ret7c_ma, market_ret, consec_months=2, scalar_gradual=True)
    ret7c_final, _ = apply_vol_target(ret7c_ts, target_vol=0.12)
    ann7c, sharpe7c, dd7c, vol7c, cum7c, _ = calculate_metrics(ret7c_final)
    results.append(('全部集成(2月止损渐进)', ann7c, sharpe7c, dd7c, vol7c, tov7c.mean()*12, avg7c))

    # ============================================================
    # 输出结果
    # ============================================================
    print("\n" + "="*80)
    print("四大优化集成验证结果")
    print("="*80)

    print(f"\n{'配置':<35} {'年化':>8} {'夏普':>6} {'回撤':>8} {'波动':>8} {'换手':>6} {'均持':>5}")
    print("-" * 80)
    for name, ann, sharpe, dd, vol, tov, avg in results:
        print(f"{name:<35} {ann:>7.2%} {sharpe:>6.2f} {dd:>7.2%} {vol:>7.2%} {tov:>5.0%} {avg:>5.0f}")

    # ============================================================
    # 子样本稳健性检验
    # ============================================================
    print("\n" + "="*80)
    print("子样本稳健性：基准 vs 全部集成")
    print("="*80)

    sub_results = []
    for start, end, label in [(None, None, '全样本'),
                               ('2005', '2012', '2005-2012'),
                               ('2013', '2018', '2013-2018'),
                               ('2019', '2025', '2019-2025')]:
        if start is None:
            mask = pd.Series(True, index=ret0_final.index)
        else:
            mask = (ret0_final.index >= start) & (ret0_final.index <= end)

        # 基准
        r0 = ret0_final[mask]
        a0, s0, d0, v0, _, _ = calculate_metrics(r0) if len(r0) > 0 else (0, 0, 0, 0, pd.Series(), 0)

        # 全部集成
        r7 = ret7_final[mask]
        a7, s7, d7, v7, _, _ = calculate_metrics(r7) if len(r7) > 0 else (0, 0, 0, 0, pd.Series(), 0)

        sub_results.append((label, a0, s0, d0, a7, s7, d7))

    print(f"\n{'时期':<15} {'基准夏普':>8} {'基准回撤':>8} {'集成夏普':>8} {'集成回撤':>8} {'夏普提升':>8}")
    print("-" * 60)
    for label, a0, s0, d0, a7, s7, d7 in sub_results:
        delta = s7 - s0
        print(f"{label:<15} {s0:>8.2f} {d0:>7.2%} {s7:>8.2f} {d7:>7.2%} {delta:>+8.2f}")

    # ============================================================
    # 可视化
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Week 10B: 四大优化集成验证', fontsize=16, fontweight='bold')

    # 图1: 累积收益对比
    ax = axes[0, 0]
    ax.plot(cum0.index, cum0.values, label='基准(MA12)', color='#95a5a6', linewidth=1.5)
    ax.plot(cum7.index, cum7.values, label='全部集成', color='#e74c3c', linewidth=2)
    ax.set_ylabel('累积净值')
    ax.set_title('累积收益对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 图2: 逐步叠加夏普
    ax = axes[0, 1]
    step_names = [r[0][:20] for r in results[:8]]
    step_sharpes = [r[2] for r in results[:8]]
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#e74c3c']
    bars = ax.barh(range(len(step_names)), step_sharpes, color=colors)
    ax.set_yticks(range(len(step_names)))
    ax.set_yticklabels(step_names, fontsize=8)
    ax.set_xlabel('夏普比率')
    ax.set_title('逐步叠加效果')
    ax.grid(True, alpha=0.3)
    # 标注数值
    for i, (name, sh) in enumerate(zip(step_names, step_sharpes)):
        ax.text(sh + 0.01, i, f'{sh:.2f}', va='center', fontsize=9)

    # 图3: 回撤对比
    ax = axes[1, 0]
    dd0_series = ((cum0 - cum0.cummax()) / cum0.cummax())
    dd7_series = ((cum7 - cum7.cummax()) / cum7.cummax())
    ax.fill_between(dd0_series.index, dd0_series.values, 0, alpha=0.3, color='#95a5a6', label='基准')
    ax.fill_between(dd7_series.index, dd7_series.values, 0, alpha=0.3, color='#e74c3c', label='集成')
    ax.set_ylabel('回撤')
    ax.set_title('回撤对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图4: 子样本夏普对比
    ax = axes[1, 1]
    periods = [r[0] for r in sub_results]
    base_sharpes = [r[2] for r in sub_results]
    opt_sharpes = [r[5] for r in sub_results]
    x = np.arange(len(periods))
    width = 0.35
    ax.bar(x - width/2, base_sharpes, width, label='基准', color='#95a5a6')
    ax.bar(x + width/2, opt_sharpes, width, label='全部集成', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_ylabel('夏普比率')
    ax.set_title('子样本稳健性')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'week10b_integration.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: results/week10b_integration.png")

    return results, sub_results


if __name__ == '__main__':
    results, sub_results = main()
