# -*- coding: utf-8 -*-
"""
最终最优组合验证 — 统一框架下对比所有候选配置
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


def calculate_metrics(returns, rf=0.0):
    if len(returns) == 0 or returns.std() == 0:
        return 0, 0, 0, 0, pd.Series(), 0
    cum_ret = (1 + returns).prod() - 1
    ann_ret = (1 + cum_ret) ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ((returns.mean() - rf / 12) / returns.std()) * np.sqrt(12) if returns.std() > 0 else 0
    cum_wealth = (1 + returns).cumprod()
    drawdown = (cum_wealth - cum_wealth.cummax()) / cum_wealth.cummax()
    max_dd = drawdown.min()
    return ann_ret, sharpe, max_dd, ann_vol, cum_wealth, 0


def apply_standardization(signal_df, method='Winsorization'):
    if method == 'Winsorization':
        lower = signal_df.rolling(24, min_periods=6).quantile(0.01)
    upper = signal_df.rolling(24, min_periods=6).quantile(0.99)
    return signal_df.clip(lower=lower, upper=upper, axis=0)
    return signal_df


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


def apply_drawdown_control(returns, soft_dd_threshold=-0.25, hard_dd_threshold=-0.35,
                           fuse_dd_threshold=-0.50, soft_position=0.80,
                           hard_position=0.40, fuse_position=0.0):
    cum_wealth = (1 + returns).cumprod()
    drawdown = (cum_wealth - cum_wealth.cummax()) / cum_wealth.cummax()
    position_levels = pd.Series(1.0, index=returns.index)
    current_position = 1.0
    stop_counter = 0
    for i, date in enumerate(returns.index):
        dd = drawdown.iloc[i]
        if dd <= fuse_dd_threshold:
            current_position = fuse_position; stop_counter = 3
        elif dd <= hard_dd_threshold:
            current_position = hard_position; stop_counter = 2
        elif dd <= soft_dd_threshold:
            current_position = soft_position; stop_counter = 1
        if stop_counter > 0:
            stop_counter -= 1
            if stop_counter == 0 and dd > soft_dd_threshold / 2:
                current_position = 1.0
        position_levels.iloc[i] = current_position
    return returns * position_levels, position_levels


def apply_realistic_costs(returns, turnover_series, commission=0.00025,
                          stamp_tax=0.001, transfer_fee=0.00002, impact_cost=0.001):
    round_trip_cost = (commission + transfer_fee + impact_cost) + (commission + stamp_tax + transfer_fee + impact_cost)
    monthly_cost = turnover_series * round_trip_cost
    return returns - monthly_cost


def run_backtest(df_ret, signal_df, topk=100, K=6, liq_filter=None):
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date]
        if liq_filter is not None and date in liq_filter.index:
            sig = sig.where(liq_filter.loc[date], other=np.nan)
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

    return port_ret, turnover_series


def main():
    print("=" * 80)
    print("最终最优组合验证")
    print("=" * 80)

    # 加载数据
    print("\n[1] 加载数据...")
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)

    # 换手率
    df_tover = pd.read_excel('data/raw/LIQ_TOVER_M.xlsx', engine='openpyxl', skiprows=[1, 2])
    df_tover.columns = ['Stkcd', 'Trdmnt', 'ToverOsM']
    df_tover['ToverOsM'] = pd.to_numeric(df_tover['ToverOsM'], errors='coerce')
    df_tover['Stkcd'] = df_tover['Stkcd'].astype(str).str.zfill(6)
    df_tover['Trdmnt'] = pd.to_datetime(df_tover['Trdmnt'], format='%Y-%m')
    tover_pivot = df_tover.pivot(index='Trdmnt', columns='Stkcd', values='ToverOsM')
    tover_pivot.index = pd.to_datetime(tover_pivot.index)
    tover_pivot.columns = tover_pivot.columns.astype(str)

    # 对齐
    common_cols = sorted(set(df_ret.columns) & set(tover_pivot.columns))
    common_idx = df_ret.index.intersection(tover_pivot.index)
    df_ret = df_ret.loc[common_idx, common_cols]
    tover_pivot = tover_pivot.loc[common_idx, common_cols]

    print(f"  数据: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")

    # 信号
    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal)
    combo_signal = 0.4 * value_signal + 0.6 * reversal_signal

    # 流动性过滤
    tover_lagged = tover_pivot.shift(1)
    liq_filter_30 = tover_lagged >= 30

    market_ret = df_ret.mean(axis=1)

    # ========================================================================
    # 候选配置
    # ========================================================================
    configs = [
        # (名称, 信号, 流动性过滤, MA, 止损, 成本)
        ('A: 基准(40V+60R)', combo_signal, False, False, False, False),
        ('B: +MA', combo_signal, False, True, False, False),
        ('C: +流动性(>30%)', combo_signal, True, False, False, False),
        ('D: +流动性+MA', combo_signal, True, True, False, False),
        ('E: +流动性+MA+极宽止损', combo_signal, True, True, True, False),
        ('F: +流动性+MA+极宽止损+成本', combo_signal, True, True, True, True),
        ('G: +MA+极宽止损+成本(无流动性)', combo_signal, False, True, True, True),
        # 纯反转
        ('H: 纯反转+MA', reversal_signal, False, True, False, False),
        ('I: 纯反转+流动性+MA', reversal_signal, True, True, False, False),
        ('J: 纯反转+MA+止损+成本', reversal_signal, False, True, True, True),
        # K=2反转
        ('K: K=2反转+MA', None, False, True, False, False),  # 特殊处理
        ('L: K=2反转+流动性+MA', None, True, True, False, False),
    ]

    results = {}

    for cfg_name, sig, use_liq, use_ma, use_dd, use_cost in configs:
        # K=2特殊处理
        if cfg_name.startswith('K: K=2'):
            k_val = 2
            sig_k2 = -df_ret.rolling(window=k_val).sum().shift(1)
            sig_k2 = apply_standardization(sig_k2)
            actual_sig = sig_k2
            actual_K = k_val
        elif cfg_name.startswith('L: K=2'):
            k_val = 2
            sig_k2 = -df_ret.rolling(window=k_val).sum().shift(1)
            sig_k2 = apply_standardization(sig_k2)
            actual_sig = sig_k2
            actual_K = k_val
        else:
            actual_sig = sig
            actual_K = K

        liq = liq_filter_30 if use_liq else None

        ret, tover = run_backtest(df_ret, actual_sig, topk=100, K=actual_K, liq_filter=liq)

        if use_ma:
            ret, _ = apply_market_filter(ret, market_returns=market_ret)
        if use_dd:
            ret, _ = apply_drawdown_control(ret)
        if use_cost:
            ret = apply_realistic_costs(ret, pd.Series(tover.mean(), index=ret.index))

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret)
        tover_ann = tover.mean() * 12

        results[cfg_name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret, 'cum_wealth': cum
        }

        print(f"  {cfg_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, "
              f"回撤={dd:.2%}, 波动={vol:.2%}, 换手={tover_ann:.0%}")

    # ========================================================================
    # 找最优
    # ========================================================================
    print("\n" + "=" * 80)
    print("排名 (按夏普)")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\n{'排名':>3} {'配置':<40} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'波动率':>8} {'换手率':>6}")
    print("-" * 90)
    for i, (name, r) in enumerate(sorted_results, 1):
        print(f"{i:>3} {name:<40} {r['ann_ret']:>7.2%} {r['sharpe']:>6.2f} "
              f"{r['max_dd']:>7.2%} {r['ann_vol']:>7.2%} {r['turnover']:>5.0%}")

    best_name, best_r = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"最优配置: {best_name}")
    print(f"  年化收益: {best_r['ann_ret']:.2%}")
    print(f"  夏普比率: {best_r['sharpe']:.2f}")
    print(f"  最大回撤: {best_r['max_dd']:.2%}")
    print(f"  年化波动率: {best_r['ann_vol']:.2%}")
    print(f"  年化换手率: {best_r['turnover']:.0%}")
    print(f"{'='*80}")

    # ========================================================================
    # 可视化
    # ========================================================================
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 累计净值
    ax1 = axes[0, 0]
    for name, r in sorted_results[:6]:
        r['cum_wealth'].plot(ax=ax1, label=name, linewidth=1.5)
    ax1.set_title('Top 6 策略: 累计净值', fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 夏普柱状图
    ax2 = axes[0, 1]
    names = [n for n, _ in sorted_results]
    sharpes = [r['sharpe'] for _, r in sorted_results]
    colors = ['gold' if i == 0 else 'steelblue' for i in range(len(names))]
    ax2.barh(range(len(names)), sharpes, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('夏普比率')
    ax2.set_title('夏普比率排名', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # 回撤对比
    ax3 = axes[1, 0]
    for name, r in sorted_results[:6]:
        cum_w = r['cum_wealth']
        dd = (cum_w - cum_w.cummax()) / cum_w.cummax()
        dd.plot(ax=ax3, label=name, linewidth=1)
    ax3.set_title('Top 6 回撤对比', fontsize=13)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 年化收益 vs 换手率
    ax4 = axes[1, 1]
    for name, r in sorted_results:
        ax4.scatter(r['turnover'], r['ann_ret'], s=80, alpha=0.7)
        ax4.annotate(name.split(':')[0], (r['turnover'], r['ann_ret']), fontsize=8)
    ax4.set_xlabel('年化换手率', fontsize=11)
    ax4.set_ylabel('年化收益', fontsize=11)
    ax4.set_title('收益 vs 换手率', fontsize=13)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'final_optimal.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'final_optimal.png'}")
    plt.close()

    return results, sorted_results


if __name__ == '__main__':
    results, sorted_results = main()
