# -*- coding: utf-8 -*-
"""
Week 7: 关键组合验证 — 跳过ML，专注有效优化+MA组合
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


def run_backtest(df_ret, signal_df, topk=100, K=6, weight_method='equal',
                 vol_lookback=12, hold_buffer=0, prev_selected_init=None):
    """
    统一回测函数
    weight_method: 'equal', 'inv_vol'(1/σ), 'inv_var'(1/σ²), 'signal_strength'
    hold_buffer: >0时启用持仓延续
    """
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    prev_selected = prev_selected_init

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date]
        ranks = sig.rank(ascending=False, method='first')

        # 选股
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
                date_idx = df_ret.index.get_loc(date)
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
            elif weight_method == 'signal_strength':
                sig_strength = (topk - ranks + 1).clip(lower=0)
                w = selected * sig_strength
                w_sum = w.sum()
                if w_sum > 0:
                    weights.loc[date] = w / w_sum
                else:
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
    turnover_series = pd.Series(turnover_list, index=port_ret.index)

    return port_ret, turnover_series


def main():
    print("=" * 80)
    print("Week 7: 关键组合验证")
    print("=" * 80)

    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)
    print(f"  数据: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")

    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal)
    combo_signal = 0.4 * value_signal + 0.6 * reversal_signal
    market_ret = df_ret.mean(axis=1)

    # ========================================================================
    # 逐层叠加验证
    # ========================================================================
    results = {}

    configs = [
        # (名称, 信号, weight_method, hold_buffer, use_ma, vol_target, vol_lookback)
        ('A:基准(40V+60R)', combo_signal, 'equal', 0, False, None, None),
        ('B:+MA', combo_signal, 'equal', 0, True, None, None),
        ('C:最小方差(1/σ²)', combo_signal, 'inv_var', 0, False, None, None),
        ('D:最小方差+MA', combo_signal, 'inv_var', 0, True, None, None),
        ('E:风险平价(1/σ)', combo_signal, 'inv_vol', 0, False, None, None),
        ('F:风险平价+MA', combo_signal, 'inv_vol', 0, True, None, None),
        ('G:持仓延续(b=50)', combo_signal, 'equal', 50, False, None, None),
        ('H:持仓延续+MA', combo_signal, 'equal', 50, True, None, None),
        ('I:最小方差+持仓延续+MA', combo_signal, 'inv_var', 50, True, None, None),
        # Vol Targeting on top of MA
        ('J:MA+VolTarget(10%,L=24)', combo_signal, 'equal', 0, True, 0.10, 24),
        ('K:MA+VolTarget(12%,L=24)', combo_signal, 'equal', 0, True, 0.12, 24),
        ('L:MA+VolTarget(15%,L=24)', combo_signal, 'equal', 0, True, 0.15, 24),
        # Triple combo
        ('M:最小方差+持仓延续+MA+VT(12%)', combo_signal, 'inv_var', 50, True, 0.12, 24),
        ('N:最小方差+持仓延续+MA+VT(10%)', combo_signal, 'inv_var', 50, True, 0.10, 24),
    ]

    for name, sig, wm, hb, use_ma, vt, vt_lb in configs:
        port_ret, tover = run_backtest(df_ret, sig, topk=100, K=K,
                                        weight_method=wm, hold_buffer=hb)

        if use_ma:
            port_ret, _ = apply_market_filter(port_ret, market_ret)

        if vt is not None:
            rolling_vol = port_ret.rolling(vt_lb).std().shift(1) * np.sqrt(12)
            vol_scalar = (vt / rolling_vol).clip(0.3, 2.0).fillna(1.0)
            port_ret = port_ret * vol_scalar

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
        results[name] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                         'ann_vol': vol, 'turnover': tover.mean() * 12,
                         'cum_wealth': cum}

        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, "
              f"波动={vol:.2%}, 换手={tover.mean()*12:.0%}")

    # ========================================================================
    # 排名
    # ========================================================================
    print("\n" + "=" * 80)
    print("最终排名 (按夏普)")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\n{'排名':>3} {'配置':<35} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'波动率':>8} {'换手率':>6}")
    print("-" * 85)
    for i, (name, r) in enumerate(sorted_results, 1):
        print(f"{i:>3} {name:<35} {r['ann_ret']:>7.2%} {r['sharpe']:>6.2f} "
              f"{r['max_dd']:>7.2%} {r['ann_vol']:>7.2%} {r['turnover']:>5.0%}")

    best_name, best_r = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"最优: {best_name}")
    print(f"  年化={best_r['ann_ret']:.2%}, 夏普={best_r['sharpe']:.2f}, "
          f"回撤={best_r['max_dd']:.2%}, 波动={best_r['ann_vol']:.2%}, 换手={best_r['turnover']:.0%}")
    print(f"{'='*80}")

    # ========================================================================
    # 可视化
    # ========================================================================
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 累计净值 top5
    ax1 = axes[0, 0]
    for name, r in sorted_results[:5]:
        r['cum_wealth'].plot(ax=ax1, label=name, linewidth=1.5)
    ax1.set_title('Top 5 累计净值', fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 夏普柱状图
    ax2 = axes[0, 1]
    names = [n for n, _ in sorted_results]
    sharpes = [r['sharpe'] for _, r in sorted_results]
    colors = ['gold' if i == 0 else 'steelblue' for i in range(len(names))]
    ax2.barh(range(len(names)), sharpes, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([n.split(':')[1] if ':' in n else n for n in names], fontsize=8)
    ax2.set_xlabel('夏普比率')
    ax2.set_title('夏普比率排名', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # 3. 回撤 top5
    ax3 = axes[1, 0]
    for name, r in sorted_results[:5]:
        cum_w = r['cum_wealth']
        dd = (cum_w - cum_w.cummax()) / cum_w.cummax()
        dd.plot(ax=ax3, label=name, linewidth=1)
    ax3.set_title('Top 5 回撤对比', fontsize=13)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. 收益 vs 波动率
    ax4 = axes[1, 1]
    for name, r in sorted_results:
        ax4.scatter(r['ann_vol'], r['ann_ret'], s=80, alpha=0.7)
        ax4.annotate(name.split(':')[1] if ':' in n else name,
                     (r['ann_vol'], r['ann_ret']), fontsize=7)
    ax4.set_xlabel('年化波动率', fontsize=11)
    ax4.set_ylabel('年化收益', fontsize=11)
    ax4.set_title('收益 vs 波动率', fontsize=13)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Week 7: 优化组合验证', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'week7_final_combo.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'week7_final_combo.png'}")
    plt.close()

    return results


if __name__ == '__main__':
    results = main()
