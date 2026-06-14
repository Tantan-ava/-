# -*- coding: utf-8 -*-
"""
价值因子有效性测试 + 纯反转策略对比

核心问题:
1. 当前价值因子是否有效？
2. 纯反转 vs 价值+反转，哪个更好？
3. 流动性过滤对两者的影响？
4. MA过滤器对两者的影响？
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
    """
    通用回测函数
    signal_df: 选股信号，值越大越优先选
    liq_filter: 布尔DataFrame，True=可交易
    """
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
    print("价值因子有效性测试 + 纯反转策略对比")
    print("=" * 80)

    # 加载数据
    print("\n[1] 加载数据...")
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)

    df_tover = pd.read_excel('data/raw/LIQ_TOVER_M.xlsx', engine='openpyxl', skiprows=[1, 2])
    df_tover.columns = ['Stkcd', 'Trdmnt', 'ToverOsM']
    df_tover['ToverOsM'] = pd.to_numeric(df_tover['ToverOsM'], errors='coerce')
    df_tover['Stkcd'] = df_tover['Stkcd'].astype(str).str.zfill(6)
    df_tover['Trdmnt'] = pd.to_datetime(df_tover['Trdmnt'], format='%Y-%m')
    tover_pivot = df_tover.pivot(index='Trdmnt', columns='Stkcd', values='ToverOsM')
    tover_pivot.index = pd.to_datetime(tover_pivot.index)
    tover_pivot.columns = tover_pivot.columns.astype(str)

    common_cols = sorted(set(df_ret.columns) & set(tover_pivot.columns))
    common_idx = df_ret.index.intersection(tover_pivot.index)
    df_ret = df_ret.loc[common_idx, common_cols]
    tover_pivot = tover_pivot.loc[common_idx, common_cols]

    print(f"  收益率: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")

    # 构建信号
    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal)

    # 流动性过滤 (t-1月换手率 >= 30%)
    tover_lagged = tover_pivot.shift(1)
    liq_filter_30 = tover_lagged >= 30
    liq_filter_50 = tover_lagged >= 50

    market_ret = df_ret.mean(axis=1)

    all_results = {}

    # ========================================================================
    # Part 1: 信号对比 — 纯反转 vs 纯价值 vs 组合
    # ========================================================================
    print("\n" + "=" * 80)
    print("Part 1: 信号对比 — 纯反转 vs 纯价值 vs 组合")
    print("=" * 80)

    signal_configs = [
        ('纯反转(100%)', reversal_signal),
        ('纯价值(100%)', value_signal),
        ('组合(40V+60R)', 0.4 * value_signal + 0.6 * reversal_signal),
        ('组合(30V+70R)', 0.3 * value_signal + 0.7 * reversal_signal),
        ('组合(20V+80R)', 0.2 * value_signal + 0.8 * reversal_signal),
        ('组合(50V+50R)', 0.5 * value_signal + 0.5 * reversal_signal),
    ]

    for name, sig in signal_configs:
        ret, tover = run_backtest(df_ret, sig, topk=100, K=K)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret)
        tover_ann = tover.mean() * 12
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")
        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret, 'cum_wealth': cum
        }

    # ========================================================================
    # Part 2: 信号对比 + MA过滤器
    # ========================================================================
    print("\n" + "=" * 80)
    print("Part 2: 信号对比 + MA市场过滤器")
    print("=" * 80)

    for name, sig in signal_configs:
        ret, tover = run_backtest(df_ret, sig, topk=100, K=K)
        ret_ma, _ = apply_market_filter(ret, market_returns=market_ret)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret_ma)
        tover_ann = tover.mean() * 12
        ma_name = f'{name}+MA'
        print(f"  {ma_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")
        all_results[ma_name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret_ma, 'cum_wealth': cum
        }

    # ========================================================================
    # Part 3: 信号对比 + 流动性过滤(>30%)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Part 3: 信号对比 + 流动性过滤(换手率>30%)")
    print("=" * 80)

    for name, sig in signal_configs:
        ret, tover = run_backtest(df_ret, sig, topk=100, K=K, liq_filter=liq_filter_30)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret)
        tover_ann = tover.mean() * 12
        liq_name = f'{name}+流动性'
        print(f"  {liq_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")
        all_results[liq_name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret, 'cum_wealth': cum
        }

    # ========================================================================
    # Part 4: 信号对比 + 流动性过滤 + MA
    # ========================================================================
    print("\n" + "=" * 80)
    print("Part 4: 信号对比 + 流动性过滤(>30%) + MA")
    print("=" * 80)

    for name, sig in signal_configs:
        ret, tover = run_backtest(df_ret, sig, topk=100, K=K, liq_filter=liq_filter_30)
        ret_ma, _ = apply_market_filter(ret, market_returns=market_ret)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret_ma)
        tover_ann = tover.mean() * 12
        full_name = f'{name}+流动性+MA'
        print(f"  {full_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")
        all_results[full_name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret_ma, 'cum_wealth': cum
        }

    # ========================================================================
    # Part 5: 全配置对比 + 止损 + 成本
    # ========================================================================
    print("\n" + "=" * 80)
    print("Part 5: 全配置对比 + 极宽止损 + 真实成本")
    print("=" * 80)

    for name, sig in signal_configs:
        ret, tover = run_backtest(df_ret, sig, topk=100, K=K, liq_filter=liq_filter_30)
        ret_ma, _ = apply_market_filter(ret, market_returns=market_ret)
        ret_dd, _ = apply_drawdown_control(ret_ma)
        ret_net = apply_realistic_costs(ret_dd, pd.Series(tover.mean(), index=ret_dd.index))
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret_net)
        tover_ann = tover.mean() * 12
        full_name = f'{name}+流动性+MA+止损+成本'
        print(f"  {full_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")
        all_results[full_name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret_net, 'cum_wealth': cum
        }

    # ========================================================================
    # Part 6: 反转K值扫描 (纯反转 + 流动性 + MA)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Part 6: 反转K值扫描 (纯反转 + 流动性 + MA)")
    print("=" * 80)

    for k_val in [1, 2, 3, 4, 6, 9, 12]:
        rev_sig = -df_ret.rolling(window=k_val).sum().shift(1)
        rev_sig = apply_standardization(rev_sig)
        ret, tover = run_backtest(df_ret, rev_sig, topk=100, K=k_val, liq_filter=liq_filter_30)
        ret_ma, _ = apply_market_filter(ret, market_returns=market_ret)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret_ma)
        tover_ann = tover.mean() * 12
        name = f'纯反转K={k_val}+流动性+MA'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")
        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret_ma, 'cum_wealth': cum
        }

    # ========================================================================
    # Part 7: 价值因子单独检验
    # ========================================================================
    print("\n" + "=" * 80)
    print("Part 7: 价值因子(12月收益)单独检验 — 它到底有没有用？")
    print("=" * 80)

    # 方法: 纯反转 vs 纯反转+价值因子加分
    # 如果加价值因子后夏普提升，说明价值因子有用
    for v_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
        combo_sig = v_weight * value_signal + (1 - v_weight) * reversal_signal
        ret, tover = run_backtest(df_ret, combo_sig, topk=100, K=K, liq_filter=liq_filter_30)
        ret_ma, _ = apply_market_filter(ret, market_returns=market_ret)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(ret_ma)
        tover_ann = tover.mean() * 12
        name = f'V={v_weight:.0%}+R={1-v_weight:.0%}+流动性+MA'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")
        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': ret_ma, 'cum_wealth': cum
        }

    # ========================================================================
    # 结果汇总
    # ========================================================================
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)

    summary_rows = []
    for name, r in all_results.items():
        if isinstance(r, dict) and 'ann_ret' in r:
            summary_rows.append({
                '策略': name,
                '年化收益': r['ann_ret'],
                '夏普比率': r['sharpe'],
                '最大回撤': r['max_dd'],
                '年化波动率': r['ann_vol'],
                '换手率(年)': r.get('turnover', 0),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('夏普比率', ascending=False)

    print(f"\n{'策略':<40} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'波动率':>8} {'换手率':>6}")
    print("-" * 85)
    for _, row in summary_df.iterrows():
        print(f"{row['策略']:<40} {row['年化收益']:>7.2%} {row['夏普比率']:>6.2f} "
              f"{row['最大回撤']:>7.2%} {row['年化波动率']:>7.2%} {row['换手率(年)']:>5.0%}")

    # 关键对比: 纯反转 vs 组合 (各配置下)
    print("\n" + "=" * 80)
    print("关键对比: 纯反转 vs 40V+60R 组合")
    print("=" * 80)

    comparisons = [
        ('无优化', '纯反转(100%)', '组合(40V+60R)'),
        ('+MA', '纯反转(100%)+MA', '组合(40V+60R)+MA'),
        ('+流动性', '纯反转(100%)+流动性', '组合(40V+60R)+流动性'),
        ('+流动性+MA', '纯反转(100%)+流动性+MA', '组合(40V+60R)+流动性+MA'),
        ('全配置', '纯反转(100%)+流动性+MA+止损+成本', '组合(40V+60R)+流动性+MA+止损+成本'),
    ]

    print(f"\n{'配置':<15} {'纯反转夏普':>10} {'组合夏普':>10} {'差异':>8} {'纯反转换手':>10} {'组合换手':>10}")
    print("-" * 70)
    for cfg_name, rev_key, combo_key in comparisons:
        if rev_key in all_results and combo_key in all_results:
            rev_sharpe = all_results[rev_key]['sharpe']
            combo_sharpe = all_results[combo_key]['sharpe']
            rev_tover = all_results[rev_key].get('turnover', 0)
            combo_tover = all_results[combo_key].get('turnover', 0)
            diff = combo_sharpe - rev_sharpe
            print(f"{cfg_name:<15} {rev_sharpe:>10.2f} {combo_sharpe:>10.2f} {diff:>+8.2f} "
                  f"{rev_tover:>9.0%} {combo_tover:>9.0%}")

    # ========================================================================
    # 可视化
    # ========================================================================
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 累计净值对比: 纯反转 vs 组合 (各配置)
    ax1 = axes[0, 0]
    key_pairs = [
        ('纯反转(100%)+流动性+MA', 'red', '纯反转'),
        ('组合(40V+60R)+流动性+MA', 'blue', '40V+60R'),
        ('组合(20V+80R)+流动性+MA', 'green', '20V+80R'),
        ('纯价值(100%)+流动性+MA', 'gray', '纯价值'),
    ]
    for key, color, label in key_pairs:
        if key in all_results and 'cum_wealth' in all_results[key]:
            all_results[key]['cum_wealth'].plot(ax=ax1, color=color, label=label, linewidth=1.5)
    ax1.set_title('信号对比 + 流动性 + MA: 累计净值', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 价值因子权重 vs 夏普
    ax2 = axes[0, 1]
    v_weights_list = []
    sharpe_list = []
    for v_w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        if v_w == 0:
            key = '纯反转(100%)+流动性+MA'
        elif v_w == 1.0:
            key = '纯价值(100%)+流动性+MA'
        else:
            key = f'V={v_w:.0%}+R={1-v_w:.0%}+流动性+MA'
        if key in all_results:
            v_weights_list.append(v_w)
            sharpe_list.append(all_results[key]['sharpe'])
    ax2.plot(v_weights_list, sharpe_list, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax2.axvline(x=0.4, color='red', linestyle='--', alpha=0.5, label='当前V=0.4')
    ax2.set_xlabel('价值因子权重', fontsize=11)
    ax2.set_ylabel('夏普比率', fontsize=11)
    ax2.set_title('价值因子权重 vs 夏普 (+流动性+MA)', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # K值扫描
    ax3 = axes[1, 0]
    k_vals = []
    k_sharpes = []
    for k_val in [1, 2, 3, 4, 6, 9, 12]:
        key = f'纯反转K={k_val}+流动性+MA'
        if key in all_results:
            k_vals.append(k_val)
            k_sharpes.append(all_results[key]['sharpe'])
    ax3.bar(range(len(k_vals)), k_sharpes, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(k_vals)))
    ax3.set_xticklabels([f'K={k}' for k in k_vals])
    ax3.set_ylabel('夏普比率', fontsize=11)
    ax3.set_title('反转K值扫描 (+流动性+MA)', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')

    # 全配置对比柱状图
    ax4 = axes[1, 1]
    cfg_names = ['无优化', '+MA', '+流动性', '+流动性\n+MA', '+流动性\n+MA+止损\n+成本']
    rev_sharpes = []
    combo_sharpes = []
    for _, rev_key, combo_key in comparisons:
        if rev_key in all_results and combo_key in all_results:
            rev_sharpes.append(all_results[rev_key]['sharpe'])
            combo_sharpes.append(all_results[combo_key]['sharpe'])
    x = np.arange(len(cfg_names))
    width = 0.35
    ax4.bar(x - width/2, rev_sharpes, width, label='纯反转', color='red', alpha=0.7)
    ax4.bar(x + width/2, combo_sharpes, width, label='40V+60R', color='blue', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(cfg_names, fontsize=8)
    ax4.set_ylabel('夏普比率', fontsize=11)
    ax4.set_title('纯反转 vs 组合: 各配置对比', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'value_factor_test.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'value_factor_test.png'}")
    plt.close()

    summary_df.to_excel(output_dir / 'value_factor_test_summary.xlsx', index=False)
    print(f"✓ 汇总表已保存: {output_dir / 'value_factor_test_summary.xlsx'}")

    return all_results, summary_df


if __name__ == '__main__':
    results, summary = main()
    print("\n" + "=" * 80)
    print("价值因子有效性测试完成！")
    print("=" * 80)
