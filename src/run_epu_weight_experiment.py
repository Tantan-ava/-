# -*- coding: utf-8 -*-
"""
EPU调整因子权重实验

基准策略: 价值(0.4) + 反转(0.6) + MA市场过滤器
实验变量: EPU调整价值/反转权重
评估指标: 年化收益、夏普、回撤、波动率、组合换手率
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


def run_backtest_with_weights(df_ret, value_signal, reversal_signal,
                              v_weight_series, r_weight_series,
                              topk=100, K=6):
    """
    逐日期用不同权重组合信号并选股，返回组合收益和换手率
    v_weight_series, r_weight_series: pd.Series, index=df_ret.index
    """
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []

    for date in df_ret.index[K:]:
        vw = v_weight_series.loc[date]
        rw = r_weight_series.loc[date]
        sig = vw * value_signal.loc[date] + rw * reversal_signal.loc[date]
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
    # 对齐换手率
    if len(turnover_list) < len(port_ret):
        turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
    elif len(turnover_list) > len(port_ret):
        turnover_list = turnover_list[-len(port_ret):]
    turnover_series = pd.Series(turnover_list, index=port_ret.index)

    return port_ret, turnover_series


def main():
    print("=" * 80)
    print("EPU调整因子权重实验")
    print("基准: 价值(0.4) + 反转(0.6) + MA市场过滤器")
    print("=" * 80)

    # 加载数据
    print("\n[1] 加载数据...")
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)

    df_epu = pd.read_excel('data/raw/SCMP_China_Policy_Uncertainty_Data.xlsx', engine='openpyxl')
    df_epu['date'] = pd.to_datetime(df_epu['year'].astype(str) + '-' + df_epu['month'].astype(str) + '-01')
    epu_series = df_epu.set_index('date')['China News-Based EPU']

    print(f"  收益率: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")
    print(f"  EPU: {len(epu_series)}个月")

    # 构建信号
    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal)

    # EPU对齐到收益率index
    epu_aligned = epu_series.reindex(df_ret.index)
    epu_median = epu_aligned.expanding().median().shift(1)
    epu_mean = epu_aligned.expanding().mean().shift(1)
    epu_p75 = epu_aligned.expanding().quantile(0.75).shift(1)
    epu_p25 = epu_aligned.expanding().quantile(0.25).shift(1)

    # 市场收益
    market_ret = df_ret.mean(axis=1)

    all_results = {}

    # ========================================================================
    # 基准1: 纯价值+反转 (无MA)
    # ========================================================================
    print("\n" + "=" * 80)
    print("基准1: 价值(0.4) + 反转(0.6), 无MA")
    print("=" * 80)

    v_w_base = pd.Series(0.4, index=df_ret.index)
    r_w_base = pd.Series(0.6, index=df_ret.index)
    base_ret, base_turnover = run_backtest_with_weights(
        df_ret, value_signal, reversal_signal, v_w_base, r_w_base, K=K)

    base_ann, base_sharpe, base_dd, base_vol, base_cum, _ = calculate_metrics(base_ret)
    base_tover_ann = base_turnover.mean() * 12
    print(f"  年化={base_ann:.2%}, 夏普={base_sharpe:.2f}, 回撤={base_dd:.2%}, "
          f"波动={base_vol:.2%}, 换手率={base_tover_ann:.0%}")

    all_results['基准(无MA)'] = {
        'ann_ret': base_ann, 'sharpe': base_sharpe, 'max_dd': base_dd,
        'ann_vol': base_vol, 'turnover': base_tover_ann,
        'returns': base_ret, 'cum_wealth': base_cum
    }

    # ========================================================================
    # 基准2: 价值(0.4) + 反转(0.6) + MA (新基准)
    # ========================================================================
    print("\n" + "=" * 80)
    print("基准2: 价值(0.4) + 反转(0.6) + MA市场过滤器 (新基准)")
    print("=" * 80)

    base_ma_ret, base_ma_scalar = apply_market_filter(base_ret, market_returns=market_ret)
    base_ma_ann, base_ma_sharpe, base_ma_dd, base_ma_vol, base_ma_cum, _ = calculate_metrics(base_ma_ret)
    # MA后的换手率仍用原始换手率（MA只调仓位不调持仓结构）
    base_ma_tover = base_turnover.reindex(base_ma_ret.index).mean() * 12
    print(f"  年化={base_ma_ann:.2%}, 夏普={base_ma_sharpe:.2f}, 回撤={base_ma_dd:.2%}, "
          f"波动={base_ma_vol:.2%}, 换手率={base_ma_tover:.0%}")

    all_results['基准(+MA)'] = {
        'ann_ret': base_ma_ann, 'sharpe': base_ma_sharpe, 'max_dd': base_ma_dd,
        'ann_vol': base_ma_vol, 'turnover': base_ma_tover,
        'returns': base_ma_ret, 'cum_wealth': base_ma_cum
    }

    # ========================================================================
    # EPU调因子权重实验
    # ========================================================================
    print("\n" + "=" * 80)
    print("EPU调整因子权重实验")
    print("  逻辑: 高EPU → 反转权重↑(不确定性高时反转更有效)")
    print("        低EPU → 价值权重↑(确定性高时价值更有效)")
    print("=" * 80)

    # EPU状态分类方法
    epu_states = {}

    # 方法A: 基于expanding median
    epu_states['A_median'] = {
        'high': epu_aligned > epu_median * 1.3,
        'low': epu_aligned < epu_median * 0.7,
        'mid': (epu_aligned >= epu_median * 0.7) & (epu_aligned <= epu_median * 1.3),
    }

    # 方法B: 基于expanding分位数
    epu_states['B_quantile'] = {
        'high': epu_aligned > epu_p75,
        'low': epu_aligned < epu_p25,
        'mid': (epu_aligned >= epu_p25) & (epu_aligned <= epu_p75),
    }

    # 方法C: 基于EPU变化率
    epu_change = epu_aligned.pct_change().replace([np.inf, -np.inf], np.nan)
    epu_states['C_change'] = {
        'high': epu_change > 0.5,
        'low': epu_change < -0.2,
        'mid': (epu_change >= -0.2) & (epu_change <= 0.5),
    }

    # 方法D: 基于EPU水平(绝对值)
    epu_states['D_level'] = {
        'high': epu_aligned > 400,
        'low': epu_aligned < 100,
        'mid': (epu_aligned >= 100) & (epu_aligned <= 400),
    }

    # 权重配置: (高EPU时V权重, 高EPU时R权重, 低EPU时V权重, 低EPU时R权重)
    weight_configs = [
        # (名称, V_high, R_high, V_low, R_low)
        ('温和(0.35/0.65, 0.45/0.55)', 0.35, 0.65, 0.45, 0.55),
        ('中等(0.30/0.70, 0.50/0.50)', 0.30, 0.70, 0.50, 0.50),
        ('激进(0.20/0.80, 0.60/0.40)', 0.20, 0.80, 0.60, 0.40),
        ('极端(0.10/0.90, 0.70/0.30)', 0.10, 0.90, 0.70, 0.30),
        ('纯反转(0.0/1.0, 0.4/0.6)', 0.00, 1.00, 0.40, 0.60),
        ('反转增强(0.25/0.75, 0.55/0.45)', 0.25, 0.75, 0.55, 0.45),
        ('价值增强(0.35/0.65, 0.60/0.40)', 0.35, 0.65, 0.60, 0.40),
    ]

    print(f"\n  EPU分类方法: {len(epu_states)}种")
    print(f"  权重配置: {len(weight_configs)}组")
    print(f"  总实验数: {len(epu_states) * len(weight_configs)}")

    best_sharpe = -999
    best_name = ''

    for state_name, state_dict in epu_states.items():
        print(f"\n  --- EPU分类: {state_name} ---")
        high_mask = state_dict['high']
        low_mask = state_dict['low']

        for wcfg_name, v_high, r_high, v_low, r_low in weight_configs:
            v_w = pd.Series(0.4, index=df_ret.index)
            r_w = pd.Series(0.6, index=df_ret.index)

            for date in df_ret.index:
                if pd.isna(epu_aligned.loc[date]):
                    continue
                if high_mask.loc[date]:
                    v_w.loc[date] = v_high
                    r_w.loc[date] = r_high
                elif low_mask.loc[date]:
                    v_w.loc[date] = v_low
                    r_w.loc[date] = r_low

            # 回测
            port_ret, port_turnover = run_backtest_with_weights(
                df_ret, value_signal, reversal_signal, v_w, r_w, K=K)

            # MA过滤器
            port_ma_ret, _ = apply_market_filter(port_ret, market_returns=market_ret)

            ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ma_ret)
            tover_ann = port_turnover.mean() * 12

            exp_name = f'{state_name}|{wcfg_name}'
            short_name = f'{state_name}|V={v_high}/{v_low}'

            print(f"    {short_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, "
                  f"回撤={dd:.2%}, 换手={tover_ann:.0%}")

            all_results[exp_name] = {
                'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                'ann_vol': vol, 'turnover': tover_ann,
                'returns': port_ma_ret, 'cum_wealth': cum,
                'v_high': v_high, 'v_low': v_low,
                'state_method': state_name
            }

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = exp_name

    print(f"\n  最优EPU调权重: {best_name}, 夏普={best_sharpe:.2f}")

    # ========================================================================
    # 额外实验: EPU连续调整 (非离散分类)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EPU连续调整: EPU水平线性映射到权重")
    print("=" * 80)

    # 将EPU标准化到[0,1]（shift(1)确保不使用当日极值）
    epu_min = epu_aligned.expanding().min().shift(1)
    epu_max = epu_aligned.expanding().max().shift(1)
    epu_norm = (epu_aligned - epu_min) / (epu_max - epu_min)
    epu_norm = epu_norm.clip(0, 1).fillna(0.5)

    for v_range in [(0.2, 0.6), (0.1, 0.7), (0.3, 0.5), (0.0, 0.8)]:
        v_min, v_max = v_range
        # EPU高 → v_weight接近v_min (反转权重高)
        # EPU低 → v_weight接近v_max (价值权重高)
        v_w_cont = v_max - epu_norm * (v_max - v_min)
        r_w_cont = 1 - v_w_cont

        port_ret, port_turnover = run_backtest_with_weights(
            df_ret, value_signal, reversal_signal, v_w_cont, r_w_cont, K=K)
        port_ma_ret, _ = apply_market_filter(port_ret, market_returns=market_ret)

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ma_ret)
        tover_ann = port_turnover.mean() * 12

        name = f'连续(V={v_min}~{v_max})'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")

        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': port_ma_ret, 'cum_wealth': cum
        }

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = name

    # ========================================================================
    # 额外实验: EPU变化率连续调整
    # ========================================================================
    print("\n" + "=" * 80)
    print("EPU变化率连续调整")
    print("=" * 80)

    epu_change_norm = epu_change.clip(-1, 1).fillna(0)
    # 变化率正 → EPU升高 → 反转权重↑
    # 变化率负 → EPU降低 → 价值权重↑

    for v_range in [(0.2, 0.6), (0.1, 0.7), (0.3, 0.5)]:
        v_min, v_max = v_range
        # epu_change_norm ∈ [-1, 1]
        # 映射: change=-1 → v=v_max, change=+1 → v=v_min
        v_w_chg = v_max - (epu_change_norm + 1) / 2 * (v_max - v_min)
        r_w_chg = 1 - v_w_chg

        port_ret, port_turnover = run_backtest_with_weights(
            df_ret, value_signal, reversal_signal, v_w_chg, r_w_chg, K=K)
        port_ma_ret, _ = apply_market_filter(port_ret, market_returns=market_ret)

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ma_ret)
        tover_ann = port_turnover.mean() * 12

        name = f'变化率(V={v_min}~{v_max})'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tover_ann:.0%}")

        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'turnover': tover_ann,
            'returns': port_ma_ret, 'cum_wealth': cum
        }

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = name

    # ========================================================================
    # 结果汇总
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验结果汇总 (按夏普排序)")
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

    # 格式化输出
    print(f"\n{'策略':<45} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'波动率':>8} {'换手率':>6}")
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(f"{row['策略']:<45} {row['年化收益']:>7.2%} {row['夏普比率']:>6.2f} "
              f"{row['最大回撤']:>7.2%} {row['年化波动率']:>7.2%} {row['换手率(年)']:>5.0%}")

    # Top 10
    print(f"\n--- Top 10 (按夏普) ---")
    top10 = summary_df.head(10)
    for _, row in top10.iterrows():
        print(f"  {row['策略']}: 夏普={row['夏普比率']:.2f}, 年化={row['年化收益']:.2%}, "
              f"回撤={row['最大回撤']:.2%}, 换手={row['换手率(年)']:.0%}")

    # ========================================================================
    # 可视化
    # ========================================================================
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 累计净值 - Top5 + 基准
    ax1 = axes[0, 0]
    top5_names = summary_df.head(5)['策略'].tolist()
    for name in ['基准(+MA)'] + top5_names:
        if name in all_results and 'cum_wealth' in all_results[name]:
            label = name if len(name) < 30 else name[:27] + '...'
            all_results[name]['cum_wealth'].plot(ax=ax1, label=label, linewidth=1.5)
    ax1.set_title('EPU调权重: 累计净值 (Top5 + 基准)', fontsize=13)
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3)

    # 夏普 vs 换手率散点图
    ax2 = axes[0, 1]
    for _, row in summary_df.iterrows():
        name = row['策略']
        if name.startswith('基准'):
            ax2.scatter(row['换手率(年)'], row['夏普比率'], color='red', s=100, zorder=5, marker='*')
            ax2.annotate('基准', (row['换手率(年)'], row['夏普比率']), fontsize=8)
        else:
            color = 'steelblue' if 'median' in name or 'quantile' in name else \
                    'orange' if '连续' in name else 'green' if '变化率' in name else 'gray'
            ax2.scatter(row['换手率(年)'], row['夏普比率'], color=color, s=30, alpha=0.6)
    ax2.set_xlabel('年化换手率', fontsize=11)
    ax2.set_ylabel('夏普比率', fontsize=11)
    ax2.set_title('夏普 vs 换手率', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # 按EPU分类方法分组: 夏普箱线图
    ax3 = axes[1, 0]
    group_data = {}
    for name, r in all_results.items():
        if name.startswith('基准'):
            continue
        if isinstance(r, dict) and 'sharpe' in r:
            method = r.get('state_method', name.split('|')[0] if '|' in name else name)
            if method not in group_data:
                group_data[method] = []
            group_data[method].append(r['sharpe'])
    if group_data:
        labels = list(group_data.keys())
        data = [group_data[k] for k in labels]
        ax3.boxplot(data, labels=labels)
        ax3.axhline(y=base_ma_sharpe, color='red', linestyle='--', alpha=0.5, label=f'基准夏普={base_ma_sharpe:.2f}')
        ax3.set_title('EPU分类方法 vs 夏普比率', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # EPU时间序列 + 高低EPU标注
    ax4 = axes[1, 1]
    epu_plot = epu_aligned.dropna()
    ax4.plot(epu_plot.index, epu_plot.values, color='steelblue', linewidth=0.8)
    ax4.axhline(y=epu_median.iloc[-1], color='orange', linestyle='--', alpha=0.5, label=f'Median={epu_median.iloc[-1]:.0f}')
    ax4.axhline(y=epu_p75.iloc[-1], color='red', linestyle='--', alpha=0.5, label=f'P75={epu_p75.iloc[-1]:.0f}')
    ax4.set_title('中国政策不确定性指数 (EPU)', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'epu_factor_weight_experiment.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'epu_factor_weight_experiment.png'}")
    plt.close()

    summary_df.to_excel(output_dir / 'epu_factor_weight_summary.xlsx', index=False)
    print(f"✓ 汇总表已保存: {output_dir / 'epu_factor_weight_summary.xlsx'}")

    return all_results, summary_df


if __name__ == '__main__':
    results, summary = main()
    print("\n" + "=" * 80)
    print("EPU调整因子权重实验完成！")
    print("=" * 80)
