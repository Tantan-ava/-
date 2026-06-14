# -*- coding: utf-8 -*-
"""
Week 11: 最终策略归因分析
1. 换手率补充
2. CH-3/CH-4因子归因
3. BARRA多因子归因（风格因子+行业因子）
4. Brinson归因（资产配置+选股+交互）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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


def apply_market_filter(returns, market_returns=None, ma_window=6,
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


def apply_vol_target(returns, target_vol=0.12, lookback=24, min_scalar=0.3, max_scalar=2.0):
    rolling_vol = returns.rolling(lookback).std().shift(1) * np.sqrt(12)
    vol_scalar = (target_vol / rolling_vol).clip(min_scalar, max_scalar)
    vol_scalar = vol_scalar.fillna(1.0)
    return returns * vol_scalar, vol_scalar


def apply_time_stop(returns, market_returns, consec_months=3, scalar_gradual=True):
    market_ret_aligned = market_returns.reindex(returns.index).fillna(0)
    underperform = returns < market_ret_aligned
    consec_under = underperform.rolling(consec_months).sum().shift(1).shift(1)
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


def build_tradeable_filter(df_ret, min_history=10):
    coverage = df_ret.rolling(12, min_periods=min_history).count().shift(1)
    is_not_new = coverage >= min_history
    is_tradable = df_ret.shift(1).abs() > 0.001
    return is_not_new & is_tradable


def run_backtest_full(df_ret, signal_df, topk=100, K=6, weight_method='inv_var',
                      vol_lookback=12, hold_buffer=80, stock_filter=None,
                      signal_threshold=0.85):
    """完整回测，返回权重和换手率"""
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    prev_selected = None

    for date in df_ret.index[K:]:
        date_idx = df_ret.index.get_loc(date)
        sig = signal_df.loc[date].copy()
        ranks = sig.rank(ascending=False, method='first')

        if stock_filter is not None and date in stock_filter.index:
            valid_stocks = stock_filter.loc[date]
            sig = sig.where(valid_stocks, other=np.nan)
            ranks = sig.rank(ascending=False, method='first')

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
            start_idx = max(0, date_idx - vol_lookback)
            hist_ret = df_ret.iloc[start_idx:date_idx]
            stock_vols = hist_ret.std()
            w_factor = 1.0 / (stock_vols ** 2).replace(0, np.nan).fillna(1)
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

    return port_ret, turnover_series, weights


# ============================================================
# CH-3 / CH-4 因子归因
# ============================================================

def construct_ch3_factors(df_ret, market_ret, value_factor=None):
    """构造CH-3因子：MKT, SMB, VMG"""
    # MKT
    mkt = market_ret

    # 市值因子（低波动≈大市值）
    stock_vol = df_ret.rolling(12, min_periods=6).std().shift(1)
    vol_median = stock_vol.median(axis=1)

    # SMB: 小波动 - 大波动
    small_mask = stock_vol.le(vol_median, axis=0)
    large_mask = stock_vol.gt(vol_median, axis=0)

    small_ret = df_ret.where(small_mask).mean(axis=1)
    large_ret = df_ret.where(large_mask).mean(axis=1)
    smb = small_ret - large_ret

    # VMG: 价值因子
    if value_factor is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'processed'
        vf_path = data_dir / 'value_factor.parquet'
        if vf_path.exists():
            value_factor = pd.read_parquet(vf_path)
            value_factor.index = pd.to_datetime(value_factor.index)
        else:
            value_factor = df_ret.rolling(12, min_periods=6).sum().shift(1)

    median_vf = value_factor.median(axis=1)

    value_mask = value_factor.le(median_vf, axis=0)
    growth_mask = value_factor.gt(median_vf, axis=0)

    value_ret = df_ret.where(value_mask).mean(axis=1)
    growth_ret = df_ret.where(growth_mask).mean(axis=1)
    vmg = value_ret - growth_ret

    return pd.DataFrame({'MKT': mkt, 'SMB': smb, 'VMG': vmg})


def construct_ch4_factors(df_ret, market_ret, value_factor=None):
    """构造CH-4因子：MKT, SMB, VMG, MOM"""
    ch3 = construct_ch3_factors(df_ret, market_ret, value_factor=value_factor)

    # MOM: 动量因子
    ret_12m = df_ret.rolling(12, min_periods=6).sum().shift(1)
    q_high = ret_12m.quantile(0.7, axis=1)
    q_low = ret_12m.quantile(0.3, axis=1)

    winner_mask = ret_12m.ge(q_high, axis=0)
    loser_mask = ret_12m.le(q_low, axis=0)

    winner_ret = df_ret.where(winner_mask).mean(axis=1)
    loser_ret = df_ret.where(loser_mask).mean(axis=1)
    mom = winner_ret - loser_ret

    ch3['MOM'] = mom
    return ch3


def factor_attribution(port_ret, factors, name='CH-3'):
    """因子归因：时间序列回归"""
    from scipy import stats

    # 对齐数据
    common_idx = port_ret.index.intersection(factors.index)
    y = port_ret.loc[common_idx]
    X = factors.loc[common_idx]

    # 添加常数项
    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    factor_names = ['Alpha'] + list(X.columns)

    # OLS回归
    beta, residuals, rank, sv = np.linalg.lstsq(X_with_const, y.values, rcond=None)

    # 计算t值
    n = len(y)
    k = X_with_const.shape[1]
    y_hat = X_with_const @ beta
    ss_res = np.sum((y.values - y_hat) ** 2)
    mse = ss_res / (n - k) if n > k else 1
    var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    se_beta = np.sqrt(np.diag(var_beta))
    t_values = beta / se_beta

    # R²
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 年化alpha
    alpha_ann = (1 + beta[0]) ** 12 - 1

    # 各因子贡献（年化）
    contributions = {}
    for i, fname in enumerate(factor_names):
        if fname == 'Alpha':
            contributions[fname] = {
                'coefficient': beta[i],
                't_value': t_values[i],
                'annual_contribution': alpha_ann,
                'p_value': 2 * (1 - stats.t.cdf(abs(t_values[i]), n - k))
            }
        else:
            factor_contrib = beta[i] * X.iloc[:, i-1].mean() * 12
            contributions[fname] = {
                'coefficient': beta[i],
                't_value': t_values[i],
                'annual_contribution': factor_contrib,
                'p_value': 2 * (1 - stats.t.cdf(abs(t_values[i]), n - k))
            }

    print(f"\n{'='*70}")
    print(f"  {name} 因子归因")
    print(f"{'='*70}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  年化Alpha = {alpha_ann:.2%}")
    print(f"\n  {'因子':<10} {'系数':>10} {'t值':>8} {'p值':>8} {'年化贡献':>10}")
    print(f"  {'-'*50}")
    for fname, info in contributions.items():
        sig = '***' if info['p_value'] < 0.01 else '**' if info['p_value'] < 0.05 else '*' if info['p_value'] < 0.1 else ''
        print(f"  {fname:<10} {info['coefficient']:>10.4f} {info['t_value']:>8.2f} "
              f"{info['p_value']:>8.4f} {info['annual_contribution']:>9.2%} {sig}")

    return contributions, r_squared, alpha_ann


# ============================================================
# BARRA 多因子归因
# ============================================================

def barra_attribution(df_ret, weights, market_ret):
    """BARRA多因子归因：风格因子+行业因子"""
    print(f"\n{'='*70}")
    print(f"  BARRA 多因子归因")
    print(f"{'='*70}")

    # 构造风格因子暴露
    # 1. Size: 市值因子（低波动=大市值）
    stock_vol = df_ret.rolling(12, min_periods=6).std().shift(1)

    # 2. Value: 价值因子（从数据文件读取）
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    vf_path = data_dir / 'value_factor.parquet'
    if vf_path.exists():
        value_factor = pd.read_parquet(vf_path)
        value_factor.index = pd.to_datetime(value_factor.index)
    else:
        value_factor = df_ret.rolling(12, min_periods=6).sum().shift(1)

    # 3. Momentum: 动量因子
    ret_6m = df_ret.rolling(6, min_periods=3).sum().shift(1)

    # 4. Reversal: 过去1M收益（反转）
    ret_1m = df_ret.shift(1)

    # 注意：Size和Volatility高度共线性（Size = -Volatility）
    # 去掉Volatility，保留4个正交因子
    factor_names = ['Size', 'Value', 'Momentum', 'Reversal']

    # 每期横截面回归
    factor_returns_list = []
    specific_returns_list = []

    for date in df_ret.index[12:]:
        if date not in weights.index:
            continue

        # 当期收益
        y = df_ret.loc[date].dropna()
        if len(y) < 50:
            continue

        # 因子暴露
        size_exp = -stock_vol.loc[date] if date in stock_vol.index else pd.Series(0, index=y.index)
        value_exp = -value_factor.loc[date] if date in value_factor.index else pd.Series(0, index=y.index)
        mom_exp = ret_6m.loc[date] if date in ret_6m.index else pd.Series(0, index=y.index)
        rev_exp = -ret_1m.loc[date] if date in ret_1m.index else pd.Series(0, index=y.index)

        # 对齐
        common = y.index
        size_exp = size_exp.reindex(common).fillna(0)
        value_exp = value_exp.reindex(common).fillna(0)
        mom_exp = mom_exp.reindex(common).fillna(0)
        rev_exp = rev_exp.reindex(common).fillna(0)

        # Z-score标准化
        def zscore(s):
            if s.std() == 0:
                return s * 0
            return (s - s.mean()) / s.std()

        X = pd.DataFrame({
            'Size': zscore(size_exp),
            'Value': zscore(value_exp),
            'Momentum': zscore(mom_exp),
            'Reversal': zscore(rev_exp)
        }, index=common)

        # 横截面回归
        try:
            model = LinearRegression()
            model.fit(X.values, y.values)
            factor_ret = pd.Series(model.coef_, index=factor_names)
            specific_ret = y - model.predict(X.values)
            factor_returns_list.append(factor_ret)
            specific_returns_list.append(specific_ret)
        except:
            continue

    if not factor_returns_list:
        print("  BARRA归因失败：无有效数据")
        return None

    factor_returns_df = pd.DataFrame(factor_returns_list)
    factor_returns_df.index = [df_ret.index[12:][i] for i in range(len(factor_returns_list))
                                if i < len(df_ret.index[12:])]

    # 组合归因
    # 计算组合在各因子上的暴露和贡献
    port_factor_contrib = {}
    for fname in factor_names:
        # 因子收益均值（年化）
        factor_ret_ann = factor_returns_df[fname].mean() * 12
        port_factor_contrib[fname] = factor_ret_ann

    # 特质收益（Alpha）
    # 组合的特质收益 = 组合收益 - 因子解释部分
    # 简化：用组合收益减去因子贡献

    print(f"\n  {'因子':<12} {'月均因子收益':>14} {'年化因子收益':>14}")
    print(f"  {'-'*44}")
    for fname in factor_names:
        monthly = factor_returns_df[fname].mean()
        annual = monthly * 12
        print(f"  {fname:<12} {monthly:>13.4%} {annual:>13.2%}")

    # 计算组合因子暴露（时间序列平均）
    print(f"\n  --- 组合因子暴露分析 ---")

    port_exposures = {}
    for date in weights.index[12:]:
        if date not in stock_vol.index:
            continue
        w = weights.loc[date]
        active = w[w > 0]
        if len(active) == 0:
            continue

        # 组合在各因子上的暴露
        size_exp = -stock_vol.loc[date].reindex(active.index).fillna(0)
        value_exp = -value_factor.loc[date].reindex(active.index).fillna(0)
        mom_exp = ret_6m.loc[date].reindex(active.index).fillna(0)
        rev_exp = -ret_1m.loc[date].reindex(active.index).fillna(0)

        # Z-score
        def zscore(s):
            if s.std() == 0:
                return s * 0
            return (s - s.mean()) / s.std()

        port_exposures[date] = {
            'Size': (active * zscore(size_exp)).sum(),
            'Value': (active * zscore(value_exp)).sum(),
            'Momentum': (active * zscore(mom_exp)).sum(),
            'Reversal': (active * zscore(rev_exp)).sum()
        }

    if port_exposures:
        port_exp_df = pd.DataFrame(port_exposures).T
        print(f"\n  {'因子':<12} {'平均暴露':>10} {'暴露标准差':>10}")
        print(f"  {'-'*36}")
        for fname in factor_names:
            print(f"  {fname:<12} {port_exp_df[fname].mean():>10.3f} {port_exp_df[fname].std():>10.3f}")

    # 因子贡献 = 暴露 × 因子收益
    if port_exposures:
        print(f"\n  --- 因子贡献分解 ---")
        print(f"  {'因子':<12} {'暴露':>8} {'因子收益':>10} {'年化贡献':>10}")
        print(f"  {'-'*44}")
        total_factor = 0
        for fname in factor_names:
            exp = port_exp_df[fname].mean()
            fret = factor_returns_df[fname].mean() * 12
            contrib = exp * fret
            total_factor += contrib
            print(f"  {fname:<12} {exp:>8.3f} {fret:>9.2%} {contrib:>9.2%}")

        print(f"  {'因子合计':<12} {'':>8} {'':>10} {total_factor:>9.2%}")

    return factor_returns_df, port_exp_df if port_exposures else None


# ============================================================
# Brinson 归因
# ============================================================

def brinson_attribution(df_ret, weights, market_ret):
    """Brinson归因：资产配置+选股+交互"""
    print(f"\n{'='*70}")
    print(f"  Brinson 归因分析")
    print(f"{'='*70}")

    # 基准：全市场等权
    # 行业分类：按波动率分5组
    stock_vol = df_ret.rolling(12, min_periods=6).std().shift(1)

    # 定义"行业"分组：按波动率五分位
    n_groups = 5
    group_labels = [f'波动率G{i+1}' for i in range(n_groups)]

    # 逐月计算Brinson归因
    brinson_results = []

    for date in df_ret.index[12:]:
        if date not in weights.index:
            continue
        if date not in stock_vol.index:
            continue

        w = weights.loc[date]
        active = w[w > 0]
        if len(active) < 10:
            continue

        # 当期收益
        ret = df_ret.loc[date]
        vol = stock_vol.loc[date]

        # 行业分组
        valid_stocks = vol.dropna().index.intersection(ret.dropna().index)
        if len(valid_stocks) < 50:
            continue

        vol_valid = vol.reindex(valid_stocks)
        ret_valid = ret.reindex(valid_stocks)

        # 分组
        try:
            groups = pd.qcut(vol_valid, n_groups, labels=group_labels, duplicates='drop')
        except:
            continue

        # 组合权重和收益（按组）
        port_group_weight = {}
        port_group_return = {}
        bench_group_weight = {}
        bench_group_return = {}

        for g in group_labels:
            g_stocks = groups[groups == g].index
            if len(g_stocks) == 0:
                continue

            # 组合权重
            w_g = active.reindex(g_stocks).fillna(0)
            pw = w_g.sum()
            if pw > 0:
                port_group_return[g] = (w_g * ret_valid.reindex(g_stocks).fillna(0)).sum() / pw if pw > 0 else 0
            else:
                port_group_return[g] = ret_valid.reindex(g_stocks).mean()
            port_group_weight[g] = pw

            # 基准权重（等权）
            bw = len(g_stocks) / len(valid_stocks)
            bench_group_weight[g] = bw
            bench_group_return[g] = ret_valid.reindex(g_stocks).mean()

        # Brinson分解
        aa = 0  # 资产配置
        ss = 0  # 选股
        ii = 0  # 交互

        for g in group_labels:
            if g not in port_group_weight:
                continue
            pw = port_group_weight[g]
            bw = bench_group_weight[g]
            pr = port_group_return.get(g, 0)
            br = bench_group_return.get(g, 0)

            aa += (pw - bw) * br
            ss += bw * (pr - br)
            ii += (pw - bw) * (pr - br)

        brinson_results.append({
            'date': date,
            'asset_allocation': aa,
            'stock_selection': ss,
            'interaction': ii,
            'active_return': aa + ss + ii
        })

    if not brinson_results:
        print("  Brinson归因失败：无有效数据")
        return None

    brinson_df = pd.DataFrame(brinson_results).set_index('date')

    # 输出结果
    print(f"\n  --- 月均分解 ---")
    print(f"  {'效应':<15} {'月均值':>10} {'t值':>8} {'年化贡献':>10}")
    print(f"  {'-'*48}")

    from scipy import stats as sp_stats
    for col in ['asset_allocation', 'stock_selection', 'interaction', 'active_return']:
        monthly = brinson_df[col]
        annual = monthly.mean() * 12
        t_val = monthly.mean() / (monthly.std() / np.sqrt(len(monthly))) if monthly.std() > 0 else 0
        label = {'asset_allocation': '资产配置', 'stock_selection': '选股',
                 'interaction': '交互效应', 'active_return': '主动收益'}[col]
        print(f"  {label:<15} {monthly.mean():>9.4%} {t_val:>8.2f} {annual:>9.2%}")

    # 按组详细分解
    print(f"\n  --- 按组分解（时间序列平均）---")
    print(f"  {'组别':<10} {'组合权重':>10} {'基准权重':>10} {'组合收益':>10} {'基准收益':>10}")
    print(f"  {'-'*54}")

    # 重新计算按组平均
    group_stats = {}
    for date in df_ret.index[12:]:
        if date not in weights.index or date not in stock_vol.index:
            continue
        w = weights.loc[date]
        active = w[w > 0]
        if len(active) < 10:
            continue
        ret = df_ret.loc[date]
        vol = stock_vol.loc[date]
        valid_stocks = vol.dropna().index.intersection(ret.dropna().index)
        if len(valid_stocks) < 50:
            continue
        vol_valid = vol.reindex(valid_stocks)
        ret_valid = ret.reindex(valid_stocks)
        try:
            groups = pd.qcut(vol_valid, n_groups, labels=group_labels, duplicates='drop')
        except:
            continue

        for g in group_labels:
            g_stocks = groups[groups == g].index
            if len(g_stocks) == 0:
                continue
            w_g = active.reindex(g_stocks).fillna(0)
            pw = w_g.sum()
            pr = (w_g * ret_valid.reindex(g_stocks).fillna(0)).sum() / pw if pw > 0 else 0
            bw = len(g_stocks) / len(valid_stocks)
            br = ret_valid.reindex(g_stocks).mean()

            if g not in group_stats:
                group_stats[g] = {'pw': [], 'bw': [], 'pr': [], 'br': []}
            group_stats[g]['pw'].append(pw)
            group_stats[g]['bw'].append(bw)
            group_stats[g]['pr'].append(pr)
            group_stats[g]['br'].append(br)

    for g in group_labels:
        if g not in group_stats:
            continue
        gs = group_stats[g]
        print(f"  {g:<10} {np.mean(gs['pw']):>9.2%} {np.mean(gs['bw']):>9.2%} "
              f"{np.mean(gs['pr']):>9.4%} {np.mean(gs['br']):>9.4%}")

    return brinson_df


# ============================================================
# 可视化
# ============================================================

def create_attribution_plots(ch3_contrib, ch4_contrib, barra_result, brinson_df, port_ret, market_ret):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Week 11: 最终策略归因分析', fontsize=16, fontweight='bold')

    # 图1: CH-3因子归因
    ax = axes[0, 0]
    if ch3_contrib:
        names = list(ch3_contrib.keys())
        contribs = [ch3_contrib[n]['annual_contribution'] for n in names]
        colors = ['#e74c3c' if n == 'Alpha' else '#3498db' for n in names]
        ax.bar(names, [c*100 for c in contribs], color=colors)
        ax.set_ylabel('年化贡献(%)')
        ax.set_title('CH-3因子归因')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    # 图2: CH-4因子归因
    ax = axes[0, 1]
    if ch4_contrib:
        names = list(ch4_contrib.keys())
        contribs = [ch4_contrib[n]['annual_contribution'] for n in names]
        colors = ['#e74c3c' if n == 'Alpha' else '#3498db' for n in names]
        ax.bar(names, [c*100 for c in contribs], color=colors)
        ax.set_ylabel('年化贡献(%)')
        ax.set_title('CH-4因子归因')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    # 图3: BARRA因子收益时序
    ax = axes[0, 2]
    if barra_result and barra_result[0] is not None:
        fr = barra_result[0]
        fr_cum = (1 + fr).cumprod()
        for col in fr.columns:
            ax.plot(fr_cum.index, fr_cum[col], label=col, linewidth=1)
        ax.set_ylabel('累积因子收益')
        ax.set_title('BARRA因子收益时序')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 图4: Brinson归因
    ax = axes[1, 0]
    if brinson_df is not None:
        cols = ['asset_allocation', 'stock_selection', 'interaction']
        labels = ['资产配置', '选股', '交互']
        means = [brinson_df[c].mean() * 12 * 100 for c in cols]
        colors = ['#3498db', '#2ecc71', '#f39c12']
        ax.bar(labels, means, color=colors)
        ax.set_ylabel('年化贡献(%)')
        ax.set_title('Brinson归因分解')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    # 图5: Brinson累积
    ax = axes[1, 1]
    if brinson_df is not None:
        for col, label, color in zip(['asset_allocation', 'stock_selection', 'interaction', 'active_return'],
                                      ['资产配置', '选股', '交互', '主动收益'],
                                      ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']):
            cum = (1 + brinson_df[col]).cumprod()
            ax.plot(cum.index, cum.values, label=label, linewidth=1.5, color=color)
        ax.set_ylabel('累积贡献')
        ax.set_title('Brinson累积贡献')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 图6: 主动收益时序
    ax = axes[1, 2]
    active_ret = port_ret - market_ret.reindex(port_ret.index).fillna(0)
    cum_active = (1 + active_ret).cumprod()
    ax.plot(cum_active.index, cum_active.values, color='#9b59b6', linewidth=1.5)
    ax.set_ylabel('累积主动收益')
    ax.set_title('累积主动收益')
    ax.axhline(y=1, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'week11_attribution.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: results/week11_attribution.png")


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*80)
    print("Week 11: 最终策略归因分析")
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

    # 构建最终策略信号
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)

    # 加载价值因子数据
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    vf_path = data_dir / 'value_factor.parquet'
    if vf_path.exists():
        value_signal = pd.read_parquet(vf_path)
        value_signal.index = pd.to_datetime(value_signal.index)
        # 对齐列名和索引
        common_cols = df_ret.columns.intersection(value_signal.columns)
        value_signal = value_signal.reindex(index=df_ret.index, columns=common_cols)
    else:
        value_signal = df_ret.rolling(12, min_periods=1).sum().shift(1)

    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    # 可交易性过滤
    tradeable_filter = build_tradeable_filter(df_ret, min_history=10)

    # 运行最终策略回测
    print("\n--- 运行最终策略回测 ---")
    port_ret, turnover, weights = run_backtest_full(
        df_ret, combined_signal, topk=100, K=K,
        weight_method='inv_var', hold_buffer=80,
        stock_filter=tradeable_filter, signal_threshold=0.85
    )

    # 应用风控
    port_ret_ma, _ = apply_market_filter(port_ret, market_ret, ma_window=6)
    port_ret_ts, _ = apply_time_stop(port_ret_ma, market_ret, consec_months=3, scalar_gradual=True)
    port_ret_final, _ = apply_vol_target(port_ret_ts, target_vol=0.12)

    # ============================================================
    # 1. 换手率信息
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  换手率分析")
    print(f"{'='*70}")

    # 对齐换手率
    turnover_aligned = turnover.reindex(port_ret_final.index).fillna(0)

    print(f"  月均换手率: {turnover_aligned.mean():.2%}")
    print(f"  年化换手率: {turnover_aligned.mean() * 12:.2%}")
    print(f"  换手率中位数(月): {turnover_aligned.median():.2%}")
    print(f"  换手率标准差(月): {turnover_aligned.std():.2%}")
    print(f"  最大月换手率: {turnover_aligned.max():.2%}")
    print(f"  最小月换手率: {turnover_aligned.min():.2%}")

    # 扣除不同成本后的净收益
    for cost_bps in [10, 20, 30, 40]:
        cost_drag = turnover_aligned * cost_bps / 10000
        net_ret = port_ret_final - cost_drag
        ann, sharpe, dd, vol, _, _ = calculate_metrics(net_ret)
        print(f"  成本{cost_bps}bps: 净年化={ann:.2%}, 净夏普={sharpe:.2f}, 净回撤={dd:.2%}")

    # ============================================================
    # 2. CH-3 / CH-4 因子归因
    # ============================================================
    ch3_factors = construct_ch3_factors(df_ret, market_ret)
    ch4_factors = construct_ch4_factors(df_ret, market_ret)

    ch3_contrib, ch3_r2, ch3_alpha = factor_attribution(port_ret_final, ch3_factors, 'CH-3')
    ch4_contrib, ch4_r2, ch4_alpha = factor_attribution(port_ret_final, ch4_factors, 'CH-4')

    # ============================================================
    # 3. BARRA 多因子归因
    # ============================================================
    # 需要用原始回测权重（未应用风控的），因为风控是组合层面的
    barra_result = barra_attribution(df_ret, weights, market_ret)

    # ============================================================
    # 4. Brinson 归因
    # ============================================================
    brinson_df = brinson_attribution(df_ret, weights, market_ret)

    # ============================================================
    # 可视化
    # ============================================================
    create_attribution_plots(ch3_contrib, ch4_contrib, barra_result, brinson_df,
                            port_ret_final, market_ret)

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  归因分析汇总")
    print(f"{'='*70}")

    ann, sharpe, dd, vol, _, _ = calculate_metrics(port_ret_final)
    print(f"\n  最终策略: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol:.2%}")
    print(f"  年化换手率: {turnover_aligned.mean() * 12:.2%}")
    print(f"  CH-3 Alpha: {ch3_alpha:.2%} (R²={ch3_r2:.4f})")
    print(f"  CH-4 Alpha: {ch4_alpha:.2%} (R²={ch4_r2:.4f})")

    return {
        'ch3': (ch3_contrib, ch3_r2, ch3_alpha),
        'ch4': (ch4_contrib, ch4_r2, ch4_alpha),
        'barra': barra_result,
        'brinson': brinson_df,
        'turnover': turnover_aligned,
        'port_ret': port_ret_final
    }


if __name__ == '__main__':
    results = main()
