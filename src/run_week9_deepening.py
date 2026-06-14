# -*- coding: utf-8 -*-
"""
Week 9: 三大优化方向实验
方向一：稳健反转 — 基本面过滤 + Fama-MacBeth交互项回归
方向二：简即是美 — 方法论提炼（文档工作，无需代码）
方向三：MA+反转互补性 — 条件分析 + 事件研究
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 工具函数（从week8复用）
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


def run_backtest(df_ret, signal_df, topk=100, K=6, weight_method='inv_var',
                 vol_lookback=12, hold_buffer=0, stock_filter=None):
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    prev_selected = None

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date].copy()
        ranks = sig.rank(ascending=False, method='first')

        # 股票过滤（基本面/壳资源等）
        if stock_filter is not None and date in stock_filter.index:
            valid_stocks = stock_filter.loc[date]
            sig = sig.where(valid_stocks, other=np.nan)
            ranks = sig.rank(ascending=False, method='first')

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

    return port_ret, turnover_series, weights


def get_market_state(returns, ma_window=12):
    """返回每期市场状态: bull/bear_oscillate/bear_trend"""
    cum_market = (1 + returns).cumprod()
    ma = cum_market.rolling(ma_window).mean().shift(1).shift(1)
    ma_slope = ma.diff(3)

    states = pd.Series('bull', index=returns.index)
    for date in returns.index:
        if pd.isna(ma.loc[date]):
            states.loc[date] = 'unknown'
            continue
        if cum_market.loc[date] < ma.loc[date]:
            if pd.notna(ma_slope.loc[date]) and ma_slope.loc[date] < 0:
                states.loc[date] = 'bear_trend'
            else:
                states.loc[date] = 'bear_oscillate'
    return states


# ============================================================
# 方向一：稳健反转实验
# ============================================================

def load_financial_data():
    """加载并清洗财务数据，构建ROE/杠杆/应计利润指标"""
    print("\n" + "="*80)
    print("方向一：稳健反转 — 加载财务数据")
    print("="*80)

    # --- FI_T2: 净利润 + ROE ---
    fi_t2 = pd.read_excel('data/raw/FI_T2.xlsx', engine='openpyxl')
    fi_t2_clean = fi_t2.iloc[2:].copy()
    fi_t2_clean.columns = ['Stkcd', 'ShortName', 'Accper', 'NetProfit', 'ROE']
    fi_t2_clean['NetProfit'] = pd.to_numeric(fi_t2_clean['NetProfit'], errors='coerce')
    fi_t2_clean['ROE'] = pd.to_numeric(fi_t2_clean['ROE'], errors='coerce')
    fi_t2_clean['Stkcd'] = fi_t2_clean['Stkcd'].astype(str).str.zfill(6)
    fi_t2_clean['Accper'] = pd.to_datetime(fi_t2_clean['Accper'], errors='coerce')
    fi_t2_clean = fi_t2_clean.dropna(subset=['Accper', 'NetProfit'])
    print(f"  FI_T2(净利润+ROE): {fi_t2_clean.shape}, 股票数={fi_t2_clean['Stkcd'].nunique()}")

    # --- FS_Combas: 账面价值(股东权益) ---
    fs_combas = pd.read_excel('data/raw/FS_Combas.xlsx', engine='openpyxl')
    fs_combas_clean = fs_combas.iloc[2:].copy()
    fs_combas_clean.columns = ['Stkcd', 'ShortName', 'Accper', 'Typrep', 'BookEquity']
    fs_combas_clean['BookEquity'] = pd.to_numeric(fs_combas_clean['BookEquity'], errors='coerce')
    fs_combas_clean['Stkcd'] = fs_combas_clean['Stkcd'].astype(str).str.zfill(6)
    fs_combas_clean['Accper'] = pd.to_datetime(fs_combas_clean['Accper'], errors='coerce')
    # 只保留合并报表(Typrep=A)
    fs_combas_clean = fs_combas_clean[fs_combas_clean['Typrep'] == 'A']
    fs_combas_clean = fs_combas_clean.dropna(subset=['Accper', 'BookEquity'])
    print(f"  FS_Combas(股东权益): {fs_combas_clean.shape}, 股票数={fs_combas_clean['Stkcd'].nunique()}")

    # --- ROEW数据: 加权ROE + 扣非净利润 ---
    roew_data = pd.read_excel('data/raw/2000-2007ROEW、net.xlsx', engine='openpyxl')
    roew_data = roew_data.rename(columns={
        '上市公司代码_Comcd': 'Stkcd',
        '信息发布日期_Infopubdt': 'InfoPubDt',
        '净资产收益率(加权)(%)_ROEW': 'ROEW',
        '扣除非经常性损益后净利润(元)_NetprfCut': 'NetprfCut'
    })
    roew_data['Stkcd'] = roew_data['Stkcd'].astype(str).str.zfill(6)
    roew_data['InfoPubDt'] = pd.to_datetime(roew_data['InfoPubDt'], errors='coerce')
    roew_data['ROEW'] = pd.to_numeric(roew_data['ROEW'], errors='coerce')
    roew_data['NetprfCut'] = pd.to_numeric(roew_data['NetprfCut'], errors='coerce')
    roew_data = roew_data.dropna(subset=['InfoPubDt'])
    print(f"  ROEW(加权ROE+扣非净利润): {roew_data.shape}, 股票数={roew_data['Stkcd'].nunique()}")

    return fi_t2_clean, fs_combas_clean, roew_data


def build_fundamental_filters(df_ret, fi_t2, fs_combas, roew_data):
    """
    构建基本面过滤条件: ROE>0, 杠杆<0.9, 应计利润占比<0.2
    返回: stock_filter (DataFrame, 每期True=保留)
    """
    print("\n  构建基本面过滤条件...")

    # 合并财务数据: 用FI_T2的净利润 + FS_Combas的股东权益 → 计算杠杆
    # 用ROEW数据的扣非净利润 → 计算应计利润

    # Step 1: 构建报告期查找表
    def get_available_report_date(trade_year, trade_month):
        if trade_month <= 4:
            return (trade_year - 1, 9)
        elif trade_month <= 8:
            return (trade_year, 3)
        elif trade_month <= 10:
            return (trade_year, 6)
        else:
            return (trade_year, 9)

    # Step 2: 为每个交易月构建基本面指标
    stock_filter = pd.DataFrame(True, index=df_ret.index, columns=df_ret.columns)
    roe_panel = pd.DataFrame(np.nan, index=df_ret.index, columns=df_ret.columns)
    leverage_panel = pd.DataFrame(np.nan, index=df_ret.index, columns=df_ret.columns)
    accrual_panel = pd.DataFrame(np.nan, index=df_ret.index, columns=df_ret.columns)

    # 预处理: 建立FI_T2查找表 (stkcd, year, month) -> NetProfit, ROE
    fi_t2['year'] = fi_t2['Accper'].dt.year
    fi_t2['month'] = fi_t2['Accper'].dt.month
    fi_t2_lookup = {}
    for _, row in fi_t2.iterrows():
        key = (row['Stkcd'], row['year'], row['month'])
        fi_t2_lookup[key] = {'NetProfit': row['NetProfit'], 'ROE': row['ROE']}

    # 预处理: 建立FS_Combas查找表 (stkcd, year, month) -> BookEquity
    fs_combas['year'] = fs_combas['Accper'].dt.year
    fs_combas['month'] = fs_combas['Accper'].dt.month
    fs_lookup = {}
    for _, row in fs_combas.iterrows():
        key = (row['Stkcd'], row['year'], row['month'])
        fs_lookup[key] = row['BookEquity']

    # 预处理: 建立ROEW查找表 (stkcd, pubdate) -> ROEW, NetprfCut
    roew_data_sorted = roew_data.sort_values(['Stkcd', 'InfoPubDt'])

    match_count = 0
    total_count = 0

    for date in df_ret.index:
        trade_year = date.year
        trade_month = date.month
        ry, rm = get_available_report_date(trade_year, trade_month)

        for stkcd in df_ret.columns:
            total_count += 1

            # 查找净利润和ROE
            fi_key = (stkcd, ry, rm)
            fi_info = fi_t2_lookup.get(fi_key)

            # 查找股东权益
            be_info = fs_lookup.get(fi_key)

            if fi_info is None or be_info is None:
                # 尝试年报
                fi_key_annual = (stkcd, ry if rm != 12 else ry - 1, 12)
                fi_info = fi_t2_lookup.get(fi_key_annual, fi_info)
                be_info = fs_lookup.get(fi_key_annual, be_info)

            if fi_info is None or be_info is None:
                stock_filter.loc[date, stkcd] = False
                continue

            net_profit = fi_info.get('NetProfit', np.nan)
            roe_val = fi_info.get('ROE', np.nan)
            book_equity = be_info

            if pd.isna(net_profit) or pd.isna(book_equity) or book_equity == 0:
                stock_filter.loc[date, stkcd] = False
                continue

            # 计算杠杆 = 总负债/总资产 ≈ 1 - 股东权益/总资产
            # 但我们没有总资产数据，用净利润/ROE估算净资产，再用BookEquity
            # 简化: 用ROE>0作为盈利筛选
            # leverage = 1 - book_equity / total_assets (无总资产，用近似)
            # 改用: 如果净利润>0且ROE>0，视为基本面稳健

            # ROE
            roe_panel.loc[date, stkcd] = roe_val

            # 应计利润: (净利润 - 经营现金流) / 总资产
            # 没有经营现金流数据，用扣非净利润近似: accrual ≈ (净利润 - 扣非净利润) / |净利润|
            # 如果扣非净利润远小于净利润 → 应计利润占比高 → 盈利质量差
            # 简化: 用FI_T2的净利润和ROEW的扣非净利润

            # 先用简化版过滤: ROE > 0 且 净利润 > 0
            if pd.notna(roe_val) and roe_val > 0 and net_profit > 0:
                stock_filter.loc[date, stkcd] = True
                match_count += 1
            else:
                stock_filter.loc[date, stkcd] = False

    coverage = match_count / total_count * 100 if total_count > 0 else 0
    print(f"  基本面过滤: 匹配{match_count}/{total_count}, 覆盖率={coverage:.1f}%")

    # 统计每期通过过滤的股票数
    filter_counts = stock_filter.sum(axis=1)
    print(f"  每期通过过滤的股票数: 均值={filter_counts.mean():.0f}, "
          f"中位数={filter_counts.median():.0f}, "
          f"最小={filter_counts.min():.0f}, 最大={filter_counts.max():.0f}")

    return stock_filter, roe_panel


def experiment_robust_reversal(df_ret, K=6):
    """方向一：稳健反转实验"""
    print("\n" + "="*80)
    print("方向一：稳健反转实验")
    print("="*80)

    # 加载财务数据
    fi_t2, fs_combas, roew_data = load_financial_data()

    # 构建基本面过滤
    stock_filter, roe_panel = build_fundamental_filters(df_ret, fi_t2, fs_combas, roew_data)

    # 构建信号
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)

    # 加载价值因子数据
    vf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'value_factor.parquet')
    if os.path.exists(vf_path):
        value_signal = pd.read_parquet(vf_path)
        value_signal.index = pd.to_datetime(value_signal.index)
        common_cols = df_ret.columns.intersection(value_signal.columns)
        value_signal = value_signal.reindex(index=df_ret.index, columns=common_cols)
    else:
        value_signal = df_ret.rolling(12, min_periods=1).sum().shift(1)

    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    # 市场收益
    market_ret = df_ret.mean(axis=1)

    print("\n--- 实验1A: 基本面过滤 vs 无过滤 ---")
    results = []

    # 无过滤
    port_ret_nf, tover_nf, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                              weight_method='inv_var', hold_buffer=80)
    port_ret_nf, _ = apply_market_filter(port_ret_nf, market_ret)
    port_ret_nf, _ = apply_vol_target(port_ret_nf, target_vol=0.12)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_nf)
    results.append(('无过滤(原始)', ann, sharpe, dd, vol))

    # ROE>0过滤
    roe_filter = roe_panel > 0
    port_ret_roe, tover_roe, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                               weight_method='inv_var', hold_buffer=80,
                                               stock_filter=roe_filter)
    port_ret_roe, _ = apply_market_filter(port_ret_roe, market_ret)
    port_ret_roe, _ = apply_vol_target(port_ret_roe, target_vol=0.12)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_roe)
    results.append(('ROE>0过滤', ann, sharpe, dd, vol))

    # 净利润>0过滤
    profit_filter = stock_filter  # 已经是ROE>0且净利润>0
    port_ret_pf, tover_pf, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                              weight_method='inv_var', hold_buffer=80,
                                              stock_filter=profit_filter)
    port_ret_pf, _ = apply_market_filter(port_ret_pf, market_ret)
    port_ret_pf, _ = apply_vol_target(port_ret_pf, target_vol=0.12)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_pf)
    results.append(('ROE>0+净利润>0过滤', ann, sharpe, dd, vol))

    # ROE top 50%过滤（只保留盈利能力前50%的股票）
    roe_ranks = roe_panel.rank(pct=True, axis=1)
    roe_top50_filter = roe_ranks >= 0.50
    port_ret_roe50, tover_roe50, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                                    weight_method='inv_var', hold_buffer=80,
                                                    stock_filter=roe_top50_filter)
    port_ret_roe50, _ = apply_market_filter(port_ret_roe50, market_ret)
    port_ret_roe50, _ = apply_vol_target(port_ret_roe50, target_vol=0.12)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_roe50)
    results.append(('ROE前50%过滤', ann, sharpe, dd, vol))

    # ROE top 30%过滤
    roe_top30_filter = roe_ranks >= 0.70
    port_ret_roe30, tover_roe30, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                                    weight_method='inv_var', hold_buffer=80,
                                                    stock_filter=roe_top30_filter)
    port_ret_roe30, _ = apply_market_filter(port_ret_roe30, market_ret)
    port_ret_roe30, _ = apply_vol_target(port_ret_roe30, target_vol=0.12)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_roe30)
    results.append(('ROE前30%过滤', ann, sharpe, dd, vol))

    print(f"\n{'配置':<20} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'波动率':>10}")
    print("-" * 60)
    for name, ann, sharpe, dd, vol in results:
        print(f"{name:<20} {ann:>9.2f}% {sharpe:>8.2f} {dd:>9.2f}% {vol:>9.2f}%")

    # --- 实验1B: Fama-MacBeth截面回归 ---
    print("\n--- 实验1B: Fama-MacBeth截面回归（反转×基本面交互项） ---")
    fm_results = fama_maceth_regression(df_ret, reversal_signal, roe_panel, value_signal)

    return results, fm_results


def fama_maceth_regression(df_ret, reversal_signal, roe_panel, value_signal):
    """
    Fama-MacBeth截面回归:
    模型1: 下月收益 = α + β₁·反转 + β₂·(反转×价值) + ε
    模型2: 下月收益 = α + β₁·反转 + β₂·(反转×ROE) + ε
    模型3: 下月收益 = α + β₁·反转 + β₂·ROE + β₃·(反转×ROE) + ε
    """
    print("  运行Fama-MacBeth回归...")

    future_ret = df_ret.shift(-1)  # 下月收益

    models = {
        '模型1: 反转+反转×12M': ['reversal', 'rev_x_value'],
        '模型2: 反转+反转×ROE': ['reversal', 'rev_x_roe'],
        '模型3: 反转+ROE+反转×ROE': ['reversal', 'roe', 'rev_x_roe'],
    }

    # 标准化信号
    rev_std = reversal_signal.rank(pct=True, axis=1) - 0.5
    val_std = value_signal.rank(pct=True, axis=1) - 0.5
    roe_std = roe_panel.rank(pct=True, axis=1) - 0.5

    # 交互项
    rev_x_value = rev_std * val_std
    rev_x_roe = rev_std * roe_std

    all_results = {}

    for model_name, vars_list in models.items():
        # 每期截面回归
        coef_list = {v: [] for v in vars_list}
        r2_list = []

        for date in df_ret.index:
            y = future_ret.loc[date].dropna()
            if len(y) < 50:
                continue

            # 构建X
            X_dict = {'reversal': rev_std.loc[date], 'value': val_std.loc[date],
                      'roe': roe_std.loc[date], 'rev_x_value': rev_x_value.loc[date],
                      'rev_x_roe': rev_x_roe.loc[date]}

            X = pd.DataFrame({v: X_dict[v].loc[y.index] for v in vars_list})
            X = X.dropna()

            if len(X) < 50:
                continue

            y_aligned = y.loc[X.index]

            # OLS回归
            try:
                X_const = np.column_stack([np.ones(len(X)), X.values])
                beta, _, _, _ = np.linalg.lstsq(X_const, y_aligned.values, rcond=None)

                for i, v in enumerate(vars_list):
                    coef_list[v].append(beta[i + 1])
                # R²
                y_pred = X_const @ beta
                ss_res = np.sum((y_aligned.values - y_pred) ** 2)
                ss_tot = np.sum((y_aligned.values - y_aligned.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2_list.append(r2)
            except:
                continue

        # Fama-MacBeth t统计量
        fm_result = {}
        for v in vars_list:
            coefs = np.array(coef_list[v])
            if len(coefs) > 0:
                mean_coef = coefs.mean()
                se_coef = coefs.std() / np.sqrt(len(coefs))
                t_stat = mean_coef / se_coef if se_coef > 0 else 0
                fm_result[v] = {'mean': mean_coef, 't': t_stat, 'n': len(coefs)}

        fm_result['avg_R2'] = np.mean(r2_list) if r2_list else 0
        all_results[model_name] = fm_result

    # 打印结果
    print(f"\n  {'模型':<30} {'变量':<15} {'均值系数':>10} {'t值':>8} {'平均R²':>8}")
    print("-" * 75)
    for model_name, fm_result in all_results.items():
        avg_r2 = fm_result.pop('avg_R2', 0)
        first = True
        for v, stats_dict in fm_result.items():
            r2_str = f"{avg_r2:.4f}" if first else ""
            name_str = model_name if first else ""
            sig = "***" if abs(stats_dict['t']) > 3.0 else ("**" if abs(stats_dict['t']) > 2.0 else ("*" if abs(stats_dict['t']) > 1.645 else ""))
            print(f"  {name_str:<30} {v:<15} {stats_dict['mean']:>10.6f} {stats_dict['t']:>7.2f}{sig} {r2_str:>8}")
            first = False

    return all_results


# ============================================================
# 方向三：MA+反转互补性实验
# ============================================================

def experiment_ma_reversal_complement(df_ret, K=6):
    """方向三：MA趋势过滤与反转的互补性机制分析"""
    print("\n" + "="*80)
    print("方向三：MA+反转互补性机制分析")
    print("="*80)

    # 构建信号
    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)

    # 加载价值因子数据
    vf_path2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'value_factor.parquet')
    if os.path.exists(vf_path2):
        value_signal = pd.read_parquet(vf_path2)
        value_signal.index = pd.to_datetime(value_signal.index)
        common_cols = df_ret.columns.intersection(value_signal.columns)
        value_signal = value_signal.reindex(index=df_ret.index, columns=common_cols)
    else:
        value_signal = df_ret.rolling(12, min_periods=1).sum().shift(1)

    combined_signal = 0.6 * reversal_signal + 0.4 * apply_standardization(value_signal)

    market_ret = df_ret.mean(axis=1)

    # 获取市场状态
    states = get_market_state(market_ret)

    # --- 实验3A: 不同市场状态下反转信号的收益 ---
    print("\n--- 实验3A: 不同市场状态下反转信号的未来收益 ---")

    future_1m = df_ret.shift(-1)  # 下1月
    future_3m = df_ret.rolling(3).sum().shift(-3)  # 下3月累计

    # 按反转信号分组: Top100 vs Bottom100
    condition_stats = []

    for state_name in ['bull', 'bear_oscillate', 'bear_trend', 'unknown']:
        state_dates = states[states == state_name].index

        if len(state_dates) == 0:
            continue

        # 对每个状态日期，计算Top100反转组合的未来收益
        ret_1m_list = []
        ret_3m_list = []
        win_count = 0
        total_count = 0

        for date in state_dates:
            sig = reversal_signal.loc[date].dropna()
            if len(sig) < 100:
                continue

            top100 = sig.nlargest(100).index
            # 等权组合下1月收益
            r1 = future_1m.loc[date, top100].mean()
            # 等权组合下3月累计收益
            r3_vals = future_3m.loc[date, top100].dropna()
            r3 = r3_vals.mean() if len(r3_vals) > 0 else np.nan

            if pd.notna(r1):
                ret_1m_list.append(r1)
                total_count += 1
                if r1 > 0:
                    win_count += 1
            if pd.notna(r3):
                ret_3m_list.append(r3)

        avg_1m = np.mean(ret_1m_list) if ret_1m_list else 0
        avg_3m = np.mean(ret_3m_list) if ret_3m_list else 0
        win_rate = win_count / total_count if total_count > 0 else 0

        state_cn = {'bull': '牛市(净值>MA12)', 'bear_oscillate': '熊市震荡(净值<MA,斜率≥0)',
                    'bear_trend': '熊市下行(净值<MA,斜率<0)', 'unknown': '未知'}
        condition_stats.append({
            'state': state_cn.get(state_name, state_name),
            'count': len(state_dates),
            'avg_1m': avg_1m,
            'avg_3m': avg_3m,
            'win_rate': win_rate
        })

    print(f"\n  {'市场状态':<30} {'触发次数':>8} {'未来1月均值':>12} {'未来3月均值':>12} {'胜率':>8}")
    print("-" * 75)
    for s in condition_stats:
        print(f"  {s['state']:<30} {s['count']:>8} {s['avg_1m']:>11.2f}% {s['avg_3m']:>11.2f}% {s['win_rate']:>7.1f}%")

    # --- 实验3B: 反转因子最差10月事件研究 ---
    print("\n--- 实验3B: 反转因子崩溃事件研究 ---")

    # 计算每月反转因子收益（Top100 - Bottom100多空）
    factor_ret = pd.Series(index=df_ret.index, dtype=float)
    for date in df_ret.index:
        sig = reversal_signal.loc[date].dropna()
        if len(sig) < 200:
            factor_ret.loc[date] = np.nan
            continue
        top100 = sig.nlargest(100).index
        bottom100 = sig.nsmallest(100).index
        long_ret = df_ret.loc[date, top100].mean()
        short_ret = df_ret.loc[date, bottom100].mean()
        factor_ret.loc[date] = long_ret - short_ret

    # 找最差10个月
    worst10 = factor_ret.dropna().nsmallest(10)
    print(f"\n  反转因子最差10个月:")
    print(f"  {'日期':<20} {'因子收益':>10} {'市场状态':>25} {'市场收益':>10}")
    print("-" * 70)

    worst_in_bear = 0
    for date, ret in worst10.items():
        state = states.loc[date] if date in states.index else 'unknown'
        state_cn = {'bull': '牛市', 'bear_oscillate': '熊市震荡', 'bear_trend': '熊市下行', 'unknown': '未知'}
        mkt_ret = market_ret.loc[date] if date in market_ret.index else np.nan
        print(f"  {str(date)[:10]:<20} {ret:>9.2f}% {state_cn.get(state, state):>25} {mkt_ret:>9.2f}%")
        if state in ('bear_oscillate', 'bear_trend'):
            worst_in_bear += 1

    print(f"\n  最差10个月中，{worst_in_bear}个月处于熊市状态({worst_in_bear/10*100:.0f}%)")

    # --- 实验3C: 反转因子在不同市场状态下的月度统计 ---
    print("\n--- 实验3C: 反转因子条件统计 ---")

    for state_name in ['bull', 'bear_oscillate', 'bear_trend']:
        state_dates = states[states == state_name].index
        state_factor_ret = factor_ret.loc[factor_ret.index.isin(state_dates)].dropna()
        if len(state_factor_ret) > 0:
            state_cn = {'bull': '牛市', 'bear_oscillate': '熊市震荡', 'bear_trend': '熊市下行'}
            print(f"  {state_cn[state_name]}: 均值={state_factor_ret.mean()*100:.2f}%, "
                  f"t值={state_factor_ret.mean()/state_factor_ret.std()*np.sqrt(len(state_factor_ret)):.2f}, "
                  f"胜率={(state_factor_ret>0).mean()*100:.1f}%, "
                  f"n={len(state_factor_ret)}")

    # --- 可视化 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MA趋势过滤与反转互补性分析', fontsize=16, fontweight='bold')

    # 图1: 不同市场状态下反转信号未来1月收益
    ax1 = axes[0, 0]
    state_labels = [s['state'][:6] for s in condition_stats]
    avg_1m_vals = [s['avg_1m'] * 100 for s in condition_stats]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
    bars = ax1.bar(state_labels, avg_1m_vals, color=colors[:len(state_labels)])
    ax1.set_ylabel('未来1月平均收益(%)')
    ax1.set_title('不同市场状态下反转信号收益')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    for bar, val in zip(bars, avg_1m_vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

    # 图2: 反转因子累计收益 + MA状态标注
    ax2 = axes[0, 1]
    cum_factor = (1 + factor_ret.dropna()).cumprod()
    ax2.plot(cum_factor.index, cum_factor.values, 'b-', linewidth=1, label='反转因子累计净值')

    # 标注熊市下行期
    bear_trend_dates = states[states == 'bear_trend'].index
    for d in bear_trend_dates:
        if d in cum_factor.index:
            ax2.axvline(x=d, color='red', alpha=0.05, linewidth=2)

    ax2.set_title('反转因子累计净值（红色阴影=熊市下行期）')
    ax2.set_ylabel('累计净值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 最差10月事件图
    ax3 = axes[1, 0]
    factor_ret_valid = factor_ret.dropna()
    ax3.bar(range(len(factor_ret_valid)), factor_ret_valid.values * 100,
            color=['#e74c3c' if r < 0 else '#2ecc71' for r in factor_ret_valid.values],
            alpha=0.5, width=1)

    # 标注最差10月
    worst_indices = [list(factor_ret_valid.index).index(d) for d in worst10.index]
    for idx in worst_indices:
        ax3.bar(idx, factor_ret_valid.iloc[idx] * 100, color='red', alpha=0.9, width=1)

    ax3.set_title('反转因子月度收益（红色=最差10月）')
    ax3.set_ylabel('月度收益(%)')
    ax3.set_xlabel('月份')
    ax3.grid(True, alpha=0.3)

    # 图4: 市场状态分布饼图 + 条件胜率
    ax4 = axes[1, 1]
    state_counts = states.value_counts()
    state_counts_cn = {'bull': '牛市', 'bear_oscillate': '熊市震荡',
                       'bear_trend': '熊市下行', 'unknown': '未知'}
    labels = [f"{state_counts_cn.get(s, s)}\n({c}月)" for s, c in state_counts.items()]
    colors_pie = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
    color_map = {'bull': '#2ecc71', 'bear_oscillate': '#f39c12',
                 'bear_trend': '#e74c3c', 'unknown': '#95a5a6'}
    pie_colors = [color_map.get(s, '#95a5a6') for s in state_counts.index]
    ax4.pie(state_counts.values, labels=labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('市场状态分布')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'week9_ma_reversal_complement.png', dpi=150, bbox_inches='tight')
    print(f"\n  图表已保存: results/week9_ma_reversal_complement.png")

    return condition_stats, worst10, worst_in_bear


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*80)
    print("Week 9: 三大优化方向实验")
    print("="*80)

    # 加载收益数据（已是宽格式）
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', engine='openpyxl')
    # 检查是否已经是宽格式
    if 'Stkcd' in df_ret.columns:
        df_ret = df_ret.pivot(index='Trdmnt', columns='Stkcd', values='Mretwd')
    else:
        # 已经是宽格式，第一列是Trdmnt
        df_ret = df_ret.set_index('Trdmnt')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret = df_ret.sort_index()
    # 数据已是小数格式（如-0.08=-8%），无需再除100
    # 剔除缺失过多的股票
    valid_stocks = df_ret.notna().sum() > len(df_ret) * 0.5
    df_ret = df_ret.loc[:, valid_stocks]
    df_ret = df_ret.fillna(0)
    print(f"收益数据: {df_ret.shape}, {df_ret.index[0]}~{df_ret.index[-1]}")

    K = 6

    # 方向一：稳健反转
    d1_results, d1_fm = experiment_robust_reversal(df_ret, K=K)

    # 方向三：MA+反转互补性
    d3_stats, d3_worst10, d3_worst_in_bear = experiment_ma_reversal_complement(df_ret, K=K)

    # 汇总
    print("\n" + "="*80)
    print("Week 9 实验汇总")
    print("="*80)

    print("\n--- 方向一：稳健反转 ---")
    print("  核心发现: 基本面过滤是否提升反转策略？")

    print("\n--- 方向三：MA+反转互补性 ---")
    print(f"  反转因子最差10月中{d3_worst_in_bear}次处于熊市({d3_worst_in_bear/10*100:.0f}%)")
    print("  MA过滤器本质是识别反转效应的生效条件")


if __name__ == '__main__':
    results = main()
