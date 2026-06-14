# -*- coding: utf-8 -*-
"""
Week 8: 教程启发优化实验
1. CH-3因子归因（检验真alpha）
2. 壳资源过滤（剔除最小30%市值）
3. EP价值因子对比实验
4. 换手约束优化
5. 子样本稳健性检验
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


def run_backtest(df_ret, signal_df, topk=100, K=6, weight_method='equal',
                 vol_lookback=12, hold_buffer=0, shell_filter=None):
    """
    shell_filter: pd.Series, 每期True=保留的股票（已剔除壳资源）
    """
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    prev_selected = None

    for date in df_ret.index[K:]:
        sig = signal_df.loc[date].copy()
        ranks = sig.rank(ascending=False, method='first')

        # 壳资源过滤
        if shell_filter is not None and date in shell_filter.index:
            valid_stocks = shell_filter.loc[date]
            sig = sig.where(valid_stocks, other=np.nan)
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


def construct_ch3_factors(df_ret, mcap_data, ep_data, rf_data):
    """
    构造CH-3因子: MKT, SMB, VMG
    按Liu-Stambaugh-Yuan(2019)方法
    """
    print("  构造CH-3因子...")

    # 剔除最小30%市值（壳资源）
    mcap_ranks = mcap_data.rank(pct=True, axis=1)
    shell_filter = mcap_ranks >= 0.30  # 保留70%大市值

    # 市场因子 MKT
    # 用70%大市值股票的市值加权收益
    mkt_ret = pd.Series(index=df_ret.index, dtype=float)
    for date in df_ret.index:
        if date not in shell_filter.index:
            mkt_ret.loc[date] = df_ret.loc[date].mean()
            continue
        valid = shell_filter.loc[date]
        if valid.sum() == 0:
            mkt_ret.loc[date] = df_ret.loc[date].mean()
            continue
        valid_mcap = mcap_data.loc[date][valid]
        valid_ret = df_ret.loc[date][valid]
        if valid_mcap.sum() > 0:
            w = valid_mcap / valid_mcap.sum()
            mkt_ret.loc[date] = (w * valid_ret).sum()
        else:
            mkt_ret.loc[date] = valid_ret.mean()

    # 无风险利率
    rf_monthly = rf_data.reindex(df_ret.index).fillna(0) / 12

    MKT = mkt_ret - rf_monthly

    # SMB和VMG: 2x3分组
    SMB_list = []
    VMG_list = []
    dates = []

    for date in df_ret.index:
        if date not in shell_filter.index or date not in ep_data.index:
            continue
        valid = shell_filter.loc[date]
        if valid.sum() < 20:
            continue

        # 获取有效股票
        valid_mcap = mcap_data.loc[date][valid]
        valid_ep = ep_data.loc[date][valid]
        valid_ret = df_ret.loc[date][valid]

        # 按市值中位数分Small/Big
        mcap_median = valid_mcap.median()
        small_mask = valid_mcap <= mcap_median
        big_mask = valid_mcap > mcap_median

        # 按EP分3组: V(前30%), M(中40%), G(后30%)
        # 负EP归入G组
        ep_positive = valid_ep[valid_ep > 0]
        if len(ep_positive) < 10:
            continue

        ep_p30 = ep_positive.quantile(0.30)
        ep_p70 = ep_positive.quantile(0.70)

        def classify_ep(ep_val):
            if ep_val <= 0:
                return 'G'
            elif ep_val <= ep_p30:
                return 'G'
            elif ep_val <= ep_p70:
                return 'M'
            else:
                return 'V'

        ep_group = valid_ep.apply(classify_ep)

        # 6个组合的市值加权收益
        port_rets = {}
        for size_label, size_mask in [('S', small_mask), ('B', big_mask)]:
            for ep_label in ['V', 'M', 'G']:
                mask = size_mask & (ep_group == ep_label)
                if mask.sum() > 0:
                    group_mcap = valid_mcap[mask]
                    group_ret = valid_ret[mask]
                    if group_mcap.sum() > 0:
                        w = group_mcap / group_mcap.sum()
                        port_rets[f'{size_label}/{ep_label}'] = (w * group_ret).sum()
                    else:
                        port_rets[f'{size_label}/{ep_label}'] = group_ret.mean()
                else:
                    port_rets[f'{size_label}/{ep_label}'] = 0.0

        # SMB = (SV+SM+SG)/3 - (BV+BM+BG)/3
        smb = (port_rets.get('S/V', 0) + port_rets.get('S/M', 0) + port_rets.get('S/G', 0)) / 3 \
            - (port_rets.get('B/V', 0) + port_rets.get('B/M', 0) + port_rets.get('B/G', 0)) / 3

        # VMG = (SV+BV)/2 - (SG+BG)/2
        vmg = (port_rets.get('S/V', 0) + port_rets.get('B/V', 0)) / 2 \
            - (port_rets.get('S/G', 0) + port_rets.get('B/G', 0)) / 2

        SMB_list.append(smb)
        VMG_list.append(vmg)
        dates.append(date)

    SMB = pd.Series(SMB_list, index=dates)
    VMG = pd.Series(VMG_list, index=dates)

    # 对齐
    common_idx = MKT.index.intersection(SMB.index).intersection(VMG.index)
    MKT = MKT.loc[common_idx]
    SMB = SMB.loc[common_idx]
    VMG = VMG.loc[common_idx]

    print(f"  CH-3因子: {len(common_idx)}个月, {common_idx[0]}~{common_idx[-1]}")
    print(f"  MKT月均={MKT.mean():.2%}, SMB月均={SMB.mean():.2%}, VMG月均={VMG.mean():.2%}")

    return MKT, SMB, VMG, shell_filter


def factor_attribution(port_ret, MKT, SMB, VMG, strategy_name="策略"):
    """
    CH-3因子归因回归
    """
    import statsmodels.api as sm

    common_idx = port_ret.index.intersection(MKT.index)
    if len(common_idx) < 24:
        print(f"  {strategy_name}: 数据不足({len(common_idx)}月), 跳过归因")
        return None

    y = port_ret.loc[common_idx]
    X = pd.DataFrame({
        'MKT': MKT.loc[common_idx],
        'SMB': SMB.loc[common_idx],
        'VMG': VMG.loc[common_idx],
    })
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HC0')

    alpha = model.params.get('const', 0)
    alpha_t = model.tvalues.get('const', 0)
    beta_mkt = model.params.get('MKT', 0)
    beta_smb = model.params.get('SMB', 0)
    beta_vmg = model.params.get('VMG', 0)
    r2 = model.rsquared

    print(f"\n  {strategy_name} CH-3归因:")
    print(f"    alpha={alpha:.4f}/月({alpha*12:.2%}/年), t={alpha_t:.2f} {'***' if abs(alpha_t)>3 else '**' if abs(alpha_t)>2 else '*' if abs(alpha_t)>1.6 else ''}")
    print(f"    β_MKT={beta_mkt:.3f}, β_SMB={beta_smb:.3f}, β_VMG={beta_vmg:.3f}")
    print(f"    R²={r2:.3f}")

    return {
        'alpha_monthly': alpha,
        'alpha_annual': alpha * 12,
        'alpha_t': alpha_t,
        'beta_mkt': beta_mkt,
        'beta_smb': beta_smb,
        'beta_vmg': beta_vmg,
        'r2': r2,
        'model': model,
    }


def main():
    print("=" * 80)
    print("Week 8: 教程启发优化实验")
    print("=" * 80)

    # 加载数据
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)

    # 市值数据
    mcap_data = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl',
                               sheet_name=1) if False else None
    # 尝试加载市值
    try:
        # 用收益文件本身估算市值权重（如果没有单独市值数据）
        # 尝试从FI_T2获取盈利数据
        fi_t2 = pd.read_excel('data/raw/FI_T2.xlsx', engine='openpyxl')
        print(f"  FI_T2数据: {fi_t2.shape}")
        print(f"  列名: {list(fi_t2.columns[:10])}")
    except Exception as e:
        print(f"  FI_T2加载失败: {e}")
        fi_t2 = None

    # 加载无风险利率
    try:
        rf_raw = pd.read_excel('data/raw/一年期定期存款.xlsx', engine='openpyxl')
        # 查找日期列和利率列
        print(f"  无风险利率原始: shape={rf_raw.shape}, columns={list(rf_raw.columns[:5])}")
        print(f"  前3行: {rf_raw.head(3).to_dict()}")

        # 尝试设置索引
        if rf_raw.shape[1] >= 2:
            rf_raw.columns = ['date', 'rate'] + [f'col{i}' for i in range(2, rf_raw.shape[1])]
            rf_raw = rf_raw.iloc[2:]  # 跳过标题行（如果有）
            rf_raw['date'] = pd.to_datetime(rf_raw['date'], format='mixed', errors='coerce')
            rf_raw['rate'] = pd.to_numeric(rf_raw['rate'], errors='coerce')
            rf_raw = rf_raw.dropna(subset=['date', 'rate'])
            rf_raw = rf_raw.set_index('date')['rate']
            rf_data = rf_raw.sort_index()
            print(f"  无风险利率: {rf_data.shape}, {rf_data.index[0]}~{rf_data.index[-1]}")
        else:
            rf_data = pd.Series(0.015, index=df_ret.index)
    except Exception as e:
        print(f"  无风险利率加载失败: {e}")
        rf_data = pd.Series(0.015, index=df_ret.index)

    # 加载盈利和账面价值数据
    try:
        fs_combas = pd.read_excel('data/raw/FS_Combas.xlsx', engine='openpyxl')
        print(f"  FS_Combas(账面价值): {fs_combas.shape}")
        print(f"  列名: {list(fs_combas.columns)}")
    except Exception as e:
        print(f"  FS_Combas加载失败: {e}")
        fs_combas = None

    # 加载ROE数据
    try:
        roe_data = pd.read_excel('data/raw/2000-2007ROEW、net.xlsx', engine='openpyxl')
        print(f"  ROE数据: {roe_data.shape}")
        print(f"  列名: {list(roe_data.columns)}")
    except Exception as e:
        print(f"  ROE加载失败: {e}")
        roe_data = None

    # 加载股东权益数据
    try:
        equity_data = pd.read_excel('data/raw/2000-2007归母所有者权.xlsx', engine='openpyxl')
        print(f"  股东权益数据: {equity_data.shape}")
        print(f"  列名: {list(equity_data.columns)}")
    except Exception as e:
        print(f"  股东权益加载失败: {e}")
        equity_data = None

    print(f"\n  收益数据: {df_ret.shape}, {df_ret.index[0]}~{df_ret.index[-1]}")

    # ========================================================================
    # 实验1: CH-3因子归因
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验1: CH-3因子归因")
    print("=" * 80)

    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)

    # 加载价值因子数据
    vf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'value_factor.parquet')
    if os.path.exists(vf_path):
        value_signal = pd.read_parquet(vf_path)
        value_signal.index = pd.to_datetime(value_signal.index)
        common_cols = df_ret.columns.intersection(value_signal.columns)
        value_signal = value_signal.reindex(index=df_ret.index, columns=common_cols)
    else:
        value_signal = df_ret.rolling(window=12).sum().shift(1)
    value_signal = apply_standardization(value_signal)
    combo_signal = 0.4 * value_signal + 0.6 * reversal_signal
    market_ret = df_ret.mean(axis=1)

    # 构造CH-3因子
    # 市值因子：用过去收益波动率（低波动≈大市值）
    mcap_proxy = df_ret.rolling(12).std().shift(1).rank(pct=True, axis=1)  # 低波动≈大市值，shift(1)避免前视偏差

    # 价值因子
    ep_proxy = value_signal.copy()  # 使用已加载的价值因子

    MKT, SMB, VMG, shell_filter = construct_ch3_factors(
        df_ret, mcap_proxy, ep_proxy, rf_data
    )

    # 运行基准策略
    port_ret_base, tover_base, _ = run_backtest(df_ret, combo_signal, topk=100, K=K)
    port_ret_ma, _ = apply_market_filter(port_ret_base, market_ret)

    # 最优策略
    port_ret_opt, tover_opt, _ = run_backtest(df_ret, combo_signal, topk=100, K=K,
                                                weight_method='inv_var', hold_buffer=50)
    port_ret_opt, _ = apply_market_filter(port_ret_opt, market_ret)

    # VT
    rolling_vol = port_ret_opt.rolling(24).std().shift(1) * np.sqrt(12)
    vol_scalar = (0.12 / rolling_vol).clip(0.3, 2.0).fillna(1.0)
    port_ret_vt = port_ret_opt * vol_scalar

    # 归因
    attr_base = factor_attribution(port_ret_base.iloc[K:], MKT, SMB, VMG, "基准(40V+60R)")
    attr_ma = factor_attribution(port_ret_ma.iloc[K:], MKT, SMB, VMG, "+MA")
    attr_opt = factor_attribution(port_ret_opt.iloc[K:], MKT, SMB, VMG, "最优(1/σ²+MA+buffer)")
    attr_vt = factor_attribution(port_ret_vt.iloc[K:], MKT, SMB, VMG, "最优+VT(12%)")

    # ========================================================================
    # 实验2: 壳资源过滤
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验2: 壳资源过滤（剔除最小30%市值）")
    print("=" * 80)

    shell_results = {}
    for filter_pct, filter_name in [(0, '无过滤'), (0.20, '剔除20%'), (0.30, '剔除30%(CH-3标准)'), (0.40, '剔除40%'), (0.50, '剔除50%')]:
        if filter_pct == 0:
            sf = None
        else:
            sf = mcap_proxy >= filter_pct

        port_ret_s, tover_s, _ = run_backtest(df_ret, combo_signal, topk=100, K=K,
                                                shell_filter=sf)
        port_ret_s, _ = apply_market_filter(port_ret_s, market_ret)

        # VT
        rv = port_ret_s.rolling(24).std().shift(1) * np.sqrt(12)
        vs = (0.12 / rv).clip(0.3, 2.0).fillna(1.0)
        port_ret_s_vt = port_ret_s * vs

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_s_vt)
        shell_results[filter_name] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                       'ann_vol': vol, 'turnover': tover_s.mean() * 12}
        print(f"  {filter_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, "
              f"波动={vol:.2%}, 换手={tover_s.mean()*12:.0%}")

    # ========================================================================
    # 实验3: EP价值因子
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验3: EP价值因子（从FI_T2数据构造）")
    print("=" * 80)

    ep_factor = None
    if fi_t2 is not None:
        # FI_T2前两行是标题行，跳过
        fi_t2_clean = fi_t2.iloc[2:].copy()
        fi_t2_clean.columns = ['Stkcd', 'ShortName', 'Accper', 'NetProfit', 'ROE']
        fi_t2_clean['NetProfit'] = pd.to_numeric(fi_t2_clean['NetProfit'], errors='coerce')
        fi_t2_clean['ROE'] = pd.to_numeric(fi_t2_clean['ROE'], errors='coerce')
        fi_t2_clean['Accper'] = pd.to_datetime(fi_t2_clean['Accper'])
        fi_t2_clean['Stkcd'] = fi_t2_clean['Stkcd'].astype(str).str.zfill(6)
        fi_t2_clean = fi_t2_clean.dropna(subset=['NetProfit', 'Accper'])
        print(f"  FI_T2清洗后: {fi_t2_clean.shape}, 股票数={fi_t2_clean['Stkcd'].nunique()}")

        # 构造EP因子: 净利润/价格
        # 按教程合并规则: 使用最近已披露的财务报告
        def get_available_report_date(trade_year, trade_month):
            if trade_month <= 4:
                return (trade_year - 1, 9)  # 用上年三季报
            elif trade_month <= 8:
                return (trade_year, 3)  # 用当年一季报
            elif trade_month <= 10:
                return (trade_year, 6)  # 用当年半年报
            else:
                return (trade_year, 9)  # 用当年三季报

        # 按股票+报告期建立查找表
        ep_lookup = {}
        for _, row in fi_t2_clean.iterrows():
            stkcd = row['Stkcd']
            accper = row['Accper']
            profit = row['NetProfit']
            key = (stkcd, accper.year, accper.month)
            ep_lookup[key] = profit

        print(f"  EP查找表: {len(ep_lookup)}条记录")

        # 构造月度EP因子
        ep_factor = pd.DataFrame(np.nan, index=df_ret.index, columns=df_ret.columns, dtype=float)
        matched = 0
        for date in df_ret.index:
            ry, rm = date.year, date.month
            report_y, report_m = get_available_report_date(ry, rm)
            for stkcd in df_ret.columns:
                # 尝试精确匹配，然后回退
                profit = None
                for dy, dm in [(report_y, report_m), (report_y, report_m - 3),
                                (report_y - 1, 12), (report_y - 1, 9)]:
                    if dm <= 0:
                        dy, dm = dy - 1, dm + 12
                    key = (stkcd, dy, dm)
                    if key in ep_lookup:
                        profit = ep_lookup[key]
                        break
                if profit is not None:
                    ep_factor.loc[date, stkcd] = profit
                    matched += 1

        print(f"  EP因子: {matched}个匹配, 覆盖率={matched / (len(df_ret.index) * len(df_ret.columns)):.1%}")

        # EP = 净利润 / 市值
        # 简化: 直接用净利润截面排名（因为净利润与市值高度相关）
        ep_factor = ep_factor.rank(pct=True, axis=1) * 2 - 1  # 映射到[-1, 1]
        print(f"  EP因子标准化完成")

    # 测试EP因子
    if ep_factor is not None and ep_factor.notna().sum().sum() > 1000:
        print("\n  EP因子有效性测试:")
        ep_results = {}

        # 纯EP因子
        port_ret_ep, tover_ep, _ = run_backtest(df_ret, ep_factor, topk=100, K=K)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_ep)
        ep_results['纯EP'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol}
        print(f"  纯EP: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

        # EP+反转组合
        for ep_w, rev_w in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]:
            combo_ep = ep_w * ep_factor + rev_w * reversal_signal
            port_ret_ce, tover_ce, _ = run_backtest(df_ret, combo_ep, topk=100, K=K)
            port_ret_ce, _ = apply_market_filter(port_ret_ce, market_ret)
            ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_ce)
            name = f'EP({ep_w})+Rev({rev_w})+MA'
            ep_results[name] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol}
            print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

        # 对比: 价值因子 vs EP
        combo_orig = 0.4 * value_signal + 0.6 * reversal_signal
        port_ret_orig, _, _ = run_backtest(df_ret, combo_orig, topk=100, K=K)
        port_ret_orig, _ = apply_market_filter(port_ret_orig, market_ret)
        ann_orig, sharpe_orig, dd_orig, vol_orig, _, _ = calculate_metrics(port_ret_orig)
        print(f"\n  对比: 价值因子+MA: 年化={ann_orig:.2%}, 夏普={sharpe_orig:.2f}")
    else:
        print("  EP因子不可用，跳过")

    # ========================================================================
    # 实验4: 换手约束优化
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验4: 换手约束优化")
    print("=" * 80)

    turnover_results = {}

    # 不同buffer值的换手率对比
    for buffer_val in [0, 20, 50, 80, 100, 150]:
        port_ret_b, tover_b, _ = run_backtest(df_ret, combo_signal, topk=100, K=K,
                                                weight_method='inv_var', hold_buffer=buffer_val)
        port_ret_b, _ = apply_market_filter(port_ret_b, market_ret)
        rv = port_ret_b.rolling(24).std().shift(1) * np.sqrt(12)
        vs = (0.12 / rv).clip(0.3, 2.0).fillna(1.0)
        port_ret_b = port_ret_b * vs

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_b)
        avg_tover = tover_b.mean() * 12
        # 扣除交易成本
        cost_bps = 20  # 20bps
        net_ret = port_ret_b - tover_b * cost_bps / 10000
        ann_net, sharpe_net, dd_net, vol_net, _, _ = calculate_metrics(net_ret)

        name = f'buffer={buffer_val}'
        turnover_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
            'turnover': avg_tover, 'ann_net': ann_net, 'sharpe_net': sharpe_net
        }
        print(f"  {name}: 夏普={sharpe:.2f}, 换手={avg_tover:.0%}, "
              f"净夏普(@20bps)={sharpe_net:.2f}, 净年化={ann_net:.2%}")

    # ========================================================================
    # 实验5: 子样本稳健性检验
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验5: 子样本稳健性检验")
    print("=" * 80)

    # 分两半: 2005-2015 vs 2016-2025
    split_date = pd.Timestamp('2016-01-01')

    sub_results = {}
    for period_name, period_mask in [
        ('全样本(2005-2025)', df_ret.index <= df_ret.index[-1]),
        ('前半(2005-2015)', df_ret.index < split_date),
        ('后半(2016-2025)', df_ret.index >= split_date),
    ]:
        df_sub = df_ret.loc[period_mask]
        if len(df_sub) < 24:
            continue

        # 重新构造信号
        rev_sub = -df_sub.rolling(window=K).sum().shift(1)
        rev_sub = apply_standardization(rev_sub)
        val_sub = value_signal.reindex(index=df_sub.index, columns=df_sub.columns)
        val_sub = apply_standardization(val_sub)
        combo_sub = 0.4 * val_sub + 0.6 * rev_sub
        mkt_sub = df_sub.mean(axis=1)

        # 最优策略
        port_ret_sub, tover_sub, _ = run_backtest(df_sub, combo_sub, topk=100, K=K,
                                                    weight_method='inv_var', hold_buffer=50)
        port_ret_sub, _ = apply_market_filter(port_ret_sub, mkt_sub)
        rv = port_ret_sub.rolling(24).std().shift(1) * np.sqrt(12)
        vs = (0.12 / rv).clip(0.3, 2.0).fillna(1.0)
        port_ret_sub = port_ret_sub * vs

        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_sub)

        # 子样本归因
        attr_sub = factor_attribution(port_ret_sub.iloc[K:], MKT, SMB, VMG,
                                       f"{period_name}")

        sub_results[period_name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
            'attr': attr_sub
        }
        print(f"  {period_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # ========================================================================
    # 汇总
    # ========================================================================
    print("\n" + "=" * 80)
    print("Week 8 实验汇总")
    print("=" * 80)

    print("\n--- 实验2: 壳资源过滤 ---")
    print(f"{'配置':<20} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'波动率':>8} {'换手率':>6}")
    print("-" * 60)
    for name, r in sorted(shell_results.items(), key=lambda x: x[1]['sharpe'], reverse=True):
        print(f"{name:<20} {r['ann_ret']:>7.2%} {r['sharpe']:>6.2f} "
              f"{r['max_dd']:>7.2%} {r['ann_vol']:>7.2%} {r['turnover']:>5.0%}")

    print("\n--- 实验4: 换手约束 ---")
    print(f"{'配置':<15} {'夏普':>6} {'净夏普':>8} {'换手率':>6} {'净年化':>8}")
    print("-" * 50)
    for name, r in sorted(turnover_results.items(), key=lambda x: x[1]['sharpe_net'], reverse=True):
        print(f"{name:<15} {r['sharpe']:>6.2f} {r['sharpe_net']:>8.2f} "
              f"{r['turnover']:>5.0%} {r['ann_net']:>7.2%}")

    print("\n--- 实验5: 子样本稳健性 ---")
    print(f"{'期间':<20} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8}")
    print("-" * 50)
    for name, r in sub_results.items():
        print(f"{name:<20} {r['ann_ret']:>7.2%} {r['sharpe']:>6.2f} {r['max_dd']:>7.2%}")

    # 可视化
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 壳资源过滤
    ax1 = axes[0, 0]
    names = list(shell_results.keys())
    sharpes = [shell_results[n]['sharpe'] for n in names]
    ax1.bar(range(len(names)), sharpes, color='steelblue')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace('剔除', '\n剔除') for n in names], fontsize=8)
    ax1.set_ylabel('夏普比率')
    ax1.set_title('壳资源过滤效果', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 换手约束
    ax2 = axes[0, 1]
    buffers = [0, 20, 50, 80, 100, 150]
    gross_sharpes = [turnover_results[f'buffer={b}']['sharpe'] for b in buffers]
    net_sharpes = [turnover_results[f'buffer={b}']['sharpe_net'] for b in buffers]
    turnovers = [turnover_results[f'buffer={b}']['turnover'] for b in buffers]
    ax2_twin = ax2.twinx()
    ax2.plot(buffers, gross_sharpes, 'b-o', label='毛夏普')
    ax2.plot(buffers, net_sharpes, 'r-s', label='净夏普(@20bps)')
    ax2_twin.plot(buffers, turnovers, 'g--^', label='换手率')
    ax2.set_xlabel('Buffer值')
    ax2.set_ylabel('夏普比率')
    ax2_twin.set_ylabel('换手率', color='g')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title('换手约束: 夏普 vs 换手率', fontsize=13)
    ax2.grid(True, alpha=0.3)

    # 3. 子样本
    ax3 = axes[1, 0]
    sub_names = list(sub_results.keys())
    sub_sharpes = [sub_results[n]['sharpe'] for n in sub_names]
    colors = ['steelblue', 'coral', 'seagreen']
    ax3.bar(range(len(sub_names)), sub_sharpes, color=colors[:len(sub_names)])
    ax3.set_xticks(range(len(sub_names)))
    ax3.set_xticklabels([n.replace('(', '\n(') for n in sub_names], fontsize=8)
    ax3.set_ylabel('夏普比率')
    ax3.set_title('子样本稳健性', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 归因对比
    ax4 = axes[1, 1]
    if attr_base and attr_ma and attr_opt and attr_vt:
        attr_data = {
            '基准': attr_base,
            '+MA': attr_ma,
            '最优': attr_opt,
            '最优+VT': attr_vt,
        }
        x = range(len(attr_data))
        alphas = [attr_data[k]['alpha_annual'] for k in attr_data]
        alpha_ts = [attr_data[k]['alpha_t'] for k in attr_data]
        ax4.bar(x, alphas, color=['steelblue', 'coral', 'seagreen', 'gold'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(list(attr_data.keys()))
        ax4.set_ylabel('年化Alpha')
        ax4.set_title('CH-3归因: Alpha', fontsize=13)
        # 标注t值
        for i, (a, t) in enumerate(zip(alphas, alpha_ts)):
            ax4.annotate(f't={t:.1f}', (i, a), ha='center', va='bottom' if a > 0 else 'top', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Week 8: 教程启发优化实验', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'week8_tutorial_inspired.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'week8_tutorial_inspired.png'}")
    plt.close()

    return {
        'shell_results': shell_results,
        'turnover_results': turnover_results,
        'sub_results': sub_results,
        'attr_base': attr_base,
        'attr_ma': attr_ma,
        'attr_opt': attr_opt,
        'attr_vt': attr_vt,
    }


if __name__ == '__main__':
    results = main()
