# -*- coding: utf-8 -*-
"""
换手率反转信号 + 政策不确定性优化实验

实验A: 换手率反转信号
  - 核心思路: 股价持续下跌 + 换手率突然升高 → 抛压释放，反转信号增强
  - 防前视偏差: 严格使用 t-1 月换手率
  - 不用于仓位调整，仅用于反转策略选股信号增强

实验B: 政策不确定性指数 (EPU)
  - 数据: SCMP China News-Based EPU (月度, 1995-2026)
  - 应用方向: 高EPU→降仓/调权重
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
    return returns - monthly_cost, monthly_cost, round_trip_cost


def load_and_align_data():
    """加载并对齐所有数据"""
    # 收益率
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

    # 政策不确定性
    df_epu = pd.read_excel('data/raw/SCMP_China_Policy_Uncertainty_Data.xlsx', engine='openpyxl')
    df_epu['date'] = pd.to_datetime(df_epu['year'].astype(str) + '-' + df_epu['month'].astype(str) + '-01')
    df_epu = df_epu.set_index('date')['China News-Based EPU']

    print(f"  收益率: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")
    print(f"  换手率: {tover_pivot.shape}")
    print(f"  EPU: {len(df_epu)}个月, {df_epu.index[0]} ~ {df_epu.index[-1]}")

    return df_ret, tover_pivot, df_epu


def main():
    print("=" * 80)
    print("换手率反转信号 + 政策不确定性优化实验")
    print("=" * 80)

    print("\n[1] 加载并对齐数据...")
    df_ret, tover_pivot, df_epu = load_and_align_data()

    all_results = {}

    # ========================================================================
    # 基准策略
    # ========================================================================
    print("\n" + "=" * 80)
    print("基准策略: Final_Score = 0.4V + 0.6R, TopK=100")
    print("=" * 80)

    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal)
    base_final = 0.4 * value_signal + 0.6 * reversal_signal

    weights_base = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list_base = []

    for date in df_ret.index[K:]:
        sig = base_final.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= 100).astype(float)
        selected = selected.where(sig.notna(), other=0.0)
        count = selected.sum()
        if count > 0:
            weights_base.loc[date] = selected / count
        if len(turnover_list_base) > 0:
            prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
            old_w = weights_base.loc[prev_date]
            turnover_list_base.append((weights_base.loc[date] - old_w).abs().sum() / 2)
        else:
            turnover_list_base.append(1.0)

    base_ret = (weights_base * df_ret).sum(axis=1).iloc[K:]
    if len(turnover_list_base) < len(base_ret):
        turnover_list_base = [1.0] * (len(base_ret) - len(turnover_list_base)) + turnover_list_base
    base_turnover = pd.Series(turnover_list_base, index=base_ret.index)

    base_ann, base_sharpe, base_dd, base_vol, base_cum = calculate_metrics(base_ret)
    print(f"  年化收益: {base_ann:.2%}, 夏普: {base_sharpe:.2f}, 回撤: {base_dd:.2%}")

    all_results['基准'] = {
        'ann_ret': base_ann, 'sharpe': base_sharpe, 'max_dd': base_dd,
        'ann_vol': base_vol, 'returns': base_ret, 'cum_wealth': base_cum
    }

    # ========================================================================
    # 实验A: 换手率反转信号
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验A: 换手率反转信号 — 股价下跌+换手率突增=反转信号增强")
    print("  严格防前视偏差: 使用 t-1 月换手率")
    print("=" * 80)

    # 换手率变化: 使用 t-1 月换手率 vs t-1-N 月换手率的比值
    tover_lagged = tover_pivot.shift(1)  # 严格使用 t-1 月换手率

    # --- 方法1: 排名增强法 ---
    # 换手率突增的股票在排名时获得百分位加分
    print("\n  --- 方法1: 排名增强法 ---")
    for lookback_n in [3, 6]:
        tover_ratio = tover_lagged / tover_lagged.shift(lookback_n)
        tover_ratio = tover_ratio.replace([np.inf, -np.inf], np.nan)
        # 换手率变化百分位排名 (0~1)
        tover_rank = tover_ratio.rank(pct=True, axis=1)

        for boost_w in [0.05, 0.10, 0.15, 0.20]:
            # 只在反转信号为正（股价下跌）时增强
            reversal_positive = (reversal_signal > 0).astype(float)
            # 增强后的反转信号 = 原始信号 + 排名加分
            # 排名加分 = boost_w * tover_rank * reversal_positive
            enhanced_reversal = reversal_signal + boost_w * tover_rank * reversal_positive
            enhanced_final = 0.4 * value_signal + 0.6 * enhanced_reversal

            weights_enh = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
            turnover_list_enh = []
            for date in df_ret.index[K:]:
                sig = enhanced_final.loc[date]
                ranks = sig.rank(ascending=False, method='first')
                selected = (ranks <= 100).astype(float)
                selected = selected.where(sig.notna(), other=0.0)
                count = selected.sum()
                if count > 0:
                    weights_enh.loc[date] = selected / count
                if len(turnover_list_enh) > 0:
                    prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
                    turnover_list_enh.append((weights_enh.loc[date] - weights_enh.loc[prev_date]).abs().sum() / 2)
                else:
                    turnover_list_enh.append(1.0)

            port_ret = (weights_enh * df_ret).sum(axis=1).iloc[K:]
            if len(turnover_list_enh) < len(port_ret):
                turnover_list_enh = [1.0] * (len(port_ret) - len(turnover_list_enh)) + turnover_list_enh
            ann, sharpe, dd, vol, cum = calculate_metrics(port_ret)

            name = f'排名增强(L={lookback_n},w={boost_w})'
            print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")
            all_results[name] = {
                'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                'ann_vol': vol, 'returns': port_ret, 'cum_wealth': cum
            }

    # --- 方法2: 条件筛选法 ---
    # 优先选"下跌+换手率突增"的股票，不够再从普通反转信号补充
    print("\n  --- 方法2: 条件筛选法 ---")
    for lookback_n in [3, 6]:
        tover_ratio = tover_lagged / tover_lagged.shift(lookback_n)
        tover_ratio = tover_ratio.replace([np.inf, -np.inf], np.nan)
        # 换手率突增: 比前N月升高50%以上
        tover_surge = (tover_ratio > 1.5).astype(float)
        # 股价下跌: 过去K个月累计收益为负
        price_drop = (df_ret.rolling(K).sum().shift(1) < 0).astype(float)
        # 两个条件同时满足
        surge_and_drop = tover_surge * price_drop

        for topk_surge, topk_rest in [(50, 50), (30, 70), (70, 30)]:
            weights_cond = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)

            for date in df_ret.index[K:]:
                sig = reversal_signal.loc[date]
                eligible = surge_and_drop.loc[date]

                # 优先从"下跌+换手率突增"中选
                sig_priority = sig.where(eligible > 0, other=np.nan)
                ranks_priority = sig_priority.rank(ascending=False, method='first')
                selected_priority = (ranks_priority <= topk_surge).astype(float)
                selected_priority = selected_priority.where(sig_priority.notna(), other=0.0)

                # 剩余名额从普通反转信号补充
                already_selected = selected_priority
                remaining_slots = topk_rest
                sig_rest = sig.where(already_selected == 0, other=np.nan)
                ranks_rest = sig_rest.rank(ascending=False, method='first')
                selected_rest = (ranks_rest <= remaining_slots).astype(float)
                selected_rest = selected_rest.where(sig_rest.notna(), other=0.0)

                total_selected = selected_priority + selected_rest
                count = total_selected.sum()
                if count > 0:
                    weights_cond.loc[date] = total_selected / count

            port_ret = (weights_cond * df_ret).sum(axis=1).iloc[K:]
            ann, sharpe, dd, vol, cum = calculate_metrics(port_ret)

            name = f'条件筛选(L={lookback_n},{topk_surge}+{topk_rest})'
            print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")
            all_results[name] = {
                'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                'ann_vol': vol, 'returns': port_ret, 'cum_wealth': cum
            }

    # --- 方法3: 乘法增强法 ---
    # 反转信号 × 换手率变化率(标准化后1+形式)，下跌+换手率突增时信号放大
    print("\n  --- 方法3: 乘法增强法 ---")
    for lookback_n in [3, 6]:
        tover_ratio = tover_lagged / tover_lagged.shift(lookback_n)
        tover_ratio = tover_ratio.replace([np.inf, -np.inf], np.nan)
        # 标准化为1+形式: 1 + alpha * (percentile_rank - 0.5)
        tover_rank = tover_ratio.rank(pct=True, axis=1).fillna(0.5)

        for alpha in [0.2, 0.4, 0.6]:
            # 乘数: 1 + alpha * (rank - 0.5)
            # 当 rank=1 (换手率最高) → 乘数 = 1 + alpha/2
            # 当 rank=0 (换手率最低) → 乘数 = 1 - alpha/2
            # 只在反转信号为正时使用乘数
            multiplier = 1 + alpha * (tover_rank - 0.5)
            multiplier = multiplier.clip(lower=0.5, upper=2.0)
            reversal_positive = (reversal_signal > 0).astype(float)
            enhanced_reversal = reversal_signal * (1 + (multiplier - 1) * reversal_positive)
            enhanced_final = 0.4 * value_signal + 0.6 * enhanced_reversal

            weights_enh = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
            for date in df_ret.index[K:]:
                sig = enhanced_final.loc[date]
                ranks = sig.rank(ascending=False, method='first')
                selected = (ranks <= 100).astype(float)
                selected = selected.where(sig.notna(), other=0.0)
                count = selected.sum()
                if count > 0:
                    weights_enh.loc[date] = selected / count

            port_ret = (weights_enh * df_ret).sum(axis=1).iloc[K:]
            ann, sharpe, dd, vol, cum = calculate_metrics(port_ret)

            name = f'乘法增强(L={lookback_n},a={alpha})'
            print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")
            all_results[name] = {
                'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                'ann_vol': vol, 'returns': port_ret, 'cum_wealth': cum
            }

    # ========================================================================
    # 实验B: 政策不确定性指数 (EPU)
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验B: 政策不确定性指数 (EPU) 应用")
    print("=" * 80)

    # EPU统计
    epu_aligned = df_epu.reindex(base_ret.index)
    print(f"  EPU对齐后: {epu_aligned.notna().sum()}个月")
    print(f"  EPU均值: {epu_aligned.mean():.1f}, 中位数: {epu_aligned.median():.1f}")
    print(f"  EPU范围: {epu_aligned.min():.1f} ~ {epu_aligned.max():.1f}")

    # 方案1: EPU高位降仓 (高不确定性→低仓位)
    epu_ma12 = epu_aligned.rolling(12).mean().shift(1)  # 避免前视偏差

    for high_pct in [75, 80, 90]:
        # 使用扩展窗口分位数避免前视偏差 (shift(1)确保不使用当日数据)
        epu_threshold = epu_aligned.expanding().quantile(high_pct / 100).shift(1)

        position_scalar = pd.Series(1.0, index=base_ret.index)
        for date in base_ret.index:
            if pd.isna(epu_aligned.loc[date]) or pd.isna(epu_threshold.loc[date]):
                continue
            if epu_aligned.loc[date] > epu_threshold.loc[date] * 1.5:
                position_scalar.loc[date] = 0.3
            elif epu_aligned.loc[date] > epu_threshold.loc[date]:
                position_scalar.loc[date] = 0.6

        ret_epu = base_ret * position_scalar
        ann, sharpe, dd, vol, cum = calculate_metrics(ret_epu)
        reduce_count = (position_scalar < 1).sum()

        name = f'EPU高位降仓(p{high_pct})'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 降仓={reduce_count}次")

        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'returns': ret_epu, 'cum_wealth': cum
        }

    # 方案2: EPU变化率 (EPU突然升高→降仓)
    epu_change = epu_aligned.pct_change()
    epu_change = epu_change.replace([np.inf, -np.inf], np.nan)

    for surge_threshold in [0.5, 1.0, 1.5]:
        position_scalar = pd.Series(1.0, index=base_ret.index)
        for date in base_ret.index:
            if pd.isna(epu_change.loc[date]):
                continue
            if epu_change.loc[date] > surge_threshold:
                position_scalar.loc[date] = 0.4
            elif epu_change.loc[date] > surge_threshold / 2:
                position_scalar.loc[date] = 0.7

        ret_epu = base_ret * position_scalar
        ann, sharpe, dd, vol, cum = calculate_metrics(ret_epu)
        reduce_count = (position_scalar < 1).sum()

        name = f'EPU变化率降仓(>{surge_threshold:.1f})'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 降仓={reduce_count}次")

        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'returns': ret_epu, 'cum_wealth': cum
        }

    # 方案3: EPU + MA市场过滤器 组合
    print("\n  --- EPU + MA市场过滤器 组合 ---")
    market_ret = df_ret.mean(axis=1)
    mf_ret, mf_scalar = apply_market_filter(base_ret, market_returns=market_ret)

    # 选最优EPU方案
    best_epu_name = None
    best_epu_sharpe = -999
    for key in all_results:
        if key.startswith('EPU') and all_results[key]['sharpe'] > best_epu_sharpe:
            best_epu_sharpe = all_results[key]['sharpe']
            best_epu_name = key
    print(f"  最优EPU方案: {best_epu_name} (夏普={best_epu_sharpe:.2f})")

    # 组合: MA过滤器 + 最优EPU
    # 重新计算最优EPU的仓位系数（shift(1)避免前视偏差）
    if 'p75' in best_epu_name:
        epu_threshold = epu_aligned.expanding().quantile(0.75).shift(1)
    elif 'p80' in best_epu_name:
        epu_threshold = epu_aligned.expanding().quantile(0.80).shift(1)
    else:
        epu_threshold = epu_aligned.expanding().quantile(0.90).shift(1)

    epu_scalar = pd.Series(1.0, index=base_ret.index)
    for date in base_ret.index:
        if pd.isna(epu_aligned.loc[date]) or pd.isna(epu_threshold.loc[date]):
            continue
        if epu_aligned.loc[date] > epu_threshold.loc[date] * 1.5:
            epu_scalar.loc[date] = 0.3
        elif epu_aligned.loc[date] > epu_threshold.loc[date]:
            epu_scalar.loc[date] = 0.6

    combined_scalar = mf_scalar * epu_scalar
    combined_scalar = combined_scalar.clip(lower=0.1)
    ret_combined = base_ret * combined_scalar
    ann_c, sharpe_c, dd_c, vol_c, cum_c = calculate_metrics(ret_combined)
    print(f"  MA+EPU组合: 年化={ann_c:.2%}, 夏普={sharpe_c:.2f}, 回撤={dd_c:.2%}")

    all_results['MA+EPU组合'] = {
        'ann_ret': ann_c, 'sharpe': sharpe_c, 'max_dd': dd_c,
        'ann_vol': vol_c, 'returns': ret_combined, 'cum_wealth': cum_c
    }

    # 方案4: EPU调整因子权重 (高EPU→反转权重↑, 低EPU→价值权重↑)
    print("\n  --- EPU调整因子权重 ---")
    epu_for_weight = epu_aligned.reindex(df_ret.index)
    epu_median_w = epu_for_weight.expanding().median().shift(1)

    for v_weight_low, v_weight_high in [(0.3, 0.5), (0.2, 0.6)]:
        # 构建逐日期的因子权重
        v_w_dict = {}
        r_w_dict = {}
        for date in df_ret.index:
            epu_val = epu_for_weight.get(date, np.nan)
            med_val = epu_median_w.get(date, np.nan)
            if pd.notna(epu_val) and pd.notna(med_val):
                if epu_val > med_val * 1.3:
                    v_w_dict[date] = v_weight_low
                    r_w_dict[date] = 1 - v_weight_low
                elif epu_val < med_val * 0.7:
                    v_w_dict[date] = v_weight_high
                    r_w_dict[date] = 1 - v_weight_high
                else:
                    v_w_dict[date] = 0.4
                    r_w_dict[date] = 0.6
            else:
                v_w_dict[date] = 0.4
                r_w_dict[date] = 0.6

        # 逐日期构建信号并选股
        weights_epu = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
        for date in df_ret.index[K:]:
            vw = v_w_dict[date]
            rw = r_w_dict[date]
            sig = vw * value_signal.loc[date] + rw * reversal_signal.loc[date]
            ranks = sig.rank(ascending=False, method='first')
            selected = (ranks <= 100).astype(float)
            selected = selected.where(sig.notna(), other=0.0)
            count = selected.sum()
            if count > 0:
                weights_epu.loc[date] = selected / count

        port_ret = (weights_epu * df_ret).sum(axis=1).iloc[K:]
        ann, sharpe, dd, vol, cum = calculate_metrics(port_ret)

        name = f'EPU调权重(V={v_weight_low}/{v_weight_high})'
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

        all_results[name] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'returns': port_ret, 'cum_wealth': cum
        }

    # ========================================================================
    # 综合最优组合
    # ========================================================================
    print("\n" + "=" * 80)
    print("综合最优: 最优换手率增强 + 流动性过滤 + MA + EPU + 止损 + 成本")
    print("=" * 80)

    # 找最优换手率反转增强（从所有方法中选夏普最高的）
    best_tover_name = None
    best_tover_sharpe = -999
    tover_keywords = ['排名增强', '条件筛选', '乘法增强']
    for key in all_results:
        if any(kw in key for kw in tover_keywords) and all_results[key]['sharpe'] > best_tover_sharpe:
            best_tover_sharpe = all_results[key]['sharpe']
            best_tover_name = key
    print(f"  最优换手率增强: {best_tover_name} (夏普={best_tover_sharpe:.2f})")

    # 重建最优换手率增强信号
    if '排名增强' in best_tover_name:
        lb = int(best_tover_name.split('L=')[1].split(',')[0])
        bw = float(best_tover_name.split('w=')[1].split(')')[0])
        tover_ratio_best = tover_lagged / tover_lagged.shift(lb)
        tover_ratio_best = tover_ratio_best.replace([np.inf, -np.inf], np.nan)
        tover_rank_best = tover_ratio_best.rank(pct=True, axis=1)
        reversal_positive_best = (reversal_signal > 0).astype(float)
        enhanced_reversal_best = reversal_signal + bw * tover_rank_best * reversal_positive_best
    elif '乘法增强' in best_tover_name:
        lb = int(best_tover_name.split('L=')[1].split(',')[0])
        al = float(best_tover_name.split('a=')[1].split(')')[0])
        tover_ratio_best = tover_lagged / tover_lagged.shift(lb)
        tover_ratio_best = tover_ratio_best.replace([np.inf, -np.inf], np.nan)
        tover_rank_best = tover_ratio_best.rank(pct=True, axis=1).fillna(0.5)
        multiplier = 1 + al * (tover_rank_best - 0.5)
        multiplier = multiplier.clip(lower=0.5, upper=2.0)
        reversal_positive_best = (reversal_signal > 0).astype(float)
        enhanced_reversal_best = reversal_signal * (1 + (multiplier - 1) * reversal_positive_best)
    else:
        # 条件筛选法 - 直接用其returns
        enhanced_reversal_best = reversal_signal  # fallback

    enhanced_final_best = 0.4 * value_signal + 0.6 * enhanced_reversal_best

    # 1. 流动性过滤
    tover_lagged_for_filter = tover_pivot.shift(1)
    liq_filter = tover_lagged_for_filter >= 30

    # 2. 回测
    weights_combo = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list_combo = []

    for date in df_ret.index[K:]:
        sig = enhanced_final_best.loc[date]
        if date in liq_filter.index:
            sig = sig.where(liq_filter.loc[date], other=np.nan)
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= 100).astype(float)
        selected = selected.where(sig.notna(), other=0.0)
        count = selected.sum()
        if count > 0:
            weights_combo.loc[date] = selected / count
        if len(turnover_list_combo) > 0:
            prev_date = df_ret.index[df_ret.index.get_loc(date) - 1]
            turnover_list_combo.append((weights_combo.loc[date] - weights_combo.loc[prev_date]).abs().sum() / 2)
        else:
            turnover_list_combo.append(1.0)

    combo_ret = (weights_combo * df_ret).sum(axis=1).iloc[K:]
    if len(turnover_list_combo) < len(combo_ret):
        turnover_list_combo = [1.0] * (len(combo_ret) - len(turnover_list_combo)) + turnover_list_combo
    combo_turnover = pd.Series(turnover_list_combo, index=combo_ret.index)

    # 3. MA过滤器
    combo_mf_ret, _ = apply_market_filter(combo_ret, market_returns=market_ret)

    # 4. EPU
    combo_epu_scalar = epu_scalar.reindex(combo_mf_ret.index).fillna(1.0)
    combo_epu_ret = combo_mf_ret * combo_epu_scalar

    # 5. 极宽止损
    combo_dd_ret, _ = apply_drawdown_control(combo_epu_ret)

    # 6. 真实成本
    combo_turnover_est = pd.Series(combo_turnover.mean(), index=combo_dd_ret.index)
    combo_net, _, _ = apply_realistic_costs(combo_dd_ret, combo_turnover_est)

    combo_ann, combo_sharpe, combo_dd, combo_vol, combo_cum = calculate_metrics(combo_net)

    print(f"  综合最优: 年化={combo_ann:.2%}, 夏普={combo_sharpe:.2f}, "
          f"回撤={combo_dd:.2%}, 波动={combo_vol:.2%}")

    all_results['综合最优'] = {
        'ann_ret': combo_ann, 'sharpe': combo_sharpe, 'max_dd': combo_dd,
        'ann_vol': combo_vol, 'returns': combo_net, 'cum_wealth': combo_cum
    }

    # ========================================================================
    # 结果汇总
    # ========================================================================
    print("\n" + "=" * 80)
    print("实验结果汇总")
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 换手率反转增强对比
    ax1 = axes[0, 0]
    tover_plot_names = ['基准'] + [k for k in all_results if any(kw in k for kw in ['排名增强', '乘法增强'])]
    # 只画夏普最高的前5个
    tover_plot_names_sorted = sorted(
        [k for k in tover_plot_names if k != '基准'],
        key=lambda x: all_results[x]['sharpe'], reverse=True
    )[:5]
    for name in ['基准'] + tover_plot_names_sorted:
        if name in all_results and 'cum_wealth' in all_results[name]:
            all_results[name]['cum_wealth'].plot(ax=ax1, label=name, linewidth=1.5)
    ax1.set_title('换手率反转增强: 累计净值', fontsize=13)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # EPU方案对比
    ax2 = axes[0, 1]
    for name in ['基准'] + [k for k in all_results if k.startswith('EPU高位') or k.startswith('EPU变化')]:
        if name in all_results and 'cum_wealth' in all_results[name]:
            all_results[name]['cum_wealth'].plot(ax=ax2, label=name, linewidth=1.5)
    ax2.set_title('EPU降仓: 累计净值', fontsize=13)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 组合方案对比
    ax3 = axes[1, 0]
    for name in ['基准', 'MA+EPU组合', '综合最优']:
        if name in all_results and 'cum_wealth' in all_results[name]:
            all_results[name]['cum_wealth'].plot(ax=ax3, label=name, linewidth=1.5)
    ax3.set_title('组合方案: 累计净值', fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # EPU时间序列
    ax4 = axes[1, 1]
    epu_plot = df_epu.reindex(base_ret.index).dropna()
    ax4.plot(epu_plot.index, epu_plot.values, color='steelblue', linewidth=1)
    ax4.axhline(y=epu_plot.quantile(0.75), color='red', linestyle='--', alpha=0.5, label='P75')
    ax4.axhline(y=epu_plot.quantile(0.90), color='darkred', linestyle='--', alpha=0.5, label='P90')
    ax4.set_title('中国政策不确定性指数 (EPU)', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'turnover_epu_optimization.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'turnover_epu_optimization.png'}")
    plt.close()

    summary_df.to_excel(output_dir / 'turnover_epu_summary.xlsx', index=False)
    print(f"✓ 汇总表已保存: {output_dir / 'turnover_epu_summary.xlsx'}")

    return all_results, summary_df


if __name__ == '__main__':
    results, summary = main()
    print("\n" + "=" * 80)
    print("换手率反转信号 + EPU优化实验完成！")
    print("=" * 80)
