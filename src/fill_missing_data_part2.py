# -*- coding: utf-8 -*-
"""补充计算缺失实验数据"""
import pandas as pd
import numpy as np
import os, sys, json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from fill_missing_data import (
    load_data, load_value_factor, calculate_metrics,
    apply_standardization, apply_market_filter, apply_vol_target, run_backtest
)

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


def main():
    print("=" * 80)
    print("补充测算缺失实验数据")
    print("=" * 80)

    df_ret = load_data()
    market_ret = df_ret.mean(axis=1)
    K = 6

    reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
    value_signal = load_value_factor(df_ret)
    combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

    results = {}

    # ====================================================================
    # 1. 换手率反转信号实验 (2.3 ④)
    # ====================================================================
    print("\n--- 1. 换手率反转信号实验 ---")
    # 需要换手率数据
    turnover_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'turnover_factor.parquet')
    if os.path.exists(turnover_path):
        tov_df = pd.read_parquet(turnover_path)
        tov_df.index = pd.to_datetime(tov_df.index)
    else:
        # 用成交量代理
        vol_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'TRD_Mnth.xlsx')
        tov_df = None

    if tov_df is not None:
        common_cols = df_ret.columns.intersection(tov_df.columns)
        tov_df = tov_df.reindex(index=df_ret.index, columns=common_cols)
        tov_change = tov_df.pct_change().shift(1)
        # 排名增强法: 信号 = reversal_rank + tov_change_rank
        rev_rank = reversal_signal.rank(pct=True, axis=1)
        tov_rank = tov_change.rank(pct=True, axis=1)
        sig_rank = 0.7 * rev_rank + 0.3 * tov_rank
        ret_r, tov_r, _ = run_backtest(df_ret, sig_rank, topk=100, K=K,
                                        weight_method='inv_var', hold_buffer=80)
        ret_r, _ = apply_market_filter(ret_r, market_ret)
        ret_r, _ = apply_vol_target(ret_r, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_r)
        results['换手率排名增强'] = {'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol,
                                      'turnover': tov_r.mean() * 12}
        print(f"  排名增强: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tov_r.mean()*12:.0f}%")

        # 条件筛选法: 换手率突增且下跌的股票
        tov_spike = tov_change > tov_change.quantile(0.8, axis=1)
        rev_sig = reversal_signal.copy()
        rev_sig[~tov_spike] = np.nan
        ret_c, tov_c, _ = run_backtest(df_ret, rev_sig.fillna(reversal_signal.min(axis=1), axis=0),
                                        topk=100, K=K, weight_method='inv_var', hold_buffer=80)
        ret_c, _ = apply_market_filter(ret_c, market_ret)
        ret_c, _ = apply_vol_target(ret_c, target_vol=0.12)
        ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_c)
        results['换手率条件筛选'] = {'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol,
                                      'turnover': tov_c.mean() * 12}
        print(f"  条件筛选: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tov_c.mean()*12:.0f}%")
    else:
        print("  换手率数据缺失，跳过")

    # ====================================================================
    # 2. 可交易性过滤实验
    # ====================================================================
    print("\n--- 2. 可交易性过滤实验 ---")
    # 简单模拟：剔除上市不足6个月的次新股（前6个月信号为NaN的股票）
    # 以及随机模拟停牌（5%概率停牌）
    np.random.seed(42)
    tradeable_filter = pd.DataFrame(True, index=df_ret.index, columns=df_ret.columns)
    for col in df_ret.columns:
        first_valid = df_ret[col].first_valid_index()
        if first_valid is not None:
            cutoff = first_valid + pd.DateOffset(months=6)
            tradeable_filter.loc[df_ret.index < cutoff, col] = False
    # 模拟停牌: 每月5%概率停牌
    for date in df_ret.index:
        mask = np.random.rand(len(df_ret.columns)) < 0.05
        tradeable_filter.loc[date, mask] = False

    ret_tf, tov_tf, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                      weight_method='inv_var', hold_buffer=80,
                                      stock_filter=tradeable_filter)
    ret_tf, _ = apply_market_filter(ret_tf, market_ret)
    ret_tf, _ = apply_vol_target(ret_tf, target_vol=0.12)
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_tf)
    results['可交易性过滤'] = {'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol,
                               'turnover': tov_tf.mean() * 12}
    print(f"  可交易性过滤: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tov_tf.mean()*12:.0f}%")

    # ====================================================================
    # 3. 最终集成策略子样本
    # ====================================================================
    print("\n--- 3. 最终集成策略子样本 ---")
    ret_base, tov_base, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                          weight_method='inv_var', hold_buffer=80,
                                          signal_threshold=0.85,
                                          stock_filter=tradeable_filter)
    ret_base, _ = apply_market_filter(ret_base, market_ret, ma_window=6)
    ret_base, _ = apply_time_stop(ret_base, market_ret, consec_months=3, scalar_gradual=True)
    ret_final, _ = apply_vol_target(ret_base, target_vol=0.12)

    for period_name, start, end in [('2005-2012', '2005-01', '2012-12'),
                                     ('2013-2018', '2013-01', '2018-12'),
                                     ('2019-2025', '2019-01', '2025-12')]:
        mask = (ret_final.index >= start) & (ret_final.index <= end)
        sub_ret = ret_final[mask]
        ann, sharpe, dd, vol, _, _ = calculate_metrics(sub_ret)
        results[f'最终集成_子样本_{period_name}'] = {'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol}
        print(f"  {period_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # 全样本
    ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_final)
    results['最终集成_全样本'] = {'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol,
                                 'turnover': tov_base.mean() * 12}
    print(f"  全样本: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 换手={tov_base.mean()*12:.0f}%")

    # ====================================================================
    # 4. 基准策略(MA12+buffer80+inv_var+VT)子样本 (用于对比)
    # ====================================================================
    print("\n--- 4. 基准策略子样本 ---")
    ret_b, tov_b, _ = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                    weight_method='inv_var', hold_buffer=80)
    ret_b, _ = apply_market_filter(ret_b, market_ret, ma_window=12)
    ret_b, _ = apply_vol_target(ret_b, target_vol=0.12)

    for period_name, start, end in [('2005-2012', '2005-01', '2012-12'),
                                     ('2013-2018', '2013-01', '2018-12'),
                                     ('2019-2025', '2019-01', '2025-12')]:
        mask = (ret_b.index >= start) & (ret_b.index <= end)
        sub_ret = ret_b[mask]
        ann, sharpe, dd, vol, _, _ = calculate_metrics(sub_ret)
        results[f'基准_子样本_{period_name}'] = {'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol}
        print(f"  {period_name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # ====================================================================
    # 保存结果
    # ====================================================================
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'fill_missing_data_part2.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    results = convert(results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")
    print("\n" + "=" * 80)
    print("补充测算完成")


if __name__ == '__main__':
    main()
