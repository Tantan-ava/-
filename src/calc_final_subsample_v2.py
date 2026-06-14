import pandas as pd, numpy as np, sys
sys.path.insert(0, '/Users/xinyutan/Documents/量化投资/quant-project/src')
from fill_missing_data import load_data, load_value_factor, apply_standardization, apply_market_filter, apply_vol_target, calculate_metrics
from run_week10b_integration import run_backtest, build_tradeable_filter, apply_time_stop

df_ret = load_data()
market_ret = df_ret.mean(axis=1)
K = 6

reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
value_signal = load_value_factor(df_ret)
combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

tradeable_filter = build_tradeable_filter(df_ret, min_history=10)

# 全部集成
ret7, tov7, _, avg7 = run_backtest(df_ret, combined_signal, topk=100, K=K,
                                     weight_method='inv_var', hold_buffer=80,
                                     signal_threshold=0.85,
                                     stock_filter=tradeable_filter)
ret7_ma, _ = apply_market_filter(ret7, market_ret, ma_window=6)
ret7_ts, _ = apply_time_stop(ret7_ma, market_ret, consec_months=3, scalar_gradual=True)
ret7_final, _ = apply_vol_target(ret7_ts, target_vol=0.12)

ann, sharpe, dd, vol, _, _ = calculate_metrics(ret7_final)
print('最终集成: 年化={:.2%}, 夏普={:.2f}, 回撤={:.2%}, 波动={:.2%}, 换手={:.0%}'.format(ann, sharpe, dd, vol, tov7.mean()*12))

periods = [
    ('2005', '2012', '2005-2012'),
    ('2013', '2018', '2013-2018'),
    ('2019', '2025', '2019-2025'),
]
for start, end, label in periods:
    mask = (ret7_final.index >= start) & (ret7_final.index <= end)
    sub = ret7_final[mask]
    if len(sub) < 6:
        continue
    ann_s, sharpe_s, dd_s, vol_s, _, _ = calculate_metrics(sub)
    print('{}: 年化={:.2%}, 夏普={:.2f}, 回撤={:.2%}, 波动={:.2%}'.format(label, ann_s, sharpe_s, dd_s, vol_s))
