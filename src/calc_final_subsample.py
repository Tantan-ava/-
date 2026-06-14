import pandas as pd, numpy as np, sys
sys.path.insert(0, '/Users/xinyutan/Documents/量化投资/quant-project/src')
from fill_missing_data import load_data, load_value_factor, apply_standardization, run_backtest, apply_market_filter, apply_vol_target, calculate_metrics

df_ret = load_data()
market_ret = df_ret.mean(axis=1)
K = 6
reversal_signal = -df_ret.rolling(K, min_periods=1).sum().shift(1)
value_signal = load_value_factor(df_ret)
combined_signal = 0.6 * apply_standardization(reversal_signal) + 0.4 * apply_standardization(value_signal)

ret_base, tov_base, _ = run_backtest(df_ret, combined_signal, topk=100, K=K, weight_method='inv_var', hold_buffer=80)
ret_ma6, _ = apply_market_filter(ret_base, market_ret, ma_window=6)

# 信号阈值85%
threshold = combined_signal.quantile(0.85, axis=1)
signal_thresh = combined_signal.copy()
for t in signal_thresh.index:
    signal_thresh.loc[t, signal_thresh.loc[t] < threshold.loc[t]] = np.nan

ret_thresh, tov_thresh, _ = run_backtest(df_ret, signal_thresh, topk=100, K=K, weight_method='inv_var', hold_buffer=80)

# 时间止损
market_ret_aligned = market_ret.reindex(ret_thresh.index).fillna(0)
underperform = ret_thresh < market_ret_aligned
consec_under = underperform.rolling(3).sum().shift(1)
time_stop_pos = pd.Series(1.0, index=ret_thresh.index)
for date in ret_thresh.index:
    if pd.isna(consec_under.loc[date]):
        continue
    if consec_under.loc[date] >= 4:
        time_stop_pos.loc[date] = 0.3
    elif consec_under.loc[date] >= 3:
        time_stop_pos.loc[date] = 0.5
ret_ts = ret_thresh * time_stop_pos

# VT
ret_final, _ = apply_vol_target(ret_ts, target_vol=0.12)

ann, sharpe, dd, vol, _, _ = calculate_metrics(ret_final)
print('最终集成: 年化={:.2%}, 夏普={:.2f}, 回撤={:.2%}, 波动={:.2%}'.format(ann, sharpe, dd, vol))

periods = [
    ('2005-01', '2012-12', '2005-2012'),
    ('2013-01', '2018-12', '2013-2018'),
    ('2019-01', '2025-12', '2019-2025'),
]
for start, end, label in periods:
    mask = (ret_final.index >= start) & (ret_final.index <= end)
    sub = ret_final[mask]
    if len(sub) < 6:
        continue
    ann_s = (1 + sub).prod() ** (12 / len(sub)) - 1
    vol_s = sub.std() * np.sqrt(12)
    sharpe_s = ann_s / vol_s if vol_s > 0 else 0
    cum = (1 + sub).cumprod()
    dd_s = (cum / cum.cummax() - 1).min()
    print('{}: 年化={:.2%}, 夏普={:.2f}, 回撤={:.2%}'.format(label, ann_s, sharpe_s, dd_s))
