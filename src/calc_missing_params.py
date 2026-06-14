"""
计算参数稳健性测试中缺失的参数组合精确数据
"""
import numpy as np
import pandas as pd
from pathlib import Path

def load_data():
    df_ret = pd.read_excel('/Users/xinyutan/Documents/量化投资/25304004_谈心语_ReversalStrategy/图表数据/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)

    vf = pd.read_parquet('data/processed/value_factor.parquet')
    vf.index = pd.to_datetime(vf.index)
    common_cols = df_ret.columns.intersection(vf.columns)
    vf = vf.reindex(index=df_ret.index, columns=common_cols)

    return df_ret, vf

def apply_standardization(signal_df):
    lower = signal_df.rolling(24, min_periods=6).quantile(0.01)
    upper = signal_df.rolling(24, min_periods=6).quantile(0.99)
    return signal_df.clip(lower=lower, upper=upper, axis=0)

def calc_signals(df_ret, fi_pivot, K):
    reversal_signal = df_ret.rolling(K).sum().shift(1)
    fi_lagged = fi_pivot.shift(1)
    fi_rank = fi_lagged.rank(pct=True, axis=1)
    value_signal = fi_rank
    combined_signal = 0.4 * apply_standardization(value_signal.fillna(0)) + 0.6 * apply_standardization(reversal_signal.fillna(0))
    return combined_signal

def run_backtest(df_ret, signal, topk=100, K=6, weight_method='equal', hold_buffer=0):
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    prev_selected = None
    for date in df_ret.index[K:]:
        sig = signal.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= topk).astype(float)
        selected = selected.where(sig.notna(), other=0.0)
        if hold_buffer > 0 and prev_selected is not None:
            prev_ranks = prev_selected.rank(ascending=False, method='first')
            keep = (prev_ranks <= hold_buffer).astype(float)
            keep = keep.where(prev_selected.notna(), other=0.0)
            new_selected = selected.copy()
            new_selected[keep > 0] = 1.0
            selected = new_selected
        count = selected.sum()
        if count > 0:
            weights.loc[date] = selected / count
        if prev_selected is not None:
            turnover_list.append((weights.loc[date] - weights.loc[df_ret.index[df_ret.index.get_loc(date) - 1]]).abs().sum() / 2)
        else:
            turnover_list.append(1.0)
        prev_selected = selected.copy()
    port_ret = (weights * df_ret).sum(axis=1).iloc[K:]
    turnover = pd.Series(turnover_list, index=port_ret.index)
    return port_ret, turnover, weights

def apply_market_filter(returns, market_returns, ma_window=12):
    ma = market_returns.rolling(ma_window).mean().shift(1)
    trend = market_returns.rolling(ma_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True).shift(1)
    position_scalar = pd.Series(1.0, index=returns.index)
    for date in returns.index:
        mkt = market_returns.get(date, np.nan)
        ma_val = ma.get(date, np.nan)
        trd = trend.get(date, np.nan)
        if pd.isna(mkt) or pd.isna(ma_val) or pd.isna(trd):
            continue
        if mkt < ma_val:
            if trd < 0:
                position_scalar.loc[date] = 0.3
            else:
                position_scalar.loc[date] = 0.5
    return returns * position_scalar, position_scalar

def apply_vol_target(returns, target_vol=0.12):
    vol = returns.rolling(24).std() * np.sqrt(12)
    vol = vol.shift(1)
    scalar = target_vol / vol
    scalar = scalar.clip(lower=0.5, upper=2.0)
    return returns * scalar, scalar

def calculate_metrics(returns):
    ann = (1 + returns.mean()) ** 12 - 1
    vol = returns.std() * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else 0
    cum = (1 + returns).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return ann, sharpe, dd, vol

def main():
    df_ret, fi_pivot = load_data()
    market_ret = df_ret.mean(axis=1)

    configs = [
        ('K=3', 3, 100, 12),
        ('K=9', 9, 100, 12),
        ('MA=24', 6, 100, 24),
    ]

    results = []
    for label, K, topk, ma_window in configs:
        combined_signal = calc_signals(df_ret, fi_pivot, K)
        ret, tov, _ = run_backtest(df_ret, combined_signal, topk=topk, K=K,
                                    weight_method='inv_var', hold_buffer=80)
        ret_ma, _ = apply_market_filter(ret, market_ret, ma_window=ma_window)
        ret_vt, _ = apply_vol_target(ret_ma, target_vol=0.12)
        ann, sharpe, dd, vol = calculate_metrics(ret_vt)
        results.append({
            'label': label,
            'K': K,
            'topk': topk,
            'ma': ma_window,
            'ann': f"{ann:.2%}",
            'sharpe': f"{sharpe:.2f}",
            'dd': f"{dd:.2%}",
            'vol': f"{vol:.2%}",
            'turnover': f"{tov.mean()*12:.0f}%",
        })
        print(f"{label}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol:.2%}, 换手={tov.mean()*12:.0f}%")

    # Save results
    import json
    with open('results/missing_params.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n结果已保存至 results/missing_params.json")

if __name__ == '__main__':
    main()
