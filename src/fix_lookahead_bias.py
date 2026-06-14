# -*- coding: utf-8 -*-
"""批量修复所有实验文件中的前视偏差"""
import re, os

files = [
    'src/run_value_factor_test.py',
    'src/run_final_optimal.py',
    'src/run_turnover_epu_experiment.py',
    'src/run_epu_weight_experiment.py',
    'src/run_week7_experiments.py',
    'src/run_week7_ml_and_combo.py',
    'src/run_week7_final.py',
    'src/run_turnover_optimization.py',
    'src/run_week9_deepening.py',
    'src/run_week8_tutorial.py',
    'src/run_week11_attribution.py',
    'src/run_optimization_experiments.py',
    'src/run_experiments.py',
]

for f in files:
    path = os.path.join('/Users/xinyutan/Documents/量化投资/quant-project', f)
    if not os.path.exists(path):
        print(f'MISSING: {f}')
        continue
    with open(path, 'r') as fh:
        content = fh.read()
    orig = content

    # Pattern 1: standardization with full-sample quantile
    if 'signal_df.quantile(0.01)' in content:
        # Multi-line version
        content = re.sub(
            r'return signal_df\.clip\(lower=signal_df\.quantile\(0\.01\),\s*\n?\s*upper=signal_df\.quantile\(0\.99\),\s*axis=1\)',
            '''lower = signal_df.rolling(24, min_periods=6).quantile(0.01)
    upper = signal_df.rolling(24, min_periods=6).quantile(0.99)
    return signal_df.clip(lower=lower, upper=upper, axis=0)''',
            content
        )
        # Single line version
        content = re.sub(
            r'return signal_df\.clip\(lower=signal_df\.quantile\(0\.01\),\s*upper=signal_df\.quantile\(0\.99\),\s*axis=1\)',
            '''lower = signal_df.rolling(24, min_periods=6).quantile(0.01)
    upper = signal_df.rolling(24, min_periods=6).quantile(0.99)
    return signal_df.clip(lower=lower, upper=upper, axis=0)''',
            content
        )

    # Pattern 2: MA without shift
    if 'cum_market.rolling(ma_window).mean()' in content:
        content = content.replace(
            'ma = cum_market.rolling(ma_window).mean()',
            'ma = cum_market.rolling(ma_window).mean().shift(1)'
        )

    # Pattern 3: vol target without shift
    if 'returns.rolling(lookback).std() * np.sqrt(12)' in content:
        content = content.replace(
            'rolling_vol = returns.rolling(lookback).std() * np.sqrt(12)',
            'rolling_vol = returns.rolling(lookback).std().shift(1) * np.sqrt(12)'
        )

    # Pattern 4: time stop without shift
    if 'underperform.rolling(consec_months).sum()' in content:
        content = content.replace(
            'consec_under = underperform.rolling(consec_months).sum()',
            'consec_under = underperform.rolling(consec_months).sum().shift(1)'
        )

    # Pattern 5: volatility state without shift
    if 'market_ret.rolling(12).std() * np.sqrt(12)' in content:
        content = content.replace(
            'mkt_vol = market_ret.rolling(12).std() * np.sqrt(12)',
            'mkt_vol = market_ret.rolling(12).std().shift(1) * np.sqrt(12)'
        )

    # Pattern 6: dual MA without shift
    if 'cum_market.rolling(6).mean()' in content and 'shift(1)' not in content:
        content = content.replace(
            'ma_short = cum_market.rolling(6).mean()',
            'ma_short = cum_market.rolling(6).mean().shift(1)'
        )
    if 'cum_market.rolling(24).mean()' in content and 'shift(1)' not in content:
        content = content.replace(
            'ma_long = cum_market.rolling(24).mean()',
            'ma_long = cum_market.rolling(24).mean().shift(1)'
        )

    if content != orig:
        with open(path, 'w') as fh:
            fh.write(content)
        print(f'FIXED: {f}')
    else:
        print(f'NO CHANGE: {f}')
