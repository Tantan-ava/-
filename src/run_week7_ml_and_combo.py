# -*- coding: utf-8 -*-
"""
Week 7 补充: ML实验(简化版) + MA过滤器组合验证
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


def run_backtest_equal_weight(df_ret, signal_df, topk=100, K=6):
    weights = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    turnover_list = []
    for date in df_ret.index[K:]:
        sig = signal_df.loc[date]
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
    if len(turnover_list) < len(port_ret):
        turnover_list = [1.0] * (len(port_ret) - len(turnover_list)) + turnover_list
    elif len(turnover_list) > len(port_ret):
        turnover_list = turnover_list[-len(port_ret):]
    turnover_series = pd.Series(turnover_list, index=port_ret.index)
    return port_ret, turnover_series, weights


def main():
    print("=" * 80)
    print("Week 7 补充: ML实验(简化) + MA组合验证")
    print("=" * 80)

    # 加载数据
    df_ret = pd.read_excel('data/raw/TRD_Mnth.xlsx', index_col=0, engine='openpyxl')
    df_ret.index = pd.to_datetime(df_ret.index)
    df_ret.columns = df_ret.columns.astype(str)
    print(f"  数据: {df_ret.shape}, {df_ret.index[0]} ~ {df_ret.index[-1]}")

    K = 6
    reversal_signal = -df_ret.rolling(window=K).sum().shift(1)
    reversal_signal = apply_standardization(reversal_signal)
    value_signal = load_value_factor(df_ret)
    value_signal = apply_standardization(value_signal)
    combo_signal = 0.4 * value_signal + 0.6 * reversal_signal
    market_ret = df_ret.mean(axis=1)

    # ========================================================================
    # 实验6 (简化): ML因子组合
    # ========================================================================
    print("\n" + "=" * 70)
    print("实验6 (简化): ML因子组合")
    print("=" * 70)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    # 构造特征
    features = {}
    for k in [1, 2, 3, 6, 9, 12]:
        feat = -df_ret.rolling(window=k).sum().shift(1)
        features[f'rev_K{k}'] = apply_standardization(feat)
    features['value_12m'] = apply_standardization(load_value_factor(df_ret))
    features['vol_6m'] = df_ret.rolling(window=6).std().shift(1)
    features['vol_12m'] = df_ret.rolling(window=12).std().shift(1)
    mom_6 = df_ret.rolling(window=6).sum().shift(1)
    mom_12 = df_ret.rolling(window=12).sum().shift(1)
    features['mom_accel'] = apply_standardization(mom_6 - mom_12)
    features['skew_6m'] = df_ret.rolling(window=6).skew().shift(1)

    # Rank标准化
    for feat_name in features:
        features[feat_name] = features[feat_name].rank(pct=True, axis=1) * 2 - 1

    target = df_ret.shift(-1)
    feature_names = list(features.keys())
    train_window = 60
    predict_start = K + train_window

    ml_results = {}

    for model_name, base_model in [('Ridge', Ridge(alpha=1.0)),
                                    ('GBDT_shallow', GradientBoostingRegressor(
                                        n_estimators=50, max_depth=2,
                                        learning_rate=0.1, min_samples_leaf=100,
                                        subsample=0.8, random_state=42))]:
        print(f"\n  训练 {model_name}...")
        ml_signal = pd.DataFrame(np.nan, index=df_ret.index, columns=df_ret.columns, dtype=float)
        dates = df_ret.index
        last_train_idx = -1
        model = None

        for i in range(predict_start, len(dates)):
            if i - last_train_idx < 12 and last_train_idx > 0 and model is not None:
                pass  # 不重新训练
            else:
                train_start_idx = max(K, i - train_window)
                X_train_list, y_train_list = [], []
                for t_idx in range(train_start_idx, i):
                    date_t = dates[t_idx]
                    feat_vals = []
                    for fn in feature_names:
                        if date_t in features[fn].index:
                            feat_vals.append(features[fn].loc[date_t].values)
                        else:
                            feat_vals.append(np.full(len(df_ret.columns), np.nan))
                    X_t = np.column_stack(feat_vals)
                    y_t = target.loc[date_t].values if date_t in target.index else np.full(len(df_ret.columns), np.nan)
                    valid_mask = ~np.isnan(y_t) & ~np.isnan(X_t).any(axis=1)
                    if valid_mask.sum() > 100:
                        X_train_list.append(X_t[valid_mask])
                        y_train_list.append(y_t[valid_mask])
                if len(X_train_list) == 0:
                    continue
                X_train = np.vstack(X_train_list)
                y_train = np.concatenate(y_train_list)
                if len(X_train) < 500:
                    continue
                model = type(base_model)(**base_model.get_params())
                model.fit(X_train, y_train)
                last_train_idx = i

            date_pred = dates[i]
            feat_vals = []
            for fn in feature_names:
                if date_pred in features[fn].index:
                    feat_vals.append(features[fn].loc[date_pred].values)
                else:
                    feat_vals.append(np.full(len(df_ret.columns), np.nan))
            X_pred = np.column_stack(feat_vals)
            valid_mask = ~np.isnan(X_pred).any(axis=1)
            if valid_mask.sum() > 0 and model is not None:
                preds = np.full(len(df_ret.columns), np.nan)
                preds[valid_mask] = model.predict(X_pred[valid_mask])
                ml_signal.loc[date_pred] = preds

        ml_signal = ml_signal.fillna(0).shift(1)

        # ML信号选股 (ml_signal预测下月收益，shift(1)对齐持仓周期避免前视偏差)
        port_ret_ml, tover_ml, _ = run_backtest_equal_weight(df_ret, ml_signal, topk=100, K=predict_start)
        ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_ml)
        ml_results[f'{model_name}选股'] = {
            'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
            'turnover': tover_ml.mean() * 12, 'returns': port_ret_ml}
        print(f"  {model_name}选股: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

        # ML + 原始信号组合
        for ml_weight in [0.3, 0.5]:
            combined = (1 - ml_weight) * combo_signal + ml_weight * ml_signal
            port_ret_comb, tover_comb, _ = run_backtest_equal_weight(df_ret, combined, topk=100, K=predict_start)
            ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_comb)
            ml_results[f'{model_name}+原始({1-ml_weight:.0%}/{ml_weight:.0%})'] = {
                'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
                'turnover': tover_comb.mean() * 12, 'returns': port_ret_comb}
            print(f"  {model_name}+原始({1-ml_weight:.0%}/{ml_weight:.0%}): 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # ========================================================================
    # 关键: 所有有效优化 + MA过滤器组合
    # ========================================================================
    print("\n" + "=" * 70)
    print("关键验证: 有效优化 + MA过滤器组合")
    print("=" * 70)

    combo_results = {}

    # 基准
    port_ret, tover, _ = run_backtest_equal_weight(df_ret, combo_signal, topk=100, K=K)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret)
    combo_results['A:基准(40V+60R)'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                         'ann_vol': vol, 'turnover': tover.mean() * 12}
    print(f"  A:基准: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # B: +MA
    port_ret_ma, _ = apply_market_filter(port_ret, market_ret)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_ma)
    combo_results['B:+MA'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                               'ann_vol': vol, 'turnover': tover.mean() * 12}
    print(f"  B:+MA: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # C: VolTarget + MA
    rolling_vol = port_ret.rolling(24).std().shift(1) * np.sqrt(12)
    for vol_target in [0.10, 0.12, 0.15]:
        vol_scalar = (vol_target / rolling_vol).clip(0.3, 2.0).fillna(1.0)
        adj_ret = port_ret * vol_scalar
        adj_ret_ma, _ = apply_market_filter(adj_ret, market_ret)
        ann, sharpe, dd, vol_adj, cum, _ = calculate_metrics(adj_ret_ma)
        name = f'C:VolTarget={vol_target:.0%}+MA'
        combo_results[name] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                'ann_vol': vol_adj, 'turnover': tover.mean() * 12}
        print(f"  {name}: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}, 波动={vol_adj:.2%}")

    # D: 最小方差(1/σ²) + MA
    vol_lookback = 12
    weights_mv = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    for date in df_ret.index[K:]:
        sig = combo_signal.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        selected = (ranks <= 100).astype(float)
        selected = selected.where(sig.notna(), other=0.0)
        if selected.sum() > 0:
            date_idx = df_ret.index.get_loc(date)
            start_idx = max(0, date_idx - vol_lookback)
            hist_ret = df_ret.iloc[start_idx:date_idx]
            stock_vols = hist_ret.std()
            inv_var = 1.0 / (stock_vols ** 2).replace(0, np.nan)
            inv_var = inv_var.fillna(0)
            w = selected * inv_var
            w_sum = w.sum()
            if w_sum > 0:
                weights_mv.loc[date] = w / w_sum
    port_ret_mv = (weights_mv * df_ret).sum(axis=1).iloc[K:]
    port_ret_mv_ma, _ = apply_market_filter(port_ret_mv, market_ret)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_mv_ma)
    combo_results['D:最小方差+MA'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                      'ann_vol': vol, 'turnover': tover.mean() * 12}
    print(f"  D:最小方差+MA: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # E: 持仓延续(buffer=50) + MA
    weights_hold = pd.DataFrame(0.0, index=df_ret.index, columns=df_ret.columns, dtype=float)
    prev_selected = None
    buffer = 50
    for date in df_ret.index[K:]:
        sig = combo_signal.loc[date]
        ranks = sig.rank(ascending=False, method='first')
        if prev_selected is not None:
            keep_mask = (ranks <= 100 + buffer) & prev_selected
            new_mask = (ranks <= 100) & ~keep_mask
            selected = keep_mask | new_mask
        else:
            selected = (ranks <= 100)
        selected = selected.astype(float)
        selected = selected.where(sig.notna(), other=0.0)
        count = selected.sum()
        if count > 0:
            weights_hold.loc[date] = selected / count
        prev_selected = selected > 0
    port_ret_hold = (weights_hold * df_ret).sum(axis=1).iloc[K:]
    port_ret_hold_ma, _ = apply_market_filter(port_ret_hold, market_ret)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(port_ret_hold_ma)
    tover_hold = []
    for i in range(1, len(weights_hold)):
        tover_hold.append((weights_hold.iloc[i] - weights_hold.iloc[i-1]).abs().sum() / 2)
    combo_results['E:持仓延续(b=50)+MA'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                             'ann_vol': vol, 'turnover': np.mean(tover_hold) * 12}
    print(f"  E:持仓延续(b=50)+MA: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # F: VolTarget(10%) + 最小方差 + MA
    rolling_vol_mv = port_ret_mv.rolling(24).std().shift(1) * np.sqrt(12)
    vol_scalar_mv = (0.10 / rolling_vol_mv).clip(0.3, 2.0).fillna(1.0)
    adj_ret_mv = port_ret_mv * vol_scalar_mv
    adj_ret_mv_ma, _ = apply_market_filter(adj_ret_mv, market_ret)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(adj_ret_mv_ma)
    combo_results['F:VolTarget(10%)+最小方差+MA'] = {'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
                                                      'ann_vol': vol, 'turnover': tover.mean() * 12}
    print(f"  F:VolTarget(10%)+最小方差+MA: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # G: VolTarget(12%) + 持仓延续 + MA
    rolling_vol_hold = port_ret_hold.rolling(24).std().shift(1) * np.sqrt(12)
    vol_scalar_hold = (0.12 / rolling_vol_hold).clip(0.3, 2.0).fillna(1.0)
    adj_ret_hold = port_ret_hold * vol_scalar_hold
    adj_ret_hold_ma, _ = apply_market_filter(adj_ret_hold, market_ret)
    ann, sharpe, dd, vol, cum, _ = calculate_metrics(adj_ret_hold_ma)
    combo_results['G:VolTarget(12%)+持仓延续+MA'] = {
        'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd, 'ann_vol': vol,
        'turnover': np.mean(tover_hold) * 12}
    print(f"  G:VolTarget(12%)+持仓延续+MA: 年化={ann:.2%}, 夏普={sharpe:.2f}, 回撤={dd:.2%}")

    # ========================================================================
    # 最终排名
    # ========================================================================
    print("\n" + "=" * 80)
    print("最终排名: 所有优化+MA组合")
    print("=" * 80)

    sorted_combo = sorted(combo_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    print(f"\n{'排名':>3} {'配置':<35} {'年化收益':>8} {'夏普':>6} {'最大回撤':>8} {'波动率':>8} {'换手率':>6}")
    print("-" * 85)
    for i, (name, r) in enumerate(sorted_combo, 1):
        print(f"{i:>3} {name:<35} {r['ann_ret']:>7.2%} {r['sharpe']:>6.2f} "
              f"{r['max_dd']:>7.2%} {r['ann_vol']:>7.2%} {r['turnover']:>5.0%}")

    best_name, best_r = sorted_combo[0]
    print(f"\n{'='*80}")
    print(f"最优配置: {best_name}")
    print(f"  年化收益: {best_r['ann_ret']:.2%}")
    print(f"  夏普比率: {best_r['sharpe']:.2f}")
    print(f"  最大回撤: {best_r['max_dd']:.2%}")
    print(f"  年化波动率: {best_r['ann_vol']:.2%}")
    print(f"  年化换手率: {best_r['turnover']:.0%}")
    print(f"{'='*80}")

    # ML结果
    print("\n" + "=" * 80)
    print("ML实验结果")
    print("=" * 80)
    for name, r in ml_results.items():
        print(f"  {name}: 年化={r['ann_ret']:.2%}, 夏普={r['sharpe']:.2f}, 回撤={r['max_dd']:.2%}")

    # 可视化
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    names = [n for n, _ in sorted_combo]
    sharpes = [r['sharpe'] for _, r in sorted_combo]
    turnovers = [r['turnover'] for _, r in sorted_combo]
    max_dds = [abs(r['max_dd']) for _, r in sorted_combo]

    x = np.arange(len(names))
    width = 0.3
    ax.bar(x - width, sharpes, width, label='夏普比率', color='steelblue')
    ax2 = ax.twinx()
    ax2.bar(x, max_dds, width, label='最大回撤(绝对值)', color='salmon', alpha=0.7)
    ax2.bar(x + width, [t/10 for t in turnovers], width, label='换手率/10', color='lightgreen', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([n.split(':')[1] if ':' in n else n for n in names], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('夏普比率')
    ax2.set_ylabel('回撤/换手率')
    ax.set_title('Week 7: 优化+MA组合对比', fontsize=13)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'week7_combo_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'week7_combo_comparison.png'}")
    plt.close()

    return combo_results, ml_results


if __name__ == '__main__':
    combo_results, ml_results = main()
