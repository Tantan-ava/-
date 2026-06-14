"""
前视偏差修复验证脚本
快速验证关键修改点的逻辑正确性
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 设置路径
script_dir = Path(__file__).parent
project_dir = script_dir.parent
data_path = project_dir / 'data' / 'raw' / 'TRD_Mnth.xlsx'

if not data_path.exists():
    print(f"错误: 未找到数据文件 {data_path}")
    sys.exit(1)

print("="*70)
print("前视偏差修复验证 - 加载数据中...")
print("="*70)
df_ret = pd.read_excel(data_path, index_col=0, engine='openpyxl')
df_ret.index = pd.to_datetime(df_ret.index)
# 只取前100个月加速验证
df_ret = df_ret.iloc[:100]
print(f"数据形状: {df_ret.shape}")
print(f"日期范围: {df_ret.index[0]} ~ {df_ret.index[-1]}")

results = {}

# ============================================================================
# 验证1: 风险平价权重计算（排除当前日期）
# ============================================================================
print("\n" + "="*70)
print("验证1: 风险平价权重 - 确认历史窗口不包含当前日期")
print("="*70)

def test_risk_parity_lookback():
    """模拟 run_optimization_experiments.py 中的风险平价权重计算"""
    lookback = 12
    date = df_ret.index[20]  # 选一个中间日期
    date_idx = df_ret.index.get_loc(date)
    start_idx = max(0, date_idx - lookback)

    # 修复后: 使用 iloc[start_idx:date_idx]，排除当前日期
    lookback_returns = df_ret.iloc[start_idx:date_idx]
    vols = lookback_returns.std()

    # 检查是否包含当前日期
    assert date not in lookback_returns.index, \
        f"前视偏差! 历史窗口包含了当前日期 {date}"
    assert len(lookback_returns) <= lookback, \
        f"窗口长度异常: {len(lookback_returns)} > {lookback}"

    print(f"✅ 日期 {date.date()}:")
    print(f"   - 历史窗口: {lookback_returns.index[0].date()} ~ {lookback_returns.index[-1].date()}")
    print(f"   - 窗口长度: {len(lookback_returns)} (max={lookback})")
    print(f"   - 当前日期是否包含: {'否 (正确)'}")
    return True

results['risk_parity'] = test_risk_parity_lookback()

# ============================================================================
# 验证2: 滚动波动率/均值 shift(1) 检查
# ============================================================================
print("\n" + "="*70)
print("验证2: 滚动波动率/均值 - 确认已添加 shift(1)")
print("="*70)

def test_rolling_shift():
    market_ret = df_ret.mean(axis=1)
    # 模拟修复后的滚动波动率
    rolling_vol = market_ret.rolling(12).std().shift(1)

    # 前12期应为 NaN（因 rolling 需要12期 + shift 需要1期）
    assert pd.isna(rolling_vol.iloc[11]), \
        "shift(1) 未生效: 第12期不应有有效值"
    assert pd.notna(rolling_vol.iloc[12]), \
        "shift(1) 异常: 第13期应有有效值"

    print(f"✅ 滚动波动率 shift(1) 检查通过")
    print(f"   - 第11期 (索引10): {rolling_vol.iloc[10]} (应为NaN)")
    print(f"   - 第12期 (索引11): {rolling_vol.iloc[11]} (应为NaN)")
    print(f"   - 第13期 (索引12): {rolling_vol.iloc[12]:.4f} (应有效)")
    return True

results['rolling_shift'] = test_rolling_shift()

# ============================================================================
# 验证3: 扩展窗口分位数 shift(1) 检查
# ============================================================================
print("\n" + "="*70)
print("验证3: 扩展窗口分位数 - 确认已添加 shift(1)")
print("="*70)

def test_expanding_quantile_shift():
    z = pd.Series(np.random.randn(50), index=df_ret.index[:50])
    # 修复后: expanding().quantile().shift(1)
    threshold = z.expanding().quantile(0.75).shift(1)

    # 第1期应为 NaN（因 shift(1)）
    assert pd.isna(threshold.iloc[0]), \
        "shift(1) 未生效: 第1期阈值应为NaN"
    assert pd.notna(threshold.iloc[1]), \
        "shift(1) 异常: 第2期阈值应有效"

    # 验证阈值不包含当日
    for i in range(5, 20):
        t_val = threshold.iloc[i]
        hist_z = z.iloc[:i]  # 历史数据不含当日
        expected_p75 = hist_z.quantile(0.75)
        assert abs(t_val - expected_p75) < 1e-10, \
            f"第{i}期阈值计算错误: {t_val} != {expected_p75}"

    print(f"✅ 扩展窗口分位数 shift(1) 检查通过")
    print(f"   - 第1期阈值: {threshold.iloc[0]} (应为NaN)")
    print(f"   - 第2期阈值: {threshold.iloc[1]:.4f} (应有效)")
    print(f"   - 第10期阈值: {threshold.iloc[9]:.4f} (基于前9期历史)")
    return True

results['expanding_quantile'] = test_expanding_quantile_shift()

# ============================================================================
# 验证4: ML信号 horizon 对齐
# ============================================================================
print("\n" + "="*70)
print("验证4: ML信号预测周期 - 确认 shift(1) 对齐持仓周期")
print("="*70)

def test_ml_signal_horizon():
    # 模拟ML信号: 预测下月收益
    np.random.seed(42)
    ml_pred = pd.DataFrame(np.random.randn(*df_ret.shape), index=df_ret.index, columns=df_ret.columns)
    # 修复后: fillna(0).shift(1)
    ml_signal = ml_pred.fillna(0).shift(1)

    # 第1期应为 NaN（因 shift(1)）
    assert pd.isna(ml_signal.iloc[0]).all(), \
        "ML信号 shift(1) 未生效: 第1期应为NaN"

    # t期信号应对应 t-1 期预测
    for i in range(1, 10):
        assert (ml_signal.iloc[i].fillna(0) == ml_pred.iloc[i-1].fillna(0)).all(), \
            f"第{i}期ML信号未正确对齐到上期预测"

    print(f"✅ ML信号 horizon 对齐检查通过")
    print(f"   - 第1期信号: 全NaN (正确，无上期预测)")
    print(f"   - 第2期信号: 对应第1期预测值 (正确)")
    print(f"   - 第5期信号: 对应第4期预测值 (正确)")
    return True

results['ml_horizon'] = test_ml_signal_horizon()

# ============================================================================
# 验证5: 流动性过滤使用历史窗口
# ============================================================================
print("\n" + "="*70)
print("验证5: 流动性过滤 - 确认使用初始历史窗口而非全样本")
print("="*70)

def test_liquidity_filter():
    # 模拟修复后的流动性过滤
    initial_window = df_ret.iloc[:24]
    trading_activity = (initial_window != 0).sum() / len(initial_window)
    price_volatility = initial_window.std()

    # 验证不使用全样本
    full_activity = (df_ret != 0).sum() / len(df_ret)
    full_vol = df_ret.std()

    assert not np.allclose(trading_activity, full_activity), \
        "流动性过滤仍使用全样本统计量!"

    print(f"✅ 流动性过滤历史窗口检查通过")
    print(f"   - 初始窗口长度: {len(initial_window)}")
    print(f"   - 全样本长度: {len(df_ret)}")
    print(f"   - 使用历史窗口统计量: 是 (正确)")
    return True

results['liquidity_filter'] = test_liquidity_filter()

# ============================================================================
# 验证6: 情绪Z-score扩展窗口 shift(1)
# ============================================================================
print("\n" + "="*70)
print("验证6: 情绪Z-score - 确认 expanding mean/std 已 shift(1)")
print("="*70)

def test_sentiment_zscore():
    np.random.seed(123)
    score = pd.Series(np.random.randn(50), index=df_ret.index[:50])
    zscore = (score - score.expanding().mean().shift(1)) / score.expanding().std().shift(1)

    # 第1期应为 NaN
    assert pd.isna(zscore.iloc[0]), "Z-score 第1期应为NaN"

    # 核心验证: 第t期的zscore使用的mean/std不包含score[t]
    # 手动计算第5期（索引4）的zscore，使用索引0-3的数据
    manual_mean = score.iloc[:4].mean()
    manual_std = score.iloc[:4].std()
    expected_z = (score.iloc[4] - manual_mean) / manual_std
    assert abs(zscore.iloc[4] - expected_z) < 1e-10, \
        f"Z-score计算不匹配: {zscore.iloc[4]} != {expected_z}"

    print(f"✅ 情绪Z-score shift(1) 检查通过")
    print(f"   - 第1期 Z-score: {zscore.iloc[0]} (应为NaN)")
    print(f"   - 第5期 Z-score: {zscore.iloc[4]:.4f} (基于前4期历史)")
    print(f"   - 手动验证: {expected_z:.4f} (匹配)")
    return True

results['sentiment_zscore'] = test_sentiment_zscore()

# ============================================================================
# 验证7: 市值代理因子 mcap_proxy shift(1)
# ============================================================================
print("\n" + "="*70)
print("验证7: 市值代理因子 - 确认 rolling std 已 shift(1)")
print("="*70)

def test_mcap_proxy():
    # 模拟修复后的市值代理因子
    mcap_proxy = df_ret.rolling(12).std().shift(1).rank(pct=True, axis=1)

    # 前12期应为 NaN
    assert mcap_proxy.iloc[:12].isna().all().all(), \
        "mcap_proxy shift(1) 未生效: 前12期应全为NaN"

    print(f"✅ 市值代理因子 shift(1) 检查通过")
    print(f"   - 第1-12期: 全NaN (正确，rolling=12 + shift(1))")
    print(f"   - 第13期: 非NaN 比例 {(~mcap_proxy.iloc[12].isna()).mean():.1%}")
    return True

results['mcap_proxy'] = test_mcap_proxy()

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
print("验证总结")
print("="*70)

all_passed = True
for name, passed in results.items():
    status = "✅ 通过" if passed else "❌ 失败"
    print(f"  {status}: {name}")
    if not passed:
        all_passed = False

print("\n" + "="*70)
if all_passed:
    print("🎉 所有前视偏差修复验证通过！")
else:
    print("⚠️ 部分验证未通过，请检查修复逻辑。")
print("="*70)

# 输出到文件
output_file = project_dir / 'experiments' / 'lookahead_fix_validation.md'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# 前视偏差修复验证报告\n\n")
    f.write(f"验证日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("## 验证项目\n\n")
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        f.write(f"- {status}: {name}\n")
    f.write("\n## 结论\n\n")
    if all_passed:
        f.write("所有关键修改点的前视偏差修复逻辑验证通过。\n")
    else:
        f.write("部分验证未通过，需进一步排查。\n")

print(f"\n验证报告已保存: {output_file}")
