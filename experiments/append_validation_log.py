"""
Append validation results to exp_log.md
"""
import os
from datetime import datetime

log_path = '/Users/xinyutan/Documents/量化投资/quant-project/experiments/exp_log.md'

content = f"""

---

## {datetime.now().strftime('%Y-%m-%d')}

### 任务：前视偏差系统性审查与修复验证

#### 背景
在策略开发过程中发现多处潜在的前视偏差（Look-ahead Bias），包括滚动/扩展窗口统计量未滞后、ML信号预测周期未对齐、风险平价权重包含当期收益、流动性过滤使用全样本统计量等问题。本次任务对代码库进行系统性审查并修复。

#### 修复文件清单

| 文件路径 | 修复内容 | 修复类型 |
|---------|---------|---------|
| `src/run_optimization_experiments.py` | 滚动夏普均值/标准差添加 `.shift(1)`；组合波动率添加 `.shift(1)`；风险预算方差添加 `.shift(1)`；风险平价历史窗口改为 `iloc[start_idx:date_idx]` 排除当前日期 | 计算方法 |
| `src/run_week8_tutorial.py` | 市值代理因子 `mcap_proxy` 的 `rolling(12).std()` 添加 `.shift(1)` | 计算方法 |
| `src/run_week7_experiments.py` | ML信号 `ml_signal.fillna(0)` 添加 `.shift(1)` 对齐持仓周期 | 逻辑对齐 |
| `src/run_week7_ml_and_combo.py` | ML信号添加 `.shift(1)` 对齐持仓周期 | 逻辑对齐 |
| `src/run_experiments.py` | 流动性过滤改为使用初始24个月数据计算统计量，避免全样本前视偏差 | 静态筛选 |
| `src/strategy/daily_tactician.py` | Z-score的 `expanding().mean/std` 添加 `.shift(1)`；阈值校准的 `expanding().quantile()` 添加 `.shift(1)`；增加NaN阈值边界保护 | 扩展窗口 |
| `src/run_epu_weight_experiment.py` | EPU扩展窗口统计量（median/mean/p75/p25）及min/max归一化均添加 `.shift(1)` | 扩展窗口 |
| `src/run_turnover_epu_experiment.py` | EPU阈值及中位数权重调整的 `expanding()` 统计添加 `.shift(1)` | 扩展窗口 |

#### 验证方法
编写 `src/validate_lookahead_fixes.py` 对7个核心修改点进行自动化验证：
1. 风险平价历史窗口不包含当前日期
2. 滚动波动率/均值已添加 `shift(1)`
3. 扩展窗口分位数已添加 `shift(1)`
4. ML信号预测周期已对齐持仓周期
5. 流动性过滤使用历史窗口而非全样本
6. 情绪Z-score扩展窗口已排除当日
7. 市值代理因子滚动标准差已添加 `shift(1)`

#### 验证结果

| 验证项 | 状态 |
|-------|------|
| 风险平价历史窗口 | ✅ 通过 |
| 滚动波动率 shift(1) | ✅ 通过 |
| 扩展窗口分位数 shift(1) | ✅ 通过 |
| ML信号 horizon 对齐 | ✅ 通过 |
| 流动性过滤历史窗口 | ✅ 通过 |
| 情绪Z-score shift(1) | ✅ 通过 |
| 市值代理因子 shift(1) | ✅ 通过 |

**结论：所有前视偏差修复逻辑验证通过。**

#### 关键风险点说明

1. **`.expanding().quantile()` 的前视偏差**：`expanding()` 在日期 `t` 的结果包含 `t` 自身，若直接用于 `t` 的决策阈值，则使用了当日信息。修复方式：`.expanding().quantile(...).shift(1)`。
2. **风险平价权重 `loc[:date]` 的隐患**：Pandas `loc[:date]` 包含 `date` 本身，`iloc[-60:]` 会取到当前期。修复方式：使用索引切片 `iloc[start_idx:date_idx]`。
3. **ML预测 horizon 不匹配**：`target = df_ret.shift(-1)` 表示预测下月收益，但回测直接使用当月信号决定当月持仓。修复方式：`ml_signal.shift(1)`。
4. **静态筛选的全样本泄露**：流动性过滤若在回测开始前用全样本计算阈值，则后续每期持仓都隐含使用了未来信息。修复方式：仅使用回测初期的历史窗口计算。

#### 后续检查清单

1. 任何 `rolling(X).mean/std/var/sum()` 用于当期决策 → 确认是否已 `.shift(1)`
2. 任何 `expanding().quantile/median/mean/std()` 用于当期阈值 → 确认是否已 `.shift(1)`
3. 任何 `.quantile()/.median()/.std()` 直接在完整 Series 上调用 → 确认是否为纯分析/作图
4. ML/预测模型 → 确认预测目标 `y` 的领先期数与回测权重应用期数匹配
5. 静态股票池筛选 → 确认筛选指标仅使用筛选时点之前的历史数据
6. 风险平价/最小方差权重 → 确认波动率计算窗口不包含当期收益

"""

with open(log_path, 'a', encoding='utf-8') as f:
    f.write(content)

print(f"已追加验证记录到 {log_path}")
