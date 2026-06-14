"""
更新 docx 中的实验数据为修复前视偏差后的最新结果
"""
from docx import Document
from pathlib import Path

doc_path = Path('/Users/xinyutan/Documents/量化投资/quant-project/results/策略总结与优化历程.docx')
doc = Document(doc_path)

# ====== 表格 3: Week 8 子样本稳健性 ======
table = doc.tables[3]
# 行 1: 全样本(2005-2025)
table.rows[1].cells[1].text = '19.88%'  # 年化收益
table.rows[1].cells[2].text = '1.04'    # 夏普
table.rows[1].cells[3].text = '-18.94%' # 最大回撤
table.rows[1].cells[4].text = '12.27%'  # Alpha
table.rows[1].cells[5].text = '5.10***' # t值

# 行 2: 前半(2005-2015)
table.rows[2].cells[1].text = '31.22%'
table.rows[2].cells[2].text = '1.31'
table.rows[2].cells[3].text = '-15.33%'
table.rows[2].cells[4].text = '15.56%'
table.rows[2].cells[5].text = '4.07***'

# 行 3: 后半(2016-2025)
table.rows[3].cells[1].text = '8.60%'
table.rows[3].cells[2].text = '0.63'
table.rows[3].cells[3].text = '-24.68%'
table.rows[3].cells[4].text = '8.19%'
table.rows[3].cells[5].text = '3.00***'

# ====== 表格 10: Week 2 实验1 形成期探索 (基线换手率) ======
table = doc.tables[10]
# 行 1: 1月 -> 基线换手率更新
table.rows[1].cells[4].text = '656%'  # 年化换手 (655.95% 四舍五入)

# ====== 表格 13: Week 3 成本敏感性 ======
table = doc.tables[13]
# 行 2: 10 bps 数据更新
table.rows[2].cells[1].text = '20.65%'  # 年化收益
table.rows[2].cells[2].text = '0.70'    # 夏普比率

# ====== 表格 14: Week 3 任务4 推荐配置 ======
table = doc.tables[14]
# 行 3: K=6+TopK=100 换手率更新
table.rows[3].cells[6].text = '656%'  # 换手率

# 保存
doc.save(doc_path)
print(f"已更新 docx 数据: {doc_path}")
