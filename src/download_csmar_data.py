# -*- coding: utf-8 -*-
"""
CSMAR 数据下载脚本
下载量化策略所需的基础数据

需要的数据表：
- TRD_Mnth: 月度个股回报率（市值数据）
- FI_T2: 财务指标（ROE、扣非净利润）
- FS_Comins: 合并利润表（净利润）
- FS_Combas: 合并资产负债表（股东权益）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CSMARDataDownloader:
    """CSMAR 数据下载器"""
    
    def __init__(self, data_dir: str = '../data/raw'):
        """
        初始化下载器
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据表映射
        self.tables = {
            'TRD_Mnth': {
                'name': '月度个股回报率文件',
                'output': 'TRD_Mnth.xlsx',
                'key_fields': ['Stkcd', 'Trdmnt'],
                'needed_columns': [
                    'Stkcd',      # 股票代码
                    'Trdmnt',     # 交易月份
                    'Mnthret',    # 月度收益率
                    'Msmvttl',    # 总市值（包括非流通股）
                    'Clsprc'      # 收盘价
                ]
            },
            'FI_T2': {
                'name': '财务指标文件',
                'output': 'FI_T2.xlsx',
                'key_fields': ['Stkcd', 'Accper'],
                'needed_columns': [
                    'Stkcd',      # 股票代码
                    'Accper',     # 会计期间
                    'Dednnpci',   # 扣除非经常性损益后的净利润
                    'Wroa'        # 加权平均净资产收益率 ROE
                ]
            },
            'FS_Comins': {
                'name': '合并利润表',
                'output': 'FS_Comins.xlsx',
                'key_fields': ['Stkcd', 'Accper'],
                'needed_columns': [
                    'Stkcd',      # 股票代码
                    'Accper',     # 会计期间
                    'B'           # 净利润
                ]
            },
            'FS_Combas': {
                'name': '合并资产负债表',
                'output': 'FS_Combas.xlsx',
                'key_fields': ['Stkcd', 'Accper'],
                'needed_columns': [
                    'Stkcd',      # 股票代码
                    'Accper',     # 会计期间
                    'B'           # 股东权益合计
                ]
            }
        }
    
    def download_from_csmar(self, table_name: str) -> pd.DataFrame:
        """
        从 CSMAR 数据库下载数据
        
        方法 1: 如果您有 CSMAR API 访问权限
        方法 2: 从已下载的 Excel 文件中读取
        
        Args:
            table_name: 表名
            
        Returns:
            DataFrame
        """
        print(f"\n正在处理 {self.tables[table_name]['name']}...")
        
        # 检查是否已经有下载好的数据
        source_file = self.data_dir / f'{table_name}_source.xlsx'
        if source_file.exists():
            print(f"从本地文件读取：{source_file}")
            df = pd.read_excel(source_file)
            return df
        
        # 尝试直接读取 CSMAR 导出的文件
        output_file = self.data_dir / self.tables[table_name]['output']
        if output_file.exists():
            print(f"从已处理文件读取：{output_file}")
            df = pd.read_excel(output_file)
            return df
        
        # 如果没有数据，提供下载指引
        print(f"\n未找到 {table_name} 数据文件")
        print(f"请从 CSMAR 数据库下载 {self.tables[table_name]['name']}")
        print(f"下载后保存为：{source_file}")
        
        return None
    
    def process_trd_mnth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理 TRD_Mnth 表（月度回报率）
        
        Args:
            df: 原始数据
            
        Returns:
            处理后的数据
        """
        print("处理 TRD_Mnth 表...")
        
        # 选择需要的列
        columns = self.tables['TRD_Mnth']['needed_columns']
        df = df[columns].copy()
        
        # 重命名
        df.columns = ['stkcd', 'month', 'ret', 'mvttl', 'price']
        
        # 数据清洗
        df = df.dropna(subset=['ret', 'mvttl'])
        df = df[df['ret'] != -999999999]
        df = df[df['mvttl'] != -999999999]
        
        # 格式化股票代码
        df['stkcd'] = df['stkcd'].astype(str).str.zfill(6)
        
        # 转换日期
        df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
        
        # 对数市值
        df['ln_mvttl'] = np.log(df['mvttl'])
        
        df = df.sort_values(['stkcd', 'month']).reset_index(drop=True)
        
        print(f"  处理后数据形状：{df.shape}")
        return df
    
    def process_fi_t2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理 FI_T2 表（财务指标）
        
        Args:
            df: 原始数据
            
        Returns:
            处理后的数据
        """
        print("处理 FI_T2 表...")
        
        columns = self.tables['FI_T2']['needed_columns']
        df = df[columns].copy()
        
        # 重命名
        df.columns = ['stkcd', 'accper', 'dednnpci', 'roe']
        
        # 数据清洗
        df = df.dropna(subset=['dednnpci', 'roe'])
        df = df[df['dednnpci'] != -999999999]
        df = df[df['roe'] != -999999999]
        
        # 格式化
        df['stkcd'] = df['stkcd'].astype(str).str.zfill(6)
        df['accper'] = pd.to_datetime(df['accper'])
        
        # 只保留年报数据（12 月 31 日）
        df = df[df['accper'].dt.month == 12]
        df['year'] = df['accper'].dt.year
        
        df = df.sort_values(['stkcd', 'accper']).reset_index(drop=True)
        
        print(f"  处理后数据形状：{df.shape}")
        return df
    
    def process_fs_comins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理 FS_Comins 表（合并利润表）
        
        Args:
            df: 原始数据
            
        Returns:
            处理后的数据
        """
        print("处理 FS_Comins 表...")
        
        columns = self.tables['FS_Comins']['needed_columns']
        df = df[columns].copy()
        
        # 重命名
        df.columns = ['stkcd', 'accper', 'netprofit']
        
        # 数据清洗
        df = df.dropna(subset=['netprofit'])
        df = df[df['netprofit'] != -999999999]
        
        # 格式化
        df['stkcd'] = df['stkcd'].astype(str).str.zfill(6)
        df['accper'] = pd.to_datetime(df['accper'])
        
        # 只保留年报数据
        df = df[df['accper'].dt.month == 12]
        df['year'] = df['accper'].dt.year
        
        df = df.sort_values(['stkcd', 'accper']).reset_index(drop=True)
        
        print(f"  处理后数据形状：{df.shape}")
        return df
    
    def process_fs_combas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理 FS_Combas 表（合并资产负债表）
        
        Args:
            df: 原始数据
            
        Returns:
            处理后的数据
        """
        print("处理 FS_Combas 表...")
        
        columns = self.tables['FS_Combas']['needed_columns']
        df = df[columns].copy()
        
        # 重命名
        df.columns = ['stkcd', 'accper', 'equity']
        
        # 数据清洗
        df = df.dropna(subset=['equity'])
        df = df[df['equity'] != -999999999]
        
        # 格式化
        df['stkcd'] = df['stkcd'].astype(str).str.zfill(6)
        df['accper'] = pd.to_datetime(df['accper'])
        
        # 只保留年报数据
        df = df[df['accper'].dt.month == 12]
        df['year'] = df['accper'].dt.year
        
        df = df.sort_values(['stkcd', 'accper']).reset_index(drop=True)
        
        print(f"  处理后数据形状：{df.shape}")
        return df
    
    def merge_all_data(self) -> pd.DataFrame:
        """
        合并所有数据表
        
        Returns:
            合并后的 DataFrame
        """
        print("\n" + "=" * 60)
        print("开始合并数据")
        print("=" * 60)
        
        # 加载并处理 TRD_Mnth
        trd_df = self.download_from_csmar('TRD_Mnth')
        if trd_df is None:
            raise ValueError("TRD_Mnth 数据不存在")
        trd_processed = self.process_trd_mnth(trd_df)
        
        # 加载并处理 FI_T2
        fi_df = self.download_from_csmar('FI_T2')
        if fi_df is None:
            raise ValueError("FI_T2 数据不存在")
        fi_processed = self.process_fi_t2(fi_df)
        
        # 加载并处理 FS_Comins（可选，作为盈利的备选）
        comins_df = self.download_from_csmar('FS_Comins')
        if comins_df is not None:
            comins_processed = self.process_fs_comins(comins_df)
        else:
            comins_processed = None
        
        # 加载并处理 FS_Combas
        combas_df = self.download_from_csmar('FS_Combas')
        if combas_df is None:
            raise ValueError("FS_Combas 数据不存在")
        combas_processed = self.process_fs_combas(combas_df)
        
        # 合并财务数据（年度）
        print("\n合并财务数据...")
        financial_df = fi_processed.merge(
            combas_processed[['stkcd', 'year', 'equity']],
            on=['stkcd', 'year'],
            how='inner'
        )
        
        if comins_processed is not None:
            financial_df = financial_df.merge(
                comins_processed[['stkcd', 'year', 'netprofit']],
                on=['stkcd', 'year'],
                how='left'
            )
        
        print(f"财务数据形状：{financial_df.shape}")
        
        # 将年度数据扩展到月度
        print("\n将年度财务数据扩展到月度...")
        financial_monthly = self.expand_to_monthly(financial_df, trd_processed)
        
        # 合并月度数据
        print("\n合并月度和财务数据...")
        merged_df = trd_processed.merge(
            financial_monthly,
            on=['stkcd', 'month'],
            how='left'
        )
        
        print(f"合并后数据形状：{merged_df.shape}")
        
        # 计算 B/M 比率
        print("\n计算 B/M 比率...")
        merged_df['bm'] = merged_df['equity'] / merged_df['mvttl']
        merged_df['ln_bm'] = np.log(merged_df['bm'])
        
        # 保存处理后的数据
        output_file = self.data_dir.parent / 'processed' / 'merged_data.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        merged_df.to_csv(output_file, index=False)
        print(f"\n✓ 合并数据已保存至：{output_file}")
        
        return merged_df
    
    def expand_to_monthly(self, financial_df: pd.DataFrame, 
                         monthly_ref: pd.DataFrame) -> pd.DataFrame:
        """
        将年度财务数据扩展到月度
        
        Args:
            financial_df: 年度财务数据
            monthly_ref: 月度参考数据（用于确定需要哪些月份）
            
        Returns:
            月度财务数据
        """
        # 获取所有需要的月份
        months = monthly_ref['month'].unique()
        
        # 为每个股票扩展年度数据到月度
        expanded_list = []
        
        for stkcd in financial_df['stkcd'].unique():
            stock_data = financial_df[financial_df['stkcd'] == stkcd]
            
            for month in months:
                # 找到该月份对应的年度数据（使用上一年的年报数据）
                year = month.year
                
                # 如果是 1-6 月，使用上一年度的数据
                # 如果是 7-12 月，使用当年度的数据
                if month.month <= 6:
                    ref_year = year - 1
                else:
                    ref_year = year
                
                year_data = stock_data[stock_data['year'] == ref_year]
                
                if len(year_data) > 0:
                    row = year_data.iloc[0].copy()
                    row['month'] = month
                    expanded_list.append(row)
        
        if expanded_list:
            expanded_df = pd.DataFrame(expanded_list)
            return expanded_df
        else:
            return pd.DataFrame()
    
    def download_and_merge(self):
        """下载并合并所有数据"""
        try:
            merged_df = self.merge_all_data()
            
            print("\n" + "=" * 60)
            print("数据合并完成！")
            print("=" * 60)
            print(f"\n最终数据形状：{merged_df.shape}")
            print(f"\n包含的变量:")
            for col in merged_df.columns:
                print(f"  - {col}")
            
            print(f"\n数据时间范围:")
            print(f"  开始：{merged_df['month'].min()}")
            print(f"  结束：{merged_df['month'].max()}")
            
            print(f"\n股票数量：{merged_df['stkcd'].nunique()}")
            
            return merged_df
            
        except Exception as e:
            print(f"\n✗ 数据合并失败：{str(e)}")
            raise


def main():
    """主函数"""
    print("=" * 60)
    print("CSMAR 数据下载与整合工具")
    print("=" * 60)
    
    downloader = CSMARDataDownloader()
    
    # 检查数据文件
    print("\n检查数据文件...")
    for table_name, table_info in downloader.tables.items():
        output_file = downloader.data_dir / table_info['output']
        source_file = downloader.data_dir / f'{table_name}_source.xlsx'
        
        if source_file.exists():
            print(f"✓ {table_name}: 找到源文件")
        elif output_file.exists():
            print(f"✓ {table_name}: 找到已处理文件")
        else:
            print(f"✗ {table_name}: 未找到文件")
    
    print("\n" + "=" * 60)
    print("请确保已从 CSMAR 下载以下数据表:")
    print("  1. TRD_Mnth - 月度个股回报率")
    print("  2. FI_T2 - 财务指标")
    print("  3. FS_Comins - 合并利润表（可选）")
    print("  4. FS_Combas - 合并资产负债表")
    print("=" * 60)
    
    choice = input("\n是否继续合并数据？(y/n): ")
    
    if choice.lower() == 'y':
        try:
            merged_df = downloader.download_and_merge()
            print("\n✓ 数据处理完成！")
        except Exception as e:
            print(f"\n✗ 处理失败：{e}")
            print("\n请确保已下载所有必需的数据文件")
    else:
        print("已取消")


if __name__ == '__main__':
    main()
