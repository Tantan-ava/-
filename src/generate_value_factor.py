# -*- coding: utf-8 -*-
"""
生成价值因子数据文件
保存为parquet格式供策略代码读取
"""
import pandas as pd
import os

def generate_value_factor():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')

    # 加载月度收益数据
    print("加载月度收益数据...")
    df_ret = pd.read_excel(os.path.join(raw_dir, 'TRD_Mnth.xlsx'))

    # 检查格式并转为宽格式
    if 'Stkcd' in df_ret.columns:
        df_ret['Stkcd'] = df_ret['Stkcd'].astype(str).str.zfill(6)
        df_ret['Trdmnt'] = pd.to_datetime(df_ret['Trdmnt'])
        df_ret = df_ret.pivot(index='Trdmnt', columns='Stkcd', values='Mretwd')
    else:
        # 已是宽格式，确保索引为日期
        if df_ret.index.dtype == object or not pd.api.types.is_datetime64_any_dtype(df_ret.index):
            if 'Trdmnt' in df_ret.columns:
                df_ret = df_ret.set_index('Trdmnt')
            df_ret.index = pd.to_datetime(df_ret.index)

    # 确保数值类型
    df_ret = df_ret.apply(pd.to_numeric, errors='coerce')
    df_ret = df_ret.sort_index()

    # 生成价值因子
    print("生成价值因子...")
    value_factor = df_ret.rolling(12, min_periods=6).sum().shift(1)

    # 保存
    output_path = os.path.join(processed_dir, 'value_factor.parquet')
    value_factor.to_parquet(output_path)
    print(f"价值因子数据已保存: {output_path}")
    print(f"  形状: {value_factor.shape}")
    print(f"  日期范围: {value_factor.index[0]} ~ {value_factor.index[-1]}")
    print(f"  非空比例: {value_factor.notna().sum().sum() / (value_factor.shape[0] * value_factor.shape[1]):.1%}")

if __name__ == '__main__':
    generate_value_factor()
