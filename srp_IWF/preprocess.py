"""
本脚本用于预处理IWF举重世锦赛数据，其核心功能是：
1. 遍历 'check_data' 文件夹下的所有 Excel (.xlsx) 文件。
2. 读取每个 Excel 文件中的所有工作表 (Sheet)。
3. 将所有工作表的数据合并到一个 Pandas DataFrame 中。
4. 在合并过程中，自动向下填充每个工作表独有的元数据（如比赛地点 'Venue'、城市 'City'、时区 'Timezone'），
   这些信息通常只出现在每个表的首行。
5. 【新增功能】根据预定义的国家列表，筛选出特定国家的运动员数据。
6. 增加一列 'competition' 用于数据溯源。
7. 清理完全空行，去掉各项目之间的空行。
8. 将最终整合后的数据保存为一个CSV文件 'iwf_data_filtered.csv'。
"""

import pandas as pd
import os
import glob

# --- 1. 设置文件路径和目标国家 ---
project_dir = os.getcwd()  
data_dir = os.path.join(project_dir, 'check_data')
# 修改输出文件名以反映数据已被筛选
output_file = os.path.join(project_dir, 'iwf_data_filter.csv')

TARGET_NATIONS = [
    'ARM', 'CHN', 'COL', 'IRI', 'JPN', 'KAZ', 'KOR', 'MEX', 'RUS', 'THA', 
    'TPE', 'UKR', 'USA'
    
]


# --- 2. 主处理逻辑 ---
def preprocess_and_combine_data(data_path, output_path):
    """
    遍历、合并、清理并筛选指定路径下的Excel文件。
    """
    if not os.path.isdir(data_path):
        print(f"错误：数据目录 '{data_path}' 不存在。请检查路径。")
        return

    excel_files = glob.glob(os.path.join(data_path, '*.xlsx'))

    if not excel_files:
        print(f"警告：在目录 '{data_path}' 中未找到任何 .xlsx 文件。")
        return

    print(f"发现 {len(excel_files)} 个Excel文件，开始处理...")

    all_sheets_data = []

    for file_path in excel_files:
        filename = os.path.basename(file_path)
        print(f"\n--- 正在读取文件: {filename} ---")
        
        try:
            xls = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            print(f"  读取文件 {filename} 时出错: {e}")
            continue

        for sheet_name, df in xls.items():
            if df.empty:
                print(f"  - 跳过空Sheet: {sheet_name}")
                continue
            
            print(f"  - 正在处理Sheet: {sheet_name}")

            # 先去掉完全空行
            df.dropna(how='all', inplace=True)

            # 向下填充元信息
            cols_to_fill = ['Venue', 'City', 'Timezone', 'Event'] 
            for col in cols_to_fill:
                if col in df.columns:
                    df[col] = df[col].ffill()
                    print(f"    已对 '{col}' 列执行向下填充。")

            # 添加来源信息列
            df['competition'] = sheet_name

            all_sheets_data.append(df)

    if not all_sheets_data:
        print("\n处理完成，但没有收集到任何有效数据。")
        return

    combined_df = pd.concat(all_sheets_data, ignore_index=True)

    # 再次确保没有完全空行
    combined_df.dropna(how='all', inplace=True)
    
    print(f"\n数据合并完成。合并后总行数: {len(combined_df)}")

    # --- 新增筛选逻辑 ---
    print(f"\n开始根据目标国家列表进行筛选...")
    print(f"目标国家: {TARGET_NATIONS}")

    if 'Nation' in combined_df.columns:
        # 预处理'Nation'列，去除前后空格并转为大写，以确保匹配准确
        combined_df['Nation'] = combined_df['Nation'].astype(str).str.strip().str.upper()
        
        original_rows = len(combined_df)
        # 执行筛选
        combined_df = combined_df[combined_df['Nation'].isin(TARGET_NATIONS)].copy()
        filtered_rows = len(combined_df)
        
        print(f"筛选完成。行数从 {original_rows} 减少到 {filtered_rows}")
    else:
        print("警告：数据中未找到 'Nation' 列，无法按国家进行筛选。将输出所有数据。")
    # --- 筛选逻辑结束 ---

    # 清理不再需要的列
    for col in ['Source_File', 'Note']:
        if col in combined_df.columns:
            combined_df.drop(columns=col, inplace=True)

    if not combined_df.empty and 'Nation' in combined_df.columns:
        print("\n--- 对最终筛选结果进行数量验证 ---")
        nation_counts_in_final_data = combined_df['Nation'].value_counts()
        print("最终数据中各目标国家的记录数：")
        print(nation_counts_in_final_data)
        print(f"验证总数: {nation_counts_in_final_data.sum()} (应与筛选后行数 {filtered_rows} 一致)")

    try:
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 成功！筛选后的数据已保存到: {output_path}")
    except Exception as e:
        print(f"\n❌ 保存文件时出错: {e}")


if __name__ == "__main__":
    preprocess_and_combine_data(data_dir, output_file)