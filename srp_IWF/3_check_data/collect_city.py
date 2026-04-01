"""
本脚本用于生成一个Excel任务分配表，以帮助团队成员
收集从各国训练基地到历届比赛场馆的旅行时长数据。

该脚本会读取 'iwf_data_with_tz.csv' 文件来获取所有比赛地点，
并结合预设的国家训练基地列表，生成一个名为 
'flight_duration_task_sheet.xlsx' 的文件。
"""

import pandas as pd
import os

# --- 检查输入文件是否存在 ---
input_csv_path = "iwf_data_with_tz.csv"
if not os.path.exists(input_csv_path):
    print(f"错误：输入文件 '{input_csv_path}' 未找到。")
    print("请确保该CSV文件与本脚本在同一目录下。")
else:
    # --- Sheet 1: 国家训练基地数据 ---
    # 根据公开信息和体育新闻整理
    training_base_data = {
        'Country_Code': [
            'ARM', 'CHN', 'COL', 'IRI', 'JPN', 'KAZ', 'KOR', 'MEX', 'RUS', 'THA', 
            'TPE', 'UKR', 'USA', 'BUL', 'CAN', 'ESP', 'FRA', 'POL', 'TUR', 'VEN', 
            'BLR', 'ECU', 'EGY', 'GEO', 'INA', 'PRK', 'UZB'
        ],
        'Training_Base_City': [
            'Yerevan', 'Beijing', 'Bogota', 'Tehran', 'Tokyo', 'Almaty', 'Seoul',
            'San Luis Potosi', 'Sochi', 'Chiang Mai', 'Kaohsiung', 'Kyiv',
            'Colorado Springs', 'Sofia', 'Montreal', 'Madrid', 'Paris', 'Warsaw',
            'Ankara', 'Caracas', 'Minsk', 'Quito', 'Cairo', 'Tbilisi', 'Jakarta',
            'Pyongyang', 'Tashkent'
        ]
    }
    df_bases = pd.DataFrame(training_base_data)

    # --- Sheet 2: 从真实的CSV文件中提取比赛场馆数据 ---
    print(f"正在从 '{input_csv_path}' 读取比赛场馆数据...")
    df_full = pd.read_csv(input_csv_path)

    # 提取所需列
    df_venues = df_full[['City', 'Event_Year']].copy()

    # 清理和去重
    df_venues.dropna(subset=['City', 'Event_Year'], inplace=True)
    df_venues = df_venues.drop_duplicates(subset=['City', 'Event_Year'])

    # 拆分 'City' 列为城市和国家
    # 使用 .str.strip() 来清理可能存在的前后空格
    df_venues[['比赛城市', '比赛国家代码']] = df_venues['City'].str.split(',', n=1, expand=True)
    df_venues['比赛城市'] = df_venues['比赛城市'].str.strip()
    df_venues['比赛国家代码'] = df_venues['比赛国家代码'].str.strip()

    # 创建最终表格，并重命名列
    df_venues_final = df_venues[['比赛国家代码', '比赛城市', 'Event_Year']].copy()
    df_venues_final.rename(columns={'Event_Year': '比赛年份'}, inplace=True)
    
    # 将年份转换为整数，便于排序
    df_venues_final['比赛年份'] = df_venues_final['比赛年份'].astype(int)

    # 排序，方便查看
    df_venues_final = df_venues_final.sort_values(by='比赛年份').reset_index(drop=True)
    print(f"成功提取并处理了 {len(df_venues_final)} 个不重复的比赛场馆记录。")

    # --- 生成Excel文件 ---
    output_filename = 'flight_duration_task_sheet.xlsx'
    with pd.ExcelWriter(output_filename) as writer:
        df_bases.to_excel(writer, sheet_name='Top Countries Training Bases', index=False)
        df_venues_final.to_excel(writer, sheet_name='Competition Venues', index=False)

    print(f"\n任务分配Excel文件 '{output_filename}' 已成功生成！")