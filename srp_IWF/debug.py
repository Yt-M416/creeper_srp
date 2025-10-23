# debug_event_columns.py
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
file_list = [
    os.path.join(base_dir, "iwf举重_1998-2017.xlsx"),
    os.path.join(base_dir, "iwf举重_2018-2025.xlsx")
]

def check_columns(df, sheet_name, file_name):
    # 去掉前后空格
    df.columns = df.columns.str.strip()
    # 打印列名和重复列
    print(f"\n文件: {file_name} | sheet: {sheet_name}")
    print("所有列名:", df.columns.tolist())
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        print("⚠️ 重复列:", dup)
    else:
        print("✅ 列名无重复")

for file_path in file_list:
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        continue

    all_sheets = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in all_sheets.items():
        if df is None or df.empty:
            continue
        check_columns(df, sheet_name, os.path.basename(file_path))



