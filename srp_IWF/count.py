'''
用于统计数据量前20的国家
'''
# analyze_two_excels.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# ========== 1. 指定要处理的两个文件 ==========
base_dir = os.path.dirname(__file__)  # 当前脚本所在目录
file_list = [
    os.path.join(base_dir, "iwf举重_1998-2017.xlsx"),
    os.path.join(base_dir, "iwf举重_2018-2025.xlsx")
]

# ========== 2. 定义辅助函数：标准化列名、清理数据 ==========
def ensure_nation_column(df):
    """
    确保 DataFrame 里有 'Nation' 列。
    如果原表列名不同（如 'country'、'国家'），这里会尝试重命名。
    """
    possible = ['Nation', 'nation', 'Country', 'country', '国家']
    for col in df.columns:
        if col in possible:
            df = df.rename(columns={col: 'Nation'})
            break
    if 'Nation' not in df.columns:
        print("⚠️ 该表未找到 Nation 列，将跳过。")
        return None
    # 去掉空格并统一为大写，保证同一个国家不会重复计数
    df['Nation'] = df['Nation'].astype(str).str.strip().str.upper()
    df = df[df['Nation'] != '']  # 删除空值
    return df

# ========== 3. 主函数 ==========
def main():
    df_list = []

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue

        print(f"读取文件: {os.path.basename(file_path)}")

        # ---- 用 sheet_name=None 读取所有 sheet ----
        # pandas 会返回一个字典：{sheet名: DataFrame}
        all_sheets = pd.read_excel(file_path, sheet_name=None)

        for sheet_name, df in all_sheets.items():
            if df is None or df.empty:
                print(f"  ⚠️ 跳过空 sheet: {sheet_name}")
                continue

            print(f"  正在处理 sheet: {sheet_name}")
            df = ensure_nation_column(df)
            if df is None:
                continue  # 没有 Nation 列的表跳过

            # 添加来源信息（可选）
            df['_file'] = os.path.basename(file_path)
            df['_sheet'] = sheet_name

            df_list.append(df)

    if not df_list:
        print("❌ 没有可用的 sheet 被读取。请检查列名是否含 'Nation'")
        return

    # ---- 合并所有 DataFrame ----
    all_data = pd.concat(df_list, ignore_index=True)
    print(f"✅ 合并完成，总行数：{len(all_data)}")

    # ---- 统计前20国家 ----
    top20 = (
        all_data['Nation']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'Nation', 'Nation': 'Count'})
        .head(20)
    )

    print("\n🌍 前20国家：")
    print(top20)

    # ---- 保存结果 ----
    out_csv = os.path.join(base_dir, "top20_nations.csv")
    top20.to_csv(out_csv, index=False)
    print(f"✅ 已保存到：{out_csv}")

    # ---- 绘图 ----
    plt.figure(figsize=(10,6))
    plt.bar(top20['Nation'], top20['Count'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 20 Nations by Data Count (1998–2025)")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "top20_nations.png"))
    plt.close()
    print("✅ 已保存图像：top20_nations.png")

if __name__ == "__main__":
    main()
