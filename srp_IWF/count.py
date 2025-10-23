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

def ensure_event_and_total(df):

    # 删除重复列，保留第一个
    df = df.loc[:, ~df.columns.duplicated()]

    """
    确保 DataFrame 有 'Event' 列（比赛项/量级）和 'Total' 列（总成绩）。
    如果列名有变体，尝试重命名；并把 Total 转为数值（无法解析的设为 NaN）。
    """
    # 常见列名变体
    event_poss = ['Event', 'event', '项目']
    total_poss = ['Total', 'total', '总成绩', 'TotalResult']

    # 统一 Event 列名
    for c in df.columns:
        if c in event_poss:
            df = df.rename(columns={c: 'Event'})
            break
    df['Event'] = df['Event'].astype(str).str.strip() # 强制转字符串


    # 统一 Total 列名
    for c in df.columns:
        if c in total_poss:
            df = df.rename(columns={c: 'Total'})
            break
    # 把 Total 尝试转为数值（非数字或 '---' 会变为 NaN）
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

    # 如果没有 Event 或 Total，尽量继续（Event 用于 top10 筛选；Total 若缺失则无法排序）
    if 'Event' not in df.columns:
        # 如果没有 Event，我们可以尝试从其它列推断，但这里直接返回 None（跳过）
        print("  ⚠️ 未找到 Event 列，将无法基于项目前10筛选该 sheet。")
        return None

    if 'Total' not in df.columns:
        print("  ⚠️ 未找到 Total 列，该 sheet 将不参与项目前10的计算（但仍计入总量统计）。")
        # 仍返回 df，但标记 Total 缺失
        return df



    return df

# ========== 3. 主函数 ==========
def main():
    df_list = []
    df_list_for_top10 = []  # 专门收集有 Event & Total 的表，用于项目前10筛选

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue

        print(f"读取文件: {os.path.basename(file_path)}")

        # ---- 用 sheet_name=None 读取所有 sheet ----
        all_sheets = pd.read_excel(file_path, sheet_name=None)

        for sheet_name, df in all_sheets.items():
            if df is None or df.empty:
                print(f"  ⚠️ 跳过空 sheet: {sheet_name}")
                continue

            print(f"  正在处理 sheet: {sheet_name}")

        
            df.columns = df.columns.str.strip()  # 去掉前后空格
            # 删除重复列，保留第一次出现的
            df = df.loc[:, ~df.columns.duplicated()]


            # 1) 标准化 Nation 列
            df = ensure_nation_column(df)
            if df is None:
                continue  # 没有 Nation 列的表跳过

            # 2) 记录来源信息（可选，便于追溯）
            df['_file'] = os.path.basename(file_path)
            df['_sheet'] = sheet_name

            # 3) 尝试标准化 Event/Total 列（如果存在）
            df_event_checked = ensure_event_and_total(df.copy())

            # 4) 将所有含 Nation 的表都纳入总数据集合
            df_list.append(df)

            # 5) 只有当该 sheet 同时含有 Event 列（且 Total 列可用或存在）时，才纳入 top10 分析集合
            if df_event_checked is not None and 'Event' in df_event_checked.columns:
                # 注意： 如果 Total 缺失（全部 NaN），该 sheet 也会被跳过 top10 逻辑
                if 'Total' in df_event_checked.columns:
                    # 仅当 Total 列存在并至少有一些非 NaN 值时，才加入
                    if df_event_checked['Total'].notna().any():
                        if 'Event' in df_event_checked.columns:
                            print(f"调试 Sheet: {sheet_name}")
                            print("列名:", df_event_checked.columns.tolist())
                            print("Event 前几行:", df_event_checked['Event'].head(10))
                            print("Event 数据类型:", df_event_checked['Event'].dtype)
                            if isinstance(df_event_checked['Event'], pd.DataFrame):
                                print("⚠️ Event 列重复，列名:", df_event_checked['Event'].columns.tolist())
                            else:
                                print("Event 数据类型:", df_event_checked['Event'].dtype)
                        df_list_for_top10.append(df_event_checked)
                    else:
                        print(f"    ⚠️ sheet {sheet_name} 的 Total 全为缺失，跳过项目前10统计。")
                else:
                    print(f"    ⚠️ sheet {sheet_name} 没有 Total 列，无法用于项目前10统计。")
            
            # 6.查看是否有重复列
            print("Sheet:", sheet_name)
            print("Columns:", df.columns.tolist())
            print("Duplicated columns:", df.columns[df.columns.duplicated()].tolist())

    if not df_list:
        print("❌ 没有可用的 sheet 被读取。请检查列名是否含 'Nation'")
        return

    # ---- 合并所有 DataFrame（总数据）----
    all_data = pd.concat(df_list, ignore_index=True)
    print(f"✅ 合并完成，总行数（含所有 sheet）：{len(all_data)}")

    # ---- 统计总数据量前20国家 ----
    total_top20 = (
        all_data['Nation']
        .value_counts()
        .head(20)               # 先取前20
        .reset_index()          # 重置索引
    )

    total_top20.columns = ['Nation', 'Count']  # 直接改列名

    # ---- 基于“项目前10”筛选并统计国家（先合并所有可用的 Event+Total sheet）----
    if df_list_for_top10:
        event_data = pd.concat(df_list_for_top10, ignore_index=True)

        # 检查合并后列名是否重复
        print("合并后列名:", event_data.columns.tolist())
        
        print("合并后 all_data 列名:", all_data.columns.tolist())
        print("合并后 event_data 列名:", event_data.columns.tolist())


        # 强制 Event 为字符串并去掉空格
        event_data['Event'] = event_data['Event'].astype(str).str.strip()

        event_data['Year'] = event_data['_sheet'].str.extract(r'(\d{4})').astype('Int64')

        top10_per_event = (
            event_data
            .sort_values(['Year', 'Event', 'Total'], ascending=[True, True, False])
            .groupby(['Year', 'Event'], group_keys=False)
            .head(10)
        )

        # 统计这些项目前10中各国家出现的次数，并取前20
        top10_top20 = (
            top10_per_event['Nation']
            .value_counts()
            .head(20)              # 先取前20
            .reset_index()          # 重置索引
        )

        top10_top20.columns = ['Nation', 'Count']  # 直接改列名
    else:
        top10_top20 = pd.DataFrame(columns=['Nation', 'Count'])
        print("⚠️ 未找到可用于项目前10统计的表（缺少 Event 或 Total 数据）")

    # ---- 计算交集 / 差集（同时在两个榜单、仅在总量榜、仅在项目前10榜）----
    set_total = set(total_top20['Nation'].tolist())
    set_top10 = set(top10_top20['Nation'].tolist())

    both = sorted(list(set_total & set_top10))
    only_total = sorted(list(set_total - set_top10))
    only_top10 = sorted(list(set_top10 - set_total))

    # ---- 输出：保存 CSV 和 一个多 sheet 的 Excel（含对比） ----
    # 1) 仍保持原来的 CSV（仅总量前20）
    out_csv = os.path.join(base_dir, "top20_nations.csv")
    total_top20.to_csv(out_csv, index=False)
    print(f"✅ 已保存总量前20（CSV）：{out_csv}")

    # 2) 保存一个含多个 sheet 的 Excel，包含：总量前20、项目前10前20、对比结果
    out_xlsx = os.path.join(base_dir, "top20_nation.xlsx")
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        total_top20.to_excel(writer, sheet_name='总数据量前20', index=False)
        top10_top20.to_excel(writer, sheet_name='项目前10的前20国家', index=False)

        # 比较表：把交集与差集写成易读的表格
        compare_df = pd.DataFrame({
            '同时在两者中': pd.Series(both),
            '只在总数据量前20': pd.Series(only_total),
            '只在项目前10前20': pd.Series(only_top10),
        })
        compare_df.to_excel(writer, sheet_name='对比结果', index=False)

    print(f"✅ 已保存多 sheet Excel：{out_xlsx}")

    # ---- 绘图：并分别保存两个榜单的条形图 ----
    if not total_top20.empty:
        plt.figure(figsize=(10,6))
        plt.bar(total_top20['Nation'], total_top20['Count'],color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 20 Nations by Total Data Count")
        plt.ylabel("Number of Records")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "top20_all.png"))
        plt.close()
        print("✅ 已保存图像：top20_all.png")

    if not top10_top20.empty:
        plt.figure(figsize=(10,6))
        plt.bar(top10_top20['Nation'], top10_top20['Count'],color='orange')
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 20 Nations Among Event Top-10")
        plt.ylabel("Count in Event Top-10")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "top20_rank10.png"))
        plt.close()
        print("✅ 已保存图像：top20_rank10.png")

if __name__ == "__main__":
    main()
