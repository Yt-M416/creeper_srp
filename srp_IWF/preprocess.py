#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作用：在原有预处理基础上，为 top13 国家添加训练时区并计算与比赛地的时差（考虑夏令时）。
输出文件：iwf_data_with_tz.csv
注意：
- 需要确保输入数据包含 'Nation' 列（国家代码）和 'Timezone' 列（比赛地时区，如 'Europe/Helsinki'）。
- 会尝试从 'competition' 字段中抽取比赛年份，如抽取不到会使用当年中期 (7月1日) 的默认年份：2015（可调整）。
"""

import os
import glob
import re
from datetime import datetime
import pandas as pd

# 优先使用 zoneinfo（Python 3.9+），否则回退到 pytz
try:
    from zoneinfo import ZoneInfo  # Python stdlib
    ZONEINFO_AVAILABLE = True
except Exception:
    import pytz
    ZONEINFO_AVAILABLE = False

# ----------------- 配置 -----------------
project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'check_data')
output_file = os.path.join(project_dir, 'iwf_data_with_tz.csv')

TARGET_NATIONS = [
    'ARM', 'CHN', 'COL', 'IRI', 'JPN', 'KAZ', 'KOR', 'MEX', 'RUS', 'THA',
    'TPE', 'UKR', 'USA'
]

# Top13 国家 -> 训练地时区（老师/你指定的代表时区）
# 你可以根据老师要求修改这些映射（例如把 RUS 指向特定城市）
nation_timezone_map = {
    'ARM': 'Asia/Yerevan',
    'CHN': 'Asia/Shanghai',
    'COL': 'America/Bogota',
    'IRI': 'Asia/Tehran',
    'JPN': 'Asia/Tokyo',
    'KAZ': 'Asia/Almaty',         # 按老师建议用 Almaty
    'KOR': 'Asia/Seoul',
    'MEX': 'America/Mexico_City', # 或者 San_Luis_Potosi -> 'America/Mexico_City' 常用代表
    'RUS': 'Europe/Moscow',       # 仅举例：若你要精确到不同训练地可细化
    'THA': 'Asia/Bangkok',
    'TPE': 'Asia/Taipei',
    'UKR': 'Europe/Kyiv',         # 注意：时区名可能因历史而变化（Kyiv vs Kiev）
    'USA': 'America/Denver'       # 老师给的是 Colorado Springs -> closer to 'America/Denver'
}

# 若无法从 competition 字段抽出年份，使用此默认年份（用于估算 DST）
FALLBACK_YEAR = 2015

# ----------------------------------------

def extract_year_from_text(s):
    """
    尝试从字符串中抽取第一个四位数年份；找不到则返回 None。
    """
    if not isinstance(s, str):
        return None
    m = re.search(r'(?<!\d)(19|20)\d{2}', s)
    if m:
        return int(m.group(0))
    return None

def get_offset_hours(tz_name, ref_dt):
    """
    返回指定时区在 ref_dt 时间点的 UTC 偏移（小时数，float）。
    支持 zoneinfo 或 pytz。
    若 tz_name 无效或偏移无法计算则返回 None。
    """
    if not isinstance(tz_name, str) or tz_name.strip() == '':
        return None
    tz_name = tz_name.strip()
    try:
        if ZONEINFO_AVAILABLE:
            # 使用 zoneinfo：用 tz-aware datetime
            aware = datetime(ref_dt.year, ref_dt.month, ref_dt.day, tzinfo=ZoneInfo(tz_name))
            offset = aware.utcoffset()
        else:
            # 使用 pytz：先本地化
            tz = pytz.timezone(tz_name)
            aware = tz.localize(datetime(ref_dt.year, ref_dt.month, ref_dt.day))
            offset = aware.utcoffset()
        if offset is None:
            return None
        return offset.total_seconds() / 3600.0
    except Exception:
        return None

def compute_timezone_diffs(df):
    """
    为 DataFrame 添加 Athlete_Timezone, Event_Year, Athlete_UTC_offset, Venue_UTC_offset, Timezone_Diff_hours 列。
    假定 df 已包含 'Nation' 和 'Timezone'（比赛地时区），以及 'competition'（可从中抽年）。
    """
    # 1) 映射运动员训练时区
    df['Athlete_Timezone'] = df['Nation'].map(nation_timezone_map).fillna('')

    # 2) 从 competition 字段尝试提取年份；若无法提取则使用 FALLBACK_YEAR
    df['Event_Year'] = df['competition'].apply(lambda x: extract_year_from_text(x) or FALLBACK_YEAR)

    # 3) 参考日选择：使用当年 7 月 1 日（通常能正确反映夏令时是否生效）
    #    也可以改成比赛具体日期（若你后续可从原数据中解析出具体日期，建议改为真实比赛日期）
    df['Reference_Date'] = df['Event_Year'].apply(lambda y: datetime(int(y), 7, 1))

    # 4) 计算 UTC 偏移（小时）并求差（Athlete - Venue）
    athlete_offsets = []
    venue_offsets = []
    tz_diffs = []

    for _, row in df.iterrows():
        ref_dt = row['Reference_Date']
        venue_tz = row.get('Timezone', '')
        athlete_tz = row.get('Athlete_Timezone', '')

        venue_off = get_offset_hours(venue_tz, ref_dt)
        athlete_off = get_offset_hours(athlete_tz, ref_dt)

        # 计算差： athlete - venue （小时）
        diff = None
        if athlete_off is not None and venue_off is not None:
            diff = athlete_off - venue_off

        athlete_offsets.append(athlete_off)
        venue_offsets.append(venue_off)
        tz_diffs.append(diff)

    df['Athlete_UTC_offset'] = athlete_offsets
    df['Venue_UTC_offset'] = venue_offsets
    df['Timezone_Diff_hours'] = tz_diffs

    # 清理临时列（如果你不想保留 Reference_Date 可取消注释下一行）
    # df.drop(columns=['Reference_Date'], inplace=True)

    return df

def preprocess_and_combine_data(data_path, output_path):
    """
    遍历并合并 Excel，然后针对 top13 国家进行筛选并计算时区差。
    """
    if not os.path.isdir(data_path):
        print(f"错误：数据目录 '{data_path}' 不存在。请检查路径。")
        return

    excel_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    if not excel_files:
        print(f"警告：在目录 '{data_path}' 中未找到任何 .xlsx 文件。")
        return

    print(f"发现 {len(excel_files)} 个Excel文件，开始处理...")

    all_sheets = []
    for file_path in excel_files:
        filename = os.path.basename(file_path)
        print(f"读取文件: {filename}")
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            print(f"  读取 {filename} 出错：{e}")
            continue

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            df.columns = df.columns.str.strip()
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.dropna(how='all')

            # 向下填充元信息
            cols_to_fill = ['Venue', 'City', 'Timezone', 'Event']
            for col in cols_to_fill:
                if col in df.columns:
                    df[col] = df[col].ffill()

            # 添加 competition 字段
            df['competition'] = sheet_name

            all_sheets.append(df)

    if not all_sheets:
        print("未收集到有效数据，结束。")
        return

    combined_df = pd.concat(all_sheets, ignore_index=True)
    combined_df.dropna(how='all', inplace=True)
    print(f"合并后总行数: {len(combined_df)}")

    # 标准化 Nation 列（去空格并大写）
    if 'Nation' not in combined_df.columns:
        print("警告：数据中没有 'Nation' 列，无法继续按国家筛选。将直接退出。")
        return
    combined_df['Nation'] = combined_df['Nation'].astype(str).str.strip().str.upper()

    # 筛选 top13 国家
    pre_filter_rows = len(combined_df)
    combined_df = combined_df[combined_df['Nation'].isin(TARGET_NATIONS)].copy()
    post_filter_rows = len(combined_df)
    print(f"按国家筛选后: {pre_filter_rows} -> {post_filter_rows} 行")

    # 计算时区相关列
    combined_df = compute_timezone_diffs(combined_df)

    # 保存结果
    try:
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 成功！输出已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存出错: {e}")

if __name__ == "__main__":
    preprocess_and_combine_data(data_dir, output_file)
