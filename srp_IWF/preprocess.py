"""
作用：在原有预处理基础上，为 top13 国家添加训练时区并计算与比赛地的时差（考虑夏令时）。
输入文件：check_data文件夹下的所有xlsx文件
输出文件：iwf_data_with_tz.csv
若 competition 字段中无年份（如 '69th MEN’s and 12th WOMEN’s WOR'），则根据届数匹配到实际年份（手动映射 1998–2010）。
"""

import os
import glob
import re
from datetime import datetime
import pandas as pd

try:
    from zoneinfo import ZoneInfo
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

nation_timezone_map = {
    'ARM': 'Asia/Yerevan',
    'CHN': 'Asia/Shanghai',
    'COL': 'America/Bogota',
    'IRI': 'Asia/Tehran',
    'JPN': 'Asia/Tokyo',
    'KAZ': 'Asia/Almaty',
    'KOR': 'Asia/Seoul',
    'MEX': 'America/Mexico_City',
    'RUS': 'Europe/Moscow',
    'THA': 'Asia/Bangkok',
    'TPE': 'Asia/Taipei',
    'UKR': 'Europe/Kyiv',
    'USA': 'America/Denver'
}

FALLBACK_YEAR = 2015

# ----------------- 新增部分：届数→年份映射 -----------------
edition_to_year_map = {
    69: 1998,
    70: 1999,
    71: 2001,
    72: 2002,
    73: 2003,
    74: 2005,
    75: 2006,
    76: 2007,
    77: 2009,
    78: 2010,
}


# ----------------------------------------

def extract_year_from_text(s):
    """优先提取年份，若无则尝试根据届数映射。"""
    if not isinstance(s, str):
        return None

    # ① 尝试直接匹配四位数年份（2011+）
    m = re.search(r'(?<!\d)(19|20)\d{2}', s)
    if m:
        return int(m.group(0))

    # ② 若无年份，则尝试匹配届数（如 69th、70TH、78Th）
    m2 = re.search(r'(\d{2,3})\s*(?:st|nd|rd|th)', s, flags=re.IGNORECASE)
    if m2:
        edition = int(m2.group(1))
        mapped_year = edition_to_year_map.get(edition)
        if mapped_year:
            # print(f"  ✅ 映射届数 {edition}th -> 年份 {mapped_year}  （原文本：{s}）")
            return mapped_year
        else:
            print(f"  ⚠️ 未找到届数 {edition} 的映射（原文本：{s}）")
            return None

    return None

# ----------------------------------------

def get_offset_hours(tz_name, ref_dt):
    """返回指定时区在 ref_dt 时间点的 UTC 偏移（小时数，float）。"""
    if not isinstance(tz_name, str) or tz_name.strip() == '':
        return None
    tz_name = tz_name.strip()
    try:
        if ZONEINFO_AVAILABLE:
            aware = datetime(ref_dt.year, ref_dt.month, ref_dt.day, tzinfo=ZoneInfo(tz_name))
            offset = aware.utcoffset()
        else:
            tz = pytz.timezone(tz_name)
            aware = tz.localize(datetime(ref_dt.year, ref_dt.month, ref_dt.day))
            offset = aware.utcoffset()
        if offset is None:
            return None
        return offset.total_seconds() / 3600.0
    except Exception:
        return None

def compute_timezone_diffs(df):
    """为 DataFrame 添加时区差值列。"""
    df['Athlete_Timezone'] = df['Nation'].map(nation_timezone_map).fillna('')
    df['Event_Year'] = df['competition'].apply(lambda x: extract_year_from_text(x) or FALLBACK_YEAR)
    df['Reference_Date'] = df['Event_Year'].apply(lambda y: datetime(int(y), 7, 1))

    athlete_offsets, venue_offsets, tz_diffs = [], [], []
    for _, row in df.iterrows():
        ref_dt = row['Reference_Date']
        venue_tz = row.get('Timezone', '')
        athlete_tz = row.get('Athlete_Timezone', '')
        venue_off = get_offset_hours(venue_tz, ref_dt)
        athlete_off = get_offset_hours(athlete_tz, ref_dt)
        diff = athlete_off - venue_off if athlete_off is not None and venue_off is not None else None
        athlete_offsets.append(athlete_off)
        venue_offsets.append(venue_off)
        tz_diffs.append(diff)

    df['Athlete_UTC_offset'] = athlete_offsets
    df['Venue_UTC_offset'] = venue_offsets
    df['Timezone_Diff_hours'] = tz_diffs
    return df

def preprocess_and_combine_data(data_path, output_path):
    """遍历并合并 Excel 文件，计算时区差值。"""
    if not os.path.isdir(data_path):
        print(f"错误：数据目录 '{data_path}' 不存在。")
        return

    excel_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    if not excel_files:
        print(f"警告：目录 '{data_path}' 中未找到任何 .xlsx 文件。")
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

            cols_to_fill = ['Venue', 'City', 'Timezone', 'Event']
            for col in cols_to_fill:
                if col in df.columns:
                    df[col] = df[col].ffill()

            df['competition'] = sheet_name
            all_sheets.append(df)

    if not all_sheets:
        print("未收集到有效数据，结束。")
        return

    combined_df = pd.concat(all_sheets, ignore_index=True)
    combined_df.dropna(how='all', inplace=True)
    print(f"合并后总行数: {len(combined_df)}")

    if 'Nation' not in combined_df.columns:
        print("警告：缺少 'Nation' 列，无法继续。")
        return
    combined_df['Nation'] = combined_df['Nation'].astype(str).str.strip().str.upper()

    pre_filter_rows = len(combined_df)
    combined_df = combined_df[combined_df['Nation'].isin(TARGET_NATIONS)].copy()
    post_filter_rows = len(combined_df)
    print(f"按国家筛选后: {pre_filter_rows} -> {post_filter_rows} 行")

    combined_df = compute_timezone_diffs(combined_df)

    try:
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 成功！输出已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存出错: {e}")

if __name__ == "__main__":
    preprocess_and_combine_data(data_dir, output_file)
