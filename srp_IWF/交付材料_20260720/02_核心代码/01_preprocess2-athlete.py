"""
preprocess2-athlete.py（修正版）

修正内容（相对原版）：
  1. 时差计算改用"圆周最短路径"（circular time difference），
     解决日期变更线附近的不合理大值问题（如 UTC+12→UTC-12 不再算出24h）。
  2. 方向定义统一为：
       d = Venue_UTC_offset - Athlete_UTC_offset
       正值 = 比赛地在运动员东边 → 运动员向东飞（Eastward）
       负值 = 比赛地在运动员西边 → 运动员向西飞（Westward）
     ★ 注意：这与原版 preprocess（athlete_off - venue_off）方向相反，
       但与 LMM_jetlag.py 中的 circular_jetlag_signed() 保持一致。
  3. 输出列名更清晰：
       - JetLag_signed：圆周最短时差，范围 [-12, 12]
       - JetLag_abs：|JetLag_signed|，范围 [0, 12]
       - 保留 Athlete_UTC_offset / Venue_UTC_offset 供下游脚本直接使用
  4. 新增运动员参赛次数分布统计，输出 athlete_participation_stats.txt
输入：check_data文件夹下的所有xlsx文件，nation_timezone_capital_overrides.csv
输出：iwf_data_top13_athlete2plus_circular.csv
"""

from __future__ import annotations

import os
import glob
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
except Exception:
    import pytz
    ZONEINFO_AVAILABLE = False


# ======================== 配置 ========================
PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "check_data")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "iwf_data_top13_athlete2plus_circular.csv")
STATS_FILE = os.path.join(PROJECT_DIR, "athlete_participation_stats.txt")

OVERRIDE_TZ_MAP_CSV = os.path.join(PROJECT_DIR, "nation_timezone_capital_overrides.csv")

FALLBACK_YEAR = 2015

# 旧研究口径：Top13 国家
TARGET_NATIONS = [
    "ARM", "CHN", "COL", "IRI", "JPN", "KAZ", "KOR",
    "MEX", "RUS", "THA", "TPE", "UKR", "USA",
]

# ======================== 国家→时区映射 ========================
# 旧流程 Top13（训练地时区，人工确认过的，优先级高）
BASE_NATION_TIMEZONE_MAP: Dict[str, str] = {
    "ARM": "Asia/Yerevan",
    "CHN": "Asia/Shanghai",
    "COL": "America/Bogota",
    "IRI": "Asia/Tehran",
    "JPN": "Asia/Tokyo",
    "KAZ": "Asia/Almaty",
    "KOR": "Asia/Seoul",
    "MEX": "America/Mexico_City",
    "RUS": "Europe/Moscow",
    "THA": "Asia/Bangkok",
    "TPE": "Asia/Taipei",
    "UKR": "Europe/Kyiv",
    "USA": "America/Denver",
}

# 额外国家的首都/主要城市时区（合理近似）
EXTRA_NATION_TIMEZONE_MAP: Dict[str, str] = {
    "BUL": "Europe/Sofia",
    "CAN": "America/Toronto",
    "ESP": "Europe/Madrid",
    "FRA": "Europe/Paris",
    "POL": "Europe/Warsaw",
    "TUR": "Europe/Istanbul",
    "VEN": "America/Caracas",
    "BLR": "Europe/Minsk",
    "ECU": "America/Guayaquil",
    "EGY": "Africa/Cairo",
    "GEO": "Asia/Tbilisi",
    "INA": "Asia/Jakarta",
    "PRK": "Asia/Pyongyang",
    "UZB": "Asia/Tashkent",
    "FIN": "Europe/Helsinki",
    "GRE": "Europe/Athens",
    "QAT": "Asia/Qatar",
    "DOM": "America/Santo_Domingo",
    "TKM": "Asia/Ashgabat",
    "KSA": "Asia/Riyadh",
    "BRN": "Asia/Bahrain",
    "NOR": "Europe/Oslo",
}


# ======================== 届数→年份映射 ========================
EDITION_TO_YEAR_MAP = {
    69: 1998, 70: 1999, 71: 2001, 72: 2002, 73: 2003,
    74: 2005, 75: 2006, 76: 2007, 77: 2009, 78: 2010,
}


def extract_year_from_text(s: object) -> Optional[int]:
    """优先提取年份；若无则尝试根据届数映射。"""
    if not isinstance(s, str):
        return None
    m = re.search(r"(?<!\d)(19|20)\d{2}", s)
    if m:
        return int(m.group(0))
    m2 = re.search(r"(\d{2,3})\s*(?:st|nd|rd|th)", s, flags=re.IGNORECASE)
    if m2:
        edition = int(m2.group(1))
        return EDITION_TO_YEAR_MAP.get(edition)
    return None


def load_override_timezone_map(path: str) -> Dict[str, str]:
    """读取外置 Nation->Timezone 映射 CSV。"""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=["Nation", "Timezone"], engine="python")
    except Exception:
        df = pd.read_csv(path, usecols=["Nation", "Timezone"], engine="python")
    df.columns = df.columns.str.strip()
    if not {"Nation", "Timezone"}.issubset(set(df.columns)):
        raise ValueError(f"覆盖映射文件缺少列：Nation/Timezone。实际列={list(df.columns)}")
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        n = str(row["Nation"]).strip().upper()
        tz = str(row["Timezone"]).strip()
        if n and tz and tz.lower() != "nan":
            out[n] = tz
    return out


def get_offset_hours(tz_name: str, ref_dt: datetime) -> Optional[float]:
    """返回指定时区在 ref_dt 日期的 UTC 偏移（小时数）。"""
    if not isinstance(tz_name, str) or tz_name.strip() == "":
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


def build_nation_timezone_map() -> Tuple[Dict[str, str], Dict[str, str]]:
    """构建 Nation->Timezone 映射（BASE > EXTRA > overrides优先级递增）。"""
    overrides = load_override_timezone_map(OVERRIDE_TZ_MAP_CSV)
    merged = {}
    merged.update(EXTRA_NATION_TIMEZONE_MAP)
    merged.update(BASE_NATION_TIMEZONE_MAP)  # 训练地优先于"首都默认"
    merged.update(overrides)                 # 用户覆盖优先级最高
    return merged, overrides


def circular_jetlag_signed(venue_off: float, athlete_off: float) -> Optional[float]:
    """
    ★ 核心修正：圆周最短路径时差（circular time difference）

    计算步骤：
      1) d = venue_off - athlete_off（行政原始时差，可能在 [-24, 24]）
      2) 若 |d| <= 12：直接用 d（在半圈以内，不需要绕远）
      3) 若 |d| > 12：绕另一个方向更近，jetlag = -sign(d) * (24 - |d|)

    返回范围：[-12, 12]
      正值 → 运动员向东飞（比赛地时间更晚/更"靠前"）
      负值 → 运动员向西飞

    示例：
      UTC+8(CHN) → UTC+2(欧洲): d = 2-8 = -6  → jetlag = -6 (向西6h) ✓
      UTC+8(CHN) → UTC-8(美西): d = -8-8 = -16 → |d|>12, jetlag = +1*（24-16）= +8 (向东8h) ✓
      UTC+12 → UTC-12:          d = -12-12 = -24 → jetlag = +1*(24-24) = 0 ✓
    """
    if venue_off is None or athlete_off is None:
        return None
    if pd.isna(venue_off) or pd.isna(athlete_off):
        return None
    d = float(venue_off) - float(athlete_off)
    if abs(d) <= 12:
        return d
    return -np.sign(d) * (24 - abs(d))


def compute_circular_jetlag(df: pd.DataFrame, nation_tz_map: Dict[str, str]) -> pd.DataFrame:
    """
    ★ 修正后的时差计算函数（替代原版 compute_abs_jetlag）

    给 df 增加：
      - Athlete_Timezone
      - Event_Year, Reference_Date
      - Athlete_UTC_offset, Venue_UTC_offset
      - JetLag_signed：圆周最短时差，[-12, 12]
      - JetLag_abs：|JetLag_signed|，[0, 12]

    与原版的区别：
      1. 使用 circular_jetlag_signed() 而非简单线性相减
      2. 方向定义为 venue - athlete（与 LMM 脚本一致）
      3. 列名改为 JetLag_signed / JetLag_abs（语义更清晰）
    """
    df["Athlete_Timezone"] = df["Nation"].map(nation_tz_map).fillna("Etc/UTC")
    df["Event_Year"] = df["competition"].apply(lambda x: extract_year_from_text(x) or FALLBACK_YEAR)
    df["Reference_Date"] = df["Event_Year"].apply(lambda y: datetime(int(y), 7, 1))

    athlete_offsets = []
    venue_offsets = []
    jetlag_signed_list = []
    jetlag_abs_list = []

    for _, row in df.iterrows():
        ref_dt = row["Reference_Date"]
        venue_tz = row.get("Timezone", "")
        athlete_tz = row.get("Athlete_Timezone", "")

        venue_off = get_offset_hours(str(venue_tz), ref_dt)
        athlete_off = get_offset_hours(str(athlete_tz), ref_dt)

        # ★ 使用圆周最短路径计算
        jl_signed = circular_jetlag_signed(venue_off, athlete_off)
        jl_abs = abs(jl_signed) if jl_signed is not None else None

        athlete_offsets.append(athlete_off)
        venue_offsets.append(venue_off)
        jetlag_signed_list.append(jl_signed)
        jetlag_abs_list.append(jl_abs)

    df["Athlete_UTC_offset"] = athlete_offsets
    df["Venue_UTC_offset"] = venue_offsets
    df["JetLag_signed"] = jetlag_signed_list    # [-12, 12]，正=东飞，负=西飞
    df["JetLag_abs"] = jetlag_abs_list          # [0, 12]

    # ★ 同时保留旧列名以兼容下游（但语义已改为圆周时差）
    df["Timezone_Diff_hours"] = jetlag_abs_list
    df["Signed_Timezone_Diff_hours"] = jetlag_signed_list

    return df


def write_participation_stats(df: pd.DataFrame, output_path: str) -> None:
    """
    ★ 新增功能：输出运动员参赛次数分布统计

    输出内容：
      - 每个参赛次数有多少运动员（如：参赛3次的运动员有120个）
      - 总运动员数、总记录数
      - 参赛次数的描述统计
    """
    counts = df["Athlete_ID"].value_counts()  # 每个运动员的参赛次数
    freq_dist = counts.value_counts().sort_index()  # 每个次数有多少运动员

    lines = []
    lines.append("=" * 70)
    lines.append("运动员参赛次数分布统计 (Athlete Participation Frequency)")
    lines.append(f"统计时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"总记录数 (rows):     {len(df)}")
    lines.append(f"总运动员数 (unique):  {counts.shape[0]}")
    lines.append(f"筛选条件:            参赛次数 >= 2")
    lines.append("")
    lines.append("-" * 50)
    lines.append(f"{'参赛次数':>10}  {'运动员人数':>12}  {'占比(%)':>10}")
    lines.append("-" * 50)

    total_athletes = counts.shape[0]
    for n_times, n_athletes in freq_dist.items():
        pct = n_athletes / total_athletes * 100
        lines.append(f"{n_times:>10}  {n_athletes:>12}  {pct:>9.1f}%")

    lines.append("-" * 50)
    lines.append(f"{'合计':>10}  {total_athletes:>12}  {'100.0%':>10}")
    lines.append("")
    lines.append("描述统计 (参赛次数):")
    lines.append(f"  均值:   {counts.mean():.2f}")
    lines.append(f"  中位数: {counts.median():.1f}")
    lines.append(f"  最小值: {counts.min()}")
    lines.append(f"  最大值: {counts.max()}")
    lines.append(f"  标准差: {counts.std():.2f}")
    lines.append("")

    # 按区间汇总
    lines.append("按参赛次数区间汇总:")
    bins = [(2, 2), (3, 3), (4, 5), (6, 10), (11, 99)]
    labels = ["恰好2次", "恰好3次", "4-5次", "6-10次", "11次及以上"]
    for (lo, hi), label in zip(bins, labels):
        n = ((counts >= lo) & (counts <= hi)).sum()
        pct = n / total_athletes * 100
        lines.append(f"  {label:>12}: {n:>6} 名运动员 ({pct:.1f}%)")

    lines.append("=" * 70)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"运动员参赛次数统计已保存: {output_path}")


def preprocess_and_combine_data(data_path: str, output_path: str) -> None:
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"数据目录不存在：{data_path}")

    excel_files = glob.glob(os.path.join(data_path, "*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"目录中未找到任何 .xlsx：{data_path}")

    print(f"发现 {len(excel_files)} 个 Excel 文件，开始读取并合并…")

    all_sheets = []
    for file_path in excel_files:
        filename = os.path.basename(file_path)
        print(f"读取文件: {filename}")
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            print(f"  读取失败，跳过 {filename}：{e}")
            continue

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            df.columns = df.columns.str.strip()
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.dropna(how="all")
            cols_to_fill = ["Venue", "City", "Timezone", "Event"]
            for col in cols_to_fill:
                if col in df.columns:
                    df[col] = df[col].ffill()
            df["competition"] = sheet_name
            all_sheets.append(df)

    if not all_sheets:
        raise RuntimeError("未收集到任何有效 sheet 数据。")

    combined_df = pd.concat(all_sheets, ignore_index=True)
    combined_df.dropna(how="all", inplace=True)
    print(f"合并后总行数: {len(combined_df)}")

    for required in ["Nation", "Name"]:
        if required not in combined_df.columns:
            raise ValueError(f"缺少必要列 '{required}'。请检查 check_data 的 Excel 格式。")

    combined_df["Nation"] = combined_df["Nation"].astype(str).str.strip().str.upper()
    combined_df["Name"] = combined_df["Name"].astype(str).str.strip()

    # 先按 Top13 国家筛选，再做后续时差与运动员频次筛选（降噪并对齐旧研究口径）
    pre_top13 = len(combined_df)
    combined_df = combined_df[combined_df["Nation"].isin(TARGET_NATIONS)].copy()
    print(f"按 Top13 国家筛选: {pre_top13} -> {len(combined_df)} 行")

    # 构建时区映射
    nation_tz_map, overrides = build_nation_timezone_map()
    print(f"Nation->Timezone 映射已加载：内置={len(nation_tz_map) - len(overrides)}，覆盖文件={len(overrides)}")

    # ★ 修正：使用圆周最短路径计算时差
    combined_df = compute_circular_jetlag(combined_df, nation_tz_map)

    # 构造 Athlete_ID，按参赛次数筛选（>=2）
    combined_df["Athlete_ID"] = combined_df["Name"].astype(str) + "_" + combined_df["Nation"].astype(str)
    counts = combined_df["Athlete_ID"].value_counts(dropna=False)
    eligible_ids = counts[counts >= 2].index
    pre_filter = len(combined_df)
    combined_df = combined_df[combined_df["Athlete_ID"].isin(eligible_ids)].copy()
    print(f"按参赛次数筛选（Athlete_ID>=2）: {pre_filter} -> {len(combined_df)} 行")

    # ======================== 剔除映射缺失的数据 ========================
    # 两类问题行：
    #   1) Athlete_Timezone 回退到 Etc/UTC（运动员所在国家没有时区映射）
    #   2) JetLag_signed 为 NA（比赛地 Timezone 缺失或无法解析）
    # 这些行的时差不可信，直接从主表移除，单独导出供排查。

    DROPPED_FILE = os.path.join(PROJECT_DIR, "dropped_rows_tz_missing.csv")

    mask_utc_fallback = combined_df["Athlete_Timezone"].eq("Etc/UTC")
    mask_jetlag_na    = combined_df["JetLag_signed"].isna()
    mask_drop = mask_utc_fallback | mask_jetlag_na

    dropped_df = combined_df[mask_drop].copy()
    n_total = len(combined_df)
    n_drop  = len(dropped_df)

    if n_drop > 0:
        # 给被剔除的行标注原因，方便排查
        reasons = []
        for _, row in dropped_df.iterrows():
            r = []
            if row["Athlete_Timezone"] == "Etc/UTC":
                r.append("Nation无时区映射(回退UTC)")
            if pd.isna(row["JetLag_signed"]):
                r.append("比赛地Timezone缺失/无效")
            reasons.append("; ".join(r))
        dropped_df["Drop_Reason"] = reasons
        dropped_df.to_csv(DROPPED_FILE, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"剔除映射缺失数据: {n_total} -> {n_total - n_drop} 行（去掉 {n_drop} 行）")
        print(f"{'='*60}")

        # 按原因分类统计
        n_utc = mask_utc_fallback.sum()
        n_jna = mask_jetlag_na.sum()
        n_both = (mask_utc_fallback & mask_jetlag_na).sum()
        print(f"  仅 Nation 无映射 (Etc/UTC):     {n_utc - n_both} 行")
        print(f"  仅比赛地 Timezone 缺失:          {n_jna - n_both} 行")
        print(f"  两者兼有:                        {n_both} 行")
        print(f"  合计去掉:                        {n_drop} 行")

        # 被剔除的 Nation 分布
        print(f"\n被剔除行的 Nation 分布（Top 20）:")
        print(dropped_df["Nation"].value_counts().head(20).to_string())

        print(f"\n被剔除行已保存: {DROPPED_FILE}")
    else:
        print("\n无映射缺失数据，无需剔除。")

    # 从主表中删除
    combined_df = combined_df[~mask_drop].copy()

    # ★ 剔除后有些运动员可能从 >=2 次降为 1 次，需重新过滤
    counts_after = combined_df["Athlete_ID"].value_counts(dropna=False)
    still_eligible = counts_after[counts_after >= 2].index
    n_before_recheck = len(combined_df)
    combined_df = combined_df[combined_df["Athlete_ID"].isin(still_eligible)].copy()
    n_recheck_drop = n_before_recheck - len(combined_df)
    if n_recheck_drop > 0:
        print(f"\n剔除后重新检查参赛次数>=2: 再去掉 {n_recheck_drop} 行（运动员参赛次数降至1次）")
    print(f"最终保留: {len(combined_df)} 行, {combined_df['Athlete_ID'].nunique()} 名运动员")

    # ★ 新增：输出参赛次数分布统计（基于剔除后的干净数据）
    write_participation_stats(combined_df, STATS_FILE)

    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n最终输出: {output_path}")
    print(f"最终行数: {len(combined_df)}, 运动员数: {combined_df['Athlete_ID'].nunique()}")


if __name__ == "__main__":
    preprocess_and_combine_data(DATA_DIR, OUTPUT_FILE)