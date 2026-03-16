"""
目标（新实验数据预处理）：
1) 合并 check_data 目录下所有比赛 Excel（每个 sheet=一届比赛/一个 competition）。
2) 不再按 Top13/Top20 国家筛选；改为筛选“参赛次数 >= 2”的运动员（Athlete_ID = Name + "_" + Nation）。
3) 计算 Jet Lag（绝对时差），而不是保留方向（东/西）。
4) 不输出/不依赖 TravelDuration（后续模型中去掉该变量）。

输出：
  - iwf_data_with_absjetlag_athlete2plus.csv

说明（关于运动员所在时区）：
  - 代码会优先使用内置的 Nation->Timezone 映射（覆盖你们旧流程中使用过的国家）。
  - 也支持通过同目录下的可选文件 nation_timezone_capital_overrides.csv 扩展/覆盖映射：
      列名要求：Nation, Timezone
    其中 Nation 为三字国家代码（IWF/NOC 常见格式），Timezone 为 IANA 时区名（如 Asia/Shanghai）。
  - 对无法映射的 Nation，会回退到 Etc/UTC（等价于“默认首都时区未知”），并在控制台打印统计信息。
"""

from __future__ import annotations

import os
import glob
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
except Exception:
    import pytz
    ZONEINFO_AVAILABLE = False


# ----------------- 配置 -----------------
PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "check_data")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "iwf_data_with_absjetlag_athlete2plus.csv")

# 可选外置映射文件：用于“首都时区”或其它你想指定的时区
OVERRIDE_TZ_MAP_CSV = os.path.join(PROJECT_DIR, "nation_timezone_capital_overrides.csv")

FALLBACK_YEAR = 2015

# 旧流程中用到的 Top13（训练地时区），保留为优先映射
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

# 你们在 travel 数据收集里出现过的额外国家（给一个“首都/主要城市时区”的合理默认值）
# 注意：这不是“运动员训练地”，而是可接受的近似（首都/主要城市时区）。
EXTRA_NATION_TIMEZONE_MAP: Dict[str, str] = {
    "BUL": "Europe/Sofia",
    "CAN": "America/Toronto",   # Montreal 位于该时区
    "ESP": "Europe/Madrid",
    "FRA": "Europe/Paris",
    "POL": "Europe/Warsaw",
    "TUR": "Europe/Istanbul",
    "VEN": "America/Caracas",
    "BLR": "Europe/Minsk",
    "ECU": "America/Guayaquil",  # Quito/Guayaquil 同一主时区（不含加拉帕戈斯）
    "EGY": "Africa/Cairo",
    "GEO": "Asia/Tbilisi",
    "INA": "Asia/Jakarta",
    "PRK": "Asia/Pyongyang",
    "UZB": "Asia/Tashkent",
    # 赛事地国家在 competition_map 中出现过的
    "FIN": "Europe/Helsinki",
    "GRE": "Europe/Athens",
    "QAT": "Asia/Qatar",
    "DOM": "America/Santo_Domingo",
    "TKM": "Asia/Ashgabat",
    "KSA": "Asia/Riyadh",
    "BRN": "Asia/Bahrain",
    "NOR": "Europe/Oslo",
}


# ----------------- 届数→年份映射（旧逻辑复用） -----------------
EDITION_TO_YEAR_MAP = {
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


def extract_year_from_text(s: object) -> Optional[int]:
    """优先提取年份；若无则尝试根据届数映射。"""
    if not isinstance(s, str):
        return None

    # ① 尝试直接匹配四位数年份（19xx/20xx）
    m = re.search(r"(?<!\d)(19|20)\d{2}", s)
    if m:
        return int(m.group(0))

    # ② 若无年份，则尝试匹配届数（如 69th、70TH、78Th）
    m2 = re.search(r"(\d{2,3})\s*(?:st|nd|rd|th)", s, flags=re.IGNORECASE)
    if m2:
        edition = int(m2.group(1))
        return EDITION_TO_YEAR_MAP.get(edition)

    return None


def load_override_timezone_map(path: str) -> Dict[str, str]:
    """
    读取外置 Nation->Timezone 映射（用于首都/自定义时区）。
    CSV 列：Nation, Timezone
    """
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)

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
    """返回指定时区在 ref_dt 日期的 UTC 偏移（小时数，float）。"""
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
    """
    构建 Nation->Timezone 映射：
      1) BASE（旧Top13训练地）
      2) EXTRA（常见国家的首都/主要城市默认时区）
      3) 可选 overrides（外置 CSV），优先级最高
    返回：
      merged_map, overrides_map
    """
    overrides = load_override_timezone_map(OVERRIDE_TZ_MAP_CSV)

    merged = {}
    merged.update(EXTRA_NATION_TIMEZONE_MAP)
    merged.update(BASE_NATION_TIMEZONE_MAP)  # 训练地优先于“首都默认”
    merged.update(overrides)  # 用户覆盖优先级最高
    return merged, overrides


def compute_abs_jetlag(df: pd.DataFrame, nation_tz_map: Dict[str, str]) -> pd.DataFrame:
    """
    给 df 增加：
      - Athlete_Timezone（Nation->timezone）
      - Event_Year、Reference_Date
      - Athlete_UTC_offset、Venue_UTC_offset
      - Timezone_Diff_hours（绝对值 JetLag）
      - Signed_Timezone_Diff_hours（带方向，保留备查）
    """
    df["Athlete_Timezone"] = df["Nation"].map(nation_tz_map).fillna("")
    df["Event_Year"] = df["competition"].apply(lambda x: extract_year_from_text(x) or FALLBACK_YEAR)
    df["Reference_Date"] = df["Event_Year"].apply(lambda y: datetime(int(y), 7, 1))

    athlete_offsets, venue_offsets, signed_diffs, abs_diffs = [], [], [], []
    for _, row in df.iterrows():
        ref_dt = row["Reference_Date"]
        venue_tz = row.get("Timezone", "")
        athlete_tz = row.get("Athlete_Timezone", "")

        venue_off = get_offset_hours(str(venue_tz), ref_dt)
        athlete_off = get_offset_hours(str(athlete_tz), ref_dt)

        signed = athlete_off - venue_off if athlete_off is not None and venue_off is not None else None
        absd = abs(signed) if signed is not None else None

        athlete_offsets.append(athlete_off)
        venue_offsets.append(venue_off)
        signed_diffs.append(signed)
        abs_diffs.append(absd)

    df["Athlete_UTC_offset"] = athlete_offsets
    df["Venue_UTC_offset"] = venue_offsets
    df["Signed_Timezone_Diff_hours"] = signed_diffs
    df["Timezone_Diff_hours"] = abs_diffs  # 与旧列名对齐：但语义变为“绝对时差”
    return df


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

            # 处理合并单元格导致的缺失（旧逻辑复用）
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

    # Nation/Name 是“参赛次数筛选”的关键字段
    for required in ["Nation", "Name"]:
        if required not in combined_df.columns:
            raise ValueError(f"缺少必要列 '{required}'。请检查 check_data 的 Excel 格式。")

    combined_df["Nation"] = combined_df["Nation"].astype(str).str.strip().str.upper()
    combined_df["Name"] = combined_df["Name"].astype(str).str.strip()

    # Nation 时区映射（训练地优先；否则首都/主要城市默认；否则可用外置 CSV 覆盖）
    nation_tz_map, overrides = build_nation_timezone_map()
    print(f"Nation->Timezone 映射已加载：内置={len(nation_tz_map) - len(overrides)}，覆盖文件={len(overrides)}")

    # 计算绝对时差 JetLag（仍保留 Signed 值备查）
    combined_df = compute_abs_jetlag(combined_df, nation_tz_map)

    # 构造 Athlete_ID，并按参赛次数筛选（>=2）
    combined_df["Athlete_ID"] = combined_df["Name"].astype(str) + "_" + combined_df["Nation"].astype(str)
    counts = combined_df["Athlete_ID"].value_counts(dropna=False)
    eligible_ids = counts[counts >= 2].index
    pre_filter = len(combined_df)
    combined_df = combined_df[combined_df["Athlete_ID"].isin(eligible_ids)].copy()
    print(f"按参赛次数筛选（Athlete_ID>=2）: {pre_filter} -> {len(combined_df)} 行")

    # 统计无法映射时区导致 JetLag 缺失的情况（便于后续补齐“首都时区表”）
    missing_tz_nations = combined_df.loc[combined_df["Athlete_Timezone"].eq(""), "Nation"].value_counts().head(20)
    missing_jetlag = combined_df["Timezone_Diff_hours"].isna().sum()
    if len(missing_tz_nations) > 0:
        print("\n⚠️ 以下 Nation 未映射到运动员时区（当前 Athlete_Timezone 为空，已回退为 UTC offset=None 导致 JetLag 缺失）：")
        print(missing_tz_nations.to_string())
        print(f"\n⚠️ JetLag（Timezone_Diff_hours）缺失行数: {missing_jetlag}")
        print(f"你可以在 {os.path.basename(OVERRIDE_TZ_MAP_CSV)} 里补齐 Nation->Timezone 映射。")

    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 输出已保存: {output_path}")


if __name__ == "__main__":
    preprocess_and_combine_data(DATA_DIR, OUTPUT_FILE)

