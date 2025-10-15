import pandas as pd
import pycountry
import pytz
from timezonefinder import TimezoneFinder

# ===== 基础设置 =====
input_path = r"D:\Code\Python\Creeper\iwf_results.xlsx"
output_path = r"D:\Code\Python\Creeper\iwf_timezone.xlsx"

# 比赛地点手动映射表（可扩展）
venue_timezones = {
    "FORDE, NOR": "Europe/Oslo",
    "PARIS, FRA": "Europe/Paris",
    "RIYADH, KSA": "Asia/Riyadh",
    "DOHA, QAT": "Asia/Qatar",
    "FUZHOU, CHN": "Asia/Shanghai", 
}

tf = TimezoneFinder()

# ===== 读取数据 =====
df = pd.read_excel(input_path)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # 删除多余空列

# ===== 核心逻辑修改：有条件地填充 Venue 列 =====

# 步骤 1: 识别哪些是空行。我们以 'Name' 列是否为空作为判断依据。
# is_data_row 会是一个布尔序列 (True/False)，True 代表该行有数据。
is_data_row = df['Name'].notna()

# 步骤 2: 只对数据行进行前向填充。
# 我们使用 .loc[is_data_row] 来选择所有数据行，然后对这些行的 'Venue' 列进行填充。
# 这样，空行（is_data_row 为 False 的行）的 'Venue' 列将保持为 NaN，不会被填充。
df.loc[is_data_row, 'Venue'] = df.loc[is_data_row, 'Venue'].ffill()


# 检查关键列
if not {"Nation", "Venue"}.issubset(df.columns):
    raise ValueError("Excel 中必须包含列：'Nation' 和 'Venue'")

# ===== 国家 -> 时区映射 (代码不变) =====
def get_country_timezone(alpha3):
    try:
        country = pycountry.countries.get(alpha_3=alpha3)
        if not country:
            return None, None
        tz = pytz.country_timezones.get(country.alpha_2)
        if tz:
            timezone = tz[0]
            offset = pd.Timestamp.now(tz=pytz.timezone(timezone)).strftime("%z")
            return timezone, f"{int(offset[:3]):+03d}:{offset[3:]}"
    except Exception:
        return None, None
    return None, None

df[["Timezone", "UTC_Offset"]] = df["Nation"].apply(
    lambda x: pd.Series(get_country_timezone(str(x).strip().upper()))
)

# ===== 比赛地点 -> 时区映射 (代码不变) =====
def get_venue_timezone(venue):
    if not isinstance(venue, str) or not venue.strip():
        return None, None
    venue_upper = venue.strip().upper()
    tz = None
    for key, value in venue_timezones.items():
        if key.upper() == venue_upper:
            tz = value
            break
    if tz:
        offset = pd.Timestamp.now(tz=pytz.timezone(tz)).strftime("%z")
        return tz, f"{int(offset[:3]):+03d}:{offset[3:]}"
    return None, None

df[["Venue_Timezone", "Venue_UTC_Offset"]] = df["Venue"].apply(
    lambda x: pd.Series(get_venue_timezone(str(x).strip()))
)

# ===== 计算时差差值（单位：小时） (代码不变) =====
def offset_to_hours(offset):
    if pd.isna(offset):
        return None
    try:
        sign = 1 if offset.startswith("+") else -1
        hours, minutes = map(int, offset[1:].split(":"))
        return sign * (hours + minutes / 60)
    except Exception:
        return None

df["UTC_Hours"] = df["UTC_Offset"].apply(offset_to_hours)
df["Venue_UTC_Hours"] = df["Venue_UTC_Offset"].apply(offset_to_hours)
df["Time_Difference_Hours"] = df["UTC_Hours"] - df["Venue_UTC_Hours"]

# ===== 输出整洁表格 (代码不变) =====
columns_order = [
    "Event", "Rank", "Name", "Nation", "Born", "Body_Weight", "Group",
    "Snatch", "Clean_Jerk", "Total", "Venue",
    "Timezone", "UTC_Offset", "Venue_Timezone", "Venue_UTC_Offset",
    "Time_Difference_Hours"
]
existing_columns = [col for col in columns_order if col in df.columns]
df = df.reindex(columns=existing_columns)


df.to_excel(output_path, index=False)
print(f"✅ 处理完成，已保存到：{output_path}")