# 用于整合组员搜集的各年份比赛的交通时间数据
import os
import pandas as pd

INPUT_DIR = "travel_time"
OUTPUT_FILE = "travel_time_merged.csv"

# =============================
# 1. 类型映射
# =============================
type_map = {
    "直飞": 0,
    "一次中转": 1,
    "二次中转": 2,
    "三次中转": 3
}

# =============================
# 2. 训练城市 → 国家代码
# =============================
training_map = {
    "Yerevan": "ARM", "Beijing": "CHN", "Bogota": "COL", "Tehran": "IRI",
    "Tokyo": "JPN", "Almaty": "KAZ", "Seoul": "KOR", "San Luis Potosi": "MEX",
    "Sochi": "RUS", "Chiang Mai": "THA", "Kaohsiung": "TPE", "Kyiv": "UKR",
    "Colorado Springs": "USA", "Sofia": "BUL", "Montreal": "CAN", "Madrid": "ESP",
    "Paris": "FRA", "Warsaw": "POL", "Ankara": "TUR", "Caracas": "VEN",
    "Minsk": "BLR", "Quito": "ECU", "Cairo": "EGY", "Tbilisi": "GEO",
    "Jakarta": "INA", "Pyongyang": "PRK", "Tashkent": "UZB"
}

# =============================
# 3. 比赛年份 → (城市, 国家)
# =============================
competition_map = {
    1998: ("Lathi", "FIN"), 1999: ("Athens", "GRE"), 2001: ("Antalya", "TUR"),
    2002: ("Warsaw", "POL"), 2003: ("Vancouver", "CAN"), 2005: ("Doha", "QAT"),
    2006: ("Santo Domingo", "DOM"), 2007: ("Chiang Mai", "THA"),
    2009: ("Goyang", "KOR"), 2010: ("Antalya", "TUR"), 2011: ("Paris", "FRA"),
    2013: ("Wroclaw", "POL"), 2014: ("Almaty", "KAZ"), 2015: ("Houston", "USA"),
    2017: ("Anaheim", "USA"), 2018: ("Ashgabat", "TKM"), 2019: ("PATTAYA", "THA"),
    2021: ("Tashkent", "UZB"), 2022: ("Bogota", "COL"), 2023: ("Riyadh", "KSA"),
    2024: ("Manama", "BRN"), 2025: ("Forde", "NOR")
}

# reverse mapping: City → Country_Code
city_to_nation = {city: ccode for _, (city, ccode) in competition_map.items()}

# =============================
# 4. 读取并合并全部文件
# =============================
year_sheets = ["2019", "2021", "2022", "2023", "2024", "2025"]
all_rows = []

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".xlsx"):
        continue

    file_path = os.path.join(INPUT_DIR, fname)
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names

    dfs_in_file = []

    if "Collect" in sheets:
        df = pd.read_excel(file_path, sheet_name="Collect")
        dfs_in_file.append(df)
    else:
        for y in year_sheets:
            if y in sheets:
                df = pd.read_excel(file_path, sheet_name=y)
                dfs_in_file.append(df)

    if len(dfs_in_file) == 0:
        continue

    df = pd.concat(dfs_in_file, ignore_index=True)

    # 清洗空行
    df = df.dropna(how="all")
    df = df.dropna(subset=["Departure_City", "Arrival_City", "Duration_Hours(h)", "Types", "Year"], how="any")

    # 去掉 Notes
    if "Notes" in df.columns:
        df = df.drop(columns=["Notes"])

    # 统一中转类型
    df["Types"] = df["Types"].map(type_map)

    # --------------------------
    # 5. 新增字段：Departure_Nation
    # --------------------------
    df["Departure_Nation"] = df["Departure_City"].map(training_map)

    # --------------------------
    # 6. Arrival_City = “城市,国家”
    # --------------------------
    def map_arrival(row):
        city = row["Arrival_City"]
        year = int(row["Year"])
        # 有的年份是写成 2,006 这种 → 去逗号
        y = int(str(year).replace(",", ""))
        if y in competition_map:
            comp_city, comp_country = competition_map[y]
            if city == comp_city:
                return f"{city},{comp_country}"
        # fallback: 用 city_to_nation
        if city in city_to_nation:
            return f"{city},{city_to_nation[city]}"
        return city  # 如果没有匹配到

    df["Arrival_City"] = df.apply(map_arrival, axis=1)
    df["Year"] = df["Year"].astype(int)

    df = df[[
        "Departure_Nation",
        "Departure_City",
        "Arrival_City",
        "Duration_Hours(h)",
        "Types",
        "Year",
    ]]

    all_rows.append(df)

merged = pd.concat(all_rows, ignore_index=True)
merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("✔ travel_time_merged.csv 生成完成")
print("总行数：", len(merged))
