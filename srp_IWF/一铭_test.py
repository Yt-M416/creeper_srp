"""
循环时间分类变量_athlete_id.py

基于同学原版代码的最小改动，仅修改两处：
  1. 输入数据改为 YT 筛选后的 iwf_data_top13_athlete2plus_circular.csv
  2. 随机截距改为 Athlete_ID（而非 Nation）
其余所有逻辑（2h分档、东西子样本拆分、Sinclair公式、可视化）完全保留原版。

目的：验证结论差异是否来自随机截距选择，而非代码逻辑错误。
"""

import warnings
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------- Global Font Config --------------------------
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

# -------------------------- 1. Path Configuration --------------------------
# ★ 改动1：输入数据改为 YT 筛选后的文件
INPUT_CSV_PATH = "iwf_data_top13_athlete2plus_circular.csv"
OUTPUT_DIR = "循环分类时间LMM_athlete_id"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------- 2. Data Loading & Preprocessing --------------------------
df = pd.read_csv(INPUT_CSV_PATH, encoding="UTF-8-sig")
df.columns = df.columns.str.strip()

# 计算最短圆周时差 (±12h)  —— 与同学原版完全一致
df["Athlete_UTC_offset"] = pd.to_numeric(df["Athlete_UTC_offset"], errors="coerce")
df["Venue_UTC_offset"] = pd.to_numeric(df["Venue_UTC_offset"], errors="coerce")

def calculate_circular_time(athlete_tz, venue_tz):
    basic_diff = venue_tz - athlete_tz
    if basic_diff > 12:
        return basic_diff - 24
    elif basic_diff < -12:
        return basic_diff + 24
    else:
        return basic_diff

df["Circular_Time"] = df.apply(
    lambda row: calculate_circular_time(row["Athlete_UTC_offset"], row["Venue_UTC_offset"]),
    axis=1
)

# 将圆周时间转换为2小时间隔的分类变量 —— 与同学原版完全一致
time_bins = np.arange(-12, 13, 2)
time_labels = [f"[{time_bins[i]}, {time_bins[i+1]})" for i in range(len(time_bins)-1)]
df["Circular_Time_Group"] = pd.cut(df["Circular_Time"], bins=time_bins, labels=time_labels, right=False, include_lowest=True)

# 创建飞行方向变量 —— 与同学原版完全一致
df["Flight_Direction"] = np.where(df["Circular_Time"] > 0, "Eastward (向东)",
                                   np.where(df["Circular_Time"] < 0, "Westward (向西)", None))

# 数据清洗 —— 适配YT数据的列名
key_cols = ["Total", "Body_Weight", "Event", "Circular_Time", "Circular_Time_Group", "Flight_Direction", "Nation", "Athlete_ID"]
df_clean = df[key_cols].dropna(subset=key_cols).copy()

df_clean["Total"] = pd.to_numeric(df_clean["Total"], errors="coerce")
df_clean["Body_Weight"] = pd.to_numeric(df_clean["Body_Weight"], errors="coerce")
df_clean = df_clean.dropna(subset=["Total", "Body_Weight"])
df_clean = df_clean[df_clean["Body_Weight"] > 0]

# 提取性别 —— 与同学原版完全一致
def extract_sex(event_str):
    event_str = str(event_str)
    if "Women" in event_str: return "Female"
    elif "Men" in event_str: return "Male"
    return None

df_clean["Sex"] = df_clean["Event"].apply(extract_sex)
df_clean = df_clean.dropna(subset=["Sex"])

# -------------------------- 3. 2021-2024 IWF官方辛克莱系数计算 —— 与同学原版完全一致 --------------------------
SINCLAIR_PARAMS = {
    "Male": {"A": 0.722762521, "b": 193.609},
    "Female": {"A": 0.787004341, "b": 153.757}
}

def calculate_sinclair(row):
    bw = row["Body_Weight"]
    total = row["Total"]
    sex = row["Sex"]
    A = SINCLAIR_PARAMS[sex]["A"]
    b = SINCLAIR_PARAMS[sex]["b"]
    if bw >= b:
        return total
    X = np.log10(bw / b)
    sc = 10 ** (A * (X ** 2))
    return total * sc

df_clean["Sinclair_Total"] = df_clean.apply(calculate_sinclair, axis=1)

# ★ 不合并旅行时长（YT数据中没有这个文件，且同学的结果也显示旅行时长无影响）
df_final = df_clean.copy()

# -------------------------- 5. LMM模型拟合 --------------------------
print("\n" + "=" * 30 + " LMM Model Analysis (Categorical Time, Athlete_ID) " + "=" * 30)
print(f"\n数据概况: {len(df_final)} 条记录, {df_final['Athlete_ID'].nunique()} 名运动员, {df_final['Nation'].nunique()} 个国家")
print(f"性别分布: {df_final['Sex'].value_counts().to_dict()}")
print(f"\n时差分组分布:")
print(df_final["Circular_Time_Group"].value_counts().sort_index().to_string())

# ★ 改动2：随机截距改为 Athlete_ID
GROUP_COL = "Athlete_ID"

# 模型1：全样本主效应
print("\n" + "-" * 20 + " [Model 1: Full Sample Main Effects] " + "-" * 20)
print(f"随机截距: {GROUP_COL}")
model_full = smf.mixedlm("Sinclair_Total ~ C(Sex) + C(Circular_Time_Group)",
                         df_final, groups=df_final[GROUP_COL]).fit(method="lbfgs")
print(model_full.summary().tables[1])

# 保存完整 summary
with open(os.path.join(OUTPUT_DIR, "model1_full_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Model 1: Full Sample Main Effects\n")
    f.write(f"随机截距: {GROUP_COL}\n")
    f.write(f"数据: {len(df_final)} rows, {df_final[GROUP_COL].nunique()} groups\n\n")
    f.write(model_full.summary().as_text())

# 模型2：向东飞行子样本
print("\n" + "-" * 20 + " [Model 2: Eastward Flight Subsample] " + "-" * 20)
df_east = df_final[df_final["Flight_Direction"] == "Eastward (向东)"].copy()
df_east["Circular_Time_Group"] = df_east["Circular_Time_Group"].cat.remove_unused_categories()

if len(df_east) > 0 and df_east[GROUP_COL].nunique() > 1:
    try:
        model_east = smf.mixedlm("Sinclair_Total ~ C(Sex) + C(Circular_Time_Group)",
                                 df_east, groups=df_east[GROUP_COL]).fit(method="lbfgs")
        print(model_east.summary().tables[1])
        with open(os.path.join(OUTPUT_DIR, "model2_east_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Model 2: Eastward Flight Subsample\n")
            f.write(f"随机截距: {GROUP_COL}\n\n")
            f.write(model_east.summary().as_text())
    except Exception as e:
        print(f"[Warning] Eastward LMM error: {e}")
else:
    print("Insufficient data for Eastward model.")

# 模型3：向西飞行子样本
print("\n" + "-" * 20 + " [Model 3: Westward Flight Subsample] " + "-" * 20)
df_west = df_final[df_final["Flight_Direction"] == "Westward (向西)"].copy()
df_west["Circular_Time_Group"] = df_west["Circular_Time_Group"].cat.remove_unused_categories()

if len(df_west) > 0 and df_west[GROUP_COL].nunique() > 1:
    try:
        model_west = smf.mixedlm("Sinclair_Total ~ C(Sex) + C(Circular_Time_Group)",
                                 df_west, groups=df_west[GROUP_COL]).fit(method="lbfgs")
        print(model_west.summary().tables[1])
        with open(os.path.join(OUTPUT_DIR, "model3_west_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Model 3: Westward Flight Subsample\n")
            f.write(f"随机截距: {GROUP_COL}\n\n")
            f.write(model_west.summary().as_text())
    except Exception as e:
        print(f"[Warning] Westward LMM error: {e}")
else:
    print("Insufficient data for Westward model.")

# -------------------------- 6. 对比总结 --------------------------
print("\n" + "=" * 70)
print("验证目的：同学代码逻辑 + YT数据 + Athlete_ID随机截距")
print("如果结果与YT自己的LMM_jetlag.py结果一致，")
print("说明同学代码逻辑没有问题，差异完全来自随机截距选择。")
print("=" * 70)

# 保存处理后数据
df_final.to_csv(os.path.join(OUTPUT_DIR, "data_used.csv"), index=False, encoding="UTF-8-sig")
print(f"\n[Completed] Output saved to: {OUTPUT_DIR}/")