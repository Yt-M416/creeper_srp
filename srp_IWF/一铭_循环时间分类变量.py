import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------- Global Font Config --------------------------
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

# -------------------------- 1. Path Configuration (修正命名) --------------------------
INPUT_CSV_PATH = r"C:\Users\2026\Desktop\srp_IWF\srp_IWF\Sinclair-LMM\iwf_data_with_tz.csv"
TRAVEL_TIME_PATH = r"C:\Users\2026\Desktop\srp_IWF\srp_IWF\travel_time_merged.csv"
# 核心修正：将「连续时间」改为「分类时间」
OUTPUT_DIR = r"C:\Users\2026\Desktop\循环分类时间LMM"

# 自动创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------- 2. Data Loading & Preprocessing --------------------------
df = pd.read_csv(INPUT_CSV_PATH, encoding="UTF-8-sig")

# 计算最短圆周时差 (±12h)
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

# 将圆周时间转换为2小时间隔的分类变量
time_bins = np.arange(-12, 13, 2)
time_labels = [f"[{time_bins[i]}, {time_bins[i+1]})" for i in range(len(time_bins)-1)]
df["Circular_Time_Group"] = pd.cut(df["Circular_Time"], bins=time_bins, labels=time_labels, right=False, include_lowest=True)

# 创建飞行方向变量
df["Flight_Direction"] = np.where(df["Circular_Time"] > 0, "Eastward (向东)",
                                   np.where(df["Circular_Time"] < 0, "Westward (向西)", None))

# 数据清洗
key_cols = ["Total", "Body_Weight", "Event", "Circular_Time", "Circular_Time_Group", "Flight_Direction", "Nation"]
df_clean = df[key_cols].dropna(subset=key_cols).copy()

df_clean["Total"] = pd.to_numeric(df_clean["Total"], errors="coerce")
df_clean["Body_Weight"] = pd.to_numeric(df_clean["Body_Weight"], errors="coerce")
df_clean = df_clean.dropna(subset=["Total", "Body_Weight"])
df_clean = df_clean[df_clean["Body_Weight"] > 0]

# 提取性别
def extract_sex(event_str):
    event_str = str(event_str)
    if "Women" in event_str: return "Female"
    elif "Men" in event_str: return "Male"
    return None

df_clean["Sex"] = df_clean["Event"].apply(extract_sex)
df_clean = df_clean.dropna(subset=["Sex"])

# -------------------------- 3. 2021-2024 IWF官方辛克莱系数计算 --------------------------
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

# -------------------------- 4. Merge Travel Data --------------------------
try:
    travel_df = pd.read_csv(TRAVEL_TIME_PATH)
    travel_df_clean = travel_df[['Departure_Nation', 'Duration_Hours(h)']].rename(
        columns={'Departure_Nation': 'Nation', 'Duration_Hours(h)': 'Travel_Duration'}
    ).drop_duplicates()
    df_final = df_clean.merge(travel_df_clean, on='Nation', how='left').dropna(subset=['Travel_Duration'])
except:
    df_final = df_clean.copy()

# -------------------------- 5. LMM模型拟合 --------------------------
print("\n" + "=" * 30 + " LMM Model Analysis (Categorical Time) " + "=" * 30)

# 模型1：全样本主效应
print("\n" + "-" * 20 + " [Model 1: Full Sample Main Effects] " + "-" * 20)
model_full = smf.mixedlm("Sinclair_Total ~ C(Sex) + Travel_Duration + C(Circular_Time_Group)",
                         df_final, groups=df_final["Nation"]).fit()
print(model_full.summary().tables[1])

# 模型2：向东飞行子样本
print("\n" + "-" * 20 + " [Model 2: Eastward Flight Subsample] " + "-" * 20)
df_east = df_final[df_final["Flight_Direction"] == "Eastward (向东)"].copy()
df_east["Circular_Time_Group"] = df_east["Circular_Time_Group"].cat.remove_unused_categories()

if len(df_east) > 0 and df_east["Nation"].nunique() > 1:
    try:
        model_east = smf.mixedlm("Sinclair_Total ~ C(Sex) + Travel_Duration + C(Circular_Time_Group)",
                                 df_east, groups=df_east["Nation"]).fit()
        print(model_east.summary().tables[1])
    except Exception as e:
        print(f"[Warning] Eastward LMM encountered an error: {e}")
        print("Switching to OLS...")
        model_east_simple = smf.ols("Sinclair_Total ~ C(Sex) + Travel_Duration + C(Circular_Time_Group)", data=df_east).fit()
        print(model_east_simple.summary().tables[1])
else:
    print("Insufficient data for Eastward model.")

# 模型3：向西飞行子样本
print("\n" + "-" * 20 + " [Model 3: Westward Flight Subsample] " + "-" * 20)
df_west = df_final[df_final["Flight_Direction"] == "Westward (向西)"].copy()
df_west["Circular_Time_Group"] = df_west["Circular_Time_Group"].cat.remove_unused_categories()

if len(df_west) > 0 and df_west["Nation"].nunique() > 1:
    try:
        model_west = smf.mixedlm("Sinclair_Total ~ C(Sex) + Travel_Duration + C(Circular_Time_Group)",
                                 df_west, groups=df_west["Nation"]).fit()
        print(model_west.summary().tables[1])
    except Exception as e:
        print(f"[Warning] Westward LMM encountered an error: {e}")
        print("Switching to OLS...")
        model_west_simple = smf.ols("Sinclair_Total ~ C(Sex) + Travel_Duration + C(Circular_Time_Group)", data=df_west).fit()
        print(model_west_simple.summary().tables[1])
else:
    print("Insufficient data for Westward model.")

# -------------------------- 6. 可视化图表生成 (修正图表标题) --------------------------

# 1. 向东飞行：圆周时差分组 vs Sinclair成绩 箱线图
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_east, x="Circular_Time_Group", y="Sinclair_Total", palette="Blues")
plt.title("Eastward Flight: Categorical Circular Timezone vs Sinclair Total", fontsize=14)
plt.xlabel("Categorical Circular Timezone (2h intervals, hours)", fontsize=12)
plt.ylabel("Sinclair Total", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/V1_Eastward_CategoricalTime_Boxplot.png")
plt.close()

# 2. 向西飞行：圆周时差分组 vs Sinclair成绩 箱线图
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_west, x="Circular_Time_Group", y="Sinclair_Total", palette="Reds")
plt.title("Westward Flight: Categorical Circular Timezone vs Sinclair Total", fontsize=14)
plt.xlabel("Categorical Circular Timezone (2h intervals, hours)", fontsize=12)
plt.ylabel("Sinclair Total", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/V2_Westward_CategoricalTime_Boxplot.png")
plt.close()

# 3. 全样本：分类圆周时差与Sinclair成绩的核心关系图
plt.figure(figsize=(16, 7))
sns.boxplot(data=df_final, x="Circular_Time_Group", y="Sinclair_Total", hue="Sex", palette="vlag")
plt.title("Core Relationship: Categorical Circular Timezone (±12h) vs Sinclair Total", fontsize=14)
plt.xlabel("Categorical Circular Timezone (2h intervals, Negative=Westward, Positive=Eastward)", fontsize=12)
plt.ylabel("Sinclair Total", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Sex")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/V3_FullSample_CategoricalTime_Boxplot.png")
plt.close()

# 4. 东向vs西向飞行的成绩分布对比箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_final, x="Flight_Direction", y="Sinclair_Total", palette="Set2")
plt.title("Performance Comparison: Eastward vs Westward (Categorical Time Analysis)", fontsize=14)
plt.xlabel("Flight Direction", fontsize=12)
plt.ylabel("Sinclair Total", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/V4_Direction_Comparison_CategoricalTime.png")
plt.close()

# -------------------------- 完成提示与数据保存 (修正文件名) --------------------------
print(f"\n[Completed] All analysis (Categorical Time) finished!")
print(f"4 visualization plots saved to: {OUTPUT_DIR}")
# 修正输出数据文件名
df_final.to_csv(f"{OUTPUT_DIR}/iwf_data_categorical_time_analysis.csv", index=False, encoding="UTF-8-sig")
print(f"Processed data saved to: {OUTPUT_DIR}/iwf_data_categorical_time_analysis.csv")