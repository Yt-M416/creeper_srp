"""
IWF举重世锦赛 EDA 整合版 - Sinclair归一化
============================================================
功能：
1. 支持两种归一化方式：Sinclair系数 & 世界纪录百分比
2. 整合箱线图、热力图、LOWESS非线性趋势分析
3. 输出详细统计摘要
4. 可视化时差与成绩的多维度关系
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 配置区：选择归一化方法
# ===================================================================
NORMALIZATION_METHOD = "sinclair"  # 可选: "sinclair" 或 "world_record"

# ===================================================================
# 步骤 0: 环境准备
# ===================================================================
def setup_environment():
    """配置中文字体、创建输出目录"""
    print("=" * 70)
    print("STEP 0: 正在配置分析环境...")
    print("=" * 70)
    
    font_paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
    font_name = 'Microsoft YaHei'
    for fp in font_paths:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            font_name = font_manager.FontProperties(fname=fp).get_name()
            print(f"-> 成功加载中文字体: {font_name}")
            break
    
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font_scale=1.2, 
                  rc={'font.sans-serif': [font_name]})
    
    output_dir = "eda_normalized_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"-> 输出目录: '{output_dir}'")
    print(f"-> 归一化方法: {NORMALIZATION_METHOD.upper()}\n")
    return output_dir

output_dir = setup_environment()

# ===================================================================
# Sinclair系数计算函数
# ===================================================================
def calculate_sinclair(row):
    total = row['Total']
    bw = row['Body_Weight']
    gender = row['Gender']
    
    sinclair_params = {
        'Men': {'b': 175.508, 'a': 0.751945},
        'Women': {'b': 153.655, 'a': 0.783497}
    }
    
    if pd.isna(total) or pd.isna(bw) or bw <= 0:
        return pd.Series([np.nan, np.nan])
    
    params = sinclair_params.get(gender)
    if not params:
        return pd.Series([np.nan, np.nan])
    
    b, a = params['b'], params['a']
    
    # 体重超出上限 → 认为系数 = 1
    if bw >= b:
        coef = 1.0
    else:
        coef = 10 ** (a * (np.log10(b / bw)) ** 2)
    
    return pd.Series([coef, total * coef])


# ===================================================================
# 世界纪录数据库（备用归一化方法）
# ===================================================================
WORLD_RECORDS = {
    'Men': {
        'pre_2018': {'56kg':307, '62kg':333, '69kg':359, '77kg':380, 
                     '85kg':396, '94kg':417, '105kg':437, '+105kg':477},
        'post_2018': {'55kg':294, '61kg':318, '67kg':339, '73kg':365, 
                      '81kg':378, '89kg':404, '96kg':417, '102kg':413, 
                      '109kg':435, '+109kg':492}
    },
    'Women': {
        'pre_2018': {'48kg':217, '53kg':233, '58kg':252, '63kg':262, 
                     '69kg':276, '75kg':296, '90kg':283, '+90kg':348},
        'post_2018': {'45kg':196, '49kg':221, '55kg':234, '59kg':248, 
                      '64kg':261, '71kg':273, '76kg':278, '81kg':284, 
                      '87kg':294, '+87kg':335}
    }
}

def normalize_by_world_record(row):
    """世界纪录百分比归一化"""
    era = 'pre_2018' if row['Event_Year'] < 2018 else 'post_2018'
    record = WORLD_RECORDS.get(row['Gender'], {}).get(era, {}).get(row['Category'])
    return row['Total'] / record if record else None

# ===================================================================
# 步骤 1: 数据加载与归一化
# ===================================================================
print("=" * 70)
print("STEP 1: 正在加载数据并进行归一化...")
print("=" * 70)

df = pd.read_csv("iwf_data_with_tz.csv")

# 基础数据清洗
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
df['Timezone_Diff_hours'] = pd.to_numeric(df['Timezone_Diff_hours'], errors='coerce')
df['Body_Weight'] = pd.to_numeric(df['Body_Weight'], errors='coerce') # 确保体重也是数字
df.dropna(subset=['Total', 'Timezone_Diff_hours'], inplace=True)

# 提取性别
df['Gender'] = df['Event'].apply(lambda x: 'Women' if 'women' in str(x).lower() else 'Men')

# --- 【核心修复】增强Category提取的鲁棒性 ---
# 原正则表达式: r'(\d{2,3}\+?kg|\+\d{2,3}kg)'
# 修正后正则表达式: 允许级别和kg之间有空格，并处理 "+" 在前或在后的情况
df['Category'] = df['Event'].str.extract(r'(\+?\d{2,3}\+?)\s*kg')[0] + 'kg'
# 进一步清理，例如将 '109+kg' 统一为 '+109kg' (如果需要)
df['Category'] = df['Category'].str.replace(r'(\d+)\+kg', r'+\1kg', regex=True)

# --- 【关键验证步骤】 ---
print("-> 正在验证'Category'列的创建情况...")
nan_count = df['Category'].isna().sum()
if nan_count > 0:
    print(f"  -> 警告: 'Category'列中仍有 {nan_count} 个空值，这些行将被忽略。")
    # 显示哪些Event无法匹配，帮助进一步调试
    unmatched_events = df[df['Category'].isna()]['Event'].unique()
    print(f"  -> 未能匹配的Event示例: {unmatched_events[:5]}")
    df.dropna(subset=['Category'], inplace=True) # 移除无法匹配的行

print(f"-> 'Category'列创建完成。唯一值示例: {df['Category'].unique()[:10]}")
# ----------------------------------------------------

# 提取体重（用于Sinclair计算）
if 'Body_Weight' not in df.columns or df['Body_Weight'].isna().all():
    print("-> 警告：数据中无有效体重列，将从级别中推断中位体重")
    def estimate_Body_Weight(category):
        if pd.isna(category): return np.nan
        weight_str = category.replace('kg', '').replace('+', '')
        try:
            return float(weight_str)
        except:
            return np.nan
    df['Body_Weight'] = df['Category'].apply(estimate_Body_Weight)

# 应用归一化
if NORMALIZATION_METHOD == "sinclair":
    print("-> 使用Sinclair系数归一化...")
    
    df[['Sinclair_Coefficient', 'Sinclair_Total']] = df.apply(calculate_sinclair, axis=1)
    df['Normalized_Total'] = df['Sinclair_Total']   # 兼容后续绘图变量
    
    norm_label = "Sinclair Total"

else:
    print("-> 使用世界纪录百分比归一化...")
    df['Normalized_Total'] = df.apply(normalize_by_world_record, axis=1)
    norm_label = "世界纪录百分比"

df.dropna(subset=['Normalized_Total'], inplace=True)
print(f"-> 归一化完成，剩余有效记录: {len(df)} 条")

# 计算基础统计
pearson_corr = df['Normalized_Total'].corr(df['Timezone_Diff_hours'], method='pearson')
spearman_corr = df['Normalized_Total'].corr(df['Timezone_Diff_hours'], method='spearman')
print(f"-> 整体Pearson相关系数: {pearson_corr:.4f}")
print(f"-> 整体Spearman相关系数: {spearman_corr:.4f}\n")
# ===================================================================
# 步骤 2: 时差分组
# ===================================================================
print("=" * 70)
print("STEP 2: 创建时差分组...")
print("=" * 70)

print("---- Debug: 时差列检查 ----")
print("时差范围调试：")
print(df['Timezone_Diff_hours'].describe())
print("样本数量:", len(df))
print("是否存在 NA:", df['Timezone_Diff_hours'].isna().sum())
print("取值前30:", df['Timezone_Diff_hours'].unique()[:30])
print(df['Timezone_Diff_hours'].describe())


bins = [-18, -6.5, -2.5, 2.5, 6.5, 18]
labels = [
    "向东大时差\n(<-6.5h)",
    "向东小时差\n(-6.5~-2.5h)",
    "微小时差\n(-2.5~2.5h)",
    "向西小时差\n(2.5~6.5h)",
    "向西大时差\n(>6.5h)"
]
df['Timezone_Group'] = pd.cut(df['Timezone_Diff_hours'], bins=bins, labels=labels)

group_stats = df.groupby('Timezone_Group')['Normalized_Total'].agg(['count', 'mean', 'median', 'std'])
print("-> 各时差组样本分布:")
print(group_stats)
print()

# ===================================================================
# 步骤 3: 可视化 - 分性别箱线图
# ===================================================================
print("=" * 70)
print("STEP 3: 生成可视化图表...")
print("=" * 70)


# 图1: 分性别的时差组箱线图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, gender in enumerate(['Men', 'Women']):
    df_gender = df[df['Gender'] == gender]
    sns.boxplot(
        data=df_gender, 
        x='Timezone_Group', 
        y='Normalized_Total', 
        ax=axes[idx],
        palette='Set2'
    )
    axes[idx].set_title(f'{gender} - 时差组对归一化成绩的影响', fontsize=14)
    axes[idx].set_xlabel('时差分组', fontsize=12)
    axes[idx].set_ylabel(norm_label, fontsize=12)
    axes[idx].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_boxplot_by_gender.png"), dpi=300)
plt.close()
print("-> [图1] 分性别箱线图已保存")

# 图2: 整体散点图（按性别着色）
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x='Timezone_Diff_hours',
    y='Normalized_Total',
    hue='Gender',
    alpha=0.4,
    s=50
)
plt.title(f'时差 vs {norm_label}（按性别）', fontsize=16)
plt.xlabel('时差（小时）', fontsize=12)
plt.ylabel(norm_label, fontsize=12)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.legend(title='性别', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_scatter_overall.png"), dpi=300)
plt.close()
print("-> [图2] 整体散点图已保存")

# 图3: LOWESS非线性趋势分析
plt.figure(figsize=(14, 7))
colors = {'Men': '#1f77b4', 'Women': '#ff7f0e'}

for gender in ['Men', 'Women']:
    df_gender = df[df['Gender'] == gender].sort_values('Timezone_Diff_hours')
    
    # 绘制散点（半透明）
    plt.scatter(
        df_gender['Timezone_Diff_hours'],
        df_gender['Normalized_Total'],
        alpha=0.15,
        s=30,
        color=colors[gender],
        label=f'{gender} 数据点'
    )
    
    # LOWESS平滑曲线
    lowess = sm.nonparametric.lowess(
        df_gender['Normalized_Total'],
        df_gender['Timezone_Diff_hours'],
        frac=0.25
    )
    plt.plot(
        lowess[:, 0],
        lowess[:, 1],
        color=colors[gender],
        linewidth=3,
        label=f'{gender} LOWESS趋势'
    )

plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='零时差')
plt.title(f'时差对{norm_label}的非线性影响（LOWESS平滑）', fontsize=16)
plt.xlabel('时差（小时）', fontsize=13)
plt.ylabel(norm_label, fontsize=13)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_lowess_trend.png"), dpi=300)
plt.close()
print("-> [图3] LOWESS非线性趋势图已保存")

# 图4: 相关系数热力图（分级别分性别）
print("-> 正在计算各级别与时差的相关系数...")


# 计算相关系数并构造成干净的表
pairs = []
for (gender, category), g in df.groupby(["Gender", "Category"]):
    r = g["Normalized_Total"].corr(g["Timezone_Diff_hours"], method="pearson")
    pairs.append([gender, category, r])

corr_by_group = pd.DataFrame(pairs, columns=["Gender", "Category", "Pearson"])


print("DEBUG: corr_by_group 预检查")
print("corr_by_group shape:", corr_by_group.shape)
print("corr_by_group head:\n", corr_by_group.head(10))
print("Pearson 列 NA 数量:", corr_by_group['Pearson'].isna().sum())
print("Gender 唯一值:", corr_by_group['Gender'].unique())
print("Category 唯一值（前20）:", corr_by_group['Category'].unique()[:20])

# 转成热力图矩阵
pivot_table = corr_by_group.pivot(
    index="Category",
    columns="Gender",
    values="Pearson"
)

plt.figure(figsize=(11, 9))
sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson相关系数'}
)
plt.title(f'时差与{norm_label}相关系数热力图（按体重级别 × 性别）', fontsize=15)
plt.xlabel('性别', fontsize=12)
plt.ylabel('体重级别', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_heatmap_correlation.png"), dpi=300)
plt.close()
print("-> [图4] 相关系数热力图已保存")


# 图5: 分组对比的小提琴图
plt.figure(figsize=(14, 6))
sns.violinplot(
    data=df,
    x='Timezone_Group',
    y='Normalized_Total',
    hue='Gender',
    split=True,
    palette='muted'
)
plt.title(f'不同时差组的{norm_label}分布（小提琴图）', fontsize=15)
plt.xlabel('时差分组', fontsize=12)
plt.ylabel(norm_label, fontsize=12)
plt.xticks(rotation=15)
plt.legend(title='性别', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_violin_plot.png"), dpi=300)
plt.close()
print("-> [图5] 小提琴图已保存")

# ===================================================================
# 步骤 4: 统计摘要输出
# ===================================================================
print("\n" + "=" * 70)
print("STEP 4: 生成统计摘要文件...")
print("=" * 70)

summary_path = os.path.join(output_dir, "summary_statistics.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write(f"IWF举重世锦赛数据分析摘要 - {norm_label}\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"归一化方法: {NORMALIZATION_METHOD.upper()}\n")
    f.write(f"总样本量: {len(df)} 条\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("1. 整体相关性分析\n")
    f.write("-" * 70 + "\n")
    f.write(f"Pearson相关系数: {pearson_corr:.4f}\n")
    f.write(f"Spearman相关系数: {spearman_corr:.4f}\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("2. 按时差分组的统计\n")
    f.write("-" * 70 + "\n")
    summary_group = df.groupby('Timezone_Group')['Normalized_Total'].agg([
        ('样本量', 'count'),
        ('均值', 'mean'),
        ('中位数', 'median'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('最大值', 'max')
    ])
    f.write(summary_group.to_string())
    f.write("\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("3. 按性别分组的统计\n")
    f.write("-" * 70 + "\n")
    summary_gender = df.groupby('Gender')['Normalized_Total'].agg([
        ('样本量', 'count'),
        ('均值', 'mean'),
        ('中位数', 'median'),
        ('标准差', 'std')
    ])
    f.write(summary_gender.to_string())
    f.write("\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("4. 各级别的相关系数\n")
    f.write("-" * 70 + "\n")
    corr_sorted = corr_by_group.sort_values('Pearson', ascending=False)
    f.write(corr_sorted.to_string(index=False))
    f.write("\n\n")
    
    # Sinclair方法的特殊说明
    if NORMALIZATION_METHOD == "sinclair":
        f.write("-" * 70 + "\n")
        f.write("Sinclair系数说明\n")
        f.write("-" * 70 + "\n")
        f.write("Sinclair Total = Total × 10^(a × (log₁₀(b/w))²)\n\n")
        f.write("参数设置:\n")
        f.write("  男性: b=175.508kg, a=0.751945\n")
        f.write("  女性: b=153.655kg, a=0.783497\n\n")
        f.write("特点:\n")
        f.write("  - 消除体重级别差异，使不同量级成绩可比\n")
        f.write("  - 轻量级运动员获得更高系数（相对力量更强）\n")
        f.write("  - 与年份无关，基于纯数学转换\n")

print(f"-> 统计摘要已保存: {summary_path}")

# 保存分组相关系数为CSV
corr_csv_path = os.path.join(output_dir, "correlation_by_category.csv")
corr_by_group.to_csv(corr_csv_path, index=False, encoding='utf-8-sig')
print(f"-> 分级别相关系数已保存: {corr_csv_path}")

# ===================================================================
# 完成
# ===================================================================
print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)
print(f"所有结果已保存至: {output_dir}")
print("\n生成的文件:")
print("  - 01_boxplot_by_gender.png (分性别箱线图)")
print("  - 02_scatter_overall.png (整体散点图)")
print("  - 03_lowess_trend.png (非线性趋势)")
print("  - 04_heatmap_correlation.png (相关系数热力图)")
print("  - 05_violin_plot.png (小提琴图)")
print("  - summary_statistics.txt (详细统计摘要)")
print("  - correlation_by_category.csv (分级别相关系数)")
print("\n" + "=" * 70)

lmm_df = df[[
    'Athlete_Name', 'Country', 'Gender', 'Category',
    'Event_Year', 'Body_Weight',
    'Timezone_Diff_hours', 'Normalized_Total'
]].dropna()

lmm_df['Gender'] = lmm_df['Gender'].astype('category')
lmm_df['Country'] = lmm_df['Country'].astype('category')
lmm_df['Category'] = lmm_df['Category'].astype('category')

lmm_csv_path = os.path.join(output_dir, "iwf_lmm_dataset.csv")
lmm_df.to_csv(lmm_csv_path, index=False, encoding='utf-8-sig')
print(f"-> LMM分析数据已保存: {lmm_csv_path}")
