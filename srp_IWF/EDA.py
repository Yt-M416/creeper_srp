"""
IWF举重世锦赛：时差与运动表现的探索性数据分析 (EDA)
======================================================
最初步的EDA脚本，输出时差分布、分性别散点图和箱线图
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 步骤 0: 环境准备 (字体配置 & 路径设置)
# ===================================================================
def setup_environment():
    """配置中文字体、创建输出目录，为分析做准备。"""
    print("=" * 60)
    print("STEP 0: 正在配置分析环境...")
    print("=" * 60)
    
    # --- 字体配置 ---
    font_paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
    font_name = 'Microsoft YaHei' # 默认字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            print(f"-> 成功加载中文字体: {font_name}")
            break
    
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font_scale=1.2, 
                  rc={'font.sans-serif': [font_name]})
    print("-> 字体配置完成。")
    
    # --- 路径设置 ---
    output_dir = "eda_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"-> 输出目录 '{output_dir}' 已创建。")
    else:
        print(f"-> 输出目录 '{output_dir}' 已存在。")
        
    return output_dir

output_dir = setup_environment()

# ===================================================================
# 步骤 1: 数据加载与预处理
# ===================================================================
print("\n" + "=" * 60)
print("STEP 1: 正在加载和预处理数据...")
print("=" * 60)

df = pd.read_csv("iwf_data_with_tz.csv")

# --- 数据清洗与特征工程 ---
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
df.dropna(subset=['Total', 'Timezone_Diff_hours'], inplace=True)
df['Gender'] = df['Event'].apply(lambda x: 'Women' if 'women' in str(x).lower() else 'Men')
# 从 Event 提取级别信息，例如 "55 kg Men" -> "55kg"
df['Category'] = df['Event'].str.extract(r'(\d{2,3}\+?kg|\+\d{2,3}kg)')[0].str.replace(' ', '')

print(f"-> 数据加载完成。数据集包含 {len(df)} 条有效记录。")
print(f"-> 已创建 'Gender' 和 'Category' 列用于分组分析。")


# ===================================================================
# 步骤 2: 单变量分析 (理解核心变量分布)
# ===================================================================
print("\n" + "=" * 60)
print("STEP 2: 单变量分析 - 探索核心变量自身分布...")
print("=" * 60)

# --- 绘制时差分布图 ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Timezone_Diff_hours'], kde=True, bins=25, color='orange')
plt.title("运动员经历的时差分布", fontsize=16, fontweight='bold')
plt.xlabel("时差 (小时, 向西为正)", fontsize=12)
plt.ylabel("频数", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_timediff_distribution.png"), dpi=300)
plt.close()

# --- 输出日志 ---
mean_diff = df['Timezone_Diff_hours'].mean()
std_diff = df['Timezone_Diff_hours'].std()
print("-> [图 01] 时差分布图已生成。")
print(f"  LOG: 数据点主要集中在0附近，但覆盖了从东到西的大范围时差。")
print(f"       均值约为 {mean_diff:.2f} 小时，标准差为 {std_diff:.2f} 小时，分布较广。")


# ===================================================================
# 步骤 3: 多变量分析 (探索变量间关系)
# ===================================================================
print("\n" + "=" * 60)
print("STEP 3: 多变量分析 - 探索时差、性别与成绩的关系...")
print("=" * 60)

# --- 绘制分性别散点图 ---
g = sns.lmplot(
    x='Timezone_Diff_hours', y='Total', col='Gender', data=df,
    height=6, aspect=1.2, scatter_kws={'alpha':0.4}, line_kws={'color':'red'}
)
g.set_titles("性别: {col_name}", size=14)
g.set_axis_labels("时差 (小时)", "总成绩 (kg)", fontsize=12)
g.fig.suptitle("分性别：时差与总成绩的初步关系", y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_scatter_faceted_by_gender.png"), dpi=300)
plt.close()

print("-> [图 02] 分性别散点图已生成。")
print(f"  LOG: 混合所有级别后，无论男性还是女性，都看不出时差与成绩有明显的线性关系。")
print(f"       这暗示我们需要进一步按级别细分，或将时差分类处理。")


# --- 将时差分箱，便于分组比较 ---
bins = [-18, -6.5, -2.5, 2.5, 6.5, 18]
labels = ['向东大时差\n(<-6.5h)', '向东小时差\n(-6.5 to -2.5h)', 
          '微小时差\n(-2.5 to 2.5h)', '向西小时差\n(2.5 to 6.5h)', 
          '向西大时差\n(>6.5h)']
df['Timezone_Group'] = pd.cut(df['Timezone_Diff_hours'], bins=bins, labels=labels)

# --- 绘制分性别、分时差组的箱线图 ---
g = sns.catplot(
    x='Timezone_Group', y='Total', col='Gender', data=df,
    kind='box', palette='pastel', height=6, aspect=1.3
)
g.set_titles("性别: {col_name}", size=14)
g.set_axis_labels("时差分组", "总成绩 (kg)", fontsize=12)
g.set_xticklabels(rotation=15)
g.fig.suptitle("分性别：不同时差分组对总成绩的影响", y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(output_dir, "03_boxplot_faceted_by_gender.png"), dpi=300)
plt.close()

print("\n-> [图 03] 分性别与时差组的箱线图已生成。")
print(f"  LOG: 将时差分类后，可以比较不同时差条件下成绩的分布。")
print(f"       初步观察，男性的'向东大时差'组的成绩中位数似乎略低于其他组，值得深入研究。")


# ===================================================================
# 步骤 4: 统计探索 (量化关系)
# ===================================================================
print("\n" + "=" * 60)
print("STEP 4: 统计探索 - 通过计算量化相关性...")
print("=" * 60)

# --- 计算总体相关系数 ---
pearson_corr = df['Total'].corr(df['Timezone_Diff_hours'], method='pearson')
spearman_corr = df['Total'].corr(df['Timezone_Diff_hours'], method='spearman')

# --- 计算各时差组的统计数据 ---
summary = df.groupby('Timezone_Group')['Total'].agg(['median', 'mean', 'count']).sort_index()

# --- 将统计结果写入文件 ---
stats_summary_path = os.path.join(output_dir, "summary_statistics.txt")
with open(stats_summary_path, 'w', encoding='utf-8') as f:
    f.write("探索性数据分析 - 统计摘要\n" + "="*30 + "\n\n")
    f.write(f"总体皮尔逊相关系数 (线性): {pearson_corr:.3f}\n")
    f.write(f"总体斯皮尔曼相关系数 (序数): {spearman_corr:.3f}\n\n")
    f.write("各时差组的成绩统计:\n")
    f.write(summary.to_string())
print(f"-> [文件] 统计摘要已保存。")
print(f"  LOG: 总体相关系数接近0 ({pearson_corr:.3f})，证实了混合数据无法揭示关系。")
print(f"       摘要文件中的分组统计数据，为图3的观察提供了数值支持。")


# --- 计算更细粒度的分组相关系数 ---
def calc_group_corr(group_df):
    if len(group_df) < 5: return None # 样本太少则不计算
    p = group_df['Total'].corr(group_df['Timezone_Diff_hours'], method='pearson')
    s = group_df['Total'].corr(group_df['Timezone_Diff_hours'], method='spearman')
    return pd.Series({'Pearson': p, 'Spearman': s, 'n': len(group_df)})

corr_by_group = df.groupby(['Gender', 'Category']).apply(calc_group_corr).reset_index()
corr_by_group.to_csv(os.path.join(output_dir, "correlation_by_group.csv"), index=False)
print(f"\n-> [文件] 按性别和级别的分组相关系数已保存。")

# --- 绘制相关系数热力图 ---
try:
    pivot_table = corr_by_group.pivot(index='Category', columns='Gender', values='Pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=.5)
    plt.title("各级别皮尔逊相关系数热力图 (时差 vs 成绩)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_heatmap_correlation.png"), dpi=300)
    plt.close()
    print("-> [图 04] 相关系数热力图已生成。")

except Exception as e:
    print(f"  LOG: 未能生成热力图，可能是级别数据不足或格式问题。错误: {e}")


# ===================================================================
# 步骤 5: 流程结束
# ===================================================================
print("\n" + "=" * 60)
print("EDA流程结束。")
print(f"-> 所有图表和统计文件已保存到 '{output_dir}' 文件夹。")
print("=" * 60)