"""
探索性数据分析 (EDA) - IWF举重世锦赛时差与表现
【V3 - 彻底修复中文显示版】
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 0. 彻底解决中文显示问题
# =====================================================

# 方法1: 使用font_manager直接指定字体文件（最可靠）
def setup_chinese_font():
    """设置中文字体的多重保险方案"""
    
    # 尝试1: 直接指定Windows字体路径
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',      # 黑体
        'C:/Windows/Fonts/msyh.ttc',        # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',      # 宋体
    ]
    
    font_added = False
    font_name = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                print(f"✓ 成功加载字体: {font_name} (路径: {font_path})")
                font_added = True
                break
            except Exception as e:
                print(f"✗ 无法加载 {font_path}: {e}")
                continue
    
    # 尝试2: 如果方法1失败，使用系统已安装字体
    if not font_added:
        print("尝试使用系统已安装字体...")
        font_name = 'Microsoft YaHei'
    
    # 【关键修复】全面设置所有可能影响的参数
    plt.rcParams['font.sans-serif'] = [font_name, 'Microsoft YaHei', 'SimHei', 'SimSun']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 强制设置所有text相关的字体
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 16
    
    # 设置seaborn（使用更新的方法）
    sns.set_theme(style="whitegrid", font_scale=1.2, 
                  rc={'font.sans-serif': [font_name, 'Microsoft YaHei', 'SimHei']})
    
    # 验证配置
    print(f"\n当前字体配置:")
    print(f"  font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"  font.family: {plt.rcParams['font.family']}")
    print(f"  axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")
    
    # 测试中文渲染
    test_chinese()
    
    return font_name

def test_chinese():
    """测试中文是否能正常显示"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, '测试中文显示：时差分析', 
            ha='center', va='center', fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig('test_chinese_display.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\n已生成测试图片: test_chinese_display.png")
    print("请检查该图片中文是否正常显示。")

# 执行字体设置
print("=" * 60)
print("开始配置中文字体...")
print("=" * 60)
setup_chinese_font()

# =====================================================
# 1. 数据读取与基本设置
# =====================================================
data_path = "iwf_data_with_tz.csv"
output_dir = "eda_plots"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n文件夹 '{output_dir}' 已创建。")

df = pd.read_csv(data_path)

# 确保主要变量存在
required_cols = ['Total', 'Timezone_Diff_hours', 'Event', 'Nation', 'Event_Year']
if not all(c in df.columns for c in required_cols):
    raise ValueError(f"缺少必要列，请检查文件。")

# =====================================================
# 2. 数据清洗
# =====================================================
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
df = df.dropna(subset=['Total', 'Timezone_Diff_hours'])
df['Gender'] = df['Event'].apply(lambda x: 'Women' if 'women' in str(x).lower() else 'Men')

# 设置绘图风格
print("\n数据加载和预处理完成，开始绘图...")

# =====================================================
# 阶段一：单变量分析
# =====================================================
print("\n阶段一：单变量分析...")

plt.figure(figsize=(10, 6))
sns.histplot(df['Timezone_Diff_hours'], kde=True, bins=25, color='orange')
plt.title("运动员经历的时差分布", fontsize=16, fontweight='bold')
plt.xlabel("Timezone_Diff_hours (小时)", fontsize=12)
plt.ylabel("频数", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_timediff_distribution.png"), dpi=300)
plt.close()

# =====================================================
# 阶段三：多变量分析
# =====================================================
print("阶段三：多变量分析...")

g = sns.lmplot(
    x='Timezone_Diff_hours', y='Total', col='Gender',
    data=df, col_wrap=2, height=6, aspect=1.2,
    scatter_kws={'alpha':0.4}, line_kws={'color':'red'}
)
g.set_titles("性别: {col_name}", size=14)
g.set_axis_labels("Timezone_Diff_hours", "Total (kg)", fontsize=12)
g.fig.suptitle("分性别时差与成绩关系", y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05a_scatter_faceted_by_gender.png"), dpi=300, bbox_inches='tight')
plt.close()

# 时差分组
bins = [-18, -6.5, -2.5, 2.5, 6.5, 18]
labels = ['Eastward_Large\n(<-6.5h)', 'Eastward_Small\n(-6.5 to -2.5h)', 
          'Minimal_Jetlag\n(-2.5 to 2.5h)', 'Westward_Small\n(2.5 to 6.5h)', 
          'Westward_Large\n(>6.5h)']
df['Timezone_Group'] = pd.cut(df['Timezone_Diff_hours'], bins=bins, labels=labels)

g = sns.catplot(
    x='Timezone_Group', y='Total', col='Gender', 
    data=df, kind='box', palette='pastel',
    height=6, aspect=1.3
)
g.set_titles("性别: {col_name}", size=14)
g.set_axis_labels("时差分组", "Total (kg)", fontsize=12)
g.set_xticklabels(rotation=15)
g.fig.suptitle("分性别比较：不同时差分组对总成绩的影响", y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(output_dir, "06b_boxplot_faceted_by_gender.png"), dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 阶段四：简单统计探索
# =====================================================
print("阶段四：简单统计探索...")

pearson_corr = df['Total'].corr(df['Timezone_Diff_hours'], method='pearson')
spearman_corr = df['Total'].corr(df['Timezone_Diff_hours'], method='spearman')

summary = df.groupby('Timezone_Group')['Total'].agg(['median', 'mean', 'count']).sort_index()

stats_summary_path = os.path.join(output_dir, "summary_statistics.txt")
with open(stats_summary_path, 'w', encoding='utf-8') as f:
    f.write("探索性数据分析 - 统计摘要\n")
    f.write("="*30 + "\n\n")
    f.write(f"Pearson相关系数 (Total vs. Timezone_Diff): {pearson_corr:.3f}\n")
    f.write(f"Spearman相关系数 (Total vs. Timezone_Diff): {spearman_corr:.3f}\n\n")
    f.write("各时差组的成绩统计:\n")
    f.write(summary.to_string())

# =====================================================
# 阶段五：自动计算多层分组的相关系数
# =====================================================
print("阶段五：自动计算性别、级别、时差与成绩的相关关系...")

# 先确保存在性别和级别字段
if 'Gender' not in df.columns:
    df['Gender'] = df['Event'].apply(lambda x: 'Women' if 'women' in str(x).lower() else 'Men')
if 'Category' not in df.columns:
    # 假设项目名中包含量级信息，例如 "Men_73kg"
    df['Category'] = df['Event'].apply(lambda x: ''.join([c for c in str(x) if c.isdigit() or c == 'k']))

# 定义一个函数计算指定分组下的相关系数
def calc_group_corr(group_df):
    """计算某一分组内的 Pearson 与 Spearman 相关系数"""
    pearson = group_df['Total'].corr(group_df['Timezone_Diff_hours'], method='pearson')
    spearman = group_df['Total'].corr(group_df['Timezone_Diff_hours'], method='spearman')
    return pd.Series({'Pearson': pearson, 'Spearman': spearman, 'n': len(group_df)})

# 计算性别 × 级别分组下的相关性
corr_gender_cat = df.groupby(['Gender', 'Category']).apply(calc_group_corr).reset_index()

# 保存结果
corr_output_path = os.path.join(output_dir, "correlation_by_gender_category.csv")
corr_gender_cat.to_csv(corr_output_path, index=False, encoding='utf-8-sig')

print(f"已生成分组相关性结果: {corr_output_path}")

# 画出一个热力图（相关系数矩阵）
pivot_table = corr_gender_cat.pivot(index='Category', columns='Gender', values='Pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("不同级别和性别下的Pearson相关系数", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "07_heatmap_corr_by_gender_category.png"), dpi=300)
plt.close()

print("阶段五完成。")


print(f"\n统计摘要已保存到: {stats_summary_path}")
print(f"\n所有图表已成功保存到 '{output_dir}' 文件夹。")
print("=" * 60)
print("EDA流程结束。请检查 test_chinese_display.png 验证中文显示。")
print("=" * 60)