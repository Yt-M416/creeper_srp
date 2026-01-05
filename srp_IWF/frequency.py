"""
数据诊断脚本：分析运动员参赛次数和数据特征
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示异常

# 配置
OUTPUT_DIR = "lmm_model1&2"
DATA_FILE = os.path.join(OUTPUT_DIR, "merged_full_with_travel.csv")

print("=" * 70)
print("数据诊断分析：运动员参赛次数与变量特征")
print("=" * 70)

# 读取数据
print(f"\n1. 加载数据: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
print(f"   总行数: {len(df)}")
print(f"   总列数: {len(df.columns)}")

# =====================================================
# 运动员参赛次数分析
# =====================================================
print("\n" + "=" * 70)
print("【运动员参赛次数分析】")
print("=" * 70)

if "Name" in df.columns:
    # 统计每个运动员参赛次数
    athlete_counts = df.groupby("Name").size().reset_index(name="Count")
    
    print(f"\n基础统计：")
    print(f"  - 唯一运动员数: {len(athlete_counts)}")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 平均每人参赛次数: {athlete_counts['Count'].mean():.2f}")
    print(f"  - 中位数参赛次数: {athlete_counts['Count'].median():.0f}")
    print(f"  - 最多参赛次数: {athlete_counts['Count'].max()}")
    
    # 参赛次数分布
    print(f"\n参赛次数分布：")
    count_dist = athlete_counts["Count"].value_counts().sort_index()
    for n_times, n_athletes in count_dist.items():
        pct = n_athletes / len(athlete_counts) * 100
        print(f"  - 参赛 {n_times:2d} 次: {n_athletes:4d} 人 ({pct:5.2f}%)")
    
    # 重复参赛统计
    repeated = athlete_counts[athlete_counts["Count"] > 1]
    single = athlete_counts[athlete_counts["Count"] == 1]
    
    print(f"\n重复参赛情况：")
    print(f"  - 只参加1次比赛: {len(single):4d} 人 ({len(single)/len(athlete_counts)*100:.2f}%)")
    print(f"  - 参加2次及以上: {len(repeated):4d} 人 ({len(repeated)/len(athlete_counts)*100:.2f}%)")
    print(f"  - 重复记录总数: {repeated['Count'].sum():4d} 条")
    
    # Top 参赛次数最多的运动员
    print(f"\n参赛次数最多的10位运动员：")
    top_athletes = athlete_counts.nlargest(10, "Count")
    for idx, row in top_athletes.iterrows():
        athlete_name = row["Name"]
        count = row["Count"]
        # 获取该运动员的国家
        nation = df[df["Name"] == athlete_name]["Nation"].iloc[0]
        print(f"  {count:2d}次 - {athlete_name:30s} ({nation})")
    
    # 可视化1：参赛次数分布柱状图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    count_dist_plot = count_dist.head(15)  # 只显示前15个
    plt.bar(count_dist_plot.index, count_dist_plot.values, color='steelblue', alpha=0.7)
    plt.xlabel("参赛次数", fontsize=12)
    plt.ylabel("运动员数量", fontsize=12)
    plt.title("运动员参赛次数分布", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for x, y in zip(count_dist_plot.index, count_dist_plot.values):
        plt.text(x, y + max(count_dist_plot.values)*0.01, str(y), 
                ha='center', va='bottom', fontsize=9)
    
    # 可视化2：累积分布
    plt.subplot(1, 2, 2)
    cumsum = count_dist.sort_index().cumsum()
    cumsum_pct = cumsum / cumsum.iloc[-1] * 100
    plt.plot(cumsum_pct.index, cumsum_pct.values, marker='o', 
            linewidth=2, markersize=6, color='darkgreen')
    plt.xlabel("参赛次数", fontsize=12)
    plt.ylabel("累积百分比 (%)", fontsize=12)
    plt.title("运动员累积分布", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%线')
    plt.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80%线')
    plt.legend()
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "athlete_participation_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 已保存图表: {output_file}")
    
else:
    print("警告：数据中没有 'Name' 列，无法进行运动员分析")

# =====================================================
# 运动员-国家关系检查
# =====================================================
print("\n" + "=" * 70)
print("【运动员-国家关系检查】")
print("=" * 70)

if "Name" in df.columns and "Nation" in df.columns:
    athlete_nations = df.groupby("Name")["Nation"].nunique()
    multi_nation = athlete_nations[athlete_nations > 1]
    
    print(f"\n国籍变化情况：")
    print(f"  - 只代表1个国家: {(athlete_nations == 1).sum()} 人")
    print(f"  - 代表多个国家: {len(multi_nation)} 人")
    
    if len(multi_nation) > 0:
        print(f"\n代表多个国家的运动员（前10位）：")
        for name in multi_nation.head(10).index:
            nations = df[df["Name"] == name]["Nation"].unique()
            print(f"  - {name}: {', '.join(nations)}")
    else:
        print("\n✓ 没有运动员换国籍的情况")

# =====================================================
# 变量相关性分析
# =====================================================
print("\n" + "=" * 70)
print("【变量相关性与共线性诊断】")
print("=" * 70)

# 选择连续变量
continuous_vars = ["Travel_Duration", "Timezone_Diff_hours", "Sinclair_Total", "Body_Weight"]
available_vars = [v for v in continuous_vars if v in df.columns]

if len(available_vars) >= 2:
    print(f"\n分析变量: {available_vars}")
    
    # 计算相关系数矩阵
    corr_data = df[available_vars].dropna()
    corr_matrix = corr_data.corr()
    
    print(f"\n相关系数矩阵：")
    print(corr_matrix.round(3))
    
    # 重点关注 Travel_Duration 和 Timezone_Diff
    if "Travel_Duration" in available_vars and "Timezone_Diff_hours" in available_vars:
        corr_value = corr_matrix.loc["Travel_Duration", "Timezone_Diff_hours"]
        print(f"\n★ Travel_Duration 与 Timezone_Diff_hours 的相关系数: {corr_value:.4f}")
        
        if abs(corr_value) > 0.7:
            print("  ⚠️ 警告：高度相关（|r| > 0.7），可能存在共线性问题！")
        elif abs(corr_value) > 0.5:
            print("  ⚠️ 注意：中度相关（|r| > 0.5），建议进行中心化处理")
        else:
            print("  ✓ 相关性可接受（|r| < 0.5）")
    
    # 可视化相关矩阵
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title("变量相关性矩阵", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "variable_correlation_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 已保存相关矩阵图: {output_file}")

# =====================================================
# 关键变量描述统计
# =====================================================
print("\n" + "=" * 70)
print("【关键变量描述统计】")
print("=" * 70)

key_vars = ["Travel_Duration", "Timezone_Diff_hours", "Sinclair_Total"]
for var in key_vars:
    if var in df.columns:
        data_var = df[var].dropna()
        print(f"\n{var}:")
        print(f"  - 样本量: {len(data_var)}")
        print(f"  - 均值: {data_var.mean():.2f}")
        print(f"  - 标准差: {data_var.std():.2f}")
        print(f"  - 最小值: {data_var.min():.2f}")
        print(f"  - 25%分位: {data_var.quantile(0.25):.2f}")
        print(f"  - 中位数: {data_var.median():.2f}")
        print(f"  - 75%分位: {data_var.quantile(0.75):.2f}")
        print(f"  - 最大值: {data_var.max():.2f}")

# =====================================================
# Travel_Duration 分布图
# =====================================================
if "Travel_Duration" in df.columns and "Timezone_Diff_hours" in df.columns:
    print("\n绘制关键变量分布图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Travel_Duration 直方图
    ax1 = axes[0, 0]
    df["Travel_Duration"].dropna().hist(bins=30, ax=ax1, color='steelblue', 
                                        alpha=0.7, edgecolor='black')
    ax1.axvline(df["Travel_Duration"].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean={df["Travel_Duration"].mean():.1f}h')
    ax1.set_xlabel("Travel Duration (hours)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Travel Duration Distribution", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Timezone_Diff 直方图
    ax2 = axes[0, 1]
    df["Timezone_Diff_hours"].dropna().hist(bins=30, ax=ax2, color='darkgreen', 
                                            alpha=0.7, edgecolor='black')
    ax2.axvline(df["Timezone_Diff_hours"].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean={df["Timezone_Diff_hours"].mean():.1f}h')
    ax2.set_xlabel("Timezone Difference (hours)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Timezone Difference Distribution", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Travel_Duration vs Timezone_Diff 散点图
    ax3 = axes[1, 0]
    plot_data = df[["Travel_Duration", "Timezone_Diff_hours"]].dropna()
    ax3.scatter(plot_data["Travel_Duration"], plot_data["Timezone_Diff_hours"], 
               alpha=0.4, s=20, color='purple')
    ax3.set_xlabel("Travel Duration (hours)", fontsize=11)
    ax3.set_ylabel("Timezone Difference (hours)", fontsize=11)
    ax3.set_title("Travel Duration vs Timezone Difference", fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 添加相关系数
    if len(plot_data) > 0:
        corr = plot_data.corr().iloc[0, 1]
        ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 按性别的 Sinclair_Total 箱线图
    ax4 = axes[1, 1]
    if "Sex_clean" in df.columns or "Sex" in df.columns:
        sex_col = "Sex_clean" if "Sex_clean" in df.columns else "Sex"
        plot_data_sex = df[[sex_col, "Sinclair_Total"]].dropna()
        plot_data_sex.boxplot(column="Sinclair_Total", by=sex_col, ax=ax4)
        ax4.set_xlabel("Sex", fontsize=11)
        ax4.set_ylabel("Sinclair Total", fontsize=11)
        ax4.set_title("Sinclair Total by Sex", fontsize=12, fontweight='bold')
        plt.sca(ax4)
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "variable_distributions.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存变量分布图: {output_file}")

# =====================================================
# 总结与建议
# =====================================================
print("\n" + "=" * 70)
print("【分析总结与建议】")
print("=" * 70)

if "Name" in df.columns:
    athlete_counts = df.groupby("Name").size()
    repeated_pct = (athlete_counts > 1).sum() / len(athlete_counts) * 100
    
    print(f"\n1. 运动员随机效应建议：")
    if repeated_pct < 10:
        print(f"   ✗ 重复参赛比例过低（{repeated_pct:.1f}%），不建议加入 (1|Athlete_ID)")
        print(f"   → 保持当前的 (1|Nation) 即可")
    elif repeated_pct < 30:
        print(f"   ⚠ 重复参赛比例中等（{repeated_pct:.1f}%），可以尝试加入 (1|Athlete_ID)")
        print(f"   → 建议先用 (1|Nation)，再比较 (1|Nation/Athlete_ID)")
    else:
        print(f"   ✓ 重复参赛比例较高（{repeated_pct:.1f}%），强烈建议加入 (1|Athlete_ID)")
        print(f"   → 使用 (1|Nation/Athlete_ID) 嵌套结构")

if "Travel_Duration" in df.columns and "Timezone_Diff_hours" in df.columns:
    corr = df[["Travel_Duration", "Timezone_Diff_hours"]].corr().iloc[0, 1]
    print(f"\n2. 变量中心化建议：")
    if abs(corr) > 0.5:
        print(f"   ✓ 相关系数 r={corr:.3f}，建议进行均值中心化")
        print(f"   → 可以降低共线性，改善模型解释性")
    else:
        print(f"   ⚠ 相关系数 r={corr:.3f}，中心化不是必须的")
        print(f"   → 但中心化仍能改善交互项的解释性")


print("\n" + "=" * 70)
print("诊断完成！")
print("=" * 70)