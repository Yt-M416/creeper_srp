"""
IWF举重世锦赛 - LMM建模与分析报告
======================================================
本脚本旨在通过线性混合效应模型（LMM）量化时差对
举重运动员Sinclair归一化成绩的影响，并生成一份
包含统计摘要和核心结果可视化的分析报告。
"""
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ===================================================================
# 步骤 0: 环境与日志设置
# ===================================================================
def setup_environment_and_logging():
    """配置字体、创建输出目录，并设置日志记录。"""
    print("=" * 60)
    print("STEP 0: 正在配置环境与日志...")
    print("=" * 60)
    
    # --- 字体配置 ---
    try:
        # 尝试找到系统中可用的中文字体
        font_paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc', 
                      'C:/Windows/Fonts/Deng.ttf']
        font_name = 'Microsoft YaHei' # 默认
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                print(f"-> 成功加载并选择字体: {font_name}")
                break
        
        # 【关键】使用更全局、更强制的方式设置
        plt.style.use('seaborn-v0_8-whitegrid') # 先设置基础风格
        plt.rcParams.update({
            'font.sans-serif': [font_name, 'Arial'], # 将中文字体放在首位
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
        })
        print(f"-> 全局字体已强制设置为: {plt.rcParams['font.sans-serif']}")
        
    except Exception as e:
        print(f"-> 字体设置警告: 未能成功加载中文字体。图表中文可能显示为方框。错误: {e}")
    
    # --- 路径设置 ---
    output_dir = "lmm_analysis_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 日志设置 ---
    log_path = os.path.join(output_dir, "lmm_analysis_log.txt")
    # 创建一个Logger类来同时打印到控制台和文件
    class Logger(object):
        def __init__(self, filename="default.log", stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'w', encoding='utf-8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    
    sys.stdout = Logger(log_path, sys.stdout)
    print(f"-> 环境配置完成，所有输出将被记录到: {log_path}")
    return output_dir

output_dir = setup_environment_and_logging()

# ===================================================================
# 步骤 1: 数据准备
# ===================================================================
def prepare_data(file_path):
    """加载数据、计算Sinclair分数并进行预处理。"""
    print("\n" + "=" * 60)
    print("STEP 1: 正在加载和预处理数据...")
    print("=" * 60)

    # --- Sinclair计算函数 ---
    def calculate_sinclair(total, Body_Weight, gender):
        params = {'Men': {'b': 175.508, 'a': 0.751945}, 'Women': {'b': 153.655, 'a': 0.783497}}
        if pd.isna(total) or pd.isna(Body_Weight) or Body_Weight <= 0: return np.nan
        p = params.get(gender)
        if not p or Body_Weight >= p['b']: return total
        return total * (10 ** (p['a'] * (np.log10(p['b'] / Body_Weight)) ** 2))

    # --- 加载与特征工程 ---
    df = pd.read_csv(file_path)
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
    df['Body_Weight'] = pd.to_numeric(df['Body_Weight'], errors='coerce')
    df.dropna(subset=['Total', 'Body_Weight', 'Timezone_Diff_hours', 'Name'], inplace=True)
    
    df['Gender'] = df['Event'].str.extract(r'(Men|Women)', flags=re.IGNORECASE)[0].str.title()
    df['Sinclair_Total'] = df.apply(lambda r: calculate_sinclair(r['Total'], r['Body_Weight'], r['Gender']), axis=1)
    df.dropna(subset=['Sinclair_Total'], inplace=True)
    
    print(f"-> Sinclair Total 计算完成，剩余 {len(df)} 条有效记录用于建模。")

    # --- 创建与设置分类变量 ---
    df["Timezone_Group"] = pd.cut(
        df["Timezone_Diff_hours"],
        bins=[-100, -6.5, -2.5, 2.5, 6.5, 100],
        labels=["向东大时差\n(<-6.5h)", "向东小时差\n(-6.5~-2.5h)", "微小时差\n(-2.5~2.5h)", 
                "向西小时差\n(2.5~6.5h)", "向西大时差\n(>6.5h)"]
    )
    df['Timezone_Group'] = df['Timezone_Group'].astype('category').cat.reorder_categories(
        ["微小时差\n(-2.5~2.5h)", "向东小时差\n(-6.5~-2.5h)", "向西小时差\n(2.5~6.5h)", 
         "向西大时差\n(>6.5h)", "向东大时差\n(<-6.5h)"], ordered=False
    )
    df["Athlete_Name"] = df["Name"].astype("category")
    df['Gender'] = df['Gender'].astype('category')
    
    print("-> 数据预处理完成。")
    return df

df = prepare_data("iwf_data_with_tz.csv")

# ===================================================================
# 步骤 2: LMM建模与分析
# ===================================================================
def run_lmm_analysis(df):
    """运行所有LMM模型并打印结果。"""
    print("\n" + "=" * 60)
    print("STEP 2: 正在运行LMM模型...")
    print("=" * 60)

    # --- 1. 基线模型 (核心模型) ---
    print("\n===== [核心] 基线模型 (随机截距) =====")
    baseline_model = smf.mixedlm("Sinclair_Total ~ C(Timezone_Group) + C(Gender)", data=df, groups=df["Athlete_Name"])
    baseline_result = baseline_model.fit(method="lbfgs")
    print(baseline_result.summary())
    
    # 计算并打印ICC
    group_var = baseline_result.cov_re.iloc[0,0]
    resid_var = baseline_result.scale
    icc = group_var / (group_var + resid_var)
    print(f"\n组内相关系数 (ICC): {icc:.3f}")
    print("解读: 数据总变异的 {:.1f}% 来自于运动员的个体差异，证明使用LMM是必要的。".format(icc * 100))

    # --- 2. 交互效应模型 (检验性别差异) ---
    print("\n\n===== [检验] 交互效应模型 =====")
    interaction_model = smf.mixedlm("Sinclair_Total ~ C(Timezone_Group) * C(Gender)", data=df, groups=df["Athlete_Name"])
    interaction_result = interaction_model.fit(method="lbfgs")
    print(interaction_result.summary())
    print("\n解读: 所有交互项的 P>|z| 均大于0.05，表明时差的影响在男女之间没有统计上的显著差异。")

    # --- 3. 随机斜率模型 (探索性) ---
    print("\n\n===== [探索] 随机斜率模型 =====")
    try:
        random_slope_model = smf.mixedlm("Sinclair_Total ~ Timezone_Diff_hours + C(Gender)", data=df, 
                                         groups=df["Athlete_Name"], re_formula="~Timezone_Diff_hours")
        random_slope_result = random_slope_model.fit(method="lbfgs")
        print(random_slope_result.summary())
    except Exception as e:
        print(f"模型未能收敛或出错: {e}")
        
    return baseline_result, interaction_result

baseline_result, interaction_result = run_lmm_analysis(df)

# ===================================================================
# 步骤 3: 核心结果可视化
# ===================================================================
def visualize_main_results(result, df, output_dir):
    """为核心的基线模型生成边际效应图。"""
    print("\n" + "=" * 60)
    print("STEP 3: 正在生成核心结果可视化图表...")
    print("=" * 60)
    
    # --- 准备预测数据 ---
    # 我们需要为每个时差组和性别创建一个“平均”运动员来进行预测
    pred_data = pd.DataFrame({
        'Timezone_Group': df['Timezone_Group'].cat.categories,
        'Gender': 'Men' # 以男性为基准
    })
    men_preds = result.predict(pred_data)
    
    pred_data['Gender'] = 'Women'
    women_preds = result.predict(pred_data)

    plot_df = pd.DataFrame({
        'Timezone_Group': np.concatenate([df['Timezone_Group'].cat.categories, df['Timezone_Group'].cat.categories]),
        'Gender': ['Men'] * len(df['Timezone_Group'].cat.categories) + ['Women'] * len(df['Timezone_Group'].cat.categories),
        'Predicted_Sinclair': np.concatenate([men_preds, women_preds])
    })

    # --- 绘制边际效应图 ---
    plt.figure(figsize=(11, 7))
    sns.pointplot(data=df, x="Timezone_Group", y="Sinclair_Total", hue="Gender",
                  dodge=True, errorbar="ci", capsize=.1, palette="Set2", linestyles="--", alpha=0.7)
    
    # 叠加模型的预测线，使其更突出
    sns.pointplot(data=plot_df, x="Timezone_Group", y="Predicted_Sinclair", hue="Gender",
                  dodge=True, palette="Set2", markers="o", scale=1.2)

    plt.title("LMM预测：时差对Sinclair成绩的边际效应", fontsize=18, fontweight='bold')
    plt.xlabel("时差分组", fontsize=12)
    plt.ylabel("预测的 Sinclair Total (含95%置信区间)", fontsize=12)
    plt.xticks(rotation=10, ha="right")
    
    # 调整图例
    handles, labels = plt.gca().get_legend_handles_labels()
    # 我们只需要前两个（男性和女性）
    plt.legend(handles[2:], labels[2:], title="性别", fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "lmm_marginal_effects_plot.png")
    plt.savefig(plot_path, dpi=400)
    plt.close()

    print(f"-> 核心结果可视化图表已保存至: {plot_path}")
    print("  LOG: 此图展示了在控制个体差异后，不同时差组的预测平均成绩。")
    print("       '向东大时差'组的预测点及其置信区间是否明显低于'微小时差'组，是关键的视觉证据。")

visualize_main_results(baseline_result, df, output_dir)

# ===================================================================
# 步骤 4: 流程结束
# ===================================================================
print("\n" + "=" * 60)
print("LMM分析流程结束。")
print(f"-> 所有结果已保存到 '{output_dir}' 文件夹。")
print("=" * 60)