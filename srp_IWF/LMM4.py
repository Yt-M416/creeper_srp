"""
完整模型对比：连续时差 vs 分组时差（包含交互项）
Model 1: Sinclair ~ Sex + Travel_Duration_c * Timezone_Diff_hours_c + (1|Athlete_ID)
Model 2: Sinclair ~ Sex + Travel_Duration_c * Timezone_Group + (1|Athlete_ID)
"""
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import chi2
import warnings
import matplotlib.pyplot as plt
 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示异常

# 配置
IWF_FILE = "iwf_data_with_tz.csv"
TRAVEL_FILE = "travel_time_merged.csv"
OUTPUT_DIR = "lmm_continuous_vs_grouped"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志
log_results = os.path.join(OUTPUT_DIR, "models_results.txt")
log_comparison = os.path.join(OUTPUT_DIR, "models_comparison.txt")
log_console = os.path.join(OUTPUT_DIR, "analysis_log.txt")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_console, encoding="utf-8"),
                             logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("lmm_comparison")

logger.info("=" * 80)
logger.info("完整模型对比：连续时差 vs 分组时差（含交互项）")
logger.info("=" * 80)

# ========== 辅助函数 ==========
def calculate_sinclair(total, body_weight, gender):
    try:
        total, bw = float(total), float(body_weight)
    except:
        return np.nan
    params = {"Men": {"a": -0.794358141, "A": 174.393},
              "Women": {"a": -0.897260740, "A": 148.026}}
    if pd.isna(total) or pd.isna(bw) or bw <= 0:
        return np.nan
    g = str(gender).strip().lower() if not pd.isna(gender) else ""
    p = params["Men"] if g.startswith("m") else params["Women"] if g.startswith("w") else params["Men"]
    A, a = p["A"], p["a"]
    coeff = 1.0 if bw > A else np.exp(a * (np.log(bw / A) ** 2))
    return total * coeff

def load_travel_table(travel_csv=TRAVEL_FILE):
    tt = pd.read_csv(travel_csv, encoding="utf-8-sig")
    tt.columns = tt.columns.str.strip()
    col_mapper = {}
    for col in tt.columns:
        cl = col.lower()
        if "departure_nation" in cl: col_mapper[col] = "Departure_Nation"
        elif "departure_city" in cl or cl == "departure": col_mapper[col] = "Departure_City"
        elif "arrival_city" in cl or cl == "arrival": col_mapper[col] = "Arrival_City"
        elif "duration" in cl: col_mapper[col] = "Duration_Hours"
        elif cl in ["types", "type"]: col_mapper[col] = "Types"
        elif cl == "year": col_mapper[col] = "Year"
    tt = tt.rename(columns=col_mapper).dropna(how="all")
    if "Duration_Hours" in tt.columns:
        tt["Duration_Hours"] = pd.to_numeric(tt["Duration_Hours"], errors="coerce")
    if "Year" in tt.columns:
        tt["Year"] = pd.to_numeric(tt["Year"], errors="coerce").astype("Int64")
    return tt

def load_iwf(iwf_path=IWF_FILE):
    iwf = pd.read_csv(iwf_path, encoding="utf-8-sig")
    iwf.columns = iwf.columns.str.strip()
    cols = {c.lower(): c for c in iwf.columns}
    rename_map = {}
    for key, new_name in [("nation", "Nation"), ("city", "Event_City"), ("total", "Total"), 
                          ("event_year", "Event_Year"), ("timezone_diff_hours", "Timezone_Diff_hours"),
                          ("name", "Name")]:
        if key in cols: rename_map[cols[key]] = new_name
    for key in ["body_weight", "body weight", "bodyweight"]:
        if key in cols: rename_map[cols[key]] = "Body_Weight"; break
    if "year" in cols and "event_year" not in cols: rename_map[cols["year"]] = "Event_Year"
    iwf = iwf.rename(columns=rename_map)
    for col in ["Event_Year", "Total", "Body_Weight"]:
        if col in iwf.columns:
            iwf[col] = pd.to_numeric(iwf[col], errors="coerce")
            if col == "Event_Year": iwf[col] = iwf[col].astype("Int64")
    if "Event" in iwf.columns and "Sex" not in iwf.columns:
        iwf["Sex"] = iwf["Event"].astype(str).str.extract(r"(Men|Women|men|women)", expand=False)
    if "Sex" in iwf.columns:
        iwf["Sex"] = iwf["Sex"].astype("string")
    if "Timezone_Diff_hours" not in iwf.columns:
        if {"Athlete_UTC_offset", "Venue_UTC_offset"}.issubset(set(iwf.columns)):
            iwf["Timezone_Diff_hours"] = pd.to_numeric(iwf["Athlete_UTC_offset"], errors="coerce") - \
                                          pd.to_numeric(iwf["Venue_UTC_offset"], errors="coerce")
    return iwf

def normalize_city_country_string(s):
    return str(s).strip().replace(", ", ",") if not pd.isna(s) else s

def merge_tables(iwf, travel):
    travel["Arrival_City_norm_lower"] = travel["Arrival_City"].apply(normalize_city_country_string).astype(str).str.lower().str.replace(" ", "")
    iwf["Event_City_norm"] = (iwf["Event_City"] if "Event_City" in iwf.columns else iwf.get("City", iwf["Event_City"])).apply(normalize_city_country_string)
    iwf["Event_City_norm_lower"] = iwf["Event_City_norm"].astype(str).str.lower().str.replace(" ", "")
    merged = iwf.merge(travel[["Departure_Nation", "Arrival_City_norm_lower", "Year", "Duration_Hours"]],
                      left_on=["Nation", "Event_City_norm_lower", "Event_Year"],
                      right_on=["Departure_Nation", "Arrival_City_norm_lower", "Year"],
                      how="left", suffixes=("", "_travel"))
    merged["Travel_Duration"] = merged["Duration_Hours"]
    return merged

def categorize_timezone(tz):
    """7组时差分类"""
    if pd.isna(tz): return np.nan
    if tz < -6.5: return "East_Large"
    elif -6.5 <= tz < -3: return "East_Medium"
    elif -3 <= tz < -1: return "East_Small"
    elif -1 <= tz <= 1: return "Minimal"
    elif 1 < tz <= 3: return "West_Small"
    elif 3 < tz <= 6.5: return "West_Medium"
    else: return "West_Large"

def apply_mean_centering(df, variables):
    centering_info = {}
    for var in variables:
        if var not in df.columns: continue
        mean_val, std_val = df[var].mean(), df[var].std()
        centered_var = f"{var}_c"
        df[centered_var] = df[var] - mean_val
        centering_info[var] = {"mean": mean_val, "std": std_val}
        logger.info(f"  {var}: mean={mean_val:.4f}, std={std_val:.4f} → {centered_var}")
    return df, centering_info

def calculate_icc(model_result):
    try:
        random_var = model_result.cov_re.iloc[0, 0]
        residual_var = model_result.scale
        return random_var / (random_var + residual_var)
    except:
        return np.nan

# ========== 主分析 ==========
def run_two_models(merged_df):
    # 数据准备
    if "Sinclair_Total" not in merged_df.columns:
        merged_df["Sinclair_Total"] = merged_df.apply(
            lambda r: calculate_sinclair(r.get("Total", np.nan), r.get("Body_Weight", np.nan), 
                                        r.get("Sex", np.nan)), axis=1)
    
    merged_df["Sex_clean"] = merged_df.get("Sex", pd.Series(np.nan, index=merged_df.index))
    merged_df["Sex_clean"] = merged_df["Sex_clean"].astype(str).str.strip().str.title()
    merged_df.loc[~merged_df["Sex_clean"].isin(["Men", "Women"]), "Sex_clean"] = np.nan
    
    if "Name" not in merged_df.columns:
        logger.error("缺少 Name 列")
        return
    merged_df["Athlete_ID"] = merged_df["Name"].astype(str) + "_" + merged_df["Nation"].astype(str)
    
    model_df = merged_df[["Sinclair_Total", "Sex_clean", "Travel_Duration", "Timezone_Diff_hours", 
                          "Nation", "Athlete_ID", "Name"]].copy()
    model_df = model_df.dropna(subset=["Sinclair_Total", "Sex_clean", "Timezone_Diff_hours"])
    complete_for_model = model_df.dropna(subset=["Travel_Duration"]).copy()
    
    logger.info(f"\n数据概况：")
    logger.info(f"  总行数: {len(complete_for_model)}")
    logger.info(f"  运动员数: {complete_for_model['Athlete_ID'].nunique()}")
    logger.info(f"  国家数: {complete_for_model['Nation'].nunique()}")
    logger.info(f"  平均每人参赛: {len(complete_for_model)/complete_for_model['Athlete_ID'].nunique():.2f}次")
    
    # 中心化
    logger.info("\n应用均值中心化：")
    complete_for_model, centering_info = apply_mean_centering(complete_for_model, 
                                                               ["Travel_Duration", "Timezone_Diff_hours"])
    
    # 时差分组
    complete_for_model["Timezone_Group"] = complete_for_model["Timezone_Diff_hours"].apply(categorize_timezone)
    complete_for_model["Timezone_Group"] = pd.Categorical(
        complete_for_model["Timezone_Group"],
        categories=["Minimal", "East_Small", "East_Medium", "East_Large", 
                   "West_Small", "West_Medium", "West_Large"], ordered=False)
    
    logger.info("\n时差分组分布：")
    for group, count in complete_for_model["Timezone_Group"].value_counts().sort_index().items():
        logger.info(f"  {group}: {count:4d} ({count/len(complete_for_model)*100:5.1f}%)")
    
    # 类型转换
    complete_for_model["Athlete_ID"] = complete_for_model["Athlete_ID"].astype("category")
    
    # 过滤：至少2次参赛
    athlete_counts = complete_for_model["Athlete_ID"].value_counts()
    valid_athletes = athlete_counts[athlete_counts >= 2].index
    complete_for_model = complete_for_model[complete_for_model["Athlete_ID"].isin(valid_athletes)].copy()
    complete_for_model["Athlete_ID"] = complete_for_model["Athlete_ID"].cat.remove_unused_categories()
    
    logger.info(f"\n过滤后（≥2次参赛）：")
    logger.info(f"  行数: {len(complete_for_model)}")
    logger.info(f"  运动员数: {complete_for_model['Athlete_ID'].nunique()}")
    
    # 保存
    complete_for_model.to_csv(os.path.join(OUTPUT_DIR, "model_input_data.csv"), 
                              index=False, encoding="utf-8-sig")
    
    # ========== 拟合两个模型 ==========
    results = {}
    
    logger.info("\n" + "=" * 80)
    logger.info("拟合模型（含交互项）")
    logger.info("=" * 80)
    
    # Model 1: 连续时差 + 交互项
    formula1 = "Sinclair_Total ~ Sex_clean + Travel_Duration_c * Timezone_Diff_hours_c"
    logger.info(f"\n【Model 1: 连续时差 + 交互项】")
    logger.info(f"  完整公式: {formula1} + (1|Athlete_ID)")
    logger.info(f"  展开: Sex + Travel_Duration_c + Timezone_Diff_hours_c + Travel_Duration_c:Timezone_Diff_hours_c")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model1 = smf.mixedlm(formula1, data=complete_for_model, 
                                groups=complete_for_model["Athlete_ID"], re_formula="1")
            res1_ml = model1.fit(method="powell", maxiter=1000, reml=False)
            res1_reml = model1.fit(method="powell", maxiter=1000, reml=True)
        icc1 = calculate_icc(res1_reml)
        logger.info(f"  ✓ 完成 | AIC={res1_ml.aic:.2f}, BIC={res1_ml.bic:.2f}, LogLik={res1_ml.llf:.4f}, ICC={icc1:.4f}")
        results["Model1_Continuous"] = {
            "ml": res1_ml, "reml": res1_reml, "formula": formula1,
            "name": "连续时差 + 交互项", "icc": icc1,
            "n_params": len(res1_reml.params)
        }
    except Exception as e:
        logger.exception(f"Model 1 失败: {e}")
        results["Model1_Continuous"] = None
    
    # Model 2: 分组时差 + 交互项
    formula2 = "Sinclair_Total ~ Sex_clean + Travel_Duration_c * Timezone_Group"
    logger.info(f"\n【Model 2: 分组时差 + 交互项】")
    logger.info(f"  完整公式: {formula2} + (1|Athlete_ID)")
    logger.info(f"  展开: Sex + Travel_Duration_c + Timezone_Group[6组] + Travel_Duration_c:Timezone_Group[6组]")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model2 = smf.mixedlm(formula2, data=complete_for_model,
                                groups=complete_for_model["Athlete_ID"], re_formula="1")
            res2_ml = model2.fit(method="powell", maxiter=1000, reml=False)
            res2_reml = model2.fit(method="powell", maxiter=1000, reml=True)
        icc2 = calculate_icc(res2_reml)
        logger.info(f"  ✓ 完成 | AIC={res2_ml.aic:.2f}, BIC={res2_ml.bic:.2f}, LogLik={res2_ml.llf:.4f}, ICC={icc2:.4f}")
        results["Model2_Grouped"] = {
            "ml": res2_ml, "reml": res2_reml, "formula": formula2,
            "name": "分组时差 + 交互项", "icc": icc2,
            "n_params": len(res2_reml.params)
        }
    except Exception as e:
        logger.exception(f"Model 2 失败: {e}")
        results["Model2_Grouped"] = None
    
    # ========== 输出与可视化 ==========
    write_results_log(results, centering_info, complete_for_model)
    compare_models(results)
    plot_comparisons(results, complete_for_model, centering_info)
    
    return results, complete_for_model

def write_results_log(results, centering_info, data):
    """输出完整模型结果"""
    with open(log_results, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("两模型完整结果：连续时差 vs 分组时差（含交互项）\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("数据概况：\n")
        f.write(f"  样本量: {len(data)}\n")
        f.write(f"  运动员数: {data['Athlete_ID'].nunique()}\n")
        f.write(f"  平均参赛次数: {len(data)/data['Athlete_ID'].nunique():.2f}\n\n")
        
        f.write("中心化信息：\n")
        for var, info in centering_info.items():
            f.write(f"  {var}: mean={info['mean']:.4f}, std={info['std']:.4f}\n")
        
        f.write("\n时差分组统计：\n")
        for group, count in data["Timezone_Group"].value_counts().sort_index().items():
            f.write(f"  {group}: {count} ({count/len(data)*100:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        for name, res in results.items():
            if res is None:
                f.write(f"【{name}】拟合失败\n\n")
                continue
            
            f.write(f"【{name}: {res['name']}】\n")
            f.write("-" * 80 + "\n")
            f.write(f"公式: {res['formula']} + (1|Athlete_ID)\n")
            f.write(f"参数数量: {res['n_params']}\n\n")
            f.write(res["reml"].summary().as_text())
            f.write(f"\n\n模型指标：\n")
            f.write(f"  AIC: {res['ml'].aic:.2f}\n")
            f.write(f"  BIC: {res['ml'].bic:.2f}\n")
            f.write(f"  Log-Likelihood (ML): {res['ml'].llf:.4f}\n")
            f.write(f"  ICC: {res['icc']:.4f}\n")
            f.write(f"  自由度: {res['n_params']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    logger.info(f"\n✓ 完整结果已保存: {log_results}")

def compare_models(results):
    """模型比较"""
    with open(log_comparison, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("模型比较：连续时差 vs 分组时差（含交互项）\n")
        f.write("=" * 80 + "\n\n")
        
        valid_results = {k: v for k, v in results.items() if v}
        
        if len(valid_results) < 2:
            f.write("警告：只有一个模型成功拟合，无法比较\n")
            logger.warning("只有一个模型成功，跳过比较")
            return
        
        # 拟合度对比
        f.write("1. 拟合度指标对比：\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模型':<25} {'参数数':<10} {'AIC':<12} {'BIC':<12} {'LogLik':<15} {'ICC':<10}\n")
        f.write("-" * 80 + "\n")
        for name, res in valid_results.items():
            f.write(f"{res['name']:<25} {res['n_params']:<10} {res['ml'].aic:<12.2f} {res['ml'].bic:<12.2f} "
                   f"{res['ml'].llf:<15.4f} {res['icc']:<10.4f}\n")
        
        # AIC/BIC 差异
        f.write("\n\n2. AIC/BIC 比较：\n")
        f.write("-" * 80 + "\n")
        models_list = list(valid_results.items())
        aic_diff = models_list[1][1]["ml"].aic - models_list[0][1]["ml"].aic
        bic_diff = models_list[1][1]["ml"].bic - models_list[0][1]["ml"].bic
        
        f.write(f"Δ AIC (Model2 - Model1): {aic_diff:+.2f}\n")
        f.write(f"Δ BIC (Model2 - Model1): {bic_diff:+.2f}\n\n")
        
        f.write("解释：\n")
        if abs(aic_diff) < 2:
            f.write("  • |ΔAIC| < 2: 两模型拟合度相当，无明显差异\n")
        elif aic_diff < -10:
            f.write(f"  • ΔAIC < -10: {models_list[0][1]['name']} 显著更好\n")
        elif aic_diff > 10:
            f.write(f"  • ΔAIC > 10: {models_list[1][1]['name']} 显著更好\n")
        elif aic_diff < 0:
            f.write(f"  • ΔAIC = {aic_diff:.2f}: {models_list[0][1]['name']} 略优\n")
        else:
            f.write(f"  • ΔAIC = {aic_diff:.2f}: {models_list[1][1]['name']} 略优\n")
        
        # LRT（如果模型嵌套）
        f.write("\n\n3. 似然比检验（LRT）：\n")
        f.write("-" * 80 + "\n")
        # Model 1 (4参数) 嵌套在 Model 2 (更多参数) 中
        if models_list[0][1]["n_params"] < models_list[1][1]["n_params"]:
            df_diff = models_list[1][1]["n_params"] - models_list[0][1]["n_params"]
            lr_stat = -2 * (models_list[0][1]["ml"].llf - models_list[1][1]["ml"].llf)
            p_value = chi2.sf(lr_stat, df=df_diff)
            
            f.write(f"H0: 连续时差模型足够（简单模型）\n")
            f.write(f"H1: 分组时差模型显著更好（复杂模型）\n\n")
            f.write(f"LR Statistic: {lr_stat:.4f}\n")
            f.write(f"自由度: {df_diff}\n")
            f.write(f"p-value: {p_value:.4f}\n\n")
            
            if p_value < 0.001:
                f.write("结论: 分组时差模型显著更好 (p<0.001) ✓\n")
                f.write("  → 时差存在显著的非线性/阈值效应\n")
            elif p_value < 0.05:
                f.write("结论: 分组时差模型显著更好 (p<0.05) ✓\n")
                f.write("  → 分组能捕捉到连续变量无法发现的效应\n")
            else:
                f.write("结论: 连续时差模型足够 (p≥0.05)\n")
                f.write("  → 分组未提供显著改善，连续假设合理\n")
        else:
            f.write("注：两模型参数数量相同或顺序不符，跳过LRT\n")
        
        # 推荐
        f.write("\n\n4. 推荐模型：\n")
        f.write("-" * 80 + "\n")
        best_aic = min(valid_results.items(), key=lambda x: x[1]["ml"].aic)
        best_bic = min(valid_results.items(), key=lambda x: x[1]["ml"].bic)
        
        f.write(f"基于 AIC: {best_aic[1]['name']} (AIC={best_aic[1]['ml'].aic:.2f})\n")
        f.write(f"基于 BIC: {best_bic[1]['name']} (BIC={best_bic[1]['ml'].bic:.2f})\n\n")
        
        # 综合建议
        f.write("综合建议：\n")
        if abs(aic_diff) < 2:
            f.write("  • AIC差异<2，选择更简单的【连续时差模型】\n")
            f.write("  • 理由：拟合度相当时，优先选择简约模型（奥卡姆剃刀）\n")
        elif aic_diff < 0:
            f.write("  • 选择【连续时差模型】\n")
            f.write("  • 理由：更低的AIC，且参数更少，解释更简洁\n")
        else:
            f.write("  • 选择【分组时差模型】\n")
            f.write("  • 理由：显著更低的AIC，说明时差存在非线性效应\n")
            f.write("  • 优势：能捕捉阈值效应（如>6小时的突变）\n")
    
    logger.info(f"✓ 模型比较已保存: {log_comparison}")

def plot_comparisons(results, df, centering_info):
    """可视化两个模型：预测 vs 真实、残差分布、交互项效果"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 提取两个模型
    res1 = results.get("Model1_Continuous", None)
    res2 = results.get("Model2_Grouped", None)

    if res1 is None or res2 is None:
        logger.error("模型拟合失败，无法绘图")
        return

    # ---------------------------------------------------------
    # 1. 预测 vs 真实
    # ---------------------------------------------------------
    df["Pred_Model1"] = res1["reml"].fittedvalues
    df["Pred_Model2"] = res2["reml"].fittedvalues

    plt.figure(figsize=(8, 6))
    plt.scatter(df["Sinclair_Total"], df["Pred_Model1"], alpha=0.5)
    plt.plot([df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             [df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             linestyle="--")
    plt.xlabel("真实 Sinclair 总分")
    plt.ylabel("预测 Sinclair（Model 1: 连续时差）")
    plt.title("预测 vs 真实（Model 1）")
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_obs_model1.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df["Sinclair_Total"], df["Pred_Model2"], alpha=0.5)
    plt.plot([df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             [df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             linestyle="--")
    plt.xlabel("真实 Sinclair 总分")
    plt.ylabel("预测 Sinclair（Model 2: 分组时差）")
    plt.title("预测 vs 真实（Model 2）")
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_obs_model2.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. 残差分布
    # ---------------------------------------------------------
    df["Residual_1"] = df["Sinclair_Total"] - df["Pred_Model1"]
    df["Residual_2"] = df["Sinclair_Total"] - df["Pred_Model2"]

    plt.figure(figsize=(10, 6))
    plt.hist(df["Residual_1"], bins=40, alpha=0.6, label="Model 1")
    plt.hist(df["Residual_2"], bins=40, alpha=0.6, label="Model 2")
    plt.xlabel("残差值")
    plt.ylabel("频数")
    plt.title("残差分布比较（Model 1 vs Model 2）")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. 交互项可视化：Travel × Timezone
    # ---------------------------------------------------------

    # 3.1 连续时差模型：绘制 3 条飞行时长水平
    dur_mean = centering_info["Travel_Duration"]["mean"]
    dur_vals = [dur_mean - 10, dur_mean, dur_mean + 10]  # 短、中、长航班
    tz_range = np.linspace(df["Timezone_Diff_hours"].min(),
                           df["Timezone_Diff_hours"].max(), 100)

    plt.figure(figsize=(8, 6))
    for d in dur_vals:
        input_df = pd.DataFrame({
            "Sex_clean": ["Men"] * len(tz_range),
            "Travel_Duration_c": d - centering_info["Travel_Duration"]["mean"],
            "Timezone_Diff_hours_c": tz_range - centering_info["Timezone_Diff_hours"]["mean"],
            "Athlete_ID": df["Athlete_ID"].iloc[0]  # dummy
        })
        preds = res1["reml"].predict(input_df)
        plt.plot(tz_range, preds, label=f"飞行时长 ≈ {d:.1f} 小时")

    plt.xlabel("时差（小时）")
    plt.ylabel("预测 Sinclair")
    plt.title("交互项效果（Model 1: 连续时差）")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "interaction_continuous.png"), dpi=300)
    plt.close()

    # 3.2 分组时差模型：画出 7 组时差的平均预测
    group_levels = df["Timezone_Group"].cat.categories
    mean_dur_c = 0  # 中心化后 0 = 平均飞行时长

    plt.figure(figsize=(9, 6))
    preds_group = []

    for g in group_levels:
        tmp = df[df["Timezone_Group"] == g]
        if len(tmp) == 0:
            preds_group.append(np.nan)
            continue
        df_pred = pd.DataFrame({
            "Sex_clean": ["Men"],
            "Travel_Duration_c": [mean_dur_c],
            "Timezone_Group": [g],
            "Athlete_ID": [tmp["Athlete_ID"].iloc[0]],
        })
        preds_group.append(res2["reml"].predict(df_pred)[0])

    plt.bar(group_levels, preds_group)
    plt.xticks(rotation=30)
    plt.ylabel("预测 Sinclair")
    plt.title("不同时差方向/幅度的影响（Model 2）")
    plt.savefig(os.path.join(OUTPUT_DIR, "interaction_grouped.png"), dpi=300)
    plt.close()


def compare_models(results):
    """模型比较与似然比检验"""
    with open(log_comparison, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n模型比较：连续时差 vs 分组时差\n" + "=" * 80 + "\n\n")

        m1 = results["Model1_Continuous"]
        m2 = results["Model2_Grouped"]

        if m1 is None or m2 is None:
            f.write("模型拟合失败，无法比较\n")
            return

        # AIC/BIC比较
        f.write("AIC / BIC：\n")
        f.write(f"  Model 1 (连续): AIC={m1['ml'].aic:.2f}, BIC={m1['ml'].bic:.2f}\n")
        f.write(f"  Model 2 (分组): AIC={m2['ml'].aic:.2f}, BIC={m2['ml'].bic:.2f}\n\n")

        # 似然比检验
        ll1 = m1["ml"].llf
        ll2 = m2["ml"].llf

        df1 = m1["n_params"]
        df2 = m2["n_params"]

        LR = 2 * (ll2 - ll1)
        df_diff = df2 - df1

        try:
            p_value = chi2.sf(LR, df_diff)
        except:
            p_value = np.nan

        f.write("似然比检验（LRT）：\n")
        f.write(f"  LR = {LR:.4f}, df = {df_diff}, p = {p_value:.5f}\n")
        if p_value < 0.05:
            f.write("  → 分组模型显著更好（p < 0.05）\n")
        else:
            f.write("  → 连续模型足够好（p ≥ 0.05）\n")

        f.write("\n" + "=" * 80 + "\n\n")

    logger.info(f"✓ 模型比较完成（见文件：{log_comparison}）")


# ==================== 绘图 =====================

def plot_comparisons(results, df, centering_info):
    """可视化两个模型：预测 vs 真实、残差分布、交互项效果"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 提取两个模型
    res1 = results.get("Model1_Continuous", None)
    res2 = results.get("Model2_Grouped", None)

    if res1 is None or res2 is None:
        logger.error("模型拟合失败，无法绘图")
        return

    # ---------------------------------------------------------
    # 1. 预测 vs 真实
    # ---------------------------------------------------------
    df["Pred_Model1"] = res1["reml"].fittedvalues
    df["Pred_Model2"] = res2["reml"].fittedvalues

    plt.figure(figsize=(8, 6))
    plt.scatter(df["Sinclair_Total"], df["Pred_Model1"], alpha=0.5)
    plt.plot([df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             [df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             linestyle="--")
    plt.xlabel("真实 Sinclair 总分")
    plt.ylabel("预测 Sinclair（Model 1: 连续时差）")
    plt.title("预测 vs 真实（Model 1）")
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_obs_model1.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df["Sinclair_Total"], df["Pred_Model2"], alpha=0.5)
    plt.plot([df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             [df["Sinclair_Total"].min(), df["Sinclair_Total"].max()],
             linestyle="--")
    plt.xlabel("真实 Sinclair 总分")
    plt.ylabel("预测 Sinclair（Model 2: 分组时差）")
    plt.title("预测 vs 真实（Model 2）")
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_obs_model2.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. 残差分布
    # ---------------------------------------------------------
    df["Residual_1"] = df["Sinclair_Total"] - df["Pred_Model1"]
    df["Residual_2"] = df["Sinclair_Total"] - df["Pred_Model2"]

    plt.figure(figsize=(10, 6))
    plt.hist(df["Residual_1"], bins=40, alpha=0.6, label="Model 1")
    plt.hist(df["Residual_2"], bins=40, alpha=0.6, label="Model 2")
    plt.xlabel("残差值")
    plt.ylabel("频数")
    plt.title("残差分布比较（Model 1 vs Model 2）")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. 交互项可视化：Travel × Timezone
    # ---------------------------------------------------------

    # 3.1 连续时差模型：绘制 3 条飞行时长水平
    dur_mean = centering_info["Travel_Duration"]["mean"]
    dur_vals = [dur_mean - 10, dur_mean, dur_mean + 10]  # 短、中、长航班
    tz_range = np.linspace(df["Timezone_Diff_hours"].min(),
                           df["Timezone_Diff_hours"].max(), 100)

    plt.figure(figsize=(8, 6))
    for d in dur_vals:
        input_df = pd.DataFrame({
            "Sex_clean": ["Men"] * len(tz_range),
            "Travel_Duration_c": d - centering_info["Travel_Duration"]["mean"],
            "Timezone_Diff_hours_c": tz_range - centering_info["Timezone_Diff_hours"]["mean"],
            "Athlete_ID": df["Athlete_ID"].iloc[0]  # dummy
        })
        preds = res1["reml"].predict(input_df)
        plt.plot(tz_range, preds, label=f"飞行时长 ≈ {d:.1f} 小时")

    plt.xlabel("时差（小时）")
    plt.ylabel("预测 Sinclair")
    plt.title("交互项效果（Model 1: 连续时差）")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "interaction_continuous.png"), dpi=300)
    plt.close()

    # 3.2 分组时差模型：画出 7 组时差的平均预测
    group_levels = df["Timezone_Group"].cat.categories
    mean_dur_c = 0  # 中心化后 0 = 平均飞行时长

    plt.figure(figsize=(9, 6))
    preds_group = []

    for g in group_levels:
        tmp = df[df["Timezone_Group"] == g]
        if len(tmp) == 0:
            preds_group.append(np.nan)
            continue
        df_pred = pd.DataFrame({
            "Sex_clean": ["Men"],
            "Travel_Duration_c": [mean_dur_c],
            "Timezone_Group": [g],
            "Athlete_ID": [tmp["Athlete_ID"].iloc[0]],
        })
        preds_group.append(res2["reml"].predict(df_pred)[0])

    plt.bar(group_levels, preds_group)
    plt.xticks(rotation=30)
    plt.ylabel("预测 Sinclair")
    plt.title("不同时差方向/幅度的影响（Model 2）")
    plt.savefig(os.path.join(OUTPUT_DIR, "interaction_grouped.png"), dpi=300)
    plt.close()



# ========================= 主程序入口 ========================

if __name__ == "__main__":
    iwf = load_iwf()
    travel = load_travel_table()
    merged = merge_tables(iwf, travel)
    results, df_model = run_two_models(merged)
