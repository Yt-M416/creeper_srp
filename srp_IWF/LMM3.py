"""
LMM with Mean-Centered Variables
均值中心化 + 似然比检验（LRT）
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
import warnings
from scipy.stats import chi2

# 配置
IWF_FILE = "iwf_data_with_tz.csv" 
TRAVEL_FILE = "travel_time_merged.csv"
OUTPUT_DIR = "lmm_time_centered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志配置
log_path = os.path.join(OUTPUT_DIR, "analysis_log.txt")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_path, encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger("lmm_centered")
logger.info("=" * 70)
logger.info("LMM with Mean-Centered Variables: START")
logger.info("=" * 70)

# ====== 函数定义 ======
def calculate_sinclair(total, body_weight, gender):
    try:
        total, bw = float(total), float(body_weight)
    except:
        return np.nan
    params = {
        "Men": {"a": -0.794358141, "A": 174.393},
        "Women": {"a": -0.897260740, "A": 148.026}
    }
    if pd.isna(total) or pd.isna(bw) or bw <= 0:
        return np.nan
    g = str(gender).strip().lower() if not pd.isna(gender) else ""
    p = params["Men"] if g.startswith("m") else params["Women"] if g.startswith("w") else params["Men"]
    A, a = p["A"], p["a"]
    coeff = 1.0 if bw > A else np.exp(a * (np.log(bw / A) ** 2))
    return total * coeff

def load_travel_table(travel_csv=TRAVEL_FILE):
    logger.info(f"加载 travel CSV: {travel_csv}")
    tt = pd.read_csv(travel_csv, encoding="utf-8-sig")
    tt.columns = tt.columns.str.strip()
    col_mapper = {}
    for col in tt.columns:
        col_lower = col.lower()
        if "departure_nation" in col_lower: col_mapper[col] = "Departure_Nation"
        elif "departure_city" in col_lower or col_lower == "departure": col_mapper[col] = "Departure_City"
        elif "arrival_city" in col_lower or col_lower == "arrival": col_mapper[col] = "Arrival_City"
        elif "duration" in col_lower: col_mapper[col] = "Duration_Hours"
        elif col_lower in ["types", "type"]: col_mapper[col] = "Types"
        elif col_lower == "year": col_mapper[col] = "Year"
    tt = tt.rename(columns=col_mapper).dropna(how="all")
    if "Types" in tt.columns:
        tt["Types"] = tt["Types"].map({"直飞": 0, "一次中转": 1, "二次中转": 2, "三次中转": 3}).fillna(tt["Types"]).astype("Int64")
    if "Year" in tt.columns:
        tt["Year"] = pd.to_numeric(tt["Year"], errors="coerce").astype("Int64")
    if "Duration_Hours" in tt.columns:
        tt["Duration_Hours"] = pd.to_numeric(tt["Duration_Hours"], errors="coerce")
    logger.info(f"travel_time 表已加载，行数={len(tt)}")
    return tt

def load_iwf(iwf_path=IWF_FILE):
    logger.info(f"加载 IWF 数据: {iwf_path}")
    iwf = pd.read_csv(iwf_path, encoding="utf-8-sig")
    iwf.columns = iwf.columns.str.strip()
    cols = {c.lower(): c for c in iwf.columns}
    rename_map = {}
    for key, new_name in [("nation", "Nation"), ("city", "Event_City"), ("total", "Total"), 
                          ("event_year", "Event_Year"), ("timezone_diff_hours", "Timezone_Diff_hours")]:
        if key in cols: rename_map[cols[key]] = new_name
    for key in ["body_weight", "body weight", "bodyweight"]:
        if key in cols: rename_map[cols[key]] = "Body_Weight"; break
    if "year" in cols and "event_year" not in cols: rename_map[cols["year"]] = "Event_Year"
    iwf = iwf.rename(columns=rename_map)
    if "Event_Year" in iwf.columns:
        iwf["Event_Year"] = pd.to_numeric(iwf["Event_Year"], errors="coerce").astype("Int64")
    if "Total" in iwf.columns:
        iwf["Total"] = pd.to_numeric(iwf["Total"], errors="coerce")
    if "Body_Weight" in iwf.columns:
        iwf["Body_Weight"] = pd.to_numeric(iwf["Body_Weight"], errors="coerce")
    if "Event" in iwf.columns and "Sex" not in iwf.columns:
        iwf["Sex"] = iwf["Event"].astype(str).str.extract(r"(Men|Women|men|women)", expand=False)
    if "Sex" in iwf.columns:
        iwf["Sex"] = iwf["Sex"].astype("string")
    if "Timezone_Diff_hours" not in iwf.columns and {"Athlete_UTC_offset", "Venue_UTC_offset"}.issubset(set(iwf.columns)):
        iwf["Timezone_Diff_hours"] = pd.to_numeric(iwf["Athlete_UTC_offset"], errors="coerce") - pd.to_numeric(iwf["Venue_UTC_offset"], errors="coerce")
    logger.info(f"IWF 表已加载，行数={len(iwf)}")
    return iwf

def normalize_city_country_string(s):
    return str(s).strip().replace(", ", ",") if not pd.isna(s) else s

def merge_tables(iwf, travel):
    logger.info("开始三键合并...")
    travel["Arrival_City_norm_lower"] = travel["Arrival_City"].apply(normalize_city_country_string).astype(str).str.lower().str.replace(" ", "")
    iwf["Event_City_norm"] = (iwf["Event_City"] if "Event_City" in iwf.columns else iwf["City"]).apply(normalize_city_country_string)
    iwf["Event_City_norm_lower"] = iwf["Event_City_norm"].astype(str).str.lower().str.replace(" ", "")
    
    merged = iwf.merge(
        travel[["Departure_Nation", "Arrival_City_norm_lower", "Year", "Duration_Hours", "Types"]],
        left_on=["Nation", "Event_City_norm_lower", "Event_Year"],
        right_on=["Departure_Nation", "Arrival_City_norm_lower", "Year"],
        how="left", suffixes=("", "_travel")
    )
    merged["Travel_Duration"] = merged["Duration_Hours"]
    if "Types" in merged.columns: merged["Travel_Types"] = merged["Types"]
    logger.info(f"合并完成，总行数={len(merged)}，Travel_Duration 有值={merged['Travel_Duration'].notna().sum()}")
    return merged

def apply_mean_centering(df, variables):
    """均值中心化"""
    logger.info("\n" + "=" * 70)
    logger.info("应用均值中心化（Grand Mean Centering）")
    logger.info("=" * 70)
    centering_info = {}
    for var in variables:
        if var not in df.columns:
            logger.warning(f"变量 {var} 不存在，跳过")
            continue
        mean_val, std_val = df[var].mean(), df[var].std()
        centered_var = f"{var}_c"
        df[centered_var] = df[var] - mean_val
        centering_info[var] = {"mean": mean_val, "std": std_val, "centered_var": centered_var}
        logger.info(f"\n{var}:")
        logger.info(f"  原始均值: {mean_val:.4f}")
        logger.info(f"  原始标准差: {std_val:.4f}")
        logger.info(f"  中心化后变量: {centered_var}")
        logger.info(f"  中心化后均值: {df[centered_var].mean():.10f} (≈0)")
    return df, centering_info

def run_models_with_centering(merged_df):
    # 计算 Sinclair
    if "Sinclair_Total" not in merged_df.columns:
        logger.info("计算 Sinclair_Total...")
        merged_df["Sinclair_Total"] = merged_df.apply(
            lambda r: calculate_sinclair(r.get("Total", np.nan), r.get("Body_Weight", np.nan), r.get("Sex", np.nan)), axis=1)
    
    # 规范化性别
    merged_df["Sex_clean"] = merged_df.get("Sex", pd.Series(np.nan, index=merged_df.index))
    merged_df["Sex_clean"] = merged_df["Sex_clean"].astype(str).str.strip().str.title()
    merged_df.loc[~merged_df["Sex_clean"].isin(["Men", "Women"]), "Sex_clean"] = np.nan
    
    # 准备数据
    model_df = merged_df[["Sinclair_Total", "Sex_clean", "Travel_Duration", "Timezone_Diff_hours", "Nation"]].copy()
    model_df = model_df.dropna(subset=["Sinclair_Total", "Sex_clean", "Timezone_Diff_hours"])
    complete_for_model = model_df.dropna(subset=["Travel_Duration"]).copy()
    logger.info(f"\n建模数据：行数={len(complete_for_model)}，国家数={complete_for_model['Nation'].nunique()}")
    
    # 中心化
    complete_for_model, centering_info = apply_mean_centering(complete_for_model, ["Travel_Duration", "Timezone_Diff_hours"])
    
    # 过滤小样本国家
    complete_for_model["Nation"] = complete_for_model["Nation"].astype("category")
    nation_counts = complete_for_model["Nation"].value_counts()
    valid_nations = nation_counts[nation_counts >= 3].index
    complete_for_model = complete_for_model[complete_for_model["Nation"].isin(valid_nations)].copy()
    complete_for_model["Nation"] = complete_for_model["Nation"].cat.remove_unused_categories()
    logger.info(f"过滤后：行数={len(complete_for_model)}，国家数={complete_for_model['Nation'].nunique()}")
    
    # 保存
    complete_for_model.to_csv(os.path.join(OUTPUT_DIR, "model_input_centered.csv"), index=False, encoding="utf-8-sig")
    
    # ====== 拟合模型 ======
    logger.info("\n" + "=" * 70)
    logger.info("拟合模型（中心化变量）")
    logger.info("=" * 70)
    
    # Model 1
    formula1 = "Sinclair_Total ~ Sex_clean + Travel_Duration_c + Timezone_Diff_hours_c"
    logger.info(f"\nModel 1: {formula1} + (1|Nation)")
    res1_reml, res1_ml = None, None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model1 = smf.mixedlm(formula1, data=complete_for_model, groups=complete_for_model["Nation"], re_formula="1")
            res1_reml = model1.fit(method="powell", maxiter=1000, reml=True)
            res1_ml = model1.fit(method="powell", maxiter=1000, reml=False)
        logger.info(f"✓ Model 1 完成 | REML LogLik={res1_reml.llf:.4f}, ML LogLik={res1_ml.llf:.4f}")
        with open(os.path.join(OUTPUT_DIR, "model1_centered_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Model 1 (Mean-Centered)\n" + "=" * 70 + "\n")
            f.write(f"Formula: {formula1} + (1|Nation)\n\nCentering Info:\n")
            for var, info in centering_info.items():
                f.write(f"  {var}: mean={info['mean']:.4f}, std={info['std']:.4f}\n")
            f.write("\n" + res1_reml.summary().as_text())
    except Exception as e:
        logger.exception(f"Model 1 失败: {e}")
    
    # Model 2
    formula2 = "Sinclair_Total ~ Sex_clean + Travel_Duration_c * Timezone_Diff_hours_c"
    logger.info(f"\nModel 2: {formula2} + (1|Nation)")
    res2_reml, res2_ml = None, None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model2 = smf.mixedlm(formula2, data=complete_for_model, groups=complete_for_model["Nation"], re_formula="1")
            res2_reml = model2.fit(method="powell", maxiter=1000, reml=True)
            res2_ml = model2.fit(method="powell", maxiter=1000, reml=False)
        logger.info(f"✓ Model 2 完成 | REML LogLik={res2_reml.llf:.4f}, ML LogLik={res2_ml.llf:.4f}")
        with open(os.path.join(OUTPUT_DIR, "model2_centered_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Model 2 (Mean-Centered with Interaction)\n" + "=" * 70 + "\n")
            f.write(f"Formula: {formula2} + (1|Nation)\n\nCentering Info:\n")
            for var, info in centering_info.items():
                f.write(f"  {var}: mean={info['mean']:.4f}, std={info['std']:.4f}\n")
            f.write("\n" + res2_reml.summary().as_text())
    except Exception as e:
        logger.exception(f"Model 2 失败: {e}")
    
    # ====== LRT ======
    if res1_ml and res2_ml:
        logger.info("\n" + "=" * 70)
        logger.info("似然比检验（LRT）")
        logger.info("=" * 70)
        lr_stat = -2 * (res1_ml.llf - res2_ml.llf)
        p_value = chi2.sf(lr_stat, df=1)
        logger.info(f"\nH0: Model 1 足够 | H1: Model 2 更好")
        logger.info(f"  Model 1 ML LogLik: {res1_ml.llf:.4f}")
        logger.info(f"  Model 2 ML LogLik: {res2_ml.llf:.4f}")
        logger.info(f"  LR Statistic: {lr_stat:.4f}, df=1, p={p_value:.4f}")
        logger.info(f"\n{'✓ Model 2 显著更好 (p<0.05)' if p_value < 0.05 else '✗ Model 1 足够 (p≥0.05)'}")
        with open(os.path.join(OUTPUT_DIR, "likelihood_ratio_test.txt"), "w", encoding="utf-8") as f:
            f.write("Likelihood Ratio Test\n" + "=" * 70 + "\n")
            f.write(f"Model 1 ML LogLik: {res1_ml.llf:.6f}\nModel 2 ML LogLik: {res2_ml.llf:.6f}\n")
            f.write(f"LR Statistic: {lr_stat:.6f}, df=1, p-value={p_value:.6f}\n")
            f.write("Conclusion: " + ("Model 2 significantly better" if p_value < 0.05 else "Model 1 sufficient"))
    
    # 绘图
    plot_models(complete_for_model, res1_reml, res2_reml, centering_info)
    return res1_reml, res2_reml, complete_for_model, centering_info

def plot_models(data, res1, res2, centering_info):
    logger.info("\n绘制图表...")
    sns.set_style("whitegrid")
    
    # 系数图
    if res1 and res2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (res, title) in enumerate([(res1, "Model 1 (Main Effects)"), (res2, "Model 2 (with Interaction)")]):
            ax = axes[idx]
            params, conf = res.params, res.conf_int()
            mask = ~params.index.str.contains("Group|Intercept", case=False, na=False)
            params_f, conf_f = params[mask], conf[mask]
            coef_df = pd.DataFrame({"term": params_f.index, "coef": params_f.values, 
                                   "ci_low": conf_f[0].values, "ci_high": conf_f[1].values})
            coef_df["term"] = coef_df["term"].str.replace("Sex_clean[T.Women]", "Sex: Women")
            coef_df["term"] = coef_df["term"].str.replace("Travel_Duration_c", "Travel Duration (c)")
            coef_df["term"] = coef_df["term"].str.replace("Timezone_Diff_hours_c", "Timezone Diff (c)")
            coef_df["term"] = coef_df["term"].str.replace(":", " × ")
            y_pos = np.arange(len(coef_df))
            colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df["coef"]]
            ax.barh(y_pos, coef_df["coef"], color=colors, alpha=0.7, height=0.6)
            for i, row in coef_df.iterrows():
                ax.plot([row["ci_low"], row["ci_high"]], [i, i], 'k-', linewidth=2.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coef_df["term"], fontsize=11)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Coefficient", fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "coefficients_centered.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ 已保存: coefficients_centered.png")
    
    # 交互图
    if res2:
        try:
            params = res2.params
            intercept = params.get("Intercept", 0)
            sex_women = params.get("Sex_clean[T.Women]", 0)
            beta_travel = params.get("Travel_Duration_c", 0)
            beta_tz = params.get("Timezone_Diff_hours_c", 0)
            beta_interaction = params.get("Travel_Duration_c:Timezone_Diff_hours_c", 0)
            
            dur_pcts_c = data["Travel_Duration_c"].quantile([0.1, 0.5, 0.9]).values
            tz_min_c, tz_max_c = data["Timezone_Diff_hours_c"].min(), data["Timezone_Diff_hours_c"].max()
            tz_grid_c = np.linspace(tz_min_c, tz_max_c, 50)
            
            travel_mean = centering_info["Travel_Duration"]["mean"]
            tz_mean = centering_info["Timezone_Diff_hours"]["mean"]
            dur_pcts_orig = dur_pcts_c + travel_mean
            tz_grid_orig = tz_grid_c + tz_mean
            
            plt.figure(figsize=(12, 7))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dur_pcts_c)))
            
            for i, (dur_c, dur_orig) in enumerate(zip(dur_pcts_c, dur_pcts_orig)):
                pred_men = intercept + beta_travel * dur_c + beta_tz * tz_grid_c + beta_interaction * dur_c * tz_grid_c
                plt.plot(tz_grid_orig, pred_men, label=f"Duration={dur_orig:.1f}h (Men)", 
                        linestyle="-", linewidth=2.5, color=colors[i])
                pred_women = intercept + sex_women + beta_travel * dur_c + beta_tz * tz_grid_c + beta_interaction * dur_c * tz_grid_c
                plt.plot(tz_grid_orig, pred_women, label=f"Duration={dur_orig:.1f}h (Women)", 
                        linestyle="--", linewidth=2.5, color=colors[i])
            
            plt.xlabel("Timezone Difference (hours)", fontsize=13)
            plt.ylabel("Predicted Sinclair Total", fontsize=13)
            plt.title("Interaction: Travel Duration × Timezone Diff (Centered)", fontsize=14, fontweight='bold')
            plt.legend(fontsize=9, loc='best', ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "interaction_centered.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✓ 已保存: interaction_centered.png")
        except Exception as e:
            logger.warning(f"交互图失败: {e}")

def main():
    try:
        travel = load_travel_table()
        iwf = load_iwf()
        merged = merge_tables(iwf, travel)
        res1, res2, data, centering_info = run_models_with_centering(merged)
        
        with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), "w", encoding="utf-8") as f:
            f.write("LMM Analysis with Mean-Centered Variables\n" + "=" * 70 + "\n")
            f.write(f"Run time: {datetime.now().isoformat()}\n\nCentering Info:\n")
            for var, info in centering_info.items():
                f.write(f"  {var}: mean={info['mean']:.4f}, std={info['std']:.4f}\n")
            f.write("\nOutput Files:\n")
            f.write("  - model1_centered_summary.txt\n  - model2_centered_summary.txt\n")
            f.write("  - likelihood_ratio_test.txt\n  - coefficients_centered.png\n  - interaction_centered.png\n")
        
        logger.info("\n" + "=" * 70)
        logger.info("分析完成！结果保存在 " + OUTPUT_DIR)
        logger.info("=" * 70)
    except Exception as e:
        logger.exception("分析失败")
        raise

if __name__ == "__main__":
    main()