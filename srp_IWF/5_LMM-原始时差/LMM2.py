"""
lmm_with_travel_duration.py (修复笛卡尔积问题)

关键修复：
1. 合并时必须同时匹配 Nation、City、Year（三键匹配）
2. 避免笛卡尔积：一个运动员只对应一条行程记录
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

# 配置
IWF_FILE = "iwf_data_with_tz.csv" 
TRAVEL_FILE = "travel_time_merged.csv"
OUTPUT_DIR = "lmm_time"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志配置
log_path = os.path.join(OUTPUT_DIR, "analysis_log.txt")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_path, encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger("lmm_travel")
logger.info("======== LMM with Travel Duration (Fixed Cartesian Product): START ========")

# Sinclair 计算
def calculate_sinclair(total, body_weight, gender):
    try:
        total = float(total)
        bw = float(body_weight)
    except Exception:
        return np.nan

    params = {
        "Men": {"a": -0.794358141, "A": 174.393},
        "Women": {"a": -0.897260740, "A": 148.026}
    }

    if pd.isna(total) or pd.isna(bw) or bw <= 0:
        return np.nan

    g = str(gender).strip().lower() if not pd.isna(gender) else ""
    if g.startswith("m"):
        p = params["Men"]
    elif g.startswith("w"):
        p = params["Women"]
    else:
        p = params["Men"]

    A, a = p["A"], p["a"]
    coeff = 1.0 if bw > A else np.exp(a * (np.log(bw / A) ** 2))
    return total * coeff

# 读取 travel_time 表
def load_travel_table(travel_csv=TRAVEL_FILE):
    if not os.path.exists(travel_csv):
        logger.error(f"Travel 数据文件不存在：{travel_csv}")
        raise FileNotFoundError(travel_csv)
    
    logger.info(f"加载 travel CSV: {travel_csv}")
    tt = pd.read_csv(travel_csv, encoding="utf-8-sig")
    tt.columns = tt.columns.str.strip()
    
    col_mapper = {}
    for col in tt.columns:
        col_lower = col.lower()
        if "departure_nation" in col_lower:
            col_mapper[col] = "Departure_Nation"
        elif "departure_city" in col_lower or col_lower == "departure":
            col_mapper[col] = "Departure_City"
        elif "arrival_city" in col_lower or col_lower == "arrival":
            col_mapper[col] = "Arrival_City"
        elif "duration" in col_lower:
            col_mapper[col] = "Duration_Hours"
        elif col_lower in ["types", "type"]:
            col_mapper[col] = "Types"
        elif col_lower == "year":
            col_mapper[col] = "Year"
    
    tt = tt.rename(columns=col_mapper).dropna(how="all")
    
    if "Types" in tt.columns:
        map_types = {"直飞": 0, "一次中转": 1, "二次中转": 2, "三次中转": 3}
        tt["Types"] = tt["Types"].map(map_types).fillna(tt["Types"]).astype("Int64")
    
    if "Year" in tt.columns:
        tt["Year"] = pd.to_numeric(tt["Year"], errors="coerce").astype("Int64")
    
    if "Duration_Hours" in tt.columns:
        tt["Duration_Hours"] = pd.to_numeric(tt["Duration_Hours"], errors="coerce")
    
    logger.info(f"travel_time 表已加载，行数={len(tt)}, 列={list(tt.columns)}")
    return tt

# 读取 IWF 表
def load_iwf(iwf_path=IWF_FILE):
    if not os.path.exists(iwf_path):
        logger.error(f"IWF 数据文件不存在：{iwf_path}")
        raise FileNotFoundError(iwf_path)
    
    logger.info(f"加载 IWF 数据: {iwf_path}")
    iwf = pd.read_csv(iwf_path, encoding="utf-8-sig")
    iwf.columns = iwf.columns.str.strip()
    
    cols = {c.lower(): c for c in iwf.columns}
    rename_map = {}
    
    for key, new_name in [("nation", "Nation"), ("city", "Event_City"), 
                          ("total", "Total"), ("event_year", "Event_Year"),
                          ("timezone_diff_hours", "Timezone_Diff_hours")]:
        if key in cols:
            rename_map[cols[key]] = new_name
    
    for key in ["body_weight", "body weight", "bodyweight"]:
        if key in cols:
            rename_map[cols[key]] = "Body_Weight"
            break
    
    if "year" in cols and "event_year" not in cols:
        rename_map[cols["year"]] = "Event_Year"
    
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
    
    if "Timezone_Diff_hours" not in iwf.columns:
        if {"Athlete_UTC_offset", "Venue_UTC_offset"}.issubset(set(iwf.columns)):
            iwf["Timezone_Diff_hours"] = pd.to_numeric(iwf["Athlete_UTC_offset"], errors="coerce") - \
                                          pd.to_numeric(iwf["Venue_UTC_offset"], errors="coerce")
    
    logger.info(f"IWF 表已加载，行数={len(iwf)}, 列={list(iwf.columns)}")
    return iwf

# 规范化城市字符串
def normalize_city_country_string(s):
    if pd.isna(s): 
        return s
    return str(s).strip().replace(", ", ",")

# 合并表格（修复版：使用三键匹配）
def merge_tables(iwf, travel):
    logger.info("开始合并 travel_time 与 IWF 表（使用三键匹配避免笛卡尔积）")
    
    required_travel_cols = ["Departure_Nation", "Arrival_City", "Year", "Duration_Hours"]
    missing_cols = [col for col in required_travel_cols if col not in travel.columns]
    if missing_cols:
        logger.error(f"Travel 表缺少必要列: {missing_cols}")
        raise ValueError(f"Travel 表缺少列: {missing_cols}")
    
    # 规范化城市名
    travel["Arrival_City_norm"] = travel["Arrival_City"].apply(normalize_city_country_string)
    travel["Arrival_City_norm_lower"] = travel["Arrival_City_norm"].astype(str).str.lower().str.replace(" ", "")
    
    if "Event_City" in iwf.columns:
        iwf["Event_City_norm"] = iwf["Event_City"].apply(normalize_city_country_string)
    elif "City" in iwf.columns:
        iwf["Event_City_norm"] = iwf["City"].apply(normalize_city_country_string)
    else:
        raise ValueError("IWF 表缺少 Event_City 或 City 列")
    
    iwf["Event_City_norm_lower"] = iwf["Event_City_norm"].astype(str).str.lower().str.replace(" ", "")
    
    # ========== 关键修复：三键匹配 ==========
    # 必须同时匹配：Nation、Event_City、Event_Year
    logger.info("执行三键合并：Nation + Event_City + Event_Year")
    
    merged = iwf.merge(
        travel[["Departure_Nation", "Arrival_City_norm_lower", "Year", "Duration_Hours", "Types"]],
        left_on=["Nation", "Event_City_norm_lower", "Event_Year"],
        right_on=["Departure_Nation", "Arrival_City_norm_lower", "Year"],
        how="left",
        suffixes=("", "_travel")
    )
    
    logger.info(f"三键合并完成！")
    logger.info(f"  合并后总行数: {len(merged)}")
    logger.info(f"  Travel_Duration 有值的行数: {merged['Duration_Hours'].notna().sum()}")
    logger.info(f"  Travel_Duration 缺失的行数: {merged['Duration_Hours'].isna().sum()}")
    
    # 检查是否还有重复
    if "Name" in merged.columns:
        dup_check = merged.groupby(["Name", "Event_Year", "Event_City_norm"]).size()
        max_dup = dup_check.max()
        if max_dup > 1:
            logger.warning(f"警告：仍存在重复记录，最大重复次数={max_dup}")
            dup_examples = dup_check[dup_check > 1].head(5)
            logger.warning(f"重复示例：\n{dup_examples}")
        else:
            logger.info("✓ 验证通过：无重复记录（每个运动员-比赛组合唯一）")
    
    # 重命名
    merged["Travel_Duration"] = merged["Duration_Hours"]
    if "Types" in merged.columns:
        merged["Travel_Types"] = merged["Types"]
    
    return merged

# 运行 LMM 模型
def run_models_and_output(merged_df):
    # 计算 Sinclair_Total
    if "Sinclair_Total" not in merged_df.columns:
        logger.info("计算 Sinclair_Total ...")
        merged_df["Sinclair_Total"] = merged_df.apply(
            lambda r: calculate_sinclair(r.get("Total", np.nan), 
                                        r.get("Body_Weight", np.nan), 
                                        r.get("Sex", np.nan)), 
            axis=1
        )
        logger.info(f"Sinclair_Total 非空数 = {merged_df['Sinclair_Total'].notna().sum()}")
    
    # 规范化性别
    merged_df["Sex_clean"] = merged_df.get("Sex", pd.Series(np.nan, index=merged_df.index))
    merged_df["Sex_clean"] = merged_df["Sex_clean"].astype(str).str.strip().str.title()
    merged_df.loc[~merged_df["Sex_clean"].isin(["Men", "Women"]), "Sex_clean"] = np.nan
    
    # 准备建模数据
    model_cols = ["Sinclair_Total", "Sex_clean", "Travel_Duration", "Timezone_Diff_hours", "Nation"]
    model_df = merged_df[model_cols].copy()
    
    logger.info(f"准备建模数据：总行数={len(model_df)}")
    
    # 去除缺失
    model_df = model_df.dropna(subset=["Sinclair_Total", "Sex_clean", "Timezone_Diff_hours"])
    logger.info(f"去除基础缺失后的行数={len(model_df)}")
    
    complete_for_model = model_df.dropna(subset=["Travel_Duration"]).copy()
    partial = model_df[model_df["Travel_Duration"].isna()].copy()
    
    logger.info(f"Travel_Duration 有效行数={len(complete_for_model)}；缺失行数={len(partial)}")
    
    # 保存数据
    merged_csv = os.path.join(OUTPUT_DIR, "merged_full_with_travel.csv")
    model_csv = os.path.join(OUTPUT_DIR, "model_input_with_travel.csv")
    complete_for_model.to_csv(model_csv, index=False, encoding="utf-8-sig")
    merged_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    logger.info(f"已保存：{merged_csv} 和 {model_csv}")
    
    if len(complete_for_model) == 0:
        logger.error("没有任何记录包含 Travel_Duration，无法建模")
        return None, None, None, None
    
    complete_for_model["Nation"] = complete_for_model["Nation"].astype("category")
    
    # 过滤样本量太小的国家
    nation_counts = complete_for_model["Nation"].value_counts()
    logger.info(f"各国样本量统计：\n{nation_counts.head(20)}")
    valid_nations = nation_counts[nation_counts >= 3].index
    complete_for_model = complete_for_model[complete_for_model["Nation"].isin(valid_nations)].copy()
    complete_for_model["Nation"] = complete_for_model["Nation"].cat.remove_unused_categories()
    
    logger.info(f"过滤后用于建模的行数={len(complete_for_model)}，国家数={complete_for_model['Nation'].nunique()}")
    
    # --------------- Model 1 ---------------
    formula1 = "Sinclair_Total ~ Sex_clean + Travel_Duration + Timezone_Diff_hours"
    logger.info(f"开始拟合 Model 1: {formula1} with (1|Nation)")
    
    res1 = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model1 = smf.mixedlm(formula1, data=complete_for_model, 
                                groups=complete_for_model["Nation"],
                                re_formula="1")
            res1 = model1.fit(method="powell", maxiter=1000, reml=True)
        logger.info("Model 1 拟合完成")
        
        with open(os.path.join(OUTPUT_DIR, "model1_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Model 1: Sinclair_Total ~ Sex + Travel_Duration + Timezone_Diff + (1|Nation)\n\n")
            f.write(res1.summary().as_text())
        logger.info("Model 1 摘要已保存")
    except Exception as e:
        logger.exception(f"Model 1 拟合失败: {e}")
    
    # --------------- Model 2 (交互项) ---------------
    formula2 = "Sinclair_Total ~ Sex_clean + Travel_Duration * Timezone_Diff_hours"
    logger.info(f"开始拟合 Model 2: {formula2} with (1|Nation)")
    
    res2 = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model2 = smf.mixedlm(formula2, data=complete_for_model, 
                                groups=complete_for_model["Nation"],
                                re_formula="1")
            res2 = model2.fit(method="powell", maxiter=1000, reml=True)
        logger.info("Model 2 拟合完成")
        
        with open(os.path.join(OUTPUT_DIR, "model2_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Model 2: Sinclair_Total ~ Sex + Travel_Duration * Timezone_Diff + (1|Nation)\n\n")
            f.write(res2.summary().as_text())
        logger.info("Model 2 摘要已保存")
    except Exception as e:
        logger.exception(f"Model 2 拟合失败: {e}")
    
    # --------------- 绘图 ---------------
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1) Model 1 预测 vs 实际
    if res1 is not None:
        try:
            complete_for_model["pred1"] = res1.fittedvalues
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="pred1", y="Sinclair_Total", hue="Sex_clean", 
                          data=complete_for_model, alpha=0.5, s=20)
            min_val = min(complete_for_model["pred1"].min(), complete_for_model["Sinclair_Total"].min())
            max_val = max(complete_for_model["pred1"].max(), complete_for_model["Sinclair_Total"].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
            plt.xlabel("Model 1 Predicted Sinclair", fontsize=12)
            plt.ylabel("Observed Sinclair Total", fontsize=12)
            plt.title("Predicted vs Observed (Model 1)", fontsize=14, fontweight='bold')
            plt.legend(title="Sex")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_obs_model1.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("已保存 Model 1 预测图")
        except Exception as e:
            logger.exception(f"绘制 Model 1 预测图失败: {e}")
    
    # 2) Model 2 预测 vs 实际
    if res2 is not None:
        try:
            complete_for_model["pred2"] = res2.fittedvalues
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="pred2", y="Sinclair_Total", hue="Sex_clean", 
                          data=complete_for_model, alpha=0.5, s=20)
            min_val = min(complete_for_model["pred2"].min(), complete_for_model["Sinclair_Total"].min())
            max_val = max(complete_for_model["pred2"].max(), complete_for_model["Sinclair_Total"].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
            plt.xlabel("Model 2 Predicted Sinclair", fontsize=12)
            plt.ylabel("Observed Sinclair Total", fontsize=12)
            plt.title("Predicted vs Observed (Model 2)", fontsize=14, fontweight='bold')
            plt.legend(title="Sex")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_obs_model2.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("已保存 Model 2 预测图")
        except Exception as e:
            logger.exception(f"绘制 Model 2 预测图失败: {e}")
    
    # 3) 交互效果
    if res2 is not None:
        try:
            logger.info("绘制交互效果图...")
            params = res2.params
            intercept = params.get("Intercept", 0)
            sex_women = params.get("Sex_clean[T.Women]", 0)
            beta_travel = params.get("Travel_Duration", 0)
            beta_tz = params.get("Timezone_Diff_hours", 0)
            beta_interaction = params.get("Travel_Duration:Timezone_Diff_hours", 0)
            
            dur_pcts = complete_for_model["Travel_Duration"].quantile([0.1, 0.5, 0.9]).values
            tz_min = complete_for_model["Timezone_Diff_hours"].min()
            tz_max = complete_for_model["Timezone_Diff_hours"].max()
            tz_grid = np.linspace(tz_min, tz_max, 50)
            
            plt.figure(figsize=(12, 7))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dur_pcts)))
            
            for i, dur in enumerate(dur_pcts):
                pred_men = intercept + beta_travel * dur + beta_tz * tz_grid + beta_interaction * dur * tz_grid
                plt.plot(tz_grid, pred_men, label=f"Duration={dur:.1f}h (Men)", 
                        linestyle="-", linewidth=2.5, color=colors[i])
                
                pred_women = intercept + sex_women + beta_travel * dur + beta_tz * tz_grid + beta_interaction * dur * tz_grid
                plt.plot(tz_grid, pred_women, label=f"Duration={dur:.1f}h (Women)", 
                        linestyle="--", linewidth=2.5, color=colors[i])
            
            plt.xlabel("Timezone Difference (hours)", fontsize=13)
            plt.ylabel("Predicted Sinclair Total", fontsize=13)
            plt.title("Interaction Effect: Travel Duration × Timezone Difference\n(Model 2)", 
                     fontsize=14, fontweight='bold')
            plt.legend(fontsize=9, loc='best', ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "interaction_travel_timezone_model2.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("已保存交互效果图")
        except Exception as e:
            logger.exception(f"绘制交互图失败: {e}")
    
    # 4) 系数图
    def save_coef_plot(res, fname, title):
        try:
            params = res.params
            conf = res.conf_int()
            
            mask = ~params.index.str.contains("Group|Intercept", case=False, na=False)
            params_filtered = params[mask]
            conf_filtered = conf[mask]
            
            if len(params_filtered) == 0:
                return
            
            coef_df = pd.DataFrame({
                "term": params_filtered.index,
                "coef": params_filtered.values,
                "ci_low": conf_filtered[0].values,
                "ci_high": conf_filtered[1].values
            })
            
            coef_df["term"] = coef_df["term"].str.replace("Sex_clean[T.Women]", "Sex: Women (vs Men)")
            coef_df["term"] = coef_df["term"].str.replace("Travel_Duration", "Travel Duration")
            coef_df["term"] = coef_df["term"].str.replace("Timezone_Diff_hours", "Timezone Diff")
            coef_df["term"] = coef_df["term"].str.replace(":", " × ")
            
            fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(coef_df))))
            y_pos = np.arange(len(coef_df))
            colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df["coef"]]
            
            ax.barh(y_pos, coef_df["coef"], color=colors, alpha=0.7, height=0.6)
            
            for i, row in coef_df.iterrows():
                ax.plot([row["ci_low"], row["ci_high"]], [i, i], 'k-', linewidth=2.5, zorder=10)
                ax.plot([row["ci_low"], row["ci_low"]], [i-0.15, i+0.15], 'k-', linewidth=2.5, zorder=10)
                ax.plot([row["ci_high"], row["ci_high"]], [i-0.15, i+0.15], 'k-', linewidth=2.5, zorder=10)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coef_df["term"], fontsize=11)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Coefficient Estimate", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"coef_{fname}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"已保存系数图: coef_{fname}.png")
        except Exception as e:
            logger.exception(f"保存系数图失败 ({fname}): {e}")
    
    if res1 is not None:
        save_coef_plot(res1, "model1", "Model 1 Coefficients (with 95% CI)")
    if res2 is not None:
        save_coef_plot(res2, "model2", "Model 2 Coefficients (with 95% CI)")
    
    return res1, res2, complete_for_model, partial

# 主流程
def main():
    try:
        travel = load_travel_table()
        iwf = load_iwf()
        merged = merge_tables(iwf, travel)
        res1, res2, used_df, skipped_df = run_models_and_output(merged)
        
        # 写最终摘要
        summary_path = os.path.join(OUTPUT_DIR, "final_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("LMM with Travel Duration - Analysis Summary\n")
            f.write(f"Run time: {datetime.now().isoformat()}\n\n")
            f.write("=" * 60 + "\n")
            f.write("Models:\n")
            f.write("  Model 1: Sinclair_Total ~ Sex + Travel_Duration + Timezone_Diff + (1|Nation)\n")
            f.write("  Model 2: Sinclair_Total ~ Sex + Travel_Duration * Timezone_Diff + (1|Nation)\n\n")
            f.write("=" * 60 + "\n")
            f.write("Data Summary:\n")
            f.write(f"  Total rows in merged data: {len(merged)}\n")
            if used_df is not None:
                f.write(f"  Rows used for modeling: {len(used_df)}\n")
            if skipped_df is not None:
                f.write(f"  Rows skipped (missing Travel_Duration): {len(skipped_df)}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Output Files:\n")
            f.write("  - model1_summary.txt: Model 1 detailed results\n")
            f.write("  - model2_summary.txt: Model 2 detailed results\n")
            f.write("  - pred_vs_obs_model1.png: Model 1 predictions\n")
            f.write("  - pred_vs_obs_model2.png: Model 2 predictions\n")
            f.write("  - interaction_travel_timezone_model2.png: Interaction effects\n")
            f.write("  - coef_model1.png: Model 1 coefficients\n")
            f.write("  - coef_model2.png: Model 2 coefficients\n")
            f.write("  - merged_full_with_travel.csv: Complete merged dataset\n")
            f.write("  - model_input_with_travel.csv: Model input data\n")
        
        logger.info(f"分析完成！结果已保存到 {OUTPUT_DIR} 目录")
        logger.info("======== LMM with Travel Duration: END ========")
        
    except Exception as e:
        logger.exception("分析过程中发生错误")
        raise

if __name__ == "__main__":
    main()