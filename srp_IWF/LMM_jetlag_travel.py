"""
LMM_jetlag_travel.py
在 LMM_jetlag.py 的基础上引入旅行时长（TravelDuration）变量，支持两种角色：

  --mode control : TravelDuration 作为协变量（控制变量），核心预测仍是 JetLagGroup_dir
                   公式：Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir) + TravelDuration_c
                   回答：控制旅行时长后，方向性时差组效应是否依然存在？

  --mode main    : TravelDuration 作为主效应，替换分组变量
                   公式：Sinclair_Total ~ C(Sex) + TravelDuration_c
                   回答：旅行时长本身是否显著影响成绩？

TravelDuration_c = TravelDuration - mean(TravelDuration)（中心化）
  → 中心化后截距有实际意义（平均旅行时长下的预测值），系数单位为"每小时"

用法：
    python LMM_jetlag_travel.py --mode control
    python LMM_jetlag_travel.py --mode main

输入：
    athlete_top13_travel.csv   （由 merge_travel.py 生成）

输出目录：
    lmm_jetlag_travel_control/
    lmm_jetlag_travel_main/
"""

from __future__ import annotations

import os
import re
import argparse
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ======================== 配置 ========================
INPUT_CSV = r"D:\Code\Python\Creeper\srp_IWF\athlete_top13_travel.csv"

# 与 LMM_jetlag.py 一致：2h bins 的标签按整数小时记为「下界-(上界-1)h」，如 [0,2)→Minimal(0-1h)
DIRECTIONAL_GROUP_ORDER = [
    "Minimal(0-1h)",
    "West_2-3h", "West_4-5h", "West_6-7h", "West_8-9h", "West_10-11h",
    "East_2-3h", "East_4-5h", "East_6-7h", "East_8-9h", "East_10-11h",
]

# ======================== 工具函数（与 LMM_jetlag.py 一致）========================

def setup_plot_style() -> None:
    try:
        from matplotlib import font_manager
        for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.sans-serif"] = [
                    font_manager.FontProperties(fname=fp).get_name(), "Arial"
                ]
                break
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font_scale=1.1)


def calculate_sinclair(total, body_weight, sex) -> float:
    try:
        total = float(total); bw = float(body_weight)
    except Exception:
        return np.nan
    params = {"Men": {"A": 0.722762521, "b": 193.609}, "Women": {"A": 0.787004341, "b": 153.757}}
    s = str(sex).strip().title()
    if s not in params or pd.isna(total) or pd.isna(bw) or bw <= 0:
        return np.nan
    A, b = params[s]["A"], params[s]["b"]
    sc = 1.0 if bw > b else 10 ** (A * np.log10(bw / b) ** 2)
    return total * sc


def extract_sex_from_event(event) -> Optional[str]:
    if not isinstance(event, str):
        return None
    m = re.search(r"(Men|Women)", event, re.IGNORECASE)
    return m.group(1).title() if m else None


def circular_jetlag_signed(venue_off: float, athlete_off: float) -> float:
    if pd.isna(venue_off) or pd.isna(athlete_off):
        return np.nan
    d = float(venue_off) - float(athlete_off)
    return d if abs(d) <= 12 else -np.sign(d) * (24 - abs(d))


def build_jetlag_group_directional(jetlag_signed: float) -> Optional[str]:
    """
    与 LMM_jetlag.py 相同规则；分组边界不变，仅标签字符串按老师要求书写。
    """
    if pd.isna(jetlag_signed):
        return None
    x = float(jetlag_signed)
    absx = abs(x)
    if absx < 2:
        return "Minimal(0-1h)"
    if x > 0:
        if absx < 4:   return "East_2-3h"
        if absx < 6:   return "East_4-5h"
        if absx < 8:   return "East_6-7h"
        if absx < 10:  return "East_8-9h"
        return "East_10-11h"
    else:
        if absx < 4:   return "West_2-3h"
        if absx < 6:   return "West_4-5h"
        if absx < 8:   return "West_6-7h"
        if absx < 10:  return "West_8-9h"
        return "West_10-11h"


def compute_icc(result) -> float:
    try:
        gv = float(result.cov_re.iloc[0, 0])
        rv = float(result.scale)
        return gv / (gv + rv) if (gv + rv) > 0 else np.nan
    except Exception:
        return np.nan


def log_write(fp, msg: str) -> None:
    fp.write(msg + "\n")
    fp.flush()


def sig_stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    if p < 0.1:    return "."
    return ""


def lrt(res_full, res_reduced) -> tuple[float, int, float]:
    """
    似然比检验（须均用 ML 拟合）。
    返回 (LR 统计量, df, p值)。
    LR = 2*(llf_full - llf_reduced)，近似服从 chi^2(df)。
    """
    lr_stat = 2.0 * (float(res_full.llf) - float(res_reduced.llf))
    df_diff = int(res_full.k_fe - res_reduced.k_fe)
    if lr_stat < 0:
        return lr_stat, df_diff, np.nan
    return lr_stat, df_diff, float(chi2.sf(lr_stat, df_diff))


# ======================== 主流程 ========================

def main() -> None:
    parser = argparse.ArgumentParser(description="LMM with TravelDuration")
    parser.add_argument(
        "--mode", type=str, default="control",
        choices=["control", "main"],
        help="control: travel as covariate; main: travel as main effect",
    )
    args = parser.parse_args()

    OUTPUT_DIR = f"lmm_jetlag_travel_{args.mode}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    setup_plot_style()
    log_path     = os.path.join(OUTPUT_DIR, "analysis_log.txt")
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
    compare_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")

    with open(log_path, "w", encoding="utf-8") as lf:
        log_write(lf, "=" * 80)
        log_write(lf, f"LMM_jetlag_travel  mode={args.mode}  - START")
        log_write(lf, f"Run time: {datetime.now().isoformat(timespec='seconds')}")
        log_write(lf, "=" * 80)

        # -------- STEP 1: 读取与基础清洗 --------
        if not os.path.exists(INPUT_CSV):
            raise FileNotFoundError(f"请先运行 merge_travel.py 生成：{INPUT_CSV}")

        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        log_write(lf, f"Loaded: {INPUT_CSV}")
        log_write(lf, f"Rows: {len(df)}, Columns: {len(df.columns)}")

        required = ["Total", "Body_Weight", "Event", "Athlete_ID",
                    "Athlete_UTC_offset", "Venue_UTC_offset", "TravelDuration"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列：{missing_cols}")

        for col in ["Total", "Body_Weight", "Athlete_UTC_offset",
                    "Venue_UTC_offset", "TravelDuration"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Sex"] = df["Event"].apply(extract_sex_from_event)
        n0 = len(df)
        df = df.dropna(subset=["Total", "Body_Weight", "Sex", "Athlete_ID",
                                "Athlete_UTC_offset", "Venue_UTC_offset"]).copy()
        log_write(lf, f"STEP 1 - Drop NA (core): {n0} → {len(df)}")

        # -------- STEP 2: JetLag + 分组 --------
        df["JetLag_signed"] = df.apply(
            lambda r: circular_jetlag_signed(r["Venue_UTC_offset"], r["Athlete_UTC_offset"]),
            axis=1,
        )
        df["JetLagGroup_dir"] = df["JetLag_signed"].apply(build_jetlag_group_directional)
        n1 = len(df)
        df = df.dropna(subset=["JetLagGroup_dir"]).copy()
        df["JetLagGroup_dir"] = pd.Categorical(
            df["JetLagGroup_dir"], categories=DIRECTIONAL_GROUP_ORDER, ordered=True
        )
        log_write(lf, f"STEP 2 - JetLag computed; drop NA: {n1} → {len(df)}")

        # -------- STEP 3: Sinclair --------
        df["Sinclair_Total"] = df.apply(
            lambda r: calculate_sinclair(r["Total"], r["Body_Weight"], r["Sex"]), axis=1
        )
        n2 = len(df)
        df = df.dropna(subset=["Sinclair_Total"]).copy()
        log_write(lf, f"STEP 3 - Sinclair computed; drop NA: {n2} → {len(df)}")

        # -------- STEP 4: TravelDuration 处理 --------
        n_miss_travel = df["TravelDuration"].isna().sum()
        log_write(lf, f"\nSTEP 4 - TravelDuration missing before drop: {n_miss_travel} / {len(df)}")

        # 只保留有旅行时长记录的行
        df = df.dropna(subset=["TravelDuration"]).copy()
        log_write(lf, f"  After dropping missing TravelDuration: {len(df)} rows")
        log_write(lf, f"  Athletes: {df['Athlete_ID'].nunique()}")

        # 中心化（减去均值）：截距 = 平均旅行时长下的预测值；系数单位 = 每小时
        travel_mean = df["TravelDuration"].mean()
        travel_std  = df["TravelDuration"].std()
        df["TravelDuration_c"] = df["TravelDuration"] - travel_mean
        log_write(lf, f"  TravelDuration  mean={travel_mean:.2f}h  std={travel_std:.2f}h")
        log_write(lf, f"  TravelDuration_c = TravelDuration - {travel_mean:.4f}  (centered, unit=h)")

        log_write(lf, f"\n  JetLagGroup_dir counts (after TravelDuration drop):")
        log_write(lf, df["JetLagGroup_dir"].value_counts().sort_index().to_string())

        df["Sex"]        = df["Sex"].astype("category")
        df["Athlete_ID"] = df["Athlete_ID"].astype("category")

        # -------- STEP 5: 拟合模型 --------
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 5 - Model fitting")
        log_write(lf, "=" * 80)

        if args.mode == "control":
            # Full：JetLagGroup_dir + TravelDuration_c（两者均在）
            formula_full    = "Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir) + TravelDuration_c"
            # Base B（无 travel）：用于检验 travel 是否显著改善拟合
            formula_base_b  = "Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir)"
            # Reduced（无 JetLag）：用于检验 JetLag 在控制 travel 后是否仍显著
            formula_no_jlag = "Sinclair_Total ~ C(Sex) + TravelDuration_c"

            log_write(lf, f"\n[Control mode]")
            log_write(lf, f"  Full    : {formula_full}")
            log_write(lf, f"  Base B  : {formula_base_b}  (no TravelDuration)")
            log_write(lf, f"  No-JLag : {formula_no_jlag}  (no JetLagGroup_dir)")

            res_full_reml   = smf.mixedlm(formula_full,    df, groups=df["Athlete_ID"]).fit(method="lbfgs")
            res_full_ml     = smf.mixedlm(formula_full,    df, groups=df["Athlete_ID"]).fit(method="lbfgs", reml=False)
            res_base_b_ml   = smf.mixedlm(formula_base_b,  df, groups=df["Athlete_ID"]).fit(method="lbfgs", reml=False)
            res_no_jlag_ml  = smf.mixedlm(formula_no_jlag, df, groups=df["Athlete_ID"]).fit(method="lbfgs", reml=False)

            icc_full  = compute_icc(res_full_reml)
            formula_report = formula_full

        else:  # main
            formula_full   = "Sinclair_Total ~ C(Sex) + TravelDuration_c"
            formula_reduced = "Sinclair_Total ~ C(Sex)"

            log_write(lf, f"\n[Main effect mode]")
            log_write(lf, f"  Full    : {formula_full}")
            log_write(lf, f"  Reduced : {formula_reduced}  (no TravelDuration)")

            res_full_reml   = smf.mixedlm(formula_full,    df, groups=df["Athlete_ID"]).fit(method="lbfgs")
            res_full_ml     = smf.mixedlm(formula_full,    df, groups=df["Athlete_ID"]).fit(method="lbfgs", reml=False)
            res_reduced_ml  = smf.mixedlm(formula_reduced, df, groups=df["Athlete_ID"]).fit(method="lbfgs", reml=False)

            icc_full = compute_icc(res_full_reml)
            formula_report = formula_full

        # -------- STEP 6: LRT --------
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 6 - Likelihood Ratio Tests (asymptotic chi-square, ML fits)")
        log_write(lf, "=" * 80)

        comp_lines = []

        if args.mode == "control":
            # LRT-A：TravelDuration 是否显著改善 Model B？
            lr_a, df_a, p_a = lrt(res_full_ml, res_base_b_ml)
            log_write(lf, "\nLRT-A: Full (JetLag + Travel) vs Base-B (JetLag only)")
            log_write(lf, f"  H0: TravelDuration_c coefficient = 0")
            log_write(lf, f"  LR={lr_a:.4f}  df={df_a}  p={p_a:.4e}" if not pd.isna(p_a) else f"  LR={lr_a:.4f}  df={df_a}  p=NA")
            if not pd.isna(p_a):
                log_write(lf, f"  → TravelDuration {'is a significant covariate' if p_a < 0.05 else 'is NOT a significant covariate'} (p {'< .05' if p_a < 0.05 else '>= .05'})")

            # LRT-B：控制 travel 后 JetLagGroup_dir 是否仍显著？
            lr_b, df_b, p_b = lrt(res_full_ml, res_no_jlag_ml)
            log_write(lf, "\nLRT-B: Full (JetLag + Travel) vs No-JLag (Travel only)")
            log_write(lf, f"  H0: all JetLagGroup_dir coefficients = 0 (given TravelDuration in model)")
            log_write(lf, f"  LR={lr_b:.4f}  df={df_b}  p={p_b:.4e}" if not pd.isna(p_b) else f"  LR={lr_b:.4f}  df={df_b}  p=NA")
            if not pd.isna(p_b):
                log_write(lf, f"  → JetLagGroup_dir {'remains significant' if p_b < 0.05 else 'is NOT significant'} after controlling TravelDuration")

            comp_lines += [
                "=" * 90,
                f"Mode: control  |  Formula: {formula_full} + (1|Athlete_ID)",
                "=" * 90,
                "",
                "LRT-A: Does TravelDuration significantly improve over base Model B (JetLag-only)?",
                f"  Full vs Base-B  |  LR={lr_a:.4f}  df={df_a}  p={'NA' if pd.isna(p_a) else f'{p_a:.4e}'}",
                f"  → {'TravelDuration is a significant covariate.' if not pd.isna(p_a) and p_a < 0.05 else 'TravelDuration does NOT significantly improve the model.'}",
                "",
                "LRT-B: Is JetLagGroup_dir still significant after controlling TravelDuration?",
                f"  Full vs No-JLag  |  LR={lr_b:.4f}  df={df_b}  p={'NA' if pd.isna(p_b) else f'{p_b:.4e}'}",
                f"  → {'JetLagGroup_dir remains significant (robust to travel control).' if not pd.isna(p_b) and p_b < 0.05 else 'JetLagGroup_dir is NOT significant after controlling travel.'}",
                "",
            ]

        else:  # main
            lr_m, df_m, p_m = lrt(res_full_ml, res_reduced_ml)
            log_write(lf, "\nLRT: TravelDuration_c overall effect (vs Sex-only model)")
            log_write(lf, f"  H0: TravelDuration_c coefficient = 0")
            log_write(lf, f"  LR={lr_m:.4f}  df={df_m}  p={p_m:.4e}" if not pd.isna(p_m) else f"  LR={lr_m:.4f}  df={df_m}  p=NA")
            if not pd.isna(p_m):
                log_write(lf, f"  → TravelDuration {'has a significant main effect' if p_m < 0.05 else 'does NOT have a significant main effect'} on Sinclair_Total")

            comp_lines += [
                "=" * 90,
                f"Mode: main  |  Formula: {formula_full} + (1|Athlete_ID)",
                "=" * 90,
                "",
                "LRT: TravelDuration_c as main effect (Full vs Sex-only)",
                f"  LR={lr_m:.4f}  df={df_m}  p={'NA' if pd.isna(p_m) else f'{p_m:.4e}'}",
                f"  → {'TravelDuration has a significant main effect on Sinclair_Total.' if not pd.isna(p_m) and p_m < 0.05 else 'TravelDuration does NOT have a significant main effect.'}",
                "",
            ]

        # -------- STEP 7: 系数报告 --------
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 7 - Fixed-effect coefficients (REML fit)")
        log_write(lf, "=" * 80)
        log_write(lf, f"  ICC = {icc_full:.4f}")
        log_write(lf, f"  AIC = {res_full_ml.aic:.1f}  BIC = {res_full_ml.bic:.1f}")

        comp_lines += [
            "=" * 90,
            "Fixed-effect coefficients (REML):",
            f"  ICC={icc_full:.4f}   AIC={res_full_ml.aic:.1f}   BIC={res_full_ml.bic:.1f}",
            "-" * 90,
        ]

        ci_df = res_full_reml.conf_int()
        for name, val in res_full_reml.fe_params.items():
            pval = res_full_reml.pvalues.get(name, np.nan)
            ci_lo = float(ci_df.loc[name, 0]) if name in ci_df.index else np.nan
            ci_hi = float(ci_df.loc[name, 1]) if name in ci_df.index else np.nan
            stars = sig_stars(pval)
            line = (f"  {name:<55} coef={val:+.4f}  "
                    f"95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}]  p={pval:.4f} {stars}")
            log_write(lf, line)
            comp_lines.append(line)

        comp_lines.append("")

        # -------- STEP 8: FDR 事后比较（仅 control 模式下 JetLag 各组 vs Minimal(0-1h)）--------
        if args.mode == "control":
            log_write(lf, "\n" + "=" * 80)
            log_write(lf, "STEP 8 - Post-hoc: each JetLagGroup vs Minimal  [FDR-BH]  (control mode)")
            log_write(lf, "=" * 80)
            log_write(lf, "  Note: model has no Sex×JetLag interaction → coef = emmeans trt.vs.ctrl contrast")

            jetlag_terms = [n for n in res_full_reml.fe_params.index if "JetLagGroup_dir" in n]
            rows = []
            for term in jetlag_terms:
                m = re.search(r"\[T\.(.+?)\]", term)
                group = m.group(1) if m else term
                coef  = float(res_full_reml.fe_params[term])
                se    = float(res_full_reml.bse[term])
                p_raw = float(res_full_reml.pvalues[term])
                ci_lo = float(ci_df.loc[term, 0])
                ci_hi = float(ci_df.loc[term, 1])
                rows.append({"Group": group, "Estimate": coef, "SE": se,
                             "CI_lo": ci_lo, "CI_hi": ci_hi, "p_raw": p_raw})

            emm_df = pd.DataFrame(rows)
            _, p_fdr_arr, _, _ = multipletests(emm_df["p_raw"].values, method="fdr_bh")
            emm_df["p_fdr"]    = p_fdr_arr
            emm_df["sig_raw"]  = emm_df["p_raw"].apply(sig_stars)
            emm_df["sig_fdr"]  = emm_df["p_fdr"].apply(sig_stars)

            hdr = f"  {'Group':<20} {'Estimate':>9} {'SE':>7} {'95%CI':^20} {'p_raw':>9} {'sig':>4}  {'p_fdr':>9} {'sig_fdr':>7}"
            sep = "  " + "-" * (len(hdr) - 2)
            log_write(lf, hdr)
            log_write(lf, sep)
            for _, row in emm_df.iterrows():
                ci_str = f"[{row['CI_lo']:+.3f},{row['CI_hi']:+.3f}]"
                log_write(lf, f"  {row['Group']:<20} {row['Estimate']:>+9.4f} {row['SE']:>7.4f} "
                              f"{ci_str:^20} {row['p_raw']:>9.4f} {row['sig_raw']:>4}  "
                              f"{row['p_fdr']:>9.4f} {row['sig_fdr']:>7}")

            emm_csv = os.path.join(OUTPUT_DIR, "emmeans_FDR_travel_control.csv")
            emm_df.to_csv(emm_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
            log_write(lf, f"\n  Saved: {emm_csv}")

            comp_lines += [
                "=" * 90,
                "Post-hoc: each JetLagGroup vs Minimal  [FDR-BH, control mode]",
                hdr, sep,
            ]
            for _, row in emm_df.iterrows():
                ci_str = f"[{row['CI_lo']:+.3f},{row['CI_hi']:+.3f}]"
                comp_lines.append(
                    f"  {row['Group']:<20} {row['Estimate']:>+9.4f} {row['SE']:>7.4f} "
                    f"{ci_str:^20} {row['p_raw']:>9.4f} {row['sig_raw']:>4}  "
                    f"{row['p_fdr']:>9.4f} {row['sig_fdr']:>7}"
                )
            comp_lines += ["  Reference: Minimal(0-2h)  |  Sig: *** <.001  ** <.01  * <.05  . <.1", ""]

        # -------- STEP 9: 可视化 --------
        # 9.1 TravelDuration 分布
        plt.figure(figsize=(10, 5))
        sns.histplot(df["TravelDuration"], bins=20, kde=True, color="#4C78A8")
        plt.axvline(travel_mean, color="red", linestyle="--", alpha=0.6, label=f"mean={travel_mean:.1f}h")
        plt.xlabel("TravelDuration (hours)")
        plt.ylabel("Count")
        plt.title("Travel Duration Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "travel_duration_distribution.png"), dpi=300)
        plt.close()

        # 9.2 TravelDuration vs Sinclair 散点
        plt.figure(figsize=(10, 6))
        for sex, color in [("Men", "#4C78A8"), ("Women", "#E45756")]:
            sub = df[df["Sex"] == sex]
            plt.scatter(sub["TravelDuration"], sub["Sinclair_Total"],
                        alpha=0.15, s=10, color=color, label=sex)
        plt.xlabel("TravelDuration (hours)")
        plt.ylabel("Sinclair_Total")
        plt.title(f"Sinclair_Total vs TravelDuration  [mode={args.mode}]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scatter_travel_sinclair.png"), dpi=300)
        plt.close()

        # -------- 写文件 --------
        with open(compare_path, "w", encoding="utf-8") as f:
            f.write("\n".join(comp_lines) + "\n")
        log_write(lf, f"\nModel comparison saved: {compare_path}")

        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write(f"Mode: {args.mode}\n")
            sf.write(f"Formula: {formula_report} + (1|Athlete_ID)\n")
            sf.write(f"ICC: {icc_full:.4f}\n\n")
            sf.write(res_full_reml.summary().as_text())
        log_write(lf, f"Model summary saved: {summary_path}")

        log_write(lf, "\n" + "=" * 80)
        log_write(lf, f"LMM_jetlag_travel  mode={args.mode}  - END")
        log_write(lf, "=" * 80)


if __name__ == "__main__":
    main()
