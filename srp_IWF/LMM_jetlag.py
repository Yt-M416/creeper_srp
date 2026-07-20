"""
LMM_jetlag.py（最终修正版）

基于原版代码的最小改动，核心修正：
  1. ★ 新增方向性分组 JetLagGroup_dir（11 档），保留原版 JetLagGroup_abs 做基线对比
     - Minimal(0-1h) 合并不区分方向（|signed|<2），作为参照组；标签按整数小时区间书写
     - East/West 各 3 档：3-6h, 6-9h, >=9h
  2. 拟合 3 个模型：A(abs-only) / B(directional) / C(continuous signed)
  3. 新增 Contrast Test：同量级下 East vs West 的系数差异检验
  4. 新增运动员参赛次数分布柱状图

JetLag 定义（24小时圆周最短路径，范围 -12~12）：
  d = Venue_UTC_offset - Athlete_UTC_offset
  if |d| <= 12: jetlag = d
  else:         jetlag = -sign(d) * (24 - |d|)
  正值 = 向东飞，负值 = 向西飞

输出目录：lmm_jetlag/
"""

from __future__ import annotations

import os
import sys
import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")          # 服务器/无 GUI 环境兼容
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore")


# ----------------- 配置 -----------------
INPUT_CSV = "iwf_data_top13_athlete2plus_circular.csv"
OUTPUT_DIR = "lmm_jetlag"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 分组类别顺序（2h一档，与同学保持一致）
ABS_GROUP_ORDER = ["0-2h", "2-4h", "4-6h", "6-8h", "8-10h", "10-12h"]
# ★ Minimal(0-1h) 必须是第一个 category（statsmodels 参照组）
# 标签含义：2h 宽 bins 用整数小时记为「下界-(上界-1)h」，如 [0,2)→0-1h，[2,4)→2-3h
DIRECTIONAL_GROUP_ORDER = [
    "Minimal(0-1h)",
    "West_2-3h", "West_4-5h", "West_6-7h", "West_8-9h", "West_10-11h",
    "East_2-3h", "East_4-5h", "East_6-7h", "East_8-9h", "East_10-11h",
]

# 图表横轴用这个顺序（从西到东，直观）
DIRECTIONAL_PLOT_ORDER = [
    "West_10-11h", "West_8-9h", "West_6-7h", "West_4-5h", "West_2-3h",
    "Minimal(0-1h)",
    "East_2-3h", "East_4-5h", "East_6-7h", "East_8-9h", "East_10-11h",
]


# ======================== 工具函数（与原版一致的部分） ========================

def setup_plot_style() -> None:
    try:
        from matplotlib import font_manager
        font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/Deng.ttf"]
        font_name = "Microsoft YaHei"
        for fp in font_paths:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                font_name = font_manager.FontProperties(fname=fp).get_name()
                break
        plt.rcParams["font.sans-serif"] = [font_name, "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font_scale=1.1)


def calculate_sinclair(total, body_weight, sex):
    """IWF 官方 Sinclair Total（2021-2024 周期）"""
    try:
        total = float(total); bw = float(body_weight)
    except Exception:
        return np.nan
    params = {"Men": {"A": 0.722762521, "b": 193.609}, "Women": {"A": 0.787004341, "b": 153.757}}
    s = str(sex).strip().title()
    if s not in params or pd.isna(total) or pd.isna(bw) or bw <= 0:
        return np.nan
    A, b = params[s]["A"], params[s]["b"]
    if bw > b:
        sc = 1.0
    else:
        X = np.log10(bw / b)       # log10，不是 ln
        sc = 10 ** (A * X**2)      # 10^(...)，不是 exp(...)
    return total * sc


def extract_sex_from_event(event: object) -> Optional[str]:
    if not isinstance(event, str):
        return None
    m = re.search(r"(Men|Women)", event, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).title()


def circular_jetlag_signed(venue_off: float, athlete_off: float) -> float:
    """
    24小时圆形差：
      d = venue_off - athlete_off  (行政原始时差，可能在 [-24, 24])
      若 |d| <= 12：jetlag = d
      否则：jetlag = -sign(d) * (24 - |d|)
    返回范围：[-12, 12]
    """
    if pd.isna(venue_off) or pd.isna(athlete_off):
        return np.nan
    d = float(venue_off) - float(athlete_off)
    if abs(d) <= 12:
        return d
    return -np.sign(d) * (24 - abs(d))


def build_jetlag_group_abs(jetlag_abs: float) -> Optional[str]:
    """按绝对值分组（2h一档，6档）。用于 Model A。"""
    if pd.isna(jetlag_abs):
        return None
    x = float(jetlag_abs)
    if x < 2:   return "0-2h"
    if x < 4:   return "2-4h"
    if x < 6:   return "4-6h"
    if x < 8:   return "6-8h"
    if x < 10:  return "8-10h"
    return "10-12h"


def build_jetlag_group_directional(jetlag_signed: float) -> Optional[str]:
    """
    ★ 方向性分组：11 档（2h 一档）。用于 Model B。
    分组边界不变；字符串标签按整数小时区间书写（[a,b)→「a-(b-1)h」）。

    |signed| < 2       → "Minimal(0-1h)"（参照组，对应 [0,2) 的整数 0,1）
    signed ∈ [+2, +4)  → "East_2-3h"
    signed ∈ [+4, +6)  → "East_4-5h"
    signed ∈ [+6, +8)  → "East_6-7h"
    signed ∈ [+8, +10) → "East_8-9h"
    signed ∈ [+10,+12] → "East_10-11h"
    signed ∈ (-4, -2]  → "West_2-3h"
    signed ∈ (-6, -4]  → "West_4-5h"
    signed ∈ (-8, -6]  → "West_6-7h"
    signed ∈ (-10,-8]  → "West_8-9h"
    signed ∈ [-12,-10] → "West_10-11h"
    """
    if pd.isna(jetlag_signed):
        return None
    x = float(jetlag_signed)
    absx = abs(x)
    if absx < 2:
        return "Minimal(0-1h)"
    if x > 0:  # 向东
        if absx < 4:   return "East_2-3h"
        if absx < 6:   return "East_4-5h"
        if absx < 8:   return "East_6-7h"
        if absx < 10:  return "East_8-9h"
        return "East_10-11h"
    else:       # 向西
        if absx < 4:   return "West_2-3h"
        if absx < 6:   return "West_4-5h"
        if absx < 8:   return "West_6-7h"
        if absx < 10:  return "West_8-9h"
        return "West_10-11h"


def log_write(fp, msg: str) -> None:
    fp.write(msg + "\n")
    fp.flush()


def compute_icc(result) -> float:
    try:
        group_var = float(result.cov_re.iloc[0, 0])
        resid_var = float(result.scale)
        return group_var / (group_var + resid_var) if (group_var + resid_var) > 0 else np.nan
    except Exception:
        return np.nan


def choose_travel_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["TravelDuration", "Travel_Duration", "Travel_Duration_c", "Duration_Hours", "Travel_Hours"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ======================== 主流程 ========================

def main() -> None:
    setup_plot_style()

    log_path     = os.path.join(OUTPUT_DIR, "analysis_log.txt")
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
    compare_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")
    data_used_path = os.path.join(OUTPUT_DIR, "data_used_for_model.csv")

    with open(log_path, "w", encoding="utf-8") as lf:
        log_write(lf, "=" * 80)
        log_write(lf, "LMM JetLag (Circular + Directional) - START")
        log_write(lf, f"Run time: {datetime.now().isoformat(timespec='seconds')}")
        log_write(lf, "=" * 80)

        if not os.path.exists(INPUT_CSV):
            raise FileNotFoundError(f"输入文件不存在：{INPUT_CSV}")

        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        log_write(lf, f"Loaded: {INPUT_CSV}")
        log_write(lf, f"Rows: {len(df)}, Columns: {len(df.columns)}")

        # -------- STEP 1: 基础清洗与关键列（与原版一致） --------
        required_cols = ["Total", "Body_Weight", "Event", "Athlete_ID", "Athlete_UTC_offset", "Venue_UTC_offset"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列：{missing_cols}。请确认预处理输出格式。")

        df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
        df["Body_Weight"] = pd.to_numeric(df["Body_Weight"], errors="coerce")
        df["Athlete_UTC_offset"] = pd.to_numeric(df["Athlete_UTC_offset"], errors="coerce")
        df["Venue_UTC_offset"] = pd.to_numeric(df["Venue_UTC_offset"], errors="coerce")

        df["Sex"] = df["Event"].apply(extract_sex_from_event)
        before_drop = len(df)
        df = df.dropna(subset=["Total", "Body_Weight", "Sex", "Athlete_ID",
                                "Athlete_UTC_offset", "Venue_UTC_offset"]).copy()
        log_write(lf, f"STEP 1 - Drop NA core fields: {before_drop} -> {len(df)}")

        # -------- STEP 2: 计算 circular JetLag + 两种分组 --------
        df["JetLag_signed"] = df.apply(
            lambda r: circular_jetlag_signed(r["Venue_UTC_offset"], r["Athlete_UTC_offset"]), axis=1)
        df["JetLag_abs"] = df["JetLag_signed"].abs()

        # 原版分组（abs-only，Model A 基线用）
        df["JetLagGroup_abs"] = df["JetLag_abs"].apply(build_jetlag_group_abs)
        # ★ 方向性分组（Model B 用）
        df["JetLagGroup_dir"] = df["JetLag_signed"].apply(build_jetlag_group_directional)

        before_jg = len(df)
        df = df.dropna(subset=["JetLagGroup_abs", "JetLagGroup_dir"]).copy()
        log_write(lf, f"STEP 2 - JetLag computed; drop NA: {before_jg} -> {len(df)}")

        df["JetLagGroup_abs"] = pd.Categorical(df["JetLagGroup_abs"],
                                                categories=ABS_GROUP_ORDER, ordered=True)
        df["JetLagGroup_dir"] = pd.Categorical(df["JetLagGroup_dir"],
                                                categories=DIRECTIONAL_GROUP_ORDER, ordered=True)

        log_write(lf, "\nJetLagGroup_abs counts (原版分组，不区分方向):")
        log_write(lf, df["JetLagGroup_abs"].value_counts().sort_index().to_string())
        log_write(lf, "\nJetLagGroup_dir counts (★方向性分组):")
        log_write(lf, df["JetLagGroup_dir"].value_counts().sort_index().to_string())

        # -------- STEP 3: Sinclair_Total（与原版一致） --------
        df["Sinclair_Total"] = df.apply(
            lambda r: calculate_sinclair(r["Total"], r["Body_Weight"], r["Sex"]), axis=1)
        before_sin = len(df)
        df = df.dropna(subset=["Sinclair_Total"]).copy()
        log_write(lf, f"\nSTEP 3 - Sinclair_Total computed; drop NA: {before_sin} -> {len(df)}")

        log_write(lf, "\nData summary (post-clean):")
        log_write(lf, f"  Rows used (pre-model): {len(df)}")
        log_write(lf, f"  Athletes (Athlete_ID): {df['Athlete_ID'].nunique()}")
        if "Nation" in df.columns:
            log_write(lf, f"  Nations: {df['Nation'].nunique()}")
        log_write(lf, f"  Mean JetLag_abs:    {df['JetLag_abs'].mean():.3f} h")
        log_write(lf, f"  Mean JetLag_signed: {df['JetLag_signed'].mean():.3f} h")
        log_write(lf, f"  Max  JetLag_abs:    {df['JetLag_abs'].max():.3f} h")

        # -------- STEP 4: 运动员参赛次数分布图 ★新增 --------
        athlete_counts = df.groupby("Athlete_ID").size()
        freq_dist = athlete_counts.value_counts().sort_index()

        log_write(lf, "\nSTEP 4 - Athlete participation frequency:")
        for n_times, n_athletes in freq_dist.items():
            log_write(lf, f"  参赛 {n_times} 次: {n_athletes} 名运动员")

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(freq_dist.index.astype(str), freq_dist.values,
                       color="#4C78A8", edgecolor="white")
        for bar_obj, val in zip(bars, freq_dist.values):
            ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Number of competitions per athlete")
        ax.set_ylabel("Number of athletes")
        ax.set_title(f"Athlete Participation Frequency (total {df['Athlete_ID'].nunique()} athletes)")
        plt.tight_layout()
        p_freq = os.path.join(OUTPUT_DIR, "athlete_participation_frequency.png")
        plt.savefig(p_freq, dpi=300)
        plt.close()
        log_write(lf, f"  Plot saved: {p_freq}")

        # -------- STEP 5: 保存模型数据 & 准备 --------
        df["Sex"] = df["Sex"].astype("category")
        df["Athlete_ID"] = df["Athlete_ID"].astype("category")
        df.to_csv(data_used_path, index=False, encoding="utf-8-sig")
        log_write(lf, f"\nSaved model input data: {data_used_path}")

        # -------- STEP 6: 拟合 3 个模型 --------
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 6 - Fitting 3 models for comparison")
        log_write(lf, "=" * 80)

        # --- Model A: 仅绝对值分组（与原版完全一致） ---
        formula_a = "Sinclair_Total ~ C(Sex) + C(JetLagGroup_abs)"
        log_write(lf, f"\nModel A (abs-only, baseline): {formula_a} + (1|Athlete_ID)")
        model_a = smf.mixedlm(formula_a, data=df, groups=df["Athlete_ID"])
        res_a_reml = model_a.fit(method="lbfgs")
        res_a_ml   = model_a.fit(method="lbfgs", reml=False)
        icc_a = compute_icc(res_a_reml)

        # --- Model B: 方向性分组 ★核心修正 ---
        formula_b = "Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir)"
        log_write(lf, f"\nModel B (directional, ★fix): {formula_b} + (1|Athlete_ID)")
        model_b = smf.mixedlm(formula_b, data=df, groups=df["Athlete_ID"])
        res_b_reml = model_b.fit(method="lbfgs")
        res_b_ml   = model_b.fit(method="lbfgs", reml=False)
        icc_b = compute_icc(res_b_reml)

        # --- Model C: 连续 JetLag_signed ---
        formula_c = "Sinclair_Total ~ C(Sex) + JetLag_signed"
        log_write(lf, f"\nModel C (continuous signed): {formula_c} + (1|Athlete_ID)")
        model_c = smf.mixedlm(formula_c, data=df, groups=df["Athlete_ID"])
        res_c_reml = model_c.fit(method="lbfgs")
        res_c_ml   = model_c.fit(method="lbfgs", reml=False)
        icc_c = compute_icc(res_c_reml)

        # -------- STEP 7.0: Likelihood Ratio Test — Model B 整体效应（JetLagGroup_dir）--------
        # 与 Full 相比，Reduced 仅去掉 C(JetLagGroup_dir)；随机效应与数据行一致；须用 ML（reml=False）比较固定效应。
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 7.0 - Likelihood Ratio Test (Model B: JetLagGroup_dir overall effect)")
        log_write(lf, "=" * 80)

        formula_b_reduced = "Sinclair_Total ~ C(Sex)"
        log_write(lf, f"Full model (ML):    {formula_b} + (1|Athlete_ID)")
        log_write(lf, f"Reduced model (ML): {formula_b_reduced} + (1|Athlete_ID)")

        model_b_reduced = smf.mixedlm(formula_b_reduced, data=df, groups=df["Athlete_ID"])
        res_b_reduced_ml = model_b_reduced.fit(method="lbfgs", reml=False)

        llf_full = float(res_b_ml.llf)
        llf_reduced = float(res_b_reduced_ml.llf)
        lr_stat = 2.0 * (llf_full - llf_reduced)
        # 嵌套模型下固定效应参数个数之差 = 被检验因子非参照水平数（此处为 11-1=10）
        df_lrt = int(res_b_ml.k_fe - res_b_reduced_ml.k_fe)
        if lr_stat < 0:
            log_write(
                lf,
                f"[WARN] LR statistic is negative ({lr_stat:.6f}); "
                "check convergence / optimization; p-value not interpretable.",
            )
            p_lrt = np.nan
        else:
            p_lrt = float(chi2.sf(lr_stat, df_lrt))

        log_write(lf, "\nLikelihood Ratio Test (asymptotic chi-square):")
        log_write(lf, f"  LogLik (full)         = {llf_full:.6f}")
        log_write(lf, f"  LogLik (reduced)      = {llf_reduced:.6f}")
        log_write(lf, f"  LR statistic (chi^2)  = {lr_stat:.6f}")
        log_write(lf, f"  df                    = {df_lrt}  (difference in k_fe)")
        log_write(lf, f"  p-value               = {p_lrt:.3e}" if not pd.isna(p_lrt) else "  p-value               = NA")
        if not pd.isna(p_lrt):
            if p_lrt < 0.05:
                log_write(lf, "  → Result: C(JetLagGroup_dir) has a significant overall effect (p < .05).")
            else:
                log_write(lf, "  → Result: C(JetLagGroup_dir) is not significant overall (p >= .05).")
        log_write(
            lf,
            "  Note: asymptotic LRT; for small samples consider sensitivity checks (e.g. parametric bootstrap).",
        )

        lrt_comp_lines = [
            "=" * 90,
            "Likelihood Ratio Test — Model B: overall effect of C(JetLagGroup_dir)",
            "  H0: all JetLagGroup_dir coefficients (vs Minimal) are jointly zero.",
            "  Full:  " + formula_b + " + (1|Athlete_ID), ML",
            "  Reduced: " + formula_b_reduced + " + (1|Athlete_ID), ML",
            "-" * 90,
            f"  LogLik (full)    = {llf_full:.6f}",
            f"  LogLik (reduced) = {llf_reduced:.6f}",
            f"  LR (chi^2)       = {lr_stat:.6f}",
            f"  df               = {df_lrt}",
            f"  p-value          = {p_lrt:.3e}" if not pd.isna(p_lrt) else "  p-value          = NA",
            "",
        ]

        # -------- STEP 7.1: Likelihood Ratio Test — Model C 整体效应（JetLag_signed 连续变量）--------
        # 复用 STEP 6 已拟合的 res_c_ml 与 STEP 7.0 已拟合的 res_b_reduced_ml，避免重复拟合。
        # df = k_fe(full) - k_fe(reduced) = 1（仅多一个连续参数 JetLag_signed）。
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 7.1 - Likelihood Ratio Test (Model C: JetLag_signed continuous effect)")
        log_write(lf, "=" * 80)
        log_write(lf, f"Full model (ML):    {formula_c} + (1|Athlete_ID)  [reused from STEP 6]")
        log_write(lf, f"Reduced model (ML): {formula_b_reduced} + (1|Athlete_ID)  [reused from STEP 7.0]")

        llf_c_full    = float(res_c_ml.llf)
        llf_c_reduced = float(res_b_reduced_ml.llf)
        lr_stat_c     = 2.0 * (llf_c_full - llf_c_reduced)
        df_lrt_c      = int(res_c_ml.k_fe - res_b_reduced_ml.k_fe)   # 应为 1

        if lr_stat_c < 0:
            log_write(
                lf,
                f"[WARN] LR statistic is negative ({lr_stat_c:.6f}); "
                "check convergence / optimization; p-value not interpretable.",
            )
            p_lrt_c = np.nan
        else:
            p_lrt_c = float(chi2.sf(lr_stat_c, df_lrt_c))

        log_write(lf, "\nLikelihood Ratio Test (asymptotic chi-square):")
        log_write(lf, f"  LogLik (full)         = {llf_c_full:.6f}")
        log_write(lf, f"  LogLik (reduced)      = {llf_c_reduced:.6f}")
        log_write(lf, f"  LR statistic (chi^2)  = {lr_stat_c:.6f}")
        log_write(lf, f"  df                    = {df_lrt_c}  (1 continuous predictor)")
        log_write(lf, f"  p-value               = {p_lrt_c:.3e}" if not pd.isna(p_lrt_c) else "  p-value               = NA")
        if not pd.isna(p_lrt_c):
            if p_lrt_c < 0.05:
                log_write(lf, "  → Result: JetLag_signed has a significant overall effect (p < .05).")
            else:
                log_write(lf, "  → Result: JetLag_signed is not significant overall (p >= .05).")
        log_write(
            lf,
            "  Note: asymptotic LRT; for small samples consider sensitivity checks (e.g. parametric bootstrap).",
        )

        lrt_c_comp_lines = [
            "=" * 90,
            "Likelihood Ratio Test — Model C: overall effect of JetLag_signed (continuous)",
            "  H0: coefficient of JetLag_signed = 0.",
            "  Full:    " + formula_c + " + (1|Athlete_ID), ML",
            "  Reduced: " + formula_b_reduced + " + (1|Athlete_ID), ML",
            "-" * 90,
            f"  LogLik (full)    = {llf_c_full:.6f}",
            f"  LogLik (reduced) = {llf_c_reduced:.6f}",
            f"  LR (chi^2)       = {lr_stat_c:.6f}",
            f"  df               = {df_lrt_c}",
            f"  p-value          = {p_lrt_c:.3e}" if not pd.isna(p_lrt_c) else "  p-value          = NA",
            "",
        ]

        # -------- STEP 7.2: Post-hoc 对比（各组 vs Minimal）+ FDR 校正 --------
        # 数学等价性说明：
        #   模型无 Sex×JetLagGroup_dir 交互项，Sex 的贡献在"g vs Minimal"对比中完全消掉，
        #   因此 C(JetLagGroup_dir)[T.xxx] 系数本身就等于 emmeans 的 trt.vs.ctrl contrast。
        #   只需对已有 10 个 p 值做 BH-FDR 多重校正即可，无需重新估计。
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 7.2 - Post-hoc contrasts: each group vs Minimal(0-1h)  [FDR-BH corrected]")
        log_write(lf, "=" * 80)
        log_write(lf, "  Method: model coefficients = trt.vs.ctrl contrasts (no Sex×Group interaction)")
        log_write(lf, "  Multiple comparison correction: FDR (Benjamini-Hochberg)")
        log_write(lf, "  95% CI from REML fit profile (Wald approximation)\n")

        jetlag_terms = [n for n in res_b_reml.fe_params.index if "JetLagGroup_dir" in n]
        ci_df = res_b_reml.conf_int()

        emm_rows = []
        for term in jetlag_terms:
            m = re.search(r"\[T\.(.+?)\]", term)
            group = m.group(1) if m else term
            coef  = float(res_b_reml.fe_params[term])
            se    = float(res_b_reml.bse[term])
            p_raw = float(res_b_reml.pvalues[term])
            ci_lo = float(ci_df.loc[term, 0])
            ci_hi = float(ci_df.loc[term, 1])
            emm_rows.append({"Group": group, "Estimate": coef, "SE": se,
                             "CI_lo": ci_lo, "CI_hi": ci_hi, "p_raw": p_raw})

        emm_df = pd.DataFrame(emm_rows)

        # BH-FDR 校正
        _, p_fdr_arr, _, _ = multipletests(emm_df["p_raw"].values, method="fdr_bh")
        emm_df["p_fdr"] = p_fdr_arr

        def sig_stars(p: float) -> str:
            if pd.isna(p):  return ""
            if p < 0.001:   return "***"
            if p < 0.01:    return "**"
            if p < 0.05:    return "*"
            if p < 0.1:     return "."
            return ""

        emm_df["sig_raw"] = emm_df["p_raw"].apply(sig_stars)
        emm_df["sig_fdr"] = emm_df["p_fdr"].apply(sig_stars)

        # 写入 analysis_log
        hdr = f"  {'Group':<20} {'Estimate':>9} {'SE':>7} {'95%CI':^20} {'p_raw':>9} {'sig':>4}  {'p_fdr':>9} {'sig_fdr':>7}"
        log_write(lf, hdr)
        log_write(lf, "  " + "-" * (len(hdr) - 2))
        for _, row in emm_df.iterrows():
            ci_str = f"[{row['CI_lo']:+.3f}, {row['CI_hi']:+.3f}]"
            log_write(
                lf,
                f"  {row['Group']:<20} {row['Estimate']:>+9.4f} {row['SE']:>7.4f} "
                f"{ci_str:^20} {row['p_raw']:>9.4f} {row['sig_raw']:>4}  "
                f"{row['p_fdr']:>9.4f} {row['sig_fdr']:>7}",
            )
        log_write(lf, "  (Reference: Minimal(0-1h), Estimate = contrast vs Minimal)")

        # 保存到 CSV
        emm_csv_path = os.path.join(OUTPUT_DIR, "emmeans_FDR.csv")
        emm_df.to_csv(emm_csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
        log_write(lf, f"\n  Saved: {emm_csv_path}")

        # 构造写入 model_comparison.txt 的段落
        emm_comp_lines = [
            "",
            "=" * 90,
            "Post-hoc contrasts: each group vs Minimal(0-1h)  [FDR-BH corrected]",
            "  (No Sex × JetLagGroup_dir interaction → model coefficient = emmeans contrast)",
            f"  {'Group':<20} {'Estimate':>9} {'SE':>7} {'95%CI':^20} {'p_raw':>9} {'sig':>4}  {'p_fdr':>9} {'sig_fdr':>7}",
            "  " + "-" * 88,
        ]
        for _, row in emm_df.iterrows():
            ci_str = f"[{row['CI_lo']:+.3f}, {row['CI_hi']:+.3f}]"
            emm_comp_lines.append(
                f"  {row['Group']:<20} {row['Estimate']:>+9.4f} {row['SE']:>7.4f} "
                f"{ci_str:^20} {row['p_raw']:>9.4f} {row['sig_raw']:>4}  "
                f"{row['p_fdr']:>9.4f} {row['sig_fdr']:>7}"
            )
        emm_comp_lines.append("  Reference: Minimal(0-1h)")
        emm_comp_lines.append("  Sig codes: *** p<.001  ** p<.01  * p<.05  . p<.1")
        emm_comp_lines.append("")

        # -------- STEP 7: 模型对比 + Contrast Test --------
        aic_a, bic_a = res_a_ml.aic, res_a_ml.bic
        aic_b, bic_b = res_b_ml.aic, res_b_ml.bic
        aic_c, bic_c = res_c_ml.aic, res_c_ml.bic

        comp_lines = []
        comp_lines.extend(lrt_comp_lines)
        comp_lines.append("=" * 90)
        comp_lines.append("Model Comparison: Abs-only vs Directional vs Continuous")
        comp_lines.append("=" * 90)
        comp_lines.append("")
        comp_lines.append(f"{'Model':<12} {'Formula':<60} {'AIC':>10} {'BIC':>10} {'ICC':>8}")
        comp_lines.append("-" * 100)
        comp_lines.append(f"{'A (abs)':<12} {formula_a:<60} {aic_a:>10.1f} {bic_a:>10.1f} {icc_a:>8.4f}")
        comp_lines.append(f"{'B (dir) *':<12} {formula_b:<60} {aic_b:>10.1f} {bic_b:>10.1f} {icc_b:>8.4f}")
        comp_lines.append(f"{'C (cont)':<12} {formula_c:<60} {aic_c:>10.1f} {bic_c:>10.1f} {icc_c:>8.4f}")
        comp_lines.append("")

        best_aic = min(aic_a, aic_b, aic_c)
        for label, aic_val in [("A", aic_a), ("B", aic_b), ("C", aic_c)]:
            comp_lines.append(f"  dAIC({label} vs best): {aic_val - best_aic:+.1f}")
        comp_lines.append("")
        comp_lines.append("Interpretation:")
        comp_lines.append("  dAIC < 2:  models essentially equivalent")
        comp_lines.append("  dAIC 2-7:  some evidence for better model")
        comp_lines.append("  dAIC > 10: strong evidence for better model")

        # Model B 各组系数
        comp_lines.append("")
        comp_lines.append("=" * 90)
        comp_lines.append("Model B (Directional) - JetLagGroup coefficients")
        comp_lines.append("  Reference group: Minimal(0-1h)")
        comp_lines.append("  Positive coef = higher Sinclair than ref; Negative = lower")
        comp_lines.append("-" * 90)
        for name, val in res_b_reml.params.items():
            if "JetLagGroup_dir" in name:
                pval = res_b_reml.pvalues.get(name, np.nan)
                sig = ""
                if not pd.isna(pval):
                    if pval < 0.001:   sig = "***"
                    elif pval < 0.01:  sig = "**"
                    elif pval < 0.05:  sig = "*"
                    elif pval < 0.1:   sig = "."
                comp_lines.append(f"  {name:<55} coef={val:+.4f}  p={pval:.4f} {sig}")

        # LRT Model C（连续变量，紧跟 Model B 系数之后）
        comp_lines.extend(lrt_c_comp_lines)

        # Post-hoc FDR 表（紧跟 LRT C 之后）
        comp_lines.extend(emm_comp_lines)

        # ★ Contrast Tests: 同量级 East vs West
        comp_lines.append("")
        comp_lines.append("=" * 90)
        comp_lines.append("Contrast Tests: East vs West at same magnitude")
        comp_lines.append("  H0: coef(East_Xh) - coef(West_Xh) = 0")
        comp_lines.append("  (both relative to Minimal(0-1h), so difference = East effect - West effect)")
        comp_lines.append("-" * 90)

        # ★ 用固定效应参数（fe_params）构造 contrast 向量
        #   res.params 包含 "Group Var" 等随机效应参数，t_test 不接受
        #   必须用 res.fe_params（长度 = k_fe），且向量必须是 2D: shape (1, k_fe)
        fe_names = list(res_b_reml.fe_params.index)
        fe_idx = {n: i for i, n in enumerate(fe_names)}

        def contrast_east_vs_west(label: str, east_term: str, west_term: str) -> None:
            """检验 coef(east_term) - coef(west_term) = 0"""
            if east_term not in fe_idx:
                comp_lines.append(f"  {label:<18}  [SKIP] {east_term} not in fixed-effect params")
                return
            if west_term not in fe_idx:
                comp_lines.append(f"  {label:<18}  [SKIP] {west_term} not in fixed-effect params")
                return
            v = np.zeros((1, res_b_reml.k_fe), dtype=float)
            v[0, fe_idx[east_term]] = 1.0
            v[0, fe_idx[west_term]] = -1.0
            try:
                t = res_b_reml.t_test(v)
                est = float(np.asarray(t.effect).flat[0])
                se  = float(np.asarray(t.sd).flat[0])
                p   = float(np.asarray(t.pvalue).flat[0])
                sig = ""
                if p < 0.001:   sig = "***"
                elif p < 0.01:  sig = "**"
                elif p < 0.05:  sig = "*"
                elif p < 0.1:   sig = "."
                comp_lines.append(f"  {label:<18}  diff={est:+.4f}  se={se:.4f}  p={p:.4f} {sig}")
            except Exception as e:
                comp_lines.append(f"  {label:<18}  [FAILED] {e}")

        contrast_east_vs_west(
            "2-3h: East-West",
            "C(JetLagGroup_dir)[T.East_2-3h]",
            "C(JetLagGroup_dir)[T.West_2-3h]",
        )
        contrast_east_vs_west(
            "4-5h: East-West",
            "C(JetLagGroup_dir)[T.East_4-5h]",
            "C(JetLagGroup_dir)[T.West_4-5h]",
        )
        contrast_east_vs_west(
            "6-7h: East-West",
            "C(JetLagGroup_dir)[T.East_6-7h]",
            "C(JetLagGroup_dir)[T.West_6-7h]",
        )
        contrast_east_vs_west(
            "8-9h: East-West",
            "C(JetLagGroup_dir)[T.East_8-9h]",
            "C(JetLagGroup_dir)[T.West_8-9h]",
        )
        contrast_east_vs_west(
            "10-11h: East-West",
            "C(JetLagGroup_dir)[T.East_10-11h]",
            "C(JetLagGroup_dir)[T.West_10-11h]",
        )

        comp_lines.append("")
        comp_lines.append("Interpretation:")
        comp_lines.append("  If p < 0.05, the effect of jet lag differs significantly by direction")
        comp_lines.append("  at that magnitude level, supporting direction-specific modeling.")
        comp_lines.append("  If all p > 0.05, abs-only grouping may be sufficient for this dataset.")
        comp_lines.append("=" * 90)

        comp_text = "\n".join(comp_lines)
        with open(compare_path, "w", encoding="utf-8") as f:
            f.write(comp_text + "\n")
        log_write(lf, f"\nModel comparison saved: {compare_path}")

        # -------- STEP 8: 写 summary（3 个模型） --------
        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write("=" * 80 + "\n")
            sf.write("Model A: Abs-only JetLagGroup (baseline, same as original)\n")
            sf.write(f"Formula: {formula_a} + (1|Athlete_ID)\n")
            sf.write(f"ICC: {icc_a:.4f}\n\n")
            sf.write(res_a_reml.summary().as_text())
            sf.write("\n\n")
            sf.write("=" * 80 + "\n")
            sf.write("* Model B: Directional JetLagGroup (FIXED version)\n")
            sf.write(f"Formula: {formula_b} + (1|Athlete_ID)\n")
            sf.write(f"Ref group: Minimal(0-1h)\n")
            sf.write(f"ICC: {icc_b:.4f}\n\n")
            sf.write(res_b_reml.summary().as_text())
            sf.write("\n\n")
            sf.write("=" * 80 + "\n")
            sf.write("Model C: Continuous JetLag_signed\n")
            sf.write(f"Formula: {formula_c} + (1|Athlete_ID)\n")
            sf.write(f"ICC: {icc_c:.4f}\n\n")
            sf.write(res_c_reml.summary().as_text())
        log_write(lf, f"Model summaries saved: {summary_path}")

        # -------- STEP 9: 可视化 --------

        # 9.1 JetLag signed 分布
        plt.figure(figsize=(11, 6))
        sns.histplot(df["JetLag_signed"], bins=25, kde=True, color="#4C78A8")
        plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        plt.title("JetLag (signed circular difference) distribution")
        plt.xlabel("JetLag_signed (hours): negative=Westward, positive=Eastward")
        plt.ylabel("Count")
        plt.tight_layout()
        p1 = os.path.join(OUTPUT_DIR, "jetlag_distribution.png")
        plt.savefig(p1, dpi=300)
        plt.close()

        # 9.2 ★ 方向性分组箱线图
        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df, x="JetLagGroup_dir", y="Sinclair_Total",
                    hue="Sex", palette="Set2", order=DIRECTIONAL_PLOT_ORDER)
        plt.title("Sinclair_Total by Directional JetLagGroup and Sex")
        plt.xlabel("JetLagGroup (West <-- --> East)")
        plt.ylabel("Sinclair_Total")
        plt.xticks(rotation=15)
        plt.tight_layout()
        p2 = os.path.join(OUTPUT_DIR, "sinclair_by_jetlaggroup_directional.png")
        plt.savefig(p2, dpi=300)
        plt.close()

        # 9.3 原版分组箱线图（对比用）
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="JetLagGroup_abs", y="Sinclair_Total",
                    hue="Sex", palette="Set2", order=ABS_GROUP_ORDER)
        plt.title("Sinclair_Total by Abs JetLagGroup and Sex (original grouping)")
        plt.xlabel("JetLagGroup (abs hours only)")
        plt.ylabel("Sinclair_Total")
        plt.tight_layout()
        p3 = os.path.join(OUTPUT_DIR, "sinclair_by_jetlaggroup_abs.png")
        plt.savefig(p3, dpi=300)
        plt.close()

        # 9.4 Model B 边际效应（方向性分组）
        try:
            pred_rows = []
            for sex in ["Men", "Women"]:
                for g in DIRECTIONAL_GROUP_ORDER:
                    pred_rows.append({"Sex": sex, "JetLagGroup_dir": g})
            pred_df = pd.DataFrame(pred_rows)
            pred_df["JetLagGroup_dir"] = pd.Categorical(
                pred_df["JetLagGroup_dir"], categories=DIRECTIONAL_GROUP_ORDER, ordered=True)
            pred_df["Pred_Sinclair"] = res_b_reml.predict(pred_df)

            plt.figure(figsize=(14, 7))
            sns.pointplot(data=pred_df, x="JetLagGroup_dir", y="Pred_Sinclair",
                          hue="Sex", dodge=0.3, markers="o", capsize=0.1,
                          order=DIRECTIONAL_PLOT_ORDER)
            plt.title("Model B: Predicted Sinclair_Total by Directional JetLagGroup")
            plt.xlabel("JetLagGroup (West <-- --> East)")
            plt.ylabel("Predicted Sinclair_Total")
            plt.xticks(rotation=15)
            plt.tight_layout()
            p4 = os.path.join(OUTPUT_DIR, "marginal_effects_directional.png")
            plt.savefig(p4, dpi=300)
            plt.close()
        except Exception as e:
            log_write(lf, f"[WARN] Marginal effects plot failed: {e}")

        # 9.5 散点图：连续 JetLag_signed vs Sinclair
        plt.figure(figsize=(12, 6))
        for sex, color in [("Men", "#4C78A8"), ("Women", "#E45756")]:
            sub = df[df["Sex"] == sex]
            plt.scatter(sub["JetLag_signed"], sub["Sinclair_Total"],
                        alpha=0.15, s=10, color=color, label=sex)
        plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        plt.xlabel("JetLag_signed (hours): negative=West, positive=East")
        plt.ylabel("Sinclair_Total")
        plt.title("Sinclair_Total vs JetLag_signed (scatter)")
        plt.legend()
        plt.tight_layout()
        p5 = os.path.join(OUTPUT_DIR, "scatter_jetlag_sinclair.png")
        plt.savefig(p5, dpi=300)
        plt.close()

        log_write(lf, "\nAll output files saved to: " + os.path.abspath(OUTPUT_DIR))
        log_write(lf, "=" * 80)
        log_write(lf, "LMM JetLag (Circular + Directional) - END")
        log_write(lf, "=" * 80)


if __name__ == "__main__":
    main()