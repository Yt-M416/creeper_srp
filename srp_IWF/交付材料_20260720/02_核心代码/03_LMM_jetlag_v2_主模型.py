"""
LMM_jetlag_v2.py

在原 `LMM_jetlag.py` 的模型结构与统计流程不变的前提下，
将主分析样本统一为“TravelDuration 可用”的 2186 条子样本，
以便与论文主文样本口径一致。

核心变化：
  1. 输入改为 `athlete_top13_travel.csv`
  2. 在完成原主分析所需清洗后，额外丢弃缺失 TravelDuration 的记录
  3. 输出目录改为 `lmm_jetlag_v2/`

其余部分（JetLag 分组、LMM 公式、LRT、FDR、East vs West contrast）
均与 `LMM_jetlag.py` 保持一致，便于和 v1 直接比较。
"""

from __future__ import annotations

import os
import re
import warnings
from datetime import datetime
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


INPUT_CSV = "athlete_top13_travel.csv"
OUTPUT_DIR = "lmm_jetlag_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABS_GROUP_ORDER = ["0-2h", "2-4h", "4-6h", "6-8h", "8-10h", "10-12h"]
DIRECTIONAL_GROUP_ORDER = [
    "Minimal(0-1h)",
    "West_2-3h",
    "West_4-5h",
    "West_6-7h",
    "West_8-9h",
    "West_10-11h",
    "East_2-3h",
    "East_4-5h",
    "East_6-7h",
    "East_8-9h",
    "East_10-11h",
]
DIRECTIONAL_PLOT_ORDER = [
    "West_10-11h",
    "West_8-9h",
    "West_6-7h",
    "West_4-5h",
    "West_2-3h",
    "Minimal(0-1h)",
    "East_2-3h",
    "East_4-5h",
    "East_6-7h",
    "East_8-9h",
    "East_10-11h",
]


def setup_plot_style() -> None:
    try:
        from matplotlib import font_manager

        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/Deng.ttf",
        ]
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


def calculate_sinclair(total: object, body_weight: object, sex: object) -> float:
    try:
        total_f = float(total)
        bw = float(body_weight)
    except Exception:
        return np.nan
    params = {"Men": {"A": 0.722762521, "b": 193.609}, "Women": {"A": 0.787004341, "b": 153.757}}
    s = str(sex).strip().title()
    if s not in params or pd.isna(total_f) or pd.isna(bw) or bw <= 0:
        return np.nan
    a, b = params[s]["A"], params[s]["b"]
    if bw > b:
        sc = 1.0
    else:
        x = np.log10(bw / b)
        sc = 10 ** (a * x**2)
    return total_f * sc


def extract_sex_from_event(event: object) -> Optional[str]:
    if not isinstance(event, str):
        return None
    m = re.search(r"(Men|Women)", event, flags=re.IGNORECASE)
    return m.group(1).title() if m else None


def circular_jetlag_signed(venue_off: float, athlete_off: float) -> float:
    if pd.isna(venue_off) or pd.isna(athlete_off):
        return np.nan
    d = float(venue_off) - float(athlete_off)
    if abs(d) <= 12:
        return d
    return -np.sign(d) * (24 - abs(d))


def build_jetlag_group_abs(jetlag_abs: float) -> Optional[str]:
    if pd.isna(jetlag_abs):
        return None
    x = float(jetlag_abs)
    if x < 2:
        return "0-2h"
    if x < 4:
        return "2-4h"
    if x < 6:
        return "4-6h"
    if x < 8:
        return "6-8h"
    if x < 10:
        return "8-10h"
    return "10-12h"


def build_jetlag_group_directional(jetlag_signed: float) -> Optional[str]:
    if pd.isna(jetlag_signed):
        return None
    x = float(jetlag_signed)
    absx = abs(x)
    if absx < 2:
        return "Minimal(0-1h)"
    if x > 0:
        if absx < 4:
            return "East_2-3h"
        if absx < 6:
            return "East_4-5h"
        if absx < 8:
            return "East_6-7h"
        if absx < 10:
            return "East_8-9h"
        return "East_10-11h"
    if absx < 4:
        return "West_2-3h"
    if absx < 6:
        return "West_4-5h"
    if absx < 8:
        return "West_6-7h"
    if absx < 10:
        return "West_8-9h"
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


def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def main() -> None:
    setup_plot_style()

    log_path = os.path.join(OUTPUT_DIR, "analysis_log.txt")
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
    compare_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")
    data_used_path = os.path.join(OUTPUT_DIR, "data_used_for_model.csv")

    with open(log_path, "w", encoding="utf-8") as lf:
        log_write(lf, "=" * 80)
        log_write(lf, "LMM JetLag V2 (Travel-available subset) - START")
        log_write(lf, f"Run time: {datetime.now().isoformat(timespec='seconds')}")
        log_write(lf, "=" * 80)

        if not os.path.exists(INPUT_CSV):
            raise FileNotFoundError(f"输入文件不存在：{INPUT_CSV}")

        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        log_write(lf, f"Loaded: {INPUT_CSV}")
        log_write(lf, f"Rows: {len(df)}, Columns: {len(df.columns)}")

        required_cols = [
            "Total",
            "Body_Weight",
            "Event",
            "Athlete_ID",
            "Athlete_UTC_offset",
            "Venue_UTC_offset",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列：{missing_cols}")

        for col in ["Total", "Body_Weight", "Athlete_UTC_offset", "Venue_UTC_offset", "TravelDuration"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Sex"] = df["Event"].apply(extract_sex_from_event)
        before_drop = len(df)
        df = df.dropna(
            subset=["Total", "Body_Weight", "Sex", "Athlete_ID", "Athlete_UTC_offset", "Venue_UTC_offset"]
        ).copy()
        log_write(lf, f"STEP 1 - Drop NA core fields: {before_drop} -> {len(df)}")

        df["JetLag_signed"] = df.apply(
            lambda r: circular_jetlag_signed(r["Venue_UTC_offset"], r["Athlete_UTC_offset"]),
            axis=1,
        )
        df["JetLag_abs"] = df["JetLag_signed"].abs()
        df["JetLagGroup_abs"] = df["JetLag_abs"].apply(build_jetlag_group_abs)
        df["JetLagGroup_dir"] = df["JetLag_signed"].apply(build_jetlag_group_directional)
        before_jg = len(df)
        df = df.dropna(subset=["JetLagGroup_abs", "JetLagGroup_dir"]).copy()
        log_write(lf, f"STEP 2 - JetLag computed; drop NA: {before_jg} -> {len(df)}")

        df["JetLagGroup_abs"] = pd.Categorical(df["JetLagGroup_abs"], categories=ABS_GROUP_ORDER, ordered=True)
        df["JetLagGroup_dir"] = pd.Categorical(
            df["JetLagGroup_dir"],
            categories=DIRECTIONAL_GROUP_ORDER,
            ordered=True,
        )

        df["Sinclair_Total"] = df.apply(
            lambda r: calculate_sinclair(r["Total"], r["Body_Weight"], r["Sex"]),
            axis=1,
        )
        before_sin = len(df)
        df = df.dropna(subset=["Sinclair_Total"]).copy()
        log_write(lf, f"STEP 3 - Sinclair_Total computed; drop NA: {before_sin} -> {len(df)}")

        n_miss_travel = int(df["TravelDuration"].isna().sum()) if "TravelDuration" in df.columns else len(df)
        log_write(lf, f"STEP 4 - TravelDuration missing before subset: {n_miss_travel} / {len(df)}")
        df = df.dropna(subset=["TravelDuration"]).copy()
        log_write(lf, f"STEP 4 - Restrict to travel-available subset: {len(df)} rows")

        log_write(lf, "\nJetLagGroup_dir counts (travel-available subset):")
        log_write(lf, df["JetLagGroup_dir"].value_counts().sort_index().to_string())

        log_write(lf, "\nData summary (post-clean):")
        log_write(lf, f"  Rows used (pre-model): {len(df)}")
        log_write(lf, f"  Athletes (Athlete_ID): {df['Athlete_ID'].nunique()}")
        if "Nation" in df.columns:
            log_write(lf, f"  Nations: {df['Nation'].nunique()}")
        log_write(lf, f"  Mean JetLag_abs:    {df['JetLag_abs'].mean():.3f} h")
        log_write(lf, f"  Mean JetLag_signed: {df['JetLag_signed'].mean():.3f} h")
        log_write(lf, f"  Max  JetLag_abs:    {df['JetLag_abs'].max():.3f} h")

        athlete_counts = df.groupby("Athlete_ID").size()
        freq_dist = athlete_counts.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(freq_dist.index.astype(str), freq_dist.values, color="#4C78A8", edgecolor="white")
        for bar_obj, val in zip(bars, freq_dist.values):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 3, str(val), ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Number of competitions per athlete")
        ax.set_ylabel("Number of athletes")
        ax.set_title(f"Athlete Participation Frequency (travel-available subset, {df['Athlete_ID'].nunique()} athletes)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "athlete_participation_frequency.png"), dpi=300)
        plt.close()

        df["Sex"] = df["Sex"].astype("category")
        df["Athlete_ID"] = df["Athlete_ID"].astype("category")
        df.to_csv(data_used_path, index=False, encoding="utf-8-sig")
        log_write(lf, f"Saved model input data: {data_used_path}")

        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 5 - Fitting 3 models for comparison")
        log_write(lf, "=" * 80)

        formula_a = "Sinclair_Total ~ C(Sex) + C(JetLagGroup_abs)"
        formula_b = "Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir)"
        formula_c = "Sinclair_Total ~ C(Sex) + JetLag_signed"

        model_a = smf.mixedlm(formula_a, data=df, groups=df["Athlete_ID"])
        res_a_reml = model_a.fit(method="lbfgs")
        res_a_ml = model_a.fit(method="lbfgs", reml=False)
        icc_a = compute_icc(res_a_reml)

        model_b = smf.mixedlm(formula_b, data=df, groups=df["Athlete_ID"])
        res_b_reml = model_b.fit(method="lbfgs")
        res_b_ml = model_b.fit(method="lbfgs", reml=False)
        icc_b = compute_icc(res_b_reml)

        model_c = smf.mixedlm(formula_c, data=df, groups=df["Athlete_ID"])
        res_c_reml = model_c.fit(method="lbfgs")
        res_c_ml = model_c.fit(method="lbfgs", reml=False)
        icc_c = compute_icc(res_c_reml)

        formula_b_reduced = "Sinclair_Total ~ C(Sex)"
        res_b_reduced_ml = smf.mixedlm(formula_b_reduced, data=df, groups=df["Athlete_ID"]).fit(method="lbfgs", reml=False)

        llf_full = float(res_b_ml.llf)
        llf_reduced = float(res_b_reduced_ml.llf)
        lr_stat = 2.0 * (llf_full - llf_reduced)
        df_lrt = int(res_b_ml.k_fe - res_b_reduced_ml.k_fe)
        p_lrt = float(chi2.sf(lr_stat, df_lrt)) if lr_stat >= 0 else np.nan

        llf_c_full = float(res_c_ml.llf)
        llf_c_reduced = float(res_b_reduced_ml.llf)
        lr_stat_c = 2.0 * (llf_c_full - llf_c_reduced)
        df_lrt_c = int(res_c_ml.k_fe - res_b_reduced_ml.k_fe)
        p_lrt_c = float(chi2.sf(lr_stat_c, df_lrt_c)) if lr_stat_c >= 0 else np.nan

        jetlag_terms = [n for n in res_b_reml.fe_params.index if "JetLagGroup_dir" in n]
        ci_df = res_b_reml.conf_int()
        emm_rows: list[dict[str, float | str]] = []
        for term in jetlag_terms:
            m = re.search(r"\[T\.(.+?)\]", term)
            group = m.group(1) if m else term
            emm_rows.append(
                {
                    "Group": group,
                    "Estimate": float(res_b_reml.fe_params[term]),
                    "SE": float(res_b_reml.bse[term]),
                    "CI_lo": float(ci_df.loc[term, 0]),
                    "CI_hi": float(ci_df.loc[term, 1]),
                    "p_raw": float(res_b_reml.pvalues[term]),
                }
            )
        emm_df = pd.DataFrame(emm_rows)
        _, p_fdr_arr, _, _ = multipletests(emm_df["p_raw"].values, method="fdr_bh")
        emm_df["p_fdr"] = p_fdr_arr
        emm_df["sig_raw"] = emm_df["p_raw"].apply(sig_stars)
        emm_df["sig_fdr"] = emm_df["p_fdr"].apply(sig_stars)
        emm_df.to_csv(os.path.join(OUTPUT_DIR, "emmeans_FDR.csv"), index=False, encoding="utf-8-sig", float_format="%.6f")

        fe_names = list(res_b_reml.fe_params.index)
        fe_idx = {n: i for i, n in enumerate(fe_names)}
        contrast_lines: list[str] = []

        def contrast_east_vs_west(label: str, east_term: str, west_term: str) -> None:
            if east_term not in fe_idx or west_term not in fe_idx:
                contrast_lines.append(f"  {label:<18}  [SKIP]")
                return
            v = np.zeros((1, res_b_reml.k_fe), dtype=float)
            v[0, fe_idx[east_term]] = 1.0
            v[0, fe_idx[west_term]] = -1.0
            t = res_b_reml.t_test(v)
            est = float(np.asarray(t.effect).flat[0])
            se = float(np.asarray(t.sd).flat[0])
            p = float(np.asarray(t.pvalue).flat[0])
            contrast_lines.append(f"  {label:<18}  diff={est:+.4f}  se={se:.4f}  p={p:.4f} {sig_stars(p)}")

        contrast_east_vs_west("2-3h: East-West", "C(JetLagGroup_dir)[T.East_2-3h]", "C(JetLagGroup_dir)[T.West_2-3h]")
        contrast_east_vs_west("4-5h: East-West", "C(JetLagGroup_dir)[T.East_4-5h]", "C(JetLagGroup_dir)[T.West_4-5h]")
        contrast_east_vs_west("6-7h: East-West", "C(JetLagGroup_dir)[T.East_6-7h]", "C(JetLagGroup_dir)[T.West_6-7h]")
        contrast_east_vs_west("8-9h: East-West", "C(JetLagGroup_dir)[T.East_8-9h]", "C(JetLagGroup_dir)[T.West_8-9h]")
        contrast_east_vs_west("10-11h: East-West", "C(JetLagGroup_dir)[T.East_10-11h]", "C(JetLagGroup_dir)[T.West_10-11h]")

        aic_a, bic_a = res_a_ml.aic, res_a_ml.bic
        aic_b, bic_b = res_b_ml.aic, res_b_ml.bic
        aic_c, bic_c = res_c_ml.aic, res_c_ml.bic

        comp_lines = [
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
            "=" * 90,
            "Model Comparison: Abs-only vs Directional vs Continuous",
            "=" * 90,
            "",
            f"{'Model':<12} {'Formula':<60} {'AIC':>10} {'BIC':>10} {'ICC':>8}",
            "-" * 100,
            f"{'A (abs)':<12} {formula_a:<60} {aic_a:>10.1f} {bic_a:>10.1f} {icc_a:>8.4f}",
            f"{'B (dir) *':<12} {formula_b:<60} {aic_b:>10.1f} {bic_b:>10.1f} {icc_b:>8.4f}",
            f"{'C (cont)':<12} {formula_c:<60} {aic_c:>10.1f} {bic_c:>10.1f} {icc_c:>8.4f}",
            "",
            "=" * 90,
            "Model B (Directional) - JetLagGroup coefficients",
            "  Reference group: Minimal(0-1h)",
            "  Positive coef = higher Sinclair than ref; Negative = lower",
            "-" * 90,
        ]
        for name, val in res_b_reml.params.items():
            if "JetLagGroup_dir" in name:
                pval = float(res_b_reml.pvalues.get(name, np.nan))
                comp_lines.append(f"  {name:<55} coef={val:+.4f}  p={pval:.4f} {sig_stars(pval)}")

        comp_lines.extend(
            [
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
                "=" * 90,
                "Post-hoc contrasts: each group vs Minimal(0-1h)  [FDR-BH corrected]",
                "  (No Sex × JetLagGroup_dir interaction → model coefficient = emmeans contrast)",
                f"  {'Group':<20} {'Estimate':>9} {'SE':>7} {'95%CI':^20} {'p_raw':>9} {'sig':>4}  {'p_fdr':>9} {'sig_fdr':>7}",
                "  " + "-" * 88,
            ]
        )
        for _, row in emm_df.iterrows():
            ci_str = f"[{row['CI_lo']:+.3f}, {row['CI_hi']:+.3f}]"
            comp_lines.append(
                f"  {row['Group']:<20} {row['Estimate']:>+9.4f} {row['SE']:>7.4f} "
                f"{ci_str:^20} {row['p_raw']:>9.4f} {row['sig_raw']:>4}  {row['p_fdr']:>9.4f} {row['sig_fdr']:>7}"
            )
        comp_lines.extend(
            [
                "  Reference: Minimal(0-1h)",
                "  Sig codes: *** p<.001  ** p<.01  * p<.05  . p<.1",
                "",
                "=" * 90,
                "Contrast Tests: East vs West at same magnitude",
                "  H0: coef(East_Xh) - coef(West_Xh) = 0",
                "  (both relative to Minimal(0-1h), so difference = East effect - West effect)",
                "-" * 90,
            ]
        )
        comp_lines.extend(contrast_lines)
        comp_lines.append("=" * 90)

        with open(compare_path, "w", encoding="utf-8") as f:
            f.write("\n".join(comp_lines) + "\n")

        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write("=" * 80 + "\n")
            sf.write("Model A: Abs-only JetLagGroup (travel-available subset)\n")
            sf.write(f"Formula: {formula_a} + (1|Athlete_ID)\n")
            sf.write(f"ICC: {icc_a:.4f}\n\n")
            sf.write(res_a_reml.summary().as_text())
            sf.write("\n\n" + "=" * 80 + "\n")
            sf.write("Model B: Directional JetLagGroup (travel-available subset)\n")
            sf.write(f"Formula: {formula_b} + (1|Athlete_ID)\n")
            sf.write("Ref group: Minimal(0-1h)\n")
            sf.write(f"ICC: {icc_b:.4f}\n\n")
            sf.write(res_b_reml.summary().as_text())
            sf.write("\n\n" + "=" * 80 + "\n")
            sf.write("Model C: Continuous JetLag_signed (travel-available subset)\n")
            sf.write(f"Formula: {formula_c} + (1|Athlete_ID)\n")
            sf.write(f"ICC: {icc_c:.4f}\n\n")
            sf.write(res_c_reml.summary().as_text())

        # 保留一个基础分布图，便于人工复核子样本
        plt.figure(figsize=(11, 6))
        sns.histplot(df["JetLag_signed"], bins=25, kde=True, color="#4C78A8")
        plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        plt.title("JetLag (signed circular difference) distribution - travel-available subset")
        plt.xlabel("JetLag_signed (hours): negative=Westward, positive=Eastward")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "jetlag_distribution.png"), dpi=300)
        plt.close()

        log_write(lf, f"Model comparison saved: {compare_path}")
        log_write(lf, f"Model summaries saved: {summary_path}")
        log_write(lf, "All output files saved to: " + os.path.abspath(OUTPUT_DIR))
        log_write(lf, "=" * 80)
        log_write(lf, "LMM JetLag V2 (Travel-available subset) - END")
        log_write(lf, "=" * 80)


if __name__ == "__main__":
    main()
