"""
LMM_jetlag_by_sex.py（最终修正版）

基于原版代码的最小改动，与 LMM_jetlag.py 保持一致：
  1. ★ 方向性分组 7 档：Minimal(0-3h) 合并做参照组 + East/West 各 3 档
  2. 每个性别拟合 2 个模型：A(abs-only) / B(directional)
  3. 新增 Contrast Test：同量级 East vs West
  4. combined_overview 输出东西方向系数对比

输出目录：lmm_jetlag_by_sex/
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

INPUT_CSV = "iwf_data_top13_athlete2plus_circular.csv"
OUTPUT_DIR = "lmm_jetlag_by_sex"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABS_GROUP_ORDER = ["0-2h", "2-4h", "4-6h", "6-8h", "8-10h", "10-12h"]
DIRECTIONAL_GROUP_ORDER = [
    "Minimal(0-2h)",
    "West_2-4h", "West_4-6h", "West_6-8h", "West_8-10h", "West_10-12h",
    "East_2-4h", "East_4-6h", "East_6-8h", "East_8-10h", "East_10-12h",
]
DIRECTIONAL_PLOT_ORDER = [
    "West_10-12h", "West_8-10h", "West_6-8h", "West_4-6h", "West_2-4h",
    "Minimal(0-2h)",
    "East_2-4h", "East_4-6h", "East_6-8h", "East_8-10h", "East_10-12h",
]


# ======================== 工具函数 ========================

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


def extract_sex_from_event(event):
    if not isinstance(event, str): return None
    m = re.search(r"(Men|Women)", event, flags=re.IGNORECASE)
    return m.group(1).title() if m else None


def circular_jetlag_signed(venue_off, athlete_off):
    if pd.isna(venue_off) or pd.isna(athlete_off): return np.nan
    d = float(venue_off) - float(athlete_off)
    if abs(d) <= 12: return d
    return -np.sign(d) * (24 - abs(d))


def build_jetlag_group_abs(jetlag_abs):
    if pd.isna(jetlag_abs): return None
    x = float(jetlag_abs)
    if x < 2: return "0-2h"
    if x < 4: return "2-4h"
    if x < 6: return "4-6h"
    if x < 8: return "6-8h"
    if x < 10: return "8-10h"
    return "10-12h"


def build_jetlag_group_directional(jetlag_signed):
    """11档方向性分组（2h一档），Minimal(0-2h)做参照组。"""
    if pd.isna(jetlag_signed): return None
    x = float(jetlag_signed)
    absx = abs(x)
    if absx < 2: return "Minimal(0-2h)"
    if x > 0:
        if absx < 4: return "East_2-4h"
        if absx < 6: return "East_4-6h"
        if absx < 8: return "East_6-8h"
        if absx < 10: return "East_8-10h"
        return "East_10-12h"
    else:
        if absx < 4: return "West_2-4h"
        if absx < 6: return "West_4-6h"
        if absx < 8: return "West_6-8h"
        if absx < 10: return "West_8-10h"
        return "West_10-12h"


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f: f.write(text)


def compute_icc(result):
    try:
        gv = float(result.cov_re.iloc[0, 0]); rv = float(result.scale)
        return gv / (gv + rv) if (gv + rv) > 0 else np.nan
    except Exception: return np.nan


def run_one_sex(df_all: pd.DataFrame, sex: str) -> dict:
    out_dir = os.path.join(OUTPUT_DIR, sex)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "analysis_log.txt")
    summary_path = os.path.join(out_dir, "model_summary.txt")
    data_path = os.path.join(out_dir, "data_used_for_model.csv")

    lines = []
    def log(msg): lines.append(msg)

    log("=" * 80)
    log(f"LMM JetLag by Sex - {sex}")
    log(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    log("=" * 80)

    df = df_all[df_all["Sex"] == sex].copy()
    log(f"Rows after sex filter: {len(df)}")

    # JetLag
    df["JetLag_signed"] = df.apply(
        lambda r: circular_jetlag_signed(r["Venue_UTC_offset"], r["Athlete_UTC_offset"]), axis=1)
    df["JetLag_abs"] = df["JetLag_signed"].abs()
    df["JetLagGroup_abs"] = df["JetLag_abs"].apply(build_jetlag_group_abs)
    df["JetLagGroup_dir"] = df["JetLag_signed"].apply(build_jetlag_group_directional)

    df = df.dropna(subset=["JetLagGroup_abs", "JetLagGroup_dir"]).copy()
    df["JetLagGroup_abs"] = pd.Categorical(df["JetLagGroup_abs"], categories=ABS_GROUP_ORDER, ordered=True)
    df["JetLagGroup_dir"] = pd.Categorical(df["JetLagGroup_dir"], categories=DIRECTIONAL_GROUP_ORDER, ordered=True)

    log("\nJetLagGroup_dir counts:")
    log(df["JetLagGroup_dir"].value_counts().sort_index().to_string())

    # Sinclair
    df["Sinclair_Total"] = df.apply(lambda r: calculate_sinclair(r["Total"], r["Body_Weight"], r["Sex"]), axis=1)
    df = df.dropna(subset=["Sinclair_Total"]).copy()
    log(f"\nRows after clean: {len(df)}")
    log(f"Athletes: {df['Athlete_ID'].nunique()}")

    df["Athlete_ID"] = df["Athlete_ID"].astype("category")
    df.to_csv(data_path, index=False, encoding="utf-8-sig")

    # --- Model A: abs-only (baseline) ---
    formula_a = "Sinclair_Total ~ C(JetLagGroup_abs)"
    model_a = smf.mixedlm(formula_a, data=df, groups=df["Athlete_ID"])
    res_a = model_a.fit(method="lbfgs")
    icc_a = compute_icc(res_a)
    res_a_ml = model_a.fit(method="lbfgs", reml=False)

    # --- Model B: directional (★ fix) ---
    formula_b = "Sinclair_Total ~ C(JetLagGroup_dir)"
    model_b = smf.mixedlm(formula_b, data=df, groups=df["Athlete_ID"])
    res_b = model_b.fit(method="lbfgs")
    icc_b = compute_icc(res_b)
    res_b_ml = model_b.fit(method="lbfgs", reml=False)

    log(f"\nModel A (abs-only): AIC={res_a_ml.aic:.1f}, BIC={res_a_ml.bic:.1f}, ICC={icc_a:.4f}")
    log(f"Model B (directional): AIC={res_b_ml.aic:.1f}, BIC={res_b_ml.bic:.1f}, ICC={icc_b:.4f}")
    log(f"dAIC(B-A) = {res_b_ml.aic - res_a_ml.aic:+.1f}")

    # Contrast tests for Model B
    log("\nContrast Tests (Model B): East vs West at same magnitude")
    fe_names = list(res_b.fe_params.index)
    fe_idx = {n: i for i, n in enumerate(fe_names)}

    contrast_results = {}
    for mag_label, east_key, west_key in [
        ("2-4h", "C(JetLagGroup_dir)[T.East_2-4h]", "C(JetLagGroup_dir)[T.West_2-4h]"),
        ("4-6h", "C(JetLagGroup_dir)[T.East_4-6h]", "C(JetLagGroup_dir)[T.West_4-6h]"),
        ("6-8h", "C(JetLagGroup_dir)[T.East_6-8h]", "C(JetLagGroup_dir)[T.West_6-8h]"),
        ("8-10h", "C(JetLagGroup_dir)[T.East_8-10h]", "C(JetLagGroup_dir)[T.West_8-10h]"),
        ("10-12h", "C(JetLagGroup_dir)[T.East_10-12h]", "C(JetLagGroup_dir)[T.West_10-12h]"),
    ]:
        if east_key not in fe_idx or west_key not in fe_idx:
            log(f"  {mag_label}: [SKIP] param not found in fe_params")
            continue
        v = np.zeros((1, res_b.k_fe), dtype=float)
        v[0, fe_idx[east_key]] = 1.0
        v[0, fe_idx[west_key]] = -1.0
        try:
            t = res_b.t_test(v)
            est = float(np.asarray(t.effect).flat[0])
            se  = float(np.asarray(t.sd).flat[0])
            p   = float(np.asarray(t.pvalue).flat[0])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
            log(f"  {mag_label}: East-West diff={est:+.4f}, se={se:.4f}, p={p:.4f} {sig}")
            contrast_results[mag_label] = {"diff": est, "p": p}
        except Exception as e:
            log(f"  {mag_label}: [FAILED] {e}")

    # Write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\nModel A: Abs-only ({sex})\n")
        f.write(f"Formula: {formula_a} + (1|Athlete_ID)\nICC: {icc_a:.4f}\n\n")
        f.write(res_a.summary().as_text())
        f.write(f"\n\n{'='*80}\n* Model B: Directional ({sex})\n")
        f.write(f"Formula: {formula_b} + (1|Athlete_ID)\n")
        f.write(f"Ref group: Minimal(0-3h)\nICC: {icc_b:.4f}\n\n")
        f.write(res_b.summary().as_text())

    # Plots
    plt.figure(figsize=(13, 6))
    sns.boxplot(data=df, x="JetLagGroup_dir", y="Sinclair_Total",
                palette="Set2", order=DIRECTIONAL_PLOT_ORDER)
    plt.title(f"Sinclair_Total by Directional JetLagGroup ({sex})")
    plt.xlabel("JetLagGroup (West <-- --> East)")
    plt.ylabel("Sinclair_Total")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sinclair_by_jetlaggroup_directional.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.histplot(df["JetLag_signed"], bins=25, kde=True, color="#4C78A8")
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.title(f"JetLag_signed distribution ({sex})")
    plt.xlabel("JetLag_signed (hours)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "jetlag_signed_distribution.png"), dpi=300)
    plt.close()

    # 边际效应（Model B）
    try:
        pred_rows = [{"JetLagGroup_dir": g} for g in DIRECTIONAL_GROUP_ORDER]
        pred_df = pd.DataFrame(pred_rows)
        pred_df["JetLagGroup_dir"] = pd.Categorical(
            pred_df["JetLagGroup_dir"], categories=DIRECTIONAL_GROUP_ORDER, ordered=True)
        pred_df["Pred_Sinclair"] = res_b.predict(pred_df)

        plt.figure(figsize=(13, 6))
        sns.pointplot(data=pred_df, x="JetLagGroup_dir", y="Pred_Sinclair",
                      markers="o", color="#4C78A8", order=DIRECTIONAL_PLOT_ORDER)
        plt.title(f"Model B: Predicted Sinclair_Total ({sex})")
        plt.xlabel("JetLagGroup (West <-- --> East)")
        plt.ylabel("Predicted Sinclair_Total")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "marginal_effects_directional.png"), dpi=300)
        plt.close()
    except Exception as e:
        log(f"[WARN] Marginal effects plot failed: {e}")

    write_text(log_path, "\n".join(lines) + "\n")

    # 返回概览信息
    dir_params = {k: v for k, v in res_b.params.items() if "JetLagGroup_dir" in k}
    dir_pvals  = {k: res_b.pvalues.get(k, np.nan) for k in dir_params}
    return {
        "sex": sex, "rows": len(df),
        "athletes": int(df["Athlete_ID"].nunique()),
        "icc_a": icc_a, "icc_b": icc_b,
        "aic_a": res_a_ml.aic, "aic_b": res_b_ml.aic,
        "bic_a": res_a_ml.bic, "bic_b": res_b_ml.bic,
        "dir_params": dir_params, "dir_pvals": dir_pvals,
        "contrasts": contrast_results,
    }


def main() -> None:
    setup_plot_style()
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"输入文件不存在：{INPUT_CSV}")

    df_all = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df_all.columns = df_all.columns.str.strip()
    for c in ["Total", "Body_Weight", "Athlete_UTC_offset", "Venue_UTC_offset"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    df_all["Sex"] = df_all["Event"].apply(extract_sex_from_event)
    df_all = df_all.dropna(subset=["Total", "Body_Weight", "Sex", "Athlete_ID",
                                    "Athlete_UTC_offset", "Venue_UTC_offset"]).copy()

    results = []
    for sex in ["Men", "Women"]:
        if (df_all["Sex"] == sex).sum() == 0: continue
        results.append(run_one_sex(df_all, sex))

    # ======================== Combined overview ========================
    ov = []
    ov.append("LMM JetLag by Sex - Overview (* with Directional Groups)")
    ov.append("=" * 90)
    ov.append(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    ov.append("")

    for r in results:
        ov.append(f"[{r['sex']}] rows={r['rows']}, athletes={r['athletes']}")
        ov.append(f"  Model A (abs-only):    AIC={r['aic_a']:.1f}, BIC={r['bic_a']:.1f}, ICC={r['icc_a']:.4f}")
        ov.append(f"  Model B (directional): AIC={r['aic_b']:.1f}, BIC={r['bic_b']:.1f}, ICC={r['icc_b']:.4f}")
        ov.append(f"  dAIC(B-A) = {r['aic_b'] - r['aic_a']:+.1f}  (negative = B better)")
        ov.append("")

        ov.append(f"  Model B coefficients (ref=Minimal(0-3h)):")
        for k in DIRECTIONAL_PLOT_ORDER:
            for pk in r["dir_params"]:
                if k in pk:
                    coef = r["dir_params"][pk]
                    pval = r["dir_pvals"].get(pk, np.nan)
                    sig = ""
                    if not pd.isna(pval):
                        if pval < 0.001: sig = "***"
                        elif pval < 0.01: sig = "**"
                        elif pval < 0.05: sig = "*"
                        elif pval < 0.1: sig = "."
                    ov.append(f"    {k:<20} coef={coef:+.4f}  p={pval:.4f} {sig}")
                    break
        ov.append("")

        # Contrast results
        if r.get("contrasts"):
            ov.append(f"  Contrast Tests (East vs West, {r['sex']}):")
            for mag, ct in r["contrasts"].items():
                sig = "*" if ct["p"] < 0.05 else ""
                ov.append(f"    {mag}: diff={ct['diff']:+.4f}, p={ct['p']:.4f} {sig}")
            ov.append("")

    ov.append("=" * 90)
    ov.append("Key question: Do East and West show different effects at the same magnitude?")
    ov.append("If contrast p < 0.05, directional modeling captures real asymmetry.")
    ov.append("If not, the direction effect may be too small to detect with this sample.")
    ov.append("=" * 90)

    write_text(os.path.join(OUTPUT_DIR, "combined_overview.txt"), "\n".join(ov) + "\n")


if __name__ == "__main__":
    main()