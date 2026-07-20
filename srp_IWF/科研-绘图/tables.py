"""
生成 Table 1/2/3 与 Supplementary Table 1（CSV）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import (
    find_latest_subdir,
    parse_heterogeneity_summary_md,
    parse_sensitivity_model_comparison,
    parse_travel_control_lrts,
)
from plot_config import PROJECT_ROOT


def calculate_sinclair(total: object, body_weight: object, sex: object) -> float:
    """IWF Sinclair Total（2021–2024），与 LMM_jetlag.py 一致。"""
    try:
        total_f = float(total)
        bw = float(body_weight)
    except Exception:
        return float("nan")
    params = {"Men": {"A": 0.722762521, "b": 193.609}, "Women": {"A": 0.787004341, "b": 153.757}}
    s = str(sex).strip().title()
    if s not in params or bw <= 0:
        return float("nan")
    a, b = params[s]["A"], params[s]["b"]
    if bw > b:
        sc = 1.0
    else:
        x = np.log10(bw / b)
        sc = 10 ** (a * x**2)
    return total_f * sc


def _mean_sd(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return ""
    return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"


def _median_iqr(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return ""
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.1f} [{q1:.1f}–{q3:.1f}]"


def build_table1_descriptive(data_csv: Path) -> pd.DataFrame:
    """Table 1：基于主分析样本（含 TravelDuration 的 2186 子样本）。"""
    df = pd.read_csv(data_csv, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if "Sex" not in df.columns:
        df["Sex"] = df["Event"].astype(str).str.contains("Women", case=False, na=False)
        df["Sex"] = np.where(
            df["Sex"],
            "Women",
            np.where(df["Event"].astype(str).str.contains("Men", case=False, na=False), "Men", np.nan),
        )

    df["Reference_Date"] = pd.to_datetime(df.get("Reference_Date"), errors="coerce")
    df["Born_dt"] = pd.to_datetime(df.get("Born"), errors="coerce")
    df["Age"] = (df["Reference_Date"] - df["Born_dt"]).dt.days / 365.25
    df["Body_Weight"] = pd.to_numeric(df.get("Body_Weight"), errors="coerce")
    if "Sinclair_Total" not in df.columns:
        df["Sinclair_Total"] = df.apply(
            lambda r: calculate_sinclair(r.get("Total"), r.get("Body_Weight"), r.get("Sex")),
            axis=1,
        )
    else:
        df["Sinclair_Total"] = pd.to_numeric(df["Sinclair_Total"], errors="coerce")

    out_rows = []
    for sex_key, title in [("Men", "Men"), ("Women", "Women"), ("All", "Total")]:
        if sex_key == "All":
            sub = df.copy()
        else:
            sub = df[df["Sex"] == sex_key].copy()
        n_obs = len(sub)
        n_ath = sub["Athlete_ID"].nunique() if "Athlete_ID" in sub.columns else 0
        miss_age = int(sub["Age"].isna().sum())
        miss_bw = int(sub["Body_Weight"].isna().sum())
        miss_sin = int(sub["Sinclair_Total"].isna().sum())
        opa = sub.groupby("Athlete_ID").size() if "Athlete_ID" in sub.columns else pd.Series(dtype=int)
        out_rows.append(
            {
                "Stratum": title,
                "Athletes_n": n_ath,
                "Observations_n": n_obs,
                "Age_mean_SD": _mean_sd(sub["Age"]),
                "Body_mass_kg_mean_SD": _mean_sd(sub["Body_Weight"]),
                "Sinclair_Total_mean_SD": _mean_sd(sub["Sinclair_Total"]),
                "Obs_per_athlete_median_IQR": _median_iqr(opa),
                "Missing_Age_n": miss_age,
                "Missing_Body_mass_n": miss_bw,
                "Missing_Sinclair_n": miss_sin,
            }
        )

    # 转为宽表（Table 1 三列 Men/Women/Total）
    wide = {}
    metrics = [
        ("Athletes (n)", "Athletes_n"),
        ("Observations (n)", "Observations_n"),
        ("Age (yr), mean ± SD", "Age_mean_SD"),
        ("Body mass (kg), mean ± SD", "Body_mass_kg_mean_SD"),
        ("Sinclair Total, mean ± SD", "Sinclair_Total_mean_SD"),
        ("Obs. per athlete, median [IQR]", "Obs_per_athlete_median_IQR"),
        ("Missing: Age (n)", "Missing_Age_n"),
        ("Missing: Body mass (n)", "Missing_Body_mass_n"),
        ("Missing: Sinclair (n)", "Missing_Sinclair_n"),
    ]
    for mlabel, key in metrics:
        wide[mlabel] = {
            "Men": str(out_rows[0][key]),
            "Women": str(out_rows[1][key]),
            "Total": str(out_rows[2][key]),
        }
    return pd.DataFrame.from_dict(wide, orient="index").reset_index().rename(columns={"index": "Metric"})


def build_table2_posthoc(
    emmeans_csv: Path,
    model_data_csv: Path,
) -> pd.DataFrame:
    """Table 2：事后比较 + 每组观测数。"""
    emm = pd.read_csv(emmeans_csv, encoding="utf-8-sig")
    dat = pd.read_csv(model_data_csv, encoding="utf-8-sig")
    dat.columns = dat.columns.str.strip()
    counts = dat.groupby("JetLagGroup_dir").size().rename("n_obs").reset_index()
    counts.rename(columns={"JetLagGroup_dir": "Group"}, inplace=True)
    out = emm.merge(counts, on="Group", how="left")
    order = [
        "West_10-11h",
        "West_8-9h",
        "West_6-7h",
        "West_4-5h",
        "West_2-3h",
        "East_2-3h",
        "East_4-5h",
        "East_6-7h",
        "East_8-9h",
        "East_10-11h",
    ]
    cat = pd.Categorical(out["Group"], categories=order, ordered=True)
    out["Group"] = cat
    return out.sort_values("Group").reset_index(drop=True)


def build_table3_heterogeneity(summary_md: Path) -> pd.DataFrame:
    df = parse_heterogeneity_summary_md(summary_md)
    if df.empty:
        return df
    # 统一列名
    df.columns = [c.strip() for c in df.columns]
    keep = df[~df["Analysis"].str.startswith("sensitivity", na=False)].copy()
    name_map = {
        "sex": ("Sex", "Men vs. Women"),
        "age": ("Age", "<25 vs. ≥25 yr"),
        "experience": ("Experience", "≤2 vs. ≥3 starts"),
        "weight": ("Weight class", "4 categories"),
    }
    rows = []
    for _, r in keep.iterrows():
        aid = str(r["Analysis"]).strip()
        if aid not in name_map:
            continue
        mod, defin = name_map[aid]
        rows.append(
            {
                "Moderator": mod,
                "Definition": defin,
                "LR": float(r["LR"]),
                "df": int(float(r["df"])),
                "p": float(r["p"]),
            }
        )
    return pd.DataFrame(rows)


def build_supp_table1_sensitivity(
    travel_control_txt: Path,
    hetero_root: Path,
) -> pd.DataFrame:
    """附录表：旅行控制 LRT + 连续敏感性 primary LRT。"""
    rows: list[dict[str, object]] = []
    txt = travel_control_txt.read_text(encoding="utf-8")
    lrts = parse_travel_control_lrts(txt)
    if lrts:
        rows.append(
            {
                "Test": "LRT-A: Travel adds vs jetlag-only",
                "LR": lrts.get("lrt_a_lr"),
                "df": lrts.get("lrt_a_df"),
                "p": lrts.get("lrt_a_p"),
            }
        )
        rows.append(
            {
                "Test": "LRT-B: JetLag joint vs travel-only (control travel)",
                "LR": lrts.get("lrt_b_lr"),
                "df": lrts.get("lrt_b_df"),
                "p": lrts.get("lrt_b_p"),
            }
        )
    for var, label in [
        ("Age_c", "Sensitivity: Age × JetLag_signed (primary LRT)"),
        ("BodyWeight_c", "Sensitivity: Body weight × JetLag_signed (primary LRT)"),
        ("Cumulative_Exp_c", "Sensitivity: Cumulative exp × JetLag_signed (primary LRT)"),
    ]:
        pdir = hetero_root
        try:
            latest = find_latest_subdir(pdir, f"sensitivity_{var}_*")
            mc = latest / "model_comparison.txt"
            d = parse_sensitivity_model_comparison(mc)
            rows.append(
                {
                    "Test": label,
                    "LR": d.get("primary_lr"),
                    "df": int(d["primary_df"]) if "primary_df" in d else None,
                    "p": d.get("primary_p"),
                }
            )
        except FileNotFoundError:
            rows.append({"Test": label, "LR": None, "df": None, "p": None})

    return pd.DataFrame(rows)


def write_all_tables(out_tables: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)
    primary_dir = PROJECT_ROOT / "lmm_jetlag_travel_control"
    main_data = primary_dir / "data_used_for_model.csv"
    build_table1_descriptive(main_data).to_csv(out_tables / "table1_descriptive.csv", index=False, encoding="utf-8-sig")

    emm = primary_dir / "emmeans_FDR_travel_control.csv"
    dused = primary_dir / "data_used_for_model.csv"
    build_table2_posthoc(emm, dused).to_csv(out_tables / "table2_posthoc_fdr.csv", index=False, encoding="utf-8-sig")

    summ = PROJECT_ROOT / "heterogeneity_results" / "heterogeneity_summary.md"
    build_table3_heterogeneity(summ).to_csv(out_tables / "table3_heterogeneity_lrt.csv", index=False, encoding="utf-8-sig")

    tc = PROJECT_ROOT / "lmm_jetlag_travel_control" / "model_comparison.txt"
    hroot = PROJECT_ROOT / "heterogeneity_results"
    build_supp_table1_sensitivity(tc, hroot).to_csv(
        out_tables / "supp_table1_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
