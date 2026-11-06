from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from matplotlib import font_manager
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")


JETLAG_ORDER = [
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

JETLAG_PLOT_ORDER = [
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


@dataclass
class AnalysisConfig:
    name: str
    display_name: str
    group_col: str
    group_levels: list[str]
    # 函数本身作为对象传来传去，函数的作用是"接收原始 DataFrame，返回带 Group 列的 DataFrame"。
    constructor: Callable[[pd.DataFrame, argparse.Namespace], pd.DataFrame]
    controls: list[str]
    include_group_main_effect: bool
    interaction_enabled: bool
    stratified_controls: list[str]
    min_stratum_n: int = 60
    extra_interactions: list[str] = field(default_factory=list)


class AnalysisLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fp = open(path, "w", encoding="utf-8")

    def close(self) -> None:
        self.fp.close()

    def line(self, text: str = "") -> None:
        self.fp.write(text + "\n")
        self.fp.flush()

    def section(self, title: str) -> None:
        self.line("\n" + "=" * 88)
        self.line(title)
        self.line("=" * 88)

    def kv(self, key: str, value: Any) -> None:
        self.line(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heterogeneity LMM framework")
    parser.add_argument(
        "--input",
        type=str,
        default="athlete_top13_travel.csv",
        help="输入CSV路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="heterogeneity_results",
        help="输出目录",
    )
    parser.add_argument(
        "--analysis",
        nargs="+",
        choices=["sex", "age", "experience", "weight", "age_adjusted"],
        default=["sex", "age", "experience", "weight", "age_adjusted"],
        help="要运行的分析",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["4", "3", "auto"],
        default="auto",
        help="体重分级模式：4级/3级/自动降级",
    )
    parser.add_argument(
        "--hard-threshold",
        type=int,
        default=3,
        help="min_obs_check硬阈值（N<该值）",
    )
    parser.add_argument(
        "--warn-threshold",
        type=int,
        default=5,
        help="min_obs_check警告阈值（N<该值）",
    )
    parser.add_argument(
        "--hard-cell-limit",
        type=int,
        default=6,
        help="当N<hard阈值的cell超过该值时，体重分析自动降级",
    )
    parser.add_argument(
        "--warn-cell-limit",
        type=int,
        default=18,
        help="当N<warn阈值的cell超过该值时，体重分析自动降级",
    )
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="运行补充敏感性分析（连续变量交互）",
    )
    return parser.parse_args()


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["axes.unicode_minus"] = False
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/Deng.ttf",
    ]
    chosen = "Arial"
    for fp in font_paths:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            chosen = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["font.sans-serif"] = [chosen, "Arial Unicode MS", "Arial"]


def extract_sex(event: Any) -> str | None:
    if not isinstance(event, str):
        return None
    text = event.lower()
    if "women" in text:
        return "Women"
    if "men" in text:
        return "Men"
    return None


def calculate_sinclair(total: Any, body_weight: Any, sex: Any) -> float:
    try:
        total_f = float(total)
        bw = float(body_weight)
    except Exception:
        return np.nan

    params = {
        "Men": {"A": 0.722762521, "b": 193.609},
        "Women": {"A": 0.787004341, "b": 153.757},
    }
    sex_s = str(sex).strip().title()
    if sex_s not in params or bw <= 0:
        return np.nan
    A = params[sex_s]["A"]
    b = params[sex_s]["b"]
    if bw > b:
        coeff = 1.0
    else:
        x = np.log10(bw / b)
        coeff = 10 ** (A * x**2)
    return total_f * coeff


def circular_jetlag_signed(venue: Any, athlete: Any) -> float:
    if pd.isna(venue) or pd.isna(athlete):
        return np.nan
    d = float(venue) - float(athlete)
    if abs(d) <= 12:
        return d
    return -np.sign(d) * (24 - abs(d))


def build_jetlag_group_directional(jetlag_signed: Any) -> str | None:
    if pd.isna(jetlag_signed):
        return None
    x = float(jetlag_signed)
    abs_x = abs(x)
    if abs_x < 2:
        return "Minimal(0-1h)"
    if x > 0:
        if abs_x < 4:
            return "East_2-3h"
        if abs_x < 6:
            return "East_4-5h"
        if abs_x < 8:
            return "East_6-7h"
        if abs_x < 10:
            return "East_8-9h"
        return "East_10-11h"
    if abs_x < 4:
        return "West_2-3h"
    if abs_x < 6:
        return "West_4-5h"
    if abs_x < 8:
        return "West_6-7h"
    if abs_x < 10:
        return "West_8-9h"
    return "West_10-11h"


def prepare_base_dataframe(input_csv: str, logger: AnalysisLogger) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    logger.section("阶段1：数据预处理")
    logger.kv("原始行数", len(df))

    # 与正式主模型保持完全相同的分析样本。最终交付数据含2,186条建模记录，
    # 异质性分析不再自行drop_duplicates，以免与主模型产生N=2,184的口径差异。
    logger.kv("样本处理", f"保留全部 {len(df)} 条输入记录（不额外去重）")

    numeric_cols = ["Total", "Body_Weight", "Athlete_UTC_offset", "Venue_UTC_offset", "TravelDuration"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Sex"] = df["Event"].apply(extract_sex)

    if "JetLag_signed" not in df.columns:
        df["JetLag_signed"] = df.apply(
            lambda r: circular_jetlag_signed(r["Venue_UTC_offset"], r["Athlete_UTC_offset"]),
            axis=1,
        )
        logger.line("[INFO] JetLag_signed 不存在，已按 Athlete/Venue UTC offset 重新计算。")
    else:
        df["JetLag_signed"] = pd.to_numeric(df["JetLag_signed"], errors="coerce")
        logger.line("[INFO] JetLag_signed 已存在，直接读取并转数值。")
    df["JetLag_abs"] = df["JetLag_signed"].abs()
    df["JetLagGroup_dir"] = df["JetLag_signed"].apply(build_jetlag_group_directional)
    df["JetLagGroup_dir"] = pd.Categorical(df["JetLagGroup_dir"], categories=JETLAG_ORDER, ordered=True)

    if "Sinclair_Total" not in df.columns:
        df["Sinclair_Total"] = df.apply(
            lambda r: calculate_sinclair(r["Total"], r["Body_Weight"], r["Sex"]),
            axis=1,
        )
        logger.line("[INFO] Sinclair_Total 不存在，已按 IWF 参数计算。")
    else:
        df["Sinclair_Total"] = pd.to_numeric(df["Sinclair_Total"], errors="coerce")
        logger.line("[INFO] Sinclair_Total 已存在，直接读取并转数值。")

    if "Reference_Date" in df.columns:
        df["Reference_Date"] = pd.to_datetime(df["Reference_Date"], errors="coerce")
    else:
        df["Reference_Date"] = pd.NaT

    born_candidates = ["Born", "Birth_Date", "BornDate"]
    born_col = next((c for c in born_candidates if c in df.columns), None)
    if born_col is not None:
        df["Born_dt"] = pd.to_datetime(df[born_col], errors="coerce")
        miss_ratio = float(df["Born_dt"].isna().mean())
        if miss_ratio > 0.5:
            logger.line(f"[WARN] {born_col} 解析后缺失率 {miss_ratio:.1%}，年龄分析可能不稳定。")
    else:
        df["Born_dt"] = pd.NaT
        logger.line("[WARN] 未找到出生日期列（Born/Birth_Date/BornDate），年龄分析将失败。")

    df["Age"] = (df["Reference_Date"] - df["Born_dt"]).dt.days / 365.25
    df["Age_c"] = df["Age"] - df["Age"].mean(skipna=True)

    df["TravelDuration_c"] = df["TravelDuration"] - df["TravelDuration"].mean(skipna=True)
    df["BodyWeight_c"] = df["Body_Weight"] - df["Body_Weight"].mean(skipna=True)

    df["__row_id"] = np.arange(len(df))
    sort_cols = ["Athlete_ID", "Reference_Date", "__row_id"]
    df = df.sort_values(sort_cols, na_position="last").copy()
    df["Cumulative_Exp"] = df.groupby("Athlete_ID").cumcount() + 1
    df["Cumulative_Exp_c"] = df["Cumulative_Exp"] - df["Cumulative_Exp"].mean(skipna=True)

    required = [
        "Sinclair_Total",
        "Sex",
        "Athlete_ID",
        "JetLagGroup_dir",
        "TravelDuration_c",
        "Body_Weight",
        "BodyWeight_c",
    ]
    before_drop = len(df)
    df = df.dropna(subset=required).copy()
    logger.kv("核心变量缺失清理", f"{before_drop} -> {len(df)}")
    logger.kv("运动员数量", df["Athlete_ID"].nunique())
    logger.line("\nJetLagGroup_dir 分布:")
    logger.line(df["JetLagGroup_dir"].value_counts().sort_index().to_string())
    return df


def construct_sex_group(df: pd.DataFrame, _: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["Group"] = out["Sex"]
    out["Group"] = pd.Categorical(out["Group"], categories=["Men", "Women"], ordered=True)
    return out


def construct_age_group(df: pd.DataFrame, _: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["Age"]).copy()
    out["Group"] = np.where(out["Age"] < 25, "Young", "Old")
    out["Group"] = pd.Categorical(out["Group"], categories=["Young", "Old"], ordered=True)
    return out


def construct_experience_group(df: pd.DataFrame, _: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["Group"] = np.where(out["Cumulative_Exp"] <= 2, "Low", "High")
    out["Cumulative_Exp_c"] = out["Cumulative_Exp"] - out["Cumulative_Exp"].mean(skipna=True)
    out["Group"] = pd.Categorical(out["Group"], categories=["Low", "High"], ordered=True)
    return out


def classify_weight_4level(row: pd.Series) -> str | None:
    sex = row["Sex"]
    bw = row["Body_Weight"]
    if pd.isna(bw) or sex not in ("Men", "Women"):
        return None
    bw = float(bw)
    if sex == "Men":
        if bw <= 67:
            return "Light"
        if bw <= 89:
            return "Middle"
        if bw <= 110:
            return "Heavy"
        return "SuperHeavy"
    if bw <= 58:
        return "Light"
    if bw <= 77:
        return "Middle"
    if bw <= 86:
        return "Heavy"
    return "SuperHeavy"


def classify_weight_3level(row: pd.Series) -> str | None:
    group4 = classify_weight_4level(row)
    if group4 in ("Heavy", "SuperHeavy"):
        return "HeavyPlus"
    return group4


def min_obs_check(
    df: pd.DataFrame,
    group_col: str,
    warn_threshold: int,
    hard_threshold: int,
) -> tuple[pd.DataFrame, int, int]:
    ctab = pd.crosstab(df[group_col], df["JetLagGroup_dir"], dropna=False)
    hard_cells = int((ctab < hard_threshold).sum().sum())
    warn_cells = int((ctab < warn_threshold).sum().sum())
    return ctab, warn_cells, hard_cells


def construct_weight_group(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    mode = args.weight_mode
    if mode == "3":
        out["Group"] = out.apply(classify_weight_3level, axis=1)
        levels = ["Light", "Middle", "HeavyPlus"]
    else:
        out["Group"] = out.apply(classify_weight_4level, axis=1)
        levels = ["Light", "Middle", "Heavy", "SuperHeavy"]
    out = out.dropna(subset=["Group"]).copy()
    out["Group"] = pd.Categorical(out["Group"], categories=levels, ordered=True)

    if mode == "auto":
        ctab, warn_cells, hard_cells = min_obs_check(
            out,
            "Group",
            warn_threshold=args.warn_threshold,
            hard_threshold=args.hard_threshold,
        )
        if (hard_cells > args.hard_cell_limit) or (warn_cells > args.warn_cell_limit):
            out["Group"] = out.apply(classify_weight_3level, axis=1)
            out["Group"] = pd.Categorical(out["Group"], categories=["Light", "Middle", "HeavyPlus"], ordered=True)
            out.attrs["weight_mode_effective"] = "3(auto-fallback)"
            out.attrs["weight_sparse_table"] = ctab
            return out
    out.attrs["weight_mode_effective"] = "4"
    return out


def build_configs() -> dict[str, AnalysisConfig]:
    return {
        "sex": AnalysisConfig(
            name="sex",
            display_name="性别",
            group_col="Group",
            group_levels=["Men", "Women"],
            constructor=construct_sex_group,
            controls=["TravelDuration_c"],
            include_group_main_effect=True,
            interaction_enabled=True,
            stratified_controls=["TravelDuration_c"],
            min_stratum_n=80,
        ),
        "age": AnalysisConfig(
            name="age",
            display_name="年龄",
            group_col="Group",
            group_levels=["Young", "Old"],
            constructor=construct_age_group,
            controls=["C(Sex)", "TravelDuration_c", "BodyWeight_c"],
            include_group_main_effect=True,
            interaction_enabled=True,
            stratified_controls=["C(Sex)", "TravelDuration_c", "BodyWeight_c"],
            min_stratum_n=80,
        ),
        "age_adjusted": AnalysisConfig(
            name="age_adjusted",
            display_name="年龄(经验调整)",
            group_col="Group",
            group_levels=["Young", "Old"],
            constructor=construct_age_group,
            controls=["C(Sex)", "TravelDuration_c", "BodyWeight_c", "Cumulative_Exp_c"],
            include_group_main_effect=True,
            interaction_enabled=True,
            stratified_controls=["C(Sex)", "TravelDuration_c", "BodyWeight_c", "Cumulative_Exp_c"],
            min_stratum_n=80,
            extra_interactions=["Cumulative_Exp_c:C(JetLagGroup_dir)"],
        ),
        "experience": AnalysisConfig(
            name="experience",
            display_name="经验",
            group_col="Group",
            group_levels=["Low", "High"],
            constructor=construct_experience_group,
            controls=["C(Sex)", "TravelDuration_c", "BodyWeight_c"],
            include_group_main_effect=True,
            interaction_enabled=True,
            stratified_controls=["C(Sex)", "TravelDuration_c", "BodyWeight_c"],
            min_stratum_n=80,
        ),
        "weight": AnalysisConfig(
            name="weight",
            display_name="体重级别",
            group_col="Group",
            group_levels=["Light", "Middle", "Heavy", "SuperHeavy"],
            constructor=construct_weight_group,
            controls=["C(Sex)", "TravelDuration_c"],
            include_group_main_effect=True,
            interaction_enabled=True,
            stratified_controls=["C(Sex)", "TravelDuration_c"],
            min_stratum_n=100,
        ),
    }


def build_formulas(config: AnalysisConfig) -> tuple[str, str]:
    jetlag_term = "C(JetLagGroup_dir)"
    terms_full: list[str] = []
    terms_reduced: list[str] = []

    terms_full.extend(config.controls)
    terms_reduced.extend(config.controls)
    if config.include_group_main_effect:
        terms_full.append("C(Group)")
        terms_reduced.append("C(Group)")
    else:
        if "C(Sex)" not in terms_full:
            terms_full.append("C(Sex)")
        if "C(Sex)" not in terms_reduced:
            terms_reduced.append("C(Sex)")
    terms_full.append(jetlag_term)
    terms_reduced.append(jetlag_term)

    if config.interaction_enabled:
        terms_full.append("C(Group):C(JetLagGroup_dir)")
    if config.extra_interactions:
        terms_full.extend(config.extra_interactions)
        terms_reduced.extend(config.extra_interactions)

    full_formula = "Sinclair_Total ~ " + " + ".join(terms_full)
    reduced_formula = "Sinclair_Total ~ " + " + ".join(terms_reduced)
    return full_formula, reduced_formula


def fit_mixedlm(formula: str, df: pd.DataFrame, reml: bool) -> Any:
    model = smf.mixedlm(formula, data=df, groups=df["Athlete_ID"])
    methods = ["lbfgs", "powell", "cg"]
    last_exc = None
    for method in methods:
        try:
            return model.fit(method=method, reml=reml, maxiter=1000)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"MixedLM failed for formula={formula}. Last error={last_exc}")


def lrt(full_ml: Any, reduced_ml: Any) -> dict[str, float]:
    lr = 2.0 * (float(full_ml.llf) - float(reduced_ml.llf))
    df_diff = int(full_ml.k_fe - reduced_ml.k_fe)
    p_val = np.nan if lr < 0 else float(chi2.sf(lr, df_diff))
    return {"lr": lr, "df": df_diff, "p": p_val}


def extract_design_a_table(
    strat_results: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level, res in strat_results.items():
        ci = res.conf_int()
        rows.append(
            {
                "design": "A",
                "group_level": level,
                "jetlag_group": "Minimal(0-1h)",
                "estimate": 0.0,
                "se": np.nan,
                "ci_lo": 0.0,
                "ci_hi": 0.0,
                "p_raw": np.nan,
            }
        )
        for jg in JETLAG_ORDER[1:]:
            term = f"C(JetLagGroup_dir)[T.{jg}]"
            if term not in res.fe_params.index:
                rows.append(
                    {
                        "design": "A",
                        "group_level": level,
                        "jetlag_group": jg,
                        "estimate": np.nan,
                        "se": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "p_raw": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "design": "A",
                    "group_level": level,
                    "jetlag_group": jg,
                    "estimate": float(res.fe_params[term]),
                    "se": float(res.bse[term]),
                    "ci_lo": float(ci.loc[term, 0]),
                    "ci_hi": float(ci.loc[term, 1]),
                    "p_raw": float(res.pvalues[term]),
                }
            )
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    for level in out["group_level"].dropna().unique():
        mask = (out["group_level"] == level) & out["p_raw"].notna()
        if mask.sum() > 0:
            _, qvals, _, _ = multipletests(out.loc[mask, "p_raw"], method="fdr_bh")
            out.loc[mask, "p_fdr"] = qvals
    return out


def _interaction_term_candidates(non_ref_group: str, jetlag: str) -> list[str]:
    return [
        f"C(Group)[T.{non_ref_group}]:C(JetLagGroup_dir)[T.{jetlag}]",
        f"C(JetLagGroup_dir)[T.{jetlag}]:C(Group)[T.{non_ref_group}]",
    ]


def find_interaction_term(fe_index: pd.Index, non_ref_group: str, jetlag: str) -> str | None:
    for cand in _interaction_term_candidates(non_ref_group, jetlag):
        if cand in fe_index:
            return cand
    # 回退：模糊匹配，兼容不同版本 statsmodels 的命名差异
    group_token = rf"{non_ref_group}"
    jetlag_token = rf"{jetlag}"
    for term in fe_index:
        if ":" not in term:
            continue
        if (group_token in term) and (jetlag_token in term):
            return term
    return None


def extract_design_b_table(full_reml: Any, group_levels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ref = group_levels[0]
    ci = full_reml.conf_int()
    for group in group_levels[1:]:
        for jg in JETLAG_ORDER[1:]:
            term = find_interaction_term(full_reml.fe_params.index, group, jg)
            label = f"{group}_vs_{ref}"
            if term is None:
                rows.append(
                    {
                        "design": "B",
                        "group_level": label,
                        "jetlag_group": jg,
                        "estimate": np.nan,
                        "se": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "p_raw": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "design": "B",
                    "group_level": label,
                    "jetlag_group": jg,
                    "estimate": float(full_reml.fe_params[term]),
                    "se": float(full_reml.bse[term]),
                    "ci_lo": float(ci.loc[term, 0]),
                    "ci_hi": float(ci.loc[term, 1]),
                    "p_raw": float(full_reml.pvalues[term]),
                }
            )
    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    mask = out["p_raw"].notna()
    if mask.sum() > 0:
        _, qvals, _, _ = multipletests(out.loc[mask, "p_raw"], method="fdr_bh")
        out.loc[mask, "p_fdr"] = qvals
    return out


def fit_stratified_models(df: pd.DataFrame, config: AnalysisConfig, logger: AnalysisLogger) -> dict[str, Any]:
    results: dict[str, Any] = {}
    rhs = " + ".join(config.stratified_controls + ["C(JetLagGroup_dir)"])
    formula = "Sinclair_Total ~ " + rhs
    logger.section("分层拟合（设计A）")
    logger.kv("分层公式", formula)
    for level in config.group_levels:
        subset = df[df["Group"] == level].copy()
        if len(subset) < config.min_stratum_n:
            logger.line(f"[WARN] {level} 样本过少（n={len(subset)}），跳过分层拟合")
            continue
        try:
            results[level] = fit_mixedlm(formula, subset, reml=True)
            logger.line(f"{level}: ok (n={len(subset)})")
        except Exception as exc:
            logger.line(f"{level}: failed -> {exc}")
    return results


def star_from_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_effect_lines(df_a: pd.DataFrame, df_b: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    x_map = {k: i for i, k in enumerate(JETLAG_PLOT_ORDER)}
    plot_df = df_a.copy()
    plot_df = plot_df[plot_df["jetlag_group"].isin(JETLAG_PLOT_ORDER)]
    plot_df["x"] = plot_df["jetlag_group"].map(x_map)

    for level, sub in plot_df.groupby("group_level"):
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["estimate"], marker="o", label=level)
        yerr = np.vstack(
            [
                (sub["estimate"] - sub["ci_lo"]).fillna(0).to_numpy(),
                (sub["ci_hi"] - sub["estimate"]).fillna(0).to_numpy(),
            ]
        )
        ax.errorbar(sub["x"], sub["estimate"], yerr=yerr, fmt="none", capsize=3, alpha=0.5)

    star_df = (
        df_b.groupby("jetlag_group", as_index=False)["p_fdr"]
        .min()
        .dropna()
    )
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    star_y = ymax + 0.06 * span
    for _, row in star_df.iterrows():
        p = float(row["p_fdr"])
        s = star_from_p(p)
        if not s:
            continue
        x = x_map.get(row["jetlag_group"])
        if x is None:
            continue
        ax.text(x, star_y, s, ha="center", va="center", color="black", fontsize=12)
    ax.set_ylim(ymin, ymax + 0.12 * span)

    ax.set_xticks(range(len(JETLAG_PLOT_ORDER)))
    ax.set_xticklabels(JETLAG_PLOT_ORDER, rotation=15)
    ax.set_ylabel("Effect vs Minimal(0-1h)")
    ax.set_xlabel("JetLagGroup_dir")
    ax.set_title(title)
    ax.legend(title="Group")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_forest_a(df_a: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    data = df_a[df_a["jetlag_group"] != "Minimal(0-1h)"].copy()
    data["label"] = data["jetlag_group"] + " | " + data["group_level"]
    data = data.sort_values(["jetlag_group", "group_level"])
    y = np.arange(len(data))
    ax.errorbar(
        data["estimate"],
        y,
        xerr=[data["estimate"] - data["ci_lo"], data["ci_hi"] - data["estimate"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=3,
    )
    ax.axvline(0, linestyle="--", color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("Effect vs Minimal(0-1h)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_cell_counts(ctab: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(ctab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def run_one_analysis(
    base_df: pd.DataFrame,
    config: AnalysisConfig,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"{config.name}_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = AnalysisLogger(out_dir / "analysis_log.txt")
    logger.section(f"{config.display_name}分析开始")

    try:
        df = config.constructor(base_df, args)
        df = df.dropna(subset=["Group", "JetLagGroup_dir"]).copy()
        df["Group"] = df["Group"].cat.remove_unused_categories()
        logger.kv("样本量", len(df))
        logger.kv("Group levels", list(df["Group"].cat.categories))

        ctab, warn_cells, hard_cells = min_obs_check(
            df,
            "Group",
            warn_threshold=args.warn_threshold,
            hard_threshold=args.hard_threshold,
        )
        logger.section("min_obs_check")
        logger.kv(f"N<{args.warn_threshold} 单元格数量", warn_cells)
        logger.kv(f"N<{args.hard_threshold} 单元格数量", hard_cells)
        logger.line(ctab.to_string())
        plot_cell_counts(ctab, out_dir / "cell_counts_heatmap.png", f"{config.display_name}: Group x JetLag counts")

        if config.name == "weight":
            mode_effective = df.attrs.get("weight_mode_effective", args.weight_mode)
            logger.kv("体重分级模式(生效)", mode_effective)
            sparse_table = df.attrs.get("weight_sparse_table")
            if sparse_table is not None:
                logger.line("四级模式原始稀疏交叉表（触发自动降级依据）：")
                logger.line(sparse_table.to_string())

        full_formula, reduced_formula = build_formulas(config)
        logger.section("模型拟合")
        logger.kv("Full formula", full_formula)
        logger.kv("Reduced formula", reduced_formula)

        full_reml = fit_mixedlm(full_formula, df, reml=True)
        full_ml = fit_mixedlm(full_formula, df, reml=False)
        reduced_ml = fit_mixedlm(reduced_formula, df, reml=False)

        lrt_res = lrt(full_ml, reduced_ml)
        logger.section("全局交互LRT")
        logger.kv("LR", f"{lrt_res['lr']:.6f}")
        logger.kv("df", lrt_res["df"])
        logger.kv("p", f"{lrt_res['p']:.6e}" if not pd.isna(lrt_res["p"]) else "NA")

        strat_results = fit_stratified_models(df, config, logger)
        design_a = extract_design_a_table(strat_results)
        design_b = extract_design_b_table(full_reml, list(df["Group"].cat.categories))
        design_b["lrt_significant"] = bool((not pd.isna(lrt_res["p"])) and (lrt_res["p"] < 0.05))
        contrasts = pd.concat([design_a, design_b], ignore_index=True)

        contrasts["sig_raw"] = contrasts["p_raw"].apply(star_from_p)
        contrasts["sig_fdr"] = contrasts["p_fdr"].apply(star_from_p)
        contrasts.to_csv(out_dir / "results_contrasts.csv", index=False, encoding="utf-8-sig")

        coef_rows = []
        ci = full_reml.conf_int()
        for name, val in full_reml.fe_params.items():
            coef_rows.append(
                {
                    "term": name,
                    "estimate": float(val),
                    "se": float(full_reml.bse[name]),
                    "ci_lo": float(ci.loc[name, 0]),
                    "ci_hi": float(ci.loc[name, 1]),
                    "p": float(full_reml.pvalues[name]),
                }
            )
        pd.DataFrame(coef_rows).to_csv(out_dir / "full_model_coefficients.csv", index=False, encoding="utf-8-sig")

        with open(out_dir / "model_summary.txt", "w", encoding="utf-8") as f:
            f.write(full_reml.summary().as_text())
        with open(out_dir / "model_comparison.txt", "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        f"analysis={config.name}",
                        f"full_formula={full_formula}",
                        f"reduced_formula={reduced_formula}",
                        f"llf_full_ml={full_ml.llf:.6f}",
                        f"llf_reduced_ml={reduced_ml.llf:.6f}",
                        f"LR={lrt_res['lr']:.6f}",
                        f"df={lrt_res['df']}",
                        f"p={lrt_res['p']:.6e}" if not pd.isna(lrt_res["p"]) else "p=NA",
                        f"aic_full={full_ml.aic:.3f}",
                        f"bic_full={full_ml.bic:.3f}",
                    ]
                )
                + "\n"
            )

        plot_effect_lines(
            design_a,
            design_b,
            out_dir / "effects_line_with_stars.png",
            f"{config.display_name}: 组内效应(线) + 组间差异显著性(*)",
        )
        plot_forest_a(
            design_a,
            out_dir / "effects_forest_designA.png",
            f"{config.display_name}: 设计A 分层效应森林图",
        )

        logger.section("输出文件")
        logger.line("results_contrasts.csv")
        logger.line("full_model_coefficients.csv")
        logger.line("model_summary.txt")
        logger.line("model_comparison.txt")
        logger.line("effects_line_with_stars.png")
        logger.line("effects_forest_designA.png")
        logger.line("cell_counts_heatmap.png")
        return {
            "analysis": config.name,
            "display_name": config.display_name,
            "output_dir": str(out_dir),
            "n": len(df),
            "lrt_lr": lrt_res["lr"],
            "lrt_df": lrt_res["df"],
            "lrt_p": lrt_res["p"],
            "failed": False,
        }
    except Exception as exc:
        logger.section("错误")
        logger.line(str(exc))
        return {
            "analysis": config.name,
            "display_name": config.display_name,
            "output_dir": str(out_dir),
            "failed": True,
            "error": str(exc),
        }
    finally:
        logger.close()


def run_continuous_sensitivity(
    base_df: pd.DataFrame,
    var_name: str,
    display_name: str,
    controls: list[str],
    output_root: Path,
) -> dict[str, Any]:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"sensitivity_{var_name}_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = AnalysisLogger(out_dir / "analysis_log.txt")
    logger.section(f"敏感性分析开始：{display_name}")

    try:
        df = base_df.dropna(subset=[var_name, "JetLag_signed", "JetLagGroup_dir", "Athlete_ID", "Sinclair_Total"]).copy()
        logger.kv("样本量", len(df))
        logger.kv("连续变量", var_name)

        # 主敏感性：连续 × 连续（1个交互参数）
        full_main = "Sinclair_Total ~ " + " + ".join(controls + ["JetLag_signed", var_name, f"{var_name}:JetLag_signed"])
        reduced_main = "Sinclair_Total ~ " + " + ".join(controls + ["JetLag_signed", var_name])
        logger.section("主敏感性（连续×连续）")
        logger.kv("Full formula", full_main)
        logger.kv("Reduced formula", reduced_main)

        full_main_reml = fit_mixedlm(full_main, df, reml=True)
        full_main_ml = fit_mixedlm(full_main, df, reml=False)
        reduced_main_ml = fit_mixedlm(reduced_main, df, reml=False)
        lrt_main = lrt(full_main_ml, reduced_main_ml)

        interaction_term = f"{var_name}:JetLag_signed"
        coef_main = float(full_main_reml.fe_params.get(interaction_term, np.nan))
        p_main = float(full_main_reml.pvalues.get(interaction_term, np.nan))
        logger.kv("主交互项", interaction_term)
        logger.kv("coef", f"{coef_main:.6f}")
        logger.kv("p", f"{p_main:.6e}" if not pd.isna(p_main) else "NA")
        logger.kv("LRT(LR,df,p)", f"{lrt_main['lr']:.6f}, {lrt_main['df']}, {lrt_main['p']:.6e}")

        # 次级敏感性：连续 × 分类（10个交互参数）
        full_secondary = "Sinclair_Total ~ " + " + ".join(
            controls + ["C(JetLagGroup_dir)", var_name, f"{var_name}:C(JetLagGroup_dir)"]
        )
        reduced_secondary = "Sinclair_Total ~ " + " + ".join(controls + ["C(JetLagGroup_dir)", var_name])
        logger.section("次级敏感性（连续×分类）")
        logger.kv("Full formula", full_secondary)
        logger.kv("Reduced formula", reduced_secondary)

        full_sec_reml = fit_mixedlm(full_secondary, df, reml=True)
        full_sec_ml = fit_mixedlm(full_secondary, df, reml=False)
        reduced_sec_ml = fit_mixedlm(reduced_secondary, df, reml=False)
        lrt_secondary = lrt(full_sec_ml, reduced_sec_ml)
        logger.kv("LRT(LR,df,p)", f"{lrt_secondary['lr']:.6f}, {lrt_secondary['df']}, {lrt_secondary['p']:.6e}")

        sec_rows = []
        ci_sec = full_sec_reml.conf_int()
        for term, coef in full_sec_reml.fe_params.items():
            if (var_name in term) and ("JetLagGroup_dir" in term):
                sec_rows.append(
                    {
                        "term": term,
                        "estimate": float(coef),
                        "se": float(full_sec_reml.bse[term]),
                        "ci_lo": float(ci_sec.loc[term, 0]),
                        "ci_hi": float(ci_sec.loc[term, 1]),
                        "p_raw": float(full_sec_reml.pvalues[term]),
                    }
                )
        sec_df = pd.DataFrame(sec_rows)
        if not sec_df.empty:
            _, p_fdr, _, _ = multipletests(sec_df["p_raw"], method="fdr_bh")
            sec_df["p_fdr"] = p_fdr
            sec_df["sig_fdr"] = sec_df["p_fdr"].apply(star_from_p)
        sec_df.to_csv(out_dir / "secondary_interactions.csv", index=False, encoding="utf-8-sig")

        with open(out_dir / "model_comparison.txt", "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        f"sensitivity_var={var_name}",
                        f"primary_full={full_main}",
                        f"primary_reduced={reduced_main}",
                        f"primary_lr={lrt_main['lr']:.6f}",
                        f"primary_df={lrt_main['df']}",
                        f"primary_p={lrt_main['p']:.6e}",
                        f"primary_interaction_coef={coef_main:.6f}",
                        f"primary_interaction_p={p_main:.6e}" if not pd.isna(p_main) else "primary_interaction_p=NA",
                        "",
                        f"secondary_full={full_secondary}",
                        f"secondary_reduced={reduced_secondary}",
                        f"secondary_lr={lrt_secondary['lr']:.6f}",
                        f"secondary_df={lrt_secondary['df']}",
                        f"secondary_p={lrt_secondary['p']:.6e}",
                    ]
                )
                + "\n"
            )

        return {
            "analysis": f"sensitivity_{var_name}",
            "display_name": f"敏感性-{display_name}",
            "output_dir": str(out_dir),
            "n": len(df),
            "lrt_lr": lrt_main["lr"],
            "lrt_df": lrt_main["df"],
            "lrt_p": lrt_main["p"],
            "failed": False,
        }
    except Exception as exc:
        logger.section("错误")
        logger.line(str(exc))
        return {
            "analysis": f"sensitivity_{var_name}",
            "display_name": f"敏感性-{display_name}",
            "output_dir": str(out_dir),
            "failed": True,
            "error": str(exc),
        }
    finally:
        logger.close()


def write_summary_report(results: list[dict[str, Any]], output_root: Path) -> None:
    lines = [
        "# Heterogeneity Summary",
        "",
        f"Run time: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Analysis | N | LR | df | p | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if r.get("failed"):
            lines.append(f"| {r['analysis']} | - | - | - | - | failed |")
        else:
            p_text = "NA" if pd.isna(r["lrt_p"]) else f"{r['lrt_p']:.3e}"
            lines.append(
                f"| {r['analysis']} | {r['n']} | {r['lrt_lr']:.3f} | {r['lrt_df']} | {p_text} | ok |"
            )
    with open(output_root / "heterogeneity_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    setup_plot_style()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    init_logger = AnalysisLogger(output_root / "run_log.txt")
    base_df = prepare_base_dataframe(args.input, init_logger)
    base_df.to_csv(output_root / "base_dataframe_snapshot.csv", index=False, encoding="utf-8-sig")
    init_logger.close()

    configs = build_configs()
    results: list[dict[str, Any]] = []
    for name in args.analysis:
        res = run_one_analysis(base_df, configs[name], args, output_root)
        results.append(res)

    if args.run_sensitivity:
        sensitivity_specs = [
            ("Age_c", "年龄(连续中心化)", ["C(Sex)", "TravelDuration_c", "BodyWeight_c"]),
            ("BodyWeight_c", "体重(连续中心化)", ["C(Sex)", "TravelDuration_c"]),
            ("Cumulative_Exp_c", "经验(连续中心化)", ["C(Sex)", "TravelDuration_c", "BodyWeight_c"]),
        ]
        for var_name, display_name, controls in sensitivity_specs:
            if var_name not in base_df.columns:
                continue
            res = run_continuous_sensitivity(
                base_df=base_df,
                var_name=var_name,
                display_name=display_name,
                controls=controls,
                output_root=output_root,
            )
            results.append(res)

    write_summary_report(results, output_root)


if __name__ == "__main__":
    main()

