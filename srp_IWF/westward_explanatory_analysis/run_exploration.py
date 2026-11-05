from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


SEED = 20261031
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
INPUT_CSV = (
    PROJECT_ROOT
    / "交付材料_20260720"
    / "01_最终数据"
    / "final_model_data_N2186.csv"
)
LOCKED_EFFECT_CSV = (
    PROJECT_ROOT
    / "交付材料_20260720"
    / "03_统计结果"
    / "01_主模型_控制旅行时长_travel_control"
    / "emmeans_FDR_travel_control.csv"
)
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"
LOG_PATH = OUTPUTS / "analysis_log.txt"

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
TARGET_GROUPS = ["West_10-11h", "East_8-9h", "East_10-11h"]
LOCKED_N = 2186
LOCKED_ATHLETES = 730
LOCKED_WEST_N = 160
LOCKED_EAST_EXTREME_N = 266
LOCKED_WEST_BETA = 5.1653
LOCKED_WEST_FDR = 0.0307
TOLERANCE_BETA = 0.001


@dataclass
class ModelFit:
    label: str
    formula: str
    n: int
    athletes: int
    success: bool
    converged: bool
    result: object | None
    error: str = ""
    warnings_text: str = ""
    design_rank: int | None = None
    design_columns: int | None = None


def ensure_dirs() -> None:
    for path in (OUTPUTS, FIGURES, NOTES):
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def metadata(stage: str, df: pd.DataFrame, filters: str) -> dict[str, object]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "input_path": str(INPUT_CSV),
        "input_rows": int(len(df)),
        "formula": "Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir) + TravelDuration_c + (1|Athlete_ID)",
        "filters": filters,
        "random_seed": SEED,
    }


def save_metadata(stage: str, df: pd.DataFrame, filters: str) -> None:
    path = OUTPUTS / f"metadata_{stage}.json"
    path.write_text(
        json.dumps(metadata(stage, df, filters), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"正式数据不存在：{INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {
        "Athlete_ID",
        "Nation",
        "Sex",
        "Event_Year",
        "competition",
        "City",
        "JetLag_signed",
        "JetLag_abs",
        "JetLagGroup_dir",
        "TravelDuration",
        "Sinclair_Total",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"正式数据缺少必要字段：{missing}")
    for col in ["JetLag_signed", "JetLag_abs", "TravelDuration", "Sinclair_Total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(required)].isna().any().any():
        bad = df[list(required)].isna().sum()
        raise ValueError(f"正式数据核心字段存在缺失：\n{bad[bad > 0].to_string()}")
    df["EventEdition"] = (
        df["Event_Year"].astype(int).astype(str)
        + " | "
        + df["competition"].astype(str).str.strip()
        + " | "
        + df["City"].astype(str).str.strip()
    )
    return prepare_subset(df)


def prepare_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TravelDuration_c"] = out["TravelDuration"] - out["TravelDuration"].mean()
    out["JetLagGroup_dir"] = pd.Categorical(
        out["JetLagGroup_dir"], categories=DIRECTIONAL_GROUP_ORDER, ordered=True
    )
    out["Sex"] = pd.Categorical(out["Sex"], categories=["Men", "Women"], ordered=True)
    out["Athlete_ID"] = out["Athlete_ID"].astype("category")
    out["Nation"] = out["Nation"].astype("category")
    out["EventEdition"] = out["EventEdition"].astype("category")
    return out


def fixed_formula(extra_terms: Iterable[str] = ()) -> str:
    terms = ["C(Sex)", "C(JetLagGroup_dir)", "TravelDuration_c", *extra_terms]
    return "Sinclair_Total ~ " + " + ".join(terms)


def design_rank(formula: str, df: pd.DataFrame) -> tuple[int, int]:
    matrix = patsy.dmatrix(formula.split("~", 1)[1], df, return_type="dataframe")
    return int(np.linalg.matrix_rank(matrix.to_numpy())), int(matrix.shape[1])


def fit_mixed_model(
    df: pd.DataFrame,
    label: str,
    extra_terms: Iterable[str] = (),
    reml: bool = True,
) -> ModelFit:
    work = prepare_subset(df)
    formula = fixed_formula(extra_terms)
    try:
        rank, columns = design_rank(formula, work)
    except Exception as exc:
        return ModelFit(label, formula, len(work), work["Athlete_ID"].nunique(), False, False, None, f"design-matrix error: {exc}")
    if rank < columns:
        return ModelFit(
            label,
            formula,
            len(work),
            work["Athlete_ID"].nunique(),
            False,
            False,
            None,
            f"rank deficient: rank={rank}, columns={columns}",
            design_rank=rank,
            design_columns=columns,
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = smf.mixedlm(formula, work, groups=work["Athlete_ID"]).fit(
                method="lbfgs", reml=reml, maxiter=2000, disp=False
            )
            warning_text = " | ".join(str(w.message) for w in caught)
            return ModelFit(
                label,
                formula,
                len(work),
                work["Athlete_ID"].nunique(),
                True,
                bool(getattr(result, "converged", False)),
                result,
                warnings_text=warning_text,
                design_rank=rank,
                design_columns=columns,
            )
        except Exception as exc:
            warning_text = " | ".join(str(w.message) for w in caught)
            return ModelFit(
                label,
                formula,
                len(work),
                work["Athlete_ID"].nunique(),
                False,
                False,
                None,
                str(exc),
                warning_text,
                rank,
                columns,
            )


def extract_jetlag_effects(fit: ModelFit) -> pd.DataFrame:
    columns = [
        "model",
        "n",
        "athletes",
        "converged",
        "group",
        "estimate",
        "se",
        "ci_low",
        "ci_high",
        "p_raw",
        "p_fdr",
        "fdr_significant",
    ]
    if not fit.success or fit.result is None:
        return pd.DataFrame(columns=columns)
    result = fit.result
    ci = result.conf_int()
    rows: list[dict[str, object]] = []
    for name, estimate in result.fe_params.items():
        if "JetLagGroup_dir" not in name:
            continue
        match = re.search(r"\[T\.(.+?)\]", name)
        group = match.group(1) if match else name
        rows.append(
            {
                "model": fit.label,
                "n": fit.n,
                "athletes": fit.athletes,
                "converged": fit.converged,
                "group": group,
                "estimate": float(estimate),
                "se": float(result.bse[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
                "p_raw": float(result.pvalues[name]),
            }
        )
    effects = pd.DataFrame(rows)
    if len(effects) != 10:
        raise RuntimeError(f"{fit.label}仅提取到{len(effects)}个非参照时差系数，预期为10个")
    effects["p_fdr"] = multipletests(effects["p_raw"], method="fdr_bh")[1]
    effects["fdr_significant"] = effects["p_fdr"] < 0.05
    return effects[columns]


def target_effects(effects: pd.DataFrame) -> pd.DataFrame:
    return effects[effects["group"].isin(TARGET_GROUPS)].copy()


def validate_baseline(df: pd.DataFrame, effects: pd.DataFrame) -> None:
    failures: list[str] = []
    if len(df) != LOCKED_N:
        failures.append(f"N={len(df)}，预期{LOCKED_N}")
    if df["Athlete_ID"].nunique() != LOCKED_ATHLETES:
        failures.append(f"运动员={df['Athlete_ID'].nunique()}，预期{LOCKED_ATHLETES}")
    counts = df["JetLagGroup_dir"].value_counts()
    if int(counts.get("West_10-11h", 0)) != LOCKED_WEST_N:
        failures.append(f"West 10–11 h={counts.get('West_10-11h', 0)}，预期{LOCKED_WEST_N}")
    if int(counts.get("East_10-11h", 0)) != LOCKED_EAST_EXTREME_N:
        failures.append(f"East 10–11 h={counts.get('East_10-11h', 0)}，预期{LOCKED_EAST_EXTREME_N}")
    west = effects.loc[effects["group"] == "West_10-11h"].iloc[0]
    if abs(float(west["estimate"]) - LOCKED_WEST_BETA) > TOLERANCE_BETA:
        failures.append(
            f"West β={west['estimate']:.6f}，锁定值={LOCKED_WEST_BETA:.4f}，超出容差{TOLERANCE_BETA}"
        )
    locked = pd.read_csv(LOCKED_EFFECT_CSV, encoding="utf-8-sig")
    locked_west = locked.loc[locked["Group"] == "West_10-11h"].iloc[0]
    if abs(float(west["estimate"]) - float(locked_west["Estimate"])) > TOLERANCE_BETA:
        failures.append("West β与锁定效果CSV不一致")
    if abs(float(west["p_fdr"]) - float(locked_west["p_fdr"])) > 0.001:
        failures.append(
            f"West FDR={west['p_fdr']:.6f}，锁定CSV={locked_west['p_fdr']:.6f}"
        )
    if failures:
        raise RuntimeError("基线复现失败：" + "；".join(failures))


def save_figure(fig: plt.Figure, stem: str) -> None:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stage_boundary(df: pd.DataFrame) -> dict[str, object]:
    log("阶段0/1：复现主模型并执行±12 h边界核验")
    save_metadata("boundary", df, "完整样本；敏感性模型仅排除JetLag_abs==12")
    full_fit = fit_mixed_model(df, "full_N2186")
    if not full_fit.success:
        raise RuntimeError(f"完整样本模型失败：{full_fit.error}")
    full_effects = extract_jetlag_effects(full_fit)
    validate_baseline(df, full_effects)
    full_effects.to_csv(OUTPUTS / "baseline_full_effects.csv", index=False, encoding="utf-8-sig")

    boundary = df.loc[np.isclose(df["JetLag_abs"], 12.0)].copy()
    neg_n = int(np.isclose(boundary["JetLag_signed"], -12.0).sum())
    pos_n = int(np.isclose(boundary["JetLag_signed"], 12.0).sum())
    if (len(boundary), neg_n, pos_n) != (103, 50, 53):
        raise RuntimeError(
            f"±12 h计数不符合锁定值：总数={len(boundary)}, -12={neg_n}, +12={pos_n}"
        )

    summary_parts: list[pd.DataFrame] = []
    for level_type, level_col in [
        ("direction", "JetLag_signed"),
        ("nation", "Nation"),
        ("event_edition", "EventEdition"),
    ]:
        grouped = (
            boundary.groupby(["JetLagGroup_dir", level_col], observed=True)
            .agg(
                n=("Sinclair_Total", "size"),
                unique_athletes=("Athlete_ID", "nunique"),
                sinclair_mean=("Sinclair_Total", "mean"),
                sinclair_median=("Sinclair_Total", "median"),
                sinclair_sd=("Sinclair_Total", "std"),
                sinclair_q1=("Sinclair_Total", lambda x: x.quantile(0.25)),
                sinclair_q3=("Sinclair_Total", lambda x: x.quantile(0.75)),
            )
            .reset_index()
            .rename(columns={level_col: "level"})
        )
        grouped.insert(0, "level_type", level_type)
        grouped["extreme_group_n"] = pd.to_numeric(
            grouped["JetLagGroup_dir"].astype(str).map(
                df["JetLagGroup_dir"].astype(str).value_counts()
            ),
            errors="coerce",
        )
        grouped["share_of_extreme_group"] = grouped["n"] / grouped["extreme_group_n"]
        summary_parts.append(grouped)
    boundary_summary = pd.concat(summary_parts, ignore_index=True)
    boundary_summary.to_csv(
        OUTPUTS / "boundary_records_summary.csv", index=False, encoding="utf-8-sig"
    )
    boundary.to_csv(OUTPUTS / "boundary_records.csv", index=False, encoding="utf-8-sig")

    no12 = df.loc[~np.isclose(df["JetLag_abs"], 12.0)].copy()
    no12_fit = fit_mixed_model(no12, "exclude_exact_12h")
    if not no12_fit.success:
        raise RuntimeError(f"排除±12 h模型失败：{no12_fit.error}")
    no12_effects = extract_jetlag_effects(no12_fit)
    all_effects = pd.concat([full_effects, no12_effects], ignore_index=True)
    compare = target_effects(all_effects)
    group_sizes = (
        pd.concat(
            [
                df.assign(model="full_N2186"),
                no12.assign(model="exclude_exact_12h"),
            ],
            ignore_index=True,
        )
        .groupby(["model", "JetLagGroup_dir"], observed=True)
        .size()
        .rename("group_n")
        .reset_index()
    )
    compare = compare.merge(
        group_sizes,
        left_on=["model", "group"],
        right_on=["model", "JetLagGroup_dir"],
        how="left",
    ).drop(columns="JetLagGroup_dir")
    full_west = compare.loc[
        (compare["model"] == "full_N2186") & (compare["group"] == "West_10-11h")
    ].iloc[0]
    no12_west = compare.loc[
        (compare["model"] == "exclude_exact_12h") & (compare["group"] == "West_10-11h")
    ].iloc[0]
    same_direction = bool(np.sign(full_west["estimate"]) == np.sign(no12_west["estimate"]))
    same_fdr_status = bool(full_west["fdr_significant"] == no12_west["fdr_significant"])
    material = not (same_direction and same_fdr_status)
    compare["boundary_material_for_west"] = material
    compare["west_same_direction"] = same_direction
    compare["west_same_fdr_status"] = same_fdr_status
    compare.to_csv(
        OUTPUTS / "boundary_model_comparison.csv", index=False, encoding="utf-8-sig"
    )

    plot_data = compare.copy()
    plot_data["display_group"] = plot_data["group"].map(
        {
            "West_10-11h": "West 10–11 h",
            "East_8-9h": "East 8–9 h",
            "East_10-11h": "East 10–11 h",
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    palette = {"full_N2186": "#315F8A", "exclude_exact_12h": "#C15F3C"}
    offsets = {"full_N2186": -0.12, "exclude_exact_12h": 0.12}
    groups = ["West 10–11 h", "East 8–9 h", "East 10–11 h"]
    x = np.arange(len(groups))
    for model, sub in plot_data.groupby("model", sort=False):
        sub = sub.set_index("display_group").loc[groups]
        pos = x + offsets[model]
        ax.errorbar(
            pos,
            sub["estimate"],
            yerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
            fmt="o",
            capsize=3,
            color=palette[model],
            label="Full sample" if model == "full_N2186" else "Exclude exact ±12 h",
        )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Adjusted difference in Sinclair Total vs Minimal")
    ax.set_title("Boundary sensitivity for focal directional jet-lag groups")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, "boundary_model_comparison")

    decision = (
        "±12 h排除后，West 10–11 h系数方向和BH-FDR结论均未改变；按预设规则，"
        "边界记录不改变统计结论，后续解释不再展开该问题。"
        if not material
        else "±12 h排除后，West 10–11 h系数方向或BH-FDR结论发生改变；按预设规则，"
        "边界记录影响实质结论，后续分析必须平行报告完整样本与排除边界样本。"
    )
    note = f"""# ±12 h边界核验

## 数据核验

- 完整建模样本：{len(df)}条记录，{df['Athlete_ID'].nunique()}名运动员。
- West 10–11 h：{int((df['JetLagGroup_dir'] == 'West_10-11h').sum())}条。
- East 10–11 h：{int((df['JetLagGroup_dir'] == 'East_10-11h').sum())}条。
- 精确12 h边界：{len(boundary)}条，其中−12 h为{neg_n}条，+12 h为{pos_n}条。
- −12 h记录占West极端组{neg_n / LOCKED_WEST_N:.1%}；+12 h记录占East极端组{pos_n / LOCKED_EAST_EXTREME_N:.1%}。

## 模型结果

| 数据版本 | West β | 95% CI | FDR P | FDR显著 |
|---|---:|---:|---:|---|
| 完整样本 | {full_west['estimate']:.3f} | [{full_west['ci_low']:.3f}, {full_west['ci_high']:.3f}] | {full_west['p_fdr']:.4f} | {bool(full_west['fdr_significant'])} |
| 排除精确±12 h | {no12_west['estimate']:.3f} | [{no12_west['ci_low']:.3f}, {no12_west['ci_high']:.3f}] | {no12_west['p_fdr']:.4f} | {bool(no12_west['fdr_significant'])} |

## 预设判定

{decision}

该判断只用于决定是否在后续解释链中继续讨论边界问题。完整效应值、置信区间和样本构成仍保存在CSV中。
"""
    (NOTES / "boundary_check.md").write_text(note, encoding="utf-8")
    log(decision)
    return {
        "boundary_material": material,
        "same_direction": same_direction,
        "same_fdr_status": same_fdr_status,
        "full_west_beta": float(full_west["estimate"]),
        "no12_west_beta": float(no12_west["estimate"]),
        "full_west_fdr": float(full_west["p_fdr"]),
        "no12_west_fdr": float(no12_west["p_fdr"]),
    }


def composition_detail(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    groups = ["All", *TARGET_GROUPS]
    overall_n = len(df)
    levels = sorted(df[dimension].astype(str).unique())
    rows: list[dict[str, object]] = []
    overall_level_counts = df[dimension].astype(str).value_counts()
    for group in groups:
        sub = df if group == "All" else df.loc[df["JetLagGroup_dir"] == group]
        counts = sub[dimension].astype(str).value_counts()
        for level in levels:
            vals = sub.loc[sub[dimension].astype(str) == level, "Sinclair_Total"]
            n = int(len(vals))
            group_share = n / len(sub) if len(sub) else np.nan
            overall_share = int(overall_level_counts.get(level, 0)) / overall_n
            enrichment = group_share / overall_share if overall_share > 0 else np.nan
            rows.append(
                {
                    "dimension": dimension,
                    "analysis_group": group,
                    "level": level,
                    "group_total_n": len(sub),
                    "n": n,
                    "group_share": group_share,
                    "unique_athletes": int(
                        sub.loc[sub[dimension].astype(str) == level, "Athlete_ID"].nunique()
                    ),
                    "sinclair_mean": vals.mean(),
                    "sinclair_median": vals.median(),
                    "sinclair_sd": vals.std(),
                    "sinclair_q1": vals.quantile(0.25),
                    "sinclair_q3": vals.quantile(0.75),
                    "overall_share": overall_share,
                    "enrichment_ratio": enrichment,
                }
            )
    return pd.DataFrame(rows)


def concentration_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dimension, group), sub in detail.groupby(["dimension", "analysis_group"]):
        positive = sub.loc[sub["n"] > 0].sort_values("group_share", ascending=False)
        shares = positive["group_share"].to_numpy(dtype=float)
        hhi = float(np.sum(shares**2))
        rows.append(
            {
                "dimension": dimension,
                "analysis_group": group,
                "nonzero_levels": int(len(positive)),
                "top1_level": positive.iloc[0]["level"] if len(positive) else "",
                "top1_share": float(shares[0]) if len(shares) else np.nan,
                "top3_share": float(shares[:3].sum()) if len(shares) else np.nan,
                "hhi": hhi,
                "effective_sources": 1 / hhi if hhi > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_composition_heatmap(detail: pd.DataFrame, dimension: str, stem: str) -> None:
    use = detail.loc[
        (detail["dimension"] == dimension) & (detail["analysis_group"] != "All")
    ].copy()
    pivot = use.pivot(index="level", columns="analysis_group", values="group_share").fillna(0)
    order = TARGET_GROUPS
    pivot = pivot[order]
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    height = max(5.0, 0.32 * len(pivot) + 1.6)
    fig, ax = plt.subplots(figsize=(8.0, height))
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.4,
        cbar_kws={"label": "Share within focal group (%)"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Nation" if dimension == "Nation" else "Event edition")
    ax.set_title(
        ("National composition" if dimension == "Nation" else "Event-edition composition")
        + " of focal jet-lag groups"
    )
    fig.tight_layout()
    save_figure(fig, stem)


def stage_composition(df: pd.DataFrame) -> dict[str, object]:
    log("阶段2：国家与赛事届次构成和集中度")
    save_metadata("composition", df, "完整样本；目标组West 10–11 h、East 8–9 h、East 10–11 h")
    nation = composition_detail(df, "Nation")
    event = composition_detail(df, "EventEdition")
    nation.to_csv(OUTPUTS / "composition_nation.csv", index=False, encoding="utf-8-sig")
    event.to_csv(OUTPUTS / "composition_event_edition.csv", index=False, encoding="utf-8-sig")
    concentration = concentration_summary(pd.concat([nation, event], ignore_index=True))
    concentration.to_csv(
        OUTPUTS / "concentration_summary.csv", index=False, encoding="utf-8-sig"
    )
    plot_composition_heatmap(nation, "Nation", "composition_nation_heatmap")
    plot_composition_heatmap(event, "EventEdition", "composition_event_heatmap")

    def top_lines(detail: pd.DataFrame, group: str) -> str:
        top = detail.loc[
            (detail["analysis_group"] == group) & (detail["n"] > 0)
        ].sort_values("group_share", ascending=False).head(5)
        return "\n".join(
            f"- {row.level}: n={int(row.n)}, 占组内{row.group_share:.1%}，"
            f"相对全样本富集{row.enrichment_ratio:.2f}倍。"
            for row in top.itertuples()
        )

    west_nation_conc = concentration.loc[
        (concentration["dimension"] == "Nation")
        & (concentration["analysis_group"] == "West_10-11h")
    ].iloc[0]
    west_event_conc = concentration.loc[
        (concentration["dimension"] == "EventEdition")
        & (concentration["analysis_group"] == "West_10-11h")
    ].iloc[0]
    note = f"""# 国家与赛事届次构成

## West 10–11 h的主要国家

{top_lines(nation, 'West_10-11h')}

国家集中度：Top 1占{west_nation_conc['top1_share']:.1%}，Top 3占{west_nation_conc['top3_share']:.1%}，HHI={west_nation_conc['hhi']:.3f}，有效来源数={west_nation_conc['effective_sources']:.2f}。

## West 10–11 h的主要赛事届次

{top_lines(event, 'West_10-11h')}

赛事集中度：Top 1占{west_event_conc['top1_share']:.1%}，Top 3占{west_event_conc['top3_share']:.1%}，HHI={west_event_conc['hhi']:.3f}，有效来源数={west_event_conc['effective_sources']:.2f}。

## 对照原则

相同指标已对East 8–9 h和East 10–11 h计算。构成比例和富集倍数用于识别候选来源，不能单独证明某个国家或赛事造成模型系数；驱动判断以逐类删除和扩展模型为准。
"""
    (NOTES / "composition_findings.md").write_text(note, encoding="utf-8")
    return {
        "west_nation_top1": str(west_nation_conc["top1_level"]),
        "west_nation_top1_share": float(west_nation_conc["top1_share"]),
        "west_event_top1": str(west_event_conc["top1_level"]),
        "west_event_top1_share": float(west_event_conc["top1_share"]),
    }


def influence_rows(
    df: pd.DataFrame,
    dimension: str,
    baseline_effects: pd.DataFrame,
) -> pd.DataFrame:
    baseline = target_effects(baseline_effects).set_index("group")
    levels = sorted(df[dimension].astype(str).unique())
    all_rows: list[dict[str, object]] = []
    for index, level in enumerate(levels, start=1):
        mask = df[dimension].astype(str) != level
        sub = df.loc[mask].copy()
        deleted_n = int((~mask).sum())
        label = f"drop_{dimension}_{level}"
        log(f"影响诊断 {dimension} {index}/{len(levels)}：删除 {level}（n={deleted_n}）")
        fit = fit_mixed_model(sub, label)
        if fit.success:
            try:
                effects = target_effects(extract_jetlag_effects(fit)).set_index("group")
            except Exception as exc:
                fit = ModelFit(
                    fit.label,
                    fit.formula,
                    fit.n,
                    fit.athletes,
                    False,
                    False,
                    None,
                    str(exc),
                    fit.warnings_text,
                    fit.design_rank,
                    fit.design_columns,
                )
                effects = pd.DataFrame()
        else:
            effects = pd.DataFrame()
        for group in TARGET_GROUPS:
            base = baseline.loc[group]
            if fit.success and group in effects.index:
                row = effects.loc[group]
                delta = float(row["estimate"] - base["estimate"])
                pct = delta / abs(float(base["estimate"])) if base["estimate"] != 0 else np.nan
                sign_changed = bool(np.sign(row["estimate"]) != np.sign(base["estimate"]))
                fdr_changed = bool(row["fdr_significant"] != base["fdr_significant"])
                influential = bool(
                    group == "West_10-11h"
                    and (sign_changed or fdr_changed or abs(pct) >= 0.20)
                )
                all_rows.append(
                    {
                        "dimension": dimension,
                        "deleted_level": level,
                        "deleted_n": deleted_n,
                        "n_after": len(sub),
                        "expected_n_after": len(df) - deleted_n,
                        "model_success": True,
                        "converged": fit.converged,
                        "target_group": group,
                        "estimate": row["estimate"],
                        "ci_low": row["ci_low"],
                        "ci_high": row["ci_high"],
                        "p_raw": row["p_raw"],
                        "p_fdr": row["p_fdr"],
                        "fdr_significant": row["fdr_significant"],
                        "baseline_estimate": base["estimate"],
                        "delta_beta": delta,
                        "relative_change": pct,
                        "sign_changed": sign_changed,
                        "fdr_status_changed": fdr_changed,
                        "west_influential": influential,
                        "error": "",
                        "warnings": fit.warnings_text,
                    }
                )
            else:
                all_rows.append(
                    {
                        "dimension": dimension,
                        "deleted_level": level,
                        "deleted_n": deleted_n,
                        "n_after": len(sub),
                        "expected_n_after": len(df) - deleted_n,
                        "model_success": False,
                        "converged": False,
                        "target_group": group,
                        "estimate": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_raw": np.nan,
                        "p_fdr": np.nan,
                        "fdr_significant": False,
                        "baseline_estimate": base["estimate"],
                        "delta_beta": np.nan,
                        "relative_change": np.nan,
                        "sign_changed": False,
                        "fdr_status_changed": False,
                        "west_influential": False,
                        "error": fit.error,
                        "warnings": fit.warnings_text,
                    }
                )
    return pd.DataFrame(all_rows)


def plot_influence(results: pd.DataFrame, dimension: str, stem: str) -> None:
    west = results.loc[
        (results["target_group"] == "West_10-11h") & results["model_success"]
    ].copy()
    west = west.sort_values("delta_beta")
    height = max(5.5, 0.31 * len(west) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, height))
    y = np.arange(len(west))
    colors = np.where(west["west_influential"], "#B54A3A", "#4B6E8C")
    for idx, (_, row) in enumerate(west.iterrows()):
        ax.errorbar(
            row["estimate"],
            idx,
            xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o",
            color=colors[idx],
            capsize=2.5,
        )
    baseline = float(west["baseline_estimate"].iloc[0]) if len(west) else np.nan
    ax.axvline(baseline, color="0.25", linestyle="--", linewidth=1, label="Full-sample β")
    ax.axvline(0, color="0.65", linestyle=":", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(west["deleted_level"], fontsize=8)
    ax.set_xlabel("West 10–11 h coefficient after deletion")
    ax.set_ylabel("Deleted " + ("nation" if dimension == "Nation" else "event edition"))
    ax.set_title("Leave-one-out influence on the West 10–11 h estimate")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, stem)


def stage_influence(df: pd.DataFrame) -> dict[str, object]:
    log("阶段3：逐赛事、逐国家删除影响诊断")
    save_metadata("influence", df, "完整样本；leave-one-nation-out和leave-one-event-edition-out")
    baseline_fit = fit_mixed_model(df, "full_N2186")
    if not baseline_fit.success:
        raise RuntimeError(f"影响分析的基线模型失败：{baseline_fit.error}")
    baseline_effects = extract_jetlag_effects(baseline_fit)
    nation_results = influence_rows(df, "Nation", baseline_effects)
    event_results = influence_rows(df, "EventEdition", baseline_effects)
    if not (nation_results["n_after"] == nation_results["expected_n_after"]).all():
        raise RuntimeError("逐国家删除样本量校验失败")
    if not (event_results["n_after"] == event_results["expected_n_after"]).all():
        raise RuntimeError("逐赛事删除样本量校验失败")
    nation_results.to_csv(
        OUTPUTS / "leave_one_nation_out.csv", index=False, encoding="utf-8-sig"
    )
    event_results.to_csv(
        OUTPUTS / "leave_one_event_out.csv", index=False, encoding="utf-8-sig"
    )
    plot_influence(nation_results, "Nation", "leave_one_nation_out_west")
    plot_influence(event_results, "EventEdition", "leave_one_event_out_west")

    nation_west = nation_results.loc[nation_results["target_group"] == "West_10-11h"].copy()
    event_west = event_results.loc[event_results["target_group"] == "West_10-11h"].copy()
    nation_candidates = nation_west.loc[nation_west["west_influential"]].sort_values(
        "delta_beta", key=lambda s: s.abs(), ascending=False
    )
    event_candidates = event_west.loc[event_west["west_influential"]].sort_values(
        "delta_beta", key=lambda s: s.abs(), ascending=False
    )

    def candidate_lines(data: pd.DataFrame) -> str:
        if data.empty:
            return "- 未识别出符合预设标准的影响来源。"
        return "\n".join(
            f"- {row.deleted_level}: 删除n={int(row.deleted_n)}后β={row.estimate:.3f}，"
            f"Δβ={row.delta_beta:+.3f}（{row.relative_change:+.1%}），"
            f"FDR P={row.p_fdr:.4f}。"
            for row in data.itertuples()
        )

    note = f"""# 逐类删除影响诊断

## 国家候选驱动来源

{candidate_lines(nation_candidates)}

## 赛事届次候选驱动来源

{candidate_lines(event_candidates)}

候选来源按预设标准识别：删除后West系数变号、BH-FDR结论改变，或|Δβ|达到完整模型β绝对值的20%。单纯因样本量下降而失去显著性、但β近似不变的结果不单独解释为驱动作用。
"""
    (NOTES / "influence_findings.md").write_text(note, encoding="utf-8")
    return {
        "nation_candidates": nation_candidates["deleted_level"].tolist(),
        "event_candidates": event_candidates["deleted_level"].tolist(),
    }


def model_fit_row(fit: ModelFit, ml_fit: ModelFit | None, baseline_ml: ModelFit | None) -> dict[str, object]:
    row: dict[str, object] = {
        "model": fit.label,
        "formula": fit.formula,
        "n": fit.n,
        "athletes": fit.athletes,
        "success": fit.success,
        "converged_reml": fit.converged,
        "design_rank": fit.design_rank,
        "design_columns": fit.design_columns,
        "error": fit.error,
        "warnings_reml": fit.warnings_text,
    }
    if ml_fit and ml_fit.success and ml_fit.result is not None:
        row.update(
            {
                "converged_ml": ml_fit.converged,
                "loglik_ml": float(ml_fit.result.llf),
                "aic_ml": float(ml_fit.result.aic),
                "bic_ml": float(ml_fit.result.bic),
                "k_fe_ml": int(ml_fit.result.k_fe),
            }
        )
        if baseline_ml and baseline_ml.success and baseline_ml.result is not None:
            lr = 2 * (float(ml_fit.result.llf) - float(baseline_ml.result.llf))
            df_diff = int(ml_fit.result.k_fe - baseline_ml.result.k_fe)
            row["lr_vs_base"] = lr
            row["lr_df_vs_base"] = df_diff
            row["lr_p_vs_base"] = float(chi2.sf(max(lr, 0), df_diff)) if df_diff > 0 else np.nan
    return row


def stage_adjustment(df: pd.DataFrame) -> dict[str, object]:
    log("阶段4：国家、赛事届次固定效应调整模型")
    save_metadata("adjustment", df, "完整样本；依次加入Nation和EventEdition固定效应")
    specs = [
        ("base", ()),
        ("plus_nation", ("C(Nation)",)),
        ("plus_event_edition", ("C(EventEdition)",)),
        ("plus_nation_event", ("C(Nation)", "C(EventEdition)")),
    ]
    reml_fits: dict[str, ModelFit] = {}
    ml_fits: dict[str, ModelFit] = {}
    for label, terms in specs:
        log(f"固定效应模型：{label}")
        reml_fits[label] = fit_mixed_model(df, label, terms, reml=True)
        ml_fits[label] = fit_mixed_model(df, label + "_ML", terms, reml=False)

    base_effects = extract_jetlag_effects(reml_fits["base"])
    base_targets = target_effects(base_effects).set_index("group")
    effects_all: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for label, _ in specs:
        fit = reml_fits[label]
        ml_fit = ml_fits[label]
        fit_rows.append(model_fit_row(fit, ml_fit, ml_fits["base"]))
        if not fit.success:
            continue
        effects = target_effects(extract_jetlag_effects(fit))
        effects["baseline_estimate"] = effects["group"].map(base_targets["estimate"])
        effects["delta_beta"] = effects["estimate"] - effects["baseline_estimate"]
        effects["relative_change"] = effects["delta_beta"] / effects["baseline_estimate"].abs()
        effects["sign_changed"] = np.sign(effects["estimate"]) != np.sign(effects["baseline_estimate"])
        effects["fdr_status_changed"] = effects["fdr_significant"] != effects["group"].map(
            base_targets["fdr_significant"]
        )
        effects_all.append(effects)
    model_fits = pd.DataFrame(fit_rows)
    model_effects = pd.concat(effects_all, ignore_index=True) if effects_all else pd.DataFrame()
    model_fits.to_csv(OUTPUTS / "adjustment_model_fits.csv", index=False, encoding="utf-8-sig")
    model_effects.to_csv(
        OUTPUTS / "adjustment_target_effects.csv", index=False, encoding="utf-8-sig"
    )

    if not model_effects.empty:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        labels = [label for label, _ in specs if label in set(model_effects["model"])]
        offsets = np.linspace(-0.24, 0.24, len(labels))
        x = np.arange(len(TARGET_GROUPS))
        palette = sns.color_palette("colorblind", len(labels))
        for label, offset, color in zip(labels, offsets, palette):
            sub = model_effects.loc[model_effects["model"] == label].set_index("group").loc[TARGET_GROUPS]
            ax.errorbar(
                x + offset,
                sub["estimate"],
                yerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
                fmt="o",
                capsize=3,
                label=label,
                color=color,
            )
        ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_", " ") for g in TARGET_GROUPS])
        ax.set_ylabel("Adjusted difference in Sinclair Total vs Minimal")
        ax.set_title("Focal estimates after nation and event-edition adjustment")
        ax.legend(frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_figure(fig, "adjustment_model_comparison")

    failures = model_fits.loc[~model_fits["success"]]
    west = model_effects.loc[model_effects["group"] == "West_10-11h"].copy()
    lines = []
    for row in west.itertuples():
        lines.append(
            f"- {row.model}: β={row.estimate:.3f}，95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}]，"
            f"FDR P={row.p_fdr:.4f}，相对基线变化{row.relative_change:+.1%}。"
        )
    if failures.empty:
        failure_text = "所有预设模型均具有满秩设计矩阵并完成拟合。"
    else:
        failure_text = "\n".join(
            f"- {row.model}: {row.error}" for row in failures.itertuples()
        )
    note = f"""# 国家与赛事届次调整模型

## West 10–11 h估计

{chr(10).join(lines) if lines else '- 无可报告结果。'}

## 可估计性

{failure_text}

固定效应模型用于判断West估计是否随国家或赛事构成调整而衰减。即使调整后关联仍存在，也不能解释为向西旅行提高成绩。
"""
    (NOTES / "adjustment_findings.md").write_text(note, encoding="utf-8")
    return {
        "failed_models": failures["model"].tolist(),
        "west_models": west[
            ["model", "estimate", "ci_low", "ci_high", "p_fdr", "relative_change"]
        ].to_dict("records"),
    }


def synthesize(df: pd.DataFrame, boundary: dict, composition: dict, influence: dict, adjustment: dict) -> None:
    nation_results = pd.read_csv(OUTPUTS / "leave_one_nation_out.csv", encoding="utf-8-sig")
    event_results = pd.read_csv(OUTPUTS / "leave_one_event_out.csv", encoding="utf-8-sig")
    adjustment_effects = pd.read_csv(
        OUTPUTS / "adjustment_target_effects.csv", encoding="utf-8-sig"
    )
    west_nation = nation_results.loc[nation_results["target_group"] == "West_10-11h"]
    west_event = event_results.loc[event_results["target_group"] == "West_10-11h"]
    influential_nations = west_nation.loc[west_nation["west_influential"], "deleted_level"].tolist()
    influential_events = west_event.loc[west_event["west_influential"], "deleted_level"].tolist()
    west_adjustment = adjustment_effects.loc[
        adjustment_effects["group"] == "West_10-11h"
    ].copy()

    if boundary["boundary_material"]:
        classification = "边界敏感"
        rationale = "排除精确±12 h后，West系数方向或BH-FDR结论发生改变。"
    elif influential_events and not influential_nations:
        classification = "赛事届次驱动或明显受赛事背景影响"
        rationale = "至少一届赛事满足预设影响标准，而未识别出满足标准的单一国家。"
    elif influential_nations and not influential_events:
        classification = "国家构成驱动或明显受国家构成影响"
        rationale = "至少一个国家满足预设影响标准，而未识别出满足标准的单一赛事。"
    elif influential_nations and influential_events:
        classification = "赛事与国家共同影响"
        rationale = "国家和赛事删除分析均识别出满足预设标准的影响来源。"
    else:
        successful_adjustment = west_adjustment.loc[west_adjustment["model"] != "base"]
        if not successful_adjustment.empty:
            stable = (
                (np.sign(successful_adjustment["estimate"]) == np.sign(boundary["full_west_beta"]))
                & (successful_adjustment["relative_change"].abs() < 0.20)
            ).all()
        else:
            stable = False
        if stable:
            classification = "多来源但在预设分析中仍较稳定"
            rationale = "未识别出单一国家或赛事驱动，国家和赛事调整后的West估计方向一致且变化均小于20%。"
        else:
            classification = "证据不足以区分来源"
            rationale = "单一删除未识别明确驱动来源，扩展模型又未提供一致、稳定的解释。"

    history_trigger = bool(influential_nations or influential_events)
    payload = {
        "classification": classification,
        "rationale": rationale,
        "history_search_triggered": history_trigger,
        "candidate_nations": influential_nations,
        "candidate_events": influential_events,
        "boundary": boundary,
        "composition": composition,
        "adjustment": adjustment,
    }
    (OUTPUTS / "synthesis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history_text = (
        "已触发外部背景调查：仅调查上述候选国家和赛事，并要求形成事件事实、样本构成与统计结果之间的证据链。"
        if history_trigger
        else "未触发外部背景调查：没有国家或赛事满足预设影响标准，不加入历史事件叙事。"
    )
    result_text_en = (
        "In post hoc explanatory analyses, the positive estimate for westward travel across the highest time-zone category was examined by national-team and championship-edition composition. "
        "The analyses were designed to identify sample concentration and influential clusters rather than to estimate a causal benefit of westward travel."
    )
    discussion_text_en = (
        "The higher Sinclair Total observed in the most extreme westward category should not be interpreted as evidence that westward travel improves performance. "
        f"The explanatory analyses classified the finding as: {classification}. "
        "Because sleep, circadian phase, acclimatization, training load, and travel schedules were not measured, the underlying mechanism remains unresolved."
    )
    limitations_text_en = (
        "The country and championship-edition analyses were post hoc and were constrained by the dependence between travel direction, team origin, and host location. "
        "They can identify concentration and influential clusters but cannot establish why performance differed across directional jet-lag categories."
    )
    note = f"""# 向西极端时差结果的综合判断

## 结论分类

**{classification}**：{rationale}

## 外部历史背景调查

{history_text}

- 候选国家：{', '.join(influential_nations) if influential_nations else '无'}
- 候选赛事：{'; '.join(influential_events) if influential_events else '无'}

## Results候选英文表述

{result_text_en}

## Discussion候选英文表述

{discussion_text_en}

## Limitations候选英文表述

{limitations_text_en}

## 中文表述边界

本轮分析用于识别样本集中和影响来源，不能证明向西旅行提高成绩。最终稿应报告已验证的国家或赛事构成结果，并把未测量的睡眠、昼夜节律相位、适应过程、训练负荷和具体行程保留为局限性，而不是把它们写成已经证实的机制。
"""
    (NOTES / "final_synthesis.md").write_text(note, encoding="utf-8")
    log(f"综合分类：{classification}。{rationale}")


def stage_all(df: pd.DataFrame) -> None:
    boundary = stage_boundary(df)
    composition = stage_composition(df)
    influence = stage_influence(df)
    adjustment = stage_adjustment(df)
    synthesize(df, boundary, composition, influence, adjustment)


def main() -> None:
    parser = argparse.ArgumentParser(description="West 10–11 h explanatory exploration")
    parser.add_argument(
        "--stage",
        choices=["boundary", "composition", "influence", "adjustment", "all"],
        default="all",
    )
    args = parser.parse_args()
    ensure_dirs()
    if LOG_PATH.exists() and args.stage == "all":
        LOG_PATH.unlink()
    df = load_data()
    log(f"读取正式数据：{INPUT_CSV}；N={len(df)}")
    if args.stage == "boundary":
        stage_boundary(df)
    elif args.stage == "composition":
        stage_composition(df)
    elif args.stage == "influence":
        stage_influence(df)
    elif args.stage == "adjustment":
        stage_adjustment(df)
    else:
        stage_all(df)
    log(f"阶段{args.stage}完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ensure_dirs()
        log(f"FATAL: {type(exc).__name__}: {exc}")
        raise
