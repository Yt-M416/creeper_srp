"""
Fig 1 森林图、Fig 2 异质性 2×2、Fig 3 敏感性、补充参赛频率图。
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from io_utils import (
    find_latest_subdir,
    parse_heterogeneity_summary_md,
    parse_lrt_block_model_b_jetlag,
    parse_sensitivity_model_comparison,
    parse_travel_control_lrts,
)
from plot_config import (
    COLORS,
    FIG_JETLAG_DIST,
    FIG_PANEL_2X2,
    FIG_SENSITIVITY,
    FIG_SINGLE_COL,
    PROJECT_ROOT,
    save_fig,
)

_ALPHA_LINE = 0.05
_X_THR = -math.log10(_ALPHA_LINE)


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


def _sig_fdr_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_fig1_forest_main_effect(emmeans_csv: Path, out_dir: Path) -> None:
    emm = pd.read_csv(emmeans_csv, encoding="utf-8-sig")
    emm = emm.set_index("Group")

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL[0] + 2.2, 5.2))

    y_pos = np.arange(len(JETLAG_PLOT_ORDER))
    for i, g in enumerate(JETLAG_PLOT_ORDER):
        if g == "Minimal(0-1h)":
            ax.plot(
                [0.0],
                [i],
                marker="D",
                markersize=5,
                color=COLORS["neutral"],
            )
            ax.text(0.15, i, "Ref", fontsize=6, va="center", color=COLORS["neutral"])
            continue
        if g not in emm.index:
            continue
        row = emm.loc[g]
        beta = float(row["Estimate"])
        lo = float(row["CI_lo"])
        hi = float(row["CI_hi"])
        p_fdr = float(row["p_fdr"])
        sig = p_fdr < 0.05
        color = COLORS["primary"] if sig else COLORS["neutral"]
        marker = "o"
        face = color if sig else "none"
        ax.errorbar(
            beta,
            i,
            xerr=[[beta - lo], [hi - beta]],
            fmt="none",
            ecolor=color,
            capsize=2,
            linewidth=1.0,
        )
        ax.plot(
            beta,
            i,
            marker=marker,
            markersize=5,
            markerfacecolor=face,
            markeredgecolor=color,
            linestyle="none",
        )
        st = _sig_fdr_stars(p_fdr)
        if st:
            span = max(abs(hi - lo), 2.0)
            ax.text(beta + 0.08 * span, i, st, va="center", fontsize=8, color=color)

    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([g.replace("_", " ") for g in JETLAG_PLOT_ORDER], fontsize=7)
    ax.set_xlabel(r"$\beta$ (Sinclair points vs Minimal)")
    ax.set_title("Directional jet lag: main effects\n(TravelDuration controlled; BH-FDR)")
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, out_dir, "fig1_forest_main_effect")
    plt.close(fig)


def _load_design_a_contrasts(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv, encoding="utf-8-sig")
    return df[df["design"] == "A"].copy()


def _pretty_group_label(analysis: str, level: str) -> str:
    if analysis == "sex":
        return level
    if analysis == "age":
        return "<25 yr" if level == "Young" else "≥25 yr"
    if analysis == "experience":
        return "≤2 starts" if level == "Low" else "≥3 starts"
    return str(level)


def _lrt_text_for(analysis: str, summ_df: pd.DataFrame) -> str:
    row = summ_df[summ_df["Analysis"].str.strip() == analysis]
    if row.empty:
        return ""
    r = row.iloc[0]
    try:
        p = float(r["p"])
        ptxt = f"{p:.3g}" if p >= 1e-4 else f"{p:.2e}"
        return f"LR={float(r['LR']):.2f}, df={int(float(r['df']))}, p={ptxt}"
    except Exception:
        return ""


def plot_fig2_heterogeneity_2x2(hetero_root: Path, summary_md: Path, out_dir: Path) -> None:
    summ = parse_heterogeneity_summary_md(summary_md)
    summ.columns = [c.strip() for c in summ.columns]

    specs = [
        ("sex", "Sex"),
        ("age", "Age"),
        ("experience", "Experience"),
        ("weight", "Weight class"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIG_PANEL_2X2, sharey=True)
    axes_flat = axes.ravel()

    all_beta = []
    for ax, (aname, title) in zip(axes_flat, specs):
        rdir = find_latest_subdir(hetero_root, f"{aname}_*")
        rc = rdir / "results_contrasts.csv"
        da = _load_design_a_contrasts(rc)
        x = np.arange(len(JETLAG_PLOT_ORDER))
        x_map = {g: j for j, g in enumerate(JETLAG_PLOT_ORDER)}
        da = da[da["jetlag_group"].isin(JETLAG_PLOT_ORDER)]
        da["x"] = da["jetlag_group"].map(x_map)

        levels = sorted(da["group_level"].dropna().unique().tolist())
        cmap = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#9467bd"]
        for li, lev in enumerate(levels):
            sub = da[da["group_level"] == lev].sort_values("x")
            color = cmap[li % len(cmap)]
            lbl = _pretty_group_label(aname, str(lev))
            ax.plot(sub["x"], sub["estimate"], marker="o", label=lbl, color=color, linewidth=1.0)
            ax.fill_between(
                sub["x"],
                sub["ci_lo"],
                sub["ci_hi"],
                alpha=0.18,
                color=color,
            )
            all_beta.extend(sub["estimate"].tolist())
            all_beta.extend(sub["ci_lo"].dropna().tolist())
            all_beta.extend(sub["ci_hi"].dropna().tolist())

        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_", "\n") for g in JETLAG_PLOT_ORDER], rotation=0, fontsize=5)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ann = _lrt_text_for(aname, summ)
        if ann:
            ax.text(0.98, 0.02, ann, transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

    if all_beta:
        lo = np.nanpercentile(all_beta, 5)
        hi = np.nanpercentile(all_beta, 95)
        pad = (hi - lo) * 0.15 + 1e-6
        for ax in axes_flat:
            ax.set_ylim(lo - pad, hi + pad)

    for ax in axes_flat:
        ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.5)
    fig.supylabel(r"$\beta$ vs Minimal (Sinclair pts)", fontsize=9)
    fig.suptitle("Subgroup-stratified effects (Design A)", fontsize=10, y=1.02)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig2_heterogeneity_2x2")
    plt.close(fig)


def _neglogp_max(label_p: list[tuple[str, float]]) -> float:
    m = 0.0
    for _, p in label_p:
        if p is not None and not (isinstance(p, float) and math.isnan(p)):
            m = max(m, -math.log10(max(float(p), 1e-300)))
    return m


def _lrt_bars_hpanel(
    ax,
    label_p: list[tuple[str, float]],
    *,
    title: str,
    color: str,
    show_legend: bool = False,
) -> None:
    """单 panel 横向条形：$-\\log_{10}(p)$；y 轴自上而下与列表同序。"""
    if not label_p:
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.axis("off")
        return
    labels = [t[0] for t in label_p]
    ps = [float(t[1]) for t in label_p]
    y = np.arange(len(labels), dtype=float)
    ps_arr = np.array(ps, dtype=float)
    xvals = -np.log10(np.clip(ps_arr, 1e-300, None))
    ax.barh(y, xvals, color=color, height=0.55, alpha=0.9)
    ax.axvline(
        _X_THR,
        color=COLORS["negative"],
        linestyle="--",
        linewidth=0.9,
        label=rf"$\alpha$={_ALPHA_LINE} ($p$=0.05)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=6)
    for i, p in enumerate(ps):
        if math.isnan(p):
            continue
        ax.text(xvals[i] + 0.12, y[i], f"p={p:.2g}", va="center", fontsize=5.8)
    if show_legend and label_p:
        ax.legend(loc="lower right", fontsize=5.5)


def plot_fig3_sensitivity(out_dir: Path) -> None:
    """
    Fig3 拆成两栏，避免把「$C$(JetLagGroup_dir) 的联合/条件联合 LRT」与
    「连续协变量×有向时差」混在同一假说层。

    左：(A) Directional groups — 联合 LRT；右：(B) 连续量×JetLag_signed 交互 LRT。
    """
    joint: list[tuple[str, float]] = []
    inter: list[tuple[str, float]] = []

    tc = PROJECT_ROOT / "lmm_jetlag_travel_control" / "model_comparison.txt"
    if tc.is_file():
        lr = parse_travel_control_lrts(tc.read_text(encoding="utf-8"))
        pb = lr.get("lrt_b_p")
        if pb is not None:
            joint.append(
                (
                    "C(JetLag) joint LRT | adjusted for travel duration",
                    float(pb),
                )
            )

    hroot = PROJECT_ROOT / "heterogeneity_results"
    for var, short in [
        ("Age_c", "Age × signed jet lag"),
        ("BodyWeight_c", "Body mass × signed jet lag"),
        ("Cumulative_Exp_c", "Cumulative starts × signed jet lag"),
    ]:
        try:
            latest = find_latest_subdir(hroot, f"sensitivity_{var}_*")
            d = parse_sensitivity_model_comparison(latest / "model_comparison.txt")
            if "primary_p" in d:
                inter.append((short, float(d["primary_p"])))
        except FileNotFoundError:
            inter.append((short, float("nan")))

    if not joint and not inter:
        return

    xmax = max(
        _neglogp_max(joint) if joint else 0.0,
        _neglogp_max(inter) if inter else 0.0,
        _X_THR * 1.1,
    )
    xmax = xmax * 1.15

    w, h = FIG_SENSITIVITY
    if joint and inter:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h), layout="constrained", gridspec_kw={"wspace": 0.4})
        _lrt_bars_hpanel(ax1, joint, title="(A) Directional groups (joint LRTs)", color=COLORS["primary"], show_legend=False)
        _lrt_bars_hpanel(
            ax2, inter, title="(B) Continuous covariate × signed jet lag", color=COLORS["secondary"], show_legend=True
        )
        ax1.set_xlim(0, xmax)
        ax2.set_xlim(0, xmax)
        fig.suptitle("Likelihood-based tests: two hypothesis families", fontsize=9.5)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(w * 0.55, h), layout="constrained")
        if joint:
            _lrt_bars_hpanel(
                ax1, joint, title="(A) Directional groups (joint LRTs)", color=COLORS["primary"], show_legend=True
            )
        else:
            _lrt_bars_hpanel(
                ax1,
                inter,
                title="(B) Continuous covariate × signed jet lag",
                color=COLORS["secondary"],
                show_legend=True,
            )
        ax1.set_xlim(0, xmax)
        fig.suptitle("Likelihood-based tests", fontsize=9.5)

    save_fig(fig, out_dir, "fig3_sensitivity")
    plt.close(fig)


def plot_supp_nation_heatmap(data_used_csv: Path, out_dir: Path) -> None:
    """
    Supp Fig：Nation × JetLagGroup_dir 观测频数热力图。
    纵轴：国家（按 East 8-9h 组观测数降序，方便读者立即核查最显著组的来源）；
    横轴：JetLagGroup_dir 11 档（West→East 序，与 Fig1 一致）；
    单元格标注频数，0 值保留空白以减少视觉噪音。
    用途：Supplementary；主文一句话引用以预防审稿人"数据偏倚"质疑。
    """
    df = pd.read_csv(data_used_csv, encoding="utf-8-sig")
    if "Nation" not in df.columns or "JetLagGroup_dir" not in df.columns:
        return

    pivot = (
        df.groupby(["Nation", "JetLagGroup_dir"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=JETLAG_PLOT_ORDER, fill_value=0)
    )
    # 按 East 8-9h 列降序排列国家（凸显最显著时差组的来源）
    sort_col = "East_8-9h"
    if sort_col in pivot.columns:
        pivot = pivot.sort_values(sort_col, ascending=False)

    n_nat, n_grp = pivot.shape
    fig_w = max(7.0, n_grp * 0.62)
    fig_h = max(3.2, n_nat * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    # 用行最大值归一化（行内比色，让每国的暴露模式更易比较）
    norm_val = pivot.div(pivot.max(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(norm_val.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    # 单元格标注原始频数；0 值空白
    for i in range(n_nat):
        for j in range(n_grp):
            v = int(pivot.iloc[i, j])
            if v > 0:
                text_col = "white" if norm_val.iloc[i, j] > 0.65 else "0.2"
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6, color=text_col)

    ax.set_xticks(range(n_grp))
    ax.set_xticklabels([g.replace("_", "\n") for g in JETLAG_PLOT_ORDER], fontsize=6.5)
    ax.set_yticks(range(n_nat))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
    ax.set_xlabel("Directional jet-lag group (West → East)", fontsize=8)
    ax.set_ylabel("Nation", fontsize=8)
    ax.set_title(
        "Nation × jet-lag group: observation counts\n(shading: row-normalised; sorted by East 8–9h)",
        fontsize=9,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, aspect=18)
    cbar.set_label("Row-normalised count", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    save_fig(fig, out_dir, "supp_nation_jetlag_heatmap")
    plt.close(fig)


def _dir_group_bar_color(g: str) -> str:
    if g == "Minimal(0-1h)":
        return COLORS["neutral"]
    if g.startswith("West"):
        return "#5B9BD5"
    if g.startswith("East"):
        return "#ED7D31"
    return COLORS["neutral"]


def plot_supp_jetlag_distribution(data_used_csv: Path, out_dir: Path) -> None:
    """
    补充图：有向时差的数据分布，与主分析 2186 条记录一致。

    (A) JetLag_signed 直方图（东向为 +，西向为 −，24h 圆形差）；
    (B) 11 个 JetLagGroup_dir 的观测频数（纵轴序与 Figure 1 一致），便于看稀疏格。
    """
    df = pd.read_csv(data_used_csv, encoding="utf-8-sig")
    if "JetLag_signed" not in df.columns or "JetLagGroup_dir" not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_JETLAG_DIST, layout="constrained")

    s = df["JetLag_signed"].dropna()
    ax1.hist(s, bins=36, color=COLORS["primary"], alpha=0.82, edgecolor="white", linewidth=0.35)
    ax1.axvline(0.0, color="0.25", ls="--", lw=0.9)
    ax1.set_xlabel(r"Signed jet lag (h), East$+$ / West$-$")
    ax1.set_ylabel("Observations (n)")
    ax1.set_title("(A) Continuous signed jet lag", fontsize=9)

    gser = df["JetLagGroup_dir"].astype(str)
    counts = gser.value_counts()
    y_pos = np.arange(len(JETLAG_PLOT_ORDER), dtype=float)
    heights = [int(counts.get(g, 0)) for g in JETLAG_PLOT_ORDER]
    colors = [_dir_group_bar_color(g) for g in JETLAG_PLOT_ORDER]
    ax2.barh(y_pos, heights, color=colors, height=0.55, edgecolor="white", linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([g.replace("_", " ") for g in JETLAG_PLOT_ORDER], fontsize=6.0)
    ax2.invert_yaxis()
    ax2.set_xlabel("n observations (same order as Figure 1)")
    ax2.set_title("(B) Directional 11-group counts", fontsize=9)
    wpad = max(max(heights), 1) * 0.02
    for y, h in zip(y_pos, heights):
        if h > 0:
            ax2.text(h + wpad, y, str(h), va="center", fontsize=5.5, color="0.3")

    fig.suptitle("Jet lag distribution (main analysis sample)", fontsize=9.5)
    save_fig(fig, out_dir, "supp_jetlag_distribution")
    plt.close(fig)


def plot_supp_athlete_participation(data_used_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(data_used_csv, encoding="utf-8-sig")
    freq = df.groupby("Athlete_ID").size().value_counts().sort_index()

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    ax.bar(freq.index.astype(int), freq.values, color=COLORS["primary"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Competitions per athlete")
    ax.set_ylabel("Count of athletes")
    ax.set_title("Athlete participation frequency")
    plt.tight_layout()
    save_fig(fig, out_dir, "supp_athlete_participation")
    plt.close(fig)


def draw_all_figures(out_fig: Path) -> None:
    out_fig.mkdir(parents=True, exist_ok=True)
    emm = PROJECT_ROOT / "lmm_jetlag_travel_control" / "emmeans_FDR_travel_control.csv"
    plot_fig1_forest_main_effect(emm, out_fig)

    summ = PROJECT_ROOT / "heterogeneity_results" / "heterogeneity_summary.md"
    plot_fig2_heterogeneity_2x2(PROJECT_ROOT / "heterogeneity_results", summ, out_fig)

    plot_fig3_sensitivity(out_fig)

    dused = PROJECT_ROOT / "lmm_jetlag_travel_control" / "data_used_for_model.csv"
    plot_supp_nation_heatmap(dused, out_fig)
    plot_supp_jetlag_distribution(dused, out_fig)
    plot_supp_athlete_participation(dused, out_fig)
