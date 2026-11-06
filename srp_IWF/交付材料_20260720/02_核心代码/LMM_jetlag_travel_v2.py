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
INPUT_CSV = "athlete_top13_travel.csv"

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


def variance_components(result) -> tuple[float, float]:
    """返回 (σ_u², σ_ε²)。"""
    try:
        return float(result.cov_re.iloc[0, 0]), float(result.scale)
    except Exception:
        return np.nan, np.nan


def compute_icc(result) -> float:
    try:
        gv, rv = variance_components(result)
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


def plot_fig1_forest(emmeans_csv: str, out_png: str) -> None:
    """
    旧版 Figure 1（保留以复现原始纵向森林图）：
    读取事后比较 CSV（Table 4 同源），按 BH-FDR p < .05 高亮显著组。
    纵轴：West 大时差在上 → Minimal → East 大时差在下（与 table2_posthoc 一致）。
    """
    emm = pd.read_csv(emmeans_csv, encoding="utf-8-sig").set_index("Group")

    plot_order = [
        "West_10-11h", "West_8-9h", "West_6-7h", "West_4-5h", "West_2-3h",
        "Minimal(0-1h)",
        "East_2-3h", "East_4-5h", "East_6-7h", "East_8-9h", "East_10-11h",
    ]
    color_sig = "#0C5DA5"
    color_nonsig = "#7F7F7F"

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y_pos = np.arange(len(plot_order))

    for i, g in enumerate(plot_order):
        if g == "Minimal(0-1h)":
            ax.plot(0.0, i, marker="D", markersize=6, color=color_nonsig, zorder=3)
            ax.text(0.15, i, "Ref", fontsize=8, va="center", color=color_nonsig)
            continue
        if g not in emm.index:
            continue
        row = emm.loc[g]
        beta = float(row["Estimate"])
        lo = float(row["CI_lo"])
        hi = float(row["CI_hi"])
        p_fdr = float(row["p_fdr"])
        sig = p_fdr < 0.05
        color = color_sig if sig else color_nonsig
        ax.errorbar(
            beta, i,
            xerr=[[beta - lo], [hi - beta]],
            fmt="none", ecolor=color, capsize=3, linewidth=1.2, zorder=2,
        )
        ax.plot(
            beta, i, marker="o", markersize=6,
            markerfacecolor=color if sig else "white",
            markeredgecolor=color, markeredgewidth=1.2, zorder=3,
        )
        stars = _sig_fdr_stars(p_fdr)
        if stars:
            span = max(abs(hi - lo), 2.0)
            ax.text(beta + 0.06 * span, i, stars, va="center", fontsize=9,
                    color=color, fontweight="bold")

    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([g.replace("_", " ") for g in plot_order], fontsize=9)
    ax.set_xlabel(r"$\beta$ (Sinclair points vs Minimal)", fontsize=10)
    ax.set_title(
        "Directional jet lag: main effects\n"
        "(TravelDuration controlled; BH-FDR)",
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_paper_figure(fig, out_png: str) -> None:
    """同时输出高分辨率 PNG 和可编辑的矢量 PDF。"""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight", facecolor="white")


def _read_effect_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"Group", "Estimate", "CI_lo", "CI_hi"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少绘图所需列：{sorted(missing)}")
    return df.set_index("Group")


def plot_fig1_faceted_main_effect(
    emmeans_csv: str,
    data_used_csv: str,
    out_png: str,
) -> None:
    """
    新版正文 Figure 1：按旅行方向分面的调整后效应图。

    两个面板共享纵轴；点为相对于 Minimal 组的 beta，误差线为普通 95% CI；
    显著性符号根据 BH-FDR 校正后的 p 值；横轴标签同时报告各组记录数。
    """
    emm = _read_effect_table(emmeans_csv)
    data = pd.read_csv(data_used_csv, encoding="utf-8-sig")
    counts = data["JetLagGroup_dir"].value_counts()
    magnitudes = ["2-3h", "4-5h", "6-7h", "8-9h", "10-11h"]
    x = np.arange(len(magnitudes))
    panel_specs = [
        ("West", "A  Westward travel", "#35608D"),
        ("East", "B  Eastward travel", "#B85C38"),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 4.25), sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    all_lows, all_highs = [], []
    for ax, (direction, panel_title, color) in zip(axes, panel_specs):
        groups = [f"{direction}_{m}" for m in magnitudes]
        rows = emm.loc[groups]
        beta = rows["Estimate"].astype(float).to_numpy()
        lo = rows["CI_lo"].astype(float).to_numpy()
        hi = rows["CI_hi"].astype(float).to_numpy()
        p_fdr = rows["p_fdr"].astype(float).to_numpy()
        all_lows.extend(lo)
        all_highs.extend(hi)

        for idx in range(len(groups)):
            significant = p_fdr[idx] < 0.05
            ax.errorbar(
                x[idx], beta[idx],
                yerr=[[beta[idx] - lo[idx]], [hi[idx] - beta[idx]]],
                fmt="none", ecolor=color, elinewidth=1.25,
                capsize=3.2, capthick=1.0, zorder=2,
            )
            ax.plot(
                x[idx], beta[idx], marker="o", markersize=6.5,
                markerfacecolor=color if significant else "white",
                markeredgecolor=color, markeredgewidth=1.35,
                linestyle="none", zorder=3,
            )
            stars = _sig_fdr_stars(p_fdr[idx])
            if stars:
                ax.annotate(
                    stars, (x[idx], hi[idx]), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=color, fontweight="bold",
                )

        labels = [
            f"{m.replace('-', '–').replace('h', ' h')}\n"
            f"n={int(counts.get(g, 0))}"
            for m, g in zip(magnitudes, groups)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_title(panel_title, fontsize=10.5, fontweight="semibold", loc="left")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.55")
        ax.spines["bottom"].set_color("0.55")

    y_span = max(all_highs) - min(all_lows)
    y_pad = max(0.8, y_span * 0.08)
    axes[0].set_ylim(min(all_lows) - y_pad, max(all_highs) + y_pad)
    axes[0].set_ylabel("Adjusted difference in Sinclair Total\n(vs Minimal)", fontsize=9.5)
    fig.supxlabel("Time-zone difference (h)", fontsize=9.5, y=0.02)
    minimal_n = int(counts.get("Minimal(0-1h)", 0))
    fig.suptitle(
        "Adjusted directional jet-lag effects",
        fontsize=11.5, y=0.99,
    )
    fig.text(
        0.5, 0.925,
        f"Reference: Minimal (0–1 h), n={minimal_n}; error bars are ordinary 95% CIs",
        ha="center", va="top", fontsize=8.3, color="0.30",
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20, top=0.84)
    _save_paper_figure(fig, out_png)
    plt.close(fig)


def plot_fig1_signed_axis_main_effect(
    emmeans_csv: str,
    data_used_csv: str,
    out_png: str,
    unadjusted_csv: Optional[str] = None,
) -> None:
    """
    正文 Figure 1：合并旅行时长调整前后结果的有向时差效应图。

    横轴按实际旅行方向编码：向西为负、向东为正，0 为 Minimal 参考组；
    纵轴为相对于 Minimal 组的 beta，误差线为普通 95% CI；灰色空心点
    表示未控制旅行时长的模型，红/蓝实心点表示控制旅行时长的主模型，
    星号依据主模型 BH-FDR 校正后的 p 值，横轴同时报告各组记录数。

    unadjusted_csv 留为可选参数，便于既有模型流程继续复现仅含主模型的图；
    交付材料的 --figures-only 流程会传入该文件并生成合并后的正式 Figure 1。
    """
    adjusted = _read_effect_table(emmeans_csv)
    unadjusted = (
        _read_effect_table(unadjusted_csv)
        if unadjusted_csv is not None else None
    )
    data = pd.read_csv(data_used_csv, encoding="utf-8-sig")
    counts = data["JetLagGroup_dir"].value_counts()

    west_groups = [
        "West_10-11h", "West_8-9h", "West_6-7h", "West_4-5h", "West_2-3h"
    ]
    east_groups = [
        "East_2-3h", "East_4-5h", "East_6-7h", "East_8-9h", "East_10-11h"
    ]
    west_x = np.array([-10.5, -8.5, -6.5, -4.5, -2.5])
    east_x = np.array([2.5, 4.5, 6.5, 8.5, 10.5])
    colors = {"West": "#35608D", "East": "#B85C38"}

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    all_lows, all_highs = [], []
    pair_offset = 0.16 if unadjusted is not None else 0.0

    for direction, groups, positions in [
        ("West", west_groups, west_x),
        ("East", east_groups, east_x),
    ]:
        adj = adjusted.loc[groups]
        adj_beta = adj["Estimate"].astype(float).to_numpy()
        adj_lo = adj["CI_lo"].astype(float).to_numpy()
        adj_hi = adj["CI_hi"].astype(float).to_numpy()
        p_fdr = adj["p_fdr"].astype(float).to_numpy()
        all_lows.extend(adj_lo)
        all_highs.extend(adj_hi)
        color = colors[direction]

        for idx, group in enumerate(groups):
            adjusted_x = positions[idx] + pair_offset

            if unadjusted is not None:
                unadj_beta = float(unadjusted.loc[group, "Estimate"])
                unadj_lo = float(unadjusted.loc[group, "CI_lo"])
                unadj_hi = float(unadjusted.loc[group, "CI_hi"])
                unadj_p_fdr = float(unadjusted.loc[group, "p_fdr"])
                unadjusted_x = positions[idx] - pair_offset
                all_lows.append(unadj_lo)
                all_highs.append(unadj_hi)

                ax.plot(
                    [unadjusted_x, adjusted_x], [unadj_beta, adj_beta[idx]],
                    color="0.72", linewidth=0.9, zorder=1,
                )
                ax.errorbar(
                    unadjusted_x, unadj_beta,
                    yerr=[[unadj_beta - unadj_lo], [unadj_hi - unadj_beta]],
                    fmt="none", ecolor="0.48", elinewidth=1.05,
                    capsize=3.0, capthick=0.9, zorder=2,
                )
                ax.plot(
                    unadjusted_x, unadj_beta, marker="o", markersize=6.0,
                    markerfacecolor="white", markeredgecolor="0.48",
                    markeredgewidth=1.2, linestyle="none", zorder=3,
                )
                unadjusted_stars = _sig_fdr_stars(unadj_p_fdr)
                if unadjusted_stars:
                    ax.annotate(
                        unadjusted_stars, (unadjusted_x, unadj_hi), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color="0.42", fontweight="bold",
                    )

            ax.errorbar(
                adjusted_x, adj_beta[idx],
                yerr=[[adj_beta[idx] - adj_lo[idx]], [adj_hi[idx] - adj_beta[idx]]],
                fmt="none", ecolor=color, elinewidth=1.25,
                capsize=3.2, capthick=1.0, zorder=2,
            )
            ax.plot(
                adjusted_x, adj_beta[idx], marker="o", markersize=6.8,
                markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=1.4,
                linestyle="none", zorder=3,
            )
            stars = _sig_fdr_stars(p_fdr[idx])
            if stars:
                ax.annotate(
                    stars, (adjusted_x, adj_hi[idx]), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.5, color=color, fontweight="bold",
                )

    minimal_n = int(counts.get("Minimal(0-1h)", 0))
    ax.plot(
        0.0, 0.0, marker="D", markersize=6.5,
        markerfacecolor="0.45", markeredgecolor="0.45", zorder=3,
    )
    ax.annotate(
        "Ref", (0.0, 0.0), xytext=(0, 8), textcoords="offset points",
        ha="center", va="bottom", fontsize=8.3, color="0.35",
    )

    positions = np.concatenate([west_x, [0.0], east_x])
    groups = west_groups + ["Minimal(0-1h)"] + east_groups
    interval_labels = [
        "−11 to −10", "−9 to −8", "−7 to −6", "−5 to −4", "−3 to −2",
        "Minimal\n(0–1 h)",
        "+2 to +3", "+4 to +5", "+6 to +7", "+8 to +9", "+10 to +11",
    ]
    labels = [
        f"{label}\nn={int(counts.get(group, 0))}"
        for label, group in zip(interval_labels, groups)
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.8)

    y_span = max(all_highs) - min(all_lows)
    y_pad = max(0.9, y_span * 0.10)
    ax.set_ylim(min(all_lows) - y_pad, max(all_highs) + y_pad)
    ax.set_xlim(-12.0, 12.0)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=1)
    ax.axvline(0.0, color="0.72", linestyle="-", linewidth=0.8, zorder=1)
    ax.axvspan(-12.0, 0.0, color=colors["West"], alpha=0.035, zorder=0)
    ax.axvspan(0.0, 12.0, color=colors["East"], alpha=0.035, zorder=0)
    ax.text(
        0.02, 0.96, "WESTWARD", transform=ax.transAxes,
        ha="left", va="top", fontsize=8.2, color=colors["West"],
        fontweight="semibold",
    )
    ax.text(
        0.98, 0.96, "EASTWARD", transform=ax.transAxes,
        ha="right", va="top", fontsize=8.2, color=colors["East"],
        fontweight="semibold",
    )
    ax.set_ylabel("Difference in Sinclair Total\n(vs Minimal)", fontsize=9.5)
    ax.set_xlabel(
        "Signed time-zone difference (h; westward < 0, eastward > 0)",
        fontsize=9.5,
    )
    ax.set_title(
        "Directional jet-lag effects before and after travel-duration adjustment",
        fontsize=11.5, pad=31,
    )
    ax.text(
        0.5, 1.035,
        "Points are model coefficients; error bars are ordinary 95% CIs; "
        "asterisks match each model's point colour and indicate BH-FDR-adjusted p<0.05",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, color="0.30",
    )

    if unadjusted is not None:
        from matplotlib.lines import Line2D
        from matplotlib.legend_handler import HandlerTuple
        legend_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=6.2,
                markerfacecolor="white", markeredgecolor="0.48",
            ),
            (
                Line2D(
                    [0], [0], marker="o", linestyle="none", markersize=6.2,
                    markerfacecolor=colors["West"], markeredgecolor=colors["West"],
                ),
                Line2D(
                    [0], [0], marker="o", linestyle="none", markersize=6.2,
                    markerfacecolor=colors["East"], markeredgecolor=colors["East"],
                ),
            ),
        ]
        ax.legend(
            handles=legend_handles,
            labels=[
                "Without travel-duration adjustment",
                "Adjusted for travel duration",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None)}, loc="upper center",
            bbox_to_anchor=(0.5, 1.155), frameon=False, ncol=2,
            fontsize=8.2, handletextpad=0.5, columnspacing=1.4,
        )
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.55")
    ax.spines["bottom"].set_color("0.55")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.76)
    _save_paper_figure(fig, out_png)
    plt.close(fig)


def plot_fig2_travel_adjustment_comparison(
    adjusted_csv: str,
    unadjusted_csv: str,
    out_png: str,
) -> None:
    """
    旧版 Figure 2（保留以复现此前旅行时长调整前后的配对系数图）。

    灰色空心点为未控制旅行时长模型，彩色实心点为控制旅行时长主模型；
    两个点以细线连接，两套估计均显示普通 95% CI。
    """
    adjusted = _read_effect_table(adjusted_csv)
    unadjusted = _read_effect_table(unadjusted_csv)
    magnitudes = ["2-3h", "4-5h", "6-7h", "8-9h", "10-11h"]
    panel_specs = [
        ("West", "A  Westward travel"),
        ("East", "B  Eastward travel"),
    ]
    adjusted_color = "#35608D"

    all_bounds = pd.concat(
        [
            adjusted[["CI_lo", "CI_hi"]],
            unadjusted[["CI_lo", "CI_hi"]],
        ],
        axis=0,
    ).astype(float)
    x_pad = max(0.8, (all_bounds["CI_hi"].max() - all_bounds["CI_lo"].min()) * 0.06)
    x_limits = (
        float(all_bounds["CI_lo"].min() - x_pad),
        float(all_bounds["CI_hi"].max() + x_pad),
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 4.35), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    y = np.arange(len(magnitudes))
    offset = 0.10
    for ax, (direction, panel_title) in zip(axes, panel_specs):
        groups = [f"{direction}_{m}" for m in magnitudes]
        adj = adjusted.loc[groups]
        unadj = unadjusted.loc[groups]

        for idx, group in enumerate(groups):
            adj_beta = float(adj.loc[group, "Estimate"])
            unadj_beta = float(unadj.loc[group, "Estimate"])
            ax.plot(
                [unadj_beta, adj_beta],
                [y[idx] - offset, y[idx] + offset],
                color="0.72", linewidth=1.0, zorder=1,
            )

            for beta, lo, hi, ypos, point_color, face_color in [
                (
                    unadj_beta,
                    float(unadj.loc[group, "CI_lo"]),
                    float(unadj.loc[group, "CI_hi"]),
                    y[idx] - offset,
                    "0.48",
                    "white",
                ),
                (
                    adj_beta,
                    float(adj.loc[group, "CI_lo"]),
                    float(adj.loc[group, "CI_hi"]),
                    y[idx] + offset,
                    adjusted_color,
                    adjusted_color,
                ),
            ]:
                ax.errorbar(
                    beta, ypos,
                    xerr=[[beta - lo], [hi - beta]],
                    fmt="none", ecolor=point_color, elinewidth=1.1,
                    capsize=2.8, capthick=0.9, zorder=2,
                )
                ax.plot(
                    beta, ypos, marker="o", markersize=5.8,
                    markerfacecolor=face_color, markeredgecolor=point_color,
                    markeredgewidth=1.2, linestyle="none", zorder=3,
                )

        ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_title(panel_title, fontsize=10.5, fontweight="semibold", loc="left")
        ax.set_xlim(*x_limits)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [m.replace("-", "–").replace("h", " h") for m in magnitudes],
            fontsize=8.5,
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.55")
        ax.spines["bottom"].set_color("0.55")
        ax.tick_params(axis="x", labelsize=8.5)

    axes[0].invert_yaxis()

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor="white", markeredgecolor="0.48",
            label="Without travel-duration adjustment",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor="#35608D", markeredgecolor="#35608D",
            label="Adjusted for travel duration",
        ),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        frameon=False, ncol=2, fontsize=8.3, handletextpad=0.5,
        columnspacing=1.4,
    )
    fig.supxlabel(r"$\beta$ (Sinclair points vs Minimal)", fontsize=9.5, y=0.025)
    fig.suptitle(
        "Effect estimates before and after travel-duration adjustment",
        fontsize=11.5, y=0.99,
    )
    fig.text(
        0.5, 0.855,
        "Points are model coefficients; error bars are ordinary 95% CIs",
        ha="center", va="top", fontsize=8.3, color="0.30",
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.78)
    _save_paper_figure(fig, out_png)
    plt.close(fig)


def generate_delivery_main_figures() -> str:
    """用交付材料内已锁定的 CSV 重新生成合并后的正文 Figure 1。"""
    code_dir = os.path.dirname(os.path.abspath(__file__))
    delivery_root = os.path.dirname(code_dir)
    adjusted_dir = os.path.join(
        delivery_root, "03_统计结果", "01_主模型_控制旅行时长_travel_control"
    )
    unadjusted_dir = os.path.join(
        delivery_root, "03_统计结果", "02_补充_不控制旅行时长_lmm_jetlag_v2"
    )
    data_csv = os.path.join(
        delivery_root, "01_最终数据", "final_model_data_N2186.csv"
    )
    figure_dir = os.path.join(delivery_root, "04_图表")
    fig1_path = os.path.join(figure_dir, "fig1_signed_axis_main_effect.png")

    # 旧版双分面主效应图调用保留，便于复现此前 Figure 1：
    # plot_fig1_faceted_main_effect(
    #     os.path.join(adjusted_dir, "emmeans_FDR_travel_control.csv"),
    #     data_csv,
    #     os.path.join(figure_dir, "fig1_faceted_main_effect.png"),
    # )
    plot_fig1_signed_axis_main_effect(
        os.path.join(adjusted_dir, "emmeans_FDR_travel_control.csv"),
        data_csv,
        fig1_path,
        unadjusted_csv=os.path.join(unadjusted_dir, "emmeans_FDR.csv"),
    )
    return fig1_path


# ======================== 主流程 ========================

def main() -> None:
    parser = argparse.ArgumentParser(description="LMM with TravelDuration")
    parser.add_argument(
        "--mode", type=str, default="control",
        choices=["control", "main"],
        help="control: travel as covariate; main: travel as main effect",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="不重跑模型，直接用交付材料中的锁定结果生成合并后的正文 Figure 1",
    )
    args = parser.parse_args()

    if args.figures_only:
        setup_plot_style()
        fig1_path = generate_delivery_main_figures()
        print(f"Saved: {fig1_path}")
        return

    OUTPUT_DIR = f"lmm_jetlag_travel_{args.mode}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    setup_plot_style()
    log_path      = os.path.join(OUTPUT_DIR, "analysis_log.txt")
    summary_path  = os.path.join(OUTPUT_DIR, "model_summary.txt")
    compare_path  = os.path.join(OUTPUT_DIR, "model_comparison.txt")
    data_used_path = os.path.join(OUTPUT_DIR, "data_used_for_model.csv")

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
        df.to_csv(data_used_path, index=False, encoding="utf-8-sig")
        log_write(lf, f"\n  Saved model input data: {data_used_path}")

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

            icc_full = compute_icc(res_full_reml)
            sigma_u2, sigma_eps2 = variance_components(res_full_reml)
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
            sigma_u2, sigma_eps2 = variance_components(res_full_reml)
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
                "Random-effects variance (REML, full model):",
                f"  sigma_u^2 = {sigma_u2:.4f}   sigma_epsilon^2 = {sigma_eps2:.4f}   ICC = {icc_full:.4f}",
                "",
                "LRT-A: Does TravelDuration significantly improve over base Model B (JetLag-only)?",
                f"  Full vs Base-B  |  LR={lr_a:.4f}  df={df_a}  p={'NA' if pd.isna(p_a) else f'{p_a:.4e}'}",
                f"  LogLik (full)    = {float(res_full_ml.llf):.6f}",
                f"  LogLik (base-B)  = {float(res_base_b_ml.llf):.6f}",
                f"  → {'TravelDuration is a significant covariate.' if not pd.isna(p_a) and p_a < 0.05 else 'TravelDuration does NOT significantly improve the model.'}",
                "",
                "LRT-B: Is JetLagGroup_dir still significant after controlling TravelDuration?",
                f"  Full vs No-JLag  |  LR={lr_b:.4f}  df={df_b}  p={'NA' if pd.isna(p_b) else f'{p_b:.4e}'}",
                f"  LogLik (full)    = {float(res_full_ml.llf):.6f}",
                f"  LogLik (no-JLag) = {float(res_no_jlag_ml.llf):.6f}",
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
                "Random-effects variance (REML, full model):",
                f"  sigma_u^2 = {sigma_u2:.4f}   sigma_epsilon^2 = {sigma_eps2:.4f}   ICC = {icc_full:.4f}",
                "",
                "LRT: TravelDuration_c as main effect (Full vs Sex-only)",
                f"  LR={lr_m:.4f}  df={df_m}  p={'NA' if pd.isna(p_m) else f'{p_m:.4e}'}",
                f"  LogLik (full)    = {float(res_full_ml.llf):.6f}",
                f"  LogLik (reduced) = {float(res_reduced_ml.llf):.6f}",
                f"  → {'TravelDuration has a significant main effect on Sinclair_Total.' if not pd.isna(p_m) and p_m < 0.05 else 'TravelDuration does NOT have a significant main effect.'}",
                "",
            ]

        # -------- STEP 7: 系数报告 --------
        log_write(lf, "\n" + "=" * 80)
        log_write(lf, "STEP 7 - Fixed-effect coefficients (REML fit)")
        log_write(lf, "=" * 80)
        log_write(lf, f"  sigma_u^2 = {sigma_u2:.4f}   sigma_epsilon^2 = {sigma_eps2:.4f}   ICC = {icc_full:.4f}")
        log_write(lf, f"  AIC = {res_full_ml.aic:.1f}  BIC = {res_full_ml.bic:.1f}")

        comp_lines += [
            "=" * 90,
            "Fixed-effect coefficients (REML):",
            f"  sigma_u^2={sigma_u2:.4f}  sigma_epsilon^2={sigma_eps2:.4f}  ICC={icc_full:.4f}  "
            f"AIC={res_full_ml.aic:.1f}  BIC={res_full_ml.bic:.1f}",
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
            emm_df["p_fdr"]     = p_fdr_arr
            sigma_eps = np.sqrt(sigma_eps2)
            emm_df["Cohens_d"]  = emm_df["Estimate"] / sigma_eps
            emm_df["sig_raw"]   = emm_df["p_raw"].apply(sig_stars)
            emm_df["sig_fdr"]   = emm_df["p_fdr"].apply(sig_stars)

            hdr = (f"  {'Group':<20} {'Estimate':>9} {'SE':>7} {'Cohen_d':>8} "
                   f"{'95%CI':^20} {'p_raw':>9} {'sig':>4}  {'p_fdr':>9} {'sig_fdr':>7}")
            sep = "  " + "-" * (len(hdr) - 2)
            log_write(lf, hdr)
            log_write(lf, sep)
            for _, row in emm_df.iterrows():
                ci_str = f"[{row['CI_lo']:+.3f},{row['CI_hi']:+.3f}]"
                log_write(lf, f"  {row['Group']:<20} {row['Estimate']:>+9.4f} {row['SE']:>7.4f} "
                              f"{row['Cohens_d']:>+8.3f} {ci_str:^20} {row['p_raw']:>9.4f} {row['sig_raw']:>4}  "
                              f"{row['p_fdr']:>9.4f} {row['sig_fdr']:>7}")

            emm_csv = os.path.join(OUTPUT_DIR, "emmeans_FDR_travel_control.csv")
            emm_df.to_csv(emm_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
            log_write(lf, f"\n  Saved: {emm_csv}")

            # 旧版纵向森林图调用保留，便于复现此前 Figure 1：
            # legacy_fig1_path = os.path.join(OUTPUT_DIR, "fig1_forest_main_effect.png")
            # plot_fig1_forest(emm_csv, legacy_fig1_path)

            # 旧版双分面主效应图调用保留，便于复现此前 Figure 1：
            # faceted_fig1_path = os.path.join(OUTPUT_DIR, "fig1_faceted_main_effect.png")
            # plot_fig1_faceted_main_effect(emm_csv, data_used_path, faceted_fig1_path)

            fig1_path = os.path.join(OUTPUT_DIR, "fig1_signed_axis_main_effect.png")
            plot_fig1_signed_axis_main_effect(emm_csv, data_used_path, fig1_path)
            log_write(lf, f"  Figure 1 (signed-axis effect plot, from {os.path.basename(emm_csv)}) saved: {fig1_path}")

            comp_lines += [
                "=" * 90,
                "Post-hoc: each JetLagGroup vs Minimal  [FDR-BH, control mode]",
                hdr, sep,
            ]
            for _, row in emm_df.iterrows():
                ci_str = f"[{row['CI_lo']:+.3f},{row['CI_hi']:+.3f}]"
                comp_lines.append(
                    f"  {row['Group']:<20} {row['Estimate']:>+9.4f} {row['SE']:>7.4f} "
                    f"{row['Cohens_d']:>+8.3f} {ci_str:^20} {row['p_raw']:>9.4f} {row['sig_raw']:>4}  "
                    f"{row['p_fdr']:>9.4f} {row['sig_fdr']:>7}"
                )
            comp_lines += [
                f"  Reference: Minimal(0-1h)  |  Cohen's d = beta / sigma_epsilon ({sigma_eps:.4f})",
                "  Sig: *** <.001  ** <.01  * <.05  . <.1",
                "",
            ]

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
            sf.write(f"sigma_u^2: {sigma_u2:.4f}\n")
            sf.write(f"sigma_epsilon^2: {sigma_eps2:.4f}\n")
            sf.write(f"ICC: {icc_full:.4f}\n\n")
            sf.write(res_full_reml.summary().as_text())
        log_write(lf, f"Model summary saved: {summary_path}")

        log_write(lf, "\n" + "=" * 80)
        log_write(lf, f"LMM_jetlag_travel  mode={args.mode}  - END")
        log_write(lf, "=" * 80)


if __name__ == "__main__":
    main()
