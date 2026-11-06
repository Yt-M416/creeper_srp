"""
将所有“文字型”实验输出（log / model_comparison / model_summary / heterogeneity summary）
合并到一份 consolidated_results.txt，方便论文写作时一口气查阅。

注意：
- CSV 表格已在 paper_outputs/tables/ 提供，这里不再重复。
- 图片已在 paper_outputs/figures/ 提供，这里不再重复。
- 本文件仅整合“文字结果”，保留原始内容不修改，仅在每段前后加分节标题。
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

from io_utils import find_latest_subdir
from plot_config import PROJECT_ROOT


SECTION_BAR = "#" * 96
SUB_BAR = "-" * 96


def _safe_read(p: Path) -> str:
    if not p.is_file():
        return f"[FILE NOT FOUND] {p}\n"
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")


def _section(title: str, src: Path, body: str) -> str:
    rel = src.as_posix()
    return (
        f"\n{SECTION_BAR}\n"
        f"# {title}\n"
        f"# Source: {rel}\n"
        f"{SECTION_BAR}\n\n"
        f"{body.rstrip()}\n"
    )


def build_consolidated_results(out_path: Path) -> None:
    """把核心文字结果拼成一个文件，供论文写作时检索。"""
    parts: list[str] = []

    parts.append(
        f"""# Consolidated Statistical Results
# Generated: {datetime.now().isoformat(timespec='seconds')}
#
# 本文件汇总了论文写作需要的全部“文字型”统计输出。
# 章节顺序约定（章节编号保持不变，但角色已按最新口径更新）：
#   1. 图形对照数据 (不控制旅行时长, LMM_jetlag_v2, 2186 obs / 730 athletes)
#      —— 仅供 Figure 1 灰色节点使用，不作为独立正文模型报告
#   2. 主分析 (Main analysis, 控制旅行时长 TravelDuration_c, control mode)
#      —— ★ 正文 headline 结果，整体检验见 2.2 节 LRT-B
#   3. 旅行时长主效应 (main mode, 仅作 Discussion 引用，不放主分析)
#   4. 异质性分析 (subgroup-stratified + interaction LRT)
#
# 注意：
#   - 主模型和异质性分析统一使用 2186 条记录 / 730 名运动员。
#   - 表格 (CSV) 见 paper_outputs/tables/，图见 paper_outputs/figures/，
#     此处不重复粘贴 CSV/图，避免冗余。
#   - Table 1 / Table 2 / Fig 1 的数据来源均为 2.x 节（lmm_jetlag_travel_control）。
"""
    )

    # 1) 不控制旅行时长的图形对照数据（仅供Figure 1灰色节点复现）
    main_dir = PROJECT_ROOT / "lmm_jetlag_v2"
    parts.append(_section(
        "1.1  Figure-only comparison (no travel control): data flow & sample composition",
        main_dir / "analysis_log.txt",
        _safe_read(main_dir / "analysis_log.txt"),
    ))
    parts.append(_section(
        "1.2  Figure-only comparison (no travel control): coefficients and post-hoc FDR",
        main_dir / "model_comparison.txt",
        _safe_read(main_dir / "model_comparison.txt"),
    ))
    parts.append(_section(
        "1.3  Figure-only comparison (no travel control): detailed REML summaries",
        main_dir / "model_summary.txt",
        _safe_read(main_dir / "model_summary.txt"),
    ))

    # 2) 主分析（控制旅行时长，正文 headline 结果）
    tc_dir = PROJECT_ROOT / "lmm_jetlag_travel_control"
    parts.append(_section(
        "2.1  Main analysis (travel-controlled): data flow & coefficients",
        tc_dir / "analysis_log.txt",
        _safe_read(tc_dir / "analysis_log.txt"),
    ))
    parts.append(_section(
        "2.2  Main analysis (travel-controlled): LRT-A / LRT-B  ★ overall test (headline) & FDR post-hoc",
        tc_dir / "model_comparison.txt",
        _safe_read(tc_dir / "model_comparison.txt"),
    ))
    parts.append(_section(
        "2.3  Main analysis (travel-controlled): REML model summary",
        tc_dir / "model_summary.txt",
        _safe_read(tc_dir / "model_summary.txt"),
    ))

    # 3) 旅行主效应
    tm_dir = PROJECT_ROOT / "lmm_jetlag_travel_main"
    parts.append(_section(
        "3.1  Travel duration as main effect: model comparison",
        tm_dir / "model_comparison.txt",
        _safe_read(tm_dir / "model_comparison.txt"),
    ))
    parts.append(_section(
        "3.2  Travel duration as main effect: REML model summary",
        tm_dir / "model_summary.txt",
        _safe_read(tm_dir / "model_summary.txt"),
    ))

    # 4) 异质性
    h_dir = PROJECT_ROOT / "heterogeneity_results"
    parts.append(_section(
        "4.1  Heterogeneity: overall LRT summary across moderators",
        h_dir / "heterogeneity_summary.md",
        _safe_read(h_dir / "heterogeneity_summary.md"),
    ))

    for tag in ["sex", "age", "experience", "weight"]:
        try:
            sub = find_latest_subdir(h_dir, f"{tag}_*")
            parts.append(_section(
                f"4.2  Heterogeneity ({tag}): full model_comparison",
                sub / "model_comparison.txt",
                _safe_read(sub / "model_comparison.txt"),
            ))
        except FileNotFoundError:
            parts.append(f"\n[Heterogeneity {tag}: subdir not found]\n")

    out_path.write_text("\n".join(parts), encoding="utf-8")
