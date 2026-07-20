"""
论文级绘图统一配置：SciencePlots + no-latex，多格式导出。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目根目录（srp_IWF）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 输出根目录
PAPER_OUTPUT_ROOT = Path(__file__).resolve().parent / "paper_outputs"

COLORS: dict[str, str] = {
    "primary": "#0C5DA5",
    "secondary": "#FF9500",
    "accent": "#00B945",
    "neutral": "#7F7F7F",
    "negative": "#D62728",
}

# 英寸尺寸
FIG_SINGLE_COL = (3.5, 2.8)
FIG_DOUBLE_COL = (7.0, 4.5)
FIG_PANEL_2X2 = (7.0, 6.0)
# Fig3：双 panel（联合 LRT vs 连续×时差交互）略宽
FIG_SENSITIVITY = (7.4, 3.2)
# 补充：有向时差连续分布 + 11 档频数（与 Fig1 纵轴序一致）
FIG_JETLAG_DIST = (7.2, 3.5)


def setup_mpl_style() -> None:
    """启用 SciencePlots（no-latex）；失败则回退到 serif 近似。"""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )


def save_fig(fig: Any, out_dir: Path, basename: str, *, tight: bool = True) -> None:
    """
    同时导出 PDF（矢量）、PNG（预览）、TIFF（投稿 600dpi LZW）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / basename
    kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.05}
    if tight:
        pass

    fig.savefig(str(base.with_suffix(".pdf")), **kwargs)
    fig.savefig(str(base.with_suffix(".png")), dpi=300, **kwargs)
    try:
        fig.savefig(
            str(base.with_suffix(".tiff")),
            dpi=600,
            pil_kwargs={"compression": "tiff_lzw"},
            **kwargs,
        )
    except Exception:
        # 部分环境 PIL 参数不同，退化为无压缩 TIFF
        fig.savefig(str(base.with_suffix(".tiff")), dpi=600, **kwargs)


def ensure_output_dirs() -> tuple[Path, Path, Path]:
    """创建 tables / figures / for_writer 子目录。"""
    root = PAPER_OUTPUT_ROOT
    tables = root / "tables"
    figures = root / "figures"
    fw = root / "for_writer"
    for d in (tables, figures, fw, fw / "data"):
        d.mkdir(parents=True, exist_ok=True)
    return tables, figures, fw
