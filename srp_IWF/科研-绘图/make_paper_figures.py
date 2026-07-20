#!/usr/bin/env python3
"""
一键生成论文用 Table 1–3、附录表、Fig 1–3、补充图及 for_writer 资料包。

用法（在 srp_IWF 目录下）:
    python 科研-绘图/make_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle import build_consolidated_results
from figures import draw_all_figures
from manifest import write_usage_manifest, write_writer_prompt
from plot_config import ensure_output_dirs, setup_mpl_style
from tables import write_all_tables
from writer_pack import write_for_writer


def main() -> None:
    setup_mpl_style()
    tables_dir, figures_dir, fw_dir = ensure_output_dirs()
    write_all_tables(tables_dir)
    draw_all_figures(figures_dir)
    write_for_writer(fw_dir)

    consolidated = fw_dir / "consolidated_results.txt"
    build_consolidated_results(consolidated)
    write_usage_manifest(fw_dir, consolidated)
    write_writer_prompt(fw_dir, consolidated)

    print("Done. Outputs ->", tables_dir.parent)


if __name__ == "__main__":
    main()
