from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize heterogeneity analysis logs")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Root result directories, e.g. heterogeneity_results heterogeneity_年龄经验调整版",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output txt path",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_first(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def extract_lrt_block(log_text: str) -> dict[str, str]:
    return {
        "lr": extract_first(r"^LR:\s*([^\n]+)$", log_text) or "NA",
        "df": extract_first(r"^df:\s*([^\n]+)$", log_text) or "NA",
        "p": extract_first(r"^p:\s*([^\n]+)$", log_text) or "NA",
    }


def summarize_analysis_dir(analysis_dir: Path) -> str:
    log_path = analysis_dir / "analysis_log.txt"
    comp_path = analysis_dir / "model_comparison.txt"
    if not log_path.exists():
        return ""

    log_text = read_text(log_path)
    comp_text = read_text(comp_path) if comp_path.exists() else ""

    sample_size = extract_first(r"^样本量:\s*([^\n]+)$", log_text) or "NA"
    group_levels = extract_first(r"^Group levels:\s*([^\n]+)$", log_text) or "NA"
    full_formula = extract_first(r"^Full formula:\s*([^\n]+)$", log_text) or "NA"
    reduced_formula = extract_first(r"^Reduced formula:\s*([^\n]+)$", log_text) or "NA"
    lrt = extract_lrt_block(log_text)

    primary_lr = extract_first(r"^primary_lr=([^\n]+)$", comp_text)
    primary_df = extract_first(r"^primary_df=([^\n]+)$", comp_text)
    primary_p = extract_first(r"^primary_p=([^\n]+)$", comp_text)
    primary_coef = extract_first(r"^primary_interaction_coef=([^\n]+)$", comp_text)
    secondary_lr = extract_first(r"^secondary_lr=([^\n]+)$", comp_text)
    secondary_df = extract_first(r"^secondary_df=([^\n]+)$", comp_text)
    secondary_p = extract_first(r"^secondary_p=([^\n]+)$", comp_text)

    lines = []
    lines.append(f"[{analysis_dir.name}]")
    lines.append(f"- path: {analysis_dir}")
    lines.append(f"- sample_size: {sample_size}")
    lines.append(f"- group_levels: {group_levels}")
    lines.append(f"- full_formula: {full_formula}")
    lines.append(f"- reduced_formula: {reduced_formula}")
    lines.append(f"- lrt: LR={lrt['lr']}, df={lrt['df']}, p={lrt['p']}")

    if primary_p is not None:
        lines.append(
            f"- sensitivity_primary(continuous×continuous): LR={primary_lr}, df={primary_df}, p={primary_p}, coef={primary_coef}"
        )
    if secondary_p is not None:
        lines.append(
            f"- sensitivity_secondary(continuous×categorical): LR={secondary_lr}, df={secondary_df}, p={secondary_p}"
        )
    lines.append("")
    return "\n".join(lines)


def summarize_root(root: Path) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append(f"ROOT: {root}")
    lines.append("=" * 100)

    md_summary = root / "heterogeneity_summary.md"
    if md_summary.exists():
        lines.append("[heterogeneity_summary.md]")
        lines.append(read_text(md_summary).strip())
        lines.append("")

    run_log = root / "run_log.txt"
    if run_log.exists():
        run_text = read_text(run_log)
        lines.append("[run_log snapshot]")
        for line in run_text.splitlines()[:30]:
            lines.append(line)
        lines.append("")

    analysis_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and (d / "analysis_log.txt").exists()],
        key=lambda p: p.name,
    )

    lines.append(f"[analysis folders found: {len(analysis_dirs)}]")
    lines.append("")
    for d in analysis_dirs:
        lines.append(summarize_analysis_dir(d))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.roots]
    output = Path(args.output)

    all_blocks = []
    all_blocks.append("Heterogeneity Results Combined Summary")
    all_blocks.append("")
    for root in roots:
        if not root.exists():
            all_blocks.append(f"[WARN] root not found: {root}")
            all_blocks.append("")
            continue
        all_blocks.append(summarize_root(root))
        all_blocks.append("")

    output.write_text("\n".join(all_blocks), encoding="utf-8")
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()

