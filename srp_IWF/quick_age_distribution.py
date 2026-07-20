"""
快速查看年龄分布（按比赛记录）并计算"截至当次比赛的累计参赛次数"。

用法：
    python quick_age_distribution.py
    python quick_age_distribution.py --input "iwf_data_top13_athlete2plus_circular.csv" --output_dir "age_distribution"
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick age distribution report")
    parser.add_argument(
        "--input",
        default="iwf_data_top13_athlete2plus_circular.csv",
        help="输入 CSV 文件路径",
    )
    parser.add_argument(
        "--output_dir",
        default="age_distribution",
        help="输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = ["Born", "Reference_Date", "Athlete_ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 出生日期示例: "Jan 08, 1980"
    df["Born_dt"] = pd.to_datetime(df["Born"], errors="coerce")
    df["Ref_dt"] = pd.to_datetime(df["Reference_Date"], errors="coerce")

    # 年龄（按比赛参考日期计算）
    df["Age"] = (df["Ref_dt"] - df["Born_dt"]).dt.days / 365.2425
    age = df["Age"].dropna()

    if age.empty:
        raise ValueError("没有可用年龄数据，请检查 Born 和 Reference_Date 格式。")

    # 累计参赛次数（截至当次比赛）
    # 关键逻辑：先按 Athlete_ID + 比赛日期排序，再 groupby.cumcount()+1
    df["__row_id"] = np.arange(len(df))
    df = df.sort_values(["Athlete_ID", "Ref_dt", "__row_id"], na_position="last").copy()
    df["Cumulative_Exp"] = df.groupby("Athlete_ID").cumcount() + 1

    # 文本统计
    summary_lines = [
        "Age Distribution Summary (record-level)",
        "=" * 60,
        f"N records (all): {len(df)}",
        f"N records (valid age): {len(age)}",
        f"Missing age: {len(df) - len(age)}",
        "",
        f"Mean age: {age.mean():.2f}",
        f"Std age: {age.std():.2f}",
        f"Median age: {age.median():.2f}",
        f"Min age: {age.min():.2f}",
        f"Max age: {age.max():.2f}",
        "",
        "Percentiles:",
        f"  P10: {age.quantile(0.10):.2f}",
        f"  P25: {age.quantile(0.25):.2f}",
        f"  P75: {age.quantile(0.75):.2f}",
        f"  P90: {age.quantile(0.90):.2f}",
        "",
        "Cumulative Experience Summary (record-level, up-to-current competition):",
        f"Mean cumulative exp: {df['Cumulative_Exp'].mean():.2f}",
        f"Median cumulative exp: {df['Cumulative_Exp'].median():.2f}",
        f"Min cumulative exp: {df['Cumulative_Exp'].min():.0f}",
        f"Max cumulative exp: {df['Cumulative_Exp'].max():.0f}",
    ]

    summary_path = os.path.join(args.output_dir, "age_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    # 5 岁一档计数
    bins = np.arange(10, 56, 5)
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    age_bin = pd.cut(age, bins=bins, right=False, labels=labels)
    bin_counts = age_bin.value_counts().sort_index()
    bin_df = bin_counts.rename_axis("Age_Bin").reset_index(name="Count")
    bin_df.to_csv(os.path.join(args.output_dir, "age_bin_counts.csv"), index=False, encoding="utf-8-sig")

    # 累计参赛次数分布（按记录）
    exp_dist = df["Cumulative_Exp"].value_counts().sort_index()
    exp_dist_df = exp_dist.rename_axis("Cumulative_Exp").reset_index(name="Record_Count")
    exp_dist_df.to_csv(
        os.path.join(args.output_dir, "cumulative_experience_record_counts.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 导出带累计参赛次数的逐条记录，便于核对
    out_cols = ["Athlete_ID", "Ref_dt", "Age", "Cumulative_Exp"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(
        os.path.join(args.output_dir, "records_with_age_and_cumulative_exp.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 直方图
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(age, bins=20, kde=True, color="#4C78A8")
    plt.axvline(age.mean(), color="red", linestyle="--", alpha=0.8, label=f"Mean={age.mean():.2f}")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.title("Age Distribution (record-level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "age_distribution_hist.png"), dpi=300)
    plt.close()

    # 累计参赛次数直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Cumulative_Exp"], bins=min(int(df["Cumulative_Exp"].max()), 20), color="#2E8B57")
    plt.xlabel("Cumulative Experience (up-to-current competition)")
    plt.ylabel("Record Count")
    plt.title("Cumulative Experience Distribution (record-level)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "cumulative_experience_hist.png"), dpi=300)
    plt.close()

    print(f"Done. Output directory: {os.path.abspath(args.output_dir)}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()

