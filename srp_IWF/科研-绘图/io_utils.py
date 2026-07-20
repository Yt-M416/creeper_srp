"""
读取 heterogeneity_summary、model_comparison、敏感性输出等。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def find_latest_subdir(parent: Path, pattern: str) -> Path:
    """按修改时间取最新子目录。"""
    if not parent.is_dir():
        raise FileNotFoundError(f"目录不存在: {parent}")
    cands = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"未找到匹配 {parent}/{pattern}")
    return cands[0]


def parse_heterogeneity_summary_md(path: Path) -> pd.DataFrame:
    """解析 heterogeneity_summary.md 中的 Markdown 表格。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        if "---" in s:
            continue
        table_lines.append(s)
    if len(table_lines) < 2:
        return pd.DataFrame()
    rows: list[list[str]] = []
    for s in table_lines:
        parts = [c.strip() for c in s.strip("|").split("|")]
        parts = [p for p in parts if p != ""]
        rows.append(parts)
    header = rows[0]
    data = rows[1:]
    # 去掉可能的空列
    while header and header[-1] == "":
        header.pop()
    data = [r[: len(header)] + [""] * (len(header) - len(r)) for r in data]
    return pd.DataFrame(data, columns=header)


def parse_lrt_block_model_b_jetlag(text: str) -> dict[str, float | int]:
    """
    从 LMM_jetlag model_comparison.txt 提取 Model B 整体 JetLag LRT（首段）。
    """
    key = "Likelihood Ratio Test — Model B"
    if key not in text:
        key = "Likelihood Ratio Test - Model B"
    i = text.find(key)
    chunk = text[i : i + 1500] if i >= 0 else text[:1500]
    m = re.search(
        r"LR \(chi\^2\)\s*=\s*([\d.+-eE]+)\s*\n\s*df\s*=\s*(\d+).*?\n\s*p-value\s*=\s*([\d.+-eE]+|NA)",
        chunk,
        re.DOTALL,
    )
    if not m:
        return {}
    lr = float(m.group(1))
    df = int(m.group(2))
    p_str = m.group(3).strip()
    p = float("nan") if p_str.upper() == "NA" else float(p_str)
    return {"lr": lr, "df": df, "p": p}


def parse_travel_control_lrts(text: str) -> dict[str, Any]:
    """解析 LMM_jetlag_travel control 的 model_comparison.txt 中 LRT-A / LRT-B。"""
    out: dict[str, Any] = {}
    m_a = re.search(r"LRT-A:.*?LR=([\d.+-eE]+)\s+df=(\d+)\s+p=([\d.+-eE]+)", text, re.DOTALL)
    m_b = re.search(r"LRT-B:.*?LR=([\d.+-eE]+)\s+df=(\d+)\s+p=([\d.+-eE]+)", text, re.DOTALL)
    if m_a:
        out["lrt_a_lr"], out["lrt_a_df"], out["lrt_a_p"] = float(m_a.group(1)), int(m_a.group(2)), float(m_a.group(3))
    if m_b:
        out["lrt_b_lr"], out["lrt_b_df"], out["lrt_b_p"] = float(m_b.group(1)), int(m_b.group(2)), float(m_b.group(3))
    return out


def parse_sensitivity_model_comparison(path: Path) -> dict[str, float]:
    """解析 sensitivity_*/model_comparison.txt 的 primary LRT 与 primary 交互 p。"""
    d: dict[str, float] = {}
    if not path.is_file():
        return d
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("primary_lr="):
            d["primary_lr"] = float(line.split("=", 1)[1])
        elif line.startswith("primary_df="):
            d["primary_df"] = float(line.split("=", 1)[1])
        elif line.startswith("primary_p="):
            d["primary_p"] = float(line.split("=", 1)[1])
    return d


def parse_icc_from_model_comparison(text: str) -> float | None:
    m = re.search(r"ICC\s*=\s*([\d.]+)", text)
    if m:
        return float(m.group(1))
    return None
