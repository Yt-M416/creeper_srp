"""
build_delivery_package.py
一次性整理论文交付材料：最终数据 + 核心代码 + 统计结果 + 图表 + Word 结果汇总。
运行：python build_delivery_package.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y%m%d")
OUT = ROOT / f"交付材料_{STAMP}"


def find_latest(pattern_prefix: str, root: Path) -> Path | None:
    cands = sorted(
        [p for p in root.glob(f"{pattern_prefix}_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    return cands[-1] if cands else None


def copy_dir(src: Path, dst: Path, patterns=("*.csv", "*.txt", "*.png")) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for f in src.glob(pat):
            shutil.copy2(f, dst / f.name)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "01_最终数据").mkdir(parents=True)
    (OUT / "02_核心代码").mkdir(parents=True)
    (OUT / "03_统计结果" / "主模型_无旅行时长_lmm_jetlag_v2").mkdir(parents=True)
    (OUT / "03_统计结果" / "稳健性_控制旅行时长_travel_control").mkdir(parents=True)
    (OUT / "03_统计结果" / "旅行时长主效应_travel_main").mkdir(parents=True)
    (OUT / "03_统计结果" / "异质性分析_heterogeneity").mkdir(parents=True)
    (OUT / "03_统计结果" / "论文表格").mkdir(parents=True)
    (OUT / "04_图表").mkdir(parents=True)

    # ---- 1. 最终数据 ----
    shutil.copy2(ROOT / "final_model_data_N2186.csv", OUT / "01_最终数据" / "final_model_data_N2186.csv")

    dict_md = OUT / "01_最终数据" / "数据字典.md"
    dict_md.write_text(
        """# final_model_data_N2186.csv 数据字典

样本：N = 2186 条记录，730 名运动员，13 个国家/地区（Top13）。
本文件是论文主模型与旅行时长稳健性模型共用的建模输入数据，可直接在 R / Python 中读取并复现论文全部主要结果。

| 列名 | 说明 |
|---|---|
| Athlete_ID | 运动员唯一标识（Name_Nation） |
| Name / Nation | 运动员姓名 / 国家代码 |
| Sex | 性别（Men / Women，从 Event 解析） |
| Born | 出生日期原始字符串 |
| Event / Event_Year / competition | 参赛项目 / 赛事年份 / 赛事名 |
| City / Timezone | 比赛地城市 / 比赛地时区 |
| Athlete_Timezone | 运动员训练地时区（按 Nation 映射） |
| Athlete_UTC_offset / Venue_UTC_offset | 运动员训练地 / 比赛地 UTC 偏移（小时，按赛事年份 7月1日计算，已含夏令时） |
| JetLag_signed | 有向时差：Venue_UTC_offset − Athlete_UTC_offset，24h 圆周最短路径折叠到 [-12, 12]；正=向东飞，负=向西飞 |
| JetLag_abs | \\|JetLag_signed\\| |
| JetLagGroup_abs | 绝对值 6 档分组（0-2h/2-4h/.../10-12h），用于备选模型 A |
| JetLagGroup_dir | **主分析用**方向性 11 档分组；参照组 Minimal(0-1h)（操作定义为 \\|JetLag_signed\\|<2h）；其余按 2h 一档，东/西各 5 档 |
| TravelDuration | 估计旅行时长（小时），部分样本缺失时已在预处理阶段整体剔除（本文件内均非缺失） |
| TravelDuration_c | TravelDuration 减去样本均值后的中心化变量，稳健性模型协变量 |
| Body_Weight | 体重 (kg) |
| Total / Snatch / Clean_Jerk | 总成绩 / 抓举 / 挺举 (kg) |
| Sinclair_Total | IWF Sinclair 标准化总成绩（结局变量） |
| Rank / Group | 名次 / 级别分组（原始表列） |
| Reference_Date | 用于计算 UTC 偏移的参考日期 |
| Note | 原始备注列 |

## 两个模型如何从本文件复现

**主模型（Model B，论文正文主要结果，对应 LR=51.82, East 8-9h β=-7.77）**：
```
Sinclair_Total ~ Sex + JetLagGroup_dir + (1|Athlete_ID)
```
事后比较：JetLagGroup_dir 各档 vs 参照组 Minimal(0-1h)，原始 p 值做 BH-FDR 校正（10 个比较）。

**稳健性模型（旅行时长控制，对应 LRT-A/LRT-B、East 8-9h 控制后 β=-5.58）**：
```
Sinclair_Total ~ Sex + JetLagGroup_dir + TravelDuration_c + (1|Athlete_ID)
```
""",
        encoding="utf-8",
    )

    # ---- 2. 核心代码（按运行顺序重命名拷贝）----
    code_map = [
        ("preprocess2-athlete.py", "01_preprocess2-athlete.py"),
        ("merge_travel.py", "02_merge_travel.py"),
        ("LMM_jetlag_v2.py", "03_LMM_jetlag_v2_主模型.py"),
        ("LMM_jetlag_travel_v2.py", "04_LMM_jetlag_travel_v2_稳健性.py"),
        ("heterogeneity_analysis.py", "05_heterogeneity_analysis_异质性敏感性.py"),
    ]
    for src_name, dst_name in code_map:
        shutil.copy2(ROOT / src_name, OUT / "02_核心代码" / dst_name)

    (OUT / "02_核心代码" / "README_运行顺序.md").write_text(
        """# 运行顺序说明

1. `01_preprocess2-athlete.py`
   读取 check_data/ 下原始 Excel，按 Top13 国家 + 参赛次数≥2 筛选，计算圆周有向时差。
   输出：iwf_data_top13_athlete2plus_circular.csv

2. `02_merge_travel.py`
   合并旅行时长数据（travel_time_merged.csv）。
   输出：athlete_top13_travel.csv

3. `03_LMM_jetlag_v2_主模型.py`
   论文正文主模型：Sinclair_Total ~ Sex + JetLagGroup_dir + (1|Athlete_ID)
   筛选：核心变量非缺失 → JetLag可分组 → Sinclair可计算 → TravelDuration非缺失（统一样本口径，N=2186）
   输出目录：lmm_jetlag_v2/（含 data_used_for_model.csv、emmeans_FDR.csv、LRT结果、图）

4. `04_LMM_jetlag_travel_v2_稳健性.py --mode control`
   稳健性模型：主模型基础上加入 TravelDuration_c 协变量。
   同一脚本 `--mode main` 可单独检验旅行时长主效应（不含JetLagGroup）。
   输出目录：lmm_jetlag_travel_control/ 、lmm_jetlag_travel_main/

5. `05_heterogeneity_analysis_异质性敏感性.py [--run-sensitivity]`
   四个调节变量（性别/年龄/经验/体重）分层 + 交互项LRT；--run-sensitivity 额外做连续变量敏感性分析。
   输出目录：heterogeneity_results/

依赖库：pandas, numpy, scipy, statsmodels, matplotlib, seaborn。
Cohen's d 计算公式：d = β / σ_ε （σ_ε 为 REML 残差标准差，即 sqrt(scale)，非残差方差）。
""",
        encoding="utf-8",
    )

    # ---- 3. 统计结果 ----
    copy_dir(ROOT / "lmm_jetlag_v2", OUT / "03_统计结果" / "主模型_无旅行时长_lmm_jetlag_v2")
    copy_dir(ROOT / "lmm_jetlag_travel_control", OUT / "03_统计结果" / "稳健性_控制旅行时长_travel_control")
    copy_dir(ROOT / "lmm_jetlag_travel_main", OUT / "03_统计结果" / "旅行时长主效应_travel_main")

    hetero_root = ROOT / "heterogeneity_results"
    hetero_out = OUT / "03_统计结果" / "异质性分析_heterogeneity"
    for prefix in ["sex", "age", "experience", "weight", "age_adjusted",
                   "sensitivity_Age_c", "sensitivity_BodyWeight_c", "sensitivity_Cumulative_Exp_c"]:
        latest = find_latest(prefix, hetero_root)
        if latest is not None:
            copy_dir(latest, hetero_out / latest.name)

    copy_dir(
        ROOT / "科研-绘图" / "paper_outputs" / "tables",
        OUT / "03_统计结果" / "论文表格",
        patterns=("*.csv",),
    )

    # ---- 4. 图表 ----
    copy_dir(
        ROOT / "科研-绘图" / "paper_outputs" / "figures",
        OUT / "04_图表",
        patterns=("*.png",),
    )

    print("交付材料已生成：", OUT)


if __name__ == "__main__":
    main()
