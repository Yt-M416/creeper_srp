"""
为论文写作人生成 README、数据字典、关键数字与方法笔记，并复制数据 CSV。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from io_utils import parse_lrt_block_model_b_jetlag, parse_travel_control_lrts
from plot_config import PROJECT_ROOT


def write_for_writer(fw: Path) -> None:
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "data").mkdir(exist_ok=True)

    readme = fw / "README.md"
    readme.write_text(
        """# 论文写作资料包

本目录由 `科研-绘图/make_paper_figures.py` 自动生成。

## 目录结构

- `../tables/` — Table 1–3 与附录表（CSV，可粘贴进 Word / LaTeX）
- `../figures/` — Fig 1–3 与补充图（PDF / PNG / TIFF）
- `data/athlete_top13_travel.csv` — 含旅行时长等变量的最终分析用数据（与主项目根目录同步拷贝）
- `data_dictionary.md` — 主要变量说明
- `key_numbers.md` — 可直接写入摘要/结果的核心数字
- `analysis_notes.md` — Methods 段可复用表述要点

## 复现统计结果

在项目根目录 `srp_IWF/` 下依次运行：

```text
python preprocess2-athlete.py
python merge_travel.py
python LMM_jetlag_travel_v2.py --mode control    # ★ 主分析（控制旅行时长），正文 headline 结果
python LMM_jetlag_travel_v2.py --mode main       # 旅行时长主效应（Discussion 引用）
python LMM_jetlag_v2.py                          # 补充分析（不控制旅行时长，仅作对照）
python heterogeneity_analysis.py --run-sensitivity
python 科研-绘图/make_paper_figures.py
```

中间结果写入 `lmm_jetlag_travel_control/`（主分析）、`lmm_jetlag_travel_main/`、`lmm_jetlag_v2/`（补充分析）、
`heterogeneity_results/`；论文用图表写入 `科研-绘图/paper_outputs/`。
""",
        encoding="utf-8",
    )

    dd = fw / "data_dictionary.md"
    dd.write_text(
        """# 数据字典（核心列）

| 列名 | 说明 |
|------|------|
| `Athlete_ID` | 运动员唯一标识 |
| `Event` | 比赛名称（含 Men/Women 用于解析性别） |
| `Total` | 总成绩 (kg) |
| `Body_Weight` | 体重 (kg) |
| `Athlete_UTC_offset` / `Venue_UTC_offset` | 运动员居住地 / 赛场 UTC 偏移 (h) |
| `JetLag_signed` | 圆形 24h 最短时差，东向为正 (h) |
| `JetLagGroup_dir` | 方向性 11 档分组；参照组 `Minimal(0-1h)` |
| `Sinclair_Total` | IWF Sinclair 标准化总分（跨级别可比） |
| `TravelDuration` | 估计旅行时长 (h)（起点为各国训练基地城市，非首都） |
| `TravelDuration_c` | TravelDuration 中心化，**主模型协变量** |
| `Reference_Date` | 比赛参考日期 |
| `Born` | 出生日期（原始字符串，解析用于年龄） |

详细预处理见仓库内 `LMM_jetlag.py`、`heterogeneity_analysis.py` 注释。
""",
        encoding="utf-8",
    )

    # key numbers from files（主分析=控制旅行时长，headline 在前；补充分析置后）
    lines_kn = ["# 关键数字（自动摘录）\n"]
    dused = PROJECT_ROOT / "lmm_jetlag_travel_control" / "data_used_for_model.csv"
    if dused.is_file():
        du = pd.read_csv(dused, encoding="utf-8-sig")
        lines_kn.append(f"- 模型用观测数 (rows): **{len(du)}**")
        lines_kn.append(f"- 运动员数 (unique Athlete_ID): **{du['Athlete_ID'].nunique()}**")
        if "Nation" in du.columns:
            lines_kn.append(f"- 国家/地区数 (unique Nation): **{du['Nation'].nunique()}**")

    tc = PROJECT_ROOT / "lmm_jetlag_travel_control" / "model_comparison.txt"
    if tc.is_file():
        t_tc = tc.read_text(encoding="utf-8")
        lr = parse_travel_control_lrts(t_tc)
        if lr.get("lrt_b_p") is not None:
            lines_kn.append(
                f"- ★ 主分析整体 LRT-B（控制旅行时长后 JetLag 联合检验）: "
                f"LR={lr.get('lrt_b_lr'):.3f}, df={lr.get('lrt_b_df')}, p={lr.get('lrt_b_p'):.3e}"
            )
        if lr.get("lrt_a_p") is not None:
            lines_kn.append(
                f"- 主分析 LRT-A（TravelDuration 本身是否显著协变量）: "
                f"LR={lr.get('lrt_a_lr'):.3f}, df={lr.get('lrt_a_df')}, p={lr.get('lrt_a_p'):.3e}"
            )
        for line in t_tc.splitlines():
            if "ICC" in line and "sigma_u" in line:
                lines_kn.append(f"- 主分析 ICC（REML 方差分量行）: `{line.strip()}`")
                break

    mc = PROJECT_ROOT / "lmm_jetlag_v2" / "model_comparison.txt"
    if mc.is_file():
        t = mc.read_text(encoding="utf-8")
        lrt = parse_lrt_block_model_b_jetlag(t)
        if lrt:
            lines_kn.append(
                f"- 补充分析（不控制旅行时长）整体 LRT，供一致性对照: "
                f"LR={lrt['lr']:.3f}, df={lrt['df']}, p={lrt['p']:.3e}"
            )

    (fw / "key_numbers.md").write_text("\n".join(lines_kn) + "\n", encoding="utf-8")

    an = fw / "analysis_notes.md"
    an.write_text(
        """# Methods 写作要点（笔记）

## 线性混合模型 (LMM) —— 主模型（论文正文 headline 结果）

- 结局：`Sinclair_Total`
- 固定效应：`C(Sex) + C(JetLagGroup_dir) + TravelDuration_c`（主分析，控制旅行时长，
  以 `LMM_jetlag_travel_v2.py --mode control` 的 2186 样本为正文口径）；随机截距：`(1 | Athlete_ID)`
- 参照组：`Minimal(0-1h)`（|signed jet lag| < 2 h）
- TravelDuration_c：旅行时长中心化协变量（TravelDuration − 样本均值）
- 软件：`statsmodels` `MixedLM`，优化 `L-BFGS`

## 整体检验

- **LRT-B（headline，正文引用）**：控制 TravelDuration_c 后，比较含/不含 JetLagGroup_dir 的模型，
  回答"时差分组作为整体是否对成绩有影响，且不能被旅行疲劳解释"。
- LRT-A：TravelDuration_c 本身是否显著改善拟合（旅行时长的独立贡献）。

## 补充分析（不控制旅行时长，一致性对照，不进正文表格）

- 固定效应：`C(Sex) + C(JetLagGroup_dir)`（`LMM_jetlag_v2.py`，同一份2186样本，去掉 TravelDuration_c）
- 用途：证明控制旅行时长前后，效应方向与显著性模式一致（仅量级从 β≈−7.77 减弱为 β≈−5.58），
  说明时差效应不完全等同于旅行疲劳。

## 多重比较

- 各时差组 vs Minimal 的事后对比：模型系数即对比（无 Sex×时差 交互）
- **FDR**：Benjamini–Hochberg，对 10 个非参照组 `p_raw` 校正得 `p_fdr`
- Table 2 / Fig 1 的数据来源均为主模型（`lmm_jetlag_travel_control`）。

## 异质性

- 分层模型（Design A）：在亚组内分别拟合同结构 LMM
- 全局交互：完整模型含 `C(Group):C(JetLagGroup_dir)` 与去交互的嵌套模型，**ML** 拟合下似然比检验

## Sinclair

- IWF 2021–2024 周期 Sinclair 系数（男/女不同 A、b 参数），见 `LMM_jetlag_travel_v2.py` 内 `calculate_sinclair`。

## Cohen's d

- d = β / σ_ε（σ_ε 为 REML 残差标准差 = sqrt(scale)，**不是**残差方差）。
""",
        encoding="utf-8",
    )

    src = PROJECT_ROOT / "athlete_top13_travel.csv"
    if src.is_file():
        shutil.copy2(src, fw / "data" / "athlete_top13_travel.csv")
