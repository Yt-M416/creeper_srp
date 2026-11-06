# 论文写作资料包

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
python LMM_jetlag_v2.py                          # Figure 1灰色节点复现（不作为独立正文模型）
python heterogeneity_analysis.py --run-sensitivity
python 科研-绘图/make_paper_figures.py
```

中间结果写入 `lmm_jetlag_travel_control/`（主分析）、`lmm_jetlag_travel_main/`、`lmm_jetlag_v2/`（补充分析）、
`heterogeneity_results/`；论文用图表写入 `科研-绘图/paper_outputs/`。
