# 运行顺序与角色说明

## 运行顺序

1. `preprocess2-athlete.py`
   读取 check_data/ 下原始 Excel，按 Top13 国家 + 参赛次数≥2 筛选，计算圆周有向时差。
   输出：iwf_data_top13_athlete2plus_circular.csv

2. `merge_travel.py`
   合并旅行时长数据（travel_time_merged.csv，按 Nation + City + Event_Year 匹配）。
   输出：athlete_top13_travel.csv

3. `LMM_jetlag_travel_v2.py --mode control`   ★ 论文正文主模型
   Sinclair_Total ~ Sex + JetLagGroup_dir + TravelDuration_c + (1|Athlete_ID)
   筛选：核心变量非缺失 → JetLag可分组 → Sinclair可计算 → TravelDuration非缺失（N=2186）
   输出目录：lmm_jetlag_travel_control/（data_used_for_model.csv、emmeans_FDR_travel_control.csv、
   LRT-A/LRT-B、Table1/2/Fig1 数据来源）
   同一脚本 `--mode main` 可单独检验旅行时长主效应（不含JetLagGroup），输出 lmm_jetlag_travel_main/。

4. `LMM_jetlag_v2.py`   补充分析（不控制旅行时长，用于对比/一致性检验，非正文主结果）
   Sinclair_Total ~ Sex + JetLagGroup_dir + (1|Athlete_ID)
   同一份 N=2186 样本，仅用于说明"控制旅行时长前后效应方向、显著性模式一致"。
   输出目录：lmm_jetlag_v2/

5. `heterogeneity_analysis.py [--run-sensitivity]`
   四个调节变量（性别/年龄/经验/体重）分层 + 交互项LRT；--run-sensitivity 额外做连续变量敏感性分析。
   输出目录：heterogeneity_results/

## 角色说明（论文正文用哪个数字）

| 结果 | 主模型 (travel_control) | 补充分析 (v2, 无旅行时长) |
|---|---|---|
| 整体 LRT | LRT-B: LR=53.27, df=10, p=6.6e-08 | LR=51.82, df=10, p=1.23e-07 |
| East 8-9h | β=-5.58, 95%CI[-8.40,-2.77], d=-0.42 | β=-7.77, 95%CI[-10.27,-5.27], d=-0.59 |
| Table 1 / Table 2 / Fig 1 | ✅ 数据来源 | 仅供文中提及对比，不进正式表格 |

依赖库：pandas, numpy, scipy, statsmodels, matplotlib, seaborn。
Cohen's d 计算公式：d = β / σ_ε （σ_ε 为 REML 残差标准差，即 sqrt(scale)，非残差方差）。
