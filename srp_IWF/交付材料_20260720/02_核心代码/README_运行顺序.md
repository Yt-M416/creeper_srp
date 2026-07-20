# 运行顺序说明

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
