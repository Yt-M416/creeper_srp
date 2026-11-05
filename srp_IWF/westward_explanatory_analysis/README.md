# West 10–11 h解释性探索

本目录用于分析West 10–11 h组Sinclair Total较高的样本构成和稳健性。分析与论文正式主模型、数据和图表隔离，不修改`交付材料_20260720`中的任何文件。

## 数据和模型

- 输入：`../交付材料_20260720/01_最终数据/final_model_data_N2186.csv`
- 结局：`Sinclair_Total`
- 主模型：`Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir) + TravelDuration_c + (1 | Athlete_ID)`
- 参照组：`Minimal(0-1h)`
- 多重比较：10个非参照时差组使用Benjamini–Hochberg FDR
- 固定效应报告：REML；不同固定效应结构比较：ML
- 随机种子：`20261031`

## 运行

在本目录或项目根目录执行：

`python srp_IWF/westward_explanatory_analysis/run_exploration.py --stage all`

本机应使用项目原分析环境的Python（当前为`D:\Program_Files\miniconda3\python.exe`），其中包含`pandas`、`numpy`、`scipy`、`statsmodels`、`patsy`、`matplotlib`和`seaborn`。

也可单独运行：

- `--stage boundary`：主模型复现和±12 h边界敏感性；
- `--stage composition`：国家和赛事届次构成；
- `--stage influence`：逐国家、逐赛事删除；
- `--stage adjustment`：国家和赛事届次固定效应调整。

## 输出

- `outputs/`：CSV、JSON和运行日志；
- `figures/`：PNG和PDF图；
- `notes/`：各阶段解释及最终Results、Discussion和Limitations候选表述。

±12 h记录保留在正式主分析中。排除精确±12 h仅用于敏感性分析；若West系数方向和BH-FDR结论均不改变，后续解释不再展开边界问题，但内部结果永久保留。

## 本轮结论索引

- `notes/boundary_check.md`：±12 h核验；
- `notes/composition_findings.md`：国家和赛事集中度；
- `notes/influence_findings.md`：逐类删除诊断；
- `notes/adjustment_findings.md`：国家和赛事固定效应模型；
- `notes/historical_context.md`：2015 Houston官方背景与使用边界；
- `notes/final_synthesis.md`：综合判断及可用于论文的中英文表述。

综合结果将West 10–11 h的正向估计归类为“赛事届次与国家构成共同影响下的不稳定观察性关联”。国家与赛事同时调整后，West β由5.165衰减至2.199，95% CI跨0，BH-FDR P=0.5362。
