# Methods 写作要点（笔记）

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

## Figure 1对照估计（不控制旅行时长，不作独立模型报告）

- 固定效应：`C(Sex) + C(JetLagGroup_dir)`（`LMM_jetlag_v2.py`，同一份2186样本，去掉 TravelDuration_c）
- 用途：仅生成Figure 1中的灰色对照节点及其BH-FDR显著性标记，直观展示加入旅行时长后估计值的变化；
  正文核心结论、正式表格与模型检验均以控制旅行时长的主模型为准。

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
