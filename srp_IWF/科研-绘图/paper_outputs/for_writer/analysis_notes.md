# Methods 写作要点（笔记）

## 线性混合模型 (LMM)

- 结局：`Sinclair_Total`
- 固定效应：`C(Sex) + C(JetLagGroup_dir)`（主分析 Model B，当前以 `LMM_jetlag_v2.py` 的 2186 子样本为主文口径）；随机截距：`(1 | Athlete_ID)`
- 参照组：`Minimal(0-1h)`（|signed jet lag| < 2 h）
- 软件：`statsmodels` `MixedLM`，优化 `L-BFGS`

## 多重比较

- 各时差组 vs Minimal 的事后对比：模型系数即对比（无 Sex×时差 交互）
- **FDR**：Benjamini–Hochberg，对 10 个非参照组 `p_raw` 校正得 `p_fdr`

## 异质性

- 分层模型（Design A）：在亚组内分别拟合同结构 LMM
- 全局交互：完整模型含 `C(Group):C(JetLagGroup_dir)` 与去交互的嵌套模型，**ML** 拟合下似然比检验

## 旅行时长稳健性

- 协变量：`TravelDuration_c`（中心化）
- `LMM_jetlag_travel.py --mode control` 报告 LRT-A（是否需旅行）与 LRT-B（控制旅行后时差是否仍联合显著）

## Sinclair

- IWF 2021–2024 周期 Sinclair 系数（男/女不同 A、b 参数），见 `LMM_jetlag.py` 内 `calculate_sinclair`。
