# Writer Prompt —— 论文代写提示 (给写作者 / AI)

> 你是一位资深运动科学 / 生物统计论文写作者，擅长 IJSPP 与 JSS 风格。
> 请基于本资料包的**既有分析结果**撰写一篇英文论文，不要再做任何统计计算；
> 所有数字必须来自下列文件，**禁止杜撰数字**。

---

## 1. 研究背景 (一句话)

国际举重联合会 (IWF) 顶级选手在跨时区国际赛中的表现是否受"有向"时差影响 (East vs West)，并在控制旅行时长后是否依然稳健，以及效应是否在性别 / 年龄 / 经验 / 体重级别间异质。

## 2. 样本 (Methods / Results 开头这么写)

- 2018–2024 间 IWF World Cup / World Championships / Olympics 等 **top-13 国家** 男女选手；
- 最终分析集：**N = 2186 观测 / 730 位运动员**；
- 结局：`Sinclair_Total` (IWF 2021–2024 周期 Sinclair 标准化总成绩)；
- 主暴露：`JetLagGroup_dir` — 将 signed jet lag (东向为正，圆形 24h 最短距离) 分为 11 档，参照组 `Minimal(0–1h)`。

## 3. 统计模型 (Methods 段抄 `for_writer/analysis_notes.md`)

主模型 (Model B)：

    Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir) + (1 | Athlete_ID)   # REML

- 10 个非参照组 × vs Minimal 的对比 p 值用 **Benjamini–Hochberg FDR** 校正；
- 稳健性：加 `TravelDuration_c` 为协变量后重复 JetLag 联合 LRT（LRT-B）；
- 异质性：分层模型 (Design A) + 交互项 LRT (Design B, ML 拟合)；
- ICC 报告自 REML 方差分量。

## 4. 主要结论 (摘要 3–4 句 + Results 首段)

1. 整体方向性时差对 Sinclair 有显著影响 —— LRT (Model B) 见 `for_writer/consolidated_results.txt` § 1.2；具体大小见 Fig 1 / Table 2。
2. 控制旅行时长 (`TravelDuration_c`) 后该效应仍联合显著 —— LRT-B 见 `for_writer/consolidated_results.txt` § 2.2。
3. 效应在性别 / 年龄 / 经验 / 体重级别亚组间**高度同质** (全部 LRT p > 0.2) —— Table 3 + Fig 2。
4. 连续调节敏感性分析同样未发现显著交互 —— Supp Table S1 + Fig 3。

## 5. 逐段提示 (Paper skeleton)

### Title
示例：*Directional Jet Lag and Weightlifting Performance in Elite Athletes: A Mixed-Effects Analysis of 2186 International Competition Entries*

### Abstract (250 字内，结构化：Purpose / Methods / Results / Conclusion)

- Purpose：评估方向性时差对 Sinclair 总分的影响及稳健性 / 同质性；
- Methods：引用 Sample + 模型 + FDR + 稳健 + 异质设计；
- Results：填入 `key_numbers.md` 中 **Model B LRT** 与 **ICC**；引用 Fig 1 显著组数量、方向；
- Conclusion：效应可观察但有限，且跨亚组一致 → 对实战 periodization 的含义。

### 1. Introduction (≤600 字)
- 背景：时差 + 运动表现 (短跑 / 游泳 / 团队项目的既有文献)；举重领域几乎空白。
- Gap：先前研究多为 absolute 时差或样本较小、未控制旅行；
- 本研究：大样本、有向、控制旅行、异质性。

### 2. Methods
- 2.1 Study design & data source：IWF 赛历 + 运动员来源国 UTC。
- 2.2 Variables：照抄 `data_dictionary.md`。
- 2.3 Outcome：Sinclair。
- 2.4 Statistical analysis：照抄 `analysis_notes.md`。强调：
  - REML；随机截距；FDR-BH；α=0.05；
  - 稳健性（旅行时长）；异质性（Design A/B）；
  - 软件：Python 3.x, statsmodels, scipy。

### 3. Results
- 3.1 Sample — 引用 Table 1；自然语言一句话覆盖 2186 / 730 / 13 国。
- 3.2 Overall jet lag effect — 报告 Model B LRT；解读 Fig 1；在文中列出显著组 β(95%CI, p_fdr) 的 1–3 条典型值（从 Table 2 挑）。
- 3.3 Travel fatigue robustness — 一句话 + 括号 LRT-A / LRT-B。
- 3.4 Homogeneity across subgroups — 引用 Table 3；四句话，每亚组一句；强调"四条线重叠" (Fig 2)。
- 3.5 Sensitivity of continuous moderators — 引用 Supp Table S1 + Fig 3。

### 4. Discussion (≤1000 字)
- 主发现 2 句总结；
- 与既有文献对比：东/西时差方向性效应 (eg. Song et al., Roach & Sargent)；
- 机理假设：昼夜节律错位 / 训练节奏中断 / 旅行疲劳可分离；
- 实践含义：跨时区赛事前的 rel locationg / 抵达时间建议；
- Limitations：观察性、无训练负荷数据、缺实际出发日 → TravelDuration 为估计值；
- Future work：更细粒度的昼夜指标（DLMO、actigraphy）。

### 5. Conclusion (3 句)
照搬 §4 主要结论 1 / 4；强调"整体存在但亚组同质"。

---

## 6. 数字查询速查

**样本 / ICC / Model B LRT** → `for_writer/key_numbers.md`（已自动抽取）。
**具体某组 β / CI / p_fdr** → `tables/table2_posthoc_fdr.csv`。
**异质性每行 LR / df / p** → `tables/table3_heterogeneity_lrt.csv`。
**稳健性 LRT-A / LRT-B** → `for_writer/consolidated_results.txt` § 2.2。
**任何其它细节** → 检索 `for_writer/consolidated_results.txt`；它按 1.x / 2.x / 3.x / 4.x 顺序组织，节标题明确。

---

## 7. 风格与格式约束

- 目标期刊：**IJSPP / JSS / 体育数据分析 Q2**。句子短、主动语态、统计数字写在括号内。
- 数字小数位：β / CI 取 **2 位**；p 值 ≥ 0.001 用 3 位小数，<0.001 写 "<0.001"。
- 表格里 FDR 显著 (p_fdr < 0.05) 的行加 **粗体** 或 `*`。
- 图不要放在正文里 inline 嵌入描述，只用 "Figure 1 shows..." 这种引用。
- 参考文献请写作者自行查 PubMed，但风格统一 (Vancouver / AMA 皆可，与目标期刊一致)。

---

## 8. 交付物

- 论文 `.docx` (或 `.tex`)，包含：Abstract / Introduction / Methods / Results / Discussion / Conclusion / References；
- 若有改动建议，可直接在文档里用修订模式注明；
- 不需要回来再问我数字，全部来自本资料包。

(End of prompt)
