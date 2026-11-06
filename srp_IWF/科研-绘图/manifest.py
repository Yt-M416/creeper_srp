"""
为论文写作者生成 USAGE_MANIFEST.md（所有文件清单+用途）
以及 WRITER_PROMPT.md（可直接复制粘贴给写手/AI 的写作提示）。

设计原则：
- 不重复已有结果，只引用路径 + 说明用途；
- 每个文件标注：论文哪个章节用 / 填什么数字；
- Prompt 里包含：研究背景、样本、结论要点、分节提示、期刊风格。
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def write_usage_manifest(fw_dir: Path, consolidated_path: Path) -> None:
    """fw_dir 为 paper_outputs/for_writer ，consolidated_path 为整合后的 txt。"""
    paper_root = fw_dir.parent  # paper_outputs
    rel_cons = _rel(consolidated_path, paper_root)

    md = f"""# 论文写作资料包 —— 使用清单 (USAGE MANIFEST)

> 自动生成于 {datetime.now():%Y-%m-%d %H:%M}。本文件告诉你：**每个文件是什么、论文哪一部分会用到、怎么填数字**。
> 所有路径均相对于 `科研-绘图/paper_outputs/`。

---

## 0. 先看这两份

| 文件 | 用途 |
|---|---|
| `for_writer/WRITER_PROMPT.md` | **给写作者/AI 的写作提示**：包含研究背景、样本、主要结论、逐段提示与风格要求。按它的框架写即可。 |
| `{rel_cons}` | **合并后的文字型统计结果**：主分析（控制旅行时长，2.x节）为正文headline，补充分析（不控制旅行时长，1.x节）为对照；事后比较 / 异质性 LRT 均在同一文件中，分节明确。**引用统计量时查这里。** |

---

## 1. 表格 (tables/) —— 直接插入论文

| 文件 | 对应论文位置 | 说明 |
|---|---|---|
| `tables/table1_descriptive.csv` | **Table 1** Participants and data description | 性别分列的 n / 年龄 / 体重 / Sinclair / per-athlete obs。粘进 Word 后手动美化即可。 |
| `tables/table2_posthoc_fdr.csv` | **Table 2** Post-hoc FDR comparisons vs Minimal | 10 个方向性时差组 vs 参照组 Minimal 的 β / SE / z / p_raw / p_fdr。显著行用加粗或 `*` 标注。 |
| `tables/table3_heterogeneity_lrt.csv` | **Table 3** Heterogeneity LRT summary | 四维度调节变量的 LR / df / p。全部不显著，用来支持"效应同质"结论。|
| `tables/supp_table1_sensitivity.csv` | **Supplementary Table S1** Continuous moderator sensitivity | Age / BodyWeight / Cumulative_Exp 作为连续变量的交互检验。|

---

## 2. 图片 (figures/) —— 直接插入论文

每张图都同时有 PDF（矢量、推荐嵌入 Word）、PNG（预览）、TIFF（投稿时使用，600 DPI LZW）。

| 文件（同名三种后缀） | 对应论文位置 | 说明 |
|---|---|---|
| `figures/fig1_forest_main_effect.*` | **Figure 1** | 11 档方向性时差的森林图（effect vs Minimal，含 95% CI，FDR 显著用实心点+星号）。 |
| `figures/fig2_heterogeneity_2x2.*` | **Figure 2** | 2×2 异质性折线图：Sex / Age / Experience / Weight 四维度的 Design-A 亚组效应。|
| `figures/fig3_sensitivity.*` | **Figure 3** 或 **Supp Fig S1** | 两栏：左 (A) 方向性时差联合 LRT（主模型 + 控制旅行 LRT-B）；右 (B) 连续协变量×有向时差交互 LRT；横轴均为 $-\\log_{10}(p)$。 |
| `figures/supp_nation_jetlag_heatmap.*` | **Supp Fig S2** | Nation × JetLagGroup_dir 观测频数热力图，按 East 8–9h 降序；回答"显著格数据是否集中于单一国家"的质疑。 |
| `figures/supp_jetlag_distribution.*` | **Supp Fig S3** | 有向时差暴露分布：左为连续 `JetLag_signed` 直方图；右为 11 档 `JetLagGroup_dir` 频数（与 Fig1 纵轴顺序一致）。 |
| `figures/supp_athlete_participation.*` | **Supp Fig S4** | 每位运动员参赛次数分布（支撑随机截距合理性）。 |

---

## 3. 数据 (for_writer/data/)

| 文件 | 用途 |
|---|---|
| `for_writer/data/athlete_top13_travel.csv` | 主文分析用最终数据表（含旅行时长）。若审稿人要求公开数据集或复现时使用。 |

---

## 4. 写作参考 (for_writer/)

| 文件 | 用途 |
|---|---|
| `for_writer/README.md` | 资料包总览与复现步骤。|
| `for_writer/data_dictionary.md` | 列名含义；写 Methods 段 Variables 时照抄或摘抄。|
| `for_writer/key_numbers.md` | **自动抽取的关键数字**：样本量、Model B LRT、ICC、旅行控制后 LRT-B，可直接粘进 Abstract / Results。|
| `for_writer/analysis_notes.md` | **Methods 段写作要点**：模型设定、FDR 校正、异质性设计、Sinclair 系数来源。 |

---

## 5. 统计量到论文段落的对照表

论文各段落需要的数字，绝大多数都能在 `{rel_cons}` 里用 Ctrl+F 找到。

| 论文段落 | 去哪找 |
|---|---|
| Abstract / Results 第一段（样本）| `for_writer/key_numbers.md` 首两条（2186 obs / 730 athletes）+ Table 1 |
| Results § Overall jet lag effect（整体 LRT，headline）| `{rel_cons}` 第 `2.2` 节 LRT-B（主分析，控制旅行时长） |
| Results § Overall jet lag effect（事后比较具体数字）| `tables/table2_posthoc_fdr.csv` + Fig 1（均来自主分析 `lmm_jetlag_travel_control`） |
| Results § Consistency without travel adjustment（一致性对照）| `{rel_cons}` 第 `1.2` 节（补充分析，不控制旅行时长，仅作对照，不放正文表格） |
| Results § Homogeneity across subgroups | `tables/table3_heterogeneity_lrt.csv` + Fig 2；细节在 `{rel_cons}` 第 4 节 |
| Methods § Statistical analysis | `for_writer/analysis_notes.md` |
| Methods § Variables | `for_writer/data_dictionary.md` |

---

## 6. 不要做的事

- 不要引用 `lmm_jetlag/` 目录（旧的 2308 样本版本）。**主分析口径为 `lmm_jetlag_travel_control/`（2186 样本，控制旅行时长）**；
  `lmm_jetlag_v2/`（同样 2186 样本，不控制旅行时长）仅作一致性对照，**不进正文 Table/Fig**，只在 Results 里补一句"结果方向和显著性一致"。
- 主模型与异质性分析现已统一使用 **2186 obs / 730 athletes**；不得再引用旧版N=2184输出。
- Table 3 故意不含 N 列，保持与正文 2186 的一致性。
- `heterogeneity_results/` 下 `*_20260416_*` 等旧时间戳目录是历史版本，**只看最新时间戳**；合并 txt 与表格已自动选最新。
"""
    (fw_dir / "USAGE_MANIFEST.md").write_text(md, encoding="utf-8")


def write_writer_prompt(fw_dir: Path, consolidated_path: Path) -> None:
    """生成可直接交给写作者 / AI 的 Prompt。"""
    paper_root = fw_dir.parent
    rel_cons = _rel(consolidated_path, paper_root)

    prompt = f"""# Writer Prompt —— 论文代写提示 (给写作者 / AI)

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

★ 主模型（正文 headline，控制旅行时长）：

    Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir) + TravelDuration_c + (1 | Athlete_ID)   # REML

补充对照模型（同一 2186 样本，不含 TravelDuration_c，仅用于一致性检验，不进正文表格）：

    Sinclair_Total ~ C(Sex) + C(JetLagGroup_dir) + (1 | Athlete_ID)   # REML

- 10 个非参照组 × vs Minimal 的对比 p 值用 **Benjamini–Hochberg FDR** 校正（主模型系数，Table 2）；
- 整体检验：控制旅行时长后 JetLag 联合 LRT-B（headline）；
- 异质性：分层模型 (Design A) + 交互项 LRT (Design B, ML 拟合)；
- ICC 报告自 REML 方差分量。

## 4. 主要结论 (摘要 3–4 句 + Results 首段)

1. 控制旅行时长后，方向性时差对 Sinclair 仍有显著整体影响 —— 主模型 LRT-B 见 `{rel_cons}` § 2.2；具体大小见 Fig 1 / Table 2。
2. 不控制旅行时长时效应方向与显著性模式一致（仅量级更大）—— 补充对照见 `{rel_cons}` § 1.2，用于说明效应不能被旅行疲劳完全解释。
3. 效应在性别 / 年龄 / 经验 / 体重级别亚组间**高度同质** (全部 LRT p > 0.2) —— Table 3 + Fig 2。
4. 连续调节敏感性分析同样未发现显著交互 —— Supp Table S1 + Fig 3。

## 5. 逐段提示 (Paper skeleton)

### Title
示例：*Directional Jet Lag and Weightlifting Performance in Elite Athletes: A Mixed-Effects Analysis of 2186 International Competition Entries*

### Abstract (250 字内，结构化：Purpose / Methods / Results / Conclusion)

- Purpose：评估方向性时差对 Sinclair 总分的影响及稳健性 / 同质性；
- Methods：引用 Sample + 模型 + FDR + 稳健 + 异质设计；
- Results：填入 `key_numbers.md` 中 **主模型 LRT-B**（headline）与 **ICC**；引用 Fig 1 显著组数量、方向；
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
- 3.2 Overall jet lag effect (travel-controlled, primary) — 报告主模型 LRT-B（括号带 LRT-A）；解读 Fig 1；
  在文中列出显著组 β(95%CI, p_fdr) 的 1–3 条典型值（从 Table 2 挑，均来自主模型）。
- 3.3 Consistency without travel adjustment — 一句话引用补充对照（`{rel_cons}` § 1.2）方向/显著性一致，量级更大。
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

**样本 / ICC / 主模型 LRT-B（headline）** → `for_writer/key_numbers.md`（已自动抽取）。
**具体某组 β / CI / p_fdr（主模型）** → `tables/table2_posthoc_fdr.csv`。
**异质性每行 LR / df / p** → `tables/table3_heterogeneity_lrt.csv`。
**主模型 LRT-A / LRT-B** → `{rel_cons}` § 2.2；**补充对照（不控制旅行时长）** → `{rel_cons}` § 1.2。
**任何其它细节** → 检索 `{rel_cons}`；它按 1.x（补充）/ 2.x（主分析）/ 3.x / 4.x 顺序组织，节标题明确。

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
"""
    (fw_dir / "WRITER_PROMPT.md").write_text(prompt, encoding="utf-8")
