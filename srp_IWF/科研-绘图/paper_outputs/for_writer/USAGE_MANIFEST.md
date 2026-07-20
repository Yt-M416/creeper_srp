# 论文写作资料包 —— 使用清单 (USAGE MANIFEST)

> 自动生成于 2026-07-20 16:38。本文件告诉你：**每个文件是什么、论文哪一部分会用到、怎么填数字**。
> 所有路径均相对于 `科研-绘图/paper_outputs/`。

---

## 0. 先看这两份

| 文件 | 用途 |
|---|---|
| `for_writer/WRITER_PROMPT.md` | **给写作者/AI 的写作提示**：包含研究背景、样本、主要结论、逐段提示与风格要求。按它的框架写即可。 |
| `for_writer/consolidated_results.txt` | **合并后的文字型统计结果**：主分析 LRT / 事后比较 / 旅行稳健性 / 异质性 LRT 全部在同一文件中，分节明确。**引用统计量时查这里。** |

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
| `figures/fig3_sensitivity.*` | **Figure 3** 或 **Supp Fig S1** | 两栏：左 (A) 方向性时差联合 LRT（主模型 + 控制旅行 LRT-B）；右 (B) 连续协变量×有向时差交互 LRT；横轴均为 $-\log_10(p)$。 |
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

论文各段落需要的数字，绝大多数都能在 `for_writer/consolidated_results.txt` 里用 Ctrl+F 找到。

| 论文段落 | 去哪找 |
|---|---|
| Abstract / Results 第一段（样本）| `for_writer/key_numbers.md` 首两条（2186 obs / 730 athletes）+ Table 1 |
| Results § Overall jet lag effect（整体 LRT）| `for_writer/consolidated_results.txt` 第 `1.2` 节 "Likelihood Ratio Test — Model B" |
| Results § Overall jet lag effect（事后比较具体数字）| `tables/table2_posthoc_fdr.csv` + Fig 1 |
| Results § Travel fatigue robustness | `for_writer/consolidated_results.txt` 第 `2.2` 节 LRT-A / LRT-B |
| Results § Homogeneity across subgroups | `tables/table3_heterogeneity_lrt.csv` + Fig 2；细节在 `for_writer/consolidated_results.txt` 第 4 节 |
| Methods § Statistical analysis | `for_writer/analysis_notes.md` |
| Methods § Variables | `for_writer/data_dictionary.md` |

---

## 6. 不要做的事

- 不要引用 `lmm_jetlag/` 目录（旧的 2308 样本版本）。主分析口径为 `lmm_jetlag_v2/`（2186 样本），已整合到本资料包。
- 不要在正文里单独列 "异质性分析样本 = 2184" —— 与主样本 2186 相差 2 条（技术性去重），可统一口径报告为 **"2186 obs / 730 athletes"**。
- Table 3 故意不含 N 列，保持与正文 2186 的一致性。
- `heterogeneity_results/` 下 `*_20260416_*` 等旧时间戳目录是历史版本，**只看最新时间戳**；合并 txt 与表格已自动选最新。
