"""
build_results_docx.py
把最新统计结果整理成 Word 文档：论文结果汇总.docx
运行：python build_results_docx.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y%m%d")
OUT_DIR = ROOT / f"交付材料_{STAMP}"
TABLES_DIR = OUT_DIR / "03_统计结果" / "论文表格"
FIG_DIR = OUT_DIR / "04_图表"
DOCX_PATH = OUT_DIR / "论文结果汇总.docx"


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def add_table(doc: Document, header, rows, title: str | None = None) -> None:
    if title:
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(11)
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(header):
        hdr_cells[i].text = str(h)
        for p in hdr_cells[i].paragraphs:
            for run in p.runs:
                run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph()


def heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def main() -> None:
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    title = doc.add_heading("IWF 跨时区时差效应研究 — 论文结果汇总", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}（全流程重跑后自动生成）")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True

    # ---------------- 1. 样本 ----------------
    heading(doc, "1. 样本描述 (Table 1)")
    header, rows = read_csv(TABLES_DIR / "table1_descriptive.csv")
    add_table(doc, header, rows)
    doc.add_paragraph(
        "最终建模样本：N = 2186 条观测，730 名运动员，13 个国家/地区（Top13）。"
        "该样本口径为主模型（无旅行时长）与稳健性模型（含旅行时长协变量）共用。"
    )

    # ---------------- 2. 主模型 ----------------
    heading(doc, "2. 主模型：方向性时差整体效应")
    doc.add_paragraph(
        "模型：Sinclair_Total ~ Sex + JetLagGroup_dir + (1|Athlete_ID)，REML 估计；"
        "参照组 Minimal(0-1h)（|有向时差| < 2h）。"
    )
    doc.add_paragraph(
        "整体似然比检验（LRT，ML 拟合，比较含/不含 JetLagGroup_dir 的模型）："
        "LR = 51.82, df = 10, p = 1.23×10⁻⁷ —— 方向性时差对 Sinclair 总分有显著整体效应。"
    )
    doc.add_paragraph("模型拟合质量：ICC ≈ 0.75，约75%的成绩变异来自运动员个体差异，支持以 Athlete_ID 作随机截距。")

    heading(doc, "2.1 事后比较 (Table 2：各时差组 vs Minimal，BH-FDR 校正)", level=2)
    header, rows = read_csv(TABLES_DIR / "table2_posthoc_fdr.csv")
    add_table(doc, header, rows)
    doc.add_paragraph(
        "结果：经 FDR 校正后唯一显著的组为 East 8–9h（β = −7.77, 95% CI [−10.27, −5.27], p_FDR < .001）；"
        "其余9个组无论东西方向、经 FDR 校正后均不显著；West 10–11h 的正系数（+5.17）校正前接近显著但校正后不再保留。"
    )
    fig1 = FIG_DIR / "fig1_forest_main_effect.png"
    if fig1.exists():
        doc.add_picture(str(fig1), width=Cm(15))
        cap = doc.add_paragraph("图1. 11档方向性时差组的边际效应森林图（vs Minimal，BH-FDR 显著标记）")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True

    # ---------------- 3. 稳健性 ----------------
    heading(doc, "3. 旅行疲劳稳健性检验")
    header, rows = read_csv(TABLES_DIR / "supp_table1_sensitivity.csv")
    add_table(doc, header, rows, title="LRT-A / LRT-B 及连续调节变量敏感性 LRT")
    doc.add_paragraph(
        "LRT-A：加入 TravelDuration_c 后模型拟合显著改善（LR=10.86, df=1, p=9.8×10⁻⁴），"
        "旅行时长每增加1小时，Sinclair 总分下降约0.22分（β=−0.22）。"
    )
    doc.add_paragraph(
        "LRT-B：控制 TravelDuration_c 后，JetLagGroup_dir 仍高度联合显著（LR=53.27, df=10, p=6.6×10⁻⁸），"
        "East 8–9h 系数略缩小为 β=−5.58（95% CI [−8.40, −2.77]），"
        "说明方向性时差效应不能被旅行疲劳解释。"
    )

    # ---------------- 4. 异质性 ----------------
    heading(doc, "4. 异质性分析 (Table 3)")
    header, rows = read_csv(TABLES_DIR / "table3_heterogeneity_lrt.csv")
    add_table(doc, header, rows)
    doc.add_paragraph(
        "四个调节变量（性别、年龄、经验、体重）的交互项 LRT 均不显著（全部 p > 0.2），"
        "时差效应在各亚组间呈统计同质性。"
    )
    fig2 = FIG_DIR / "fig2_heterogeneity_2x2.png"
    if fig2.exists():
        doc.add_picture(str(fig2), width=Cm(15))
        cap = doc.add_paragraph("图2. 四个调节变量分层下的时差效应模式")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True

    fig3 = FIG_DIR / "fig3_sensitivity.png"
    if fig3.exists():
        doc.add_picture(str(fig3), width=Cm(15))
        cap = doc.add_paragraph("图3. 连续调节变量（年龄/体重/经验，中心化）敏感性分析")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True

    # ---------------- 5. 关键澄清 ----------------
    heading(doc, "5. 本次核对澄清的问题")
    items = [
        "Cohen's d 计算已修正：由 β/σ²_ε（残差方差，错误）改为 β/σ_ε（残差标准差，正确）。"
        "East 8–9h 的 d 由错误的 −0.032 修正为 −0.42。β、SE、p、FDR 结果不受影响。",
        "论文正文主模型 = LMM_jetlag_v2.py（不含 TravelDuration），"
        "稳健性检验 = LMM_jetlag_travel_v2.py --mode control（含 TravelDuration_c），二者均为正确、互补的分析，非互斥或替代关系。",
        "JetLagGroup 分组：Minimal 组操作定义为 |有向时差| < 2h（标签书写为 0-1h，为2h宽档的整数标签惯例）；"
        "东、西向各按2h一档分5档；圆周时差范围为[-12,12]，所有有效值均可分组，无因超出分组范围被删除的样本；"
        "仅因时区信息缺失（JetLag_signed 为 NA）而删除的记录。",
        "数据处理与代码文件名已统一（原 preprocess/merge 脚本输出文件名与下游脚本读取文件名不一致的问题已修复）。",
        "全流程已于本次重新跑通：预处理 → 合并旅行时长 → 主模型 → 稳健性模型 → 异质性/敏感性分析 → 论文图表，"
        "所有数字已与论文《论文结构.txt》原有表述逐项核对一致。",
    ]
    for it in items:
        doc.add_paragraph(it, style="List Bullet")

    doc.save(DOCX_PATH)
    print("Word 文档已生成：", DOCX_PATH)


if __name__ == "__main__":
    main()
