"""Build an AWS-CN blog .docx from a markdown source, following the
"AWS-CN 博客撰写规范" conventions used by the published EKS GPU blog:

- 标题1 = blog title (once); 标题2 = chapters; 标题3 / 标题4 = sub-headings
- code blocks = 1x1 table, gray F2F2F2, Consolas 10pt, first line = language name
- inline code = Consolas 10pt with F2F2F2 shading
- tables = dark-blue 1F4E79 header (white bold) + EAF1F8 zebra rows
- images = own paragraph, 6in wide, caption below in 【图 N. ...】 (italic, 595959, 10pt)
- images are embedded as SVG (asvg:svgBlip) with a PNG fallback rendered by rsvg-convert

usage: build_docx.py <template.docx> <source.md> <out.docx>
"""
import copy
import os
import re
import subprocess
import sys

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

TEMPLATE, SRC, OUT = sys.argv[1:4]
SRC_DIR = os.path.dirname(os.path.abspath(SRC))
PNG_DIR = os.path.join(os.path.dirname(os.path.abspath(OUT)), "img-png")
os.makedirs(PNG_DIR, exist_ok=True)

doc = Document(TEMPLATE)
body = doc.element.body
for child in list(body):
    if child.tag != qn("w:sectPr"):
        body.remove(child)
# drop the template's images / hyperlinks that the cleared body no longer references
for rid, rel in list(doc.part.rels.items()):
    if rel.reltype in (RT.IMAGE, RT.HYPERLINK):
        doc.part.drop_rel(rid)

STYLE = {1: "heading 1", 2: "heading 2", 3: "heading 3", 4: "heading 4"}
LANG_LABEL = {"bash": "Bash", "sh": "Bash", "python": "Python", "yaml": "YAML", "hcl": "HCL", "json": "JSON"}
SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
ASVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


# ---------- inline runs ----------
INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def shade(rpr_or_tcpr, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    rpr_or_tcpr.append(shd)


def code_run(par, text):
    r = par.add_run(text)
    r.font.name = "Consolas"
    r._element.rPr.rFonts.set(qn("w:cs"), "Consolas")
    r.font.size = Pt(10)
    shade(r._element.get_or_add_rPr(), "F2F2F2")
    return r


def add_hyperlink(par, text, url, bold=False):
    rid = par.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(color)
    rpr.append(u)
    if bold:
        rpr.append(OxmlElement("w:b"))
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    par._p.append(h)


def add_inline(par, text, bold=False, color=None):
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            add_inline(par, tok[2:-2], bold=True, color=color)
        elif tok.startswith("`") and tok.endswith("`"):
            r = code_run(par, tok[1:-1])
            if bold:
                r.bold = True
        elif tok.startswith("[") and "](" in tok:
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            add_hyperlink(par, m.group(1), m.group(2), bold=bold)
        else:
            r = par.add_run(tok)
            if bold:
                r.bold = True
            if color:
                r.font.color.rgb = RGBColor.from_string(color)


# ---------- blocks ----------
def heading(level, text):
    p = doc.add_paragraph(style=STYLE[level])
    add_inline(p, text)


def paragraph(text):
    p = doc.add_paragraph()
    add_inline(p, text)


def bullet(text, level):
    p = doc.add_paragraph(style="List Bullet 2" if level > 0 else "List Bullet")
    add_inline(p, text)


LIST_STATE = {"in_list": False, "num_id": None}


def new_num_instance():
    """Each separate numbered list gets its own w:num (same abstractNum, restart at 1)."""
    numbering = doc.part.numbering_part.numbering_definitions._numbering
    style_num_id = doc.styles["List Number"].element.pPr.numPr.numId.val
    abs_id = numbering.num_having_numId(style_num_id).abstractNumId.val
    num = numbering.add_num(abs_id)
    num.add_lvlOverride(ilvl=0).add_startOverride(1)
    return num.numId


def numbered(text):
    if not LIST_STATE["in_list"]:
        LIST_STATE["num_id"] = new_num_instance()
        LIST_STATE["in_list"] = True
    p = doc.add_paragraph(style="List Number")
    numpr = p._p.get_or_add_pPr().get_or_add_numPr()
    numpr.get_or_add_ilvl().val = 0
    numpr.get_or_add_numId().val = LIST_STATE["num_id"]
    add_inline(p, text)


def quote(text):
    p = doc.add_paragraph(style="Quote")
    add_inline(p, text)


def code_block(lang, lines):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))  # 代码块不跨页断开
    cell = tbl.rows[0].cells[0]
    shade(cell._tc.get_or_add_tcPr(), "F2F2F2")
    label = LANG_LABEL.get(lang.lower()) if lang else None
    all_lines = ([label] if label else []) + lines
    first = True
    for line in all_lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        pf = p.paragraph_format
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        r = p.add_run(line)
        r.font.name = "Consolas"
        r._element.rPr.rFonts.set(qn("w:cs"), "Consolas")
        r.font.size = Pt(10)
    doc.add_paragraph()  # 表格后留一行，避免与下一段粘连


def table(rows):
    header, data = rows[0], rows[1:]
    tbl = doc.add_table(rows=1 + len(data), cols=len(header))
    tbl.style = doc.styles["Table Grid"]
    tbl.autofit = False
    # column widths proportional to content (CJK counts double), page text width = 6in
    def disp(t):
        t = re.sub(r"\*\*|`|\[([^\]]+)\]\([^)]+\)", lambda m: m.group(1) or "", t)
        return sum(2 if ord(ch) > 0x2E7F else 1 for ch in t)
    weights = []
    for ci in range(len(header)):
        w = max(disp(r[ci]) if ci < len(r) else 0 for r in rows)
        weights.append(min(max(w, 8), 70))
    total, minw = 8640, 1300
    widths = [max(minw, int(total * w / sum(weights))) for w in weights]
    scale = total / sum(widths)
    widths = [int(w * scale) for w in widths]
    for row in tbl.rows:
        for ci, cell in enumerate(row.cells):
            cell.width = Twips(widths[ci])
    # repeat header row on page breaks
    trpr = tbl.rows[0]._tr.get_or_add_trPr()
    trpr.append(OxmlElement("w:tblHeader"))
    for ci, txt in enumerate(header):
        cell = tbl.rows[0].cells[ci]
        shade(cell._tc.get_or_add_tcPr(), "1F4E79")
        p = cell.paragraphs[0]
        add_inline(p, txt, bold=True, color="FFFFFF")
    for ri, row in enumerate(data, start=1):
        for ci in range(len(header)):
            cell = tbl.rows[ri].cells[ci]
            if ri % 2 == 0:
                shade(cell._tc.get_or_add_tcPr(), "EAF1F8")
            add_inline(cell.paragraphs[0], row[ci] if ci < len(row) else "")
    doc.add_paragraph()


_svg_count = 0


def image(caption, rel_path):
    global _svg_count
    svg_path = os.path.join(SRC_DIR, rel_path)
    png_path = os.path.join(PNG_DIR, os.path.splitext(os.path.basename(svg_path))[0] + ".png")
    subprocess.run(["rsvg-convert", "-w", "1760", svg_path, "-o", png_path], check=True)

    p = doc.add_paragraph()
    run = p.add_run()
    run.add_picture(png_path, width=Inches(6))
    pic_name = os.path.basename(svg_path).replace(".svg", ".png")

    # attach the SVG part and reference it from the blip (PNG stays as fallback)
    _svg_count += 1
    with open(svg_path, "rb") as f:
        svg_part = Part(PackURI(f"/word/media/fig_svg{_svg_count}.svg"), "image/svg+xml", f.read(), doc.part.package)
    svg_rid = doc.part.relate_to(svg_part, RT.IMAGE)
    blip = run._element.find(".//" + qn("a:blip"))
    ext_lst = OxmlElement("a:extLst")
    ext = OxmlElement("a:ext")
    ext.set("uri", SVG_EXT_URI)
    from lxml import etree
    svg_blip = etree.SubElement(ext, f"{{{ASVG_NS}}}svgBlip", nsmap={"asvg": ASVG_NS})
    svg_blip.set(qn("r:embed"), svg_rid)
    ext_lst.append(ext)
    blip.append(ext_lst)
    for c in run._element.iter(qn("pic:cNvPr")):
        c.set("name", pic_name)
    docpr = run._element.find(".//" + qn("wp:docPr"))
    docpr.set("descr", caption)

    cap = doc.add_paragraph()
    r = cap.add_run(f"【{caption}】")
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor.from_string("595959")


# ---------- markdown parsing ----------
def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


lines = open(SRC, encoding="utf-8").read().splitlines()
i = 0
title = None
while i < len(lines):
    line = lines[i]
    s = line.strip()
    if not s or s == "---":
        i += 1
        continue
    if not re.match(r"^\d+\.\s+", s):
        LIST_STATE["in_list"] = False
    m = re.match(r"^(#{1,4})\s+(.*)$", s)
    if m:
        level = len(m.group(1))
        if level == 1:
            title = m.group(2)
        heading(level, m.group(2))
        i += 1
        continue
    if s.startswith("```"):
        lang = s[3:].strip()
        i += 1
        block = []
        while not lines[i].strip().startswith("```"):
            block.append(lines[i])
            i += 1
        code_block(lang, block)
        i += 1
        continue
    if s.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            row = split_row(lines[i])
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in row):
                rows.append(row)
            i += 1
        table(rows)
        continue
    m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
    if m:
        image(m.group(1), m.group(2))
        i += 1
        continue
    m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
    if m:
        bullet(m.group(2), 1 if len(m.group(1)) >= 2 else 0)
        i += 1
        continue
    m = re.match(r"^\d+\.\s+(.*)$", s)
    if m:
        numbered(m.group(1))
        i += 1
        continue
    if s.startswith(">"):
        quote(s.lstrip("> ").strip())
        i += 1
        continue
    paragraph(s)
    i += 1

cp = doc.core_properties
cp.title = title or ""
cp.author = "Kevin Zhao"
cp.subject = "Qwen3-30B-A3B MoE inference on AWS Trainium2"
cp.keywords = "AWS Trainium2, Amazon EKS, Neuron, NKI, MoE, vLLM, Qwen3, DeepSeek-V3"
doc.save(OUT)
print("saved", OUT)
