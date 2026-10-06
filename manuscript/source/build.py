"""Markdown -> pandoc docx -> journal-style post-processing.
Usage: python build.py <in.md> <out.docx> [--supp] [--layout A|B]"""
import re, sys, subprocess, copy, os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

args = sys.argv[1:]
src, out = args[0], args[1]
SUPP = '--supp' in args
LAYOUT = args[args.index('--layout') + 1] if '--layout' in args else 'A'

FONT = 'Times New Roman'
if LAYOUT == 'A':      # journal-like single column
    BODY_PT, LINE, INDENT, PAR_AFTER, MARG = 11, 1.2, 0.6, 0, 2.3
else:                   # submission style
    BODY_PT, LINE, INDENT, PAR_AFTER, MARG = 12, 1.5, 0.0, 6, 2.5
PAGE_W, PAGE_H = 21.0, 29.7
TEXT_W_CM = PAGE_W - 2 * MARG


# ----------------------------------------------------------------------------- 1. markdown pre-processing
def preprocess(md):
    lines = md.split('\n')
    outl = []
    for ln in lines:
        m = re.match(r'^@@(\w+)\s?(.*)$', ln)
        if not m:
            outl.append(ln); continue
        tag, txt = m.group(1), m.group(2)
        style = {'TITLE': 'Title', 'AUTHORS': 'Author', 'AFFIL': 'Affiliation', 'HEAD': 'Front Heading',
                 'HL': 'Highlight', 'ABS': 'Abstract', 'KEY': 'Keywords', 'TOC': 'Contents Entry'}.get(tag)
        if tag == 'REFS':
            refs = open(os.path.join(HERE, 'references.md')).read().strip().split('\n\n')
            for r in refs:
                outl += ['::: {custom-style="Reference"}', r.strip(), ':::', '']
            continue
        if tag == 'ALG':
            alg = open(os.path.join(HERE, 'algorithm.md')).read().strip().split('\n')
            for a in alg:
                if a.strip():
                    outl += ['::: {custom-style="Algorithm"}', a.strip(), ':::', '']
            continue
        outl += [f'::: {{custom-style="{style}"}}', txt, ':::']
    return '\n'.join(outl)


md = open(src).read()
md = preprocess(md)
tmp_md = out + '.tmp.md'
open(tmp_md, 'w').write(md)
raw = out + '.raw.docx'
subprocess.run(['pandoc', tmp_md, '-f', 'markdown+superscript+subscript+tex_math_dollars+pipe_tables+fenced_divs',
                '-t', 'docx', '-o', raw, '--resource-path', os.path.join(HERE, '..')], check=True)

# ----------------------------------------------------------------------------- 2. post-processing
doc = Document(raw)
styles = doc.styles


def set_font(style_or_run_font, size=None, bold=None, italic=None, color=None):
    f = style_or_run_font
    f.name = FONT
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    if color is not None: f.color.rgb = RGBColor.from_string(color)


def set_east_asia(style):
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.insert(0, rf)
    for a in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
        rf.set(qn(a), FONT)
    for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]


def get_style(name, base='Normal'):
    try:
        return styles[name]
    except KeyError:
        s = styles.add_style(name, 1)
        s.base_style = styles[base]
        return s


def para_fmt(style, align=None, before=0, after=0, line=None, indent=None, left=None, keep_next=None, hanging=None):
    pf = style.paragraph_format
    if align is not None: pf.alignment = align
    pf.space_before = Pt(before); pf.space_after = Pt(after)
    if line is not None:
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE; pf.line_spacing = line
    if indent is not None: pf.first_line_indent = Cm(indent)
    if left is not None: pf.left_indent = Cm(left)
    if hanging is not None:
        pf.left_indent = Cm(hanging); pf.first_line_indent = Cm(-hanging)
    if keep_next is not None: pf.keep_with_next = keep_next


J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
for nm in ['Normal', 'Body Text', 'First Paragraph', 'Compact']:
    s = get_style(nm); set_font(s.font, BODY_PT, color='000000'); set_east_asia(s)
    para_fmt(s, J, 0, PAR_AFTER, LINE, INDENT)
para_fmt(get_style('First Paragraph'), J, 0, PAR_AFTER, LINE, INDENT)
para_fmt(get_style('Compact'), L, 0, 0, 1.0, 0)

s = get_style('Title'); set_font(s.font, 16 if LAYOUT == 'A' else 16, True, False, '000000'); set_east_asia(s)
para_fmt(s, C, 0, 10, 1.1, 0)
s = get_style('Author'); set_font(s.font, 11.5, False, False, '000000'); set_east_asia(s); para_fmt(s, C, 0, 4, 1.1, 0)
s = get_style('Affiliation'); set_font(s.font, 9.5, False, True, '000000'); set_east_asia(s); para_fmt(s, C, 0, 2, 1.1, 0)
s = get_style('Front Heading'); set_font(s.font, 11, True, False, '000000'); set_east_asia(s); para_fmt(s, L, 12, 4, 1.1, 0, keep_next=True)
s = get_style('Highlight'); set_font(s.font, BODY_PT - 0.5, False, False, '000000'); set_east_asia(s); para_fmt(s, L, 0, 2, 1.15, 0)
s.paragraph_format.left_indent = Cm(0.5); s.paragraph_format.first_line_indent = Cm(-0.4)
s = get_style('Abstract'); set_font(s.font, BODY_PT - 0.5, False, False, '000000'); set_east_asia(s); para_fmt(s, J, 0, 4, LINE * 0.97, 0)
s = get_style('Keywords'); set_font(s.font, BODY_PT - 0.5, False, False, '000000'); set_east_asia(s); para_fmt(s, J, 2, 6, 1.15, 0)
s = get_style('Contents Entry'); set_font(s.font, BODY_PT, False, False, '000000'); set_east_asia(s); para_fmt(s, L, 0, 2, 1.15, 0)
for nm, sz, b, it, bf, af in [('Heading 1', 12, True, False, 14, 6), ('Heading 2', 11, True, False, 10, 4),
                               ('Heading 3', 11, False, True, 8, 3)]:
    s = get_style(nm); set_font(s.font, sz, b, it, '000000'); set_east_asia(s)
    para_fmt(s, L, bf, af, 1.1, 0, keep_next=True)
s = get_style('Image Caption'); set_font(s.font, 9.5, False, False, '000000'); set_east_asia(s); para_fmt(s, J, 3, 10, 1.1, 0)
s = get_style('Table Caption'); set_font(s.font, 9.5, False, False, '000000'); set_east_asia(s); para_fmt(s, L, 10, 3, 1.1, 0, keep_next=True)
s = get_style('Captioned Figure'); para_fmt(s, C, 8, 0, 1.0, 0, keep_next=True)
s = get_style('Figure'); para_fmt(s, C, 8, 0, 1.0, 0, keep_next=True)
s = get_style('Table Note'); set_font(s.font, 8.5, False, False, '000000'); set_east_asia(s); para_fmt(s, J, 2, 8, 1.05, 0)
s = get_style('Table Text'); set_font(s.font, 8.5, False, False, '000000'); set_east_asia(s); para_fmt(s, L, 1, 1, 1.0, 0)
s = get_style('Reference'); set_font(s.font, 9, False, False, '000000'); set_east_asia(s); para_fmt(s, J, 0, 2, 1.1, hanging=0.75)
s = get_style('Equation'); para_fmt(s, L, 4, 4, 1.0, 0)
s = get_style('Algorithm'); set_font(s.font, 9.5, False, False, '000000'); set_east_asia(s); para_fmt(s, L, 0, 1, 1.1, 0)
s.paragraph_format.left_indent = Cm(0.3)

# page setup
for sec in doc.sections:
    sec.page_width, sec.page_height = Cm(PAGE_W), Cm(PAGE_H)
    sec.left_margin = sec.right_margin = Cm(MARG)
    sec.top_margin = Cm(2.4); sec.bottom_margin = Cm(2.2)
    sec.header_distance = Cm(1.2); sec.footer_distance = Cm(1.1); sec.gutter = Cm(0)
    # page number
    fp = sec.footer.paragraphs[0]; fp.alignment = C
    r = fp.add_run()
    for t, txt in [('begin', None), (None, ' PAGE '), ('end', None)]:
        if t:
            e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), t); r._r.append(e)
        else:
            e = OxmlElement('w:instrText'); e.set(qn('xml:space'), 'preserve'); e.text = txt; r._r.append(e)
    r.font.size = Pt(9); r.font.name = FONT

body = doc.element.body
TW = int(TEXT_W_CM / 2.54 * 1440)


def ptext(p):
    return ''.join(t.text or '' for t in p.iter(qn('w:t')))


def set_pstyle(p, name):
    pPr = p.find(qn('w:pPr'))
    if pPr is None:
        pPr = OxmlElement('w:pPr'); p.insert(0, pPr)
    ps = pPr.find(qn('w:pStyle'))
    if ps is None:
        ps = OxmlElement('w:pStyle'); pPr.insert(0, ps)
    ps.set(qn('w:val'), styles[name].style_id)


# --- equations: merge {{EQ:n}} marker into previous display-math paragraph
paras = list(body.iter(qn('w:p')))
for p in paras:
    t = ptext(p).strip()
    m = re.fullmatch(r'\{\{EQ:(S?\d+)\}\}', t)
    if not m:
        continue
    num = m.group(1)
    prev = p.getprevious()
    while prev is not None and prev.tag != qn('w:p'):
        prev = prev.getprevious()
    omp = prev.find(f'{{{M_NS}}}oMathPara') if prev is not None else None
    if omp is None:
        print('WARN: no math before', num); p.getparent().remove(p); continue
    omaths = omp.findall(f'{{{M_NS}}}oMath')
    newp = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr'); newp.append(pPr)
    ps = OxmlElement('w:pStyle'); ps.set(qn('w:val'), styles['Equation'].style_id); pPr.append(ps)
    tabs = OxmlElement('w:tabs')
    for val, pos in [('center', TW // 2), ('right', TW)]:
        tb = OxmlElement('w:tab'); tb.set(qn('w:val'), val); tb.set(qn('w:pos'), str(pos)); tabs.append(tb)
    pPr.append(tabs)
    def run_with(tab=False, text=None):
        r = OxmlElement('w:r'); rpr = OxmlElement('w:rPr')
        rf = OxmlElement('w:rFonts'); [rf.set(qn(a), FONT) for a in ('w:ascii', 'w:hAnsi', 'w:cs')]; rpr.append(rf)
        sz = OxmlElement('w:sz'); sz.set(qn('w:val'), str(int(BODY_PT * 2))); rpr.append(sz); r.append(rpr)
        if tab: r.append(OxmlElement('w:tab'))
        if text:
            tt = OxmlElement('w:t'); tt.text = text; r.append(tt)
        return r
    newp.append(run_with(tab=True))
    for om in omaths:
        newp.append(om)
    newp.append(run_with(tab=True, text=f'({num})'))
    prev.addprevious(newp)
    prev.getparent().remove(prev)
    p.getparent().remove(p)

# --- tables: three-line style, captions, notes
tbl_count = 0
for tbl in list(body.iter(qn('w:tbl'))):
    tblPr = tbl.find(qn('w:tblPr'))
    for tag in ('w:tblStyle', 'w:tblBorders', 'w:tblW', 'w:tblLayout', 'w:jc', 'w:tblLook', 'w:tblCellMar'):
        e = tblPr.find(qn(tag))
        if e is not None: tblPr.remove(e)
    w = OxmlElement('w:tblW'); w.set(qn('w:type'), 'pct'); w.set(qn('w:w'), '5000'); tblPr.append(w)
    jc = OxmlElement('w:jc'); jc.set(qn('w:val'), 'center'); tblPr.append(jc)
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
    mar = OxmlElement('w:tblCellMar')
    for side, v in [('top', 15), ('left', 60), ('bottom', 15), ('right', 60)]:
        e = OxmlElement(f'w:{side}'); e.set(qn('w:w'), str(v)); e.set(qn('w:type'), 'dxa'); mar.append(e)
    tblPr.append(mar)
    rows = tbl.findall(qn('w:tr'))
    for ri, tr in enumerate(rows):
        trPr = tr.find(qn('w:trPr'))
        if trPr is None:
            trPr = OxmlElement('w:trPr'); tr.insert(0, trPr)
        cs = OxmlElement('w:cantSplit'); trPr.append(cs)
        if ri == 0:
            th = OxmlElement('w:tblHeader'); trPr.append(th)
        for tc in tr.findall(qn('w:tc')):
            tcPr = tc.find(qn('w:tcPr'))
            if tcPr is None:
                tcPr = OxmlElement('w:tcPr'); tc.insert(0, tcPr)
            b = OxmlElement('w:tcBorders')
            def border(side, sz):
                e = OxmlElement(f'w:{side}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), str(sz))
                e.set(qn('w:space'), '0'); e.set(qn('w:color'), '000000'); b.append(e)
            if ri == 0:
                border('top', 12); border('bottom', 6)
            if ri == len(rows) - 1:
                border('bottom', 12)
            tcPr.append(b)
            va = OxmlElement('w:vAlign'); va.set(qn('w:val'), 'center'); tcPr.append(va)
            for p in tc.findall(qn('w:p')):
                pPr = p.find(qn('w:pPr'))
                align = None
                if pPr is not None and pPr.find(qn('w:jc')) is not None:
                    align = pPr.find(qn('w:jc')).get(qn('w:val'))
                set_pstyle(p, 'Table Text')
                pPr = p.find(qn('w:pPr'))
                if align:
                    jc = pPr.find(qn('w:jc'))
                    if jc is None:
                        jc = OxmlElement('w:jc'); pPr.append(jc)
                    jc.set(qn('w:val'), align)
                if ri == 0:
                    for r in p.iter(qn('w:r')):
                        rpr = r.find(qn('w:rPr'))
                        if rpr is None:
                            rpr = OxmlElement('w:rPr'); r.insert(0, rpr)
                        if rpr.find(qn('w:b')) is None:
                            rpr.append(OxmlElement('w:b'))
                for r in p.iter(qn('w:r')):
                    rpr = r.find(qn('w:rPr'))
                    if rpr is None:
                        rpr = OxmlElement('w:rPr'); r.insert(0, rpr)
                    nw = OxmlElement('w:noProof'); rpr.append(nw)
    # caption: pandoc puts it before table as Table Caption paragraph
    prev = tbl.getprevious()
    if prev is not None and prev.tag == qn('w:p') and ptext(prev).strip() == 'Nomenclature':
        continue
    if prev is not None and prev.tag == qn('w:p'):
        txt = ptext(prev)
        tbl_count += 1
        # prepend bold "Table n." unless supplementary label already present
        m = re.match(r'^(Supplementary Table S\d+\.)\s*(.*)$', txt)
        for r in list(prev.findall(qn('w:r'))) + list(prev.iter(qn('w:hyperlink'))):
            pass
        if m:
            lab, rest = m.group(1), m.group(2)
        else:
            lab, rest = f'Table {tbl_count}.', txt
        for r in list(prev):
            if r.tag != qn('w:pPr'):
                prev.remove(r)
        def mk(text, bold=False):
            r = OxmlElement('w:r'); rpr = OxmlElement('w:rPr')
            if bold: rpr.append(OxmlElement('w:b'))
            r.append(rpr); t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve'); t.text = text; r.append(t)
            return r
        prev.append(mk(lab + ' ', True)); prev.append(mk(rest))
        set_pstyle(prev, 'Table Caption')

# table notes "TN:"
for p in list(body.iter(qn('w:p'))):
    t = ptext(p)
    if t.startswith('TN:'):
        set_pstyle(p, 'Table Note')
        # strip prefix
        for tt in p.iter(qn('w:t')):
            if tt.text and 'TN:' in tt.text:
                tt.text = tt.text.replace('TN: ', '', 1).replace('TN:', '', 1)
                break

# figures: caption paragraphs and image paragraphs
for p in body.iter(qn('w:p')):
    pPr = p.find(qn('w:pPr'))
    if pPr is not None and pPr.find(qn('w:pStyle')) is not None:
        sid = pPr.find(qn('w:pStyle')).get(qn('w:val'))
        if sid in (styles['Captioned Figure'].style_id, styles['Figure'].style_id):
            pass

# highlights: add bullet character
for p in body.iter(qn('w:p')):
    pPr = p.find(qn('w:pPr'))
    if pPr is not None and pPr.find(qn('w:pStyle')) is not None and \
            pPr.find(qn('w:pStyle')).get(qn('w:val')) == styles['Highlight'].style_id:
        r = OxmlElement('w:r'); t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve'); t.text = '•  '
        r.append(t)
        first = pPr.getnext()
        pPr.addnext(r)

# abstract block rules: top border on "Abstract" heading, bottom border on keywords
def add_border(p, side):
    pPr = p.find(qn('w:pPr'))
    bdr = pPr.find(qn('w:pBdr'))
    if bdr is None:
        bdr = OxmlElement('w:pBdr'); pPr.append(bdr)
    e = OxmlElement(f'w:{side}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '6'); e.set(qn('w:space'), '4')
    e.set(qn('w:color'), '000000'); bdr.append(e)

allp = list(body.iter(qn('w:p')))
for i, p in enumerate(allp):
    if ptext(p).strip() == 'Abstract':
        add_border(p, 'top')
    if ptext(p).startswith('Keywords:'):
        add_border(p, 'bottom')
    if ptext(p).strip() == 'Highlights':
        add_border(p, 'top')

# page break before main text (after keywords) and before references / supplementary sections
def page_break_before(p):
    pPr = p.find(qn('w:pPr'))
    e = OxmlElement('w:pageBreakBefore'); pPr.append(e)

for p in body.iter(qn('w:p')):
    t = ptext(p).strip()
    if not SUPP and t == 'Nomenclature':
        page_break_before(p)
    if SUPP and t.startswith('Supplementary Note S1.') and p.find(qn('w:pPr')) is not None and \
            p.find(qn('w:pPr')).find(qn('w:pStyle')) is not None and \
            p.find(qn('w:pPr')).find(qn('w:pStyle')).get(qn('w:val')).startswith('Heading'):
        page_break_before(p)
    if SUPP and t in ('Supplementary Tables', 'Supplementary Figures', 'Supplementary Algorithm S1'):
        page_break_before(p)

# images: limit width to text width
for d in body.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline'):
    ext = d.find('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent')
    cx, cy = int(ext.get('cx')), int(ext.get('cy'))
    maxw = int(Cm(TEXT_W_CM))
    if cx > maxw:
        ratio = maxw / cx
        ext.set('cx', str(maxw)); ext.set('cy', str(int(cy * ratio)))
        for a in d.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}ext'):
            a.set('cx', str(maxw)); a.set('cy', str(int(cy * ratio)))


# ----------------------------------------------------------------------------- schema order normalisation
ORD = {
 'pPr': 'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange',
 'rPr': 'rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath',
 'tblPr': 'tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription',
 'tcPr': 'cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark',
 'style': 'name aliases basedOn next link autoRedefine hidden uiPriority semiHidden unhideWhenUsed qFormat locked personal personalCompose personalReply rsid pPr rPr tblPr trPr tcPr tblStylePr',
 'tcBorders': 'top start left bottom end right insideH insideV tl2br tr2bl',
 'pBdr': 'top left bottom right between bar',
 'tblCellMar': 'top start left bottom end right',
}
ORD = {k: v.split() for k, v in ORD.items()}


def normalise(root):
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        ns, _, local = el.tag[1:].partition('}')
        if ns == W_NS and local in ORD:
            order = ORD[local]
            kids = list(el)
            def key(c):
                n = c.tag.split('}')[-1] if isinstance(c.tag, str) else ''
                return order.index(n) if n in order else len(order)
            seen = set(); keep = []
            for c in sorted(kids, key=key):
                n = c.tag.split('}')[-1] if isinstance(c.tag, str) else ''
                if local in ('pPr', 'rPr', 'tblPr', 'tcPr') and n in seen and n in order:
                    continue
                seen.add(n); keep.append(c)
            for c in kids:
                el.remove(c)
            for c in keep:
                c.tail = None
                el.append(c)
        elif ns == W_NS and local == 'trPr':
            seen = set()
            for c in list(el):
                n = c.tag.split('}')[-1]
                if n in seen:
                    el.remove(c)
                seen.add(n)
        elif ns == M_NS and local == 'rPr':
            if el.find(f'{{{M_NS}}}nor') is not None:
                for c in el.findall(f'{{{M_NS}}}sty'):
                    el.remove(c)
            order = ['lit', 'nor', 'scr', 'sty', 'brk', 'aln']
            kids = sorted(list(el), key=lambda c: order.index(c.tag.split('}')[-1]) if c.tag.split('}')[-1] in order else 9)
            for c in list(el): el.remove(c)
            for c in kids: el.append(c)
        elif ns == M_NS and local == 'dPr':
            order = ['begChr', 'sepChr', 'endChr', 'grow', 'shp', 'ctrlPr']
            kids = sorted(list(el), key=lambda c: order.index(c.tag.split('}')[-1]) if c.tag.split('}')[-1] in order else 9)
            for c in list(el): el.remove(c)
            for c in kids: el.append(c)
        elif ns == M_NS and local == 'mcPr':
            kids = sorted(list(el), key=lambda c: 0 if c.tag.endswith('}count') else 1)
            for c in list(el): el.remove(c)
            for c in kids: el.append(c)


normalise(doc.element.body)
normalise(doc.styles.element)

# remove pandoc's default "Compact"/"Body Text" leftovers are fine; set document language
doc.core_properties.title = 'Manuscript' if not SUPP else 'Supplementary material'
doc.core_properties.author = ''
doc.save(out)
os.remove(raw); os.remove(tmp_md)

# ----------------------------------------------------------------------------- zip-level fixes (settings order, numbering nsid)
import zipfile, shutil
SETTINGS_ORDER = """writeProtection view zoom removePersonalInformation removeDateAndTime doNotDisplayPageBoundaries displayBackgroundShape
printPostScriptOverText printFractionalCharacterWidth printFormsData embedTrueTypeFonts embedSystemFonts saveSubsetFonts saveFormsData mirrorMargins
alignBordersAndEdges bordersDoNotSurroundHeader bordersDoNotSurroundFooter gutterAtTop hideSpellingErrors hideGrammaticalErrors activeWritingStyle
proofState formsDesign attachedTemplate linkStyles stylePaneFormatFilter stylePaneSortMethod documentType mailMerge revisionView trackRevisions
doNotTrackMoves doNotTrackFormatting documentProtection autoFormatOverride styleLockTheme styleLockQFSet defaultTabStop autoHyphenation
consecutiveHyphenLimit hyphenationZone doNotHyphenateCaps showEnvelope summaryLength clickAndTypeStyle defaultTableStyle evenAndOddHeaders
bookFoldRevPrinting bookFoldPrinting bookFoldPrintingSheets drawingGridHorizontalSpacing drawingGridVerticalSpacing displayHorizontalDrawingGridEvery
displayVerticalDrawingGridEvery doNotUseMarginsForDrawingGridOrigin drawingGridHorizontalOrigin drawingGridVerticalOrigin doNotShadeFormData
noPunctuationKerning characterSpacingControl printTwoOnOne strictFirstAndLastChars noLineBreaksAfter noLineBreaksBefore savePreviewPicture
doNotValidateAgainstSchema saveInvalidXml ignoreMixedContent alwaysShowPlaceholderText doNotDemarcateInvalidXml saveXmlDataOnly useXSLTWhenSaving
saveThroughXslt showXMLTags alwaysMergeEmptyNamespace updateFields hdrShapeDefaults footnotePr endnotePr compat docVars rsids mathPr attachedSchema
themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats doNotAutoCompressPictures forceUpgrade captions readModeInkLockDown smartTagType
schemaLibrary shapeDefaults doNotEmbedSmartTags decimalSymbol listSeparator""".split()


def fix_zip(path):
    tmp = path + '.fix'
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == 'word/settings.xml':
                root = etree.fromstring(data)
                kids = list(root)
                key = lambda c: SETTINGS_ORDER.index(c.tag.split('}')[-1]) if c.tag.split('}')[-1] in SETTINGS_ORDER else len(SETTINGS_ORDER)
                for c in kids: root.remove(c)
                for c in sorted(kids, key=key):
                    c.tail = None; root.append(c)
                data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
            elif item.filename == 'word/numbering.xml':
                txt = data.decode('utf-8')
                txt = re.sub(r'<w:nsid w:val="([0-9A-Fa-f]+)"/>', lambda m: f'<w:nsid w:val="{m.group(1).upper().rjust(8, "0")[-8:]}"/>', txt)
                data = txt.encode('utf-8')
            zout.writestr(item, data)
    shutil.move(tmp, path)


fix_zip(out)

print('saved', out, 'tables', tbl_count)
