# Build the GSDSEF 2027 digital presentation from the official template.
# Plain style per template rules: black text, white background, Arial only,
# 12 slides max, data displayed on slides, no links.
import copy
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

ROOT = r"C:\Users\moonw\EndoscopicMonocularSLAM"
TPL = ROOT + r"\presentation\template.pptx"
import os as _os
OUT = _os.environ.get("ARGUS_DECK_OUT") or ROOT + r"\presentation\Argus_GSDSEF_2027_Digital_Presentation.pptx"
BLACK = RGBColor(0, 0, 0)
FONT = "Arial"

prs = Presentation(TPL)

# ---------- structural: delete the directions slide (slide 1) ----------
xml_slides = prs.slides._sldIdLst
slide_ids = list(xml_slides)
rId = slide_ids[0].get(qn("r:id"))
prs.part.drop_rel(rId)
xml_slides.remove(slide_ids[0])

slides = list(prs.slides)  # now 12: title, abstract, intro, materials, m1, m2, r1, r2, r3, disc, concl, refs

# ---------- helpers ----------
def _style_run(r, size, bold, italic=False):
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = BLACK

def _no_bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "0")
    pPr.set("indent", "0")
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    pPr.append(pPr.makeelement(qn("a:buNone"), {}))

def _bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "228600")
    pPr.set("indent", "-228600")
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    f = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
    c = pPr.makeelement(qn("a:buChar"), {"char": "\u2022"})
    pPr.append(f)
    pPr.append(c)

def para(tf, runs, size=13, bullet=False, level=0, after=5, first=False, align=None):
    """runs: list of (text, bold) or (text, bold, italic)."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.level = level
    for t in runs:
        txt, bold = t[0], t[1]
        italic = t[2] if len(t) > 2 else False
        r = p.add_run()
        r.text = txt
        _style_run(r, size, bold, italic)
    p.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    if bullet:
        _bullet(p)
    else:
        _no_bullet(p)
    return p

def body_of(slide):
    for sh in slide.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 1:
            return sh
    raise RuntimeError("no body placeholder")

def clear_body(slide, top=None, height=None):
    sh = body_of(slide)
    tf = sh.text_frame
    tf.clear()
    tf.word_wrap = True
    # always write a full explicit frame: a partial xfrm collapses width to zero
    sh.left = Inches(0.5)
    sh.width = Inches(9.0)
    sh.top = Inches(top if top is not None else 1.75)
    sh.height = Inches(height if height is not None else 4.95)
    return tf, sh

def drop_autoshapes(slide):
    for sh in list(slide.shapes):
        if sh.shape_type == 1:  # AUTO_SHAPE decorations (wingding arrows)
            sh._element.getparent().remove(sh._element)

def plain_table(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for el in tblPr.findall(qn("a:tableStyleId")):
        el.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # Table Grid: black lines, no fill

def style_cell(cell, text, size=12, bold=False):
    tf = cell.text_frame
    tf.word_wrap = True
    tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    _style_run(r, size, bold)
    _no_bullet(p)
    cell.margin_left = Inches(0.05)
    cell.margin_right = Inches(0.05)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    cell.fill.background()

def add_table(slide, rows_data, x, y, w, col_widths=None, size=12, header=True, row_h=0.28):
    rows, cols = len(rows_data), len(rows_data[0])
    gfx = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(row_h * rows))
    table = gfx.table
    plain_table(table)
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Inches(cw)
    for ri, row in enumerate(rows_data):
        table.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            style_cell(table.cell(ri, ci), val, size=size, bold=(header and ri == 0))
    return gfx

def fit(path, max_w, max_h):
    with Image.open(path) as im:
        w, h = im.size
    s = min(max_w / w, max_h / h)
    return w * s, h * s

def add_fig(slide, path, x, y, max_w, max_h, caption=None, cap_w=None):
    w, h = fit(path, max_w, max_h)
    x = x + (max_w - w) / 2  # center within the allotted cell
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    if caption:
        tb = slide.shapes.add_textbox(Inches(x), Inches(y + h + 0.02), Inches(cap_w or max_w), Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        para(tf, [(caption, False, True)], size=12, first=True, after=0)
    return w, h

def placeholder_box(slide, x, y, w, h, label):
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = RGBColor(255, 255, 255)
    b.line.color.rgb = BLACK
    b.line.width = Pt(1)
    ln = b.line._get_or_add_ln()
    dash = ln.makeelement(qn("a:prstDash"), {"val": "dash"})
    ln.append(dash)
    b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "INSERT IMAGE"
    _style_run(r, 12, True)
    p.alignment = PP_ALIGN.CENTER
    _no_bullet(p)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = label
    _style_run(r2, 12, False, italic=True)
    p2.alignment = PP_ALIGN.CENTER
    _no_bullet(p2)
    return b

def caption_box(slide, text, x, y, w, size=12):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    para(tf, [(text, False, True)], size=size, first=True, after=0)

# ================= SLIDE 1: TITLE =================
s = slides[0]
for sh in list(s.shapes):  # remove the instruction text box
    if sh.shape_type == 17:
        sh._element.getparent().remove(sh._element)
title_ph = None
for sh in s.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 0:
        title_ph = sh
title_ph.top = Inches(0.55)
title_ph.height = Inches(2.6)
tf = title_ph.text_frame
tf.clear()
tf.word_wrap = True
para(tf, [("Argus: Monocular 3D Coverage Mapping to Reveal Blind Spots in Colonoscopy", True)], size=30, first=True, after=14, align=PP_ALIGN.CENTER)
para(tf, [("Biomedical/Health Sciences, and Biomedical Engineering", False)], size=20, after=8, align=PP_ALIGN.CENTER)
para(tf, [("Kathleen Wang, 11th Grade", False)], size=20, after=0, align=PP_ALIGN.CENTER)
w, h = fit(ROOT + r"\results\blindspot_map.png", 5.6, 3.1)
s.shapes.add_picture(ROOT + r"\results\blindspot_map.png", Inches((10 - w) / 2), Inches(3.35), Inches(w), Inches(h))
caption_box(s, "A reconstructed colon segment with its blind-spot map: patches seen fewer than three times are flagged (my figure).", 2.35, 3.38 + h + 0.03, 5.3)
add_fig(s, ROOT + r"\presentation\shots\app_home.png", 0.30, 3.55, 1.75, 2.55)
caption_box(s, "Argus Lite, the phone app (my screenshot)", 0.10, 6.14, 2.15)
add_fig(s, ROOT + r"\presentation\shots\app_live_study.png", 7.95, 3.55, 1.75, 2.55)
caption_box(s, "a live camera study in the app (my screenshot)", 7.80, 6.14, 2.15)

# ================= SLIDE 2: ABSTRACT =================
s = slides[1]
drop_autoshapes(s)
tf, sh = clear_body(s, top=1.55, height=3.95)
ABSTRACT = ("Colonoscopy prevents cancer only where the colon wall is actually examined, yet roughly one in four "
            "precancerous polyps is missed, largely because parts of the wall are never brought into view and no "
            "standard instrument tells the endoscopist which parts those were. Argus answers that question from an "
            "unmodified endoscope's own video. The pipeline chains a domain-fine-tuned monocular depth network, a "
            "from-scratch direct visual odometry tracker, fusion of the depth maps into a 3D surface, per-patch "
            "view counting, and a cascaded polyp detector, running end to end on one workstation and on an "
            "ordinary Android phone. Development followed a strict engineering discipline: twenty-three design "
            "iterations, each judged against pass or fail criteria committed to a public git record before the test "
            "ran, with failures published alongside their mechanisms; two upgrades passed their gates and were "
            "adopted. On phantom ground truth the final pipeline tracks the camera to 0.42 mm mean trajectory "
            "error, reconstructs the surface to 1.3 to 2.4 mm median error, and cuts false blind-spot alarms from "
            "7 to 41 percent of wall down to 0.5 to 3.9 percent through depth fine-tuning, the project's "
            "central finding: generic AI fails inside the body, and targeted fine-tuning rescues it. The same "
            "finding repeated in detection, where a redesigned dense-sampling protocol raised detection on unseen "
            "clinical footage from 13 of 25 to 19 of 25 polyp clips (paired, p = 0.0012). Head to head, the "
            "tracker draws with the standard tool ORB-SLAM3 on accuracy while posing every frame of video.")
para(tf, [(ABSTRACT, False)], size=15, first=True, after=0)
print("ABSTRACT WORDS:", len(ABSTRACT.split()))
add_fig(s, ROOT + r"\presentation\shots\app_surface_xray.png", 0.5, 5.55, 2.9, 1.1)
add_fig(s, ROOT + r"\results\polyp_find_size_demo.png", 3.55, 5.55, 2.9, 1.1)
add_fig(s, ROOT + r"\results\polyp_examples\good_detection_dice0.99.png", 6.6, 5.55, 2.9, 1.1)
caption_box(s, "Left: the app's 3D map, blind spots shown through the surface (my screenshot). Center: automatic polyp find-and-size. Right: segmentation vs expert outline.", 0.5, 6.70, 9.0)

# ================= SLIDE 3: INTRODUCTION =================
s = slides[2]
tf, sh = clear_body(s, top=1.45, height=5.45)
intro = [
    ([("Problem. ", True),
      ("Colonoscopy is the gold standard for preventing colorectal cancer, but tandem studies estimate that about "
       "a quarter of adenomas are missed, and a major mechanism is simple: parts of the colon wall are hidden "
       "behind folds, passed too quickly, or seen only at a glancing angle, and the endoscopist has no record of "
       "which parts those were. Procedure quality is currently measured by proxies such as adenoma detection rate "
       "and withdrawal time, which describe the doctor on average but say nothing about the wall of this patient's "
       "procedure.", False)], 13),
    ([("What is already known. ", True),
      ("Commercial AI for colonoscopy (computer-aided detection) highlights polyps that appear on screen; by "
       "construction it cannot see wall that never appears. General-purpose visual SLAM such as ORB-SLAM3 is "
       "mature, and monocular depth networks such as Depth Anything V2 are strong on everyday scenes, but the "
       "colon violates their assumptions with weak texture, specular glare, and deforming tissue. Research "
       "reconstruction systems typically need added hardware or offline processing, and the C3VD phantom datasets "
       "are the only public source of trajectory ground truth.", False)], 13),
    ([("Engineering goal. ", True),
      ("Build and honestly validate a system that turns a standard, unmodified endoscope's video into a 3D "
       "reconstruction with a per-patch coverage map that shows what was examined, what was only glanced at, and "
       "where the map has holes, running on consumer hardware, with every stage measured against ground truth and "
       "every claim carrying its trivial baseline.", False)], 13),
    ([("Origin and hypothesis evolution. ", True),
      ("The project began with the hypothesis that specular glare breaks endoscopic SLAM and that removing it "
       "would help. Controlled experiments largely disproved this: feature-based tracking is already glare-robust, "
       "and naive glare inpainting actively hurt. That disproof produced a from-scratch dense tracker and a pivot "
       "to the coverage question, where a staged ablation isolated learned depth as the true bottleneck.", False)], 13),
]
first = True
total = 0
for runs, sz in intro:
    para(tf, runs, size=sz, first=first, after=7)
    first = False
    total += sum(len(t[0].split()) for t in runs)
print("INTRO WORDS:", total)

# ================= SLIDE 4: MATERIALS =================
s = slides[3]
tf, sh = clear_body(s, top=1.55, height=5.3)
mats = [
    [("Datasets. ", True),
     ("C3VD and C3VDv2: silicone colon phantoms filmed with a real endoscope, with 6-degree-of-freedom camera "
      "ground truth, the basis of all trajectory and reconstruction validation (5 headline sequences plus full "
      "screening withdrawals). EndoSLAM: ex-vivo porcine colon with OptiTrack camera ground truth, used for "
      "cross-dataset tests on real tissue (5 trajectories, 457 to 942 frames each). HyperKvasir and public "
      "clinical endoscopy clips: unseen human footage for the two frozen detection tests (n = 41 and n = 48 clips).", False)],
    [("Hardware. ", True),
     ("One Windows workstation with a single NVIDIA RTX A1000 (8 GB) graphics card, which also performed all "
      "neural-network fine-tuning, and one ordinary Android phone that runs the complete pipeline on-device, "
      "including from its own live camera.", False)],
    [("Software and models. ", True),
     ("Python with PyTorch, OpenCV, and Open3D for the pipeline; Depth Anything V2 (fine-tuned on phantom data) "
      "for depth; a visual odometry tracker written from scratch for this project; DeepLabV3+ with an "
      "EfficientNet-B4 backbone (fine-tuned) for polyp segmentation; Kotlin and Android Studio for the phone "
      "application; a containerized ORB-SLAM3 build as the comparison baseline.", False)],
    [("The record. ", True),
     ("A public git repository holds every pre-registered pass or fail criterion (committed before each test ran), "
      "all raw result artifacts, and the audit scripts that reproduce every published number.", False)],
]
first = True
for runs in mats:
    para(tf, runs, size=13, bullet=True, first=first, after=8)
    first = False
y_m = 5.90
add_fig(s, ROOT + r"\results\c3vd_undistort.png", 0.5, y_m, 2.9, 0.82)
add_fig(s, ROOT + r"\results\screening_polyp\c0_0_f0318.jpg", 3.55, y_m, 2.9, 0.82)
add_fig(s, ROOT + r"\results\hk_clip1_sample.jpg", 6.6, y_m, 2.9, 0.82)
caption_box(s, "The three data domains: ground-truth phantom, screening-pace withdrawal, unseen clinical footage.", 0.5, y_m + 0.85, 8.9)

# ================= SLIDE 5: METHODS (block diagram + validation) =================
s = slides[4]
tf, sh = clear_body(s, top=4.05, height=1.70)

def box(slide, x, y, w, h, lines, size=12):
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = RGBColor(255, 255, 255)
    b.line.color.rgb = BLACK
    b.line.width = Pt(1.25)
    b.shadow.inherit = False
    btf = b.text_frame
    btf.word_wrap = True
    btf.margin_left = Inches(0.04)
    btf.margin_right = Inches(0.04)
    btf.margin_top = Inches(0.02)
    btf.margin_bottom = Inches(0.02)
    for i, (txt, bold) in enumerate(lines):
        p = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
        r = p.add_run()
        r.text = txt
        _style_run(r, size, bold)
        p.alignment = PP_ALIGN.CENTER
        _no_bullet(p)
    return b

def harrow(slide, x, y):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(0.24), Inches(0.18))
    a.fill.solid()
    a.fill.fore_color.rgb = BLACK
    a.line.color.rgb = BLACK
    a.shadow.inherit = False

BY, BH, BW = 1.55, 1.15, 1.58
boxes = [
    [("Raw video", True), ("standard endoscope, no added hardware", False)],
    [("Learned depth", True), ("fine-tuned Depth Anything V2", False)],
    [("Tracking", True), ("from-scratch direct visual odometry", False)],
    [("3D fusion", True), ("depth maps merged into a surface", False)],
    [("Coverage map", True), ("views counted per wall patch", False)],
]
x = 0.5
for i, lines in enumerate(boxes):
    box(s, x, BY, BW, BH, lines)
    x += BW
    if i < 4:
        harrow(s, x + 0.015, BY + BH / 2 - 0.09)
        x += 0.27
varr = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(2.95), Inches(BY + BH + 0.05), Inches(0.18), Inches(0.22))
varr.fill.solid()
varr.fill.fore_color.rgb = BLACK
varr.line.color.rgb = BLACK
varr.shadow.inherit = False
box(s, 1.10, BY + BH + 0.30, 3.9, 0.78,
    [("Polyp detection cascade", True), ("fast student screens every frame, teacher confirms; findings pinned and sized in the 3D map", False)])
caption_box(s, "Block diagram of the full program, drawn by me.", 5.3, BY + BH + 0.55, 4.0)
meth = [
    [("Validation by staged ablation. ", True),
     ("On phantom ground truth, each estimated input is swapped for its ground-truth version to isolate error "
      "sources (ground-truth depth with estimated poses, estimated depth with ground-truth poses, and so on), so "
      "every error is attributed to a stage, not guessed at. All measurements are metric (millimeters).", False)],
    [("Pre-registration. ", True),
     ("Every experiment's pass or fail criteria, data lists, and planned outcomes were committed to the public git "
      "record before the test ran, then never changed; testing stops at the first failed gate, and failures are "
      "published with their mechanisms. Every trajectory error is printed beside the same sequence's trivial "
      "constant-point baseline so a flattering but meaningless number cannot hide.", False)],
]
first = True
for runs in meth:
    para(tf, runs, size=13, first=first, after=6)
    first = False
add_fig(s, ROOT + r"\presentation\shots\gate_commit.png", 0.5, 5.85, 2.9, 1.0)
add_fig(s, ROOT + r"\presentation\shots\studio_console.png", 3.55, 5.85, 2.9, 1.0)
add_fig(s, ROOT + r"\presentation\shots\workstation.jpg", 6.6, 5.85, 2.9, 1.0)
caption_box(s, "Left: a pass-or-fail gate committed to GitHub before the experiment ran. Center: the Argus Studio demo console. Right: the live pipeline at my workstation. My screenshots.", 0.5, 6.88, 9.0)

# ================= SLIDE 6: METHODS con'd (design -> test -> redesign) =================
s = slides[5]
tf, sh = clear_body(s, top=1.45, height=3.1)
m2 = [
    [("Design, test, redesign. ", True),
     ("The project ran 23 pre-registered design iterations; 2 passed their gates and were adopted, and the rest "
      "were rejected by their own pre-stated criteria and published anyway. Adoption 1: an improved tracking "
      "kernel (robust in-loop weighting), adopted only after a staged re-validation of every tracking benchmark, "
      "cut mean trajectory error from 0.91 to 0.42 mm. Adoption 2: a dense frame-sampling protocol for detection, "
      "adopted only after passing three gates on a second frozen test of 48 fresh clinical clips.", False)],
    [("Redesigns the evidence rejected ", True),
     ("(each against its own committed bar): glare removal as a tracking aid, two inpainting methods, six "
      "deformation compensations, three generations of loop closure, keyframe tracking, two depth-resolution "
      "speedups, and porting Adoption 1 to the phone's tracker, which failed its gate on one of two "
      "pre-registered windows and was kept off the phone: an adoption is validated per configuration, never "
      "assumed to generalize.", False)],
    [("Self-audit. ", True),
     ("A 35-figure audit reproduces every published number from raw artifacts; a red-team audit re-derives the "
      "headline results with independently written code (15 of 15 checks, including an exact re-track of the "
      "0.42 mm mean); two self-caught metric errors were retracted and are reported as results.", False)],
]
first = True
for runs in m2:
    para(tf, runs, size=13, first=first, after=6)
    first = False
y_fig = 4.75
w1, h1 = fit(ROOT + r"\results\poster\fig13_path_overlay.png", 4.35, 1.95)
add_fig(s, ROOT + r"\results\poster\fig13_path_overlay.png", 0.5, y_fig, 4.35, 1.95)
add_fig(s, ROOT + r"\results\poster\fig2_depth_fix.png", 5.15, y_fig, 4.35, 1.95)
caption_box(s, "Left: estimated camera path against phantom ground truth. Right: off-the-shelf vs fine-tuned depth, the central redesign. Figures produced by me.", 0.5, y_fig + max(h1, fit(ROOT + r"\results\poster\fig2_depth_fix.png", 4.35, 1.95)[1]) + 0.04, 9.0)

# ================= SLIDE 7: RESULTS (core accuracy) =================
s = slides[6]
drop_autoshapes(s)
tf, sh = clear_body(s, top=1.40, height=0.55)
para(tf, [("All results below are against phantom ground truth; n = 5 sequences unless stated; every trajectory "
           "error is reported beside its sequence's constant-point trivial baseline.", False)], size=13, first=True, after=0)
rows = [
    ["Stage (what was measured)", "Result", "Baseline / sample size"],
    ["Camera tracking, mean trajectory error", "0.42 mm (v1: 0.91 mm)", "constant-point baselines 33 to 74 mm; n = 5 sequences, every frame posed"],
    ["Depth accuracy, held-out (leave-one-sequence-out)", "68% off-the-shelf to 92 to 99.7% fine-tuned", "n = 5 held-out sequences"],
    ["3D reconstruction, median surface error", "1.3 to 2.4 mm", "4 of 5 sequences, from video alone"],
    ["False blind-spot alarms (the headline)", "7 to 41% of wall, cut to 0.5 to 3.9%", "same fixed metric before vs after fine-tuning"],
    ["Coverage at screening pace", "about 40% of mapped wall under-seen", "validated against a perfect-tracking twin instrument, 3 withdrawals"],
    ["Withdrawal frame-rate requirement", "tracking fails at ~2 effective fps, holds at ~9", "n = 5 complete withdrawals, ~43 m of ground-truth path"],
]
add_table(s, rows, 0.5, 2.0, 9.0, col_widths=[3.1, 2.7, 3.2], size=12, row_h=0.42)
y_fig = 5.05
hA = add_fig(s, ROOT + r"\results\poster\fig1_ablation.png", 0.5, y_fig, 4.35, 1.65)[1]
add_fig(s, ROOT + r"\results\poster\fig1b_falsealarm.png", 5.15, y_fig, 4.35, 1.65)
caption_box(s, "Left: staged ablation isolating depth as the bottleneck. Right: false-alarm rate before and after depth fine-tuning. Figures produced by me.", 0.5, y_fig + 1.68, 9.0)

# ================= SLIDE 8: RESULTS con'd (unseen clinical footage + baselines) =================
s = slides[7]
tf, sh = clear_body(s, top=1.40, height=0.85)
para(tf, [("Frozen tests on unseen human clinical footage: the detector and its threshold were frozen before any "
           "test clip was downloaded, and no test clip was ever used for tuning.", False)], size=13, first=True, after=0)
rows = [
    ["Test (all pre-registered)", "Result", "Statistics"],
    ["First frozen test, n = 41 clips", "11/21 polyp clips vs 3/20 controls; every flagged control held real labeled pathology", "Fisher exact p = 0.020"],
    ["Miss autopsy, then redesign", "8 of 10 misses traced to sparse frame sampling; dense protocol pre-registered", "candidate frozen before new data"],
    ["Second frozen test, n = 48 fresh clips", "19/25 polyp clips (76%) vs 13/25 (52%), old protocol, identical clips; 2 clean false alarms (pre-stated allowance)", "paired, p = 0.0012"],
    ["Head-to-head vs ORB-SLAM3 (phantom, 50 runs)", "a draw on accuracy: means 0.42 vs 0.39 mm, per-sequence 2-2-1; baseline poses sparse keyframes, Argus every frame", "overlapping bootstrap CIs"],
    ["Cross-dataset check (real porcine tissue)", "four-way statistical tie, 39.6 to 41.4 mm mean; ORB-SLAM3 completes 65 to 95% of each sequence", "n = 5 trajectories, 5 runs per config"],
]
add_table(s, rows, 0.5, 2.25, 9.0, col_widths=[2.6, 4.5, 1.9], size=12, row_h=0.5)
y_fig = 5.55
add_fig(s, ROOT + r"\results\poster\fig10_detector_fix.png", 0.5, y_fig, 4.35, 1.15)
add_fig(s, ROOT + r"\results\figures\orbslam3_endoslam_ate.png", 5.15, y_fig, 4.35, 1.15)
caption_box(s, "Left: the detector's domain-gap failure and its fix. Right: trajectory error on real tissue, all systems. Figures produced by me.", 0.5, y_fig + 1.18, 9.0)

# ================= SLIDE 9: RESULTS con'd (negative results, published) =================
s = slides[8]
tf, sh = clear_body(s, top=1.40, height=0.85)
para(tf, [("Negative results, published under their own pre-registered gates. ", True),
          ("Each failure below carries a measured mechanism; none was quietly dropped. These bound what the "
           "system can honestly claim.", False)], size=13, first=True, after=0)
rows = [
    ["Question (pre-registered gate)", "Measured answer", "Mechanism"],
    ["Can loop closure fix whole-colon drift?", "no: a ground-truth oracle recovers at most 14% on screening paths", "a withdrawal is topologically a line; too few revisits"],
    ["Can a site marked in one procedure be re-found in the next? (bar: 10 mm)", "no: median 26.3 mm across 45 sites, discrimination 29% vs the 80% bar", "per-session map error compounds across two sessions"],
    ["Can tissue deformation be compensated? (6 attempts)", "no: all six rejected; best miss was -18% vs a 20% bar", "measured 9x cost; learned weights did not generalize"],
    ["Is the phone video-rate?", "no: about 3 fps on-device (depth 199 ms, track 25, fuse 50, detect 76)", "depth network is the bottleneck; resolution speedup failed its accuracy gate"],
]
add_table(s, rows, 0.5, 2.30, 9.0, col_widths=[3.0, 3.3, 2.7], size=12, row_h=0.5)
y_fig = 5.45
add_fig(s, ROOT + r"\results\poster\fig12_framerate.png", 0.5, y_fig, 4.35, 1.25)
add_fig(s, ROOT + r"\results\poster\fig15_loopclosure.png", 5.15, y_fig, 4.35, 1.25)
caption_box(s, "Left: the frame-rate law from complete withdrawals. Right: the loop-closure oracle bound. Figures produced by me.", 0.5, y_fig + 1.28, 9.0)

# ================= SLIDE 10: DISCUSSION =================
s = slides[9]
tf, sh = clear_body(s, top=1.45, height=3.30)
disc = [
    [("The central finding, demonstrated twice. ", True),
     ("The same disease and cure appeared in two independent AI components: the depth network and the polyp "
      "detector were both strong on their home data and failed in the colon's domain, and in both cases targeted "
      "fine-tuning or protocol redesign recovered them. Competence ends at the edge of the training data; this "
      "matches the published experience of medical AI generally and is the project's most transferable result.", False)],
    [("What the coverage number means. ", True),
     ("A blind spot is a wall patch with fewer than three adequate views, including zero; wall never captured at "
      "all appears as visible holes in the reconstruction rather than in the percentage, so the figure is stated "
      "as a fraction of the mapped wall. The threshold was sensitivity-tested and does not drive the conclusions.", False)],
    [("Errors and surprises. ", True),
     ("Two metrics flattered the system while measuring nothing and were self-caught and retracted: an agreement "
      "score inflated by class imbalance, and an aligned trajectory error that collapses under divergence. Both "
      "produced standing safeguards (trivial baselines beside every number; the guard-rejection rate as the "
      "primary metric under guarding). The original glare hypothesis was itself largely disproven, which redirected "
      "the whole project.", False)],
    [("Why the redesigns happened. ", True),
     ("Each redesign was forced by a measured mechanism, not taste: the detection miss autopsy traced misses to "
      "frame sampling, so the protocol was redesigned and re-frozen; the tracker upgrade failed its phone gate, so "
      "the phone kept its own kernel; drift resisted loop closure for a measured topological reason, so the claim "
      "was bounded instead of forced.", False)],
]
first = True
for runs in disc:
    para(tf, runs, size=13, first=first, after=7)
    first = False
y_d = 4.90
add_fig(s, ROOT + r"\results\poster\fig16_coverage_validation.png", 0.5, y_d, 4.35, 1.70)
add_fig(s, ROOT + r"\results\poster\fig6_crossdataset.png", 5.15, y_d, 4.35, 1.70)
caption_box(s, "Left: coverage validated against the perfect-tracking twin instrument. Right: cross-dataset transfer to real tissue. Figures produced by me.", 0.5, y_d + 1.73, 9.0)

# ================= SLIDE 11: CONCLUSION =================
s = slides[10]
tf, sh = clear_body(s, top=1.45, height=3.70)
conc = [
    [("Answer to the engineering goal. ", True),
     ("A standard endoscope's video alone, processed on consumer hardware, can be turned into a 3D reconstruction "
      "accurate to about 2 mm of surface error with a validated per-patch coverage map; at screening pace about "
      "40% of the mapped wall is genuinely under-seen, which is precisely the information no current instrument "
      "provides.", False)],
    [("Against existing designs. ", True),
     ("Commercial computer-aided detection answers a different question (is a polyp on screen now?) and cannot see "
      "wall that never appears; Argus measures exposure itself. Against the standard open-source tracker, "
      "ORB-SLAM3, the from-scratch tracker draws on accuracy on the phantom and on real tissue while posing every "
      "frame with zero tracking loss; the honest tracking claim is density, glare handling, and a measured "
      "operating envelope, not accuracy supremacy.", False)],
    [("Honest limits. ", True),
     ("Validated where ground truth exists (phantom; real-tissue cross-checks); whole-colon drift of roughly "
      "100 mm is proven unfixable by loop closure here; the phone runs near real time at about 3 fps, below the "
      "measured 9 fps needed for live guidance; cross-patient generalization needs data that does not publicly "
      "exist. Each limit ships with its mechanism.", False)],
    [("Applications and next steps. ", True),
     ("Today: a retrospective procedure report card (coverage grade, blind regions in square millimeters, camera "
      "speed) for quality review and training, a stratification signal rather than a diagnosis. Next: reaching the "
      "measured 9 fps target for live guidance, and a clinical study tying measured coverage to outcomes, which "
      "this project showed requires data that must be collected prospectively.", False)],
]
first = True
for runs in conc:
    para(tf, runs, size=13, first=first, after=7)
    first = False
y_c = 5.30
add_fig(s, ROOT + r"\results\poster\fig7_measurement.png", 0.5, y_c, 2.9, 1.35)
add_fig(s, ROOT + r"\presentation\shots\report_card.png", 3.55, y_c, 2.9, 1.35)
add_fig(s, ROOT + r"\presentation\shots\app_pointcloud.png", 6.6, y_c, 2.9, 1.35)
caption_box(s, "Left: monocular sizing, 0.60 mm median error on 3 to 30 mm targets (my figure). Center: a procedure report card from Argus Studio. Right: the blind-spot map on the phone.", 0.5, y_c + 1.38, 9.0)

# ================= SLIDE 12: REFERENCES / ACKNOWLEDGEMENTS =================
s = slides[11]
tf, sh = clear_body(s, top=1.40, height=5.5)
refs = [
    "Shaukat A. et al. ACG Clinical Guidelines: Colorectal Cancer Screening 2021. American Journal of Gastroenterology 116(3), 2021.",
    "Corley D.A. et al. Adenoma Detection Rate and Risk of Colorectal Cancer and Death. New England Journal of Medicine 370, 2014.",
    "Zhao S. et al. Magnitude, Risk Factors, and Factors Associated With Adenoma Miss Rate of Tandem Colonoscopy. Gastroenterology 156(6), 2019.",
    "Bobrow T.L. et al. Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration. Medical Image Analysis, 2023.",
    "Ozyoruk K.B. et al. EndoSLAM Dataset and an Unsupervised Monocular Visual Odometry and Depth Estimation Approach for Endoscopic Videos. Medical Image Analysis 71, 2021.",
    "Yang L. et al. Depth Anything V2. Advances in Neural Information Processing Systems (NeurIPS), 2024.",
    "Campos C. et al. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM. IEEE Transactions on Robotics 37(6), 2021.",
    "Engel J. et al. Direct Sparse Odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence 40(3), 2018.",
    "Chen L.-C. et al. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. European Conference on Computer Vision (ECCV), 2018.",
    "Borgli H. et al. HyperKvasir, a Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy. Scientific Data 7, 2020.",
    "Umeyama S. Least-Squares Estimation of Transformation Parameters Between Two Point Patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence 13(4), 1991.",
]
para(tf, [("References (one format throughout):", True)], size=13, first=True, after=3)
for rtext in refs:
    para(tf, [(rtext, False)], size=12, bullet=True, after=2)
para(tf, [("Acknowledgements. ", True),
          ("I thank the authors of the validation dataset for correspondence on clinical framing and evaluation "
           "priorities. I used an AI assistant (Anthropic's Claude) as an engineering collaborator for code "
           "drafting and analysis under my direction; all experiments ran on my own hardware against pass or fail "
           "criteria committed in advance, every published number is reproducible from the repository's raw "
           "artifacts, and the responsibility for all of it is mine. This project used no institutional lab and "
           "no mentorship program.", False)], size=12, after=0)

prs.save(OUT)
print("SAVED:", OUT)
