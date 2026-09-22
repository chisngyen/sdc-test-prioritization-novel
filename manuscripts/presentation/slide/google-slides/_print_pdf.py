#!/usr/bin/env python
"""Print the deck to a REAL-TEXT PDF (selectable text + embedded fonts), NOT images.

Route #3 (same as the MEMRES-to-CGAR reference): stack every <section> into one
harness HTML with an exact @page size, then let headless Chrome print it via
Skia/PDF. Text stays vector + searchable; only logos/raster assets become images.
Output: RoadFury-to-SE2RoadNet-text.pdf in the slide/ folder."""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "RoadFury to SE2RoadNet.dc.html")
OUT  = os.path.abspath(os.path.join(HERE, "..", "RoadFury-to-SE2RoadNet-text.pdf"))
CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800'
         '&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">')

# @page size in px -> Chrome maps 1920x1080px to 1440x810pt (16:9), matching the reference.
HEAD = ('<!doctype html><html><head><meta charset="utf-8">' + FONTS + '<style>'
        '@page{size:1920px 1080px;margin:0;}'
        'html,body{margin:0;padding:0;background:#fff;}'
        '*{-webkit-print-color-adjust:exact;print-color-adjust:exact;}'
        'section{width:1920px;height:1080px;box-sizing:border-box;overflow:hidden;'
        'break-after:page;page-break-after:always;break-inside:avoid;}'
        'section:last-of-type{break-after:auto;page-break-after:avoid;}'
        '</style></head><body>')

deck = open(DECK, encoding="utf-8").read()
secs = re.findall(r'<section[\s\S]*?</section>', deck)
print("deck sections:", len(secs))

# The clickable "Sources" slide lives only in the pptx build (build_pptx.py), not in the
# HTML deck. Append a theme-matched, clickable equivalent so the text PDF also satisfies
# the professor's image-source rule. Kept here (not in the deck) to avoid a duplicate
# Sources slide when build_pptx.py appends its own.
SOURCES = [
    ("Logo HCMUS",                       "https://www.hcmus.edu.vn"),
    ("Logo ICST 2026",                   "https://conf.researchr.org/home/icst-2026"),
    ("Video demo (Google Drive)",        "https://drive.google.com/file/d/1JC0NY3qfW-if9cM74Zi3d0VaTcA-1el_/view?usp=sharing"),
    ("Video demo (Facebook)",            "https://www.facebook.com/reel/1528755028625867"),
    ("Ảnh đường & histogram độ cong",    "https://github.com/christianbirchler-org/sensodat"),
    ("Ảnh chụp mô phỏng (BeamNG.tech)",  "https://www.beamng.tech/"),
]
rows = "".join(
    '<div style="display:flex;align-items:baseline;gap:18px;margin-bottom:22px;">'
    '<span style="flex:none;min-width:430px;font-weight:700;color:#F6F4EF;font-size:30px;">%s</span>'
    '<a href="%s" style="color:#E0772B;font-family:\'JetBrains Mono\',monospace;font-size:24px;'
    'word-break:break-all;text-decoration:none;border-bottom:1px solid rgba(224,119,43,0.4);">%s</a>'
    '</div>' % (label, url, url) for label, url in SOURCES)
NOTE = ("Sơ đồ kiến trúc là hình TikZ tự vẽ (fig_arch, fig_arch_se2); các biểu đồ APFD/leaderboard/focal "
        "là SVG/CSS tự vẽ từ số liệu thí nghiệm của nhóm &mdash; không cần dẫn nguồn. Ảnh đường &amp; histogram "
        "độ cong vẽ từ bộ dữ liệu SensoDat [1]; ảnh chụp mô phỏng từ BeamNG.tech. Thumbnail video sẽ được "
        "thay bằng khung hình video của nhóm.")
sources_sec = (
    '<section data-label="Sources" style="font-family:\'Be Vietnam Pro\',sans-serif;'
    'background:#16242A;color:#F6F4EF;padding:58px 96px 44px;display:flex;flex-direction:column;overflow:hidden;">'
    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:24px;letter-spacing:3px;'
    'text-transform:uppercase;color:#8BA0A2;">Sources &middot; Image credits</span>'
    '<h1 style="font-size:58px;font-weight:800;margin:14px 0 48px;color:#F6F4EF;">Nguồn hình ảnh / Image sources</h1>'
    '<div style="flex:1;">' + rows + '</div>'
    '<p style="font-size:22px;font-style:italic;color:#8BA0A2;line-height:1.5;margin:0;">' + NOTE + '</p>'
    '</section>')
secs.append(sources_sec)
print("total sections (+Sources):", len(secs))

harness = os.path.join(HERE, "_harness_pdf.html")
open(harness, "w", encoding="utf-8").write(HEAD + "\n".join(secs) + "</body></html>")

profile = tempfile.mkdtemp(prefix="chr_pdf_")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                "--user-data-dir=" + profile, "--force-device-scale-factor=1",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=12000",
                "--print-to-pdf=" + OUT, harness],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote:", OUT, os.path.getsize(OUT) if os.path.exists(OUT) else "MISSING")
