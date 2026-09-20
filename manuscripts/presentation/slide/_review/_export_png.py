"""Export every <section> of the (fixed) deck to a sharp 2x PNG (3840x2160).
Renders the HTML directly via headless Chrome (vector-crisp, real fonts, no LibreOffice
fallback artifacts). Output: slides_png/slide-NN.png  (1-indexed).
Headless Chrome reserves ~96 CSS px of window height, so render at window height 1176
(=> a real 1080px viewport) with device-scale-factor 2, then crop the top 3840x2160."""
import os, re, subprocess
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDE = os.path.join(HERE, os.pardir)
DECK = os.path.join(SLIDE, "google-slides", "RoadFury to SE2RoadNet.dc.html")
OUT = os.path.join(SLIDE, "slides_png")
TMP = os.path.join(SLIDE, "google-slides")  # render rel to deck so uploads/ imgs resolve
CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
os.makedirs(OUT, exist_ok=True)

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,700'
         '&family=JetBrains+Mono:ital,wght@0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">')
TPL = ('<!doctype html><html><head><meta charset="utf-8">' + FONTS +
       '<style>html,body{margin:0;padding:0;background:#fff}'
       'section{width:1920px;height:1080px;box-sizing:border-box;overflow:hidden}</style></head><body>%s</body></html>')

deck = open(DECK, encoding="utf-8").read()
secs = re.findall(r'<section[\s\S]*?</section>', deck)
print("sections:", len(secs))
for i, sec in enumerate(secs):
    html = TPL % sec
    tmp = os.path.join(TMP, "_ptmp_%02d.html" % i)
    open(tmp, "w", encoding="utf-8").write(html)
    out = os.path.join(OUT, "slide-%02d.png" % (i + 1))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--screenshot=" + out, "--window-size=2040,1176",
                    "--force-device-scale-factor=2", "--virtual-time-budget=4000", tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(tmp)
    im = Image.open(out)
    if im.size != (3840, 2160):
        im.crop((0, 0, 3840, 2160)).save(out)
    print("  slide-%02d.png  %dx%d" % (i + 1, *Image.open(out).size))
print("done ->", OUT)
