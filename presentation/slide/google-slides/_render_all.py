#!/usr/bin/env python
"""Render every <section> of the deck to a true 1920x1080 PNG.
Headless Chrome reserves ~96px of window height, so we render at window height 1176
(=> a real 1080px viewport) and crop the top 1080px. Output: s-NN.png in this dir."""
import os, re, subprocess, sys, tempfile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "RoadFury to SE2RoadNet.dc.html")
CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"
FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800'
         '&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">')
TPL = ('<!doctype html><html><head><meta charset="utf-8">' + FONTS +
       '<style>html,body{margin:0;padding:0;background:#fff}'
       'section{width:1920px;height:1080px;box-sizing:border-box;overflow:hidden}</style></head><body>%s</body></html>')

deck = open(DECK, encoding="utf-8").read()
secs = re.findall(r'<section[\s\S]*?</section>', deck)
print("sections:", len(secs))
only = sys.argv[1:]  # optional list of indices to render; default all

for i, sec in enumerate(secs):
    if only and str(i) not in only and str(i).zfill(2) not in only:
        continue
    html = TPL % sec
    tmp = os.path.join(HERE, "_rtmp_%02d.html" % i)
    open(tmp, "w", encoding="utf-8").write(html)
    out = os.path.join(HERE, "s-%02d.png" % i)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--screenshot=" + out, "--window-size=2040,1200",
                    "--force-device-scale-factor=1", "--virtual-time-budget=3500", tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(tmp)
    im = Image.open(out)
    if im.size != (1920, 1080):
        im.crop((0, 0, 1920, 1080)).save(out)
    print("  s-%02d.png  %s" % (i, (re.search(r'data-label="([^"]*)"', sec) or [None,'?'])[1]))

print("done")
