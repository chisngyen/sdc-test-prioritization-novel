#!/usr/bin/env python
"""Assemble _sections/sNN.html fragments into the full .dc.html deck (DeckCanvas scaffold)."""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SECDIR = os.path.join(HERE, "_sections")
OUT = os.path.join(HERE, "RoadFury to SE2RoadNet.dc.html")

HEAD = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;}
  body{margin:0;}
  ::selection{background:#E0772B;color:#fff;}
</style>
<script src="image-slot.js"></script>
</helmet>
<x-import component-from-global-scope="deck-stage" from="./deck-stage.js" width="1920" height="1080" hint-size="100%,100%">

"""

TAIL = "\n</x-import>\n</x-dc>\n</body>\n</html>\n"

def strip_fences(s):
    s = s.strip()
    # remove accidental markdown fences
    s = re.sub(r'^```[a-zA-Z]*\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    return s.strip()

files = sorted(glob.glob(os.path.join(SECDIR, "s*.html")))
if not files:
    sys.exit("No section files in %s" % SECDIR)

parts = []
labels = []
for f in files:
    with open(f, encoding="utf-8") as fh:
        html = strip_fences(fh.read())
    if "<section" not in html:
        print("WARN: %s has no <section> tag" % os.path.basename(f))
    m = re.search(r'data-label="([^"]*)"', html)
    labels.append((os.path.basename(f), m.group(1) if m else "?"))
    parts.append(html)

deck = HEAD + "\n\n".join(parts) + TAIL
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(deck)

print("Wrote %s  (%d sections)" % (os.path.normpath(OUT), len(parts)))
for fn, lab in labels:
    print("  %-10s %s" % (fn, lab))
