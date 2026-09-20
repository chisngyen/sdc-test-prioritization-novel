"""Render a built pptx to per-slide PNGs + contact sheets for visual verification.
usage: python _verify_render.py <pdf_path> <out_prefix>   (pdf already produced by soffice)
"""
import sys, os, glob, fitz
from PIL import Image, ImageDraw, ImageFont

pdf = sys.argv[1] if len(sys.argv) > 1 else "_review/_full.pdf"
prefix = sys.argv[2] if len(sys.argv) > 2 else "n"
outdir = "_review/new_png"
os.makedirs(outdir, exist_ok=True)

doc = fitz.open(pdf)
mat = fitz.Matrix(150/72, 150/72)
for i, p in enumerate(doc):
    p.get_pixmap(matrix=mat).save(f"{outdir}/{prefix}-{i:02d}.png")
print("rendered", doc.page_count, "->", outdir)

files = sorted(glob.glob(f"{outdir}/{prefix}-*.png"))
cols, rows = 3, 5
pad, labelh, tw = 14, 30, 760
sheets = [files[i:i+cols*rows] for i in range(0, len(files), cols*rows)]
for si, sheet in enumerate(sheets):
    im0 = Image.open(sheet[0]); th = int(tw*im0.height/im0.width)
    cw, ch = tw+pad, th+labelh+pad
    canvas = Image.new("RGB", (cols*cw+pad, rows*ch+pad), (245,245,245))
    d = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("arial.ttf", 22)
    except: font = ImageFont.load_default()
    for k, f in enumerate(sheet):
        r, c = divmod(k, cols); x = pad+c*cw; y = pad+r*ch
        canvas.paste(Image.open(f).resize((tw, th)), (x, y+labelh))
        d.text((x+4, y+4), os.path.basename(f).replace(".png",""), fill=(0,0,0), font=font)
    canvas.save(f"_review/sheet_new_{si}.png")
    print("sheet", si)
