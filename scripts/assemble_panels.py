#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# each entry: output name -> (orientation, [source png stems], figure width in")
# orientation: "row" (side by side) or "col" (stacked)
LAYOUTS = {
    "fig1_data": {
        "orient": "col",
        "sources": ["F1_concentrations", "F1_eda_slices"],
        "width": 8.0,
        "hratios": [1.0, 1.0],
    },
    # Fig 2 = architectures, drawn by hand: blank placeholder
    "_unused_fig2_placeholder": {
        "orient": "col",
        "sources": [],           # no source PNGs; emits a labelled blank panel
        "width": 8.0,
        "blank_height": 5.0,     # inches of reserved space
    },
    "fig3_accuracy": {
        "orient": "col",
        "sources": ["F2_accuracy", "F2_metrics"],
        "width": 8.0,
        "hratios": [1.0, 1.0],
    },
    "fig4_midplane": {
        "orient": "col",
        "sources": ["F3_recon_slices", "F3_midplane_r2"],
        "width": 8.0,
        "hratios": [2.4, 1.0],
    },
    # Fig 5 = cost (speedup) alone, in the main text.
    "fig5_cost": {
        "orient": "col",
        "sources": ["F4_speedup"],
        "width": 6.0,
    },
    # Fig 6 = Sobol ranking, main text.
    "fig6_sobol": {
        "orient": "col",
        "sources": ["F4_sobol"],
        "width": 6.5,
    },
    # Fig 7 = SMoRe ParS on the ABM: (a) sensitivity vs recovery and
    # (b) admissible-region widths, both in one PNG.
    "fig7_smore": {
        "orient": "row",
        "sources": ["F5_smore_regions"],
        "width": 8.0,
        "label_x": [0.0, 0.40],
    },
}

LETTERS = "abcdefgh"
PDF_FAILED = []


def _imread(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(str(path))


def assemble_pdf(name, spec, indir, outdir, labels, dpi):
    """Vector PDF of the assembled figure: the panels' own PDFs placed on one page with pdflatex
    (graphicx and geometry only). Returns False, with the reason printed, when that is not possible."""
    import shutil, subprocess, tempfile
    from PIL import Image

    stems = spec["sources"]
    pdfs = [indir / f"{s}.pdf" for s in stems]
    out_pdf = outdir / f"{name}.pdf"
    xs = spec.get("label_x")
    if shutil.which("pdflatex") is None:
        print(f"  [{name}: no pdflatex in PATH, no PDF]"); return False
    missing = [p.name for p in pdfs if not p.exists()]
    if missing:
        print(f"  [{name}: missing panel PDFs {', '.join(missing)}, no PDF]"); return False
    if spec["orient"] != "col" and len(stems) > 1 or spec.get("panel_x"):
        print(f"  [{name}: layout not supported for PDF]"); return False

    # panel sizes in inches: the PNGs share the PDFs' bounding box
    sizes = [Image.open(indir / f"{s}.png").size for s in stems]
    W = max(w for w, _ in sizes) / dpi
    heights = [h / dpi for _, h in sizes]
    strip = max(0.08, 0.05 * max(heights))          # letter strip above a panel
    gap = 0.7 * strip                                # space between panels
    letter = lambda i: "\\textbf{\\Large (" + LETTERS[i] + ")}"
    body, H = [], 0.0
    with_letters = labels and not (len(stems) == 1 and xs is None)
    if not with_letters:
        for i, p in enumerate(pdfs):
            body.append(f"\\makebox[{W:.3f}in][c]{{\\includegraphics{{{p.as_posix()}}}}}\\par")
            H += heights[i]
            if i < len(pdfs) - 1:
                body.append(f"\\vspace{{{gap:.3f}in}}"); H += gap
    elif xs is not None:
        marks = "".join(f"\\makebox[0pt][l]{{\\hspace{{{xf * W + 0.06:.3f}in}}{letter(i)}}}" for i, xf in enumerate(xs))
        body.append(f"\\makebox[{W:.3f}in][l]{{\\rule{{0pt}}{{{strip:.3f}in}}{marks}}}\\par")
        body.append(f"\\makebox[{W:.3f}in][c]{{\\includegraphics{{{pdfs[0].as_posix()}}}}}\\par")
        H = strip + heights[0]
    else:
        for i, p in enumerate(pdfs):
            body.append(f"\\makebox[{W:.3f}in][l]{{\\rule{{0pt}}{{{strip:.3f}in}}\\hspace{{0.06in}}{letter(i)}}}\\par")
            body.append(f"\\makebox[{W:.3f}in][c]{{\\includegraphics{{{p.as_posix()}}}}}\\par")
            H += strip + heights[i]
            if i < len(pdfs) - 1:
                body.append(f"\\vspace{{{gap:.3f}in}}"); H += gap
    # the panels are stacked in a box, the page is cut to that box and the box is shipped out directly
    body = [l.replace("\\vspace{", "\\vskip ").replace("in}", "in") if l.startswith("\\vspace") else l for l in body]
    tex = ("\\documentclass{article}\n\\usepackage{graphicx}\n\\newsavebox{\\figbox}\n"
           "\\begin{document}\n"
           f"\\sbox{{\\figbox}}{{\\vbox{{\\hsize={W:.3f}in \\offinterlineskip \\parindent=0pt \\parskip=0pt\n"
           + "\n".join(body) + "\n}}\n"
           "\\pdfpagewidth=\\dimexpr\\wd\\figbox+0.06in\\relax\n"
           "\\pdfpageheight=\\dimexpr\\ht\\figbox+\\dp\\figbox+0.06in\\relax\n"
           "\\hoffset=\\dimexpr-1in+0.03in\\relax \\voffset=\\dimexpr-1in+0.03in\\relax\n"
           "\\shipout\\box\\figbox\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{name}.tex"
        src.write_text(tex)
        r = subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error", f"-output-directory={td}", str(src)],
                           capture_output=True, text=True)
        built = Path(td) / f"{name}.pdf"
        if r.returncode != 0 or not built.exists():
            log = Path(td) / f"{name}.log"
            tail = log.read_text(errors="replace").splitlines()[-15:] if log.exists() else [r.stdout[-800:]]
            print(f"  [{name}: pdflatex failed]\n    " + "\n    ".join(tail)); return False
        shutil.copyfile(built, out_pdf)
    print(f"  -> {out_pdf}  (vector)")
    return True


def assemble(name, spec, indir, outdir, labels=True, dpi=300):
    stems = spec["sources"]

    # blank placeholder panel
    if not stems:
        width = spec["width"]
        height = spec.get("blank_height", 5.0)
        fig, ax = plt.subplots(figsize=(width, height))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False,
                     ls="--", ec="0.6", lw=1.2, transform=ax.transAxes))
        ax.text(0.5, 0.5, "architecture diagram\n(placeholder ,  hand-drawn)",
                ha="center", va="center", fontsize=13, color="0.5",
                style="italic", transform=ax.transAxes)
        outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png",):
            fig.savefig(outdir / f"{name}.{ext}", dpi=dpi,
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  -> {outdir / f'{name}.png'}  (blank placeholder)")
        return True

    paths = [indir / f"{s}.png" for s in stems]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"  [skip {name}] missing: {', '.join(p.name for p in missing)}")
        return False

    from PIL import Image, ImageDraw, ImageFont
    import numpy as _np

    def _pil(path):
        im = Image.open(path).convert("RGB")
        return im

    pil_imgs = [_pil(p) for p in paths]
    n = len(pil_imgs)
    orient = spec["orient"]
    xs = spec.get("label_x")

    # a bold font, falling back to default if DejaVu isn't present
    def _font(size):
        for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _pad_to_width(im, target_w):
        """Centre-pad an image with white to reach target width (px)."""
        if im.width >= target_w:
            return im
        canvas = Image.new("RGB", (target_w, im.height), "white")
        canvas.paste(im, ((target_w - im.width) // 2, 0))
        return canvas

    def _add_strip_and_letter(im, letter, strip_px, x_px, font):
        """Add a white strip on top of the image and draw the letter in it."""
        out = Image.new("RGB", (im.width, im.height + strip_px), "white")
        out.paste(im, (0, strip_px))
        draw = ImageDraw.Draw(out)
        # vertically centre the letter in the strip
        try:
            bbox = draw.textbbox((0, 0), letter, font=font)
            th = bbox[3] - bbox[1]
        except Exception:
            th = strip_px // 2
        draw.text((x_px, (strip_px - th) // 2), letter, fill="black", font=font)
        return out

    LABEL_X_PX = 18 # fixed left offset for a letter (px in final img)

    if not labels or (n == 1 and xs is None):
        # single panel, or labels off: just stack/place with no letters
        if orient == "row" and n > 1:
            H = max(im.height for im in pil_imgs)
            scaled = [im.resize((int(im.width * H / im.height), H)) for im in pil_imgs]
            total_w = sum(im.width for im in scaled)
            out = Image.new("RGB", (total_w, H), "white")
            x = 0
            for im in scaled:
                out.paste(im, (x, 0)); x += im.width
        else:
            W = max(im.width for im in pil_imgs)
            padded = [_pad_to_width(im, W) for im in pil_imgs]
            total_h = sum(im.height for im in padded)
            out = Image.new("RGB", (W, total_h), "white")
            y = 0
            for im in padded:
                out.paste(im, (0, y)); y += im.height

    elif xs is not None:
        # single image already holding several internal panels: one white strip
        # across the top, letters at the given x-fractions.
        base = pil_imgs[0]
        strip_px = max(24, int(0.06 * base.height))
        font = _font(int(strip_px * 0.7))
        out = Image.new("RGB", (base.width, base.height + strip_px), "white")
        out.paste(base, (0, strip_px))
        draw = ImageDraw.Draw(out)
        for i, xf in enumerate(xs):
            lx = int(xf * base.width) + LABEL_X_PX
            try:
                bbox = draw.textbbox((0, 0), f"({LETTERS[i]})", font=font)
                th = bbox[3] - bbox[1]
            except Exception:
                th = strip_px // 2
            draw.text((lx, (strip_px - th) // 2), f"({LETTERS[i]})",
                      fill="black", font=font)

    else:
        # multi-panel: common width, strip + letter on each, then stack
        if orient == "row":
            H = max(im.height for im in pil_imgs)
            scaled = [im.resize((int(im.width * H / im.height), H)) for im in pil_imgs]
            strip_px = max(24, int(0.07 * H))
            font = _font(int(strip_px * 0.7))
            labelled = [_add_strip_and_letter(im, f"({LETTERS[i]})", strip_px,
                        LABEL_X_PX, font) for i, im in enumerate(scaled)]
            total_w = sum(im.width for im in labelled)
            out = Image.new("RGB", (total_w, H + strip_px), "white")
            x = 0
            for im in labelled:
                out.paste(im, (x, 0)); x += im.width
        else:  # col
            W = max(im.width for im in pil_imgs)
            padded = [_pad_to_width(im, W) for im in pil_imgs]
            strip_px = max(24, int(0.05 * max(im.height for im in padded)))
            font = _font(int(strip_px * 0.75))
            if spec.get("panel_x"):
                # several letters per source (a source holding internal panels)
                labelled, k = [], 0
                for im, fx in zip(padded, spec["panel_x"]):
                    out_i = Image.new("RGB", (im.width, im.height + strip_px), "white")
                    out_i.paste(im, (0, strip_px)); dr = ImageDraw.Draw(out_i)
                    for f_ in fx:
                        dr.text((int(f_ * im.width) + LABEL_X_PX, strip_px // 6),
                                f"({LETTERS[k]})", fill="black", font=font); k += 1
                    labelled.append(out_i)
            else:
                labelled = [_add_strip_and_letter(im, f"({LETTERS[i]})", strip_px,
                            LABEL_X_PX, font) for i, im in enumerate(padded)]
            total_h = sum(im.height for im in labelled)
            out = Image.new("RGB", (W, total_h), "white")
            y = 0
            for im in labelled:
                out.paste(im, (0, y)); y += im.height

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{name}.png"
    out.save(out_png, dpi=(dpi, dpi))
    print(f"  -> {out_png}  (from {', '.join(stems)})")
    if not assemble_pdf(name, spec, indir, outdir, labels, dpi):
        PDF_FAILED.append(name)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", default="figures_3d",
                    help="dir with the source F*.png panels")
    ap.add_argument("--out", dest="outdir", default="paper3d/figures",
                    help="dir for the assembled paper figures (PNG and PDF)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="assemble only these (e.g. fig3_accuracy fig6_recovery)")
    ap.add_argument("--labels", choices=["on", "off"], default="on",
                    help="stamp (a)/(b)/(c) letters (default on)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    labels = args.labels == "on"

    names = args.only if args.only else list(LAYOUTS.keys())
    print(f"Assembling {len(names)} figure(s) from {indir}/ -> {outdir}/")
    ok = 0
    for name in names:
        if name not in LAYOUTS:
            print(f"  [unknown] {name}")
            continue
        if assemble(name, LAYOUTS[name], indir, outdir, labels, args.dpi):
            ok += 1
    print(f"{ok}/{len(names)} assembled -> {outdir}/")
    if PDF_FAILED:
        raise SystemExit(f"no vector PDF for: {', '.join(PDF_FAILED)}")


if __name__ == "__main__":
    main()