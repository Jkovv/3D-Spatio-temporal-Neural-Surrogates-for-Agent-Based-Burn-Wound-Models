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
    # Fig 2 = architectures, drawn by hand. This reserves a blank, correctly-
    # sized page so numbering and layout stay consistent with the rest.
    "fig2_architectures": {
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
    "fig7_recovery": {
        "orient": "row",
        "sources": ["F5_recovery"],
        "width": 12.0,
        "wratios": [1.0],
        # F5_recovery is a single PNG that already holds 3 internal panels.
        # Stamp 3 letters at these x-fractions; label_y raised so they clear
        # the panel titles.
        "label_x": [0.02, 0.36, 0.69],
        "label_y": 1.005,
    },
}

LETTERS = "abcdefgh"


def _imread(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(str(path))


def assemble(name, spec, indir, outdir, labels=True, dpi=300):
    stems = spec["sources"]

    # blank reserved-space case (e.g. hand-drawn architectures figure) 
    if not stems:
        width = spec["width"]
        height = spec.get("blank_height", 5.0)
        fig, ax = plt.subplots(figsize=(width, height))
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False,
                     ls="--", ec="0.6", lw=1.2, transform=ax.transAxes))
        ax.text(0.5, 0.5, "architecture diagram\n(placeholder — hand-drawn)",
                ha="center", va="center", fontsize=13, color="0.5",
                style="italic", transform=ax.transAxes)
        outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"{name}.{ext}", dpi=dpi,
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  -> {outdir / f'{name}.pdf'}  (blank placeholder)")
        return True

    paths = [indir / f"{s}.png" for s in stems]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"  [skip {name}] missing: {', '.join(p.name for p in missing)}")
        return False

    # load 
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
        # multi-panel: normalise to common width, add a strip + letter to each,
        # then stack. Because every panel is the same width and the letter sits
        # at the same pixel x, the letters line up perfectly under one another.
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
            labelled = [_add_strip_and_letter(im, f"({LETTERS[i]})", strip_px,
                        LABEL_X_PX, font) for i, im in enumerate(padded)]
            total_h = sum(im.height for im in labelled)
            out = Image.new("RGB", (W, total_h), "white")
            y = 0
            for im in labelled:
                out.paste(im, (0, y)); y += im.height

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{name}.png"
    out_pdf = outdir / f"{name}.pdf"
    out.save(out_png, dpi=(dpi, dpi))
    out.save(out_pdf, "PDF", resolution=float(dpi))
    print(f"  -> {out_pdf}  (from {', '.join(stems)})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", default="figures_3d",
                    help="dir with the source F*.png panels")
    ap.add_argument("--out", dest="outdir", default="paper3d/figures",
                    help="dir for the assembled paper figures (PDF+PNG)")
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
    print(f"Done: {ok}/{len(names)} assembled -> {outdir}/")
    if ok:
        print("\nDrop into paper.tex, replacing each \\panelplaceholder{...}{...} with:")
        print("  \\includegraphics[width=\\linewidth]{figures/<name>.pdf}")


if __name__ == "__main__":
    main()