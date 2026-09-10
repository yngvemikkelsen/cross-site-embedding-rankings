#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pillow"]
# ///
"""Paper 19 Figure 2 - cross-site ranking transfer and variance decomposition.

Values are the corrected best-chunk primary analysis (contrastive 8-model panel),
matching Tables 3 and 5. Regenerated after the per-chunk instruction-prefix fix.

Usage:  python3 gen_fig2.py [-o figure2.png]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DPI = 140
GREEN = "#1b7a4b"
DARK = "#1f4e79"
LIGHT = "#7ba7d7"
TITLE = "#1f5c99"

# Panel A: genre, tau, CI low, CI high
TAU = [("Discharge summaries", 0.857, 0.643, 1.000),
       ("Imaging reports",     0.643, 0.357, 0.857)]

# Panel B: term, eta2, CI low, CI high, is_main_effect
ETA = [("Genre",                 0.563, 0.454, 0.633, True),
       ("Model",                 0.311, 0.242, 0.390, True),
       ("Model \u00d7 genre",    0.097, 0.073, 0.125, False),
       ("Model \u00d7 site \u00d7 genre", 0.011, 0.006, 0.026, False),
       ("Model \u00d7 site",     0.008, 0.004, 0.021, False),
       ("Site \u00d7 genre",     0.009, 0.000, 0.057, False),
       ("Site/corpus",           0.000, 0.000, 0.027, True)]


def main(out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.95), dpi=DPI,
                                   gridspec_kw={"height_ratios": [1, 1.55]})

    # ---------------- Panel A ----------------
    ys = list(range(len(TAU)))[::-1]
    for y, (_, tau, lo, hi) in zip(ys, TAU):
        ax1.plot([lo, hi], [y, y], color=GREEN, lw=2.6, solid_capstyle="round",
                 zorder=2)
        ax1.plot([tau], [y], "o", color=GREEN, ms=11, zorder=3)
        ax1.annotate("\u03c4 = %.2f" % tau, (tau, y), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=12.5,
                     fontweight="bold", color=GREEN)
    ax1.axvline(0, color="#c0392b", ls="--", lw=1.4, zorder=1)
    ax1.text(0.012, len(TAU) - 0.72, "\u03c4 = 0: no rank concordance",
             color="#c0392b", fontsize=10.5, va="center")
    ax1.set_yticks(ys)
    ax1.set_yticklabels([t[0] for t in TAU], fontsize=12.5)
    ax1.set_xlim(-0.02, 1.07)
    ax1.set_ylim(-0.62, len(TAU) - 0.32)
    ax1.set_xlabel("Cross-site Kendall \u03c4 (95% CI)", fontsize=12)
    ax1.set_title("(A) Cross-site ranking transfer by genre", fontsize=13.5,
                  fontweight="bold", color=TITLE, pad=12)
    ax1.grid(axis="x", color="#d9d9d9", lw=0.8)
    ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=11.5)

    # ---------------- Panel B ----------------
    ys2 = list(range(len(ETA)))[::-1]
    for y, (_, val, lo, hi, main_eff) in zip(ys2, ETA):
        c = DARK if main_eff else LIGHT
        ax2.plot([lo, hi], [y, y], color=c, lw=2.6, solid_capstyle="round",
                 zorder=2)
        ax2.plot([val], [y], "o", color=c, ms=11, zorder=3)
        ax2.annotate("%.3f" % val, (hi, y), textcoords="offset points",
                     xytext=(9, 0), va="center", fontsize=12)
    ax2.set_yticks(ys2)
    ax2.set_yticklabels([e[0] for e in ETA], fontsize=12.5)
    ax2.set_xlim(-0.008, 0.72)
    ax2.set_ylim(-0.65, len(ETA) - 0.35)
    ax2.set_xlabel("\u03b7\u00b2 of cell-level MRR@10 (95% CI)", fontsize=12)
    ax2.set_title("(B) Sources of variation in retrieval effectiveness",
                  fontsize=13.5, fontweight="bold", color=TITLE, pad=12)
    ax2.grid(axis="x", color="#d9d9d9", lw=0.8)
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=11.5)
    ax2.legend(handles=[Line2D([], [], color=DARK, lw=5, label="Main effect"),
                        Line2D([], [], color=LIGHT, lw=5, label="Interaction")],
               loc="lower right", fontsize=11.5, frameon=True, framealpha=1,
               edgecolor="#bfbfbf")

    fig.tight_layout(h_pad=2.4)
    fig.savefig(out, dpi=DPI, facecolor="white")
    # flatten to plain RGB and guarantee JMIR's 1200x1200 px maximum
    from PIL import Image
    im = Image.open(out)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1] if im.mode in ("RGBA", "LA") else None)
        im = bg
    else:
        im = im.convert("RGB")
    if max(im.size) > 1200:
        scale = 1200.0 / max(im.size)
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    im.save(out, format="PNG", optimize=True)
    print("wrote %s  %dx%d px  mode=%s" % (out, im.width, im.height, im.mode))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="figure2.png")
    main(ap.parse_args().out)
