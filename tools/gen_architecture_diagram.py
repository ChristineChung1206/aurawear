#!/usr/bin/env python3
"""
AuraWear system overview diagram — publication-ready (v4).
Output: figures/system_architecture.pdf + .png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial"],
    "font.size": 10,
    "axes.linewidth": 0,
})

P = {
    "bg":       "#FAFCFD",
    "main":     "#E0F2F1",  "main_e":  "#00796B",  "main_t": "#004D40",
    "user":     "#FFF8E1",  "user_e":  "#F9A825",  "user_t": "#E65100",
    "tech":     "#F5F5F5",  "tech_e":  "#BDBDBD",  "tech_t": "#616161",
    "flow":     "#00695C",
    "feed":     "#7B1FA2",  "feed_bg": "#F3E5F5",
    "title":    "#263238",
}

fig, ax = plt.subplots(1, 1, figsize=(13.5, 6.6))
fig.patch.set_facecolor(P["bg"])
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 6.6)
ax.set_aspect("equal")
ax.axis("off")

# ── Helpers ─────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, text, fs=10, bold=False,
         tc="#222", lw=1.5, rad=0.15, z=2, va="center"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va=va,
                fontsize=fs, color=tc, weight="bold" if bold else "normal",
                zorder=z+1, linespacing=1.35)

def pill(x, y, w, text, fs=7.5):
    rbox(x, y, w, 0.30, P["tech"], P["tech_e"], text,
         fs=fs, tc=P["tech_t"], lw=0.7, rad=0.11, z=3)

def arr(x1, y1, x2, y2, c=None, lw=2.0, sty="-|>",
        conn="arc3,rad=0", sA=4, sB=4, z=4):
    c = c or P["flow"]
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=sty, color=c, lw=lw,
        connectionstyle=conn, shrinkA=sA, shrinkB=sB, zorder=z))

def lbl(x, y, text, c=None, fs=8):
    c = c or P["flow"]
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=c, style="italic", zorder=5)


# ================================================================
#  TITLE
# ================================================================
ax.text(6.75, 6.25, "AuraWear Overview",
        ha="center", va="center", fontsize=15, weight="bold",
        color=P["title"], zorder=5)


# ================================================================
#  MAIN FLOW — 5 boxes
# ================================================================
W, H = 2.05, 1.15
Y = 3.95
GAP = 0.35
xs = [0.35 + i * (W + GAP) for i in range(5)]

defs = [
    ("Selfie Upload\n+ Initial Style Prompt",    None,                                     False),
    ("Personal Color\nAnalysis",                  "12-season type\nfrom facial colors",      False),
    ("Palette\nCuration",                         "User selects colors\nfrom seasonal palette", True),
    ("Intent\nClarification",                     "A/B intent options\nUser selects one",    True),
    ("Recommendation\n+ Explanations",            "Ranked recommendations\nItem-level explanations", False),
]

for i, (title, sub, is_u) in enumerate(defs):
    fc = P["user"]   if is_u else P["main"]
    ec = P["user_e"] if is_u else P["main_e"]
    tc = P["user_t"] if is_u else P["main_t"]
    rbox(xs[i], Y, W, H, fc, ec, title, fs=11, bold=True,
         tc=tc, lw=2.0 if is_u else 1.5)
    if sub:
        ax.text(xs[i] + W/2, Y - 0.15, sub, ha="center", va="top",
                fontsize=8, color="#555", linespacing=1.3, zorder=5)

# Step circles
for i in range(5):
    cx, cy = xs[i] + W/2, Y + H + 0.22
    ax.add_patch(plt.Circle((cx, cy), 0.16,
                 fc=P["main_e"], ec="white", lw=1.5, zorder=5))
    ax.text(cx, cy, str(i+1), ha="center", va="center",
            fontsize=9, weight="bold", color="white", zorder=6)

# USER badges
for i in [2, 3]:
    ax.text(xs[i]+W-0.15, Y+H-0.15, "USER", ha="center", va="center",
            fontsize=6, weight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc=P["user_e"],
                      ec="none", alpha=0.9), zorder=6)

# Main flow arrows
for i in range(4):
    arr(xs[i]+W, Y+H/2, xs[i+1], Y+H/2, lw=2.3)


# ================================================================
#  TECH PILLS
# ================================================================
py = 2.90
pill(xs[1]+0.05,  py, 1.9,  "BiSeNet  |  ONNX Runtime")
pill(xs[2]+0.05,  py, 1.9,  "18-color seasonal palette")
pill(xs[3]-0.15,  py, 2.30, "GPT-4.1-mini  |  CLIP ViT-B-32")
pill(xs[4]+0.02,  py, 2.0,  "CIE Lab  |  CLIP similarity")


# ================================================================
#  PRODUCT CATALOG  (#3 — tighter to Step 5, clear arrow)
# ================================================================
cat_x = xs[4] + 0.25
cat_y = Y + H + 0.55
rbox(cat_x, cat_y, 1.55, 0.42, P["tech"], P["tech_e"],
     "Product Catalog\n12,701 items", fs=8, tc=P["tech_t"],
     lw=0.8, rad=0.10, z=3)
# Arrow from catalog down to Step 5 top
arr(cat_x + 0.77, cat_y, xs[4] + W/2, Y + H,
    c=P["tech_e"], lw=1.3, sA=2, sB=4)


# ================================================================
#  FEEDBACK LOOP  (#1 — three even columns)
# ================================================================
fb_x  = xs[1] - 0.1
fb_y  = 1.35
fb_w  = 9.4
fb_h  = 1.05

# Background
rbox(fb_x, fb_y, fb_w, fb_h, P["feed_bg"], P["feed"],
     "", fs=1, lw=1.5, rad=0.15, z=2)

# Title
ax.text(fb_x + fb_w/2, fb_y + fb_h - 0.18, "Feedback Loop",
        ha="center", va="top", fontsize=10, weight="bold",
        color=P["feed"], zorder=3)

# USER badge
ax.text(fb_x + fb_w - 0.35, fb_y + fb_h - 0.16, "USER",
        ha="center", va="center",
        fontsize=6, weight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.13", fc=P["feed"],
                  ec="none", alpha=0.9), zorder=6)

# Three action columns — evenly spaced
col_w = fb_w / 3
actions = [
    "Like\nupdate preference",
    "Dislike\nsuppress + avoid terms",
    "Cart\nstronger preference update",
]
for j, act in enumerate(actions):
    col_cx = fb_x + col_w * j + col_w / 2
    # Split into label + description
    parts = act.split("\n")
    # Action name (bold)
    ax.text(col_cx, fb_y + 0.42, parts[0],
            ha="center", va="center", fontsize=9,
            weight="bold", color=P["feed"], zorder=3)
    # Arrow symbol + description
    ax.text(col_cx, fb_y + 0.17, parts[1],
            ha="center", va="center", fontsize=7.5,
            color=P["feed"], zorder=3, alpha=0.85)

# Thin vertical separators between columns
for j in [1, 2]:
    sep_x = fb_x + col_w * j
    ax.plot([sep_x, sep_x], [fb_y + 0.08, fb_y + fb_h - 0.30],
            color=P["feed"], lw=0.5, alpha=0.3, zorder=3)


# ================================================================
#  FEEDBACK ARROWS  (#2, #6 — unified, thinner, better placed)
# ================================================================
FEED_LW   = 1.2           # #6 — thinner than before
FEED_FS   = 8             # #2 — same font size for all labels

# (a) Step 5 -> Feedback: "provide feedback"
reco_cx = xs[4] + W/2
arr(reco_cx, Y, reco_cx, fb_y + fb_h,
    c=P["feed"], lw=FEED_LW)
lbl(reco_cx + 0.65, (Y + fb_y + fb_h)/2 + 0.15,
    "provide\nfeedback", c=P["feed"], fs=FEED_FS)

# (b) Feedback -> Palette: "update preferences"  (#6 — gentler arc)
arr(fb_x + 0.3, fb_y + fb_h * 0.4,
    xs[2] + W/2, Y,
    c=P["feed"], lw=FEED_LW, conn="arc3,rad=-0.20")
lbl(xs[1] + 0.3, 2.62, "update\npreferences", c=P["feed"], fs=FEED_FS)

# (c) Feedback -> Intent: "refine intent"
intent_x = xs[3] + W/2 + 0.4
arr(intent_x, fb_y + fb_h,
    intent_x, Y,
    c=P["feed"], lw=FEED_LW)
lbl(intent_x + 0.55, 2.62, "refine\nintent", c=P["feed"], fs=FEED_FS)


# ================================================================
#  ANNOTATION
# ================================================================
ax.text(6.75, 0.65,
        "Iterative co-decision: users can revise palette, "
        "clarify intent, and update recommendations through feedback.",
        ha="center", va="center", fontsize=8, color="#666",
        style="italic", zorder=5)


# ================================================================
#  LEGEND  (#7 — even smaller, minimal)
# ================================================================
lx0, ly = 10.5, 0.18
for i, (fc, ec, lab) in enumerate([
    (P["main"], P["main_e"], "System stage"),
    (P["user"], P["user_e"], "User decision"),
    (P["feed_bg"], P["feed"], "Feedback loop"),
]):
    lx = lx0 + i * 1.05
    ax.add_patch(FancyBboxPatch(
        (lx, ly), 0.15, 0.12,
        boxstyle="round,pad=0,rounding_size=0.04",
        fc=fc, ec=ec, lw=0.6, zorder=2))
    ax.text(lx + 0.22, ly + 0.06, lab,
            fontsize=6, va="center", color="#666", zorder=3)


# ── Save ────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/system_architecture.{ext}",
                bbox_inches="tight", dpi=300, facecolor=P["bg"])
    print(f"\u2705 figures/system_architecture.{ext}")
plt.close(fig)
