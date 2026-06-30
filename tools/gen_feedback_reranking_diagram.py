#!/usr/bin/env python3
"""Generate Figure 4 — Feedback Loop & Reranking (v5).

v4→v5 changes:
  1. All small text bumped +1–2 pt for print legibility
  2. Product Catalog removed entirely
  3. Session State: single-line "Name -- description" format, darker desc text
  4. Regenerate → "Regenerate Results" with "rescore candidates" label
  5. Legend removed
  6. User Feedback: verified uniform spacing
  7. Pipeline step 2: two lines (title + signal list on separate line)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────
TEAL      = "#2A9D8F"
TEAL_L    = "#E0F2F0"
GOLD      = "#D4A843"
GOLD_L    = "#FDF6E3"
PURPLE    = "#7B2D8E"
PURPLE_L  = "#EDE0F0"
RED_SOFT  = "#D95F4B"
GREY_BG   = "#F5F5F5"
GREY_BD   = "#BBBBBB"
WHITE     = "#FFFFFF"
DARK      = "#2D2D2D"
MID       = "#555555"
LIGHT     = "#777777"

FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "text.color": DARK})

# ── Canvas ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.2)
ax.axis("off")
fig.patch.set_facecolor(WHITE)

# ── Helpers ──────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.3, zorder=2, **kw):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.14",
                        fc=fc, ec=ec, lw=lw, zorder=zorder, **kw)
    ax.add_patch(b)

def arr(x1, y1, x2, y2, color=DARK, lw=1.6, style="-|>", zorder=3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)

def carr(x1, y1, x2, y2, rad=0.3, color=DARK, lw=1.6, zorder=3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle=f"arc3,rad={rad}",
                                color=color, lw=lw),
                zorder=zorder)

def T(x, y, s, sz=9, ha="center", va="center", wt="normal", c=DARK, ff=FONT, **kw):
    ax.text(x, y, s, fontsize=sz, ha=ha, va=va, fontweight=wt,
            color=c, fontfamily=ff, **kw)


# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════
T(5, 7.95, "Feedback Loop & Reranking", sz=15, wt="bold")


# ══════════════════════════════════════════════════════════════
#  NODE A — RECOMMENDED GALLERY  (top center)
# ══════════════════════════════════════════════════════════════
A_x, A_y, A_w, A_h = 2.0, 6.85, 6.0, 0.80
rbox(A_x, A_y, A_w, A_h, TEAL_L, TEAL, lw=2.0)
T(A_x + A_w/2, A_y + A_h/2 + 0.12,
  "Recommended Gallery", sz=12, wt="bold", c=TEAL)
T(A_x + A_w/2, A_y + A_h/2 - 0.18,
  "Each item shows a short, signal-grounded explanation",
  sz=8.5, c=MID)


# ══════════════════════════════════════════════════════════════
#  NODE B — USER FEEDBACK  (right side)
# ══════════════════════════════════════════════════════════════
B_x, B_y, B_w, B_h = 6.6, 4.10, 3.1, 2.30
rbox(B_x, B_y, B_w, B_h, GOLD_L, GOLD, lw=1.6)
T(B_x + B_w/2, B_y + B_h - 0.28,
  "User Feedback", sz=12, wt="bold", c="#9A7B1D")

# Three actions — uniform: 10pt bold name, 8pt desc, identical x + gap
actions = [
    ("Like",        "strengthen preference direction", TEAL),
    ("Add to Cart", "stronger preference signal",      TEAL),
    ("Dislike",     "suppress + extract avoid terms",   RED_SOFT),
]
act_left = B_x + 0.35
act_top  = B_y + B_h - 0.72
act_gap  = 0.55
for i, (name, desc, clr) in enumerate(actions):
    yy = act_top - i * act_gap
    T(act_left, yy, name, sz=10, ha="left", wt="bold", c=clr)
    T(act_left, yy - 0.22, desc, sz=8, ha="left", c=LIGHT)

# Arrow A → B
arr(A_x + A_w, A_y + A_h/2,
    B_x + B_w/2, B_y + B_h,
    color=GOLD, lw=1.5)
T(8.85, 6.70, "user acts\non items", sz=8, c="#9A7B1D", ha="center",
  style="italic")


# ══════════════════════════════════════════════════════════════
#  NODE C — SESSION STATE UPDATE  (bottom center)
#  Single-line format: "Name -- description"
# ══════════════════════════════════════════════════════════════
C_x, C_y, C_w, C_h = 2.5, 1.05, 5.0, 2.60
rbox(C_x, C_y, C_w, C_h, GREY_BG, GREY_BD, lw=1.4)
T(C_x + C_w/2, C_y + C_h - 0.25,
  "Session State Update", sz=11, wt="bold", c=DARK)

state_lines = [
    ("Preference vector",  "from likes and carts",              TEAL),
    ("Negative vectors",   "from dislikes",                     RED_SOFT),
    ("Intent vector",      "from selected A/B interpretation",  TEAL),
    ("Avoid terms",        "from LLM dislike analysis",         PURPLE),
    ("Seen set",           "for novelty tracking",              MID),
]
sl_left = C_x + 0.40
sl_top  = C_y + C_h - 0.62
sl_gap  = 0.36
for i, (name, desc, clr) in enumerate(state_lines):
    yy = sl_top - i * sl_gap
    T(sl_left, yy, name, sz=9, ha="left", wt="bold", c=clr)
    T(sl_left, yy - 0.20, desc, sz=8, ha="left", c=MID)

# Arrow B → C
arr(B_x + B_w/2, B_y,
    C_x + C_w, C_y + C_h,
    color=GOLD, lw=1.5)
T(7.45, 3.65, "updates state", sz=8, c="#9A7B1D", ha="center",
  style="italic")


# ── LLM Dislike Analysis  (attached to B→C path) ────────────
llm_x, llm_y, llm_w, llm_h = 7.5, 2.65, 2.2, 0.60
rbox(llm_x, llm_y, llm_w, llm_h, PURPLE_L, PURPLE, lw=1.0)
T(llm_x + llm_w/2, llm_y + llm_h/2 + 0.08,
  "LLM Dislike Analysis", sz=8.5, wt="bold", c=PURPLE)
T(llm_x + llm_w/2, llm_y + llm_h/2 - 0.16,
  "infers semantic avoid terms", sz=7.5, c=PURPLE)
# dislike area → LLM
carr(B_x + B_w/2 + 0.2, B_y + 0.10,
     llm_x + llm_w/2, llm_y + llm_h,
     rad=-0.15, color=PURPLE, lw=1.0)
# LLM → state
carr(llm_x, llm_y + llm_h/2,
     C_x + C_w + 0.05, C_y + C_h/2 + 0.35,
     rad=-0.35, color=PURPLE, lw=1.0)


# ══════════════════════════════════════════════════════════════
#  NODE D — RERANKING PIPELINE  (left side)
# ══════════════════════════════════════════════════════════════
D_x, D_y, D_w, D_h = 0.3, 4.10, 2.6, 2.30
rbox(D_x, D_y, D_w, D_h, TEAL_L, TEAL, lw=1.6)
T(D_x + D_w/2, D_y + D_h - 0.25,
  "Reranking Pipeline", sz=12, wt="bold", c=TEAL)

# Three-step pipeline
pipe_steps = [
    ("1", "Filter",  ["color gate + dislike ban"]),
    ("2", "Score",   ["multi-signal score",
                      "(color, preference, intent,",
                      "suppression, diversity, avoid, novelty)"]),
    ("3", "Select",  ["greedy top-k + LLM explain"]),
]
step_y = D_y + D_h - 0.60
for num, title, descs in pipe_steps:
    circ = plt.Circle((D_x + 0.32, step_y), 0.14,
                       fc=TEAL, ec="none", zorder=4)
    ax.add_patch(circ)
    T(D_x + 0.32, step_y, num, sz=9, wt="bold", c=WHITE, zorder=5)
    T(D_x + 0.58, step_y + 0.02, title, sz=9.5, ha="left", wt="bold", c=TEAL)
    for j, d in enumerate(descs):
        T(D_x + 0.58, step_y - 0.21 - j * 0.17, d,
          sz=7.5, ha="left", c=MID, va="top")
    step_y -= 0.50 + max(0, (len(descs) - 1) * 0.15)


# ── Regenerate Results  (on the State → Pipeline path) ───────
regen_cx = 2.50
regen_cy = 3.25

# arrow: state left → Regenerate pill
arr(C_x, C_y + C_h/2 + 0.20,
    regen_cx + 0.92, regen_cy,
    color=TEAL, lw=1.5)
# arrow: Regenerate pill → pipeline bottom
arr(regen_cx - 0.25, regen_cy + 0.08,
    D_x + D_w/2, D_y - 0.02,
    color=TEAL, lw=1.5)

rbox(regen_cx - 0.90, regen_cy - 0.25, 1.82, 0.50,
     WHITE, TEAL, lw=1.5)
T(regen_cx + 0.01, regen_cy + 0.02,
  "Regenerate Results", sz=9.5, wt="bold", c=TEAL)

# label on arrow from Regenerate to pipeline
T(1.10, 3.72, "rescore\ncandidates", sz=7.5, c=TEAL, ha="center",
  style="italic")


# ── Arrow D → A  (pipeline → gallery) ───────────────────────
arr(D_x + D_w/2, D_y + D_h,
    A_x, A_y + A_h/2,
    color=TEAL, lw=1.5)
T(1.10, 6.85, "new results", sz=8, c=TEAL, ha="center", style="italic")


# ── Save ─────────────────────────────────────────────────────
out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "feedback_reranking.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out / "feedback_reranking.png", bbox_inches="tight", dpi=300)
print("Done:  figures/feedback_reranking.pdf")
print("Done:  figures/feedback_reranking.png")
