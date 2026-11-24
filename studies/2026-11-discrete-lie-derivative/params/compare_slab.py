"""Figure and table of the discrete spectrum in the homogeneous slab (from eig_slab.py): the frequencies of the Bloch
blocks (n1, n2, n3) = (1, 0, n3) against the axial wave number, for N3 = 12 and 13, both models, with the exact
shear-Alfven, fast and slow branches. Branch of a discrete mode: shear Alfven if its velocity is mostly u2 (the wave
vector lies in the x-z plane, B0 = e_z), else fast or slow (u1, u3, p); modes with omega ~ 0 are the zero modes.

Writes slab_spectrum.png and slab_spectrum.md (beside this file or in $STUDY_RUNS).

Run: python compare_slab.py
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("STUDY_RUNS", HERE))
C = {"q": "#2a78d6", "var": "#eb6834"}
LAB = {"q": "q model (Pi^1)", "var": "variational (L2)"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                     "font.size": 10})


def load(model, nz):
    return json.load(open(RUNS / f"eig_slab_{model}_z{nz}_k10.json"))


def exact_curves(d, kz):
    L, beta, g = d["L"], d["beta"], d["gamma_ad"]
    kx = 2 * np.pi / L[0]
    k2 = kx**2 + kz**2
    cs2 = g * beta / 2
    s = k2 * (1 + cs2) / 2
    dd = np.sqrt(s**2 - k2 * kz**2 * cs2)
    return np.abs(kz), np.sqrt(s + dd), np.sqrt(np.maximum(s - dd, 0))


def modes(row):
    """(omega, is_alfven) of the modes with omega > 0 (each +-omega pair once) and the number of zero modes."""
    om = np.array(row["omega"])
    u2 = np.array(row["content"]["u_comp"])[:, 1] / np.maximum(np.array(row["content"]["u"]), 1e-300)
    pos = [(o, f > 0.5) for o, f in zip(om, u2) if o > 1e-6]
    return pos, int(np.sum(np.abs(om) < 1e-6))


fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
lines = ["Homogeneous slab, B0 = e_z, beta = 0.1, 8 x 8 x N3, block (n1, n2, n3) = (1, 0, n3): discrete omega / exact", "",
         "| N3 | n3 | model | shear Alfven | fast | slow | zero modes |", "|---|---|---|---|---|---|---|"]
for ax, nz in zip(axs, (12, 13)):
    d0 = load("var", nz)
    Lz = d0["L"][2]
    kf = np.linspace(0, 2 * np.pi * (nz // 2) / Lz, 200)
    ea, ef, es = exact_curves(d0, kf)
    for y, lab in ((ea, "exact shear Alfven"), (ef, "exact fast"), (es, "exact slow")):
        ax.plot(kf, y, "k-" if "Alfven" in lab else ("k--" if "fast" in lab else "k:"), lw=1, label=lab)
    cs = np.sqrt(d0["gamma_ad"] * d0["beta"] / 2)
    ax.plot(kf, cs * np.sqrt((2 * np.pi / d0["L"][0]) ** 2 + kf**2), "-", color="#bbb", lw=1,
            label="pure sound c_s |k| (no magnetic stiffness)")
    for model, dx in (("var", -0.012), ("q", 0.012)):
        d = load(model, nz)
        for row in d["rows"]:
            kz = 2 * np.pi * row["n"][2] / Lz
            pos, nzero = modes(row)
            for o, alf in pos:
                ax.plot(kz + dx, o, "o" if alf else "^", color=C[model], ms=5, mfc="none" if alf else C[model])
            if nzero > 1 and row["exact"]["alfven"] > 1e-9:  # one zero mode is physical (div b), the others spurious (k_z > 0)
                ax.plot(kz + dx, 0.0, "x", color=C[model], ms=9, mew=2)
                if model == "var":
                    ax.annotate("spurious omega = 0:\nshear Alfven and slow", (kz, 0.05), (kz - 0.75, 2.6),
                                fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
            ex = row["exact"]
            alf = [o for o, a_ in pos if a_]
            oth = sorted(o for o, a_ in pos if not a_)
            fast = oth[-1] if oth else 0.0
            slow = oth[0] if len(oth) > 1 else 0.0
            if row["n"][2] >= nz // 2 - 2:
                lines.append(f"| {nz} | {row['n'][2]} | {model} | {(alf[0] if alf else 0.0):.3f} / {ex['alfven']:.3f} | "
                             f"{fast:.3f} / {ex['fast']:.3f} | {slow:.3f} / {ex['slow']:.3f} | {nzero} |")
    for model in ("var", "q"):
        ax.plot([], [], "o", color=C[model], mfc="none", label=f"{LAB[model]}: shear Alfven (u2)")
        ax.plot([], [], "^", color=C[model], label=f"{LAB[model]}: fast / slow")
        ax.plot([], [], "x", color=C[model], mew=2, label=f"{LAB[model]}: spurious omega = 0")
    ax.axvline(np.pi * nz / Lz, color="#888", lw=0.8)
    ax.set(xlabel="axial wave number k_z = 2 pi n3 / Lz", title=f"N3 = {nz}" + (" (n3 = 6: alternating wave)" if nz == 12
                                                                                  else " (no alternating wave)"))
axs[0].set_ylabel("frequency omega")
axs[1].legend(fontsize=7, loc="upper left")
fig.suptitle("Discrete spectrum of the homogeneous slab (k_x = 2 pi): exact branches (black) vs discrete modes", fontsize=11)
fig.tight_layout()
fig.savefig(RUNS / "slab_spectrum.png", dpi=150, bbox_inches="tight")
open(RUNS / "slab_spectrum.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote slab_spectrum.png, slab_spectrum.md in", RUNS)
