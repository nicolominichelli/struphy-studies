"""Compare the runs of the study: the relative L2 deviation of the binned distributions from t=0, the change of the
distribution f(t_end) - f(0) in the poloidal plane and in (r, v_par), and a table (compare.md).

Run in the folder of the runs (or set STUDY_RUNS to it) after run_case.sh and the pproc of every run:
    python compare.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS_DIR = Path(os.environ.get("STUDY_RUNS", HERE))

# run -> label (in this order in the figures and the table)
RUNS = {
    "base": "base: canonical, shift=Bmax",
    "shift_none": "canonical, shift=None",
    "shift_b0": "canonical, shift=B0",
    "gyro": "gyro-Maxwellian (control)",
    "eps0.02": r"$\epsilon$ = 0.02",
    "eps0.1": r"$\epsilon$ = 0.1",
    "dg2": "discrete gradient 2nd order",
    "rk4": "explicit RK4",
}
# binned slices (with the binned quantity) -> label
SLICES = {
    "e1_e2_density": "poloidal plane",
    "e1_v1_density": r"$(r, v_\parallel)$",
    "v1_v2_density": r"$(v_\parallel, \mu)$",
    "e1_density": "r",
}


def _available():
    return {name: label for name, label in RUNS.items() if (RUNS_DIR / name / "results" / "time.npy").exists()}


def _load(name, quantity):
    path = RUNS_DIR / name / "results" / f"{quantity}.npy"
    return np.load(path) if path.exists() else None


# 1D slices of the deviation curves (they separate the runs; the 2D slices are dominated by the noise of small bins)
CURVE_SLICES = {"e1_density": "r", "v1_density": r"$v_\parallel$"}
# running median over this many saved times (removes the spikes of single markers next to the axis)
MEDIAN_WINDOW = 7
# time windows of the table: early and late medians of the deviation
EARLY, LATE = (5.0, 25.0), (75.0, 100.0)


def _running_median(values, window=MEDIAN_WINDOW):
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[k : k + window]) for k in range(len(values))])


def compare_deviation(runs):
    """Running median of the relative L2 deviation from t = 0 of the 1D slices, all runs."""
    colors = plt.cm.tab10(np.arange(len(runs)))
    fig, axs = plt.subplots(1, len(CURVE_SLICES), figsize=(6.2 * len(CURVE_SLICES), 4.4), squeeze=False)
    for ax, (slice_name, slice_label) in zip(axs[0], CURVE_SLICES.items()):
        for (name, label), color in zip(runs.items(), colors):
            dev = _load(name, f"deviation_{slice_name}")
            if dev is None:
                continue
            time = _load(name, "time")
            ax.plot(time[1:], _running_median(dev[1:]), color=color, label=label)
        ax.set_title(f"binned in {slice_label}")
        ax.set_xlabel("t")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
    axs[0][0].set_ylabel(f"relative L2 deviation from t = 0 (running median of {MEDIAN_WINDOW})")
    axs[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RUNS_DIR / "compare_deviation.png", dpi=110)
    plt.close(fig)


def compare_change(runs, views=("pol", "rv"), filename="compare_change.png", width=3.3, height=3.4):
    """f(t_end) - f(0) per run in the given views (poloidal plane, (r, v_par)), on one color scale per view."""
    from struphy import Output

    sys.path.insert(0, str(HERE))
    from pproc_tokamak_equilibrium import PERCENTILE, _Panels, _scale

    panels = {}
    for name in runs:
        out = Output(RUNS_DIR / name)
        out.pproc()
        panels[name] = _Panels(out)
    labels = {"pol": "poloidal plane", "rv": r"$(r, v_\parallel)$"}
    views = [(view, labels[view]) for view in views]

    def change(p, view):
        f = getattr(p, view)[0]
        return (f[-1] - f[0]) / _scale(f[0])

    limits = {view: max(_scale(change(p, view)) for p in panels.values()) for view, _ in views}
    fig, axs = plt.subplots(len(views), len(runs), figsize=(width * len(runs), height * len(views)), squeeze=False)
    for col, (name, label) in enumerate(runs.items()):
        p = panels[name]
        for row, (view, view_label) in enumerate(views):
            ax = axs[row][col]
            im = p.draw(ax, view, change(p, view), "RdBu_r", -limits[view], limits[view], label)
            if col > 0:
                ax.set_ylabel("")
            if row == 0:
                ax.set_xlabel("")
        for row in range(len(views)):
            if col == len(runs) - 1:
                fig.colorbar(axs[row][col].collections[0], ax=axs[row].tolist(), shrink=0.85)
    for row, (_, view_label) in enumerate(views):
        axs[row][0].annotate(
            view_label, xy=(-0.45, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=11
        )
    fig.suptitle(
        f"f(t_end) - f(0), normalized to the {PERCENTILE}th percentile of f(0), common color scale per row",
        fontsize=11,
    )
    fig.savefig(RUNS_DIR / filename, dpi=110, bbox_inches="tight")
    plt.close(fig)


def table(runs):
    """Medians of the deviation (1D slices) at early and late times, energy drift and lost markers (compare.md)."""
    head = "| run | " + " | ".join(f"dev. {label} t = {EARLY[0]:g}-{EARLY[1]:g} / {LATE[0]:g}-{LATE[1]:g}" for label in ("r", "v_par"))
    lines = [head + " | max energy drift | lost markers |", "|---|---|---|---|---|"]
    for name in runs:
        time = _load(name, "time")[1:]
        cells = []
        for slice_name in CURVE_SLICES:
            dev = _load(name, f"deviation_{slice_name}")[1:]
            early = np.median(dev[(time > EARLY[0]) & (time <= EARLY[1])])
            late = np.median(dev[(time >= LATE[0]) & (time <= LATE[1])])
            cells.append(f"{early:.3f} / {late:.3f} ({100 * (late / early - 1):+.0f}%)")
        en = _load(name, "en_tot")
        drift = f"{np.max(np.abs(en - en[0])) / abs(en[0]):.1e}"
        lost = int(_load(name, "n_lost_particles")[-1])
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {drift} | {lost} |")
    text = "\n".join(lines) + "\n"
    (RUNS_DIR / "compare.md").write_text(text)
    print(text)


if __name__ == "__main__":
    runs = _available()
    print(f"runs: {list(runs)}")
    compare_deviation(runs)
    compare_change(runs)
    # the key figure: exact equilibrium (base) vs shift B0 vs gyro-Maxwellian, in (r, v_par)
    key = {name: runs[name] for name in ("base", "shift_b0", "gyro") if name in runs}
    compare_change(key, views=("rv",), filename="compare_change_rv.png", width=4.6, height=4.6)
    table(runs)
