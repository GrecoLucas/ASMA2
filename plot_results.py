"""
plot_results.py  —  Offline analysis of backflip training curves.

Reads the TensorBoard event files written to ./tb_logs/ and plots:
  - Episode reward (rollout/ep_rew_mean)
  - Flip success rate      (flip/success_rate)
  - Max rotation reached   (flip/max_rotation_deg)
  - Max height reached     (flip/max_height)
  - Episode length         (flip/episode_length)
  - SAC entropy coefficient (train/ent_coef)

Usage:
    python plot_results.py
    python plot_results.py --logdir ./tb_logs --smooth 50 --out results.png
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ──────────────────────────────────────────────────────────────
# 1. Load scalars from TensorBoard logs
# ──────────────────────────────────────────────────────────────

def load_all_scalars(logdir: str) -> dict[str, tuple[list, list]]:
    """
    Walks all subdirectories under logdir, loads every EventAccumulator,
    and merges all scalar tags into a single dict:
        { "tag": ([step, ...], [value, ...]) }
    Steps are sorted and de-duplicated.
    """
    # Find every directory that contains event files
    event_dirs = set()
    for path in glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True):
        event_dirs.add(os.path.dirname(path))

    if not logdir in event_dirs:
        # Also try the root itself
        if glob.glob(os.path.join(logdir, "events.out.tfevents.*")):
            event_dirs.add(logdir)

    if not event_dirs:
        raise FileNotFoundError(
            f"No TensorBoard event files found under '{logdir}'.\n"
            "Make sure training has started and tensorboard_log is set in the SAC constructor."
        )

    print(f"Found event data in {len(event_dirs)} director(y/ies):")
    for d in sorted(event_dirs):
        print(f"  {d}")

    merged: dict[str, dict[int, float]] = {}  # tag -> {step: value}

    for d in sorted(event_dirs):
        ea = EventAccumulator(d)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if tag not in merged:
                merged[tag] = {}
            for e in events:
                merged[tag][e.step] = e.value  # last write wins on duplicate steps

    # Convert to sorted lists
    result: dict[str, tuple[list, list]] = {}
    for tag, step_val in merged.items():
        pairs = sorted(step_val.items())
        result[tag] = ([p[0] for p in pairs], [p[1] for p in pairs])

    return result


# ──────────────────────────────────────────────────────────────
# 2. Smoothing helper
# ──────────────────────────────────────────────────────────────

def smooth(values: list[float], window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

def smooth_steps(steps: list[int], window: int) -> np.ndarray:
    if window <= 1 or len(steps) < window:
        return np.array(steps)
    return np.array(steps[window - 1:])


# ──────────────────────────────────────────────────────────────
# 3. Panel definitions
# ──────────────────────────────────────────────────────────────

PANELS = [
    # (title, y-label, [tb_tags_to_try_in_order], optional_horizontal_reference)
    ("Episode Reward",         "Mean reward",               ["rollout/ep_rew_mean"],          None),
    ("Flip Success Rate",      "Fraction (last 200 ep)",    ["flip/success_rate"],             None),
    ("Max Rotation Reached",   "Degrees (neg = backward)",  ["flip/max_rotation_deg"],         -360.0),
    ("Max Height Reached",     "Units above ground",        ["flip/max_height"],               None),
    ("Episode Length",         "Steps",                     ["flip/episode_length",
                                                              "rollout/ep_len_mean"],           None),
    ("SAC Entropy Coefficient","ent_coef",                  ["train/ent_coef"],                None),
]

COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8", "#80CBC4"]
DARK_BG   = "#0D1117"
PANEL_BG  = "#161B22"
GRID_CLR  = "#21262D"
BORDER    = "#30363D"
TEXT_DIM  = "#8B949E"
TEXT_MAIN = "#E6EDF3"


# ──────────────────────────────────────────────────────────────
# 4. Build the figure
# ──────────────────────────────────────────────────────────────

def make_figure(data: dict, smooth_window: int, out_path: str | None):
    n_cols = 2
    n_rows = (len(PANELS) + 1) // n_cols

    fig = plt.figure(figsize=(14, n_rows * 3.6), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.38)

    available_tags = sorted(data.keys())
    print(f"\nAvailable tags in log: {available_tags}\n")

    for idx, (title, ylabel, tags, href) in enumerate(PANELS):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.set_title(title, color=TEXT_MAIN, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Training steps", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, color=GRID_CLR, linewidth=0.7, linestyle="--", alpha=0.8)

        color = COLORS[idx % len(COLORS)]
        plotted = False

        for tag in tags:
            if tag in data:
                steps, values = data[tag]
                # Raw trace (very faint)
                ax.plot(steps, values, color=color, alpha=0.12, linewidth=0.7)
                # Smoothed trace
                sv = smooth(values, smooth_window)
                ss = smooth_steps(steps, smooth_window)
                ax.plot(ss, sv, color=color, linewidth=2.2,
                        label=f"{tag.split('/')[-1]} (smooth={smooth_window})")
                ax.legend(fontsize=7, labelcolor=TEXT_DIM,
                          facecolor=PANEL_BG, edgecolor=BORDER, loc="best")

                # Optional horizontal reference line (e.g. -360° = full backflip)
                if href is not None:
                    ax.axhline(href, color="#FF5252", linewidth=1.2,
                               linestyle="--", alpha=0.7, label="full flip")

                plotted = True
                break

        if not plotted:
            ax.text(0.5, 0.5,
                    f"No data yet\n({' / '.join(t.split('/')[-1] for t in tags)})",
                    ha="center", va="center", color="#484F58",
                    transform=ax.transAxes, fontsize=9)

    fig.suptitle("🤸 Backflip Agent — Training Progress",
                 color=TEXT_MAIN, fontsize=15, fontweight="bold", y=1.02)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Plot saved → {out_path}")
    else:
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────
# 5. Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot backflip training results from TensorBoard logs")
    parser.add_argument("--logdir", default="./tb_logs",
                        help="TensorBoard log directory (default: ./tb_logs)")
    parser.add_argument("--smooth", type=int, default=30,
                        help="Moving-average window (default: 30 data points)")
    parser.add_argument("--out",    default=None,
                        help="Save to file instead of showing interactively (e.g. results.png)")
    args = parser.parse_args()

    print(f"Loading scalars from '{args.logdir}' ...")
    data = load_all_scalars(args.logdir)
    print(f"Loaded {len(data)} scalar tag(s).")

    make_figure(data, smooth_window=args.smooth, out_path=args.out)


if __name__ == "__main__":
    main()
