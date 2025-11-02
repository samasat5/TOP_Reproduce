import os, glob, csv, argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# ---------- TB -> CSV ----------
def export_tb_scalars(logdir: str) -> pd.DataFrame:
    """Read all events files in a logdir and return a DataFrame of scalars."""
    rows = []
    pattern = os.path.join(logdir, "events.out.tfevents.*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {logdir}")

    for f in files:
        ea = event_accumulator.EventAccumulator(
            f, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        for tag in ea.Tags().get('scalars', []):
            for e in ea.Scalars(tag):
                rows.append({
                    "file": os.path.basename(f),
                    "tag": tag,
                    "step": e.step,
                    "wall_time": e.wall_time,
                    "value": e.value
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No scalar data found in: {logdir}")
    return df


def ema(x: np.ndarray, alpha: float = 0.1) -> np.ndarray: # Smoothing
    y, m = np.empty_like(x, dtype=float), 0.0
    for i, v in enumerate(x):
        m = alpha * v + (1 - alpha) * m if i else v
        y[i] = m
    return y


def reward_curve(df: pd.DataFrame, tag="Reward/Test") -> Tuple[np.ndarray, np.ndarray]:
    r = (df[df["tag"] == tag][["step", "value"]]
         .rename(columns={"value": "reward"})
         .sort_values("step")
         .drop_duplicates("step", keep="last"))
    return r["step"].to_numpy(), r["reward"].to_numpy()

def infer_label_from_tb(df: pd.DataFrame) -> Optional[str]:
    """Try to infer 'Optimistic (β=0)' or 'Pessimistic (β=-1)' from Distributions/optimism."""
    s = (df[df["tag"] == "Distributions/optimism"]
         .sort_values("step")["value"].round().astype(int))
    if s.empty:
        return None
    mode = s.mode().iloc[0]
    return "Optimistic (β=0)" if mode == 0 else "Pessimistic (β=-1)" if mode == -1 else f"β={mode}"

def reward_curves_by_mode(df: pd.DataFrame,
                          reward_tag="Reward/Test",
                          beta_tag="Distributions/optimism"):
    """
    Splits Reward/Test into two curves based on most recent beta (optimism) value.
    Returns a dict: { "Optimistic (β=0)": (steps, rewards), "Pessimistic (β=-1)": (...) }
    """
    # rewards
    r = (df[df["tag"] == reward_tag][["step", "value"]]
           .rename(columns={"value": "reward"})
           .sort_values("step")
           .drop_duplicates("step", keep="last"))
    # optimism values
    b = (df[df["tag"] == beta_tag][["step", "value"]]
           .rename(columns={"value": "beta"})
           .sort_values("step")
           .drop_duplicates("step", keep="last"))
    if r.empty or b.empty:
        return {}

    # align each reward step with the most recent beta (backward merge)
    merged = pd.merge_asof(r, b, on="step", direction="backward")
    merged = merged.dropna(subset=["beta"])
    merged["mode"] = merged["beta"].round().astype(int).map({
        -1: "Pessimistic (β=-1)",
         0: "Optimistic (β=0)"
    })

    curves = {}
    for name, g in merged.groupby("mode"):
        curves[name] = (g["step"].to_numpy(), g["reward"].to_numpy())
    return curves


def plot_runs(
    runs: List[Tuple[str, str]], 
    alpha: float,
    out_csv: Optional[str],
    out_png: Optional[str],
    title: str,
):
    paper_bg   = "#f0f2f5"
    grid_color = "#ffffff"
    colors = ["#f2a900", "#d14b4b", "#4c8bf5", "#7aa37a"]  

    plt.rcParams.update({
        "figure.figsize": (9.5, 6),
        "axes.facecolor": paper_bg,
        "axes.edgecolor": "none",
        "axes.grid": True,
        "grid.color": grid_color,
        "grid.linewidth": 1.2,
        "grid.alpha": 1.0,
        "font.size": 16,
    })

    plt.figure()
    all_export_rows = []

    for idx, (logdir, maybe_label) in enumerate(runs):
        df = export_tb_scalars(logdir)
        
        if out_csv:
            tmp = df.copy()
            tmp["source"] = os.path.basename(os.path.normpath(logdir))
            all_export_rows.append(tmp)

        # split automatically into optimism/pessimism curves
        curves = reward_curves_by_mode(df)
        if not curves:
            print(f"[warn] Could not split by mode in {logdir}, using global Reward/Test only.")
            x, y = reward_curve(df, tag="Reward/Test")
            y_s = ema(y, alpha=alpha) if alpha > 0 else y
            plt.plot(x, y_s, label=maybe_label or "Reward/Test", linewidth=3, color=colors[idx % len(colors)])
            continue

        for j, (lbl, (x, y)) in enumerate(curves.items()):
            y_s = ema(y, alpha=alpha) if alpha > 0 else y
            plt.plot(x, y_s, label=lbl, linewidth=3, color=colors[(idx + j) % len(colors)])

    plt.xlabel("Training steps")
    plt.ylabel("Reward (smoothed)" if alpha > 0 else "Reward")
    plt.title(title)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[ok] Saved figure: {out_png}")
    else:
        plt.show()

    if out_csv and all_export_rows:
        out = pd.concat(all_export_rows, ignore_index=True)
        out.to_csv(out_csv, index=False)
        print(f"[ok] Saved merged scalars CSV: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Export TB scalars and plot Reward/Test from one or more logdirs.")
    ap.add_argument("--logdir", nargs="+", required=True,
                    help="One or more TensorBoard log directories (each containing events.out.tfevents.*).")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional labels for each logdir; if omitted, inferred from TB or folder name.")
    ap.add_argument("--alpha", type=float, default=0.1, help="EMA smoothing factor in [0,1]. Use 0 to disable.")
    ap.add_argument("--out-csv", type=str, default=None, help="Optional path to save merged scalars CSV.")
    ap.add_argument("--out-png", type=str, default=None, help="Optional path to save the figure.")
    ap.add_argument("--title", type=str, default="TOP under fixed beta | HalfCheetah-v2  ",
                    help="Plot title.")
    args = ap.parse_args()

    labels = args.labels or [None] * len(args.logdir)
    if len(labels) != len(args.logdir):
        ap.error("If --labels is provided, it must have the same length as --logdir.")

    runs = list(zip(args.logdir, labels))
    plot_runs(runs, alpha=args.alpha, out_csv=args.out_csv, out_png=args.out_png, title=args.title)

if __name__ == "__main__":
    main()
