import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.decomposition import NMF

np.random.seed(42)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    n_components: int = 3


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def make_matrix(s: pd.Series) -> pd.DataFrame:
    # Build year x month matrix of nonnegative values (shift to positive if needed)
    df = s.to_frame("value")
    minv = df["value"].min()
    if minv <= 0:
        df["value"] = df["value"] - minv + 1e-6
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="value")
    pivot = pivot.dropna()
    return pivot


def main():
    cfg = Config()
    s = load_series(cfg)
    M = make_matrix(s)

    nmf = NMF(
        n_components=cfg.n_components, init="nndsvda", random_state=42, max_iter=1000
    )
    W = nmf.fit_transform(M.values)
    H = nmf.components_
    recon = W @ H
    # Compute reconstruction MAE
    mae = float(np.mean(np.abs(M.values - recon)))
    print(f"NMF reconstruction MAE: {mae:.3f}")

    # Plot components (basis seasonal patterns)
    plt.figure(figsize=(9, 4))
    for k in range(cfg.n_components):
        plt.plot(range(1, 13), H[k], label=f"basis {k}")
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Component strength")
    plt.legend()
    save_fig("eia_mfle_components.png")

    # Plot actual vs reconstructed last year
    last_year = M.index.max()
    y_true = M.loc[last_year].values
    y_hat = recon[M.index.get_loc(last_year)]
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, 13), y_true, label="actual")
    plt.plot(range(1, 13), y_hat, label="reconstructed")
    plt.xticks(range(1, 13))
    plt.legend()
    save_fig("eia_mfle_reconstruction.png")


if __name__ == "__main__":
    main()
