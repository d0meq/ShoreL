"""
visualize.py
============
Produkty wynikowe projektu:
1. Mapa zmian       - wszystkie linie brzegowe naniesione kolorami wg roku,
                      z przekrojami pokolorowanymi wg tempa erozji/akumulacji.
2. Wykres EPR       - tempo zmian (m/rok) dla kolejnych przekrojow.
3. Animacja GIF     - ewolucja linii brzegowej w czasie.
"""

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # tryb bez okna (zapis do plikow)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_change_map(results, transects, epr_df, out_path):
    """Mapa: linie brzegowe (kolor = rok) + przekroje (kolor = EPR)."""
    fig, ax = plt.subplots(figsize=(10, 9))

    results = sorted(results, key=lambda r: r.year)
    years = [r.year for r in results]
    cmap_years = cm.get_cmap("viridis", len(years))

    # linie brzegowe
    for i, r in enumerate(results):
        x, y = r.line.xy
        ax.plot(x, y, color=cmap_years(i), lw=2, label=str(r.year))

    # przekroje pokolorowane wg EPR
    epr_vals = epr_df["EPR_m_yr"].to_numpy(dtype=float)
    finite = epr_vals[np.isfinite(epr_vals)]
    vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap_epr = cm.get_cmap("RdYlGn")  # czerwony=erozja, zielony=akumulacja

    for t in transects:
        e = epr_df["EPR_m_yr"].get(t["id"], np.nan)
        x, y = t["line"].xy
        color = cmap_epr(norm(e)) if np.isfinite(e) else "lightgray"
        ax.plot(x, y, color=color, lw=1.2, alpha=0.9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_epr)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("EPR [m/rok]   (< 0 erozja,  > 0 akumulacja)")

    ax.set_title("Mapa zmian linii brzegowej")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.legend(title="Rok", loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_epr_chart(epr_df, out_path):
    """Slupkowy wykres EPR dla kolejnych przekrojow."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ids = epr_df.index.to_numpy()
    vals = epr_df["EPR_m_yr"].to_numpy(dtype=float)
    colors = ["#d73027" if v < 0 else "#1a9850" for v in np.nan_to_num(vals)]
    ax.bar(ids, vals, color=colors, width=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Tempo zmian linii brzegowej (EPR) wg przekroju")
    ax.set_xlabel("Numer przekroju (wzdluz brzegu)")
    ax.set_ylabel("EPR [m/rok]")
    ax.grid(axis="y", alpha=0.3)

    mean_epr = np.nanmean(vals)
    ax.axhline(mean_epr, color="navy", ls="--", lw=1,
               label=f"srednia = {mean_epr:.2f} m/rok")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def make_animation(results, out_path, fps=1.5):
    """Animacja GIF pokazujaca przesuwanie sie linii brzegowej rok po roku."""
    results = sorted(results, key=lambda r: r.year)

    # wspolny zasieg dla wszystkich klatek
    xs, ys = [], []
    for r in results:
        x, y = r.line.xy
        xs += list(x)
        ys += list(y)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05

    frames = []
    for i, r in enumerate(results):
        fig, ax = plt.subplots(figsize=(7, 7))
        # poprzednie lata w tle (szare)
        for prev in results[:i]:
            px, py = prev.line.xy
            ax.plot(px, py, color="lightgray", lw=1)
        # biezacy rok wyrozniony
        x, y = r.line.xy
        ax.plot(x, y, color="#c0392b", lw=2.5)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal")
        ax.set_title(f"Linia brzegowa - rok {r.year}")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return out_path
