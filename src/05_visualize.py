"""
ETAP 5 — Wizualizacja wyników
Produkty:
  A. Mapa zmian EPR — kartograficzna mapa stref erozji/akumulacji
  B. Animacja GIF — ewolucja linii brzegowej w czasie
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
import imageio.v2 as imageio
from shapely.ops import unary_union

# ─────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────
OUTPUT_DIR = Path("output")
LINES_DIR  = OUTPUT_DIR / "shorelines"
MAPS_DIR   = OUTPUT_DIR / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

GIF_FPS     = 1.5    # klatki na sekundę
GIF_DPI     = 120    # rozdzielczość klatek GIF


# ─────────────────────────────────────────
# POMOCNICZE
# ─────────────────────────────────────────
def load_shorelines(lines_dir: Path) -> list[dict]:
    files = sorted(lines_dir.glob("shoreline_*.geojson"))
    result = []
    for f in files:
        gdf = gpd.read_file(f)
        if gdf.empty:
            continue
        parts    = f.stem.split("_")
        date_str = parts[1] if len(parts) > 1 else "19700101"
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            date = datetime(1970, 1, 1)
        result.append({"date": date, "geom": gdf.geometry.iloc[0], "crs": gdf.crs})
    result.sort(key=lambda x: x["date"])
    return result


# ─────────────────────────────────────────
# PRODUKT A — Mapa zmian EPR
# ─────────────────────────────────────────
def map_epr(shorelines: list[dict]) -> None:
    """Mapa kartograficzna z transektami EPR pokolorowanymi wg tempa zmian."""

    epr_path = OUTPUT_DIR / "epr_transects.geojson"
    base_path = OUTPUT_DIR / "baseline.geojson"

    if not epr_path.exists():
        print("  ⚠ Brak epr_transects.geojson — uruchom najpierw 04_analysis_epr.py")
        return

    gdf_epr  = gpd.read_file(epr_path)
    gdf_base = gpd.read_file(base_path) if base_path.exists() else None

    fig, ax = plt.subplots(figsize=(14, 9))

    # Tło — wszystkie linie brzegowe
    colors_shore = cm.Blues(np.linspace(0.3, 0.9, len(shorelines)))
    for sl, col in zip(shorelines, colors_shore):
        geom = sl["geom"]
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                xs, ys = line.xy
                ax.plot(xs, ys, color=col, linewidth=0.6, alpha=0.5)
        elif geom.geom_type == "LineString":
            xs, ys = geom.xy
            ax.plot(xs, ys, color=col, linewidth=0.6, alpha=0.5)

    # Baseline
    if gdf_base is not None and not gdf_base.empty:
        bgeom = gdf_base.geometry.iloc[0]
        if hasattr(bgeom, "xy"):
            ax.plot(*bgeom.xy, color="black", linewidth=1.2,
                    linestyle="--", label="Baseline", zorder=3)

    # Transekty pokolorowane wg EPR
    epr_vals = gdf_epr["epr_m_yr"]
    vmax     = max(abs(epr_vals.min()), abs(epr_vals.max()), 0.5)
    norm     = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap     = cm.RdYlGn

    for _, row in gdf_epr.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        color = cmap(norm(row["epr_m_yr"]))
        xs, ys = geom.xy
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8, zorder=4)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("EPR [m/rok]", fontsize=10)

    # Legenda linii brzegowych
    handles = []
    for sl, col in zip(shorelines[::max(1, len(shorelines)//5)], colors_shore):
        handles.append(mpatches.Patch(color=col,
                                      label=sl["date"].strftime("%Y-%m-%d")))
    handles.append(plt.Line2D([0], [0], color="black", ls="--", label="Baseline"))
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              title="Linia brzegowa", title_fontsize=9)

    ax.set_title("Mapa zmian linii brzegowej — Mierzeja Wiślana\n"
                 "(zielony = akumulacja, czerwony = erozja)", fontsize=12)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_aspect("equal")

    plt.tight_layout()
    out = MAPS_DIR / "mapa_zmian_epr.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Mapa EPR: {out.name}")


# ─────────────────────────────────────────
# PRODUKT B — Animacja GIF
# ─────────────────────────────────────────
def animate_shorelines(shorelines: list[dict]) -> None:
    """Animacja poklatkowa ewolucji linii brzegowej."""

    if len(shorelines) < 2:
        print("  ⚠ Potrzebne co najmniej 2 sceny do animacji.")
        return

    # Wspólny extent (wszystkich linii)
    all_bounds = []
    for sl in shorelines:
        geom = sl["geom"]
        if not geom.is_empty:
            all_bounds.append(geom.bounds)

    if not all_bounds:
        print("  ⚠ Brak geometrii do animacji.")
        return

    x_min = min(b[0] for b in all_bounds)
    y_min = min(b[1] for b in all_bounds)
    x_max = max(b[2] for b in all_bounds)
    y_max = max(b[3] for b in all_bounds)
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    frames_dir = MAPS_DIR / "_frames"
    frames_dir.mkdir(exist_ok=True)
    frame_paths = []

    colors = cm.viridis(np.linspace(0, 1, len(shorelines)))

    for i, sl in enumerate(shorelines):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Wszystkie poprzednie linie (wygaszone)
        for j in range(i):
            geom = shorelines[j]["geom"]
            col  = (*colors[j][:3], 0.25)   # przezroczysty
            _plot_line(ax, geom, color=col, lw=0.8)

        # Bieżąca linia (pełna, pogrubiona)
        _plot_line(ax, sl["geom"], color=colors[i], lw=2.0)

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal")
        ax.set_title(f"Linia brzegowa — Mierzeja Wiślana\n"
                     f"{sl['date'].strftime('%d %B %Y')}",
                     fontsize=12)
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

        # Legenda dat
        handles = [
            mpatches.Patch(color=colors[j],
                           label=shorelines[j]["date"].strftime("%Y-%m-%d"),
                           alpha=0.4 if j < i else 1.0)
            for j in range(i + 1)
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")

        frame_path = frames_dir / f"frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=GIF_DPI, bbox_inches="tight")
        plt.close()
        frame_paths.append(frame_path)
        print(f"  Klatka {i+1}/{len(shorelines)}: {sl['date'].strftime('%Y-%m-%d')}")

    # Złożenie GIF
    out_gif = OUTPUT_DIR / "animacja_linii_brzegowej.gif"
    images  = [imageio.imread(str(p)) for p in frame_paths]
    imageio.mimsave(str(out_gif), images, fps=GIF_FPS, loop=0)
    print(f"  ✓ Animacja GIF: {out_gif}")

    # Usuń klatki tymczasowe
    for p in frame_paths:
        p.unlink()
    frames_dir.rmdir()


def _plot_line(ax, geom, color, lw):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        ax.plot(*geom.xy, color=color, linewidth=lw)
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            ax.plot(*g.xy, color=color, linewidth=lw)
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            if g.geom_type in ("LineString", "MultiLineString"):
                _plot_line(ax, g, color, lw)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Etap 5 — Wizualizacja wyników")
    print("=" * 55)

    print("\nWczytywanie linii brzegowych...")
    shorelines = load_shorelines(LINES_DIR)
    print(f"  Wczytano: {len(shorelines)} scen(y)\n")

    if not shorelines:
        print("✗ Brak linii brzegowych. Uruchom etap 3.")
        return

    print("[A] Generowanie mapy zmian EPR...")
    map_epr(shorelines)

    print("\n[B] Generowanie animacji GIF...")
    animate_shorelines(shorelines)

    print(f"\n{'='*55}")
    print("  ✓ Etap 5 zakończony. Projekt gotowy!")
    print(f"\n  Pliki wynikowe w: {OUTPUT_DIR.resolve()}")
    print("   • output/maps/mapa_zmian_epr.png")
    print("   • output/animacja_linii_brzegowej.gif")
    print("   • output/epr_results.csv")
    print("   • output/epr_transects.geojson")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
