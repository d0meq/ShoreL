"""
ETAP 4 — Analiza EPR (End Point Rate)
Kroki:
  1. Wczytanie wszystkich linii brzegowych GeoJSON
  2. Wyznaczenie linii bazowej (baseline) równoległej do brzegu
  3. Generowanie transektów prostopadłych co SPACING metrów
  4. Wyznaczenie punktów przecięcia transekt × linia brzegowa
  5. Obliczenie EPR = (pozycja_ostatnia − pozycja_pierwsza) / liczba_lat
  6. Zapis wyników jako GeoJSON + wykres statystyczny
"""

from pathlib import Path
from datetime import datetime
import matplotlib.patches as mpatches

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely.ops import unary_union

# ─────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────
LINES_DIR  = Path("output/shorelines")
OUTPUT_DIR = Path("output")
MAPS_DIR   = OUTPUT_DIR / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

SPACING        = 100   # rozstaw transektów [m]
TRANSECT_HALF  = 500   # długość transektu w każdą stronę od baseline [m]
MIN_INTERSECTIONS = 2  # minimalna liczba scen żeby transekt był liczony


# ─────────────────────────────────────────
# KROK 1 — Wczytanie linii brzegowych
# ─────────────────────────────────────────
def load_shorelines(lines_dir: Path) -> list[dict]:
    files = sorted(lines_dir.glob("shoreline_*.geojson"))
    if not files:
        raise FileNotFoundError(f"Brak plików GeoJSON w {lines_dir}")

    shorelines = []
    for f in files:
        gdf = gpd.read_file(f)
        if gdf.empty:
            print(f"  ⚠ Pominięto pusty plik: {f.name}")
            continue

        parts    = f.stem.split("_")
        date_str = parts[1] if len(parts) > 1 else "19700101"
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            date = datetime(1970, 1, 1)

        geom = gdf.geometry.iloc[0]
        if geom.geom_type in ("LineString", "MultiLineString"):
            line = geom
        elif geom.geom_type == "GeometryCollection":
            parts_geom = [g for g in geom.geoms
                          if g.geom_type in ("LineString", "MultiLineString")]
            line = unary_union(parts_geom) if parts_geom else None
        else:
            line = None

        if line is None or line.is_empty:
            print(f"  ⚠ Brak linii w: {f.name}")
            continue

        year_frac = date.year + date.timetuple().tm_yday / 365.25
        shorelines.append({"file": f.name, "date": date,
                            "year": year_frac, "geom": line, "crs": gdf.crs})
        print(f"  ✓ {f.name}  →  {date.strftime('%Y-%m-%d')}")

    shorelines.sort(key=lambda x: x["date"])
    return shorelines


# ─────────────────────────────────────────
# KROK 2 — Linia bazowa
# ─────────────────────────────────────────
def build_baseline(shorelines: list[dict]) -> LineString:
    all_geoms = [s["geom"] for s in shorelines]
    merged    = unary_union(all_geoms)

    if merged.geom_type == "MultiLineString":
        longest = max(merged.geoms, key=lambda g: g.length)
    else:
        longest = merged

    baseline = longest.simplify(200, preserve_topology=True)
    print(f"  Baseline: {baseline.length/1000:.1f} km, "
          f"{len(list(baseline.coords))} wierzchołków")
    return baseline


# ─────────────────────────────────────────
# KROK 3 — Transekty
# ─────────────────────────────────────────
def generate_transects(baseline: LineString,
                       spacing: float, half_len: float) -> list[dict]:
    transects = []
    length    = baseline.length
    dist      = 0.0

    while dist <= length:
        pt    = baseline.interpolate(dist)
        dist2 = min(dist + 1.0, length)
        pt2   = baseline.interpolate(dist2)
        dx    = pt2.x - pt.x
        dy    = pt2.y - pt.y
        norm  = np.hypot(dx, dy)
        if norm < 1e-10:
            dist += spacing
            continue

        px = -dy / norm
        py =  dx / norm
        p1 = Point(pt.x + px * half_len, pt.y + py * half_len)
        p2 = Point(pt.x - px * half_len, pt.y - py * half_len)

        transects.append({
            "geom":     LineString([p1, p2]),
            "base_pt":  pt,
            "dist_km":  dist / 1000,
        })
        dist += spacing

    print(f"  Wygenerowano {len(transects)} transektów (co {spacing} m)")
    return transects


# ─────────────────────────────────────────
# KROK 4 — Przecięcia
# ─────────────────────────────────────────
def intersection_dist(transect_geom: LineString,
                      shore: LineString,
                      base_pt: Point) -> float | None:
    try:
        inter = transect_geom.intersection(shore)
    except Exception:
        return None

    if inter.is_empty:
        return None

    if inter.geom_type == "Point":
        pt = inter
    elif inter.geom_type == "MultiPoint":
        pt = inter.geoms[0]
    elif inter.geom_type == "GeometryCollection":
        pts = [g for g in inter.geoms if g.geom_type == "Point"]
        pt  = pts[0] if pts else None
    else:
        pt = None

    if pt is None:
        return None

    raw = base_pt.distance(pt)
    # Znak: strona transektu względem baseline
    d_start  = Point(transect_geom.coords[0]).distance(pt)
    d_base   = Point(transect_geom.coords[0]).distance(base_pt)
    sign     = 1.0 if d_start < d_base else -1.0
    return sign * raw


# ─────────────────────────────────────────
# KROK 5 — EPR
# ─────────────────────────────────────────
def calculate_epr(transects: list[dict],
                  shorelines: list[dict]) -> gpd.GeoDataFrame:
    records = []

    for t in transects:
        positions = {}
        for sl in shorelines:
            d = intersection_dist(t["geom"], sl["geom"], t["base_pt"])
            if d is not None:
                positions[sl["year"]] = d

        if len(positions) < MIN_INTERSECTIONS:
            continue

        years = sorted(positions)
        pos   = [positions[y] for y in years]
        dt    = years[-1] - years[0]
        epr   = (pos[-1] - pos[0]) / dt if dt > 0 else 0.0

        lrr = epr
        if len(years) >= 3:
            coeffs = np.polyfit(years, pos, 1)
            lrr    = float(coeffs[0])

        records.append({
            "dist_km":   round(t["dist_km"], 3),
            "epr_m_yr":  round(epr, 3),
            "lrr_m_yr":  round(lrr, 3),
            "n_obs":     len(positions),
            "pos_first": round(pos[0], 2),
            "pos_last":  round(pos[-1], 2),
            "geometry":  t["geom"],
        })

    return gpd.GeoDataFrame(records, crs=shorelines[0]["crs"])


# ─────────────────────────────────────────
# KROK 6 — Wykres
# ─────────────────────────────────────────
def plot_epr(gdf: gpd.GeoDataFrame, shorelines: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    date_range = (f"{shorelines[0]['date'].strftime('%Y-%m-%d')} → "
                  f"{shorelines[-1]['date'].strftime('%Y-%m-%d')}")
    fig.suptitle(f"Analiza EPR — Costa Brava\n{date_range}", fontsize=13)

    x   = gdf["dist_km"]
    epr = gdf["epr_m_yr"]
    colors = ["#e53935" if v < 0 else "#43a047" for v in epr]

    axes[0].bar(x, epr, width=SPACING / 1000 * 0.9, color=colors, alpha=0.85)
    axes[0].axhline(0,          color="black", lw=0.8, ls="--")
    axes[0].axhline(epr.mean(), color="navy",  lw=1.5, ls=":",
                    label=f"Średnia: {epr.mean():.2f} m/rok")
    axes[0].set_xlabel("Odległość wzdłuż brzegu [km]")
    axes[0].set_ylabel("EPR [m/rok]")
    axes[0].set_title("Tempo zmian linii brzegowej")
    axes[0].legend(handles=[
        mpatches.Patch(color="#e53935", label="Erozja (< 0)"),
        mpatches.Patch(color="#43a047", label="Akumulacja (> 0)"),
        plt.Line2D([0], [0], color="navy", ls=":",
                   label=f"Średnia: {epr.mean():.2f} m/rok"),
    ], fontsize=9)

    axes[1].hist(epr, bins=30, color="#546e7a", edgecolor="white", alpha=0.85)
    axes[1].axvline(0,            color="black",      lw=1.0, ls="--")
    axes[1].axvline(epr.mean(),   color="navy",       lw=1.5, ls=":",
                    label=f"Średnia: {epr.mean():.2f} m/rok")
    axes[1].axvline(epr.median(), color="darkorange", lw=1.5, ls="-.",
                    label=f"Mediana: {epr.median():.2f} m/rok")
    axes[1].set_xlabel("EPR [m/rok]")
    axes[1].set_ylabel("Liczba transektów")
    axes[1].set_title("Rozkład wartości EPR")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    out = MAPS_DIR / "epr_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Wykres: {out.name}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Etap 4 — Analiza EPR")
    print("=" * 55)

    print("\n[1] Wczytywanie linii brzegowych...")
    shorelines = load_shorelines(LINES_DIR)
    print(f"    Wczytano: {len(shorelines)} scen(y)")

    if len(shorelines) < 2:
        print("\n✗ Potrzebne minimum 2 linie z różnych dat.")
        print("  Pobierz więcej scen przez 01_download.py,")
        print("  przetwórz przez 02 i 03, a potem wróć tutaj.")
        return

    print("\n[2] Linia bazowa...")
    baseline = build_baseline(shorelines)

    print("\n[3] Transekty...")
    transects = generate_transects(baseline, SPACING, TRANSECT_HALF)

    print("\n[4+5] Obliczanie EPR...")
    gdf = calculate_epr(transects, shorelines)
    print(f"    Transektów z wynikiem: {len(gdf)}")

    if gdf.empty:
        print("✗ Brak wyników — linie mogą być zbyt krótkie lub nie przecinać transektów.")
        return

    epr = gdf["epr_m_yr"]
    print(f"\n  ── Statystyki EPR ──────────────────────")
    print(f"  Średnia   : {epr.mean():.2f} m/rok")
    print(f"  Mediana   : {epr.median():.2f} m/rok")
    print(f"  Min / Max : {epr.min():.2f} / {epr.max():.2f} m/rok")
    print(f"  Erozja    : {(epr < 0).sum()} transektów ({(epr < 0).mean()*100:.0f}%)")
    print(f"  Akumulacja: {(epr > 0).sum()} transektów ({(epr > 0).mean()*100:.0f}%)")
    print(f"  ────────────────────────────────────────")

    gdf.to_file(OUTPUT_DIR / "epr_transects.geojson", driver="GeoJSON")
    gdf.drop(columns="geometry").to_csv(OUTPUT_DIR / "epr_results.csv", index=False)
    gpd.GeoDataFrame(geometry=[baseline], crs=shorelines[0]["crs"]).to_file(
        OUTPUT_DIR / "baseline.geojson", driver="GeoJSON"
    )
    print(f"\n  ✓ epr_transects.geojson")
    print(f"  ✓ epr_results.csv")
    print(f"  ✓ baseline.geojson")

    print("\n[6] Wykres...")
    plot_epr(gdf, shorelines)

    print(f"\n{'='*55}")
    print("  ✓ Etap 4 zakończony.")
    print("  Następny krok: python src/05_visualize.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
