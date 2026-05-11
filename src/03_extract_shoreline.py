"""
ETAP 3 — Ekstrakcja linii brzegowej z obrazów SAR
Kroki:
  1. Wczytanie przetworzonego rastra sigma0 [dB]
  2. Progowanie Otsu — automatyczny podział na wodę i ląd
  3. Morfologia matematyczna — czyszczenie maski binarnej
  4. Wektoryzacja — konwersja rastrowej granicy na linię wektorową
  5. Filtrowanie — zachowanie tylko głównej linii brzegowej
  6. Zapis jako GeoJSON + wizualizacja przycięta do AOI
"""

import re
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from shapely.geometry import shape, LineString
from shapely.ops import unary_union
from scipy.ndimage import (
    binary_closing, binary_opening,
    binary_fill_holes, label
)
from pyproj import Transformer

# ─────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("output")
MAPS_DIR      = OUTPUT_DIR / "maps"
LINES_DIR     = OUTPUT_DIR / "shorelines"

for d in [MAPS_DIR, LINES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Próg dB — None = automatyczny Otsu
MANUAL_THRESHOLD = -17.0

# Minimalna powierzchnia obszaru lądowego [piksele]
MIN_LAND_AREA = 10_000

# Obszar zainteresowania — Costa Brava: Platja d'Aro / Palamós (WGS84)
AOI_WGS84 = {
    "lon_min": 3.05, "lon_max": 3.22,
    "lat_min": 41.78, "lat_max": 41.90,
}


# ─────────────────────────────────────────
# POMOCNICZE — konwersja AOI → piksele
# ─────────────────────────────────────────
def aoi_to_pixel_window(src: rasterio.DatasetReader) -> tuple[int, int, int, int]:
    """
    Konwertuje AOI w WGS84 na okno pikselowe w rastrze src.
    Używa rasterio.windows.from_bounds — poprawna metoda bez ręcznej inwersji.
    Zwraca (row_min, row_max, col_min, col_max).
    """
    from rasterio.windows import from_bounds as window_from_bounds

    try:
        crs_epsg    = src.crs.to_epsg()
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True)

        x_min, y_min = transformer.transform(AOI_WGS84["lon_min"], AOI_WGS84["lat_min"])
        x_max, y_max = transformer.transform(AOI_WGS84["lon_max"], AOI_WGS84["lat_max"])

        print(f"\n     Raster bounds (UTM): {src.bounds}")
        print(f"     AOI bounds   (UTM): x={x_min:.0f}–{x_max:.0f}, y={y_min:.0f}–{y_max:.0f}")

        # Sprawdź czy AOI pokrywa się z rasterem
        rb = src.bounds
        if x_max < rb.left or x_min > rb.right or y_max < rb.bottom or y_min > rb.top:
            print("     ⚠ AOI całkowicie poza zasięgiem rastra — używam całego obrazu")
            return _downsampled_window(src)

        # Przytnij AOI do granic rastra
        x_min = max(x_min, rb.left);  x_max = min(x_max, rb.right)
        y_min = max(y_min, rb.bottom); y_max = min(y_max, rb.top)

        # Właściwa konwersja przez rasterio
        win   = window_from_bounds(x_min, y_min, x_max, y_max, src.transform)
        r0    = max(0, int(win.row_off))
        c0    = max(0, int(win.col_off))
        r1    = min(src.height, int(win.row_off + win.height))
        c1    = min(src.width,  int(win.col_off + win.width))

        print(f"     AOI pixel window: rows {r0}–{r1}, cols {c0}–{c1}  ({r1-r0}×{c1-c0} px)")

        if r1 - r0 < 100 or c1 - c0 < 100:
            print("     ⚠ Zbyt małe okno — używam całego obrazu")
            return _downsampled_window(src)

        return r0, r1, c0, c1

    except Exception as e:
        print(f"     ⚠ Błąd AOI: {e} — używam całego obrazu")
        return _downsampled_window(src)


def _downsampled_window(src: rasterio.DatasetReader) -> tuple[int, int, int, int]:
    """Fallback: zwraca pełne okno rastra."""
    return 0, src.height, 0, src.width


# ─────────────────────────────────────────
# KROK 2 — Progowanie metodą Otsu
# ─────────────────────────────────────────
def otsu_threshold(data: np.ndarray) -> float:
    valid = data[np.isfinite(data)].flatten()
    counts, bins = np.histogram(valid, bins=512, range=(-30, 5))
    bin_centers  = (bins[:-1] + bins[1:]) / 2
    total = counts.sum()

    best_thresh = 0.0
    best_var    = 0.0
    weight_bg   = 0.0
    sum_bg      = 0.0
    total_sum   = np.sum(bin_centers * counts)

    for i, count in enumerate(counts):
        weight_bg += count
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg  += bin_centers[i] * count
        mean_bg  = sum_bg / weight_bg
        mean_fg  = (total_sum - sum_bg) / weight_fg
        var_b    = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_b > best_var:
            best_var    = var_b
            best_thresh = bin_centers[i]

    return float(best_thresh)


# ─────────────────────────────────────────
# KROK 3 — Morfologia matematyczna
# ─────────────────────────────────────────
def clean_mask(water_mask: np.ndarray) -> np.ndarray:
    struct3 = np.ones((3, 3), dtype=bool)
    struct5 = np.ones((5, 5), dtype=bool)
    mask = binary_closing(water_mask, structure=struct5, iterations=2)
    mask = binary_opening(mask,       structure=struct3, iterations=2)
    mask = binary_fill_holes(mask)
    return mask


def keep_large_land(land_mask: np.ndarray, min_area: int) -> np.ndarray:
    labeled, _ = label(land_mask)
    sizes       = np.bincount(labeled.ravel())
    sizes[0]    = 0
    keep        = sizes >= min_area
    return keep[labeled]


# ─────────────────────────────────────────
# KROK 4 — Wektoryzacja
# ─────────────────────────────────────────
def vectorize_shoreline(land_mask, transform, crs) -> gpd.GeoDataFrame:
    shapes_gen = rasterio.features.shapes(
        land_mask.astype(np.uint8),
        mask=land_mask.astype(np.uint8),
        transform=transform,
    )
    polygons = [shape(geom) for geom, val in shapes_gen if val == 1]
    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    lines = []
    for poly in polygons:
        if poly.is_valid and not poly.is_empty:
            lines.append(poly.exterior)
            for interior in poly.interiors:
                lines.append(LineString(interior))

    merged = unary_union(lines)
    return gpd.GeoDataFrame(geometry=[merged], crs=crs)


# ─────────────────────────────────────────
# KROK 6 — Wizualizacja (3 panele)
# ─────────────────────────────────────────
def save_preview(data_db_full: np.ndarray,
                 land_mask_full: np.ndarray,
                 water_mask_full: np.ndarray,
                 nodata_mask_full: np.ndarray,
                 threshold: float,
                 scene_id: str,
                 date_str: str,
                 pixel_window: tuple) -> None:

    r0, r1, c0, c1 = pixel_window

    # Wytnij do AOI
    data_db   = data_db_full[r0:r1, c0:c1]
    land      = land_mask_full[r0:r1, c0:c1]
    water     = water_mask_full[r0:r1, c0:c1]
    nodata    = nodata_mask_full[r0:r1, c0:c1]

    # Jeśli wycięty fragment jest pusty lub za duży — downsample
    MAX_DIM = 3000
    if data_db.size == 0:
        print("\n  ⚠ Pusty wycinek AOI — podgląd pominięty.")
        return
    if data_db.shape[0] > MAX_DIM or data_db.shape[1] > MAX_DIM:
        step = max(data_db.shape[0] // MAX_DIM, data_db.shape[1] // MAX_DIM, 1)
        data_db = data_db[::step, ::step]
        land    = land[::step, ::step]
        water   = water[::step, ::step]
        nodata  = nodata[::step, ::step]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"Costa Brava — {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}  (ID: {scene_id})",
        fontsize=13, fontweight="bold"
    )

    # Panel 1 — Obraz SAR dB (przycięty do AOI)
    valid = data_db[np.isfinite(data_db)]
    if valid.size > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
    else:
        vmin, vmax = -25, 0
    im = axes[0].imshow(data_db, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Sigma0 VV [dB]\n(próg Otsu: {threshold:.2f} dB)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.04, label="dB")

    # Panel 2 — Maska ląd / woda / nodata
    overlay = np.zeros((*data_db.shape, 3), dtype=np.uint8)
    overlay[land]   = [139, 195,  74]   # ląd     — zielony
    overlay[water]  = [ 30, 136, 229]   # woda    — niebieski
    overlay[nodata] = [ 60,  60,  60]   # nodata  — ciemnoszary

    axes[1].imshow(overlay)

    # Czerwona linia brzegowa (kontur lądu)
    try:
        from skimage import measure
        contours = measure.find_contours(land.astype(float), 0.5)
        for cnt in contours:
            axes[1].plot(cnt[:, 1], cnt[:, 0], "r-", linewidth=0.6, alpha=0.8)
    except ImportError:
        pass

    axes[1].set_title("Klasyfikacja ląd / woda\n(czerwona = linia brzegowa)")
    axes[1].axis("off")

    patch_l = mpatches.Patch(color="#8bc34a", label="Ląd")
    patch_w = mpatches.Patch(color="#1e88e5", label="Woda")
    patch_n = mpatches.Patch(color="#3c3c3c", label="Brak danych")
    patch_r = mpatches.Patch(color="red",     label="Linia brzegowa")
    axes[1].legend(handles=[patch_l, patch_w, patch_n, patch_r],
                   loc="lower right", fontsize=8)

    # Panel 3 — Histogram dB z progiem
    valid_full = data_db_full[np.isfinite(data_db_full)].flatten()
    axes[2].hist(valid_full, bins=300, range=(-30, 5),
                 color="#546e7a", alpha=0.8, density=True)
    axes[2].axvline(threshold, color="red", linewidth=2,
                    label=f"Próg Otsu: {threshold:.2f} dB")
    axes[2].set_xlabel("Sigma0 [dB]")
    axes[2].set_ylabel("Gęstość")
    axes[2].set_title("Histogram wartości dB\n(cała scena)")
    axes[2].legend(fontsize=9)
    axes[2].annotate("← WODA", xy=(threshold - 1, axes[2].get_ylim()[1] * 0.8),
                     ha="right", color="#1e88e5", fontsize=9)
    axes[2].annotate("LĄD →",  xy=(threshold + 1, axes[2].get_ylim()[1] * 0.8),
                     ha="left",  color="#4caf50", fontsize=9)

    plt.tight_layout()
    out_path = MAPS_DIR / f"preview_{date_str}_{scene_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Podgląd PNG: {out_path.name}")


# ─────────────────────────────────────────
# PRZETWARZANIE JEDNEJ SCENY
# ─────────────────────────────────────────
def extract_shoreline(tif_path: Path) -> Path | None:
    name = tif_path.stem
    m = re.search(r"(\d{8})_([A-Z0-9]+)_utm", name)
    date_str = m.group(1) if m else "unknown"
    scene_id = m.group(2) if m else name

    out_geojson = LINES_DIR / f"shoreline_{date_str}_{scene_id}.geojson"

    print(f"\n{'─'*55}")
    print(f"  Plik: {tif_path.name}")
    print(f"  Data: {date_str}  |  ID: {scene_id}")
    print(f"{'─'*55}")

    # Krok 1 — wczytanie
    print("  → Wczytywanie rastra ...", end="", flush=True)
    with rasterio.open(tif_path) as src:
        data_db   = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        pixel_win = aoi_to_pixel_window(src)
    r0, r1, c0, c1 = pixel_win
    print(f" ✓  ({data_db.shape[1]}×{data_db.shape[0]} px, AOI: wiersze {r0}–{r1}, kol {c0}–{c1})")

    # Krok 2 — próg
    if MANUAL_THRESHOLD is not None:
        threshold = MANUAL_THRESHOLD
        print(f"  → Próg ręczny: {threshold:.2f} dB")
    else:
        print("  → Wyznaczanie progu Otsu ...", end="", flush=True)
        threshold = otsu_threshold(data_db)
        print(f" ✓  próg = {threshold:.2f} dB")

    nodata_mask = ~np.isfinite(data_db)
    water_mask  = (data_db < threshold) & np.isfinite(data_db)
    land_mask   = ~water_mask & np.isfinite(data_db)

    w_pct = water_mask.sum() / np.isfinite(data_db).sum() * 100
    print(f"     Woda: {w_pct:.1f}%  |  Ląd: {100-w_pct:.1f}%  |  Nodata: {nodata_mask.sum()/data_db.size*100:.1f}%")

    # Krok 3 — morfologia
    print("  → Czyszczenie maski ...", end="", flush=True)
    water_clean = clean_mask(water_mask)
    land_clean  = ~water_clean & np.isfinite(data_db)
    land_clean  = keep_large_land(land_clean, MIN_LAND_AREA)
    water_clean = ~land_clean & np.isfinite(data_db)  # uaktualnij wodę
    print(" ✓")

    # Krok 4 — wektoryzacja
    print("  → Wektoryzacja ...", end="", flush=True)
    gdf = vectorize_shoreline(land_clean, transform, crs)
    if gdf.empty:
        print(" ✗  Nie znaleziono geometrii!")
        return None
    print(f" ✓  ({len(gdf)} obiekt(ów))")

    # Krok 5 — zapis GeoJSON (nadpisz jeśli istnieje)
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"  ✓ GeoJSON: {out_geojson.name}")

    # Krok 6 — wizualizacja
    print("  → Generowanie podglądu ...", end="", flush=True)
    save_preview(
        data_db_full=data_db,
        land_mask_full=land_clean,
        water_mask_full=water_clean,
        nodata_mask_full=nodata_mask,
        threshold=threshold,
        scene_id=scene_id,
        date_str=date_str,
        pixel_window=pixel_win,
    )

    return out_geojson


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Etap 3 — Ekstrakcja linii brzegowej")
    print("=" * 55)

    tif_files = sorted(PROCESSED_DIR.glob("*_utm.tif"))
    if not tif_files:
        print("✗ Brak plików *_utm.tif w data/processed/")
        return

    print(f"Znaleziono {len(tif_files)} plik(ów).\n")

    results = []
    for tif in tif_files:
        r = extract_shoreline(tif)
        if r:
            results.append(r)

    print(f"\n{'='*55}")
    if results:
        print(f"  ✓ Wyekstrahowano {len(results)} linię/linii brzegową.")
        print(f"  GeoJSON → {LINES_DIR.resolve()}")
        print(f"  PNG     → {MAPS_DIR.resolve()}")
    else:
        print("  ✗ Nie udało się wyekstrahować linii brzegowej.")
        print("\n  Wskazówki jeśli wynik wciąż niepoprawny:")
        print("  • Zmień MANUAL_THRESHOLD (np. -16.0 lub -10.0)")
        print("  • Sprawdź histogram w PNG — próg powinien leżeć")
        print("    między dwoma szczytami (woda i ląd)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
