"""
demo.py
=======
Pelna demonstracja dzialania projektu BEZ potrzeby pobierania danych z
satelity. Skrypt:

1. Generuje 5 sztucznych scen SAR (GeoTIFF) z brzegiem, ktory z roku na rok
   sie zmienia: na polnocy silna erozja, na poludniu lekka akumulacja.
   Sceny zawieraja realistyczny szum plamkowy (speckle), tak jak prawdziwe
   obrazy radarowe.
2. Uruchamia caly potok analizy (ten sam, ktory dziala na prawdziwych danych).
3. Zapisuje produkty wynikowe do data/output/.

Uruchomienie:   python demo.py
"""

import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

from coastline import extract_coastline
from analysis import build_baseline, make_transects, measure_positions, compute_epr
from analysis import export_coastlines, export_epr_points
from visualize import plot_change_map, plot_epr_chart, make_animation

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "output")

# --- parametry sceny ---
H, W = 500, 600          # rozmiar obrazu w pikselach
PIX = 10                 # rozdzielczosc 10 m/piksel (jak Sentinel-1 GRD)
CRS = "EPSG:32634"       # UTM strefa 34N (Polska / Baltyk) - wspolrzedne w metrach
TRANSFORM = from_origin(500000, 6000000, PIX, PIX)
YEARS = [2018, 2020, 2021, 2023, 2025]


def make_scene(year, seed):
    """Tworzy jedna sztuczna scene SAR i zwraca tablice intensywnosci."""
    rng = np.random.default_rng(seed)
    rows = np.arange(H).reshape(-1, 1)
    cols = np.arange(W).reshape(1, -1)

    # pozycja brzegu (kolumna) zalezna od wiersza (falisty brzeg)
    base = 320 + 30 * np.sin(2 * np.pi * rows / 180.0)
    # tempo zmian rozne wzdluz brzegu: gora erozja (-), dol akumulacja (+)
    rate = np.linspace(-5.0, 2.0, H).reshape(-1, 1)   # kolumny / rok
    shoreline = base + rate * (year - YEARS[0])

    land = cols < shoreline          # lad po lewej, woda po prawej

    # intensywnosc wstecznego rozproszenia (linear): lad jasny, woda ciemna
    img = np.where(land, 0.25, 0.04).astype("float32")

    # szum plamkowy (speckle) - multiplikatywny, rozklad gamma (multi-look L=4)
    L = 4
    speckle = rng.gamma(shape=L, scale=1.0 / L, size=(H, W)).astype("float32")
    img *= speckle
    return img


def save_geotiff(arr, path):
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="float32", crs=CRS, transform=TRANSFORM,
    ) as dst:
        dst.write(arr, 1)


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    # 1. generowanie scen
    print("1) Generowanie sztucznych scen SAR...")
    paths = []
    for i, yr in enumerate(YEARS):
        p = os.path.join(RAW, f"sar_{yr}.tif")
        save_geotiff(make_scene(yr, seed=100 + i), p)
        paths.append((yr, p))
        print(f"   - {os.path.basename(p)}")

    # 2. ekstrakcja linii brzegowej z kazdej sceny
    print("2) Wykrywanie linii brzegowej...")
    results = [extract_coastline(p, year=yr) for yr, p in paths]

    # 3. analiza wieloczasowa
    print("3) Analiza zmian (linia bazowa, przekroje, EPR)...")
    oldest = min(results, key=lambda r: r.year)
    baseline = build_baseline(oldest, offset_m=800, simplify_m=30)
    transects = make_transects(baseline, oldest.line, spacing_m=200, length_m=2000)
    positions = measure_positions(transects, results)
    epr = compute_epr(positions, geometry=transects)

    # 4. zapis wynikow
    print("4) Zapis produktow wynikowych...")
    epr.to_csv(os.path.join(OUT, "epr_table.csv"))
    positions.to_csv(os.path.join(OUT, "shoreline_positions.csv"))
    export_coastlines(results, os.path.join(OUT, "linie_brzegowe.geojson"))
    export_epr_points(epr, oldest.crs, os.path.join(OUT, "epr_punkty.geojson"))
    plot_change_map(results, transects, epr, os.path.join(OUT, "mapa_zmian.png"))
    plot_epr_chart(epr, os.path.join(OUT, "wykres_epr.png"))
    make_animation(results, os.path.join(OUT, "animacja.gif"))

    # 5. podsumowanie
    mean_epr = np.nanmean(epr["EPR_m_yr"])
    erosion = (epr["EPR_m_yr"] < 0).sum()
    accretion = (epr["EPR_m_yr"] > 0).sum()
    print("\n=== PODSUMOWANIE ===")
    print(f"Liczba przekrojow:        {len(epr)}")
    print(f"Srednie tempo EPR:        {mean_epr:.2f} m/rok")
    print(f"Przekroje z erozja:       {erosion}")
    print(f"Przekroje z akumulacja:   {accretion}")
    print(f"Najszybsza erozja:        {epr['EPR_m_yr'].min():.2f} m/rok")
    print(f"Najszybsza akumulacja:    {epr['EPR_m_yr'].max():.2f} m/rok")
    print(f"\nWyniki zapisane w: {OUT}")


if __name__ == "__main__":
    main()
