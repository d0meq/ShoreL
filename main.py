"""
main.py
=======
Uruchomienie analizy na PRAWDZIWYCH danych (pliki GeoTIFF z obrazami SAR).

Zalozenia dotyczace danych wejsciowych:
  * jeden plik GeoTIFF = jedna scena (jeden rok),
  * w nazwie pliku musi byc rok, np. "sentinel1_2019.tif", "S1_2021_vv.tif",
  * dane powinny byc w ukladzie metrycznym (UTM), zeby EPR wyszlo w m/rok.

Przyklad:
    python main.py --input data/raw --output data/output \
                   --spacing 200 --length 2000 --offset 800

Skad wziac dane Sentinel-1 (stan na 2025/2026):
  Stara biblioteka 'sentinelsat' oraz Copernicus Open Access Hub NIE dzialaja.
  Dane pobiera sie z Copernicus Data Space Ecosystem:
    - recznie:  https://browser.dataspace.copernicus.eu  (Copernicus Browser),
    - przez API (OData / STAC / openEO) po zalozeniu darmowego konta.
  Polecany produkt: Sentinel-1 GRD, polaryzacja VV. Dla najlepszych wynikow
  warto wczesniej wykonac w programie ESA SNAP: kalibracje radiometryczna,
  filtr speckle i korekcje terenu (Range-Doppler) z zapisem do GeoTIFF (UTM).
"""

import argparse
import os
import re
import sys

import numpy as np

from coastline import extract_coastline
from analysis import build_baseline, make_transects, measure_positions, compute_epr
from analysis import export_coastlines, export_epr_points
from visualize import plot_change_map, plot_epr_chart, make_animation

YEAR_RE = re.compile(r"(19|20)\d{2}")


def find_year(filename):
    """Wyciaga rok z nazwy pliku (pierwsza liczba 4-cyfrowa 19xx/20xx)."""
    m = YEAR_RE.search(filename)
    return int(m.group(0)) if m else None


def collect_inputs(folder):
    """Zwraca posortowana liste (rok, sciezka) plikow .tif z folderu."""
    items = []
    for name in os.listdir(folder):
        if name.lower().endswith((".tif", ".tiff")):
            yr = find_year(name)
            if yr is None:
                print(f"  [pomijam] brak roku w nazwie: {name}")
                continue
            items.append((yr, os.path.join(folder, name)))
    items.sort(key=lambda x: x[0])
    return items


def main():
    ap = argparse.ArgumentParser(description="Monitoring zmian linii brzegowej (SAR).")
    ap.add_argument("--input", required=True, help="folder z plikami GeoTIFF")
    ap.add_argument("--output", default="data/output", help="folder na wyniki")
    ap.add_argument("--spacing", type=float, default=200, help="odstep przekrojow [m]")
    ap.add_argument("--length", type=float, default=2000, help="dlugosc przekroju [m]")
    ap.add_argument("--offset", type=float, default=800, help="odsuniecie linii bazowej [m]")
    ap.add_argument("--speckle", type=int, default=5, help="rozmiar filtra speckle [px]")
    ap.add_argument("--no-db", action="store_true", help="nie przeliczaj na dB")
    args = ap.parse_args()

    inputs = collect_inputs(args.input)
    if len(inputs) < 2:
        sys.exit("Potrzebne sa co najmniej 2 sceny z roznych lat.")

    print(f"Znaleziono {len(inputs)} scen: {[y for y, _ in inputs]}")
    os.makedirs(args.output, exist_ok=True)

    # 1. ekstrakcja linii brzegowej
    print("Wykrywanie linii brzegowej...")
    results = []
    for yr, path in inputs:
        r = extract_coastline(path, year=yr,
                              speckle_size=args.speckle, to_db=not args.no_db)
        results.append(r)
        print(f"  - {yr}: OK")

    # ostrzezenie, jesli dane nie sa w metrach
    crs = results[0].crs
    if crs is not None and getattr(crs, "is_geographic", False):
        print("  UWAGA: dane sa w stopniach (uklad geograficzny). "
              "EPR nie bedzie w metrach - przeprojektuj do UTM.")

    # 2. analiza
    print("Analiza wieloczasowa...")
    oldest = min(results, key=lambda r: r.year)
    baseline = build_baseline(oldest, offset_m=args.offset, simplify_m=args.spacing / 5)
    transects = make_transects(baseline, oldest.line,
                               spacing_m=args.spacing, length_m=args.length)
    positions = measure_positions(transects, results)
    epr = compute_epr(positions, geometry=transects)

    # 3. zapis wynikow
    print("Zapis wynikow...")
    epr.to_csv(os.path.join(args.output, "epr_table.csv"))
    positions.to_csv(os.path.join(args.output, "shoreline_positions.csv"))
    export_coastlines(results, os.path.join(args.output, "linie_brzegowe.geojson"))
    export_epr_points(epr, oldest.crs, os.path.join(args.output, "epr_punkty.geojson"))
    plot_change_map(results, transects, epr, os.path.join(args.output, "mapa_zmian.png"))
    plot_epr_chart(epr, os.path.join(args.output, "wykres_epr.png"))
    make_animation(results, os.path.join(args.output, "animacja.gif"))

    mean_epr = np.nanmean(epr["EPR_m_yr"])
    print(f"\nGotowe. Srednie EPR = {mean_epr:.2f} m/rok. Wyniki w: {args.output}")


if __name__ == "__main__":
    main()
