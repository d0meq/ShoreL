"""
analysis.py
===========
Analiza wieloczasowa zmian linii brzegowej i obliczanie wskaznika EPR.

Metoda jest uproszczona wersja podejscia DSAS (Digital Shoreline Analysis
System) stosowanego w geologii wybrzeza:

1. Z najstarszej linii brzegowej tworzymy LINIE BAZOWA (baseline) lezaca po
   stronie ladu, rownolegla do brzegu.
2. Co zadany odstep stawiamy PRZEKROJE (transekty) prostopadle do brzegu,
   biegnace od linii bazowej w strone morza.
3. Dla kazdego przekroju i kazdego roku znajdujemy punkt przeciecia z linia
   brzegowa  ->  odleglosc = pozycja brzegu w danym roku.
4. EPR (End Point Rate) = (pozycja_najnowsza - pozycja_najstarsza)
                          / (rok_najnowszy - rok_najstarszy)   [m / rok]

Interpretacja znaku (linia bazowa po stronie ladu):
    EPR < 0  ->  EROZJA   (brzeg cofa sie w glab ladu)
    EPR > 0  ->  AKUMULACJA (brzeg przyrasta w strone morza)
"""

import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from shapely.geometry import LineString, Point


def export_coastlines(results, path):
    """Zapisuje wszystkie linie brzegowe do pliku GeoJSON (do QGIS)."""
    import geopandas as gpd
    rows = sorted(results, key=lambda r: r.year)
    gdf = gpd.GeoDataFrame(
        {"year": [r.year for r in rows]},
        geometry=[r.line for r in rows],
        crs=rows[0].crs,
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def export_epr_points(epr_df, crs, path):
    """Zapisuje punkty przekrojow z wartoscia EPR do GeoJSON (do QGIS)."""
    import geopandas as gpd
    df = epr_df.dropna(subset=["x", "y"]).copy()
    gdf = gpd.GeoDataFrame(
        df.drop(columns=["x", "y"]),
        geometry=[Point(xy) for xy in zip(df["x"], df["y"])],
        crs=crs,
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def _point_on_land(x, y, land_mask, transform):
    """Sprawdza, czy wspolrzedna mapowa (x, y) trafia w piksel ladu."""
    r, c = rowcol(transform, x, y)
    if 0 <= r < land_mask.shape[0] and 0 <= c < land_mask.shape[1]:
        return bool(land_mask[r, c])
    return False


def build_baseline(result, offset_m, simplify_m=None):
    """
    Tworzy linie bazowa przez przesuniecie najstarszej linii brzegowej
    o `offset_m` metrow w strone LADU.

    result : CoastlineResult najstarszej sceny
    offset_m : float          odleglosc przesuniecia [m]
    simplify_m : float|None   tolerancja wygladzania linii (Douglas-Peucker)
    """
    line = result.line
    if simplify_m:
        line = line.simplify(simplify_m)

    # przesuwamy w obie strony i wybieramy te, ktora lezy na ladzie
    candidates = []
    for side in ("left", "right"):
        try:
            cand = line.parallel_offset(offset_m, side, join_style=2)
        except Exception:
            continue
        if cand.is_empty:
            continue
        # parallel_offset moze zwrocic MultiLineString - bierzemy najdluzszy fragment
        if cand.geom_type == "MultiLineString":
            cand = max(cand.geoms, key=lambda g: g.length)
        # ile punktow tej linii lezy na ladzie?
        pts = [cand.interpolate(t, normalized=True) for t in np.linspace(0, 1, 25)]
        land_frac = np.mean([
            _point_on_land(p.x, p.y, result.land_mask, result.transform) for p in pts
        ])
        candidates.append((land_frac, cand))

    if not candidates:
        raise RuntimeError("Nie udalo sie zbudowac linii bazowej.")
    # wybieramy linie najbardziej "ladowa"
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def make_transects(baseline, reference_line, spacing_m, length_m):
    """
    Stawia przekroje prostopadle do linii bazowej, skierowane w strone morza.

    baseline : LineString      linia bazowa (po stronie ladu)
    reference_line : LineString linia brzegowa odniesienia (do okreslenia
                               kierunku "w strone morza")
    spacing_m : float          odstep miedzy przekrojami [m]
    length_m : float           dlugosc przekroju [m]

    Zwraca liste slownikow: {id, line(LineString), origin(Point)}
    """
    transects = []
    n = max(int(baseline.length // spacing_m), 1)
    eps = spacing_m * 0.05  # maly krok do wyznaczenia stycznej

    for i in range(n + 1):
        d = min(i * spacing_m, baseline.length)
        origin = baseline.interpolate(d)
        # styczna z roznicy dwoch bliskich punktow
        p1 = baseline.interpolate(max(d - eps, 0))
        p2 = baseline.interpolate(min(d + eps, baseline.length))
        tx, ty = p2.x - p1.x, p2.y - p1.y
        norm = np.hypot(tx, ty)
        if norm == 0:
            continue
        tx, ty = tx / norm, ty / norm
        # dwa kierunki prostopadle
        nx, ny = -ty, tx
        # wybor kierunku "w strone morza" = ten, ktory zbliza nas do linii brzegu
        cand_a = Point(origin.x + nx * length_m, origin.y + ny * length_m)
        cand_b = Point(origin.x - nx * length_m, origin.y - ny * length_m)
        if reference_line.distance(cand_a) <= reference_line.distance(cand_b):
            seg = LineString([origin, cand_a])
        else:
            seg = LineString([origin, cand_b])
        transects.append({"id": i, "line": seg, "origin": origin})
    return transects


def measure_positions(transects, results):
    """
    Dla kazdego przekroju mierzy pozycje linii brzegowej w kazdym roku.

    transects : lista z make_transects
    results : lista CoastlineResult (rozne lata)

    Zwraca pandas.DataFrame: wiersze = przekroje, kolumny = lata,
    wartosci = odleglosc od linii bazowej [m] (NaN gdy brak przeciecia).
    """
    years = sorted(r.year for r in results)
    by_year = {r.year: r.line for r in results}

    rows = []
    for t in transects:
        row = {"transect_id": t["id"]}
        for y in years:
            inter = t["line"].intersection(by_year[y])
            if inter.is_empty:
                row[y] = np.nan
                continue
            # przy wielu przecieciach bierzemy najblizsze poczatkowi przekroju
            if inter.geom_type == "Point":
                pt = inter
            elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
                pts = [g for g in inter.geoms if g.geom_type == "Point"]
                if not pts:
                    row[y] = np.nan
                    continue
                pt = min(pts, key=lambda p: t["origin"].distance(p))
            else:  # linia - bierzemy pierwszy punkt
                pt = Point(inter.coords[0])
            row[y] = t["origin"].distance(pt)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("transect_id")
    return df


def compute_epr(positions, geometry=None):
    """
    Oblicza EPR oraz NSM dla kazdego przekroju.

    positions : DataFrame z measure_positions
    geometry : opcjonalna lista transektow (do dolaczenia punktu origin)

    Zwraca DataFrame z kolumnami:
        first_year, last_year, NSM [m], EPR [m/rok], (x, y origin)
    """
    years = list(positions.columns)
    out = []
    for tid, row in positions.iterrows():
        valid = row.dropna()
        if len(valid) < 2:
            out.append({"transect_id": tid, "first_year": np.nan,
                        "last_year": np.nan, "NSM_m": np.nan, "EPR_m_yr": np.nan})
            continue
        y0, y1 = valid.index.min(), valid.index.max()
        nsm = valid[y1] - valid[y0]
        epr = nsm / (y1 - y0) if y1 != y0 else np.nan
        out.append({"transect_id": tid, "first_year": y0, "last_year": y1,
                    "NSM_m": round(nsm, 2), "EPR_m_yr": round(epr, 3)})

    df = pd.DataFrame(out).set_index("transect_id")

    if geometry is not None:
        coords = {t["id"]: (t["origin"].x, t["origin"].y) for t in geometry}
        df["x"] = [coords.get(i, (np.nan, np.nan))[0] for i in df.index]
        df["y"] = [coords.get(i, (np.nan, np.nan))[1] for i in df.index]
    return df
