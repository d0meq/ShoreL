"""
coastline.py
============
Wykrywanie linii brzegowej z pojedynczego obrazu radarowego SAR (GeoTIFF).

Zasada działania (zgodna z opisem projektu):
woda na obrazach SAR odbija sygnal "lustrzanie" w kierunku przeciwnym do
satelity, wiec jest CIEMNA (niski wspolczynnik odbicia). Lad odbija sygnal
rozproszenie i jest JASNY (wysoki wspolczynnik odbicia). Dzieki temu prostym
progowaniem (metoda Otsu) mozna oddzielic lad od wody, a granica miedzy nimi
to linia brzegowa.

Kroki:
1. Wczytanie rastra (rasterio) wraz z geotransformacja i ukladem wspolrzednych.
2. (opcjonalnie) konwersja do skali decybelowej (dB) - rozklad sygnalu SAR
   jest wtedy bardziej symetryczny, co poprawia progowanie.
3. Redukcja szumu plamkowego (speckle) filtrem medianowym.
4. Progowanie metoda Otsu  ->  maska binarna (lad / woda).
5. Czyszczenie maski (usuniecie malych obiektow i dziur, wybor najwiekszego
   spojnego obszaru ladu).
6. Wyznaczenie konturu maski (granica lad-woda)  ->  linia brzegowa.
7. Zamiana wspolrzednych pikselowych na wspolrzedne geograficzne (mapowe).
"""

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label
from skimage.morphology import remove_small_holes, remove_small_objects
from shapely.geometry import LineString


@dataclass
class CoastlineResult:
    """Wynik ekstrakcji dla jednej sceny."""
    year: int                 # rok sceny
    line: LineString          # linia brzegowa we wspolrzednych mapowych
    land_mask: np.ndarray     # maska binarna ladu (True = lad)
    transform: object         # affine transform rastra
    crs: object               # uklad wspolrzednych rastra
    bounds: tuple             # zasieg (minx, miny, maxx, maxy)


def _remove_small_holes(mask, size):
    """Zgodne z roznymi wersjami scikit-image."""
    try:
        return remove_small_holes(mask, max_size=size)
    except TypeError:
        return remove_small_holes(mask, area_threshold=size)


def _remove_small_objects(mask, size):
    try:
        return remove_small_objects(mask, max_size=size)
    except TypeError:
        return remove_small_objects(mask, min_size=size)


def _to_decibel(arr):
    """Konwersja intensywnosci sygnalu na skale dB: 10*log10(x)."""
    arr = np.where(arr <= 0, np.nan, arr)
    return 10.0 * np.log10(arr)


def _drop_border_points(contour, shape, margin=2):
    """
    Usuwa z konturu fragmenty biegnace po krawedzi obrazu (to nie jest realna
    linia brzegowa, tylko ramka kadru). Zwraca najdluzszy ciagly fragment
    lezacy wewnatrz obrazu.
    """
    h, w = shape
    rows, cols = contour[:, 0], contour[:, 1]
    inside = (
        (rows > margin) & (rows < h - 1 - margin) &
        (cols > margin) & (cols < w - 1 - margin)
    )
    if inside.all() or not inside.any():
        return contour
    # podzial na ciagle odcinki "wewnetrzne"
    runs, cur = [], []
    for pt, ok in zip(contour, inside):
        if ok:
            cur.append(pt)
        elif cur:
            runs.append(np.array(cur))
            cur = []
    if cur:
        runs.append(np.array(cur))
    return max(runs, key=len) if runs else contour


def extract_coastline(
    tif_path,
    year,
    speckle_size=5,
    to_db=True,
    min_object_px=500,
    keep_largest=True,
    border_margin_px=2,
):
    """
    Wyznacza linie brzegowa z jednego pliku GeoTIFF.

    Parametry
    ---------
    tif_path : str            sciezka do pliku GeoTIFF (1 kanal SAR, np. VV)
    year : int                rok sceny (do analizy wieloczasowej)
    speckle_size : int        rozmiar okna filtra medianowego (redukcja szumu)
    to_db : bool              czy przeliczyc na decybele przed progowaniem
    min_object_px : int       minimalny rozmiar obiektu/dziury do usuniecia (px)
    keep_largest : bool       zostaw tylko najwiekszy spojny obszar ladu

    Zwraca
    ------
    CoastlineResult
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        bounds = tuple(src.bounds)
        nodata = src.nodata

    # maska brakujacych danych
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    work = arr.copy()
    if to_db:
        work = _to_decibel(work)

    # uzupelnienie brakow mediana, zeby filtr i Otsu nie wariowaly
    fill_value = np.nanmedian(work[valid]) if valid.any() else 0.0
    work = np.where(np.isfinite(work), work, fill_value)

    # 3. redukcja szumu plamkowego (speckle)
    work = median_filter(work, size=speckle_size)

    # 4. progowanie Otsu  ->  lad = wartosci wysokie (jasne)
    thresh = threshold_otsu(work)
    land = work > thresh

    # 5. czyszczenie maski
    if min_object_px > 0:
        land = _remove_small_holes(land, min_object_px)
        land = _remove_small_objects(land, min_object_px)
    if keep_largest and land.any():
        lbl = label(land)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0  # tlo
        land = lbl == counts.argmax()

    # 6. kontur granicy lad-woda na poziomie 0.5
    contours = find_contours(land.astype(float), level=0.5)
    if not contours:
        raise RuntimeError(f"Nie znaleziono linii brzegowej w {tif_path}")
    # najdluzszy kontur traktujemy jako glowna linie brzegowa
    contour = max(contours, key=len)
    # odciecie fragmentow biegnacych po krawedzi kadru
    if border_margin_px > 0:
        contour = _drop_border_points(contour, land.shape, margin=border_margin_px)

    # 7. piksele (row, col) -> wspolrzedne mapowe (x, y)
    rows, cols = contour[:, 0], contour[:, 1]
    xs, ys = transform_xy(transform, rows, cols)
    line = LineString(list(zip(xs, ys)))

    return CoastlineResult(
        year=year,
        line=line,
        land_mask=land,
        transform=transform,
        crs=crs,
        bounds=bounds,
    )
