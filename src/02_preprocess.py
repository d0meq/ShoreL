"""
ETAP 2 — Preprocessing danych Sentinel-1 GRD
Kroki:
  1. Rozpakowanie archiwów .zip
  2. Odczyt kanału VV (polaryzacja najlepsza do detekcji wody)
  3. Kalibracja radiometryczna → współczynnik sigma0
  4. Konwersja do skali decybelowej [dB]
  5. Filtr speckle (Lee 7x7) — redukcja szumu radarowego
  6. Reprojekcja do EPSG:32634 (UTM strefa 34N — Polska)
  7. Zapis jako GeoTIFF
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d

# ─────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:32630"   # zamiast 32634
FILTER_SIZE = 7             # rozmiar okna filtra Lee


# ─────────────────────────────────────────
# KROK 1 — Rozpakowanie archiwów .zip
# ─────────────────────────────────────────
def unzip_scenes(raw_dir: Path) -> list[Path]:
    """Rozpakowuje wszystkie .zip i zwraca listę folderów .SAFE."""
    safe_dirs = []
    zip_files = list(raw_dir.glob("*.SAFE.zip"))

    if not zip_files:
        print("✗ Brak plików .SAFE.zip w katalogu data/raw/")
        return []

    print(f"Znaleziono {len(zip_files)} archiw(ów) .zip\n")

    for zf in zip_files:
        # POPRAWKA: zf.stem dla "plik.SAFE.zip" daje "plik.SAFE"
        # — to już jest poprawna nazwa folderu, bez doklejania ".SAFE"
        safe_name = zf.stem                  # np. "S1A_...0295.SAFE"
        safe_path = raw_dir / safe_name

        if safe_path.exists():
            print(f"   ↷ Pominięto (już rozpakowany): {safe_name}")
        else:
            print(f"   📦 Rozpakowuję: {zf.name} ...", end="", flush=True)
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(raw_dir)
            print(" ✓")

        safe_dirs.append(safe_path)

    return safe_dirs


# ─────────────────────────────────────────
# KROK 2 — Odnajdywanie pliku VV
# ─────────────────────────────────────────
def find_vv_tiff(safe_dir: Path) -> Path | None:
    """Zwraca ścieżkę do pliku GeoTIFF z polaryzacją VV."""
    # Użycie rglob dla pewności — szuka rekurencyjnie
    tiffs = [f for f in safe_dir.rglob("*-vv-*.tiff")]
    if not tiffs:
        tiffs = [f for f in safe_dir.rglob("*vv*.tif")]
    if not tiffs:
        print(f"   ✗ Nie znaleziono pliku VV w: {safe_dir.name}")
        print(f"     Dostępne pliki w measurement/:")
        for f in safe_dir.glob("measurement/*"):
            print(f"       {f.name}")
        return None
    # Wybierz plik z folderu measurement (nie annotation)
    measurement_tiffs = [f for f in tiffs if "measurement" in str(f)]
    return measurement_tiffs[0] if measurement_tiffs else tiffs[0]


# ─────────────────────────────────────────
# KROK 3 — Kalibracja radiometryczna
# ─────────────────────────────────────────
def get_calibration_lut(safe_dir: Path) -> np.ndarray | None:
    """Odczytuje LUT sigmaNought z pliku XML kalibracji."""
    cal_files = list(safe_dir.rglob("calibration/calibration*vv*.xml"))
    if not cal_files:
        print("   ⚠ Brak pliku kalibracji XML — użyta kalibracja uproszczona.")
        return None

    tree = ET.parse(cal_files[0])
    root = tree.getroot()

    lut_values = []
    for cal_vec in root.iter("calibrationVector"):
        sigma_el = cal_vec.find("sigmaNought")
        if sigma_el is not None:
            vals = [float(v) for v in sigma_el.text.split()]
            lut_values.append(vals)

    return np.array(lut_values[0]) if lut_values else None


def calibrate(data: np.ndarray, lut: np.ndarray | None) -> np.ndarray:
    """sigma0 = DN² / LUT² (lub z przybliżonym współczynnikiem)."""
    dn = data.astype(np.float32)

    if lut is not None:
        x_lut  = np.linspace(0, dn.shape[1] - 1, len(lut))
        x_img  = np.arange(dn.shape[1])
        interp = interp1d(x_lut, lut, kind="linear", fill_value="extrapolate")
        lut_row = interp(x_img).astype(np.float32)
        cal_sq  = lut_row[np.newaxis, :] ** 2
        sigma0  = np.where(cal_sq > 0, (dn ** 2) / cal_sq, 0.0)
    else:
        A2     = 83.0 ** 2
        sigma0 = np.where(dn > 0, (dn ** 2) / A2, 0.0)

    return sigma0.astype(np.float32)


# ─────────────────────────────────────────
# KROK 4 — Konwersja do dB
# ─────────────────────────────────────────
def to_db(sigma0: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(sigma0 > 0, 10 * np.log10(sigma0), np.nan)
    return db.astype(np.float32)


# ─────────────────────────────────────────
# KROK 5 — Filtr Lee
# ─────────────────────────────────────────
def lee_filter(img: np.ndarray, size: int = 7) -> np.ndarray:
    """Uproszczony filtr Lee dla redukcji szumu speckle."""
    img_filled = np.nan_to_num(img, nan=0.0)
    img_mean   = uniform_filter(img_filled, size=size)
    img_sq     = uniform_filter(img_filled ** 2, size=size)
    img_var    = np.maximum(img_sq - img_mean ** 2, 0.0)

    enl        = 4.4  # Equivalent Number of Looks — Sentinel-1 GRD
    noise_var  = (img_mean ** 2) / enl
    weight     = np.where(img_var > 0, img_var / (img_var + noise_var), 0.0)
    filtered   = img_mean + weight * (img_filled - img_mean)

    filtered[img == 0] = np.nan
    return filtered.astype(np.float32)


# ─────────────────────────────────────────
# KROK 6 — Reprojekcja do UTM 34N
# ─────────────────────────────────────────
def reproject_to_utm(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(32634)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": "float32",
            "nodata": np.nan,
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )


# ─────────────────────────────────────────
# PRZETWARZANIE JEDNEJ SCENY
# ─────────────────────────────────────────
def process_scene(safe_dir: Path) -> Path | None:
    scene_name = safe_dir.stem

    # Wyciągnięcie daty + unikalnego ID (ostatnie 4 znaki) z nazwy folderu
    # Format: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_..._XXXXX_YYYYYY_ZZZZ.SAFE
    date_match = re.search(r"_(\d{8})T", scene_name)
    date_str   = date_match.group(1) if date_match else "unknown"
    unique_id  = scene_name.split("_")[-1]   # np. "0295" lub "D32C"

    # Unikalna nazwa pliku wynikowego (data + ID)
    out_stem   = f"sigma0_vv_{date_str}_{unique_id}"
    tmp_path   = PROCESSED_DIR / f"{out_stem}_tmp.tif"
    final_path = PROCESSED_DIR / f"{out_stem}_utm.tif"

    if final_path.exists():
        print(f"   ↷ Pominięto (już przetworzony): {final_path.name}")
        return final_path

    print(f"\n{'─'*55}")
    print(f"  Przetwarzam: {scene_name}")
    print(f"  Data: {date_str}  |  ID: {unique_id}")
    print(f"{'─'*55}")

    # Krok 2 — znajdź plik VV
    vv_tiff = find_vv_tiff(safe_dir)
    if vv_tiff is None:
        return None
    print(f"  ✓ Plik VV: {vv_tiff.name}")

    # Krok 3 — odczyt + kalibracja
    print("  → Kalibracja radiometryczna ...", end="", flush=True)
    with rasterio.open(vv_tiff) as src:
        meta = src.meta.copy()
        raw  = src.read(1)
    lut    = get_calibration_lut(safe_dir)
    sigma0 = calibrate(raw, lut)
    print(" ✓")

    # Krok 5 — filtr speckle
    print(f"  → Filtr Lee ({FILTER_SIZE}×{FILTER_SIZE}) ...", end="", flush=True)
    sigma0_f = lee_filter(sigma0, size=FILTER_SIZE)
    print(" ✓")

    # Krok 4 — dB
    print("  → Konwersja do dB ...", end="", flush=True)
    sigma0_db = to_db(sigma0_f)
    print(" ✓")

    # Zapis tymczasowy
    meta.update({"dtype": "float32", "count": 1, "nodata": np.nan})
    with rasterio.open(tmp_path, "w", **meta) as dst:
        dst.write(sigma0_db, 1)

    # Krok 6 — reprojekcja
    print(f"  → Reprojekcja do {TARGET_CRS} ...", end="", flush=True)
    reproject_to_utm(tmp_path, final_path)
    tmp_path.unlink()
    print(" ✓")

    size_mb = final_path.stat().st_size / 1e6
    print(f"  ✓ Zapisano: {final_path.name}  ({size_mb:.1f} MB)")
    return final_path


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Etap 2 — Preprocessing SAR (Sentinel-1 GRD)")
    print("=" * 55)

    safe_dirs = unzip_scenes(RAW_DIR)
    if not safe_dirs:
        return

    results = []
    for safe_dir in safe_dirs:
        result = process_scene(safe_dir)
        if result:
            results.append(result)

    print(f"\n{'='*55}")
    if results:
        print(f"  ✓ Przetworzono {len(results)} scen(y) pomyślnie.")
        print(f"  Pliki w: {PROCESSED_DIR.resolve()}")
        for f in results:
            print(f"   • {f.name}")
    else:
        print("  ✗ Żadna scena nie została przetworzona.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
