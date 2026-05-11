"""
ETAP 1 — Pobieranie danych Sentinel-1 GRD
Źródło: Copernicus Data Space Ecosystem (CDSE)
API: OData / REST
Obszar: Costa Brava (okolice Platja d'Aro / Palamós, Katalonia)
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────
# KONFIGURACJA — uzupełnij swoimi danymi
# ─────────────────────────────────────────
USERNAME = "dominikskornicki68@gmail.com"   # email z konta Copernicus
PASSWORD = "Ner1fymn0xh3lR>"              # hasło Copernicus

# Obszar zainteresowania (AOI) — Costa Brava: Platja d'Aro / Palamós (bounding box)
# Format: min_lon, min_lat, max_lon, max_lat
# Linia brzegowa dobrze widoczna w SAR — mały obszar, mniej danych do pobrania
AOI_WKT = "POLYGON((3.05 41.78, 3.22 41.78, 3.22 41.90, 3.05 41.90, 3.05 41.78))"

# Zakres dat (format: YYYY-MM-DD) — lato = minimalne zachmurzenie nad Katalonią
DATE_START = "2022-07-01"
DATE_END   = "2022-07-31"

# Katalog zapisu danych
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Maksymalna liczba scen do pobrania (test: 2)
MAX_RESULTS = 1


# ─────────────────────────────────────────
# KROK 1 — Pobranie tokenu OAuth2
# ─────────────────────────────────────────
def get_access_token(username: str, password: str) -> str:
    """Pobiera token dostępu z CDSE."""
    url = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    r = requests.post(url, data=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Błąd autoryzacji ({r.status_code}): {r.text}")
    token = r.json()["access_token"]
    print("✓ Token OAuth2 uzyskany pomyślnie.")
    return token


# ─────────────────────────────────────────
# KROK 2 — Wyszukiwanie scen Sentinel-1
# ─────────────────────────────────────────
def search_scenes(token: str) -> list[dict]:
    """Przeszukuje katalog CDSE i zwraca listę scen."""
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # Budowanie filtra OData
    date_filter = (
        f"ContentDate/Start gt {DATE_START}T00:00:00.000Z "
        f"and ContentDate/Start lt {DATE_END}T23:59:59.000Z"
    )
    collection_filter = "Collection/Name eq 'SENTINEL-1'"
    product_filter    = "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD')"
    geo_filter        = f"OData.CSC.Intersects(area=geography'SRID=4326;{AOI_WKT}')"

    full_filter = f"{date_filter} and {collection_filter} and {product_filter} and {geo_filter}"

    params = {
        "$filter": full_filter,
        "$orderby": "ContentDate/Start asc",
        "$top": MAX_RESULTS,
        "$expand": "Attributes",
    }

    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, params=params, headers=headers, timeout=60)

    if r.status_code != 200:
        raise RuntimeError(f"Błąd wyszukiwania ({r.status_code}): {r.text}")

    scenes = r.json().get("value", [])
    print(f"✓ Znaleziono {len(scenes)} scen(y) dla podanych kryteriów.")

    for s in scenes:
        print(f"   • {s['Name']}  |  {s['ContentDate']['Start'][:10]}")

    return scenes


# ─────────────────────────────────────────
# KROK 3 — Pobieranie scen
# ─────────────────────────────────────────
def download_scene(scene: dict, token: str) -> Path:
    """Pobiera pojedynczą scenę jako plik .zip."""
    scene_id   = scene["Id"]
    scene_name = scene["Name"]
    out_path   = OUTPUT_DIR / f"{scene_name}.zip"

    if out_path.exists():
        print(f"   ↷ Pominięto (już istnieje): {scene_name}")
        return out_path

    url = (
        f"https://zipper.dataspace.copernicus.eu"
        f"/odata/v1/Products({scene_id})/$value"
    )
    headers = {"Authorization": f"Bearer {token}"}

    print(f"   ↓ Pobieranie: {scene_name} ...", end="", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r   ↓ {scene_name}: {pct:.1f}%  ", end="", flush=True)

    print(f"\n   ✓ Zapisano: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ─────────────────────────────────────────
# KROK 4 — Zapis metadanych
# ─────────────────────────────────────────
def save_metadata(scenes: list[dict]) -> None:
    """Zapisuje metadane wyszukanych scen do JSON."""
    meta_path = OUTPUT_DIR / "scenes_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadane zapisane: {meta_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Monitoring erozji linii brzegowej — Pobieranie danych")
    print("=" * 55)
    print(f"  Obszar : Costa Brava (Platja d'Aro / Palamós)")
    print(f"  Okres  : {DATE_START}  →  {DATE_END}")
    print(f"  Katalog: {OUTPUT_DIR.resolve()}")
    print("=" * 55)

    try:
        token  = get_access_token(USERNAME, PASSWORD)
        scenes = search_scenes(token)

        if not scenes:
            print("Brak scen spełniających kryteria. Spróbuj poszerzyć zakres dat lub AOI.")
            return

        save_metadata(scenes)

        print(f"\nRozpoczynam pobieranie {len(scenes)} scen(y)...")
        for i, scene in enumerate(scenes, 1):
            print(f"\n[{i}/{len(scenes)}]")
            download_scene(scene, token)

        print("\n✓ Wszystkie sceny pobrane pomyślnie!")
        print(f"  Pliki: {OUTPUT_DIR.resolve()}")

    except Exception as e:
        print(f"\n✗ Błąd: {e}")
        raise


if __name__ == "__main__":
    main()