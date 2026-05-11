"""Diagnostyka struktury pobranych plików Sentinel-1"""
from pathlib import Path

RAW_DIR = Path("data/raw")

# Pokaż wszystko co jest w data/raw
print("=== Zawartość data/raw/ ===")
for f in sorted(RAW_DIR.iterdir()):
    print(f"  {f.name}  ({'katalog' if f.is_dir() else f'plik, {f.stat().st_size/1e6:.1f} MB'})")

# Dla każdego folderu .SAFE pokaż strukturę
print("\n=== Struktura folderów .SAFE ===")
for safe in sorted(RAW_DIR.glob("*.SAFE")):
    print(f"\n📁 {safe.name}/")
    for f in sorted(safe.rglob("*")):
        if f.is_file():
            rel = f.relative_to(safe)
            print(f"   {rel}  ({f.stat().st_size/1e6:.2f} MB)")
