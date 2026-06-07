# Monitoring erozji linii brzegowej (SAR / Sentinel-1)

Prosty, ale w pełni działający projekt w Pythonie do:

1. **wykrywania linii brzegowej** z obrazów radarowych SAR (GeoTIFF),
2. **analizy, jak linia brzegowa przesuwa się przez kilka lat** na wybranym terenie,
3. obliczania wskaźnika **EPR (End Point Rate)** — tempa zmian w metrach na rok.

Projekt celowo ograniczono do najważniejszych funkcji (wykrywanie + analiza
wieloczasowa), zgodnie z założeniem „jak najprościej, ale ma działać".

---

## Co dostajesz na wyjściu

| Plik | Opis |
|------|------|
| `mapa_zmian.png` | Wszystkie linie brzegowe (kolor = rok) + przekroje pokolorowane wg tempa erozji/akumulacji |
| `wykres_epr.png` | Słupkowy wykres EPR dla kolejnych przekrojów wzdłuż brzegu |
| `animacja.gif` | Animacja pokazująca przesuwanie się brzegu rok po roku |
| `epr_table.csv` | Tabela: NSM [m] i EPR [m/rok] dla każdego przekroju |
| `shoreline_positions.csv` | Pozycja brzegu (odległość od linii bazowej) w każdym roku |
| `linie_brzegowe.geojson` | Linie brzegowe do otwarcia w QGIS |
| `epr_punkty.geojson` | Punkty z wartością EPR do otwarcia w QGIS |

**Interpretacja EPR:** wartość **ujemna = erozja** (brzeg się cofa),
wartość **dodatnia = akumulacja** (brzeg przyrasta).

---

## Instalacja

```bash
pip install -r requirements.txt
```

## Szybki start — demo (bez pobierania danych)

Najszybszy sposób, żeby zobaczyć, że wszystko działa. Skrypt sam generuje
5 sztucznych scen SAR z realistycznym szumem plamkowym i brzegiem, który
z roku na rok się zmienia, a następnie uruchamia całą analizę:

```bash
python demo.py
```

Wyniki pojawią się w `data/output/`.

## Użycie na prawdziwych danych

1. Wrzuć pliki GeoTIFF do folderu (np. `data/raw/`). Wymagania:
   - jeden plik = jeden rok,
   - **rok w nazwie pliku**, np. `sentinel1_2019.tif`, `S1_2021_vv.tif`,
   - dane w układzie metrycznym (UTM), żeby EPR wyszło w m/rok.
2. Uruchom:

```bash
python main.py --input data/raw --output data/output
```

Parametry (opcjonalne):

```bash
python main.py --input data/raw \
    --spacing 200 \    # odstęp między przekrojami [m]
    --length 2000 \    # długość przekroju [m]
    --offset 800 \     # odsunięcie linii bazowej w głąb lądu [m]
    --speckle 5        # rozmiar filtra redukcji szumu [px]
```

---

## Skąd wziąć dane Sentinel-1 (ważne!)

Wymieniona w pierwotnym opisie biblioteka **`sentinelsat` już nie działa** —
stary Copernicus Open Access Hub (SciHub) został trwale zamknięty. Dane
pobiera się teraz z **Copernicus Data Space Ecosystem (CDSE)**:

- **Ręcznie (najprościej):** [Copernicus Browser](https://browser.dataspace.copernicus.eu)
  → zaznacz obszar (np. Mierzeja Wiślana), wybierz misję **Sentinel-1**,
  produkt **GRD**, pobierz sceny z różnych lat.
- **Przez API** (po założeniu darmowego konta): OData / STAC / openEO, albo
  nieoficjalna biblioteka `cdse-client` (następca sentinelsat).

**Zalecane przygotowanie danych** (program ESA **SNAP**, darmowy):
kalibracja radiometryczna → filtr speckle → korekcja terenu (Range-Doppler)
→ zapis do GeoTIFF w układzie UTM, polaryzacja **VV**. Bez tego skrypt też
zadziała, ale wyniki będą mniej dokładne.

---

## Jak to działa (w skrócie)

**Wykrywanie brzegu** (`coastline.py`): na obrazach SAR woda jest ciemna
(niskie odbicie), a ląd jasny. Po redukcji szumu plamkowego (filtr medianowy)
i progowaniu metodą **Otsu** powstaje maska ląd/woda, a jej kontur to linia
brzegowa. Współrzędne pikselowe są przeliczane na geograficzne.

**Analiza wieloczasowa** (`analysis.py`): uproszczona metoda **DSAS** —
z najstarszej linii tworzona jest linia bazowa po stronie lądu, prostopadle
do niej stawiane są przekroje, a w każdym przekroju mierzona jest pozycja
brzegu w kolejnych latach. Z różnicy pozycji liczony jest EPR.

**Wizualizacja** (`visualize.py`): mapa, wykres i animacja.

## Struktura projektu

```
coastline_monitor/
├── coastline.py      # wykrywanie linii brzegowej z 1 sceny
├── analysis.py       # linia bazowa, przekroje, EPR, eksport GeoJSON
├── visualize.py      # mapa, wykres EPR, animacja
├── demo.py           # dane testowe + pełna demonstracja
├── main.py           # uruchomienie na prawdziwych danych (CLI)
├── requirements.txt
└── data/
    ├── raw/          # tu wrzucasz pliki GeoTIFF
    └── output/       # tu trafiają wyniki
```

## Ograniczenia

- Progowanie Otsu może mieć problem przy bardzo wzburzonym morzu (wiatr
  podnosi odbicie wody). Pomaga wcześniejsza korekcja terenu i filtr speckle.
- Metoda przekrojów działa najlepiej dla brzegów o w miarę regularnym
  przebiegu; przy mocno poszarpanej linii przekroje mogą się krzyżować.
