"""
ETAP 6 — Interfejs graficzny (GUI)
Aplikacja do wizualizacji zmian linii brzegowej Costa Brava.

Funkcje:
  • Lista dostępnych zobrazowań z datami
  • Podgląd pojedynczej sceny (linia brzegowa na tle SAR)
  • Porównanie dwóch dat na jednej mapie
  • Wyświetlenie mapy EPR (erozja / akumulacja)

Uruchomienie:
  python src/06_gui.py
"""

import re
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from shapely.ops import unary_union

# ─────────────────────────────────────────
# ŚCIEŻKI
# ─────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = BASE_DIR / "output"
LINES_DIR   = OUTPUT_DIR / "shorelines"
MAPS_DIR    = OUTPUT_DIR / "maps"
EPR_PATH    = OUTPUT_DIR / "epr_transects.geojson"
BASE_PATH   = OUTPUT_DIR / "baseline.geojson"

# ─────────────────────────────────────────
# KOLORY / STYL
# ─────────────────────────────────────────
BG_DARK     = "#1e1e2e"
BG_PANEL    = "#2a2a3e"
BG_CARD     = "#313145"
FG_TEXT     = "#cdd6f4"
FG_MUTED    = "#7f849c"
ACCENT      = "#89b4fa"
ACCENT2     = "#a6e3a1"
RED         = "#f38ba8"
BORDER      = "#45475a"

FONT_TITLE  = ("Segoe UI", 13, "bold")
FONT_LABEL  = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)


# ─────────────────────────────────────────
# ŁADOWANIE DANYCH
# ─────────────────────────────────────────
def load_shorelines() -> list[dict]:
    """Wczytuje wszystkie GeoJSON z katalogu shorelines."""
    files = sorted(LINES_DIR.glob("shoreline_*.geojson"))
    result = []
    for f in files:
        gdf = gpd.read_file(f)
        if gdf.empty:
            continue
        parts    = f.stem.split("_")
        date_str = parts[1] if len(parts) > 1 else "19700101"
        scene_id = parts[2] if len(parts) > 2 else "????"
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            date = datetime(1970, 1, 1)
        geom = gdf.geometry.iloc[0]
        result.append({
            "file":     f,
            "date":     date,
            "date_str": date_str,
            "scene_id": scene_id,
            "label":    date.strftime("%d %b %Y") + f"  [{scene_id}]",
            "geom":     geom,
            "crs":      gdf.crs,
        })
    result.sort(key=lambda x: x["date"])
    return result


def load_epr() -> gpd.GeoDataFrame | None:
    if EPR_PATH.exists():
        return gpd.read_file(EPR_PATH)
    return None


# ─────────────────────────────────────────
# RYSOWANIE
# ─────────────────────────────────────────
def _plot_geom(ax, geom, **kwargs):
    """Rysuje LineString lub MultiLineString na osi."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        ax.plot(*geom.xy, **kwargs)
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            ax.plot(*g.xy, **kwargs)
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            if g.geom_type in ("LineString", "MultiLineString"):
                _plot_geom(ax, g, **kwargs)


def common_extent(geoms: list, pad_frac: float = 0.05):
    """Oblicza wspólny zasięg dla listy geometrii."""
    bounds = [g.bounds for g in geoms if g is not None and not g.is_empty]
    if not bounds:
        return None
    x0 = min(b[0] for b in bounds)
    y0 = min(b[1] for b in bounds)
    x1 = max(b[2] for b in bounds)
    y1 = max(b[3] for b in bounds)
    px = (x1 - x0) * pad_frac or 500
    py = (y1 - y0) * pad_frac or 500
    return x0 - px, y0 - py, x1 + px, y1 + py


def draw_single(ax, entry: dict):
    """Rysuje pojedynczą linię brzegową."""
    ax.clear()
    ax.set_facecolor("#0d1117")
    _plot_geom(ax, entry["geom"], color=ACCENT, linewidth=1.8, label=entry["label"])
    ext = common_extent([entry["geom"]])
    if ext:
        ax.set_xlim(ext[0], ext[2])
        ax.set_ylim(ext[1], ext[3])
    ax.set_title(
        f"Linia brzegowa — Costa Brava\n{entry['date'].strftime('%d %B %Y')}  (ID: {entry['scene_id']})",
        color=FG_TEXT, fontsize=11, pad=10
    )
    ax.set_xlabel("Easting [m]",  color=FG_MUTED, fontsize=9)
    ax.set_ylabel("Northing [m]", color=FG_MUTED, fontsize=9)
    ax.tick_params(colors=FG_MUTED, labelsize=8)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
              labelcolor=FG_TEXT, loc="best")
    ax.set_aspect("equal")


def draw_compare(ax, entry1: dict, entry2: dict):
    """Rysuje dwie linie brzegowe na jednej mapie."""
    ax.clear()
    ax.set_facecolor("#0d1117")

    _plot_geom(ax, entry1["geom"], color=ACCENT,  linewidth=1.8,
               label=f"Wcześniej: {entry1['date'].strftime('%d %b %Y')}", zorder=3)
    _plot_geom(ax, entry2["geom"], color=RED,     linewidth=1.8,
               label=f"Później:   {entry2['date'].strftime('%d %b %Y')}", zorder=4)

    # Oblicz czas między scenami
    dt_days = (entry2["date"] - entry1["date"]).days
    dt_years = dt_days / 365.25

    ext = common_extent([entry1["geom"], entry2["geom"]])
    if ext:
        ax.set_xlim(ext[0], ext[2])
        ax.set_ylim(ext[1], ext[3])

    ax.set_title(
        f"Porównanie linii brzegowej — Costa Brava\n"
        f"Δt = {dt_days} dni ({dt_years:.1f} lat)",
        color=FG_TEXT, fontsize=11, pad=10
    )
    ax.set_xlabel("Easting [m]",  color=FG_MUTED, fontsize=9)
    ax.set_ylabel("Northing [m]", color=FG_MUTED, fontsize=9)
    ax.tick_params(colors=FG_MUTED, labelsize=8)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER,
              labelcolor=FG_TEXT, loc="best")
    ax.set_aspect("equal")


def draw_epr(ax, shorelines: list[dict], gdf_epr: gpd.GeoDataFrame):
    """Rysuje mapę EPR z transektami pokolorowanymi wg tempa zmian."""
    ax.clear()
    ax.set_facecolor("#0d1117")

    # Wszystkie linie brzegowe jako tło
    colors_bg = cm.Blues(np.linspace(0.3, 0.85, len(shorelines)))
    for sl, col in zip(shorelines, colors_bg):
        _plot_geom(ax, sl["geom"], color=(*col[:3], 0.45), linewidth=0.8)

    # Baseline
    if BASE_PATH.exists():
        gdf_b = gpd.read_file(BASE_PATH)
        if not gdf_b.empty:
            bgeom = gdf_b.geometry.iloc[0]
            if hasattr(bgeom, "xy"):
                ax.plot(*bgeom.xy, color="white", lw=1.0, ls="--",
                        alpha=0.6, label="Baseline", zorder=3)

    # Transekty EPR
    epr_vals = gdf_epr["epr_m_yr"]
    vmax = max(abs(epr_vals.min()), abs(epr_vals.max()), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = cm.RdYlGn

    for _, row in gdf_epr.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        color = cmap(norm(row["epr_m_yr"]))
        ax.plot(*geom.xy, color=color, linewidth=1.6, alpha=0.9, zorder=4)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("EPR [m/rok]", color=FG_TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=FG_MUTED, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG_TEXT)

    # Statystyki w rogu
    mean_epr = epr_vals.mean()
    erosion_pct = (epr_vals < 0).mean() * 100
    info = (f"n={len(gdf_epr)} transektów\n"
            f"Śr. EPR: {mean_epr:+.2f} m/rok\n"
            f"Erozja: {erosion_pct:.0f}%")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            color=FG_TEXT, fontfamily="monospace",
            bbox=dict(facecolor=BG_CARD, edgecolor=BORDER, alpha=0.85, boxstyle="round,pad=0.4"))

    ext = common_extent([sl["geom"] for sl in shorelines])
    if ext:
        ax.set_xlim(ext[0], ext[2])
        ax.set_ylim(ext[1], ext[3])

    ax.set_title("Mapa EPR — Costa Brava\n(zielony = akumulacja, czerwony = erozja)",
                 color=FG_TEXT, fontsize=11, pad=10)
    ax.set_xlabel("Easting [m]",  color=FG_MUTED, fontsize=9)
    ax.set_ylabel("Northing [m]", color=FG_MUTED, fontsize=9)
    ax.tick_params(colors=FG_MUTED, labelsize=8)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_aspect("equal")

    handles = [plt.Line2D([0], [0], color="white", ls="--", alpha=0.6, label="Baseline")]
    handles += [mpatches.Patch(facecolor=cm.RdYlGn(norm(v)),
                               label=f"{sl['date'].strftime('%Y-%m-%d')}")
                for sl, v in zip(shorelines, np.linspace(-vmax * 0.5, vmax * 0.5, len(shorelines)))]
    ax.legend(handles=handles, fontsize=7, facecolor=BG_CARD,
              edgecolor=BORDER, labelcolor=FG_TEXT, loc="lower right")


# ─────────────────────────────────────────
# GŁÓWNA KLASA APLIKACJI
# ─────────────────────────────────────────
class ShoreLineApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ShoreL — Monitoring linii brzegowej · Costa Brava")
        self.configure(bg=BG_DARK)
        self.geometry("1280x780")
        self.minsize(900, 600)

        # Dane
        self.shorelines = load_shorelines()
        self.gdf_epr    = load_epr()

        self._build_ui()

        if self.shorelines:
            self.scene_list.selection_set(0)
            self._on_scene_select(None)
        else:
            self._show_no_data()

    # ─── Budowa UI ───────────────────────
    def _build_ui(self):
        # ── Lewa kolumna (panel sterowania)
        left = tk.Frame(self, bg=BG_PANEL, width=260)
        left.pack(side="left", fill="y", padx=0, pady=0)
        left.pack_propagate(False)

        self._build_left_panel(left)

        # ── Separator
        sep = tk.Frame(self, bg=BORDER, width=1)
        sep.pack(side="left", fill="y")

        # ── Prawa kolumna (mapa)
        right = tk.Frame(self, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        self._build_map_area(right)
        self._build_statusbar(right)

    def _build_left_panel(self, parent):
        # Nagłówek
        hdr = tk.Frame(parent, bg=BG_PANEL)
        hdr.pack(fill="x", padx=16, pady=(20, 8))

        tk.Label(hdr, text="🌊 ShoreL", font=("Segoe UI", 15, "bold"),
                 bg=BG_PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(hdr, text="Costa Brava · Sentinel-1 SAR", font=FONT_SMALL,
                 bg=BG_PANEL, fg=FG_MUTED).pack(anchor="w")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=10)

        # ── Sekcja: lista scen
        tk.Label(parent, text="DOSTĘPNE SCENY", font=("Segoe UI", 8, "bold"),
                 bg=BG_PANEL, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(4, 4))

        list_frame = tk.Frame(parent, bg=BG_PANEL)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        scrollbar = tk.Scrollbar(list_frame, bg=BG_CARD, troughcolor=BG_PANEL,
                                 relief="flat", bd=0)
        scrollbar.pack(side="right", fill="y")

        self.scene_list = tk.Listbox(
            list_frame,
            bg=BG_CARD, fg=FG_TEXT, selectbackground=ACCENT,
            selectforeground=BG_DARK, font=FONT_MONO,
            relief="flat", bd=0, highlightthickness=0,
            activestyle="none",
            yscrollcommand=scrollbar.set
        )
        self.scene_list.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.scene_list.yview)

        for sl in self.shorelines:
            self.scene_list.insert("end", f"  {sl['date'].strftime('%Y-%m-%d')}  {sl['scene_id']}")

        self.scene_list.bind("<<ListboxSelect>>", self._on_scene_select)

        btn_preview = tk.Button(
            parent, text="👁  Pokaż scenę",
            command=self._show_single,
            bg=ACCENT, fg=BG_DARK, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=10, pady=7,
            cursor="hand2", activebackground="#74a8f0", activeforeground=BG_DARK
        )
        btn_preview.pack(fill="x", padx=12, pady=(2, 10))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Sekcja: porównanie
        tk.Label(parent, text="PORÓWNAJ DWA ZDJĘCIA", font=("Segoe UI", 8, "bold"),
                 bg=BG_PANEL, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(4, 6))

        labels = ["Data 1 (wcześniej):", "Data 2 (później):"]
        self.combo_vars = []

        date_options = [sl["date"].strftime("%Y-%m-%d") + f"  {sl['scene_id']}"
                        for sl in self.shorelines]

        for i, lbl in enumerate(labels):
            tk.Label(parent, text=lbl, font=FONT_SMALL,
                     bg=BG_PANEL, fg=FG_TEXT).pack(anchor="w", padx=16, pady=(4, 2))

            var = tk.StringVar()
            combo = ttk.Combobox(parent, textvariable=var, values=date_options,
                                 state="readonly", font=FONT_SMALL)
            combo.pack(fill="x", padx=12, pady=(0, 4))

            if self.shorelines:
                idx = min(i, len(self.shorelines) - 1)
                combo.current(idx)

            self.combo_vars.append(var)

        btn_compare = tk.Button(
            parent, text="🔍  Porównaj",
            command=self._show_compare,
            bg=ACCENT2, fg=BG_DARK, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=10, pady=7,
            cursor="hand2", activebackground="#8dd9a0", activeforeground=BG_DARK
        )
        btn_compare.pack(fill="x", padx=12, pady=(4, 10))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Sekcja: EPR
        tk.Label(parent, text="ANALIZA EPR", font=("Segoe UI", 8, "bold"),
                 bg=BG_PANEL, fg=FG_MUTED).pack(anchor="w", padx=16, pady=(4, 6))

        epr_status = "✓ dostępna" if self.gdf_epr is not None else "✗ brak — uruchom etap 4"
        epr_color  = ACCENT2 if self.gdf_epr is not None else RED

        tk.Label(parent, text=epr_status, font=FONT_SMALL,
                 bg=BG_PANEL, fg=epr_color).pack(anchor="w", padx=16, pady=(0, 6))

        btn_epr = tk.Button(
            parent, text="📊  Mapa EPR",
            command=self._show_epr,
            bg="#cba6f7", fg=BG_DARK, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, padx=10, pady=7,
            cursor="hand2", activebackground="#b591e5", activeforeground=BG_DARK,
            state="normal" if self.gdf_epr is not None else "disabled"
        )
        btn_epr.pack(fill="x", padx=12, pady=(0, 16))

        # Info na dole
        n_scenes = len(self.shorelines)
        tk.Label(parent, text=f"Załadowano {n_scenes} scen(y)", font=FONT_SMALL,
                 bg=BG_PANEL, fg=FG_MUTED).pack(side="bottom", pady=10)

    def _build_map_area(self, parent):
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(9, 6.5),
                                          facecolor=BG_DARK)
        self.ax.set_facecolor("#0d1117")
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.09)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Pasek narzędzi matplotlib
        toolbar_frame = tk.Frame(parent, bg=BG_DARK)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(bg=BG_DARK)
        toolbar.update()

    def _build_statusbar(self, parent):
        bar = tk.Frame(parent, bg=BG_PANEL, height=26)
        bar.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="Gotowy · wybierz scenę z listy")
        tk.Label(bar, textvariable=self.status_var, font=FONT_SMALL,
                 bg=BG_PANEL, fg=FG_MUTED, anchor="w").pack(side="left", padx=12)

    # ─── Akcje ───────────────────────────
    def _get_selected_scene(self) -> dict | None:
        sel = self.scene_list.curselection()
        if not sel:
            return None
        return self.shorelines[sel[0]]

    def _on_scene_select(self, event):
        entry = self._get_selected_scene()
        if entry:
            self.status_var.set(f"Wybrano: {entry['date'].strftime('%d %B %Y')}  [ID: {entry['scene_id']}]")

    def _show_single(self):
        entry = self._get_selected_scene()
        if not entry:
            messagebox.showwarning("Brak wyboru", "Wybierz scenę z listy.")
            return
        self.status_var.set(f"Wyświetlam: {entry['date'].strftime('%d %B %Y')} …")
        self.update_idletasks()
        draw_single(self.ax, entry)
        self.canvas.draw()
        self.status_var.set(f"✓ Scena: {entry['date'].strftime('%d %B %Y')}  |  ID: {entry['scene_id']}")

    def _resolve_combo(self, var: tk.StringVar) -> dict | None:
        val = var.get().strip()
        if not val:
            return None
        for sl in self.shorelines:
            key = sl["date"].strftime("%Y-%m-%d") + f"  {sl['scene_id']}"
            if key == val:
                return sl
        return None

    def _show_compare(self):
        e1 = self._resolve_combo(self.combo_vars[0])
        e2 = self._resolve_combo(self.combo_vars[1])

        if not e1 or not e2:
            messagebox.showwarning("Brak danych", "Wybierz obie daty do porównania.")
            return
        if e1["date"] == e2["date"]:
            messagebox.showwarning("Identyczne daty", "Wybierz dwie różne daty.")
            return

        # Upewnij się że e1 < e2
        if e1["date"] > e2["date"]:
            e1, e2 = e2, e1

        self.status_var.set("Porównuję dwie sceny …")
        self.update_idletasks()
        draw_compare(self.ax, e1, e2)
        self.canvas.draw()
        dt = (e2["date"] - e1["date"]).days
        self.status_var.set(
            f"✓ Porównanie: {e1['date'].strftime('%Y-%m-%d')} vs {e2['date'].strftime('%Y-%m-%d')} "
            f"(Δt = {dt} dni)"
        )

    def _show_epr(self):
        if self.gdf_epr is None:
            messagebox.showerror("Brak danych EPR",
                                 "Nie znaleziono pliku epr_transects.geojson.\n"
                                 "Uruchom najpierw: python src/04_analysis_epr.py")
            return
        if not self.shorelines:
            messagebox.showerror("Brak linii", "Brak danych o liniach brzegowych.")
            return

        self.status_var.set("Ładuję mapę EPR …")
        self.update_idletasks()

        # Usuń stary colorbar jeśli istnieje
        if len(self.fig.axes) > 1:
            for cax in self.fig.axes[1:]:
                cax.remove()

        draw_epr(self.ax, self.shorelines, self.gdf_epr)
        self.canvas.draw()
        n = len(self.gdf_epr)
        mean_epr = self.gdf_epr["epr_m_yr"].mean()
        self.status_var.set(
            f"✓ Mapa EPR  |  {n} transektów  |  Śr. EPR: {mean_epr:+.2f} m/rok"
        )

    def _show_no_data(self):
        self.ax.clear()
        self.ax.set_facecolor("#0d1117")
        self.ax.text(
            0.5, 0.55,
            "Brak danych linii brzegowych",
            transform=self.ax.transAxes,
            ha="center", va="center",
            color=RED, fontsize=14, fontweight="bold"
        )
        self.ax.text(
            0.5, 0.44,
            "Uruchom etapy 1 → 2 → 3, aby pobrać\ni przetworzyć dane Sentinel-1.",
            transform=self.ax.transAxes,
            ha="center", va="center",
            color=FG_MUTED, fontsize=10
        )
        self.ax.axis("off")
        self.canvas.draw()
        self.status_var.set("⚠ Brak scen — najpierw uruchom etapy 1–3")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    app = ShoreLineApp()
    app.mainloop()
