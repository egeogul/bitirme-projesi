"""
Tez grafik export scripti.
6 adet PNG üretir → figures/ klasörüne kaydeder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
RAW        = ROOT / "data" / "raw" / "train.csv"
PROC       = ROOT / "data" / "processed"
FEAT_PARQ  = PROC / "train_features.parquet"
PRED_PARQ  = PROC / "test_predictions.parquet"
FI_CSV     = PROC / "feature_importance.csv"
FIGURES    = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
})
PRIMARY   = "#2563EB"
SECONDARY = "#F59E0B"
ACCENT    = "#10B981"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Günlük satış trendi + 30-günlük hareketli ortalama
# ─────────────────────────────────────────────────────────────────────────────
print("[1/6] Günlük satış trendi + 30-günlük MA …")
df_raw = pd.read_csv(RAW, parse_dates=["date"])
daily  = df_raw.groupby("date")["sales"].sum().reset_index()
daily["ma30"] = daily["sales"].rolling(30, center=True).mean()

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(daily["date"], daily["sales"],
        color=PRIMARY, alpha=0.35, linewidth=0.8, label="Günlük Toplam Satış")
ax.plot(daily["date"], daily["ma30"],
        color=SECONDARY, linewidth=2.2, label="30-Günlük Hareketli Ortalama")
ax.set_title("Günlük Toplam Satış Trendi (2013–2017)")
ax.set_xlabel("Tarih")
ax.set_ylabel("Toplam Satış (tüm mağaza × ürün)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(framealpha=0.9)
fig.savefig(FIGURES / "01_daily_sales_trend.png")
plt.close(fig)
print("   → figures/01_daily_sales_trend.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Aylık mevsimsellik bar grafiği
# ─────────────────────────────────────────────────────────────────────────────
print("[2/6] Aylık mevsimsellik …")
df_raw["month"] = df_raw["date"].dt.month
monthly_avg = df_raw.groupby("month")["sales"].mean()
month_names = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(monthly_avg.index, monthly_avg.values,
              color=[ACCENT if v == monthly_avg.max()
                     else (SECONDARY if v == monthly_avg.min() else PRIMARY)
                     for v in monthly_avg.values],
              edgecolor="white", linewidth=0.6, width=0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title("Aylık Ortalama Satış Mevsimselliği (2013–2017)")
ax.set_xlabel("Ay")
ax.set_ylabel("Ortalama Günlük Satış")
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, monthly_avg.max() * 1.12)
# Legend notu
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ACCENT,    label="En Yüksek (Tem)"),
                   Patch(facecolor=SECONDARY, label="En Düşük (Oca)"),
                   Patch(facecolor=PRIMARY,   label="Diğer Aylar")]
ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9)
fig.savefig(FIGURES / "02_monthly_seasonality.png")
plt.close(fig)
print("   → figures/02_monthly_seasonality.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ACF / PACF grafiği
# ─────────────────────────────────────────────────────────────────────────────
print("[3/6] ACF / PACF …")
# Store 1, Item 1 serisi (günlük)
s1i1 = df_raw[(df_raw["store"] == 1) & (df_raw["item"] == 1)].sort_values("date")["sales"]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
plot_acf(s1i1, lags=60, ax=axes[0], color=PRIMARY,
         title="Otokorelasyon Fonksiyonu (ACF) — Store 1 / Item 1")
plot_pacf(s1i1, lags=60, ax=axes[1], color=SECONDARY,
          title="Kısmi Otokorelasyon Fonksiyonu (PACF) — Store 1 / Item 1",
          method="ywm")
for ax in axes:
    ax.set_xlabel("Gecikme (gün)")
    ax.set_ylabel("Korelasyon")
    # Vurgu çizgileri: lag 7 ve 14
    for lag in [7, 14, 30]:
        ax.axvline(lag, color="red", linestyle="--", alpha=0.4, linewidth=0.9)
axes[0].annotate("lag=7",  xy=(7, 0),  xytext=(7.4, 0.05), fontsize=8, color="red")
axes[0].annotate("lag=14", xy=(14, 0), xytext=(14.4, 0.05), fontsize=8, color="red")
plt.tight_layout()
fig.savefig(FIGURES / "03_acf_pacf.png")
plt.close(fig)
print("   → figures/03_acf_pacf.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Örnek store-item: rolling average + lag görselleştirmesi
# ─────────────────────────────────────────────────────────────────────────────
print("[4/6] Rolling average + lag grafiği …")
df_feat = pd.read_parquet(FEAT_PARQ)
s1i1_feat = (df_feat[(df_feat["store"] == 1) & (df_feat["item"] == 1)]
             .sort_values("date")
             .tail(365))   # son 1 yıl (2017)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Üst: rolling mean 7 & 30 + gerçek satış
axes[0].plot(s1i1_feat["date"], s1i1_feat["sales"],
             color=PRIMARY, alpha=0.4, linewidth=0.8, label="Gerçek Satış")
axes[0].plot(s1i1_feat["date"], s1i1_feat["rolling_mean_7"],
             color=SECONDARY, linewidth=1.8, label="7-günlük MA")
axes[0].plot(s1i1_feat["date"], s1i1_feat["rolling_mean_30"],
             color=ACCENT, linewidth=1.8, label="30-günlük MA")
axes[0].set_title("Store 1 / Item 1 — Rolling Ortalama (2017)")
axes[0].set_ylabel("Satış")
axes[0].legend(framealpha=0.9)

# Alt: lag_7 ve lag_365 karşılaştırması
axes[1].plot(s1i1_feat["date"], s1i1_feat["sales"],
             color=PRIMARY, alpha=0.4, linewidth=0.8, label="Gerçek Satış")
axes[1].plot(s1i1_feat["date"], s1i1_feat["lag_7"],
             color=SECONDARY, linewidth=1.5, linestyle="--", label="Lag-7")
axes[1].plot(s1i1_feat["date"], s1i1_feat["lag_365"],
             color="#EF4444", linewidth=1.5, linestyle=":", label="Lag-365 (geçen yıl)")
axes[1].set_title("Store 1 / Item 1 — Lag Özellikleri (2017)")
axes[1].set_xlabel("Tarih")
axes[1].set_ylabel("Satış")
axes[1].legend(framealpha=0.9)

plt.tight_layout()
fig.savefig(FIGURES / "04_rolling_lag_features.png")
plt.close(fig)
print("   → figures/04_rolling_lag_features.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Gerçek vs Tahmin (2017 test seti) — Store 1 / Item 1
# ─────────────────────────────────────────────────────────────────────────────
print("[5/6] Gerçek vs Tahmin (2017) …")
df_pred = pd.read_parquet(PRED_PARQ)
df_pred["date"] = pd.to_datetime(df_pred["date"])
s1i1_pred = (df_pred[(df_pred["store"] == 1) & (df_pred["item"] == 1)]
             .sort_values("date"))

mae_val  = s1i1_pred["error"].abs().mean()
smape_val = (2 * s1i1_pred["error"].abs() /
             (s1i1_pred["sales"].abs() + s1i1_pred["predicted"].abs())).mean() * 100

fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                         gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

# Üst: gerçek vs tahmin
axes[0].plot(s1i1_pred["date"], s1i1_pred["sales"],
             color=PRIMARY, linewidth=1.5, label="Gerçek Satış", alpha=0.85)
axes[0].plot(s1i1_pred["date"], s1i1_pred["predicted"],
             color=SECONDARY, linewidth=1.5, linestyle="--",
             label=f"LightGBM Tahmini  (MAE={mae_val:.2f}, SMAPE={smape_val:.1f}%)")
axes[0].fill_between(s1i1_pred["date"],
                     s1i1_pred["sales"], s1i1_pred["predicted"],
                     alpha=0.12, color="gray")
axes[0].set_title("Gerçek vs Tahmin — Store 1 / Item 1 (2017 Test Seti)")
axes[0].set_ylabel("Günlük Satış")
axes[0].legend(framealpha=0.9)

# Alt: hata (residual)
axes[1].bar(s1i1_pred["date"], s1i1_pred["error"],
            color=[ACCENT if e >= 0 else "#EF4444" for e in s1i1_pred["error"]],
            width=1, alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Tahmin Hatası (Gerçek − Tahmin)")
axes[1].set_xlabel("Tarih")
axes[1].set_ylabel("Hata")

plt.tight_layout()
fig.savefig(FIGURES / "05_actual_vs_predicted.png")
plt.close(fig)
print("   → figures/05_actual_vs_predicted.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Importance — Top 20
# ─────────────────────────────────────────────────────────────────────────────
print("[6/6] Feature importance top-20 …")
fi = pd.read_csv(FI_CSV).sort_values("importance", ascending=True).tail(20)

# Renk: kategori bazlı
def fi_color(name):
    if "lag" in name:     return "#7C3AED"   # mor → lag
    if "rolling" in name: return SECONDARY    # sarı → rolling
    if "store" in name or "item" in name or "store_item" in name:
        return ACCENT                          # yeşil → etkileşim
    return PRIMARY                             # mavi → diğer

colors = [fi_color(f) for f in fi["feature"]]

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="white", linewidth=0.5)
ax.set_title("LightGBM — Top 20 Özellik Önemi (Gain)")
ax.set_xlabel("Önem Skoru (Gain)")
ax.set_ylabel("")
for bar, val in zip(bars, fi["importance"]):
    ax.text(val + fi["importance"].max() * 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlim(0, fi["importance"].max() * 1.14)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#7C3AED", label="Lag Özellikleri"),
    Patch(facecolor=SECONDARY, label="Rolling İstatistikler"),
    Patch(facecolor=ACCENT,    label="Mağaza×Ürün Etkileşimi"),
    Patch(facecolor=PRIMARY,   label="Tarih / Diğer"),
]
ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9, loc="lower right")
fig.savefig(FIGURES / "06_feature_importance_top20.png")
plt.close(fig)
print("   → figures/06_feature_importance_top20.png")

# ─────────────────────────────────────────────────────────────────────────────
print("\n✓ Tüm grafikler figures/ klasörüne kaydedildi.")
print(f"  Konum: {FIGURES}")
for p in sorted(FIGURES.glob("*.png")):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name:<45} {size_kb:6.1f} KB")
