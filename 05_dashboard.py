"""
Supply Chain Decision Support System — Streamlit Dashboard
Çalıştırmak için:
    streamlit run 05_dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Sayfa yapılandırması ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain DSS",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Renk sabitleri ────────────────────────────────────────────────────────────
ACTION_COLORS = {
    "CRITICAL":  "#d62728",
    "LOW":       "#ff7f0e",
    "OK":        "#2ca02c",
    "OVERSTOCK": "#9467bd",
}
ACTION_ICONS = {
    "CRITICAL":  "🔴",
    "LOW":       "🟠",
    "OK":        "🟢",
    "OVERSTOCK": "🟣",
}

PROCESSED = Path("data/processed")


# ─── Veri yükleme (önbelleklenmiş) ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    pred = pd.read_parquet(PROCESSED / "test_predictions.parquet")
    pred["date"] = pd.to_datetime(pred["date"])

    rec = pd.read_parquet(PROCESSED / "inventory_recommendations.parquet")

    fi = pd.read_csv(PROCESSED / "feature_importance.csv")

    with open(PROCESSED / "model_meta.json") as f:
        meta = json.load(f)

    return pred, rec, fi, meta


pred_df, rec_df, fi_df, model_meta = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Supply Chain DSS")
    st.caption("AI-Powered Decision Support System")
    st.divider()

    st.subheader("🔍 Filtreler")

    stores = sorted(rec_df["store"].unique())
    items  = sorted(rec_df["item"].unique())

    sel_stores = st.multiselect(
        "Mağaza", stores, default=stores, key="stores"
    )
    sel_items = st.multiselect(
        "Ürün", items, default=items[:10], key="items"
    )
    sel_actions = st.multiselect(
        "Aksiyon",
        options=["CRITICAL", "LOW", "OK", "OVERSTOCK"],
        default=["CRITICAL", "LOW", "OK", "OVERSTOCK"],
        key="actions",
    )

    st.divider()
    st.caption(
        f"**Model:** {model_meta['model_name']}  \n"
        f"**Train:** {model_meta['train_period']['start']} → "
        f"{model_meta['train_period']['end']}  \n"
        f"**Test:** {model_meta['test_period']['start']} → "
        f"{model_meta['test_period']['end']}"
    )

    st.divider()
    st.markdown("**Renk Kodları**")
    st.markdown(
        "🔴 **CRITICAL** — Stok güvenlik seviyesinin altında, acil sipariş gerekli  \n"
        "🟠 **LOW** — Yeniden sipariş noktasının altında, sipariş verilmeli  \n"
        "🟢 **OK** — Stok seviyesi normal aralıkta  \n"
        "🟣 **OVERSTOCK** — Fazla stok, yeni alım durdurulmalı"
    )

# ─── Veri filtreleme ──────────────────────────────────────────────────────────
if not sel_stores or not sel_items:
    st.warning("Lütfen en az bir mağaza ve ürün seçin.")
    st.stop()

rec_f  = rec_df[
    rec_df["store"].isin(sel_stores) &
    rec_df["item"].isin(sel_items) &
    rec_df["action"].isin(sel_actions)
].copy()

pred_f = pred_df[
    pred_df["store"].isin(sel_stores) &
    pred_df["item"].isin(sel_items)
].copy()

# ─── Nasıl Kullanılır? ────────────────────────────────────────────────────────
with st.expander("💡 Nasıl Kullanılır?", expanded=False):
    st.markdown(
        """
        Bu dashboard, yapay zeka destekli bir **tedarik zinciri karar destek sistemi**dir.
        LightGBM ile üretilen satış tahminleri ve envanter optimizasyon algoritmaları
        kullanılarak mağaza–ürün bazında aksiyonlar önerilir.

        ---

        ### Filtreler (Sol Panel)
        | Filtre | Açıklama |
        |--------|----------|
        | **Mağaza** | Analiz edilecek mağazaları seçin. Birden fazla seçilebilir. |
        | **Ürün** | Görüntülenecek ürünleri seçin. |
        | **Aksiyon** | Yalnızca belirli risk seviyelerini listelemek için filtreleyin. |

        > Tüm sekmeler bu filtreleri **eş zamanlı** olarak uygular.

        ---

        ### Sekmeler
        | Sekme | Ne Gösterir? |
        |-------|-------------|
        | 📊 **Genel Bakış** | Toplam kombinasyon sayısı, aksiyon dağılımı (pasta & bar grafik) ve mağaza×ürün ısı haritası. Tüm envanterin tek bakışta özetini verir. |
        | 📈 **Tahmin Grafikleri** | Seçilen mağaza ve ürün için 2017 yılı gerçek satış vs LightGBM tahmini. ±RMSE bandı, residual dağılımı, aylık MAE/RMSE içerir. |
        | 📦 **Stok & Aksiyonlar** | Gauge göstergesi (Safety Stock, ROP, EOQ eşikleri), renk kodlu aksiyon tablosu ve stok vs ROP karşılaştırma grafiği. |
        | ⚠️ **Risk Analizi** | Stok tükenme olasılıkları, risk matrisi, mağaza bazında risk özeti ve hizmet düzeyi duyarlılık analizi. |
        | 🤖 **Model Bilgisi** | Test metrikleri (MAE, RMSE, MAPE), feature importance sıralaması, özellik grup katkıları ve model parametreleri. |

        ---

        ### Renk Kodları
        | Renk | Aksiyon | Stok Durumu | Önerilen Adım |
        |------|---------|-------------|---------------|
        | 🔴 **Kırmızı** | CRITICAL | Mevcut stok < Safety Stock | Acil sipariş ver (EOQ kadar) |
        | 🟠 **Turuncu** | LOW | Safety Stock ≤ stok < ROP | Sipariş oluştur |
        | 🟢 **Yeşil** | OK | ROP ≤ stok ≤ ROP + 2×EOQ | Herhangi bir aksiyon gerekmez |
        | 🟣 **Mor** | OVERSTOCK | Stok > ROP + 2×EOQ | Yeni alım durdur, tüketimi bekle |

        ---

        ### Temel Kavramlar
        - **Safety Stock (SS):** Talep dalgalanmalarına karşı tampon stok.
          Formül: `z × σ_talep × √(Tedarik süresi + Gözden geçirme periyodu)`
        - **ROP (Reorder Point):** Sipariş verilmesi gereken stok seviyesi.
          Dinamik ROP, LightGBM'nin gelecek 7 günlük tahminini kullanır.
        - **EOQ (Economic Order Quantity):** Sipariş ve elde tutma maliyetlerini
          minimize eden optimal sipariş miktarı (Wilson formülü).
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SEKME YAPISI
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_forecast, tab_inventory, tab_risk, tab_model = st.tabs([
    "📊 Overview",
    "📈 Forecast Charts",
    "📦 Inventory and Actions",
    "⚠️ Risk Analysis",
    "🤖 Model Information",
])

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GENEL BAKIŞ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Genel Bakış")

    # KPI kartları
    total_comb   = len(rec_f)
    n_critical   = (rec_f["action"] == "CRITICAL").sum()
    n_low        = (rec_f["action"] == "LOW").sum()
    n_ok         = (rec_f["action"] == "OK").sum()
    n_overstock  = (rec_f["action"] == "OVERSTOCK").sum()
    total_order  = rec_f["order_qty"].sum()
    avg_ss       = rec_f["safety_stock"].mean()

    st.caption("Seçili mağaza ve ürünlerdeki stok durumunun anlık özeti. Kırmızı ve turuncu sayılar ne kadar acil aksiyon gerektiğini gösterir.")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Toplam Kombinasyon", f"{total_comb}")
    k2.metric("🔴 Kritik",   f"{n_critical}",
              delta=f"{n_critical/total_comb:.0%}" if total_comb else "—",
              delta_color="inverse")
    k3.metric("🟠 Düşük",    f"{n_low}",
              delta=f"{n_low/total_comb:.0%}" if total_comb else "—",
              delta_color="inverse")
    k4.metric("🟢 Normal",   f"{n_ok}")
    k5.metric("🟣 Fazla",    f"{n_overstock}")
    k6.metric("Toplam Sipariş", f"{total_order:,.0f} br")

    st.divider()
    col_left, col_right = st.columns([1, 2])

    # Pasta grafik
    with col_left:
        st.caption("Tüm ürünlerin kaçının acil durumda, kaçının normal seviyede olduğunu tek bakışta gösterir. Kırmızı dilim büyüdükçe envanter yönetimi daha kritik hale gelir.")
        action_counts = rec_f["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        fig_pie = px.pie(
            action_counts,
            names="action", values="count",
            color="action",
            color_discrete_map=ACTION_COLORS,
            hole=0.45,
            title="Aksiyon Dağılımı",
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Mağaza bazında aksiyon dağılımı
    with col_right:
        st.caption("Her mağazanın ürün bazında hangi oranda sorunlu olduğunu karşılaştırır. Kırmızı barı yüksek olan mağaza en fazla acil sipariş bekleyen yerdir.")
        store_action = (
            rec_f.groupby(["store", "action"])
                  .size()
                  .reset_index(name="count")
        )
        fig_bar = px.bar(
            store_action,
            x="store", y="count", color="action",
            color_discrete_map=ACTION_COLORS,
            barmode="stack",
            title="Mağaza Bazında Aksiyon Dağılımı",
            labels={"store": "Mağaza", "count": "Ürün Sayısı"},
        )
        fig_bar.update_layout(height=380, legend_title_text="Aksiyon")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Isı haritası
    st.subheader("Mağaza × Ürün Aksiyon Isı Haritası")
    st.caption("Her karenin rengi, o mağaza–ürün çiftinin stok durumunu gösterir. Kırmızı kare = o ürün o mağazada tükenmek üzere; yeşil kare = stok yeterli. Satır satır tarayarak hangi mağazanın en çok sorun yaşadığını görebilirsiniz.")
    heat_vals = {"OK": 0, "OVERSTOCK": 1, "LOW": 2, "CRITICAL": 3}
    pivot_all = rec_df.pivot(index="store", columns="item", values="action").replace(heat_vals)
    pivot_filt = rec_f.pivot(index="store", columns="item", values="action").replace(heat_vals)

    # Sadece seçilen satır/sütunlar
    rows = [s for s in sel_stores if s in pivot_filt.index]
    cols_show = [i for i in sel_items if i in pivot_filt.columns]
    if rows and cols_show:
        pivot_show = pivot_filt.loc[rows, cols_show]
        fig_heat = go.Figure(go.Heatmap(
            z=pivot_show.values,
            x=[str(c) for c in pivot_show.columns],
            y=[str(r) for r in pivot_show.index],
            colorscale=[
                [0.00, "#2ca02c"],
                [0.33, "#9467bd"],
                [0.66, "#ff7f0e"],
                [1.00, "#d62728"],
            ],
            zmin=0, zmax=3,
            colorbar=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["OK", "OVERSTOCK", "LOW", "CRITICAL"],
            ),
            hovertemplate="Mağaza %{y} — Ürün %{x}<extra></extra>",
        ))
        fig_heat.update_layout(
            xaxis_title="Ürün", yaxis_title="Mağaza",
            height=max(200, len(rows) * 40 + 80),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TAHMİN GRAFİKLERİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.header("Tahmin Grafikleri")

    fc1, fc2 = st.columns(2)
    sel_store_f = fc1.selectbox("Mağaza", sel_stores, key="fc_store")
    sel_item_f  = fc2.selectbox("Ürün",   sel_items,  key="fc_item")

    ts = pred_df[
        (pred_df["store"] == sel_store_f) &
        (pred_df["item"]  == sel_item_f)
    ].sort_values("date")

    if ts.empty:
        st.info("Seçilen kombinasyon için veri yok.")
    else:
        mae_val  = ts["error"].abs().mean()
        rmse_val = np.sqrt((ts["error"] ** 2).mean())
        mape_val = (ts["error"].abs() / ts["sales"].replace(0, np.nan)).mean() * 100

        st.caption("Bu üç sayı modelin ne kadar yanıldığını gösterir. MAE ortalama kaç birim hata yapıldığını, MAPE ise bu hatanın gerçek satışın yüzde kaçı olduğunu söyler. Düşük değerler daha iyi tahmin anlamına gelir.")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{mae_val:.2f}")
        m2.metric("RMSE", f"{rmse_val:.2f}")
        m3.metric("MAPE", f"{mape_val:.1f}%")

        # Gerçek vs Tahmin
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts["date"], y=ts["sales"],
            mode="lines", name="Gerçek Satış",
            line=dict(color="#636efa", width=1.5),
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts["date"], y=ts["predicted"],
            mode="lines", name="LightGBM Tahmini",
            line=dict(color="#ef553b", width=1.5, dash="dash"),
        ))
        # ±RMSE bandı
        fig_ts.add_trace(go.Scatter(
            x=pd.concat([ts["date"], ts["date"][::-1]]),
            y=pd.concat([ts["predicted"] + rmse_val,
                         (ts["predicted"] - rmse_val)[::-1]]),
            fill="toself",
            fillcolor="rgba(239,85,59,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±RMSE Bandı",
        ))
        fig_ts.update_layout(
            title=f"Mağaza {sel_store_f} — Ürün {sel_item_f} | 2017 Tahmin",
            xaxis_title="Tarih", yaxis_title="Satış",
            legend=dict(orientation="h", y=1.05),
            height=420,
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption("Mavi çizgi gerçekte kaç ürün satıldığını, kırmızı kesikli çizgi ise yapay zekanın önceden ne kadar satılacağını tahmin ettiğini gösterir. İki çizgi birbirine ne kadar yakınsa model o kadar başarılıdır. Gölgeli alan tahminin hata payını temsil eder.")

        # Hata dağılımı + scatter
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.caption("Modelin ne kadar yanıldığının dağılımı. Çubukların sıfır etrafında toplanmış ve simetrik olması, modelin sistematik bir hata yapmadığını gösterir. Sağa veya sola yatık bir dağılım sürekli eksik ya da fazla tahmin yapıldığına işaret eder.")
            fig_hist = px.histogram(
                ts, x="error", nbins=50,
                title="Residual Dağılımı",
                labels={"error": "Gerçek − Tahmin"},
                color_discrete_sequence=["#636efa"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.add_vline(
                x=ts["error"].mean(),
                line_dash="dot", line_color="orange",
                annotation_text=f"Ort={ts['error'].mean():.2f}",
            )
            fig_hist.update_layout(height=320)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_res2:
            st.caption("Her nokta bir günü temsil eder: yatay eksende modelin tahmini, dikey eksende gerçek satış. Noktalar kırmızı çizgiye ne kadar yakınsa tahminler o kadar isabetli demektir.")
            fig_sc = px.scatter(
                ts, x="predicted", y="sales",
                title="Gerçek vs Tahmin",
                labels={"predicted": "Tahmin", "sales": "Gerçek"},
                opacity=0.5,
                color_discrete_sequence=["#636efa"],
            )
            mn = min(ts["predicted"].min(), ts["sales"].min())
            mx = max(ts["predicted"].max(), ts["sales"].max())
            fig_sc.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines", name="İdeal",
                line=dict(color="red", dash="dash"),
            ))
            fig_sc.update_layout(height=320)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Aylık hata analizi
        st.subheader("Aylık Performans")
        st.caption("Modelin yıl içinde hangi aylarda daha çok yanıldığını gösterir. Yüksek hata değerleri genellikle sezonluk zirve dönemlerine (yaz, yılbaşı gibi) denk gelir; bu aylar için stok tamponunu artırmak faydalıdır.")
        ts["month"] = ts["date"].dt.month
        monthly = ts.groupby("month").apply(
            lambda g: pd.Series({
                "MAE" : g["error"].abs().mean(),
                "RMSE": np.sqrt((g["error"] ** 2).mean()),
                "MAPE": (g["error"].abs() / g["sales"].replace(0, np.nan)).mean() * 100,
                "Ort Satış": g["sales"].mean(),
                "Ort Tahmin": g["predicted"].mean(),
            })
        ).reset_index()
        monthly["Ay"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")

        fig_monthly = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Aylık MAE & RMSE", "Aylık Ort. Satış vs Tahmin"],
        )
        fig_monthly.add_trace(
            go.Bar(x=monthly["Ay"], y=monthly["MAE"],
                   name="MAE", marker_color="#636efa"),
            row=1, col=1,
        )
        fig_monthly.add_trace(
            go.Bar(x=monthly["Ay"], y=monthly["RMSE"],
                   name="RMSE", marker_color="#ef553b"),
            row=1, col=1,
        )
        fig_monthly.add_trace(
            go.Scatter(x=monthly["Ay"], y=monthly["Ort Satış"],
                       mode="lines+markers", name="Gerçek",
                       line=dict(color="#636efa")),
            row=1, col=2,
        )
        fig_monthly.add_trace(
            go.Scatter(x=monthly["Ay"], y=monthly["Ort Tahmin"],
                       mode="lines+markers", name="Tahmin",
                       line=dict(color="#ef553b", dash="dash")),
            row=1, col=2,
        )
        fig_monthly.update_layout(height=350, barmode="group")
        st.plotly_chart(fig_monthly, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. STOK & AKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_inventory:
    st.header("Stok Seviyeleri & Aksiyon Önerileri")

    if rec_f.empty:
        st.info("Seçilen filtreler için kayıt yok.")
    else:
        # Özet metrikler
        st.caption("Seçili ürünler için ortalama günlük satış miktarı, güvenlik tamponu, sipariş tetik noktası ve tek seferde önerilen sipariş büyüklüğü.")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Ort. Günlük Talep",   f"{rec_f['mean_demand_daily'].mean():.1f} br/gün")
        i2.metric("Ort. Safety Stock",   f"{rec_f['safety_stock'].mean():.0f} br")
        i3.metric("Ort. Dinamik ROP",    f"{rec_f['rop_dynamic'].mean():.0f} br")
        i4.metric("Ort. EOQ",            f"{rec_f['eoq_wilson'].mean():.0f} br")

        st.divider()

        # Stok seviyesi waterfall — mevcut stok vs eşikler (seçili 1 kombinasyon)
        st.subheader("Stok Seviyesi Göstergesi")
        st.caption("Gösterge ibresi mevcut stok miktarını gösterir. Kırmızı bölge tehlike, sarı bölge sipariş zamanı, yeşil bölge normal, mor bölge fazla stok anlamına gelir. Kırmızı çizgi sipariş verilmesi gereken eşiği işaretler.")
        inv1, inv2 = st.columns(2)
        sel_s = inv1.selectbox("Mağaza", sel_stores, key="inv_store")
        sel_i = inv2.selectbox("Ürün",   sel_items,  key="inv_item")

        row = rec_df[
            (rec_df["store"] == sel_s) & (rec_df["item"] == sel_i)
        ]
        if not row.empty:
            row = row.iloc[0]
            cur   = row["current_stock"]
            ss    = row["safety_stock"]
            rop   = row["rop_dynamic"]
            eoq   = row["eoq_wilson"]
            act   = row["action"]
            color = ACTION_COLORS[act]

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=cur,
                delta={"reference": rop, "valueformat": ".0f",
                       "suffix": " (vs ROP)"},
                title={"text": f"Mevcut Stok — Mağaza {sel_s} / Ürün {sel_i}"},
                gauge={
                    "axis": {"range": [0, max(cur * 1.5, rop * 2, 1)]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0, ss],          "color": "#ffcccc"},
                        {"range": [ss, rop],         "color": "#ffe5b4"},
                        {"range": [rop, rop + eoq],  "color": "#d4edda"},
                        {"range": [rop + eoq, max(cur * 1.5, rop * 2, 1)],
                         "color": "#e8d5ff"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.85,
                        "value": rop,
                    },
                },
            ))
            fig_gauge.update_layout(height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Aksiyon kartı
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Aksiyon", f"{ACTION_ICONS[act]} {act}")
            ac2.metric("Safety Stock", f"{ss:.0f}")
            ac3.metric("Dinamik ROP",  f"{rop:.0f}")
            ac4.metric("Önerilen Sipariş", f"{row['order_qty']:.0f} br")

        st.divider()

        # Aksiyon tablosu
        st.subheader("Aksiyon Önerileri Tablosu")
        st.caption("Her satır bir mağaza–ürün çiftini temsil eder. 'Sipariş Miktarı' sütunu o ürün için önerilen sipariş adedini, 'Sipariş Günü' sütunu ise sipariş verilmesi için kaç gün kaldığını gösterir. Negatif değer stokun zaten tükendiği anlamına gelir.")

        sort_col = st.selectbox(
            "Sıralama",
            ["priority_score", "days_to_reorder", "mean_demand_daily", "order_qty"],
            index=0,
        )

        display_df = (
            rec_f
            .sort_values(sort_col, ascending=False if sort_col == "priority_score" else True)
            [[
                "store", "item", "action", "current_stock",
                "safety_stock", "rop_dynamic", "eoq_wilson",
                "order_qty", "days_to_reorder", "mean_demand_daily",
            ]]
        )

        def color_action(val):
            colors_map = {
                "CRITICAL":  "background-color:#ffcccc; color:#7b0000; font-weight:bold",
                "LOW":       "background-color:#ffe5b4; color:#7a3e00; font-weight:bold",
                "OK":        "background-color:#d4edda; color:#155724",
                "OVERSTOCK": "background-color:#e8d5ff; color:#4a0080",
            }
            return colors_map.get(val, "")

        styled = (
            display_df.round(1)
            .rename(columns={
                "store": "Mağaza", "item": "Ürün", "action": "Aksiyon",
                "current_stock": "Mevcut Stok", "safety_stock": "Safety Stock",
                "rop_dynamic": "ROP (Dyn)", "eoq_wilson": "EOQ",
                "order_qty": "Sipariş Miktarı",
                "days_to_reorder": "Sipariş Günü",
                "mean_demand_daily": "Ort. Talep/gün",
            })
            .style.applymap(color_action, subset=["Aksiyon"])
        )
        st.dataframe(styled, use_container_width=True, height=420)

        # Stok seviyeleri — scatter: current stock vs ROP
        st.subheader("Mevcut Stok vs ROP Karşılaştırması")
        st.caption("Her nokta bir mağaza–ürün çiftini temsil eder. Kesik çizginin altında kalan noktalar sipariş verilmesi gereken ürünleri gösterir; ne kadar aşağıda olursa o kadar acil demektir. Renk, durumun ne kadar kritik olduğunu belirtir.")
        fig_scatter = px.scatter(
            rec_f,
            x="rop_dynamic", y="current_stock",
            color="action", color_discrete_map=ACTION_COLORS,
            hover_data=["store", "item", "order_qty", "days_to_reorder"],
            labels={"rop_dynamic": "Dinamik ROP", "current_stock": "Mevcut Stok"},
            title="Mevcut Stok vs Dinamik ROP — Her nokta bir store×item",
        )
        # 45° çizgi
        lim = max(rec_f["rop_dynamic"].max(), rec_f["current_stock"].max()) * 1.05
        fig_scatter.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim],
            mode="lines", name="Stok = ROP",
            line=dict(color="gray", dash="dash", width=1.5),
        ))
        fig_scatter.update_layout(height=430)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RİSK ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.header("Risk Analizi")

    # Stok tükenme riski
    from scipy.stats import norm

    rec_risk = rec_f.copy()
    rec_risk["stockout_prob"] = rec_risk.apply(
        lambda r: (1.0 - norm.cdf(r["current_stock"] / r["std_demand_daily"]))
                  if r["std_demand_daily"] > 0 else 0.0,
        axis=1,
    )
    rec_risk["excess_stock"]  = (rec_risk["current_stock"] - rec_risk["rop_dynamic"]).clip(lower=0)
    rec_risk["holding_cost_annual"] = rec_risk["safety_stock"] * 0.20 * 10  # H * unit_cost

    st.caption("Stok tükenmesi ne kadar olası? Kaç ürün yüksek risk altında? Fazla stok depolamak ne kadara mal oluyor? Bu dört kart cevapları özetler.")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Ort. Stok Tükenme Riski",
              f"{rec_risk['stockout_prob'].mean():.1%}")
    r2.metric("Yüksek Riskli Kombinasyon (>50%)",
              f"{(rec_risk['stockout_prob'] > 0.5).sum()}")
    r3.metric("Toplam Fazla Stok",
              f"{rec_risk['excess_stock'].sum():,.0f} br")
    r4.metric("Yıllık Holding Maliyeti",
              f"{rec_risk['holding_cost_annual'].sum():,.0f} TL")

    st.divider()
    risk_left, risk_right = st.columns(2)

    # Stok tükenme riski dağılımı
    with risk_left:
        st.caption("Her ürünün rafının boşalma ihtimalini gösterir. Kırmızı çizginin sağında kalan ürünlerin yarısından fazlasının tükeneceği tahmin ediliyor; bu ürünler en önce sipariş edilmelidir.")
        fig_risk = px.histogram(
            rec_risk, x="stockout_prob", nbins=30,
            color="action", color_discrete_map=ACTION_COLORS,
            title="Stok Tükenme Riski Dağılımı",
            labels={"stockout_prob": "Tükenme Olasılığı"},
            barmode="overlay",
        )
        fig_risk.add_vline(x=0.5, line_dash="dash", line_color="red",
                           annotation_text="50% eşiği")
        fig_risk.update_layout(height=370)
        st.plotly_chart(fig_risk, use_container_width=True)

    # Risk matrisi: talep belirsizliği vs stok açığı
    with risk_right:
        st.caption("Sağ üst köşedeki noktalar en riskli ürünleri işaretler: hem satış miktarı tahmin edilmesi zor hem de mevcut stoğu yetersiz. Nokta büyüklüğü günlük ortalama satışı, renk ise aksiyon aciliyetini gösterir.")
        rec_risk["stock_gap"] = rec_risk["rop_dynamic"] - rec_risk["current_stock"]
        fig_matrix = px.scatter(
            rec_risk,
            x="std_demand_daily", y="stock_gap",
            color="action", color_discrete_map=ACTION_COLORS,
            size="mean_demand_daily",
            hover_data=["store", "item", "stockout_prob"],
            title="Risk Matrisi: Talep Belirsizliği vs Stok Açığı",
            labels={"std_demand_daily": "Talep Std (günlük)",
                    "stock_gap": "Stok Açığı (ROP − Mevcut)"},
        )
        fig_matrix.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_matrix.update_layout(height=370)
        st.plotly_chart(fig_matrix, use_container_width=True)

    # Mağaza bazında toplam risk
    st.subheader("Mağaza Bazında Risk Özeti")
    st.caption("Hangi mağazanın en fazla sorun yaşadığını karşılaştırır. Kırmızıya yakın hücreler yüksek riski, maviye yakın hücreler ise yüksek sipariş maliyetini işaret eder. Bu tablo mağaza müdürlerine öncelik sırası belirlemede yardımcı olur.")
    store_risk = rec_risk.groupby("store").agg(
        ort_tükenme_riski   = ("stockout_prob", "mean"),
        kritik_ürün_sayısı  = ("action", lambda x: (x == "CRITICAL").sum()),
        toplam_sipariş      = ("order_qty", "sum"),
        yıllık_holding_tl   = ("holding_cost_annual", "sum"),
        ort_talep_günlük    = ("mean_demand_daily", "mean"),
    ).reset_index().rename(columns={"store": "Mağaza"})

    store_risk_styled = store_risk.round(2).style.background_gradient(
        subset=["ort_tükenme_riski", "kritik_ürün_sayısı"], cmap="RdYlGn_r"
    ).background_gradient(
        subset=["toplam_sipariş", "yıllık_holding_tl"], cmap="Blues"
    )
    st.dataframe(store_risk_styled, use_container_width=True)

    # Hizmet düzeyi duyarlılık grafiği (5 seviye)
    st.subheader("Hizmet Düzeyi — Safety Stock — Maliyet Ödünleşimi")
    st.caption("'Müşteriye her zaman ürün sunmak istiyorsak ne kadar fazla stok tutmamız gerekir ve bunun maliyeti ne olur?' sorusunun cevabı. Hizmet düzeyi yükseldikçe hem tampon stok hem de depolama maliyeti artar; bu grafik hangi noktada durmak gerektiğine karar vermeye yardımcı olur.")
    sls  = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    rows_sens = []
    for sl in sls:
        z_sl   = norm.ppf(sl)
        ss_all = rec_f["std_demand_daily"] * z_sl * np.sqrt(7 + 7)
        rows_sens.append({
            "Hizmet Düzeyi": f"{sl:.0%}",
            "Ort. SS": ss_all.mean(),
            "Toplam Holding (TL)": (ss_all * 0.20 * 10).sum(),
        })
    sens_df = pd.DataFrame(rows_sens)

    fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sens.add_trace(
        go.Bar(x=sens_df["Hizmet Düzeyi"], y=sens_df["Ort. SS"],
               name="Ort. Safety Stock (br)", marker_color="#636efa"),
        secondary_y=False,
    )
    fig_sens.add_trace(
        go.Scatter(x=sens_df["Hizmet Düzeyi"], y=sens_df["Toplam Holding (TL)"],
                   mode="lines+markers", name="Toplam Holding Maliyet (TL)",
                   line=dict(color="#ef553b", width=2)),
        secondary_y=True,
    )
    fig_sens.update_yaxes(title_text="Safety Stock (br)", secondary_y=False)
    fig_sens.update_yaxes(title_text="Yıllık Holding Maliyeti (TL)", secondary_y=True)
    fig_sens.update_layout(
        title="Hizmet Düzeyi Duyarlılık Analizi",
        height=380, legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL BİLGİSİ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.header("Model Bilgisi")

    # Test metrikleri
    st.subheader("Test Seti Performansı (2017)")
    st.caption("Yapay zekanın hiç görmediği 2017 verisi üzerindeki başarısı. MAE ortalama kaç birim yanıldığını, MAPE gerçek satışın yüzde kaçı kadar hata yapıldığını gösterir. Bu sayılar stok kararlarının ne kadar güvenilir olduğunu belirler.")
    m = model_meta["test_metrics"]
    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("MAE",   f"{m['MAE']:.3f}")
    tm2.metric("RMSE",  f"{m['RMSE']:.3f}")
    tm3.metric("MAPE",  f"{m['MAPE']:.2f}%")
    tm4.metric("SMAPE", f"{m['SMAPE']:.2f}%")

    st.divider()

    # Feature Importance
    st.subheader("Tahminlere En Çok Etki Eden Faktörler (Top 20)")
    st.caption("Yapay zekanın satış tahmini yaparken hangi bilgilere en çok baktığını gösterir. Barı uzun olan faktör tahmini en çok etkileyen faktördür. Örneğin 'geçen haftaki satış' barı uzunsa model geçmiş satış verisine çok güveniyor demektir.")
    top_fi = fi_df.nlargest(20, "importance")
    fig_fi = px.bar(
        top_fi.sort_values("importance"),
        x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        title=f"LightGBM Feature Importance (Top 20)",
        labels={"importance": "Importance", "feature": "Özellik"},
    )
    fig_fi.update_layout(
        height=520, coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Tüm özelliklerin dağılımı
    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        st.subheader("Tahmin Faktörü Grupları")
        st.caption("Faktörleri kategorilere ayırarak hangi bilgi türünün tahmini en çok yönlendirdiğini gösterir. Örneğin 'Geçmiş Satış (Lag)' dilimi büyükse model ağırlıklı olarak geçmiş verilere dayanıyor; 'Tarih' dilimi büyükse mevsim ve gün bilgisi belirleyici demektir.")
        def feature_group(name):
            if name.startswith("lag_"):        return "Lag"
            if name.startswith("rolling_"):    return "Rolling"
            if name.startswith("store_item"):  return "Etkileşim"
            if name in ("store_daily_total", "item_daily_total"): return "Etkileşim"
            if name in ("month_sin","month_cos","day_of_week_sin","day_of_week_cos",
                        "week_of_year_sin","week_of_year_cos"): return "Döngüsel"
            if name in ("year","month","day","day_of_week","week_of_year",
                        "day_of_year","quarter","is_weekend","is_month_start",
                        "is_month_end","season"):  return "Tarih"
            if name in ("days_since_start","year_progress"): return "Trend"
            return "Diğer"

        fi_df["group"] = fi_df["feature"].apply(feature_group)
        group_fi = fi_df.groupby("group")["importance"].sum().reset_index()

        fig_group = px.pie(
            group_fi, names="group", values="importance",
            title="Özellik Grubu Katkısı",
            hole=0.4,
        )
        fig_group.update_traces(textposition="outside", textinfo="percent+label")
        fig_group.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_group, use_container_width=True)

    with col_fi2:
        st.subheader("Model Parametreleri")
        params = model_meta.get("params", {})
        key_params = {k: v for k, v in params.items()
                      if k in ("n_estimators","learning_rate","num_leaves",
                               "max_depth","subsample","colsample_bytree",
                               "reg_alpha","reg_lambda","min_child_samples")}
        st.json(key_params)

        st.subheader("Eğitim Bilgisi")
        st.info(
            f"**Model:** {model_meta['model_name']}  \n"
            f"**Eğitim dönemi:** {model_meta['train_period']['start']} → "
            f"{model_meta['train_period']['end']}  \n"
            f"**Test dönemi:** {model_meta['test_period']['start']} → "
            f"{model_meta['test_period']['end']}  \n"
            f"**Özellik sayısı:** {len(model_meta['feature_cols'])}  \n"
            f"**Hedef değişken:** {model_meta['target_col']}"
        )

    # Tüm özellik önemi tablosu
    with st.expander("Tüm Faktörler — Detaylı Tablo"):
        st.caption("Modelin tahmin yaparken kullandığı tüm faktörlerin tam listesi ve her birinin ne kadar etkili olduğu. Koyu mavi hücre daha yüksek etkiyi temsil eder.")
        st.dataframe(
            fi_df[["feature", "group", "importance"]]
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
            .style.background_gradient(subset=["importance"], cmap="Blues"),
            use_container_width=True,
            height=400,
        )
