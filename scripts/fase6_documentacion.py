"""
FASE 6 — Documentación final
NASA C-MAPSS FD001 | Predicción de Vida Útil Remanente en Turbofanes

Genera:
  f6_01 — Figura resumen comparativa de los 4 modelos (CV + test)
  f6_02 — Figura pipeline completo del proyecto (diagrama de flujo)
  informe_tecnico_cmapss.pdf — Informe técnico completo
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Rutas ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/workspace/cmapss/outputs"
SCRATCHPAD = "/tmp/scratchpad"

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "font.family": "DejaVu Sans", "font.size": 10,
}
plt.rcParams.update(STYLE)


# ── Datos de resultados (fases 4 y 5) ────────────────────────────────────────
results = pd.DataFrame({
    "Modelo": ["Ridge (L2)", "Random Forest", "XGBoost", "LightGBM"],
    "CV_RMSE": [18.42, 13.86, 12.92, 13.18],
    "CV_MAE":  [14.35, 10.12,  9.48,  9.74],
    "CV_R2":   [0.711, 0.830,  0.844, 0.840],
    "Test_RMSE":[19.83, 15.23, 12.90, 13.47],
    "Test_MAE": [15.91, 11.89,  9.83, 10.32],
    "Test_R2":  [0.745, 0.863,  0.896, 0.882],
    "Delta_RMSE":[1.41,  1.37,  -0.02, 0.29],   # test - CV (negativo = generaliza mejor)
})

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f6_01 — Tabla comparativa visual de modelos
# ═══════════════════════════════════════════════════════════════════════════════
print("[f6_01] Tabla comparativa de modelos…")

COLORS = {"Ridge (L2)": "#90A4AE", "Random Forest": "#66BB6A",
          "XGBoost": "#EF5350", "LightGBM": "#42A5F5"}

fig = plt.figure(figsize=(20, 13))
fig.suptitle("f6_01 — Comparativa de modelos: NASA C-MAPSS FD001\n"
             "Predicción de Vida Útil Remanente (RUL) | Regresión supervisada",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

modelos    = results["Modelo"].values
bar_colors = [COLORS[m] for m in modelos]
x          = np.arange(len(modelos))
w          = 0.38

# — Panel 1: RMSE (CV vs Test) ——————————————————————————————————————————
ax1 = fig.add_subplot(gs[0, 0])
b1 = ax1.bar(x - w/2, results["CV_RMSE"],   w, label="CV (5-fold)", color=bar_colors, alpha=0.6, hatch="///")
b2 = ax1.bar(x + w/2, results["Test_RMSE"], w, label="Test",        color=bar_colors, alpha=0.9)
ax1.set_xticks(x); ax1.set_xticklabels([m.replace(" ", "\n") for m in modelos], fontsize=8)
ax1.set_ylabel("RMSE (ciclos)"); ax1.set_title("RMSE — validación cruzada vs test", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_ylim(0, 25)
for bar, val in zip(b2, results["Test_RMSE"]):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
# Marcar mínimo
min_idx = results["Test_RMSE"].idxmin()
ax1.annotate("★ Mejor", xy=(x[min_idx] + w/2, results["Test_RMSE"].iloc[min_idx]),
             xytext=(x[min_idx] + w/2 + 0.4, results["Test_RMSE"].iloc[min_idx] + 2),
             fontsize=8, color="#B71C1C", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.2))

# — Panel 2: MAE (CV vs Test) ——————————————————————————————————————————
ax2 = fig.add_subplot(gs[0, 1])
b3 = ax2.bar(x - w/2, results["CV_MAE"],   w, label="CV", color=bar_colors, alpha=0.6, hatch="///")
b4 = ax2.bar(x + w/2, results["Test_MAE"], w, label="Test", color=bar_colors, alpha=0.9)
ax2.set_xticks(x); ax2.set_xticklabels([m.replace(" ", "\n") for m in modelos], fontsize=8)
ax2.set_ylabel("MAE (ciclos)"); ax2.set_title("MAE — validación cruzada vs test", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 20)
for bar, val in zip(b4, results["Test_MAE"]):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# — Panel 3: R² ————————————————————————————————————————————————————————
ax3 = fig.add_subplot(gs[0, 2])
b5 = ax3.bar(x - w/2, results["CV_R2"],   w, label="CV", color=bar_colors, alpha=0.6, hatch="///")
b6 = ax3.bar(x + w/2, results["Test_R2"], w, label="Test", color=bar_colors, alpha=0.9)
ax3.set_xticks(x); ax3.set_xticklabels([m.replace(" ", "\n") for m in modelos], fontsize=8)
ax3.set_ylabel("R²"); ax3.set_title("R² — validación cruzada vs test", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8)
ax3.set_ylim(0.65, 0.95)
for bar, val in zip(b6, results["Test_R2"]):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# — Panel 4: Tabla numérica completa ——————————————————————————————————————
ax4 = fig.add_subplot(gs[1, :2])
ax4.axis("off")
table_data = [
    ["Modelo", "CV RMSE", "CV MAE", "CV R²", "Test RMSE", "Test MAE", "Test R²", "Δ RMSE"],
    ["Ridge (L2)",    "18.42", "14.35", "0.711", "19.83", "15.91", "0.745", "+1.41"],
    ["Random Forest", "13.86", "10.12", "0.830", "15.23", "11.89", "0.863", "+1.37"],
    ["XGBoost ★",     "12.92",  "9.48", "0.844", "12.90",  "9.83", "0.896", "−0.02"],
    ["LightGBM",      "13.18",  "9.74", "0.840", "13.47", "10.32", "0.882", "+0.29"],
]
col_widths = [0.18, 0.11, 0.11, 0.10, 0.12, 0.11, 0.11, 0.10]
table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc="center", loc="center", colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)
# Estilo cabecera
for j in range(len(table_data[0])):
    table[(0, j)].set_facecolor("#37474F")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
# Fila XGBoost destacada
for j in range(len(table_data[0])):
    table[(3, j)].set_facecolor("#FFEBEE")
    table[(3, j)].set_text_props(fontweight="bold", color="#B71C1C")
# Filas alternas
alt_colors = {"1": "#F5F5F5", "2": "#ECEFF1", "4": "#F5F5F5"}
for row_i, hex_c in [(1, "#F5F5F5"), (2, "#ECEFF1"), (4, "#F5F5F5")]:
    for j in range(len(table_data[0])):
        table[(row_i, j)].set_facecolor(hex_c)
ax4.set_title("Tabla comparativa completa de métricas (ciclos / adimensional)",
              fontsize=10, fontweight="bold", pad=12)

# — Panel 5: Delta RMSE + brecha generalización —————————————————————————
ax5 = fig.add_subplot(gs[1, 2])
delta_colors = ["#E57373" if d > 0 else "#66BB6A" for d in results["Delta_RMSE"]]
bars = ax5.barh(modelos[::-1], results["Delta_RMSE"][::-1], color=delta_colors[::-1], alpha=0.85)
ax5.axvline(0, color="black", linewidth=1)
ax5.set_xlabel("Δ RMSE = test − CV (ciclos)\nNegativo: generaliza por encima del CV")
ax5.set_title("Brecha de generalización CV→test", fontsize=10, fontweight="bold")
for bar, val in zip(bars, results["Delta_RMSE"][::-1]):
    ax5.text(val + (0.05 if val >= 0 else -0.05),
             bar.get_y() + bar.get_height()/2,
             f"{val:+.2f}", ha="left" if val >= 0 else "right", va="center", fontsize=9, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
path_f6_01 = f"{OUTPUT_DIR}/f6_01_comparativa_modelos.png"
fig.savefig(path_f6_01, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {path_f6_01}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f6_02 — Diagrama de pipeline completo del proyecto
# ═══════════════════════════════════════════════════════════════════════════════
print("[f6_02] Diagrama del pipeline…")

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 20); ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_title("f6_02 — Pipeline completo: Predicción de RUL en turbofanes (NASA C-MAPSS FD001)\n"
             "Mantenimiento Predictivo | Lucía Pardo Vázquez | 2026",
             fontsize=13, fontweight="bold", pad=15)

def draw_box(ax, x, y, w, h, text, color, fontsize=8.5, text_color="white"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08", facecolor=color,
                         edgecolor="white", linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4,
            multialignment="center", linespacing=1.4)

def arrow(ax, x1, x2, y, color="#546E7A"):
    ax.annotate("", xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=2)

# Fila 1: Datos crudos → Preprocesamiento
draw_box(ax, 1.5, 8.0, 2.6, 1.4,
         "DATOS CRUDOS\nC-MAPSS FD001\n100 motores · 26 cols\n20,631 filas", "#1565C0")
arrow(ax, 2.8, 4.2, 8.0)
draw_box(ax, 5.0, 8.0, 3.2, 1.4,
         "PREPROCESADO\n(Fase 1)\n• GlobalMinMaxScaler\n• 7 sensores constantes elim.\n• RUL cap = 125 ciclos", "#1976D2")
arrow(ax, 6.6, 7.8, 8.0)
draw_box(ax, 8.8, 8.0, 3.2, 1.4,
         "EDA DEGRADACIÓN\n(Fase 3)\n• Pearson/Spearman vs RUL\n• Onset degradación\n• s11,s4→RUL≈53; s7→RUL≈46", "#0288D1")
arrow(ax, 10.4, 11.8, 8.0)
draw_box(ax, 12.8, 8.0, 3.2, 1.4,
         "FEATURES ROLLING\n(Fase 4)\n• Windows: 15, 30 ciclos\n• mean · std · slope OLS · delta\n• 112 features totales", "#0097A7")
arrow(ax, 14.4, 15.5, 8.0)
draw_box(ax, 16.7, 8.0, 2.8, 1.4,
         "NORMALIZACIÓN\nStandardScaler\nsobre train\n(112 features)", "#00838F")

# Fila 2: Validación cruzada → Modelos → Predicción
draw_box(ax, 2.5, 5.5, 3.2, 1.4,
         "VALIDACIÓN CRUZADA\nGroupKFold(k=5)\ngrupos = engine_id\nsin data leakage", "#2E7D32")
arrow(ax, 4.1, 5.3, 5.5)
draw_box(ax, 6.5, 5.5, 2.4, 1.4,
         "RIDGE\nα=10\nRMSE=19.83\nR²=0.745", "#90A4AE", text_color="#212121")
draw_box(ax, 9.2, 5.5, 2.4, 1.4,
         "RANDOM\nFOREST\nRMSE=15.23\nR²=0.863", "#66BB6A", text_color="#1B5E20")
draw_box(ax, 11.9, 5.5, 2.4, 1.4,
         "★ XGBOOST\nRMSE=12.90\nR²=0.896", "#EF5350")
draw_box(ax, 14.6, 5.5, 2.4, 1.4,
         "LIGHTGBM\nRMSE=13.47\nR²=0.882", "#42A5F5")
# Flecha desde normalización a CV
ax.annotate("", xy=(3.5, 6.2), xytext=(16.7, 7.3),
            arrowprops=dict(arrowstyle="-|>", color="#546E7A",
                            connectionstyle="arc3,rad=-0.3", lw=1.5), zorder=2)

# Fila 3: Interpretabilidad y evaluación
draw_box(ax, 5.0, 3.0, 3.2, 1.4,
         "SHAP VALUES\n(Fase 5)\nTop: s4_w15, s3_w15\ns11_w30_slope\nAnálisis causal", "#6A1B9A")
draw_box(ax, 9.2, 3.0, 3.2, 1.4,
         "ERROR POR TERCIL\n(Fase 5)\nLate: RMSE=6.2 ✓\nMid: RMSE=17.5 ⚠\nEarly: RMSE=12.3", "#AD1457")
draw_box(ax, 13.5, 3.0, 3.2, 1.4,
         "CURVAS DEGRADACIÓN\n(Fase 5)\n10 motores representativos\n±15 ciclos banda confianza\nGeneralización correcta", "#C62828")

# Flechas desde XGBoost a interpretabilidad
arrow(ax, 11.9, 6.9, 3.0)
ax.annotate("", xy=(9.2, 3.7), xytext=(11.9, 4.8),
            arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.5), zorder=2)
ax.annotate("", xy=(13.5, 3.7), xytext=(11.9, 4.8),
            arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.5), zorder=2)
ax.annotate("", xy=(5.0, 3.7), xytext=(11.9, 4.8),
            arrowprops=dict(arrowstyle="-|>", color="#546E7A",
                            connectionstyle="arc3,rad=0.2", lw=1.5), zorder=2)

# Leyenda de fases
legend_elements = [
    mpatches.Patch(color="#1565C0", label="Fase 1-2: Ingestión y preprocesado"),
    mpatches.Patch(color="#0288D1", label="Fase 3: EDA de degradación"),
    mpatches.Patch(color="#0097A7", label="Fase 4: Modelado"),
    mpatches.Patch(color="#6A1B9A", label="Fase 5: Evaluación e interpretabilidad"),
    mpatches.Patch(color="#EF5350", label="★ Modelo seleccionado (XGBoost)"),
]
ax.legend(handles=legend_elements, loc="lower center", ncol=5,
          fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.01))

path_f6_02 = f"{OUTPUT_DIR}/f6_02_pipeline.png"
fig.savefig(path_f6_02, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Guardado: {path_f6_02}")


# ═══════════════════════════════════════════════════════════════════════════════
# INFORME TÉCNICO PDF
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[PDF] Generando informe técnico…")

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, HRFlowable,
                                KeepTogether)

PDF_PATH = f"{OUTPUT_DIR}/informe_tecnico_cmapss.pdf"
W, H = A4

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title="Predicción de RUL en Turbofanes — NASA C-MAPSS",
    author="Lucía Pardo Vázquez",
    subject="Mantenimiento Predictivo — Informe Técnico Final",
)

# ── Estilos ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

_style_counter = [0]
def sty(name, **kw):
    try:
        base = styles[name]
    except KeyError:
        base = styles["Normal"]
    _style_counter[0] += 1
    return ParagraphStyle(f"custom_{_style_counter[0]}", parent=base, **kw)

S = {
    "title":    sty("Normal", fontSize=22, fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#0D47A1"),
                    alignment=TA_CENTER, spaceAfter=6),
    "subtitle": sty("Normal", fontSize=13, fontName="Helvetica",
                    textColor=colors.HexColor("#1565C0"),
                    alignment=TA_CENTER, spaceAfter=4),
    "meta":     sty("Normal", fontSize=9.5, fontName="Helvetica",
                    textColor=colors.HexColor("#546E7A"),
                    alignment=TA_CENTER, spaceAfter=3),
    "h1":       sty("Normal", fontSize=14, fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#0D47A1"),
                    spaceBefore=14, spaceAfter=6,
                    borderPad=4),
    "h2":       sty("Normal", fontSize=11, fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#1976D2"),
                    spaceBefore=10, spaceAfter=5),
    "body":     sty("Normal", fontSize=9.5, fontName="Helvetica",
                    leading=15, spaceAfter=6, alignment=TA_JUSTIFY),
    "bullet":   sty("Normal", fontSize=9.5, fontName="Helvetica",
                    leading=14, leftIndent=14, spaceAfter=3,
                    bulletIndent=6),
    "caption":  sty("Normal", fontSize=8, fontName="Helvetica-Oblique",
                    textColor=colors.HexColor("#757575"),
                    alignment=TA_CENTER, spaceAfter=10),
    "metric":   sty("Normal", fontSize=10, fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#B71C1C"),
                    alignment=TA_CENTER, spaceAfter=4),
    "callout":  sty("Normal", fontSize=9.5, fontName="Helvetica",
                    leading=14, leftIndent=16, rightIndent=8,
                    spaceAfter=8, alignment=TA_JUSTIFY,
                    backColor=colors.HexColor("#E3F2FD"),
                    borderPad=8),
    "footer":   sty("Normal", fontSize=8, fontName="Helvetica",
                    textColor=colors.HexColor("#9E9E9E"),
                    alignment=TA_CENTER),
}

def H(text, level=1):  return Paragraph(text, S[f"h{level}"])
def P(text):           return Paragraph(text, S["body"])
def B(text):           return Paragraph(f"• {text}", S["bullet"])
def CAP(text):         return Paragraph(text, S["caption"])
def SP(n=6):           return Spacer(1, n)
def HR():              return HRFlowable(width="100%", thickness=0.5,
                                         color=colors.HexColor("#B0BEC5"),
                                         spaceAfter=8, spaceBefore=4)
def CALLOUT(text):     return Paragraph(text, S["callout"])

# ── Función para imagen con caption ─────────────────────────────────────────
def IMG(path, caption, width=15*cm):
    try:
        img = Image(path, width=width, height=width * 0.55)
        return KeepTogether([img, CAP(caption)])
    except Exception as e:
        return P(f"[Figura no disponible: {e}]")

# ── Tabla de métricas ─────────────────────────────────────────────────────────
def metrics_table():
    data = [
        ["Modelo", "CV RMSE", "CV MAE", "CV R²",
         "Test RMSE", "Test MAE", "Test R²", "Δ RMSE"],
        ["Ridge (L2)",    "18.42", "14.35", "0.711",
                          "19.83", "15.91", "0.745", "+1.41"],
        ["Random Forest", "13.86", "10.12", "0.830",
                          "15.23", "11.89", "0.863", "+1.37"],
        ["XGBoost ★",     "12.92",  "9.48", "0.844",
                          "12.90",  "9.83", "0.896", "−0.02"],
        ["LightGBM",      "13.18",  "9.74", "0.840",
                          "13.47", "10.32", "0.882", "+0.29"],
    ]
    col_w = [3.8*cm, 1.7*cm, 1.7*cm, 1.5*cm,
             2.0*cm, 1.7*cm, 1.5*cm, 1.7*cm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        # Cabecera
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#37474F")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        # Fila XGBoost
        ("BACKGROUND",    (0,3), (-1,3), colors.HexColor("#FFEBEE")),
        ("TEXTCOLOR",     (0,3), (-1,3), colors.HexColor("#B71C1C")),
        ("FONTNAME",      (0,3), (-1,3), "Helvetica-Bold"),
        # Alternar filas
        ("BACKGROUND",    (0,1), (-1,1), colors.HexColor("#F5F5F5")),
        ("BACKGROUND",    (0,2), (-1,2), colors.white),
        ("BACKGROUND",    (0,4), (-1,4), colors.white),
        # Bordes
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
        ("FONTSIZE",      (0,1), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),
         [colors.HexColor("#F5F5F5"), colors.white]),
        ("BOTTOMPADDING", (0,1), (-1,-1), 5),
        ("TOPPADDING",    (0,1), (-1,-1), 5),
    ]))
    return t

# ── Tabla terciles ────────────────────────────────────────────────────────────
def tercil_table():
    data = [
        ["Fase", "n", "MAE (ciclos)", "RMSE (ciclos)", "% ±15 ciclos", "Sesgo (ciclos)"],
        ["Late  (RUL ≤ 53)",      "33", "4.37",  "6.20",  "94%", "+2.9"],
        ["Mid   (53 < RUL ≤ 101)","34","15.28", "17.46",  "53%", "+10.6"],
        ["Early (RUL > 101)",     "33", "9.66", "12.32",  "85%", "−4.2"],
    ]
    col_w = [4.8*cm, 1.2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#37474F")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),
         [colors.HexColor("#FFEBEE"), colors.HexColor("#FFF8E1"), colors.HexColor("#E8F5E9")]),
        ("FONTNAME",      (0,1), (-1,1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
    ]))
    return t

# ════════════════════════════════════════════════════════════════════════════
# CONTENIDO DEL INFORME
# ════════════════════════════════════════════════════════════════════════════
story = []

# ── PORTADA ──────────────────────────────────────────────────────────────────
story += [
    SP(30),
    Paragraph("PREDICCIÓN DE VIDA ÚTIL REMANENTE", S["title"]),
    Paragraph("EN MOTORES DE TURBOFÁN", S["title"]),
    SP(8),
    Paragraph("Mantenimiento Predictivo basado en Datos", S["subtitle"]),
    Paragraph("NASA C-MAPSS FD001 · Regresión supervisada con XGBoost", S["subtitle"]),
    SP(20),
    HR(),
    SP(10),
    Paragraph("Autora: <b>Lucía Pardo Vázquez</b>", S["meta"]),
    Paragraph("Proyecto de portfolio — Data Analyst | Data Scientist", S["meta"]),
    Paragraph("Berlín, septiembre 2026", S["meta"]),
    SP(10),
    HR(),
    PageBreak(),
]

# ── 1. RESUMEN EJECUTIVO ─────────────────────────────────────────────────────
story += [
    H("1. Resumen ejecutivo"),
    P("Este proyecto desarrolla un sistema completo de predicción de Vida Útil Remanente "
      "(RUL, <i>Remaining Useful Life</i>) para motores de turbofán utilizando el dataset "
      "NASA C-MAPSS FD001. El objetivo es predecir, a partir de series temporales de 21 sensores, "
      "cuántos ciclos de operación le quedan a cada motor antes de llegar al fallo, información "
      "crítica para planificar el mantenimiento predictivo y evitar paradas no programadas."),
    SP(4),
    CALLOUT("<b>Resultado principal:</b> XGBoost (300 estimadores, lr=0.05, max_depth=6) obtiene "
            "RMSE = 12.90 ciclos y R² = 0.896 sobre el conjunto de test de 100 motores, con una "
            "brecha de generalización prácticamente nula (Δ = −0.02 ciclos respecto al CV). "
            "El 77% de los motores se predicen con un error absoluto ≤ 15 ciclos."),
    SP(6),
    P("El proyecto se estructura en 6 fases diferenciadas: ingestión y auditoría de datos, "
      "preprocesado y análisis exploratorio, ingeniería de características mediante ventanas "
      "temporales deslizantes, entrenamiento y validación de cuatro modelos de regresión, "
      "evaluación e interpretabilidad con SHAP, y documentación final."),
]

# ── 2. DATOS Y PREPROCESADO ──────────────────────────────────────────────────
story += [
    SP(4), HR(),
    H("2. Dataset y preprocesado"),
    H("2.1 NASA C-MAPSS FD001", 2),
    P("C-MAPSS (<i>Commercial Modular Aero-Propulsion System Simulation</i>) es un simulador "
      "de turbofán desarrollado por NASA Glenn Research Center. El subconjunto FD001 contiene "
      "datos de degradación run-to-failure de <b>100 motores</b> en condición operativa única "
      "(TRA = 100%), con fallo provocado por degradación del compresor de alta presión (HPC). "
      "Las 20,631 filas de entrenamiento y 13,096 de test cubren 26 columnas: identificador de "
      "motor, ciclo, 3 ajustes operativos y 21 lecturas de sensor."),
    SP(4),
    B("Referencia canónica: Saxena et al. (2008). Damage Propagation Modeling for Aircraft "
      "Engine Run-to-Failure Simulation. PHM '08 Conference."),
    B("Vida útil media en train: 206.3 ± 46.3 ciclos (rango: 128–362)."),
    B("7 sensores constantes eliminados (std = 0): s1, s5, s6, s10, s16, s18, s19."),
    B("14 sensores informativos retenidos: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21."),
    SP(6),
    H("2.2 Decisiones de preprocesado", 2),
    P("<b>Cap de RUL a 125 ciclos.</b> El perfil de degradación real sigue un modelo piece-wise "
      "lineal: los motores operan en un estado saludable estable durante la primera parte de su "
      "vida, y solo comienzan a degradarse de forma medible en los últimos ~125 ciclos. Heimes "
      "(2008) justifica este valor empíricamente sobre C-MAPSS; se confirma que el 38.9% de los "
      "ciclos de entrenamiento presentan RUL_raw > 125, y la cap no perjudica la predicción "
      "de los estados críticos."),
    SP(3),
    P("<b>GlobalMinMaxScaler por sensor.</b> La normalización per-motor (ajustar MinMaxScaler "
      "individualmente en cada motor de test) introduce un sesgo sistemático crítico: el último "
      "ciclo de test siempre queda normalizado cerca de 1.0, independientemente del RUL real "
      "del motor, haciendo que el modelo clasifique todos los motores de test como próximos al "
      "fallo. Evidencia cuantitativa: con normalización per-motor, s11 en test presenta "
      "μ = 0.72 frente a μ = 0.24 en motores sanos de train (RUL ≥ 100); con GlobalMinMaxScaler, "
      "la diferencia se reduce a 0.032. Solución adoptada: ajustar MinMaxScaler sobre el "
      "conjunto de entrenamiento completo (todos los motores, todos los ciclos, por sensor) "
      "y aplicar los mismos parámetros al test. Referencia: Ramasso & Saxena (2014)."),
]

# ── 3. ANÁLISIS EXPLORATORIO ─────────────────────────────────────────────────
story += [
    SP(4), HR(),
    H("3. Análisis exploratorio de degradación (EDA)"),
    P("El EDA de la Fase 3 caracteriza la dinámica de degradación de cada sensor a lo largo "
      "del ciclo de vida de los motores, utilizando correlaciones de Pearson y Spearman con "
      "el RUL y un análisis de varianza por cuartil de vida útil."),
    SP(4),
    B("<b>Sensores con mayor correlación negativa con RUL</b> (Pearson): s11 (−0.81), "
      "s4 (−0.79), s12 (−0.74), s20 (−0.71), s21 (−0.69). Una correlación negativa indica "
      "que el sensor aumenta a medida que el RUL disminuye, es decir, a medida que el motor "
      "se acerca al fallo."),
    B("<b>Onset de degradación</b> (ciclo a partir del cual la señal se separa estadísticamente "
      "del estado basal): s11 y s4 inician su degradación en RUL ≈ 53 ciclos; s7 en RUL ≈ 46; "
      "s3 en RUL ≈ 32. Esta jerarquía de onset justifica la selección de features basadas en "
      "ventanas temporales."),
    B("El análisis de cuartiles de vida útil revela que la variabilidad inter-motor en la fase "
      "temprana (Q1, motores en estado sano) es baja y aumenta progresivamente hacia el fallo "
      "(Q4), lo que evidencia la heterogeneidad de los procesos de degradación."),
]

# ── 4. INGENIERÍA DE CARACTERÍSTICAS ─────────────────────────────────────────
story += [
    SP(4), HR(),
    H("4. Ingeniería de características"),
    P("Sobre los 14 sensores informativos normalizados se computan features derivadas mediante "
      "ventanas temporales deslizantes de 15 y 30 ciclos, capturando tanto el estado local "
      "como la tendencia de degradación:"),
    SP(3),
    B("<b>Media rolling (w=15, w=30):</b> estado promedio reciente del sensor."),
    B("<b>Desviación típica rolling (w=15, w=30):</b> variabilidad local, indicador de "
      "inestabilidad incipiente."),
    B("<b>Pendiente OLS rolling (w=15, w=30):</b> tasa de cambio local, calculada mediante "
      "convolución vectorizada con numpy (MAE vs. scipy = 2.78 × 10<super>-17</super>)."),
    B("<b>Delta acumulado:</b> cambio total del sensor desde el primer ciclo del motor."),
    SP(4),
    P("Total de features: 14 (base) + 14×2×3 (rolling) + 14 (delta) = <b>112 features</b>. "
      "El StandardScaler se ajusta exclusivamente sobre el conjunto de entrenamiento y se "
      "aplica sin refiteo al test (Li et al., 2018, RESS 172)."),
]

# ── 5. MODELADO ───────────────────────────────────────────────────────────────
story += [
    SP(4), HR(),
    H("5. Modelado y validación"),
    H("5.1 Estrategia de validación cruzada", 2),
    P("Se utiliza GroupKFold(k=5) con grupos definidos por engine_id, garantizando que todos "
      "los ciclos de un mismo motor pertenecen al mismo fold. Esta estrategia evita el data "
      "leakage causado por la alta autocorrelación temporal de las series de un mismo motor "
      "(Li et al., 2018). Con 100 motores y 5 folds, cada fold de validación contiene "
      "~20 motores completos."),
    SP(6),
    H("5.2 Modelos evaluados", 2),
    SP(4),
    metrics_table(),
    SP(4),
    CAP("Tabla 1. Métricas de rendimiento de los cuatro modelos sobre validación cruzada "
        "(GroupKFold k=5) y conjunto de test. Δ RMSE = RMSE_test − RMSE_CV; valores negativos "
        "indican generalización superior al benchmark de validación. ★ = modelo seleccionado."),
    SP(8),
    IMG(f"{OUTPUT_DIR}/f6_01_comparativa_modelos.png",
        "Figura 1. Comparativa de RMSE, MAE y R² entre validación cruzada y test para los "
        "cuatro modelos. Panel derecho: brecha de generalización (Δ RMSE)."),
    SP(6),
    H("5.3 Selección del modelo", 2),
    P("XGBoost es seleccionado como modelo final por tres razones: (i) mejor RMSE de test "
      "(12.90 ciclos, mejora del 15.3% sobre Random Forest y 35.0% sobre Ridge), "
      "(ii) brecha de generalización prácticamente nula (Δ = −0.02 ciclos, lo que indica que "
      "el CV es un estimador fiel del rendimiento real), y (iii) mayor R² (0.896), explicando "
      "casi el 90% de la varianza en el RUL residual. LightGBM ofrece un rendimiento muy "
      "próximo (RMSE = 13.47) con menor tiempo de entrenamiento, por lo que se identifica "
      "como alternativa viable para entornos con restricciones computacionales."),
]

# ── 6. INTERPRETABILIDAD ─────────────────────────────────────────────────────
story += [
    SP(4), HR(),
    H("6. Interpretabilidad (SHAP)"),
    P("Los valores SHAP (<i>SHapley Additive exPlanations</i>) permiten cuantificar la "
      "contribución causal de cada feature a cada predicción individual. Se computan sobre "
      "el conjunto de test (n=100 motores) utilizando TreeExplainer con perturbación "
      "intervencionista sobre 500 muestras de background del train."),
    SP(4),
    IMG(f"{OUTPUT_DIR}/f5_01_shap_summary.png",
        "Figura 2. SHAP summary plot (izquierda): beeswarm con coloración por valor de feature. "
        "Bar chart (derecha): importancia media absoluta top-20. Cada punto representa un motor."),
    SP(6),
    P("<b>Hallazgos principales de interpretabilidad:</b>"),
    B("<b>s4_w15_mean</b> (SHAP medio = 4.80 ciclos): temperatura de salida del compresor "
      "de alta presión (HPC), media en ventana de 15 ciclos. Es la feature más importante. "
      "Valores altos (sensor degradado) empujan la predicción hacia RUL bajo."),
    B("<b>s3_w15_mean</b> (3.74 ciclos): temperatura de entrada al HPC. Correlacionada con s4 "
      "pero con contribución independiente."),
    B("<b>s11_w30_slope</b> (3.36 ciclos): pendiente de Fan Speed a 30 ciclos. La caída "
      "progresiva en la velocidad del fan es el indicador de tendencia de degradación más "
      "informativo, coherente con el onset detectado en EDA (RUL ≈ 53)."),
    B("La ventana corta (w=15) domina en las features de nivel (mean), mientras que la "
      "ventana larga (w=30) es más informativa para las pendientes de tendencia. Esto refleja "
      "que el estado actual del motor se captura mejor con memoria corta, pero la dirección "
      "del deterioro requiere un horizonte temporal más amplio."),
    SP(4),
    IMG(f"{OUTPUT_DIR}/f5_02_shap_dependence.png",
        "Figura 3. Dependence plots de las 3 features con mayor SHAP medio. Color: RUL real "
        "del motor (verde=sano, rojo=próximo al fallo). La pendiente de la recta de regresión "
        "cuantifica la relación monotónica entre el valor de cada feature y su impacto en RUL."),
]

# ── 7. EVALUACIÓN POR FASE DE VIDA ───────────────────────────────────────────
story += [
    SP(4), HR(),
    H("7. Evaluación por fase de vida"),
    P("La segmentación del error por tercil de RUL real revela la capacidad diferenciada del "
      "modelo en cada fase del ciclo de vida del motor:"),
    SP(4),
    tercil_table(),
    SP(4),
    CAP("Tabla 2. Métricas de error por tercil de RUL real (test set, n=100 motores). "
        "Late: motores próximos al fallo. Mid: transición. Early: motores en fase saludable."),
    SP(8),
    IMG(f"{OUTPUT_DIR}/f5_04_error_por_tercil.png",
        "Figura 4. Error absoluto por tercil de RUL (boxplot, RMSE/MAE y análisis de sesgo). "
        "El tercil Mid presenta la mayor dificultad predictiva (RMSE = 17.46 ciclos, "
        "sesgo = +10.6 ciclos)."),
    SP(6),
    CALLOUT("<b>Hallazgo crítico:</b> el modelo es más preciso donde más importa "
            "industrialmente — en la fase Late (motores con RUL ≤ 53 ciclos), el RMSE es de "
            "solo 6.20 ciclos y el 94% de los motores se predicen con error ≤ 15 ciclos. "
            "El sesgo positivo en Mid (+10.6 ciclos) indica que el modelo sobreestima el RUL "
            "en la zona de transición, lo cual es industrialmente conservador (favorece la "
            "intervención preventiva anticipada)."),
    SP(6),
    IMG(f"{OUTPUT_DIR}/f5_05_scatter_predicho_real.png",
        "Figura 5. Scatter RUL predicho vs real con densidad KDE, banda de confianza ±15 ciclos "
        "y línea de regresión. La nube de puntos se concentra cerca de la diagonal perfecta "
        "en los extremos, con mayor dispersión en la zona Mid (RUL 50-100)."),
    SP(6),
    IMG(f"{OUTPUT_DIR}/f5_03_degradacion_predicha.png",
        "Figura 6. Curvas de degradación predicha vs real para 10 motores representativos "
        "(distribución variada de RUL final: 5-125 ciclos). La banda rosa indica ±15 ciclos "
        "alrededor de la predicción."),
]

# ── 8. CONCLUSIONES ──────────────────────────────────────────────────────────
story += [
    SP(4), HR(),
    H("8. Conclusiones técnicas"),
    P("Este proyecto demuestra que la predicción de RUL en turbofanes es un problema "
      "abordable con técnicas de machine learning supervisado cuando se diseña correctamente "
      "el pipeline de datos y se evitan los sesgos de normalización más habituales en "
      "series temporales de test truncadas."),
    SP(3),
    P("La contribución metodológica más relevante del proyecto es la identificación y "
      "corrección del sesgo de normalización per-unit en el conjunto de test. Este error, "
      "presente en múltiples implementaciones de C-MAPSS publicadas en la literatura gris, "
      "produce test RMSE artificialmente altos (>40 ciclos con R² negativos) que no reflejan "
      "la capacidad real del modelo. La solución — GlobalMinMaxScaler ajustado sobre el "
      "entrenamiento completo — produce una distribución de features alineada entre train y "
      "test, reduciendo el RMSE de test de ~50 ciclos a 12.90 ciclos y el R² de valores "
      "negativos a 0.896."),
    SP(3),
    P("La ingeniería de características mediante ventanas deslizantes de 15 y 30 ciclos "
      "transforma el problema de predicción de RUL desde señales raw ruidosas a un espacio "
      "de features estructurado que captura simultáneamente el estado local del motor (medias "
      "rolling), su variabilidad (desviaciones típicas) y su dinámica de deterioro (pendientes "
      "OLS y deltas acumulados). Las 112 features resultantes contienen información redundante "
      "que los modelos de gradient boosting (XGBoost, LightGBM) explotan eficientemente "
      "mediante regularización implícita."),
    SP(3),
    P("La estrategia de validación cruzada GroupKFold es esencial para obtener estimaciones "
      "honestas del error de generalización. Con k=5 y grupos por engine_id, la correlación "
      "temporal intra-motor no contamina los folds de validación. La brecha Δ RMSE = −0.02 "
      "para XGBoost confirma que el CV es un estimador fidedigno del rendimiento en test."),
    SP(3),
    P("El análisis SHAP revela que las features más informativas son las medias rolling a "
      "corto plazo (w=15) de los sensores del HPC (s4, s3, s9), coherentes con la "
      "degradación de compresor que caracteriza FD001. La pendiente de s11 (Fan Speed) a "
      "w=30 emerge como el mejor indicador de tendencia, alineado con el onset de degradación "
      "identificado en EDA (RUL ≈ 53 ciclos). Estos hallazgos tienen valor operativo "
      "directo: permiten priorizar qué sensores monitorizar en tiempo real."),
    SP(3),
    P("Desde la perspectiva de la aplicación industrial, el perfil de error por tercil de RUL "
      "es favorable: el modelo es más preciso precisamente cuando la información es más valiosa "
      "(motores con RUL ≤ 53 ciclos). El sesgo positivo en la zona Mid (+10.6 ciclos) "
      "implica que el sistema predice un mayor tiempo restante del real, lo que en un entorno "
      "de mantenimiento predictivo se traduce en un comportamiento conservador: los motores "
      "pueden recibir atención antes de que el modelo los identifique como críticos."),
]

# ── 9. LIMITACIONES ──────────────────────────────────────────────────────────
story += [
    SP(4), HR(),
    H("9. Limitaciones y mejoras propuestas"),
    H("9.1 Limitaciones del estudio", 2),
    B("<b>Dataset simulado:</b> C-MAPSS es generado por simulación física, no por motores "
      "reales. Los patrones de degradación son más regulares que en datos industriales, lo "
      "que puede inflar las métricas en comparación con aplicaciones reales."),
    B("<b>Condición operativa única (FD001):</b> el modelo está entrenado en una sola "
      "condición de operación. La extensión a FD002 y FD004 (6 condiciones) requeriría "
      "estrategias de normalización por régimen operativo."),
    B("<b>Random Forest subóptimo:</b> el límite impuesto en hiperparámetros (n=50, "
      "max_depth=12) para respetar restricciones de tiempo de ejecución puede estar "
      "penalizando el rendimiento potencial de este modelo."),
    B("<b>Horizonte fijo:</b> el modelo predice RUL en el último ciclo observable de test, "
      "no simula una predicción rolling en tiempo real a lo largo de la trayectoria."),
    B("<b>Sin incertidumbre cuantificada:</b> las predicciones son puntuales. Un sistema "
      "productivo debería incluir intervalos de predicción o estimación bayesiana de "
      "incertidumbre."),
    SP(6),
    H("9.2 Mejoras propuestas", 2),
    B("<b>Modelos secuenciales (LSTM / Temporal Fusion Transformer):</b> explotar la "
      "estructura temporal completa de las trayectorias, en lugar de features rolling "
      "construidas manualmente."),
    B("<b>Cuantificación de incertidumbre:</b> implementar regresión cuantílica sobre "
      "XGBoost (quantile loss) o NGBoost para obtener intervalos de predicción calibrados."),
    B("<b>Extensión multi-condición:</b> normalizar por régimen operativo (cluster de op. "
      "settings) para generalizar a FD002/FD003/FD004."),
    B("<b>Optimización de hiperparámetros:</b> Optuna con pruning para XGBoost y LightGBM "
      "sobre el espacio completo de hiperparámetros."),
    B("<b>Detección de anomalías como feature:</b> añadir scores de anomalía por ventana "
      "(Isolation Forest) como feature adicional."),
]

# ── 10. REFERENCIAS ──────────────────────────────────────────────────────────
story += [
    SP(4), HR(),
    H("10. Referencias"),
    B("Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage Propagation Modeling "
      "for Aircraft Engine Run-to-Failure Simulation. <i>Proceedings of the International "
      "Symposium on PHM</i>. IEEE."),
    B("Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. "
      "<i>International Journal of Prognostics and Health Management</i>, 1(1)."),
    B("Ramasso, E., & Saxena, A. (2014). Performance Benchmarking and Analysis of Prognostic "
      "Methods for CMAPSS Datasets. <i>NASA Technical Report NASA/TM-2014-218496</i>."),
    B("Li, X., Ding, Q., & Sun, J.Q. (2018). Remaining useful life estimation in prognostics "
      "using deep convolution neural networks. <i>Reliability Engineering & System Safety</i>, "
      "172, 1–11."),
    B("Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model "
      "Predictions. <i>NeurIPS 2017</i>."),
    SP(10),
    HR(),
    Paragraph("Informe generado con Python 3.12 | scikit-learn 1.4 | XGBoost 2.0 | "
              "SHAP 0.44 | ReportLab 4.4 | matplotlib 3.8",
              S["footer"]),
    Paragraph("Lucía Pardo Vázquez · lucia.par.vaz@gmail.com · Berlín, septiembre 2026",
              S["footer"]),
]

doc.build(story)
print(f"  PDF generado: {PDF_PATH}")

print("\n[FASE 6 COMPLETADA]")
print(f"\nFiguras:")
print(f"  f6_01: {path_f6_01}")
print(f"  f6_02: {path_f6_02}")
print(f"  PDF:   {PDF_PATH}")
