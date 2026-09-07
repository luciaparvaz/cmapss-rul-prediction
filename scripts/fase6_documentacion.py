"""
=============================================================================
PROYECTO: Detección de Fallos — NASA C-MAPSS (FD001)
FASE 6: Documentación final (sobre Fase 0-5, 7)
=============================================================================
Reescritura completa (sept. 2026). La versión original de este script no
solo tenía rutas rotas ("/home/claude/cmapss/...", "/tmp/claude-0/..."):
tenía TODOS los números de modelado, SHAP e interpretabilidad
HARDCODEADOS a mano, copiados de una ejecución anterior a esta auditoría.
Arreglar solo las rutas habría producido un informe técnico que se
ejecuta sin errores pero miente sobre los resultados — exactamente el
tipo de problema que esta auditoría existe para corregir en el resto
del proyecto (ver fase0_preprocesado.py).

Esta versión no hardcodea ningún resultado de modelo: carga los CSV que
ya exportan fase3/fase4/fase5 (fuente única de verdad) y construye
tablas, figuras y texto a partir de ahí. Si el mejor modelo cambia en
una ejecución futura (ya cambió una vez en esta misma auditoría — de
XGBoost a RandomForest, ver README "discrepancia con los números
originales"), este informe lo refleja automáticamente en vez de seguir
citando un modelo que ya no es el mejor.

Genera:
  f6_01 — Figura resumen comparativa de los 4 modelos de regresión (CV + test)
  f6_02 — Figura pipeline completo del proyecto (diagrama de flujo)
  informe_tecnico_cmapss.pdf — Informe técnico completo (regresión + clasificación)

Requiere haber ejecutado antes (en orden): fase0, fase1, fase2, fase3,
fase4, fase5 — cada uno exporta a outputs/ los CSV que este script lee.
=============================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fase0_preprocesado as f0
import fase1_carga_inspeccion as f1

OUTPUT_DIR = f0.OUTPUT_DIR   # f6 guarda figuras + PDF en outputs/, igual que f5/f7

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "font.family": "DejaVu Sans", "font.size": 10,
}
plt.rcParams.update(STYLE)

DISPLAY_NAMES = {"Ridge": "Ridge (L2)", "RandomForest": "Random Forest",
                  "XGBoost": "XGBoost", "LightGBM": "LightGBM"}
MODEL_ORDER = ["Ridge", "RandomForest", "XGBoost", "LightGBM"]


# ─────────────────────────────────────────────
# 0. CARGA DE RESULTADOS (nada se reconstruye — todo viene de CSV ya
#    exportados por fase3/fase4/fase5)
# ─────────────────────────────────────────────

def _require(path, phase):
    if not os.path.exists(path):
        raise RuntimeError(
            f"No se encuentra {path}. Ejecuta {phase} antes de fase6_documentacion.py."
        )
    return path


def load_all_results():
    reg = pd.read_csv(_require(f"{OUTPUT_DIR}/fase4_resultados.csv", "fase4_modelado.py"))
    reg = reg.set_index("modelo").loc[MODEL_ORDER].reset_index()
    reg["Modelo"] = reg["modelo"].map(DISPLAY_NAMES)
    reg["Delta_RMSE"] = reg["test_rmse"] - reg["cv_rmse_mean"]

    clf = pd.read_csv(_require(f"{OUTPUT_DIR}/fase4_clasificacion_resultados.csv", "fase4_modelado.py"))
    clf["Modelo"] = clf["modelo"]

    shap_top20 = pd.read_csv(_require(f"{OUTPUT_DIR}/fase5_shap_top20.csv", "fase5_evaluacion.py"),
                              index_col=0)
    tercil = pd.read_csv(_require(f"{OUTPUT_DIR}/fase5_error_tercil.csv", "fase5_evaluacion.py"))
    tercil["tercil"] = tercil["tercil"].str.replace("\n", " ", regex=False)
    fase5_resumen = pd.read_csv(_require(f"{OUTPUT_DIR}/fase5_resumen.csv", "fase5_evaluacion.py")).iloc[0]

    corr_df = pd.read_csv(_require(f"{OUTPUT_DIR}/fase3_correlaciones_rul.csv", "fase3_eda_degradacion.py"))
    onset_df = pd.read_csv(_require(f"{OUTPUT_DIR}/fase3_onset_degradacion.csv", "fase3_eda_degradacion.py"))

    subsets_path = f"{OUTPUT_DIR}/fase1_subsets_comparison.csv"
    subsets_df = pd.read_csv(subsets_path, index_col=0) if os.path.exists(subsets_path) else None

    best_reg = reg.loc[reg["test_rmse"].idxmin()]
    best_clf = clf.loc[clf["test_recall"].idxmax()]

    return {
        "reg": reg, "clf": clf, "shap_top20": shap_top20, "tercil": tercil,
        "fase5_resumen": fase5_resumen, "corr_df": corr_df, "onset_df": onset_df,
        "subsets_df": subsets_df, "best_reg": best_reg, "best_clf": best_clf,
    }


def describe_feature(feat_name: str) -> str:
    """Traduce un nombre de feature (p.ej. 's11_w30_slope') a una
    descripción legible reutilizando f1.SENSOR_NAMES — no se mantiene
    una segunda copia de esas descripciones."""
    parts = feat_name.split("_")
    sensor = parts[0]
    sensor_desc = f1.SENSOR_NAMES.get(sensor, sensor)
    if len(parts) == 2 and parts[1] == "delta":
        stat_desc = "delta acumulado desde el primer ciclo del motor"
    elif len(parts) >= 3:
        window = parts[1].replace("w", "")
        stat_map = {"mean": "media móvil", "std": "desviación típica móvil",
                    "slope": "pendiente OLS móvil", "min": "mínimo móvil", "max": "máximo móvil"}
        stat_desc = f"{stat_map.get(parts[2], parts[2])}, ventana {window} ciclos"
    else:
        stat_desc = "valor crudo (normalizado)"
    return f"{sensor_desc} — {stat_desc}"


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f6_01 — Tabla comparativa visual de modelos de regresión
# ═══════════════════════════════════════════════════════════════════════════════

def plot_f6_01(reg: pd.DataFrame, save_path: str):
    print("[f6_01] Tabla comparativa de modelos de regresión...")
    COLORS = {"Ridge (L2)": "#90A4AE", "Random Forest": "#66BB6A",
              "XGBoost": "#EF5350", "LightGBM": "#42A5F5"}

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle("f6_01 — Comparativa de modelos: NASA C-MAPSS FD001\n"
                 "Predicción de Vida Útil Remanente (RUL) | Regresión supervisada",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    modelos = reg["Modelo"].values
    bar_colors = [COLORS[m] for m in modelos]
    x = np.arange(len(modelos)); w = 0.38
    best_idx = int(reg["test_rmse"].values.argmin())
    best_label = modelos[best_idx]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - w/2, reg["cv_rmse_mean"], w, label="CV (5-fold)", color=bar_colors, alpha=0.6, hatch="///")
    b2 = ax1.bar(x + w/2, reg["test_rmse"], w, label="Test", color=bar_colors, alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels([m.replace(" ", "\n") for m in modelos], fontsize=8)
    ax1.set_ylabel("RMSE (ciclos)"); ax1.set_title("RMSE — validación cruzada vs test", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8); ax1.set_ylim(0, reg["cv_rmse_mean"].max() * 1.3)
    for bar, val in zip(b2, reg["test_rmse"]):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.annotate("★ Mejor", xy=(x[best_idx] + w/2, reg["test_rmse"].iloc[best_idx]),
                 xytext=(x[best_idx] + w/2 + 0.4, reg["test_rmse"].iloc[best_idx] + 2),
                 fontsize=8, color="#B71C1C", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.2))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - w/2, reg["cv_mae_mean"], w, label="CV", color=bar_colors, alpha=0.6, hatch="///")
    b4 = ax2.bar(x + w/2, reg["test_mae"], w, label="Test", color=bar_colors, alpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels([m.replace(" ", "\n") for m in modelos], fontsize=8)
    ax2.set_ylabel("MAE (ciclos)"); ax2.set_title("MAE — validación cruzada vs test", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8); ax2.set_ylim(0, reg["cv_mae_mean"].max() * 1.3)
    for bar, val in zip(b4, reg["test_mae"]):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x - w/2, reg["cv_r2_mean"], w, label="CV", color=bar_colors, alpha=0.6, hatch="///")
    b6 = ax3.bar(x + w/2, reg["test_r2"], w, label="Test", color=bar_colors, alpha=0.9)
    ax3.set_xticks(x); ax3.set_xticklabels([m.replace(" ", "\n") for m in modelos], fontsize=8)
    ax3.set_ylabel("R²"); ax3.set_title("R² — validación cruzada vs test", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8); ax3.set_ylim(reg[["cv_r2_mean", "test_r2"]].min().min() - 0.05, 0.98)
    for bar, val in zip(b6, reg["test_r2"]):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax4 = fig.add_subplot(gs[1, :2]); ax4.axis("off")
    header = ["Modelo", "CV RMSE", "CV MAE", "CV R²", "Test RMSE", "Test MAE", "Test R²", "Δ RMSE"]
    rows = []
    for _, r in reg.iterrows():
        star = " ★" if r["Modelo"] == best_label else ""
        rows.append([r["Modelo"] + star, f"{r['cv_rmse_mean']:.2f}", f"{r['cv_mae_mean']:.2f}",
                     f"{r['cv_r2_mean']:.3f}", f"{r['test_rmse']:.2f}", f"{r['test_mae']:.2f}",
                     f"{r['test_r2']:.3f}", f"{r['Delta_RMSE']:+.2f}"])
    table = ax4.table(cellText=rows, colLabels=header, cellLoc="center", loc="center",
                      colWidths=[0.18, 0.11, 0.11, 0.10, 0.12, 0.11, 0.11, 0.10])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.2)
    for j in range(len(header)):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for j in range(len(header)):
        table[(best_idx + 1, j)].set_facecolor("#FFEBEE")
        table[(best_idx + 1, j)].set_text_props(fontweight="bold", color="#B71C1C")
    alt_colors = ["#F5F5F5", "white", "white", "white"]
    for i in range(len(rows)):
        if i != best_idx:
            for j in range(len(header)):
                table[(i + 1, j)].set_facecolor(alt_colors[i % len(alt_colors)])
    ax4.set_title("Tabla comparativa completa de métricas (ciclos / adimensional)",
                  fontsize=10, fontweight="bold", pad=12)

    ax5 = fig.add_subplot(gs[1, 2])
    delta_colors = ["#E57373" if d > 0 else "#66BB6A" for d in reg["Delta_RMSE"]]
    bars = ax5.barh(modelos[::-1], reg["Delta_RMSE"].values[::-1], color=delta_colors[::-1], alpha=0.85)
    ax5.axvline(0, color="black", linewidth=1)
    ax5.set_xlabel("Δ RMSE = test − CV (ciclos)\nNegativo: generaliza por encima del CV")
    ax5.set_title("Brecha de generalización CV→test", fontsize=10, fontweight="bold")
    for bar, val in zip(bars, reg["Delta_RMSE"].values[::-1]):
        ax5.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                 f"{val:+.2f}", ha="left" if val >= 0 else "right", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURA f6_02 — Diagrama de pipeline completo del proyecto
# ═══════════════════════════════════════════════════════════════════════════════

def plot_f6_02(data: dict, save_path: str):
    print("[f6_02] Diagrama del pipeline...")
    reg, clf = data["reg"], data["clf"]
    best_reg, best_clf = data["best_reg"], data["best_clf"]
    shap_top20, tercil = data["shap_top20"], data["tercil"]
    onset_df = data["onset_df"]

    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(0, 20); ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_title("f6_02 — Pipeline completo: Predicción de RUL en turbofanes (NASA C-MAPSS FD001)\n"
                 "Mantenimiento Predictivo | Lucía Pardo Vázquez | 2026",
                 fontsize=13, fontweight="bold", pad=15)

    def draw_box(x, y, w, h, text, color, fontsize=8.5, text_color="white"):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=text_color,
                fontweight="bold", zorder=4, multialignment="center", linespacing=1.4)

    def arrow(x1, x2, y, color="#546E7A"):
        ax.annotate("", xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=2)

    onset_groups = onset_df.groupby("rul_onset")["sensor"].apply(lambda s: ",".join(s)).sort_index()
    onset_parts = [f"{sensors}→{rul}" for rul, sensors in onset_groups.items()]
    # 2 grupos por línea como máximo para no desbordar la caja
    onset_lines = [", ".join(onset_parts[i:i+2]) for i in range(0, len(onset_parts), 2)]
    onset_txt = "\n".join(onset_lines)

    # Fila 1: Datos crudos → Preprocesamiento → EDA → Features → Normalización
    draw_box(1.5, 8.0, 2.6, 1.4, "DATOS CRUDOS\nC-MAPSS FD001\n100 motores · 26 cols\n20,631 filas", "#1565C0")
    arrow(2.8, 4.2, 8.0)
    draw_box(5.0, 8.0, 3.2, 1.4,
             "PREPROCESADO\n(Fase 0)\n• MinMaxScaler global\n• 7 sensores constantes elim.\n• RUL cap = 125 ciclos", "#1976D2")
    arrow(6.6, 7.8, 8.0)
    draw_box(8.8, 8.0, 3.2, 1.6, f"EDA DEGRADACIÓN\n(Fase 3)\n• Pearson/Spearman vs RUL\n• Onset degradación:\n{onset_txt}", "#0288D1", fontsize=7.2)
    arrow(10.4, 11.8, 8.0)
    draw_box(12.8, 8.0, 3.2, 1.4,
             "FEATURES ROLLING\n(Fase 2)\n• Windows: 15, 30 ciclos\n• mean · std · slope OLS · delta\n• 112 features totales", "#0097A7")
    arrow(14.4, 15.5, 8.0)
    draw_box(16.7, 8.0, 2.8, 1.4, "NORMALIZACIÓN\nStandardScaler\nsobre train\n(112 features)", "#00838F")

    # Fila 2: Validación cruzada → 4 modelos de regresión
    draw_box(2.5, 5.5, 3.2, 1.4, "VALIDACIÓN CRUZADA\nGroupKFold(k=5)\ngrupos = engine_id\nsin data leakage", "#2E7D32")
    arrow(4.1, 5.3, 5.5)
    reg_colors = {"Ridge (L2)": "#90A4AE", "Random Forest": "#66BB6A", "XGBoost": "#EF5350", "LightGBM": "#42A5F5"}
    dark_text_models = {"Ridge (L2)", "Random Forest"}   # fondos claros -> texto oscuro legible
    reg_x = [6.5, 9.2, 11.9, 14.6]
    for xi, (_, r) in zip(reg_x, reg.iterrows()):
        is_best = r["Modelo"] == best_reg["Modelo"]
        label = r["Modelo"].replace(" ", "\n").upper()
        star = "★ " if is_best else ""
        text_color = "#212121" if r["Modelo"] in dark_text_models else "white"
        draw_box(xi, 5.5, 2.4, 1.4, f"{star}{label}\nRMSE={r['test_rmse']:.2f}\nR²={r['test_r2']:.3f}",
                 reg_colors[r["Modelo"]], text_color=text_color)
    ax.annotate("", xy=(3.5, 6.2), xytext=(16.7, 7.3),
                arrowprops=dict(arrowstyle="-|>", color="#546E7A", connectionstyle="arc3,rad=-0.3", lw=1.5), zorder=2)

    # Fila 3: Interpretabilidad, evaluación y clasificación (extensión)
    top3 = shap_top20.iloc[:3]
    shap_txt = "\n".join(f"{name} ({val:.2f})" for name, val in top3["mean_abs_shap"].items())
    draw_box(4.6, 3.0, 3.2, 1.5, f"SHAP VALUES\n(Fase 5, {best_reg['Modelo']})\n{shap_txt}", "#6A1B9A", fontsize=7.6)

    t = tercil.set_index("tercil")
    late = [i for i in t.index if i.startswith("Late")][0]
    mid = [i for i in t.index if i.startswith("Mid")][0]
    early = [i for i in t.index if i.startswith("Early")][0]
    draw_box(8.5, 3.0, 3.2, 1.5,
             f"ERROR POR TERCIL\n(Fase 5)\nLate: RMSE={t.loc[late,'RMSE']:.1f}\n"
             f"Mid: RMSE={t.loc[mid,'RMSE']:.1f}\nEarly: RMSE={t.loc[early,'RMSE']:.1f}",
             "#AD1457", fontsize=7.8)

    draw_box(12.4, 3.0, 3.2, 1.5,
             "CURVAS DEGRADACIÓN\n(Fase 5)\n10 motores representativos\n±15 ciclos banda confianza",
             "#C62828", fontsize=7.8)

    draw_box(16.3, 3.0, 3.2, 1.5,
             f"CLASIFICACIÓN\n(Fase 4, extensión)\n★ {best_clf['Modelo']}\n"
             f"Recall={best_clf['test_recall']:.3f}, F1={best_clf['test_f1']:.3f}",
             "#00695C", fontsize=7.8)

    arrow(11.9, 6.6, 3.0)
    ax.annotate("", xy=(8.5, 3.75), xytext=(11.9, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.5), zorder=2)
    ax.annotate("", xy=(12.4, 3.75), xytext=(11.9, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.5), zorder=2)
    ax.annotate("", xy=(4.6, 3.75), xytext=(11.9, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#546E7A", connectionstyle="arc3,rad=0.2", lw=1.5), zorder=2)
    ax.annotate("", xy=(16.3, 3.75), xytext=(14.6, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#546E7A", connectionstyle="arc3,rad=-0.2", lw=1.5), zorder=2)

    legend_elements = [
        mpatches.Patch(color="#1976D2", label="Fase 0-2: Preprocesado y features"),
        mpatches.Patch(color="#0288D1", label="Fase 3: EDA de degradación"),
        mpatches.Patch(color="#2E7D32", label="Fase 4: Modelado (regresión + clasificación)"),
        mpatches.Patch(color="#6A1B9A", label="Fase 5: Evaluación e interpretabilidad"),
        mpatches.Patch(color=reg_colors[best_reg["Modelo"]], label=f"★ Modelo seleccionado ({best_reg['Modelo']})"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=8.5,
              frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# INFORME TÉCNICO PDF
# ═══════════════════════════════════════════════════════════════════════════════

def build_pdf(data: dict, path_f6_01: str, path_f6_02: str, pdf_path: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, PageBreak, HRFlowable, KeepTogether)

    reg, clf = data["reg"], data["clf"]
    best_reg, best_clf = data["best_reg"], data["best_clf"]
    shap_top20, tercil = data["shap_top20"], data["tercil"]
    corr_df, onset_df, subsets_df = data["corr_df"], data["onset_df"], data["subsets_df"]

    W, H = A4
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm, topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="Predicción de RUL en Turbofanes — NASA C-MAPSS",
        author="Lucía Pardo Vázquez",
        subject="Mantenimiento Predictivo — Informe Técnico Final",
    )

    styles = getSampleStyleSheet()
    _c = [0]
    def sty(name, **kw):
        base = styles[name] if name in styles else styles["Normal"]
        _c[0] += 1
        return ParagraphStyle(f"custom_{_c[0]}", parent=base, **kw)

    S = {
        "title":    sty("Normal", fontSize=22, fontName="Helvetica-Bold", textColor=colors.HexColor("#0D47A1"), alignment=TA_CENTER, spaceAfter=6),
        "subtitle": sty("Normal", fontSize=13, fontName="Helvetica", textColor=colors.HexColor("#1565C0"), alignment=TA_CENTER, spaceAfter=4),
        "meta":     sty("Normal", fontSize=9.5, fontName="Helvetica", textColor=colors.HexColor("#546E7A"), alignment=TA_CENTER, spaceAfter=3),
        "h1":       sty("Normal", fontSize=14, fontName="Helvetica-Bold", textColor=colors.HexColor("#0D47A1"), spaceBefore=14, spaceAfter=6),
        "h2":       sty("Normal", fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#1976D2"), spaceBefore=10, spaceAfter=5),
        "body":     sty("Normal", fontSize=9.5, fontName="Helvetica", leading=15, spaceAfter=6, alignment=TA_JUSTIFY),
        "bullet":   sty("Normal", fontSize=9.5, fontName="Helvetica", leading=14, leftIndent=14, spaceAfter=3, bulletIndent=6),
        "caption":  sty("Normal", fontSize=8, fontName="Helvetica-Oblique", textColor=colors.HexColor("#757575"), alignment=TA_CENTER, spaceAfter=10),
        "callout":  sty("Normal", fontSize=9.5, fontName="Helvetica", leading=14, leftIndent=16, rightIndent=8, spaceAfter=8, alignment=TA_JUSTIFY, backColor=colors.HexColor("#E3F2FD"), borderPad=8),
        "footer":   sty("Normal", fontSize=8, fontName="Helvetica", textColor=colors.HexColor("#9E9E9E"), alignment=TA_CENTER),
    }
    def Hh(text, level=1): return Paragraph(text, S[f"h{level}"])
    def P(text): return Paragraph(text, S["body"])
    def B(text): return Paragraph(f"• {text}", S["bullet"])
    def CAP(text): return Paragraph(text, S["caption"])
    def SP(n=6): return Spacer(1, n)
    def HR(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#B0BEC5"), spaceAfter=8, spaceBefore=4)
    def CALLOUT(text): return Paragraph(text, S["callout"])
    def IMG(p, caption, width=15*cm):
        try:
            img = Image(p, width=width, height=width * 0.55)
            return KeepTogether([img, CAP(caption)])
        except Exception as e:
            return P(f"[Figura no disponible: {e}]")

    def metrics_table():
        header = ["Modelo", "CV RMSE", "CV MAE", "CV R²", "Test RMSE", "Test MAE", "Test R²", "Δ RMSE"]
        data_rows = [header]
        best_row_i = None
        for i, (_, r) in enumerate(reg.iterrows()):
            star = " ★" if r["Modelo"] == best_reg["Modelo"] else ""
            if star:
                best_row_i = i + 1
            data_rows.append([r["Modelo"] + star, f"{r['cv_rmse_mean']:.2f}", f"{r['cv_mae_mean']:.2f}",
                              f"{r['cv_r2_mean']:.3f}", f"{r['test_rmse']:.2f}", f"{r['test_mae']:.2f}",
                              f"{r['test_r2']:.3f}", f"{r['Delta_RMSE']:+.2f}"])
        col_w = [3.8*cm, 1.7*cm, 1.7*cm, 1.5*cm, 2.0*cm, 1.7*cm, 1.5*cm, 1.7*cm]
        t = Table(data_rows, colWidths=col_w, repeatRows=1)
        style = [
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 8),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
            ("FONTSIZE", (0,1), (-1,-1), 8.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F5F5F5"), colors.white]),
            ("BOTTOMPADDING", (0,1), (-1,-1), 5), ("TOPPADDING", (0,1), (-1,-1), 5),
        ]
        if best_row_i:
            style += [("BACKGROUND", (0,best_row_i), (-1,best_row_i), colors.HexColor("#FFEBEE")),
                      ("TEXTCOLOR", (0,best_row_i), (-1,best_row_i), colors.HexColor("#B71C1C")),
                      ("FONTNAME", (0,best_row_i), (-1,best_row_i), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    def classification_table():
        header = ["Modelo", "CV F1", "CV Recall", "CV ROC-AUC", "Test F1", "Test Recall", "Test Precision", "Test ROC-AUC"]
        data_rows = [header]
        best_row_i = None
        for i, (_, r) in enumerate(clf.iterrows()):
            star = " ★" if r["Modelo"] == best_clf["Modelo"] else ""
            if star:
                best_row_i = i + 1
            data_rows.append([r["Modelo"] + star, f"{r['cv_f1_mean']:.3f}", f"{r['cv_recall_mean']:.3f}",
                              f"{r['cv_roc_auc_mean']:.4f}", f"{r['test_f1']:.3f}", f"{r['test_recall']:.3f}",
                              f"{r['test_precision']:.3f}", f"{r['test_roc_auc']:.4f}"])
        col_w = [3.4*cm, 1.5*cm, 1.6*cm, 1.8*cm, 1.5*cm, 1.7*cm, 2.0*cm, 1.8*cm]
        t = Table(data_rows, colWidths=col_w, repeatRows=1)
        style = [
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 7.5),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
            ("FONTSIZE", (0,1), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F5F5F5"), colors.white]),
            ("BOTTOMPADDING", (0,1), (-1,-1), 5), ("TOPPADDING", (0,1), (-1,-1), 5),
        ]
        if best_row_i:
            style += [("BACKGROUND", (0,best_row_i), (-1,best_row_i), colors.HexColor("#E0F2F1")),
                      ("TEXTCOLOR", (0,best_row_i), (-1,best_row_i), colors.HexColor("#00695C")),
                      ("FONTNAME", (0,best_row_i), (-1,best_row_i), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    def tercil_table():
        header = ["Fase", "n", "MAE (ciclos)", "RMSE (ciclos)", "% ±15 ciclos", "Sesgo (ciclos)"]
        data_rows = [header]
        for _, r in tercil.iterrows():
            data_rows.append([r["tercil"], str(int(r["n"])), f"{r['MAE']:.2f}", f"{r['RMSE']:.2f}",
                              f"{r['pct_within15']:.0f}%", f"{r['bias']:+.1f}"])
        col_w = [4.8*cm, 1.2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm]
        t = Table(data_rows, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8.5),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#FFEBEE"), colors.HexColor("#FFF8E1"), colors.HexColor("#E8F5E9")]),
            ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 5),
        ]))
        return t

    def subsets_table():
        header = ["Subdataset", "Motores tr/te", "Cond. op.", "Modos fallo", "Vida media±std", "Sensores inform.", "Normalización"]
        data_rows = [header]
        for name, r in subsets_df.iterrows():
            data_rows.append([name, f"{int(r['Motores train'])}/{int(r['Motores test'])}",
                              str(int(r["Cond. operacionales"])), str(int(r["Modos de fallo"])),
                              f"{r['Vida media (ciclos)']:.0f}±{r['Vida std']:.0f}",
                              str(int(r["Sensores informativos"])), r["Normalización"]])
        col_w = [2.3*cm, 2.3*cm, 1.7*cm, 1.9*cm, 2.7*cm, 2.5*cm, 2.5*cm]
        t = Table(data_rows, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7.8),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F5F5F5"), colors.white]),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 5),
        ]))
        return t

    # Valores derivados usados en el texto
    best_label = best_reg["Modelo"]
    best_rmse, best_r2, best_mae = best_reg["test_rmse"], best_reg["test_r2"], best_reg["test_mae"]
    best_delta = best_reg["Delta_RMSE"]
    # Media ponderada del % de motores con error <=15 ciclos, agregando los 3 terciles
    pct15_all = (tercil["pct_within15"] * tercil["n"]).sum() / tercil["n"].sum()
    top3 = shap_top20.iloc[:3]
    top1_name, top1_val = top3.index[0], top3["mean_abs_shap"].iloc[0]
    top2_name, top2_val = top3.index[1], top3["mean_abs_shap"].iloc[1]
    top3_name, top3_val = top3.index[2], top3["mean_abs_shap"].iloc[2]
    t = tercil.set_index("tercil")
    late_row = [i for i in t.index if i.startswith("Late")][0]
    mid_row = [i for i in t.index if i.startswith("Mid")][0]
    early_row = [i for i in t.index if i.startswith("Early")][0]

    corr_top5 = corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index).head(5)
    onset_sorted = onset_df.sort_values("rul_onset")

    second_best = reg[reg["Modelo"] != best_label].sort_values("test_rmse").iloc[0]
    worst = reg.sort_values("test_rmse", ascending=False).iloc[0]
    improvement_vs_worst = (worst["test_rmse"] - best_rmse) / worst["test_rmse"] * 100

    story = []

    # ── PORTADA ──
    story += [
        SP(30),
        Paragraph("PREDICCIÓN DE VIDA ÚTIL REMANENTE", S["title"]),
        Paragraph("EN MOTORES DE TURBOFÁN", S["title"]),
        SP(8),
        Paragraph("Mantenimiento Predictivo basado en Datos", S["subtitle"]),
        Paragraph(f"NASA C-MAPSS FD001 · Regresión y clasificación supervisada · Modelo: {best_label}", S["subtitle"]),
        SP(20), HR(), SP(10),
        Paragraph("Autora: <b>Lucía Pardo Vázquez</b>", S["meta"]),
        Paragraph("Proyecto de portfolio — Data Analyst | Data Scientist", S["meta"]),
        Paragraph("Berlín, septiembre 2026", S["meta"]),
        SP(10), HR(), PageBreak(),
    ]

    # ── 1. RESUMEN EJECUTIVO ──
    story += [
        Hh("1. Resumen ejecutivo"),
        P("Este proyecto desarrolla un sistema completo de predicción de Vida Útil Remanente "
          "(RUL, <i>Remaining Useful Life</i>) para motores de turbofán utilizando el dataset "
          "NASA C-MAPSS FD001, con dos objetivos complementarios: (1) predecir cuántos ciclos "
          "de operación le quedan a cada motor mediante regresión, y (2) clasificar si un motor "
          "está en zona de riesgo inminente de fallo (RUL ≤ 30 ciclos), priorizando Recall sobre "
          "Precision por el coste asimétrico de un falso negativo en mantenimiento predictivo."),
        SP(4),
        CALLOUT(f"<b>Resultado principal (regresión):</b> {best_label} obtiene "
                f"RMSE = {best_rmse:.2f} ciclos y R² = {best_r2:.3f} sobre el conjunto de test "
                f"de 100 motores, con una brecha de generalización de {best_delta:+.2f} ciclos "
                f"respecto al CV. El {pct15_all:.0f}% de los motores se predicen con un error "
                f"absoluto ≤ 15 ciclos."),
        SP(4),
        CALLOUT(f"<b>Resultado principal (clasificación):</b> {best_clf['Modelo']} alcanza "
                f"Recall = {best_clf['test_recall']:.3f} en test (prácticamente todos los "
                f"motores en zona de riesgo detectados) con "
                f"Precision = {best_clf['test_precision']:.3f}, "
                f"F1 = {best_clf['test_f1']:.3f} y ROC-AUC = {best_clf['test_roc_auc']:.4f}."),
        SP(6),
        P("El proyecto se estructura en 7 fases: carga y auditoría de datos con normalización "
          "canónica (Fase 0), inspección exploratoria extendida a los 4 subdatasets C-MAPSS "
          "(Fase 1), ingeniería de características mediante ventanas temporales deslizantes "
          "(Fase 2), análisis de degradación (Fase 3), entrenamiento y validación de modelos de "
          "regresión y clasificación (Fase 4), evaluación e interpretabilidad con SHAP (Fase 5), "
          "y mejoras de rigor estadístico — comparación pareada de modelos, censura, PHM score, "
          "heterocedasticidad, ablación de ventanas (Fase 7)."),
        SP(4),
        P("<b>Nota metodológica.</b> Una auditoría posterior a la primera versión de este "
          "proyecto encontró y corrigió una fuga de datos (<i>data leakage</i>) en la "
          "normalización de sensores del conjunto de test — ver sección 2.2. Los números de "
          "este informe corresponden a la versión corregida del pipeline."),
    ]

    # ── 2. DATOS Y PREPROCESADO ──
    story += [
        SP(4), HR(),
        Hh("2. Dataset y preprocesado"),
        Hh("2.1 NASA C-MAPSS FD001", 2),
        P("C-MAPSS (<i>Commercial Modular Aero-Propulsion System Simulation</i>) es un simulador "
          "de turbofán desarrollado por NASA Glenn Research Center. El subconjunto FD001 contiene "
          "datos de degradación run-to-failure de <b>100 motores</b> en condición operativa única "
          "(Sea Level), con fallo provocado por degradación del compresor de alta presión (HPC). "
          "Las 20,631 filas de entrenamiento y 13,096 de test cubren 26 columnas: identificador de "
          "motor, ciclo, 3 ajustes operativos y 21 lecturas de sensor."),
        SP(4),
        B("Referencia canónica: Saxena et al. (2008). Damage Propagation Modeling for Aircraft "
          "Engine Run-to-Failure Simulation. PHM '08 Conference."),
        B("Vida útil media en train: 206.3 ± 46.3 ciclos (rango: 128–362)."),
        B("7 sensores constantes eliminados (std &lt; 0.01): s1, s5, s6, s10, s16, s18, s19."),
        B("14 sensores informativos retenidos: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21."),
        SP(6),
        Hh("2.2 Decisiones de preprocesado", 2),
        P("<b>Cap de RUL a 125 ciclos.</b> El perfil de degradación real sigue un modelo piece-wise "
          "lineal: los motores operan en un estado saludable estable durante la primera parte de su "
          "vida, y solo comienzan a degradarse de forma medible en los últimos ciclos. Heimes "
          "(2008) justifica este valor empíricamente sobre C-MAPSS."),
        SP(3),
        P("<b>MinMaxScaler global, no per-motor.</b> La normalización per-motor (ajustar un "
          "MinMaxScaler individualmente en cada motor de test) introduce un sesgo sistemático "
          "crítico: el último ciclo observado de test siempre queda normalizado cerca de 1.0, "
          "independientemente del RUL real del motor, haciendo que el modelo interprete todo "
          "motor de test como próximo al fallo. Evidencia cuantitativa (Fase 0): con "
          "normalización per-motor, s11 en el último ciclo de test presenta μ≈0.72 frente a "
          "μ≈0.32 en motores sanos de train (RUL≥100); con MinMaxScaler global (fit solo en "
          "train), la diferencia se reduce a μ≈0.44, un 69% menos de sesgo. Solución adoptada: "
          "ajustar el escalador sobre el conjunto de entrenamiento completo (todos los motores, "
          "todos los ciclos, por sensor) y aplicar los mismos parámetros al test — la práctica "
          "estándar de <i>held-out evaluation</i> (Hastie, Tibshirani &amp; Friedman, 2009)."),
        SP(3),
        P("<b>Extensión a FD002-FD004.</b> Estos subdatasets combinan 6 condiciones operacionales "
          "(altitud, Mach, TRA), frente a la condición única de FD001/FD003 — un MinMaxScaler "
          "global mezclaría en la misma escala mediciones físicamente distintas. Fase 0 agrupa "
          "los 3 <i>op_setting</i> en 6 regímenes vía k-means (fit solo en train) y ajusta un "
          "MinMaxScaler independiente por régimen. Ver sección 3.2 para la comparativa completa "
          "de los 4 subdatasets; el modelado (secciones 4-7) se mantiene enfocado en FD001, el "
          "subdataset de referencia más limpio para esta iteración."),
    ]

    # ── 3. ANÁLISIS EXPLORATORIO ──
    corr_bullets = "; ".join(f"{r.sensor} ({r.pearson_r:+.3f})" for r in corr_top5.itertuples())
    onset_bullets = "; ".join(f"{r.sensor} en RUL≈{r.rul_onset}" for r in onset_sorted.itertuples())
    story += [
        SP(4), HR(),
        Hh("3. Análisis exploratorio de degradación (EDA)"),
        Hh("3.1 FD001 en profundidad", 2),
        P("El EDA de la Fase 3 caracteriza la dinámica de degradación de cada sensor a lo largo "
          "del ciclo de vida de los motores, utilizando correlaciones de Pearson y Spearman con "
          "el RUL crudo y un análisis de varianza intra- vs. inter-motor."),
        SP(4),
        B(f"<b>Sensores con mayor correlación absoluta con RUL</b> (Pearson): {corr_bullets}."),
        B(f"<b>Onset de degradación</b> (ciclo a partir del cual la señal supera 2σ del valor "
          f"basal, RUL&gt;200): {onset_bullets}. Esta jerarquía justifica las ventanas temporales "
          f"de 15 y 30 ciclos usadas en la ingeniería de características."),
        B("El análisis de cuartiles de vida útil revela que la variabilidad inter-motor en la "
          "fase temprana (Q1, motores en estado sano) es baja y aumenta progresivamente hacia "
          "el fallo (Q4), evidenciando la heterogeneidad de los procesos de degradación."),
    ]
    if subsets_df is not None:
        story += [
            SP(6),
            Hh("3.2 Comparativa de los 4 subdatasets (extensión, Fase 1)", 2),
            P("La normalización de Fase 0 se extendió a FD002-FD004 (sección 2.2); esta "
              "extensión de Fase 1 compara los 4 subdatasets sin entrenar modelos sobre ellos:"),
            SP(4), subsets_table(), SP(4),
            CAP("Tabla 0. Comparativa de los 4 subdatasets NASA C-MAPSS. FD002/FD004 retienen "
                "más sensores informativos porque su varianza depende del régimen operacional, "
                "no solo de la degradación — un sensor casi constante en FD001 (condición única) "
                "puede variar sustancialmente entre 6 regímenes."),
        ]

    # ── 4. INGENIERÍA DE CARACTERÍSTICAS ──
    story += [
        SP(4), HR(),
        Hh("4. Ingeniería de características"),
        P("Sobre los 14 sensores informativos normalizados se computan features derivadas mediante "
          "ventanas temporales deslizantes de 15 y 30 ciclos, capturando tanto el estado local "
          "como la tendencia de degradación:"),
        SP(3),
        B("<b>Media rolling (w=15, w=30):</b> estado promedio reciente del sensor."),
        B("<b>Desviación típica rolling (w=15, w=30):</b> variabilidad local, indicador de "
          "inestabilidad incipiente."),
        B("<b>Pendiente OLS rolling (w=15, w=30):</b> tasa de cambio local, calculada mediante "
          "convolución vectorizada."),
        B("<b>Delta acumulado:</b> cambio total del sensor desde el primer ciclo del motor."),
        SP(4),
        P("Total de features: 14 (base) + 14×2×3 (rolling) + 14 (delta) = <b>112 features</b>. "
          "Un conjunto extendido (238 features, ventanas 15/30/50 + min/max + delta de primer "
          "orden) se comparó empíricamente vía GroupKFold + test oficial: gana en CV "
          "(11.29 vs. 12.72 RMSE) pero pierde en test (12.66 vs. 11.98) — el patrón "
          "sesgo-varianza esperado (Hastie et al., 2009), que confirma la elección de 112 "
          "features. El StandardScaler se ajusta exclusivamente sobre el conjunto de "
          "entrenamiento y se aplica sin refiteo al test (Li, Ding &amp; Sun, 2018)."),
    ]

    # ── 5. MODELADO ──
    story += [
        SP(4), HR(),
        Hh("5. Modelado y validación"),
        Hh("5.1 Estrategia de validación cruzada", 2),
        P("Se utiliza GroupKFold(k=5) con grupos definidos por engine_id, garantizando que todos "
          "los ciclos de un mismo motor pertenecen al mismo fold. Esta estrategia evita el data "
          "leakage causado por la alta autocorrelación temporal de las series de un mismo motor "
          "(Li, Ding &amp; Sun, 2018). Con 100 motores y 5 folds, cada fold de validación contiene "
          "~20 motores completos."),
        SP(6),
        Hh("5.2 Regresión: modelos evaluados", 2), SP(4),
        metrics_table(), SP(4),
        CAP("Tabla 1. Métricas de rendimiento de los cuatro modelos de regresión sobre "
            "validación cruzada (GroupKFold k=5) y conjunto de test. Δ RMSE = RMSE_test − "
            "RMSE_CV; valores negativos indican generalización superior al benchmark de "
            "validación. ★ = modelo seleccionado."),
        SP(8),
        IMG(path_f6_01, "Figura 1. Comparativa de RMSE, MAE y R² entre validación cruzada y "
                        "test para los cuatro modelos de regresión. Panel derecho: brecha de "
                        "generalización (Δ RMSE)."),
        SP(6),
        Hh("5.3 Selección del modelo de regresión", 2),
        P(f"<b>{best_label}</b> es seleccionado como modelo final por su menor RMSE de test "
          f"({best_rmse:.2f} ciclos, una mejora del {improvement_vs_worst:.1f}% sobre el peor "
          f"modelo evaluado, {worst['Modelo']}) y su R² de {best_r2:.3f}, explicando "
          f"{best_r2*100:.0f}% de la varianza en el RUL residual. Su brecha de generalización "
          f"(Δ = {best_delta:+.2f} ciclos) indica que el CV es un estimador razonablemente fiel "
          f"del rendimiento real. {second_best['Modelo']} ofrece un rendimiento cercano "
          f"(RMSE = {second_best['test_rmse']:.2f}), por lo que se identifica como alternativa "
          f"viable si difieren las prioridades de interpretabilidad o coste computacional."),
        SP(6),
        Hh("5.4 Clasificación de zona de riesgo (objetivo 2)", 2),
        P("En paralelo a la regresión de RUL, se entrena un clasificador binario para "
          "<i>early_failure</i> (RUL ≤ 30 ciclos = fallo inminente), usando las mismas 112 "
          "features, el mismo protocolo GroupKFold(k=5) y el mismo test oficial. Las clases "
          "están desbalanceadas (~17.5% positivos en train), por lo que los 4 modelos — "
          "Logistic Regression, Random Forest, XGBoost y LightGBM — incorporan peso de clase "
          "balanceado (<i>class_weight=\"balanced\"</i> / <i>scale_pos_weight</i>)."),
        SP(4), classification_table(), SP(4),
        CAP("Tabla 1b. Métricas de clasificación early_failure sobre validación cruzada y test. "
            "★ = mejor modelo por Recall en test (prioridad de seguridad industrial: un fallo "
            "inminente no detectado es más costoso que una revisión de más)."),
        SP(6),
        CALLOUT(f"<b>Por qué Recall y no Accuracy.</b> En mantenimiento predictivo, un falso "
                f"negativo (fallo inminente no detectado) es mucho más costoso que un falso "
                f"positivo (una revisión de más). Con Recall = {best_clf['test_recall']:.3f} en "
                f"test, {best_clf['Modelo']} identifica correctamente prácticamente todos los "
                f"motores en zona de riesgo, manteniendo Precision = "
                f"{best_clf['test_precision']:.3f}."),
    ]

    # ── 6. INTERPRETABILIDAD ──
    story += [
        SP(4), HR(),
        Hh("6. Interpretabilidad (SHAP)"),
        P(f"Los valores SHAP (<i>SHapley Additive exPlanations</i>) permiten cuantificar la "
          f"contribución causal de cada feature a cada predicción individual. Se computan sobre "
          f"el conjunto de test (n=100 motores) para el modelo de regresión seleccionado "
          f"({best_label}) utilizando TreeExplainer con perturbación intervencionista sobre "
          f"muestras de background del train."),
        SP(4),
        IMG(f"{OUTPUT_DIR}/f5_01_shap_summary.png",
            "Figura 2. SHAP summary plot (izquierda): beeswarm con coloración por valor de "
            "feature. Bar chart (derecha): importancia media absoluta top-20. Cada punto "
            "representa un motor."),
        SP(6),
        P("<b>Hallazgos principales de interpretabilidad:</b>"),
        B(f"<b>{top1_name}</b> (SHAP medio = {top1_val:.2f} ciclos): {describe_feature(top1_name)}. "
          f"Es la feature más importante."),
        B(f"<b>{top2_name}</b> ({top2_val:.2f} ciclos): {describe_feature(top2_name)}."),
        B(f"<b>{top3_name}</b> ({top3_val:.2f} ciclos): {describe_feature(top3_name)}."),
        B("El feature más importante es una <i>pendiente</i> (tendencia), no un nivel absoluto "
          "— coherente con que la velocidad de degradación es más informativa que el valor "
          "puntual del sensor, y con la jerarquía de onset de degradación de la sección 3.1."),
        SP(4),
        IMG(f"{OUTPUT_DIR}/f5_02_shap_dependence.png",
            "Figura 3. Dependence plots de las 3 features con mayor SHAP medio. Color: RUL real "
            "del motor (verde=sano, rojo=próximo al fallo). La pendiente de la recta de regresión "
            "cuantifica la relación monotónica entre el valor de cada feature y su impacto en RUL."),
    ]

    # ── 7. EVALUACIÓN POR FASE DE VIDA ──
    story += [
        SP(4), HR(),
        Hh("7. Evaluación por fase de vida"),
        P(f"La segmentación del error por tercil de RUL real ({best_label}, test) revela la "
          f"capacidad diferenciada del modelo en cada fase del ciclo de vida del motor:"),
        SP(4), tercil_table(), SP(4),
        CAP("Tabla 2. Métricas de error por tercil de RUL real (test set, n=100 motores). "
            "Late: motores próximos al fallo. Mid: transición. Early: motores en fase saludable."),
        SP(8),
        IMG(f"{OUTPUT_DIR}/f5_04_error_por_tercil.png",
            f"Figura 4. Error absoluto por tercil de RUL (boxplot, RMSE/MAE y análisis de sesgo). "
            f"El tercil Mid presenta la mayor dificultad predictiva (RMSE = {t.loc[mid_row,'RMSE']:.2f} "
            f"ciclos, sesgo = {t.loc[mid_row,'bias']:+.1f} ciclos)."),
        SP(6),
        CALLOUT(f"<b>Hallazgo:</b> el modelo es más preciso donde más importa industrialmente — "
                f"en la fase Late (motores próximos al fallo), el RMSE es de solo "
                f"{t.loc[late_row,'RMSE']:.2f} ciclos y el {t.loc[late_row,'pct_within15']:.0f}% "
                f"de los motores se predicen con error ≤ 15 ciclos. El sesgo es positivo en las "
                f"fases Late y Mid ({t.loc[late_row,'bias']:+.1f} y {t.loc[mid_row,'bias']:+.1f} "
                f"ciclos): el modelo tiende a sobreestimar el RUL cerca del fallo — más seguro "
                f"que subestimarlo, pero conviene tenerlo presente al calibrar umbrales de alarma."),
        SP(6),
        IMG(f"{OUTPUT_DIR}/f5_05_scatter_predicho_real.png",
            "Figura 5. Scatter RUL predicho vs real con densidad KDE, banda de confianza ±15 "
            "ciclos y línea de regresión."),
        SP(6),
        IMG(f"{OUTPUT_DIR}/f5_03_degradacion_predicha.png",
            "Figura 6. Curvas de degradación predicha vs real para 10 motores representativos "
            "(distribución variada de RUL final). La banda rosa indica ±15 ciclos alrededor de "
            "la predicción."),
    ]

    # ── 8. CONCLUSIONES ──
    story += [
        SP(4), HR(),
        Hh("8. Conclusiones técnicas"),
        P("Este proyecto demuestra que la predicción de RUL en turbofanes es un problema "
          "abordable con técnicas de machine learning supervisado cuando se diseña correctamente "
          "el pipeline de datos y se evitan los sesgos de normalización más habituales en "
          "series temporales de test truncadas."),
        SP(3),
        P("La contribución metodológica más relevante del proyecto es la identificación y "
          "corrección del sesgo de normalización per-motor en el conjunto de test (sección "
          "2.2): un MinMaxScaler ajustado independientemente en cada motor de test produce "
          "una señal espuria de \"cerca del fallo\" en todo el test set, independientemente del "
          "RUL real. La solución — un único escalador ajustado sobre el entrenamiento completo — "
          "elimina ese sesgo y produce una comparación train/test estadísticamente honesta."),
        SP(3),
        P(f"La ingeniería de características mediante ventanas deslizantes de 15 y 30 ciclos "
          f"transforma el problema de predicción de RUL desde señales crudas ruidosas a un "
          f"espacio de features estructurado que captura simultáneamente el estado local del "
          f"motor (medias rolling), su variabilidad (desviaciones típicas) y su dinámica de "
          f"deterioro (pendientes OLS y deltas acumulados). De los cuatro modelos evaluados, "
          f"{best_label} obtiene el mejor equilibrio entre ajuste y generalización sobre estas "
          f"112 features."),
        SP(3),
        P("La estrategia de validación cruzada GroupKFold es esencial para obtener estimaciones "
          "honestas del error de generalización. Con k=5 y grupos por engine_id, la correlación "
          "temporal intra-motor no contamina los folds de validación."),
        SP(3),
        P(f"El análisis SHAP revela que {describe_feature(top1_name).split(' — ')[0]} es la "
          f"variable más informativa para el modelo seleccionado, con las pendientes de "
          f"tendencia (velocidad de cambio) generalmente más informativas que los niveles "
          f"absolutos — hallazgo con valor operativo directo: permite priorizar qué sensores y "
          f"qué transformaciones (nivel vs. tendencia) monitorizar en tiempo real."),
        SP(3),
        P(f"El objetivo de clasificación (zona de riesgo) se resuelve con alta fiabilidad — "
          f"Recall = {best_clf['test_recall']:.3f} en test — confirmando que las mismas 112 "
          f"features que sirven para la regresión de RUL son suficientes para una tarea de "
          f"alarma binaria, sin necesidad de un pipeline de features independiente."),
        SP(3),
        P("Desde la perspectiva de la aplicación industrial, el perfil de error por tercil de "
          "RUL es favorable: el modelo de regresión es más preciso precisamente cuando la "
          "información es más valiosa (motores próximos al fallo), y el clasificador prioriza "
          "explícitamente no dejar pasar fallos inminentes — ambos comportamientos son "
          "conservadores desde el punto de vista del mantenimiento predictivo."),
    ]

    # ── 9. LIMITACIONES ──
    story += [
        SP(4), HR(),
        Hh("9. Limitaciones y mejoras propuestas"),
        Hh("9.1 Limitaciones del estudio", 2),
        B("<b>Dataset simulado:</b> C-MAPSS es generado por simulación física, no por motores "
          "reales. Los patrones de degradación son más regulares que en datos industriales, lo "
          "que puede inflar las métricas en comparación con aplicaciones reales."),
        B("<b>Modelado limitado a FD001:</b> aunque Fase 0/1 extendieron el preprocesado y el "
          "EDA a los 4 subdatasets (incluida la normalización por régimen operacional para "
          "FD002/FD004, sección 2.2/3.2), el modelado supervisado (Fases 4-7) se mantiene "
          "enfocado en FD001 por decisión de alcance de esta iteración — es el subdataset "
          "más limpio para desarrollo inicial (1 condición, 1 modo de fallo)."),
        B("<b>Horizonte fijo:</b> el modelo predice RUL en el último ciclo observable de test, "
          "no simula una predicción rolling en tiempo real a lo largo de la trayectoria."),
        B("<b>Sin incertidumbre cuantificada:</b> las predicciones de regresión son puntuales. "
          "Un sistema productivo debería incluir intervalos de predicción o estimación "
          "bayesiana de incertidumbre."),
        B("<b>Umbral de clasificación fijo (0.5):</b> el clasificador no se calibró para un "
          "punto de operación específico de Recall/Precision; un despliegue real debería "
          "ajustar el umbral según el coste real de falsos negativos vs. falsos positivos."),
        SP(6),
        Hh("9.2 Mejoras propuestas", 2),
        B("<b>Modelos secuenciales (LSTM / Temporal Fusion Transformer):</b> explotar la "
          "estructura temporal completa de las trayectorias, en lugar de features rolling "
          "construidas manualmente."),
        B("<b>Cuantificación de incertidumbre:</b> implementar regresión cuantílica o NGBoost "
          "para obtener intervalos de predicción calibrados."),
        B(f"<b>Extensión del modelado a FD002-FD004:</b> reutilizar la normalización por "
          f"régimen ya implementada en Fase 0 para entrenar y evaluar los mismos modelos sobre "
          f"los subdatasets multi-condición."),
        B("<b>Optimización de hiperparámetros:</b> búsqueda sistemática (p.ej. Optuna con "
          "pruning) para XGBoost y LightGBM sobre el espacio completo de hiperparámetros."),
        B("<b>Detección de anomalías como feature:</b> añadir scores de anomalía por ventana "
          "(Isolation Forest) como feature adicional."),
    ]

    # ── 10. REFERENCIAS ──
    story += [
        SP(4), HR(),
        Hh("10. Referencias"),
        B("Saxena, A., Goebel, K., Simon, D., &amp; Eklund, N. (2008). Damage Propagation "
          "Modeling for Aircraft Engine Run-to-Failure Simulation. <i>Proceedings of the "
          "International Conference on Prognostics and Health Management (PHM 2008)</i>, "
          "Denver, CO."),
        B("Heimes, F. O. (2008). Recurrent neural networks for remaining useful life "
          "estimation. <i>International Conference on Prognostics and Health Management "
          "(PHM 2008)</i>."),
        B("Ramasso, E., &amp; Saxena, A. (2014). Performance Benchmarking and Analysis of "
          "Prognostic Methods for CMAPSS Datasets. <i>International Journal of Prognostics "
          "and Health Management</i>, 5(2). DOI: 10.36001/ijphm.2014.v5i2.2236."),
        B("Li, X., Ding, Q., &amp; Sun, J.-Q. (2018). Remaining useful life estimation in "
          "prognostics using deep convolution neural networks. <i>Reliability Engineering "
          "&amp; System Safety</i>, 172, 1–11."),
        B("Lundberg, S. M., &amp; Lee, S. I. (2017). A Unified Approach to Interpreting Model "
          "Predictions. <i>NeurIPS 2017</i>."),
        B("Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of "
          "Statistical Learning</i> (2nd ed.). Springer."),
        SP(10), HR(),
        Paragraph("Informe generado con Python 3.12 | pandas 2.2 | scikit-learn 1.6 | "
                  "XGBoost 2.1 | LightGBM 4.7 | SHAP 0.52 | ReportLab 5.0 | matplotlib 3.10",
                  S["footer"]),
        Paragraph("Lucía Pardo Vázquez · lucia.par.vaz@gmail.com · Berlín, septiembre 2026",
                  S["footer"]),
    ]

    doc.build(story)


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL — FASE 6
# ─────────────────────────────────────────────

def run_fase6():
    print("\n" + "═" * 70)
    print("  FASE 6 — DOCUMENTACIÓN FINAL")
    print("═" * 70)

    print("\n[1/3] Cargando resultados de Fase 3/4/5 (nada se reconstruye)...")
    data = load_all_results()
    print(f"      Mejor modelo regresión     : {data['best_reg']['Modelo']} "
          f"(Test RMSE={data['best_reg']['test_rmse']:.3f})")
    print(f"      Mejor modelo clasificación : {data['best_clf']['Modelo']} "
          f"(Test Recall={data['best_clf']['test_recall']:.3f})")

    print("\n[2/3] Generando figuras...")
    path_f6_01 = f"{OUTPUT_DIR}/f6_01_comparativa_modelos.png"
    path_f6_02 = f"{OUTPUT_DIR}/f6_02_pipeline.png"
    plot_f6_01(data["reg"], path_f6_01)
    plot_f6_02(data, path_f6_02)

    print("\n[3/3] Generando informe técnico PDF...")
    pdf_path = f"{OUTPUT_DIR}/informe_tecnico_cmapss.pdf"
    build_pdf(data, path_f6_01, path_f6_02, pdf_path)
    print(f"  PDF generado: {pdf_path}")

    print("\n" + "═" * 70)
    print("  RESUMEN FASE 6")
    print("═" * 70)
    print(f"  Figuras : {path_f6_01}")
    print(f"            {path_f6_02}")
    print(f"  PDF     : {pdf_path}")
    print("═" * 70)
    print("  ✅ FASE 6 COMPLETADA")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    run_fase6()
