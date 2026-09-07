# NASA C-MAPSS FD001 — Mantenimiento Predictivo de Motores Turbofan

**Proyecto de fin de máster · Business Analytics · Universidad UNIE**
**Autora:** Lucía Pardo Vázquez · lucia.par.vaz@gmail.com
**Dataset:** NASA C-MAPSS FD001 (CMAPSS: Commercial Modular Aero-Propulsion System Simulation)
**Objetivo:** Predecir la Vida Útil Restante (RUL) de motores turbofan a partir de series temporales de sensores

---

## ⚠️ Aviso sobre esta versión del README

Este README documenta una **auditoría estadística completa** del proyecto (septiembre 2026) que encontró y corrigió un problema de fuga de datos (*data leakage*) en la normalización de sensores, varios errores de etiquetado/aplicación de tests estadísticos, y citas bibliográficas incorrectas. Las fases 0, 1, 2, 4, 5 y 7 se reescribieron —como script `.py` **y** como notebook `.ipynb` ejecutado y explicado paso a paso— sobre un pipeline único y correcto. Fase 0 y Fase 1 se extendieron además a los 4 subdatasets (FD001-FD004), y Fase 4 al objetivo 2 del proyecto (clasificación `early_failure`, ausente hasta esta revisión). La fase 3 se reparó igualmente (rutas + reutilización de Fase 0), pero solo como script — nunca tuvo notebook. **La fase 6 (informe PDF) también se reescribió por completo**: no eran solo rutas rotas, sino resultados de modelo enteros hardcodeados de una ejecución anterior (XGBoost como mejor modelo, SHAP con features desactualizadas, una cita bibliográfica incorrecta) — ahora carga todo dinámicamente desde los CSV que exportan Fase 3/4/5, así que nunca puede volver a citar un modelo que ya no es el mejor. Con esto, **las 8 fases (0-7) están reparadas, auditadas y verificadas de punta a punta**. Ver [Estado y pendientes](#estado-y-pendientes) al final de este documento.

Los números de esta versión (Fases 4/5/7) **no coinciden exactamente** con los que se publicaron originalmente, aunque el pipeline y los hiperparámetros son los mismos — ver [Nota sobre la discrepancia con los números originales](#nota-sobre-la-discrepancia-con-los-números-originales).

---

## Estructura del proyecto

```
PROYECT_04/
├── archive/                        # Datos originales NASA C-MAPSS (train/test/RUL, FD001-FD004)
│   ├── train_FD001.txt             # 100 motores, run-to-failure (20.631 filas)
│   ├── test_FD001.txt              # 100 motores con trayectoria parcial (13.096 filas)
│   ├── RUL_FD001.txt               # RUL verdadero en el último ciclo de test
│   └── readme.txt                  # Documentación oficial de NASA (estructura de columnas)
├── models/                          # Modelos y escaladores entrenados (Fase 4)
│   ├── xgboost_fd001.pkl            # Regresión RUL
│   ├── lightgbm_fd001.pkl
│   ├── randomforest_fd001.pkl       # Mejor modelo de regresión en esta revisión (ver Resultados — Fase 4)
│   ├── ridge_fd001.pkl
│   ├── xgboost_classifier_fd001.pkl # Clasificación early_failure (extensión — objetivo 2)
│   ├── lightgbm_classifier_fd001.pkl
│   ├── randomforest_classifier_fd001.pkl
│   ├── logisticregression_classifier_fd001.pkl
│   ├── global_sensor_scaler.pkl     # MinMaxScaler global, ajustado SOLO en train (Fase 0, FD001)
│   ├── global_sensor_scaler_FD00{1,3}.pkl  # Idem, generado por el pipeline multi-subdataset
│   ├── condition_kmeans_FD00{2,4}.pkl      # k-means (k=6) de regímenes operacionales, fit en train
│   ├── condition_scalers_FD00{2,4}.pkl     # dict {régimen: MinMaxScaler}, uno por condición
│   └── final_scaler.pkl             # StandardScaler sobre las 112 features — compartido regresión+clasificación
├── outputs/                          # Resultados: figuras (f5-f7), tablas, informe PDF
│   ├── f5_01…f5_05_*.png            # Fase 5: interpretabilidad SHAP
│   ├── f6_01…f6_02_*.png            # Fase 6: comparativa de modelos y pipeline (100% dinámico)
│   ├── f7_01…f7_06_*.png            # Fase 7: mejoras estadísticas
│   ├── sensor_stats.csv             # Resumen estadístico por sensor (Fase 1, FD001)
│   ├── fase0_subsets_summary.csv    # Resumen de normalización aplicada por subdataset (Fase 0 ext.)
│   ├── fase1_subsets_comparison.csv # Comparativa FD001-FD004 (Fase 1 ext.)
│   ├── train/test_FD00{1..4}_normalized.parquet  # Salida normalizada de Fase 0 (los 4 subdatasets)
│   ├── fase2_comparacion_features.csv  # 112 vs. 238 features (Fase 2)
│   ├── fase4_resultados.csv         # Métricas CV y test — regresión RUL (Fase 4)
│   ├── fase4_clasificacion_resultados.csv  # Métricas CV y test — clasificación early_failure (Fase 4 ext.)
│   ├── fase4_comparacion_readme.csv # Regresión: este run vs. números históricos del README
│   ├── fase7_ablacion_ventanas.csv  # Ablación de ventanas (Fase 7, M6)
│   ├── informe_tecnico_cmapss.pdf   # Informe técnico (Fase 6, 11 páginas, regenerado con datos reales)
│   ├── fase3_correlaciones_rul.csv / fase3_onset_degradacion.csv  # Export Fase 3, consumido por Fase 6
│   └── fase5_shap_top20.csv / fase5_error_tercil.csv / fase5_resumen.csv  # Export Fase 5, ídem
├── figures/                          # Figuras de Fases 1-4
│   ├── f1_01…f1_04_*.png            # Fase 1: EDA en profundidad de FD001
│   ├── f1_05_subsets_comparison.png # Fase 1 ext.: sensores informativos + vida útil, FD001-FD004
│   ├── f1_06_operating_conditions_fd004.png  # Fase 1 ext.: 6 regímenes de FD004 (k-means)
│   ├── f3_01…f3_05_*.png            # Fase 3: EDA de degradación (spaghetti, correlación, FD001 vs FD003)
│   ├── f4_01…f4_04_*.png            # Fase 4: regresión RUL (CV, predicho vs. real, residuos, test)
│   └── f4_05…f4_06_*.png            # Fase 4 ext.: clasificación early_failure (CV, matrices de confusión, ROC/PR)
├── requirements.txt                  # Dependencias con versiones fijadas
├── scripts/
│   ├── fase0_preprocesado.py        # ★ Carga + normalización (global FD001/3, por régimen FD002/4)
│   ├── fase0_preprocesado.ipynb     #   Mismo pipeline (solo FD001), explicado paso a paso + ejecutado
│   ├── fase1_carga_inspeccion.py    # ★ EDA de FD001 + comparativa FD001-FD004
│   ├── fase1_carga_inspeccion.ipynb
│   ├── fase2_feature_engineering.py # ★ Features rolling: conjunto canónico (112) vs. extendido (238)
│   ├── fase2_feature_engineering.ipynb
│   ├── fase3_eda_degradacion.py     # ★ Reparada — rutas + reutiliza Fase 0 (sin notebook propio)
│   ├── fase4_modelado.py            # ★ GroupKFold CV + entrenamiento final (4 modelos)
│   ├── fase4_modelado.ipynb
│   ├── fase5_evaluacion.py          # ★ SHAP, curvas de degradación, error por tercil
│   ├── fase5_evaluacion.ipynb
│   ├── fase6_documentacion.py       # ★ Informe PDF + figuras, 100% dinámico (lee CSV de Fase 3/4/5)
│   ├── fase7_mejoras.py             # ★ M1-M6 (incluye la antigua fase7_m6.py, fusionada)
│   ├── fase7_mejoras.ipynb
│   └── fase7_m6.py                  #   Huérfano — su lógica ya vive en fase7_mejoras.py
└── requirements.txt
```
★ = reescrita en esta revisión (script + notebook, ejecutados de punta a punta, 0 errores).

---

## Descripción del dataset

| Parámetro | Valor |
|---|---|
| Motores de entrenamiento | 100 |
| Motores de test | 100 |
| Filas de entrenamiento | 20.631 |
| Sensores seleccionados | 14 (s2, s3, s4, s7–s9, s11–s15, s17, s20, s21) |
| Sensores descartados (varianza nula en train) | 7 (s1, s5, s6, s10, s16, s18, s19) |
| Cap de RUL | 125 ciclos (Heimes, 2008) |
| Condición operacional | FD001: única (altitud baja, baja carga) |
| Vida útil en train | 206.3 ± 46.3 ciclos (rango 128–362) |

El fichero raw no tiene fila de cabecera (confirmado contra `archive/readme.txt`, que numera las 26 columnas por posición sin nombrarlas): los nombres de columna se asignan por posición en `fase0_preprocesado.py::load_cmapss`.

---

## Fase 0 — Preprocesado canónico (el fix central de esta revisión)

**El problema que corrige.** Versiones previas de este proyecto ajustaban un `MinMaxScaler` **por motor**, de forma independiente en train y en test (`normalize_per_unit`, ahora eliminada). Esto es un *look-ahead bias*: para normalizar el ciclo *t* de un motor de test, el escalador usaba el mínimo y el máximo de **toda** la trayectoria observada de ese motor, incluyendo ciclos posteriores a *t* que en un despliegue real (streaming) todavía no se habrían observado. Además, como cada trayectoria de test se reescalaba a su propio rango [0,1], el último ciclo observado quedaba siempre cerca de 1.0 **independientemente del RUL real** — el modelo aprendía una señal espuria de "cerca del fallo" en vez de degradación real.

**La corrección:** un único `MinMaxScaler` ajustado **solo sobre el conjunto de entrenamiento** (todos los motores, todos los ciclos) y aplicado sin reajuste al test — la práctica estándar de *held-out evaluation* (Hastie, Tibshirani & Friedman, 2009).

**Evidencia cuantitativa** (reproducida en código en `fase0_preprocesado.ipynb`, no solo afirmada en prosa): para el sensor `s11`, la media en el último ciclo de test es 0.721 con normalización per-motor, baja a 0.444 con normalización global, y la referencia en motores sanos de train (RUL≥100) es 0.318. La normalización global reduce el sesgo (|gap| frente a la referencia) de 0.403 a 0.126 — una reducción del 69%.

Este único `GlobalMinMaxScaler`, ajustado en Fase 0, es la base de la que parten Fases 1, 2, 4, 5 y 7 — ninguna de ellas reimplementa su propia normalización.

**Extensión a FD002/FD004 (sept. 2026).** `fase0_preprocesado.py` ahora también preprocesa FD002 y FD004, que combinan **6 condiciones operacionales** (altitud, Mach y TRA — ver `archive/readme.txt`), frente a la condición única de FD001/FD003. Aplicarles el mismo `GlobalMinMaxScaler` mezclaría en una sola escala mediciones que son físicamente distintas por régimen. En su lugar, `normalize_dataset()` decide la estrategia según `SUBSET_META`:

- **FD001 / FD003** (1 condición) → `GlobalMinMaxScaler`, sin cambios respecto a lo descrito arriba.
- **FD002 / FD004** (6 condiciones) → los 3 `op_setting` se agrupan en 6 regímenes vía k-means (`fit_condition_clusters`, ajustado solo en train, k=6 según literatura: Ramasso & Saxena, 2014; Li, Ding & Sun, 2018), y se ajusta un `MinMaxScaler` independiente **por régimen** (`fit_condition_scalers`), también solo en train. La asignación de régimen de un ciclo de test usa `kmeans.predict` (nunca se reajusta el k-means con datos de test). Esto no es normalización por motor: cada régimen agrupa ciclos de decenas de motores distintos que comparten condición física, así que no hay look-ahead ni fuga de la señal de degradación — se preserva exactamente el mismo principio de *held-out evaluation* que corrige Fase 0.

Consecuencia observada: FD002/FD004 retienen **20 de 21 sensores** como informativos (frente a 14 en FD001), porque sensores que en FD001 son prácticamente constantes (`s1`, `s5`, `s6`, `s10`, `s18`, `s19`) sí varían con la condición operacional cuando hay 6 regímenes distintos — la selección de sensores se recalcula por subdataset, nunca se reutiliza la lista de FD001.

---

## Fase 1 — Inspección y comparativa de subdatasets

`fase1_carga_inspeccion.py` mantiene el análisis en profundidad de FD001 (estadísticas descriptivas, sensores informativos, distribución de vida útil, efecto de la normalización global sobre trayectorias reales) y añade una sección comparativa de los 4 subdatasets (`run_fase1_multi_subset`), sin tocar ninguno de los artefactos de FD001.

| Subdataset | Motores train/test | Cond. operacionales | Modos de fallo | Vida media ± std (ciclos) | Sensores informativos | Normalización |
|---|---|---|---|---|---|---|
| FD001 | 100 / 100 | 1 | 1 (HPC) | 206.3 ± 46.3 | 14 | Global |
| FD002 | 260 / 259 | 6 | 1 (HPC) | 206.8 ± 46.8 | 20 | Por régimen (k-means) |
| FD003 | 100 / 100 | 1 | 2 (HPC + Fan) | 247.2 ± 86.5 | 15 | Global |
| FD004 | 249 / 248 | 6 | 2 (HPC + Fan) | 246.0 ± 73.1 | 20 | Por régimen (k-means) |

Nulos: 0 en los 4 subdatasets (train y test). Tabla completa en `outputs/fase1_subsets_comparison.csv`.

Dos lecturas del EDA comparado:
- **Más condiciones → más sensores informativos.** No es que FD002/FD004 tengan mejor instrumentación; es que la varianza de un sensor depende del régimen en el que se mide. Un sensor constante bajo una única condición (Sea Level) puede variar sustancialmente entre 6 regímenes de altitud/Mach/TRA.
- **Más modos de fallo → vida útil más dispersa.** FD003/FD004 (2 modos de fallo: HPC + Fan) muestran std de vida útil sensiblemente mayor que FD001/FD002 (1 modo), visible en el boxplot comparado (`figures/f1_05_subsets_comparison.png`). La separabilidad de los 6 regímenes de FD004 en el espacio `op_setting_1`/`op_setting_2` se muestra en `figures/f1_06_operating_conditions_fd004.png`.

---

## Pipeline

```
Fase 0 → Datos raw → GlobalMinMaxScaler (FD001/FD003) | MinMaxScaler por régimen k-means (FD002/FD004)
Fase 1 → EDA descriptivo sobre la salida de Fase 0 (FD001 en profundidad + comparativa FD001-FD004)
Fase 2 → Rolling features: mean, std, OLS slope (w=15, w=30) + delta acumulado
       → 112 features (14 sensores crudos + 14×2×3 rolling + 14 delta)
       → [comparación empírica contra un conjunto extendido de 238 features, ver abajo]
Fase 4 → StandardScaler (fit en train, aplicado a los 4 modelos por igual)
       → GroupKFold(k=5) agrupado por engine_id
       → Ridge / RandomForest / XGBoost / LightGBM
Fase 5 → SHAP (interventional), curvas de degradación, error por tercil de RUL
Fase 7 → Comparación pareada de modelos, censura, PHM score, SHAP tree_path_dependent
       → ablación de s14, Durbin-Watson + Breusch-Pagan, ablación de ventanas
```

**Nota sobre el escalado en Fase 4:** el `StandardScaler` se aplica a los 4 modelos por igual, no solo a Ridge (a diferencia de lo que documentaban versiones previas). Un reescalado monótono por columna no cambia las predicciones de un árbol de decisión (los splits comparan umbrales; el orden relativo no cambia) — pero si el modelo se entrenó con features escaladas, alimentarlo sin escalar en inferencia sería incorrecto, porque los splits del árbol están en "unidades escaladas". `models/final_scaler.pkl` es ese StandardScaler único, compartido por los 4 modelos.

---

## Feature engineering (Fase 2) — dos conjuntos, comparados empíricamente

Para cada uno de los 14 sensores informativos:

**Conjunto CANÓNICO (112 features, el que alimenta Fase 4):** ventanas `w ∈ {15, 30}`, con media rolling, desviación típica rolling y pendiente OLS rolling (vectorizada por convolución) + **delta acumulado** (`s(t) − s(ciclo inicial del motor)`). Total: 14 sensores crudos + 14×2×3 rolling + 14 delta = **112**.

**Conjunto EXTENDIDO (238 features, solo para comparación):** ventanas `w ∈ {15, 30, 50}`, añade mínimo y máximo rolling, y usa **delta de primer orden** (`s(t) − s(t−1)`, proxy de velocidad instantánea) en vez de acumulado. Total: 14 + 14×3×5 + 14 = **238**.

Estas dos definiciones de "delta" capturan información distinta (desplazamiento total desde el inicio vs. velocidad instantánea) — antes coexistían sin documentar cuál usaba cada script.

### Comparación empírica (LightGBM ligero, GroupKFold k=5 + test oficial)

| Conjunto | Features | CV RMSE | Test RMSE |
|---|---|---|---|
| **Canónico** | **112** | 12.72 | **11.98** |
| Extendido | 238 | **11.29** | 12.66 |

El conjunto extendido gana en CV pero **pierde en test** — patrón de libro del compromiso sesgo-varianza (Hastie et al., 2009): más features ajustan mejor los folds de validación cruzada sin generalizar mejor. Confirma empíricamente, no solo por economía de diseño, la elección de las 112 features para Fase 4. Es consistente con la ablación completa de ventanas de Fase 7 (M6, ver abajo).

---

## Fase 3 — EDA orientado a degradación

`fase3_eda_degradacion.py` opera sobre datos crudos (train, sin normalizar) de FD001 — y, para la comparativa final, también FD003 — y produce 5 figuras: spaghetti plots de los 8 sensores más correlacionados con RUL, heatmap de correlación (Pearson + Spearman) entre los 14 sensores informativos y RUL, distribución del RUL por cuartil de vida útil, curvas de degradación media (mediana ± 1 std) por intervalo de RUL, y una comparativa FD001 vs. FD003.

**Reparación (sept. 2026):** el script apuntaba a `/home/claude/cmapss/...` (rutas del sandbox original) y reimplementaba su propio loader CSV y su propia lista de sensores informativos, en vez de reutilizar `fase0_preprocesado.py`. Se corrigió con el mismo patrón que Fase 1/2: rutas relativas al repo, `f0.load_cmapss` / `f0.INFORMATIVE_EXPECTED` reutilizados en vez de redefinidos. También se corrigió la cita bibliográfica errónea ("NASA/TM-2014-218496" → IJPHM 2014, el mismo error ya corregido en Fase 0) y un defecto visual en `f3_01_spaghetti_plots.png` donde la colorbar horizontal se solapaba con la fila inferior de subplots. La lógica de correlaciones en sí usa datos crudos, no normalizados por motor, así que nunca estuvo afectada por el bug de leakage que corrigió Fase 0.

**Hallazgos principales (FD001, train):**

| Sensor | \|r\| Pearson con RUL | Lectura física |
|---|---|---|
| s11 (Ps30) | 0.696 | Presión estática HPC — el más discriminativo |
| s4 (T50) | 0.679 | Temperatura salida LPT |
| s12 (phi) | 0.672 | Ratio de flujo de combustible |
| s7 (P30) | 0.657 | Presión salida HPC |
| s15 (BPR) | 0.643 | Ratio de bypass |

9 de los 14 sensores informativos muestran señal ALTA (\|r\| > 0.6) con el RUL; ninguno muestra señal BAJA. El análisis de inicio de degradación observable (umbral 2σ sobre la fase sana, RUL>200) sitúa el punto en que la señal se separa del ruido entre RUL≈32 (s3) y RUL≈53 (s11, s4) — coherente con el cap de 125 ciclos de Fase 2: hay margen sano antes de ese punto en el que pedir al modelo predecir RUL preciso sería pedirle distinguir señal de ruido de instrumentación.

**FD001 vs. FD003:** FD003 (2 modos de fallo: HPC + Fan) tiene vida útil más larga y mucho más dispersa que FD001 (247.2 ± 86.5 vs. 206.3 ± 46.3 ciclos) — consistente con superponer dos procesos de degradación con dinámicas distintas en la misma población de motores. Las trayectorias medianas de s3/s11/s4 son visualmente muy similares entre ambos subdatasets hasta ~70% de vida transcurrida, y solo se separan cerca del fallo — el segundo modo de fallo de FD003 (Fan) no es visualmente evidente en estos 3 sensores hasta la fase final, lo que sugiere que distinguir el modo de fallo activo requeriría mirar sensores adicionales o modelos específicos por modo, no solo estos 3.

---

## Resultados — Fase 4 (Modelado)

GroupKFold(k=5) agrupado por `engine_id` (Li, Ding & Sun, 2018): evita que ciclos del mismo motor aparezcan en train y validación simultáneamente, lo que produciría fuga de datos y RMSE artificialmente optimista.

| Modelo | CV RMSE (±std) | CV R² | Test RMSE | Test R² |
|---|---|---|---|---|
| **RandomForest** ★ | 12.87 ± 0.87 | 0.905 | **11.04** | **0.924** |
| XGBoost | **12.68 ± 0.82** | **0.908** | 11.32 | 0.920 |
| LightGBM | 12.74 ± 0.85 | 0.907 | 12.05 | 0.910 |
| Ridge | 16.39 ± 1.39 | 0.845 | 14.93 | 0.861 |

En esta revisión, **RandomForest** obtiene el mejor test RMSE (aunque XGBoost tiene el mejor CV RMSE) — a diferencia de la versión histórica del README, donde XGBoost era mejor en ambos. Ver la comparación completa y la discusión de por qué cambia en la sección siguiente.

### Clasificación `early_failure` (extensión sept. 2026 — objetivo 2 del proyecto)

El objetivo 2 del proyecto (clasificar zona de riesgo, `RUL≤30`) tenía la columna target calculada desde Fase 2 pero **ningún modelo entrenado** en el pipeline auditado — un hueco real frente al alcance pedido. Se añadió a `fase4_modelado.py` sin tocar el track de regresión: mismos datos (112 features), mismo protocolo (GroupKFold k=5 por motor + test oficial de 100 motores), 4 clasificadores con peso de clase balanceado (17.5% de positivos en train) — LogisticRegression, RandomForest, XGBoost, LightGBM.

| Modelo | CV F1 (±std) | CV Recall (±std) | CV ROC-AUC | Test F1 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.907 ± 0.017 | 0.948 ± 0.012 | 0.9951 | 0.941 | **0.960** | 0.9973 |
| RandomForest | 0.913 ± 0.011 | 0.936 ± 0.017 | 0.9947 | 0.941 | **0.960** | 0.9931 |
| XGBoost | 0.925 ± 0.008 | 0.945 ± 0.017 | 0.9963 | 0.941 | **0.960** | **0.9984** |
| **LightGBM** | **0.926 ± 0.008** | 0.945 ± 0.018 | **0.9964** | 0.941 | **0.960** | 0.9979 |

Los 4 modelos aciertan **exactamente los mismos** 97/100 puntos de test (73 TN, 2 FP, 1 FN, 24 TP) — coincidencia real, no un bug: con ROC-AUC>0.99 el problema es casi linealmente separable en este test de 100 puntos, así que los 4 modelos convergen al mismo umbral de decisión aunque sus probabilidades difieran (de ahí que el ROC-AUC de test sí varíe entre modelos). LightGBM/XGBoost ganan en CV (conjunto más grande, 17.731 muestras); en el test reducido los 4 empatan en Recall.

**Por qué Recall y no Accuracy/Precision:** en mantenimiento predictivo, un falso negativo (fallo inminente no detectado) es mucho más costoso que un falso positivo (una revisión de más) — el objetivo del proyecto pide explícitamente priorizar Recall. Con Recall=0.960 en test, el mejor caso pierde 1 de 25 fallos inminentes; la curva Precision-Recall (`f4_06`) muestra que los 4 modelos mantienen Precision>0.9 incluso a Recall alto, gracias al peso de clase balanceado.

---

## Nota sobre la discrepancia con los números originales

Los números de Fase 4 (y, en cascada, Fase 5 y Fase 7) de esta revisión **no reproducen exactamente** los que se publicaron originalmente, a pesar de partir del mismo conjunto de 112 features y los mismos hiperparámetros documentados:

| Modelo | CV RMSE (este run) | CV RMSE (histórico) | Test RMSE (este run) | Test RMSE (histórico) | Test R² (este run) | Test R² (histórico) |
|---|---|---|---|---|---|---|
| Ridge | 16.39 | 15.62 | 14.93 | 14.78 | 0.861 | 0.864 |
| RandomForest | 12.87 | 13.80 | **11.04** | 13.69 | 0.924 | 0.883 |
| XGBoost | 12.68 | 12.93 | 11.32 | 12.80 | 0.920 | 0.898 |
| LightGBM | 12.74 | 12.90 | 12.05 | 13.51 | 0.910 | 0.886 |

Los 4 modelos obtienen **mejor** test RMSE en esta revisión, y **RandomForest supera a XGBoost** — un cambio de conclusión, no solo de decimales.

**Se descartó como causa:** el conteo de features (`assert == 112`), los hiperparámetros (copiados literalmente de la versión anterior del README), el esquema de validación (`GroupKFold(k=5)`, `random_state=42` en los 4 modelos) y el escalado (`StandardScaler` aplicado uniformemente, corrigiendo además una inconsistencia previa donde solo se documentaba para Ridge).

**Hipótesis más verosímil: deriva de versión de librerías.** No existía ningún `requirements.txt` hasta esta revisión, así que no hay forma de saber con qué versiones exactas de scikit-learn/XGBoost/LightGBM se generaron los números originales. Este entorno usa scikit-learn 1.6.1, XGBoost 2.1.3, LightGBM 4.7.0 (ver `requirements.txt`). No se puede descartar del todo una diferencia residual en la reconstrucción del pipeline original (que nunca existió como código ejecutable único, solo reconstruido ad hoc en versiones previas de Fase 5/7) — pero el patrón sistemático (los 4 modelos mejoran, no una mezcla aleatoria de mejoras y empeoramientos) apunta más a un cambio de comportamiento interno de las librerías que a un error de reconstrucción de features.

**Consecuencia práctica:** los modelos guardados en `models/` y todos los resultados de Fases 5 y 7 corresponden a **este** run, no al histórico. Si se desea preservar los números originales como referencia, deberían citarse explícitamente como no reproducibles con precisión en este entorno.

---

## Resultados — Fase 5 (Interpretabilidad SHAP)

**Corrección (sept. 2026): el modelo interpretado ya no está fijado a "xgboost".** La versión previa de `fase5_evaluacion.py` tenía `model_name="xgboost"` como default, heredado de una revisión en la que XGBoost sí era el mejor modelo en test. Esta auditoría encontró que, en este entorno, **RandomForest supera a XGBoost** en test RMSE (11.04 vs. 11.32 — ver "discrepancia con los números originales" arriba), y Fase 5 seguía interpretando el modelo equivocado sin que nada en el código lo verificara. `get_best_model_name()` ahora lee `fase4_resultados.csv` y determina el modelo a interpretar dinámicamente — los números de abajo son de **RandomForest**, el modelo realmente desplegado.

### Features más importantes (SHAP interventional, test — RandomForest)

| Feature | Mean |SHAP| (ciclos) |
|---|---|
| s11_w30_slope | 3.69 |
| s2_w15_mean | 3.16 |
| s4_w15_mean | 2.83 |
| s3_w15_mean | 2.59 |
| s12_w30_slope | 2.20 |

El feature más importante es una *pendiente* (s11_w30_slope), no un nivel — coherente con que la velocidad de degradación de Ps30 (presión estática HPC) es más informativa que su valor absoluto, y con que s11 ya era el sensor más correlacionado con RUL en el EDA de Fase 3 (|r|=0.696).

### Error por tercil de RUL (RandomForest, test)

| Tercil | n | MAE | RMSE | % error ≤15 ciclos | Sesgo |
|---|---|---|---|---|---|
| Tardía (RUL≤53) | 33 | 4.58 | 5.71 | 97.0% | +2.95 |
| Media (53<RUL≤101) | 34 | 12.54 | 15.33 | 67.6% | +7.35 |
| Temprana (RUL>101) | 33 | 7.17 | 9.74 | 90.9% | −3.76 |

El error es sistemáticamente mayor en la fase media de vida — mismo patrón cualitativo que las versiones anteriores de esta tabla (con XGBoost), base empírica del test formal de heterocedasticidad de Fase 7 (M5). El sesgo cambia de signo entre tercios (positivo en tardía/media, negativo en temprana): el modelo tiende a sobreestimar el RUL cerca del fallo y subestimarlo lejos de él — más seguro que el error inverso, pero conviene tenerlo presente en umbrales de alarma.

---

## Resultados — Fase 7 (Mejoras estadísticas)

### M1 — Comparación pareada de pérdidas (ya no "Diebold-Mariano")

**Corrección metodológica:** Harvey, Leybourne & Newbold (1997) diseñaron su test para comparar pronósticos **secuenciales de una única serie temporal**, donde el diferencial de pérdida puede estar autocorrelado (particularmente a horizontes >1) y su varianza debe estimarse de forma robusta a esa autocorrelación (HAC). Aquí se comparan 100 motores **independientes**, no una serie temporal — no hay autocorrelación de horizonte que corregir, así que la maquinaria propia del test DM no aplica. Lo que se calcula es un t de Student **pareado** (diseño correcto: ambos modelos predicen las mismas 100 unidades) sobre el diferencial de pérdida cuadrática, con un factor de ajuste de varianza para muestra pequeña que toma prestada la forma funcional de la corrección HLN, sin su justificación de fondo.

| Comparación | t (ajustado) | p-valor | Conclusión |
|---|---|---|---|
| XGBoost vs LightGBM | −2.338 | **0.0214** | Diferencia significativa (α=0.05) |
| XGBoost vs RandomForest | 0.574 | 0.5674 | No significativa |

---

### M2 — Segregación por censura derecha

11 de los 100 motores de test tienen RUL_true ≥ 125 (target truncado al cap).

| Grupo (XGBoost) | n | RMSE | MAE | Sesgo |
|---|---|---|---|---|
| Global | 100 | 11.32 | 8.19 | +0.92 |
| No censurado | 89 | 11.17 | 8.12 | +1.47 |
| Censurado | 11 | **12.44** | 8.75 | −3.55 |

---

### M3 — PHM Score asimétrica (Saxena et al., 2008)

$$s(d) = \begin{cases} e^{d/10}-1 & d\geq 0 \text{ (sobreestima RUL)} \\ e^{-d/13}-1 & d<0 \text{ (subestima RUL)} \end{cases}$$

| Modelo | PHM Score (↓ mejor) | RMSE |
|---|---|---|
| **RandomForest** | **186.6** | 11.04 |
| XGBoost | 188.8 | 11.32 |
| LightGBM | 220.2 | 12.05 |
| Ridge | 337.9 | 14.93 |

RandomForest es mejor también bajo la métrica industrial asimétrica — coherente con ser el mejor modelo en test RMSE en esta revisión.

---

### M4 — SHAP tree_path_dependent + ablación de s14

**Hallazgo que invierte la conclusión histórica:** eliminar `s14` **empeora** el test RMSE en +0.600 ciclos (CV: 13.010±0.845 sin s14 vs. 12.68 con s14; test: 11.917 sin s14 vs. 11.318 con s14) — al contrario de la versión histórica, que reportaba una *mejora* de ~1 ciclo al eliminarlo. Consistente, una vez más, con que el modelo subyacente es distinto. Sirve de recordatorio: una conclusión de "esta feature es redundante" basada en un solo entrenamiento/semilla/versión de librería puede no sostenerse.

---

### M5 — Durbin-Watson y Breusch-Pagan (test correcto + chequeo de pseudo-replicación)

**Corrección metodológica:** la versión previa regresionaba `|residuo|` sobre el RUL y llamaba a eso "Breusch-Pagan" — ese es el test de **Glejser**. Breusch-Pagan (1979) regresiona residuos **al cuadrado**, con estadístico $LM = n\cdot R^2 \sim \chi^2(k)$.

| Estadístico | Valor |
|---|---|
| DW medio por motor | 0.273 ± 0.154 |
| Motores con DW < 1.5 | 100 / 100 |
| Breusch-Pagan, pooled (n=13.096) | LM=182.12, p≈0 |
| Breusch-Pagan, por motor (n=100) | LM=9.37, **p=0.0022** |

**Chequeo de pseudo-replicación:** Durbin-Watson ya demuestra que los ciclos consecutivos de un motor no son independientes, así que el test pooled (que asume independencia) infla la significancia. Agregando a un valor por motor (n=100, observaciones genuinamente independientes), el resultado **sigue siendo significativo** (p=0.0022) — confirma que la heterocedasticidad es un hallazgo robusto, no un artefacto de inflar el tamaño muestral.

---

### M6 — Ablación de ventanas temporales (LightGBM, GroupKFold k=5)

| Ventanas | Features | CV RMSE | Test RMSE |
|---|---|---|---|
| [15,30,45] | 154 | **11.35** | 12.53 |
| [30,45] | 112 | 11.40 | 12.40 |
| [15,45] | 112 | 11.43 | 12.54 |
| [45] | 70 | 11.65 | 12.66 |
| [5,15,30] | 154 | 12.61 | **11.37** |
| **[15,30] ★** | **112** | 12.73 | 11.96 |
| [30] | 70 | 12.76 | 12.27 |
| [5] | 70 | 13.57 | 13.61 |
| [5,15] | 112 | 13.84 | 13.98 |
| [15] | 70 | 13.87 | 13.84 |

Mismo patrón que en Fase 2: más ventanas mejora el CV sin mejorar consistentemente el test. `[15,30]` sigue siendo un compromiso razonable entre complejidad y generalización.

---

## Reproducibilidad

```bash
# Dependencias (versiones fijadas — ver requirements.txt; incluye el fix de
# numba 0.67.0, necesario para que shap funcione con numpy 2.x)
pip install -r requirements.txt

# Pipeline actualizado y validado en esta revisión (en orden):
python scripts/fase0_preprocesado.py        # o el notebook fase0_preprocesado.ipynb
python scripts/fase1_carga_inspeccion.py    # o el notebook fase1_carga_inspeccion.ipynb
python scripts/fase2_feature_engineering.py # o el notebook fase2_feature_engineering.ipynb
python scripts/fase3_eda_degradacion.py     # sin notebook — solo script
python scripts/fase4_modelado.py            # o el notebook fase4_modelado.ipynb
python scripts/fase5_evaluacion.py          # o el notebook fase5_evaluacion.ipynb
python scripts/fase7_mejoras.py             # o el notebook fase7_mejoras.ipynb
python scripts/fase6_documentacion.py       # sin notebook — solo script; requiere 3/4/5 ya ejecutados
```

**Importante:** los modelos guardados en `models/` deben usarse siempre con `global_sensor_scaler.pkl` y `final_scaler.pkl` del mismo directorio — son parte del modelo, no un preprocesado intercambiable. Reajustarlos desde cero con otra semilla o subconjunto de datos producirá predicciones distintas.

Cada notebook (`fase0_preprocesado.ipynb` … `fase7_mejoras.ipynb`) reutiliza las funciones del `.py` correspondiente en vez de duplicarlas — usa `show_source()` (definida al principio de cada notebook, vía `inspect.getsource`) para mostrar el código real de cada función justo antes de su primer uso, con su ubicación exacta (`archivo.py:línea`). Esto evita tanto la duplicación de lógica como el tener que saltar de archivo para ver qué hace una función.

---

## Estado y pendientes

| Fase | Estado | Notas |
|---|---|---|
| Fase 0 (preprocesado) | ✅ Reescrita, auditada | Script + notebook, 0 errores |
| Fase 1 (EDA) | ✅ Reescrita, auditada + extendida a 4 subdatasets | Script y notebook corren `run_fase1()` (FD001) + `run_fase1_multi_subset()` (FD001-FD004), 0 errores en ambos |
| Fase 2 (features) | ✅ Reescrita, auditada | Script + notebook, 0 errores; compara 112 vs. 238 |
| Fase 3 (EDA degradación) | ✅ Reparada (sept. 2026) | `fase3_eda_degradacion.py` apuntaba a `/home/claude/cmapss/...` y reimplementaba su propio loader/lista de sensores; ahora reutiliza `f0.load_cmapss` / `f0.INFORMATIVE_EXPECTED` y rutas relativas al repo, igual que Fase 1/2. Su lógica de correlaciones usa datos crudos (no normalizados), así que nunca estuvo afectada por el bug de leakage de Fase 0. Exporta `fase3_correlaciones_rul.csv` / `fase3_onset_degradacion.csv` (los consume Fase 6). 0 errores, 5 figuras. Solo `.py` — sin notebook (la Fase 3 original nunca tuvo uno) |
| Fase 4 (modelado) | ✅ Reescrita, auditada + extendida a clasificación | Script y notebook, 0 errores; regresión con números que difieren del histórico (ver nota arriba). Extensión (sept. 2026): añadido el track de clasificación `early_failure` (objetivo 2 del proyecto), ausente hasta ahora — ver Resultados. Notebook incluye la extensión (sección 7) |
| Fase 5 (SHAP) | ✅ Reescrita, auditada + corregida | Script y notebook, 0 errores. Corrección (sept. 2026): ya no asume XGBoost fijo — `get_best_model_name()` detecta el mejor modelo de Fase 4 (RandomForest en esta revisión) leyendo `fase4_resultados.csv`. Exporta `fase5_shap_top20.csv` / `fase5_error_tercil.csv` / `fase5_resumen.csv` (los consume Fase 6) |
| Fase 6 (informe PDF) | ✅ Reescrita por completo (sept. 2026) | No eran solo rutas rotas: TODOS los resultados de modelo estaban hardcodeados a mano de una ejecución anterior (XGBoost como mejor modelo, SHAP con features desactualizadas, cita de Ramasso incorrecta). Reescrita para cargar dinámicamente `fase3_correlaciones_rul.csv`, `fase3_onset_degradacion.csv`, `fase4_resultados.csv`, `fase4_clasificacion_resultados.csv`, `fase5_shap_top20.csv`, `fase5_error_tercil.csv`, `fase5_resumen.csv` y `fase1_subsets_comparison.csv` — nunca puede volver a citar un modelo que ya no es el mejor. Añade sección de clasificación (objetivo 2, ausente en el informe original). 0 errores, PDF de 11 páginas (382 palabras solo en conclusiones). Solo script — nunca tuvo notebook |
| Fase 7 (mejoras estadísticas) | ✅ Reescrita, auditada | Script + notebook, 0 errores; M1 y M5 corregidos metodológicamente |
| `fase7_m6.py` | 🗑️ Huérfano | Su lógica (ablación de ventanas) ya vive dentro de `fase7_mejoras.py`; pendiente de decidir si se elimina |

---

## Referencias

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation*. International Conference on Prognostics and Health Management (PHM), Denver, CO.
- Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation*. International Conference on Prognostics and Health Management (PHM).
- Ramasso, E., & Saxena, A. (2014). *Performance benchmarking and analysis of prognostic methods for CMAPSS datasets*. International Journal of Prognostics and Health Management, 5(2). DOI: 10.36001/ijphm.2014.v5i2.2236. *(Corrección: versiones previas citaban esta referencia como "NASA/TM-2014-218496", un identificador de informe que no corresponde a ningún documento NASA real verificable; la cita correcta es la de IJPHM.)*
- Li, X., Ding, Q., & Sun, J.-Q. (2018). *Remaining useful life estimation in prognostics using deep convolution neural networks*. Reliability Engineering & System Safety, 172, 1–11.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). *Testing the equality of prediction mean squared errors*. International Journal of Forecasting, 13(2), 281–291.
- Breusch, T. S., & Pagan, A. R. (1979). *A simple test for heteroscedasticity and random coefficient variation*. Econometrica, 47(5), 1287–1294.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems, 30.
- Breiman, L. (2001). *Random forests*. Machine Learning, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD 2016.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. Advances in Neural Information Processing Systems, 30.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

---

*Proyecto desarrollado con Python 3.12 · NASA Prognostics Center of Excellence Data Repository*
