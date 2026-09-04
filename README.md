# NASA C-MAPSS FD001 — Mantenimiento Predictivo de Motores Turbofan

**Proyecto de fin de máster · Business Analytics · Universidad UNIE**  
**Autora:** Lucía Pardo Vázquez · lucia.par.vaz@gmail.com  
**Dataset:** NASA C-MAPSS FD001 (CMAPSS: Commercial Modular Aero-Propulsion System Simulation)  
**Objetivo:** Predecir la Vida Útil Restante (RUL) de motores turbofan a partir de series temporales de sensores

---

## Estructura del proyecto

```
cmapss/
├── data/                          # Datos originales NASA C-MAPSS
│   ├── train_FD001.txt            # 100 motores, run-to-failure (20.631 filas)
│   ├── test_FD001.txt             # 100 motores con trayectoria parcial
│   └── RUL_FD001.txt             # RUL verdadero en el último ciclo de test
├── models/                        # Modelos y escaladores guardados
│   ├── xgboost_fd001.pkl          # Mejor modelo (RMSE=12.80, R²=0.898)
│   ├── lightgbm_fd001.pkl
│   ├── randomforest_fd001.pkl
│   ├── ridge_fd001.pkl
│   ├── global_sensor_scaler.pkl   # MinMaxScaler global (ajustado en train)
│   └── final_scaler.pkl           # StandardScaler (ajustado en train, 112 features)
├── outputs/                       # Figuras y tablas de resultados
│   ├── f5_01 … f5_05_*.png        # Fase 5: interpretabilidad
│   ├── f6_01 … f6_02_*.png        # Fase 6: comparativas y pipeline
│   ├── f7_01 … f7_06_*.png        # Fase 7: mejoras estadísticas
│   ├── fase4_resultados.csv       # Métricas CV y test de los 4 modelos
│   ├── fase7_ablacion_ventanas.csv
│   └── informe_tecnico_cmapss.pdf # Informe técnico completo (PDF)
├── fase1_carga_inspeccion.py      # EDA inicial, varianza de sensores
├── fase2_feature_engineering.py   # Rolling features (w=15,30), OLS slope
├── fase3_eda_degradacion.py       # Análisis degradación, correlaciones
├── fase4_modelado.py              # Entrenamiento, GroupKFold CV, selección
├── fase5_evaluacion.py            # SHAP, curvas de degradación, terciles
├── fase6_documentacion.py         # Informe técnico PDF y figuras comparativas
└── fase7_mejoras.py               # 6 mejoras estadísticas (Fase 7)
```

---

## Descripción del dataset

| Parámetro | Valor |
|---|---|
| Motores de entrenamiento | 100 |
| Motores de test | 100 |
| Filas de entrenamiento | 20.631 |
| Sensores seleccionados | 14 (s2, s3, s4, s7–s9, s11–s15, s17, s20, s21) |
| Sensores descartados (varianza nula) | 7 (s1, s5, s6, s10, s16, s18, s19) |
| Cap de RUL | 125 ciclos (Heimes 2008) |
| Condición operacional | FD001: única (altitud baja, baja carga) |

---

## Pipeline

```
Datos raw  →  GlobalMinMaxScaler (14 sensores)
           →  Rolling features: mean, std, OLS slope (w=15, w=30)
           →  Delta feature (diferencia respecto al ciclo inicial)
           →  112 features totales
           →  StandardScaler
           →  GroupKFold(k=5) agrupado por engine_id
           →  XGBoost / LightGBM / RandomForest / Ridge
```

**Nota crítica de normalización:** El escalador global (MinMaxScaler) se ajusta sobre TODOS los ciclos del conjunto de entrenamiento y se aplica sin reajuste al test. Ajustar el escalador por motor en el test introduciría sesgo severo al llevar todos los valores al rango [0,1] independientemente del RUL real.

---

## Feature engineering

Para cada uno de los 14 sensores y cada ventana temporal `w ∈ {15, 30}` se calculan:

- **Media rolling** (`s_w{w}_mean`): tendencia central del sensor en los últimos `w` ciclos
- **Desviación estándar rolling** (`s_w{w}_std`): variabilidad reciente
- **Pendiente OLS rolling** (`s_w{w}_slope`): tasa de cambio lineal mediante regresión vectorizada con convolución
- **Delta** (`s_delta`): diferencia acumulada respecto al primer ciclo del motor

Total: 14 × (1 + 2×3) + 14 = **112 features**

---

## Resultados — Fase 4 (Modelado)

| Modelo | CV RMSE (±std) | CV R² | Test RMSE | Test R² |
|---|---|---|---|---|
| **XGBoost** ★ | **12.93 ± 0.73** | **0.903** | **12.80** | **0.898** |
| LightGBM | 12.90 ± 0.80 | 0.904 | 13.51 | 0.886 |
| RandomForest | 13.80 ± 0.81 | 0.890 | 13.69 | 0.883 |
| Ridge | 15.62 ± 0.58 | 0.859 | 14.78 | 0.864 |

Gap CV–Test: **0.12 ciclos** (XGBoost) → generalización excelente.  
94% de los motores se predicen con error ≤ 15 ciclos en la fase tardía (RUL < 42).

---

## Resultados — Fase 5 (Interpretabilidad SHAP)

### Features más importantes (SHAP interventional, test)

| Feature | |SHAP| medio |
|---|---|
| s4_w15_mean | 5.091 |
| s3_w15_mean | 4.117 |
| s9_w15_mean | 3.357 |
| s2_w15_mean | 3.186 |
| s11_w30_slope | 2.572 |

**Interpretación:** El sensor 4 (presión total en la salida del compresor de baja presión) y el sensor 3 (presión total en la entrada del compresor) dominan la predicción. Las pendientes de ventana larga capturan la aceleración de la degradación.

### Error por tercil de RUL

| Fase | RMSE | % motores con error ≤ 15 ciclos |
|---|---|---|
| Tardía (RUL < 42) | 6.20 | 94% |
| Media (42–83) | 17.46 | 53% |
| Temprana (RUL > 83) | 12.32 | 85% |

---

## Resultados — Fase 7 (Mejoras estadísticas)

### M1 — Test Diebold-Mariano con corrección Harvey-Leybourne-Newbold

Contraste estadístico formal de la superioridad de modelos en predicción de series temporales (Harvey et al. 1997). La corrección HLN ajusta el estadístico t para muestras pequeñas (n=100).

| Comparación | t (HLN) | p-valor | Conclusión |
|---|---|---|---|
| XGBoost vs LightGBM | −2.191 | **0.031** | Diferencia significativa (α=0.05) |
| XGBoost vs RandomForest | −1.380 | 0.171 | No significativa |

**Interpretación:** XGBoost supera estadísticamente a LightGBM (p=0.031), a pesar de que la diferencia de CV RMSE era de solo 0.03 ciclos. La selección del modelo final queda así justificada estadísticamente, no solo por la mejor media de CV.

---

### M2 — Segregación por censura derecha

11 de los 100 motores de test tienen RUL_true ≥ 125, lo que significa que su target fue truncado a 125 y el error de predicción está contaminado.

| Grupo | n | RMSE | Bias |
|---|---|---|---|
| Global | 100 | 12.90 | +3.18 |
| No censurado (RUL_true < 125) | 89 | **12.16** | +3.79 |
| Censurado (RUL_true ≥ 125) | 11 | **17.77** | −1.70 |

**Interpretación:** El RMSE "real" del modelo (sin contaminación por censura) es 12.16 ciclos. Los 11 motores censurados arrastran el RMSE global hacia arriba de forma artificiosa porque el target no refleja el verdadero RUL.

---

### M3 — PHM Score asimétrica (Saxena et al. 2008)

La métrica estándar del PHM Challenge penaliza asimétricamente: sobreestimar el RUL (el motor falla antes de lo predicho) es exponencialmente más costoso que subestimarlo.

```
s(d) = exp(d/10) − 1    si d ≥ 0  (sobreestimación)
s(d) = exp(−d/13) − 1   si d < 0  (subestimación)
```

| Modelo | PHM Score (↓ mejor) | RMSE |
|---|---|---|
| **XGBoost** | **253.6** | 12.90 |
| LightGBM | 284.4 | 13.51 |
| RandomForest | 310.2 | 13.69 |
| Ridge | 450.5 | 15.89 |

**Interpretación:** Con los scalers correctos el ranking PHM coincide con el de RMSE. XGBoost mantiene su posición como mejor modelo también bajo la métrica industrial asimétrica.

---

### M4 — SHAP tree_path_dependent + ablación de s14

**Multicolinealidad s9–s14:** Pearson r = 0.963. La alta correlación distorsiona las importancias SHAP interventional al distribuir el peso entre ambas features.

| Perturbación SHAP | Top feature |SHAP| |
|---|---|---|
| interventional | s4_w15_mean = 5.091 | Elimina efecto de correlación |
| tree_path_dependent | s4_w15_mean = 6.654 | Incorpora efecto conjunto |

**Ablación de s14 en XGBoost:**

| Configuración | CV RMSE | Test RMSE | Δ Test |
|---|---|---|---|
| Con s14 (original) | 12.925 | 12.804 | — |
| Sin s14 | 13.004 | **11.805** | **−0.999** |

**Interpretación:** Eliminar s14 *mejora* el test RMSE en 1 ciclo. s14 es prescindible — su información ya está contenida en s9 y añadirla solo introduce ruido por multicolinealidad.

---

### M5 — Durbin-Watson y Breusch-Pagan sobre trayectorias completas

El análisis se realiza sobre las trayectorias completas de todos los ciclos de test (no solo el último ciclo), prediciendo ciclo a ciclo con el modelo entrenado.

**Durbin-Watson (autocorrelación de residuos):**

| Estadístico | Valor |
|---|---|
| DW medio por motor | **0.173 ± 0.132** |
| Motores con DW < 1.5 | 100 / 100 |
| Interpretación | Autocorrelación positiva muy fuerte |

DW = 2 indica ausencia de autocorrelación. DW = 0.173 indica que el error en el ciclo t predice fuertemente el error en el ciclo t+1. XGBoost, al ser un modelo de árbol sin memoria, no aprovecha esta estructura temporal.

**Breusch-Pagan (heteroscedasticidad):**

| Estadístico | Valor |
|---|---|
| F de Breusch-Pagan | 631.13 |
| p-valor | < 0.0001 |
| β_RUL | −0.068 |
| Interpretación | Heteroscedasticidad severa |

El coeficiente negativo (β_RUL = −0.068) indica que el error es mayor cuando el RUL es alto (fases tempranas de vida del motor), donde la señal de degradación aún es débil.

**Implicación:** Modelos secuenciales (LSTM, TCN) captarían la autocorrelación temporal en los residuos que el árbol de decisión ignora.

---

### M6 — Ablación de ventanas temporales (LightGBM, GroupKFold k=5)

Justificación empírica de la elección w = {15, 30}.

| Ventanas | Features | CV RMSE | Test RMSE |
|---|---|---|---|
| [15,30,45] | 154 | **11.40** | 12.53 |
| [15,45] | 112 | 11.48 | 12.72 |
| [30,45] | 112 | 11.56 | 12.43 |
| [45] | 70 | 11.61 | 12.79 |
| [5,15,30] | 154 | 12.65 | 11.45 |
| **[15,30] ★** | **112** | **12.83** | **11.64** |
| [30] | 70 | 12.83 | 11.81 |
| [5] | 70 | 13.61 | 13.65 |
| [5,15] | 112 | 13.86 | 13.81 |
| [15] | 70 | 13.90 | 13.33 |

**Interpretación:** Las combinaciones con w=45 mejoran el CV (11.40–11.61) pero el beneficio en test es marginal y a costa de +40 features (154 vs 112). La configuración original [15,30] ofrece el mejor equilibrio entre complejidad y generalización. Añadir w=5 perjudica consistentemente (ruido a corto plazo). La ventana de 30 ciclos captura la tendencia de degradación media-larga sin sobreajustar.

---

## Reproducibilidad

```bash
# Dependencias
pip install numpy pandas scikit-learn xgboost lightgbm shap joblib matplotlib scipy reportlab

# Ejecutar por fases (en orden)
python fase1_carga_inspeccion.py
python fase2_feature_engineering.py
python fase3_eda_degradacion.py
python fase4_modelado.py
python fase5_evaluacion.py
python fase6_documentacion.py
python fase7_mejoras.py   # Mejoras 1–5
python fase7_m6.py        # Mejora 6 + resumen final
```

**IMPORTANTE:** Los modelos guardados en `models/` deben usarse siempre con `global_sensor_scaler.pkl` y `final_scaler.pkl` del mismo directorio. Reajustar los scalers desde cero produce predicciones incorrectas (RMSE ≈19.8 en lugar de 12.8) porque los parámetros del StandardScaler difieren ligeramente.

---

## Referencias

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation*. International Conference on Prognostics and Health Management (PHM), Denver, CO.
- Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation*. International Conference on Prognostics and Health Management (PHM).
- Harvey, D., Leybourne, S., & Newbold, P. (1997). *Testing the equality of prediction mean squared errors*. International Journal of Forecasting, 13(2), 281–291.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems, 30.

---

*Proyecto desarrollado con Python 3.12 · NASA Prognostics Center of Excellence Data Repository*
