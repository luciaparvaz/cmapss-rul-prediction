# NASA C-MAPSS FD001 — Mantenimiento Predictivo de Motores Turbofan

**Proyecto de fin de máster · Business Analytics · Universidad UNIE**
**Autora:** Lucía Pardo Vázquez · lucia.par.vaz@gmail.com
**Dataset:** NASA C-MAPSS FD001 (Commercial Modular Aero-Propulsion System Simulation)
**Objetivo:** Predecir la Vida Útil Restante (RUL) de motores turbofan a partir de series temporales de 21 sensores.

🌐 **English version: [`english/README.md`](english/README.md)** — full mirror (code, docs), run end to end and reproducing the same numbers.

---

## Cómo leer este README

Todas las cifras de este documento se generan por `train.py` y se leen directamente de
`outputs/resultados_finales.csv` (y del resto de CSV/JSON de `outputs/`) — no hay ningún
número transcrito a mano. Si vuelves a ejecutar el pipeline y las cifras cambian, este
README queda desactualizado hasta que se regenere.

El pipeline anterior (`scripts/fase1..fase7`) tenía tres definiciones de features
incompatibles entre sí, un bug de fuga temporal en la normalización, decisiones de
modelado (selección de modelo, ablación de sensores, tamaño de ventana) tomadas mirando
el test set, y varios contrastes estadísticos mal aplicados o directamente inexistentes
en el código pese a estar citados en el informe. Este README describe el pipeline
**reconstruido** (`train.py` + `src/cmapss/`), que sustituye por completo al anterior.

---

## Estructura del proyecto

```
cmapss-rul-prediction/
├── archive/                       # Datos crudos NASA C-MAPSS (solo FD001)
│   ├── train_FD001.txt            # 100 motores, run-to-failure (20.631 filas)
│   ├── test_FD001.txt             # 100 motores con trayectoria parcial (13.096 filas)
│   ├── RUL_FD001.txt              # RUL verdadero en el último ciclo de test
│   └── readme.txt                 # Descripción oficial del dataset (NASA)
├── src/cmapss/                    # Única fuente de verdad del pipeline
│   ├── config.py                  # Rutas relativas, semillas, hiperparámetros de CV
│   ├── data.py                    # Carga, identificación de sensores, target RUL
│   ├── features.py                # Ingeniería de features (una sola definición)
│   ├── modeling.py                # Pipelines sklearn, espacios de búsqueda, baselines
│   └── stats_eval.py              # PHM score, tests estadísticos corregidos
├── train.py                       # Entrypoint único: ejecuta todo el pipeline
├── models/                        # Modelos entrenados + scaler global
├── outputs/                       # Tablas (CSV/JSON) — fuente de verdad de las cifras
├── figures/                       # Figuras generadas por train.py
├── requirements.txt               # Versiones exactas verificadas
└── LICENSE
```

---

## Descripción del dataset

| Parámetro | Valor |
|---|---|
| Motores de entrenamiento | 100 |
| Motores de test | 100 |
| Filas de entrenamiento | 20.631 |
| Condición operacional | FD001: única (nivel del mar) |
| Modo de fallo | Degradación del compresor de alta presión (HPC) |
| Cap de RUL | 125 ciclos, piecewise-linear (Heimes 2008) |
| Motores de test censurados (RUL_true ≥ 125) | 11 / 100 |

**Identificación física de sensores** (Saxena et al. 2008, Tabla 2) — s3 (T30) y s4 (T50)
son **temperaturas**, no presiones, corrigiendo un error del README anterior:

| Sensor | Descripción |
|---|---|
| s2 | T24 — Temperatura total salida LPC (°R) |
| s3 | T30 — Temperatura total salida HPC (°R) |
| s4 | T50 — Temperatura total salida LPT (°R) |
| s7 | P30 — Presión total salida HPC (psia) |
| s8 / s9 | Nf / Nc — Velocidad física del fan / core (rpm) |
| s11 | Ps30 — Presión estática salida HPC (psia) |
| s12 | phi — Ratio caudal combustible / Ps30 |
| s13 / s14 | NRf / NRc — Velocidad corregida del fan / core (rpm) |
| s15 | BPR — Ratio de bypass |
| s17 | htBleed — Entalpía de sangrado |
| s20 / s21 | W31 / W32 — Sangrado de refrigerante HPT / LPT (lbm/s) |

Sensores descartados por varianza nula (std < 0.01, calculado en train): s1, s5, s6, s10,
s16, s18, s19.

---

## Pipeline (sin fuga de datos)

```
Datos crudos FD001
  -> MinMaxScaler global, AJUSTADO SOLO EN TRAIN (14 sensores informativos)
  -> Features de ventana deslizante: mean, std, slope OLS (causales/trailing)
  -> Delta (deriva acumulada desde el primer ciclo del motor)
  -> Ablación de ventanas por GroupKFold CV -> tamaño de ventana elegido por CV
  -> Ablación de s14 (mismo pipeline en ambos brazos) -> decisión por CV
  -> RandomizedSearchCV + GroupKFold(k=5) -> hiperparámetros por modelo
  -> Selección de modelo por CV out-of-fold (nunca por el test)
  -> Refit sobre 100% del train -> UNA sola evaluación en test
```

El proyecto anterior normalizaba cada sensor **por motor**, con un `MinMaxScaler`
ajustado sobre la trayectoria **completa** de ese motor (incluido el propio test) — el
valor normalizado del ciclo *t* dependía de ciclos futuros del mismo motor. Aquí el
escalador es un único `MinMaxScaler` ajustado exclusivamente sobre train y aplicado sin
reajuste a test, verificado en código:

```python
global_scaler = MinMaxScaler()
train_df[informative] = global_scaler.fit_transform(train_df[informative])
test_df[informative] = global_scaler.transform(test_df[informative])   # nunca .fit en test
```

---

## Feature engineering

Para cada sensor informativo y cada ventana `w` (elegida por CV, ver más abajo) se
calculan `mean`, `std` y la pendiente OLS (`slope`, vectorizada por convolución,
estrictamente trailing: en el ciclo *t* solo usa ciclos ≤ *t* del mismo motor). Se añade
además `delta = valor(t) − valor(primer ciclo del motor)`.

### Ablación de ventanas temporales (decidida por CV, no a mano)

Modelo sonda: LightGBM con hiperparámetros fijos modestos, `GroupKFold(k=5)` agrupado
por `engine_id`.

| Ventanas | Nº features | CV RMSE (± std) |
|---|---|---|
| [15, 30, 50] | 154 | 10.953 ± 0.666 |
| **[30, 50] ★** | **112** | **11.067 ± 0.692** |
| [15, 30, 45] | 154 | 11.148 ± 0.553 |
| [15, 45] | 112 | 11.175 ± 0.663 |
| [15, 30] | 112 | 12.610 ± 1.019 |

**Ventana elegida por CV:** `[30, 50]` → **112 features** totales (14 sensores × 2
ventanas × 3 estadísticos + 14 deltas). `[15, 30, 50]` tiene el RMSE medio más bajo,
pero la diferencia (0.11 ciclos) está dentro de 1 desviación estándar de `[30, 50]`, que
usa 42 features menos — se prefiere por parsimonia, no por una diferencia estadística
demostrada.

### Ablación del sensor s14 (mismo pipeline en ambos brazos)

s9 y s14 (velocidad física y corregida del core) tienen correlación de Pearson r≈0.96.
A diferencia del proyecto anterior — donde el brazo "con s14" y el brazo "sin s14" se
entrenaban con pipelines distintos (features, escalado y semilla diferentes) y la
decisión se leía del test — aquí ambos brazos usan exactamente el mismo pipeline y se
deciden por CV:

| Variante | CV RMSE (± std) |
|---|---|
| Con s14 | 11.067 ± 0.692 |
| Sin s14 | 11.148 ± 0.695 |

t-test pareado por fold: t=−0.73, p=0.506 → **decisión (por CV): conservar s14**. La
diferencia no es significativa; a diferencia del proyecto anterior (que sí eliminaba
s14 apoyándose en una mejora de test que resultó ser un artefacto de comparar dos
pipelines distintos), aquí se conserva por defecto ante la ausencia de evidencia de que
retirarlo ayude.

---

## Modelado y validación

- **Búsqueda de hiperparámetros:** `RandomizedSearchCV` + `GroupKFold(k=5)` para Ridge,
  RandomForest, XGBoost y LightGBM (espacios de búsqueda en `src/cmapss/modeling.py`).
  RandomForest usa menos combinaciones (8 vs 20) porque su ajuste por split exacto es
  ~10-20× más lento que el de los modelos por histograma en este dataset — es una
  limitación de cómputo declarada, no un atajo oculto.
- **Selección de modelo:** por RMSE promedio en predicciones out-of-fold
  (`cross_val_predict`, `GroupKFold(k=5)`) con los hiperparámetros ya elegidos. El test
  **no interviene** en esta decisión.
- **Comparaciones entre modelos:** t-test pareado sobre el error cuadrático medio por
  motor (agregado desde las predicciones OOF de train, nunca del test), con corrección
  de Holm sobre las 6 comparaciones posibles entre los 4 modelos.
- **Test set:** se toca **una sola vez**, al final, para reportar las métricas de todos
  los modelos ya congelados.

### Resultados finales (tabla única, `outputs/resultados_finales.csv`)

| Modelo | CV RMSE (± std) | Test RMSE | Test MAE | Test R² | PHM score |
|---|---|---|---|---|---|
| **LightGBM ★ (elegido por CV)** | **10.824 ± 0.701** | 12.986 | 9.410 | 0.895 | 273.6 |
| RandomForest (runner-up CV) | 10.893 ± 0.627 | **11.855** | **8.574** | **0.912** | **234.2** |
| XGBoost | 10.905 ± 0.635 | 12.552 | 8.752 | 0.902 | 250.3 |
| Ridge | 14.152 ± 1.163 | 14.076 | 11.363 | 0.877 | 298.4 |
| Baseline (vida media − ciclos) | — | 36.084 | 26.943 | 0.189 | 21 032 |
| Baseline (media del train) | — | 41.942 | 34.830 | −0.095 | 33 354 |
| Baseline (mediana del train) | — | 49.203 | 38.010 | −0.508 | 166 491 |

*(Cifras de PHM regeneradas tras corregir un bug real: los parámetros
`a_early`/`a_late` de `phm_score` estaban intercambiados respecto al estándar
de Saxena — a2=10 debe aplicarse a la sobreestimación (d≥0, la dirección
peligrosa) y a1=13 a la subestimación (d<0), y una versión anterior de este
mismo README los tenía al revés. Con los valores corregidos, el ranking
relativo entre los 4 modelos no cambia — RandomForest sigue teniendo el mejor
PHM score en test — pero las magnitudes sí, y no deben leerse como las de una
versión previa de esta tabla.)*

**Modelo elegido por CV:** LightGBM (runner-up: RandomForest, diferencia de CV RMSE de
solo 0.07 ciclos, no significativa — ver Holm más abajo).

**Nota de honestidad metodológica:** en el test set (tocado una única vez), RandomForest
obtiene mejor RMSE que LightGBM (11.86 vs 12.99). Esto es exactamente la clase de
variación muestral que el diseño de este proyecto está pensado para no ocultar: el
modelo se eligió por CV *antes* de mirar el test, y esa elección no cambia a posteriori
solo porque el test hubiera favorecido a otro modelo — hacerlo sería la misma fuga de
decisión que denunció la auditoría del proyecto anterior. El test de Holm (más abajo)
muestra que RandomForest, XGBoost y LightGBM son estadísticamente indistinguibles entre
sí en CV; solo Ridge queda claramente por detrás. Con esa información, la diferencia
observada en el único test set es coherente con ruido de muestreo entre tres modelos
empatados, no con que LightGBM sea realmente peor.

Los tres baselines triviales (media, mediana y "vida media del train − ciclos
observados") confirman que los modelos aprenden señal real: LightGBM reduce el RMSE de
test un 64% frente al mejor baseline (36.08 → 12.99); RandomForest, un 67% (36.08 →
11.86).

**Censura a la derecha:** 11 de los 100 motores de test tienen RUL_true ≥ 125 (su
target real quedó truncado por el cap). La columna `test_rmse_no_censurado` de
`outputs/resultados_finales.csv` es el RMSE sobre el 89% no censurado — se reporta como
**cota inferior**, nunca como "el RMSE real" (a diferencia del README anterior).

### Comparaciones estadísticas entre modelos (Holm, `outputs/comparaciones_modelos_holm.csv`)

t-test pareado sobre el error cuadrático medio por motor (OOF-CV). **No es un test de
Diebold-Mariano**: DM exige una única serie temporal con horizonte fijo y varianza de
largo plazo (HAC); aquí la unidad de comparación son 100 motores distintos, una
comparación transversal — el nombre "Diebold-Mariano" del proyecto anterior era
incorrecto, igual que su corrección Harvey-Leybourne-Newbold (estaba aplicada al revés:
dividía el estadístico t en vez de multiplicarlo).

| Comparación | t | p | p (Holm) | ¿Significativa? |
|---|---|---|---|---|
| Ridge vs RandomForest | 8.23 | 7.8e-13 | 3.1e-12 | Sí |
| Ridge vs XGBoost | 8.89 | 2.8e-14 | 1.7e-13 | Sí |
| Ridge vs LightGBM | 8.89 | 2.8e-14 | 1.7e-13 | Sí |
| RandomForest vs XGBoost | 0.82 | 0.416 | 0.832 | No |
| RandomForest vs LightGBM | 1.35 | 0.179 | 0.536 | No |
| XGBoost vs LightGBM | 0.65 | 0.519 | 0.832 | No |

Ridge queda significativamente por detrás de los tres modelos de árboles incluso tras
la corrección de Holm sobre las 6 comparaciones. Entre RandomForest, XGBoost y LightGBM
**no hay diferencia estadísticamente significativa** — están empatados dentro del ruido
de la validación cruzada, lo que contextualiza el cambio de ranking en el test set
descrito arriba.

---

## Interpretabilidad (SHAP)

`figures/f08_shap_summary.png` muestra la importancia SHAP global (TreeExplainer) del
modelo ganador sobre el test set. Es un análisis de interpretabilidad, no una decisión
del pipeline — el test ya estaba "gastado" en la evaluación final cuando se generó.

---

## Rigor estadístico sobre los residuos

Se reconstruye la trayectoria completa de RUL real para cada motor de test (fijando un
error de desfase de un ciclo que tenía el proyecto anterior: `arange(n_cyc+rul_true,
rul_true, -1)` terminaba en `rul_true+1`, no en `rul_true`) y se predice ciclo a ciclo
con el modelo ganador.

**Los contrastes se calculan solo sobre `RUL_real <= 125`** (n=89 motores; excluye los 11
motores censurados y el tramo temprano de cada trayectoria donde `RUL_real > 125`). El
motivo, visible en `figures/f07_residuals_heteroscedasticity.png` (gris = excluido, azul =
usado): el modelo se entrenó sobre el target **capeado** a 125, así que para ciclos con
RUL real muy por encima del cap la predicción queda mecánicamente topada en ~125 mientras
el eje de RUL real sigue subiendo sin límite — eso produce una tendencia casi perfectamente
lineal que no es heterocedasticidad del modelo, es aritmética del cap. Incluir ese tramo
(como hacía una versión anterior de este mismo pipeline) infla artificialmente el
estadístico de Breusch-Pagan.

- **Heteroscedasticidad (Breusch-Pagan real):** agregado por motor (n=89, no sobre los
  ~13.000 ciclos individuales — el proyecto original calculaba lo que en realidad es un
  test de Glejser con los grados de libertad inflados ×132 por pseudo-replicación
  intra-motor). Sin filtrar por el cap, LM=69.03, p=9.7e-17 parecía indicar
  heterocedasticidad fuerte; **filtrando a RUL<=125, LM=1.59, p=0.207 — no hay evidencia
  significativa de heterocedasticidad.** El hallazgo "el error crece con el nivel de RUL"
  de una versión anterior de este README era, en su mayor parte, el artefacto del cap
  descrito arriba, no una propiedad real del modelo.
- **Autocorrelación (Ljung-Box, no Durbin-Watson):** sobre residuos **centrados por
  motor** (el sesgo medio de cada motor inflaba artificialmente el DW original), mismo
  filtro RUL<=125. El 92.5% de los motores muestra autocorrelación significativa
  (lag=10) — a diferencia de la heterocedasticidad, esto se mantiene tras el filtro y sí
  es un resultado esperable: un modelo de árboles sin memoria sobre una trayectoria de
  degradación monótona. Un modelo secuencial (LSTM, TCN) o una corrección autorregresiva
  del residuo podrían aprovechar esa estructura; queda fuera del alcance de este
  proyecto.

---

## Limitaciones

- **Un único régimen operativo y un único modo de fallo** (FD001: nivel del mar,
  degradación de HPC). Las cifras no son transferibles a FD002/FD003/FD004 sin
  reentrenar; ese ejercicio queda fuera de este repositorio.
- **Target capeado a 125 ciclos** (piecewise-linear, Heimes 2008): todas las métricas
  están calculadas sobre RUL truncado, no sobre el RUL real sin censurar.
- **11 motores de test censurados** (RUL_true ≥ 125): su error de predicción
  está contaminado por la censura; se reporta por separado, nunca oculto.
- **Sin cuantificación de incertidumbre por predicción** (intervalos de predicción,
  conformal prediction): los modelos devuelven un punto, no una distribución.
- **Dirección del sesgo y seguridad:** un sesgo positivo del error (`RUL_real −
  RUL_pred`, ver `figures/f07_residuals_heteroscedasticity.png`) significa que el modelo
  predijo **menos** vida útil de la real (subestimación, lado conservador). El sesgo
  peligroso en mantenimiento de motores de avión es el opuesto — **sobreestimar** el RUL
  retrasa un mantenimiento necesario. Cualquier despliegue real debería monitorizar y
  penalizar explícitamente ese lado del error (la métrica PHM de Saxena ya lo hace,
  penalizando la sobreestimación con un factor exponencial mayor).
- **No apto para decisiones de aeronavegabilidad**: este es un ejercicio de portfolio
  sobre datos simulados de la NASA, no un sistema certificado. Cualquier uso operativo
  real requeriría validación adicional, supervisión humana y cumplimiento normativo.

---

## Reproducibilidad

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
python train.py
```

Un único comando ejecuta todo el pipeline: carga, ablaciones por CV, búsqueda de
hiperparámetros, entrenamiento final, evaluación en test, bloque estadístico, guardado
de modelos (con verificación de que se recargan y reproducen el RMSE de test) y figuras.
Tiempo medido en la última ejecución completa: **~19 minutos en CPU**, de los cuales
~8.4 min son la búsqueda de hiperparámetros de RandomForest (8 combinaciones) — con
diferencia el paso más lento del pipeline por usar split exacto en vez de histograma
(ver nota en `src/cmapss/modeling.py`) — pese a probar menos combinaciones que XGBoost o
LightGBM (20 cada uno, ambos completados en ~4 y ~3 min respectivamente). El tiempo total
depende de la máquina; la cifra de arriba es de un run real, no una estimación.

**Nota de compatibilidad de versiones:** `xgboost` combinado con versiones recientes de
`scikit-learn` tiene un bug conocido (`AttributeError` en `__sklearn_tags__` al envolver
`XGBRegressor` en un `sklearn.Pipeline`). `requirements.txt` fija versiones donde ya está
corregido — no bajar `xgboost` sin volver a verificar ese punto.

El modelo XGBoost se guarda en formato nativo `models/xgboost_fd001.json` (estable entre
versiones de la librería), no como `.pkl`: el `.pkl` del proyecto anterior era
irrecuperable entre versiones de xgboost (devolvía RMSE≈68 al recargarlo). Los demás
modelos (Ridge, RandomForest, LightGBM) se guardan con `joblib`.

---

## Referencias

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling
  for Aircraft Engine Run-to-Failure Simulation*. 1st International Conference on
  Prognostics and Health Management (PHM08), Denver, CO.
- Heimes, F. O. (2008). *Recurrent Neural Networks for Remaining Useful Life
  Estimation*. International Conference on Prognostics and Health Management (PHM).
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*.
  Scandinavian Journal of Statistics, 6(2), 65–70.
- Breusch, T. S., & Pagan, A. R. (1979). *A Simple Test for Heteroscedasticity and
  Random Coefficient Variation*. Econometrica, 47(5), 1287–1294.
- Ljung, G. M., & Box, G. E. P. (1978). *On a Measure of Lack of Fit in Time Series
  Models*. Biometrika, 65(2), 297–303.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model
  Predictions*. Advances in Neural Information Processing Systems, 30.

---

*Proyecto desarrollado en Python 3.12 · NASA Prognostics Center of Excellence Data
Repository.*
