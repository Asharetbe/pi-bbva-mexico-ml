# Estimación de Probabilidad de Incumplimiento (PI) - BBVA México

Proyecto académico para el desarrollo e implementación de **cuatro familias de modelos de machine learning** (Regresión Logística, Random Forest, XGBoost y Redes Neuronales Profundas) orientados a estimar la **Probabilidad de Incumplimiento (PI)** crediticia de clientes BBVA (Datos Sintéticos).

## 📊 Resumen de Resultados

Se evaluaron los modelos base bajo las mismas métricas (optimizando threshold por métrica KS). A continuación, el desempeño calculado sobre la partición fuera de tiempo / validación final (OOS):

| Modelo | AUC OOS | Gini OOS | KS OOS | Recall OOS | Precision OOS | Gap Train-OOS | PSI OOS |
|---|---|---|---|---|---|---|---|
| **Regresión Logística (WOE)** | 0.7370 | 0.4740 | 0.3834 | 0.6456 | 0.1579 | 0.0627 | 0.0024 |
| **Random Forest** | 0.7570 | 0.5140 | 0.4290 | 0.6460 | 0.1910 | 0.1760 | 0.0520 |
| **XGBoost** | **0.7900** | **0.5800** | **0.4700** | 0.3671 | **0.3494** | 0.2057 | 0.0382 |
| **Deep Learning (SimpleMLP)** | 0.7578 | 0.5156 | 0.4078 | - | - | 0.0989 | 0.0123 |
| **Deep Learning (ResAutoInt)** | 0.7552 | 0.5105 | 0.4149 | - | - | **0.0723** | **0.0039** |

**Conclusiones Principales:**
- **XGBoost** obtiene el mayor poder discriminante (AUC y KS), pero presenta el mayor nivel de sobreajuste (brecha de AUC mayor al 20% vs el set de entrenamiento).
- **Regresión Logística (Scorecard)** es el modelo más congruente y estable en base a su distribución (PSI virtualmente 0 y mucha robustez de caída entre Train-OOS), haciéndolo una opción excelente y fácil de explicar.
- Las arquitecturas de **Deep Learning** logran una excelente estabilidad poblacional (PSI OOS < 0.02) y, mediante fine-tuning y arquitecturas orientadas a datos tabulares (ResAutoInt), superan fácilmente la robustez del Random Forest asimilando resultados consistentes.

## 🛠️ Reproducibilidad

Todo el código está diseñado para resolver las rutas automáticamente desde la base del repositorio y sin paths *hardcodeados*. 

1. **Clonar e preparar el entorno:**
   ```bash
   git clone https://github.com/Asharetbe/pi-bbva-mexico-ml.git
   cd pi-bbva-mexico-ml
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Preparar Subconjuntos (Splits) de Datos:**
   Si cuentas con el archivo original crudo (`.xlsx`), colócalo en `data/raw/BasePI.xlsx`.  
   *Nota: Por seguridad, los datos crudos no se suben al control de versiones.*  
   Ejecuta el notebook `notebooks/P3_RandomForest/01_p3_splits_diagnostico.ipynb` para generar correctamente las particiones pre-procesadas `[X_train, y_train, X_test, etc.]` dentro de `data/splits/`.

3. **Análisis Bivariado Integrado:**
   Ejecuta `notebooks/P3_RandomForest/02_p3_bivariado_final.ipynb` para crear las transformaciones de Variables y sus valores calculados en formato `.csv` guardados en `data/variables_bivariadas/`. (Estos archivos son de requerimiento para cargar en los notebooks de XGBoost, LogReg, DL, etc.).

4. **Entrenar y Evaluar Modelos:**
   Puedes ejecutar los demás archivos e investigaciones en los notebooks de tu elección, ya que encontrarán sus dependencias adecuadamente:
   - **LogReg**: `notebooks/P1_LogReg/regresionLog_proyectoFinal.ipynb`
   - **XGBoost**: `notebooks/P2_XGBoost/01_P2_Modelo_XGBoost.ipynb`
   - **Random Forest**: `notebooks/P3_RandomForest/03_p3_rf_baseline.ipynb` o el de *tuning* `04_p3_...`
   - **Deep Learning**: `cd notebooks/P4_NeuralNet/ && python busqueda_focalizada.py` (o `gran_busqueda.py`)

## 📁 Estructura del Repositorio

- `data/` : Subdirectorios de particiones reproducibles (`splits/`), outputs temporales (`variables_bivariadas/`) y archivo crudo inicial (`raw/`).
- `notebooks/` : Secciones segregadas por modelo.
  - `P1_LogReg/` : Scorecard y Regresión Logística usando técnica de WOEs.
  - `P2_XGBoost/` : Investigación del algoritmo Gradient Boosting base.
  - `P3_RandomForest/` : Notebooks fundacionales (diagnóstico, split poblacional y bivariado), y estudio de Bosques Aleatorios.
  - `P4_NeuralNet/` : Scripts parametrizados en python crudo para arquitecturas Deep Learning, incluyendo Optuna HP Tuning.
- `src/utils/` : Funciones centralizadas transversales. Fundamental para `evaluation.py` donde se homogeniza la métrica `KS` y `Population Stability Index (PSI)`.
- `reports/` : Almacenamiento local automatizado de gráficos / métricas .csv para documentación rápida.
- `models/` : Almacenamiento local automatizado para archivos de validación de modelos u operaciones generadas.
