# Estimación de Probabilidad de Incumplimiento (PI) - BBVA México

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow?style=flat-square)
![Contributors](https://img.shields.io/badge/Contributors-4-blue?style=flat-square)

## 📋 Descripción del Proyecto

Proyecto académico para el desarrollo e implementación de **cuatro modelos de machine learning** orientados a estimar la **Probabilidad de Incumplimiento (PI)** crediticia de clientes BBVA México.

**Objetivo**: Crear modelos predictivos robustos que identifiquen riesgo de incumplimiento y evaluar su desempeño mediante métricas estándar de riesgo crediticio.

**Base de datos**: 10,000 observaciones con 120 variables predictoras y 1 variable binaria de incumplimiento (8% de incumplimiento).

**Entrega final**: 30 de mayo de 2026 — Reporte en PDF + Word + Notebooks reproducibles.

---

## 👥 Responsables y Modelos Asignados

| Integrante | Modelo | Secciones del Informe |
|-----------|--------|----------------------|
| **P1** | Regresión Logística + WOE | §2.1 Problemática, §3 Conclusiones, Scorecard |
| **P2** | XGBoost / Gradient Boosting | §1.1 BBVA México, §2.4 Comparación de modelos |
| **P3** | Random Forest | §2.2.2 Análisis bivariado, Infraestructura splits |
| **P4** | Red Neuronal MLP | §2.2.1 Análisis univariado, Integración documento |

---

## 📁 Estructura del Proyecto

```
pi-bbva-mexico-ml/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── .gitignore                         # Configuración Git
├── setup.py                           # (Opcional) Instalación como paquete
│
├── data/
│   ├── raw/
│   │   └── BasePI.xlsx                # Base de datos original (NO VERSIONADO)
│   ├── processed/
│   │   ├── train_70_seed42.csv        # Split entrenamiento (70%)
│   │   ├── test_20_seed42.csv         # Split prueba (20%)
│   │   └── oot_10_seed42.csv          # Split fuera de tiempo (10%)
│   └── metadata/
│       └── diccionario_completo.csv   # Diccionario de variables
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── evaluation.py              # Función común de evaluación
│   │   ├── splits.py                  # Generador de splits estratificados
│   │   └── preprocessing.py           # Utilidades de preprocesamiento común
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py     # M1: Regresión Logística
│   │   ├── xgboost_model.py           # M2: XGBoost
│   │   ├── random_forest.py           # M3: Random Forest
│   │   └── neural_network.py          # M4: Red Neuronal MLP
│   └── config.py                      # Configuración global (seed, paths, etc.)
│
├── notebooks/
│   ├── P1_Logistic_Regression/
│   │   ├── 01_P1_EDA_LR.ipynb         # Análisis exploratorio
│   │   ├── 02_P1_WOE_Engineering.ipynb # WOE y feature engineering
│   │   ├── 03_P1_Model_Training.ipynb # Entrenamiento del modelo
│   │   └── 04_P1_Scorecard.ipynb      # Construcción del scorecard
│   │
│   ├── P2_XGBoost/
│   │   ├── 01_P2_EDA_XGB.ipynb        # Análisis exploratorio
│   │   ├── 02_P2_Feature_Selection.ipynb # Selección de features
│   │   ├── 03_P2_Hyperparameter_Tuning.ipynb # Tuning
│   │   └── 04_P2_Model_Evaluation.ipynb # Evaluación y comparación
│   │
│   ├── P3_Random_Forest/
│   │   ├── 01_P3_EDA_RF.ipynb         # Análisis exploratorio
│   │   ├── 02_P3_Bivariate_Analysis.ipynb # Análisis bivariado
│   │   ├── 03_P3_Model_Training.ipynb # Entrenamiento
│   │   └── 04_P3_Feature_Importance.ipynb # Importancia de variables
│   │
│   └── P4_Neural_Network/
│       ├── 01_P4_EDA_NN.ipynb         # Análisis exploratorio
│       ├── 02_P4_Univariate_Analysis.ipynb # Análisis univariado
│       ├── 03_P4_Data_Normalization.ipynb # Normalización
│       └── 04_P4_Model_Training.ipynb # Entrenamiento MLP
│
├── models/
│   ├── M1_logistic_regression.pkl     # Modelo entrenado P1
│   ├── M2_xgboost.json                # Modelo entrenado P2
│   ├── M3_random_forest.pkl           # Modelo entrenado P3
│   └── M4_neural_network.h5           # Modelo entrenado P4
│
├── results/
│   ├── metrics/
│   │   ├── M1_performance_metrics.csv
│   │   ├── M2_performance_metrics.csv
│   │   ├── M3_performance_metrics.csv
│   │   └── M4_performance_metrics.csv
│   ├── plots/
│   │   ├── roc_curves_comparison.png
│   │   ├── feature_importance.png
│   │   └── confusion_matrices.png
│   └── predictions/
│       ├── M1_predictions_test.csv
│       ├── M2_predictions_test.csv
│       ├── M3_predictions_test.csv
│       └── M4_predictions_test.csv
│
├── docs/
│   ├── PROJECT_CHARTER.md             # Alcance y objetivos
│   ├── TECHNICAL_SPECS.md             # Especificaciones técnicas
│   ├── BRANCHING_STRATEGY.md          # Estrategia de ramas Git
│   └── COMMIT_CONVENTION.md           # Convención de commits
│
└── reports/
    ├── Informe_Final_PI_BBVA.pdf      # Informe final (entrega)
    ├── Informe_Final_PI_BBVA.docx     # Versión Word
    └── Resumen_Ejecutivo.pdf          # Resumen para presentación
```

---

## 🔄 Split de Datos

Los datos se dividen **estratificados** con `random_state=42`:

- **Train (70%)**: 7,000 observaciones → Entrenamiento
- **Test (20%)**: 2,000 observaciones → Evaluación del desempeño
- **OOT (10%)**: 1,000 observaciones → Prueba fuera de tiempo (validación de estabilidad)

**Comando para generar splits**:
```bash
python src/utils/splits.py --input data/raw/BasePI.xlsx --output data/processed/
```

---

## 📦 Instalación y Setup

### Requisitos previos
- Python 3.9 o superior
- pip o conda
- Git

### Pasos de instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Asharetbe/pi-bbva-mexico-ml.git
   cd pi-bbva-mexico-ml
   ```

2. **Crear entorno virtual**:
   ```bash
   python -m venv venv
   ```
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generar splits de datos** (ejecutar una sola vez):
   ```bash
   python src/utils/splits.py --input data/raw/BasePI.xlsx --output data/processed/
   ```

5. **Verificar setup**:
   ```bash
   python -c "import pandas, sklearn, xgboost, tensorflow, keras; print('✓ Setup OK')"
   ```

---

## 🚀 Instrucciones de Reproducción

### Para cada integrante

**Paso 1: Clonar y actualizar el repositorio**
```bash
git clone https://github.com/Asharetbe/pi-bbva-mexico-ml.git
cd pi-bbva-mexico-ml
git pull origin main
```

**Paso 2: Crear rama personal** (usar convención `[P#]/feature-description`)
```bash
git checkout -b P1/woe-scorecard        # P1
git checkout -b P2/xgboost-tuning       # P2
git checkout -b P3/random-forest-features # P3
git checkout -b P4/neural-network       # P4
```

**Paso 3: Instalar entorno local**
```bash
python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # macOS/Linux
pip install -r requirements.txt
python src/utils/splits.py --input data/raw/BasePI.xlsx --output data/processed/
```

**Paso 4: Trabajar en notebooks**
- Abrir `notebooks/P#_ModelName/` correspondiente
- Seguir plantilla de secciones predefinidas
- Guardar resultados en `results/`

**Paso 5: Hacer commit y push**
```bash
git add .
git commit -m "[P1] feat: implementar WOE y scorecard"  # Usar convención
git push origin P1/woe-scorecard
```

**Paso 6: Crear Pull Request**
- En GitHub: New Pull Request
- Base: `main` ← Compare: `P1/woe-scorecard`
- Descripción: explicar cambios principales
- Esperar revisión de P4 (integrador)

---

## 📅 Cronograma de Hitos

| Fecha | Hito | Responsable |
|-------|------|------------|
| **May 26** | ✅ Setup repositorio, splits, utils | P4 |
| **May 27** | Modelos M1 y M2 entrenados | P1, P2 |
| **May 28** | Modelos M3 y M4 entrenados | P3, P4 |
| **May 29** | Informe combinado (secciones integradas) | P4 |
| **May 30** | ✅ Revisión final, deploy, entrega | Todos |

---

## 🔧 Métricas de Evaluación Comunes

Todos los modelos se evalúan con:

- **AUROC** (Area Under ROC Curve) — Discriminación
- **KS Statistic** — Poder de separación
- **Gini Coefficient** — Concentración del riesgo
- **Precisión, Recall, F1-Score** — Exactitud
- **Matriz de Confusión** — Distribución de predicciones
- **Curva ROC** — Visualización
- **PSI (Population Stability Index)** — Estabilidad temporal

**Función común**: `src/utils/evaluation.py`
```python
from src.utils.evaluation import comprehensive_evaluation
metrics = comprehensive_evaluation(y_true, y_pred, y_proba, model_name="M1_LR")
```

---

## 🌳 Convención de Ramas Git

```
main/                          # Rama principal (entrega final)
├── P1/woe-scorecard           # P1 — Regresión Logística
├── P2/xgboost-tuning          # P2 — XGBoost
├── P3/random-forest-features  # P3 — Random Forest
└── P4/neural-network          # P4 — Red Neuronal

develop/ (opcional)            # Rama de integración (si se usa)
```

---

## 💬 Convención de Commits

**Formato**:
```
[P#] type: descripción corta

Descripción detallada (opcional)
- Cambio 1
- Cambio 2
```

**Ejemplos válidos**:
```
[P1] feat: implementar WOE para variables continuas
[P2] feat: tuning hiperparámetros XGBoost con Optuna
[P3] feat: análisis bivariado y selección de features
[P4] feat: normalización de datos para red neuronal
[P4] docs: integrar secciones del informe final
[P1] fix: corregir cálculo de scorecard
[P2] refactor: mejorar modularidad código evaluación
```

**Tipos permitidos**:
- `feat` — Nueva funcionalidad
- `fix` — Corrección de bugs
- `docs` — Documentación
- `refactor` — Mejora de código
- `test` — Tests o validación
- `chore` — Tareas de configuración

---

## 📞 Contacto y Preguntas

Si tienes dudas sobre:
- **Splits y datos**: contactar a P3 (infraestructura)
- **Evaluación común**: contactar a P4 (integración)
- **Modelo X**: contactar a Px responsable

---

## 📄 Licencia

MIT License — Proyecto académico para BBVA México (2026).

---

**Última actualización**: 2026-05-28  
**Estado**: En desarrollo — Setup y utilidades completadas ✓
