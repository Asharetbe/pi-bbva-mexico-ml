# Project Charter - PI BBVA México

## Información General

**Nombre del Proyecto**: Estimación de Probabilidad de Incumplimiento (PI) Crediticia - BBVA México

**Objetivo Principal**: Desarrollar e implementar cuatro modelos de machine learning para estimar la probabilidad de incumplimiento de clientes BBVA México.

**Tipo de Proyecto**: Académico — Entrega: 30 de mayo de 2026

**Repositorio**: https://github.com/Asharetbe/pi-bbva-mexico-ml.git

---

## Alcance

### Dentro del alcance
✓ Desarrollar 4 modelos: LR, XGBoost, RF, MLP
✓ Análisis exploratorio de datos (EDA)
✓ Preprocesamiento y feature engineering
✓ Validación en train, test, OOT
✓ Métricas estándar de riesgo crediticio (AUROC, KS, Gini)
✓ Informe final integrado (PDF + Word + notebooks)

### Fuera del alcance
✗ Implementación en producción
✗ API de predicción en tiempo real
✗ Dashboard interactivo
✗ Monitoring continuo

---

## Integrantes y Responsabilidades

### P1 - Modelo 1: Regresión Logística con WOE
- **Modelo**: Regresión Logística + Weight of Evidence + Scorecard
- **Notebooks**:
  - 01_P1_EDA_LR.ipynb
  - 02_P1_WOE_Engineering.ipynb
  - 03_P1_Model_Training.ipynb
  - 04_P1_Scorecard.ipynb
- **Secciones del informe**:
  - §2.1 Problemática (definición de default, contexto crediticio)
  - §3 Conclusiones (hallazgos principales)
- **Rama Git**: `P1/woe-scorecard`
- **Modelo guardado**: `models/M1_logistic_regression.pkl`

### P2 - Modelo 2: XGBoost / Gradient Boosting
- **Modelo**: XGBoost con tuning de hiperparámetros (Optuna)
- **Notebooks**:
  - 01_P2_EDA_XGB.ipynb
  - 02_P2_Feature_Selection.ipynb
  - 03_P2_Hyperparameter_Tuning.ipynb
  - 04_P2_Model_Evaluation.ipynb
- **Secciones del informe**:
  - §1.1 BBVA México (contexto de la institución)
  - §2.4 Comparación de modelos (tabla comparativa M1-M4)
- **Rama Git**: `P2/xgboost-tuning`
- **Modelo guardado**: `models/M2_xgboost.json`

### P3 - Modelo 3: Random Forest
- **Modelo**: Random Forest con tuning
- **Notebooks**:
  - 01_P3_EDA_RF.ipynb
  - 02_P3_Bivariate_Analysis.ipynb
  - 03_P3_Model_Training.ipynb
  - 04_P3_Feature_Importance.ipynb
- **Secciones del informe**:
  - §2.2.2 Análisis bivariado (correlaciones, relaciones entre features)
  - Infraestructura de splits (§2.3 o referencia)
- **Rama Git**: `P3/random-forest-features`
- **Modelo guardado**: `models/M3_random_forest.pkl`

### P4 - Modelo 4: Red Neuronal MLP
- **Modelo**: Red Neuronal Multicapa con TensorFlow/Keras
- **Notebooks**:
  - 01_P4_EDA_NN.ipynb
  - 02_P4_Univariate_Analysis.ipynb
  - 03_P4_Data_Normalization.ipynb
  - 04_P4_Model_Training.ipynb
- **Secciones del informe**:
  - §2.2.1 Análisis univariado (distribuciones, estadísticas por variable)
  - **Integración del documento final** (compilar y editar todas las secciones)
- **Rama Git**: `P4/neural-network`
- **Modelo guardado**: `models/M4_neural_network.h5`
- **Rol especial**: Integrador (merge de PRs, compilación del informe)

---

## Base de Datos

**Archivo**: `BasePI.xlsx`
- **Observaciones**: 10,000
- **Features**: 120 variables (x1 - x120)
- **Target**: Variable binaria (0=Buen pagador, 1=Incumplidor)
- **Distribución**: 8% incumplimiento (800 malos / 9,200 buenos)
- **Estratificación**: Mantener 8% en cada split

**Splits** (estratificados, seed=42):
- **Train**: 70% (7,000 obs)
- **Test**: 20% (2,000 obs)
- **OOT**: 10% (1,000 obs — Out-of-Time)

**Ubicación después de procesamiento**:
- `data/processed/train_70_seed42.csv`
- `data/processed/test_20_seed42.csv`
- `data/processed/oot_10_seed42.csv`

---

## Métricas de Evaluación (estándar)

Todos los modelos se evalúan con:

| Métrica | Interpretación |
|---------|----------------|
| **AUROC** | Área bajo curva ROC (0-1, mayor es mejor) |
| **KS Statistic** | Máxima separación entre buenos y malos (0-1) |
| **Gini Coefficient** | Concentración de riesgo: Gini = 2×AUROC - 1 |
| **Precisión** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | Media armónica de precisión y recall |
| **PSI** | Population Stability Index (cambio en distribución) |

**Función común**: `src/utils/evaluation.py:comprehensive_evaluation()`

---

## Cronograma

### Semana 1 (26-27 May)
- **May 26 (Miércoles)**: Setup repositorio, splits, utilidades comunes
  - Crear estructura de carpetas ✓
  - Generar splits (70/20/10) ✓
  - Crear evaluation.py y splits.py ✓
  - Plantillas de notebooks ✓

- **May 27 (Jueves)**: Primeros modelos
  - **P1**: Entrenar M1 (Logística) → PR a main
  - **P2**: Entrenar M2 (XGBoost) → PR a main

### Semana 2 (28-29 May)
- **May 28 (Viernes)**: Modelos finales
  - **P3**: Entrenar M3 (Random Forest) → PR a main
  - **P4**: Entrenar M4 (Red Neuronal) → PR a main

- **May 29 (Sábado)**: Compilación del informe
  - **P4**: Integrar todas las secciones
  - Crear documento final (PDF + Word)
  - Revisar referencias y formato

### Semana 3 (30 May)
- **May 30 (Domingo)**: Entrega final ✅
  - Último push a main
  - Archivo `Informe_Final_PI_BBVA.pdf`
  - Archivo `Informe_Final_PI_BBVA.docx`
  - Todos los notebooks reproducibles

---

## Dependencias y Stack Técnico

**Lenguaje**: Python 3.9+

**Librerías principales**:
- `pandas`, `numpy` — Manipulación de datos
- `scikit-learn` — ML estándar
- `xgboost` — Gradient Boosting
- `tensorflow`, `keras` — Redes neuronales
- `optuna` — Tuning de hiperparámetros
- `scorecardpy` — WOE y scorecards
- `matplotlib`, `seaborn` — Visualización

**Gestión de versiones**: Git + GitHub

**Documentación**: Markdown + Jupyter Notebooks

---

## Criterios de Éxito

### Técnicos
✓ Todos los 4 modelos entrenados correctamente
✓ Métricas AUROC > 0.7 en test (mínimo aceptable)
✓ Validación cruzada train/test/OOT sin overfitting severo
✓ Código reproducible con seed=42
✓ Splits estratificados correctamente (8% en cada uno)

### Documentación
✓ Informe final integrado (PDF + Word)
✓ Todos los notebooks ejecutables
✓ README.md con instrucciones completas
✓ Commit history limpia con buenos mensajes

### Entrega
✓ Repositorio actualizado en main
✓ Todos los archivos versionados en Git
✓ Fecha de entrega: 30 de mayo de 2026

---

## Restricciones y Supuestos

### Restricciones
- Fecha límite: 30 de mayo de 2026 (no negociable)
- Base de datos fixed (no hay nuevos datos)
- 4 integrantes (roles no intercambiables)
- Python como lenguaje único

### Supuestos
- Se dispondrá de BasePI.xlsx en tiempo
- Entorno Python ya instalado
- Conexión a GitHub disponible
- Colaboración sin conflictos de código

---

## Riesgos Identificados

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|-----------|
| Conflictos en merge | Media | Alto | Usar ramas separadas + rebase |
| Modelos con bajo desempeño | Media | Medio | Probar múltiples arquitecturas |
| Datos inconsistentes | Baja | Alto | Usar splits.py centralizado |
| Entrega incompleta | Baja | Crítico | Cronograma claro + checkpoints |

---

## Plan de Comunicación

**Reuniones**: Diarias (10 min, mañana)
- Status de cada integrante
- Bloqueadores
- Preguntas

**Canal de comunicación**: 
- Comentarios en PRs para discusión técnica
- Chat/email para urgencias

**Documentación compartida**:
- README.md (punto de entrada)
- docs/ (detalles técnicos)
- GitHub Issues (si es necesario)

---

## Definición de "Hecho" (Definition of Done)

Un modelo se considera **completo** cuando:

1. ✓ Notebook(s) ejecutable(s) sin errores
2. ✓ Modelo entrenado y guardado
3. ✓ Evaluación completa (AUROC, KS, Gini, etc.)
4. ✓ Métricas guardadas en CSV
5. ✓ Gráficas generadas (ROC, Confusión, Feature Importance)
6. ✓ Predicciones guardadas en CSV
7. ✓ PR creado y aprobado
8. ✓ Merged a main
9. ✓ Secciones del informe escritas
10. ✓ Commit history limpia

---

## Aprobación del Charter

| Rol | Nombre | Firma | Fecha |
|-----|--------|-------|-------|
| P1 | | | |
| P2 | | | |
| P3 | | | |
| P4 | | 2026-05-28 | |

---

**Estado**: Activo ✓
**Versión**: 1.0
**Última actualización**: 2026-05-28
