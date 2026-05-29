"""
Función común de evaluación de modelos de PI crediticia.
Calcula métricas estándar de riesgo crediticio.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_ks_statistic(y_true, y_proba):
    """
    Calcula el estadístico KS (Kolmogorov-Smirnov).
    Mide la máxima diferencia entre las distribuciones acumuladas de buenos y malos.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)
    return ks


def calculate_gini(y_true, y_proba):
    """
    Calcula el coeficiente de Gini.
    Mide la concentración de riesgo: Gini = 2*AUROC - 1
    """
    auroc = roc_auc_score(y_true, y_proba)
    gini = 2 * auroc - 1
    return gini


def calculate_psi(actual_dist, expected_dist, bins=10):
    """
    Calcula el Population Stability Index (PSI).
    Mide cambios en la distribución entre dos períodos.

    Interpretación:
    - PSI < 0.05: estable
    - 0.05 < PSI < 0.10: pequeño cambio
    - PSI > 0.10: cambio significativo
    """
    def calculate_psi_bin(expected, actual):
        if actual == 0 or expected == 0:
            return 0
        return (actual - expected) * np.log(actual / expected)

    psi = 0
    for i in range(len(actual_dist)):
        psi += calculate_psi_bin(expected_dist[i], actual_dist[i])
    return psi


def comprehensive_evaluation(y_true, y_pred, y_proba, model_name="Model", verbose=True):
    """
    Evaluación comprehensiva de un modelo de PI crediticia.

    Parámetros:
    -----------
    y_true : array-like
        Valores verdaderos (0/1)
    y_pred : array-like
        Predicciones binarias del modelo (0/1)
    y_proba : array-like
        Probabilidades predichas (scores entre 0 y 1)
    model_name : str
        Nombre del modelo para identificar en reportes
    verbose : bool
        Si True, imprime reporte detallado

    Retorna:
    --------
    dict : Diccionario con todas las métricas calculadas
    """

    # Validaciones básicas
    assert len(y_true) == len(y_pred) == len(y_proba), "Las longitudes no coinciden"

    # ==================== MÉTRICAS DE DISCRIMINACIÓN ====================

    # AUROC
    auroc = roc_auc_score(y_true, y_proba)

    # KS Statistic
    ks_stat = calculate_ks_statistic(y_true, y_proba)

    # Gini Coefficient
    gini = calculate_gini(y_true, y_proba)

    # ==================== MÉTRICAS DE CLASIFICACIÓN ====================

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Tasa de Verdaderos Positivos (Recall / Sensitivity)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Tasa de Falsos Positivos
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Precisión
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Especificidad
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # F1-Score
    f1 = f1_score(y_true, y_pred)

    # Exactitud (Accuracy)
    accuracy = (tp + tn) / len(y_true)

    # ==================== COMPOSICIÓN DEL CONJUNTO ====================

    n_total = len(y_true)
    n_bad = np.sum(y_true)
    n_good = n_total - n_bad
    pct_bad = (n_bad / n_total) * 100
    pct_good = (n_good / n_total) * 100

    # ==================== REPORTE TEXTUAL ====================

    if verbose:
        print("\n" + "="*70)
        print(f"EVALUACIÓN DEL MODELO: {model_name}".center(70))
        print("="*70)

        print("\n📊 COMPOSICIÓN DEL CONJUNTO")
        print(f"  Total: {n_total:,} observaciones")
        print(f"  Malos: {n_bad:,} ({pct_bad:.2f}%)")
        print(f"  Buenos: {n_good:,} ({pct_good:.2f}%)")

        print("\n🎯 MÉTRICAS DE DISCRIMINACIÓN")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  KS Statistic: {ks_stat:.4f}")
        print(f"  Gini Coefficient: {gini:.4f}")

        print("\n🔍 MÉTRICAS DE CLASIFICACIÓN")
        print(f"  Precisión: {precision:.4f}")
        print(f"  Recall (Sensibilidad): {tpr:.4f}")
        print(f"  Especificidad: {specificity:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Exactitud: {accuracy:.4f}")

        print("\n📈 MATRIZ DE CONFUSIÓN")
        print(f"  Verdaderos Negativos (TN): {tn:,}")
        print(f"  Falsos Positivos (FP): {fp:,}")
        print(f"  Falsos Negativos (FN): {fn:,}")
        print(f"  Verdaderos Positivos (TP): {tp:,}")

        print("\n" + "="*70)

    # ==================== RETORNO DE MÉTRICAS ====================

    metrics = {
        # Discriminación
        'auroc': auroc,
        'ks_statistic': ks_stat,
        'gini': gini,

        # Clasificación
        'precision': precision,
        'recall': tpr,
        'specificity': specificity,
        'f1_score': f1,
        'accuracy': accuracy,

        # Matriz de confusión
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,

        # Composición
        'n_total': n_total,
        'n_bad': n_bad,
        'n_good': n_good,
        'pct_bad': pct_bad,
        'pct_good': pct_good,

        # FPR y TPR (para gráficas)
        'fpr': fpr,
        'tpr': tpr,
    }

    return metrics


def plot_roc_curve(y_true, y_proba, model_name="Model", ax=None):
    """
    Grafica la curva ROC del modelo.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auroc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUROC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Curva ROC - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    return ax


def plot_confusion_matrix(y_true, y_pred, model_name="Model", ax=None):
    """
    Grafica la matriz de confusión.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Verdadero')
    ax.set_title(f'Matriz de Confusión - {model_name}')
    ax.set_xticklabels(['Bueno (0)', 'Malo (1)'])
    ax.set_yticklabels(['Bueno (0)', 'Malo (1)'])

    return ax


def save_metrics_to_csv(metrics, model_name, filepath):
    """
    Guarda las métricas en un archivo CSV.
    """
    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, 'model_name', model_name)
    metrics_df.to_csv(filepath, index=False)
    print(f"✓ Métricas guardadas en: {filepath}")
