import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)


def ks_statistic(y_true, y_score):
    # Tabla ordenada de mayor a menor probabilidad estimada.
    data = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_score,
    }).sort_values("y_score", ascending=False)

    total_bad = data["y_true"].sum()
    total_good = len(data) - total_bad

    if total_bad == 0 or total_good == 0:
        return np.nan

    # KS = máxima diferencia entre malos acumulados y buenos acumulados.
    data["cum_bad"] = data["y_true"].cumsum() / total_bad
    data["cum_good"] = (1 - data["y_true"]).cumsum() / total_good
    data["ks"] = np.abs(data["cum_bad"] - data["cum_good"])

    return data["ks"].max()


def threshold_by_ks(y_true, y_score):
    # Este umbral se debe calcular solo con train.
    data = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_score,
    }).sort_values("y_score", ascending=False)

    total_bad = data["y_true"].sum()
    total_good = len(data) - total_bad

    if total_bad == 0 or total_good == 0:
        raise ValueError("No se puede calcular KS sin ambas clases en y_true.")

    data["cum_bad"] = data["y_true"].cumsum() / total_bad
    data["cum_good"] = (1 - data["y_true"]).cumsum() / total_good
    data["ks"] = np.abs(data["cum_bad"] - data["cum_good"])

    best_row = data.loc[data["ks"].idxmax()]
    return float(best_row["y_score"])


def evaluate_binary_model(y_true, y_score, threshold):
    """
    Calcula las métricas principales del modelo binario.
    Recibe probabilidades y un threshold ya elegido en train.
    """
    # Convierte probabilidades a 0/1 usando el mismo threshold para todos los splits.
    y_pred = (y_score >= threshold).astype(int)

    # Métricas de discriminación del modelo.
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    ks = ks_statistic(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # Métricas usando el threshold elegido.
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Matriz de confusión: verdaderos/falsos positivos y negativos.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def psi_table(expected_score, actual_score, n_bins=10):
    """
    Calcula una tabla de PSI usando los cortes del score esperado.
    En este proyecto, expected_score normalmente es train y actual_score es test u oos.
    """
    expected = pd.Series(expected_score).dropna()
    actual = pd.Series(actual_score).dropna()

    if expected.nunique() <= 1:
        raise ValueError("No se puede calcular PSI si el score esperado no tiene variacion.")

    _, bins = pd.qcut(expected, q=n_bins, retbins=True, duplicates="drop")
    bins = np.unique(bins)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_bin = pd.cut(expected, bins=bins, include_lowest=True)
    actual_bin = pd.cut(actual, bins=bins, include_lowest=True)

    tabla = pd.DataFrame({
        "score_bin": expected_bin.cat.categories.astype(str),
        "expected_n": expected_bin.value_counts(sort=False).values,
        "actual_n": actual_bin.value_counts(sort=False).values,
    })

    eps = 0.000001
    tabla["expected_pct"] = tabla["expected_n"] / tabla["expected_n"].sum()
    tabla["actual_pct"] = tabla["actual_n"] / tabla["actual_n"].sum()
    tabla["psi_component"] = (
        (tabla["actual_pct"] - tabla["expected_pct"])
        * np.log((tabla["actual_pct"] + eps) / (tabla["expected_pct"] + eps))
    )

    return tabla


def population_stability_index(expected_score, actual_score, n_bins=10):
    """Regresa el PSI total entre dos distribuciones de score."""
    return float(psi_table(expected_score, actual_score, n_bins)["psi_component"].sum())


def psi_interpretation(psi):
    """Clasificacion practica del PSI."""
    if pd.isna(psi):
        return "No calculable"
    if psi < 0.10:
        return "Estable"
    if psi < 0.25:
        return "Cambio moderado"
    return "Cambio fuerte"
