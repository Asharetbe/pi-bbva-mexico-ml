import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def ks_univariado(x, y):
    """
    Calcula el KS usando una sola variable como score.
    """
    datos = pd.DataFrame({"x": x, "y": y}).dropna()

    if datos["x"].nunique() <= 1:
        return np.nan

    datos = datos.sort_values("x", ascending=False)

    total_malos = datos["y"].sum()
    total_buenos = len(datos) - total_malos

    if total_malos == 0 or total_buenos == 0:
        return np.nan

    datos["malos_acum"] = datos["y"].cumsum() / total_malos
    datos["buenos_acum"] = (1 - datos["y"]).cumsum() / total_buenos
    datos["ks"] = abs(datos["malos_acum"] - datos["buenos_acum"])

    return float(datos["ks"].max())


def woe_table_variable(x, y, variable, n_bins=10):
    """
    Arma la tabla WOE de una variable.
    Los missings se dejan como un bin separado.
    """
    datos = pd.DataFrame({"x": x, "y": y})
    sin_missing = datos[datos["x"].notna()].copy()
    con_missing = datos[datos["x"].isna()].copy()

    if sin_missing["x"].nunique() <= 1:
        return pd.DataFrame()

    try:
        bins = pd.qcut(sin_missing["x"], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    sin_missing["bin_label"] = bins.astype(str)
    sin_missing["bin_order"] = bins.cat.codes

    if len(con_missing) > 0:
        con_missing["bin_label"] = "missing"
        con_missing["bin_order"] = int(sin_missing["bin_order"].max()) + 1

    datos_bineados = pd.concat([sin_missing, con_missing], axis=0)

    tabla = datos_bineados.groupby(["bin_order", "bin_label"], observed=False)["y"].agg(
        n="count",
        malos="sum",
    ).reset_index()
    tabla["buenos"] = tabla["n"] - tabla["malos"]
    tabla["bad_rate"] = tabla["malos"] / tabla["n"]

    total_malos = tabla["malos"].sum()
    total_buenos = tabla["buenos"].sum()

    if total_malos == 0 or total_buenos == 0:
        return pd.DataFrame()

    tabla["dist_malos"] = tabla["malos"] / total_malos
    tabla["dist_buenos"] = tabla["buenos"] / total_buenos

    eps = 0.000001
    tabla["woe"] = np.log((tabla["dist_buenos"] + eps) / (tabla["dist_malos"] + eps))
    tabla["iv_parcial"] = (tabla["dist_buenos"] - tabla["dist_malos"]) * tabla["woe"]
    tabla.insert(0, "variable", variable)

    return tabla.sort_values(["variable", "bin_order"]).reset_index(drop=True)


def iv_por_cuantiles(x, y, variable, n_bins=10):
    """Calcula el IV total de una variable a partir de su tabla WOE."""
    tabla = woe_table_variable(x, y, variable, n_bins=n_bins)
    if tabla.empty:
        return np.nan
    return float(tabla["iv_parcial"].sum())


def es_monotono_woe(tabla_woe):
    """
    Verifica si el WOE es monotono (creciente o decreciente) ignorando el bin de missings.
    Retorna True si es monotono, False si no lo es, np.nan si no hay suficientes bins.
    """
    sin_missing = tabla_woe[tabla_woe["bin_label"] != "missing"].copy()
    sin_missing = sin_missing.sort_values("bin_order")

    if len(sin_missing) < 2:
        return np.nan

    diferencias = np.diff(sin_missing["woe"].values)
    es_creciente = bool(np.all(diferencias >= 0))
    es_decreciente = bool(np.all(diferencias <= 0))
    return es_creciente or es_decreciente


def auc_univariado(x, y):
    """
    Calcula AUC con una sola variable.
    Usa max(AUC, 1-AUC) para no castigar variables con relacion negativa.
    """
    filas_validas = pd.Series(x).notna()
    x_valido = pd.Series(x)[filas_validas]
    y_valido = pd.Series(y)[filas_validas]

    if x_valido.nunique() <= 1 or y_valido.nunique() <= 1:
        return np.nan

    try:
        auc_raw = roc_auc_score(y_valido, x_valido)
        return float(max(auc_raw, 1 - auc_raw))
    except ValueError:
        return np.nan


def categoria_iv(iv):
    """Clasifica el IV en rangos faciles de interpretar."""
    if pd.isna(iv):
        return "No calculable"
    if iv < 0.02:
        return "Sin poder predictivo"
    if iv < 0.10:
        return "Debil"
    if iv < 0.30:
        return "Medio"
    if iv < 0.50:
        return "Fuerte"
    return "Muy fuerte / revisar posible leakage"


def comentario_iv(iv):
    """Genera un comentario automatico segun el valor del IV."""
    if pd.isna(iv):
        return "Variable no evaluable por baja variabilidad o problemas de binning."
    if iv >= 0.50:
        return "Variable altamente predictiva; revisar posible leakage."
    if iv >= 0.10:
        return "Variable candidata para modelacion."
    return "Bajo poder predictivo individual."


def resumen_bivariado(X, y, diccionario=None, n_bins=10):
    """
    Calcula missing, IV, KS, AUC univariado y guarda tambien el detalle WOE.
    Regresa dos tablas: resumen por variable y WOE por bin.
    """
    resultados = []
    tablas_woe = []

    for variable in X.columns:
        x = X[variable]
        tabla_woe = woe_table_variable(x, y, variable, n_bins=n_bins)
        if not tabla_woe.empty:
            tablas_woe.append(tabla_woe)

        resultados.append({
            "variable": variable,
            "missing_rate": x.isna().mean(),
            "n_unique": x.nunique(dropna=True),
            "iv": np.nan if tabla_woe.empty else tabla_woe["iv_parcial"].sum(),
            "ks": ks_univariado(x, y),
            "auc_univariado": auc_univariado(x, y),
            "woe_monotone": np.nan if tabla_woe.empty else es_monotono_woe(tabla_woe),
        })

    resumen = pd.DataFrame(resultados)

    if diccionario is not None:
        resumen = resumen.merge(diccionario, left_on="variable", right_on="Variable", how="left")
        resumen = resumen.rename(columns={
            "Descripcion": "descripcion",
            "Descripción": "descripcion",
            "Correlacion Esperada": "correlacion_esperada",
        })

    resumen["categoria_iv"] = resumen["iv"].apply(categoria_iv)
    resumen["comentario"] = resumen["iv"].apply(comentario_iv)
    resumen = resumen.sort_values("iv", ascending=False).reset_index(drop=True)

    columnas = [
        "variable",
        "descripcion",
        "correlacion_esperada",
        "missing_rate",
        "n_unique",
        "iv",
        "ks",
        "auc_univariado",
        "woe_monotone",
        "categoria_iv",
        "comentario",
    ]
    columnas = [col for col in columnas if col in resumen.columns]
    resumen = resumen[columnas]

    woe = pd.concat(tablas_woe, ignore_index=True) if tablas_woe else pd.DataFrame()

    return resumen, woe
