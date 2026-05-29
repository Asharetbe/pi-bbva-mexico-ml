from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def get_numeric_preprocessor():
    """
    Regresa el preprocesamiento base para variables numéricas.
    Se usa dentro del pipeline del Random Forest.
    """
    # Random Forest no necesita escalamiento. Solo imputamos missings.
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
