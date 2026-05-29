"""
Configuración global del proyecto PI BBVA México.
Define paths, semillas, y parámetros comunes.
"""

from pathlib import Path

# ==================== PATHS ====================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
DATA_METADATA_DIR = PROJECT_ROOT / 'data' / 'metadata'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# ==================== DATOS ====================

RANDOM_STATE = 42
TARGET_COL = 'target'
TRAIN_SIZE = 0.70
TEST_SIZE = 0.20
OOT_SIZE = 0.10

# Archivos de datos
TRAIN_FILE = DATA_PROCESSED_DIR / f'train_{int(TRAIN_SIZE*100)}_seed{RANDOM_STATE}.csv'
TEST_FILE = DATA_PROCESSED_DIR / f'test_{int(TEST_SIZE*100)}_seed{RANDOM_STATE}.csv'
OOT_FILE = DATA_PROCESSED_DIR / f'oot_{int(OOT_SIZE*100)}_seed{RANDOM_STATE}.csv'
DICT_FILE = DATA_METADATA_DIR / 'diccionario_completo.csv'

# ==================== MODELOS ====================

MODELS = {
    'M1_logistic_regression': {
        'owner': 'P1',
        'file': MODELS_DIR / 'M1_logistic_regression.pkl',
        'description': 'Regresión Logística con WOE'
    },
    'M2_xgboost': {
        'owner': 'P2',
        'file': MODELS_DIR / 'M2_xgboost.json',
        'description': 'XGBoost / Gradient Boosting'
    },
    'M3_random_forest': {
        'owner': 'P3',
        'file': MODELS_DIR / 'M3_random_forest.pkl',
        'description': 'Random Forest'
    },
    'M4_neural_network': {
        'owner': 'P4',
        'file': MODELS_DIR / 'M4_neural_network.h5',
        'description': 'Red Neuronal MLP'
    }
}

# ==================== MÉTRICAS ====================

METRICS_TO_CALCULATE = [
    'auroc',
    'ks_statistic',
    'gini',
    'precision',
    'recall',
    'specificity',
    'f1_score',
    'accuracy'
]

# ==================== HIPERPARÁMETROS ====================

# Para tuning con Optuna
OPTUNA_CONFIG = {
    'n_trials': 50,
    'timeout': 600,  # segundos
    'show_progress_bar': True
}

# Random Forest
RF_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# XGBoost
XGB_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE
}

# Regresión Logística
LR_DEFAULT_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'solver': 'lbfgs'
}

# Red Neuronal
NN_DEFAULT_PARAMS = {
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'batch_size': 32,
    'epochs': 50,
    'dropout_rate': 0.3,
    'learning_rate': 0.001
}

# ==================== INTEGRANTES ====================

INTEGRANTES = {
    'P1': {
        'nombre': 'Participante 1',
        'modelo': 'M1 - Regresión Logística',
        'secciones': ['§2.1 Problemática', '§3 Conclusiones']
    },
    'P2': {
        'nombre': 'Participante 2',
        'modelo': 'M2 - XGBoost',
        'secciones': ['§1.1 BBVA México', '§2.4 Comparación de modelos']
    },
    'P3': {
        'nombre': 'Participante 3',
        'modelo': 'M3 - Random Forest',
        'secciones': ['§2.2.2 Análisis bivariado']
    },
    'P4': {
        'nombre': 'Participante 4',
        'modelo': 'M4 - Red Neuronal MLP',
        'secciones': ['§2.2.1 Análisis univariado']
    }
}

# ==================== VALIDACIONES ====================

def validate_config():
    """Valida que todos los paths y configuraciones sean válidas."""
    # Validar que las proporciones sumen 1
    total = TRAIN_SIZE + TEST_SIZE + OOT_SIZE
    assert abs(total - 1.0) < 0.001, f"Proporciones no suman 1: {total}"

    # Validar que los directorios existan o puedan crearse
    for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("✓ Configuración validada correctamente")


if __name__ == '__main__':
    validate_config()
