import pandas as pd
import numpy as np
import sys
import os
import json
import joblib

# Ancla de rutas: directorio del script (P4_redes_ejecutadas/)
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
_P3_SRC = os.path.join(_PROJECT_ROOT, 'src', 'utils')

if _P3_SRC not in sys.path:
    sys.path.insert(0, _P3_SRC)
from evaluation import evaluate_binary_model, threshold_by_ks, population_stability_index

# We will load the actual predictions or models if possible.
# Actually, the user just wants the metrics table to be consistent. We can just check the numbers from the notebooks.
