"""
Generador de splits estratificados para el proyecto PI BBVA México.
Crea train (70%), test (20%) y OOT (10%) con stratificación en el target.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def generate_stratified_splits(
    input_file,
    output_dir,
    target_col='target',
    train_size=0.70,
    test_size=0.20,
    oot_size=0.10,
    random_state=42,
    verbose=True
):
    """
    Genera splits estratificados de datos para PI crediticia.

    Parámetros:
    -----------
    input_file : str
        Ruta al archivo de datos original (Excel o CSV)
    output_dir : str
        Directorio donde guardar los splits
    target_col : str
        Nombre de la columna target (default: 'target')
    train_size : float
        Proporción para entrenamiento (default: 0.70)
    test_size : float
        Proporción para test (default: 0.20)
    oot_size : float
        Proporción para OOT (default: 0.10)
    random_state : int
        Seed para reproducibilidad (default: 42)
    verbose : bool
        Si True, imprime información del proceso

    Retorna:
    --------
    dict : Diccionario con los tres DataFrames (train, test, oot)
    """

    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "="*70)
        print("GENERADOR DE SPLITS ESTRATIFICADOS".center(70))
        print("="*70)
        print(f"\n📂 Leyendo archivo: {input_file}")

    # Leer datos
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    n_original = len(df)
    if verbose:
        print(f"✓ Datos cargados: {n_original:,} filas × {len(df.columns)} columnas")

    # Verificar que el target existe
    if target_col not in df.columns:
        raise ValueError(f"Columna '{target_col}' no encontrada. Columnas disponibles: {list(df.columns)}")

    # Validar proporciones
    if not np.isclose(train_size + test_size + oot_size, 1.0):
        raise ValueError(f"Las proporciones no suman 1.0: {train_size + test_size + oot_size}")

    # Estadísticas iniciales
    n_bad = (df[target_col] == 1).sum()
    n_good = (df[target_col] == 0).sum()
    pct_bad = (n_bad / n_original) * 100

    if verbose:
        print(f"\n📊 Composición original:")
        print(f"  Malos (target=1): {n_bad:,} ({pct_bad:.2f}%)")
        print(f"  Buenos (target=0): {n_good:,} ({100-pct_bad:.2f}%)")

    # Paso 1: Separar train+test (90%) de OOT (10%)
    df_train_test, df_oot = train_test_split(
        df,
        test_size=oot_size,
        stratify=df[target_col],
        random_state=random_state
    )

    # Paso 2: Separar train (70%) de test (20%) dentro de train_test
    # Proporción ajustada: test_size_ajustado = 0.20 / 0.90 ≈ 0.222
    test_size_adjusted = test_size / (train_size + test_size)
    df_train, df_test = train_test_split(
        df_train_test,
        test_size=test_size_adjusted,
        stratify=df_train_test[target_col],
        random_state=random_state
    )

    # Verificar proporciones
    n_train = len(df_train)
    n_test = len(df_test)
    n_oot = len(df_oot)

    if verbose:
        print(f"\n✂️  Splits generados:")
        print(f"  Train (70%): {n_train:,} filas ({n_train/n_original*100:.1f}%)")
        print(f"    - Malos: {(df_train[target_col]==1).sum():,}")
        print(f"    - Buenos: {(df_train[target_col]==0).sum():,}")
        print(f"  Test (20%): {n_test:,} filas ({n_test/n_original*100:.1f}%)")
        print(f"    - Malos: {(df_test[target_col]==1).sum():,}")
        print(f"    - Buenos: {(df_test[target_col]==0).sum():,}")
        print(f"  OOT (10%): {n_oot:,} filas ({n_oot/n_original*100:.1f}%)")
        print(f"    - Malos: {(df_oot[target_col]==1).sum():,}")
        print(f"    - Buenos: {(df_oot[target_col]==0).sum():,}")

    # Guardar splits
    output_path = Path(output_dir)

    train_file = output_path / f"train_70_seed{random_state}.csv"
    test_file = output_path / f"test_20_seed{random_state}.csv"
    oot_file = output_path / f"oot_10_seed{random_state}.csv"

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)
    df_oot.to_csv(oot_file, index=False)

    if verbose:
        print(f"\n💾 Archivos guardados:")
        print(f"  ✓ {train_file}")
        print(f"  ✓ {test_file}")
        print(f"  ✓ {oot_file}")
        print(f"\nℹ️  Usa random_state={random_state} en todos tus modelos para reproducibilidad")
        print("="*70 + "\n")

    return {
        'train': df_train,
        'test': df_test,
        'oot': df_oot
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera splits estratificados para proyecto PI BBVA'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Ruta al archivo de datos (Excel o CSV)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/',
        help='Directorio de salida para los splits'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='target',
        help='Nombre de la columna target'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed para reproducibilidad'
    )

    args = parser.parse_args()

    generate_stratified_splits(
        input_file=args.input,
        output_dir=args.output,
        target_col=args.target,
        random_state=args.seed,
        verbose=True
    )
