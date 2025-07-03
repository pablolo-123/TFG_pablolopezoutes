#!/usr/bin/env python
# coding: utf-8

# CONFIGURACIÓN
input_csv = '../datasets/CTMTL.csv'
smiles_column = 'Canonical SMILES'
label_column = 'Toxicity Value'

# IMPORTS
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

#  CARGAR CSV
try:
    df = pd.read_csv(input_csv)
    print(f" Dataset cargado con {len(df)} compuestos")
except Exception as e:
    print(f" Error al cargar el archivo CSV: {e}")

# LIMPIEZA Y SMILES CANÓNICOS
valid_smiles = []
canonical_smiles = []
discarded = 0

for idx, row in df.iterrows():
    smiles = row[smiles_column]
    label = row[label_column]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            valid_smiles.append(smiles)
            canonical_smiles.append(can_smiles)
        else:
            discarded += 1
    except:
        discarded += 1

print(f" {discarded} compuestos descartados por errores de parseo")
df[smiles_column] = pd.Series(canonical_smiles)

#  ELIMINAR CONFLICTOS Y DUPLICADOS
duplicate_stats = df.duplicated(subset=[smiles_column], keep=False)
grouped = df.groupby(smiles_column)[label_column]
conflicts = grouped.nunique() > 1
if conflicts.any():
    problematic_smiles = conflicts[conflicts].index.tolist()
    df = df[~df[smiles_column].isin(problematic_smiles)].copy()
    print(f"❗ {len(problematic_smiles)} compuestos conflictivos eliminados")

df = df.drop_duplicates(subset=[smiles_column], keep='first')
print(f"Tras eliminar duplicados: {len(df)} compuestos únicos")

#  RENOMBRAR COLUMNAS PARA CONSISTENCIA
df['label_multi'] = df[label_column]
df = df.rename(columns={smiles_column: 'SMILES', label_column: 'label_binary'})

#  CROSS-VALIDATION EN TRAINVAL
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(df, df['label_binary']))

for fold, (trainval_idx, test_idx) in enumerate(folds):
    # Este fold será el test
    test_df = df.iloc[test_idx]
    # Los otros 4 folds serán train + valid
    remaining_df = df.iloc[trainval_idx]

    # Dividir train+valid
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=0.2,  # o ajusta según tu criterio
        stratify=remaining_df['label_binary'],
        random_state=42
    )

    # Crear directorios y guardar
    base_path = f'MMGIN/folds/fold_{fold}/'
    os.makedirs(base_path + 'train/raw/', exist_ok=True)
    os.makedirs(base_path + 'val/raw/', exist_ok=True)
    os.makedirs(base_path + 'test/raw/', exist_ok=True)

    train_df.to_csv(base_path + 'train/raw/train_graph_dataset.csv', index=False)
    val_df.to_csv(base_path + 'val/raw/val_graph_dataset.csv', index=False)
    test_df.to_csv(base_path + 'test/raw/test_graph_dataset.csv', index=False)

    print(f"[Fold {fold}] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

#  BORRAR PROCESADOS ANTIGUOS SI EXISTEN
for folder in ['train', 'val', 'test']:
    try:
        os.remove(f'MMGIN/feng/{folder}/processed/process.pt')
    except FileNotFoundError:
        pass
