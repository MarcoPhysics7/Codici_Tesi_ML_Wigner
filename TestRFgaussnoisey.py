#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testa Random Forest sui dati con rumore gaussiano sullo spazio delle fasi, per vedere la performance su diversi valori di  
deviazioni standard sui punti dello spazio delle fasi

@author: marcoascari
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# - CONFIGURAZIONE PARAMETRI -
RESOLUTION = "20x20"
N_ESTIMATORS = 100  # imposta il valore corretto se diverso
MODEL_FOLDER = f"rf_results/{RESOLUTION}/trees{N_ESTIMATORS}/final_model"
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER, "best_rf_model.pkl")
SCALER_PATH = "scaler.pkl"  # Lo scaler salvato nella cartella corrente
DATASET_BASE_PATH = "dataset_wigner_rumore_gaussiano"
NUM_FAMILIES = 10

def load_data_from_folder(folder_path):
    X_list = []
    y_list = []
    if not os.path.isdir(folder_path):
        print(f"[!] Cartella non trovata: {folder_path}")
        return None, None
    for family_idx in range(1, NUM_FAMILIES + 1):
        matrices_path = os.path.join(folder_path, f"family_{family_idx}", "matrices")
        if not os.path.isdir(matrices_path):
            print(f"[!] Cartella non trovata: {matrices_path}")
            continue
        files = sorted([f for f in os.listdir(matrices_path) if f.endswith(".npy")])
        for f in files:
            mat = np.load(os.path.join(matrices_path, f))
            X_list.append(mat.flatten())
            y_list.append(family_idx - 1)
    if len(X_list) == 0:
        return None, None
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

# - Caricamento modello e scaler -
print("Caricamento modello e scaler...")
model = joblib.load(BEST_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# - Loop su cartelle devstd_X -
std_values = []
accuracies = []

for folder_name in sorted(os.listdir(DATASET_BASE_PATH)):
    if not folder_name.startswith("devstd_"):
        continue

    try:
        std_str = folder_name.split("_")[1]
        dev_std = float(std_str)
    except ValueError:
        print(f"[!] Nome cartella non valido: {folder_name}")
        continue

    print(f"\nTesting Gaussian noise std = {dev_std:.4f}")
    base_path = os.path.join(DATASET_BASE_PATH, folder_name, RESOLUTION)

    X_test, y_test = load_data_from_folder(base_path)
    if X_test is None or y_test is None:
        print(f"[!] Nessun dato trovato per std={dev_std:.3f}")
        accuracies.append(np.nan)
        continue

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    std_values.append(dev_std)
    accuracies.append(acc)

# - Plot -
if std_values:
    std_values, accuracies = zip(*sorted(zip(std_values, accuracies)))

    plt.figure(figsize=(8, 5))
    plt.plot(std_values, accuracies, marker='o')
    plt.xlabel("Deviazione standard del rumore (Gaussian noise on phase space)")
    plt.ylabel("Accuracy")
    plt.title(f"Random Forest accuracy vs Gaussian noise std ({RESOLUTION})")
    plt.grid(True)
    plt.xscale("log")
    plt.tight_layout()
    plt.show()
else:
    print("[!] Nessun dato da plottare.")
