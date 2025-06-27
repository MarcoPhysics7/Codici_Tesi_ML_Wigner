#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testa MLP sui dati con rumore gaussiano sullo spazio delle fasi, per vedere la performance su diversi valori di  
deviazioni standard sui punti dello spazio delle fasi

@author: marcoascari
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# -CONFIGURAZIONE PARAMETRI-
resolution = "20x20"
model_path = f"mlp_results/{resolution}/hidden10/final_model/best_final_model.pkl"
scaler_path = "scaler.pkl"  # Lo scaler salvato nella cartella corrente
dataset_base_path = "dataset_wigner_rumore_gaussiano"

# -Carica il modello MLP salvato-
model = joblib.load(model_path)

# -Carica lo scaler-
scaler = joblib.load(scaler_path)

# -Prepara array per il plot-
std_values = []
accuracies = []

# -Cicla sulle cartelle devstd_X-
for folder_name in sorted(os.listdir(dataset_base_path)):
   

    if not folder_name.startswith("devstd_"):
        continue

    try:
        std_str = folder_name.split("_")[1]  # estrae la stringa dopo "devstd_"
        dev_std = float(std_str)
    except ValueError:
        print(f"[!] Nome cartella non valido: {folder_name}")
        continue

    path_20x20 = os.path.join(dataset_base_path, folder_name, resolution)
    if not os.path.exists(path_20x20):
        print(f"[!] Risoluzione {resolution} non trovata in {folder_name}")
        continue

    X, y = [], []
    for family_id in range(1, 11):
        family_path = os.path.join(path_20x20, f"family_{family_id}", "matrices")
        if not os.path.exists(family_path):
            continue

        for fname in os.listdir(family_path):
            if fname.endswith(".npy"):
                mat = np.load(os.path.join(family_path, fname))
                X.append(mat.flatten())
                y.append(family_id - 1)

    if len(X) == 0:
        print(f"[!] Nessun dato trovato per dev_std={dev_std}")
        continue

    X = np.array(X)
    y = np.array(y)

    # Usa lo scaler già salvato
    X = scaler.transform(X)

    # Predizione e accuracy
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    std_values.append(dev_std)
    print(f"Dev. Std. = {dev_std:.4f} | Accuracy = {acc:.4f}")


# - Plot -
if std_values:
    # Ordina per dev_std crescente
    std_values, accuracies = zip(*sorted(zip(std_values, accuracies)))

    plt.figure(figsize=(8, 5))
    plt.plot(std_values, accuracies, marker='o')
    plt.xlabel("Deviazione standard del rumore (Gaussian noise on phase space)")
    plt.ylabel("Accuracy")
    plt.title(f"MLP accuracy vs Gaussian noise std ({resolution})")
    plt.grid(True)
    plt.xscale("log")
    plt.tight_layout()
    plt.show()
else:
    print("[!] Nessun dato da plottare.")

