#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testa RandomForest sui dati mixed, per vedere la performance su diversi valori di mixing 
delle funzioni di Wigner con il vuoto

@author: marcoascari
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Parametri principali
RESOLUTION = "20x20"
N_ESTIMATORS = 100  # metti il numero giusto
MODEL_FOLDER = f"rf_results/{RESOLUTION}/trees{N_ESTIMATORS}/final_model"
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER, "best_rf_model.pkl")
SCALER_PATH = "scaler.pkl"  # nella cartella corrente

alphas = [0.50, 0.80, 0.90, 0.95, 0.99]
NUM_FAMILIES = 10
#carica i dati delle Wigner mixed
def load_data_from_folder(folder_path):
    X_list = []
    y_list = []
    if not os.path.isdir(folder_path):
        print(f"[!] Cartella non trovata: {folder_path}")
        return None, None
    for family_idx in range(1, NUM_FAMILIES + 1):
        family = f"family_{family_idx}"
        family_path = os.path.join(folder_path, family)
        matrices_path = os.path.join(family_path, "matrices")
        if not os.path.isdir(matrices_path):
            print(f"[!] Cartella non trovata: {matrices_path}")
            return None, None
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

print("Caricamento modello e scaler...")
model = joblib.load(BEST_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

accuracies = []
#testing
for alpha in alphas:
    print(f"\nTesting mix alpha = {alpha:.2f}")
    mix_str = f"mix_{alpha:.2f}"
    base_path = os.path.join("dataset_wigner_mixed", mix_str, RESOLUTION)

    X_test, y_test = load_data_from_folder(base_path)
    if X_test is None or y_test is None:
        print(f"[!] Nessun dato trovato per mix={alpha:.2f}")
        accuracies.append(np.nan)
        continue

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    accuracies.append(acc)
#plot accuracy-mixing
plt.figure(figsize=(8,5))
plt.plot(alphas, accuracies, marker='o')
plt.xlabel("Mix parameter α")
plt.ylabel("Accuracy")
#plt.title("Random Forest accuracy on mixed noisy data")
plt.grid(True)
plt.show()



