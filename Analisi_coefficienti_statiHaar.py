#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi dei coefficienti per testare la loro dispersione randomica
Crea un grafico del valor medio del modulo quadro del coefficiente c_n in funzione dell'indice (n) per ogni famiglia
ricreando la dispersione per la famiglia N-esima <|c_N|^2> circa = 1/(N+1)

@author: marcoascari
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# - PARAMETRI -
dataset_root = "dataset_wigner"
res_reference = "10x10"  # analizziamo solo una risoluzione (basta 100x100)
families = range(1, 11)

# - FUNZIONI -
def load_coeffs_from_file(filepath):
    coeffs = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(":")[1].split("+")
            real = float(parts[0].strip())
            imag = float(parts[1].replace("j", "").strip())
            coeffs.append(real**2 + imag**2)
    return coeffs

# - CARICAMENTO COEFFICIENTI -
family_means = defaultdict(list)

for n_max in families:
    print(f"Analizzando family_{n_max}...")
    coeffs_path = os.path.join(dataset_root, res_reference, f"family_{n_max}", "matrices")
    all_coeffs = []

    for file in sorted(os.listdir(coeffs_path)):
        if file.endswith("_coeffs.txt"):
            coeff_path = os.path.join(coeffs_path, file)
            coeff_sq = load_coeffs_from_file(coeff_path)
            all_coeffs.append(coeff_sq)

    all_coeffs = np.array(all_coeffs)  # shape: (num_states, num_coeffs)
    mean_per_n = all_coeffs.mean(axis=0)
    family_means[n_max] = mean_per_n

# - GRAFICO -
plt.figure(figsize=(10, 6))
for n_max, means in family_means.items():
    plt.plot(range(len(means)), means, label=f"Family {n_max}")

plt.xlabel("n (indice coefficiente $|n⟩$)")
plt.ylabel(r"Media $\langle |c_n|^2 \rangle$")
plt.title("Distribuzione media dei coefficienti $|c_n|^2$ per famiglia")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
output_plot_dir = os.path.join(dataset_root, res_reference, "plots")
os.makedirs(output_plot_dir, exist_ok=True)
plt.savefig(os.path.join(output_plot_dir, "media_coeffs_per_famiglia.png"), dpi=300)

plt.show()
