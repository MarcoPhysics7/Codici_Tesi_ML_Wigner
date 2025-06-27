#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crea dataset di test, per testare i modelli con rumore gaussiano
sullo spazio delle fasi per diverse deviazioni standard

@author: marcoascari


"""

import os
import numpy as np
from qutip import Qobj, wigner
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# - CONFIGURAZIONE -
INPUT_DATASET_DIR = "dataset_wigner"
OUTPUT_DATASET_DIR = "dataset_wigner_rumore_gaussiano"
RESOLUTIONS = [10, 20, 50, 70, 100]
DEV_STDS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
SEED = 42  # Per riprodurre lo stesso test set
XMAX = 5  # Range per x e p

# - FUNZIONI -
def get_test_coeff_paths(res):
    filenames = []
    labels = []
    coeff_paths = []
    for family_id in range(1, 11):
        base_path = os.path.join(INPUT_DATASET_DIR, res, f"family_{family_id}", "matrices")
        for fname in os.listdir(base_path):
            if fname.endswith(".npy") and "coeffs" not in fname:
                filenames.append(os.path.join(base_path, fname))
                labels.append(family_id - 1)
    _, test_files, _, _ = train_test_split(
        filenames, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    coeff_paths = []
    for fpath in test_files:
        family_path, matrix_name = os.path.split(fpath)
        base_name = matrix_name.replace(".npy", "_coeffs.txt")
        coeff_paths.append(os.path.join(family_path, base_name))
    return coeff_paths

def load_coeffs(filepath):
    coeffs = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                _, value = line.strip().split(":")
                value = value.replace(" ", "").replace("+-", "-")
                coeffs.append(complex(value))
            except Exception as e:
                print(f"Errore parsing riga '{line.strip()}' in file {filepath}: {e}")
    return coeffs

def build_density_matrix(coeffs):
    ket = Qobj(np.array(coeffs, dtype=np.complex128).reshape(-1, 1))
    rho = ket * ket.dag()  # |ψ⟩⟨ψ|
    return rho


def generate_noisy_grid(xmax, resolution, dev_std):
    x = np.linspace(-xmax, xmax, resolution)
    p = np.linspace(-xmax, xmax, resolution)
    x_noisy = x + np.random.normal(0, dev_std, size=x.shape)
    p_noisy = p + np.random.normal(0, dev_std, size=p.shape)
    return x_noisy, p_noisy



def save_wigner_image(matrix, filepath):
    vmax = abs(matrix).max()
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(1.6, 1.6), dpi=100)
    plt.imshow(matrix, cmap="RdBu_r", origin="lower", norm=norm)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close()


# -GENERAZIONE -
for dev_std in DEV_STDS:
    print(f">> Generating Wigner with dev_std = {dev_std}")
    for res in RESOLUTIONS:
        coeff_paths = get_test_coeff_paths(f"{res}x{res}")
        for coeff_path in tqdm(coeff_paths):
            try:
                coeffs = load_coeffs(coeff_path)
                rho = build_density_matrix(coeffs)
                x_noisy, p_noisy = generate_noisy_grid(XMAX, res, dev_std)
                W = wigner(rho, x_noisy, p_noisy)
            except Exception as e:
                print(f"Errore con {coeff_path}: {e}")
                continue

            parts = coeff_path.split(os.sep)
            family = parts[-3]
            fname = parts[-1].replace("_coeffs.txt", "")
            out_base = os.path.join(OUTPUT_DATASET_DIR, f"devstd_{dev_std}", f"{res}x{res}", family)
            os.makedirs(os.path.join(out_base, "matrices"), exist_ok=True)
            os.makedirs(os.path.join(out_base, "images"), exist_ok=True)
            np.save(os.path.join(out_base, "matrices", f"{fname}.npy"), W)
            save_wigner_image(W, os.path.join(out_base, "images", f"{fname}.png"))



