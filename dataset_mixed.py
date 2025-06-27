#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crea dataset di test, per testare i modelli con stati mixed con il vuoto
 per diverse percentuali di vuoto

@author: marcoascari

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from qutip import basis, ket2dm, wigner
from sklearn.model_selection import train_test_split

# - PARAMETRI -
INPUT_DATASET_DIR = "dataset_wigner" #percorso dataset 'pulito'
OUTPUT_BASE_DIR = "dataset_wigner_mixed" #dataset di output che dovrà contenere le wigner utilizzate per testare gli altri modelli, ma mischiate al vuoto

resolutions = ["10x10", "20x20", "50x50", "70x70", "100x100"] #lo fa per tutte queste risoluzioni
ALPHAS = [0.99, 0.95, 0.9, 0.8, 0.5]#percentuale della Wigner originale nella combinazione con il vuoto

xmax = pmax = 5 #estremi del dominio dov'è calcolata la Wigner
hilbert_dim = 12 #dimensione >10 tetto massimo dimensione dello spazio di Hilbert

# -FUNZIONI-
#restituisce la funzione di Wigner del vuoto su una griglia resxres
def compute_vacuum_wigner(res):
    x = np.linspace(-xmax, xmax, res)
    p = np.linspace(-pmax, pmax, res)
    vac = basis(hilbert_dim, 0) #stato di vuoto
    rho_vac = ket2dm(vac) #matrice densità del vuoto
    return wigner(rho_vac, x, p) 

def load_wigner_matrix(path): #carica la matrice .npy delle Wigner pulite
    return np.load(path)

def save_wigner_image(wigner_matrix, path): #salva l'immagine corrispondente alla Wigner
    vmax = abs(wigner_matrix).max()
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=100)
    ax.imshow(wigner_matrix, cmap="RdBu_r", origin="lower", norm=norm)#usa colormap simetrica RedBLue
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
#restituisce solo i path dei file utilizzati nel testing
def get_test_filepaths(res):
    filenames = []
    labels = []
    for family_id in range(1, 11):
        path = os.path.join(INPUT_DATASET_DIR, res, f"family_{family_id}", "matrices")
        for fname in os.listdir(path):
            if fname.endswith(".npy") and "coeffs" not in fname:
                filenames.append(os.path.join(path, fname))
                labels.append(family_id - 1)
    _, test_files, _, _ = train_test_split(
        filenames, labels, test_size=0.2, stratify=labels, random_state=42#random_state=42 stesso seed utilizzato per il testset
    )
    return set(test_files)

# per ogni alpha e ogni risoluzione calcola le Wigner miste e le mette nel dataset di output
for alpha in ALPHAS:
    mix_folder = f"mix_{alpha:.2f}"
    print(f"\n🌀 Generazione dataset misto (solo test) con alpha = {alpha}")

    for res in resolutions:
        print(f"  ➤ Risoluzione: {res}")
        res_num = int(res.split("x")[0])
        W_vacuum = compute_vacuum_wigner(res_num)
        test_files = get_test_filepaths(res)

        for family_id in range(1, 11):
            input_dir = os.path.join(INPUT_DATASET_DIR, res, f"family_{family_id}", "matrices")
            output_matrix_dir = os.path.join(OUTPUT_BASE_DIR, mix_folder, res, f"family_{family_id}", "matrices")
            output_image_dir = os.path.join(OUTPUT_BASE_DIR, mix_folder, res, f"family_{family_id}", "images")

            os.makedirs(output_matrix_dir, exist_ok=True)
            os.makedirs(output_image_dir, exist_ok=True)

            for fname in os.listdir(input_dir):
                if not fname.endswith(".npy") or "coeffs" in fname:
                    continue

                full_path = os.path.join(input_dir, fname)
                if full_path not in test_files:
                    continue

                W_original = load_wigner_matrix(full_path)
                W_mixed = alpha * W_original + (1 - alpha) * W_vacuum

                name_no_ext = fname.replace(".npy", "")
                new_matrix_path = os.path.join(output_matrix_dir, name_no_ext + ".npy")
                new_image_path = os.path.join(output_image_dir, name_no_ext + ".png")

                np.save(new_matrix_path, W_mixed)
                save_wigner_image(W_mixed, new_image_path)

        print(f"✅ Completato: alpha = {alpha}, risoluzione = {res}")

print("\n🏁 Tutti i dataset MISTI (solo test) sono stati generati.")

