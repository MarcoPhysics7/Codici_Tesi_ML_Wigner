#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Questo codice genera dataset di funzioni di Wigner a diverse risoluzioni
(10x10, 20x20, ..., 100x100) per 10 famiglie di stati quantistici.
Ogni famiglia corrisponde a una sovrapposizione randomica (seconda la misura di Haar)
di stati di Fock fino a un numero quantico massimo 'n_max' (da 1 a 10)

Per ogni stato vengono salvati:
- La matrice della funzione di Wigner (in formato '.npy')
- L'immagine della funzione di Wigner (in formato '.png')
- I coefficienti della sovrapposizione nello spazio di Fock (in formato '.txt')
@author: marcoascari
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from qutip import Qobj, wigner, ket2dm
from numpy.random import default_rng
from qutip import rand_unitary,basis

# - PARAMETRI GLOBALI(modificabili a piacere per risultati diversi) -
output_base = "dataset_wigner"          #Cartella principale di output
resolutions = [10, 20, 50, 70, 100]     #Risoluzioni delle immagini Wigner
samples_per_family = 1500               #Numero di stati per ogni famiglia
families = range(1, 11)                 #Famiglie da n_max = 1 a 10
hilbert_dim = 12                        #Dimensione dello spazio di Hilbert totale
xmax = pmax = 5                         #Estensione dell'asse x e p nello spazio delle fasi
rng = default_rng()                     #Generatore di numeri casuali

# === FUNZIONI ===
def random_fock_superposition(n_max, dim):
    """
    Genera una sovrapposizione Haar-random di stati di Fock da |0⟩ a |n_max⟩,
    garantendo che il coefficiente di |n_max⟩ sia diverso da zero.
    
    Parametri:
        n_max (int): Numero quantico massimo incluso nella sovrapposizione
        dim (int): Dimensione dello spazio di Hilbert (usata per il padding)
        
    Restituisce:
        Qobj: Stato quantistico normalizzato come Qutip ket (dimensione `dim`)
    """
    d_eff = n_max + 1

    if d_eff == 1:
        padded = np.zeros(dim, dtype=complex)
        padded[0] = 1.0
        return Qobj(padded, dims=[[dim], [1]])

    while True:
        U = rand_unitary(d_eff)         # matrice unitaria Haar-random
        psi = U @ basis(d_eff, 0)       # ruota lo stato |0⟩ nel sottospazio di hilbert di dimensione d_eff
        coeffs = psi.full().flatten()   # array complesso dei coefficienti

        if abs(coeffs[-1]) > 1e-3:       # Verifica che |n_max⟩ sia significativamente occupato
            padded = np.zeros(dim, dtype=complex)
            padded[:d_eff] = coeffs
            return Qobj(padded, dims=[[dim], [1]])

def save_wigner_image(wigner_matrix, path, img_res):
    """
  Salva la matrice della funzione di Wigner come immagine colorata.
  
  Parametri:
      wigner_matrix (ndarray): Matrice 2D della funzione di Wigner
      path (str): Percorso per il salvataggio dell'immagine
      img_res (int): Risoluzione dell'immagine (lato in pixel)
  """
    vmax = abs(wigner_matrix).max()
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.figure(figsize=(2, 2), dpi=img_res // 2)
    plt.axis('off')
    plt.imshow(wigner_matrix, origin="lower", cmap="RdBu_r", norm=norm, extent=[-xmax, xmax, -pmax, pmax])
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# - GENERAZIONE BASE: stati quantistici -
print("Generazione stati base...")
all_states = {n_max: [random_fock_superposition(n_max, hilbert_dim) for _ in range(samples_per_family)]
              for n_max in families}

# - GENERAZIONE IMMAGINI E MATRICI -
for res in resolutions:
    print(f"\nGenerazione per risoluzione {res}x{res}...")
    x = np.linspace(-xmax, xmax, res)
    p = np.linspace(-pmax, pmax, res)

    for n_max in families:
        print(f"  Famiglia {n_max}...")

        #cartelle per salvataggio
        base_dir = os.path.join(output_base, f"{res}x{res}", f"family_{n_max}")
        img_dir = os.path.join(base_dir, "images")
        mat_dir = os.path.join(base_dir, "matrices")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mat_dir, exist_ok=True)

         # Per ogni stato: calcola Wigner, salva matrice, immagine e coeffs # Per ogni stato: calcola Wigner, salva matrice, immagine e coeffs
        for i, state in enumerate(all_states[n_max]):
            rho = ket2dm(state) #stato puro -> matrice densità
            W = wigner(rho, x, p) #Funzioe di Wigner

            filename = f"state_{i:04d}"
            #Salva matrice .npy
            np.save(os.path.join(mat_dir, filename + ".npy"), W)
            #Salva immagine .png
            save_wigner_image(W, os.path.join(img_dir, filename + ".png"), img_res=res)

            # Salva coeff. complessi fino a n_max in .txt
            coeffs_path = os.path.join(mat_dir, filename + "_coeffs.txt")
            with open(coeffs_path, 'w') as f:
                for n in range(n_max + 1):
                    c = state.full()[n, 0]
                    f.write(f"{n}: {c.real:.6f} + {c.imag:.6f}j\n")
