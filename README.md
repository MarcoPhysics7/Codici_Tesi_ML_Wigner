#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:47:21 2025

@author: marcoascari
"""

La corrente repository contiene i codici necessari per creare un dataset di funzioni di Wigner per diversi
stati quantistici Haar random e per allenare diversi modelli di Machine Learning a classificarli in base al loro supporto sulla base di Fock.
In particolare:
-dataset_wigner_pulito -> crea il dataset originale per allenare i modelli.
-dataset_wigner_noisy_grid -> crea dati rumorosi per testare i modelli ML su punti con Wigner generate da punti rumorosi sullo spazio delle fasi
-dataset_mixed -> crea dati per testare modelli ML introducendo una percentuale di mix dello stato originale con lo stato di vuoto
- i file NOME_MODELLO_SEARCH -> cercano i migliori iperparametri per il modello in considerazione con alcuni parametri fissati
- i file TestNomeModelloRumore testano i modelli su Wigner blurred e mixed con il vuoto
