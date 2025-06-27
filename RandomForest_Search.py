#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codice che sceglie il miglior modello Random Forest secondo alcuni parametri, e lo salva come .pth
I risultati della cross validation e del miglior modello sono strutturati nella cartella RF_results
Testa il miglior modello su dati mai visti
plotta il grafico finale di train e validaton loss

@author: marcoascari
"""
import time
import signal
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold #per divisione dei dati per cross validation
from sklearn.preprocessing import MinMaxScaler #per normalizzare i dati
from sklearn.ensemble import RandomForestClassifier #importo Random Forest
from sklearn.metrics import accuracy_score #
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib #per salvare il modello migliore
import matplotlib.pyplot as plt

# - PARAMETRI FISSATI -
DATASET_DIR = "dataset_wigner" #dataset di partenza con le 10 famiglie di stati
RESOLUTION = "20x20" #risoluzione fissata della Wigner Function
N_ESTIMATORS = 50  #numero fisso di alberi nella foresta, da fissare a scelta

# - CARTELLE OUTPUT -
BASE_DIR = os.path.join("rf_results", RESOLUTION, f"trees{N_ESTIMATORS}")
CV_DIR = os.path.join(BASE_DIR, "cv_results")
FINAL_DIR = os.path.join(BASE_DIR, "final_model")
os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

start_time = time.time()  #prendo il tempo dell'intero processo

"""-Opzionale Timeout per interrompere la compilazione se dura troppo-

def timeout_handler(signum, frame):
    print("Tempo massimo di esecuzione raggiunto. Interruzione.")
    sys.exit(1)

Opzionale Timeout per interrompere la compilazione se dura troppo
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3600)  # Timeout dopo 3600 secondi (1 ora)
"""
# - Funzione Caricamento dati da dataset_wigner -
def load_wigner_dataset(dataset_dir, resolution, flatten=True): 
    X, y = [], []
    for family_id in range(1, 11): #ciclo sulle famiglie da 1 a 10
        path = os.path.join(dataset_dir, resolution, f"family_{family_id}", "matrices") #percorso dei dati
        for fname in os.listdir(path): #ciclo sulle Wigner
            if fname.endswith(".npy") and "coeffs" not in fname: #fa attenzione a non prendere i file txt con i coefficienti dei rispetivi stati delle Wigner
                matrix = np.load(os.path.join(path, fname)) #
                X.append(matrix.flatten() if flatten else matrix) #prende le matrici di Wigner flattenate
                y.append(family_id - 1) #assegna la label da 0 a 9 per ogni matrice di Wigner
    return np.array(X), np.array(y) #restituisce le immagini(matrici.npy) come array flatten

X, y = load_wigner_dataset(DATASET_DIR, RESOLUTION) #carica X e y con i dati usando la funzione preccedente
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) #80% train+validation 20%testing data

scaler = MinMaxScaler(feature_range=(-1, 1)) #normalizzazione tra -1,1 dei dai per aiutare convergenza del modello
X_trainval = scaler.fit_transform(X_trainval) #la applico a train+val
X_test = scaler.transform(X_test)#la applico al test

# - Grid Search manuale con Cross Validation -
param_grid = {  #parametri che vengono testati in ogni combinazione tra loro per scoprire quali sono i migliori
    'max_depth': [5, 10, None], #profondità degli alberi, quante decisioni prendono, None indica profondità ultima
    'max_features': ['sqrt', 'log2'], #features per il bootstrap dei dati
    'criterion': ['gini', 'entropy'] #criterio di decisione di ogni albero
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)#impostazione 5 fold cross validation
best_score = 0#inizializza il miglior risultato
best_params = {} #inizializza migliori parametri

log_file_path = os.path.join(CV_DIR, "cv_results.txt")#apro il file per salvare i dati della cross validation
log_file = open(log_file_path, "w")
#inizia la grid-search
for depth in param_grid['max_depth']:
    for feat in param_grid['max_features']:
        for crit in param_grid['criterion']:
            comb_str = f"depth={depth}, max_features={feat}, criterion={crit}"
            print(f"\nTesting: {comb_str}")
            log_file.write(f"\nTesting: {comb_str}\n")
            scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
                X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
                y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

                rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=depth,
                                            max_features=feat, criterion=crit, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                acc = accuracy_score(y_val, rf.predict(X_val))
                scores.append(acc)
                print(f" Fold {fold+1} accuracy: {acc:.4f}")
                log_file.write(f" Fold {fold+1} accuracy: {acc:.4f}\n")

            avg_score = np.mean(scores)
            print(f"Average CV accuracy: {avg_score:.4f}")
            log_file.write(f" Average CV accuracy: {avg_score:.4f}\n")

            if avg_score > best_score:
                best_score = avg_score
                best_params = {'max_depth': depth, 'max_features': feat, 'criterion': crit}

log_file.write(f"\n=== MIGLIORI IPERPARAMETRI ===\n{best_params}\nAccuracy media CV: {best_score:.4f}\n")
log_file.close()

print("\n=== MIGLIORI IPERPARAMETRI ===")
print(best_params)
print(f"Accuracy media CV: {best_score:.4f}")

# - Allenamento finale -
final_model = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                     max_depth=best_params['max_depth'],
                                     max_features=best_params['max_features'],
                                     criterion=best_params['criterion'],
                                     random_state=42, n_jobs=-1)

final_model.fit(X_trainval, y_trainval)
joblib.dump(final_model, os.path.join(FINAL_DIR, "best_rf_model.pkl"))

# - Test finale -
y_test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\nTest accuracy finale: {test_acc:.4f}")

# - Confusion Matrix -
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))  # Etichette da 0 a 9
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "confusion_matrix.png"))  # Salva l'immagine
plt.show()  # Mostra a schermo

#Salva i risultati
with open(os.path.join(FINAL_DIR, "final_test_accuracy.txt"), "w") as f:
    f.write(f"Test accuracy finale: {test_acc:.4f}\n")
    f.write(f"Migliori iperparametri: {best_params}\n")
    

# Se finisce prima, disattiva l'allarme
#signal.alarm(0)

# Fine timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Tempo totale: {elapsed_time:.2f} secondi ({elapsed_time/60:.2f} minuti)")