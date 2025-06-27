#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codice che sceglie il miglior modello MLP secondo alcuni parametri, e lo salva come .pkl
Salva i risultati nella cartella MLP_results
Testa il miglior modello su dati mai visti
plotta il grafico finale di train e validaton loss
@author: marcoascari
"""
import time
import signal
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import joblib
#import warnings

#warnings.filterwarnings("ignore")


#  -Parametri fissi-
DATASET_DIR = "dataset_wigner" #Directory principale del dataset
RESOLUTION = "20x20" #Risoluzione scelta per immagini di Wigner functions
N_HIDDEN=10 #Numero di neuroni nello strato nascosto dell'MLP
EPOCHS = 1000 #Numero massimo di epoche
PATIENCE = 10  #Early stopping: stop se la validation loss non migliora per 10 epoche
HIDDEN_LAYER_SIZES = (N_HIDDEN,)  #Struttura della rete: un solo hidden layer con N_HIDDEN neuroni

# -Directory di Output per salvataggi-
BASE_DIR = os.path.join("mlp_results", RESOLUTION, f"hidden{N_HIDDEN}")
CV_DIR = os.path.join(BASE_DIR, "cv_results") #Dove salvare i risultati della cross-validation
FINAL_DIR = os.path.join(BASE_DIR, "final_model")#Dove salvare il modello finale
os.makedirs(CV_DIR, exist_ok=True) #controllo sull'esistenza directory, le crea solo se non esistono
os.makedirs(FINAL_DIR, exist_ok=True)

start_time = time.time()  # Inizio il conteggio del tempo
""" Opzionale, fisso tempo massimo per evitare esecuzioni troppo lunghe
def timeout_handler(signum, frame):
    print("Tempo massimo di esecuzione raggiunto. Interruzione.")
    sys.exit(1)

 Imposta il segnale
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3600)  # Timeout dopo 3600 secondi (1 ora)
"""

# -Funzione per caricare dataset-
def load_wigner_dataset(dataset_dir, resolution, flatten=False):
    """
    Carica tutte le matrici di Wigner in formato .npy da una data risoluzione.
    Ritorna le immagini (X) e le etichette corrispondenti (y).
    Se flatten=True, le matrici NxN vengono trasformate in vettori di lunghezza N^2
    """
    X, y = [], []
    for family_id in range(1, 11):#10 famiglie, etichette da 0 a 9
        path = os.path.join(dataset_dir, resolution, f"family_{family_id}", "matrices")
        for fname in os.listdir(path):
            if fname.endswith(".npy") and "coeffs" not in fname:
                matrix = np.load(os.path.join(path, fname))
                X.append(matrix.flatten() if flatten else matrix)
                y.append(family_id - 1)
    return np.array(X), np.array(y)

# -Caricamento dati-
X, y = load_wigner_dataset(DATASET_DIR, RESOLUTION, flatten=True)

#Suddivisione in training+validation (80%) e test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Normalizzazione: MinMax scaler mappa i valori in [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_trainval = scaler.fit_transform(X_trainval)
X_test = scaler.transform(X_test)

#Salvataggio dello scaler per uso futuro
joblib.dump(scaler, os.path.join(FINAL_DIR, "scaler.pkl"))

# -Griglia di iperparametri da testare-
param_grid = {
    'learning_rate_init': [0.001, 0.0001],
    'batch_size': [10, 20, 40],
    'activation':['tanh','relu']
}
#Cross-validation stratificata (5 fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = {}
#File per salvare i risultati
log_file_path = os.path.join(CV_DIR, "cv_results.txt")
log_file = open(log_file_path, "w")
#Inizio grid search
for activation in param_grid['activation']:
    for lr in param_grid['learning_rate_init']:
        for bs in param_grid['batch_size']:
            combination_str = f"activation={activation}, lr={lr}, batch_size={bs}"
            print(f"\nTesting: {combination_str}")
            log_file.write(f"\nTesting: {combination_str}\n")
            scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
                X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
                y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
                classes = np.unique(y_train)

                mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=activation,
                                    learning_rate_init=lr, batch_size=bs, solver='adam',
                                    max_iter=1, warm_start=True, random_state=42)

                best_val_loss = float('inf')
                epochs_no_improve = 0

                for epoch in range(EPOCHS):
                    mlp.partial_fit(X_train, y_train, classes=classes)#fit su un epoca (max_iter=1 + warm_state=True)
                    val_loss = log_loss(y_val, mlp.predict_proba(X_val))
                    #Early stopping
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        joblib.dump(mlp, os.path.join(CV_DIR, f"best_model_fold{fold}.pkl"))
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= PATIENCE:
                        break
                    #salvataggio del modello e dei risultati
                mlp = joblib.load(os.path.join(CV_DIR, f"best_model_fold{fold}.pkl"))
                val_acc = accuracy_score(y_val, mlp.predict(X_val))
                scores.append(val_acc)
                print(f" Fold {fold+1} finished with accuracy: {val_acc:.4f}")
                log_file.write(f" Fold {fold+1}: accuracy={val_acc:.4f}\n")
                #media sui 5 fold e salvataggio
            avg_score = np.mean(scores)
            print(f"Average CV accuracy: {avg_score:.4f}")
            log_file.write(f" Average CV accuracy: {avg_score:.4f}\n")
            #scelta miglior modello
            if avg_score > best_score:
                best_score = avg_score
                best_params = {
                    'activation': activation,
                    'learning_rate_init': lr,
                    'batch_size': bs
                }

#Stampa dei parametri del miglior modello
print("\n=== MIGLIORI IPERPARAMETRI ===")
print(best_params)
#Stampa su file
print(f"Accuracy media CV: {best_score:.4f}")
log_file.write(f"\n=== MIGLIORI IPERPARAMETRI ===\n{best_params}\nAccuracy media CV: {best_score:.4f}\n")
log_file.close()


# -Allenamento finale-
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=42)
classes = np.unique(y_train)

mlp_final = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, activation=best_params['activation'],
                          learning_rate_init=best_params['learning_rate_init'],
                          batch_size=best_params['batch_size'],
                          solver='adam', max_iter=1, warm_start=True, random_state=42)

train_loss, val_loss, train_acc, val_acc = [], [], [], []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    mlp_final.partial_fit(X_train, y_train, classes=classes)

    y_train_pred = mlp_final.predict(X_train)
    y_val_pred = mlp_final.predict(X_val)

    train_loss.append(mlp_final.loss_)
    val_loss_epoch = log_loss(y_val, mlp_final.predict_proba(X_val))
    val_loss.append(val_loss_epoch)

    train_acc.append(accuracy_score(y_train, y_train_pred))
    val_acc.append(accuracy_score(y_val, y_val_pred))

  
    #Ealry stopping
    if val_loss_epoch < best_val_loss - 1e-4:
        best_val_loss = val_loss_epoch
        joblib.dump(mlp_final, os.path.join(FINAL_DIR, "best_final_model.pkl"))
        
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print("Early stopping attivato")
        break

# -Plots-
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Train loss")
plt.plot(val_loss, label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Train acc")
plt.plot(val_acc, label="Val acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(FINAL_DIR, "loss_accuracy_plot.png"))
plt.close()


# -Test finale-
best_model = joblib.load(os.path.join(FINAL_DIR, "best_final_model.pkl"))
test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nTest accuracy finale: {test_acc:.4f}")
with open(os.path.join(FINAL_DIR, "final_test_accuracy.txt"), "w") as f:
    f.write(f"Test accuracy finale: {test_acc:.4f}\n")
    f.write(f"Migliori iperparametri: {best_params}\n")
    
    
# Calcolo della confusion matrix
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

# Visualizza e salva la confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=True)
plt.title("Confusion Matrix - Test Set")
plt.savefig(os.path.join(FINAL_DIR, "confusion_matrix.png"))
plt.close()

# Salva anche la matrice come testo
np.savetxt(os.path.join(FINAL_DIR, "confusion_matrix.txt"), cm, fmt='%d')

"""  
    Se finisce prima, disattiva l'allarme, aggiungere se si tiene il tempo massimo impostato all'inizio
    signal.alarm(0)
"""
#fine tempo compilazione
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n Tempo totale: {elapsed_time:.2f} secondi ({elapsed_time/60:.2f} minuti)")