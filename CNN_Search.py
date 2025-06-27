#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codice che sceglie il miglior modello CNN secondo alcuni parametri, e lo salva come .pth
I risultati della cross validation e del miglior modello sono strutturati nella cartella CNN_results
Testa il miglior modello su dati mai visti
plotta il grafico finale di train e validaton loss

@author: marcoascari
"""
import time
import signal
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# === PARAMETRI ===
DATASET_DIR = "dataset_wigner"
RESOLUTION = "20x20" #fisso la risoluzione
EPOCHS = 1000 #fisso massimo di epoche
PATIENCE = 10 #pazienza per l'early stopping
NUM_CONV_LAYERS = 1 #fisso il numero di convoluzioni+pooling
USE_FC_HIDDEN = False #booleano che decide se utilizzare lo strato nascosto
FC_HIDDEN_DIM = 10 #dimensione dello strato nascosto

# -Struttura output organizzata-
BASE_DIR = os.path.join("CNN_results", RESOLUTION)
MODEL_DIR_NAME = f"conv{NUM_CONV_LAYERS}_hidden{FC_HIDDEN_DIM if USE_FC_HIDDEN else 0}"
MODEL_DIR = os.path.join(BASE_DIR, MODEL_DIR_NAME)
CV_DIR = os.path.join(MODEL_DIR, "crossval")
FINAL_DIR = os.path.join(MODEL_DIR, "final_model")

os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

start_time = time.time()  # ⏱️ Inizio timing
""" Opzionale fissa un tempo massimo della run
def timeout_handler(signum, frame):
    print("Tempo massimo di esecuzione raggiunto. Interruzione.")
    sys.exit(1)

# Imposta il segnale
#signal.signal(signal.SIGALRM, timeout_handler)
#signal.alarm(3600)  # Timeout dopo 3600 secondi (1 ora)
"""
# -Caricamento del dataset-
def load_wigner_dataset(dataset_dir, resolution):
    X, y = [], []
    for family_id in range(1, 11):
        path = os.path.join(dataset_dir, resolution, f"family_{family_id}", "matrices")
        for fname in os.listdir(path):
            if fname.endswith(".npy") and "coeffs" not in fname:
                matrix = np.load(os.path.join(path, fname))
                X.append(matrix)
                y.append(family_id - 1)
    return np.array(X), np.array(y)
#Divisione train+val(80%) e test(20%)
X, y = load_wigner_dataset(DATASET_DIR, RESOLUTION)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalizza in [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_trainval_flat = scaler.fit_transform(X_trainval.reshape(len(X_trainval), -1))
X_test_flat = scaler.transform(X_test.reshape(len(X_test), -1))
X_trainval = X_trainval_flat.reshape((-1, 1, X.shape[1], X.shape[2]))
X_test = X_test_flat.reshape((-1, 1, X.shape[1], X.shape[2]))

# -CostruzioneCNN-
def build_cnn_model(input_shape, num_classes, num_filters, kernel_size, activation_fn):
    layers = []
    in_channels = 1
    for _ in range(NUM_CONV_LAYERS):
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
        layers.append(activation_fn())
        layers.append(nn.MaxPool2d(2))
        in_channels = num_filters
    layers.append(nn.Flatten())

    dummy_input = torch.zeros((1, 1, *input_shape))
    dummy_output = nn.Sequential(*layers)(dummy_input)
    flatten_dim = dummy_output.shape[1]

    if USE_FC_HIDDEN:
        layers.append(nn.Linear(flatten_dim, FC_HIDDEN_DIM))
        layers.append(activation_fn())
        layers.append(nn.Linear(FC_HIDDEN_DIM, num_classes))
    else:
        layers.append(nn.Linear(flatten_dim, num_classes))

    return nn.Sequential(*layers)

#  Grid search con cross validation
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [10, 20],
    'activation': [nn.ReLU, nn.Tanh],
    'num_filters': [8, 16],
    'kernel_size': [3, 5]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = {}

log_file = open(os.path.join(CV_DIR, "cv_results.txt"), "w")

for activation_fn in param_grid['activation']:
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for nf in param_grid['num_filters']:
                for ks in param_grid['kernel_size']:
                    scores = []
                    combination_str = f"activation={activation_fn.__name__}, lr={lr}, batch_size={bs}, filters={nf}, kernel={ks}"
                    print(f"\nTesting: {combination_str}")
                    log_file.write(f"\nTesting: {combination_str}\n")

                    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
                        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
                        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

                        train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=bs, shuffle=True)
                        val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long()), batch_size=bs)

                        model = build_cnn_model(X.shape[1:], 10, nf, ks, activation_fn)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()

                        best_val_loss = float('inf')
                        epochs_no_improve = 0

                        for epoch in range(EPOCHS):
                            model.train()
                            for xb, yb in train_loader:
                                optimizer.zero_grad()
                                loss = criterion(model(xb), yb)
                                loss.backward()
                                optimizer.step()

                            model.eval()
                            val_losses = []
                            preds, true = [], []
                            with torch.no_grad():
                                for xb, yb in val_loader:
                                    outputs = model(xb)
                                    val_losses.append(criterion(outputs, yb).item())
                                    preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                                    true.extend(yb.cpu().numpy())

                            val_loss = np.mean(val_losses)
                            if val_loss < best_val_loss - 1e-4:
                                best_val_loss = val_loss
                                best_val_preds = preds.copy()  # memorizza le predizioni migliori
                                best_val_true = true.copy()
                                epochs_no_improve = 0
                            else:
                                epochs_no_improve += 1
                            
                            if epochs_no_improve >= PATIENCE:
                                break

                        acc = accuracy_score(best_val_true, best_val_preds)
                        scores.append(acc)
                        log_file.write(f" Fold {fold+1}: accuracy={acc:.4f}\n")
                        print(f" Fold {fold+1} accuracy: {acc:.4f}")

                    avg_score = np.mean(scores)
                    log_file.write(f" Average CV accuracy: {avg_score:.4f}\n")
                    #ricerc miglior modello
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'activation': activation_fn.__name__,
                            'learning_rate': lr,
                            'batch_size': bs,
                            'num_filters': nf,
                            'kernel_size': ks
                        }
#stampa migliori iperparametri
print("\n- MIGLIORI IPERPARAMETRI -")
print(best_params)
#stampa su file migliori iperparametri
print(f"Accuracy media CV: {best_score:.4f}")
log_file.write(f"\n- MIGLIORI IPERPARAMETRI -\n{str(best_params)}\nAccuracy media CV: {best_score:.4f}\n")
log_file.close()

# -Train miglior modello CNN e test sul test set mai visto prima-
print("\n- TRAINING FINALE CON I MIGLIORI IPERPARAMETRI -")

# Riassegna i parametri migliori
act_fn = getattr(nn, best_params['activation'])
lr = best_params['learning_rate']
bs = best_params['batch_size']
nf = best_params['num_filters']
ks = best_params['kernel_size']

# Split train/validation interno
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=bs, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long()), batch_size=bs)

model = build_cnn_model(X.shape[1:], 10, nf, ks, act_fn)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

    # Validation
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            val_batch_losses.append(criterion(out, yb).item())
    val_loss = np.mean(val_batch_losses)
    val_losses.append(val_loss)

   # print(f"Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_final_weights = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered.")
        break

# - Plot train/val loss -
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FINAL_DIR, "loss_curve.png"))
plt.close()

# - Test sul test set iniziale-
model.load_state_dict(best_final_weights)
torch.save(model, os.path.join(FINAL_DIR, "best_model.pth"))
model.eval()

test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=bs)
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy finale sul test set: {test_acc:.4f}")

with open(os.path.join(FINAL_DIR, "final_test_accuracy.txt"), "w") as f:
    f.write(f"Accuracy finale sul test set: {test_acc:.4f}\n")
    f.write("Parametri finali:\n")
    for k, v in best_params.items():
        f.write(f"  {k}: {v}\n")
        
# Se finisce prima, disattiva l'allarme
#signal.alarm(0)

# - Confusion Matrix -
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'F{i}' for i in range(10)])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "confusion_matrix.png"))
plt.close()
# Fine timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTempo totale: {elapsed_time:.2f} secondi ({elapsed_time/60:.2f} minuti)")