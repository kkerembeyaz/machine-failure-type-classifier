# 1) IMPORTS
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn

# 2) VERİ YÜKLEME VE İLK İNCELEME (EDA)
df = pd.read_csv("ai4i2020.csv")
print(df.head())
print(df.info())
print(df.describe())

# 3) VERİ TUTARLILIK KONTROLÜ VE TEMİZLİK
#    - "Machine failure=1 ama hiçbir tür işaretli değil" (unknown) satırları
#    - Birden fazla arıza türünün aynı anda 1 olduğu (overlap) satırları çıkar
failure_count = df[['TWF','HDF','PWF','OSF','RNF']].sum(axis=1)  # 5 arıza türü sütununun satır bazında toplamı
mask_unknown = (df['Machine failure'] == 1) & (failure_count == 0)  # Durum 1: Machine failure=1 ama hiçbir tür işaretlenmemiş
mask_overlap = failure_count > 1  # Durum 2: Birden fazla tür aynı anda 1 (örtüşen arızalar)
mask_to_drop = mask_unknown | mask_overlap  # İkisini birleştir (| = veya)

print("Unknown failure satır sayısı:", mask_unknown.sum())
print("Overlap satır sayısı:", mask_overlap.sum())
print("Toplam çıkarılacak satır:", mask_to_drop.sum())
df = df[~mask_to_drop].reset_index(drop=True)

# 4) HEDEF ETİKETİ OLUŞTURMA (Failure_Type)
#    5 binary arıza sütununu tek bir multiclass etiket sütununda birleştir
conditions = [
    df['TWF'] == 1,
    df['HDF'] == 1,
    df['PWF'] == 1,
    df['OSF'] == 1,
    df['RNF'] == 1,
]
choices = [
    'TWF',
    'HDF',
    'PWF',
    'OSF',
    'RNF',
]
df['Failure_Type'] = np.select(conditions, choices, default='No Failure')

# Kontrol
print(df['Failure_Type'].value_counts())

# 5) GİRDİ (X) / HEDEF (y) AYRIMI VE ENCODING
feature_cols = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

X = df[feature_cols]
y = df['Failure_Type']

# --- Type sütunu: Ordinal Encoding (L=0, M=1, H=2) ---
type_order = {'L': 0, 'M': 1, 'H': 2}
X['Type'] = X['Type'].map(type_order)

# --- Failure_Type sütunu: Manuel Mapping (string -> class index) ---
label_map = {
    "No Failure": 0,
    "TWF": 1,
    "HDF": 2,
    "PWF": 3,
    "OSF": 4,
    "RNF": 5,
}
y_encoded = y.map(label_map)

# 6) STRATIFIED TRAIN / VAL / TEST SPLIT (70/15/15)
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=
0.3, stratify= y_encoded,random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify= y_temp,
    random_state=42
)

print(X_train.shape, X_val.shape, X_test.shape)

# 7) SAYISAL SÜTUNLARI SCALE ETME (yalnızca train'e fit, val/test'e transform)
from sklearn.preprocessing import StandardScaler
numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 8) TENSOR DÖNÜŞÜMÜ
import torch
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.long)
y_val_t = torch.tensor(y_val.values, dtype=torch.long)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

print(X_train_t.shape, y_train_t.shape)
print(X_val_t.shape, y_val_t.shape)
print(X_test_t.shape, y_test_t.shape)

# 9) MODEL MİMARİSİ (FailureClassifier)
import torch.nn as nn
class FailureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(6,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,6)
        )
    def forward(self,x):
        return self.linear_layer_stack(x)   
model = FailureClassifier()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

import torchmetrics
num_classes = 6
accuracy_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)

# 10) DENEME 1 — WEIGHT'SİZ EĞİTİM (model)
#     Sonuç: model çoğunluk sınıfına ("No Failure") kilitleniyor, macro accuracy ~%16-17
num_epochs = 200

train_losses = []
val_losses = []
val_accuracies =[]
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    #Training
    model.train()
    logits = model(X_train_t)
    loss = loss_fn(logits, y_train_t)

    pred = torch.argmax(logits, dim=1)
    acc = accuracy_fn(pred, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    train_accuracies.append(acc)

    model.eval()
    with torch.inference_mode():
        val_logits = model(X_val_t)
        val_loss = loss_fn(val_logits, y_val_t)
        val_pred = torch.argmax(val_logits, dim=1)
        val_acc = accuracy_fn(y_val_t, val_pred)

    val_losses.append(val_loss.item())
    val_accuracies.append(val_acc)

    if epoch % 20 == 0:
        print(f"Epoch:{epoch} | Loss:{loss:.4f} | Accuracy:{acc.item()*100:.2f}% | "
              f"Val Loss:{val_loss:.4f} | Val Accuracy:{val_acc.item()*100:.2f}%") 

print(torch.unique(pred, return_counts=True))

# 11) DENEME 2 — BALANCED CLASS WEIGHTS İLE EĞİTİM (model2)
#     Sonuç: azınlık sınıfların recall'u iyileşiyor, ama No Failure precision ~%50'ye düşüyor
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train_t.numpy())
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_t.numpy()
)

class_weights = torch.tensor(weights, dtype=torch.float32)
print(class_weights)

loss_fn2 = nn.CrossEntropyLoss(weight=class_weights)

model2 = FailureClassifier()

from torch import optim
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

train_losses2 = []
val_losses2 = []
train_accuracies2 = []
val_accuracies2 = []

for epoch in range(num_epochs):
    model2.train()
    logits = model2(X_train_t)
    loss = loss_fn2(logits, y_train_t)

    pred = torch.argmax(logits, dim=1)
    acc = accuracy_fn(pred, y_train_t)

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

    train_losses2.append(loss.item())
    train_accuracies2.append(acc.item())

    model2.eval()
    with torch.inference_mode():
        val_logits = model2(X_val_t)
        val_loss = loss_fn2(val_logits, y_val_t)
        val_pred = torch.argmax(val_logits, dim=1)
        val_acc = accuracy_fn(val_pred, y_val_t)

    val_losses2.append(val_loss.item())
    val_accuracies2.append(val_acc.item())

    if epoch % 20 == 0:
        print(f"Epoch:{epoch} | Loss:{loss:.4f} | Accuracy:{acc.item()*100:.2f}% | "
              f"Val Loss:{val_loss:.4f} | Val Accuracy:{val_acc.item()*100:.2f}%")

    from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

model2.eval()
with torch.inference_mode():
    val_logits = model2(X_val_t)
    val_pred = torch.argmax(val_logits, dim=1)

label_map_inv = {v: k for k, v in label_map.items()}
cm = confusion_matrix(y_val_t.numpy(), val_pred.numpy())
class_names = [label_map_inv[i] for i in sorted(label_map_inv.keys())]

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix - Validation Set')
plt.show()

print(classification_report(y_val_t.numpy(), val_pred.numpy(), target_names=class_names))

# 12) DENEME 3 — TAVANLI (CAPPED) CLASS WEIGHTS İLE EĞİTİM (model3)
#     Sonuç: No Failure precision ~%99'a çıkıyor, ama TWF ve RNF recall %0'a düşüyor
model3 = FailureClassifier()
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

train_losses3 = []
val_losses3 = []
train_accuracies3 = []
val_accuracies3 = []

weights_capped = np.minimum(weights, 10.0)
class_weights_capped = torch.tensor(weights_capped, dtype=torch.float32)
print(class_weights_capped)

loss_fn3 = nn.CrossEntropyLoss(weight=class_weights_capped)

for epoch in range(num_epochs):
    model3.train()
    logits = model3(X_train_t)
    loss = loss_fn3(logits, y_train_t)

    pred = torch.argmax(logits, dim=1)
    acc = accuracy_fn(pred, y_train_t)

    optimizer3.zero_grad()
    loss.backward()
    optimizer3.step()

    train_losses3.append(loss.item())
    train_accuracies3.append(acc.item())

    model3.eval()
    with torch.inference_mode():
        val_logits = model3(X_val_t)
        val_loss = loss_fn3(val_logits, y_val_t)
        val_pred = torch.argmax(val_logits, dim=1)
        val_acc = accuracy_fn(val_pred, y_val_t)

    val_losses3.append(val_loss.item())
    val_accuracies3.append(val_acc.item())

    if epoch % 20 == 0:
        print(f"Epoch:{epoch} | Loss:{loss:.4f} | Accuracy:{acc.item()*100:.2f}% | "
              f"Val Loss:{val_loss:.4f} | Val Accuracy:{val_acc.item()*100:.2f}%")

    model3.eval()
with torch.inference_mode():
    val_logits = model3(X_val_t)
    val_pred = torch.argmax(val_logits, dim=1)

cm = confusion_matrix(y_val_t.numpy(), val_pred.numpy())

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix - model3 (capped weights)')
plt.show()

print(classification_report(y_val_t.numpy(), val_pred.numpy(), target_names=class_names))

# 13) DENEME 4 — MANUEL, SINIF BAZLI AĞIRLIKLARLA EĞİTİM (model4) — FİNAL MODEL
#     Sonuç: No Failure ve TWF arasında daha dengeli bir trade-off
manual_weights = {
    0: 0.5,   # No Failure
    1: 20.0,  # TWF
    2: 10.0,  # HDF
    3: 12.0,  # PWF
    4: 12.0,  # OSF
    5: 15.0,  # RNF
}

class_weights_manual = torch.tensor(
    [manual_weights[i] for i in sorted(manual_weights.keys())],
    dtype=torch.float32
)
print(class_weights_manual)

loss_fn4 = nn.CrossEntropyLoss(weight=class_weights_manual)

model4 = FailureClassifier()
optimizer4 = optim.Adam(model4.parameters(), lr=0.001)

train_losses4 = []
val_losses4 = []
train_accuracies4 = []
val_accuracies4 = []

for epoch in range(num_epochs):
    model4.train()
    logits = model4(X_train_t)
    loss = loss_fn4(logits, y_train_t)

    pred = torch.argmax(logits, dim=1)
    acc = accuracy_fn(pred, y_train_t)

    optimizer4.zero_grad()
    loss.backward()
    optimizer4.step()

    train_losses4.append(loss.item())
    train_accuracies4.append(acc.item())

    model4.eval()
    with torch.inference_mode():
        val_logits = model4(X_val_t)
        val_loss = loss_fn4(val_logits, y_val_t)
        val_pred = torch.argmax(val_logits, dim=1)
        val_acc = accuracy_fn(val_pred, y_val_t)

    val_losses4.append(val_loss.item())
    val_accuracies4.append(val_acc.item())

    if epoch % 20 == 0:
        print(f"Epoch:{epoch} | Loss:{loss:.4f} | Accuracy:{acc.item()*100:.2f}% | "
              f"Val Loss:{val_loss:.4f} | Val Accuracy:{val_acc.item()*100:.2f}%")

model4.eval()
with torch.inference_mode():
    val_logits = model4(X_val_t)
    val_pred = torch.argmax(val_logits, dim=1)

cm = confusion_matrix(y_val_t.numpy(), val_pred.numpy())

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix - model4 (capped weights)')
plt.show()

print(classification_report(y_val_t.numpy(), val_pred.numpy(), target_names=class_names))

# 14) NİHAİ TEST SETİ DEĞERLENDİRMESİ (model4, tek seferlik)
model4.eval()
with torch.inference_mode():
    test_logits = model4(X_test_t)
    test_loss = loss_fn4(test_logits, y_test_t)
    test_pred = torch.argmax(test_logits, dim=1)
    test_acc = accuracy_fn(test_pred, y_test_t)

print(f"Test Loss: {test_loss.item():.4f} | Test Accuracy (macro): {test_acc.item()*100:.2f}%")

cm_test = confusion_matrix(y_test_t.numpy(), test_pred.numpy())

plt.figure(figsize=(8,6))
sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix - Test Set (model4, final)')
plt.show()

print(classification_report(y_test_t.numpy(), test_pred.numpy(), target_names=class_names))

# 15) MODEL / SCALER / MAPPING KAYDETME (state_dict + torch.save)
from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#model
MODEL_NAME = "machine-failure-classifier.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model4.state_dict(), f=MODEL_SAVE_PATH)

#scale ve mappingler
torch.save(obj=scaler, f=MODEL_PATH / "scaler.pth")
torch.save(obj=type_order, f=MODEL_PATH / "type_order.pth")
torch.save(obj=label_map, f=MODEL_PATH / "label_map.pth")
torch.save(obj=label_map_inv, f=MODEL_PATH / "label_map_inv.pth")

print(MODEL_SAVE_PATH)

# 16) MANUEL DOĞRULAMA — WEB APP İLE AYNI PREPROCESSING'İ TEKRARLAYIP TEST ETME
def manual_predict(type_val, air_temp, process_temp, rpm, torque, tool_wear):
    type_encoded = type_order[type_val]
    numeric = scaler.transform([[air_temp, process_temp, rpm, torque, tool_wear]])[0]
    features = [type_encoded] + list(numeric)
    x = torch.tensor([features], dtype=torch.float32)

    model4.eval()
    with torch.inference_mode():
        logits = model4(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()

    pred_label = label_map_inv[pred_idx]
    prob_dict = {label_map_inv[i]: round(probs[0][i].item(), 4) for i in range(len(label_map_inv))}

    print("Tahmin:", pred_label)
    print("Olasılıklar:", prob_dict)
    return pred_label

# test
manual_predict('L', 307, 295, 1457, 43.2, 0)
