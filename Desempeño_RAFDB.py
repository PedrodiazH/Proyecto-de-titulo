''' Registro de accuracy: 97.29'''
''' Registro de accuracy con randomHorizontalFlip() 97.33'''
# Desempeño del modelo bajo datos de test y matriz de confusion
import torch                                        
import torch.nn as nn   
from tqdm import tqdm                               # barra de progreso
from torch.utils.data import DataLoader
from torchvision.transforms import transforms       # preprocesamiento por torch
from Approach.ResEmoteNet import ResEmoteNet        # arquitectura
from get_dataset import Four4All                    # clase para cargar imagen y preparar la data de test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Selección del dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")                   # por gpu nvidia asi q es mas rápido, pero debe ser instalado previamente
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'Device que se usa: {device}')

# Preprocesamiento para test con torch
transform = transforms.Compose([
    transforms.Resize((64, 64)),                        # redimensionar
    transforms.Grayscale(num_output_channels=3),        # conversion a grayscale con 3 canales
    #transforms.RandomHorizontalFlip(),                  # aumentar datos      
    transforms.ToTensor(),                              # convierte imagen a tensor [0,1]
    transforms.Normalize(                               # normaliza tensor para usar pesos entrenados de la cnn
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Datos de test (Probado en FER2013)
fer_dataset_test = Four4All(csv_file='RAF-DB/DATASET/subConjunto_paraExperimento/test_explicabilidad_experimentoOriginalesREMAPPED.csv',
                            img_dir='RAF-DB/DATASET/subConjunto_paraExperimento', transform=transform)

''' csv_file = 'RAF-DB/DATASET/test_explicabilidad_Remapped.csv' 
    img_dir= 'RAF-DB/DATASET/explicabilidad_oculta' '''
''' 'RAF-DB/test_labels_diseñadoRemmaped.csv'
    'RAF-DB/DATASET/test2'
'''

data_test_loader = DataLoader(fer_dataset_test, batch_size=64, shuffle=False) # batch 64
test_image, test_label = next(iter(data_test_loader))
criterion = torch.nn.CrossEntropyLoss()   

# Cargar modelo y pesos
model = ResEmoteNet()
numParams = sum(p.numel() for p in model.parameters())         # Para conocer el numero de parámetros
print("Número total de parámetros del modelo:", numParams)
checkpoint = torch.load('./Weights/rafdb_model.pth')             # Pesos entregados previamente por autor 
model.load_state_dict(checkpoint['model_state_dict'])          
model.to(device)                                               # se mueve el modelo al device

# Evaluación del modelo
model.eval()                    # pone la red en modo evaluación, evita overfitting
final_test_loss = 0.0
final_test_correct = 0          # inicialización de métricas de desempeño
final_test_total = 0
preds = []                      # lista de predicciones para matriz de conf
all_labels = []                 # lista de las labels para matriz de conf
probs_all = []
with torch.no_grad():
    for data in tqdm(data_test_loader, desc="Testeando el modelo: "):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        final_test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        final_test_total += labels.size(0)
        final_test_correct += (predicted == labels).sum().item()
        preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Obtener clasificador softmax para metricas basada en score
        probs = torch.nn.functional.softmax(outputs, dim=1)
        probs_all.extend(probs.cpu().numpy())
#Resultados de la evaluación
final_test_loss = final_test_loss / len(data_test_loader)
final_test_acc = final_test_correct / final_test_total

print(f"Test Loss: {final_test_loss}")
print(f"\nTest Accuracy: {final_test_acc} --> ({final_test_acc*100:.2f}%)")

# Matriz de confusión
labelsOrden = ['enojo', 'disgusto', 'miedo', 'feliz', 'neutral', 'tristeza', 'sorpresa']
matrizConf = confusion_matrix(all_labels, preds,normalize='true')
fig, ax = plt.subplots(figsize=(6, 4))  # Compacto para paper
disp = ConfusionMatrixDisplay(confusion_matrix=matrizConf, display_labels=labelsOrden)
disp.plot(cmap=plt.cm.YlOrBr, ax=ax, colorbar=True,values_format='.2f')
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)
plt.title(f"ResEmoteNet con RAF-DB. Acc={final_test_acc*100:.2f}%", fontsize=11, fontweight='semibold')
plt.xlabel("Predicción", fontsize=10)
plt.ylabel("Real", fontsize=10)
plt.tight_layout()
plt.show()

# Classification report 
print("\nReporte de clasificación por clase:\n")
classificationReport = classification_report(all_labels, preds, target_names=labelsOrden, digits=3)
print(classificationReport)

# Mas métricas
f1_macro = f1_score(all_labels, preds, average='macro')
f1_weighted = f1_score(all_labels, preds, average='weighted')
recall_macro = recall_score(all_labels, preds, average='macro')
precision_macro = precision_score(all_labels, preds, average='macro')

print(f"F1-score (Macro): {f1_macro:.4f}")
print(f"F1-score (Ponderado): {f1_weighted:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"Precisión (Macro): {precision_macro:.4f}")

# Batplot
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds)
df_metrics = pd.DataFrame({
    'Clase': labelsOrden,
    'Precisión': precision,
    'Recall': recall,
    'F1-score': f1
})

df_metrics.set_index('Clase').plot(kind='bar', figsize=(6, 4), ylim=(0, 1), colormap='plasma')
#plt.title("Métricas por clase en FER2013")
plt.ylabel("Puntaje")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 
# One vs Rest
y_true_bin = label_binarize(all_labels, classes=np.arange(len(labelsOrden)))

# Para cada clase, calcula su curva
plt.figure(figsize=(6, 4))
for i, class_name in enumerate(labelsOrden):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], np.array(probs_all)[:, i])
    ap_score = average_precision_score(y_true_bin[:, i], np.array(probs_all)[:, i])
    plt.plot(recall, precision, lw=2, label=f'{class_name} (AP={ap_score:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.title("Curvas Precision-Recall por clase (OvR)",fontsize=11, fontweight='semibold')
plt.legend(loc='best', fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()