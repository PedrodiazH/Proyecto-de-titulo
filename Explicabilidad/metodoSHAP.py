import shap
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Approach.ResEmoteNet import ResEmoteNet

# Configuracion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando el hardware: {device}")
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Cargar modelo
model = ResEmoteNet().to(device)
checkpoint = torch.load('./Weights/fer_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocesamiento requerido por el modelo
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funcion de prediccion para SHAP, se usa como modelo explicable
def predict(images_np):
    images_tensor = torch.tensor(images_np).permute(0, 3, 1, 2).float() / 255.0 # Convierte imagen en tensor
    for i in range(images_tensor.shape[0]):
        img = transforms.ToPILImage()(images_tensor[i]) # Aplica transformacion
        images_tensor[i] = transform(img)
    images_tensor = images_tensor.to(device)
    with torch.no_grad():
        outputs = model(images_tensor)
        probs = F.softmax(outputs, dim=1) # Retorna probabilidades softmax
    return probs.cpu().numpy()

# Datos de entrada
image_path = "FER2013/subConjuntoFER2013_paraExperimento/sad/PrivateTest_1414350.jpg"
img_pil = Image.open(image_path).convert('RGB')
img_resized = img_pil.resize((64, 64))
img_np = np.array(img_resized)

# Probabilidades
probs = predict(np.array([img_np]))[0]
pred_index = np.argmax(probs)

# Mascara SHAP
segments_slic = slic(img_np, n_segments=20, compactness=10, sigma=1) # segmenta las imagenes en n regiones 
masker = shap.maskers.Image("inpaint_telea", img_np.shape) # Se define estrategia que SHAP usa para ocultar regiones

# Cálculo correcto: SHAP por clase (para métricas)
mean_abs_shap = []                                          # Importancia SHAP media
# Explicacion SHAP por clase
for i in range(len(emotions)):
    explainer_i = shap.Explainer(lambda x: predict(x)[:, i], masker, algorithm="partition") # explainer independiente por clase
    shap_values_i = explainer_i(np.array([img_np]))
    mean_val = np.abs(shap_values_i.values[0]).mean()
    mean_abs_shap.append(mean_val)

# === Mostrar tabla de métricas ===
print("\n=== Métricas SHAP por emoción ===")
print(f"{'Emoción':<10} {'Probabilidad':<12} {'Importancia SHAP (|media|)':<25} {'Predicción'}")
for i in range(len(emotions)):
    pred_marker = "✓" if i == pred_index else ""
    print(f"{emotions[i]:<10} {probs[i]:<12.6f} {mean_abs_shap[i]:<25.6f} {pred_marker}")

# === Visualización de la tabla con matplotlib ===
df = pd.DataFrame({
    'Emoción': emotions,
    'Probabilidad': [f"{v:.6f}" for v in probs],
    'Importancia SHAP (|mean|)': [f"{v:.6f}" for v in mean_abs_shap],
    'Predicción': ['✓' if i == pred_index else '' for i in range(len(emotions))]
})

fig, ax = plt.subplots()
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2,1.2)
ax.set_title("Métricas SHAP por emoción", fontweight="bold", fontsize=14, pad=20)
plt.show()

# === SHAP global para visualización (solo para image_plot)
explainer_vis = shap.Explainer(predict, masker, algorithm="partition")
shap_values_vis = explainer_vis(np.array([img_np]))

# === Mostrar imagen con SHAP
print("\n[Debug] Mostrando visualización SHAP para todas las emociones...")
shap.image_plot(shap_values_vis, labels=emotions)


''' Prueba con entropia y SHAP mean (guardar .csv) '''
import glob
from tqdm import tqdm
base_dir = "FER2013/subConjuntoFER2013_paraExperimento"
resultados = [] # inicializar todos los results

# Recorrer por clase
for clase in emotions:
    print(f"\nProcesando clase: {clase}")
    dir_clase = os.path.join(base_dir, clase)
    imagenes_clase = sorted(glob.glob(os.path.join(dir_clase, "*.jpg")))[:3]

    if len(imagenes_clase) == 0:
        print(f"[Advertencia] No se encontraron imágenes en {dir_clase}")
        continue

    for path in tqdm(imagenes_clase, desc=f"Clase {clase}"):
        try:
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil.resize((64, 64)))

            # Predicción
            probs = predict(np.array([img_np]))[0]
            pred_index = np.argmax(probs)

            # SHAP por clase
            masker = shap.maskers.Image("inpaint_telea", img_np.shape)
            shap_means = []
            for i in range(len(emotions)):
                explainer_i = shap.Explainer(lambda x: predict(x)[:, i], masker, algorithm="partition")
                shap_values_i = explainer_i(np.array([img_np]))
                mean_val = np.abs(shap_values_i.values[0]).mean()
                shap_means.append(mean_val)

            # Guardar métricas
            resultado = {
                'imagen': os.path.basename(path),
                'clase_esperada': clase,
                'clase_predicha': emotions[pred_index],
                **{f'prob_{emo}': probs[i] for i, emo in enumerate(emotions)},
                **{f'shap_{emo}': shap_means[i] for i, emo in enumerate(emotions)}
            }
            resultados.append(resultado)

        except Exception as e:
            print(f"[Error] Falló imagen {path}: {e}")

# === Guardar CSV ===
os.makedirs("Explicabilidad/Exp_Entropia_SHAPmean", exist_ok=True)
df_final = pd.DataFrame(resultados)
df_final.to_csv("Explicabilidad/Exp_Entropia_SHAPmean/shap_metrica_700.csv", index=False)
print("\nCSV guardado exitosamente como 'shap_metrica_700.csv'")