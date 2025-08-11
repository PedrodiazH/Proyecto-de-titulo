import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import sys, os
import pandas as pd
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from lime.lime_image import LimeImageExplainer
import cv2 as cv2

def detectar_y_recortar_cara(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return Image.open(image_path).convert('RGB')  # fallback

    (x, y, w, h) = faces[0]
    cara = img[y:y+h, x:x+w]
    return Image.fromarray(cv2.cvtColor(cara, cv2.COLOR_BGR2RGB))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Approach.ResEmoteNet import ResEmoteNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels
#labelsOrden = ['enojo', 'disgusto', 'miedo', 'feliz', 'neutral', 'tristeza', 'sorpresa']
labelsOrden = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
model = ResEmoteNet()
checkpoint = torch.load('./Weights/fer_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Convertir imagenes a tensor, predecir y devolver probabilidades softmax
def batch_predict(images):
    model.eval()
    images = torch.stack([transform(Image.fromarray(img)).to(device) for img in images], dim=0)
    outputs = model(images)
    return outputs.softmax(dim=1).detach().cpu().numpy()

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = 'FER2013/test/fear/PrivateTest_166793.jpg'  # Agregar ruta de imagen a testear

''' Imagenes aprobadas para resultados: 'RAF-DB/DATASET/test2/surprise/test_0008_aligned.jpg' 
    RAF-DB/DATASET/train/happy/train_00016_aligned.jpg
    PROBAR CON ESTAS:
    RAF-DB/DATASET/explicabilidad_oculta/disgust/test_2321_aligned.jpg
    RAF-DB/DATASET/explicabilidad_oculta/fear/test_2278_aligned.jpg
    RAF-DB/DATASET/explicabilidad_oculta/happy/test_0026_aligned.jpg
    RAF-DB/DATASET/explicabilidad_oculta/surprise/test_0245_aligned.jpg

    PARA FER2013

'''
# Obtener clase
nombre_clase = os.path.basename(os.path.dirname(image_path)).lower()

if nombre_clase not in labelsOrden:
    print(f"Clase '{nombre_clase}' no está en labelsOrden. Revisa el path o labelsOrden.")
    sys.exit(1)

label_real = labelsOrden.index(nombre_clase)

# Preparar imagen y usar lime
img = detectar_y_recortar_cara(image_path)
img_np = np.array(img.resize((128, 128)))  # Lime espera arrays uint8

explainer = LimeImageExplainer() # 

explanation = explainer.explain_instance( # metodo principal para obtener explicaciones
    image=img_np,
    classifier_fn=batch_predict,
    labels=[label_real],        # Clase que buscara explicar lime
    hide_color=0,
    num_samples=5
)
print("Clases explicadas por LIME:", explanation.top_labels) # CLASES MAS RELEVANTES EN EL MUESTREO DE EXPLICABILIDAD, clases activadas por modelo LIME

label_pred = explanation.top_labels[0]

top_label = explanation.top_labels[0]
print(f"\nClase real: {labelsOrden[label_real]} | Clase predicha: {labelsOrden[label_pred]}")
#if label_real != label_pred:
#    print("El modelo no predijo correctamente esta imagen.")

# Obtener imagen explicada con LIME para la clase predicha
try:
    temp, mask = explanation.get_image_and_mask( # mask es 
        label=label_pred,
        positive_only=True,
        num_features=10,     # numero de superpixeles relevantes
        hide_rest=False
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME - Clase predicha: {labelsOrden[label_pred]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

except KeyError:
    print(f"LIME no generó explicacion para la clase predicha: {label_pred}")

# Mapa de calor con superpixeles
segmentos = explanation.segments  # matriz 2D de índices por superpíxel
pesos = dict(explanation.local_exp[label_pred])  # importancia de cada superpíxel
heatmap = np.zeros_like(segmentos, dtype=np.float32)
# asignar pesos de importancia por superpixeles
for sp_index, weight in pesos.items():
    heatmap[segmentos == sp_index] = weight
heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) # normalizacion

plt.figure(figsize=(6, 6))
plt.imshow(img_np)
plt.imshow(heatmap_norm, cmap='spring', alpha=0.4)  # superposición
plt.title(f"Mapa de calor LIME - Clase: {labelsOrden[label_pred]}")
plt.axis('off')
plt.colorbar(label='Importancia')
plt.tight_layout()
plt.show()


# Curva explicabilidad
''' ordenar de mayor a menor
    predecir antes y despues

    ocultar lo q predice positivo
    y hacer nuevo dataset de test con imagenes con partes explicables tapadas

    mapear lo q dice lime a vectoresde caracteristicas comun

    shap revisar


'''
'''
from tqdm import tqdm
def ocultar_regiones_importantes(img_np, mask, hide_color=(0, 0, 0)):
    output = img_np.copy()
    output[mask == 1] = hide_color
    return output

directorio_test = 'FER2013/test'
nuevo_dataset = 'FER2013/explicabilidad_oculta'
os.makedirs(nuevo_dataset, exist_ok=True)

imagenes_originales = []
segmentaciones = []
explicaciones = []
clase_real = []
for clase in labelsOrden:
    carpeta_clase = os.path.join(directorio_test, clase)
    carpeta_salida = os.path.join(nuevo_dataset, clase)
    os.makedirs(carpeta_salida, exist_ok=True)

    imagenes = sorted(os.listdir(carpeta_clase))[:100]
    for img_file in tqdm(imagenes, desc=f"Procesando {clase}"):
        path_img = os.path.join(carpeta_clase, img_file)
        if not imagenes:
            print(f"[Advertencia] Clase '{clase}' no tiene imágenes.")
            continue
        try:
            img = detectar_y_recortar_cara(path_img)
            img_np = np.array(img.resize((128, 128)))

            label_index = labelsOrden.index(clase)

            explanation = explainer.explain_instance(
                image=img_np,
                classifier_fn=batch_predict,
                labels=[label_index],
                num_samples=5
            )
            imagenes_originales.append(img_np)
            segmentaciones.append(explanation.segments)
            explicaciones.append(explanation)
            clase_real.append(label_index)

            # Si LIME no devuelve esa clase, usar top_label como fallback solo para neutral/happy
            if label_index not in explanation.local_exp:
                if clase in ['neutral', 'happy']:
                    print(f"[Aviso] Clase '{clase}' no explicada, usando top_label en {img_file}")
                    label_index = explanation.top_labels[0]
                else:
                    print(f"[Advertencia] Clase '{clase}' no explicada, imagen descartada: {img_file}")
                    continue  # Salta solo si no es neutral/happy

            _, mask = explanation.get_image_and_mask(
                label=label_index,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )

            img_oculta = ocultar_regiones_importantes(img_np, mask)
            Image.fromarray(img_oculta).save(os.path.join(carpeta_salida, img_file))

        except Exception as e:
            print(f"Error con {path_img}: {e}")


# Superpixeles
from skimage.segmentation import mark_boundaries

def mostrar_superpixeles(imagen, segmentacion):
    # Asegúrate de que la imagen esté en formato RGB (o usa cmap si es escala de grises)
    if imagen.ndim == 2:  # Escala de grises
        imagen_rgb = np.stack([imagen]*3, axis=-1)
    elif imagen.shape[2] == 1:
        imagen_rgb = np.concatenate([imagen]*3, axis=-1)
    else:
        imagen_rgb = imagen

    # Visualización con los límites de los superpixeles
    imagen_bordeada = mark_boundaries(imagen_rgb / 255.0, segmentacion)

    plt.figure(figsize=(6, 4))
    plt.imshow(imagen_bordeada)
    plt.title("Superpíxeles demarcados",fontsize=11, fontweight='semibold')
    plt.axis('off')
    plt.show()
mostrar_superpixeles(img_np, segmentos) '''
