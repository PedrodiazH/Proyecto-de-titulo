''' Este codigo sirve para crear un .csv con las etiquetas dadas la distribución del dataset descargado (FER2013)
    Luego se tiene que re-mapear en caso de ser necesario
    '''
import os
import pandas as pd

# Ruta base del conjunto de test
base_path = "RAF-DB/DATASET/subConjunto_paraExperimento"  #"FER2013/test" "RAF-DB/DATASET/test2"
class_to_idx = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6} # distribucion del dataset

data = []

for class_name, class_idx in class_to_idx.items():
    class_folder = os.path.join(base_path, class_name)
    for fname in os.listdir(class_folder):
        if fname.lower().endswith((".jpg", ".png")):
            rel_path = os.path.join(class_name, fname)  # relativo a base_path
            data.append([rel_path, class_idx])

# Guardar el CSV
df = pd.DataFrame(data, columns=["filename", "label"])
df.to_csv("RAF-DB/DATASET/subConjunto_paraExperimento/test_explicabilidad_experimentoOriginales.csv", index=False)

print(f"CSV generado con {len(df)} imágenes.")
labels = pd.read_csv("RAF-DB/DATASET/subConjunto_paraExperimento/test_explicabilidad_experimentoOriginales.csv")["label"]
print(labels.value_counts().sort_index())