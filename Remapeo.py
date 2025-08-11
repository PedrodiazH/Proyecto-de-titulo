# Este código se realiza en el caso de existir un re-mapeo de las etiquetas de las clases 
# dadas por el dataset y las q tiene el modelo
''' from torchvision.datasets import ImageFolder
    dataset = ImageFolder(root="FER2013/test", transform=...)
    print(dataset.class_to_idx)                     # Asi se comprueba las distribución del diccionario de clases           
    Actualmente este es: {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}''' 
import pandas as pd
# Cargar el archivo de labels
df = pd.read_csv('RAF-DB/DATASET/subConjunto_paraExperimento/test_explicabilidad_experimentoOriginales.csv')

# Mapeo de clase anterior a clase nueva
remap = {0: 3, 1: 4, 2: 5, 3: 0, 4: 6, 5: 2, 6: 1}
# remap_AffectNet = {0: 3, 1: 4, 2: 5, 3: 0, 4: 6, 5: 2, 6: 1} # Dataset no oficial affectnet7 por lo que se tuvo q reacondicionar

# Remapeo de las clases
df['label'] = df['label'].map(remap)

# Guardar el nuevo CSV
df.to_csv('RAF-DB/DATASET/subConjunto_paraExperimento/test_explicabilidad_experimentoOriginalesREMAPPED.csv', index=False)
print('Remapeo exitoso')