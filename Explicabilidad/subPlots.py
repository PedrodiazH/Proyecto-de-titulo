# Codigo para hacer subplots
from matplotlib import pyplot as plt
from PIL import Image
# Cargar imágenes directamente
img1 = Image.open("RAF-DB/DATASET/explicabilidad_oculta/disgust/test_2321_aligned.jpg")
img2 = Image.open("RAF-DB/DATASET/explicabilidad_oculta/fear/test_2278_aligned.jpg")
img3 = Image.open("RAF-DB/DATASET/explicabilidad_oculta/happy/test_0026_aligned.jpg")
img4 = Image.open("RAF-DB/DATASET/explicabilidad_oculta/surprise/test_0245_aligned.jpg")

imagenes = [img1, img2, img3, img4]
titulos = ['clase disgusto', 'clase miedo', 'clase feliz', 'clase sorpresa']
# Mostrar en layout 2x2
plt.figure(figsize=(6, 4))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(imagenes[i])
    plt.axis('off')
    plt.title(titulos[i])
plt.tight_layout()
plt.show()

