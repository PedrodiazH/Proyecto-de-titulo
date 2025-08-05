
---   
<div align="center">    

# Desarrollo de algoritmos de IA explicables para el reconocimiento de emociones basado en videos de expresiones faciales

Pedro Díaz Herrera & Marcos Carripan Moya

Profesor guía: Christopher Flores Jara

[![Documento](https://img.shields.io/badge/Documento%20de%20tesis-8A2BE2)](#)
[![Revisión bibliográfica](https://img.shields.io/badge/Google%20Drive-Revisión%20bibliográfica-285F4?logo=googledrive&logoColor=fff)](https://drive.google.com/drive/u/0/folders/1PsqGXum6_dIXtJz5SpGycjOnqgmWVMOc)
[![Extras](https://img.shields.io/badge/Material%20extra-00FFFF)]()

</div>

## Descripción general

Este proyecto es realizado para optar por el título de Ingeniería Civil en Automatización en la Universidad del Bío-Bío.

Consiste en una revisión del estado del arte del reconocimiento de expresiones faciales, la implementación de algoritmos que interpreten un estado emocional en una escena, y explicabilidad del modelo de aprendizaje utilizado.

### Exploración conjuntos de datos 
* [FER2013](https://github.com/PedrodiazH/Proyecto-de-titulo/blob/main/Extras/AnalisisFER2013.ipynb)
* [AffectNet](https://github.com/PedrodiazH/Proyecto-de-titulo/blob/main/Extras/Analisis_AffectNet.ipynb)
* [RAF-DB](https://github.com/PedrodiazH/Proyecto-de-titulo/blob/main/Extras/Analisis_RAF.ipynb)

## Requerimientos (fue probado bajo estas versiones)
* Python 3.9+
* PyTorch 2.7.1+cu118
* torchvision 0.22.1
* numpy
* scikit-learn 1.7.0
* opencv-python 4.12.0
* matplotlib 3.10.3
* seaborn
* lime 0.2.0.1
* shap 0.48.0
* Bibliotecas adicionales en: [requirements.txt]()

## Flujo de trabajo
<div align="center">
  <img src="https://github.com/PedrodiazH/Proyecto-de-titulo/blob/main/Extras/Workflow.jpg?raw=true" alt="Flujo de trabajo" width="540"/>
</div>

## Arquitectura del modelo
El modelo consta de una CNN, un bloque SE para potenciar la CNN y una red residual para mejorar la clasificación de expresiones faciales
<div align="center">
  <img src="https://github.com/PedrodiazH/Proyecto-de-titulo/blob/main/Extras/ResEmoteNet.jpg?raw=true" alt="Imagen grande" width="420"/>
  <img src="https://github.com/PedrodiazH/Proyecto-de-titulo/blob/main/Extras/bloque_SE_Redresidual.jpg?raw=true" alt="Imagen pequeña 1" width="360"/>
</div>



