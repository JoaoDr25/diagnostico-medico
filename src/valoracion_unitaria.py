import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Se define la ruta del modelo entrenado.
ruta_modelo_entrenado = os.path.join('.', 'models', 'modelo_diagnostico_neumonia.h5')

try:
    modelo = keras.models.load_model(ruta_modelo_entrenado)
except OSError as e:
    print(f"Error al cargar el modelo: {e}. Asegúrate de que el archivo '{ruta_modelo_entrenado}' exista.")
    exit()

# Se define la ruta de una imagen de prueba.
ruta_imagen_nueva = os.path.join('.', 'data', 'chest_xray', 'test', 'PNEUMONIA', 'person3_virus_15.jpeg')
IMG_SIZE = 150

try:
    # Se carga la imagen y se procesa.
    img = cv2.imread(ruta_imagen_nueva)
    if img is None:
        print(f"Error: No se pudo cargar la imagen de la ruta {ruta_imagen_nueva}")
        exit()
    
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_redimensionada = cv2.resize(img_gris, (IMG_SIZE, IMG_SIZE))
    img_normalizada = img_redimensionada / 255.0
    
    # Se le da el formato correcto para que el modelo la pueda procesar.
    imagen_final = np.expand_dims(img_normalizada, axis=0)
    imagen_final = np.expand_dims(imagen_final, axis=-1)
    
except Exception as e:
    print(f"Error durante el preprocesamiento de la imagen: {e}")
    exit()

# Se realiza la predicción.
prediccion_prob = modelo.predict(imagen_final)
probabilidad_neumonia = prediccion_prob[0][0]
print(f"\nProbabilidad de Neumonía: {probabilidad_neumonia:.4f}")

# Se interpreta el resultado.
if probabilidad_neumonia > 0.5:
    diagnostico = "Neumonía"
else:
    diagnostico = "Normal"

print(f"Diagnóstico del modelo: {diagnostico}")