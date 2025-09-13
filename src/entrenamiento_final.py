import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Se define la ruta de los datos procesados. Cambiar a ruta local.
ruta_datos_procesados = os.path.join('D:\Juan Camilo\Escritorio\Python\SenaSoft Inteligencia Artificial 2025\diagnostico-medico', 'models', 'datos_procesados.npz') 

try:
    # Se carga el archivo de datos.
    with np.load(ruta_datos_procesados, allow_pickle=True) as data:
        imagenes = data['imagenes']
        etiquetas = data['etiquetas']
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{ruta_datos_procesados}'.")
    exit()

# Se reformatea el arreglo de imágenes para que sea compatible con la entrada de la CNN.
X = np.array(imagenes.tolist()).reshape(-1, 150, 150, 1)
y = np.array(etiquetas.tolist())

# Se dividen los datos en conjuntos de entrenamiento y prueba.
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Se define la arquitectura de la red neuronal.
modelo = Sequential([
    # Capa convolucional para extraer características.
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D((2, 2)), # Capa de pooling para reducir la dimensionalidad.
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(), # Aplanar la salida para la capa densa.
    Dense(512, activation='relu'),
    Dropout(0.5), # Capa de dropout para prevenir el sobreajuste.
    # Capa de salida con activación 'sigmoid' para clasificación binaria.
    Dense(1, activation='sigmoid')
])

# Compilar el modelo con el optimizador, la función de pérdida y las métricas.
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento.
historial = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=10, batch_size=32, validation_data=(X_prueba, y_prueba))

# Se guarda el modelo entrenado para su uso posterior.
modelo_guardar_ruta = os.path.join('.', 'models', 'modelo_diagnostico_neumonia.h5')
modelo.save(modelo_guardar_ruta)
print(f"\n¡Entrenamiento finalizado! El modelo se ha guardado en '{modelo_guardar_ruta}'.")

# Visualizar el progreso del entrenamiento.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del Modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend()
plt.show()