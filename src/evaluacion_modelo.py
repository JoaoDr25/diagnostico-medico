import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Se define la ruta de los datos procesados.
ruta_datos_procesados = os.path.join('.', 'models', 'datos_procesados.npz')

try:
    with np.load(ruta_datos_procesados, allow_pickle=True) as data:
        imagenes = data['imagenes']
        etiquetas = data['etiquetas']
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{ruta_datos_procesados}'.")
    exit()

X = np.array(imagenes.tolist()).reshape(-1, 150, 150, 1)
y = np.array(etiquetas.tolist())

# Se dividen los datos nuevamente, asegurando que el conjunto de prueba sea el mismo.
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Se define la ruta del modelo entrenado.
ruta_modelo_entrenado = os.path.join('..', 'models', 'modelo_diagnostico_neumonia.h5')

try:
    modelo = keras.models.load_model(ruta_modelo_entrenado)
except OSError as e:
    print(f"Error al cargar el modelo: {e}. Asegúrate de que el archivo '{ruta_modelo_entrenado}' exista.")
    exit()

# Hacer predicciones sobre el conjunto de prueba.
y_pred_prob = modelo.predict(X_prueba)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Generar el reporte de clasificación y la matriz de confusión.
print("Reporte de Clasificación:")
print(classification_report(y_prueba, y_pred))

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_prueba, y_pred)
print(cm)

# Visualizar la matriz de confusión.
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Neumonía'], yticklabels=['Normal', 'Neumonía'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()