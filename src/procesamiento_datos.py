import os
import cv2
import numpy as np
import pandas as pd # Se importa para usar el DataFrame

# Definir la ruta de los datos, usando 'os.path.join' para asegurar la compatibilidad con todos los sistemas operativos. Cambiar a ruta local.
ruta_principal = os.path.join('D:\Juan Camilo\Escritorio\Python\SenaSoft Inteligencia Artificial 2025\diagnostico-medico', 'data', 'chest_xray') 
ruta_entrenamiento = os.path.join(ruta_principal, 'train')

# Se define el tamaño deseado para las imágenes. Todas las imágenes serán redimensionadas a esta dimensión.
IMG_SIZE = 150

# Ruta del archivo donde se guardarán los datos procesados.
DATOS_PROCESADOS_FILENAME = os.path.join('', 'models', 'datos_procesados.npz')

# Lista para almacenar las imágenes y sus etiquetas correspondientes.
datos = []

def procesar_imagenes_y_etiquetar(ruta_carpeta, etiqueta):
    for nombre_archivo in os.listdir(ruta_carpeta):
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
        
        # Ignorar subcarpetas.
        if os.path.isdir(ruta_completa):
            continue

        try:
            # Cargar la imagen y convertirla a escala de grises.
            img = cv2.imread(ruta_completa)
            if img is None: continue
            img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Redimensionar la imagen a un tamaño uniforme.
            img_redimensionada = cv2.resize(img_gris, (IMG_SIZE, IMG_SIZE))
            
            # Normalizar los píxeles (valores entre 0 y 1).
            img_normalizada = img_redimensionada / 255.0
            
            # Agregar la imagen y su etiqueta a la lista.
            datos.append([img_normalizada, etiqueta])
        except Exception as e:
            print(f"Error al procesar la imagen {ruta_completa}: {e}")

# Procesar las imágenes para cada categoría.
print("Procesando imágenes normales...")
procesar_imagenes_y_etiquetar(os.path.join(ruta_entrenamiento, 'NORMAL'), 0) # 0 = Normal
print("Procesando imágenes de neumonía...")
procesar_imagenes_y_etiquetar(os.path.join(ruta_entrenamiento, 'PNEUMONIA'), 1) # 1 = Neumonía

# Convertir la lista en un DataFrame de Pandas para su manipulación
df_imagenes = pd.DataFrame(datos, columns=['imagen', 'etiqueta'])

# Mezclar los datos de manera aleatoria para evitar sesgos durante el entrenamiento.
df_imagenes = df_imagenes.sample(frac=1).reset_index(drop=True)

# Guardar las imágenes y etiquetas procesadas en un archivo comprimido.
np.savez_compressed(DATOS_PROCESADOS_FILENAME, imagenes=df_imagenes['imagen'].tolist(), etiquetas=df_imagenes['etiqueta'].tolist())
print(f"Datos procesados guardados en '{DATOS_PROCESADOS_FILENAME}'.")