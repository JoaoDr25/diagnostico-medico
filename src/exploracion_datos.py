import os
import cv2
import matplotlib.pyplot as plt

# Definir la ruta de los datos, usando 'os.path.join' para asegurar la compatibilidad con todos los sistemas operativos. Cambiar a ruta local.
ruta_principal = os.path.join('D:\Juan Camilo\Escritorio\Python\SenaSoft Inteligencia Artificial 2025\diagnostico-medico', 'data', 'chest_xray') 
ruta_entrenamiento = os.path.join(ruta_principal, 'train')
ruta_normal = os.path.join(ruta_entrenamiento, 'NORMAL')
ruta_neumonia = os.path.join(ruta_entrenamiento, 'PNEUMONIA')

# Contar el número de archivos en cada categoría para verificar el balance del dataset.
num_normal = len(os.listdir(ruta_normal))
num_neumonia = len(os.listdir(ruta_neumonia))

print(f'Número de imágenes normales: {num_normal}')
print(f'Número de imágenes con neumonía: {num_neumonia}')

# Obtener una lista de los nombres de archivos para cada categoría.
archivos_normal = [f for f in os.listdir(ruta_normal) if os.path.isfile(os.path.join(ruta_normal, f))]
archivos_neumonia = [f for f in os.listdir(ruta_neumonia) if os.path.isfile(os.path.join(ruta_neumonia, f))]

if not archivos_normal or not archivos_neumonia:
    print("No se encontraron archivos en una o ambas categorías.")
else:
    # Seleccionar la primera imagen de cada categoría como ejemplo.
    imagen_normal_ejemplo = archivos_normal[0]
    imagen_neumonia_ejemplo = archivos_neumonia[0]

    # Construir las rutas completas a las imágenes de ejemplo.
    ruta_imagen_normal = os.path.join(ruta_normal, imagen_normal_ejemplo)
    ruta_imagen_neumonia = os.path.join(ruta_neumonia, imagen_neumonia_ejemplo)

    # Cargar las imágenes usando OpenCV.
    imagen_normal = cv2.imread(ruta_imagen_normal)
    imagen_neumonia = cv2.imread(ruta_imagen_neumonia)

    if imagen_normal is not None and imagen_neumonia is not None:
        # Convertir las imágenes a escala de grises, que es el formato ideal para las radiografías.
        imagen_normal_grises = cv2.cvtColor(imagen_normal, cv2.COLOR_BGR2GRAY)
        imagen_neumonia_grises = cv2.cvtColor(imagen_neumonia, cv2.COLOR_BGR2GRAY)

        # Crear una figura para mostrar ambos gráficos.
        plt.figure(figsize=(10,5))
        
        # Mostrar la radiografía normal en el primer subplot.
        plt.subplot(1,2,1)
        plt.imshow(imagen_normal_grises, cmap='gray')
        plt.title('Radiografía Normal')
        plt.axis('off')

        # Mostrar la radiografía con neumonía en el segundo subplot.
        plt.subplot(1,2,2)
        plt.imshow(imagen_neumonia_grises, cmap='gray')
        plt.title('Radiografía con Neumonía')
        plt.axis('off')
        
        # Mostrar los gráficos en una ventana.
        plt.show()

        print("La exploración de datos se ha completado exitosamente.")
    else:
        print("Error al cargar las imágenes. Verifique las rutas y los archivos.")