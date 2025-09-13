import os
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Definir el tamaño de la imagen que el modelo espera.
IMG_SIZE = 150

@st.cache_resource
def load_model():
    """Carga el modelo de Keras desde el archivo .h5."""
    try:
        # Se construye la ruta absoluta al modelo.
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join('..', 'models', 'modelo_diagnostico_neumonia.h5')
        
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}. Asegúrate de que el archivo 'modelo_diagnostico_neumonia.h5' esté en la carpeta 'models'.")
        return None

def preprocess_image(image):
    """Convierte la imagen cargada por el usuario al formato que el modelo espera."""
    # Lee el objeto de Streamlit como un arreglo de bytes y luego lo decodifica.
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, 1)
    
    # Convierte a escala de grises.
    img_gris = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Redimensiona la imagen.
    img_redimensionada = cv2.resize(img_gris, (IMG_SIZE, IMG_SIZE))
    
    # Normaliza los píxeles.
    img_normalizada = img_redimensionada / 255.0
    
    # Añade dimensiones extra para que coincida con el formato de entrada del modelo.
    img_final = np.expand_dims(img_normalizada, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)
    
    return img_final

st.set_page_config(layout="wide")
# Título y descripción de la aplicación.
st.title("🩺 Diagnóstico de Neumonía por IA")
st.write("Carga una radiografía de tórax para obtener un diagnóstico. Es una herramienta de apoyo, no un reemplazo para el diagnóstico médico.")

# Carga el modelo de IA.
model = load_model()

# Widget para que el usuario pueda subir un archivo.
uploaded_file = st.file_uploader("Elige una imagen de radiografía...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Muestra la imagen que el usuario cargó.
    st.image(uploaded_file, caption='Radiografía Cargada', width="stretch")
    st.write("")
    
    if model:
        # Crea un botón para iniciar el diagnóstico.
        if st.button('Diagnosticar'):
            with st.spinner('Analizando la radiografía...'):
                # Preprocesar la imagen.
                processed_image = preprocess_image(uploaded_file)
                
                # Realizar la predicción.
                prediccion_prob = model.predict(processed_image)
                probabilidad_neumonia = prediccion_prob[0][0]
                
                # Interpretar y mostrar el resultado.
                if probabilidad_neumonia > 0.5:
                    diagnostico = "Neumonía"
                    color = "red"
                else:
                    diagnostico = "Normal"
                    color = "green"
                    
                st.markdown(f"**Resultado:** <span style='color:{color}; font-size: 24px;'>{diagnostico}</span>", unsafe_allow_html=True)
                st.write(f"Probabilidad de Neumonía: **{probabilidad_neumonia:.2f}**")