🩺 Sistema de Diagnóstico de Neumonía por IA


- Descripción del Proyecto:
Este proyecto es una solución de Inteligencia Artificial para el diagnóstico de neumonía a partir de radiografías de tórax. Utiliza una Red Neuronal Convolucional (CNN), un modelo de aprendizaje profundo, para clasificar imágenes como Normal o Neumonía.


- El objetivo principal es servir como una herramienta de apoyo para el personal médico, ayudando a acelerar el proceso de diagnóstico y mejorando la eficiencia, especialmente en entornos con acceso limitado a especialistas.


- Características Principales:
Diagnóstico Preciso: El modelo ha sido evaluado con métricas clave como la precisión y la matriz de confusión, demostrando una alta tasa de aciertos y un bajo número de falsos negativos, que es el error más crítico en este contexto médico.


- Transparencia Ética: La solución se presenta como un complemento para ofrecer una segunda opinión, no como un reemplazo de un médico, minimizando el riesgo de diagnósticos erróneos y promoviendo un uso responsable de la IA.


- Tecnología de Vanguardia: Implementado con Python y las librerías de machine learning de la industria, incluyendo TensorFlow y Keras.


- Tecnologías Utilizadas:

Python
TensorFlow / Keras
NumPy
OpenCV
Streamlit
Matplotlib / Seaborn
scikit-learn


- Cómo Ejecutar el Proyecto:

- Configurar el entorno: Asegúrate de tener Python y Conda instalados.

conda create --name diagnostico-datos-medicos python=3.9
conda activate diagnostico-datos-medicos


- Instalar librerías:

pip install -r requirements.txt


- Obtener los datos: Descarga el conjunto de datos de radiografías y colócalo en la ruta data/chest_xray.


- Ejecutar los scripts en orden:

src/exploracion_datos.py: Explora el dataset por primera vez.

src/procesamiento_datos.py: Prepara las imágenes y guarda el dataset.

src/entrenamiento_final.py: Construye, entrena y guarda el modelo de IA.

src/evaluacion_modelo.py: Muestra el rendimiento del modelo con métricas y una matriz de confusión.

src/valoracion_unitaria.py: Permite hacer un diagnóstico con una sola imagen.

streamlit run src/app.py: Ejecutar la aplicación web.

