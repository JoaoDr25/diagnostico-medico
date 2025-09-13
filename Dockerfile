# Usa una imagen de Python base.
FROM python:3.9-bullseye

# Instala las dependencias del sistema operativo necesarias para OpenCV.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1

# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copia los archivos de requerimientos y los instala.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el código del proyecto al contenedor.
COPY . .

# Expone el puerto por el que correrá la aplicación de Streamlit.
EXPOSE 8501

# Comando para ejecutar la aplicación cuando se inicie el contenedor.
CMD ["streamlit", "run", "src/app.py"]
