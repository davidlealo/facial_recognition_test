import cv2
import numpy as np
import os

# Crear el detector de rostros
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# Crear el reconocedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Carpeta con imágenes de entrenamiento
data_folder = '/Users/davidlealolivares/Documents/repos/facial_recognition_test/detectar_estudiante/img'
images = []
labels = []

for label_folder in os.listdir(data_folder):
    label_path = os.path.join(data_folder, label_folder)
    
    # Ignorar archivos como .DS_Store
    if not os.path.isdir(label_path):
        continue
    
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        
        # Asegurarse de que sea un archivo de imagen válido
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(int(label_folder))  # Suponiendo que los nombres de las carpetas son etiquetas numéricas

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.save('/Users/davidlealolivares/Documents/repos/facial_recognition_test/detectar_estudiante/trained_model.yml')