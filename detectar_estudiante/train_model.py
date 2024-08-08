import cv2
import numpy as np
import os

# Crear el detector de rostros
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

# Crear el reconocedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Carpeta con imágenes de entrenamiento
data_folder = 'img'

faces = []
labels = []

for label in os.listdir(data_folder):
    for image_file in os.listdir(os.path.join(data_folder, label)):
        img_path = os.path.join(data_folder, label, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces_detected = face_cascade.detectMultiScale(img)
        for (x, y, w, h) in faces_detected:
            faces.append(img[y:y+h, x:x+w])
            labels.append(int(label))

recognizer.train(faces, np.array(labels))
recognizer.save('trained_model.yml')
