
import cv2
import numpy as np
import csv
from datetime import datetime

# Cargar el clasificador de rostros y el modelo LBPH para reconocimiento
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cargar el modelo previamente entrenado
recognizer.read('/Users/davidlealolivares/Documents/repos/facial_recognition_test/detectar_estudiante/trained_model.yml')

# Diccionario para nombres de personas
names = {0: "Jackie Chan", 1: "Pedro Pascal", 2: "David Leal", 3: "Ana Maria"}  # Agrega más nombres según tu entrenamiento

# Abrir archivo CSV para guardar las detecciones
with open('detections.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Nombre", "Fecha y Hora"])  # Escribir encabezados

    # Captura de video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        exit()

    while True:
        # Captura frame por frame
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir el frame.")
            break

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extraer la región de interés (ROI) para el rostro detectado
            roi_gray = gray[y:y+h, x:x+w]
            
            # Reconocer el rostro
            id_, confidence = recognizer.predict(roi_gray)
            
            # Mostrar el nombre y el nivel de confianza en el marco
            if confidence < 100:
                name = names.get(id_, "Desconocido")
            else:
                name = "Desconocido"
            
            # Guardar la detección con la marca de tiempo en el archivo CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([name, timestamp])

            # Obtener el tamaño del texto
            (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Coordenadas del rectángulo de fondo
            rect_x1, rect_y1 = x, y - text_height - 10
            rect_x2, rect_y2 = x + text_width, y

            # Dibujar el rectángulo de fondo
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), cv2.FILLED)  # Fondo negro

            # Dibujar el texto sobre el rectángulo
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Mostrar el frame resultante
        cv2.imshow('Frame', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura
    cap.release()
    cv2.destroyAllWindows()
