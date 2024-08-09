import cv2
import numpy as np

# Cargar el clasificador de rostros y el modelo LBPH para reconocimiento
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Cargar el modelo previamente entrenado
recognizer.read('trained_model.yml')

# Diccionario para nombres de personas
names = {0: "David Leal"}  # Agrega más nombres según tu entrenamiento

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
            name = names[id_]
        else:
            name = "Desconocido"
        
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar la imagen con reconocimiento facial
    cv2.imshow('Face Recognition', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
