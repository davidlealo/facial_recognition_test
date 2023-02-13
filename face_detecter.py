import cv2
rostros_cascada = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
captura = cv2.VideoCapture(0)

while True:
    _, img = captura.read()

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros = rostros_cascada.detectMultiScale(gris, 1.3, 5)
 
    for (x, y, w, h) in rostros:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Tu carita', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
captura.release()
cv2.destroyAllWindows()