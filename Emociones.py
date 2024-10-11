#pip install deepface
#pip install opencv
#pip install mediapipe
#pip install tf-keras

# Importamos las librerias
from deepface import DeepFace
import cv2
import mediapipe as mp
import numpy as np

# Inicializamos la detección de rostros
detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence=0.8, model_selection=0)
cap = cv2.VideoCapture(0)

# Función para seleccionar la imagen de la emoción detectada
def emotionImage(emotion):
    # Emojis
    if emotion == 'Feliz': return cv2.imread('Emociones/Felicidad.png')
    if emotion == 'Enojado': return cv2.imread('Emociones/Enojado.png')
    if emotion == 'Sorprendido': return cv2.imread('Emociones/Sorpresa.png')
    if emotion == 'Triste': return cv2.imread('Emociones/Tristeza.png')
    if emotion == 'Neutral': return cv2.imread('Emociones/Neutral.png')
    if emotion == 'Disgustado': return cv2.imread('Emociones/Disgustado.png')
    if emotion == 'Miedoso': return cv2.imread('Emociones/Miedo.png')

while True:
    # Leemos los fotogramas
    ret, frame = cap.read()
    if not ret:
        break

    # Corrección de color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Creamos un espacio para el emoji y el nombre de la emoción
    emoji_space = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
    nFrame = cv2.hconcat([frame, emoji_space])

    # Procesamos
    resrostros = rostros.process(rgb)

    # Detección
    if resrostros.detections is not None:
        # Registramos
        for rostro in resrostros.detections:
            # Extraemos información de ubicación
            al, an, c = frame.shape
            box = rostro.location_data.relative_bounding_box
            xi, yi, w, h = int(box.xmin * an), int(box.ymin * al), int(box.width * an), int(box.height * al)
            xf, yf = xi + w, yi + h

            # Dibujamos el cuadro de detección del rostro
            cv2.rectangle(nFrame, (xi, yi), (xf, yf), (255, 255, 0), 2)

            # Analizamos la emoción
            info = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)

            emociones = 'neutral'
            if len(info) > 0:
                emociones = info[0]['dominant_emotion']

            # Traducimos las emociones al español y se asigna a la variable emociones
            # el nombre de la emoción detectada
            if emociones == 'disgust': emociones = 'Disgustado'
            if emociones == 'fear': emociones = 'Miedoso'
            if emociones == 'happy': emociones = 'Feliz'
            if emociones == 'sad': emociones = 'Triste'
            if emociones == 'surprise': emociones = 'Sorprendido'
            if emociones == 'angry': emociones = 'Enojado'
            if emociones == 'neutral': emociones = 'Neutral'

            # Leemos la imagen del emoji correspondiente a la emoción detectada y se carga 
            # en la variable img para ser mostrada en la parte derecha de la ventana
            img = emotionImage(emociones)
            if img is not None:
                img = cv2.resize(img, (emoji_space.shape[1], emoji_space.shape[0]))  # Redimensionamos la imagen del emoji
                nFrame[:, frame.shape[1]:] = img  # Insertamos el emoji en el espacio adicional
            else:
                print('No se encontró la imagen')

            # Mostramos el nombre de la emoción en el espacio adicional
            cv2.putText(nFrame, str(emociones), (frame.shape[1] + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostramos los fotogramas
    cv2.imshow("Detección de Emociones", nFrame)

    # Leemos el teclado, si se presiona escape se cierra la ventana
    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
