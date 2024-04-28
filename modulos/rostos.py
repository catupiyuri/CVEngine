import cv2
import os

# Obtém o diretório atual do script
script_dir = os.path.dirname(__file__)

# Caminho para o arquivo haarcascade_frontalface_default.xml
cascade_path = os.path.join(script_dir, "../modelos/haarcascade_rostos.xml")

# Carregar o classificador pré-treinado para detecção de rostos
face_cascade = cv2.CascadeClassifier(cascade_path)

# Função para detectar e desenhar quadrados ao redor dos rostos
def detect_faces(frame):
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Desenhar um quadrado ao redor de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    

    
    # Chamar a função para detectar rostos
    frame = detect_faces(frame)
    
    # Exibir o frame resultante
    cv2.imshow('CVEngine: Detecção de Rostos', frame)
    
    # Parar o loop quando 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()

