import cv2
import os

# Obtém o diretório atual do script
script_dir = os.path.dirname(__file__)

# Caminho para o arquivo haarcascade_cars.xml
cascade_path = os.path.join(script_dir, "../modelos/haarcascade_carros.xml")

# Carregar o classificador pré-treinado para detecção de carros
car_cascade = cv2.CascadeClassifier(cascade_path)

# Função para detectar e desenhar quadrados ao redor dos carros
def detect_cars(frame):
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar carros na imagem
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Desenhar um quadrado ao redor de cada carro detectado
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    
    # Inverter horizontalmente o frame
    frame = cv2.flip(frame, 1)
    
    # Chamar a função para detectar carros
    frame = detect_cars(frame)
    
    # Exibir o frame resultante
    cv2.imshow('Deteccao de Carros', frame)
    
    # Parar o loop quando 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
