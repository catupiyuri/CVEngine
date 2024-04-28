import cv2
import os

# Obtém o diretório atual do script
script_dir = os.path.dirname(__file__)

# Caminho para o arquivo haarcascade_bicicletas.xml
cascade_path = os.path.join(script_dir, "../modelos/haarcascade_bicicletas.xml")

# Carregar o classificador pré-treinado para detecção de bicicletas
bike_cascade = cv2.CascadeClassifier(cascade_path)

# Função para detectar e desenhar quadrados ao redor das bicicletas
def detect_bikes(frame):
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar equalização de histograma para melhorar o contraste
    gray = cv2.equalizeHist(gray)
    
    # Detectar bicicletas na imagem
    bikes = bike_cascade.detectMultiScale(gray,1.01, 1)
    
    # Desenhar um quadrado ao redor de cada bicicleta detectada
    for (x, y, w, h) in bikes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame

# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    
    # Inverter horizontalmente o frame
    frame = cv2.flip(frame, 1)
    
    # Chamar a função para detectar bicicletas
    frame = detect_bikes(frame)
    
    # Exibir o frame resultante
    cv2.imshow('Deteccao de Bicicletas', frame)
    
    # Parar o loop quando 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
