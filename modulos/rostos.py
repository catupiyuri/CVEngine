from flask import Flask, render_template, Response, jsonify
import cv2
import freenect
import numpy as np

app = Flask(__name__)

# Carregar o classificador pré-treinado para detecção de rostos
face_cascade = cv2.CascadeClassifier("../modelos/haarcascade_rostos.xml")

face_count = 0  # Variável global para armazenar o número de rostos detectados

# Modifique a função detect_faces para atualizar o número de rostos detectados
def detect_faces(frame):
    global face_count
    
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=5, minSize=(30, 30))
    
    # Atualizar o número de rostos detectados
    face_count = len(faces)
    
    # Desenhar um quadrado ao redor de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

# Adicione uma nova rota para retornar o número de rostos detectados
@app.route('/face_count')
def get_face_count():
    return jsonify({'count': face_count})

def generate_frames():
    while True:
        # Capturar frame do Kinect
        frame, _ = freenect.sync_get_video()
        
        # Converter para RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Chamar a função para detectar rostos
        frame = detect_faces(frame)
        
        # Codificar o frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        # Converter o buffer para bytes
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('rostos.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

