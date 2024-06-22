from flask import Flask, render_template, Response, jsonify, request
import cv2
import freenect
import numpy as np

app = Flask(__name__)

# Carregar o classificador pré-treinado para detecção de carros
car_cascade = cv2.CascadeClassifier("../modelos/haarcascade_carros.xml")

car_count = 0  # Variável global para armazenar o número de carros detectados
traffic_count = 0  # Variável global para armazenar o número de carros que cruzaram a linha
tracked_cars = []  # Lista para armazenar os centróides dos carros rastreados
use_kinect = True  # Variável global para alternar entre Kinect e vídeo

# Configurar a resolução desejada (320x240)
VIDEO_W = 320
VIDEO_H = 240

# Definir a linha na parte inferior da tela (10 pixels a partir da borda inferior)
LINE_POSITION = VIDEO_H - 104

# Distância máxima para associar um carro detectado a um carro rastreado
MAX_DISTANCE = 50

def update_tracked_cars(new_centroids):
    global tracked_cars, traffic_count
    updated_tracked_cars = []

    for (new_x, new_y) in new_centroids:
        matched = False
        for (tracked_x, tracked_y) in tracked_cars:
            distance = np.sqrt((new_x - tracked_x)**2 + (new_y - tracked_y)**2)
            if distance < MAX_DISTANCE:
                updated_tracked_cars.append((new_x, new_y))
                matched = True
                break
        if not matched:
            updated_tracked_cars.append((new_x, new_y))
            if new_y > LINE_POSITION:
                traffic_count += 1

    tracked_cars = updated_tracked_cars

def detect_cars(frame):
    global car_count
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=6, minSize=(5, 5))
    car_count = len(cars)
    cv2.line(frame, (0, LINE_POSITION), (VIDEO_W, LINE_POSITION), (0, 255, 0), 2)
    new_centroids = []

    for (x, y, w, h) in cars:
        centroid_x = x + w // 2
        centroid_y = y + h // 2
        new_centroids.append((centroid_x, centroid_y))
    
    update_tracked_cars(new_centroids)
    
    return frame

def generate_frames():
    global use_kinect
    cap = cv2.VideoCapture('video.avi')  # Carregar o vídeo pré-gravado

    while True:
        if use_kinect:
            frame, _ = freenect.sync_get_video()
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        frame = detect_cars(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('carros.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/car_count')
def get_car_count():
    return jsonify({'count': car_count})

@app.route('/traffic_count')
def get_traffic_count():
    return jsonify({'traffic': traffic_count})

@app.route('/reset_traffic_count', methods=['POST'])
def reset_traffic_count():
    global traffic_count, tracked_cars
    traffic_count = 0
    tracked_cars = []
    return '', 204

@app.route('/toggle_source', methods=['POST'])
def toggle_source():
    global use_kinect
    use_kinect = not use_kinect
    return jsonify({'use_kinect': use_kinect})

if __name__ == "__main__":
    app.run(debug=True)

