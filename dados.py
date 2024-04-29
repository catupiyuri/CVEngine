import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.datasets import cifar10 
from sklearn.datasets import fetch_lfw_people 
from skimage.transform import resize 
import numpy as np 
from skimage.color import rgb2gray 
from PIL import Image

def load_lfw_data(min_faces_per_person=70, resize=0.4):
    lfw_dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    X = lfw_dataset.images
    Y = np.ones(lfw_dataset.target.shape)
    return X, Y

def load_non_face_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    non_face_images = np.concatenate((X_train, X_test))
    batch_size = 1000 # Ajuste este valor de acordo com a quantidade de memória disponível
    non_face_images_gray = np.empty((0, 32, 32))
    for i in range(0, len(non_face_images), batch_size):
        batch = non_face_images[i:i+batch_size]
        batch_gray = rgb2gray(batch)
        non_face_images_gray = np.append(non_face_images_gray, batch_gray, axis=0)
    Y_non_face = np.zeros(len(non_face_images_gray))
    return non_face_images_gray, Y_non_face

X_face, Y_face = load_lfw_data()
X_non_face, Y_non_face = load_non_face_data()

# Redimensiona as imagens do LFW para terem a mesma forma que as imagens do CIFAR-10
X_face_resized = [resize(image, (X_non_face.shape[1], X_non_face.shape[2])) for image in X_face]
X_face_resized = np.array(X_face_resized)

X = np.concatenate((X_face_resized, X_non_face))
Y = np.concatenate((Y_face, Y_non_face))

print("X_face_resized shape:", X_face_resized.shape)
print("X_non_face shape:", X_non_face.shape)
print("X shape:", X.shape)
print("Y_face shape:", Y_face.shape)
print("Y_non_face shape:", Y_non_face.shape)
print("Y shape:", Y.shape)

# Carrega os dados
X_face, Y_face = load_lfw_data()