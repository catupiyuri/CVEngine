from src.data.dados import X, Y
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras import regularizers
from logs import TrainingLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from src.genetic.genetics import criar_populacao_inicial, avaliar_individuo, selecionar_individuo, crossover, mutacao, evoluir_populacao

from sklearn.model_selection import train_test_split
X = tf.cast(X, tf.float32)
import numpy as np
import math
import cv2

def processar_imagem(modelo, imagem):
    # Redimensiona a imagem para o tamanho esperado pelo modelo
    imagem_resized = cv2.resize(imagem, (32, 32))

    # Normaliza a imagem
    imagem_norm = imagem_resized / 255.0

    # Adiciona uma dimensão extra para corresponder à entrada esperada pelo modelo
    imagem_input = np.expand_dims(imagem_norm, axis=0)

    # Faz a previsão
    predicao = modelo.predict(imagem_input)

    # Se a previsão for maior que 0.5, consideramos que a IA reconheceu um rosto
    if predicao > 0.5:
        # Desenha um retângulo ao redor do rosto
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe a imagem
    cv2.imshow('Imagem', imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Iniciando o modelo de treinamento")

X_np = X.numpy()

train_data, validation_data, train_labels, validation_labels = train_test_split(X_np, Y, test_size=0.2)

def criar_datagen():
    # Cria um gerador de imagens
    datagen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        zoom_range = 0.1
    )
    return datagen


def criar_modelo(individuo):
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (2, 2), padding='same',activation='relu'))  # Reduza o tamanho do filtro
    model.add(MaxPooling2D((2, 2)))  # Adicione uma segunda camada de MaxPooling
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))  # Adicione uma terceira camada convolucional 
    model.add(MaxPooling2D((2, 2)))  # Adicione uma terceira camada de MaxPooling
    model.add(BatchNormalization())
    
    
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling2D())  # Adicione uma camada GlobalAveragePooling2D
    model.add(Dense(128 , activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=individuo['taxa_aprendizado']), 
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    
    return model

def treinar_modelo(model, train_data, train_labels, epochs):
    print("Iniciando o treinamento do modelo")
    logger = TrainingLogger()
    datagen = criar_datagen()
    train_data = train_data.reshape(-1, 32, 32, 1)
    datagen.fit(train_data)
    model.fit(datagen.flow(train_data, train_labels, batch_size=16), steps_per_epoch=math.ceil(len(train_data) / 16), epochs=epochs, callbacks=[logger])

populacao = criar_populacao_inicial(10)
print("População inicial criada")

for i in range(20):
    aptidoes = []
    print(f"Gerando {len(populacao)} indivíduos para a geração {i+1}")
    for individuo in populacao:
        print("Criando modelo para o indivíduo")
        model = criar_modelo(individuo)
        print(f"Treinando o modelo para {individuo['epochs']} épocas")
        treinar_modelo(model, train_data, train_labels, individuo['epochs'])
        aptidao = avaliar_individuo(model, X_treino, Y_treino, validation_data, validation_labels)
        print(f"Aptidão do indivíduo: {aptidao}")
        aptidoes.append(aptidao)
    nova_populacao = []
    for i in range(len(populacao)):
        individuo = selecionar_individuo(populacao, aptidoes)
        nova_populacao.append(individuo)
    populacao = evoluir_populacao(nova_populacao)
melhor_individuo = max(populacao, key=avaliar_individuo)

def avaliar_modelo(model, test_data, test_labels):
    loss, acc = model.evaluate(test_data, test_labels, verbose=2)
    print('\nTest accuracy:', acc)