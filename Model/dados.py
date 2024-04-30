import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from rich.progress import track
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from joblib import dump, load

console = Console()
# Crie um Progress personalizado
progress = Progress(console=console, auto_refresh=False)

console.print(Panel.fit(Text("Iniciando o programa de normalização de dados...", style="bold blue on white"), title="Status", border_style="blue"), justify="center")

import time
import importlib

modules = [
    'numpy',
    'string',
    'collections',
    'skimage.transform',
    'skimage.color',
    'PIL.Image',
    'PIL.ImageFile',
    'sklearn.datasets',
    'sklearn.utils',
    'keras.datasets',
    'colorama',
]

def load_module(module, progress, task):
    try:
        importlib.import_module(module)
        progress.update(task, advance=1)
        progress.refresh()  # Simula o tempo de carregamento
    except ImportError:
        console.print(f"Erro ao carregar o módulo {module}", style="red")

with progress:
    task1 = progress.add_task("[cyan]Carregando módulos...", total=len(modules))
    for module in modules:
        load_module(module, progress, task1)

import numpy as np
import string
from collections import Counter
from skimage.transform import resize
from skimage.color import rgb2gray
from PIL import ImageFile
from sklearn.datasets import fetch_lfw_people
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from colorama import Fore, Style

ImageFile.LOAD_TRUNCATED_IMAGES = True

def print_separator():
    console.rule(style="blue")

print_separator()

def load_lfw_data(min_faces_per_person=1, resize=0.4):
    console.print("Carregamento dos dados LFW...", style="red", end="")

    lfw_dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    num_faces = len(lfw_dataset.images)

    for _ in track(range(2020), description="[cyan]Carregando dados LFW..."):
        time.sleep(0.00001)
    console.print("\rCarregamento dos dados LFW concluído!", style="yellow")


    face_counts = Counter(lfw_dataset.target)
    single_face_ids = [person_id for person_id, count in face_counts.items() if count == 1]
    single_face_indices = [i for i, target in enumerate(lfw_dataset.target) if target in single_face_ids]
    image_labels = ["rh." + string.ascii_lowercase[i % 26] for i in range(len(single_face_indices))]
    print("Rótulos antes do embaralhamento: " + ', '.join(image_labels[:5]))  # Imprime os primeiros 5 rótulos antes do embaralhamento
    single_face_indices, image_labels = shuffle(single_face_indices, image_labels)  # Embaralha os índices e os rótulos
    print("Rótulos após o embaralhamento: " + ', '.join(image_labels[:5]))  # Imprime os primeiros 5 rótulos após o embaralhamento
    X = lfw_dataset.images[single_face_indices]
    Y = np.ones(len(single_face_indices))
    num_faces = len(X)
    console.print(f"[green]Carregadas {num_faces} imagens LFW[/green]")
    print_separator()
    return X, Y, image_labels

def load_non_face_data(num_images=1010):
    console.print("Carregamento dos dados non-face...", style="red", end="")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    non_face_images = np.concatenate((X_train, X_test))
    non_face_images = non_face_images[:num_images]
    num_images = len(non_face_images)
    for _ in track(range(num_images), description="[cyan]Carregando dados non-face..."):
        time.sleep(0.001)
    console.print("\rCarregamento dos dados non-face concluído!", style="yellow")
    image_labels = ["ia." + string.ascii_lowercase[i % 26] for i in range(len(non_face_images))]
    print("Rótulos antes do embaralhamento: " + ', '.join(image_labels[:5]))  # Imprime os primeiros 5 rótulos antes do embaralhamento
    non_face_images, image_labels = shuffle(non_face_images, image_labels)  # Embaralha as imagens e os rótulos
    print("Rótulos após o embaralhamento: " + ', '.join(image_labels[:5]))  # Imprime os primeiros 5 rótulos após o embaralhamento
    batch_size = 2020
    non_face_images_gray = np.empty((0, 32, 32))
    for i in range(0, len(non_face_images), batch_size):
        batch = non_face_images[i:i+batch_size]
        batch_gray = rgb2gray(batch)
        non_face_images_gray = np.append(non_face_images_gray, batch_gray, axis=0)
    Y_non_face = np.zeros(len(non_face_images_gray))
    num_non_faces = len(non_face_images_gray)
    console.print(f"[green]Carregadas {num_non_faces} imagens non-face[/green]")
    print_separator()
    return non_face_images_gray, Y_non_face, image_labels

def print_info(label, value):
    if isinstance(value, tuple):
        value = value + (0,) * (3 - len(value))
        value = ' '.join(f'{i:>4}' for i in value)
    elif isinstance(value, bool):
        value = 'Sim' if value else 'Não'
    elif isinstance(value, np.bool_):
        value = 'Sim' if value else 'Não'
    else:
        value = f'{value:>4}'
    print("{:<30}: {}".format(label, value))   

cache_file = 'Image_data_cache.joblib'

def check_data(cache_data):
    return len(cache_data) == 6 and all(isinstance(i, np.ndarray) for i in cache_data)

def generate_data():
    console.print("Gerando dados...", style="yellow")
    X_face, Y_face, image_labels = load_lfw_data()
    X_non_face, Y_non_face, image_labels_non_face = load_non_face_data()

    console.print("Normalizando dados LFW para Non-face", style="green")

    for _ in track(range(len(X_face)), description="[cyan]Redimensionando imagens..."):
            time.sleep(0.00001)

    # Resize LFW images to have the same shape as CIFAR-10 images
    X_face_resized = [resize(image, (X_non_face.shape[1], X_non_face.shape[2])) for image in X_face]
    X_face_resized = np.array(X_face_resized)

    # Concatenação dos dados
    X = np.concatenate((X_face_resized, X_non_face))
    Y = np.concatenate((Y_face, Y_non_face))

    return X, Y, X_face_resized, X_non_face, Y_face, Y_non_face

if os.path.exists(cache_file):
    console.print("Carregando dados do cache...", style="yellow")
    cache_data = load(cache_file)
    if not check_data(cache_data):
        console.print("Dados de cache inválidos, gerando novos dados...", style = "red")
        cache_data = generate_data()
        dump(cache_data, cache_file)
    X, Y, X_face_resized, X_non_face, Y_face, Y_non_face = cache_data
else:
    cache_data = generate_data()
    dump(cache_data, cache_file)
    X, Y, X_face_resized, X_non_face, Y_face, Y_non_face = cache_data

print_info("Dimensões LFW Redimensionado", X_face_resized.shape)
print_info("Dimensões Non-face", X_non_face.shape)
print_info("Dimensões de LFW + Non-face", X.shape)
print_info("LFW classificado?", np.all(Y_face == 1))
print_info("Non-face classificado?", np.all(Y_non_face == 0))
print_info("LFW + Non-face classificado?", np.array_equal(Y, np.concatenate((Y_face, Y_non_face))))

# Divisão dos dados
X_treino, X_validacao, Y_treino, Y_validacao = train_test_split(X, Y, test_size=0.2, random_state=42)

console.print("Dados LFW normalizados!", style="yellow")
print()