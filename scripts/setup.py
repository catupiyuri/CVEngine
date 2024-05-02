import os
import subprocess
def verificar_e_criar_diretorio(diretorio):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
        print(f"Diretório {diretorio} criado com sucesso.")

def verificar_e_criar_arquivo(arquivo):
    if not os.path.exists(arquivo):
        print(f"Criando arquivo {arquivo}")
        open(arquivo, 'w').close()

def configurar_github(repositorio):
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"])
    subprocess.run(["git", "remote", "add", "origin", repositorio])
    subprocess.run(["git", "pull", "origin", "Tensorflow"])
    result = subprocess.run(["git", "push", "-u", "origin", "master"], capture_output=True)
    print(result.stdout.decode())
    if result.returncode != 0:
        print("Não foi possível puxar do branch 'Tensor flow'. Verifique se o branch existe no repositorio remoto.")

def main():
    diretorios = ['src/data', 'src/model', 'src/genetic', 'src/logs']
    arquivos = ['src/data/config.py', 'src/data/dados.py', 'src/data/__init__.py', 'src/genetic/genetics.py', 'src/genetic/__init__.py', 'src/model/modelo.py', 'src/model/requirements.txt', 'src/model/__init__.py', 'tests/test_genetics.py', 'cache/Image_data_cache.joblib']

    for diretorio in diretorios:
        verificar_e_criar_diretorio(diretorio)

    for arquivo in arquivos:
        verificar_e_criar_arquivo(arquivo)

    configurar_github("https://github.com/Desuoka/CVEngine")

if __name__ == "__main__":
    main()