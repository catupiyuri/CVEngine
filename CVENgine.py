import os

def limpar():
    if os.name == "nt":  
        os.system("cls")  
    else:
        os.system("clear")  

def centralizar(texto, largura):
    return texto.center(largura)

def bem_vindo():
    limpar()
    largura = os.get_terminal_size().columns
    print(centralizar("Seja bem-vindo ao CVEngine!", largura))
    print(centralizar("Versão ALPHA", largura))
    input(centralizar("Pressione Enter para continuar...", largura))

def menu():
    limpar()
    largura = os.get_terminal_size().columns
    print(centralizar("Menu de Opções:", largura))
    print(centralizar("1. Rostos", largura))
    print(centralizar("2. Motos", largura))
    print(centralizar("3. Carros", largura))
    print(centralizar("4. Bicicletas", largura))
    print(centralizar("5. Pedestres", largura))
    print(centralizar("6. Sair", largura))

def abrir_arquivo(nome_arquivo):
    caminho_arquivo = os.path.join("modulos", nome_arquivo + ".py")
    if os.path.exists(caminho_arquivo):
        os.system(f"python {caminho_arquivo}")
    else:
        print(f"O arquivo {nome_arquivo}.py não foi encontrado na pasta modulos.")

def processar_opcao(opcao):
    if opcao == '1':
        abrir_arquivo("rostos")
    elif opcao == '2':
        print("Opa! Essa opção ainda não está disponível!")
    elif opcao == '3':
        abrir_arquivo("carros")
    elif opcao == '4':
        print("Opa! Essa opção ainda não está disponível!")
    elif opcao == '5':
        print("Opa! Essa opção ainda não está disponível!")
    elif opcao == '6':
        print("Saindo...")
        return False
    else:
        print("Opção inválida. Escolha novamente.")
    input("Pressione Enter para continuar...")
    return True

def main():
    bem_vindo()
    while True:
        menu()
        escolha = input("\nEscolha uma opção: ")
        continuar = processar_opcao(escolha)
        if not continuar:
            break

if __name__ == "__main__":
    main()

