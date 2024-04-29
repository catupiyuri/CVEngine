import numpy as np

def criar_populacao_inicial(tamanho_populacao):
    # Retorna uma lista de dicionários com hiperparâmetros aleatórios
    return [
        {
            'camadas_conv': np.random.randint(1, 4),
            'camadas_dense': np.random.randint(1, 4),
            'neuronios_dense': np.random.randint(64, 256),
            'dropout': np.random.uniform(0.1, 0.5),
            'l2': np.random.uniform(0.01, 0.1),
            'batch_size': np.random.choice([32, 64, 128]),
            'epochs': np.random.randint(5, 20),
            'num_camadas': np.random.randint(1, 3),
            'num_neuronios': np.random.randint(10, 100),
            'taxa_aprendizado': np.random.uniform(0.0001, 0.1),
            'momento': np.random.uniform(0.9, 0.99),
            'funcao_ativacao': np.random.choice(['relu', 'tanh', 'sigmoid']),
            'taxa_dropout': np.random.uniform(0.1, 0.5),
            'taxa_l2': np.random.uniform(0.01, 0.1),
            'taxa_batch_size': np.random.choice([32, 64, 128]),
            'taxa_epochs': np.random.randint(5, 20),
            'taxa_multiplicador': np.random.uniform(0.0001, 0.1),
            'taxa_mutacao': np.random.uniform(0.01, 0.1),
        }
        for _ in range(tamanho_populacao)
    ]

def avaliar_individuo(model ,modelo, X_treino, Y_treino, X_validacao, Y_validacao):
    # Treina a rede com os hiperparâmetros de cada indivíduo e retorna a acurácia no conjunto de validação
    modelo.fit(X_treino, Y_treino, epochs=5, validation_data=(X_validacao, Y_validacao))
    return modelo.evaluate(X_validacao, Y_validacao)

def selecionar_individuo(populacao, aptidoes):
    # Seleciona um indivíduo da população com base em suas aptidões (usando seleção de roleta)
    soma_aptidoes = sum(aptidoes)
    roleta = np.random.uniform(0, soma_aptidoes)
    for i, aptidao in enumerate(aptidoes):
        if roleta < aptidao:
            return populacao[i]
        roleta -= aptidao

def crossover(individuo1, individuo2):
    # Combina os hiperparâmetros de dois indivíduos para criar um novo indivíduo
    novo_individuo = individuo1.copy()
    for param in individuo2:
        if np.random.rand() < 0.5:
            novo_individuo[param] = individuo2[param]
    return novo_individuo

def mutacao(individuo):
    # Altera aleatoriamente alguns dos hiperparâmetros de um indivíduo
    if np.random.rand() < 0.1:
        individuo['num_camadas'] = np.random.randint(1, 5)
    if np.random.rand() < 0.1:
        individuo['num_neuronios'] = np.random.randint(10, 256)
    if np.random.rand() < 0.1:
        individuo['taxa_aprendizado'] = np.random.uniform(0.001, 0.1)
    return individuo

def evoluir_populacao(populacao, modelo, X_treino, Y_treino, X_validacao, Y_validacao):
    aptidoes = [avaliar_individuo(individuo, modelo, X_treino, Y_treino, X_validacao, Y_validacao) for individuo in populacao]
    nova_populacao = [selecionar_individuo(populacao, aptidoes) for _ in range(len(populacao))]
    nova_populacao = [crossover(nova_populacao[i], nova_populacao[(i+1)%len(nova_populacao)]) for i in range(len(nova_populacao))]
    nova_populacao = [mutacao(individuo) for individuo in nova_populacao]
    return nova_populacao