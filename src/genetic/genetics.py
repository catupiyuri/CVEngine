import numpy as np
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pprint import pprint
from data.dados import X_treino, Y_treino, X_validacao, Y_validacao
logging.basicConfig(level=logging.INFO)

class IndividuoGenetico:
    def __init__(self, label):
        self.label = label
        self.parametros_geneticos = {}
        self.cache_desempenho = {}        
        self.inicializar_parametros_geneticos()

    def inicializar_parametros_geneticos(self):
        self.parametros_geneticos = {
            'camadas_conv': np.random.randint(MIN_CAMADAS_CONV, MAX_CAMADAS_CONV + 1),
            'camadas_dense': np.random.randint(MIN_CAMADAS_DENSE, MAX_CAMADAS_DENSE + 1),
            'neuronios_dense': np.random.randint(MIN_NEURONIOS_DENSE, MAX_NEURONIOS_DENSE + 1),
            'dropout': round(np.random.uniform(MIN_DROPOUT, MAX_DROPOUT), 2),
            'l2': round(np.random.uniform(MIN_L2, MAX_L2), 2),
            'batch_size': OPCOES_BATCH_SIZE,
            'epochs': np.random.randint(MIN_EPOCHS, MAX_EPOCHS + 1),
            'num_camadas': np.random.randint(MIN_NUM_CAMADAS, MAX_NUM_CAMADAS + 1),
            'num_neuronios': np.random.randint(MIN_NUM_NEURONIOS, MAX_NUM_NEURONIOS + 1),
            'total_neuronios': np.random.randint(MIN_TOTAL_NEURONIOS, MAX_TOTAL_NEURONIOS + 1),
            'taxa_aprendizado': round(np.random.uniform(MIN_TAXA_APRENDIZADO, MAX_TAXA_APRENDIZADO), 4),
            'momento': round(np.random.uniform(MIN_MOMENTO, MAX_MOMENTO), 2),
            'funcao_ativacao': np.random.choice(OPCOES_FUNCAO_ATIVACAO),
            'taxa_mutacao': round(np.random.uniform(MIN_TAXA_MUTACAO, MAX_TAXA_MUTACAO), 2),
            'num_camadas_conv': np.random.randint(MIN_NUM_CAMADAS_CONV, MAX_NUM_CAMADAS_CONV + 1),
            'num_camadas_dense': np.random.randint(MIN_NUM_CAMADAS_DENSE, MAX_NUM_CAMADAS_DENSE + 1),
            'num_neuronios_dense': np.random.randint(MIN_NUM_NEURONIOS_DENSE, MAX_NUM_NEURONIOS_DENSE + 1),
            'valor_dropout': round(np.random.uniform(MIN_VALOR_DROPOUT, MAX_VALOR_DROPOUT), 2),
            'valor_l2': round(np.random.uniform(MIN_VALOR_L2, MAX_VALOR_L2), 2),
            'tamanho_batch': np.random.randint(MIN_TAMANHO_BATCH, MAX_TAMANHO_BATCH + 1),
            'num_epochs': np.random.randint(MIN_NUM_EPOCHS, MAX_NUM_EPOCHS + 1),
            'total_camadas': np.random.randint(MIN_TOTAL_CAMADAS, MAX_TOTAL_CAMADAS + 1),
            'valor_momento': round(np.random.uniform(MIN_VALOR_MOMENTO, MAX_VALOR_MOMENTO), 2)
        }
        logging.info("Parâmetros genéticos inicializados.")

    def avaliar_desempenho(self, modelo_treinamento, X_treino, Y_treino, X_validacao, Y_validacao):
        chave_cache = str(self.parametros_geneticos)
        if chave_cache in self.cache_desempenho:
            return self.cache_desempenho[chave_cache]
        try:
            # Verificar se os parâmetros genéticos estão dentro dos limites esperados
            # Aqui, adicionamos verificações para cada parâmetro genético
            modelo_treinamento.fit(X_treino, Y_treino, epochs=self.parametros_geneticos['num_epochs'], batch_size=self.parametros_geneticos['tamanho_batch'],validation_data=(X_validacao, Y_validacao))
            desempenho = modelo_treinamento.evaluate(X_validacao, Y_validacao)
            self.cache_desempenho[chave_cache] = desempenho
            logging.info("Desempenho do indivíduo genético avaliado com sucesso.")
            return desempenho
        except ValueError as e:
            logging.error(f"Erro ao avaliar o desempenho do indivíduo genético: {e}")
            return None
        except RuntimeError as e:
            logging.error(f"Erro ao avaliar o desempenho do indivíduo genético: {e}. Verifique se o modelo de treinamento está configurado corretamente.")
            return None

    def aplicar_mutacao(self, chave_parametro, valor_minimo=None, valor_maximo=None, opcoes=None):
        if np.random.rand() < self.parametros_geneticos['taxa_mutacao']:
            if opcoes is not None:
                self.parametros_geneticos[chave_parametro] = np.random.choice(opcoes)
            elif valor_minimo is not None and valor_maximo is not None:
                novo_valor = np.random.uniform(valor_minimo, valor_maximo)
                # Se o valor original era um inteiro, mantenha o novo valor como inteiro
                if isinstance(self.parametros_geneticos[chave_parametro], int):
                    self.parametros_geneticos[chave_parametro] = int(novo_valor)
                else:
                    # Caso contrário, arredonde para duas casas decimais
                    self.parametros_geneticos[chave_parametro] = round(novo_valor, 2)
            else:
                logging.warning(f"Não foi possível aplicar mutação ao parâmetro {chave_parametro} porque nenhum valor mínimo, máximo ou opções foram fornecidos.")
        return self

    def realizar_crossover(self, outro_individuo_genetico):
        for parametro in outro_individuo_genetico.parametros_geneticos:
            if np.random.rand() < 0.5:
                self.parametros_geneticos[parametro] = outro_individuo_genetico.parametros_geneticos[parametro]
        return self

class PopulacaoGenetica:
    def __init__(self, tamanho_populacao, label):
        self.label = label
        self.individuos_geneticos = [IndividuoGenetico(f"Individuo{i+1}") for i in range(tamanho_populacao)]

    def selecionar_individuo_genetico(self, aptidoes):
        soma_aptidoes = sum(aptidoes)
        roleta_selecao = np.random.uniform(0, soma_aptidoes)
        for i, aptidao in enumerate(aptidoes):
            if roleta_selecao < aptidao:
                return self.individuos_geneticos[i]
            roleta_selecao -= aptidao

    def evoluir_populacao(self, modelo_treinamento, X_treino, Y_treino, X_validacao, Y_validacao):
        try:
            aptidoes = [individuo_genetico.avaliar_desempenho(modelo_treinamento, X_treino, Y_treino, X_validacao, Y_validacao) for individuo_genetico in self.individuos_geneticos]
            tamanho_populacao = len(self.individuos_geneticos)
            nova_populacao = [self.selecionar_individuo_genetico(aptidoes) for _ in range(tamanho_populacao)]
            for i in range(tamanho_populacao):
                individuo_genetico = nova_populacao[i]
                individuo_genetico.realizar_crossover(nova_populacao[(i+1)%tamanho_populacao])
                for chave_parametro in individuo_genetico.parametros_geneticos.keys():
                    # Aqui você precisa fornecer os argumentos necessários para o método aplicar_mutacao
                    valor_minimo = getattr(config, f"MIN_{chave_parametro.upper()}", None)
                    valor_maximo = getattr(config, f"MAX_{chave_parametro.upper()}", None)
                    opcoes = getattr(config, f"OPCOES_{chave_parametro.upper()}", None)
                    individuo_genetico.aplicar_mutacao(chave_parametro, valor_minimo, valor_maximo, opcoes)
            logging.info("População genética evoluída com sucesso.")
            return nova_populacao
        except ValueError as e:
            logging.error(f"Erro ao evoluir a população genética: {e}. Verifique se os valores de aptidão estão corretos.")
            return None