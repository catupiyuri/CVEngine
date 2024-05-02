import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unittest
import numpy as np
from src.genetic.genetics import IndividuoGenetico, PopulacaoGenetica
from keras.models import Sequential
import pprint

# Defina o tamanho da população e o rótulo
tamanho_populacao = 100  # ou qualquer valor que você deseja
label = "PopulacaoTeste"  # ou qualquer string que você deseja

# Crie uma instância da classe PopulacaoGenetica
populacao = PopulacaoGenetica(tamanho_populacao, label)

class TestIndividuoGenetico(unittest.TestCase):
    def setUp(self):
        self.individuo = IndividuoGenetico('test_label')

    def test_inicializar_parametros_geneticos(self):
        self.individuo.inicializar_parametros_geneticos()
        parametros = self.individuo.parametros_geneticos

        self.assertIsInstance(parametros['camadas_conv'], int)
        self.assertGreaterEqual(parametros['camadas_conv'], MIN_CAMADAS_CONV)
        self.assertLessEqual(parametros['camadas_conv'], MAX_CAMADAS_CONV + 1)

        # Add more assertions for other parameters

    def test_avaliar_desempenho(self):
        # Create a mock model and data for testing
        model = MockModel()
        X_train = np.random.rand(100, 10)
        Y_train = np.random.randint(2, size=(100,))
        X_valid = np.random.rand(50, 10)
        Y_valid = np.random.randint(2, size=(50,))

        desempenho = self.individuo.avaliar_desempenho(model, X_train, Y_train, X_valid, Y_valid)

        self.assertIsNotNone(desempenho)
        self.assertIsInstance(desempenho, float)

    def test_aplicar_mutacao(self):
        chave_parametro = 'camadas_conv'
        valor_minimo = 1
        valor_maximo = 10

        self.individuo.aplicar_mutacao(chave_parametro, valor_minimo, valor_maximo)

        self.assertGreaterEqual(self.individuo.parametros_geneticos[chave_parametro], valor_minimo)
        self.assertLessEqual(self.individuo.parametros_geneticos[chave_parametro], valor_maximo)

        # Add more assertions for other parameters

    def test_realizar_crossover(self):
        outro_individuo = IndividuoGenetico('other_label')

        self.individuo.realizar_crossover(outro_individuo)

        # Add assertions to check if parameters have been crossed over correctly

    def test_inicializacao_parametros_geneticos(self):
        individuo = IndividuoGenetico("IndividuoTeste")
        for chave, valor in individuo.parametros_geneticos.items():
            valor_minimo = getattr(config, f"MIN_{chave.upper()}", None)
            valor_maximo = getattr(config, f"MAX_{chave.upper()}", None)
            if valor_minimo is not None and valor_maximo is not None:
                assert valor_minimo <= valor <= valor_maximo, f"O valor do parâmetro {chave} está fora dos limites esperados."
        print("Parâmetros genéticos após a inicialização:")
        pprint(individuo.parametros_geneticos)

    def test_aplicar_mutacao(self):
        individuo = IndividuoGenetico("IndividuoTeste")
        individuo.parametros_geneticos['taxa_mutacao'] = 1  # Garantir que a mutação ocorra
        parametros_originais = individuo.parametros_geneticos.copy()
        for chave in parametros_originais.keys():
            valor_minimo = getattr(config, f"MIN_{chave.upper()}", None)
            valor_maximo = getattr(config, f"MAX_{chave.upper()}", None)
            opcoes = getattr(config, f"OPCOES_{chave.upper()}", None)
            individuo.aplicar_mutacao(chave, valor_minimo, valor_maximo, opcoes)
        assert individuo.parametros_geneticos != parametros_originais, "A mutação não alterou os parâmetros genéticos."
        print("Parâmetros genéticos após a mutação:")
        pprint(individuo.parametros_geneticos)

    def test_realizar_crossover(self):
        individuo1 = IndividuoGenetico("IndividuoTeste1")
        individuo2 = IndividuoGenetico("IndividuoTeste2")
        parametros_originais = individuo1.parametros_geneticos.copy()
        individuo1.realizar_crossover(individuo2)
        assert individuo1.parametros_geneticos != parametros_originais, "O crossover não alterou os parâmetros genéticos."
        print("Parâmetros genéticos após o crossover:")
        pprint(individuo1.parametros_geneticos)

    def test_avaliar_desempenho(self):
        # Criando um modelo de treinamento fictício
        modelo_treinamento = Sequential()
        modelo_treinamento.add(Dense(1, input_dim=1, activation='linear'))
        modelo_treinamento.compile(loss='mean_squared_error', optimizer=Adam())

        # Criando um conjunto de dados fictício
        X_treino = np.array([1, 2, 3, 4, 5])
        Y_treino = np.array([2, 4, 6, 8, 10])
        X_validacao = np.array([6, 7, 8, 9, 10])
        Y_validacao = np.array([12, 14, 16, 18, 20])

        # Criando um indivíduo genético
        individuo = IndividuoGenetico("IndividuoTeste")
        individuo.parametros_geneticos['num_epochs'] = 1
        individuo.parametros_geneticos['tamanho_batch'] = 1

        # Testando a função avaliar_desempenho
        desempenho = individuo.avaliar_desempenho(modelo_treinamento, X_treino, Y_treino, X_validacao, Y_validacao)
        assert desempenho is not None, "A função avaliar_desempenho não retornou um valor."
        print(f"Desempenho retornado pela função avaliar_desempenho: {desempenho}")

class TestGenetics(unittest.TestCase):
    def setUp(self):
        # Crie uma população genética e um modelo de treinamento fictício aqui
        self.populacao = [...]
        self.modelo_treinamento = [...]

    def test_evoluir_populacao(self):
        nova_populacao = evoluir_populacao(self.populacao, self.modelo_treinamento)

        # Verifique se a nova população é uma lista (ou qualquer outra estrutura de dados que você esteja usando)
        self.assertIsInstance(nova_populacao, list)

        # Verifique se a nova população tem o mesmo tamanho que a população original
        self.assertEqual(len(nova_populacao), len(self.populacao))

        # Adicione mais verificações aqui conforme necessário

if __name__ == '__main__':
    unittest.main()