#==============================================================================
# Trabalho 2 - Estimar o preço de um imóvel a partir de suas características
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

import math
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

#------------------------------------------------------------------------------
# Importar os conjuntos de teste e treinamento (retirando as colunas dos id's)
#------------------------------------------------------------------------------

caminho_conjunto_de_teste = Path('../data') / 'conjunto_de_teste.csv'
caminho_conjunto_de_treinamento = Path('../data') / 'conjunto_de_treinamento.csv'
dados_treinamento = pd.read_csv(caminho_conjunto_de_treinamento)
dados_teste = pd.read_csv(caminho_conjunto_de_teste)
ids_dados_teste = dados_teste['Id']
dados_teste = dados_teste.iloc[:, 1:]
dados_treinamento = dados_treinamento.iloc[:, 1:]

# ------------------------------------------------------------------------------
#  Exibição das primeiras 10 linhas do conjunto de treinamento através da função head()
# ------------------------------------------------------------------------------

print("\n\t\t-----Dez primeiras linhas do conjunto de treinamento-----\n")
print(dados_treinamento.head(n=10))

# ------------------------------------------------------------------------------
#  Descrição dos dados de treinamento (como número de linhas, tipo de cada
#  atributo e número de valores não nulos) através da função info()
# ------------------------------------------------------------------------------

print("\n\n\t-----Descrição dos dados do conjunto de treinamento-----\n")
dados_treinamento.info()

# ------------------------------------------------------------------------------
#  Descobrindo quais categorias existem nas features e no alvo, além de quantos
#  dígitos pertencem a cada categoria, usando a função value_counts()
# ------------------------------------------------------------------------------

print("\n\n\t-----Categorias das features e do alvo, com suas respectivas quantidades-----\n")
for feature in list(dados_treinamento.columns):
    print("\n", dados_treinamento[feature].value_counts())

# ------------------------------------------------------------------------------
#  Resumo dos atributos numéricos do conjunto de treinamento através da
#  função describe()
# ------------------------------------------------------------------------------

print("\n\n\t-----Resumo dos atributos numéricos-----\n")
print(dados_treinamento.describe())

# ------------------------------------------------------------------------------
# Exibindo as features do dataset e seus tipos
# ------------------------------------------------------------------------------

print("\n\n\t-----Features disponíveis-----\n")
print(list(dados_treinamento.columns))

print("\n\n-----Tipos das features-----\n")
print(dados_treinamento.dtypes)

# ------------------------------------------------------------------------------
# Exibindo o histograma entre as quantidades e os valores do alvo
# ------------------------------------------------------------------------------

print(f"\n\n\t-----Histograma do alvo-----\n")
grafico = dados_treinamento['preco'].plot.hist(bins=100)
grafico.set(title='preco', xlabel='Quantidades', ylabel='Valores')
plt.show()

# ------------------------------------------------------------------------------
#  Melhor exibição das classes das features, pois os describe() não exibiu todas
# ------------------------------------------------------------------------------

print("\n\n\t-----Melhor exibição das classes das features-----\n")
for feature in list(dados_treinamento.columns):
    print(f"\nClasses {feature}: ", list(dados_treinamento[feature].unique()))

# ------------------------------------------------------------------------------
# Criação e implementação de uma função para aplicar a classe OneHotEncoder em
# colunas categóricas, mantendo as demais inalteradas.
# ------------------------------------------------------------------------------

def apply_one_hot_encoder(data, features, data_type='training', target='target'):
    "Função que aplica a classe OneHotEncoder em features categóricas, mantendo as demais inalteradas."

    # Concatenar o DataFrame codificado com as demais features e o alvo.
    if data_type == 'training':

        # Separar a coluna do alvo das demais features.
        data_target = data[target]
        data_features = data.drop(target, axis=1)

        # Instanciar o OneHotEncoder.
        one_hot_encoder = OneHotEncoder(sparse_output=False)

        # Aplicar o OneHotEncoder às colunas categóricas.
        data_codificado = one_hot_encoder.fit_transform(data_features[features])

        # Colhetando os nomes das features codificadas.
        features_codificadas = one_hot_encoder.get_feature_names_out(features)

        # Converter o resultado para DataFrame.
        data_frame_codificado = pd.DataFrame(data_codificado, columns=features_codificadas, index=data.index)

        # Remover as features categóricas originais.
        data_features = data_features.drop(columns=features)

        data_final = pd.concat([data_features, data_frame_codificado, data_target], axis=1)
        return data_final

    elif data_type == 'test':

        # Instanciar o OneHotEncoder.
        one_hot_encoder = OneHotEncoder(sparse_output=False)

        # Aplicar o OneHotEncoder às colunas categóricas.
        data_codificado = one_hot_encoder.fit_transform(data[features])

        # Colhetando os nomes das features codificadas.
        features_codificadas = one_hot_encoder.get_feature_names_out(features)

        # Converter o resultado para DataFrame.
        data_frame_codificado = pd.DataFrame(data_codificado, columns=features_codificadas, index=data.index)

        # Remover as features categóricas originais.
        data = data.drop(columns=features)

        data_final = pd.concat([data, data_frame_codificado], axis=1)
        return data_final

    else:
        raise ValueError(f'\nData type "{data_type}" is not supported\n')

# ------------------------------------------------------------------------------
# Implementação da função nas features categóricas dos conjuntos de dados.
# ------------------------------------------------------------------------------

features_categoricas = ['tipo', 'bairro', 'tipo_vendedor', 'diferenciais']
dados_treinamento = apply_one_hot_encoder(dados_treinamento, features_categoricas, 'training', 'preco')
dados_teste = apply_one_hot_encoder(dados_teste, features_categoricas, 'test')

# ------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
# ------------------------------------------------------------------------------

dados_treinamento_embaralhados = dados_treinamento.sample(frac=1, random_state=11012005)
dados_teste_embaralhados = dados_teste.sample(frac=1, random_state=11012005)

#------------------------------------------------------------------------------
# Separar o conjunto de treinamento em arrays X e Y, exibindo suas dimensões
#------------------------------------------------------------------------------

# Separando as features do alvo.
X = dados_treinamento_embaralhados.iloc[:, :-1].values
y = dados_treinamento_embaralhados.iloc[:, -1].values

# Conjunto de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=11012005)

# Conjunto de teste final
X_teste_final = dados_teste_embaralhados.iloc[:, :].values

# ------------------------------------------------------------------------------
# Aplicação da escala no X de treino e de teste
# ------------------------------------------------------------------------------

# escala = MinMaxScaler()
escala = StandardScaler()

escala.fit(X_treino)
X_treino_com_escala = escala.transform(X_treino)
X_teste_com_escala = escala.transform(X_teste)

# ------------------------------------------------------------------------------
# Treinando o modelo KNeighborsClassifier, com k variando entre 1 e 30
# ------------------------------------------------------------------------------
# Primeiro teste: k = 21 -- RMSE = 1478825.5182 -- R2 = -3.8099

print("\n\n\t-----Regressor com KNN-----\n")

for k in range(1, 31):

    # Instanciando o regressor KNN.
    regressor_knn = KNeighborsRegressor(n_neighbors=k, weights="uniform")
    regressor_knn = regressor_knn.fit(X_treino_com_escala, y_treino)

    # Predições.
    y_resposta_treino = regressor_knn.predict(X_treino_com_escala)
    y_resposta_teste = regressor_knn.predict(X_teste_com_escala)

    # Calculando RMSE e o R2 Score.
    rmse_treino = math.sqrt(mean_squared_error(y_treino, y_resposta_treino))
    rmse_teste = math.sqrt(mean_squared_error(y_teste, y_resposta_teste))
    r2_score_treino = r2_score(y_treino, y_resposta_treino)
    r2_score_teste = r2_score(y_teste, y_resposta_teste)

    print(f'\nK = {k}')
    print(f'RMSE Treino: {rmse_treino:.4f}')
    print(f'R2 Score Treino: {r2_score_treino:.4f}')
    print(f'RMSE Teste: {rmse_teste:.4f}')
    print(f'R2 Score Teste: {r2_score_teste:.4f}')

    # Obtendo a matriz de confusão.
    # matriz_confusao = confusion_matrix(y_treino, y_resposta)

# ------------------------------------------------------------------------------
#  Treinando o modelo Regressão Linear
# ------------------------------------------------------------------------------
# Primeiro teste: RMSE = 27325327877087870976.0000 -- R2 = -1642234633302646893198704640.0000

print("\n\n\t-----Regressor com Regressão Linear-----\n")

# Instanciando o regressor Regressão Linear.
regressor_regressao_linear = LinearRegression()
regressor_regressao_linear = regressor_regressao_linear.fit(X_treino_com_escala, y_treino)

# Predições.
y_resposta_treino = regressor_regressao_linear.predict(X_treino_com_escala)
y_resposta_teste = regressor_regressao_linear.predict(X_teste_com_escala)

# Calculando RMSE e o R2 Score.
rmse_treino = math.sqrt(mean_squared_error(y_treino, y_resposta_treino))
rmse_teste = math.sqrt(mean_squared_error(y_teste, y_resposta_teste))
r2_score_treino = r2_score(y_treino, y_resposta_treino)
r2_score_teste = r2_score(y_teste, y_resposta_teste)

print(f'RMSE Treino: {rmse_treino:.4f}')
print(f'R2 Score Treino: {r2_score_treino:.4f}')
print(f'RMSE Teste: {rmse_teste:.4f}')
print(f'R2 Score Teste: {r2_score_teste:.4f}')

# ------------------------------------------------------------------------------
# Treinando o modelo Regressão Polinomial
# ------------------------------------------------------------------------------
# Primeiro teste: Erro de não possuir espaço suficiente para alocar memória.

print("\n\n\t-----Regressor com Regressão Polinomial-----\n")

for grau in range(1, 11):

    # Instanciando o metodo PolynomialFeatures.
    polynomial_features = PolynomialFeatures(degree=grau)
    polynomial_features = polynomial_features.fit(X_treino)
    X_treino_poly = polynomial_features.transform(X_treino_com_escala)
    X_teste_poly = polynomial_features.transform(X_teste_com_escala)

    # Instanciando o regressor Regressão Linear.
    regressor_regressao_linear = LinearRegression()
    regressor_regressao_linear = regressor_regressao_linear.fit(X_treino_poly, y_treino)

    # Predições.
    y_resposta_treino = regressor_regressao_linear.predict(X_treino_poly)
    y_resposta_teste = regressor_regressao_linear.predict(X_teste_poly)

    # Calculando RMSE e o R2 Score.
    rmse_treino = math.sqrt(mean_squared_error(y_treino, y_resposta_treino))
    rmse_teste = math.sqrt(mean_squared_error(y_teste, y_resposta_teste))
    r2_score_treino = r2_score(y_treino, y_resposta_treino)
    r2_score_teste = r2_score(y_teste, y_resposta_teste)

    print(f'\nGrau = {grau}')
    print(f'RMSE Treino: {rmse_treino:.4f}')
    print(f'R2 Score Treino: {r2_score_treino:.4f}')
    print(f'RMSE Teste: {rmse_teste:.4f}')
    print(f'R2 Score Teste: {r2_score_teste:.4f}')