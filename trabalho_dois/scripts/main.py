#==============================================================================
# Trabalho 2 - Estimar o preço de um imóvel a partir de suas características
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
