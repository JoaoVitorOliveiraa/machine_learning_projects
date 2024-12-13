#==============================================================================
# Trabalho 1 - Sistema de apoio à decisão para aprovação de crédito
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

#------------------------------------------------------------------------------
# Importar os conjuntos de dados de teste e treinamento (retirando as colunas dos id's)
#------------------------------------------------------------------------------

caminho_conjunto_de_teste = Path('../data') / 'conjunto_de_teste.csv'
caminho_conjunto_de_treinamento = Path('../data') / 'conjunto_de_treinamento.csv'
dados_teste = pd.read_csv(caminho_conjunto_de_teste)
dados_treinamento = pd.read_csv(caminho_conjunto_de_treinamento)
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
#  Descobrindo quais categorias existem e quantos dígitos pertencem a cada
#  categoria usando a função value_counts()
# ------------------------------------------------------------------------------

print("\n\n\t-----Categorias do alvo, com suas respectivas quantidades-----\n")
print(dados_treinamento["inadimplente"].value_counts())

# ------------------------------------------------------------------------------
#  Resumo dos atributos numéricos do conjunto de treinamento através da
#  função describe()
# ------------------------------------------------------------------------------

print("\n\n\t-----Resumo dos atributos numéricos-----\n")
print(dados_treinamento.describe())

#------------------------------------------------------------------------------
# Separar o conjunto de treinamento em atributos e alvo, exibindo suas dimensões
#------------------------------------------------------------------------------

# atributos = dados_treinamento.iloc[:, :-1].to_numpy()   # ou :-1].values
# rotulos = dados_treinamento.iloc[:, -1].to_numpy()      # ou :-1].values
# print("\n\n\t-----Dimensões-----")
# print(f"\nDimensão das features: {atributos.shape}")
# print(f"Dimensão dos rótulos: {rotulos.shape}\n")

#------------------------------------------------------------------------------
# Exibir as colunas do dataset de treinamento
#------------------------------------------------------------------------------

colunas = dados_treinamento.columns
print("\n\n\t-----Colunas disponíveis-----\n")
print(colunas)

#------------------------------------------------------------------------------
# Exibir o coeficiente de Pearson de cada atributo (entre o mesmo e o alvo)
#------------------------------------------------------------------------------

# for coluna in colunas:
#     print('%10s = %6.3f , p-value = %.9f' % (
#         coluna,
#         pearsonr(dados_treinamento[coluna], dados_treinamento['inadimplente'])[0],
#         pearsonr(dados_treinamento[coluna], dados_treinamento['inadimplente'])[1]
#         )
#     )

# ------------------------------------------------------------------------------
#  Histograma dos dados
#  Eixo vertical: Número de instâncias
#  Eixo horizontal: Determinado intervalo valores
# ------------------------------------------------------------------------------

# dados_treinamento.hist(bins=50, figsize=(35, 35))
# plt.show()

# ------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
# ------------------------------------------------------------------------------

dados_embaralhados = dados_treinamento.sample(frac=1, random_state=11012005)