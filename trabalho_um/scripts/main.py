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
from pyexpat import features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

#------------------------------------------------------------------------------
# Importar os conjuntos de teste e treinamento (retirando as colunas dos id's)
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
#  Remoção de features inúteis
#  grau_instrucao: Totalmente preenchida com zeros
#  possui_telefone_celular: Totalmente preenchida com "N"
#  qtde_contas_bancarias_especiais: Conteúdo idêntico à "qtde_contas_bancarias"
# ------------------------------------------------------------------------------

features_inuteis = ["grau_instrucao", "possui_telefone_celular", "qtde_contas_bancarias_especiais"]
dados_treinamento.drop(features_inuteis, axis=1, inplace=True)

# ------------------------------------------------------------------------------
#  Substituindo espaços vazios do atributo 'sexo' por 'N' (não informado)
# ------------------------------------------------------------------------------

dados_treinamento['sexo'] = dados_treinamento['sexo'].str.strip().replace('', 'N')

# ------------------------------------------------------------------------------
#  Exibindo a quantidade de cada categoria do atributo "sexo"
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#  Criação de uma função para calcular a taxa de inadimplência de cada classe
#  das features categóricas.
# ------------------------------------------------------------------------------

def porcentagem_de_inadimplencia_classe(data, feature, target='inadimplente'):
    "Função que calcula a taxa de inadimplência de cada classe das features categóricas."

    # Substituindo espaços vazios por 'N' (não informado), caso seja a feature "sexo".
    if feature == "sexo":
        data[feature] = data[feature].str.strip().replace('', 'N')

    # Exibindo a quantidade de cada categoria na coluna.
    print(f"\n\n\t-----Categorias do feature '{feature}'-----\n")
    dicionario_feature = dict(data[feature].value_counts())
    print(data[feature].value_counts())

    # Calculando e exibindo a porcentagem de inadimplência para cada categoria.
    print(f"\n\n\t-----Porcentagem de inadimplência para as categorias da feature '{feature}'-----\n")
    for categoria, quantidade in dicionario_feature.items():
        quantidade_inadimplentes = data[data[feature] == categoria][target].sum()
        porcentagem_inadimplentes = (quantidade_inadimplentes / quantidade) * 100
        print(f"Categoria: {categoria}")
        print(f"Quantidade Total: {quantidade}")
        print(f"Quantidade Inadimplentes: {quantidade_inadimplentes}")
        print(f"Porcentagem de Inadimplência: {porcentagem_inadimplentes:.3f}%\n")

# # Lista de features categóricas para processar
# features_categoricas = ['sexo', 'outra_feature', 'mais_uma_feature']
#
# # Processar cada feature categórica
# for feature in features_categoricas:
#     processar_feature_categorica(dados_treinamento, feature)

print("\n\n\t-----Categorias do atributo 'sexo'-----\n")
feature_sexo = dados_treinamento["sexo"]
dicionario_sexo = dict(feature_sexo.value_counts())
print(feature_sexo.value_counts())


# ------------------------------------------------------------------------------
#  Calculando a porcentagem de inadimplência do sexo feminino
# ------------------------------------------------------------------------------

print("\n\n\t-----Porcentagem de inadimplência do sexo 'feminino'-----\n")
quantidade_f = dicionario_sexo["F"]
quantidade_f_inadimplentes = dados_treinamento[feature_sexo=='F']['inadimplente'].sum()
porcentagem_f_inadimplentes = (quantidade_f_inadimplentes/quantidade_f) * 100
print(f"Quantidade Sexo Feminino: {(quantidade_f)}\n")
print(f"Quantidade Sexo Feminino Inadimplente: {(quantidade_f_inadimplentes)}\n")
print(f"Porcentagem Inadimplência: {(porcentagem_f_inadimplentes):.3f}%\n")

# ------------------------------------------------------------------------------
#  Calculando a porcentagem de inadimplência do sexo masculino
# ------------------------------------------------------------------------------

print("\n\n\t-----Porcentagem de inadimplência do sexo 'masculino'-----\n")
quantidade_m = dicionario_sexo["M"]
quantidade_m_inadimplentes = dados_treinamento[feature_sexo=='M']['inadimplente'].sum()
porcentagem_m_inadimplentes = (quantidade_m_inadimplentes/quantidade_m) * 100
print(f"Quantidade Sexo Masculino: {(quantidade_m)}\n")
print(f"Quantidade Sexo Masculino Inadimplente: {(quantidade_m_inadimplentes)}\n")
print(f"Porcentagem Inadimplência: {(porcentagem_m_inadimplentes):.3f}%\n")

# ------------------------------------------------------------------------------
#  Calculando a porcentagem de inadimplência do sexo não informado
# ------------------------------------------------------------------------------

print("\n\n\t-----Porcentagem de inadimplência do sexo 'não informado'-----\n")
quantidade_n = dicionario_sexo["N"]
quantidade_n_inadimplentes = dados_treinamento[feature_sexo=='N']['inadimplente'].sum()
porcentagem_n_inadimplentes = (quantidade_n_inadimplentes/quantidade_n) * 100
print(f"Quantidade Sexo Não Informado: {(quantidade_n)}\n")
print(f"Quantidade Sexo Não Informado Inadimplente: {(quantidade_n_inadimplentes)}\n")
print(f"Porcentagem Inadimplência: {(porcentagem_n_inadimplentes):.3f}%\n")


#------------------------------------------------------------------------------
# Separar o conjunto de treinamento em atributos e alvo, exibindo suas dimensões
#------------------------------------------------------------------------------

# atributos = dados_treinamento.iloc[:, :-1].to_numpy()   # ou :-1].values
# rotulos = dados_treinamento.iloc[:, -1].to_numpy()      # ou :-1].values
# print("\n\n\t-----Dimensões-----")
# print(f"\nDimensão das features: {atributos.shape}")
# print(f"Dimensão dos rótulos: {rotulos.shape}\n")

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