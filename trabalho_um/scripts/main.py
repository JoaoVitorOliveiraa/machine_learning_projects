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
#  Melhor exibição das classes das features, pois os describe() não exibiu todas
# ------------------------------------------------------------------------------

print("\n\n\t-----Melhor exibição das classes das features-----\n")
for feature in list(dados_treinamento.columns):
    print(f"\nClasses {feature}: ", list(dados_treinamento[feature].unique()))

# ------------------------------------------------------------------------------
#  Remoção de features inúteis
#  grau_instrucao: Totalmente preenchida com zeros
#  possui_telefone_celular: Totalmente preenchida com "N"
#  qtde_contas_bancarias_especiais: Conteúdo idêntico à "qtde_contas_bancarias"
#  meses_no_trabalho: Vasta maioria com valor zero.
# ------------------------------------------------------------------------------

features_inuteis = ["grau_instrucao", "possui_telefone_celular", "qtde_contas_bancarias_especiais", "meses_no_trabalho"]
dados_treinamento.drop(features_inuteis, axis=1, inplace=True)

# ------------------------------------------------------------------------------
#  Criação de uma função para substituir o valor de uma classe em uma feature, e
#  sua aplicação nas features
# ------------------------------------------------------------------------------

def substituir_valores_da_classe(data, features_list, old_value, new_value):
    "Função que substitui o valor de uma classe em uma feature."

    for feature in features_list:
        data[feature] = data[feature].replace({old_value: new_value})

# Substituindo as classes binárias 'N' e 'Y' em 0 e 1 de cada feature categórica que apresenta essas classes.
features_com_classes_y_n = ['possui_telefone_residencial', 'vinculo_formal_com_empresa', 'possui_telefone_trabalho']
substituir_valores_da_classe(dados_treinamento, features_com_classes_y_n, 'Y', 1)
substituir_valores_da_classe(dados_treinamento, features_com_classes_y_n, 'N', 0)

# Dicionário onde as chaves e valores são as regiões do Brasil e listas com siglas de estados.
dict_regioes_do_brasil = {'regiao_norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
                          'regiao_nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
                          'regiao_centro_oeste': ['DF', 'GO', 'MT', 'MS'],
                          'regiao_sudeste': ['ES', 'MG', 'RJ', 'SP'],
                          'regiao_sul': ['PR', 'RS', 'SC']}

# Lista com todas as features que possuem siglas de estados brasileiros como classes.
features_siglas_estados_brasileiros = ['estado_onde_trabalha', 'estado_onde_nasceu', 'estado_onde_reside']

# Substituindo cada estado por sua respectiva região, em cada feature listada.
for regiao, classes in dict_regioes_do_brasil.items():
    for classe in classes:
        substituir_valores_da_classe(dados_treinamento, features_siglas_estados_brasileiros, classe, regiao)

# ------------------------------------------------------------------------------
#  Substituindo os espaços ausentes das features, que estavam incompletas e que o
#  significado de suas classes não foi especificado, pela mediana dos valores de
#  suas classes, pois os valores não estão uniformemente distribuídos.
# ------------------------------------------------------------------------------

features_incompletas = ['tipo_residencia', 'meses_na_residencia', 'profissao', 'ocupacao', 'profissao_companheiro', 'sexo',
                        'grau_instrucao_companheiro', 'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho',
                        'estado_onde_trabalha', 'estado_onde_nasceu', 'estado_onde_reside']

for feature in features_incompletas:

    # Substituindo espaços ' ' por 'N' (não informado).
    if feature == 'sexo':
        substituir_valores_da_classe(dados_treinamento, [feature], ' ', 'N')

    # Substituindo os espaços ' ' por "classe_invalida".
    elif feature in features_siglas_estados_brasileiros:
        substituir_valores_da_classe(dados_treinamento, features_siglas_estados_brasileiros, ' ', 'classe_invalida')

    # Substituindo os espaços ' ' pela média dos valores (após transformá-los em números).
    elif feature in ['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho']:

        # Primeiramente, substituímos os espaços vazios por None, a fim de realizar a conversão da coluna.
        substituir_valores_da_classe(dados_treinamento, [feature], ' ', None)

        # Converte os valores str da coluna em numéricos. Coerce substitui strings inválidas por NaN.
        dados_treinamento[feature] = pd.to_numeric(dados_treinamento[feature], errors='coerce')

        # Substituindo os valores NaN pela mediana.
        mediana_feature = dados_treinamento[feature].median()
        dados_treinamento[feature] = dados_treinamento[feature].fillna(mediana_feature)

    else:
        mediana_feature = dados_treinamento[feature].median()
        dados_treinamento[feature] = dados_treinamento[feature].fillna(mediana_feature)

# ------------------------------------------------------------------------------
#  Criação de uma função para calcular a taxa de inadimplência de cada classe
#  das features categóricas.
# ------------------------------------------------------------------------------

def calcular_taxa_de_inadimplencia_das_classes(data, features_list, target='inadimplente'):
    "Função que calcula a taxa de inadimplência de cada classe das features categóricas."

    for feature in features_list:
        # Exibindo a quantidade de cada categoria na coluna.
        print(f"\n\n\t-----Categorias da feature '{feature}'-----\n")
        dicionario_feature = dict(data[feature].value_counts())
        print(data[feature].value_counts())

        # Calculando e exibindo a taxa de inadimplência para cada categoria.
        print(f"\n\n\t-----Taxa de inadimplência para as categorias da feature '{feature}'-----\n")
        for categoria, quantidade in dicionario_feature.items():
            quantidade_inadimplentes = data[data[feature] == categoria][target].sum()
            taxa_inadimplencia = (quantidade_inadimplentes / quantidade) * 100
            print(f"Categoria: {categoria}")
            print(f"Quantidade Total: {quantidade}")
            print(f"Quantidade Inadimplentes: {quantidade_inadimplentes}")
            print(f"Taxa de Inadimplência: {taxa_inadimplencia:.3f}%\n")

# ------------------------------------------------------------------------------
#  Calculando a taxa de inadimplência das classes de cada feature categórica
#  Obs: Como foi observado que nas features que apresentavam apenas duas classes,
#  a taxa de inadimplência era de 49%-51% para cada, só foram adicionadas à lista
#  abaixo as features que possuíam mais de duas classes
# ------------------------------------------------------------------------------

features_categoricas = ['produto_solicitado', 'forma_envio_solicitacao', 'tipo_endereco', 'sexo', 'estado_civil',
                        'nacionalidade', 'estado_onde_nasceu', 'estado_onde_reside', 'tipo_residencia',
                        'estado_onde_trabalha', 'profissao', 'ocupacao', 'profissao_companheiro',
                        'grau_instrucao_companheiro', 'local_onde_reside', 'local_onde_trabalha']

calcular_taxa_de_inadimplencia_das_classes(dados_treinamento, features_categoricas)

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