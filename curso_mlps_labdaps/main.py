#==============================================================================
# Machine Learning para Predições em Saúde - Curso de Verão 2025 - FSP - LABDAPS
#==============================================================================

#------------------------------------------------------------------------------
# Importando bibliotecas
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             accuracy_score, brier_score_loss)
from sklearn.model_selection import (KFold, cross_val_score, train_test_split,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from dfply import *
from dtreeviz.trees import *                # Visualização de árvores de decisão

#------------------------------------------------------------------------------
# Configurando a fonte utilizada em gráficos gerados pelo Matplotlib
#------------------------------------------------------------------------------

rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans'], 'size': 10})
rc('mathtext', **{'default': 'regular'})

# rc('font', {...}) -> Define as configurações gerais da fonte nos gráficos do Matplotlib.
# 'family': 'sans-serif' -> Define a família de fontes como sans-serif (sem serifa).
# 'sans-serif': ['DejaVu Sans'] -> Especifica que a fonte a ser usada dentro da família sans-serif será a DejaVu Sans (padrão do Matplotlib).
# 'size': 10 -> Define o tamanho padrão do texto nos gráficos como 10 pontos.
# 'mathtext': {'default': 'regular'} -> Define que os textos matemáticos (escritos com LaTeX dentro do Matplotlib) serão renderizados com fonte regular, sem itálico por padrão.

#------------------------------------------------------------------------------
# Definindo a semente de aleatoriedade
#------------------------------------------------------------------------------

np.random.seed(30)

#------------------------------------------------------------------------------
# Obtendo o conjunto de dados
#------------------------------------------------------------------------------

dados = pd.read_csv('https://raw.githubusercontent.com/laderast/cvdRiskData/master/data-raw/fullPatientData.csv')

#------------------------------------------------------------------------------
# Verificando o número de linhas e colunas do conjunto de dados
#------------------------------------------------------------------------------

print(f"\n\n\t\t-----Shape do dataset-----\n\n{dados.shape}\n")

# ------------------------------------------------------------------------------
#  Exibindo as primeiras 10 linhas do conjunto de dados 
# ------------------------------------------------------------------------------

print("\n\t\t-----Dez primeiras linhas do conjunto de dados-----\n")
print(dados.head(n=10))

# ------------------------------------------------------------------------------
#  Exibindo as primeiras 10 linhas do conjunto de dados (de maneira transposta)
# ------------------------------------------------------------------------------

print("\n\t\t-----Dez primeiras linhas do conjunto de dados (de maneira transposta)-----\n")
print(dados.head(n=10).T)

# ------------------------------------------------------------------------------
#  Obtendo uma descrição do conjunto de dados 
# ------------------------------------------------------------------------------

print("\n\n\t-----Descrição do conjunto de dados-----\n")
dados.info()

# ------------------------------------------------------------------------------
#  Descobrindo quais categorias existem nas features e no alvo, além de quantos
#  dígitos pertencem a cada categoria, usando a função value_counts()
# ------------------------------------------------------------------------------

print("\n\n\t-----Categorias das features e do alvo, com suas respectivas quantidades-----\n")
for coluna in list(dados.columns):
    print("\n", dados[coluna].value_counts())

# ------------------------------------------------------------------------------
#  Resumo dos atributos numéricos do conjunto de dados 
# ------------------------------------------------------------------------------

print("\n\n\t-----Resumo dos atributos numéricos-----\n")
print(dados.describe())

# ------------------------------------------------------------------------------
#  Melhor exibição das classes das colunas, pois o describe() não exibiu todas
# ------------------------------------------------------------------------------

print("\n\n\t-----Melhor exibição das classes das colunas-----\n")
for coluna in list(dados.columns):
    print(f"\nClasses {coluna}: ", list(dados[coluna].unique()))

# ------------------------------------------------------------------------------
#  Filtrando o conjunto de dados
# ------------------------------------------------------------------------------

# Iremos considerar individuos com idade superior a 55 anos.
dados = dados[dados["numAge"] > 55]

# Removendo colunas.
dados = dados.drop(["patientID", "age", "treat"], axis=1)
# dados.drop(["patientID", "age", "treat"], axis=1, inplace=True)

# ------------------------------------------------------------------------------
#  Aplicando o OneHotEncoding nas variáveis categóricas
# ------------------------------------------------------------------------------

# Inserimos o comando get_dummies para a realização do one hot encoding para as 
# variáveis categórias que possuem mais de duas categorias.
# As variáveis dicotômicas não precisam passar pelo one hot encoding. Podemos 
# somente aplicar o label encoding, substintuindo os valores por 0 e 1.

dados =  pd.get_dummies(dados, columns=['race'], dtype=int)

# ------------------------------------------------------------------------------
#  Aplicando o LabelEncoder nas variáveis dicotômicas
# ------------------------------------------------------------------------------

# O LabelEncoder utiliza por padrão a ordem alfabética. Dessa forma, como nossos
# dados estavam com categorias "N" e "Y", "N" assume o valor 0 e "Y" o valor 1.
# Para o sexo: F -> 0,  M -> 1
# Caso este não seja o cenário desejado, é possível invertar a ordem pelo "inverse_transform".

dados[['htn', 'smoking', 't2d', 'gender']] = dados[['htn', 'smoking', 't2d', 'gender']].apply(LabelEncoder().fit_transform)

# ------------------------------------------------------------------------------
#  Obtendo os conjuntos de treino e teste
# ------------------------------------------------------------------------------

features = dados.iloc[:, dados.columns != 'cvd'] # Separamos as nossas variáveis preditoras do nosso desfecho/target ==> conjunto X.
target = dados.iloc[:, dados.columns == 'cvd'] # Criamos um vetor (única coluna) selecionando somente a variável desfecho ==> conjunto y.

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    stratify = target,  # Estratificando com base no alvo.
                                                    train_size = 0.70,
                                                    random_state = 30)

# ------------------------------------------------------------------------------
#  Ordenando as colunas, alocando as contínuas nas primeiras posições
# ------------------------------------------------------------------------------

# Standarscaler com passthrough tem um problema de ordenação das colunas para algumas versões do SKLearn.
# Quando aplicado, o método fornece o resultado com as colunas padronizadas em primeiro, seguidas das demais colunas.
# Para resolver este problema, iremos ordenar as nossas colunas alocando as contínuas nas primeiras posições

X_train = X_train.loc[:,['numAge', 'bmi', 'tchol', 'sbp', # variaveis continuas
                          'htn', 'smoking', 't2d', 'gender','race_AmInd', 'race_Asian/PI', 'race_Black/AfAm', 'race_White']]
X_test = X_test.loc[:,['numAge', 'bmi', 'tchol', 'sbp', # variaveis continuas
                          'htn', 'smoking', 't2d', 'gender','race_AmInd', 'race_Asian/PI', 'race_Black/AfAm', 'race_White']]

X_train_columns = X_train.columns
X_test_columns = X_test.columns

# ------------------------------------------------------------------------------
#  Escalonando as features com valores contínuos 
# ------------------------------------------------------------------------------

# Variáveis contínuas que serão padronizadas.
continuous_features = ['numAge', 'bmi', 'tchol', 'sbp']

# Criando o ColumnTransformer.
scaler = ColumnTransformer([
    ('scaler', StandardScaler(), continuous_features)
], remainder='passthrough')  # Mantém as outras colunas sem alterações.

# Realizamos o fit no treino e aplicamos os valores obtidos (média e desvio padrão) para a padronização do treino e do teste.
scaler.fit(X_train) 

X_train_scaled = scaler.transform(X_train) # transformando (padronizando) os dados de treino.
X_test_scaled = scaler.transform(X_test) # transformando (padronizando) os dados de teste, com as informações do treino.

#               ---------- OPÇÃO 2 ----------
# Criando e ajustando o scaler SOMENTE nas colunas contínuas
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(X_train[continuous_features])  # Ajusta apenas nas colunas numéricas
# X_train[continuous_features] = scaler.transform(X_train[continuous_features])
# X_test[continuous_features] = scaler.transform(X_test[continuous_features])

# ------------------------------------------------------------------------------
#  Evitando a exibição dos dados em notacao científica
# ------------------------------------------------------------------------------

# Esse comando configura o pandas para exibir os números float com três casas
# decimais ao imprimir DataFrames ou Séries.
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ------------------------------------------------------------------------------
#  Transformando X_train_scaled e X_test_scaled em DataFrame.
# ------------------------------------------------------------------------------

# O escalonamento retorna os dados em formato array. 
# Precisamos transformá-los novamente para data.frame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_columns)

# ------------------------------------------------------------------------------
#  Aplicando o LabelEncoder no target
# ------------------------------------------------------------------------------

# transformando a variável target: Y --> 1 e N --> 0.
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)