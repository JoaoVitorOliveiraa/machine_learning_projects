#==============================================================================
# Trabalho 2 - Estimar o preço de um imóvel a partir de suas características
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures

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
#  Remoção de features que possuíam uma classe extremamente dominante
# ------------------------------------------------------------------------------

features_classes_dominantes = [
's_jogos',
's_ginastica',
'tipo_Loft',
'tipo_Quitinete',
'bairro_Aflitos',
'bairro_Afogados',
'bairro_Agua Fria',
'bairro_Apipucos',
'bairro_Areias',
'bairro_Arruda',
'bairro_Barro',
'bairro_Beira Rio',
'bairro_Benfica',
'bairro_Boa Vista',
'bairro_Bongi',
'bairro_Cajueiro',
'bairro_Campo Grande',
'bairro_Caxanga',
'bairro_Centro',
'bairro_Cid Universitaria',
'bairro_Coelhos',
'bairro_Cohab',
'bairro_Cordeiro',
'bairro_Derby',
'bairro_Dois Irmaos',
'bairro_Engenho do Meio',
'bairro_Estancia',
'bairro_Guabiraba',
'bairro_Hipodromo',
'bairro_Ilha do Leite',
'bairro_Ilha do Retiro',
'bairro_Imbiribeira',
'bairro_Ipsep',
'bairro_Iputinga',
'bairro_Jaqueira',
'bairro_Jd S Paulo',
'bairro_Lagoa do Araca',
'bairro_Macaxeira',
'bairro_Monteiro',
'bairro_Paissandu',
'bairro_Piedade',
'bairro_Pina',
'bairro_Poco',
'bairro_Poco da Panela',
'bairro_Ponto de Parada',
'bairro_Prado',
'bairro_Recife',
'bairro_S Jose',
'bairro_San Martin',
'bairro_Sancho',
'bairro_Santana',
'bairro_Setubal',
'bairro_Soledade',
'bairro_Sto Amaro',
'bairro_Sto Antonio',
'bairro_Tamarineira',
'bairro_Tejipio',
'bairro_Torreao',
'bairro_Varzea',
'bairro_Zumbi',
'diferenciais_campo de futebol e copa',
'diferenciais_campo de futebol e esquina',
'diferenciais_campo de futebol e estacionamento visitantes',
'diferenciais_campo de futebol e playground',
'diferenciais_campo de futebol e quadra poliesportiva',
'diferenciais_campo de futebol e salao de festas',
'diferenciais_children care',
'diferenciais_children care e playground',
'diferenciais_churrasqueira',
'diferenciais_churrasqueira e campo de futebol',
'diferenciais_churrasqueira e copa',
'diferenciais_churrasqueira e esquina',
'diferenciais_churrasqueira e estacionamento visitantes',
'diferenciais_churrasqueira e frente para o mar',
'diferenciais_churrasqueira e playground',
'diferenciais_churrasqueira e sala de ginastica',
'diferenciais_churrasqueira e salao de festas',
'diferenciais_churrasqueira e sauna',
'diferenciais_copa',
'diferenciais_copa e esquina',
'diferenciais_copa e estacionamento visitantes',
'diferenciais_copa e playground',
'diferenciais_copa e quadra poliesportiva',
'diferenciais_copa e sala de ginastica',
'diferenciais_copa e salao de festas',
'diferenciais_esquina',
'diferenciais_esquina e estacionamento visitantes',
'diferenciais_esquina e playground',
'diferenciais_esquina e quadra poliesportiva',
'diferenciais_esquina e sala de ginastica',
'diferenciais_esquina e salao de festas',
'diferenciais_estacionamento visitantes',
'diferenciais_estacionamento visitantes e playground',
'diferenciais_estacionamento visitantes e sala de ginastica',
'diferenciais_estacionamento visitantes e salao de festas',
'diferenciais_frente para o mar',
'diferenciais_frente para o mar e campo de futebol',
'diferenciais_frente para o mar e copa',
'diferenciais_frente para o mar e esquina',
'diferenciais_frente para o mar e playground',
'diferenciais_frente para o mar e quadra poliesportiva',
'diferenciais_frente para o mar e salao de festas',
'diferenciais_piscina e children care',
'diferenciais_piscina e esquina',
'diferenciais_piscina e estacionamento visitantes',
'diferenciais_piscina e frente para o mar',
'diferenciais_piscina e hidromassagem',
'diferenciais_piscina e quadra de squash',
'diferenciais_piscina e quadra poliesportiva',
'diferenciais_piscina e sala de ginastica',
'diferenciais_piscina e salao de jogos',
'diferenciais_piscina e copa',
'diferenciais_piscina e campo de futebol',
'diferenciais_piscina',
'diferenciais_playground',
'diferenciais_playground e quadra poliesportiva',
'diferenciais_playground e sala de ginastica',
'diferenciais_playground e salao de jogos',
'diferenciais_quadra poliesportiva',
'diferenciais_quadra poliesportiva e salao de festas',
'diferenciais_sala de ginastica',
'diferenciais_sala de ginastica e salao de festas',
'diferenciais_sala de ginastica e salao de jogos',
'diferenciais_salao de festas e salao de jogos',
'diferenciais_salao de festas e vestiario',
'diferenciais_salao de jogos',
'diferenciais_sauna',
'diferenciais_sauna e campo de futebol',
'diferenciais_sauna e copa',
'diferenciais_sauna e esquina',
'diferenciais_sauna e frente para o mar',
'diferenciais_sauna e playground',
'diferenciais_sauna e quadra poliesportiva',
'diferenciais_sauna e sala de ginastica',
'diferenciais_sauna e salao de festas',
'diferenciais_vestiario']

dados_treinamento.drop(features_classes_dominantes, axis=1, inplace=True)
dados_teste.drop(features_classes_dominantes, axis=1, inplace=True)

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

escala = MinMaxScaler()
# escala = StandardScaler()

escala.fit(X_treino)
X_treino_com_escala = escala.transform(X_treino)
X_teste_com_escala = escala.transform(X_teste)

# ------------------------------------------------------------------------------
# Treinando o modelo KNeighborsClassifier, com k variando entre 1 e 30
# ------------------------------------------------------------------------------
# Primeiro teste: k = 21 -- RMSE = 1478825.5182 -- R2 = -3.8099
# Segundo teste: k = 31 -- RMSE = 791209.1434 -- R2 = -0.3769 (StandardScaler)
# Terceiro teste: k = 31 -- RMSE = 724420.5889 -- R2 = -0.1542 (MinMaxScaler)

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
# Segundo teste: RMSE = 2547988.6239 -- R2 = -13.2791 (StandardScaler)
# Terceiro teste: RMSE = 2151041.3687 -- R2 = -9.1766 (MinMaxScaler)

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
# Segundo teste: Grau = 1 -- RMSE: 2151041.3687 -- R2: -9.1766 (StandardScaler)
# Terceiro teste: Erro de não possuir espaço suficiente para alocar memória. (MinMaxScaler)

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

# ------------------------------------------------------------------------------
# Treinando o modelo Regressão Polinomial com regularização Ridge (L2)
# ------------------------------------------------------------------------------
# Primeiro teste: Erro de não possuir espaço suficiente para alocar memória.
# Segundo teste: a = 520 -- RMSE = 1723217.7234 -- R2 =- 5.5311 (StandardScaler)
# Terceiro teste: a = 64000 -- RMSE = 722421.7111 -- R2 =- -0.1479 (MinMaxScaler)

print("\n\n\t-----Regressor com Regressão Polinomial com regularização Ridge (L2)-----\n")

print('   ALPHA\t     RMSE Treino      R2 Score       RMSE Teste      R2 Score Teste')
print(' ---------- \t -----------    ------------    -------------    ---------------')

# No teste 2, os melhores valores deste laço foram em a = 1000.
# No teste 3, os melhores valores deste laço foram em a = 100000.
# for a in [0.001, 0.010, 0.100, 1.000, 10.00, 100.0, 1000, 10000, 100000, 1000000]:

# No teste 2, os melhores valores deste laço foram em a = 500.
#for a in [100.0, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 5000, 8000, 10000]:
# No teste 2, os melhores valores deste laço foram em a = 520.
# for a in [400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600]:

# No teste 3, os melhores valores deste laço foram em a = 80000.
# for a in [10000, 20000, 50000, 80000, 100000, 200000, 500000, 800000, 1000000]:
# No teste 3, os melhores valores deste laço foram em a = 65000.
# for a in [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]:
# No teste 3, os melhores valores deste laço foram em a = 64000.
for a in [60000, 61000, 62000, 63000, 63500, 64000, 64500,  65000, 65200, 65400, 65600, 65800, 66000, 67000, 68000, 69000, 70000]:

    # Instanciando o metodo PolynomialFeatures.
    polynomial_features = PolynomialFeatures(degree=3)
    polynomial_features = polynomial_features.fit(X_treino)
    X_treino_poly = polynomial_features.transform(X_treino_com_escala)
    X_teste_poly = polynomial_features.transform(X_teste_com_escala)

    # Instanciando a regularização Ridge (L2).
    regularizacao_ridge = Ridge(alpha=a)
    regularizacao_ridge = regularizacao_ridge.fit(X_treino_poly, y_treino)

    # Predições.
    y_resposta_treino = regularizacao_ridge.predict(X_treino_poly)
    y_resposta_teste = regularizacao_ridge.predict(X_teste_poly)

    # Calculando RMSE e o R2 Score.
    rmse_treino = math.sqrt(mean_squared_error(y_treino, y_resposta_treino))
    rmse_teste = math.sqrt(mean_squared_error(y_teste, y_resposta_teste))
    r2_score_treino = r2_score(y_treino, y_resposta_treino)
    r2_score_teste = r2_score(y_teste, y_resposta_teste)

    print(f'  {a} ', f'\t\t{rmse_treino:.4f} ', f'\t\t{r2_score_treino:.4f} ', f'\t\t{rmse_teste:.4f}', f'\t\t{r2_score_teste:.4f}')

    # print(f'\nAlpha = {a}')
    # print(f'RMSE Treino: {rmse_treino:.4f}')
    # print(f'R2 Score Treino: {r2_score_treino:.4f}')
    # print(f'RMSE Teste: {rmse_teste:.4f}')
    # print(f'R2 Score Teste: {r2_score_teste:.4f}')

# -------------------------------------------------------------------------------
# Treinando a primeira submissão para o kaggle (Regressor com Regressão Polinomial com regularização Ridge (L2))
# -------------------------------------------------------------------------------

# Utilizando todos os dados.
X_treino_submissao = X
X_teste_submissao = X_teste_final
y_treino_submissao = y

# Colocando em escala.
escala.fit(X_treino_submissao)
X_treino_submissao_com_escala = escala.transform(X_treino_submissao)
X_teste_submissao_com_escala = escala.transform(X_teste_submissao)

# Aplicando o modelo
a = 64000

# Instanciando o metodo PolynomialFeatures.
polynomial_features = PolynomialFeatures(degree=3)
polynomial_features = polynomial_features.fit(X_treino_submissao)
X_treino_poly = polynomial_features.transform(X_treino_submissao_com_escala)
X_teste_poly = polynomial_features.transform(X_teste_submissao_com_escala)

# Instanciando a regularização Ridge (L2).
regularizacao_ridge = Ridge(alpha=a)
regularizacao_ridge = regularizacao_ridge.fit(X_treino_poly, y_treino_submissao)

# Predição.
y_resposta_teste_submissao = regularizacao_ridge.predict(X_teste_poly)

# Criando o DataFrame de submissão.
primeira_submissao_kaggle = pd.DataFrame({
    'Id': ids_dados_teste,
    'preco': y_resposta_teste_submissao
})

# Salvando em CSV
primeira_submissao_kaggle.to_csv('primeira_submissao_kaggle.csv', index=False)
print("Arquivo salvo como 'primeira_submissao_kaggle.csv'")
