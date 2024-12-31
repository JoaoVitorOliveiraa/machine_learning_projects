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

def apply_one_hot_encoder(data, features, target=False):
    "Função que aplica a classe OneHotEncoder em features categóricas, mantendo as demais inalteradas."

    # Separar a coluna do alvo das demais features.
    if target:
        data_target = data[target]
        data_features = data.drop(target, axis=1)

    else:
        data_target = None
        data_features = data

    # Substituindo espaços (' ') por underlines ('_').
    for feature in features:
        data_features[feature] = data_features[feature].str.replace(' ', '_')

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

# ------------------------------------------------------------------------------
# Implementação da função nas features categóricas dos conjuntos de dados.
# ------------------------------------------------------------------------------

features_categoricas = ['tipo', 'bairro', 'tipo_vendedor', 'diferenciais']
dados_treinamento = apply_one_hot_encoder(dados_treinamento, features_categoricas, 'preco')
dados_teste = apply_one_hot_encoder(dados_teste, features_categoricas)

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
'bairro_Agua_Fria',
'bairro_Apipucos',
'bairro_Areias',
'bairro_Arruda',
'bairro_Barro',
'bairro_Beira_Rio',
'bairro_Benfica',
'bairro_Boa_Vista',
'bairro_Bongi',
'bairro_Cajueiro',
'bairro_Campo_Grande',
'bairro_Caxanga',
'bairro_Centro',
'bairro_Cid_Universitaria',
'bairro_Coelhos',
'bairro_Cohab',
'bairro_Cordeiro',
'bairro_Derby',
'bairro_Dois_Irmaos',
'bairro_Engenho_do_Meio',
'bairro_Estancia',
'bairro_Guabiraba',
'bairro_Hipodromo',
'bairro_Ilha_do_Leite',
'bairro_Ilha_do_Retiro',
'bairro_Imbiribeira',
'bairro_Ipsep',
'bairro_Iputinga',
'bairro_Jaqueira',
'bairro_Jd_S_Paulo',
'bairro_Lagoa_do_Araca',
'bairro_Macaxeira',
'bairro_Monteiro',
'bairro_Paissandu',
'bairro_Piedade',
'bairro_Pina',
'bairro_Poco',
'bairro_Poco_da_Panela',
'bairro_Ponto_de_Parada',
'bairro_Prado',
'bairro_Recife',
'bairro_S_Jose',
'bairro_San_Martin',
'bairro_Sancho',
'bairro_Santana',
'bairro_Setubal',
'bairro_Soledade',
'bairro_Sto_Amaro',
'bairro_Sto_Antonio',
'bairro_Tamarineira',
'bairro_Tejipio',
'bairro_Torreao',
'bairro_Varzea',
'bairro_Zumbi',
'diferenciais_campo_de_futebol_e_copa',
'diferenciais_campo_de_futebol_e_esquina',
'diferenciais_campo_de_futebol_e_estacionamento_visitantes',
'diferenciais_campo_de_futebol_e_playground',
'diferenciais_campo_de_futebol_e_quadra_poliesportiva',
'diferenciais_campo_de_futebol_e_salao_de_festas',
'diferenciais_children_care',
'diferenciais_children_care_e_playground',
'diferenciais_churrasqueira',
'diferenciais_churrasqueira_e_campo_de_futebol',
'diferenciais_churrasqueira_e_copa',
'diferenciais_churrasqueira_e_esquina',
'diferenciais_churrasqueira_e_estacionamento_visitantes',
'diferenciais_churrasqueira_e_frente_para_o_mar',
'diferenciais_churrasqueira_e_playground',
'diferenciais_churrasqueira_e_sala_de_ginastica',
'diferenciais_churrasqueira_e_salao_de_festas',
'diferenciais_churrasqueira_e_sauna',
'diferenciais_copa',
'diferenciais_copa_e_esquina',
'diferenciais_copa_e_estacionamento_visitantes',
'diferenciais_copa_e_playground',
'diferenciais_copa_e_quadra_poliesportiva',
'diferenciais_copa_e_sala_de_ginastica',
'diferenciais_copa_e_salao_de_festas',
'diferenciais_esquina',
'diferenciais_esquina_e_estacionamento_visitantes',
'diferenciais_esquina_e_playground',
'diferenciais_esquina_e_quadra_poliesportiva',
'diferenciais_esquina_e_sala_de_ginastica',
'diferenciais_esquina_e_salao_de_festas',
'diferenciais_estacionamento_visitantes',
'diferenciais_estacionamento_visitantes_e_playground',
'diferenciais_estacionamento_visitantes_e_sala_de_ginastica',
'diferenciais_estacionamento_visitantes_e_salao_de_festas',
'diferenciais_frente_para_o_mar',
'diferenciais_frente_para_o_mar_e_campo_de_futebol',
'diferenciais_frente_para_o_mar_e_copa',
'diferenciais_frente_para_o_mar_e_esquina',
'diferenciais_frente_para_o_mar_e_playground',
'diferenciais_frente_para_o_mar_e_quadra_poliesportiva',
'diferenciais_frente_para_o_mar_e_salao_de_festas',
'diferenciais_piscina_e_children_care',
'diferenciais_piscina_e_esquina',
'diferenciais_piscina_e_estacionamento_visitantes',
'diferenciais_piscina_e_frente_para_o_mar',
'diferenciais_piscina_e_hidromassagem',
'diferenciais_piscina_e_quadra_de_squash',
'diferenciais_piscina_e_quadra_poliesportiva',
'diferenciais_piscina_e_sala_de_ginastica',
'diferenciais_piscina_e_salao_de_jogos',
'diferenciais_piscina_e_copa',
'diferenciais_piscina_e_campo_de_futebol',
'diferenciais_piscina',
'diferenciais_playground',
'diferenciais_playground_e_quadra_poliesportiva',
'diferenciais_playground_e_sala_de_ginastica',
'diferenciais_playground_e_salao_de_jogos',
'diferenciais_quadra_poliesportiva',
'diferenciais_quadra_poliesportiva_e_salao_de_festas',
'diferenciais_sala_de_ginastica',
'diferenciais_sala_de_ginastica_e_salao_de_festas',
'diferenciais_sala_de_ginastica_e_salao_de_jogos',
'diferenciais_salao_de_festas_e_salao_de_jogos',
'diferenciais_salao_de_festas_e_vestiario',
'diferenciais_salao_de_jogos',
'diferenciais_sauna',
'diferenciais_sauna_e_campo_de_futebol',
'diferenciais_sauna_e_copa',
'diferenciais_sauna_e_esquina',
'diferenciais_sauna_e_frente_para_o_mar',
'diferenciais_sauna_e_playground',
'diferenciais_sauna_e_quadra_poliesportiva',
'diferenciais_sauna_e_sala_de_ginastica',
'diferenciais_sauna_e_salao_de_festas',
'diferenciais_vestiario']

for feature in features_classes_dominantes:
    dados_treinamento.drop(feature, axis=1, inplace=True)
    
    # Algumas dessas colunas não estão presentes no conjunto de teste.
    if feature in dados_teste.columns:
        dados_teste.drop(feature, axis=1, inplace=True)

# ------------------------------------------------------------------------------
#  Remoção de features que estavam somente no conjunto de teste
# ------------------------------------------------------------------------------

features_somente_dados_teste = [
'bairro_Beberibe',
'bairro_Fundao',
'bairro_Ibura',
'diferenciais_campo_de_futebol_e_sala_de_ginastica',
'diferenciais_churrasqueira_e_children_care',
'diferenciais_copa_e_hidromassagem',
'diferenciais_estacionamento_visitantes_e_hidromassagem',
'diferenciais_estacionamento_visitantes_e_salao_de_jogos',
'diferenciais_frente_para_o_mar_e_children_care',
'diferenciais_frente_para_o_mar_e_hidromassagem',
'diferenciais_hidromassagem_e_salao_de_festas']

dados_teste.drop(features_somente_dados_teste, axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Exibindo os coeficientes de Pearson de cada atributo (entre o mesmo e o alvo)
# ------------------------------------------------------------------------------

print("\n\n\t-----Coeficiente de Pearson-----\n")
for coluna in dados_treinamento.columns:
    coef_pearsonr = pearsonr(dados_treinamento[coluna], dados_treinamento['preco'])[0]
    p_value = pearsonr(dados_treinamento[coluna], dados_treinamento['preco'])[1]
    print(f'{coluna}: {coef_pearsonr:.3f}, p-value: {p_value:.3f}\n')

# ------------------------------------------------------------------------------
# Remoção de features que possuíam um coeficiente de Pearson menor que 0.01
# ------------------------------------------------------------------------------

drop_list_pearson = ['area_extra', 'estacionamento', 'piscina', 'quadra', 's_festas', 'sauna', 'vista_mar', 'tipo_Apartamento', 
                'tipo_Casa', 'bairro_Boa_Viagem', 'bairro_Casa_Amarela', 'bairro_Encruzilhada', 'bairro_Espinheiro',
                'bairro_Gracas', 'bairro_Madalena', 'bairro_Parnamirim', 'bairro_Rosarinho', 'diferenciais_piscina_e_playground',
                'diferenciais_piscina_e_salao_de_festas', 'diferenciais_piscina_e_sauna', 'diferenciais_playground_e_salao_de_festas']

dados_treinamento.drop(drop_list_pearson, axis=1, inplace=True)
dados_teste.drop(drop_list_pearson, axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino esteja isenta de qualquer viés de seleção
# ------------------------------------------------------------------------------

dados_treinamento_embaralhados = dados_treinamento.sample(frac=1, random_state=11012005)

#------------------------------------------------------------------------------
# Separar o conjunto de treinamento em arrays X e Y, exibindo suas dimensões
#------------------------------------------------------------------------------

# Separando as features do alvo.
X = dados_treinamento_embaralhados.iloc[:, :-1].values
y = dados_treinamento_embaralhados.iloc[:, -1].values

# Conjunto de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=11012005)

# Conjunto de teste final
X_teste_final = dados_teste

# ------------------------------------------------------------------------------
# Aplicação da escala no X de treino e de teste
# ------------------------------------------------------------------------------

# escala = MinMaxScaler()
escala = StandardScaler()

escala.fit(X_treino)
X_treino_com_escala = escala.transform(X_treino)
X_teste_com_escala = escala.transform(X_teste)

# ------------------------------------------------------------------------------
# Criação de uma função para calcular o RMSPE
# ------------------------------------------------------------------------------

def rmspe(y_true, y_pred):
    "Função que retorna o valor do RMSPE."
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Verificar se há zeros em y_true para evitar divisão por zero
    if np.any(y_true == 0):
        raise ValueError("Os valores reais (y_true) não podem conter zeros ao calcular o RMSPE.")
    
    # Calcular o RMSPE
    percentage_errors = ((y_true - y_pred) / y_true) ** 2
    rmspe_value = np.sqrt(np.mean(percentage_errors))
    return rmspe_value

# ------------------------------------------------------------------------------
# Treinando o modelo KNeighborsRegressor, com k variando entre 1 e 50
# ------------------------------------------------------------------------------

print("\n\n\t-----Regressor com KNN-----\n")

# Para este laço, o melhor resultado foi em k = 1 com RMSPE = 0.5265 e R2 Score = 0.6374
for k in range(1, 51):    

    # Instanciando o regressor KNN.
    regressor_knn = KNeighborsRegressor(n_neighbors=k, weights="uniform")
    regressor_knn = regressor_knn.fit(X_treino_com_escala, y_treino)

    # Predições.
    y_resposta_treino = regressor_knn.predict(X_treino_com_escala)
    y_resposta_teste = regressor_knn.predict(X_teste_com_escala)

    # Calculando RMSPE e o R2 Score.
    rmspe_treino = rmspe(y_treino, y_resposta_treino)
    rmspe_teste = rmspe(y_teste, y_resposta_teste)
    r2_score_treino = r2_score(y_treino, y_resposta_treino)
    r2_score_teste = r2_score(y_teste, y_resposta_teste)

    print(f'\nK = {k}')
    print(f'RMSPE Treino: {rmspe_treino:.4f}')
    print(f'R2 Score Treino: {r2_score_treino:.4f}')
    print(f'RMSPE Teste: {rmspe_teste:.4f}')
    print(f'R2 Score Teste: {r2_score_teste:.4f}')

# ------------------------------------------------------------------------------
#  Treinando o modelo Regressão Linear
# ------------------------------------------------------------------------------

print("\n\n\t-----Regressor com Regressão Linear-----\n")

# Instanciando o regressor Regressão Linear.
regressor_regressao_linear = LinearRegression()
regressor_regressao_linear = regressor_regressao_linear.fit(X_treino_com_escala, y_treino)

# Predições.
y_resposta_treino = regressor_regressao_linear.predict(X_treino_com_escala)
y_resposta_teste = regressor_regressao_linear.predict(X_teste_com_escala)

# Calculando RMSPE e o R2 Score.
rmspe_treino = rmspe(y_treino, y_resposta_treino)
rmspe_teste = rmspe(y_teste, y_resposta_teste)
r2_score_teste = r2_score(y_teste, y_resposta_teste)

print(f'RMSPE Treino: {rmspe_treino:.4f}')
print(f'R2 Score Treino: {r2_score_treino:.4f}')
print(f'RMSPE Teste: {rmspe_teste:.4f}')
print(f'R2 Score Teste: {r2_score_teste:.4f}')

# ------------------------------------------------------------------------------
# Treinando o modelo Regressão Polinomial
# ------------------------------------------------------------------------------

print("\n\n\t-----Regressor com Regressão Polinomial-----\n")

# Para este laço, o melhor resultado foi em grau = 1 com RMSPE = 5.5511 e R2 Score = -9.0472
# for grau in range(1, 11):

# Por fim, escolhemos grau = 1.
grau = 1    

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

# Calculando RMSPE e o R2 Score.
rmspe_treino = rmspe(y_treino, y_resposta_treino)
rmspe_teste = rmspe(y_teste, y_resposta_teste)
r2_score_treino = r2_score(y_treino, y_resposta_treino)
r2_score_teste = r2_score(y_teste, y_resposta_teste)

print(f'\nGrau = {grau}')
print(f'RMSPE Treino: {rmspe_treino:.4f}')
print(f'R2 Score Treino: {r2_score_treino:.4f}')
print(f'RMSPE Teste: {rmspe_teste:.4f}')
print(f'R2 Score Teste: {r2_score_teste:.4f}')

# ------------------------------------------------------------------------------
# Treinando o modelo Regressão Polinomial com regularização Ridge (L2)
# ------------------------------------------------------------------------------

print("\n\n\t-----Regressor com Regressão Polinomial com regularização Ridge (L2)-----\n")
print('   ALPHA\t     RMSPE Treino      R2 Score       RMSPE Teste      R2 Score Teste')
print(' ---------- \t -----------    ------------    -------------    ---------------')

# Para este laço, o melhor resultado foi em a = 10000000000 com RMSPE = 1.6604 e R2 Score = -0.1581
# for a in [0.001, 0.010, 0.100, 1.000, 10.00, 100.0, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]:

# Para este laço, o melhor resultado foi em a = 5000000000 com RMSPE = 1.6604  e R2 Score = -0.1581
# for a in [100000000, 200000000, 300000000, 400000000, 500000000, 600000000, 700000000, 800000000, 900000000, 900000000, 1000000000,
          # 1500000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000]:
    
# Por fim, escolhemos a = 5000000000.
a = 5000000000

# Instanciando o metodo PolynomialFeatures.
polynomial_features = PolynomialFeatures(degree=2)
polynomial_features = polynomial_features.fit(X_treino)
X_treino_poly = polynomial_features.transform(X_treino_com_escala)
X_teste_poly = polynomial_features.transform(X_teste_com_escala)

# Instanciando a regularização Ridge (L2).
regularizacao_ridge = Ridge(alpha=a)
regularizacao_ridge = regularizacao_ridge.fit(X_treino_poly, y_treino)

# Predições.
y_resposta_treino = regularizacao_ridge.predict(X_treino_poly)
y_resposta_teste = regularizacao_ridge.predict(X_teste_poly)

# Calculando RMSPE e o R2 Score.
rmspe_treino = rmspe(y_treino, y_resposta_treino)
rmspe_teste = rmspe(y_teste, y_resposta_teste)
r2_score_treino = r2_score(y_treino, y_resposta_treino)
r2_score_teste = r2_score(y_teste, y_resposta_teste)

print(f'  {a} ', f'\t\t   {rmspe_treino:.4f} ', f'\t\t   {r2_score_treino:.4f} ', f'\t\t   {rmspe_teste:.4f}', f'\t\t   {r2_score_teste:.4f}')

# -------------------------------------------------------------------------------
# Treinando a submissão final para o Kaggle, com o modelo que obteve o melhor 
# RMSPE e R2 Score (KNN).
# -------------------------------------------------------------------------------

# Utilizando todos os dados.
X_treino_submissao = X
X_teste_submissao = X_teste_final
y_treino_submissao = y

# Colocando em escala.
escala = StandardScaler()
escala.fit(X_treino_submissao)
X_treino_submissao_com_escala = escala.transform(X_treino_submissao)
X_teste_submissao_com_escala = escala.transform(X_teste_submissao)

# Aplicando o modelo
k = 1
regressor_knn = KNeighborsRegressor(n_neighbors=k, weights="uniform")
regressor_knn = regressor_knn.fit(X_treino_submissao_com_escala, y_treino_submissao)
y_resposta_teste_submissao = regressor_knn.predict(X_teste_submissao_com_escala)

# Criando o DataFrame de submissão.
submissao_final_kaggle = pd.DataFrame({
    'Id': ids_dados_teste,
    'preco': y_resposta_teste_submissao
})

# Salvando em CSV
submissao_final_kaggle.to_csv('submissao_final_kaggle.csv', index=False)
print("Arquivo salvo como 'submissao_final_kaggle.csv'")