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
# Exibindo as features do dataset e seus tipos uas comparações gráficas com o alvo
# ------------------------------------------------------------------------------

print("\n\n\t-----Features disponíveis-----\n")
print(list(dados_treinamento.columns))

print("\n\n-----Tipos das features-----\n")
print(dados_treinamento.dtypes)

# ------------------------------------------------------------------------------
# Exibindo o histograma entre as quantidades e os valores do alvo
# ------------------------------------------------------------------------------

print(f"\n\n\t-----Histograma do alvo-----\n")
grafico = dados_treinamento['inadimplente'].plot.hist(bins=30)
grafico.set(title='inadimplente', xlabel='Quantidades', ylabel='Valores')
plt.show()

# ------------------------------------------------------------------------------
#  Melhor exibição das classes das features, pois os describe() não exibiu todas
# ------------------------------------------------------------------------------

print("\n\n\t-----Melhor exibição das classes das features-----\n")
for feature in list(dados_treinamento.columns):
    print(f"\nClasses {feature}: ", list(dados_treinamento[feature].unique()))

# ------------------------------------------------------------------------------
#  Remoção das features inúteis, notificadas no arquivo 'dicionario_de_dados'
#  grau_instrucao: Totalmente preenchida com zeros
#  possui_telefone_celular: Totalmente preenchida com "N"
#  qtde_contas_bancarias_especiais: Conteúdo idêntico à "qtde_contas_bancarias"
#  meses_no_trabalho: Vasta maioria com valor zero.
# ------------------------------------------------------------------------------

features_inuteis = ["grau_instrucao", "possui_telefone_celular", "qtde_contas_bancarias_especiais", "meses_no_trabalho"]
dados_treinamento.drop(features_inuteis, axis=1, inplace=True)
dados_teste.drop(features_inuteis, axis=1, inplace=True)

# ------------------------------------------------------------------------------
#  Remoção de features que possuíam uma classe extremamente dominante
#  nacionalidade: Vasta maioria com valor um (1: 19152)
#  valor_patrimonio_pessoal: Vasta maioria com valor zero (0: 19072)
#  grau_instrucao_companheiro: Vasta maioria com valor zero (0: 19345)
#  tipo_endereco: Vasta maioria com valor um (1: 19873)
#  possui_cartao_diners: Vasta maioria com valor zero (0: 19968)
#  possui_cartao_amex: Vasta maioria com valor zero (0: 19959)
#  possui_outros_cartoes: Vasta maioria com valor zero (0: 19955)
# ------------------------------------------------------------------------------

features_classes_dominantes = ["nacionalidade", "valor_patrimonio_pessoal", "grau_instrucao_companheiro", "tipo_endereco",
                               'possui_cartao_diners', 'possui_cartao_amex', 'possui_outros_cartoes']

dados_treinamento.drop(features_classes_dominantes, axis=1, inplace=True)
dados_teste.drop(features_classes_dominantes, axis=1, inplace=True)

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
substituir_valores_da_classe(dados_teste, features_com_classes_y_n, 'Y', 1)
substituir_valores_da_classe(dados_teste, features_com_classes_y_n, 'N', 0)

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
        substituir_valores_da_classe(dados_teste, features_siglas_estados_brasileiros, classe, regiao)

# ------------------------------------------------------------------------------
#  Substituindo os espaços ausentes das features, que estavam incompletas e que o
#  significado de suas classes não foi especificado, pela mediana dos valores de
#  suas classes, pois os valores não estão uniformemente distribuídos.
# ------------------------------------------------------------------------------

features_incompletas = ['tipo_residencia', 'meses_na_residencia', 'profissao', 'ocupacao', 'profissao_companheiro', 'sexo',
                        'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho',
                        'estado_onde_trabalha', 'estado_onde_nasceu', 'estado_onde_reside']

for feature in features_incompletas:

    # Substituindo espaços ' ' por 'N' (não informado).
    if feature == 'sexo':
        substituir_valores_da_classe(dados_treinamento, [feature], ' ', 'N')
        substituir_valores_da_classe(dados_teste, [feature], ' ', 'N')

    # Substituindo os espaços ' ' por "classe_invalida".
    elif feature in features_siglas_estados_brasileiros:
        substituir_valores_da_classe(dados_treinamento, features_siglas_estados_brasileiros, ' ', 'classe_invalida')
        substituir_valores_da_classe(dados_teste, features_siglas_estados_brasileiros, ' ', 'classe_invalida')

    # Substituindo os espaços ' ' pela média dos valores (após transformá-los em números).
    elif feature in ['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho']:

        # Primeiramente, substituímos os espaços vazios por None, a fim de realizar a conversão da coluna.
        substituir_valores_da_classe(dados_treinamento, [feature], ' ', None)
        substituir_valores_da_classe(dados_teste, [feature], ' ', None)

        # Converte os valores str da coluna em numéricos. Coerce substitui strings inválidas por NaN.
        dados_treinamento[feature] = pd.to_numeric(dados_treinamento[feature], errors='coerce')
        dados_teste[feature] = pd.to_numeric(dados_teste[feature], errors='coerce')

        # Substituindo os valores NaN pela mediana.
        mediana_feature_treinamento = dados_treinamento[feature].median()
        mediana_feature_teste = dados_teste[feature].median()
        dados_treinamento[feature] = dados_treinamento[feature].fillna(mediana_feature_treinamento)
        dados_teste[feature] = dados_teste[feature].fillna(mediana_feature_teste)

    else:
        mediana_feature_treinamento = dados_treinamento[feature].median()
        mediana_feature_teste = dados_teste[feature].median()
        dados_treinamento[feature] = dados_treinamento[feature].fillna(mediana_feature_treinamento)
        dados_teste[feature] = dados_teste[feature].fillna(mediana_feature_teste)

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

features_categoricas = ['produto_solicitado', 'forma_envio_solicitacao', 'estado_civil', 'estado_onde_nasceu',
                        'estado_onde_reside', 'tipo_residencia', 'estado_onde_trabalha', 'profissao', 'ocupacao',
                        'profissao_companheiro', 'sexo', 'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho']

calcular_taxa_de_inadimplencia_das_classes(dados_treinamento, features_categoricas)

# ------------------------------------------------------------------------------
# Criação de uma função para dividir as classes numéricas de uma ou mais features
# em 4 partes, através dos quartis.
# ------------------------------------------------------------------------------

def dividir_classes_por_quartis(data, features_list):
    "Função que divide as classes numéricas de uma ou mais features em 4 partes, através dos quartis."

    for feature in features_list:
        # Garantir que a coluna seja numérica
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

        # Obter os quartis e estatísticas descritivas
        descricao_estatistica_feature = dict(data[feature].describe())
        minimo = descricao_estatistica_feature['min']
        primeiro_quartil = descricao_estatistica_feature['25%']
        mediana = descricao_estatistica_feature['50%']
        terceiro_quartil = descricao_estatistica_feature['75%']
        maximo = descricao_estatistica_feature['max']

        # Função interna para classificar os valores.
        def classificar_por_quartil(classe):

            # Ignorar valores NaN e ' '.
            if pd.isna(classe) or (classe == ' '):
                return classe

            elif minimo <= classe <= primeiro_quartil:
                return 'primeira_particao'

            elif primeiro_quartil < classe <= mediana:
                return 'segunda_particao'

            elif mediana < classe <= terceiro_quartil:
                return 'terceira_particao'

            elif terceiro_quartil < classe <= maximo:
                return 'quarta_particao'

        # Aplicar a classificação à cada classe da coluna.
        data[feature] = data[feature].apply(classificar_por_quartil)

# ------------------------------------------------------------------------------
# Aplicação da função 'dividir_classes_por_quartis' em features que possuem
# números como classe, onde o significado desses números não foi informado
# ------------------------------------------------------------------------------

features_significado_nao_informado = ['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho', 'estado_civil',
                                      'tipo_residencia', 'profissao', 'ocupacao', 'profissao_companheiro']

dividir_classes_por_quartis(dados_treinamento, features_significado_nao_informado)
dividir_classes_por_quartis(dados_teste, features_significado_nao_informado)

# ------------------------------------------------------------------------------
# Criação e implementação de uma função para aplicar a classe OneHotEncoder em
# colunas categóricas, mantendo as demais inalteradas.
# ------------------------------------------------------------------------------

def aplicar_one_hot_encoder(data, features, data_type='training', target='target'):
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
        raise ValueError(f'Data type "{data_type}" is not supported')

# Implementação da função.
dados_treinamento = aplicar_one_hot_encoder(dados_treinamento, features_categoricas, 'training', 'inadimplente')
dados_teste = aplicar_one_hot_encoder(dados_teste, features_categoricas, 'test')

#------------------------------------------------------------------------------
# Removendo as features dos estados inválidos.
#------------------------------------------------------------------------------

estados_invalidos = ['estado_onde_nasceu_classe_invalida', 'estado_onde_trabalha_classe_invalida']
dados_treinamento.drop(estados_invalidos, axis=1, inplace=True)
dados_teste.drop(estados_invalidos, axis=1, inplace=True)

#------------------------------------------------------------------------------
# Remoção de features que possuíam uma classe extremamente dominante, após a
# implementação da função 'aplicar_one_hot_encoder'.
# sexo_N: Vasta maioria com valor zero (0: 19968)
#------------------------------------------------------------------------------

dados_treinamento.drop('sexo_N', axis=1, inplace=True)
dados_teste.drop('sexo_N', axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Exibindo os histogramas entre as quantidades e os valores de cada feature
# ------------------------------------------------------------------------------

for feature in list(dados_treinamento.columns):
    print(f"\n\n\t-----Histograma da feature {feature}-----\n")
    grafico = dados_treinamento[feature].plot.hist(bins=100)
    grafico.set(title=feature, xlabel='Valores', ylabel='Quantidades')
    plt.show()

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

# # Separando as features do alvo.
# X_treino = dados_treinamento_embaralhados.iloc[:, :-1].values
# y_treino = dados_treinamento_embaralhados.iloc[:, -1].values
#
# # Conjunto de teste
# X_teste = dados_teste_embaralhados.iloc[:, :].values

# print("\n\n\t-----Dimensões-----")
# print(f"\nDimensão X treino: {X_treino.shape}")
# print(f"Dimensão y treino: {y_treino.shape}")
# print(f"Dimensão X teste: {X_teste.shape}")
# print(f"Dimensão y teste: {y_teste.shape}\n")

#------------------------------------------------------------------------------
# Exibindo os coeficientes de Pearson de cada atributo (entre o mesmo e o alvo)
#------------------------------------------------------------------------------

print("\n\n\t-----Coeficiente de Pearson-----\n")
for coluna in dados_treinamento.columns:
    coef_pearsonr = pearsonr(dados_treinamento[coluna], dados_treinamento['inadimplente'])[0]
    p_value = pearsonr(dados_treinamento[coluna], dados_treinamento['inadimplente'])[1]
    print(f'{coluna}: {coef_pearsonr:.3f}, p-value: {p_value:.3f}\n')

# ------------------------------------------------------------------------------
# Aplicação da escala no X de treino e de teste
# ------------------------------------------------------------------------------

# escala = MinMaxScaler()
escala = StandardScaler()

escala.fit(X_treino)
X_treino_com_escala = escala.fit_transform(X_treino)
X_teste_com_escala = escala.transform(X_teste)

# X_treino_com_escala = escala.fit_transform(X_treino.astype(np.float64))

# ------------------------------------------------------------------------------
# Treinando o modelo KNeighborsClassifier, com k variando entre 1 e 30
# ------------------------------------------------------------------------------

print("\n\n\t-----Classificador com KNN-----\n")
for k in range(1, 31):

    # Instanciando o classificador KNN.
    classificador_knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    classificador_knn = classificador_knn.fit(X_treino_com_escala, y_treino)

    y_resposta_treino = classificador_knn.predict(X_treino_com_escala)
    y_resposta_teste = classificador_knn.predict(X_teste_com_escala)

    acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
    acuracia_teste  = accuracy_score(y_teste, y_resposta_teste)

    print(f'\nK = {k}')
    print(f'Acurácia Treino: {(acuracia_treino*100):.4f}%')
    print(f'Taxa de Erro Treino: {((1-acuracia_treino)*100):.4f}%')
    print(f'Acurácia Teste: {(acuracia_teste*100):.4f}%')
    print(f'Taxa de Erro Teste: {((1-acuracia_teste)*100):.4f}%')

    # # Realizando predições a partir da validação cruzada (4 folds).
    # y_resposta = cross_val_predict(classificador_knn, X_treino_com_escala, y_treino, cv=4)
    #
    # # Obtendo a acurácia de cada um dos 4 folds da validação cruzada.
    # acuracia_4_folds = cross_val_score(classificador_knn, X_treino_com_escala, y_treino, cv=4, scoring='accuracy')
    #
    # # Obtendo a média das acurácias.
    # media_acuracias = acuracia_4_folds.mean()
    #
    # # Obtendo a matriz de confusão.
    # matriz_confusao = confusion_matrix(y_treino, y_resposta)

    # #print(f'Predições: {y_resposta}')
    # #print(f'Acurácias: {acuracia_4_folds}')
    # print(f'Média das Acurácias: {(media_acuracias*100):.4f}%')
    # print(f'Taxa de Erro Médio: {((1-media_acuracias)*100):.4f}%')
    # #print(f'Matriz de Confusão: {matriz_confusao}')

# -------------------------------------------------------------------------------
# Treinando o modelo LogisticRegression, com penalidade L2
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador com Regressão Logística (Regularização L2)-----\n")
print("\n             C TREINO  TESTE")
print(" ------------- ------  -----")

# Para este laço, o melhor resultado foi em C=0.001000 (59.7% de acurácia).
# for c in [0.000001, 0.000010, 0.000100, 0.001, 0.010, 0.100,
#           1, 10, 100, 1000, 10000, 100000, 1000000]:

# Para este laço, o melhor resultado foi em C=0.002000 (59.8% de acurácia).
#for c in [0.000100, 0.000200, 0.000500, 0.001000, 0.002000, 0.005000, 0.010000]:

# Para este laço, os valores 0.001500, 0.002000 e 0.002200 possuem 59.8% de acurácia.
#for c in [0.001, 0.0012, 0.0015, 0.002, 0.0022, 0.0025, 0.003, 0.0035, 0.004, 0.0045,0.005]:

# Para este laço, como vários valores possuem 59.8% de acurácia, tomamos 0.002000 como C.
#for c in [0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.00205, 0.0021, 0.00215, 0.0022]:

c = 0.002000
classificador_lr = LogisticRegression(penalty='l2', C=c, max_iter=100000)
classificador_lr = classificador_lr.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_lr.predict(X_treino_com_escala)
y_resposta_teste = classificador_lr.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%14.6f" % c,
    "%6.1f" % (100 * acuracia_treino),
    "%6.1f" % (100 * acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o modelo LogisticRegressionCV, com penalidade L2 e validação cruzada
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador com Regressão Logística (Regularização L2 e Validação Cruzada)-----\n")
print("\n             C TREINO  TESTE")
print(" ------------- ------  -----")

classificador_lr_cv = LogisticRegressionCV(cv=4, max_iter=100000, penalty='l2',
                                     # Último laço da Regressão Logística sem validação cruzada.
                                     Cs=[0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.00205, 0.0021, 0.00215, 0.0022])

classificador_lr_cv = classificador_lr_cv.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_lr_cv.predict(X_treino_com_escala)
y_resposta_teste = classificador_lr_cv.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%14.6f" % c,
    "%6.1f" % (100 * acuracia_treino),
    "%6.1f" % (100 * acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Bayesiano Ingênuo
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Bayesiano Ingênuo-----\n")

print('\nBernoulliNB:\n')
classificador_bernoullinb = BernoulliNB()
classificador_bernoullinb = classificador_bernoullinb.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_bernoullinb.predict(X_treino_com_escala)
y_resposta_teste = classificador_bernoullinb.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(f'Acurácia Treino: {(acuracia_treino*100):.4f}%')
print(f'Taxa de Erro Treino: {((1-acuracia_treino)*100):.4f}%')
print(f'Acurácia Teste: {(acuracia_teste*100):.4f}%')
print(f'Taxa de Erro Teste: {((1-acuracia_teste)*100):.4f}%')


print('\nMultinomial:\n')
classificador_multinomial = MultinomialNB()
classificador_multinomial = classificador_multinomial.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_multinomial.predict(X_treino_com_escala)
y_resposta_teste = classificador_multinomial.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(f'Acurácia Treino: {(acuracia_treino*100):.4f}%')
print(f'Taxa de Erro Treino: {((1-acuracia_treino)*100):.4f}%')
print(f'Acurácia Teste: {(acuracia_teste*100):.4f}%')
print(f'Taxa de Erro Teste: {((1-acuracia_teste)*100):.4f}%')

print('\nGaussianNB:\n')
classificador_gaussiannb = GaussianNB()
classificador_gaussiannb = classificador_gaussiannb.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_gaussiannb.predict(X_treino_com_escala)
y_resposta_teste = classificador_gaussiannb.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(f'Acurácia Treino: {(acuracia_treino*100):.4f}%')
print(f'Taxa de Erro Treino: {((1-acuracia_treino)*100):.4f}%')
print(f'Acurácia Teste: {(acuracia_teste*100):.4f}%')
print(f'Taxa de Erro Teste: {((1-acuracia_teste)*100):.4f}%')

# -------------------------------------------------------------------------------
# Treinando o classificador Support Vector Machine Linear
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Support Vector Machine Linear-----\n")

print("\n           C  ACCTRE  ACCTES  ERRTRE  ERRTES")
print(" -----------  ------  ------  ------  ------")

# Para este laço, os melhores resultados foram em C=0.000100 e C=0.001000, ambos com 59.5% de acurácia.
# for C in [0.000001, 0.000010, 0.000100, 0.001000, 0.010000, 0.100000]:

# Para este laço, os melhores resultados foram em C=0.000050 e C=0.000200, ambos com 59.7% de acurácia.
#for C in [0.000010, 0.000020, 0.000050, 0.000100, 0.000200, 0.000500, 0.001000, 0.002000, 0.005000, 0.010000]:

# Para este laço, o melhor resultado foi em C=0.000050 com 59.7% de acurácia.
#for C in [0.000020, 0.000030, 0.000040, 0.000050, 0.000060, 0.000070, 0.000080, 0.000090, 0.000100]:

# Para este laço, os melhores resultados foram em C=0.000150, C=0.000175, C=0.000250 e C=0.000300, todos com 59.8% de acurácia.
#for C in [0.000100, 0.000150, 0.000175, 0.000200, 0.000225, 0.000250, 0.000300, 0.000350, 0.000400, 0.000450, 0.000500]:

# Para este laço, o melhor resultado foi em C=0.000160, com 59.84% de acurácia.
#for C in [0.000100, 0.000110, 0.000120, 0.000130, 0.000140, 0.000150, 0.000160, 0.000170, 0.000180, 0.000190, 0.000200]:

# Para este laço, os melhores resultados foram em C=0.000235 e C=0.000260, ambos com 59.86% de acurácia.
# for C in [0.000225, 0.000230, 0.000235, 0.000240, 0.000245, 0.000250, 0.000260, 0.000270, 0.000280, 0.000290, 0.000300,
#           0.000310, 0.000320, 0.000330, 0.000340, 0.000350]:

# Para este laço, podemos considerar que o melhor resultado foi em C=0.000235, com 59,86% de acurácia.
#for C in [0.000230, 0.000231, 0.000232, 0.000233, 0.000234, 0.000235, 0.000236, 0.000237, 0.000238, 0.000239, 0.000240]:

# Para este laço, podemos considerar que o melhor resultado foi em C=0.000260, com 59,86% de acurácia.
# for C in [0.000251, 0.000252, 0.000253, 0.000254, 0.000255, 0.000256, 0.000257, 0.000258, 0.000259, 0.000260,
#           0.000261, 0.000262, 0.000263, 0.000264, 0.000265, 0.000266, 0.000267, 0.000268, 0.000269, 0.000270]:

# Por fim, escolhemos C=0.000260.
C = 0.000260

classificador_linear_svc = LinearSVC(penalty='l2', C=C, max_iter=10000000, dual=True, random_state=11012005)
classificador_linear_svc = classificador_linear_svc.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_linear_svc.predict(X_treino_com_escala)
y_resposta_teste = classificador_linear_svc.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    # "%3d"%k,
    "%11.6f" % C,
    "%8.2f" % (100*acuracia_treino),
    "%7.2f" % (100*acuracia_teste),
    "%7.2f" % (100 - 100*acuracia_treino),
    "%7.2f" % (100 - 100*acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Support Vector Machine com Kernel
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Support Vector Machine com Kernel-----\n")
print("\n       C    GAMMA  TREINO  TESTE")
print(" -------  -------  ------  ------")

# g = 1/64
# for g in [0.0009, 0.0010, 0.0011, 0.0012, 0.0013, 0.0014]:
#     for c in [0.5, 1, 2, 5, 10]:

# Para este laço, o melhor resultado foi em c=100 e g=0.0001, com 60.24% de acurácia.
# for g in [0.000100, 0.001, 0.010, 0.100, 1, 10, 100, 1000, 10000, 100000, 1000000]:
#     for c in [0.010, 0.100, 1, 10, 100, 1000, 10000, 100000, 1000000]:

# Para este laço, o melhor resultado foi em c=100 e g=0.0001, com 60.24% de acurácia.
# for g in [0.000100, 0.001, 0.010, 0.100, 1, 10, 100, 1000, 10000, 100000, 1000000]:
#     for c in [50, 100, 200]:

# Para este laço, o melhor resultado foi em c=100 e g=0.0002, com 60.34% de acurácia.
# c = 100
# for g in [0.000020, 0.000050, 0.000100, 0.000200, 0.000500]:

# Por fim, escolhemos c = 100 e g = 0.0002.
c = 100
g = 0.0002

classificador_svc = SVC(kernel='rbf', C=c, gamma=g, max_iter=100000000)
classificador_svc = classificador_svc.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_svc.predict(X_treino_com_escala)
y_resposta_teste = classificador_svc.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%9.4f" % c,
    "%9.4f" % g,
    "%6.2f" % (100*acuracia_treino),
    "%6.2f" % (100*acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Árvore de Decisão, variando a profundidade
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Árvore de Decisão-----\n")

print("\n  D TREINO  TESTE")
print(" -- ------ ------")

# Para este laço, o melhor resultado foi em d=6 e criterion='gini', com 58.18% de acurácia.
#for d in range(2, 21):

d = 6
# criterion = 'gini', 'entropy' ou 'log_loss'
classificador_arvore_decisao = DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=11012005)
classificador_arvore_decisao = classificador_arvore_decisao.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_arvore_decisao.predict(X_treino_com_escala)
y_resposta_teste = classificador_arvore_decisao.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%3d" % d,
    "%6.2f" % (100*acuracia_treino),
    "%6.2f" % (100*acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Floresta Aleatória
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Floresta Aleatória-----\n")
print("\n  K   D TREINO  TESTE   ERRO  oob_score")
print("  -- -- ------ ------ ------ -----")

#Para este laço, o melhor resultado foi em k=90 e d=10, com 59.76% de acurácia.
# d = 10
# for k in range(5, 201, 5):

# Para este laço, o melhor resultado foi em k=189 e d=10, com 58.69% de acurácia.
#for k in [185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]:

# Para este laço, o melhor resultado foi em k=189 e d=12, com 60.48% de acurácia.
# k = 189
# for d in range(2, 21):

k = 189
d = 12

classificador_floresta_aleatoria = RandomForestClassifier(
    n_estimators=k,
    max_features='sqrt',
    oob_score=True,
    max_depth=d,
    random_state=11012005
)

classificador_floresta_aleatoria = classificador_floresta_aleatoria.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_floresta_aleatoria.predict(X_treino_com_escala)
y_resposta_teste = classificador_floresta_aleatoria.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%3d" % k,
    "%3d" % d,
    "%6.2f" % (100*acuracia_treino),
    "%6.2f" % (100*acuracia_teste),
    "%6.2f" % (100*(1-acuracia_teste)),
    "%6.2f" % (100*classificador_floresta_aleatoria.oob_score_)
)