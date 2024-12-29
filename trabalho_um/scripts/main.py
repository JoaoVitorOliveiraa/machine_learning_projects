#==============================================================================
# Trabalho 1 - Sistema de apoio à decisão para aprovação de crédito
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Importar os conjuntos de teste e treinamento (retirando as colunas dos id's)
# ------------------------------------------------------------------------------

caminho_conjunto_de_teste = Path('../data') / 'conjunto_de_teste.csv'
caminho_conjunto_de_treinamento = Path('../data') / 'conjunto_de_treinamento.csv'
dados_teste = pd.read_csv(caminho_conjunto_de_teste)
dados_treinamento = pd.read_csv(caminho_conjunto_de_treinamento)
ids_solicitantes_dados_teste = dados_teste['id_solicitante']
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

print("\n\n\t-----Histograma do alvo-----\n")
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
#  Criação de uma função para substituir o valor de uma classe em uma feature
# ------------------------------------------------------------------------------

def replace_class_value(data, features, old_value, new_value):
    "Função que substitui o valor de uma classe em uma feature."

    for feature in features:
        data[feature] = data[feature].replace({old_value: new_value})

# ------------------------------------------------------------------------------
#  Aplicação da função de substituir valores de classes
# ------------------------------------------------------------------------------

# Substituindo as classes binárias 'N' e 'Y' em 0 e 1 de cada feature categórica que apresenta essas classes.
features_com_classes_y_n = ['possui_telefone_residencial', 'vinculo_formal_com_empresa', 'possui_telefone_trabalho']
replace_class_value(dados_treinamento, features_com_classes_y_n, 'Y', 1)
replace_class_value(dados_treinamento, features_com_classes_y_n, 'N', 0)
replace_class_value(dados_teste, features_com_classes_y_n, 'Y', 1)
replace_class_value(dados_teste, features_com_classes_y_n, 'N', 0)

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
        replace_class_value(dados_treinamento, features_siglas_estados_brasileiros, classe, regiao)
        replace_class_value(dados_teste, features_siglas_estados_brasileiros, classe, regiao)

# ------------------------------------------------------------------------------
#  Substituindo os espaços ausentes das features, que estavam incompletas e que o
#  significado de suas classes não foi especificado, pela mediana dos valores de
#  suas classes, pois os valores não estão uniformemente distribuídos.
# ------------------------------------------------------------------------------

features_incompletas = ['tipo_residencia', 'meses_na_residencia', 'profissao', 'ocupacao', 'profissao_companheiro','sexo',
                        'codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho',
                        'estado_onde_trabalha', 'estado_onde_nasceu', 'estado_onde_reside']

for feature in features_incompletas:

    # Substituindo espaços ' ' por 'N' (não informado).
    if feature == 'sexo':
        replace_class_value(dados_treinamento, [feature], ' ', 'N')
        replace_class_value(dados_teste, [feature], ' ', 'N')

    # Substituindo os espaços ' ' por "classe_invalida".
    elif feature in features_siglas_estados_brasileiros:
        replace_class_value(dados_treinamento, [feature], ' ', 'classe_invalida')
        replace_class_value(dados_teste, [feature], ' ', 'classe_invalida')

    # Substituindo os espaços ' ' pela média dos valores (após transformá-los em números).
    elif feature in ['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho']:

        # Primeiramente, substituímos os espaços vazios por None, a fim de realizar a conversão da coluna.
        replace_class_value(dados_treinamento, [feature], ' ', None)
        replace_class_value(dados_teste, [feature], ' ', None)

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

def calculate_classes_target_rate(data, target, features=False):
    "Função que calcula a taxa do alvo de cada classe das features categóricas."

    if features:
        features_list = features

    else:
        features_list = list(data.columns)

    for feature in features_list:
        print(f"\n\n\t-----Taxa de '{target}' para as categorias da feature '{feature}'-----\n")
        dicionario_feature = dict(data[feature].value_counts())
        for categoria, quantidade in dicionario_feature.items():
            quantidade_target = data[data[feature] == categoria][target].sum()
            taxa_target = (quantidade_target / quantidade) * 100
            print(f"Categoria: {categoria}")
            print(f"Quantidade Total: {quantidade}")
            print(f"Quantidade {target.title()}: {quantidade_target}")
            print(f"Taxa de {target.title()}: {taxa_target:.3f}%\n")

# ------------------------------------------------------------------------------
#  Calculando a taxa de inadimplência das classes de cada feature categórica
#  Obs: Como foi observado que nas features que apresentavam apenas duas classes,
#  a taxa de inadimplência era de 49%-51% para cada, só foram adicionadas à lista
#  abaixo as features que possuíam mais de duas classes
# ------------------------------------------------------------------------------

features_categoricas = ['produto_solicitado', 'forma_envio_solicitacao', 'estado_civil', 'estado_onde_nasceu',
                        'estado_onde_reside', 'tipo_residencia', 'estado_onde_trabalha', 'profissao', 'ocupacao',
                        'profissao_companheiro', 'sexo', 'codigo_area_telefone_residencial',
                        'codigo_area_telefone_trabalho']

calculate_classes_target_rate(dados_treinamento, "inadimplente", features_categoricas)

# ------------------------------------------------------------------------------
# Criação de uma função para dividir as classes numéricas de uma ou mais features
# em 4 partes, através dos quartis.
# ------------------------------------------------------------------------------

def divide_classes_by_quartiles(data, features):
    "Função que divide as classes numéricas de uma ou mais features em 4 partes, através dos quartis."

    for feature in features:
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
        def classify_by_quartiles(classe):

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
        data[feature] = data[feature].apply(classify_by_quartiles)

# ------------------------------------------------------------------------------
# Aplicação da função 'divide_classes_by_quartiles' em features que possuem
# números como classe, onde o significado desses números não foi informado
# ------------------------------------------------------------------------------

features_significado_nao_informado = ['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho', 'estado_civil',
                                      'tipo_residencia', 'profissao', 'ocupacao', 'profissao_companheiro']

divide_classes_by_quartiles(dados_treinamento, features_significado_nao_informado)
divide_classes_by_quartiles(dados_teste, features_significado_nao_informado)

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
        data_features[feature] = data_features[feature].replace({' ': '_'})

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


# Implementação da função.
dados_treinamento = apply_one_hot_encoder(dados_treinamento, features_categoricas, 'inadimplente')
dados_teste = apply_one_hot_encoder(dados_teste, features_categoricas)

# ------------------------------------------------------------------------------
# Removendo as features dos estados inválidos.
# ------------------------------------------------------------------------------

estados_invalidos = ['estado_onde_nasceu_classe_invalida', 'estado_onde_trabalha_classe_invalida']
dados_treinamento.drop(estados_invalidos, axis=1, inplace=True)
dados_teste.drop(estados_invalidos, axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Exibindo os coeficientes de Pearson de cada atributo (entre o mesmo e o alvo)
# ------------------------------------------------------------------------------

print("\n\n\t-----Coeficiente de Pearson-----\n")
for coluna in dados_treinamento.columns:
    coef_pearsonr = pearsonr(dados_treinamento[coluna], dados_treinamento['inadimplente'])[0]
    p_value = pearsonr(dados_treinamento[coluna], dados_treinamento['inadimplente'])[1]
    print(f'{coluna}: {coef_pearsonr:.3f}, p-value: {p_value:.3f}\n')

# ------------------------------------------------------------------------------
# Remoção de features que possuíam um coeficiente de Pearson menor que 0.01
# ------------------------------------------------------------------------------

drop_list_pearson = ['codigo_area_telefone_residencial_terceira_particao', 'ocupacao_quarta_particao',
              'estado_onde_trabalha_regiao_sul', 'estado_onde_trabalha_regiao_sudeste', 'estado_onde_trabalha_regiao_norte',
              'estado_onde_reside_regiao_sudeste', 'estado_onde_nasceu_regiao_sudeste', 'estado_onde_nasceu_regiao_norte',
              'forma_envio_solicitacao_internet', 'produto_solicitado_2', 'vinculo_formal_com_empresa', 'possui_cartao_visa',
              'renda_extra', 'renda_mensal_regular', 'possui_email', 'sexo_N']

dados_treinamento.drop(drop_list_pearson, axis=1, inplace=True)
dados_teste.drop(drop_list_pearson, axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Exibindo os histogramas entre as quantidades e valores de cada feature
# ------------------------------------------------------------------------------

for feature in list(dados_treinamento.columns):
    print(f"\n\n\t-----Histograma da feature {feature}-----\n")
    grafico = dados_treinamento[feature].plot.hist(bins=100)
    grafico.set(title=feature, xlabel='Valores', ylabel='Quantidades')
    plt.show()

# ------------------------------------------------------------------------------
# Embaralhar o conjunto de dados de treino para garantir que a divisão entre os
# dados esteja isenta de qualquer viés de seleção
# ------------------------------------------------------------------------------

dados_treinamento_embaralhados = dados_treinamento.sample(frac=1, random_state=11012005)

# ------------------------------------------------------------------------------
# Separar o conjunto de treinamento em arrays X e Y
# ------------------------------------------------------------------------------

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

# Neste caso, os melhores resultados foram encontrados com o StandardScaler()
# escala = MinMaxScaler()
escala = StandardScaler()

escala.fit(X_treino)
X_treino_com_escala = escala.transform(X_treino)
X_teste_com_escala = escala.transform(X_teste)

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
    acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

    print(f'\nK = {k}')
    print(f'Acurácia Treino: {(acuracia_treino * 100):.4f}%')
    print(f'Taxa de Erro Treino: {((1 - acuracia_treino) * 100):.4f}%')
    print(f'Acurácia Teste: {(acuracia_teste * 100):.4f}%')
    print(f'Taxa de Erro Teste: {((1 - acuracia_teste) * 100):.4f}%')

# -------------------------------------------------------------------------------
# Treinando o modelo LogisticRegression, com penalidade L2
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador com Regressão Logística (Regularização L2)-----\n")
print("\n             C TREINO  TESTE")
print(" ------------- ------  -----")

# Para este laço, o melhor resultado foi em C=0.010000 (59.9% de acurácia).
# for c in [0.000001, 0.000010, 0.000100, 0.001, 0.010, 0.100,
          # 1, 10, 100, 1000, 10000, 100000, 1000000]:

# Para este laço, os melhores resultados foram em C=0.002000/acurácia=60.0% e em C=0.005000/acurácia=60.0%.
# for c in [0.00100, 0.00200, 0.00500, 0.01000, 0.02000, 0.05000, 0.10000]:

# Para este laço, o melhor resultado foi em C=0.002500 (60.1% de acurácia)
# for c in [0.00100, 0.00150, 0.00180, 0.00200, 0.00250, 0.00300, 0.00350, 0.00400, 0.00450, 0.00500, 0.00550, 0.00600, 0.00700, 0.00800, 0.010000]:

# Por fim, decidimos que o melhor valor de c é 0.002500.
c = 0.002500

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
                                           Cs=[0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.00205, 0.0021, 0.00215,
                                               0.0022, 0.0025])

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

print(f'Acurácia Treino: {(acuracia_treino * 100):.4f}%')
print(f'Taxa de Erro Treino: {((1 - acuracia_treino) * 100):.4f}%')
print(f'Acurácia Teste: {(acuracia_teste * 100):.4f}%')
print(f'Taxa de Erro Teste: {((1 - acuracia_teste) * 100):.4f}%')

# print('\nMultinomial:\n')
# classificador_multinomial = MultinomialNB()
# classificador_multinomial = classificador_multinomial.fit(X_treino_com_escala, y_treino)
#
# y_resposta_treino = classificador_multinomial.predict(X_treino_com_escala)
# y_resposta_teste = classificador_multinomial.predict(X_teste_com_escala)
#
# acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
# acuracia_teste = accuracy_score(y_teste, y_resposta_teste)
#
# print(f'Acurácia Treino: {(acuracia_treino*100):.4f}%')
# print(f'Taxa de Erro Treino: {((1-acuracia_treino)*100):.4f}%')
# print(f'Acurácia Teste: {(acuracia_teste*100):.4f}%')
# print(f'Taxa de Erro Teste: {((1-acuracia_teste)*100):.4f}%')

print('\nGaussianNB:\n')
classificador_gaussiannb = GaussianNB()
classificador_gaussiannb = classificador_gaussiannb.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_gaussiannb.predict(X_treino_com_escala)
y_resposta_teste = classificador_gaussiannb.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(f'Acurácia Treino: {(acuracia_treino * 100):.4f}%')
print(f'Taxa de Erro Treino: {((1 - acuracia_treino) * 100):.4f}%')
print(f'Acurácia Teste: {(acuracia_teste * 100):.4f}%')
print(f'Taxa de Erro Teste: {((1 - acuracia_teste) * 100):.4f}%')

# -------------------------------------------------------------------------------
# Treinando o classificador Support Vector Machine Linear
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Support Vector Machine Linear-----\n")

print("\n           C  ACCTRE  ACCTES  ERRTRE  ERRTES")
print(" -----------  ------  ------  ------  ------")

# Para este laço, os melhores resultados foram em C=0.001000 com 59.94% de acurácia.
# for C in [0.000001, 0.000010, 0.000100, 0.001000, 0.010000, 0.100000]:

# Para este laço, o melhor resultado foi em C=0.000900 com 60.04% de acurácia.
# for C in [0.00010, 0.00020, 0.00050, 0.00080, 0.00090, 0.00100, 0.00200,  0.00300, 0.00500,  0.00800, 0.001000]:

# Para este laço, o melhor resultado ainda foi em C=0.000900 com 60.04% de acurácia.
# for C in [0.00080, 0.00081, 0.00082, 0.00083, 0.00084, 0.00085, 0.00086, 0.00087, 0.00088, 0.00089, 0.00090,
# 0.00091, 0.00092, 0.00093, 0.00094, 0.00095, 0.00096, 0.00097, 0.00098, 0.00099, 0.00100]:

# Por fim, escolhemos C=0.000900.
C = 0.000900

classificador_linear_svc = LinearSVC(penalty='l2', C=C, max_iter=10000000, dual=True, random_state=11012005)
classificador_linear_svc = classificador_linear_svc.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_linear_svc.predict(X_treino_com_escala)
y_resposta_teste = classificador_linear_svc.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    # "%3d"%k,
    "%11.6f" % C,
    "%8.2f" % (100 * acuracia_treino),
    "%7.2f" % (100 * acuracia_teste),
    "%7.2f" % (100 - 100 * acuracia_treino),
    "%7.2f" % (100 - 100 * acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Support Vector Machine com Kernel
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Support Vector Machine com Kernel-----\n")
print("\n       C    GAMMA  TREINO  TESTE")
print(" -------  -------  ------  ------")

# Para este laço, o melhor resultado foi em c=10/g=0.0010/acurácia=60.32%.
# for g in [0.000100, 0.001, 0.010, 0.100, 1, 10, 100, 1000, 10000, 100000, 1000000]:
# for c in [0.010, 0.100, 1, 10, 100, 1000, 10000, 100000, 1000000]:

# Para este laço, o melhor resultado foi em c=4/acurácia=60.50%.
# g = 0.001
# for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

# Por fim, escolhemos c = 4 e g = 0.0010 --> acurácia = 60.50%
c = 4
g = 0.0010

classificador_svc = SVC(kernel='rbf', C=c, gamma=g, max_iter=100000000)
classificador_svc = classificador_svc.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_svc.predict(X_treino_com_escala)
y_resposta_teste = classificador_svc.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%9.4f" % c,
    "%9.4f" % g,
    "%6.2f" % (100 * acuracia_treino),
    "%6.2f" % (100 * acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Árvore de Decisão, variando a profundidade
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Árvore de Decisão-----\n")
print("\n  D TREINO  TESTE")
print(" -- ------ ------")

# Para este laço, o melhor resultado foi em d=5 e criterion='gini', com 57.98% de acurácia.
# for d in range(2, 21):

d = 5
# criterion = 'gini', 'entropy' ou 'log_loss'
classificador_arvore_decisao = DecisionTreeClassifier(criterion='gini', max_depth=d, random_state=11012005)
classificador_arvore_decisao = classificador_arvore_decisao.fit(X_treino_com_escala, y_treino)

y_resposta_treino = classificador_arvore_decisao.predict(X_treino_com_escala)
y_resposta_teste = classificador_arvore_decisao.predict(X_teste_com_escala)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%3d" % d,
    "%6.2f" % (100 * acuracia_treino),
    "%6.2f" % (100 * acuracia_teste)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Floresta Aleatória
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Floresta Aleatória-----\n")
print("\n  K   D TREINO  TESTE   ERRO  oob_score")
print("  -- -- ------ ------ ------ -----")

# Para este laço, o melhor resultado foi em k=65/acurácia=60.90%/max_features=9.
# d = 10
# for k in range(5, 501, 5):

# Para este laço, o melhor resultado foi em k=65/d=10/acurácia=60.90%/max_features=9.
# d = 10
# for k in [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:

# Para este laço, o melhor resultado ainda foi em k=65/d=10/acurácia=60.90%/max_features=9.
# k = 65
# for d in range(2, 21):

# Por fim, escolhemos k = 65, d = 10, max_features = 9 --> acurácia = 60.90%.
k = 65
d = 10

classificador_floresta_aleatoria = RandomForestClassifier(
    n_estimators=k,
    max_features=9,
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
    "%6.2f" % (100 * acuracia_treino),
    "%6.2f" % (100 * acuracia_teste),
    "%6.2f" % (100 * (1 - acuracia_teste)),
    "%6.2f" % (100 * classificador_floresta_aleatoria.oob_score_)
)

# -------------------------------------------------------------------------------
# Treinando o classificador Floresta Aleatória com Filtro de Feature Importance
# -------------------------------------------------------------------------------

print("\n\n\t-----Classificador Floresta Aleatória (Filtro de Feature Importance)-----\n")
print("\n  K   D TREINO  TESTE   ERRO")
print("  -- -- ------ ------ ------")

# Treinamento de um modelo inicial (tendo como base os valores anteriores de 'k' e 'd')
k = 65
d = 10
classificador_floresta_aleatoria = RandomForestClassifier(
    n_estimators=k,
    max_features=9,
    oob_score=True,
    max_depth=d,
    random_state=11012005
)
classificador_floresta_aleatoria = classificador_floresta_aleatoria.fit(X_treino_com_escala, y_treino)

# Filtrando as features, para que apenas as que possuem importância superior a 0.01 sejam mantidas.
X_treino_reduzido = X_treino_com_escala[:, classificador_floresta_aleatoria.feature_importances_ > 0.01]
X_teste_reduzido = X_teste_com_escala[:, classificador_floresta_aleatoria.feature_importances_ > 0.01]


# Para este laço, o melhor resultado foi em k=205/d=10/acurácia=61.10%/max_features=8.
# d = 10
# for k in range(5, 251, 5):

# Para este laço, o melhor resultado ainda foi em k=205/d=10/acurácia=61.10%/max_features=8.
# d = 10
# for k in [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210]:

# Para este laço, o melhor resultado ainda foi em k=205/d=10/acurácia=61.10%/max_features=8.
# k = 170
# for d in range(2, 21):

# Por fim, escolhemos k = 205, d = 10, max_features = 8 --> acurácia = 61.10%.
k = 205
d = 10

# Treinamento final do modelo.
classificador_floresta_aleatoria = RandomForestClassifier(
    n_estimators=k,
    max_features=8,
    oob_score=False,
    max_depth=d,
    random_state=11012005
)

classificador_floresta_aleatoria = classificador_floresta_aleatoria.fit(X_treino_reduzido, y_treino)

y_resposta_treino = classificador_floresta_aleatoria.predict(X_treino_reduzido)
y_resposta_teste = classificador_floresta_aleatoria.predict(X_teste_reduzido)

acuracia_treino = accuracy_score(y_treino, y_resposta_treino)
acuracia_teste = accuracy_score(y_teste, y_resposta_teste)

print(
    "%3d" % k,
    "%3d" % d,
    "%6.2f" % (100*acuracia_treino),
    "%6.2f" % (100*acuracia_teste),
    "%6.2f" % (100*(1-acuracia_teste))
)