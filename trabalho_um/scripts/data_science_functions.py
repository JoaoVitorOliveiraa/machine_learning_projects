#==============================================================================
# Arquivo que contém funções úteis para aplicações de Ciência de Dados
#==============================================================================

#------------------------------------------------------------------------------
# Importações das bibliotecas
#------------------------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#------------------------------------------------------------------------------
# Declarações das funções
#------------------------------------------------------------------------------

def show_data_head(lines_number, data, columns=False):
    "Função que exibe as primeiras 'lines_number' linhas do DataFrame 'data'."

    if columns:
        for column in columns:
            print(f'\n{data[column].head(n=lines_number)}\n')

    else:
        print(f'\n{data.head(n=lines_number)}\n')


def show_data_tail(lines_number, data, columns=False):
    "Função que exibe as últimas 'lines_number' linhas do DataFrame 'data'."

    if columns:
        for column in columns:
            print(f'\n{data[column].tail(n=lines_number)}\n')

    else:
        print(f'\n{data.tail(n=lines_number)}\n')


def show_data_info(data, columns=False):
    "Função que exibe informações sobre o DataFrame 'data', incluindo o tipo de índice e colunas, valores não nulos e uso de memória."

    if columns:
        for column in columns:
            print(f'\n{data[column].info()}\n')

    else:
        print(f'\n{data.info()}\n')


def show_data_value_counts(data, columns=False):
    "Função que exibe a frequência das classes de cada coluna no DataFrame 'data'."

    if columns:
        for column in columns:
            print(f'\n{data[column].value_counts()}\n')

    else:
        for column in data.columns:
            print(f'\n{data[column].value_counts()}\n')


def show_data_description(data, columns=False):
    "Função que exibe uma descrição estatística do DataFrame 'data', incluindo os três quartis, média e desvio padrão."

    if columns:
        for column in columns:
            print(f'\n{data[column].describe()}\n')

    else:
        print(f'\n{data.describe()}\n')


def return_data_columns(data):
    "Função que retorna e exibe as colunas do DataFrame 'data'."

    print(list(data.columns))
    return list(data.columns)


def show_data_columns_types(data, columns=False):
    "Função que exibe os tipos de cada coluna do DataFrame 'data'."

    if columns:
        for column in columns:
            print(f'\n{data[column].dtypes}\n')

    else:
        print(f'\n{data.dtypes}\n')


def show_colums_classes(data, columns=False):
    "Função que exibe as classes de cada coluna do DataFrame 'data'."

    if columns:
        for column in columns:
            print(f"\nColumn {column}: ", data[column].unique().tolist())

    else:
        for column in list(data.columns):
            print(f"\nColumn {column}: ", data[column].unique().tolist())


def drop_data_columns(data, columns):
    "Função que exclui colunas do DataFrame 'data'."

    data.drop(columns, axis=1, inplace=True)


def replace_class_value(data, features, old_value, new_value):
    "Função que substitui o valor de uma classe em uma feature."

    for feature in features:
        data[feature] = data[feature].replace({old_value: new_value})


def calculate_classes_target_rate(data, features, target='inadimplente'):
    "Função que calcula a taxa do alvo de cada classe das features categóricas."

    for feature in features:
        print(f"\n\n\t-----Taxa de inadimplência para as categorias da feature '{feature}'-----\n")
        dicionario_feature = dict(data[feature].value_counts())
        for categoria, quantidade in dicionario_feature.items():
            quantidade_inadimplentes = data[data[feature] == categoria][target].sum()
            taxa_inadimplencia = (quantidade_inadimplentes / quantidade) * 100
            print(f"Categoria: {categoria}")
            print(f"Quantidade Total: {quantidade}")
            print(f"Quantidade Inadimplentes: {quantidade_inadimplentes}")
            print(f"Taxa de Inadimplência: {taxa_inadimplencia:.3f}%\n")


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