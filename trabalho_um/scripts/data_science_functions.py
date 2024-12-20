#==============================================================================
# Arquivo que contém funções úteis para aplicações de Ciência de Dados
#==============================================================================

def show_data_head(lines_number, data, features=False):
    "Função que exibe as primeiras 'lines_number' linhas do DataFrame 'data'."

    if features:
        for feature in features:
            print(f'\n{data[feature].head(n=lines_number)}\n')

    else:
        print(f'\n{data.head(n=lines_number)}\n')


def show_data_tail(lines_number, data, features=False):
    "Função que exibe as últimas 'lines_number' linhas do DataFrame 'data'."

    if features:
        for feature in features:
            print(f'\n{data[feature].tail(n=lines_number)}\n')

    else:
        print(f'\n{data.tail(n=lines_number)}\n')


def show_data_info(data, features=False):
    "Função que exibe informações sobre o DataFrame 'data', incluindo o tipo de índice e colunas, valores não nulos e uso de memória."

    if features:
        for feature in features:
            print(f'\n{data[feature].info()}\n')

    else:
        print(f'\n{data.info()}\n')


def show_data_value_counts(data, features=False):
    "Função que exibe a frequência das classes de cada feature no DataFrame 'data'."

    if features:
        for feature in features:
            print(f'\n{data[feature].value_counts()}\n')

    else:
        for feature in data.columns:
            print(f'\n{data[feature].value_counts()}\n')


def show_data_description(data, features=False):
    "Função que exibe uma descrição estatística do DataFrame 'data', incluindo os três quartis, média e desvio padrão."

    if features:
        for feature in features:
            print(f'\n{data[feature].describe()}\n')

    else:
        print(f'\n{data.describe()}\n')


def return_data_features(data):
    "Função que retorna e exibe as features do DataFrame 'data'."

    print(list(data.columns))
    return list(data.columns)