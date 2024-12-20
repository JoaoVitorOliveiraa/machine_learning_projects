#==============================================================================
# Arquivo que contém funções úteis para aplicações de Ciência de Dados
#==============================================================================

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