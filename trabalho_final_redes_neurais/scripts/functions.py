#==============================================================================
# Arquivo que contém as principais funções do arquivo principal
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#------------------------------------------------------------------------------

def media_ponderada_notas(linha):
    """
    Calcula a média ponderada das avaliações de 1★, 3★ e 5★, com base no 
    total de avaliações.
    """
    return (linha['Lowest★'] * 1 + linha['Medium★★★'] * 3 + linha['Highest★★★★★'] * 5) / max(linha['Total_ratings'], 1)

#------------------------------------------------------------------------------

def calcular_metricas_agrupadas(linha, coluna_alvo, status_df):
    """
    Transforma listas agrupadas (como 'Ação, Comédia' em gêneros) em contagens e médias úteis para análise.

    Parâmetros:
        linha (pd.Series): Linha do DataFrame sendo processada.
        coluna_alvo (str): Nome da coluna a ser analisada (ex: 'Director', 'Genre').
        status_df (pd.DataFrame): DataFrame com estatísticas pré-calculadas.

    Retorna:
        pd.Series: Colunas com as métricas calculadas.
    """

    valores_na_linha = linha[coluna_alvo]

    # Caso 1: Valor único
    if len(valores_na_linha) == 1:
        valor = valores_na_linha[0]
        stats = status_df[status_df[coluna_alvo] == valor].iloc[0]
        return pd.Series({
            f'{coluna_alvo}_count': stats[f'qtd_filmes_{coluna_alvo.lower()}'],
            f'{coluna_alvo}_mean_rating': stats[f'media_notas_{coluna_alvo.lower()}']
        })

    # Caso 2: Valores únicos (todos com count == 1)
    elif all(status_df[status_df[coluna_alvo] == v][f'qtd_filmes_{coluna_alvo.lower()}'].iloc[0] == 1
             for v in valores_na_linha):
        return pd.Series({
            f'{coluna_alvo}_count': 1.0,
            f'{coluna_alvo}_mean_rating': linha['Average_rating']
        })

    # Caso 3: Valores com histórico (calcula médias)
    else:
        stats = status_df[status_df[coluna_alvo].isin(valores_na_linha)]
        return pd.Series({
            f'{coluna_alvo}_count': stats[f'qtd_filmes_{coluna_alvo.lower()}'].mean(),
            f'{coluna_alvo}_mean_rating': stats[f'media_notas_{coluna_alvo.lower()}'].mean()
        })

# ------------------------------------------------------------------------------