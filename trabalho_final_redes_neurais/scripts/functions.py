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

    # Caso 1: Valor único.
    if len(valores_na_linha) == 1:
        valor = valores_na_linha[0]
        status = status_df[status_df[coluna_alvo] == valor].iloc[0]
        
        # Verifica se não está vazio.
        if not status.empty:
            return pd.Series({
                f'{coluna_alvo}_count': status[f'qtd_filmes_{coluna_alvo.lower()}'],
                f'{coluna_alvo}_mean_rating': status[f'media_ponderada_{coluna_alvo.lower()}'],
                f'{coluna_alvo}_mean_per_qtd': status[f'media_ponderada_{coluna_alvo.lower()}'] / status[f'qtd_filmes_{coluna_alvo.lower()}']
            })
        
        # Se estiver, retorna valores zerados.
        else:
            return pd.Series({f'{coluna_alvo}_count': 0.0, f'{coluna_alvo}_mean_rating': 0.0, f'{coluna_alvo}_mean_per_qtd': 0.0})

    # Caso 2: Valores únicos (todos com count == 1).
    elif all(status_df[status_df[coluna_alvo] == valor][f'qtd_filmes_{coluna_alvo.lower()}'].iloc[0] == 1
             for valor in valores_na_linha):
        
        # Neste caso, podemos tomar o primeiro valor. Pois as métricas de todos são iguais.
        valor = valores_na_linha[0]
        status = status_df[status_df[coluna_alvo] == valor].iloc[0]
        
        # Verifica se não está vazio.
        if not status.empty:
            return pd.Series({
                f'{coluna_alvo}_count': 1.0,
                f'{coluna_alvo}_mean_rating': status[f'media_ponderada_{coluna_alvo.lower()}'],
                f'{coluna_alvo}_mean_per_qtd': status[f'media_ponderada_{coluna_alvo.lower()}'] / 1.0
            })

        # Se estiver, retorna valores zerados.
        else:
            return pd.Series({f'{coluna_alvo}_count': 0.0, f'{coluna_alvo}_mean_rating': 0.0, f'{coluna_alvo}_mean_per_qtd': 0.0})

    # Caso 3: Quantidade de filmes diferentes entre os diretores.
    else:

        # Seleciona apenas as linhas dos valores da linha.        
        status = status_df[status_df[coluna_alvo].isin(valores_na_linha)]
    
        # Verifica se não está vazio.
        if not status.empty:
            quantidade_media_filmes = status[f'qtd_filmes_{coluna_alvo.lower()}'].mean()
            media_ponderada_media = status[f'media_ponderada_{coluna_alvo.lower()}'].mean()
            
            return pd.Series({
                f'{coluna_alvo}_count': quantidade_media_filmes,
                f'{coluna_alvo}_mean_rating': media_ponderada_media,
                f'{coluna_alvo}_mean_per_qtd': media_ponderada_media / quantidade_media_filmes
            })
         
        # Se estiver, retorna valores zerados.
        else: 
            return pd.Series({f'{coluna_alvo}_count': 0.0, f'{coluna_alvo}_mean_rating': 0.0, f'{coluna_alvo}_mean_per_qtd': 0.0})
 
# ------------------------------------------------------------------------------