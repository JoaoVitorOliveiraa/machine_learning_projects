#==============================================================================
# Arquivo que contém as principais funções do arquivo principal
#==============================================================================

#------------------------------------------------------------------------------
# Importar bibliotecas
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau, spearmanr

#------------------------------------------------------------------------------

def media_ponderada_notas(linha):
    """
    Calcula a média ponderada das avaliações de 1★, 3★ e 5★, com base no 
    total de avaliações.
    """
    return (linha['lowest'] * 1 + linha['medium'] * 3 + linha['highest'] * 5) / max(linha['Total_ratings'], 1)

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
                f'{coluna_alvo}_count': status['qtd_filmes'],
                f'{coluna_alvo}_mean_rating': status['media_ponderada'],
                f'{coluna_alvo}_mean_per_qtd': status['media_ponderada'] * np.log1p(status['qtd_filmes'])
            })
        
        # Se estiver, retorna valores zerados.
        else:
            return pd.Series({f'{coluna_alvo}_count': 0.0, f'{coluna_alvo}_mean_rating': 0.0, f'{coluna_alvo}_mean_per_qtd': 0.0})

    # Caso 2: Valores únicos (todos com count == 1).
    elif all(status_df[status_df[coluna_alvo] == valor]['qtd_filmes'].iloc[0] == 1
             for valor in valores_na_linha):
        
        # Neste caso, podemos tomar o primeiro valor. Pois as métricas de todos são iguais.
        valor = valores_na_linha[0]
        status = status_df[status_df[coluna_alvo] == valor].iloc[0]
        
        # Verifica se não está vazio.
        if not status.empty:
            return pd.Series({
                f'{coluna_alvo}_count': 1.0,
                f'{coluna_alvo}_mean_rating': status['media_ponderada'],
                f'{coluna_alvo}_mean_per_qtd': status['media_ponderada'] * np.log1p(1.0)
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
            quantidade_media_filmes = status['qtd_filmes'].mean()
            media_ponderada_media = status['media_ponderada'].mean()
            
            return pd.Series({
                f'{coluna_alvo}_count': quantidade_media_filmes,
                f'{coluna_alvo}_mean_rating': media_ponderada_media,
                f'{coluna_alvo}_mean_per_qtd': media_ponderada_media * np.log1p(quantidade_media_filmes)
            })
         
        # Se estiver, retorna valores zerados.
        else: 
            return pd.Series({f'{coluna_alvo}_count': 0.0, f'{coluna_alvo}_mean_rating': 0.0, f'{coluna_alvo}_mean_per_qtd': 0.0})
 
# ------------------------------------------------------------------------------
        
def show_correlations(data, target, columns=False):
    """
    Exibe coeficientes de Pearson e Kendall Tau com p-values
    entre as colunas fornecidas e o alvo 'target'.
    
    Parâmetros:
    - data: DataFrame com os dados.
    - target: string com o nome da coluna alvo.
    - columns: lista de colunas a analisar (opcional). Se False, usa todas.
    """
    
    if columns:
        columns_list = columns

    else:
        columns_list = list(data.columns)

    print(" ")
    print("COMPARAÇÃO DE CORRELAÇÃO COM O ALVO:", target)
    print(" ")
    print(f"{'Variável':<33} {'Pearson':<12} {'P-Value':<10} {'Kendall':<12} {'P-Value':<10} {'Spearman':<12} {'P-Value'}")
    print("-" * 102)

    for column in columns_list:
        coef_pearsonr, p_value_pearsonr = pearsonr(data[column], data[target])
        coef_kendalltau, p_value_kendalltau = kendalltau(data[column], data[target])
        coef_spearmanr, p_value_spearmanr = spearmanr(data[column], data[target])
        print(f"{column:<30} {coef_pearsonr:10.4f} {p_value_pearsonr:12.5f} {coef_kendalltau:10.4f} {p_value_kendalltau:12.5f} {coef_spearmanr:10.4f} {p_value_spearmanr:12.5f} ")
        
# ------------------------------------------------------------------------------
 
def exibir_histogramas(df, bins=30, n_colunas_grade=3, colunas=None):
    """
    Função que:
        -Exibe um histograma para cada coluna numérica de um DataFrame
        -Organiza todos os histogramas em uma única figura
        -Ajusta dinamicamente o layout com base no número de colunas
    """
    
    # Se o usuário não fornecer colunas, seleciona todas as colunas numéricas automaticamente.
    if colunas is None:
        colunas = df.select_dtypes(include=np.number).columns.tolist()

    n_colunas_plotadas = len(colunas)
    n_linhas_grade = int(np.ceil(n_colunas_plotadas / n_colunas_grade))
    plt.figure(figsize=(6 * n_colunas_grade, 4 * n_linhas_grade))

    for i, coluna in enumerate(colunas):
        ax = plt.subplot(n_linhas_grade, n_colunas_grade, i + 1)
        ax.hist(df[coluna].dropna(), bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(coluna)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frequência")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------

def show_correlation_matrix(data, method='spearman'):
    """
    Exibe a matriz de correlação entre as variáveis do conjunto de dados.
    """
    
    # Calculando matriz de correlação (Spearman por padrão).
    corr_matrix = data.corr(method=method)
    
    # Gerando o heatmap.
    plt.figure(figsize=(12, 12))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,            # Mostra valores nas células
        fmt=".2f",             # Formato com 2 casas decimais
        cmap='coolwarm',       # Mapa de cores (azul-quente)
        center=0,              # Centraliza o branco no 0
        vmin=-1,               # Valor mínimo da escala
        vmax=1,                # Valor máximo da escala
        linewidths=.5,         # Largura das linhas entre células
        square=True            # Mantém as células quadradas
    )
    
    # Ajustando título e labels.
    plt.title(f'Matriz de Correlação ({method.title()})', fontsize=14, pad=20)
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    
    # Mostrando o gráfico.
    plt.tight_layout()
    plt.show()








