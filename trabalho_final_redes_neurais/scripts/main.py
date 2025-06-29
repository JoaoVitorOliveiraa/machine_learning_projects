#==============================================================================
# Trabalho Final - Classificação de Filmes do Letterbox
#==============================================================================

#------------------------------------------------------------------------------
# Importando bibliotecas
#------------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------------------
# Importando os conjuntos de dados e retirando a colunas dos id's
# ------------------------------------------------------------------------------

caminho_conjunto_dados = Path('../data') / 'Letterbox-Movie-Classification-Dataset.csv'
dados = pd.read_csv(caminho_conjunto_dados)
dados = dados.iloc[:, 1:]

# ------------------------------------------------------------------------------
# Retirando as colunas dos títulos e descrições dos filmes
# ------------------------------------------------------------------------------

dados.drop(["Film_title", "Description"], axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Reduzindo a coluna 'Original_language' para '5 línguas mais comuns' + 'outros'
# ------------------------------------------------------------------------------

linguas_comuns = dados['Original_language'].value_counts().nlargest(5).index.tolist()
dados['Original_language'] = dados['Original_language'].apply(lambda x: x.lower() if x in linguas_comuns else 'others')

# ------------------------------------------------------------------------------
# Criando colunas: quantidade_filmes_diretor e media_notas_diretor/quatidade_filmes_diretor
# ------------------------------------------------------------------------------

# Padronização das strings dos diretores, removendo espaços antes da vírgula e separando em listas.
dados['Director'] = dados['Director'].str.replace(r'\s*,\s*', ', ', regex=True).str.split(', ')

# Cria um DataFrame onde há uma linha para cada par de string de diretores, ou seja, um diretor por linha.
dados_explodidos = dados.explode('Director')

# Calcula as estatísticas por diretor.
status_diretor = dados_explodidos.groupby('Director')['Average_rating'].agg(
    qtd_filmes_diretor=('count'),
    media_notas_diretor=('mean')
).reset_index()


def calcular_metricas(linha):
    "Função que calcula as métricas de cada diretor."
    
    # Armazena os diretores daquela linha.
    diretores_na_linha = linha['Director']
    
    # Caso haja apenas 1 diretor (usa os valores diretos de status_diretor).
    if len(diretores_na_linha) == 1:
        diretor = diretores_na_linha[0]
        status = status_diretor[status_diretor['Director'] == diretor].iloc[0]
        
        return pd.Series({
            'director_film_count': status['qtd_filmes_diretor'],
            'director_mean_rating': status['qtd_filmes_diretor'] / status['media_notas_diretor']  
        })
    
    # Caso haja uma dupla única (ambos com count == 1).
    elif all(status_diretor[status_diretor['Director'] == diretor]['qtd_filmes_diretor'].iloc[0] == 1 for diretor in diretores_na_linha):
        return pd.Series({
            'director_film_count': 1.0,
            'director_mean_rating': 1.0 / linha['Average_rating']
        })
    
    # Caso haja pelo menos um diretor com histórico (calcula médias).
    else:
        status = status_diretor[status_diretor['Director'].isin(diretores_na_linha)]
        media_qtd_filmes = status['qtd_filmes_diretor'].mean()
        media_notas = status['media_notas_diretor'].mean()
        
        return pd.Series({
            'director_film_count': media_qtd_filmes,
            'director_mean_rating': media_qtd_filmes / media_notas
        })


# Aplicação da função
dados[['director_film_count', 'director_mean_rating']] = dados.apply(calcular_metricas, axis=1)


# Removendo a coluna com os nomes dos diretores.
dados = dados.drop(columns=['Director'])




