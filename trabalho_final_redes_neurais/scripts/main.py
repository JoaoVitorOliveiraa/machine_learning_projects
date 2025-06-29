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
from functions import media_ponderada_notas, calcular_metricas_agrupadas

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
# Substituindo a coluna "Director" por três novas colunas:
# - A quantidade de filmes de cada diretor;
# - A média ponderada das avaliações recebidas pelos filmes dos diretores;
# - A razão entre essa média ponderada e a quantidade de filmes (métrica ajustada por volume).
# ------------------------------------------------------------------------------

# Padronização das strings dos diretores, removendo espaços antes da vírgula e separando em listas.
dados['Director'] = dados['Director'].str.replace(r'\s*,\s*', ', ', regex=True).str.split(', ')

# Cria um DataFrame onde há uma linha para cada par de string de diretores, ou seja, um diretor por linha.
dados_explodidos = dados.explode('Director')

# Cria uma coluna com a média ponderada das avaliações das 3 eestrelas pelo total de avaliações.
dados_explodidos['media_ponderada_notas'] = dados_explodidos.apply(media_ponderada_notas, axis=1)

# Calcula as estatísticas por diretor.
status_diretor = dados_explodidos.groupby('Director')['media_ponderada_notas'].agg(
    qtd_filmes_director=('count'),
    media_ponderada_director=('mean')
).reset_index()

# Para diretores (usando o mesmo status_diretor original)
dados[['director_count', 'director_mean_rating', 'director_mean_per_qtd']] = dados.apply(
    lambda linha: calcular_metricas_agrupadas(linha, 'Director', status_diretor), 
    axis=1
)

# Removendo a coluna com os nomes dos diretores.
dados = dados.drop(columns=['Director'])









