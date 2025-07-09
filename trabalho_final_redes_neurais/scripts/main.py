#==============================================================================
# Trabalho Final - Classificação de Filmes do Letterbox
#==============================================================================

#------------------------------------------------------------------------------
# Importando bibliotecas
#------------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from functions import (media_ponderada_notas, calcular_metricas_agrupadas, 
                       show_correlations, exibir_histogramas, show_correlation_matrix)

# ------------------------------------------------------------------------------
# Importando os conjuntos de dados e retirando a colunas dos id's
# ------------------------------------------------------------------------------

caminho_conjunto_dados = Path('../data') / 'Letterbox-Movie-Classification-Dataset.csv'
dados = pd.read_csv(caminho_conjunto_dados)
dados = dados.iloc[:, 1:]

# ------------------------------------------------------------------------------
# Renomeação de colunas.
# ------------------------------------------------------------------------------

novas_colunas = {'Lowest★': 'lowest', 'Medium★★★': 'medium', 'Highest★★★★★': 'highest'}
dados = dados.rename(columns=novas_colunas)

# ------------------------------------------------------------------------------
# Correções de distribuições assimétricas com a Transformação de Yeo-Johnson
# ------------------------------------------------------------------------------

# Lista de colunas com assimetria/calda
colunas_com_assimetria = ['Watches', 'Likes', 'Fans', 'Total_ratings', 
                  'highest', 'medium', 'lowest', 'List_appearances']

# Aplicando a Transformação de Yeo-Johnson.
power_transformer = PowerTransformer(method='yeo-johnson')
dados[colunas_com_assimetria] = power_transformer.fit_transform(dados[colunas_com_assimetria])

# Alternativa: Aplicar log1p em cada coluna
#for coluna in colunas_com_assimetria:
#    dados[coluna] = np.log1p(dados[coluna])  # log(1 + x)

# ------------------------------------------------------------------------------
# Retirando as colunas dos títulos e descrições dos filmes
# ------------------------------------------------------------------------------

dados.drop(["Film_title", "Description"], axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Reduzindo a coluna 'Original_language' para 'english' e 'outros'
# ------------------------------------------------------------------------------

linguas_comuns = dados['Original_language'].value_counts().nlargest(1).index.tolist()
dados['Original_language'] = dados['Original_language'].apply(lambda x: x.lower() if x in linguas_comuns else 'others')

# ------------------------------------------------------------------------------
# Aplicando o OneHotEncoding na coluna 'Original_language' e deixando somente
# a coluna da língua inglesa.
# ------------------------------------------------------------------------------

dados =  pd.get_dummies(dados, columns=['Original_language'], dtype=int)
dados = dados.drop(columns=['Original_language_others'])  

# ------------------------------------------------------------------------------
# Substituindo as colunas 'Director', 'Genres' e 'Studios' por três novas colunas:
# - A quantidade de filmes de cada classe da coluna;
# - A média ponderada das avaliações recebidas pelos filmes das classes das colunas;
# - A multiplicação entre essa média ponderada e a quantidade de filmes.
# ------------------------------------------------------------------------------

colunas = ['Director', 'Genres', 'Studios']

for coluna in colunas:
    
    # Removendo possíveis espaços em branco das strings das listas das colunas 'Genres' e 'Studios'.
    if coluna in ['Genres', 'Studios']:       
        dados[coluna] = dados[coluna].apply(lambda lista: [item.strip() for item in lista])
        
    # Padronização das strings dos diretores, removendo espaços antes da vírgula e separando em listas.
    else:
        dados[coluna] = dados[coluna].str.replace(r'\s*,\s*', ', ', regex=True).str.split(', ')
        
    # Cria um DataFrame onde há uma linha para cada conjunto de strings de gêneros/estúdios.
    dados_explodidos = dados.explode(coluna)
    
    # Cria uma coluna com a média ponderada das avaliações das 3 eestrelas pelo total de avaliações.
    dados_explodidos['media_ponderada_notas'] = dados_explodidos.apply(media_ponderada_notas, axis=1)

    # Calcula as estatísticas da coluna.
    status_coluna = dados_explodidos.groupby(coluna)['media_ponderada_notas'].agg(
        qtd_filmes=('count'),
        media_ponderada=('mean')
    ).reset_index()

    # Para diretores (usando o mesmo status_diretor original)
    dados[[f'{coluna.lower()}_count', f'{coluna.lower()}_mean_rating', f'{coluna.lower()}_mean_per_qtd']] = dados.apply(
        lambda linha: calcular_metricas_agrupadas(linha, coluna, status_coluna), 
        axis=1
    )

    # Removendo a coluna de referência.
    dados = dados.drop(columns=[coluna])

# ------------------------------------------------------------------------------
# Exibindo os coeficientes de Pierson, Kendall Tau e Spearman de cada coluna 
# em relação ao alvo.
# ------------------------------------------------------------------------------

show_correlations(dados, "Average_rating")

# ------------------------------------------------------------------------------
# Removendo variáveis que possuiam coefs de Pearson e Kendall menores que 0,1
# ------------------------------------------------------------------------------

dados = dados.drop(columns=['genres_mean_rating', 'studios_count', 
                            'studios_mean_rating', 'studios_mean_per_qtd'])

# ------------------------------------------------------------------------------
# Exibindo um histograma para cada coluna numérica do Dataset.
# ------------------------------------------------------------------------------

exibir_histogramas(dados, bins=15, n_colunas_grade=2)

# ------------------------------------------------------------------------------
# Exibindo a matriz de correlação do conjunto de dados 
# ------------------------------------------------------------------------------

show_correlation_matrix(dados)

# ------------------------------------------------------------------------------
# Removendo variáveis altamente correlacionadas com a variável "Fans"
# ------------------------------------------------------------------------------

dados = dados.drop(columns=['Watches', 'Likes', 'Total_ratings',
                            'List_appearances', 'highest', 
                            'director_mean_per_qtd', 'medium',
                            'director_mean_rating'])

# ------------------------------------------------------------------------------
# Embaralhando o conjunto de dados para garantir que a divisão entre os dados 
# esteja isenta de qualquer viés de seleção
# ------------------------------------------------------------------------------

dados = dados.sample(frac=1, random_state=30)