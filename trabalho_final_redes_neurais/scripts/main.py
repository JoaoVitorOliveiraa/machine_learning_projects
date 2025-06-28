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

