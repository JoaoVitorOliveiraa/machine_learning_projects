#==============================================================================
# Trabalho Final - Classificação de Filmes do Letterbox
#==============================================================================

#------------------------------------------------------------------------------
# Importando bibliotecas
#------------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from functions import (media_ponderada_notas, calcular_metricas_agrupadas,
                       show_correlations, exibir_histogramas, show_correlation_matrix,
                       aplicar_pca)

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

# ------------------------------------------------------------------------------
# Separação do DataFrame em Features (X) e Alvo (y)
# X conterá todas as colunas de entrada (features)
# y conterá a coluna que queremos prever (o alvo 'Average_rating')
# ------------------------------------------------------------------------------

X = dados.drop(columns=['Average_rating'])
y = dados['Average_rating']

print("Dimensões de X (features):", X.shape)
print("Dimensões de y (alvo):", y.shape)
print("-" * 50)

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=42))
])

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(model_pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')

y_pred_aggregated = cross_val_predict(model_pipeline, X, y, cv=kfold)
final_rmse = np.sqrt(mean_squared_error(y, y_pred_aggregated))
final_r2 = r2_score(y, y_pred_aggregated)
print(f"RMSE geral (agregado) do melhor modelo: {final_rmse:.4f}")
print(f"R² geral (agregado) do melhor modelo: {final_r2:.4f}")

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred_aggregated, alpha=0.5, edgecolors='k')
perfect_line = [min(y), max(y)]
plt.plot(perfect_line, perfect_line, 'r--', lw=2, label='Previsão Perfeita (y=x)')
plt.title('Valores Previstos vs. Reais (MELHOR Modelo)', fontsize=16)
plt.xlabel('Valores Reais', fontsize=12)
plt.ylabel('Valores Previstos', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.savefig('previsto_vs_real_melhor_modelo.png')
plt.show()
print("Gráfico 'previsto_vs_real_melhor_modelo.png' foi salvo.")

# ------------------------------------------------------------------------------
# 2. DEFINIÇÃO DO ESPAÇO DE BUSCA DE HIPERPARÂMETROS
# Aqui definimos um dicionário com os hiperparâmetros que queremos testar.
# RandomizedSearchCV irá sortear combinações a partir daqui.
# ------------------------------------------------------------------------------
print("2. Definindo o espaço de busca de hiperparâmetros...")
param_distributions = {
    'mlp__hidden_layer_sizes': [(32, 16), (64, 32), (64, 32, 16), (100, 50)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [0.001, 0.01, 0.1, 1],
    'mlp__learning_rate_init': [0.001, 0.005, 0.01]
}

# ------------------------------------------------------------------------------
# 3. CONFIGURAÇÃO E EXECUÇÃO DO RANDOMIZEDSEARCHCV
# O Pipeline garante que o scaler seja aplicado corretamente em cada fold.
# Usamos 'neg_root_mean_squared_error' como métrica, pois o Scikit-learn
# tenta maximizar a pontuação (erro menor = pontuação maior/mais próxima de zero).
# ------------------------------------------------------------------------------
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(max_iter=500, random_state=42))  # Parâmetros fixos
])

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# n_iter: Número de combinações a serem testadas. Aumente para uma busca mais completa.
random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_distributions,
    n_iter=30,  # Testará 30 combinações diferentes
    cv=kfold,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,  # Usa todos os cores da CPU
    random_state=42,
    verbose=1  # Mostra o progresso
)

print("3. Iniciando a busca de hiperparâmetros...")
random_search.fit(X, y)
print("Busca finalizada!")
print("-" * 60)

# ------------------------------------------------------------------------------
# 4. ANÁLISE E VISUALIZAÇÃO DOS RESULTADOS DA BUSCA
# Vamos analisar os resultados para entender o que funcionou melhor.
# ------------------------------------------------------------------------------
print("4. Analisando os resultados da busca...")
print(f"Melhor pontuação (RMSE): {-random_search.best_score_:.4f}")
print("Melhor combinação de hiperparâmetros encontrada:")
print(random_search.best_params_)
print("-" * 60)

# Criar um DataFrame com os resultados da busca para facilitar a visualização
results_df = pd.DataFrame(random_search.cv_results_)
results_df['mean_test_score'] = -results_df['mean_test_score']  # Converte para RMSE positivo


# Função para plotar os resultados
def plot_search_results(df):
    params_to_plot = ['param_mlp__solver', 'param_mlp__activation', 'param_mlp__hidden_layer_sizes']

    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(20, 6), sharey=True)
    fig.suptitle('Desempenho (RMSE) por Hiperparâmetro', fontsize=16)

    for i, param in enumerate(params_to_plot):
        # Converte tuplas para strings para plotagem
        if param == 'param_mlp__hidden_layer_sizes':
            df[param] = df[param].astype(str)

        sns.boxplot(x=param, y='mean_test_score', data=df, ax=axes[i], palette='viridis')
        axes[i].set_title(f'RMSE por {param.replace("param_mlp__", "")}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('RMSE' if i == 0 else '')
        axes[i].tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparacao_hiperparametros.png')
    plt.show()


plot_search_results(results_df)
print("Gráfico 'comparacao_hiperparametros.png' foi salvo.")
print("-" * 60)

# ------------------------------------------------------------------------------
# 5. AVALIAÇÃO DETALHADA DO MELHOR MODELO ENCONTRADO
# Agora usamos o melhor pipeline encontrado para gerar os gráficos de diagnóstico.
# ------------------------------------------------------------------------------
print("5. Avaliando o melhor modelo encontrado em detalhe...")
best_model_pipeline = random_search.best_estimator_

# Gerando o Box Plot de RMSE para o MELHOR modelo
scores_best_model = cross_val_score(best_model_pipeline, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
rmse_scores_best_model = -scores_best_model

plt.figure(figsize=(10, 6))
sns.boxplot(x=rmse_scores_best_model, palette='plasma', width=0.4)
sns.stripplot(x=rmse_scores_best_model, color='black', size=5, jitter=0.1, label='RMSE por Fold')
plt.title('Box Plot do RMSE do MELHOR Modelo nos 10 Folds', fontsize=16)
plt.xlabel('Raiz do Erro Quadrático Médio (RMSE) - Menor é Melhor', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('boxplot_rmse_melhor_modelo.png')
plt.show()
print("Gráfico 'boxplot_rmse_melhor_modelo.png' foi salvo.\n")

# Gerando o Gráfico Previsto vs. Real para o MELHOR modelo
y_pred_aggregated = cross_val_predict(best_model_pipeline, X, y, cv=kfold)
final_rmse = np.sqrt(mean_squared_error(y, y_pred_aggregated))
final_r2 = r2_score(y, y_pred_aggregated)
print(f"RMSE geral (agregado) do melhor modelo: {final_rmse:.4f}")
print(f"R² geral (agregado) do melhor modelo: {final_r2:.4f}")

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred_aggregated, alpha=0.5, edgecolors='k')
perfect_line = [min(y), max(y)]
plt.plot(perfect_line, perfect_line, 'r--', lw=2, label='Previsão Perfeita (y=x)')
plt.title('Valores Previstos vs. Reais (MELHOR Modelo)', fontsize=16)
plt.xlabel('Valores Reais', fontsize=12)
plt.ylabel('Valores Previstos', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.savefig('previsto_vs_real_melhor_modelo.png')
plt.show()
print("Gráfico 'previsto_vs_real_melhor_modelo.png' foi salvo.")

# ==============================================================================
# 6. (NOVO) IDEIA FÁCIL: ENSEMBLE DOS MELHORES MODELOS
# ==============================================================================
print("6. Testando uma melhoria fácil: Ensemble dos Top 3 Modelos...")

# Pega o DataFrame de resultados e ordena pelos melhores scores
results_df_sorted = results_df.sort_values(by='rank_test_score')

# Seleciona os 3 melhores conjuntos de parâmetros
top_n = 3
best_n_params = results_df_sorted.head(top_n)['params'].tolist()

all_predictions = []

print(f"Gerando previsões para os {top_n} melhores modelos...")
for i, params in enumerate(best_n_params):
    print(f"  - Modelo {i + 1}/{top_n}...")

    # Cria um novo pipeline com os parâmetros do ranking
    current_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(max_iter=500, random_state=42))
    ])
    current_pipeline.set_params(**params)

    # Gera as previsões para este modelo
    y_pred = cross_val_predict(current_pipeline, X, y, cv=kfold)
    all_predictions.append(y_pred)

# Calcula a média das previsões dos 3 modelos
ensemble_prediction = np.mean(all_predictions, axis=0)

# Avalia o resultado do Ensemble
print("\n--- Resultados do Modelo Ensemble ---")
final_rmse_ensemble = np.sqrt(mean_squared_error(y, ensemble_prediction))
final_r2_ensemble = r2_score(y, ensemble_prediction)

print(f"RMSE geral (agregado) do ENSEMBLE: {final_rmse_ensemble:.4f}")
print(f"R² geral (agregado) do ENSEMBLE: {final_r2_ensemble:.4f}")

# Gerando o gráfico de dispersão para o ENSEMBLE
print("\nGerando o Gráfico Previsto vs. Real para o Modelo Ensemble...")
plt.figure(figsize=(8, 8))
plt.scatter(y, ensemble_prediction, alpha=0.5, edgecolors='k', color='green')
perfect_line = [min(y), max(y)]
plt.plot(perfect_line, perfect_line, 'r--', lw=2, label='Previsão Perfeita (y=x)')
plt.title('Valores Previstos vs. Reai', fontsize=16)
plt.xlabel('Valores Reais', fontsize=12)
plt.ylabel('Valores Previstos', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.savefig('previsto_vs_real_ensemble.png')
plt.show()
print("Gráfico 'previsto_vs_real_ensemble.png' foi salvo.")