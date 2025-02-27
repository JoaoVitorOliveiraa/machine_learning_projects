# EEL891 - 2024.02 - Trabalho 1

# Sobre
- Repositório referente ao Trabalho 1 da disciplina Introdução ao Aprendizado de Máquina.

# Tema
- Sistema de apoio à decisão para aprovação de crédito.

# Descrição  do Projeto
- Este é o Trabalho 1 de avaliação da disciplina EEL891 
(Introdução ao Aprendizado de Máquina) para a turma do período 2024-2.


- Neste trabalho você construirá uma classificação para apoio à decisão 
de aprovação de crédito.


- A ideia é identificar, entre os clientes que solicitam um produto de 
crédito (como um cartão de crédito ou um empréstimo pessoal, por 
exemplo) e que cumprem os pré-requisitos essenciais para a aprovação 
do crédito, aqueles que apresentam alto risco de não conseguirem 
honrar o pagamento, tornando-se inadimplentes.


- Para isso, você receberá um arquivo com dados históricos de 20.000 
transações de produtos de créditos que foram aprovados pela instituição, 
acompanhados do desenvolvimento imediato, ou seja, acompanhados da 
indicação de quais desses solicitantes receberam honrar os pagamentos 
e quais ficaram inadimplentes.


- Com base nesses dados históricos, você deverá construir um classificador 
que, a partir dos dados de uma nova solicitação de crédito, tente prever
se este solicitante será um bom ou mau pagador.


- O objetivo da disputa é disputar com seus colegas que conseguem obter a acurácia
mais alta em um conjunto de 5.000 transações de crédito aprovadas (diferentes 
das 20.000 anteriores) discussões (quitação da dívida ou inadimplência) são 
negociações ocultas no site do Kaggle, que medirá automaticamente a taxa de 
acerto das enviadas pelos concorrentes, sem revelar o "gabarito".

# Arquivos Fornecidos
- conjunto_de_treinamento.csv - Dados históricos fornecidos para treinamento de modelos preditivos. 
Este arquivo contém 20.000 amostras de concessões de crédito com id do solicitante, dados da 
solicitação e encerramento do contrato (dívida quitada ou inadimplência).


- conjunto_de_teste.csv - Dados históricos que serão usados na competição para verificar e comparar o 
desempenho dos modelos preditivos construídos pelos concorrentes. Este arquivo contém 5.000 amostras 
de concessões de crédito (diferentes das fornecidas no conjunto de treinamento) com o id do solicitante
e os dados da solicitação, mas sem o desfecho do contrato, que é a variável-alvo a ser respondida pelo
modelo preditivo.


- exemplo_arquivo_respostas.csv - Exemplo de arquivo de respostas para envio (envio). Este arquivo deve 
conter o desfecho previsto pelo modelo para cada uma das 5.000 amostras do conjunto de teste.


- dicionario_de_dados.xlsx - Planilha Excel contendo a descrição de todos os campos existentes nos arquivos CSV acima.