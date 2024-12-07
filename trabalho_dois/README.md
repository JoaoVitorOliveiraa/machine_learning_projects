# EEL891 - 2024.02 - Trabalho 2

# Sobre
- Repositório referente ao Trabalho 2 da disciplina Introdução ao Aprendizado de Máquina.

# Tema
- Estimar o preço de um imóvel a partir de suas características.

# Descrição do Projeto
- Este é o segundo trabalho de avaliação da disciplina EEL891 (Introdução ao Aprendizado de Máquina) para a turma do semestre 2024.02.


- Neste trabalho você utilizará técnicas de regressão multivariável 
para estimar o preço de um imóvel a partir de características como 
o tipo de imóvel (apartamento, casa, loft ou quitinete), bairro onde
está localizado, número de quartos, número de vagas, área útil , área 
extra e presença de elementos diferenciais em relação a outros imóveis, 
tais como churrasqueira, estacionamento para visitantes, piscina, playground, 
quadra esportiva, campo de futebol, salão de festas, salão de jogos, sala de 
ginástica, sauna e vista para o março.

# Descrição do Conjunto de Dados
- Trata-se de um conjunto de dados contendo informações a respeito de 6.683 imóveis 
residenciais de uma cidade brasileira, coletadas de ofertas de venda publicadas 
em site especializado. São fornecidos um conjunto de treinamento, um conjunto de 
teste e um exemplo de arquivo de envio de resposta, todos em formato CSV.


- No conjunto de treinamento, composto por 4683 imóveis, são fornecidas 20 características
de cada imóvel, acompanhadas do preço de venda. No conjunto de teste, composto pelos 2.000 
imóveis restantes, são fornecidos somente como características do imóvel, cabendo ao concorrente
estimar o preço de venda.

# Arquivos Fornecidos
- conjunto_de_treinamento.csv - Dados para treinamento (características dos imóveis e relativos preços)


- conjunto_de_teste.csv - Dados para teste (somente as características dos imóveis)


- exemplo_arquivo_respostas.csv - Exemplo de arquivo para envio das respostas (preços estimados para o conjunto de teste)

# Descrição dos Campos
- Id – Identificação única do imóvel


- tipo - Tipo de imóvel (apartamento, casa, loft ou quitinete)


- bairro - Nome do bairro onde o imóvel se localiza


- tipo_vendedor - Tipo de vendedor (imobiliário ou pessoa física)


- quartos - Número de quartos


- suites - Número de suítes


- vagas - Número de vagas de garagem


- area_util - Àrea útil, em metros quadrados


- area_extra - Àrea extra, em metros quadrados


- diferenciais - Descrição textual das duas principais características que diferenciam o imóvel


- churrasqueira -O anúncio menciona churrasqueira ( 1 = menciona ; 0 = não menciona )


- estacionamento - O anúncio menciona estacionamento para visitantes ( 1 = menciona ; 0 = não menciona )


- piscina - O anúncio menciona piscina ( 1 = menciona ; 0 = não menciona )


- playground - O anúncio menciona playground ( 1 = menciona ; 0 = não menciona )


- quadra - O anúncio menciona quadra esportiva ou campo de futebol ( 1 = menciona ; 0 = não menciona )


- s_festas - O anúncio menciona salão de festas ( 1 = menciona ; 0 = não menciona )


- s_jogos - O anúncio menciona salão de jogos ( 1 = menciona ; 0 = não menciona )


- s_ginastica - O anúncio menciona sala de ginástica ( 1 = menciona ; 0 = não menciona )


- sauna - O anúncio menciona sauna ( 1 = menciona ; 0 = não menciona )


- vista_mar - O anúncio menciona vista para o mar ( 1 = menciona ; 0 = não menciona )


- preco - Preço de venda (valor a ser estimado, informado apenas no conjunto de treinamento


- Obs: Os 10 campos seguintes ao campo diferenciais se referem à existência ou não de determinadas 
palavras ou expressões na descrição textual contida nesse campo.