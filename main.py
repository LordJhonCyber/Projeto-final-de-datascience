import pandas as pd
import numpy as np

# Simulando um dataset fictício de vendas de e-commerce com 20 produtos
data = {
    'order_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020],
    'produto': ['Laptop', 'Telefone', 'Tablet', 'Fones de Ouvido', 'Teclado', 'Mouse', 'Monitor', 'Impressora', 'Câmera', 'Smartwatch',
                'Drone', 'Carregador', 'Pen Drive', 'Caixa de Som', 'Controle', 'Microfone', 'SSD', 'Cabo HDMI', 'Fonte de Alimentação', 'Modem'],
    'categoria': ['Eletrônicos', 'Eletrônicos', 'Eletrônicos', 'Acessórios', 'Acessórios', 'Acessórios', 'Eletrônicos', 'Acessórios', 'Eletrônicos', 'Eletrônicos',
                  'Eletrônicos', 'Acessórios', 'Acessórios', 'Acessórios', 'Acessórios', 'Acessórios', 'Eletrônicos', 'Acessórios', 'Acessórios', 'Eletrônicos'],
    'quantidade': [1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 2, 3, 4, 1, 2, 1, 2, 3, 1, 1],
    'preço': [1200, 800, 300, 150, 100, 50, 500, 300, 700, 250, 1200, 50, 30, 100, 80, 150, 200, 20, 100, 400],
    'data_pedido': ['2023-09-01', '2023-09-02', '2023-09-03', '2023-09-03', '2023-09-04', '2023-09-05', '2023-09-05', '2023-09-06', '2023-09-06', '2023-09-07',
                    '2023-09-08', '2023-09-08', '2023-09-09', '2023-09-09', '2023-09-10', '2023-09-11', '2023-09-11', '2023-09-12', '2023-09-13', '2023-09-13'],
    'cliente': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack', 'Kathy', 'Leo', 'Mike', 'Nina', 'Oscar', 'Paul', 'Quinn', 'Rachel', 'Sam', 'Tom'],
    'país': ['EUA', 'EUA', 'Canadá', 'Canadá', 'EUA', 'Reino Unido', 'EUA', 'Canadá', 'Reino Unido', 'EUA', 'Canadá', 'EUA', 'EUA', 'Reino Unido', 'EUA', 'Canadá', 'Reino Unido', 'EUA', 'Canadá', 'EUA']
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Exibindo o DataFrame original
print(df)

# Verificando e removendo duplicatas (se houver)
df = df.drop_duplicates()

# Verificando valores nulos
print(df.isnull().sum())  # Não há valores nulos neste dataset fictício

# Convertendo a coluna 'data_pedido' para datetime
df['data_pedido'] = pd.to_datetime(df['data_pedido'])

# Exibindo o DataFrame após a conversão
print(df.dtypes)

# Exemplo de tratamento de valores incorretos (se necessário)
df['quantidade'] = df['quantidade'].apply(lambda x: np.abs(x))

# Exibindo o DataFrame limpo
print(df)

# Calculando o total de vendas (quantidade * preço)
df['vendas'] = df['quantidade'] * df['preço']

# Estatísticas descritivas básicas
estatisticas_descritivas = df['vendas'].describe()
print(estatisticas_descritivas)

# Calculando média, desvio padrão, mínimo e máximo das vendas
media_vendas = df['vendas'].mean()
desvio_padrao_vendas = df['vendas'].std()
min_vendas = df['vendas'].min()
max_vendas = df['vendas'].max()

print(f"Média de vendas: {media_vendas}")
print(f"Desvio padrão das vendas: {desvio_padrao_vendas}")
print(f"Vendas mínimas: {min_vendas}")
print(f"Vendas máximas: {max_vendas}")

import matplotlib.pyplot as plt
import seaborn as sns

# Agrupando as vendas por data do pedido
vendas_por_data = df.groupby('data_pedido')['vendas'].sum().reset_index()

###Visualizações Gráficas

# Criando o gráfico de linha de vendas ao longo do tempo
plt.figure(figsize=(10, 6))
sns.lineplot(x='data_pedido', y='vendas', data=vendas_por_data)
plt.title('Tendência de Vendas ao Longo do Tempo')
plt.xlabel('Data do Pedido')
plt.ylabel('Total de Vendas')
plt.xticks(rotation=45)
plt.show()

### Distribuição de vendas

# Criando um histograma para ver a distribuição das vendas
plt.figure(figsize=(10, 6))
sns.histplot(df['vendas'], bins=10, kde=True)
plt.title('Distribuição das Vendas')
plt.xlabel('Total de Vendas')
plt.ylabel('Frequência')
plt.show()


## Vendas por Categoria de Produtos

# Agrupando as vendas por categoria de produtos
vendas_por_categoria = df.groupby('categoria')['vendas'].sum().reset_index()

# Criando um gráfico de barras para vendas por categoria
plt.figure(figsize=(10, 6))
sns.barplot(x='categoria', y='vendas', data=vendas_por_categoria)
plt.title('Vendas por Categoria de Produto')
plt.xlabel('Categoria')
plt.ylabel('Total de Vendas')
plt.show()


### Análise de Correlação

# Calculando a correlação entre as variáveis numéricas
correlacao = df[['quantidade', 'preço', 'vendas']].corr()
print(correlacao)

# Exibindo a matriz de correlação com heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação entre Quantidade, Preço e Vendas')
plt.show()


### Gráfico de Dispersão

# Gráfico de dispersão entre preço e quantidade
plt.figure(figsize=(10, 6))
sns.scatterplot(x='preço', y='quantidade', data=df, hue='categoria', size='vendas', sizes=(50, 200))
plt.title('Relação entre Preço e Quantidade')
plt.xlabel('Preço')
plt.ylabel('Quantidade')
plt.show()

### Analise de preço por país
 
# Agrupando as vendas por país
vendas_por_pais = df.groupby('país')['vendas'].sum().reset_index()

# Criando um gráfico de barras para vendas por país
plt.figure(figsize=(10, 6))
sns.barplot(x='país', y='vendas', data=vendas_por_pais)
plt.title('Vendas por País')
plt.xlabel('País')
plt.ylabel('Total de Vendas')
plt.show()

### Vendas Acumuladas ao Longo do Tempo

# Calculando as vendas acumuladas
vendas_por_data['vendas_acumuladas'] = vendas_por_data['vendas'].cumsum()

# Gráfico de vendas acumuladas ao longo do tempo
plt.figure(figsize=(10, 6))
sns.lineplot(x='data_pedido', y='vendas_acumuladas', data=vendas_por_data)
plt.title('Vendas Acumuladas ao Longo do Tempo')
plt.xlabel('Data do Pedido')
plt.ylabel('Total de Vendas Acumuladas')
plt.xticks(rotation=45)
plt.show()


### Aplicação do Teste ANOVA

from scipy import stats

# Agrupando os dados por categorias de produtos
eletronicos = df[df['categoria'] == 'Eletrônicos']['vendas']
acessorios = df[df['categoria'] == 'Acessórios']['vendas']

# Realizando o teste ANOVA para comparar as médias entre as categorias
f_stat, p_value = stats.f_oneway(eletronicos, acessorios)

# Exibindo os resultados do teste
print(f"Estatística F: {f_stat}")
print(f"P-valor: {p_value}")

# Verificando se rejeitamos ou não a hipótese nula
alpha = 0.05  # Nível de significância de 5%
if p_value < alpha:
    print("Rejeitamos a hipótese nula. Há uma diferença significativa entre as vendas das categorias.")
else:
    print("Falhamos em rejeitar a hipótese nula. Não há diferença significativa entre as vendas das categorias.")


# Exemplo do Teste t de Student para comparar Eletrônicos e Acessórios
t_stat, p_value_t = stats.ttest_ind(eletronicos, acessorios)

print(f"Estatística T: {t_stat}")
print(f"P-valor: {p_value_t}")

if p_value_t < alpha:
    print("Rejeitamos a hipótese nula. Há uma diferença significativa entre as vendas de Eletrônicos e Acessórios.")
else:
    print("Falhamos em rejeitar a hipótese nula. Não há diferença significativa entre as vendas de Eletrônicos e Acessórios.")
