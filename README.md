# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Simulação de um dataset fictício de vendas de e-commerce (substitua por seu próprio dataset, se necessário)
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

# Criando o DataFrame a partir dos dados
df = pd.DataFrame(data)

# 1. Pré-processamento e Preparação dos Dados

# Criando a coluna 'vendas', que é a multiplicação da quantidade pelo preço
df['vendas'] = df['quantidade'] * df['preço']

# Como algumas colunas são categóricas (como 'categoria' e 'país'), precisamos transformá-las em valores numéricos.
# Usaremos o LabelEncoder para converter essas colunas.

# Convertendo a coluna 'categoria' e 'país' para valores numéricos
le_categoria = LabelEncoder()
le_pais = LabelEncoder()
df['categoria'] = le_categoria.fit_transform(df['categoria'])  # Converte categorias em números (e.g., Eletrônicos = 0, Acessórios = 1)
df['país'] = le_pais.fit_transform(df['país'])  # Converte países em números

# Verificando os primeiros registros do DataFrame após a conversão
print(df.head())

# 2. Divisão dos Dados em Conjuntos de Treino e Teste

# Definindo as variáveis preditoras (X) e a variável alvo (y)
X = df[['quantidade', 'preço', 'categoria', 'país']]  # Features (colunas que vamos usar para prever as vendas)
y = df['vendas']  # Target (a variável que queremos prever)

# Dividindo o dataset em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modelo de Machine Learning - Regressão Linear

# Criando o modelo de Regressão Linear
modelo_lr = LinearRegression()

# Treinando o modelo de Regressão Linear com os dados de treino
modelo_lr.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred_lr = modelo_lr.predict(X_test)

# Avaliando o desempenho do modelo de Regressão Linear
mse_lr = mean_squared_error(y_test, y_pred_lr)  # Erro quadrático médio
mae_lr = mean_absolute_error(y_test, y_pred_lr)  # Erro absoluto médio
r2_lr = r2_score(y_test, y_pred_lr)  # Coeficiente de determinação (R²)

# Exibindo os resultados da Regressão Linear
print("Modelo de Regressão Linear:")
print(f"Mean Squared Error (MSE): {mse_lr}")
print(f"Mean Absolute Error (MAE): {mae_lr}")
print(f"R²: {r2_lr}")

# 4. Modelo de Machine Learning - Árvore de Decisão

# Criando o modelo de Árvores de Decisão
modelo_dt = DecisionTreeRegressor(random_state=42)

# Treinando o modelo de Árvore de Decisão com os dados de treino
modelo_dt.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred_dt = modelo_dt.predict(X_test)

# Avaliando o desempenho do modelo de Árvores de Decisão
mse_dt = mean_squared_error(y_test, y_pred_dt)  # Erro quadrático médio
mae_dt = mean_absolute_error(y_test, y_pred_dt)  # Erro absoluto médio
r2_dt = r2_score(y_test, y_pred_dt)  # Coeficiente de determinação (R²)

# Exibindo os resultados da Árvore de Decisão
print("\nModelo de Árvores de Decisão:")
print(f"Mean Squared Error (MSE): {mse_dt}")
print(f"Mean Absolute Error (MAE): {mae_dt}")
print(f"R²: {r2_dt}")

# 5. Conclusão:
# Comparando o desempenho dos modelos, podemos escolher aquele que melhor se ajusta aos dados.
# Se o modelo de Regressão Linear tiver um R² mais próximo de 1 e menores MSE e MAE, ele pode ser a melhor opção.
# Caso contrário, o modelo de Árvores de Decisão pode ser mais adequado para os dados, especialmente se houver não-linearidades.
