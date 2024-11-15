import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.datasets import get_rdataset
import matplotlib.pyplot as plt

# Função para calcular métricas de avaliação
def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# 1. Leitura dos dados da base AirPassengers
data = get_rdataset("AirPassengers", "datasets").data
data.columns = ['tempo', 'valor']

# Convertendo a coluna 'tempo' para índice datetime
data['tempo'] = pd.date_range(start='1949-01', periods=len(data), freq='M')
data.set_index('tempo', inplace=True)

# Exibindo os primeiros registros da base de dados
print("Primeiros registros da base de dados:")
print(data.head())

# Visualizando a série temporal
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['valor'], label='AirPassengers', color='blue')
plt.title('Série Temporal - AirPassengers')
plt.xlabel('Tempo')
plt.ylabel('Número de Passageiros')
plt.legend()
plt.show()

# 2. Divisão em treino (80%) e teste (20%)
train_size = int(0.8 * len(data))
train, test = data.iloc[:train_size], data.iloc[train_size:]

y_train, y_test = train['valor'], test['valor']

# Inicializando uma lista para armazenar os resultados
resultados = []

# 3.1 Regressão Linear
X_train = np.arange(len(y_train)).reshape(-1, 1)
X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
resultados.append(['Regressão Linear', *calcular_metricas(y_test, y_pred_lr)])

# 3.2 Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
resultados.append(['Random Forest', *calcular_metricas(y_test, y_pred_rf)])

# 3.3 Suavização Exponencial (Holt-Winters)
model_hw = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12)
fit_hw = model_hw.fit()
y_pred_hw = fit_hw.forecast(len(y_test))
resultados.append(['Holt-Winters', *calcular_metricas(y_test, y_pred_hw)])

# 3.4 Encontrando o Melhor ARIMA Automaticamente
print("Ajustando o melhor modelo ARIMA...")
auto_model = auto_arima(y_train,
                        start_p=1, start_q=1,
                        max_p=5, max_q=5,
                        seasonal=False,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)

print(f"Melhor modelo ARIMA: {auto_model.order}")

fit_arima = auto_model.fit(y_train)
y_pred_arima = fit_arima.predict(n_periods=len(y_test))
resultados.append(['ARIMA (Melhor Ajuste)', *calcular_metricas(y_test, y_pred_arima)])

# 4. Exibindo as Métricas em um DataFrame
df_resultados = pd.DataFrame(resultados, columns=['Modelo', 'MAE', 'RMSE', 'R²'])
print("\nMétricas de Avaliação dos Modelos:")
print(df_resultados)

# Identificando o melhor modelo com base no RMSE
melhor_modelo = df_resultados.loc[df_resultados['RMSE'].idxmin()]
print(f"\nMelhor Modelo: {melhor_modelo['Modelo']} com RMSE={melhor_modelo['RMSE']}")

# 5. Prevendo valores futuros com o melhor modelo
n_futuro = 12  # Previsão para 12 meses à frente

if melhor_modelo['Modelo'] == 'Regressão Linear':
    X_futuro = np.arange(len(data), len(data) + n_futuro).reshape(-1, 1)
    previsao_futura = lr.predict(X_futuro)

elif melhor_modelo['Modelo'] == 'Random Forest':
    X_futuro = np.arange(len(data), len(data) + n_futuro).reshape(-1, 1)
    previsao_futura = rf.predict(X_futuro)

elif melhor_modelo['Modelo'] == 'Holt-Winters':
    previsao_futura = fit_hw.forecast(n_futuro)

elif melhor_modelo['Modelo'] == 'ARIMA (Melhor Ajuste)':
    previsao_futura = fit_arima.predict(n_periods=n_futuro)

# 6. Visualizando a previsão futura
datas_futuras = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=n_futuro, freq='M')

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['valor'], label='Dados Originais', color='blue')
plt.plot(datas_futuras, previsao_futura, label='Previsão Futura', color='red')
plt.title(f'Previsão Futura - {melhor_modelo["Modelo"]}')
plt.xlabel('Tempo')
plt.ylabel('Número de Passageiros')
plt.legend()
plt.show()
