import pandas as pd
import numpy as np

# Script para verificar cálculos de retornos acumulados
print("=== DIAGNÓSTICO DE RETORNOS ACUMULADOS ===")

# Cargar datos
data = pd.read_excel('precios_limpios.xlsx')
data['DATES'] = pd.to_datetime(data['DATES'])
data.set_index('DATES', inplace=True)

# Obtener precios
price_columns = [col for col in data.columns 
                if col not in ['IPSA Index', 'IPSA index'] and not col.startswith('Unnamed')]
prices = data[price_columns].dropna()
ipsa_prices = data['IPSA Index'].loc[prices.index]

# Calcular retornos
stock_returns = np.log(prices / prices.shift(1)).dropna()
ipsa_returns = np.log(ipsa_prices / ipsa_prices.shift(1)).dropna()

print(f"Período de datos: {stock_returns.index.min()} a {stock_returns.index.max()}")
print(f"Número de observaciones: {len(stock_returns)}")

# Verificar retornos acumulados del IPSA
ipsa_cumulative_simple = (1 + ipsa_returns).cumprod()
print(f"\nRetorno total IPSA (método simple): {(ipsa_cumulative_simple.iloc[-1] - 1) * 100:.2f}%")

# Verificar retorno desde precios
price_return = (ipsa_prices.iloc[-1] / ipsa_prices.iloc[0]) - 1
print(f"Retorno total IPSA (desde precios): {price_return * 100:.2f}%")

# Probar con portafolio equiponderado
equal_weights = np.ones(len(stock_returns.columns)) / len(stock_returns.columns)
portfolio_returns = (stock_returns * equal_weights).sum(axis=1)
portfolio_cumulative = (1 + portfolio_returns).cumprod()
print(f"Retorno total portafolio equiponderado: {(portfolio_cumulative.iloc[-1] - 1) * 100:.2f}%")

# Verificar tracking error
tracking_error = np.std(portfolio_returns - ipsa_returns) * np.sqrt(252)
print(f"Tracking Error (equiponderado): {tracking_error:.4f}")

print("\n=== VERIFICACIÓN COMPLETADA ===")
