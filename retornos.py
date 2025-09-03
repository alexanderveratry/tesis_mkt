import pandas as pd
import numpy as np

# Cargar datos
data = pd.read_excel('precios_limpios.xlsx')



# Convertir fechas
if 'DATES' in data.columns:
    data['DATES'] = pd.to_datetime(data['DATES'])
    data.set_index('DATES', inplace=True)

# Separar variable dependiente (IPSA Index)
ipsa_column = 'IPSA Index' if 'IPSA Index' in data.columns else None

if ipsa_column:
    ipsa_prices = data[ipsa_column].copy()
else:
    # Crear IPSA sintético como promedio equiponderado
    price_columns = [col for col in data.columns if not col.startswith('Unnamed')]
    ipsa_prices = data[price_columns].mean(axis=1)

# Variables independientes (todas las demás columnas excepto IPSA)
independent_columns = [col for col in data.columns 
                     if col not in [ipsa_column, 'DATES'] and not col.startswith('Unnamed')]

stock_prices = data[independent_columns].copy()

# Eliminar filas con valores faltantes
stock_prices = stock_prices.dropna()
ipsa_prices = ipsa_prices.loc[stock_prices.index]

# Calcular retornos logarítmicos
stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
ipsa_returns = np.log(ipsa_prices / ipsa_prices.shift(1)).dropna()

# Alinear fechas
common_dates = stock_returns.index.intersection(ipsa_returns.index)
stock_returns = stock_returns.loc[common_dates]
ipsa_returns = ipsa_returns.loc[common_dates]

print(f"Variable dependiente: {ipsa_column if ipsa_column else 'IPSA sintético'}")
print(f"Variables independientes: {len(stock_returns.columns)} activos")
print(f"Período: {stock_returns.index.min()} a {stock_returns.index.max()}")
print(f"Observaciones: {len(stock_returns)}")

def plot_returns(ipsa_returns, stock_returns):
    import matplotlib.pyplot as plt
    
    # Calcular retornos acumulados correctamente
    # Convertir retornos logarítmicos a simples y luego calcular productoria
    ipsa_simple_returns = np.exp(ipsa_returns) - 1
    stock_simple_returns = np.exp(stock_returns) - 1
    
    ipsa_cumulative = (1 + ipsa_simple_returns).cumprod()
    stock_cumulative = (1 + stock_simple_returns).cumprod()
    
    # Encontrar top 5 por retorno acumulado final
    final_returns = stock_cumulative.iloc[-1] - 1
    top5_stocks = final_returns.nlargest(5).index.tolist()
    
    # Graficar
    plt.figure(figsize=(12, 8))
    
    # IPSA
    plt.plot(ipsa_cumulative.index, (ipsa_cumulative - 1) * 100, 
             'red', linewidth=2, label='IPSA Index')
    
    # Top 5 stocks
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i, stock in enumerate(top5_stocks):
        plt.plot(stock_cumulative.index, (stock_cumulative[stock] - 1) * 100,
                color=colors[i], linewidth=1.5, label=stock)
    
    plt.title('Retornos Acumulados: IPSA vs Top 5 Activos (Cálculo Corregido)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Retorno Acumulado (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadísticas
    print(f"\nTop 5 activos por retorno acumulado (cálculo corregido):")
    for i, stock in enumerate(top5_stocks):
        ret = final_returns[stock] * 100
        print(f"{i+1}. {stock}: {ret:.2f}%")
    
    ipsa_final = (ipsa_cumulative.iloc[-1] - 1) * 100
    print(f"\nIPSA Index: {ipsa_final:.2f}%")

# Calcular retorno simple (precio_final/precio_inicial - 1) y comparar con retorno acumulado
print("\n" + "=" * 80)
print("COMPARACIÓN: RETORNO SIMPLE vs RETORNO ACUMULADO")
print("=" * 80)

# Calcular retornos simples
simple_returns = (stock_prices.iloc[-1] / stock_prices.iloc[0] - 1) * 100

# Calcular retorno acumulado correcto: productoria de (1 + retorno_t)
# Convertir retornos logarítmicos a retornos simples para el cálculo acumulado
simple_daily_returns = np.exp(stock_returns) - 1
cumulative_returns = ((1 + simple_daily_returns).cumprod().iloc[-1] - 1) * 100

# Crear DataFrame de comparación
comparison_df = pd.DataFrame({
    'Retorno_Simple_%': simple_returns,
    'Retorno_Acumulado_%': cumulative_returns,
    'Diferencia_%': simple_returns - cumulative_returns
})

# Ordenar por retorno simple descendente
comparison_df = comparison_df.sort_values('Retorno_Simple_%', ascending=False)

print(comparison_df.round(4))

# Estadísticas de la diferencia
print(f"\nEstadísticas de la diferencia (Simple - Acumulado):")
print(f"Media: {comparison_df['Diferencia_%'].mean():.6f}%")
print(f"Desviación estándar: {comparison_df['Diferencia_%'].std():.6f}%")
print(f"Máxima diferencia: {comparison_df['Diferencia_%'].max():.6f}%")
print(f"Mínima diferencia: {comparison_df['Diferencia_%'].min():.6f}%")

# Para IPSA
ipsa_simple = (ipsa_prices.iloc[-1] / ipsa_prices.iloc[0] - 1) * 100
# Calcular retorno acumulado correcto para IPSA
ipsa_simple_daily_returns = np.exp(ipsa_returns) - 1
ipsa_cumulative = ((1 + ipsa_simple_daily_returns).cumprod().iloc[-1] - 1) * 100
ipsa_difference = ipsa_simple - ipsa_cumulative

print(f"\nIPSA Index:")
print(f"Retorno Simple: {ipsa_simple:.4f}%")
print(f"Retorno Acumulado: {ipsa_cumulative:.4f}%")
print(f"Diferencia: {ipsa_difference:.6f}%")

# Ejecutar función
plot_returns(ipsa_returns, stock_returns)