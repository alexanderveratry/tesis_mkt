import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analisis_profundo_tracking():
    """
    Análisis más profundo para entender por qué el tracking error es tan alto
    """
    print("=== ANÁLISIS PROFUNDO DEL PROBLEMA DE TRACKING ===\n")
    
    # 1. Cargar datos
    data = pd.read_excel('precios_limpios.xlsx')
    if 'DATES' in data.columns:
        data['DATES'] = pd.to_datetime(data['DATES'])
        data.set_index('DATES', inplace=True)
    
    price_columns = [col for col in data.columns 
                    if col not in ['IPSA Index', 'IPSA index'] and not col.startswith('Unnamed')]
    
    prices = data[price_columns].dropna()
    ipsa_prices = data['IPSA Index'].dropna()
    
    # Retornos simples
    stock_returns = prices.pct_change().dropna()
    ipsa_returns = ipsa_prices.pct_change().dropna()
    
    # Alinear
    common_dates = stock_returns.index.intersection(ipsa_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    ipsa_returns = ipsa_returns.loc[common_dates]
    
    print("1. ANÁLISIS DE CORRELACIONES:")
    correlations = stock_returns.corrwith(ipsa_returns).sort_values(ascending=False)
    print("Top 10 correlaciones con IPSA:")
    print(correlations.head(10))
    print(f"\nCorrelación promedio: {correlations.mean():.3f}")
    print(f"Correlación mínima: {correlations.min():.3f}")
    print(f"Correlación máxima: {correlations.max():.3f}")
    
    # 2. Análisis del modelo de regresión completo
    print("\n2. REGRESIÓN LINEAL EN TODO EL PERÍODO:")
    
    lr = LinearRegression()
    lr.fit(stock_returns, ipsa_returns)
    weights_lr = lr.coef_
    
    # Normalizar pesos
    weights_lr_pos = np.maximum(weights_lr, 0)  # Solo pesos positivos
    if weights_lr_pos.sum() > 0:
        weights_lr_normalized = weights_lr_pos / weights_lr_pos.sum()
    else:
        weights_lr_normalized = np.ones(len(weights_lr)) / len(weights_lr)
    
    # Predicciones
    predicted_returns = (stock_returns * weights_lr_normalized).sum(axis=1)
    
    # Métricas
    correlation_pred_actual = np.corrcoef(predicted_returns, ipsa_returns)[0,1]
    tracking_error_period = np.std(predicted_returns - ipsa_returns) * np.sqrt(252)
    
    print(f"R² del modelo: {lr.score(stock_returns, ipsa_returns):.4f}")
    print(f"Correlación predicho vs real: {correlation_pred_actual:.4f}")
    print(f"Tracking Error anualizado: {tracking_error_period:.4f}")
    
    # Retornos acumulados
    cum_predicted = (1 + predicted_returns).cumprod().iloc[-1] - 1
    cum_ipsa = (1 + ipsa_returns).cumprod().iloc[-1] - 1
    
    print(f"Retorno acumulado predicho: {cum_predicted:.4f} ({cum_predicted*100:.2f}%)")
    print(f"Retorno acumulado IPSA: {cum_ipsa:.4f} ({cum_ipsa*100:.2f}%)")
    print(f"Diferencia: {abs(cum_predicted - cum_ipsa):.4f} ({abs(cum_predicted - cum_ipsa)*100:.2f}%)")
    
    # 3. Pesos del modelo de regresión
    print("\n3. PESOS DEL MODELO DE REGRESIÓN:")
    weights_series = pd.Series(weights_lr_normalized, index=stock_returns.columns).sort_values(ascending=False)
    print("Top 15 pesos:")
    print(weights_series.head(15))
    print(f"\nPesos top 10 suman: {weights_series.head(10).sum():.3f}")
    print(f"Número de pesos > 0: {(weights_series > 0).sum()}")
    
    # 4. Análisis por subperíodos
    print("\n4. ANÁLISIS POR SUBPERÍODOS:")
    
    n_periods = len(stock_returns)
    period_size = 252  # 1 año
    
    period_results = []
    
    for start_idx in range(0, n_periods - period_size, period_size//2):  # Overlap 50%
        end_idx = min(start_idx + period_size, n_periods)
        
        period_stocks = stock_returns.iloc[start_idx:end_idx]
        period_ipsa = ipsa_returns.iloc[start_idx:end_idx]
        
        if len(period_stocks) < 100:
            continue
            
        # Regresión en el período
        lr_period = LinearRegression()
        lr_period.fit(period_stocks, period_ipsa)
        
        weights_period = lr_period.coef_
        weights_period_pos = np.maximum(weights_period, 0)
        if weights_period_pos.sum() > 0:
            weights_period_norm = weights_period_pos / weights_period_pos.sum()
        else:
            weights_period_norm = np.ones(len(weights_period)) / len(weights_period)
        
        # Predicciones
        pred_period = (period_stocks * weights_period_norm).sum(axis=1)
        
        # Métricas
        te_period = np.std(pred_period - period_ipsa) * np.sqrt(252)
        corr_period = np.corrcoef(pred_period, period_ipsa)[0,1]
        r2_period = lr_period.score(period_stocks, period_ipsa)
        
        cum_pred_period = (1 + pred_period).cumprod().iloc[-1] - 1
        cum_ipsa_period = (1 + period_ipsa).cumprod().iloc[-1] - 1
        
        period_results.append({
            'start_date': period_stocks.index[0],
            'end_date': period_stocks.index[-1],
            'tracking_error': te_period,
            'correlation': corr_period,
            'r2': r2_period,
            'return_diff': abs(cum_pred_period - cum_ipsa_period)
        })
    
    # Mostrar resultados por período
    df_periods = pd.DataFrame(period_results)
    print(f"Análisis de {len(df_periods)} subperíodos:")
    print(f"Tracking Error promedio: {df_periods['tracking_error'].mean():.4f}")
    print(f"Correlación promedio: {df_periods['correlation'].mean():.4f}")
    print(f"R² promedio: {df_periods['r2'].mean():.4f}")
    print(f"Diferencia de retorno promedio: {df_periods['return_diff'].mean():.4f}")
    
    # Períodos con peor performance
    print("\nPeríodos con mayor tracking error:")
    worst_periods = df_periods.nlargest(3, 'tracking_error')
    for _, period in worst_periods.iterrows():
        print(f"{period['start_date'].strftime('%Y-%m-%d')} a {period['end_date'].strftime('%Y-%m-%d')}: TE={period['tracking_error']:.4f}")
    
    # 5. Verificar si el problema está en los datos del IPSA
    print("\n5. ANÁLISIS DE CONSISTENCIA DEL IPSA:")
    
    # Calcular un IPSA sintético como promedio equiponderado
    ipsa_synthetic = stock_returns.mean(axis=1)
    correlation_synthetic = np.corrcoef(ipsa_returns, ipsa_synthetic)[0,1]
    
    cum_ipsa_real = (1 + ipsa_returns).cumprod().iloc[-1] - 1
    cum_ipsa_synthetic = (1 + ipsa_synthetic).cumprod().iloc[-1] - 1
    
    print(f"Correlación IPSA real vs sintético: {correlation_synthetic:.4f}")
    print(f"Retorno IPSA real: {cum_ipsa_real:.4f} ({cum_ipsa_real*100:.2f}%)")
    print(f"Retorno IPSA sintético: {cum_ipsa_synthetic:.4f} ({cum_ipsa_synthetic*100:.2f}%)")
    print(f"Diferencia: {abs(cum_ipsa_real - cum_ipsa_synthetic)*100:.2f}%")
    
    # 6. Crear gráfico de diagnóstico
    print("\n6. CREANDO GRÁFICO DE DIAGNÓSTICO...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Retornos acumulados
    cum_predicted_series = (1 + predicted_returns).cumprod()
    cum_ipsa_series = (1 + ipsa_returns).cumprod()
    cum_synthetic_series = (1 + ipsa_synthetic).cumprod()
    
    axes[0,0].plot(cum_predicted_series.index, cum_predicted_series, label='Modelo Regresión', linewidth=2)
    axes[0,0].plot(cum_ipsa_series.index, cum_ipsa_series, label='IPSA Real', linewidth=2)
    axes[0,0].plot(cum_synthetic_series.index, cum_synthetic_series, label='IPSA Sintético', linewidth=1, alpha=0.7)
    axes[0,0].set_title('Retornos Acumulados')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Diferencia de retornos diarios
    axes[0,1].plot((predicted_returns - ipsa_returns).index, (predicted_returns - ipsa_returns) * 100)
    axes[0,1].set_title('Diferencia de Retornos Diarios (%)')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].grid(True, alpha=0.3)
    
    # Distribución de correlaciones
    axes[1,0].hist(correlations, bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Distribución de Correlaciones con IPSA')
    axes[1,0].axvline(x=correlations.mean(), color='red', linestyle='--', label=f'Media: {correlations.mean():.3f}')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Tracking error por período
    axes[1,1].plot(df_periods['start_date'], df_periods['tracking_error'], 'o-')
    axes[1,1].set_title('Tracking Error por Período')
    axes[1,1].axhline(y=df_periods['tracking_error'].mean(), color='red', linestyle='--', 
                     label=f'Media: {df_periods["tracking_error"].mean():.4f}')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('diagnostico_tracking_profundo.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado como: diagnostico_tracking_profundo.png")
    plt.show()
    
    return {
        'correlations': correlations,
        'weights_optimal': weights_series,
        'period_analysis': df_periods,
        'model_performance': {
            'r2': lr.score(stock_returns, ipsa_returns),
            'tracking_error': tracking_error_period,
            'return_difference': abs(cum_predicted - cum_ipsa)
        }
    }

if __name__ == "__main__":
    resultados = analisis_profundo_tracking()
