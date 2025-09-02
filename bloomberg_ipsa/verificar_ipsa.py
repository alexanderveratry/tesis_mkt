import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def verificar_datos_ipsa():
    """
    Verificar si el problema está en la inclusión de dividendos o datos incorrectos
    """
    print("=== VERIFICACIÓN DE DATOS DEL IPSA ===\n")
    
    # Cargar datos
    data = pd.read_excel('precios_limpios.xlsx')
    if 'DATES' in data.columns:
        data['DATES'] = pd.to_datetime(data['DATES'])
        data.set_index('DATES', inplace=True)
    
    # Separar precios de acciones e IPSA
    price_columns = [col for col in data.columns 
                    if col not in ['IPSA Index', 'IPSA index'] and not col.startswith('Unnamed')]
    
    prices = data[price_columns].dropna()
    ipsa_prices = data['IPSA Index'].dropna()
    
    print("1. INFORMACIÓN BÁSICA:")
    print(f"Período de datos: {prices.index.min()} a {prices.index.max()}")
    print(f"Número de acciones: {len(price_columns)}")
    print(f"Observaciones: {len(prices)}")
    
    # Estadísticas de precios
    print(f"\n2. ESTADÍSTICAS DE PRECIOS:")
    print(f"IPSA - Precio inicial: {ipsa_prices.iloc[0]:.2f}")
    print(f"IPSA - Precio final: {ipsa_prices.iloc[-1]:.2f}")
    print(f"IPSA - Cambio total: {(ipsa_prices.iloc[-1]/ipsa_prices.iloc[0] - 1)*100:.2f}%")
    
    # Verificar comportamiento del IPSA vs acciones individuales
    print(f"\n3. COMPARACIÓN IPSA VS ACCIONES:")
    
    # Retornos simples
    stock_returns = prices.pct_change().dropna()
    ipsa_returns = ipsa_prices.pct_change().dropna()
    
    # Alinear fechas
    common_dates = stock_returns.index.intersection(ipsa_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    ipsa_returns = ipsa_returns.loc[common_dates]
    
    # Calcular índice equiponderado de las acciones
    equal_weighted_index = stock_returns.mean(axis=1)
    
    # Retornos acumulados
    cum_ipsa = (1 + ipsa_returns).cumprod()
    cum_equal_weighted = (1 + equal_weighted_index).cumprod()
    
    print(f"Retorno total IPSA: {(cum_ipsa.iloc[-1] - 1)*100:.2f}%")
    print(f"Retorno total Equiponderado: {(cum_equal_weighted.iloc[-1] - 1)*100:.2f}%")
    print(f"Diferencia: {(cum_ipsa.iloc[-1] - cum_equal_weighted.iloc[-1])*100:.2f}%")
    
    # Calcular pesos implícitos del IPSA (market cap weighted approximation)
    print(f"\n4. ANÁLISIS DE PESOS IMPLÍCITOS:")
    
    # Usar correlaciones y betas como aproximación de pesos
    betas = {}
    for stock in stock_returns.columns:
        # Calcular beta de cada acción vs IPSA
        cov_matrix = np.cov(stock_returns[stock].dropna(), ipsa_returns.loc[stock_returns[stock].dropna().index])
        beta = cov_matrix[0,1] / cov_matrix[1,1]
        betas[stock] = beta
    
    betas_series = pd.Series(betas).sort_values(ascending=False)
    print("Top 10 Betas vs IPSA:")
    print(betas_series.head(10))
    
    # Crear índice ponderado por capitalización aproximada
    # Usar promedios de precios como proxy de market cap
    market_caps = prices.mean()  # Precio promedio como proxy
    market_caps_normalized = market_caps / market_caps.sum()
    
    cap_weighted_returns = (stock_returns * market_caps_normalized).sum(axis=1)
    cum_cap_weighted = (1 + cap_weighted_returns).cumprod()
    
    print(f"\n5. COMPARACIÓN CON PONDERACIÓN POR CAPITALIZACIÓN:")
    print(f"Retorno total Cap-weighted: {(cum_cap_weighted.iloc[-1] - 1)*100:.2f}%")
    print(f"Diferencia vs IPSA: {(cum_ipsa.iloc[-1] - cum_cap_weighted.iloc[-1])*100:.2f}%")
    
    # Verificar si hay saltos o discontinuidades en IPSA
    print(f"\n6. ANÁLISIS DE DISCONTINUIDADES:")
    ipsa_daily_changes = ipsa_returns.abs()
    large_changes = ipsa_daily_changes[ipsa_daily_changes > 0.05]  # Cambios > 5%
    
    print(f"Días con cambios > 5%: {len(large_changes)}")
    if len(large_changes) > 0:
        print("Fechas con grandes cambios:")
        for date, change in large_changes.head(10).items():
            print(f"  {date.strftime('%Y-%m-%d')}: {change*100:.2f}%")
    
    # Verificar períodos específicos problemáticos
    print(f"\n7. ANÁLISIS POR AÑOS:")
    years = sorted(ipsa_returns.index.year.unique())
    
    for year in years:
        year_mask = ipsa_returns.index.year == year
        year_ipsa = ipsa_returns.loc[year_mask]
        year_stocks = stock_returns.loc[year_mask]
        
        if len(year_ipsa) > 10:  # Solo años con datos suficientes
            year_return_ipsa = (1 + year_ipsa).prod() - 1
            year_return_equal = (1 + year_stocks.mean(axis=1)).prod() - 1
            
            print(f"  {year}: IPSA={year_return_ipsa*100:.1f}%, Equal-weighted={year_return_equal*100:.1f}%, Diff={abs(year_return_ipsa-year_return_equal)*100:.1f}%")
    
    # Crear gráfico de diagnóstico
    print(f"\n8. CREANDO GRÁFICO COMPARATIVO...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Evolución de índices
    axes[0,0].plot(cum_ipsa.index, (cum_ipsa - 1) * 100, label='IPSA Real', linewidth=2)
    axes[0,0].plot(cum_equal_weighted.index, (cum_equal_weighted - 1) * 100, label='Equal-weighted', linewidth=2)
    axes[0,0].plot(cum_cap_weighted.index, (cum_cap_weighted - 1) * 100, label='Cap-weighted', linewidth=2)
    axes[0,0].set_title('Evolución de Retornos Acumulados (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Diferencia diaria IPSA vs Equal-weighted
    diff_daily = (ipsa_returns - equal_weighted_index) * 100
    axes[0,1].plot(diff_daily.index, diff_daily)
    axes[0,1].set_title('Diferencia Diaria: IPSA vs Equal-weighted (%)')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].grid(True, alpha=0.3)
    
    # Distribución de betas
    axes[1,0].hist(betas_series, bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Distribución de Betas vs IPSA')
    axes[1,0].axvline(x=1, color='red', linestyle='--', label='Beta = 1')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Rolling correlation IPSA vs Equal-weighted
    rolling_corr = ipsa_returns.rolling(window=60).corr(equal_weighted_index)
    axes[1,1].plot(rolling_corr.index, rolling_corr)
    axes[1,1].set_title('Correlación Rolling (60 días): IPSA vs Equal-weighted')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('verificacion_datos_ipsa.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado como: verificacion_datos_ipsa.png")
    plt.show()
    
    # Resultado final
    print(f"\n=== CONCLUSIONES ===")
    return_diff = abs(cum_ipsa.iloc[-1] - cum_equal_weighted.iloc[-1]) * 100
    
    if return_diff > 20:
        print("🚨 PROBLEMA IDENTIFICADO: Gran diferencia entre IPSA y acciones individuales")
        print("   Posibles causas:")
        print("   - IPSA incluye dividendos, acciones no")
        print("   - Datos de IPSA incorrectos o de diferente fuente")
        print("   - Diferencias en ajustes por splits/dividendos")
        print(f"   - Diferencia total: {return_diff:.1f}%")
    elif return_diff > 10:
        print("⚠️  DIFERENCIA MODERADA: Puede deberse a ponderación por capitalización")
        print(f"   Diferencia: {return_diff:.1f}%")
    else:
        print("✅ DATOS CONSISTENTES: Diferencia aceptable")
        print(f"   Diferencia: {return_diff:.1f}%")
    
    return {
        'return_difference': return_diff,
        'ipsa_total_return': (cum_ipsa.iloc[-1] - 1) * 100,
        'equal_weighted_return': (cum_equal_weighted.iloc[-1] - 1) * 100,
        'cap_weighted_return': (cum_cap_weighted.iloc[-1] - 1) * 100,
        'betas': betas_series,
        'large_changes_count': len(large_changes)
    }

if __name__ == "__main__":
    resultados = verificar_datos_ipsa()
