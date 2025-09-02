import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Cargar y examinar los datos
def diagnosticar_datos():
    print("=== DIAGNÓSTICO DEL PROBLEMA ===\n")
    
    # 1. Cargar datos
    data = pd.read_excel('precios_limpios.xlsx')
    
    # Convertir fechas
    if 'DATES' in data.columns:
        data['DATES'] = pd.to_datetime(data['DATES'])
        data.set_index('DATES', inplace=True)
    
    print("1. ESTRUCTURA DE DATOS:")
    print(f"Forma de datos: {data.shape}")
    print(f"Columnas: {list(data.columns)}")
    print(f"Período: {data.index.min()} a {data.index.max()}")
    
    # 2. Identificar columnas de precios vs IPSA
    ipsa_column = None
    for col_name in ['IPSA Index', 'IPSA index', 'IPSA_Index']:
        if col_name in data.columns:
            ipsa_column = col_name
            break
    
    price_columns = [col for col in data.columns 
                    if col not in ['IPSA Index', 'IPSA index'] and not col.startswith('Unnamed')]
    
    prices = data[price_columns].dropna()
    if ipsa_column:
        ipsa_prices = data[ipsa_column].dropna()
    else:
        ipsa_prices = prices.mean(axis=1)
    
    print(f"\n2. COLUMNAS IDENTIFICADAS:")
    print(f"IPSA Column: {ipsa_column}")
    print(f"Número de acciones: {len(price_columns)}")
    print(f"Primeras 5 acciones: {price_columns[:5]}")
    
    # 3. Calcular retornos
    stock_returns = np.log(prices / prices.shift(1)).dropna()
    ipsa_returns = np.log(ipsa_prices / ipsa_prices.shift(1)).dropna()
    
    # Alinear fechas
    common_dates = stock_returns.index.intersection(ipsa_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    ipsa_returns = ipsa_returns.loc[common_dates]
    
    print(f"\n3. RETORNOS CALCULADOS:")
    print(f"Períodos de retornos: {len(stock_returns)}")
    print(f"Fecha inicio: {stock_returns.index.min()}")
    print(f"Fecha fin: {stock_returns.index.max()}")
    
    # 4. Verificar escalas de retornos
    print(f"\n4. ESTADÍSTICAS DE RETORNOS:")
    print(f"Retorno promedio IPSA (diario): {ipsa_returns.mean():.6f}")
    print(f"Volatilidad IPSA (diaria): {ipsa_returns.std():.6f}")
    print(f"Retorno promedio acciones (diario): {stock_returns.mean().mean():.6f}")
    print(f"Volatilidad promedio acciones (diaria): {stock_returns.std().mean():.6f}")
    
    # 5. Simular una regresión simple y verificar
    print(f"\n5. PRUEBA DE REGRESIÓN SIMPLE:")
    
    # Tomar una muestra de datos para prueba
    sample_size = min(252, len(stock_returns))  # 1 año o menos
    sample_stocks = stock_returns.iloc[-sample_size:]
    sample_ipsa = ipsa_returns.iloc[-sample_size:]
    
    from sklearn.linear_model import LinearRegression
    
    # Regresión simple
    lr = LinearRegression()
    lr.fit(sample_stocks, sample_ipsa)
    weights_lr = lr.coef_
    weights_lr = weights_lr / weights_lr.sum()  # Normalizar
    
    # Calcular retornos predichos
    predicted_returns = (sample_stocks * weights_lr).sum(axis=1)
    
    # Comparar
    cumulative_pred = (1 + predicted_returns).cumprod().iloc[-1] - 1
    cumulative_ipsa = (1 + sample_ipsa).cumprod().iloc[-1] - 1
    
    print(f"Retorno acumulado IPSA (muestra): {cumulative_ipsa:.4f} ({cumulative_ipsa*100:.2f}%)")
    print(f"Retorno acumulado Predicho: {cumulative_pred:.4f} ({cumulative_pred*100:.2f}%)")
    print(f"Error de tracking: {abs(cumulative_pred - cumulative_ipsa):.4f}")
    
    # 6. Verificar normalización de datos
    print(f"\n6. VERIFICACIÓN DE NORMALIZACIÓN:")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    sample_stocks_scaled = pd.DataFrame(
        scaler.fit_transform(sample_stocks),
        index=sample_stocks.index,
        columns=sample_stocks.columns
    )
    
    # Regresión con datos normalizados
    lr_scaled = LinearRegression()
    lr_scaled.fit(sample_stocks_scaled, sample_ipsa)
    weights_lr_scaled = lr_scaled.coef_
    weights_lr_scaled = weights_lr_scaled / weights_lr_scaled.sum()
    
    predicted_returns_scaled = (sample_stocks * weights_lr_scaled).sum(axis=1)
    cumulative_pred_scaled = (1 + predicted_returns_scaled).cumprod().iloc[-1] - 1
    
    print(f"Retorno con normalización: {cumulative_pred_scaled:.4f} ({cumulative_pred_scaled*100:.2f}%)")
    
    # 7. Verificar dates de rebalanceo
    print(f"\n7. SIMULACIÓN DE FECHAS DE REBALANCEO:")
    
    def get_third_friday(year, month):
        from datetime import datetime, timedelta
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(days=14)
        return third_friday.date()
    
    # Generar algunas fechas de prueba
    start_year = stock_returns.index.min().year
    end_year = stock_returns.index.max().year
    rebalance_months = [3, 6, 9, 12]
    
    test_dates = []
    for year in range(start_year, min(start_year + 2, end_year + 1)):
        for month in rebalance_months:
            third_friday = get_third_friday(year, month)
            target_date = pd.to_datetime(third_friday)
            available_dates = stock_returns.index[stock_returns.index <= target_date]
            if len(available_dates) > 0:
                test_dates.append(available_dates[-1])
    
    print(f"Primeras fechas de rebalanceo: {test_dates[:5]}")
    
    # 8. Probar cálculo acumulativo correcto
    print(f"\n8. VERIFICACIÓN DE CÁLCULO ACUMULATIVO:")
    
    # Método correcto: usar retornos simples para acumulación
    simple_returns_ipsa = ipsa_prices.pct_change().dropna()
    simple_returns_stocks = prices.pct_change().dropna()
    
    # Alinear
    common_simple = simple_returns_ipsa.index.intersection(simple_returns_stocks.index)
    simple_returns_ipsa = simple_returns_ipsa.loc[common_simple]
    simple_returns_stocks = simple_returns_stocks.loc[common_simple]
    
    # Tomar misma muestra
    sample_simple_stocks = simple_returns_stocks.iloc[-sample_size:]
    sample_simple_ipsa = simple_returns_ipsa.iloc[-sample_size:]
    
    # Calcular acumulativo con retornos simples
    cumulative_ipsa_simple = (1 + sample_simple_ipsa).cumprod().iloc[-1] - 1
    
    print(f"IPSA con retornos simples: {cumulative_ipsa_simple:.4f} ({cumulative_ipsa_simple*100:.2f}%)")
    print(f"IPSA con retornos log: {cumulative_ipsa:.4f} ({cumulative_ipsa*100:.2f}%)")
    
    return {
        'stock_returns': stock_returns,
        'ipsa_returns': ipsa_returns,
        'simple_returns_stocks': simple_returns_stocks,
        'simple_returns_ipsa': simple_returns_ipsa,
        'prices': prices,
        'ipsa_prices': ipsa_prices
    }

if __name__ == "__main__":
    data_diagnostico = diagnosticar_datos()
