import pandas as pd
import numpy as np

# Cargar datos
data = pd.read_excel('precios_limpios.xlsx')

# Convertir fechas
if 'DATES' in data.columns:
    data['DATES'] = pd.to_datetime(data['DATES'])
    data.set_index('DATES', inplace=True)

# Separar variable dependiente (IPSA Index)
ipsa_column = 'IPSA Index' 


ipsa_prices = data[ipsa_column].copy()


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


def robust_lasso_tracking_model(ipsa_returns, stock_returns):
    from sklearn.linear_model import Lasso, HuberRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.covariance import LedoitWolf
    from scipy.optimize import minimize
    from scipy.stats import trim_mean
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    print("Iniciando modelo LASSO robusto con optimización de alpha...")
    
    # Preparar datos
    stock_simple_returns = np.exp(stock_returns) - 1
    stock_cumulative = (1 + stock_simple_returns).cumprod()
    ipsa_simple_returns = np.exp(ipsa_returns) - 1
    ipsa_cumulative = (1 + ipsa_simple_returns).cumprod()
   
    # PREPROCESSING ROBUSTO
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(stock_returns)
    y = ipsa_returns.values
    
    # Winsorization para outliers
    def winsorize_data(data, limits=(0.05, 0.05)):
        from scipy.stats.mstats import winsorize
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            result[:, i] = winsorize(data[:, i], limits=limits)
        return result
    
    X_robust = winsorize_data(X_scaled, limits=(0.05, 0.05))
    
    # Covarianza robusta
    lw = LedoitWolf()
    robust_cov = lw.fit(stock_returns).covariance_
    
    # BÚSQUEDA DE ALPHA ÓPTIMO
    def find_optimal_alpha():
        print("\n" + "="*50)
        print("BÚSQUEDA DE ALPHA ÓPTIMO MEDIANTE VALIDACIÓN CRUZADA")
        print("="*50)
        
        alpha_candidates = [  1, 2, 3, 4 ,5 ]
        n_folds = 4
        fold_size = len(X_robust) // n_folds
        
        results = {}
        
        for alpha_base in alpha_candidates:
            print(f"\nProbando alpha = {alpha_base}...")
            fold_errors = []
            fold_diversifications = []
            
            # Validación cruzada temporal
            for fold in range(n_folds):
                train_end = (fold + 1) * fold_size
                if train_end >= len(X_robust) - 50:  # Dejar datos suficientes para test
                    break
                    
                test_start = train_end
                test_end = min(train_end + fold_size, len(X_robust))
                
                if test_end - test_start < 30:  # Mínimo 30 observaciones para test
                    continue
                
                X_train = X_robust[:train_end]
                y_train = y[:train_end]
                X_test = X_robust[test_start:test_end]
                y_test = y[test_start:test_end]
                
                # Función objetivo para validación cruzada
                def cv_objective(weights):
                    portfolio_returns = X_train @ weights
                    tracking_diff = portfolio_returns - y_train
                    
                    # Error base (MSE)
                    mse_error = np.mean(tracking_diff**2)
                    
                    # Penalizaciones con alpha
                    l1_penalty = alpha_base * np.sum(np.abs(weights))
                    l2_penalty = alpha_base * 0.0 * np.sum(weights**2)
                    
                    # Penalty de concentración
                    abs_weights = np.abs(weights)
                    if np.sum(abs_weights) > 0:
                        norm_weights = abs_weights / np.sum(abs_weights)
                        entropy = -np.sum(norm_weights * np.log(norm_weights + 1e-8))
                        max_entropy = np.log(len(weights))
                        conc_penalty = alpha_base * 0.5 * (1 - entropy / max_entropy)
                    else:
                        conc_penalty = 0
                    
                    return mse_error + l1_penalty + l2_penalty + conc_penalty
                
                # Restricciones
                def sum_constraint(w):
                    return np.sum(w) - 1.0
                
                def top10_constraint(w):
                    abs_w = np.abs(w)
                    top10 = np.sort(abs_w)[-10:]
                    print(top10[-1])
                    return np.sum(top10) - 0.6
                
                # Optimización
                bounds = [(-0.15, 0.15) for _ in range(len(X_train[0]))]
                x0 = np.ones(len(X_train[0])) / len(X_train[0])
                
                constraints = [
                    {'type': 'eq', 'fun': sum_constraint},
                    {'type': 'ineq', 'fun': top10_constraint}
                ]
                
                try:
                    result = minimize(cv_objective, x0, method='SLSQP',
                                    bounds=bounds, constraints=constraints,
                                    options={'maxiter': 3000})
                    
                    if result.success:
                        # Evaluar en conjunto de prueba
                        test_returns = X_test @ result.x
                        test_error = np.mean((test_returns - y_test)**2)
                        fold_errors.append(test_error)
                        
                        # Calcular diversificación
                        abs_w = np.abs(result.x)
                        if np.sum(abs_w) > 0:
                            norm_w = abs_w / np.sum(abs_w)
                            hhi = np.sum(norm_w**2)
                            eff_stocks = 1 / hhi
                            fold_diversifications.append(eff_stocks)
                        
                except Exception as e:
                    print(f"    Error en fold {fold}: {str(e)}")
                    continue
            
            # Guardar resultados
            if fold_errors:
                avg_error = np.mean(fold_errors)
                std_error = np.std(fold_errors) if len(fold_errors) > 1 else 0
                avg_div = np.mean(fold_diversifications) if fold_diversifications else 10
                
                # Score combinado: menor error + estabilidad + diversificación
                score = avg_error + 0.1 * std_error - 0.001 * avg_div
                
                results[alpha_base] = {
                    'avg_error': avg_error,
                    'std_error': std_error,
                    'avg_diversification': avg_div,
                    'score': score,
                    'n_folds': len(fold_errors)
                }
                
                print(f"    Error: {avg_error:.6f} ± {std_error:.6f}")
                print(f"    Diversificación: {avg_div:.1f} acciones")
                print(f"    Score: {score:.6f}")
            else:
                print(f"    No se pudo optimizar")
        
        # Seleccionar mejor alpha
        if results:
            best_alpha = min(results.keys(), key=lambda a: results[a]['score'])
            print(f"\n🎯 ALPHA ÓPTIMO SELECCIONADO: {best_alpha}")
            print(f"   Error CV: {results[best_alpha]['avg_error']:.6f}")
            print(f"   Diversificación: {results[best_alpha]['avg_diversification']:.1f}")
            return best_alpha, results
        else:
            print("\n⚠️ Usando alpha por defecto: 0.1")
            return 0.1, {}
    
    # Encontrar alpha óptimo
    optimal_alpha, alpha_results = find_optimal_alpha()
    
    # OPTIMIZACIÓN FINAL CON ALPHA ÓPTIMO
    print(f"\n" + "="*50)
    print(f"OPTIMIZACIÓN FINAL CON ALPHA = {optimal_alpha}")
    print("="*50)
    
    def final_objective(weights):
        portfolio_returns = X_robust @ weights
        tracking_diff = portfolio_returns - y
        
        # Error principal con Huber loss (robusto)
        def huber_loss(residuals, delta=0.01):
            abs_res = np.abs(residuals)
            quad = np.minimum(abs_res, delta)
            linear = abs_res - quad
            return np.sum(0.5 * quad**2 + delta * linear)
        
        huber_error = huber_loss(tracking_diff)
        
        # Penalizaciones con alpha óptimo
        l1_penalty = optimal_alpha * 1000 * np.sum(np.abs(weights))
        l2_penalty = optimal_alpha * 500 * np.sum(weights**2)
        
        # Penalty de concentración (entropía)
        abs_weights = np.abs(weights)
        if np.sum(abs_weights) > 0:
            norm_weights = abs_weights / np.sum(abs_weights)
            entropy = -np.sum(norm_weights * np.log(norm_weights + 1e-8))
            max_entropy = np.log(len(weights))
            conc_penalty = optimal_alpha * 800 * (1 - entropy / max_entropy)
        else:
            conc_penalty = 0
        
        # Risk penalty
        risk_penalty = optimal_alpha * 50 * weights.T @ robust_cov @ weights
        
        return huber_error + l1_penalty + l2_penalty + conc_penalty + risk_penalty
    
    # Restricciones finales
    def constraint_sum(w):
        return np.sum(w) - 1.0
    
    def constraint_top10(w):
        abs_w = np.abs(w)
        top10 = np.sort(abs_w)[-10:]
        return np.sum(top10) - 0.6
    
    def constraint_max_weight(w):
        return 0.15 - np.max(np.abs(w))
    
    # Optimización con múltiples inicializaciones
    bounds = [(-0.12, 0.12) for _ in range(len(stock_returns.columns))]
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_top10},
        {'type': 'ineq', 'fun': constraint_max_weight}
    ]
    
    # Múltiples puntos de inicio
    best_result = None
    best_obj = float('inf')
    
    # Inicialización 1: equiponderado
    x0_equal = np.ones(len(stock_returns.columns)) / len(stock_returns.columns)
    
    # Inicialización 2: basado en correlación con IPSA
    correlations = []
    for col in stock_returns.columns:
        corr = stock_returns[col].corr(ipsa_returns)
        correlations.append(abs(corr) if not np.isnan(corr) else 0)
    correlations = np.array(correlations)
    x0_corr = correlations / np.sum(correlations) if np.sum(correlations) > 0 else x0_equal
    
    # Inicialización 3: basado en volatilidad inversa
    volatilities = stock_returns.std().values
    inv_vol = 1 / (volatilities + 1e-8)
    x0_vol = inv_vol / np.sum(inv_vol)
    
    initializations = [x0_equal, x0_corr, x0_vol]
    
    for i, x0 in enumerate(initializations):
        try:
            print(f"Probando inicialización {i+1}/3...")
            result = minimize(final_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 5000, 'ftol': 1e-9})
            
            if result.success and result.fun < best_obj:
                best_result = result
                best_obj = result.fun
                print(f"  ✓ Exitosa (objetivo: {result.fun:.4f})")
            elif result.success:
                print(f"  ✓ Exitosa pero no mejor (objetivo: {result.fun:.4f})")
            else:
                print(f"  ✗ Falló")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    # Verificar resultado
    if best_result and best_result.success:
        lasso_weights = best_result.x
        print(f"\n✅ Optimización exitosa!")
        print(f"Objetivo final: {best_obj:.4f}")
        
        # Verificar restricciones
        abs_w = np.abs(lasso_weights)
        top10_sum = np.sum(np.sort(abs_w)[-10:])
        max_weight = np.max(abs_w)
        eff_stocks = 1 / np.sum((abs_w / np.sum(abs_w))**2)
        
        print(f"✓ Suma de pesos: {np.sum(lasso_weights):.6f}")
        print(f"✓ Top 10 suma: {top10_sum*100:.1f}% (mín: 60%)")
        print(f"✓ Peso máximo: {max_weight*100:.1f}% (máx: 15%)")
        print(f"✓ Acciones efectivas: {eff_stocks:.1f}")
        
    else:
        print("❌ Todas las optimizaciones fallaron, usando pesos equiponderados")
        lasso_weights = np.ones(len(stock_returns.columns)) / len(stock_returns.columns)
        top10_sum = 0.6  # Para evitar errores
    
    # CALCULAR RETORNOS DEL PORTAFOLIO
    lasso_weights_dict = dict(zip(stock_returns.columns, lasso_weights))
    
    # Retornos del portafolio
    portfolio_simple_returns = np.zeros(len(stock_returns))
    for stock, weight in lasso_weights_dict.items():
        portfolio_simple_returns += stock_simple_returns[stock].values * weight
    
    portfolio_simple_returns = pd.Series(portfolio_simple_returns, index=stock_returns.index)
    portfolio_cumulative = (1 + portfolio_simple_returns).cumprod()
    
    # MÉTRICAS ROBUSTAS
    tracking_diff = portfolio_simple_returns - ipsa_simple_returns
    
    robust_metrics = {
        'mean_diff': np.mean(tracking_diff),
        'median_diff': np.median(tracking_diff),
        'std_diff': np.std(tracking_diff),
        'mad_diff': np.median(np.abs(tracking_diff - np.median(tracking_diff))),
        'var_95': np.percentile(tracking_diff, 5),
        'cvar_95': np.mean(tracking_diff[tracking_diff <= np.percentile(tracking_diff, 5)])
    }
    
    # VISUALIZACIONES
    print("\nGenerando visualizaciones...")
    
    # 1. Gráfico principal de performance
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(ipsa_cumulative.index, (ipsa_cumulative - 1) * 100, 
             'red', linewidth=3, label='IPSA', alpha=0.8)
    plt.plot(portfolio_cumulative.index, (portfolio_cumulative - 1) * 100,
             'black', linewidth=3, label=f'Portafolio LASSO (α={optimal_alpha})')
    
    plt.title('Performance: Portafolio vs IPSA', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Retorno Acumulado (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. Resultados de validación cruzada
    plt.subplot(2, 2, 2)
    if alpha_results:
        alphas = list(alpha_results.keys())
        errors = [alpha_results[a]['avg_error'] for a in alphas]
        
        plt.semilogx(alphas, errors, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=optimal_alpha, color='red', linestyle='--', alpha=0.7,
                   label=f'Óptimo: {optimal_alpha}')
        plt.title('Validación Cruzada - Error vs Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('Error CV')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No hay resultados\nde validación cruzada', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Validación Cruzada')
    
    # 3. Composición del portafolio
    plt.subplot(2, 2, 3)
    
    significant_weights = [(stock, weight) for stock, weight in lasso_weights_dict.items() 
                          if abs(weight) > 0.005]
    significant_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if significant_weights:
        top_stocks = [x[0] for x in significant_weights[:10]]
        top_weights = [x[1] * 100 for x in significant_weights[:10]]
        colors = ['green' if w > 0 else 'red' for w in top_weights]
        
        bars = plt.bar(range(len(top_stocks)), top_weights, color=colors, alpha=0.7)
        plt.title('Top 10 Holdings del Portafolio')
        plt.xlabel('Acciones')
        plt.ylabel('Peso (%)')
        plt.xticks(range(len(top_stocks)), top_stocks, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Línea de referencia para Top 10
        top10_actual = sum(abs(w) for w in top_weights)
        plt.text(0.7, 0.9, f'Top 10: {top10_actual:.1f}%\n(Mín: 60%)', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", 
                facecolor='yellow', alpha=0.7))
    
    # 4. Distribución de tracking differences
    plt.subplot(2, 2, 4)
    plt.hist(tracking_diff.values, bins=30, alpha=0.7, color='skyblue', density=True)
    plt.axvline(x=robust_metrics['mean_diff'], color='red', linestyle='-', 
                label=f"Media: {robust_metrics['mean_diff']:.4f}")
    plt.axvline(x=robust_metrics['median_diff'], color='green', linestyle='--', 
                label=f"Mediana: {robust_metrics['median_diff']:.4f}")
    
    plt.title('Distribución de Tracking Differences')
    plt.xlabel('Tracking Difference')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Gráfico de verificación de restricciones
    plt.figure(figsize=(14, 6))
    
    # Distribución de pesos
    plt.subplot(1, 2, 1)
    abs_weights_sorted = np.sort(np.abs(lasso_weights))[::-1]
    plt.bar(range(1, len(abs_weights_sorted[:15])+1), abs_weights_sorted[:15] * 100,
           color=['red' if i < 10 else 'blue' for i in range(15)], alpha=0.7)
    
    plt.axvline(x=10.5, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Top 10')
    plt.title('Distribución de Pesos (Top 15)')
    plt.xlabel('Ranking')
    plt.ylabel('Peso Absoluto (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Peso acumulado
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(abs_weights_sorted)
    plt.plot(range(1, len(cumsum[:20])+1), cumsum[:20] * 100, 'bo-', linewidth=2)
    plt.axhline(y=60, color='red', linestyle='--', label='Mínimo 60%')
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.7)
    plt.scatter([10], [cumsum[9] * 100], color='red', s=100, zorder=5,
               label=f'Top 10: {cumsum[9]*100:.1f}%')
    
    plt.title('Peso Acumulado')
    plt.xlabel('Top N Acciones')
    plt.ylabel('Peso Acumulado (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20)
    
    plt.tight_layout()
    plt.show()
    
    # ESTADÍSTICAS FINALES
    final_portfolio_return = (portfolio_cumulative.iloc[-1] - 1) * 100
    final_ipsa_return = (ipsa_cumulative.iloc[-1] - 1) * 100
    
    print(f"\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"Alpha óptimo: {optimal_alpha}")
    print(f"Retorno portafolio: {final_portfolio_return:.2f}%")
    print(f"Retorno IPSA: {final_ipsa_return:.2f}%")
    print(f"Excess return: {final_portfolio_return - final_ipsa_return:.2f}%")
    
    print(f"\nMétricas robustas de tracking:")
    for metric, value in robust_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print(f"\nVerificación de restricciones:")
    print(f"  Top 10 suma: {top10_sum*100:.1f}% (≥60%)")
    print(f"  Peso máximo: {np.max(np.abs(lasso_weights))*100:.1f}% (≤15%)")
    print(f"  Acciones efectivas: {1/np.sum((np.abs(lasso_weights)/np.sum(np.abs(lasso_weights)))**2):.1f}")
    
    print(f"\nTop 10 holdings:")
    for i, (stock, weight) in enumerate(significant_weights[:10]):
        sign = '+' if weight > 0 else ''
        print(f"  {i+1:2d}. {stock:12s}: {sign}{weight*100:6.2f}%")
    
    return lasso_weights_dict, portfolio_cumulative, robust_metrics, optimal_alpha, alpha_results

# Ejemplo de uso:
weights, cumulative, metrics, alpha, results = robust_lasso_tracking_model(ipsa_returns, stock_returns)