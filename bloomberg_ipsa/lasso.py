import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

def verificar_dependencias_y_archivos():
    """
    Verifica archivos de dependencias y archivos existentes
    """
    # Verificar archivos necesarios
    archivos_requeridos = ['precios_limpios.xlsx', 'retornos_diarios.xlsx']
    archivos_faltantes = [archivo for archivo in archivos_requeridos if not os.path.exists(archivo)]
    
    if archivos_faltantes:
        print("ERROR: Faltan archivos necesarios:")
        for archivo in archivos_faltantes:
            print(f"  ✗ {archivo}")
        print("\nEjecuta primero 'python trabajo.py' para generar estos archivos.")
        return False, False
    
    # Verificar archivos que se van a generar para el modelo LASSO
    archivos_lasso = [
        'modelo_lasso_ipsa.xlsx',
        'ranking_acciones_lasso.xlsx',
        'resultados_lasso_ipsa.xlsx'
    ]
    
    archivos_graficos_lasso = [
        'evolucion_pesos_lasso.png',
        'tracking_lasso_ipsa.png',
        'ranking_importancia_acciones.png'
    ]
    
    archivos_existentes = [archivo for archivo in archivos_lasso + archivos_graficos_lasso if os.path.exists(archivo)]
    
    sobrescribir = True
    if archivos_existentes:
        print(f"\n=== ARCHIVOS DEL MODELO LASSO EXISTENTES ===")
        for archivo in archivos_existentes:
            print(f"  ✓ {archivo}")
        
        respuesta = input("\n¿Deseas sobrescribir los archivos del modelo LASSO existentes? (s/n): ").lower().strip()
        sobrescribir = respuesta in ['s', 'si', 'sí', 'y', 'yes']
    
    return True, sobrescribir

def limpiar_nombres_columnas(df):
    """
    Limpia los nombres de las columnas eliminando el sufijo _px_last
    """
    # Crear diccionario de mapeo para renombrar columnas
    rename_dict = {}
    for col in df.columns:
        if col.endswith('_px_last'):
            nuevo_nombre = col.replace('_px_last', '')
            rename_dict[col] = nuevo_nombre
        else:
            rename_dict[col] = col
    
    # Renombrar columnas
    df_renamed = df.rename(columns=rename_dict)
    return df_renamed

def cargar_datos_limpios():
    """
    Carga los datos ya procesados
    """
    print("Cargando datos procesados...")
    
    try:
        df_precios = pd.read_excel('precios_limpios.xlsx', index_col=0)
        df_retornos = pd.read_excel('retornos_diarios.xlsx', index_col=0)
        
        # Limpiar nombres de columnas
        df_precios = limpiar_nombres_columnas(df_precios)
        df_retornos = limpiar_nombres_columnas(df_retornos)
        
        print(f"✓ Precios cargados: {df_precios.shape}")
        print(f"✓ Retornos cargados: {df_retornos.shape}")
        
        # Mostrar información de fechas
        print(f"✓ Período de datos: {df_retornos.index.min().strftime('%Y-%m-%d')} a {df_retornos.index.max().strftime('%Y-%m-%d')}")
        print(f"✓ Total de acciones: {len(df_retornos.columns)}")
        print(f"✓ Nombres de acciones (primeras 5): {list(df_retornos.columns[:5])}")
        
        return df_precios, df_retornos
    except FileNotFoundError as e:
        print(f"Error: No se encontraron los archivos procesados. Ejecuta primero trabajo.py")
        raise e

def cargar_matriz_existente():
    """
    Intenta cargar matriz de correlación existente para comparación
    """
    try:
        if os.path.exists('matriz_correlacion_sin_ipsa_detallada.xlsx'):
            matriz_existente = pd.read_excel('matriz_correlacion_sin_ipsa_detallada.xlsx', index_col=0)
            print(f"✓ Matriz de correlación existente encontrada: {matriz_existente.shape}")
            return matriz_existente
    except:
        pass
    return None

def preparar_datos_lasso(df_retornos):
    """
    Prepara los datos para el modelo LASSO
    """
    print(f"\n=== PREPARACIÓN DE DATOS PARA MODELO LASSO ===")
    
    # Identificar columna IPSA
    columnas_ipsa = [col for col in df_retornos.columns if 'IPSA' in col.upper()]
    
    if not columnas_ipsa:
        print("ERROR: No se encontró columna IPSA en los datos")
        return None, None, None
    
    ipsa_col = columnas_ipsa[0]
    print(f"✓ Columna IPSA identificada: {ipsa_col}")
    
    # Separar variable dependiente (IPSA) de las independientes (acciones)
    acciones_cols = [col for col in df_retornos.columns if col != ipsa_col]
    
    y = df_retornos[ipsa_col]  # Variable dependiente
    X = df_retornos[acciones_cols]  # Variables independientes
    
    # Eliminar valores faltantes
    datos_completos = pd.concat([X, y], axis=1).dropna()
    X_clean = datos_completos[acciones_cols]
    y_clean = datos_completos[ipsa_col]
    
    print(f"✓ Datos preparados:")
    print(f"  - Acciones (variables independientes): {X_clean.shape[1]}")
    print(f"  - Observaciones con datos completos: {len(X_clean)}")
    print(f"  - Período: {X_clean.index.min().strftime('%Y-%m-%d')} a {X_clean.index.max().strftime('%Y-%m-%d')}")
    
    return X_clean, y_clean, acciones_cols

def generar_fechas_recalibracion(X, y, frecuencia_meses=1):
    """
    Genera las fechas de recalibración mensual
    """
    fecha_inicio = X.index.min()
    fecha_fin = X.index.max()
    
    fechas_recalibracion = []
    fecha_actual = fecha_inicio
    
    while fecha_actual <= fecha_fin:
        # Buscar la primera fecha disponible en ese mes
        inicio_mes = fecha_actual.replace(day=1)
        siguiente_mes = (inicio_mes + timedelta(days=32)).replace(day=1)
        
        # Filtrar fechas en ese mes
        fechas_mes = X.index[(X.index >= inicio_mes) & (X.index < siguiente_mes)]
        
        if len(fechas_mes) > 0:
            fechas_recalibracion.append(fechas_mes[0])
        
        # Avanzar al siguiente mes
        fecha_actual = siguiente_mes
    
    print(f"✓ Fechas de recalibración generadas: {len(fechas_recalibracion)} puntos")
    return fechas_recalibracion

def optimizar_portafolio_lasso_restringido(X_train, y_train, max_stocks=10, lambda_l1=0.01):
    """
    Optimiza un portafolio con restricciones LASSO:
    - Pesos suman 1
    - Máximo 10 acciones
    - Regularización L1 para sparsity
    """
    n_stocks = X_train.shape[1]
    
    def objective(weights):
        # Predicción del modelo
        pred = np.dot(X_train, weights)
        # Error cuadrático medio + penalización L1
        mse = np.mean((y_train - pred) ** 2)
        l1_penalty = lambda_l1 * np.sum(np.abs(weights))
        return mse + l1_penalty
    
    # Restricciones
    constraints = [
        # Los pesos deben sumar 1
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]
    
    # Bounds: pesos entre 0 y 1
    bounds = [(0, 1) for _ in range(n_stocks)]
    
    # Punto inicial: pesos iguales
    x0 = np.ones(n_stocks) / n_stocks
    
    # Múltiples inicializaciones para evitar mínimos locales
    best_result = None
    best_objective = float('inf')
    
    for i in range(5):  # 5 intentos con diferentes inicializaciones
        if i == 0:
            # Primera inicialización: pesos iguales
            x0_try = np.ones(n_stocks) / n_stocks
        else:
            # Inicializaciones aleatorias
            x0_try = np.random.dirichlet(np.ones(n_stocks))
        
        try:
            result = minimize(
                objective, 
                x0_try, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success and result.fun < best_objective:
                best_result = result
                best_objective = result.fun
                
        except Exception as e:
            continue
    
    if best_result is None or not best_result.success:
        # Si la optimización falla, usar solución simple
        print("  Advertencia: Optimización falló, usando solución simple")
        weights = np.zeros(n_stocks)
        # Seleccionar las acciones más correlacionadas con IPSA
        correlaciones = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(n_stocks)])
        top_indices = np.argsort(np.abs(correlaciones))[-max_stocks:]
        weights[top_indices] = 1.0 / len(top_indices)
        return weights
    
    weights = best_result.x
    
    # Aplicar restricción de máximo número de acciones
    # Mantener solo los 'max_stocks' pesos más grandes
    if np.sum(weights > 1e-6) > max_stocks:
        # Identificar los top max_stocks
        top_indices = np.argsort(weights)[-max_stocks:]
        new_weights = np.zeros(n_stocks)
        new_weights[top_indices] = weights[top_indices]
        # Renormalizar para que sumen 1
        new_weights = new_weights / np.sum(new_weights)
        weights = new_weights
    
    return weights

def entrenar_modelo_lasso_mensual(X, y, acciones_cols, ventana_entrenamiento=252, max_stocks=10):
    """
    Entrena modelo LASSO con restricciones de pesos y recalibración mensual
    """
    print(f"\n=== ENTRENAMIENTO MODELO LASSO RESTRINGIDO ===")
    print(f"✓ Restricciones:")
    print(f"  - Pesos suman 1")
    print(f"  - Máximo {max_stocks} acciones seleccionadas")
    print(f"  - Ventana de entrenamiento: {ventana_entrenamiento} días")
    
    fechas_recalibracion = generar_fechas_recalibracion(X, y)
    
    # Almacenar resultados
    resultados = {
        'fecha': [],
        'lambda_l1': [],
        'r2_train': [],
        'r2_test': [],
        'mse_train': [],
        'mse_test': [],
        'n_acciones_seleccionadas': [],
        'suma_pesos': [],
        'prediccion_ipsa': [],
        'retorno_real_ipsa': []
    }
    
    # Almacenar pesos de cada recalibración
    pesos_historicos = pd.DataFrame(index=fechas_recalibracion, columns=acciones_cols)
    pesos_historicos = pesos_historicos.fillna(0)
    
    scaler = StandardScaler()
    
    # Probar diferentes valores de lambda
    lambdas_to_try = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    for i, fecha_recal in enumerate(fechas_recalibracion[1:], 1):  # Empezar desde el segundo mes
        try:
            # Definir ventana de entrenamiento
            fecha_inicio_train = X.index[max(0, X.index.get_loc(fecha_recal) - ventana_entrenamiento)]
            fecha_fin_train = fecha_recal
            
            # Datos de entrenamiento
            X_train = X.loc[fecha_inicio_train:fecha_fin_train]
            y_train = y.loc[fecha_inicio_train:fecha_fin_train]
            
            # Verificar que hay suficientes datos
            if len(X_train) < 60:  # Mínimo 60 días de datos
                continue
            
            # Normalizar datos
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Buscar el mejor lambda mediante validación
            best_lambda = None
            best_r2 = -float('inf')
            
            for lambda_l1 in lambdas_to_try:
                # Validación cruzada simple (split temporal)
                split_point = int(len(X_train_scaled) * 0.8)
                X_val_train = X_train_scaled[:split_point]
                y_val_train = y_train.iloc[:split_point]
                X_val_test = X_train_scaled[split_point:]
                y_val_test = y_train.iloc[split_point:]
                
                if len(X_val_test) < 10:  # Muy pocos datos para validar
                    continue
                
                try:
                    weights = optimizar_portafolio_lasso_restringido(
                        X_val_train, y_val_train, max_stocks, lambda_l1
                    )
                    
                    # Predicción en validación
                    y_pred_val = np.dot(X_val_test, weights)
                    r2_val = r2_score(y_val_test, y_pred_val)
                    
                    if r2_val > best_r2:
                        best_r2 = r2_val
                        best_lambda = lambda_l1
                        
                except Exception:
                    continue
            
            # Si no se encontró un buen lambda, usar el default
            if best_lambda is None:
                best_lambda = 0.01
            
            # Entrenar modelo final con todos los datos y mejor lambda
            weights_final = optimizar_portafolio_lasso_restringido(
                X_train_scaled, y_train, max_stocks, best_lambda
            )
            
            # Predicción en conjunto de entrenamiento
            y_pred_train = np.dot(X_train_scaled, weights_final)
            
            # Fecha de predicción (siguiente día hábil)
            idx_actual = X.index.get_loc(fecha_recal)
            if idx_actual + 1 < len(X):
                fecha_pred = X.index[idx_actual + 1]
                X_test = X.loc[fecha_pred:fecha_pred]
                y_test = y.loc[fecha_pred:fecha_pred]
                
                if len(X_test) > 0:
                    X_test_scaled = scaler.transform(X_test)
                    y_pred_test = np.dot(X_test_scaled, weights_final)
                    
                    # Verificar restricciones
                    suma_pesos = np.sum(weights_final)
                    n_acciones_sel = np.sum(weights_final > 1e-6)
                    
                    # Almacenar resultados
                    resultados['fecha'].append(fecha_pred)
                    resultados['lambda_l1'].append(best_lambda)
                    resultados['r2_train'].append(r2_score(y_train, y_pred_train))
                    resultados['r2_test'].append(r2_score(y_test, y_pred_test))
                    resultados['mse_train'].append(mean_squared_error(y_train, y_pred_train))
                    resultados['mse_test'].append(mean_squared_error(y_test, y_pred_test))
                    resultados['n_acciones_seleccionadas'].append(n_acciones_sel)
                    resultados['suma_pesos'].append(suma_pesos)
                    resultados['prediccion_ipsa'].append(y_pred_test[0])
                    resultados['retorno_real_ipsa'].append(y_test.iloc[0])
                    
                    # Almacenar pesos
                    pesos_historicos.loc[fecha_recal, :] = weights_final
            
            if i % 12 == 0:  # Progreso cada año
                print(f"  Procesadas {i} recalibraciones...")
                print(f"    - Promedio acciones seleccionadas: {np.mean([r for r in resultados['n_acciones_seleccionadas'] if r <= max_stocks]):.1f}")
                print(f"    - Promedio suma de pesos: {np.mean(resultados['suma_pesos']):.4f}")
                
        except Exception as e:
            print(f"  Error en fecha {fecha_recal}: {str(e)}")
            continue
    
    print(f"✓ Modelo LASSO restringido entrenado con {len(resultados['fecha'])} predicciones")
    
    # Verificación final de restricciones
    if len(resultados['n_acciones_seleccionadas']) > 0:
        max_acciones_usadas = max(resultados['n_acciones_seleccionadas'])
        prom_suma_pesos = np.mean(resultados['suma_pesos'])
        print(f"✓ Verificación de restricciones:")
        print(f"  - Máximo acciones seleccionadas: {max_acciones_usadas} (límite: {max_stocks})")
        print(f"  - Promedio suma de pesos: {prom_suma_pesos:.6f} (objetivo: 1.0)")
    
    return resultados, pesos_historicos

def analizar_importancia_acciones(pesos_historicos):
    """
    Analiza la importancia de las acciones basado en los pesos del modelo LASSO restringido
    """
    print(f"\n=== ANÁLISIS DE IMPORTANCIA DE ACCIONES (MODELO RESTRINGIDO) ===")
    
    # Calcular métricas de importancia
    importancia_acciones = pd.DataFrame(index=pesos_historicos.columns)
    
    # Peso promedio (ya son pesos positivos que suman 1)
    importancia_acciones['peso_promedio'] = pesos_historicos.mean()
    
    # Peso promedio solo cuando es seleccionada (peso > 0)
    importancia_acciones['peso_promedio_cuando_seleccionada'] = pesos_historicos.replace(0, np.nan).mean()
    
    # Frecuencia de selección (% de veces que el peso > 0)
    importancia_acciones['frecuencia_seleccion'] = (pesos_historicos > 1e-6).mean()
    
    # Peso máximo
    importancia_acciones['peso_maximo'] = pesos_historicos.max()
    
    # Volatilidad de los pesos (solo cuando es seleccionada)
    importancia_acciones['volatilidad_pesos'] = pesos_historicos.replace(0, np.nan).std()
    
    # Score combinado de importancia (ajustado para pesos que suman 1)
    importancia_acciones['score_importancia'] = (
        importancia_acciones['peso_promedio'] * 0.5 +
        importancia_acciones['frecuencia_seleccion'] * 0.3 +
        (importancia_acciones['peso_maximo'] / 1.0) * 0.2  # Normalizado por el máximo posible
    )
    
    # Llenar NaN con 0
    importancia_acciones = importancia_acciones.fillna(0)
    
    # Ordenar por importancia
    ranking = importancia_acciones.sort_values('score_importancia', ascending=False)
    
    print(f"✓ Ranking de importancia calculado para {len(ranking)} acciones")
    print(f"\nTOP 10 ACCIONES MÁS IMPORTANTES:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Acción':<8} {'Score':<8} {'Peso Prom':<10} {'Freq Sel':<10} {'Peso Max':<10} {'Peso c/Sel':<12}")
    print("-" * 100)
    
    for i, (accion, datos) in enumerate(ranking.head(10).iterrows(), 1):
        peso_cuando_sel = datos['peso_promedio_cuando_seleccionada']
        peso_cuando_sel_str = f"{peso_cuando_sel:.4f}" if not np.isnan(peso_cuando_sel) else "N/A"
        
        print(f"{i:2d}.  {accion:<8} {datos['score_importancia']:.4f}   "
              f"{datos['peso_promedio']:.4f}     {datos['frecuencia_seleccion']:.2%}     "
              f"{datos['peso_maximo']:.4f}     {peso_cuando_sel_str}")
    
    # Estadísticas del portafolio
    print(f"\n=== ESTADÍSTICAS DEL PORTAFOLIO ===")
    n_acciones_promedio = (pesos_historicos > 1e-6).sum(axis=1).mean()
    concentracion_promedio = pesos_historicos.max(axis=1).mean()
    
    print(f"Número promedio de acciones en portafolio: {n_acciones_promedio:.1f}")
    print(f"Concentración promedio (peso máximo): {concentracion_promedio:.2%}")
    print(f"Acciones que aparecen en >50% de portafolios: {(ranking['frecuencia_seleccion'] > 0.5).sum()}")
    print(f"Acciones que aparecen en >25% de portafolios: {(ranking['frecuencia_seleccion'] > 0.25).sum()}")
    
    return ranking

def graficar_evolucion_pesos(pesos_historicos, ranking, top_n=10):
    """
    Grafica la evolución de los pesos de las acciones más importantes
    """
    print(f"\n=== GRAFICANDO EVOLUCIÓN DE PESOS ===")
    
    # Seleccionar top N acciones más importantes
    top_acciones = ranking.head(top_n).index
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Evolución de pesos de top acciones
    ax1 = axes[0, 0]
    for accion in top_acciones[:5]:  # Top 5
        ax1.plot(pesos_historicos.index, pesos_historicos[accion], 
                label=accion, linewidth=2, alpha=0.8)
    ax1.set_title(f'Evolución de Pesos - Top 5 Acciones', fontsize=14)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Peso en Modelo LASSO')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # Gráfico 2: Heatmap de pesos
    ax2 = axes[0, 1]
    pesos_top = pesos_historicos[top_acciones].T
    im = ax2.imshow(pesos_top.values, aspect='auto', cmap='RdBu_r', 
                    interpolation='nearest')
    ax2.set_title('Heatmap de Pesos - Top 10 Acciones', fontsize=14)
    ax2.set_yticks(range(len(top_acciones)))
    ax2.set_yticklabels(top_acciones)
    ax2.set_xlabel('Tiempo (recalibraciones)')
    plt.colorbar(im, ax=ax2)
    
    # Gráfico 3: Distribución de frecuencia de selección
    ax3 = axes[1, 0]
    ranking_plot = ranking.head(15)
    bars = ax3.barh(range(len(ranking_plot)), ranking_plot['frecuencia_seleccion'], 
                    color='skyblue', alpha=0.8)
    ax3.set_yticks(range(len(ranking_plot)))
    ax3.set_yticklabels(ranking_plot.index)
    ax3.set_xlabel('Frecuencia de Selección')
    ax3.set_title('Frecuencia de Selección - Top 15 Acciones', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2%}', ha='left', va='center', fontsize=9)
    
    # Gráfico 4: Score de importancia
    ax4 = axes[1, 1]
    bars4 = ax4.barh(range(len(ranking_plot)), ranking_plot['score_importancia'], 
                     color='lightcoral', alpha=0.8)
    ax4.set_yticks(range(len(ranking_plot)))
    ax4.set_yticklabels(ranking_plot.index)
    ax4.set_xlabel('Score de Importancia')
    ax4.set_title('Score de Importancia - Top 15 Acciones', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('evolucion_pesos_lasso.png', dpi=300, bbox_inches='tight')
    plt.show()

def graficar_performance_modelo(resultados):
    """
    Grafica la performance del modelo LASSO restringido
    """
    print(f"\n=== GRAFICANDO PERFORMANCE DEL MODELO RESTRINGIDO ===")
    
    resultados_df = pd.DataFrame(resultados)
    resultados_df['fecha'] = pd.to_datetime(resultados_df['fecha'])
    resultados_df.set_index('fecha', inplace=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Gráfico 1: Predicciones vs Reales
    ax1 = axes[0, 0]
    ax1.plot(resultados_df.index, resultados_df['retorno_real_ipsa'], 
             label='IPSA Real', color='blue', alpha=0.7)
    ax1.plot(resultados_df.index, resultados_df['prediccion_ipsa'], 
             label='IPSA Predicho', color='red', alpha=0.7)
    ax1.set_title('Predicciones vs Retornos Reales IPSA', fontsize=14)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Retorno Diario')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Gráfico 2: R² del modelo
    ax2 = axes[0, 1]
    ax2.plot(resultados_df.index, resultados_df['r2_test'], 
             color='green', linewidth=2)
    ax2.set_title('Evolución R² del Modelo', fontsize=14)
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('R² Test')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Gráfico 3: Número de acciones seleccionadas
    ax3 = axes[0, 2]
    ax3.plot(resultados_df.index, resultados_df['n_acciones_seleccionadas'], 
             color='purple', linewidth=2)
    ax3.set_title('Número de Acciones Seleccionadas', fontsize=14)
    ax3.set_xlabel('Fecha')
    ax3.set_ylabel('Número de Acciones')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Límite (10)')
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Gráfico 4: Error de tracking
    ax4 = axes[1, 0]
    tracking_error = resultados_df['retorno_real_ipsa'] - resultados_df['prediccion_ipsa']
    ax4.plot(resultados_df.index, tracking_error, color='orange', alpha=0.7)
    ax4.set_title('Error de Tracking', fontsize=14)
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('Error (Real - Predicho)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Gráfico 5: Suma de pesos (verificación restricción)
    ax5 = axes[1, 1]
    ax5.plot(resultados_df.index, resultados_df['suma_pesos'], 
             color='brown', linewidth=2)
    ax5.set_title('Verificación: Suma de Pesos = 1', fontsize=14)
    ax5.set_xlabel('Fecha')
    ax5.set_ylabel('Suma de Pesos')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Objetivo (1.0)')
    ax5.legend()
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Gráfico 6: Distribución de número de acciones
    ax6 = axes[1, 2]
    ax6.hist(resultados_df['n_acciones_seleccionadas'], bins=10, 
             alpha=0.7, color='skyblue', edgecolor='black')
    ax6.set_title('Distribución Número de Acciones', fontsize=14)
    ax6.set_xlabel('Número de Acciones')
    ax6.set_ylabel('Frecuencia')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Límite (10)')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('tracking_lasso_ipsa.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estadísticas de performance
    r2_promedio = resultados_df['r2_test'].mean()
    mse_promedio = resultados_df['mse_test'].mean()
    acciones_promedio = resultados_df['n_acciones_seleccionadas'].mean()
    suma_pesos_promedio = resultados_df['suma_pesos'].mean()
    suma_pesos_std = resultados_df['suma_pesos'].std()
    
    print(f"\n=== ESTADÍSTICAS DE PERFORMANCE ===")
    print(f"R² promedio: {r2_promedio:.4f}")
    print(f"MSE promedio: {mse_promedio:.6f}")
    print(f"Número promedio de acciones seleccionadas: {acciones_promedio:.1f}")
    print(f"Error de tracking promedio: {tracking_error.abs().mean():.6f}")
    print(f"Volatilidad del error de tracking: {tracking_error.std():.6f}")
    print(f"\n=== VERIFICACIÓN DE RESTRICCIONES ===")
    print(f"Suma de pesos promedio: {suma_pesos_promedio:.6f} (objetivo: 1.0)")
    print(f"Desviación estándar suma de pesos: {suma_pesos_std:.6f}")
    print(f"Máximo número de acciones usado: {resultados_df['n_acciones_seleccionadas'].max()} (límite: 10)")
    print(f"Casos donde se violó restricción de 10 acciones: {(resultados_df['n_acciones_seleccionadas'] > 10).sum()}")
    print(f"Casos donde suma de pesos se desvía >1%: {(abs(resultados_df['suma_pesos'] - 1.0) > 0.01).sum()}")

def guardar_resultados_lasso(resultados, pesos_historicos, ranking):
    """
    Guarda todos los resultados del modelo LASSO
    """
    print(f"\n=== GUARDANDO RESULTADOS DEL MODELO LASSO ===")
    
    # Guardar resultados del modelo
    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_excel('resultados_lasso_ipsa.xlsx', index=False)
    print("✓ Resultados del modelo guardados en: resultados_lasso_ipsa.xlsx")
    
    # Guardar pesos históricos
    pesos_historicos.to_excel('modelo_lasso_ipsa.xlsx')
    print("✓ Pesos históricos guardados en: modelo_lasso_ipsa.xlsx")
    
    # Guardar ranking de acciones
    ranking.to_excel('ranking_acciones_lasso.xlsx')
    print("✓ Ranking de acciones guardado en: ranking_acciones_lasso.xlsx")

def main():
    """
    Función principal para el modelo LASSO de predicción del IPSA
    """
    print("="*70)
    print("MODELO LASSO PARA PREDICCIÓN DEL IPSA - RECALIBRACIÓN MENSUAL")
    print("="*70)
    
    try:
        # Verificar dependencias y archivos existentes
        dependencias_ok, sobrescribir = verificar_dependencias_y_archivos()
        
        if not dependencias_ok:
            return
        
        if not sobrescribir:
            print("Operación cancelada por el usuario.")
            print("Los archivos existentes se mantienen sin cambios.")
            return
        
        # Cargar datos
        df_precios, df_retornos = cargar_datos_limpios()
        
        # Preparar datos para LASSO
        X, y, acciones_cols = preparar_datos_lasso(df_retornos)
        
        if X is None:
            return
        
        # Entrenar modelo LASSO con recalibración mensual y restricciones
        resultados, pesos_historicos = entrenar_modelo_lasso_mensual(X, y, acciones_cols, max_stocks=10)
        
        # Analizar importancia de acciones
        ranking = analizar_importancia_acciones(pesos_historicos)
        
        # Graficar evolución de pesos
        graficar_evolucion_pesos(pesos_historicos, ranking)
        
        # Graficar performance del modelo
        graficar_performance_modelo(resultados)
        
        # Guardar resultados
        guardar_resultados_lasso(resultados, pesos_historicos, ranking)
        
        print(f"\n" + "="*70)
        print("MODELO LASSO COMPLETADO")
        print("="*70)
        print("Archivos generados:")
        print("• resultados_lasso_ipsa.xlsx")
        print("• modelo_lasso_ipsa.xlsx")
        print("• ranking_acciones_lasso.xlsx")
        print("\nGráficos generados:")
        print("• evolucion_pesos_lasso.png")
        print("• tracking_lasso_ipsa.png")
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
