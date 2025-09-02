import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class BaseIPSATracker:
    """
    Clase base para todos los modelos de tracking del IPSA
    Contiene funcionalidad común para cargar datos, calcular retornos y crear visualizaciones
    """
    
    def __init__(self, data_path='precios_limpios.xlsx'):
        self.data_path = data_path
        self.data = None
        self.returns_data = None
        self.scaler = StandardScaler()
        self.current_weights = None
        self.rebalance_dates = []
        self.tracking_results = []
        
    def load_data(self):
        """Cargar y preparar los datos de precios"""
        print("Cargando datos...")
        self.data = pd.read_excel(self.data_path)
        
        # Convertir la columna de fechas
        if 'DATES' in self.data.columns:
            self.data['DATES'] = pd.to_datetime(self.data['DATES'])
            self.data.set_index('DATES', inplace=True)
        
        # Eliminar columnas que no sean precios de acciones (excluir IPSA Index y columnas no deseadas)
        price_columns = [col for col in self.data.columns 
                        if col not in ['IPSA Index', 'IPSA index'] and not col.startswith('Unnamed')]
        
        self.prices = self.data[price_columns].copy()
        
        # Obtener el índice IPSA si está disponible (probar diferentes nombres)
        ipsa_column = None
        for col_name in ['IPSA Index', 'IPSA index', 'IPSA_Index']:
            if col_name in self.data.columns:
                ipsa_column = col_name
                break
        
        if ipsa_column:
            self.ipsa_prices = self.data[ipsa_column].copy()
            print(f"IPSA encontrado en columna: {ipsa_column}")
        else:
            print("Advertencia: No se encontró la columna del IPSA")
            # Crear un índice sintético como promedio ponderado equiponderado
            self.ipsa_prices = self.prices.mean(axis=1)
        
        # Eliminar filas con valores faltantes
        self.prices = self.prices.dropna()
        self.ipsa_prices = self.ipsa_prices.loc[self.prices.index]
        
        print(f"Datos cargados: {len(self.prices)} observaciones, {len(self.prices.columns)} acciones")
        print(f"Período: {self.prices.index.min()} a {self.prices.index.max()}")
        
    def calculate_returns(self):
        """Calcular retornos logarítmicos"""
        self.stock_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        self.ipsa_returns = np.log(self.ipsa_prices / self.ipsa_prices.shift(1)).dropna()
        
        # Alinear las fechas
        common_dates = self.stock_returns.index.intersection(self.ipsa_returns.index)
        self.stock_returns = self.stock_returns.loc[common_dates]
        self.ipsa_returns = self.ipsa_returns.loc[common_dates]
        
        print(f"Retornos calculados para {len(common_dates)} períodos")
    
    def get_third_friday(self, year, month):
        """Obtener el tercer viernes de un mes específico"""
        # Primer día del mes
        first_day = datetime(year, month, 1)
        
        # Encontrar el primer viernes
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Tercer viernes
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday.date()
    
    def get_fourth_week_start(self, year, month):
        """Obtener el primer día de la 4ta semana de un mes específico"""
        # Primer día del mes
        first_day = datetime(year, month, 1)
        
        # Encontrar el primer lunes del mes
        days_until_monday = (7 - first_day.weekday()) % 7
        if first_day.weekday() == 0:  # Si el primer día es lunes
            days_until_monday = 0
        first_monday = first_day + timedelta(days=days_until_monday)
        
        # La 4ta semana comienza 3 semanas después del primer lunes
        fourth_week_start = first_monday + timedelta(weeks=3)
        
        return fourth_week_start.date()
    
    def optimize_unrestricted_weights(self, X_train, y_train):
        """
        Optimizar pesos SIN restricciones usando diferentes métodos para minimizar tracking error
        """
        best_weights = None
        best_score = float('inf')
        best_method = None
        
        # Método 1: Regresión lineal simple (OLS)
        try:
            ols = LinearRegression()
            ols.fit(X_train, y_train)
            weights_ols = ols.coef_
            
            # Normalizar para que sumen 1
            if abs(weights_ols.sum()) > 1e-10:
                weights_ols = weights_ols / weights_ols.sum()
            
                # Calcular tracking error
                predictions = X_train @ weights_ols
                mse = np.mean((y_train - predictions) ** 2)
                
                if mse < best_score:
                    best_score = mse
                    best_weights = weights_ols.copy()
                    best_method = "OLS"
        except:
            pass
            
        # Método 2: Ridge con diferentes alphas
        alphas_ridge = [0.001, 0.01, 0.1, 1.0, 10.0]
        for alpha in alphas_ridge:
            try:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train, y_train)
                weights_ridge = ridge.coef_
                
                # Normalizar para que sumen 1
                if abs(weights_ridge.sum()) > 1e-10:
                    weights_ridge = weights_ridge / weights_ridge.sum()
                
                    # Calcular tracking error
                    predictions = X_train @ weights_ridge
                    mse = np.mean((y_train - predictions) ** 2)
                    
                    if mse < best_score:
                        best_score = mse
                        best_weights = weights_ridge.copy()
                        best_method = f"Ridge(α={alpha})"
            except:
                continue
                
        # Método 3: ElasticNet con diferentes parámetros
        alphas_elastic = [0.001, 0.01, 0.1, 1.0]
        l1_ratios = [0.1, 0.5, 0.7, 0.9]
        
        for alpha in alphas_elastic:
            for l1_ratio in l1_ratios:
                try:
                    elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
                    elastic.fit(X_train, y_train)
                    weights_elastic = elastic.coef_
                    
                    # Normalizar para que sumen 1
                    if abs(weights_elastic.sum()) > 1e-10:
                        weights_elastic = weights_elastic / weights_elastic.sum()
                    
                        # Calcular tracking error
                        predictions = X_train @ weights_elastic
                        mse = np.mean((y_train - predictions) ** 2)
                        
                        if mse < best_score:
                            best_score = mse
                            best_weights = weights_elastic.copy()
                            best_method = f"ElasticNet(α={alpha}, l1={l1_ratio})"
                except:
                    continue
        
        # Si no se encuentra solución, usar equiponderado
        if best_weights is None:
            print("Advertencia: Usando pesos equiponderados como fallback")
            best_weights = np.ones(len(X_train.columns)) / len(X_train.columns)
            best_method = "Equiponderado"
        
        # Verificación final
        assert abs(best_weights.sum() - 1.0) < 1e-8, f"ERROR: Los pesos no suman 1: {best_weights.sum()}"
        
        return best_weights, best_method
    
    def get_summary_statistics(self):
        """Obtener estadísticas resumen del backtest"""
        if not self.tracking_results:
            print("No hay resultados disponibles. Ejecutar backtest primero.")
            return None
        
        # Calcular estadísticas agregadas
        avg_tracking_error = np.mean([r['tracking_error'] for r in self.tracking_results])
        avg_max_weight = np.mean([r['max_weight'] for r in self.tracking_results])
        avg_negative_weights = np.mean([r['negative_weights'] for r in self.tracking_results])
        avg_active_stocks = np.mean([r['active_stocks'] for r in self.tracking_results])
        total_excess_return = np.sum([r['excess_return'] for r in self.tracking_results])
        
        summary = {
            'total_rebalances': len(self.tracking_results),
            'avg_tracking_error': avg_tracking_error,
            'avg_max_weight': avg_max_weight,
            'avg_negative_weights': avg_negative_weights,
            'avg_active_stocks': avg_active_stocks,
            'total_excess_return': total_excess_return
        }
        
        return summary
    
    def get_current_weights(self):
        """Obtener los pesos actuales (última recalibración)"""
        if self.tracking_results:
            latest_result = self.tracking_results[-1]
            weights_dict = dict(zip(self.stock_returns.columns, latest_result['weights']))
            return pd.Series(weights_dict).sort_values(ascending=False)
        return None
    
    def create_annual_returns_visualization(self, portfolio_series, ipsa_series, model_name, filename_suffix):
        """
        Crear visualización de retornos anuales reiniciados
        """
        print("Creando gráfico de retornos anuales reiniciados...")
        
        # Crear una nueva figura para el gráfico anual
        fig = plt.figure(figsize=(20, 12))
        
        # Crear subplot para el gráfico anual
        ax1 = plt.subplot(2, 1, 1)
        
        # Calcular retornos anuales reiniciados
        portfolio_df = pd.DataFrame({'returns': portfolio_series.pct_change().fillna(0)}, index=portfolio_series.index)
        ipsa_df = pd.DataFrame({'returns': ipsa_series.pct_change().fillna(0)}, index=ipsa_series.index)
        
        # Agrupar por año y calcular retornos acumulados por año
        portfolio_df['year'] = portfolio_df.index.year
        ipsa_df['year'] = ipsa_df.index.year
        
        # Colores para diferentes años
        years = sorted(portfolio_df['year'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
        
        for i, year in enumerate(years):
            # Filtrar datos por año
            port_year = portfolio_df[portfolio_df['year'] == year]
            ipsa_year = ipsa_df[ipsa_df['year'] == year]
            
            # Calcular retornos acumulados reiniciando en 1 para cada año
            port_cumulative_year = (1 + port_year['returns']).cumprod()
            ipsa_cumulative_year = (1 + ipsa_year['returns']).cumprod()
            
            # Crear índice de días del año (1 a N)
            days_in_year = range(1, len(port_cumulative_year) + 1)
            
            # Plotear líneas para este año
            plt.plot(days_in_year, (port_cumulative_year - 1) * 100, 
                    color=colors[i], linewidth=2, linestyle='-', 
                    label=f'Portafolio {year}', alpha=0.8)
            plt.plot(days_in_year, (ipsa_cumulative_year - 1) * 100, 
                    color=colors[i], linewidth=2, linestyle='--', 
                    label=f'IPSA {year}', alpha=0.8)
        
        plt.title(f'Retornos Acumulados Anuales Reiniciados - {model_name} (Cada año comienza en 0%)', fontsize=14, fontweight='bold')
        plt.xlabel('Día del Año')
        plt.ylabel('Retorno Acumulado Anual (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Crear tabla de estadísticas anuales
        print("Creando tabla de estadísticas anuales...")
        ax2 = plt.subplot(2, 1, 2)
        ax2.axis('tight')
        ax2.axis('off')
        
        # Calcular estadísticas por año
        annual_stats = []
        for year in years:
            port_year = portfolio_df[portfolio_df['year'] == year]
            ipsa_year = ipsa_df[ipsa_df['year'] == year]
            
            if len(port_year) > 0 and len(ipsa_year) > 0:
                port_annual_return = ((1 + port_year['returns']).prod() - 1) * 100
                ipsa_annual_return = ((1 + ipsa_year['returns']).prod() - 1) * 100
                excess_annual = port_annual_return - ipsa_annual_return
                
                # Tracking error anual
                annual_te = np.std(port_year['returns'] - ipsa_year['returns']) * np.sqrt(252)
                
                annual_stats.append([
                    str(year),
                    f"{port_annual_return:.2f}%",
                    f"{ipsa_annual_return:.2f}%", 
                    f"{excess_annual:.2f}%",
                    f"{annual_te:.4f}"
                ])
        
        # Crear tabla
        headers = ['Año', 'Retorno Portafolio', 'Retorno IPSA', 'Excess Return', 'Tracking Error']
        table = ax2.table(cellText=annual_stats, colLabels=headers, 
                         cellLoc='center', loc='center', 
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Estilo de la tabla
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title(f'Estadísticas de Performance por Año - {model_name}', fontsize=14, fontweight='bold', pad=20)
        
        # Ajustar layout y guardar gráfico anual
        plt.tight_layout()
        
        # Guardar gráfico anual
        output_file_annual = f'analisis_modelo_{filename_suffix}_retornos_anuales.png'
        plt.savefig(output_file_annual, dpi=300, bbox_inches='tight')
        print(f"Gráficos anuales guardados en: {output_file_annual}")
        
        plt.show()
        
        return annual_stats

class IPSALassoTracker(BaseIPSATracker):
    """
    Modelo LASSO para seguir al índice IPSA con restricciones específicas:
    1. Suma de pesos = 1
    2. Recalibración cada 3er viernes de marzo, junio, septiembre y diciembre
    3. Top 10 acciones deben sumar al menos 60% del portafolio
    4. No forward looking - solo datos pasados para recalibración
    """
    
    def __init__(self, data_path='precios_limpios.xlsx'):
        super().__init__(data_path)  # Llamar al constructor de la clase base
        
    def generate_rebalance_dates(self):
        """Generar fechas de recalibración (3er viernes de mar, jun, sep, dic)"""
        start_year = self.stock_returns.index.min().year
        end_year = self.stock_returns.index.max().year
        
        rebalance_months = [3, 6, 9, 12]  # Marzo, Junio, Septiembre, Diciembre
        
        for year in range(start_year, end_year + 1):
            for month in rebalance_months:
                third_friday = self.get_third_friday(year, month)
                
                # Encontrar la fecha de trading más cercana (no forward looking)
                target_date = pd.to_datetime(third_friday)
                available_dates = self.stock_returns.index[self.stock_returns.index <= target_date]
                
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]  # Última fecha disponible antes o igual al target
                    if closest_date not in self.rebalance_dates:
                        self.rebalance_dates.append(closest_date)
        
        self.rebalance_dates.sort()
        print(f"Fechas de recalibración generadas: {len(self.rebalance_dates)} fechas")
        
    def optimize_lasso_weights(self, X_train, y_train, min_top10_weight=0.6):
        """
        Optimizar pesos usando LASSO con restricciones
        """
        # Probar diferentes valores de alpha para LASSO
        alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        best_weights = None
        best_score = float('inf')
        
        for alpha in alphas:
            # Entrenar modelo LASSO
            lasso = Lasso(alpha=alpha, max_iter=2000, positive=True)
            
            try:
                lasso.fit(X_train, y_train)
                weights = lasso.coef_
                
                # Asegurar que los pesos sean positivos y normalizar para que sumen exactamente 1
                weights = np.maximum(weights, 0)  # Forzar pesos positivos
                
                if weights.sum() > 0:
                    weights = weights / weights.sum()  # Normalizar para que sumen exactamente 1
                else:
                    continue
                
                # Verificar que efectivamente sumen 1
                assert abs(weights.sum() - 1.0) < 1e-10, f"Los pesos no suman 1: {weights.sum()}"
                
                # Verificar restricción de top 10
                sorted_weights = np.sort(weights)[::-1]
                top10_sum = sorted_weights[:10].sum()
                
                if top10_sum >= min_top10_weight:
                    # Calcular error de tracking
                    predictions = X_train @ weights
                    mse = np.mean((y_train - predictions) ** 2)
                    
                    if mse < best_score:
                        best_score = mse
                        best_weights = weights.copy()
            
            except Exception as e:
                continue
        
        # Si no se encuentra solución, usar optimización alternativa
        if best_weights is None:
            print("Advertencia: No se pudo satisfacer restricción de top 10, usando optimización alternativa")
            # Usar Ridge como fallback
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=0.1, positive=True)
            ridge.fit(X_train, y_train)
            best_weights = ridge.coef_
            best_weights = np.maximum(best_weights, 0)  # Asegurar pesos positivos
            if best_weights.sum() > 0:
                best_weights = best_weights / best_weights.sum()  # Normalizar para que sumen 1
            else:
                # Último recurso: pesos equiponderados
                best_weights = np.ones(len(X_train.columns)) / len(X_train.columns)
        
        # Verificación final
        assert abs(best_weights.sum() - 1.0) < 1e-10, f"ERROR: Los pesos finales no suman 1: {best_weights.sum()}"
        
        return best_weights
    
    def backtest_model(self, lookback_window=252):
        """
        Realizar backtest del modelo con recalibraciones periódicas
        """
        results = []
        
        for i, rebal_date in enumerate(self.rebalance_dates):
            print(f"Procesando recalibración {i+1}/{len(self.rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Determinar ventana de entrenamiento (solo datos pasados)
            train_end = rebal_date
            train_start_idx = max(0, self.stock_returns.index.get_loc(train_end) - lookback_window)
            train_start = self.stock_returns.index[train_start_idx]
            
            # Datos de entrenamiento
            X_train = self.stock_returns.loc[train_start:train_end]
            y_train = self.ipsa_returns.loc[train_start:train_end]
            
            if len(X_train) < 50:  # Mínimo de observaciones
                continue
            
            # Normalizar datos
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            
            # Optimizar pesos
            weights = self.optimize_lasso_weights(X_train_scaled, y_train)
            
            # Determinar período de aplicación (hasta siguiente recalibración)
            if i < len(self.rebalance_dates) - 1:
                period_end = self.rebalance_dates[i + 1]
            else:
                period_end = self.stock_returns.index.max()
            
            # Aplicar pesos y calcular performance (incluir fecha de recalibración)
            period_mask = (self.stock_returns.index >= rebal_date) & (self.stock_returns.index <= period_end)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                # Calcular retornos del portafolio
                portfolio_returns = (period_returns * weights).sum(axis=1)
                
                # Métricas de performance
                tracking_error = np.std(portfolio_returns - period_ipsa_returns) * np.sqrt(252)
                cumulative_port = (1 + portfolio_returns).cumprod().iloc[-1] - 1
                cumulative_ipsa = (1 + period_ipsa_returns).cumprod().iloc[-1] - 1
                
                # Información de los pesos
                top10_weights = np.sort(weights)[::-1][:10].sum()
                non_zero_stocks = np.sum(weights > 0.001)
                
                result = {
                    'rebalance_date': rebal_date,
                    'period_end': period_end,
                    'tracking_error': tracking_error,
                    'portfolio_return': cumulative_port,
                    'ipsa_return': cumulative_ipsa,
                    'excess_return': cumulative_port - cumulative_ipsa,
                    'top10_weight_sum': top10_weights,
                    'active_stocks': non_zero_stocks,
                    'weights': weights.copy()
                }
                
                results.append(result)
                
                print(f"  Tracking Error: {tracking_error:.4f}")
                print(f"  Top 10 weight sum: {top10_weights:.3f}")
                print(f"  Active stocks: {non_zero_stocks}")
        
        self.tracking_results = results
        return results
    
    def get_summary_statistics(self):
        """Obtener estadísticas resumen del backtest"""
        if not self.tracking_results:
            print("No hay resultados disponibles. Ejecutar backtest primero.")
            return None
        
        # Calcular estadísticas agregadas
        avg_tracking_error = np.mean([r['tracking_error'] for r in self.tracking_results])
        avg_top10_weight = np.mean([r['top10_weight_sum'] for r in self.tracking_results])
        avg_active_stocks = np.mean([r['active_stocks'] for r in self.tracking_results])
        total_excess_return = np.sum([r['excess_return'] for r in self.tracking_results])
        
        summary = {
            'total_rebalances': len(self.tracking_results),
            'avg_tracking_error': avg_tracking_error,
            'avg_top10_weight_sum': avg_top10_weight,
            'avg_active_stocks': avg_active_stocks,
            'total_excess_return': total_excess_return,
            'min_top10_constraint_met': all(r['top10_weight_sum'] >= 0.59 for r in self.tracking_results)
        }
        
        return summary
    
    def get_current_weights(self):
        """Obtener los pesos actuales (última recalibración)"""
        if self.tracking_results:
            latest_result = self.tracking_results[-1]
            weights_dict = dict(zip(self.stock_returns.columns, latest_result['weights']))
            return pd.Series(weights_dict).sort_values(ascending=False)
        return None
    
    def run_full_analysis(self):
        """Ejecutar análisis completo"""
        print("=== MODELO LASSO PARA TRACKING DEL IPSA ===\n")
        
        # Cargar y preparar datos
        self.load_data()
        self.calculate_returns()
        self.generate_rebalance_dates()
        
        # Ejecutar backtest
        print(f"\nEjecutando backtest con {len(self.rebalance_dates)} recalibraciones...")
        results = self.backtest_model()
        
        # Mostrar estadísticas resumen
        print("\n=== ESTADÍSTICAS RESUMEN ===")
        summary = self.get_summary_statistics()
        if summary:
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        # Mostrar pesos actuales
        print("\n=== PESOS ACTUALES (TOP 15) ===")
        current_weights = self.get_current_weights()
        if current_weights is not None:
            print(current_weights.head(15))
            print(f"\nSuma total de pesos: {current_weights.sum():.10f}")
            print(f"Número de acciones con peso > 0: {(current_weights > 0).sum()}")
            print(f"Peso de las top 10: {current_weights.head(10).sum():.4f}")
        
        return results
    
    def create_visualizations(self):
        """
        Crear visualizaciones del modelo LASSO
        """
        if not self.tracking_results:
            print("No hay resultados disponibles. Ejecutar backtest primero.")
            return
        
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. GRÁFICO DE PESOS A LO LARGO DEL TIEMPO
        print("Creando gráfico de evolución de pesos...")
        ax1 = plt.subplot(3, 1, 1)
        
        # Preparar datos de pesos
        dates = [result['rebalance_date'] for result in self.tracking_results]
        weights_matrix = np.array([result['weights'] for result in self.tracking_results])
        stock_names = self.stock_returns.columns
        
        # Seleccionar las 10 acciones con mayor peso promedio
        avg_weights = np.mean(weights_matrix, axis=0)
        top_10_indices = np.argsort(avg_weights)[-10:]
        top_10_names = [stock_names[i] for i in top_10_indices]
        
        # Graficar evolución de pesos de top 10
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_10_indices)))
        for i, (idx, name) in enumerate(zip(top_10_indices, top_10_names)):
            plt.plot(dates, weights_matrix[:, idx], 
                    label=name, linewidth=2, color=colors[i], marker='o', markersize=4)
        
        plt.title('Evolución de Pesos del Portafolio (Top 10 Acciones)', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha de Recalibración')
        plt.ylabel('Peso en el Portafolio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. GRÁFICO DE TRACKING ERROR A LO LARGO DEL TIEMPO
        print("Creando gráfico de tracking error...")
        ax2 = plt.subplot(3, 1, 2)
        
        tracking_errors = [result['tracking_error'] for result in self.tracking_results]
        
        plt.plot(dates, tracking_errors, 'r-o', linewidth=2, markersize=6, 
                label='Tracking Error')
        plt.axhline(y=np.mean(tracking_errors), color='blue', linestyle='--', 
                   label=f'Promedio: {np.mean(tracking_errors):.4f}')
        
        plt.title('Evolución del Tracking Error', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha de Recalibración')
        plt.ylabel('Tracking Error (Anualizado)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. GRÁFICO DE RETORNO COMPUESTO: PORTAFOLIO VS IPSA
        print("Creando gráfico de retornos acumulados...")
        ax3 = plt.subplot(3, 1, 3)
        
        # Calcular serie temporal completa de retornos
        all_dates = []
        portfolio_cumulative = []
        ipsa_cumulative = []
        
        cumulative_port = 1.0
        cumulative_ipsa = 1.0
        
        for i, result in enumerate(self.tracking_results):
            # Período de aplicación de estos pesos
            start_date = result['rebalance_date']
            if i < len(self.tracking_results) - 1:
                end_date = self.tracking_results[i + 1]['rebalance_date']
            else:
                end_date = self.stock_returns.index.max()
            
            # Retornos en este período
            period_mask = (self.stock_returns.index > start_date) & (self.stock_returns.index <= end_date)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                weights = result['weights']
                portfolio_period_returns = (period_returns * weights).sum(axis=1)
                
                # Acumular retornos día a día
                for date in period_returns.index:
                    port_ret = portfolio_period_returns.loc[date]
                    ipsa_ret = period_ipsa_returns.loc[date]
                    
                    cumulative_port *= (1 + port_ret)
                    cumulative_ipsa *= (1 + ipsa_ret)
                    
                    all_dates.append(date)
                    portfolio_cumulative.append(cumulative_port)
                    ipsa_cumulative.append(cumulative_ipsa)
        
        # Convertir a Series para graficar
        portfolio_series = pd.Series(portfolio_cumulative, index=all_dates)
        ipsa_series = pd.Series(ipsa_cumulative, index=all_dates)
        
        plt.plot(portfolio_series.index, (portfolio_series - 1) * 100, 
                'blue', linewidth=2, label='Portafolio LASSO')
        plt.plot(ipsa_series.index, (ipsa_series - 1) * 100, 
                'red', linewidth=2, label='IPSA')
        
        # Marcar fechas de recalibración
        for date in dates:
            if date in portfolio_series.index:
                plt.axvline(x=date, color='gray', linestyle=':', alpha=0.7)
        
        plt.title('Retornos Acumulados: Portafolio vs IPSA', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Ajustar layout y guardar gráfico principal
        fig.tight_layout()
        
        # Guardar gráfico principal
        output_file = 'analisis_modelo_lasso.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGráficos principales guardados en: {output_file}")
        
        plt.show()
        
        # Crear gráfico de retornos anuales reiniciados usando la funcionalidad de la clase base
        annual_stats = self.create_annual_returns_visualization(
            portfolio_series, ipsa_series, "LASSO", "lasso"
        )
        
        # Mostrar estadísticas finales
        final_port_return = (portfolio_series.iloc[-1] - 1) * 100
        final_ipsa_return = (ipsa_series.iloc[-1] - 1) * 100
        excess_return = final_port_return - final_ipsa_return
        
        print(f"\n=== ESTADÍSTICAS FINALES - LASSO ===")
        print(f"Retorno acumulado Portafolio: {final_port_return:.2f}%")
        print(f"Retorno acumulado IPSA: {final_ipsa_return:.2f}%")
        print(f"Excess Return: {excess_return:.2f}%")
        print(f"Tracking Error promedio: {np.mean(tracking_errors):.4f}")
        
        # Estadísticas anuales en consola
        print(f"\n=== ESTADÍSTICAS ANUALES - LASSO ===")
        for stat in annual_stats:
            print(f"Año {stat[0]}: Port={stat[1]}, IPSA={stat[2]}, Excess={stat[3]}, TE={stat[4]}")
        
        return {
            'portfolio_cumulative': portfolio_series,
            'ipsa_cumulative': ipsa_series,
            'tracking_errors': tracking_errors,
            'weights_evolution': weights_matrix,
            'dates': dates
        }

class IPSATrackerSinRestricciones(BaseIPSATracker):
    """
    Modelo sin restricciones de peso para minimizar tracking error con IPSA
    Objetivo: Minimizar TE sin restricciones de peso individual o de concentración
    Solo mantiene: suma de pesos = 1 y recalibración trimestral
    """
    
    def __init__(self, data_path='precios_limpios.xlsx'):
        super().__init__(data_path)  # Llamar al constructor de la clase base
        
    def generate_rebalance_dates(self):
        """Generar fechas de recalibración (3er viernes de mar, jun, sep, dic)"""
        start_year = self.stock_returns.index.min().year
        end_year = self.stock_returns.index.max().year
        
        rebalance_months = [3, 6, 9, 12]  # Marzo, Junio, Septiembre, Diciembre
        
        for year in range(start_year, end_year + 1):
            for month in rebalance_months:
                third_friday = self.get_third_friday(year, month)
                
                # Encontrar la fecha de trading más cercana (no forward looking)
                target_date = pd.to_datetime(third_friday)
                available_dates = self.stock_returns.index[self.stock_returns.index <= target_date]
                
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]  # Última fecha disponible antes o igual al target
                    if closest_date not in self.rebalance_dates:
                        self.rebalance_dates.append(closest_date)
        
        self.rebalance_dates.sort()
        print(f"Fechas de recalibración generadas: {len(self.rebalance_dates)} fechas")
        
    def backtest_model(self, lookback_window=252):
        """
        Realizar backtest del modelo sin restricciones
        """
        results = []
        
        for i, rebal_date in enumerate(self.rebalance_dates):
            print(f"Procesando recalibración {i+1}/{len(self.rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Determinar ventana de entrenamiento (solo datos pasados)
            train_end = rebal_date
            train_start_idx = max(0, self.stock_returns.index.get_loc(train_end) - lookback_window)
            train_start = self.stock_returns.index[train_start_idx]
            
            # Datos de entrenamiento
            X_train = self.stock_returns.loc[train_start:train_end]
            y_train = self.ipsa_returns.loc[train_start:train_end]
            
            if len(X_train) < 50:  # Mínimo de observaciones
                continue
            
            # Normalizar datos
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            
            # Optimizar pesos sin restricciones
            weights, method = self.optimize_unrestricted_weights(X_train_scaled, y_train)
            
            # Determinar período de aplicación (hasta siguiente recalibración)
            if i < len(self.rebalance_dates) - 1:
                period_end = self.rebalance_dates[i + 1]
            else:
                period_end = self.stock_returns.index.max()
            
            # Aplicar pesos y calcular performance (incluir fecha de recalibración)
            period_mask = (self.stock_returns.index >= rebal_date) & (self.stock_returns.index <= period_end)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                # Calcular retornos del portafolio
                portfolio_returns = (period_returns * weights).sum(axis=1)
                
                # Métricas de performance
                tracking_error = np.std(portfolio_returns - period_ipsa_returns) * np.sqrt(252)
                cumulative_port = (1 + portfolio_returns).cumprod().iloc[-1] - 1
                cumulative_ipsa = (1 + period_ipsa_returns).cumprod().iloc[-1] - 1
                
                # Información de los pesos
                max_weight = np.max(weights)
                min_weight = np.min(weights)
                negative_weights = np.sum(weights < 0)
                non_zero_stocks = np.sum(np.abs(weights) > 0.001)
                
                result = {
                    'rebalance_date': rebal_date,
                    'period_end': period_end,
                    'tracking_error': tracking_error,
                    'portfolio_return': cumulative_port,
                    'ipsa_return': cumulative_ipsa,
                    'excess_return': cumulative_port - cumulative_ipsa,
                    'max_weight': max_weight,
                    'min_weight': min_weight,
                    'negative_weights': negative_weights,
                    'active_stocks': non_zero_stocks,
                    'method': method,
                    'weights': weights.copy()
                }
                
                results.append(result)
                
                print(f"  Tracking Error: {tracking_error:.4f}")
                print(f"  Método: {method}")
                print(f"  Peso máximo: {max_weight:.3f}")
                print(f"  Pesos negativos: {negative_weights}")
        
        self.tracking_results = results
        return results
    
    def get_summary_statistics(self):
        """Obtener estadísticas resumen del backtest sin restricciones"""
        if not self.tracking_results:
            print("No hay resultados disponibles. Ejecutar backtest primero.")
            return None
        
        # Calcular estadísticas agregadas
        avg_tracking_error = np.mean([r['tracking_error'] for r in self.tracking_results])
        avg_max_weight = np.mean([r['max_weight'] for r in self.tracking_results])
        avg_negative_weights = np.mean([r['negative_weights'] for r in self.tracking_results])
        avg_active_stocks = np.mean([r['active_stocks'] for r in self.tracking_results])
        total_excess_return = np.sum([r['excess_return'] for r in self.tracking_results])
        
        summary = {
            'total_rebalances': len(self.tracking_results),
            'avg_tracking_error': avg_tracking_error,
            'avg_max_weight': avg_max_weight,
            'avg_negative_weights': avg_negative_weights,
            'avg_active_stocks': avg_active_stocks,
            'total_excess_return': total_excess_return
        }
        
        return summary
    
    def get_current_weights(self):
        """Obtener los pesos actuales (última recalibración)"""
        if self.tracking_results:
            latest_result = self.tracking_results[-1]
            weights_dict = dict(zip(self.stock_returns.columns, latest_result['weights']))
            return pd.Series(weights_dict).sort_values(ascending=False)
        return None
    
    def run_full_analysis(self):
        """Ejecutar análisis completo sin restricciones"""
        print("=== MODELO SIN RESTRICCIONES PARA TRACKING DEL IPSA ===\n")
        
        # Cargar y preparar datos
        self.load_data()
        self.calculate_returns()
        self.generate_rebalance_dates()
        
        # Ejecutar backtest
        print(f"\nEjecutando backtest sin restricciones con {len(self.rebalance_dates)} recalibraciones...")
        results = self.backtest_model()
        
        # Mostrar estadísticas resumen
        print("\n=== ESTADÍSTICAS RESUMEN (SIN RESTRICCIONES) ===")
        summary = self.get_summary_statistics()
        if summary:
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        # Mostrar pesos actuales
        print("\n=== PESOS ACTUALES - SIN RESTRICCIONES (TOP 15) ===")
        current_weights = self.get_current_weights()
        if current_weights is not None:
            print(current_weights.head(15))
            print(f"\nSuma total de pesos: {current_weights.sum():.10f}")
            print(f"Número de acciones con peso > 0: {(current_weights > 0).sum()}")
            print(f"Número de acciones con peso < 0: {(current_weights < 0).sum()}")
            print(f"Peso máximo: {current_weights.max():.4f}")
            print(f"Peso mínimo: {current_weights.min():.4f}")
        
        return results

class IPSATrackerMensual(BaseIPSATracker):
    """
    Modelo sin restricciones de peso con recalibración MENSUAL
    Objetivo: Minimizar TE con rebalanceo más frecuente (mensual)
    Restricciones: Solo suma de pesos = 1 y recalibración al comienzo de la 4ta semana de cada mes
    """
    
    def __init__(self, data_path='precios_limpios.xlsx'):
        super().__init__(data_path)  # Llamar al constructor de la clase base
        
    def generate_monthly_rebalance_dates(self):
        """Generar fechas de recalibración MENSUAL (comienzo de la 4ta semana)"""
        start_date = self.stock_returns.index.min()
        end_date = self.stock_returns.index.max()
        
        start_year = start_date.year
        start_month = start_date.month
        end_year = end_date.year
        end_month = end_date.month
        
        # Generar fechas para cada mes
        current_year = start_year
        current_month = start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            try:
                # Obtener el comienzo de la 4ta semana del mes
                fourth_week_start = self.get_fourth_week_start(current_year, current_month)
                target_date = pd.to_datetime(fourth_week_start)
                
                # Verificar que la fecha esté dentro del rango de datos
                if target_date >= start_date and target_date <= end_date:
                    # Encontrar la fecha de trading más cercana (no forward looking)
                    available_dates = self.stock_returns.index[self.stock_returns.index <= target_date]
                    
                    if len(available_dates) > 0:
                        closest_date = available_dates[-1]  # Última fecha disponible antes o igual al target
                        if closest_date not in self.rebalance_dates:
                            self.rebalance_dates.append(closest_date)
            
            except ValueError:
                # Manejar casos donde el mes no tiene suficientes días
                pass
            
            # Avanzar al siguiente mes
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        self.rebalance_dates.sort()
        print(f"Fechas de recalibración mensual (4ta semana) generadas: {len(self.rebalance_dates)} fechas")
        
    def backtest_model(self, lookback_window=126):  # 6 meses para rebalanceo mensual
        """
        Realizar backtest del modelo con recalibración mensual
        """
        results = []
        
        for i, rebal_date in enumerate(self.rebalance_dates):
            if i % 6 == 0:  # Mostrar cada 6 meses
                print(f"Procesando recalibración {i+1}/{len(self.rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Determinar ventana de entrenamiento (solo datos pasados)
            train_end = rebal_date
            train_start_idx = max(0, self.stock_returns.index.get_loc(train_end) - lookback_window)
            train_start = self.stock_returns.index[train_start_idx]
            
            # Datos de entrenamiento
            X_train = self.stock_returns.loc[train_start:train_end]
            y_train = self.ipsa_returns.loc[train_start:train_end]
            
            if len(X_train) < 30:  # Mínimo de observaciones para rebalanceo mensual
                continue
            
            # Normalizar datos
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            
            # Optimizar pesos sin restricciones
            weights, method = self.optimize_unrestricted_weights(X_train_scaled, y_train)
            
            # Determinar período de aplicación (hasta siguiente recalibración)
            if i < len(self.rebalance_dates) - 1:
                period_end = self.rebalance_dates[i + 1]
            else:
                period_end = self.stock_returns.index.max()
            
            # Aplicar pesos y calcular performance (incluir fecha de recalibración)
            period_mask = (self.stock_returns.index >= rebal_date) & (self.stock_returns.index <= period_end)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                # Calcular retornos del portafolio
                portfolio_returns = (period_returns * weights).sum(axis=1)
                
                # Métricas de performance
                tracking_error = np.std(portfolio_returns - period_ipsa_returns) * np.sqrt(252)
                cumulative_port = (1 + portfolio_returns).cumprod().iloc[-1] - 1
                cumulative_ipsa = (1 + period_ipsa_returns).cumprod().iloc[-1] - 1
                
                # Información de los pesos
                max_weight = np.max(weights)
                min_weight = np.min(weights)
                negative_weights = np.sum(weights < 0)
                non_zero_stocks = np.sum(np.abs(weights) > 0.001)
                
                result = {
                    'rebalance_date': rebal_date,
                    'period_end': period_end,
                    'tracking_error': tracking_error,
                    'portfolio_return': cumulative_port,
                    'ipsa_return': cumulative_ipsa,
                    'excess_return': cumulative_port - cumulative_ipsa,
                    'max_weight': max_weight,
                    'min_weight': min_weight,
                    'negative_weights': negative_weights,
                    'active_stocks': non_zero_stocks,
                    'method': method,
                    'weights': weights.copy()
                }
                
                results.append(result)
        
        self.tracking_results = results
        return results
    
    def get_summary_statistics(self):
        """Obtener estadísticas resumen del backtest mensual"""
        if not self.tracking_results:
            print("No hay resultados disponibles. Ejecutar backtest primero.")
            return None
        
        # Calcular estadísticas agregadas
        avg_tracking_error = np.mean([r['tracking_error'] for r in self.tracking_results])
        avg_max_weight = np.mean([r['max_weight'] for r in self.tracking_results])
        avg_negative_weights = np.mean([r['negative_weights'] for r in self.tracking_results])
        avg_active_stocks = np.mean([r['active_stocks'] for r in self.tracking_results])
        total_excess_return = np.sum([r['excess_return'] for r in self.tracking_results])
        
        summary = {
            'total_rebalances': len(self.tracking_results),
            'avg_tracking_error': avg_tracking_error,
            'avg_max_weight': avg_max_weight,
            'avg_negative_weights': avg_negative_weights,
            'avg_active_stocks': avg_active_stocks,
            'total_excess_return': total_excess_return
        }
        
        return summary
    
    def get_current_weights(self):
        """Obtener los pesos actuales (última recalibración)"""
        if self.tracking_results:
            latest_result = self.tracking_results[-1]
            weights_dict = dict(zip(self.stock_returns.columns, latest_result['weights']))
            return pd.Series(weights_dict).sort_values(ascending=False)
        return None
    
    def run_full_analysis(self):
        """Ejecutar análisis completo con recalibración mensual en la 4ta semana"""
        print("=== MODELO MENSUAL (4ta SEMANA) SIN RESTRICCIONES PARA TRACKING DEL IPSA ===\n")
        
        # Cargar y preparar datos
        self.load_data()
        self.calculate_returns()
        self.generate_monthly_rebalance_dates()
        
        # Ejecutar backtest
        print(f"\nEjecutando backtest mensual con {len(self.rebalance_dates)} recalibraciones...")
        results = self.backtest_model()
        
        # Mostrar estadísticas resumen
        print("\n=== ESTADÍSTICAS RESUMEN (REBALANCEO MENSUAL 4ta SEMANA) ===")
        summary = self.get_summary_statistics()
        if summary:
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        # Mostrar pesos actuales
        print("\n=== PESOS ACTUALES - REBALANCEO MENSUAL 4ta SEMANA (TOP 15) ===")
        current_weights = self.get_current_weights()
        if current_weights is not None:
            print(current_weights.head(15))
            print(f"\nSuma total de pesos: {current_weights.sum():.10f}")
            print(f"Número de acciones con peso > 0: {(current_weights > 0).sum()}")
            print(f"Número de acciones con peso < 0: {(current_weights < 0).sum()}")
            print(f"Peso máximo: {current_weights.max():.4f}")
            print(f"Peso mínimo: {current_weights.min():.4f}")
        
        return results
    
    def create_visualizations(self):
        """
        Crear visualizaciones del modelo con rebalanceo mensual
        """
        if not self.tracking_results:
            print("No hay resultados disponibles. Ejecutar backtest primero.")
            return
        
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. GRÁFICO DE PESOS A LO LARGO DEL TIEMPO
        print("Creando gráfico de evolución de pesos...")
        ax1 = plt.subplot(3, 1, 1)
        
        # Preparar datos de pesos
        dates = [result['rebalance_date'] for result in self.tracking_results]
        weights_matrix = np.array([result['weights'] for result in self.tracking_results])
        stock_names = self.stock_returns.columns
        
        # Seleccionar las acciones con mayor peso absoluto promedio
        avg_abs_weights = np.mean(np.abs(weights_matrix), axis=0)
        top_indices = np.argsort(avg_abs_weights)[-10:]  # Top 10 por peso absoluto
        top_names = [stock_names[i] for i in top_indices]
        
        # Graficar evolución de pesos (muestrear cada 3 meses para legibilidad)
        sample_indices = range(0, len(dates), 3)  # Cada 3 meses
        sampled_dates = [dates[i] for i in sample_indices]
        sampled_weights = weights_matrix[sample_indices, :]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_indices)))
        for i, (idx, name) in enumerate(zip(top_indices, top_names)):
            plt.plot(sampled_dates, sampled_weights[:, idx], 
                    label=name, linewidth=2, color=colors[i], marker='o', markersize=3)
        
        # Línea en cero para visualizar pesos negativos
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title('Evolución de Pesos - REBALANCEO MENSUAL 4ta Semana (Top 10, muestra trimestral)', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha de Recalibración')
        plt.ylabel('Peso en el Portafolio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. GRÁFICO DE TRACKING ERROR A LO LARGO DEL TIEMPO
        print("Creando gráfico de tracking error...")
        ax2 = plt.subplot(3, 1, 2)
        
        tracking_errors = [result['tracking_error'] for result in self.tracking_results]
        
        # Muestrear cada 3 meses para legibilidad
        sampled_te = [tracking_errors[i] for i in sample_indices]
        
        plt.plot(sampled_dates, sampled_te, 'r-o', linewidth=2, markersize=4, 
                label='Tracking Error')
        plt.axhline(y=np.mean(tracking_errors), color='blue', linestyle='--', 
                   label=f'Promedio: {np.mean(tracking_errors):.4f}')
        
        plt.title('Evolución del Tracking Error - REBALANCEO MENSUAL 4ta Semana', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha de Recalibración')
        plt.ylabel('Tracking Error (Anualizado)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. GRÁFICO DE RETORNO COMPUESTO: PORTAFOLIO VS IPSA
        print("Creando gráfico de retornos acumulados...")
        ax3 = plt.subplot(3, 1, 3)
        
        # Calcular serie temporal completa de retornos
        all_dates = []
        portfolio_cumulative = []
        ipsa_cumulative = []
        
        cumulative_port = 1.0
        cumulative_ipsa = 1.0
        
        for i, result in enumerate(self.tracking_results):
            # Período de aplicación de estos pesos
            start_date = result['rebalance_date']
            if i < len(self.tracking_results) - 1:
                end_date = self.tracking_results[i + 1]['rebalance_date']
            else:
                end_date = self.stock_returns.index.max()
            
            # Retornos en este período (incluir fecha de recalibración)
            period_mask = (self.stock_returns.index >= start_date) & (self.stock_returns.index <= end_date)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                weights = result['weights']
                portfolio_period_returns = (period_returns * weights).sum(axis=1)
                
                # Acumular retornos día a día
                for date in period_returns.index:
                    port_ret = portfolio_period_returns.loc[date]
                    ipsa_ret = period_ipsa_returns.loc[date]
                    
                    cumulative_port *= (1 + port_ret)
                    cumulative_ipsa *= (1 + ipsa_ret)
                    
                    all_dates.append(date)
                    portfolio_cumulative.append(cumulative_port)
                    ipsa_cumulative.append(cumulative_ipsa)
        
        # Convertir a Series para graficar
        portfolio_series = pd.Series(portfolio_cumulative, index=all_dates)
        ipsa_series = pd.Series(ipsa_cumulative, index=all_dates)
        
        plt.plot(portfolio_series.index, (portfolio_series - 1) * 100, 
                'blue', linewidth=2, label='Portafolio Mensual')
        plt.plot(ipsa_series.index, (ipsa_series - 1) * 100, 
                'red', linewidth=2, label='IPSA')
        
        # Marcar fechas de recalibración (muestrear)
        for date in sampled_dates:
            if date in portfolio_series.index:
                plt.axvline(x=date, color='gray', linestyle=':', alpha=0.5)
        
        plt.title('Retornos Acumulados: Portafolio Mensual 4ta Semana vs IPSA', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. GRÁFICO DE RETORNOS ANUALES REINICIADOS (NUEVO)
        print("Creando gráfico de retornos anuales reiniciados...")
        
        # Crear una nueva figura para el gráfico anual
        fig2 = plt.figure(figsize=(20, 12))
        
        # Crear subplot para el gráfico anual
        ax4 = plt.subplot(2, 1, 1)
        
        # Calcular retornos anuales reiniciados
        portfolio_df = pd.DataFrame({'returns': portfolio_series.pct_change().fillna(0)}, index=portfolio_series.index)
        ipsa_df = pd.DataFrame({'returns': ipsa_series.pct_change().fillna(0)}, index=ipsa_series.index)
        
        # Agrupar por año y calcular retornos acumulados por año
        portfolio_df['year'] = portfolio_df.index.year
        ipsa_df['year'] = ipsa_df.index.year
        
        # Colores para diferentes años
        years = sorted(portfolio_df['year'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
        
        for i, year in enumerate(years):
            # Filtrar datos por año
            port_year = portfolio_df[portfolio_df['year'] == year]
            ipsa_year = ipsa_df[ipsa_df['year'] == year]
            
            # Calcular retornos acumulados reiniciando en 1 para cada año
            port_cumulative_year = (1 + port_year['returns']).cumprod()
            ipsa_cumulative_year = (1 + ipsa_year['returns']).cumprod()
            
            # Crear índice de días del año (1 a N)
            days_in_year = range(1, len(port_cumulative_year) + 1)
            
            # Plotear líneas para este año
            plt.plot(days_in_year, (port_cumulative_year - 1) * 100, 
                    color=colors[i], linewidth=2, linestyle='-', 
                    label=f'Portafolio {year}', alpha=0.8)
            plt.plot(days_in_year, (ipsa_cumulative_year - 1) * 100, 
                    color=colors[i], linewidth=2, linestyle='--', 
                    label=f'IPSA {year}', alpha=0.8)
        
        plt.title('Retornos Acumulados Anuales Reiniciados - REBALANCEO MENSUAL (Cada año comienza en 0%)', fontsize=14, fontweight='bold')
        plt.xlabel('Día del Año')
        plt.ylabel('Retorno Acumulado Anual (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 5. TABLA DE ESTADÍSTICAS ANUALES
        print("Creando tabla de estadísticas anuales...")
        ax5 = plt.subplot(2, 1, 2)
        ax5.axis('tight')
        ax5.axis('off')
        
        # Calcular estadísticas por año
        annual_stats = []
        for year in years:
            port_year = portfolio_df[portfolio_df['year'] == year]
            ipsa_year = ipsa_df[ipsa_df['year'] == year]
            
            if len(port_year) > 0 and len(ipsa_year) > 0:
                port_annual_return = ((1 + port_year['returns']).prod() - 1) * 100
                ipsa_annual_return = ((1 + ipsa_year['returns']).prod() - 1) * 100
                excess_annual = port_annual_return - ipsa_annual_return
                
                # Tracking error anual
                annual_te = np.std(port_year['returns'] - ipsa_year['returns']) * np.sqrt(252)
                
                annual_stats.append([
                    str(year),
                    f"{port_annual_return:.2f}%",
                    f"{ipsa_annual_return:.2f}%", 
                    f"{excess_annual:.2f}%",
                    f"{annual_te:.4f}"
                ])
        
        # Crear tabla
        headers = ['Año', 'Retorno Portafolio', 'Retorno IPSA', 'Excess Return', 'Tracking Error']
        table = ax5.table(cellText=annual_stats, colLabels=headers, 
                         cellLoc='center', loc='center', 
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Estilo de la tabla
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Estadísticas de Performance por Año - REBALANCEO MENSUAL', fontsize=14, fontweight='bold', pad=20)
        
        # Ajustar layout y guardar gráfico anual
        plt.tight_layout()
        
        # Guardar gráfico anual
        output_file_annual = 'analisis_modelo_mensual_retornos_anuales.png'
        plt.savefig(output_file_annual, dpi=300, bbox_inches='tight')
        print(f"Gráficos anuales guardados en: {output_file_annual}")
        
        plt.show()
        
        # Ajustar layout y guardar gráfico principal
        fig.tight_layout()
        
        # Guardar gráfico principal
        output_file = 'analisis_modelo_mensual_4ta_semana.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGráficos principales guardados en: {output_file}")
        
        # Mostrar estadísticas finales
        final_port_return = (portfolio_series.iloc[-1] - 1) * 100
        final_ipsa_return = (ipsa_series.iloc[-1] - 1) * 100
        excess_return = final_port_return - final_ipsa_return
        
        print(f"\n=== ESTADÍSTICAS FINALES - REBALANCEO MENSUAL 4ta SEMANA ===")
        print(f"Retorno acumulado Portafolio: {final_port_return:.2f}%")
        print(f"Retorno acumulado IPSA: {final_ipsa_return:.2f}%")
        print(f"Excess Return: {excess_return:.2f}%")
        print(f"Tracking Error promedio: {np.mean(tracking_errors):.4f}")
        print(f"Número total de recalibraciones: {len(self.tracking_results)}")
        
        # Estadísticas anuales en consola
        print(f"\n=== ESTADÍSTICAS ANUALES - REBALANCEO MENSUAL ===")
        for stat in annual_stats:
            print(f"Año {stat[0]}: Port={stat[1]}, IPSA={stat[2]}, Excess={stat[3]}, TE={stat[4]}")
        
        # No mostrar automáticamente - el usuario puede decidir
        
        return {
            'portfolio_cumulative': portfolio_series,
            'ipsa_cumulative': ipsa_series,
            'tracking_errors': tracking_errors,
            'weights_evolution': weights_matrix,
            'dates': dates
        }
    
    def create_visualizations(self):
        """
        Crear visualizaciones del modelo sin restricciones
        """
        if not self.tracking_results:
            print("No hay resultados disponibles. Ejecutar backtest primero.")
            return
        
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. GRÁFICO DE PESOS A LO LARGO DEL TIEMPO (incluyendo pesos negativos)
        print("Creando gráfico de evolución de pesos...")
        ax1 = plt.subplot(3, 1, 1)
        
        # Preparar datos de pesos
        dates = [result['rebalance_date'] for result in self.tracking_results]
        weights_matrix = np.array([result['weights'] for result in self.tracking_results])
        stock_names = self.stock_returns.columns
        
        # Seleccionar las acciones con mayor peso absoluto promedio
        avg_abs_weights = np.mean(np.abs(weights_matrix), axis=0)
        top_indices = np.argsort(avg_abs_weights)[-15:]  # Top 15 por peso absoluto
        top_names = [stock_names[i] for i in top_indices]
        
        # Graficar evolución de pesos
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_indices)))
        for i, (idx, name) in enumerate(zip(top_indices, top_names)):
            plt.plot(dates, weights_matrix[:, idx], 
                    label=name, linewidth=2, color=colors[i], marker='o', markersize=4)
        
        # Línea en cero para visualizar pesos negativos
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title('Evolución de Pesos del Portafolio - SIN RESTRICCIONES (Top 15)', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha de Recalibración')
        plt.ylabel('Peso en el Portafolio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. GRÁFICO DE TRACKING ERROR A LO LARGO DEL TIEMPO
        print("Creando gráfico de tracking error...")
        ax2 = plt.subplot(3, 1, 2)
        
        tracking_errors = [result['tracking_error'] for result in self.tracking_results]
        
        plt.plot(dates, tracking_errors, 'r-o', linewidth=2, markersize=6, 
                label='Tracking Error')
        plt.axhline(y=np.mean(tracking_errors), color='blue', linestyle='--', 
                   label=f'Promedio: {np.mean(tracking_errors):.4f}')
        
        plt.title('Evolución del Tracking Error - SIN RESTRICCIONES', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha de Recalibración')
        plt.ylabel('Tracking Error (Anualizado)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. GRÁFICO DE RETORNO COMPUESTO: PORTAFOLIO VS IPSA
        print("Creando gráfico de retornos acumulados...")
        ax3 = plt.subplot(3, 1, 3)
        
        # Calcular serie temporal completa de retornos
        all_dates = []
        portfolio_cumulative = []
        ipsa_cumulative = []
        
        cumulative_port = 1.0
        cumulative_ipsa = 1.0
        
        for i, result in enumerate(self.tracking_results):
            # Período de aplicación de estos pesos
            start_date = result['rebalance_date']
            if i < len(self.tracking_results) - 1:
                end_date = self.tracking_results[i + 1]['rebalance_date']
            else:
                end_date = self.stock_returns.index.max()
            
            # Retornos en este período
            period_mask = (self.stock_returns.index > start_date) & (self.stock_returns.index <= end_date)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                weights = result['weights']
                portfolio_period_returns = (period_returns * weights).sum(axis=1)
                
                # Acumular retornos día a día
                for date in period_returns.index:
                    port_ret = portfolio_period_returns.loc[date]
                    ipsa_ret = period_ipsa_returns.loc[date]
                    
                    cumulative_port *= (1 + port_ret)
                    cumulative_ipsa *= (1 + ipsa_ret)
                    
                    all_dates.append(date)
                    portfolio_cumulative.append(cumulative_port)
                    ipsa_cumulative.append(cumulative_ipsa)
        
        # Convertir a Series para graficar
        portfolio_series = pd.Series(portfolio_cumulative, index=all_dates)
        ipsa_series = pd.Series(ipsa_cumulative, index=all_dates)
        
        plt.plot(portfolio_series.index, (portfolio_series - 1) * 100, 
                'blue', linewidth=2, label='Portafolio Sin Restricciones')
        plt.plot(ipsa_series.index, (ipsa_series - 1) * 100, 
                'red', linewidth=2, label='IPSA')
        
        # Marcar fechas de recalibración
        for date in dates:
            if date in portfolio_series.index:
                plt.axvline(x=date, color='gray', linestyle=':', alpha=0.7)
        
        plt.title('Retornos Acumulados: Portafolio Sin Restricciones vs IPSA', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. GRÁFICO DE RETORNOS ANUALES REINICIADOS (NUEVO)
        print("Creando gráfico de retornos anuales reiniciados...")
        
        # Crear una nueva figura para el gráfico anual
        fig2 = plt.figure(figsize=(20, 12))
        
        # Crear subplot para el gráfico anual
        ax4 = plt.subplot(2, 1, 1)
        
        # Calcular retornos anuales reiniciados
        portfolio_df = pd.DataFrame({'returns': portfolio_series.pct_change().fillna(0)}, index=portfolio_series.index)
        ipsa_df = pd.DataFrame({'returns': ipsa_series.pct_change().fillna(0)}, index=ipsa_series.index)
        
        # Agrupar por año y calcular retornos acumulados por año
        portfolio_df['year'] = portfolio_df.index.year
        ipsa_df['year'] = ipsa_df.index.year
        
        # Colores para diferentes años
        years = sorted(portfolio_df['year'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
        
        for i, year in enumerate(years):
            # Filtrar datos por año
            port_year = portfolio_df[portfolio_df['year'] == year]
            ipsa_year = ipsa_df[ipsa_df['year'] == year]
            
            # Calcular retornos acumulados reiniciando en 1 para cada año
            port_cumulative_year = (1 + port_year['returns']).cumprod()
            ipsa_cumulative_year = (1 + ipsa_year['returns']).cumprod()
            
            # Crear índice de días del año (1 a N)
            days_in_year = range(1, len(port_cumulative_year) + 1)
            
            # Plotear líneas para este año
            plt.plot(days_in_year, (port_cumulative_year - 1) * 100, 
                    color=colors[i], linewidth=2, linestyle='-', 
                    label=f'Portafolio {year}', alpha=0.8)
            plt.plot(days_in_year, (ipsa_cumulative_year - 1) * 100, 
                    color=colors[i], linewidth=2, linestyle='--', 
                    label=f'IPSA {year}', alpha=0.8)
        
        plt.title('Retornos Acumulados Anuales Reiniciados - SIN RESTRICCIONES (Cada año comienza en 0%)', fontsize=14, fontweight='bold')
        plt.xlabel('Día del Año')
        plt.ylabel('Retorno Acumulado Anual (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 5. TABLA DE ESTADÍSTICAS ANUALES
        print("Creando tabla de estadísticas anuales...")
        ax5 = plt.subplot(2, 1, 2)
        ax5.axis('tight')
        ax5.axis('off')
        
        # Calcular estadísticas por año
        annual_stats = []
        for year in years:
            port_year = portfolio_df[portfolio_df['year'] == year]
            ipsa_year = ipsa_df[ipsa_df['year'] == year]
            
            if len(port_year) > 0 and len(ipsa_year) > 0:
                port_annual_return = ((1 + port_year['returns']).prod() - 1) * 100
                ipsa_annual_return = ((1 + ipsa_year['returns']).prod() - 1) * 100
                excess_annual = port_annual_return - ipsa_annual_return
                
                # Tracking error anual
                annual_te = np.std(port_year['returns'] - ipsa_year['returns']) * np.sqrt(252)
                
                annual_stats.append([
                    str(year),
                    f"{port_annual_return:.2f}%",
                    f"{ipsa_annual_return:.2f}%", 
                    f"{excess_annual:.2f}%",
                    f"{annual_te:.4f}"
                ])
        
        # Crear tabla
        headers = ['Año', 'Retorno Portafolio', 'Retorno IPSA', 'Excess Return', 'Tracking Error']
        table = ax5.table(cellText=annual_stats, colLabels=headers, 
                         cellLoc='center', loc='center', 
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Estilo de la tabla
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Estadísticas de Performance por Año - SIN RESTRICCIONES', fontsize=14, fontweight='bold', pad=20)
        
        # Ajustar layout y guardar gráfico anual
        plt.tight_layout()
        
        # Guardar gráfico anual
        output_file_annual = 'analisis_modelo_sin_restricciones_retornos_anuales.png'
        plt.savefig(output_file_annual, dpi=300, bbox_inches='tight')
        print(f"Gráficos anuales guardados en: {output_file_annual}")
        
        plt.show()
        
        # Ajustar layout y guardar gráfico principal
        fig.tight_layout()
        
        # Guardar gráfico principal
        output_file = 'analisis_modelo_sin_restricciones.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGráficos principales guardados en: {output_file}")
        
        # Mostrar estadísticas finales
        final_port_return = (portfolio_series.iloc[-1] - 1) * 100
        final_ipsa_return = (ipsa_series.iloc[-1] - 1) * 100
        excess_return = final_port_return - final_ipsa_return
        
        print(f"\n=== ESTADÍSTICAS FINALES - SIN RESTRICCIONES ===")
        print(f"Retorno acumulado Portafolio: {final_port_return:.2f}%")
        print(f"Retorno acumulado IPSA: {final_ipsa_return:.2f}%")
        print(f"Excess Return: {excess_return:.2f}%")
        print(f"Tracking Error promedio: {np.mean(tracking_errors):.4f}")
        
        # Estadísticas anuales en consola
        print(f"\n=== ESTADÍSTICAS ANUALES - SIN RESTRICCIONES ===")
        for stat in annual_stats:
            print(f"Año {stat[0]}: Port={stat[1]}, IPSA={stat[2]}, Excess={stat[3]}, TE={stat[4]}")
        
        # No mostrar automáticamente - el usuario puede decidir
        
        return {
            'portfolio_cumulative': portfolio_series,
            'ipsa_cumulative': ipsa_series,
            'tracking_errors': tracking_errors,
            'weights_evolution': weights_matrix,
            'dates': dates
        }

# Uso del modelo
if __name__ == "__main__":
    print("="*80)
    print("🎯 MODELOS DE TRACKING DEL IPSA")
    print("="*80)
    
    # Selección del modelo
    print("\nSeleccione el modelo a ejecutar:")
    print("1. Modelo LASSO con restricciones (Top 10 >= 60%, rebalanceo trimestral)")
    print("2. Modelo sin restricciones de peso (rebalanceo trimestral)")
    print("3. Modelo sin restricciones de peso (rebalanceo MENSUAL - 4ta semana)")
    
    while True:
        try:
            opcion = int(input("\nIngrese su opción (1, 2 o 3): "))
            if opcion in [1, 2, 3]:
                break
            else:
                print("Por favor ingrese 1, 2 o 3")
        except ValueError:
            print("Por favor ingrese un número válido")
    
    if opcion == 1:
        print("\n🔒 EJECUTANDO MODELO CON RESTRICCIONES (TRIMESTRAL)...")
        tracker = IPSALassoTracker()
        results = tracker.run_full_analysis()
        model_name = "LASSO_CON_RESTRICCIONES"
        
    elif opcion == 2:
        print("\n🔓 EJECUTANDO MODELO SIN RESTRICCIONES (TRIMESTRAL)...")
        tracker = IPSATrackerSinRestricciones()
        results = tracker.run_full_analysis()
        model_name = "SIN_RESTRICCIONES_TRIMESTRAL"
        
    else:  # opcion == 3
        print("\n📅 EJECUTANDO MODELO SIN RESTRICCIONES (MENSUAL - 4ta SEMANA)...")
        tracker = IPSATrackerMensual()
        results = tracker.run_full_analysis()
        model_name = "SIN_RESTRICCIONES_MENSUAL_4ta_SEMANA"
    
    # Preguntar al usuario si quiere ver las visualizaciones
    print("\n" + "="*60)
    activar_funcion = input("¿Desea ver los gráficos de análisis? (escriba 'si' para activar): ").strip().lower()
    
    if activar_funcion == 'si':
        print(f"\n🎯 ACTIVANDO FUNCIÓN DE VISUALIZACIÓN PARA {model_name}...")
        try:
            if hasattr(tracker, 'create_visualizations'):
                viz_results = tracker.create_visualizations()
            else:
                print("❌ Función de visualización no disponible para este modelo")
            print("✅ Visualizaciones completadas exitosamente!")
        except Exception as e:
            print(f"❌ Error al crear visualizaciones: {e}")
    else:
        print("🔄 Análisis completado sin visualizaciones.")