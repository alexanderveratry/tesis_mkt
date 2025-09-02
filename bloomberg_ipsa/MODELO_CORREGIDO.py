import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class IPSALassoTrackerCorregido:
    """
    VERSIÓN CORREGIDA del modelo LASSO para seguir al índice IPSA
    Correcciones principales:
    1. Uso consistente de retornos simples para acumulación
    2. Máscaras de períodos consistentes
    3. Lógica de rebalanceo corregida
    """
    
    def __init__(self, data_path='precios_limpios.xlsx'):
        self.data_path = data_path
        self.data = None
        self.prices = None
        self.ipsa_prices = None
        self.stock_returns = None
        self.ipsa_returns = None
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
        
        # Eliminar columnas que no sean precios de acciones
        price_columns = [col for col in self.data.columns 
                        if col not in ['IPSA Index', 'IPSA index'] and not col.startswith('Unnamed')]
        
        self.prices = self.data[price_columns].copy()
        
        # Obtener el índice IPSA
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
            self.ipsa_prices = self.prices.mean(axis=1)
        
        # Eliminar filas con valores faltantes
        self.prices = self.prices.dropna()
        self.ipsa_prices = self.ipsa_prices.loc[self.prices.index]
        
        print(f"Datos cargados: {len(self.prices)} observaciones, {len(self.prices.columns)} acciones")
        print(f"Período: {self.prices.index.min()} a {self.prices.index.max()}")
        
    def calculate_returns(self):
        """
        CORREGIDO: Calcular retornos simples (no logarítmicos) para acumulación correcta
        """
        # Usar retornos simples para acumulación correcta
        self.stock_returns = self.prices.pct_change().dropna()
        self.ipsa_returns = self.ipsa_prices.pct_change().dropna()
        
        # Alinear las fechas
        common_dates = self.stock_returns.index.intersection(self.ipsa_returns.index)
        self.stock_returns = self.stock_returns.loc[common_dates]
        self.ipsa_returns = self.ipsa_returns.loc[common_dates]
        
        print(f"Retornos simples calculados para {len(common_dates)} períodos")
        print(f"Retorno promedio IPSA (diario): {self.ipsa_returns.mean():.6f}")
        print(f"Volatilidad IPSA (diaria): {self.ipsa_returns.std():.6f}")
    
    def get_third_friday(self, year, month):
        """Obtener el tercer viernes de un mes específico"""
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(days=14)
        return third_friday.date()
    
    def generate_rebalance_dates(self):
        """Generar fechas de recalibración (3er viernes de mar, jun, sep, dic)"""
        start_year = self.stock_returns.index.min().year
        end_year = self.stock_returns.index.max().year
        
        rebalance_months = [3, 6, 9, 12]
        
        for year in range(start_year, end_year + 1):
            for month in rebalance_months:
                third_friday = self.get_third_friday(year, month)
                target_date = pd.to_datetime(third_friday)
                available_dates = self.stock_returns.index[self.stock_returns.index <= target_date]
                
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    if closest_date not in self.rebalance_dates:
                        self.rebalance_dates.append(closest_date)
        
        self.rebalance_dates.sort()
        print(f"Fechas de recalibración generadas: {len(self.rebalance_dates)} fechas")
        
    def optimize_lasso_weights(self, X_train, y_train, min_top10_weight=0.6):
        """Optimizar pesos usando LASSO con restricciones"""
        alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        best_weights = None
        best_score = float('inf')
        
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=2000, positive=True)
            
            try:
                lasso.fit(X_train, y_train)
                weights = lasso.coef_
                
                # Normalizar para que sumen 1
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    
                    # Verificar restricción de top 10
                    sorted_weights = np.sort(weights)[::-1]
                    top10_sum = sorted_weights[:10].sum()
                    
                    if top10_sum >= min_top10_weight:
                        # Calcular tracking error
                        predictions = X_train @ weights
                        mse = np.mean((y_train - predictions) ** 2)
                        
                        if mse < best_score:
                            best_score = mse
                            best_weights = weights.copy()
            except:
                continue
        
        # Si no se encuentra solución, usar métodos alternativos
        if best_weights is None:
            print("  Warning: Usando Ridge como fallback")
            ridge = Ridge(alpha=0.1)
            ridge.fit(X_train, y_train)
            weights = ridge.coef_
            weights = np.maximum(weights, 0)  # Asegurar pesos positivos
            if weights.sum() > 0:
                best_weights = weights / weights.sum()
            else:
                best_weights = np.ones(len(X_train.columns)) / len(X_train.columns)
        
        # Verificación final
        assert abs(best_weights.sum() - 1.0) < 1e-10, f"ERROR: Los pesos no suman 1: {best_weights.sum()}"
        
        return best_weights
    
    def backtest_model(self, lookback_window=252):
        """
        CORREGIDO: Backtest con lógica de períodos consistente
        """
        results = []
        
        for i, rebal_date in enumerate(self.rebalance_dates):
            print(f"Procesando recalibración {i+1}/{len(self.rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Ventana de entrenamiento (solo datos ANTERIORES a la fecha de rebalanceo)
            train_end_idx = self.stock_returns.index.get_loc(rebal_date)
            train_start_idx = max(0, train_end_idx - lookback_window)
            
            train_data = self.stock_returns.iloc[train_start_idx:train_end_idx]  # NO incluir la fecha de rebalanceo
            train_ipsa = self.ipsa_returns.iloc[train_start_idx:train_end_idx]
            
            if len(train_data) < 50:
                continue
            
            # Normalizar datos de entrenamiento
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(train_data),
                index=train_data.index,
                columns=train_data.columns
            )
            
            # Optimizar pesos
            weights = self.optimize_lasso_weights(X_train_scaled, train_ipsa)
            
            # CORREGIDO: Período de aplicación (DESDE la fecha de rebalanceo)
            if i < len(self.rebalance_dates) - 1:
                period_end = self.rebalance_dates[i + 1]
                # Aplicar hasta el día ANTERIOR al siguiente rebalanceo
                period_mask = (self.stock_returns.index >= rebal_date) & (self.stock_returns.index < period_end)
            else:
                # Último período: hasta el final
                period_mask = self.stock_returns.index >= rebal_date
            
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                # CORREGIDO: Calcular retornos del portafolio
                portfolio_returns = (period_returns * weights).sum(axis=1)
                
                # CORREGIDO: Métricas con retornos simples
                tracking_error = np.std(portfolio_returns - period_ipsa_returns) * np.sqrt(252)
                
                # Retornos acumulados correctos
                cumulative_portfolio = (1 + portfolio_returns).prod() - 1
                cumulative_ipsa = (1 + period_ipsa_returns).prod() - 1
                
                # Información de pesos
                top10_weights = np.sort(weights)[-10:].sum()
                active_stocks = np.sum(weights > 0.001)
                
                result = {
                    'rebalance_date': rebal_date,
                    'period_end': period_returns.index[-1] if len(period_returns) > 0 else rebal_date,
                    'tracking_error': tracking_error,
                    'portfolio_return': cumulative_portfolio,
                    'ipsa_return': cumulative_ipsa,
                    'excess_return': cumulative_portfolio - cumulative_ipsa,
                    'top10_weight_sum': top10_weights,
                    'active_stocks': active_stocks,
                    'weights': weights.copy(),
                    'period_days': len(period_returns)
                }
                
                results.append(result)
                
                print(f"  Período: {len(period_returns)} días")
                print(f"  Tracking Error: {tracking_error:.4f}")
                print(f"  Portfolio Return: {cumulative_portfolio:.4f}")
                print(f"  IPSA Return: {cumulative_ipsa:.4f}")
                print(f"  Top 10 weight: {top10_weights:.3f}")
        
        self.tracking_results = results
        return results
    
    def get_summary_statistics(self):
        """Obtener estadísticas resumen del backtest"""
        if not self.tracking_results:
            print("No hay resultados disponibles.")
            return None
        
        avg_tracking_error = np.mean([r['tracking_error'] for r in self.tracking_results])
        avg_top10_weight = np.mean([r['top10_weight_sum'] for r in self.tracking_results])
        avg_active_stocks = np.mean([r['active_stocks'] for r in self.tracking_results])
        
        # CORREGIDO: Calcular retorno total acumulado correctamente
        total_portfolio_return = 1.0
        total_ipsa_return = 1.0
        
        for result in self.tracking_results:
            total_portfolio_return *= (1 + result['portfolio_return'])
            total_ipsa_return *= (1 + result['ipsa_return'])
        
        total_portfolio_return -= 1
        total_ipsa_return -= 1
        
        summary = {
            'total_rebalances': len(self.tracking_results),
            'avg_tracking_error': avg_tracking_error,
            'avg_top10_weight_sum': avg_top10_weight,
            'avg_active_stocks': avg_active_stocks,
            'total_portfolio_return': total_portfolio_return,
            'total_ipsa_return': total_ipsa_return,
            'total_excess_return': total_portfolio_return - total_ipsa_return,
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
        """Ejecutar análisis completo CORREGIDO"""
        print("=== MODELO LASSO CORREGIDO PARA TRACKING DEL IPSA ===\n")
        
        # Cargar y preparar datos
        self.load_data()
        self.calculate_returns()
        self.generate_rebalance_dates()
        
        # Ejecutar backtest
        print(f"\nEjecutando backtest corregido con {len(self.rebalance_dates)} recalibraciones...")
        results = self.backtest_model()
        
        # Mostrar estadísticas resumen
        print("\n=== ESTADÍSTICAS RESUMEN CORREGIDAS ===")
        summary = self.get_summary_statistics()
        if summary:
            print(f"Total de rebalanceos: {summary['total_rebalances']}")
            print(f"Tracking Error promedio: {summary['avg_tracking_error']:.4f}")
            print(f"Top 10 weight promedio: {summary['avg_top10_weight_sum']:.3f}")
            print(f"Acciones activas promedio: {summary['avg_active_stocks']:.1f}")
            print(f"Retorno total Portfolio: {summary['total_portfolio_return']:.4f} ({summary['total_portfolio_return']*100:.2f}%)")
            print(f"Retorno total IPSA: {summary['total_ipsa_return']:.4f} ({summary['total_ipsa_return']*100:.2f}%)")
            print(f"Excess Return total: {summary['total_excess_return']:.4f} ({summary['total_excess_return']*100:.2f}%)")
            print(f"Restricción Top 10 cumplida: {summary['min_top10_constraint_met']}")
        
        # Mostrar pesos actuales
        print("\n=== PESOS ACTUALES (TOP 15) ===")
        current_weights = self.get_current_weights()
        if current_weights is not None:
            print(current_weights.head(15))
            print(f"\nSuma total de pesos: {current_weights.sum():.10f}")
            print(f"Número de acciones con peso > 0: {(current_weights > 0).sum()}")
            print(f"Peso de las top 10: {current_weights.head(10).sum():.4f}")
        
        return results

# Uso del modelo corregido
if __name__ == "__main__":
    print("="*80)
    print("🔧 MODELO LASSO CORREGIDO PARA TRACKING DEL IPSA")
    print("="*80)
    
    tracker = IPSALassoTrackerCorregido()
    results = tracker.run_full_analysis()
