"""
Modelo de Machine Learning para replicar el IPSA con menor tracking error
Recalibración mensual del portafolio
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class IPSATrackingModel:
    def __init__(self, data_path='data_limpia.xlsx'):
        """
        Inicializa el modelo de tracking del IPSA
        
        Args:
            data_path (str): Ruta al archivo de datos limpios
        """
        self.data_path = data_path
        self.data = None
        self.returns_data = None
        self.models = {}
        self.results = {}
        self.monthly_weights = {}
        self.tracking_errors = {}
        
    def load_data(self):
        """Carga los datos desde el archivo Excel"""
        print("Cargando datos...")
        self.data = pd.read_excel(self.data_path)
        
        # Convertir la columna de fechas
        self.data['DATES'] = pd.to_datetime(self.data['DATES'])
        self.data.set_index('DATES', inplace=True)
        
        # Ordenar por fecha
        self.data.sort_index(inplace=True)
        
        print(f"Datos cargados: {len(self.data)} filas, {len(self.data.columns)} columnas")
        print(f"Período: {self.data.index.min()} a {self.data.index.max()}")
        
    def calculate_returns(self, method='simple'):
        """
        Calcula los retornos de todas las series
        
        Args:
            method (str): 'simple' o 'log' para tipo de retorno
        """
        print(f"Calculando retornos ({method})...")
        
        if method == 'log':
            self.returns_data = np.log(self.data / self.data.shift(1))
        else:
            self.returns_data = self.data.pct_change()
        
        # Eliminar valores nulos
        self.returns_data.dropna(inplace=True)
        
        print(f"Retornos calculados: {len(self.returns_data)} observaciones")
        
    def prepare_features_target(self):
        """Prepara las variables independientes y dependiente"""
        # Variable dependiente: retornos del IPSA
        if 'IPSA Index' in self.returns_data.columns:
            self.y = self.returns_data['IPSA Index']
            # Variables independientes: retornos de todas las acciones (excepto IPSA)
            self.X = self.returns_data.drop('IPSA Index', axis=1)
        else:
            raise ValueError("No se encontró la columna IPSA Index en los datos")
        
        print(f"Variables independientes: {len(self.X.columns)} acciones")
        print(f"Observaciones disponibles: {len(self.X)}")
        
    def get_monthly_periods(self):
        """Genera los períodos mensuales para recalibración"""
        dates = self.returns_data.index
        monthly_dates = []
        
        # Agrupar por año-mes
        for year_month, group in self.returns_data.groupby([
            self.returns_data.index.year, 
            self.returns_data.index.month
        ]):
            monthly_dates.append({
                'start': group.index.min(),
                'end': group.index.max(),
                'year_month': f"{year_month[0]}-{year_month[1]:02d}"
            })
        
        return monthly_dates
    
    def train_models(self):
        """Entrena diferentes modelos de ML"""
        # Definir modelos a probar
        models_config = {
            'Linear_Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        monthly_periods = self.get_monthly_periods()
        
        for model_name, model in models_config.items():
            print(f"\n=== Entrenando modelo: {model_name} ===")
            
            self.models[model_name] = []
            self.monthly_weights[model_name] = {}
            model_predictions = []
            model_actual = []
            model_dates = []
            
            # Necesitamos al menos 3 meses de datos para entrenar
            for i in range(3, len(monthly_periods)):
                # Datos de entrenamiento: 3 meses anteriores
                train_periods = monthly_periods[i-3:i]
                train_start = train_periods[0]['start']
                train_end = train_periods[-1]['end']
                
                # Datos de predicción: mes actual
                pred_period = monthly_periods[i]
                pred_start = pred_period['start']
                pred_end = pred_period['end']
                
                # Preparar datos de entrenamiento
                X_train = self.X.loc[train_start:train_end]
                y_train = self.y.loc[train_start:train_end]
                
                # Preparar datos de predicción
                X_pred = self.X.loc[pred_start:pred_end]
                y_pred = self.y.loc[pred_start:pred_end]
                
                # Verificar que tenemos datos suficientes
                if len(X_train) < 10 or len(X_pred) == 0:
                    continue
                
                # Normalizar datos (importante para algunos modelos)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_pred_scaled = scaler.transform(X_pred)
                
                try:
                    # Entrenar modelo
                    model_copy = models_config[model_name]
                    model_copy.fit(X_train_scaled, y_train)
                    
                    # Predecir
                    predictions = model_copy.predict(X_pred_scaled)
                    
                    # Guardar resultados
                    model_predictions.extend(predictions)
                    model_actual.extend(y_pred.values)
                    model_dates.extend(y_pred.index)
                    
                    # Guardar pesos del modelo (si está disponible)
                    if hasattr(model_copy, 'coef_'):
                        # Normalizar pesos para que sumen 1 (interpretación como portafolio)
                        raw_weights = model_copy.coef_
                        normalized_weights = raw_weights / np.sum(np.abs(raw_weights))
                        
                        weights = dict(zip(X_train.columns, normalized_weights))
                        raw_weights_dict = dict(zip(X_train.columns, raw_weights))
                        
                        self.monthly_weights[model_name][pred_period['year_month']] = {
                            'weights_normalized': weights,
                            'weights_raw': raw_weights_dict,
                            'sum_abs_weights': np.sum(np.abs(normalized_weights)),
                            'sum_weights': np.sum(normalized_weights)
                        }
                    
                    self.models[model_name].append({
                        'model': model_copy,
                        'scaler': scaler,
                        'period': pred_period['year_month'],
                        'train_period': f"{train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}",
                        'pred_period': f"{pred_start.strftime('%Y-%m-%d')} to {pred_end.strftime('%Y-%m-%d')}"
                    })
                    
                except Exception as e:
                    print(f"Error en período {pred_period['year_month']}: {e}")
                    continue
            
            # Calcular métricas del modelo
            if model_predictions:
                self.results[model_name] = {
                    'predictions': np.array(model_predictions),
                    'actual': np.array(model_actual),
                    'dates': model_dates,
                    'mse': mean_squared_error(model_actual, model_predictions),
                    'rmse': np.sqrt(mean_squared_error(model_actual, model_predictions)),
                    'mae': mean_absolute_error(model_actual, model_predictions),
                    'r2': r2_score(model_actual, model_predictions)
                }
                
                # Calcular tracking error (desviación estándar de la diferencia de retornos)
                tracking_error = np.std(np.array(model_actual) - np.array(model_predictions)) * np.sqrt(252)  # Anualizado
                self.tracking_errors[model_name] = tracking_error
                
                print(f"Tracking Error: {tracking_error:.4f} ({tracking_error*100:.2f}%)")
                print(f"R²: {self.results[model_name]['r2']:.4f}")
                print(f"RMSE: {self.results[model_name]['rmse']:.6f}")
    
    def evaluate_models(self):
        """Evalúa y compara todos los modelos"""
        print("\n" + "="*60)
        print("RESUMEN DE RESULTADOS")
        print("="*60)
        
        results_summary = []
        
        for model_name, result in self.results.items():
            results_summary.append({
                'Modelo': model_name,
                'Tracking_Error': self.tracking_errors[model_name],
                'R²': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'N_Predicciones': len(result['predictions'])
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Tracking_Error')
        
        print(results_df.round(6))
        
        # Mejor modelo
        best_model = results_df.iloc[0]['Modelo']
        print(f"\n🏆 MEJOR MODELO: {best_model}")
        print(f"Tracking Error: {results_df.iloc[0]['Tracking_Error']:.4f} ({results_df.iloc[0]['Tracking_Error']*100:.2f}%)")
        
        return results_df
    
    def plot_results(self):
        """Genera gráficos de los resultados"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Modelos de Tracking del IPSA', fontsize=16, fontweight='bold')
        
        # 1. Tracking Errors
        ax1 = axes[0, 0]
        models = list(self.tracking_errors.keys())
        errors = list(self.tracking_errors.values())
        
        bars = ax1.bar(models, errors, color='skyblue', alpha=0.7)
        ax1.set_title('Tracking Error por Modelo', fontweight='bold')
        ax1.set_ylabel('Tracking Error (Anualizado)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Resaltar el mejor modelo
        min_idx = np.argmin(errors)
        bars[min_idx].set_color('gold')
        bars[min_idx].set_alpha(1.0)
        
        # 2. R² Score
        ax2 = axes[0, 1]
        r2_scores = [self.results[model]['r2'] for model in models]
        bars2 = ax2.bar(models, r2_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('R² Score por Modelo', fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Resaltar el mejor R²
        max_idx = np.argmax(r2_scores)
        bars2[max_idx].set_color('gold')
        bars2[max_idx].set_alpha(1.0)
        
        # 3. Predicciones vs Real (mejor modelo)
        best_model = min(self.tracking_errors.keys(), key=lambda x: self.tracking_errors[x])
        ax3 = axes[1, 0]
        
        actual = self.results[best_model]['actual']
        predicted = self.results[best_model]['predictions']
        
        ax3.scatter(actual, predicted, alpha=0.6, color='blue')
        ax3.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax3.set_xlabel('Retornos Reales IPSA')
        ax3.set_ylabel('Retornos Predichos')
        ax3.set_title(f'Predicciones vs Real - {best_model}', fontweight='bold')
        
        # 4. Serie temporal del mejor modelo
        ax4 = axes[1, 1]
        dates = self.results[best_model]['dates']
        
        ax4.plot(dates, actual, label='IPSA Real', color='blue', alpha=0.7)
        ax4.plot(dates, predicted, label='IPSA Predicho', color='red', alpha=0.7)
        ax4.set_title(f'Serie Temporal - {best_model}', fontweight='bold')
        ax4.set_xlabel('Fecha')
        ax4.set_ylabel('Retornos')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('resultados_tracking_ipsa.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_weights(self, model_name=None):
        """Analiza la evolución de los pesos del portafolio"""
        if model_name is None:
            # Usar el mejor modelo
            model_name = min(self.tracking_errors.keys(), key=lambda x: self.tracking_errors[x])
        
        if model_name not in self.monthly_weights:
            print(f"No hay información de pesos para el modelo {model_name}")
            return
        
        weights_data = self.monthly_weights[model_name]
        if not weights_data:
            print(f"No hay datos de pesos para el modelo {model_name}")
            return
        
        # Extraer pesos normalizados y información de suma
        normalized_weights = {}
        weight_sums = {}
        
        for period, data in weights_data.items():
            if isinstance(data, dict) and 'weights_normalized' in data:
                normalized_weights[period] = data['weights_normalized']
                weight_sums[period] = {
                    'sum_abs_weights': data.get('sum_abs_weights', 0),
                    'sum_weights': data.get('sum_weights', 0)
                }
            else:
                # Fallback para formato anterior (solo pesos raw)
                raw_weights = np.array(list(data.values()))
                norm_weights = raw_weights / np.sum(np.abs(raw_weights))
                normalized_weights[period] = dict(zip(data.keys(), norm_weights))
                weight_sums[period] = {
                    'sum_abs_weights': np.sum(np.abs(norm_weights)),
                    'sum_weights': np.sum(norm_weights)
                }
        
        # Convertir a DataFrame
        weights_df = pd.DataFrame(normalized_weights).T
        weights_df.index = pd.to_datetime(weights_df.index + '-01')
        weights_df = weights_df.sort_index()
        
        # Verificar sumas de pesos por período
        print(f"\n=== VERIFICACIÓN DE PESOS - {model_name} ===")
        print("Suma de valores absolutos de pesos por período (debería ser ~1.0):")
        for period, sums in weight_sums.items():
            print(f"{period}: |pesos| = {sums['sum_abs_weights']:.6f}, Σpesos = {sums['sum_weights']:+.6f}")
        
        # Gráfico de evolución de pesos
        plt.figure(figsize=(15, 10))
        
        # Seleccionar las 10 acciones con mayor peso promedio absoluto
        avg_weights = weights_df.abs().mean().sort_values(ascending=False)
        top_stocks = avg_weights.head(10).index
        
        for stock in top_stocks:
            plt.plot(weights_df.index, weights_df[stock], label=stock.replace('_px_last', ''), linewidth=2)
        
        plt.title(f'Evolución de Pesos del Portafolio (Normalizados) - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Peso en el Portafolio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'evolucion_pesos_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Estadísticas de pesos
        print(f"\n=== ANÁLISIS DE PESOS - {model_name} ===")
        print("Top 10 acciones por peso promedio absoluto:")
        for i, (stock, weight) in enumerate(avg_weights.head(10).items(), 1):
            avg_actual_weight = weights_df[stock].mean()
            print(f"{i:2d}. {stock.replace('_px_last', ''):15s}: |peso|={weight:7.4f}, peso_prom={avg_actual_weight:+7.4f}")
        
        # Verificación final de normalización
        total_abs_weights = weights_df.abs().sum(axis=1)
        print(f"\nVerificación de normalización:")
        print(f"Suma de |pesos| promedio: {total_abs_weights.mean():.6f} (debería ser ~1.0)")
        print(f"Rango suma |pesos|: [{total_abs_weights.min():.6f}, {total_abs_weights.max():.6f}]")
        
        return weights_df
    
    def verify_weight_normalization(self):
        """Verifica que todos los pesos estén correctamente normalizados"""
        print("\n🔍 VERIFICACIÓN DE NORMALIZACIÓN DE PESOS")
        print("="*60)
        
        for model_name, weights_data in self.monthly_weights.items():
            if not weights_data:
                continue
                
            print(f"\nModelo: {model_name}")
            print("-" * 40)
            
            normalization_issues = []
            
            for period, data in weights_data.items():
                if isinstance(data, dict) and 'weights_normalized' in data:
                    sum_abs = data.get('sum_abs_weights', 0)
                    sum_weights = data.get('sum_weights', 0)
                    
                    # Verificar que la suma de valores absolutos sea ~1
                    if abs(sum_abs - 1.0) > 0.001:
                        normalization_issues.append(f"{period}: |suma| = {sum_abs:.6f}")
                    
                    print(f"{period}: ✓ |pesos|={sum_abs:.6f}, Σpesos={sum_weights:+.6f}")
                else:
                    normalization_issues.append(f"{period}: Formato de pesos incorrecto")
            
            if normalization_issues:
                print(f"\n⚠️  Problemas de normalización encontrados:")
                for issue in normalization_issues:
                    print(f"   - {issue}")
            else:
                print(f"\n✅ Todos los pesos están correctamente normalizados")
    
    def calculate_portfolio_returns(self, model_name=None):
        """Calcula los retornos reales del portafolio usando los pesos normalizados"""
        if model_name is None:
            model_name = min(self.tracking_errors.keys(), key=lambda x: self.tracking_errors[x])
        
        if model_name not in self.monthly_weights:
            print(f"No hay pesos disponibles para {model_name}")
            return None
        
        print(f"\n📊 CALCULANDO RETORNOS REALES DEL PORTAFOLIO - {model_name}")
        
        # Obtener retornos de acciones (sin IPSA)
        stock_returns = self.returns_data.drop('IPSA_px_last', axis=1)
        ipsa_returns = self.returns_data['IPSA_px_last']
        
        portfolio_returns = []
        dates_used = []
        
        for period, data in self.monthly_weights[model_name].items():
            if isinstance(data, dict) and 'weights_normalized' in data:
                weights = data['weights_normalized']
                
                # Encontrar fechas de este período
                period_start = pd.to_datetime(period + '-01')
                
                if period == list(self.monthly_weights[model_name].keys())[-1]:
                    # Último período
                    period_end = stock_returns.index.max()
                else:
                    # Calcular siguiente mes
                    next_month = period_start + pd.DateOffset(months=1)
                    period_end = next_month - pd.Timedelta(days=1)
                
                # Filtrar datos del período
                period_mask = (stock_returns.index >= period_start) & (stock_returns.index <= period_end)
                period_returns = stock_returns[period_mask]
                
                if len(period_returns) == 0:
                    continue
                
                # Calcular retornos del portafolio
                weights_array = np.array([weights.get(col, 0) for col in period_returns.columns])
                
                for date_idx in range(len(period_returns)):
                    date = period_returns.index[date_idx]
                    daily_returns = period_returns.iloc[date_idx].values
                    portfolio_return = np.sum(daily_returns * weights_array)
                    
                    portfolio_returns.append(portfolio_return)
                    dates_used.append(date)
        
        if portfolio_returns:
            portfolio_df = pd.DataFrame({
                'fecha': dates_used,
                'retorno_portafolio': portfolio_returns,
                'retorno_ipsa': [ipsa_returns[date] for date in dates_used]
            })
            
            portfolio_df['tracking_error'] = portfolio_df['retorno_portafolio'] - portfolio_df['retorno_ipsa']
            
            # Calcular métricas
            te_daily = portfolio_df['tracking_error'].std()
            te_annual = te_daily * np.sqrt(252)
            correlation = portfolio_df['retorno_portafolio'].corr(portfolio_df['retorno_ipsa'])
            
            print(f"Tracking Error Diario: {te_daily:.6f}")
            print(f"Tracking Error Anual: {te_annual:.4f} ({te_annual*100:.2f}%)")
            print(f"Correlación: {correlation:.6f}")
            print(f"Días analizados: {len(portfolio_df)}")
            
            return portfolio_df
        
        return None
    
    def plot_portfolio_vs_ipsa_comparison(self, model_name=None):
        """Crea gráfico comparativo de retornos portafolio vs IPSA"""
        if model_name is None:
            model_name = min(self.tracking_errors.keys(), key=lambda x: self.tracking_errors[x])
        
        print(f"\n📊 GENERANDO GRÁFICO COMPARATIVO - {model_name}")
        
        # Calcular retornos del portafolio
        portfolio_df = self.calculate_portfolio_returns(model_name)
        
        if portfolio_df is None or len(portfolio_df) == 0:
            print("❌ No se pudieron calcular los retornos del portafolio")
            return
        
        # Crear figura con múltiples subgráficos
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comparación Retornos: Portafolio Optimizado vs IPSA - {model_name}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Serie temporal de retornos
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df['fecha'], portfolio_df['retorno_portafolio'] * 100, 
                label='Portafolio Optimizado', linewidth=1.5, color='blue', alpha=0.8)
        ax1.plot(portfolio_df['fecha'], portfolio_df['retorno_ipsa'] * 100, 
                label='IPSA', linewidth=1.5, color='red', alpha=0.8)
        ax1.set_title('Retornos Diarios (%)', fontweight='bold')
        ax1.set_ylabel('Retorno (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Scatter plot: Portafolio vs IPSA
        ax2 = axes[0, 1]
        ax2.scatter(portfolio_df['retorno_ipsa'] * 100, 
                   portfolio_df['retorno_portafolio'] * 100, 
                   alpha=0.6, color='purple', s=20)
        
        # Línea de regresión perfecta (y = x)
        min_ret = min(portfolio_df['retorno_ipsa'].min(), portfolio_df['retorno_portafolio'].min()) * 100
        max_ret = max(portfolio_df['retorno_ipsa'].max(), portfolio_df['retorno_portafolio'].max()) * 100
        ax2.plot([min_ret, max_ret], [min_ret, max_ret], 'r--', linewidth=2, label='Tracking Perfecto')
        
        # Línea de regresión real
        correlation = portfolio_df['retorno_portafolio'].corr(portfolio_df['retorno_ipsa'])
        z = np.polyfit(portfolio_df['retorno_ipsa'] * 100, portfolio_df['retorno_portafolio'] * 100, 1)
        p = np.poly1d(z)
        ax2.plot([min_ret, max_ret], p([min_ret, max_ret]), 'g-', linewidth=2, 
                label=f'Regresión (R²={correlation**2:.4f})')
        
        ax2.set_title('Retornos: Portafolio vs IPSA', fontweight='bold')
        ax2.set_xlabel('Retorno IPSA (%)')
        ax2.set_ylabel('Retorno Portafolio (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Tracking Error a lo largo del tiempo
        ax3 = axes[1, 0]
        
        # Calcular tracking error móvil (21 días)
        portfolio_df_sorted = portfolio_df.sort_values('fecha')
        rolling_te = portfolio_df_sorted['tracking_error'].rolling(21, min_periods=5).std() * np.sqrt(252)
        
        ax3.plot(portfolio_df_sorted['fecha'], rolling_te * 100, 
                color='orange', linewidth=2, label='TE Móvil (21d)')
        ax3.axhline(y=rolling_te.mean() * 100, color='red', linestyle='--', 
                   label=f'TE Promedio: {rolling_te.mean()*100:.2f}%')
        ax3.set_title('Tracking Error Móvil (Anualizado)', fontweight='bold')
        ax3.set_ylabel('Tracking Error (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Retornos acumulados
        ax4 = axes[1, 1]
        
        # Calcular retornos acumulados
        portfolio_cumret = (1 + portfolio_df_sorted['retorno_portafolio']).cumprod()
        ipsa_cumret = (1 + portfolio_df_sorted['retorno_ipsa']).cumprod()
        
        ax4.plot(portfolio_df_sorted['fecha'], (portfolio_cumret - 1) * 100, 
                label='Portafolio Optimizado', linewidth=2.5, color='blue')
        ax4.plot(portfolio_df_sorted['fecha'], (ipsa_cumret - 1) * 100, 
                label='IPSA', linewidth=2.5, color='red', alpha=0.8)
        
        ax4.set_title('Retornos Acumulados (%)', fontweight='bold')
        ax4.set_ylabel('Retorno Acumulado (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'comparacion_portafolio_vs_ipsa_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir estadísticas detalladas
        print(f"\n📈 ESTADÍSTICAS COMPARATIVAS - {model_name}")
        print("="*60)
        
        # Calcular métricas
        te_daily = portfolio_df['tracking_error'].std()
        te_annual = te_daily * np.sqrt(252)
        correlation = portfolio_df['retorno_portafolio'].corr(portfolio_df['retorno_ipsa'])
        
        portf_ret_annual = portfolio_df['retorno_portafolio'].mean() * 252
        ipsa_ret_annual = portfolio_df['retorno_ipsa'].mean() * 252
        
        portf_vol_annual = portfolio_df['retorno_portafolio'].std() * np.sqrt(252)
        ipsa_vol_annual = portfolio_df['retorno_ipsa'].std() * np.sqrt(252)
        
        final_portf = portfolio_cumret.iloc[-1]
        final_ipsa = ipsa_cumret.iloc[-1]
        
        print(f"Período analizado    : {portfolio_df['fecha'].min().strftime('%Y-%m-%d')} a {portfolio_df['fecha'].max().strftime('%Y-%m-%d')}")
        print(f"Días de trading      : {len(portfolio_df)}")
        print(f"")
        print(f"RETORNOS ANUALIZADOS:")
        print(f"Portafolio          : {portf_ret_annual*100:+7.2f}%")
        print(f"IPSA                : {ipsa_ret_annual*100:+7.2f}%")
        print(f"Diferencia          : {(portf_ret_annual-ipsa_ret_annual)*100:+7.2f}%")
        print(f"")
        print(f"VOLATILIDAD ANUALIZADA:")
        print(f"Portafolio          : {portf_vol_annual*100:7.2f}%")
        print(f"IPSA                : {ipsa_vol_annual*100:7.2f}%")
        print(f"")
        print(f"TRACKING ERROR:")
        print(f"Diario              : {te_daily*100:7.4f}%")
        print(f"Anualizado          : {te_annual*100:7.2f}%")
        print(f"")
        print(f"CORRELACIÓN         : {correlation:7.6f}")
        print(f"")
        print(f"RETORNOS ACUMULADOS:")
        print(f"Portafolio          : {(final_portf-1)*100:+7.2f}%")
        print(f"IPSA                : {(final_ipsa-1)*100:+7.2f}%")
        print(f"Diferencia          : {((final_portf/final_ipsa-1)*100):+7.2f}%")
        
        return portfolio_df
    
    def save_results(self):
        """Guarda los resultados en archivos Excel"""
        print("\nGuardando resultados...")
        
        # Resumen de modelos
        results_summary = []
        for model_name, result in self.results.items():
            results_summary.append({
                'Modelo': model_name,
                'Tracking_Error': self.tracking_errors[model_name],
                'Tracking_Error_Pct': self.tracking_errors[model_name] * 100,
                'R_Squared': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'N_Predicciones': len(result['predictions'])
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Tracking_Error')
        
        # Guardar en Excel
        with pd.ExcelWriter('resultados_modelo_tracking_ipsa.xlsx') as writer:
            results_df.to_excel(writer, sheet_name='Resumen_Modelos', index=False)
            
            # Predicciones del mejor modelo
            best_model = results_df.iloc[0]['Modelo']
            pred_df = pd.DataFrame({
                'Fecha': self.results[best_model]['dates'],
                'IPSA_Real': self.results[best_model]['actual'],
                'IPSA_Predicho': self.results[best_model]['predictions'],
                'Diferencia': self.results[best_model]['actual'] - self.results[best_model]['predictions']
            })
            pred_df.to_excel(writer, sheet_name=f'Predicciones_{best_model}', index=False)
            
            # Pesos del mejor modelo
            if best_model in self.monthly_weights:
                weights_data = self.monthly_weights[best_model]
                if weights_data:
                    # Extraer pesos normalizados
                    normalized_weights = {}
                    raw_weights = {}
                    weight_stats = {}
                    
                    for period, data in weights_data.items():
                        if isinstance(data, dict) and 'weights_normalized' in data:
                            normalized_weights[period] = data['weights_normalized']
                            raw_weights[period] = data['weights_raw']
                            weight_stats[period] = {
                                'sum_abs_weights': data.get('sum_abs_weights', 0),
                                'sum_weights': data.get('sum_weights', 0)
                            }
                        else:
                            # Fallback para formato anterior
                            raw_weights[period] = data
                            norm_vals = np.array(list(data.values()))
                            norm_vals = norm_vals / np.sum(np.abs(norm_vals))
                            normalized_weights[period] = dict(zip(data.keys(), norm_vals))
                    
                    # Guardar pesos normalizados (principales)
                    if normalized_weights:
                        norm_df = pd.DataFrame(normalized_weights).T
                        norm_df.to_excel(writer, sheet_name=f'Pesos_Normalizados_{best_model}')
                    
                    # Guardar pesos raw (para referencia)
                    if raw_weights:
                        raw_df = pd.DataFrame(raw_weights).T
                        raw_df.to_excel(writer, sheet_name=f'Pesos_Raw_{best_model}')
                    
                    # Guardar estadísticas de pesos
                    if weight_stats:
                        stats_df = pd.DataFrame(weight_stats).T
                        stats_df.to_excel(writer, sheet_name=f'Estadisticas_Pesos_{best_model}')
        
        print("✅ Resultados guardados en 'resultados_modelo_tracking_ipsa.xlsx'")
    
    def run_complete_analysis(self):
        """Ejecuta el análisis completo"""
        print("🚀 INICIANDO ANÁLISIS DE TRACKING DEL IPSA")
        print("="*50)
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Calcular retornos
        self.calculate_returns(method='simple')  # Puedes cambiar a 'log' si prefieres
        
        # 3. Preparar variables
        self.prepare_features_target()
        
        # 4. Entrenar modelos
        self.train_models()
        
        # 5. Evaluar modelos
        results_summary = self.evaluate_models()
        
        # 6. Visualizar resultados
        self.plot_results()
        
        # 7. Verificar normalización de pesos
        self.verify_weight_normalization()
        
        # 8. Analizar pesos
        self.analyze_weights()
        
        # 9. Calcular retornos reales del portafolio
        portfolio_returns = self.calculate_portfolio_returns()
        
        # 10. Crear gráfico comparativo detallado
        comparison_df = self.plot_portfolio_vs_ipsa_comparison()
        
        # 11. Guardar resultados
        self.save_results()
        
        print("\n✅ ANÁLISIS COMPLETADO")
        return results_summary

def main():
    """Función principal"""
    # Crear instancia del modelo
    modelo = IPSATrackingModel('data_limpia.xlsx')
    
    # Ejecutar análisis completo
    resultados = modelo.run_complete_analysis()
    
    return modelo, resultados

if __name__ == "__main__":
    modelo, resultados = main()
