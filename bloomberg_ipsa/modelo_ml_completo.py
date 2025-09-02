#!/usr/bin/env python3
"""
Modelo de Machine Learning completo para tracking del IPSA
Con restricción de que las top 10 acciones sumen al menos 60%
Permite forward looking para descubrir pesos óptimos
Objetivo: minimizar tracking error con IPSA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from scipy.optimize import minimize, differential_evolution
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar la clase base del archivo data.py
from data import BaseIPSATracker

class AdvancedMLIPSATracker(BaseIPSATracker):
    """
    Modelo avanzado de ML para tracking del IPSA con múltiples algoritmos y optimización
    """
    
    def __init__(self, data_path='precios_limpios.xlsx'):
        super().__init__(data_path)
        self.model_name = "Advanced ML"
        self.best_model = None
        self.feature_importance = None
        
    def generate_rebalance_dates(self):
        """Generar fechas de recalibración"""
        start_year = self.stock_returns.index.min().year
        end_year = self.stock_returns.index.max().year
        
        rebalance_months = [3, 6, 9, 12]  # Marzo, Junio, Septiembre, Diciembre
        
        for year in range(start_year, end_year + 1):
            for month in rebalance_months:
                third_friday = self.get_third_friday(year, month)
                
                # Encontrar la fecha de trading más cercana (forward looking permitido)
                target_date = pd.to_datetime(third_friday)
                available_dates = self.stock_returns.index
                
                if len(available_dates) > 0:
                    distances = np.abs((available_dates - target_date).days)
                    closest_idx = np.argmin(distances)
                    closest_date = available_dates[closest_idx]
                    
                    if closest_date not in self.rebalance_dates:
                        self.rebalance_dates.append(closest_date)
        
        self.rebalance_dates.sort()
        print(f"Fechas de recalibración generadas: {len(self.rebalance_dates)} fechas")
    
    def create_features(self, returns_data, ipsa_returns, lookback_days=21):
        """
        Crear features adicionales para mejorar el modelo ML
        """
        features = returns_data.copy()
        
        # Features de momentum
        for window in [5, 10, 21]:
            # Media móvil de retornos
            features[f'ma_{window}'] = returns_data.rolling(window).mean().fillna(0)
            # Volatilidad rolling
            features[f'vol_{window}'] = returns_data.rolling(window).std().fillna(0)
        
        # Features de correlación con IPSA
        rolling_corr = returns_data.corrwith(ipsa_returns, axis=0).fillna(0)
        features['ipsa_corr'] = rolling_corr
        
        # Features de retornos relativos vs IPSA
        ipsa_broadcast = ipsa_returns.values.reshape(-1, 1)
        relative_returns = returns_data.values - ipsa_broadcast
        features_rel = pd.DataFrame(relative_returns, 
                                   index=returns_data.index, 
                                   columns=[f'{col}_rel_ipsa' for col in returns_data.columns])
        features = pd.concat([features, features_rel], axis=1)
        
        return features.fillna(0)
    
    def train_ensemble_model(self, X_train, y_train):
        """
        Entrenar un ensemble de modelos y seleccionar el mejor
        """
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'ElasticNet': ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                random_state=42
            ),
            'Ridge': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
        best_score = float('inf')
        best_model = None
        best_name = None
        
        # Evaluación con Time Series Split
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, model in models.items():
            try:
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)
                    mse = np.mean((y_val - y_pred) ** 2)
                    scores.append(mse)
                
                avg_score = np.mean(scores)
                print(f"  {name}: MSE = {avg_score:.6f}")
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                print(f"  Error con {name}: {e}")
                continue
        
        if best_model is not None:
            # Entrenar el mejor modelo con todos los datos
            best_model.fit(X_train, y_train)
            print(f"  Mejor modelo: {best_name}")
            
            # Obtener feature importance si está disponible
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                self.feature_importance = np.abs(best_model.coef_)
            else:
                self.feature_importance = np.ones(len(X_train.columns)) / len(X_train.columns)
                
        return best_model, best_name
    
    def optimize_weights_advanced(self, X_train, y_train, X_future, y_future, min_top10_weight=0.6):
        """
        Optimización avanzada de pesos con restricciones
        """
        print("  Entrenando modelos ML...")
        
        # Combinar datos (forward looking)
        X_combined = pd.concat([X_train, X_future])
        y_combined = pd.concat([y_train, y_future])
        
        # Crear features adicionales
        X_features = self.create_features(X_combined, y_combined)
        
        # Seleccionar solo las columnas originales para los pesos
        original_columns = X_train.columns
        X_original = X_features[original_columns]
        
        # Entrenar ensemble de modelos
        best_model, model_name = self.train_ensemble_model(X_features, y_combined)
        
        if best_model is None:
            print("  Error: No se pudo entrenar ningún modelo, usando pesos equiponderados")
            return np.ones(len(original_columns)) / len(original_columns)
        
        # Obtener pesos iniciales basados en importancia de features
        if len(self.feature_importance) >= len(original_columns):
            initial_weights = self.feature_importance[:len(original_columns)]
        else:
            initial_weights = np.ones(len(original_columns))
        
        initial_weights = initial_weights / initial_weights.sum()
        
        # Función objetivo mejorada
        def advanced_objective(weights):
            # Retornos del portafolio
            portfolio_returns = (X_original * weights).sum(axis=1)
            
            # Tracking error
            tracking_error = np.std(portfolio_returns - y_combined) * np.sqrt(252)
            
            # Penalización por concentración excesiva
            max_weight_penalty = max(0, np.max(weights) - 0.15) * 10  # Penalizar pesos > 15%
            
            # Penalización por demasiada dispersión
            active_stocks = np.sum(weights > 0.001)
            if active_stocks > 25:  # Penalizar si hay más de 25 acciones activas
                dispersion_penalty = (active_stocks - 25) * 0.01
            else:
                dispersion_penalty = 0
            
            return tracking_error + max_weight_penalty + dispersion_penalty
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1.0},  # Suma = 1
            {'type': 'ineq', 'fun': lambda w: np.sort(w)[::-1][:10].sum() - min_top10_weight},  # Top 10 >= 60%
            {'type': 'ineq', 'fun': lambda w: w}  # Pesos positivos
        ]
        
        bounds = [(0, 0.20) for _ in range(len(initial_weights))]  # Máximo 20% por acción
        
        # Optimización con múltiples algoritmos
        optimization_results = []
        
        # 1. SLSQP
        try:
            result_slsqp = minimize(
                advanced_objective, initial_weights, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            if result_slsqp.success:
                optimization_results.append(('SLSQP', result_slsqp.x, result_slsqp.fun))
        except:
            pass
        
        # 2. Differential Evolution (algoritmo genético)
        try:
            def de_objective(weights):
                # Verificar restricciones manualmente
                if abs(weights.sum() - 1.0) > 1e-6:
                    return 1e6
                if np.sort(weights)[::-1][:10].sum() < min_top10_weight:
                    return 1e6
                if np.any(weights < 0):
                    return 1e6
                return advanced_objective(weights)
            
            bounds_de = [(0, 0.20) for _ in range(len(initial_weights))]
            result_de = differential_evolution(
                de_objective, bounds_de, 
                seed=42, maxiter=300, popsize=15
            )
            
            if result_de.success and result_de.fun < 1e5:
                # Normalizar para asegurar suma = 1
                weights_de = result_de.x / result_de.x.sum()
                optimization_results.append(('DifferentialEvolution', weights_de, de_objective(weights_de)))
        except:
            pass
        
        # Seleccionar mejor resultado
        if optimization_results:
            best_method, best_weights, best_score = min(optimization_results, key=lambda x: x[2])
            print(f"  Mejor optimización: {best_method} (score: {best_score:.6f})")
        else:
            print("  Todas las optimizaciones fallaron, usando pesos basados en ML")
            best_weights = initial_weights
        
        # Normalización final y verificación
        best_weights = best_weights / best_weights.sum()
        
        # Verificar restricción de top 10
        top10_sum = np.sort(best_weights)[::-1][:10].sum()
        if top10_sum < min_top10_weight:
            print(f"  Ajustando pesos: Top 10 suma {top10_sum:.3f}")
            # Redistribuir peso hacia las top 10
            sorted_indices = np.argsort(best_weights)[::-1]
            deficit = min_top10_weight - top10_sum
            
            for i in range(10):
                if sorted_indices[i] < len(best_weights):
                    best_weights[sorted_indices[i]] += deficit * best_weights[sorted_indices[i]] / top10_sum
            
            # Normalizar
            best_weights = best_weights / best_weights.sum()
        
        # Verificaciones finales
        assert abs(best_weights.sum() - 1.0) < 1e-8, f"Pesos no suman 1: {best_weights.sum()}"
        final_top10 = np.sort(best_weights)[::-1][:10].sum()
        print(f"  Pesos finales - Top 10: {final_top10:.3f}, Max: {np.max(best_weights):.3f}")
        
        return best_weights
    
    def backtest_model(self, lookback_window=252, forward_window=63):
        """
        Backtest del modelo avanzado
        """
        results = []
        
        for i, rebal_date in enumerate(self.rebalance_dates):
            print(f"Recalibración {i+1}/{len(self.rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Ventana de entrenamiento
            train_end = rebal_date
            train_start_idx = max(0, self.stock_returns.index.get_loc(train_end) - lookback_window)
            train_start = self.stock_returns.index[train_start_idx]
            
            # Ventana forward looking
            future_start = rebal_date
            try:
                future_end_idx = min(len(self.stock_returns.index) - 1,
                                   self.stock_returns.index.get_loc(future_start) + forward_window)
                future_end = self.stock_returns.index[future_end_idx]
            except:
                future_end = self.stock_returns.index[-1]
            
            # Datos
            X_train = self.stock_returns.loc[train_start:train_end]
            y_train = self.ipsa_returns.loc[train_start:train_end]
            X_future = self.stock_returns.loc[future_start:future_end]
            y_future = self.ipsa_returns.loc[future_start:future_end]
            
            if len(X_train) < 100 or len(X_future) < 20:
                continue
            
            # Normalizar
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index, columns=X_train.columns
            )
            X_future_scaled = pd.DataFrame(
                self.scaler.transform(X_future),
                index=X_future.index, columns=X_future.columns
            )
            
            # Optimizar pesos
            weights = self.optimize_weights_advanced(
                X_train_scaled, y_train, X_future_scaled, y_future
            )
            
            # Período de aplicación
            if i < len(self.rebalance_dates) - 1:
                period_end = self.rebalance_dates[i + 1]
            else:
                period_end = self.stock_returns.index.max()
            
            # Calcular performance
            period_mask = (self.stock_returns.index >= rebal_date) & (self.stock_returns.index <= period_end)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                portfolio_returns = (period_returns * weights).sum(axis=1)
                
                # Métricas
                tracking_error = np.std(portfolio_returns - period_ipsa_returns) * np.sqrt(252)
                cumulative_port = (1 + portfolio_returns).cumprod().iloc[-1] - 1
                cumulative_ipsa = (1 + period_ipsa_returns).cumprod().iloc[-1] - 1
                
                # Info de pesos
                top10_sum = np.sort(weights)[::-1][:10].sum()
                max_weight = np.max(weights)
                active_stocks = np.sum(weights > 0.001)
                
                result = {
                    'rebalance_date': rebal_date,
                    'period_end': period_end,
                    'tracking_error': tracking_error,
                    'portfolio_return': cumulative_port,
                    'ipsa_return': cumulative_ipsa,
                    'excess_return': cumulative_port - cumulative_ipsa,
                    'top10_weight_sum': top10_sum,
                    'max_weight': max_weight,
                    'negative_weights': 0,  # Siempre positivos
                    'active_stocks': active_stocks,
                    'weights': weights.copy()
                }
                
                results.append(result)
                
                print(f"  TE: {tracking_error:.4f}, Top10: {top10_sum:.3f}, Max: {max_weight:.3f}, Activas: {active_stocks}")
        
        self.tracking_results = results
        return results
    
    def run_full_analysis(self):
        """Ejecutar análisis completo"""
        print("=== MODELO AVANZADO DE ML PARA TRACKING DEL IPSA ===\n")
        
        self.load_data()
        self.calculate_returns()
        self.generate_rebalance_dates()
        
        print(f"\nEjecutando backtest con {len(self.rebalance_dates)} recalibraciones...")
        results = self.backtest_model()
        
        # Estadísticas
        print("\n=== ESTADÍSTICAS RESUMEN ===")
        summary = self.get_summary_statistics()
        if summary:
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        # Pesos actuales
        print("\n=== PESOS ACTUALES (TOP 15) ===")
        current_weights = self.get_current_weights()
        if current_weights is not None:
            print(current_weights.head(15))
            print(f"\nSuma total: {current_weights.sum():.10f}")
            print(f"Acciones activas: {(current_weights > 0.001).sum()}")
            print(f"Top 10 suma: {current_weights.head(10).sum():.4f}")
        
        return results
    
    def create_visualizations(self):
        """Crear visualizaciones avanzadas"""
        if not self.tracking_results:
            print("No hay resultados disponibles.")
            return
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 18))
        
        # Datos para gráficos
        dates = [r['rebalance_date'] for r in self.tracking_results]
        weights_matrix = np.array([r['weights'] for r in self.tracking_results])
        tracking_errors = [r['tracking_error'] for r in self.tracking_results]
        
        # 1. Evolución de pesos (Top 10)
        ax1 = plt.subplot(4, 1, 1)
        avg_weights = np.mean(weights_matrix, axis=0)
        top_10_indices = np.argsort(avg_weights)[-10:]
        
        colors = plt.cm.tab20(np.linspace(0, 1, 10))
        for i, idx in enumerate(top_10_indices):
            stock_name = self.stock_returns.columns[idx]
            plt.plot(dates, weights_matrix[:, idx], 
                    label=stock_name, linewidth=2, color=colors[i], marker='o', markersize=3)
        
        plt.title('Evolución de Pesos - Top 10 Acciones (ML Avanzado)', fontsize=12, fontweight='bold')
        plt.ylabel('Peso')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Tracking Error
        ax2 = plt.subplot(4, 1, 2)
        plt.plot(dates, tracking_errors, 'r-o', linewidth=2, markersize=4)
        plt.axhline(y=np.mean(tracking_errors), color='blue', linestyle='--', 
                   label=f'Promedio: {np.mean(tracking_errors):.4f}')
        plt.title('Evolución del Tracking Error', fontsize=12, fontweight='bold')
        plt.ylabel('Tracking Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Concentración de pesos
        ax3 = plt.subplot(4, 1, 3)
        top10_sums = [r['top10_weight_sum'] for r in self.tracking_results]
        max_weights = [r['max_weight'] for r in self.tracking_results]
        
        plt.plot(dates, top10_sums, 'g-o', linewidth=2, markersize=4, label='Top 10 Suma')
        plt.plot(dates, max_weights, 'purple', linewidth=2, marker='s', markersize=4, label='Peso Máximo')
        plt.axhline(y=0.6, color='red', linestyle='--', label='Mínimo Top 10 (60%)')
        plt.title('Concentración de Pesos', fontsize=12, fontweight='bold')
        plt.ylabel('Peso')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. Retornos acumulados
        ax4 = plt.subplot(4, 1, 4)
        
        # Calcular serie temporal de retornos
        all_dates = []
        portfolio_cumulative = []
        ipsa_cumulative = []
        
        cumulative_port = 1.0
        cumulative_ipsa = 1.0
        
        for i, result in enumerate(self.tracking_results):
            start_date = result['rebalance_date']
            if i < len(self.tracking_results) - 1:
                end_date = self.tracking_results[i + 1]['rebalance_date']
            else:
                end_date = self.stock_returns.index.max()
            
            period_mask = (self.stock_returns.index > start_date) & (self.stock_returns.index <= end_date)
            period_returns = self.stock_returns.loc[period_mask]
            period_ipsa_returns = self.ipsa_returns.loc[period_mask]
            
            if len(period_returns) > 0:
                weights = result['weights']
                portfolio_period_returns = (period_returns * weights).sum(axis=1)
                
                for date in period_returns.index:
                    port_ret = portfolio_period_returns.loc[date]
                    ipsa_ret = period_ipsa_returns.loc[date]
                    
                    cumulative_port *= (1 + port_ret)
                    cumulative_ipsa *= (1 + ipsa_ret)
                    
                    all_dates.append(date)
                    portfolio_cumulative.append(cumulative_port)
                    ipsa_cumulative.append(cumulative_ipsa)
        
        portfolio_series = pd.Series(portfolio_cumulative, index=all_dates)
        ipsa_series = pd.Series(ipsa_cumulative, index=all_dates)
        
        plt.plot(portfolio_series.index, (portfolio_series - 1) * 100, 
                'blue', linewidth=2, label='Portafolio ML Avanzado')
        plt.plot(ipsa_series.index, (ipsa_series - 1) * 100, 
                'red', linewidth=2, label='IPSA')
        
        # Marcar recalibraciones
        for date in dates:
            if date in portfolio_series.index:
                plt.axvline(x=date, color='gray', linestyle=':', alpha=0.5)
        
        plt.title('Retornos Acumulados: Portafolio vs IPSA', fontsize=12, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Guardar
        output_file = 'analisis_modelo_ml_avanzado.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGráficos guardados en: {output_file}")
        
        plt.show()
        
        # Crear gráfico anual
        annual_stats = self.create_annual_returns_visualization(
            portfolio_series, ipsa_series, "ML Avanzado", "ml_avanzado"
        )
        
        # Estadísticas finales
        final_port_return = (portfolio_series.iloc[-1] - 1) * 100
        final_ipsa_return = (ipsa_series.iloc[-1] - 1) * 100
        excess_return = final_port_return - final_ipsa_return
        
        print(f"\n=== RESULTADOS FINALES - ML AVANZADO ===")
        print(f"Retorno Portafolio: {final_port_return:.2f}%")
        print(f"Retorno IPSA: {final_ipsa_return:.2f}%")
        print(f"Excess Return: {excess_return:.2f}%")
        print(f"Tracking Error promedio: {np.mean(tracking_errors):.4f}")
        
        return {
            'portfolio_series': portfolio_series,
            'ipsa_series': ipsa_series,
            'tracking_errors': tracking_errors,
            'annual_stats': annual_stats
        }


def main():
    """Función principal para ejecutar el modelo completo"""
    print("="*60)
    print("MODELO DE ML AVANZADO PARA TRACKING DEL IPSA")
    print("="*60)
    print("Restricciones:")
    print("- Las 10 acciones con mayor peso deben sumar ≥ 60%")
    print("- Forward looking permitido para descubrir pesos óptimos")
    print("- Objetivo: minimizar tracking error con IPSA")
    print("- Recalibración: 3er viernes de mar, jun, sep, dic")
    print("="*60)
    
    # Crear y ejecutar modelo
    tracker = AdvancedMLIPSATracker()
    results = tracker.run_full_analysis()
    
    # Crear visualizaciones
    print("\n" + "="*40)
    print("CREANDO VISUALIZACIONES...")
    print("="*40)
    viz_data = tracker.create_visualizations()
    
    # Análisis de resultados
    if results:
        tracking_errors = [r['tracking_error'] for r in results]
        top10_sums = [r['top10_weight_sum'] for r in results]
        excess_returns = [r['excess_return'] for r in results]
        
        print("\n" + "="*40)
        print("ANÁLISIS DE RESULTADOS")
        print("="*40)
        print(f"Número de recalibraciones: {len(results)}")
        print(f"Tracking Error promedio: {np.mean(tracking_errors):.4f}")
        print(f"Tracking Error mínimo: {np.min(tracking_errors):.4f}")
        print(f"Tracking Error máximo: {np.max(tracking_errors):.4f}")
        print(f"Top 10 peso promedio: {np.mean(top10_sums):.3f}")
        print(f"Restricción Top 10 cumplida: {all(x >= 0.59 for x in top10_sums)}")
        print(f"Excess return total: {np.sum(excess_returns):.4f}")
        
        # Estadísticas por año si hay datos anuales
        if 'annual_stats' in viz_data:
            print(f"\nEstadísticas anuales guardadas en archivo PNG")
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    
    return tracker, results, viz_data


if __name__ == "__main__":
    tracker, results, viz_data = main()
