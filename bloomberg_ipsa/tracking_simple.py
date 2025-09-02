"""
Modelo Simplificado de Tracking IPSA con Recalibración Mensual
Enfoque en minimizar tracking error usando regresión lineal con regularización
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleIPSATracker:
    def __init__(self, data_path='data_limpia.xlsx'):
        """
        Modelo simple para tracking del IPSA
        
        Args:
            data_path (str): Ruta al archivo de datos
        """
        self.data_path = data_path
        self.data = None
        self.returns = None
        self.portfolio_weights = {}
        self.portfolio_returns = []
        self.benchmark_returns = []
        self.tracking_error = None
        
    def load_and_prepare_data(self):
        """Carga y prepara los datos"""
        print("📊 Cargando datos...")
        
        # Cargar datos
        self.data = pd.read_excel(self.data_path)
        self.data['DATES'] = pd.to_datetime(self.data['DATES'])
        self.data.set_index('DATES', inplace=True)
        self.data.sort_index(inplace=True)
        
        # Calcular retornos diarios
        self.returns = self.data.pct_change().dropna()
        
        print(f"✅ Datos cargados: {len(self.returns)} días de trading")
        print(f"📅 Período: {self.returns.index.min().strftime('%Y-%m-%d')} a {self.returns.index.max().strftime('%Y-%m-%d')}")
        print(f"📈 Acciones disponibles: {len([col for col in self.returns.columns if col != 'IPSA_px_last'])}")
        
    def get_monthly_rebalancing_dates(self):
        """Obtiene las fechas de rebalanceo mensual"""
        # Primer día hábil de cada mes
        monthly_dates = []
        
        for year_month, group in self.returns.groupby([
            self.returns.index.year, 
            self.returns.index.month
        ]):
            monthly_dates.append({
                'date': group.index.min(),
                'year_month': f"{year_month[0]}-{year_month[1]:02d}",
                'end_date': group.index.max()
            })
        
        return monthly_dates[3:]  # Empezar después de 3 meses para tener datos de entrenamiento
    
    def fit_tracking_model(self, X_train, y_train, alpha=1.0):
        """
        Ajusta el modelo de tracking usando Ridge Regression
        
        Args:
            X_train: Retornos de las acciones (variables independientes)
            y_train: Retornos del IPSA (variable dependiente)
            alpha: Parámetro de regularización
            
        Returns:
            tuple: (modelo entrenado, scaler, pesos)
        """
        # Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Entrenar modelo Ridge
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_scaled, y_train)
        
        # Obtener pesos del portafolio
        weights = dict(zip(X_train.columns, model.coef_))
        
        return model, scaler, weights
    
    def run_monthly_tracking(self):
        """Ejecuta el modelo con recalibración mensual"""
        print("\n🔄 Iniciando modelo con recalibración mensual...")
        
        # Obtener fechas de rebalanceo
        rebalancing_dates = self.get_monthly_rebalancing_dates()
        
        # Variables para almacenar resultados
        monthly_results = []
        cumulative_portfolio = []
        cumulative_benchmark = []
        
        # Separar variables
        ipsa_returns = self.returns['IPSA_px_last']
        stock_returns = self.returns.drop('IPSA_px_last', axis=1)
        
        print(f"📅 Períodos de rebalanceo: {len(rebalancing_dates)}")
        
        for i, period in enumerate(rebalancing_dates):
            print(f"\n--- Período {i+1}/{len(rebalancing_dates)}: {period['year_month']} ---")
            
            # Definir períodos de entrenamiento y predicción
            current_date = period['date']
            end_date = period['end_date']
            
            # Datos de entrenamiento: 3 meses anteriores
            train_end = current_date - pd.Timedelta(days=1)
            train_start = train_end - pd.Timedelta(days=90)  # Aproximadamente 3 meses
            
            # Filtrar datos de entrenamiento
            train_mask = (stock_returns.index >= train_start) & (stock_returns.index <= train_end)
            X_train = stock_returns[train_mask]
            y_train = ipsa_returns[train_mask]
            
            # Datos de predicción: mes actual
            pred_mask = (stock_returns.index >= current_date) & (stock_returns.index <= end_date)
            X_pred = stock_returns[pred_mask]
            y_pred = ipsa_returns[pred_mask]
            
            if len(X_train) < 30 or len(X_pred) == 0:  # Necesitamos al menos 30 días de entrenamiento
                print(f"⚠️  Datos insuficientes para {period['year_month']}")
                continue
            
            try:
                # Entrenar modelo
                model, scaler, weights = self.fit_tracking_model(X_train, y_train)
                
                # Hacer predicciones
                X_pred_scaled = scaler.transform(X_pred)
                portfolio_pred = model.predict(X_pred_scaled)
                
                # Calcular retornos reales del portafolio usando los pesos
                # Normalizar pesos para que sumen 1
                weights_array = np.array(list(weights.values()))
                weights_normalized = weights_array / np.sum(np.abs(weights_array))
                
                # Retornos del portafolio basado en pesos
                portfolio_returns_actual = (X_pred * weights_normalized).sum(axis=1)
                
                # Almacenar resultados
                for date_idx in range(len(X_pred)):
                    monthly_results.append({
                        'date': X_pred.index[date_idx],
                        'period': period['year_month'],
                        'portfolio_return': portfolio_returns_actual.iloc[date_idx],
                        'benchmark_return': y_pred.iloc[date_idx],
                        'prediction': portfolio_pred[date_idx],
                        'tracking_error_daily': portfolio_returns_actual.iloc[date_idx] - y_pred.iloc[date_idx]
                    })
                
                # Guardar pesos del período
                self.portfolio_weights[period['year_month']] = {
                    'weights': weights,
                    'weights_normalized': dict(zip(X_pred.columns, weights_normalized)),
                    'r2_score': model.score(scaler.transform(X_train), y_train),
                    'n_train_days': len(X_train),
                    'n_pred_days': len(X_pred)
                }
                
                print(f"✅ R² entrenamiento: {model.score(scaler.transform(X_train), y_train):.4f}")
                print(f"📊 Días entrenamiento: {len(X_train)}, Días predicción: {len(X_pred)}")
                
            except Exception as e:
                print(f"❌ Error en {period['year_month']}: {e}")
                continue
        
        # Convertir resultados a DataFrame
        self.results_df = pd.DataFrame(monthly_results)
        
        if len(self.results_df) > 0:
            # Calcular tracking error
            tracking_errors = self.results_df['tracking_error_daily']
            self.tracking_error = np.std(tracking_errors) * np.sqrt(252)  # Anualizado
            
            print(f"\n🎯 TRACKING ERROR TOTAL: {self.tracking_error:.4f} ({self.tracking_error*100:.2f}%)")
            
            # Calcular retornos acumulados
            self.results_df['portfolio_cumret'] = (1 + self.results_df['portfolio_return']).cumprod()
            self.results_df['benchmark_cumret'] = (1 + self.results_df['benchmark_return']).cumprod()
            
            return self.results_df
        else:
            print("❌ No se pudieron generar resultados")
            return None
    
    def analyze_performance(self):
        """Analiza el desempeño del modelo"""
        if self.results_df is None or len(self.results_df) == 0:
            print("❌ No hay resultados para analizar")
            return
        
        print("\n📈 ANÁLISIS DE DESEMPEÑO")
        print("="*50)
        
        # Estadísticas básicas
        portfolio_ret = self.results_df['portfolio_return']
        benchmark_ret = self.results_df['benchmark_return']
        
        stats = {
            'Retorno Promedio Portafolio (%)': portfolio_ret.mean() * 100,
            'Retorno Promedio IPSA (%)': benchmark_ret.mean() * 100,
            'Volatilidad Portafolio (%)': portfolio_ret.std() * np.sqrt(252) * 100,
            'Volatilidad IPSA (%)': benchmark_ret.std() * np.sqrt(252) * 100,
            'Tracking Error (%)': self.tracking_error * 100,
            'Correlación': portfolio_ret.corr(benchmark_ret),
            'Días de Trading': len(self.results_df)
        }
        
        for metric, value in stats.items():
            print(f"{metric:30s}: {value:8.4f}")
        
        # Retornos acumulados finales
        final_portfolio = self.results_df['portfolio_cumret'].iloc[-1]
        final_benchmark = self.results_df['benchmark_cumret'].iloc[-1]
        
        print(f"\n💰 RETORNOS ACUMULADOS:")
        print(f"Portafolio: {(final_portfolio-1)*100:+.2f}%")
        print(f"IPSA:       {(final_benchmark-1)*100:+.2f}%")
        print(f"Diferencia: {((final_portfolio/final_benchmark-1)*100):+.2f}%")
    
    def plot_results(self):
        """Genera gráficos de los resultados"""
        if self.results_df is None or len(self.results_df) == 0:
            print("❌ No hay datos para graficar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Tracking del IPSA - Modelo Simplificado', fontsize=16, fontweight='bold')
        
        # 1. Retornos acumulados
        ax1 = axes[0, 0]
        ax1.plot(self.results_df['date'], self.results_df['portfolio_cumret'], 
                label='Portafolio Tracking', linewidth=2, color='blue')
        ax1.plot(self.results_df['date'], self.results_df['benchmark_cumret'], 
                label='IPSA Benchmark', linewidth=2, color='red', alpha=0.8)
        ax1.set_title('Retornos Acumulados', fontweight='bold')
        ax1.set_ylabel('Retorno Acumulado')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Tracking Error a lo largo del tiempo
        ax2 = axes[0, 1]
        rolling_te = self.results_df['tracking_error_daily'].rolling(21).std() * np.sqrt(252)  # 21 días móvil
        ax2.plot(self.results_df['date'], rolling_te * 100, color='orange', linewidth=2)
        ax2.set_title('Tracking Error Móvil (21 días)', fontweight='bold')
        ax2.set_ylabel('Tracking Error (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribución de errores de tracking diarios
        ax3 = axes[1, 0]
        ax3.hist(self.results_df['tracking_error_daily'] * 100, bins=50, alpha=0.7, color='green')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Distribución de Errores de Tracking Diarios', fontweight='bold')
        ax3.set_xlabel('Error de Tracking (%)')
        ax3.set_ylabel('Frecuencia')
        ax3.grid(True, alpha=0.3)
        
        # 4. Retornos diarios: Portafolio vs IPSA
        ax4 = axes[1, 1]
        ax4.scatter(self.results_df['benchmark_return'] * 100, 
                   self.results_df['portfolio_return'] * 100, 
                   alpha=0.6, color='purple')
        
        # Línea de regresión perfecta
        min_ret = min(self.results_df['benchmark_return'].min(), self.results_df['portfolio_return'].min()) * 100
        max_ret = max(self.results_df['benchmark_return'].max(), self.results_df['portfolio_return'].max()) * 100
        ax4.plot([min_ret, max_ret], [min_ret, max_ret], 'r--', linewidth=2, label='Tracking Perfecto')
        
        ax4.set_title('Retornos Diarios: Portafolio vs IPSA', fontweight='bold')
        ax4.set_xlabel('Retorno IPSA (%)')
        ax4.set_ylabel('Retorno Portafolio (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tracking_ipsa_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_weights(self):
        """Analiza la evolución de los pesos del portafolio"""
        if not self.portfolio_weights:
            print("❌ No hay información de pesos disponible")
            return
        
        print("\n⚖️  ANÁLISIS DE PESOS DEL PORTAFOLIO")
        print("="*50)
        
        # Convertir pesos a DataFrame
        weights_data = {}
        for period, data in self.portfolio_weights.items():
            weights_data[period] = data['weights_normalized']
        
        weights_df = pd.DataFrame(weights_data).T
        weights_df.index = pd.to_datetime(weights_df.index + '-01')
        weights_df = weights_df.sort_index()
        
        # Top acciones por peso promedio
        avg_weights = weights_df.abs().mean().sort_values(ascending=False)
        
        print("🏆 TOP 10 ACCIONES POR PESO PROMEDIO:")
        for i, (stock, weight) in enumerate(avg_weights.head(10).items(), 1):
            ticker = stock.replace('_px_last', '')
            print(f"{i:2d}. {ticker:12s}: {weight:7.4f} ({weight*100:5.1f}%)")
        
        # Gráfico de evolución de pesos principales
        plt.figure(figsize=(15, 8))
        
        top_stocks = avg_weights.head(8).index
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_stocks)))
        
        for stock, color in zip(top_stocks, colors):
            ticker = stock.replace('_px_last', '')
            plt.plot(weights_df.index, weights_df[stock], 
                    label=ticker, linewidth=2.5, color=color)
        
        plt.title('Evolución de Pesos del Portafolio (Top 8 Acciones)', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Peso en el Portafolio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evolucion_pesos_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return weights_df
    
    def save_results(self):
        """Guarda todos los resultados en Excel"""
        if self.results_df is None:
            print("❌ No hay resultados para guardar")
            return
        
        print("\n💾 Guardando resultados...")
        
        with pd.ExcelWriter('tracking_ipsa_simple_results.xlsx') as writer:
            # Resultados diarios
            self.results_df.to_excel(writer, sheet_name='Resultados_Diarios', index=False)
            
            # Estadísticas de pesos por período
            if self.portfolio_weights:
                weights_summary = []
                for period, data in self.portfolio_weights.items():
                    weights_summary.append({
                        'Periodo': period,
                        'R2_Entrenamiento': data['r2_score'],
                        'Dias_Entrenamiento': data['n_train_days'],
                        'Dias_Prediccion': data['n_pred_days']
                    })
                
                weights_summary_df = pd.DataFrame(weights_summary)
                weights_summary_df.to_excel(writer, sheet_name='Resumen_Periodos', index=False)
                
                # Pesos detallados
                weights_data = {}
                for period, data in self.portfolio_weights.items():
                    weights_data[period] = data['weights_normalized']
                
                if weights_data:
                    weights_df = pd.DataFrame(weights_data).T
                    weights_df.to_excel(writer, sheet_name='Pesos_Portafolio')
        
        print("✅ Resultados guardados en 'tracking_ipsa_simple_results.xlsx'")
    
    def run_complete_analysis(self):
        """Ejecuta el análisis completo"""
        print("🚀 MODELO SIMPLIFICADO DE TRACKING DEL IPSA")
        print("="*50)
        
        # 1. Cargar y preparar datos
        self.load_and_prepare_data()
        
        # 2. Ejecutar modelo mensual
        results = self.run_monthly_tracking()
        
        if results is not None:
            # 3. Analizar desempeño
            self.analyze_performance()
            
            # 4. Generar gráficos
            self.plot_results()
            
            # 5. Analizar pesos
            self.analyze_weights()
            
            # 6. Guardar resultados
            self.save_results()
            
            print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        
        return self

def main():
    """Función principal"""
    tracker = SimpleIPSATracker('data_limpia.xlsx')
    tracker.run_complete_analysis()
    return tracker

if __name__ == "__main__":
    tracker = main()
