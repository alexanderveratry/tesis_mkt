"""
Script simple para generar gráfico de comparación:
Retorno Portafolio (pesos optimizados * retornos activos) vs Retorno IPSA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(data_path='data_limpia.xlsx'):
    """Carga y prepara los datos"""
    print("📊 Cargando datos...")
    data = pd.read_excel(data_path)
    data['DATES'] = pd.to_datetime(data['DATES'])
    data.set_index('DATES', inplace=True)
    data.sort_index(inplace=True)
    
    # Calcular retornos diarios
    returns = data.pct_change().dropna()
    
    print(f"✅ Datos cargados: {len(returns)} días de trading")
    print(f"📅 Período: {returns.index.min().strftime('%Y-%m-%d')} a {returns.index.max().strftime('%Y-%m-%d')}")
    
    return returns

def simulate_simple_tracking_portfolio(returns_data):
    """
    Simula un portafolio simple de tracking usando regresión lineal por períodos
    """
    print("🔄 Simulando portafolio de tracking...")
    
    # Separar variables
    ipsa_returns = returns_data['IPSA Index']
    stock_returns = returns_data.drop('IPSA Index', axis=1)
    
    # Resultados
    portfolio_results = []
    rebalancing_dates = []  # Para guardar las fechas de recalibración
    
    # Agrupar por meses para recalibración
    for year_month, month_data in returns_data.groupby([
        returns_data.index.year, 
        returns_data.index.month
    ]):
        
        # Guardar fecha de recalibración (primer día del mes)
        rebalancing_date = month_data.index.min()
        rebalancing_dates.append(rebalancing_date)
        
        # Necesitamos al menos 2 meses de datos históricos
        end_date = month_data.index.min()
        start_date = end_date - pd.DateOffset(months=2)
        
        # Datos de entrenamiento
        train_mask = (returns_data.index >= start_date) & (returns_data.index < end_date)
        X_train = stock_returns[train_mask]
        y_train = ipsa_returns[train_mask]
        
        if len(X_train) < 20:  # Necesitamos al menos 20 días de datos
            # Usar pesos iguales como fallback
            weights = np.ones(len(stock_returns.columns)) / len(stock_returns.columns)
        else:
            # Regresión lineal simple
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y_train)
            
            # Normalizar pesos
            raw_weights = model.coef_
            weights = raw_weights / np.sum(np.abs(raw_weights))
        
        # Aplicar pesos al mes actual
        month_stock_returns = stock_returns.loc[month_data.index]
        month_ipsa_returns = ipsa_returns.loc[month_data.index]
        
        for date in month_data.index:
            daily_stock_returns = month_stock_returns.loc[date].values
            portfolio_return = np.sum(daily_stock_returns * weights)
            ipsa_return = month_ipsa_returns.loc[date]
            
            portfolio_results.append({
                'fecha': date,
                'retorno_portafolio': portfolio_return,
                'retorno_ipsa': ipsa_return,
                'tracking_error': portfolio_return - ipsa_return,
                'periodo': f"{year_month[0]}-{year_month[1]:02d}"
            })
    
    return pd.DataFrame(portfolio_results), rebalancing_dates

def create_comparison_plot(results_df, rebalancing_dates, save_name='comparacion_portafolio_ipsa.png'):
    """Crea el gráfico de comparación solicitado"""
    print("📈 Creando gráfico de comparación...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Crear figura con subgráficos
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparación: Retorno Portafolio Optimizado vs Retorno IPSA\n(Líneas verticales indican recalibraciones mensuales)', 
                 fontsize=16, fontweight='bold')
    
    # Función para añadir líneas verticales de recalibración
    def add_rebalancing_lines(ax, dates_list, results_df):
        # Solo mostrar algunas líneas para no saturar el gráfico
        dates_to_show = [d for i, d in enumerate(dates_list) if i % 3 == 0]  # Cada 3 meses
        for date in dates_to_show:
            if date >= results_df['fecha'].min() and date <= results_df['fecha'].max():
                ax.axvline(x=date, color='gray', linestyle=':', alpha=0.6, linewidth=1)
    
    # 1. Serie temporal de retornos diarios
    ax1 = axes[0, 0]
    ax1.plot(results_df['fecha'], results_df['retorno_portafolio'] * 100, 
            label='Portafolio (Pesos Optimizados)', linewidth=1.2, color='#2E86AB', alpha=0.8)
    ax1.plot(results_df['fecha'], results_df['retorno_ipsa'] * 100, 
            label='IPSA', linewidth=1.2, color='#A23B72', alpha=0.8)
    
    # Añadir líneas de recalibración
    add_rebalancing_lines(ax1, rebalancing_dates, results_df)
    
    ax1.set_title('Retornos Diarios', fontweight='bold', pad=20)
    ax1.set_ylabel('Retorno (%)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter plot con línea de regresión
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['retorno_ipsa'] * 100, 
                         results_df['retorno_portafolio'] * 100, 
                         alpha=0.6, color='#F18F01', s=25, edgecolors='white', linewidth=0.5)
    
    # Línea de tracking perfecto
    min_ret = min(results_df['retorno_ipsa'].min(), results_df['retorno_portafolio'].min()) * 100
    max_ret = max(results_df['retorno_ipsa'].max(), results_df['retorno_portafolio'].max()) * 100
    ax2.plot([min_ret, max_ret], [min_ret, max_ret], '--', color='red', linewidth=2, 
            label='Tracking Perfecto (y=x)', alpha=0.8)
    
    # Línea de regresión real
    correlation = results_df['retorno_portafolio'].corr(results_df['retorno_ipsa'])
    z = np.polyfit(results_df['retorno_ipsa'] * 100, results_df['retorno_portafolio'] * 100, 1)
    p = np.poly1d(z)
    ax2.plot([min_ret, max_ret], p([min_ret, max_ret]), '-', color='green', linewidth=2, 
            label=f'Regresión Real (R²={correlation**2:.4f})', alpha=0.8)
    
    ax2.set_title('Correlación: Portafolio vs IPSA', fontweight='bold', pad=20)
    ax2.set_xlabel('Retorno IPSA (%)', fontweight='bold')
    ax2.set_ylabel('Retorno Portafolio (%)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Tracking Error a lo largo del tiempo
    ax3 = axes[1, 0]
    
    # Tracking error móvil
    rolling_te = results_df['tracking_error'].rolling(21, min_periods=5).std() * np.sqrt(252)
    ax3.plot(results_df['fecha'], rolling_te * 100, 
            color='#C73E1D', linewidth=2.5, label='TE Móvil (21 días)')
    
    # Añadir líneas de recalibración más visibles para tracking error
    dates_to_show = [d for i, d in enumerate(rebalancing_dates) if i % 2 == 0]  # Cada 2 meses
    for date in dates_to_show:
        if date >= results_df['fecha'].min() and date <= results_df['fecha'].max():
            ax3.axvline(x=date, color='orange', linestyle=':', alpha=0.8, linewidth=1.5)
    
    # Línea promedio
    avg_te = rolling_te.mean() * 100
    ax3.axhline(y=avg_te, color='#3C1518', linestyle='--', linewidth=2,
               label=f'TE Promedio: {avg_te:.2f}%', alpha=0.8)
    
    ax3.set_title('Tracking Error (Anualizado) - Líneas naranjas: Recalibraciones', fontweight='bold', pad=20)
    ax3.set_ylabel('Tracking Error (%)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Retornos acumulados
    ax4 = axes[1, 1]
    
    # Calcular retornos acumulados
    portfolio_cumret = (1 + results_df['retorno_portafolio']).cumprod()
    ipsa_cumret = (1 + results_df['retorno_ipsa']).cumprod()
    
    ax4.plot(results_df['fecha'], (portfolio_cumret - 1) * 100, 
            label='Portafolio Optimizado', linewidth=3, color='#2E86AB', alpha=0.9)
    ax4.plot(results_df['fecha'], (ipsa_cumret - 1) * 100, 
            label='IPSA', linewidth=3, color='#A23B72', alpha=0.9)
    
    # Añadir líneas de recalibración
    add_rebalancing_lines(ax4, rebalancing_dates, results_df)
    
    ax4.set_title('Performance Acumulada', fontweight='bold', pad=20)
    ax4.set_ylabel('Retorno Acumulado (%)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Añadir nota sobre las líneas verticales
    fig.text(0.02, 0.02, 'Líneas verticales grises: Recalibraciones mensuales (cada 3 meses mostradas)', 
             fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Información sobre recalibraciones
    print(f"📅 Total de recalibraciones: {len(rebalancing_dates)}")
    print(f"📅 Primera recalibración: {rebalancing_dates[0].strftime('%Y-%m-%d')}")
    print(f"📅 Última recalibración: {rebalancing_dates[-1].strftime('%Y-%m-%d')}")
    
    return fig

def create_rebalancing_timeline(results_df, rebalancing_dates, save_name='timeline_recalibraciones.png'):
    """Crea un gráfico timeline mostrando claramente los períodos de recalibración"""
    print("📊 Creando timeline de recalibraciones...")
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Calcular retornos acumulados
    portfolio_cumret = (1 + results_df['retorno_portafolio']).cumprod()
    ipsa_cumret = (1 + results_df['retorno_ipsa']).cumprod()
    
    # Gráfico principal
    ax.plot(results_df['fecha'], (portfolio_cumret - 1) * 100, 
            label='Portafolio Optimizado', linewidth=2.5, color='#2E86AB', alpha=0.9)
    ax.plot(results_df['fecha'], (ipsa_cumret - 1) * 100, 
            label='IPSA', linewidth=2.5, color='#A23B72', alpha=0.9)
    
    # Añadir todas las líneas de recalibración (más visibles)
    for i, date in enumerate(rebalancing_dates):
        if date >= results_df['fecha'].min() and date <= results_df['fecha'].max():
            ax.axvline(x=date, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Añadir etiquetas cada 6 meses para no saturar
            if i % 6 == 0:
                ax.text(date, ax.get_ylim()[1] * 0.9, f"{date.strftime('%Y-%m')}", 
                       rotation=90, fontsize=8, ha='right', va='top', alpha=0.8)
    
    ax.set_title('Performance Acumulada con Períodos de Recalibración Mensual', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Retorno Acumulado (%)', fontweight='bold')
    ax.set_xlabel('Fecha', fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Información en el gráfico
    ax.text(0.02, 0.98, f'Total Recalibraciones: {len(rebalancing_dates)}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.text(0.02, 0.02, 'Líneas rojas verticales: Recalibraciones mensuales del portafolio', 
            transform=ax.transAxes, fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def analyze_tracking_error_around_rebalancing(results_df, rebalancing_dates, save_name='tracking_error_detallado.png'):
    """Analiza el comportamiento del tracking error alrededor de las recalibraciones"""
    print("🔍 Analizando tracking error alrededor de recalibraciones...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Análisis Detallado del Tracking Error y Recalibraciones', 
                 fontsize=16, fontweight='bold')
    
    # 1. Tracking Error diario con todas las recalibraciones
    ax1 = axes[0]
    
    # Tracking error diario absoluto
    daily_te = np.abs(results_df['tracking_error']) * 100
    ax1.plot(results_df['fecha'], daily_te, 
            color='#C73E1D', linewidth=1, alpha=0.6, label='TE Diario (absoluto)')
    
    # Tracking error móvil
    rolling_te = results_df['tracking_error'].rolling(21, min_periods=5).std() * np.sqrt(252)
    ax1.plot(results_df['fecha'], rolling_te * 100, 
            color='#8B0000', linewidth=2.5, label='TE Móvil 21d (anualizado)')
    
    # Añadir todas las líneas de recalibración
    for i, date in enumerate(rebalancing_dates):
        if date >= results_df['fecha'].min() and date <= results_df['fecha'].max():
            ax1.axvline(x=date, color='blue', linestyle='--', alpha=0.7, linewidth=1)
            
            # Etiquetas cada 6 meses
            if i % 6 == 0:
                ax1.text(date, ax1.get_ylim()[1] * 0.9, f"Recal.\n{date.strftime('%Y-%m')}", 
                        rotation=0, fontsize=8, ha='center', va='top', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    ax1.set_title('Tracking Error Diario y Móvil con Recalibraciones Mensuales', fontweight='bold')
    ax1.set_ylabel('Tracking Error (%)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histograma de tracking error antes y después de recalibraciones
    ax2 = axes[1]
    
    # Calcular TE en ventanas alrededor de recalibraciones
    pre_rebal_te = []
    post_rebal_te = []
    
    for date in rebalancing_dates:
        # Buscar índice de la fecha de recalibración
        date_mask = results_df['fecha'] >= date
        if date_mask.any():
            idx = results_df[date_mask].index[0]
            
            # 5 días antes de recalibración
            pre_window = results_df.loc[max(0, idx-5):idx-1, 'tracking_error'] if idx > 5 else []
            # 5 días después de recalibración  
            post_window = results_df.loc[idx:min(len(results_df)-1, idx+5), 'tracking_error'] if idx < len(results_df)-5 else []
            
            if len(pre_window) > 0:
                pre_rebal_te.extend(np.abs(pre_window) * 100)
            if len(post_window) > 0:
                post_rebal_te.extend(np.abs(post_window) * 100)
    
    # Crear histogramas
    if pre_rebal_te and post_rebal_te:
        ax2.hist(pre_rebal_te, bins=30, alpha=0.7, label='5 días antes recalibración', 
                color='red', density=True)
        ax2.hist(post_rebal_te, bins=30, alpha=0.7, label='5 días después recalibración', 
                color='green', density=True)
        
        ax2.axvline(np.mean(pre_rebal_te), color='red', linestyle='--', linewidth=2,
                   label=f'Promedio pre: {np.mean(pre_rebal_te):.3f}%')
        ax2.axvline(np.mean(post_rebal_te), color='green', linestyle='--', linewidth=2,
                   label=f'Promedio post: {np.mean(post_rebal_te):.3f}%')
    
    ax2.set_title('Distribución del Tracking Error: Antes vs Después de Recalibraciones', fontweight='bold')
    ax2.set_xlabel('Tracking Error Absoluto (%)', fontweight='bold')
    ax2.set_ylabel('Densidad', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Estadísticas
    if pre_rebal_te and post_rebal_te:
        print(f"\n📊 ESTADÍSTICAS TRACKING ERROR ALREDEDOR DE RECALIBRACIONES:")
        print(f"TE promedio 5 días antes de recalibración : {np.mean(pre_rebal_te):.4f}%")
        print(f"TE promedio 5 días después de recalibración: {np.mean(post_rebal_te):.4f}%")
        print(f"Reducción promedio en TE                   : {np.mean(pre_rebal_te) - np.mean(post_rebal_te):+.4f}%")
        print(f"Mediana pre-recalibración                  : {np.median(pre_rebal_te):.4f}%")
        print(f"Mediana post-recalibración                 : {np.median(post_rebal_te):.4f}%")
    
    return fig

def print_performance_stats(results_df):
    """Imprime estadísticas de performance detalladas"""
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS DE PERFORMANCE")
    print("="*70)
    
    # Calcular métricas
    te_daily = results_df['tracking_error'].std()
    te_annual = te_daily * np.sqrt(252)
    correlation = results_df['retorno_portafolio'].corr(results_df['retorno_ipsa'])
    
    # Retornos anualizados
    portf_ret_annual = results_df['retorno_portafolio'].mean() * 252
    ipsa_ret_annual = results_df['retorno_ipsa'].mean() * 252
    
    # Volatilidades anualizadas
    portf_vol_annual = results_df['retorno_portafolio'].std() * np.sqrt(252)
    ipsa_vol_annual = results_df['retorno_ipsa'].std() * np.sqrt(252)
    
    # Retornos acumulados
    portfolio_cumret = (1 + results_df['retorno_portafolio']).cumprod()
    ipsa_cumret = (1 + results_df['retorno_ipsa']).cumprod()
    final_portf = portfolio_cumret.iloc[-1]
    final_ipsa = ipsa_cumret.iloc[-1]
    
    # Sharpe ratios (asumiendo risk-free rate = 0)
    sharpe_portf = portf_ret_annual / portf_vol_annual if portf_vol_annual > 0 else 0
    sharpe_ipsa = ipsa_ret_annual / ipsa_vol_annual if ipsa_vol_annual > 0 else 0
    
    print(f"📅 Período de análisis   : {results_df['fecha'].min().strftime('%Y-%m-%d')} a {results_df['fecha'].max().strftime('%Y-%m-%d')}")
    print(f"📈 Días de trading       : {len(results_df):,}")
    print(f"")
    print(f"💰 RETORNOS ANUALIZADOS:")
    print(f"   Portafolio           : {portf_ret_annual*100:+8.2f}%")
    print(f"   IPSA                 : {ipsa_ret_annual*100:+8.2f}%")
    print(f"   Diferencia           : {(portf_ret_annual-ipsa_ret_annual)*100:+8.2f}%")
    print(f"")
    print(f"📊 VOLATILIDAD ANUALIZADA:")
    print(f"   Portafolio           : {portf_vol_annual*100:8.2f}%")
    print(f"   IPSA                 : {ipsa_vol_annual*100:8.2f}%")
    print(f"")
    print(f"🎯 TRACKING ERROR:")
    print(f"   Diario               : {te_daily*100:8.4f}%")
    print(f"   Anualizado           : {te_annual*100:8.2f}%")
    print(f"")
    print(f"🔗 CORRELACIÓN          : {correlation:8.6f}")
    print(f"📈 R² (Coef. Determinación): {correlation**2:8.6f}")
    print(f"")
    print(f"⚡ SHARPE RATIO:")
    print(f"   Portafolio           : {sharpe_portf:8.4f}")
    print(f"   IPSA                 : {sharpe_ipsa:8.4f}")
    print(f"")
    print(f"🏆 RETORNOS ACUMULADOS:")
    print(f"   Portafolio           : {(final_portf-1)*100:+8.2f}%")
    print(f"   IPSA                 : {(final_ipsa-1)*100:+8.2f}%")
    print(f"   Outperformance       : {((final_portf/final_ipsa-1)*100):+8.2f}%")
    
    # Análisis de tracking error por percentiles
    te_percentiles = np.percentile(np.abs(results_df['tracking_error']) * 100, [50, 75, 90, 95, 99])
    print(f"")
    print(f"📊 DISTRIBUCIÓN TRACKING ERROR (%):")
    print(f"   Mediana (P50)        : {te_percentiles[0]:8.4f}%")
    print(f"   P75                  : {te_percentiles[1]:8.4f}%")
    print(f"   P90                  : {te_percentiles[2]:8.4f}%")
    print(f"   P95                  : {te_percentiles[3]:8.4f}%")
    print(f"   P99                  : {te_percentiles[4]:8.4f}%")

def main():
    """Función principal"""
    print("🚀 ANÁLISIS DE COMPARACIÓN: PORTAFOLIO vs IPSA")
    print("="*60)
    
    # 1. Cargar datos
    returns_data = load_data('data_limpia.xlsx')
    
    # 2. Simular portafolio de tracking
    results_df, rebalancing_dates = simulate_simple_tracking_portfolio(returns_data)
    
    if len(results_df) == 0:
        print("❌ No se pudieron generar resultados")
        return None
    
    # 3. Crear gráfico comparativo con líneas de recalibración
    fig = create_comparison_plot(results_df, rebalancing_dates)
    
    # 3.1. Crear timeline de recalibraciones
    timeline_fig = create_rebalancing_timeline(results_df, rebalancing_dates)
    
    # 3.2. Análisis detallado del tracking error
    te_analysis_fig = analyze_tracking_error_around_rebalancing(results_df, rebalancing_dates)
    
    # 4. Imprimir estadísticas
    print_performance_stats(results_df)
    
    # 5. Guardar resultados
    results_df.to_excel('resultados_comparacion_portafolio_ipsa.xlsx', index=False)
    print(f"\n💾 Resultados guardados en 'resultados_comparacion_portafolio_ipsa.xlsx'")
    
    print(f"\n✅ ANÁLISIS COMPLETADO")
    return results_df, rebalancing_dates

if __name__ == "__main__":
    results, rebalancing_dates = main()
