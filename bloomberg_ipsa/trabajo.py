
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



def calcular_retornos_diarios(df_precios):
    """
    Calcula retornos diarios para cada acción
    """
    print(f"\n=== CALCULANDO RETORNOS DIARIOS ===")
    
    # Calcular retornos logarítmicos
    df_retornos = np.log(df_precios / df_precios.shift(1))
    
    # Eliminar la primera fila (que será NA por el shift)
    df_retornos = df_retornos.dropna()
    
    print(f"Retornos calculados: {df_retornos.shape}")
    print(f"Período de retornos: {df_retornos.index.min()} a {df_retornos.index.max()}")
    
    # Estadísticas descriptivas de retornos
    print(f"\n=== ESTADÍSTICAS DE RETORNOS ===")
    stats = df_retornos.describe()
    print(f"Retorno promedio diario (%):")
    print((stats.loc['mean'] * 100).round(4))
    
    print(f"\nVolatilidad diaria (%):")
    print((stats.loc['std'] * 100).round(4))
    
    print(f"\nRetorno anualizado (%):")
    print((stats.loc['mean'] * 252 * 100).round(2))
    
    print(f"\nVolatilidad anualizada (%):")
    print((stats.loc['std'] * np.sqrt(252) * 100).round(2))
    
    return df_retornos

def crear_graficos_individuales(df_precios, df_retornos):
    """
    Crea gráficos individuales por separado
    """
    print(f"\n=== CREANDO GRÁFICOS ===")
    
    # 1. Gráfico de evolución de precios normalizados
    plt.figure(figsize=(12, 8))
    df_normalizado = df_precios / df_precios.iloc[0] * 100
    
    # Mostrar todas las acciones con transparencia
    for col in df_normalizado.columns:
        plt.plot(df_normalizado.index, df_normalizado[col], alpha=0.6, linewidth=1)
    
    plt.title('Evolución de Precios Normalizados (Base 100) - Todas las Acciones', fontsize=14)
    plt.xlabel('Fecha')
    plt.ylabel('Precio Normalizado')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trabajo_res/evolucion_precios_normalizados.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Gráfico de distribución de retornos promedio
    plt.figure(figsize=(10, 6))
    retornos_promedio = df_retornos.mean() * 100
    plt.hist(retornos_promedio, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribución de Retornos Promedio Diarios', fontsize=14)
    plt.xlabel('Retorno Promedio Diario (%)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    plt.axvline(retornos_promedio.mean(), color='red', linestyle='--', 
                label=f'Media: {retornos_promedio.mean():.4f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('trabajo_res/distribucion_retornos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Gráfico de volatilidad por acción
    plt.figure(figsize=(12, 8))
    volatilidades = df_retornos.std() * 100
    volatilidades_ordenadas = volatilidades.sort_values(ascending=True)
    volatilidades_ordenadas.plot(kind='barh', color='lightcoral')
    plt.title('Volatilidad Diaria por Acción', fontsize=14)
    plt.xlabel('Volatilidad Diaria (%)')
    plt.tight_layout()
    plt.savefig('trabajo_res/volatilidad_acciones.png', dpi=300, bbox_inches='tight')
    plt.show()

def analizar_matriz_correlacion(df_retornos):
    """
    Análisis detallado de la matriz de correlación
    """
    print(f"\n=== ANÁLISIS DETALLADO DE MATRIZ DE CORRELACIÓN ===")
    
    # Identificar y excluir IPSA si existe
    df_retornos_sin_ipsa = df_retornos.copy()
    columnas_ipsa = [col for col in df_retornos.columns if 'IPSA' in col.upper()]
    
    if columnas_ipsa:
        print(f"Excluyendo del análisis de correlación: {columnas_ipsa}")
        df_retornos_sin_ipsa = df_retornos_sin_ipsa.drop(columns=columnas_ipsa)
        print(f"Acciones para análisis de correlación: {len(df_retornos_sin_ipsa.columns)}")
    
    # Calcular matriz de correlación sin IPSA
    correlaciones = df_retornos_sin_ipsa.corr()
    
    # Crear gráfico de matriz de correlación completa
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlaciones, dtype=bool))
    sns.heatmap(correlaciones, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación de Retornos (Sin IPSA) - Triángulo Superior', fontsize=14)
    plt.tight_layout()
    plt.savefig('trabajo_res/matriz_correlacion_completa_sin_ipsa.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # Estadísticas de correlación
    corr_values = correlaciones.values
    mask_triangular = ~np.eye(corr_values.shape[0], dtype=bool)
    correlaciones_unicas = corr_values[np.triu(mask_triangular, k=1)]
    
    print(f"Estadísticas de correlación:")
    print(f"  - Correlación promedio: {correlaciones_unicas.mean():.4f}")
    print(f"  - Correlación mediana: {np.median(correlaciones_unicas):.4f}")
    print(f"  - Correlación mínima: {correlaciones_unicas.min():.4f}")
    print(f"  - Correlación máxima: {correlaciones_unicas.max():.4f}")
    print(f"  - Desviación estándar: {correlaciones_unicas.std():.4f}")
    
    # Encontrar pares con mayor y menor correlación
    print(f"\n=== PARES DE ACCIONES CON MAYOR CORRELACIÓN ===")
    correlaciones_flat = correlaciones.where(np.triu(np.ones_like(correlaciones, dtype=bool), k=1))
    correlaciones_stack = correlaciones_flat.stack().sort_values(ascending=False)
    
    print("Top 10 pares más correlacionados:")
    for i, ((accion1, accion2), corr) in enumerate(correlaciones_stack.head(10).items()):
        print(f"  {i+1:2d}. {accion1} - {accion2}: {corr:.4f}")
    
    print(f"\n=== PARES DE ACCIONES CON MENOR CORRELACIÓN ===")
    print("Top 10 pares menos correlacionados:")
    for i, ((accion1, accion2), corr) in enumerate(correlaciones_stack.tail(10).items()):
        print(f"  {i+1:2d}. {accion1} - {accion2}: {corr:.4f}")
    
    # Análisis por sectores (basado en nombres de acciones)
    print(f"\n=== ANÁLISIS DE CORRELACIÓN POR GRUPOS ===")
    
    # Gráfico de distribución de correlaciones
    plt.figure(figsize=(10, 6))
    plt.hist(correlaciones_unicas, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title('Distribución de Correlaciones entre Acciones', fontsize=14)
    plt.xlabel('Coeficiente de Correlación')
    plt.ylabel('Frecuencia')
    plt.axvline(correlaciones_unicas.mean(), color='red', linestyle='--', 
                label=f'Media: {correlaciones_unicas.mean():.4f}')
    plt.axvline(np.median(correlaciones_unicas), color='green', linestyle='--', 
                label=f'Mediana: {np.median(correlaciones_unicas):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trabajo_res/distribucion_correlaciones.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlaciones

def analizar_datos_limpios(df_precios, df_retornos):
    """
    Análisis exploratorio de los datos limpios
    """
    print(f"\n=== ANÁLISIS DE DATOS LIMPIOS ===")
    
    # Crear gráficos individuales
    crear_graficos_individuales(df_precios, df_retornos)
    
    # Análisis de correlación detallado
    matriz_correlacion = analizar_matriz_correlacion(df_retornos)
    
    # Estadísticas generales
    retornos_promedio = df_retornos.mean() * 100
    volatilidades = df_retornos.std() * 100
    
    # Mostrar top y bottom performers
    print(f"\n=== TOP 5 ACCIONES POR RETORNO PROMEDIO ===")
    top_retornos = retornos_promedio.sort_values(ascending=False).head()
    for accion, retorno in top_retornos.items():
        print(f"{accion}: {retorno:.4f}% diario ({retorno*252:.2f}% anual)")
    
    print(f"\n=== BOTTOM 5 ACCIONES POR RETORNO PROMEDIO ===")
    bottom_retornos = retornos_promedio.sort_values(ascending=True).head()
    for accion, retorno in bottom_retornos.items():
        print(f"{accion}: {retorno:.4f}% diario ({retorno*252:.2f}% anual)")
    
    print(f"\n=== TOP 5 ACCIONES MÁS VOLÁTILES ===")
    top_volatilidad = volatilidades.sort_values(ascending=False).head()
    for accion, vol in top_volatilidad.items():
        print(f"{accion}: {vol:.4f}% diario ({vol*np.sqrt(252):.2f}% anual)")
    
    print(f"\n=== TOP 5 ACCIONES MENOS VOLÁTILES ===")
    bottom_volatilidad = volatilidades.sort_values(ascending=True).head()
    for accion, vol in bottom_volatilidad.items():
        print(f"{accion}: {vol:.4f}% diario ({vol*np.sqrt(252):.2f}% anual)")
    
    return matriz_correlacion

def guardar_resultados(df_precios, df_retornos, matriz_correlacion):
    """
    Guarda los resultados en archivos Excel
    """
    print(f"\n=== GUARDANDO RESULTADOS ===")
    
    # Guardar precios limpios
    df_precios.to_excel('trabajo_res/precios_limpios.xlsx')
    print("✓ Precios limpios guardados en: trabajo_res/precios_limpios.xlsx")
    # Guardar retornos
    df_retornos.to_excel('trabajo_res/retornos_diarios.xlsx')
    print("✓ Retornos diarios guardados en: trabajo_res/retornos_diarios.xlsx")
    # Guardar matriz de correlación sin IPSA
    matriz_correlacion.to_excel('trabajo_res/matriz_correlacion_sin_ipsa.xlsx')
    print("✓ Matriz de correlación (sin IPSA) guardada en: trabajo_res/matriz_correlacion_sin_ipsa.xlsx")
    # Crear un resumen estadístico
    resumen = pd.DataFrame({
        'Retorno_Promedio_Diario': df_retornos.mean(),
        'Volatilidad_Diaria': df_retornos.std(),
        'Retorno_Anualizado': df_retornos.mean() * 252,
        'Volatilidad_Anualizada': df_retornos.std() * np.sqrt(252),
        'Ratio_Sharpe_Anual': (df_retornos.mean() * 252) / (df_retornos.std() * np.sqrt(252)),
        'Min_Retorno': df_retornos.min(),
        'Max_Retorno': df_retornos.max(),
        'Observaciones': df_retornos.count()
    })
    
    resumen.to_excel('trabajo_res/resumen_estadistico.xlsx')
    print("✓ Resumen estadístico guardado en: trabajo_res/resumen_estadistico.xlsx")
    
    # Crear resumen de correlaciones
    corr_values = matriz_correlacion.values
    mask_triangular = ~np.eye(corr_values.shape[0], dtype=bool)
    correlaciones_unicas = corr_values[np.triu(mask_triangular, k=1)]
    
    resumen_correlacion = pd.DataFrame({
        'Estadistica': ['Promedio', 'Mediana', 'Mínima', 'Máxima', 'Desviación Estándar'],
        'Valor': [
            correlaciones_unicas.mean(),
            np.median(correlaciones_unicas),
            correlaciones_unicas.min(),
            correlaciones_unicas.max(),
            correlaciones_unicas.std()
        ]
    })
    
    resumen_correlacion.to_excel('trabajo_res/resumen_correlaciones.xlsx', index=False)
    print("✓ Resumen de correlaciones guardado en: trabajo_res/resumen_correlaciones.xlsx")


def main():
    """
    Función principal: parte desde precios_limpios.xlsx y realiza análisis, gráficos y guardado de resultados.
    """
    print("="*60)
    print("PROCESAMIENTO DE DATOS FINANCIEROS - IPSA")
    print("="*60)
    try:
        # 1. Cargar precios limpios
        print("Cargando precios limpios desde precios_limpios.xlsx ...")
        df_precios_limpios = pd.read_excel('precios_limpios.xlsx', index_col=0)

        # 2. Calcular retornos
        df_retornos = calcular_retornos_diarios(df_precios_limpios)

        # 3. Análisis (incluye matriz de correlación)
        matriz_correlacion = analizar_datos_limpios(df_precios_limpios, df_retornos)

        # 4. Guardar resultados
        guardar_resultados(df_precios_limpios, df_retornos, matriz_correlacion)

        print(f"\n" + "="*60)
        print("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("Archivos generados en trabajo_res/:")
        print("• precios_limpios.xlsx")
        print("• retornos_diarios.xlsx")
        print("• matriz_correlacion_sin_ipsa.xlsx")
        print("• resumen_estadistico.xlsx")
        print("• resumen_correlaciones.xlsx")
        print("\nGráficos generados en trabajo_res/:")
        print("• evolucion_precios_normalizados.png")
        print("• distribucion_retornos.png")
        print("• volatilidad_acciones.png")
        print("• matriz_correlacion_completa_sin_ipsa.png")
        print("• distribucion_correlaciones.png")
    except Exception as e:
        print(f"Error durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()