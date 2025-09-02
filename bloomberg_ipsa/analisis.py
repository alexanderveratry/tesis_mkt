import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
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
    
    # Verificar archivos que se van a generar
    archivos_analisis = [
        'matriz_correlacion_sin_ipsa_detallada.xlsx',
        'estadisticas_correlacion.xlsx',
        'acciones_correlacion_promedio.xlsx'
    ]
    
    archivos_graficos = [
        'correlacion_detallada_sin_ipsa.png',
        'estadisticas_correlacion.png',
        'correlaciones_extremas.png'
    ]
    
    archivos_existentes = [archivo for archivo in archivos_analisis + archivos_graficos if os.path.exists(archivo)]
    
    sobrescribir = True
    if archivos_existentes:
        print(f"\n=== ARCHIVOS DE ANÁLISIS EXISTENTES ===")
        for archivo in archivos_existentes:
            print(f"  ✓ {archivo}")
        
        respuesta = input("\n¿Deseas sobrescribir los archivos de análisis existentes? (s/n): ").lower().strip()
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

def analizar_correlaciones_sin_ipsa(df_retornos):
    """
    Análisis detallado de correlaciones excluyendo IPSA
    """
    print(f"\n=== ANÁLISIS DE CORRELACIONES SIN IPSA ===")
    
    # Identificar y excluir columnas relacionadas con IPSA
    columnas_originales = list(df_retornos.columns)
    columnas_ipsa = [col for col in df_retornos.columns if 'IPSA' in col.upper()]
    
    print(f"Columnas totales: {len(columnas_originales)}")
    if columnas_ipsa:
        print(f"Columnas IPSA encontradas y excluidas: {columnas_ipsa}")
        df_sin_ipsa = df_retornos.drop(columns=columnas_ipsa)
    else:
        print("No se encontraron columnas IPSA")
        df_sin_ipsa = df_retornos.copy()
    
    print(f"Acciones para análisis: {len(df_sin_ipsa.columns)}")
    
    # Calcular matriz de correlación
    correlaciones = df_sin_ipsa.corr()
    
    # Heatmap completo con anotaciones
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(correlaciones, dtype=bool))
    sns.heatmap(correlaciones, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8}, 
                annot_kws={'size': 8})
    plt.title('Matriz de Correlación - Acciones IPSA (Excluyendo Índice)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('correlacion_detallada_sin_ipsa.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlaciones, df_sin_ipsa

def estadisticas_correlacion(correlaciones):
    """
    Análisis estadístico de las correlaciones
    """
    print(f"\n=== ESTADÍSTICAS DE CORRELACIÓN ===")
    
    # Obtener valores únicos de correlación (triángulo superior, sin diagonal)
    corr_values = correlaciones.values
    mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)
    correlaciones_unicas = corr_values[mask]
    
    # Estadísticas básicas
    stats = {
        'Media': correlaciones_unicas.mean(),
        'Mediana': np.median(correlaciones_unicas),
        'Desviación Estándar': correlaciones_unicas.std(),
        'Mínimo': correlaciones_unicas.min(),
        'Máximo': correlaciones_unicas.max(),
        'Q1': np.percentile(correlaciones_unicas, 25),
        'Q3': np.percentile(correlaciones_unicas, 75)
    }
    
    print("Estadísticas de correlación entre acciones:")
    for stat, valor in stats.items():
        print(f"  {stat}: {valor:.4f}")
    
    # Distribución de correlaciones
    plt.figure(figsize=(12, 8))
    
    # Histograma
    plt.subplot(2, 2, 1)
    plt.hist(correlaciones_unicas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribución de Correlaciones')
    plt.xlabel('Coeficiente de Correlación')
    plt.ylabel('Frecuencia')
    plt.axvline(stats['Media'], color='red', linestyle='--', label=f"Media: {stats['Media']:.3f}")
    plt.axvline(stats['Mediana'], color='green', linestyle='--', label=f"Mediana: {stats['Mediana']:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(correlaciones_unicas)
    plt.title('Box Plot de Correlaciones')
    plt.ylabel('Coeficiente de Correlación')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot aproximado
    plt.subplot(2, 2, 3)
    sorted_corr = np.sort(correlaciones_unicas)
    n = len(sorted_corr)
    theoretical_quantiles = np.linspace(0, 1, n)
    plt.scatter(theoretical_quantiles, sorted_corr, alpha=0.6)
    plt.title('Distribución Acumulada')
    plt.xlabel('Cuantiles Teóricos')
    plt.ylabel('Correlaciones Observadas')
    plt.grid(True, alpha=0.3)
    
    # Densidad
    plt.subplot(2, 2, 4)
    plt.hist(correlaciones_unicas, bins=30, density=True, alpha=0.7, color='lightcoral')
    plt.title('Densidad de Correlaciones')
    plt.xlabel('Coeficiente de Correlación')
    plt.ylabel('Densidad')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('estadisticas_correlacion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats, correlaciones_unicas

def pares_correlacion_extremos(correlaciones, n_pares=10):
    """
    Identifica pares con correlaciones más altas y más bajas
    """
    print(f"\n=== PARES CON CORRELACIONES EXTREMAS ===")
    
    # Crear máscara para obtener solo el triángulo superior
    mask = np.triu(np.ones_like(correlaciones, dtype=bool), k=1)
    correlaciones_masked = correlaciones.where(mask)
    
    # Convertir a Series y ordenar
    correlaciones_serie = correlaciones_masked.stack().sort_values(ascending=False)
    
    print(f"\nTOP {n_pares} PARES MÁS CORRELACIONADOS:")
    print("-" * 50)
    for i, ((stock1, stock2), corr) in enumerate(correlaciones_serie.head(n_pares).items(), 1):
        print(f"{i:2d}. {stock1:10s} - {stock2:10s}: {corr:6.4f}")
    
    print(f"\nTOP {n_pares} PARES MENOS CORRELACIONADOS:")
    print("-" * 50)
    for i, ((stock1, stock2), corr) in enumerate(correlaciones_serie.tail(n_pares).items(), 1):
        print(f"{i:2d}. {stock1:10s} - {stock2:10s}: {corr:6.4f}")
    
    # Crear gráfico de correlaciones extremas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top correlaciones
    top_corr = correlaciones_serie.head(n_pares)
    top_labels = [f"{pair[0]}-{pair[1]}" for pair in top_corr.index]
    ax1.barh(range(len(top_corr)), top_corr.values, color='darkgreen', alpha=0.7)
    ax1.set_yticks(range(len(top_corr)))
    ax1.set_yticklabels(top_labels, fontsize=9)
    ax1.set_xlabel('Coeficiente de Correlación')
    ax1.set_title(f'Top {n_pares} Correlaciones Más Altas')
    ax1.grid(True, alpha=0.3)
    
    # Bottom correlaciones
    bottom_corr = correlaciones_serie.tail(n_pares)
    bottom_labels = [f"{pair[0]}-{pair[1]}" for pair in bottom_corr.index]
    ax2.barh(range(len(bottom_corr)), bottom_corr.values, color='darkred', alpha=0.7)
    ax2.set_yticks(range(len(bottom_corr)))
    ax2.set_yticklabels(bottom_labels, fontsize=9)
    ax2.set_xlabel('Coeficiente de Correlación')
    ax2.set_title(f'Top {n_pares} Correlaciones Más Bajas')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlaciones_extremas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlaciones_serie

def guardar_analisis_correlacion(correlaciones, stats, df_sin_ipsa):
    """
    Guarda los resultados del análisis de correlación
    """
    print(f"\n=== GUARDANDO ANÁLISIS DE CORRELACIÓN ===")
    
    # Guardar matriz de correlación sin IPSA
    correlaciones.to_excel('matriz_correlacion_sin_ipsa_detallada.xlsx')
    print("✓ Matriz de correlación guardada en: matriz_correlacion_sin_ipsa_detallada.xlsx")
    
    # Guardar estadísticas
    stats_df = pd.DataFrame(list(stats.items()), columns=['Estadística', 'Valor'])
    stats_df.to_excel('estadisticas_correlacion.xlsx', index=False)
    print("✓ Estadísticas guardadas en: estadisticas_correlacion.xlsx")
    
    # Guardar lista de acciones analizadas
    acciones_df = pd.DataFrame({
        'Accion': df_sin_ipsa.columns,
        'Correlacion_Promedio': [correlaciones[col].drop(col).mean() for col in correlaciones.columns]
    })
    acciones_df = acciones_df.sort_values('Correlacion_Promedio', ascending=False)
    acciones_df.to_excel('acciones_correlacion_promedio.xlsx', index=False)
    print("✓ Ranking de acciones por correlación promedio guardado en: acciones_correlacion_promedio.xlsx")

def main():
    """
    Función principal para análisis de correlaciones
    """
    print("="*60)
    print("ANÁLISIS DE CORRELACIONES - ACCIONES IPSA (SIN ÍNDICE)")
    print("="*60)
    
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
        
        # Intentar cargar matriz existente para comparación
        matriz_existente = cargar_matriz_existente()
        
        # Análisis de correlaciones sin IPSA
        correlaciones, df_sin_ipsa = analizar_correlaciones_sin_ipsa(df_retornos)
        
        # Comparar con matriz existente si existe
        if matriz_existente is not None:
            diferencias = abs(correlaciones - matriz_existente).max().max()
            print(f"\n=== COMPARACIÓN CON ANÁLISIS ANTERIOR ===")
            print(f"Diferencia máxima en correlaciones: {diferencias:.6f}")
            if diferencias < 0.0001:
                print("✓ Los datos son prácticamente idénticos al análisis anterior")
            else:
                print("⚠ Se detectaron diferencias significativas")
        
        # Estadísticas detalladas
        stats, correlaciones_unicas = estadisticas_correlacion(correlaciones)
        
        # Pares extremos
        serie_correlaciones = pares_correlacion_extremos(correlaciones)
        
        # Guardar resultados
        guardar_analisis_correlacion(correlaciones, stats, df_sin_ipsa)
        
        print(f"\n" + "="*60)
        print("ANÁLISIS DE CORRELACIONES COMPLETADO")
        print("="*60)
        print("Archivos generados:")
        print("• matriz_correlacion_sin_ipsa_detallada.xlsx")
        print("• estadisticas_correlacion.xlsx")
        print("• acciones_correlacion_promedio.xlsx")
        print("\nGráficos generados:")
        print("• correlacion_detallada_sin_ipsa.png")
        print("• estadisticas_correlacion.png")
        print("• correlaciones_extremas.png")
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
