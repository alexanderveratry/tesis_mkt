import pandas as pd
import numpy as np
from datetime import datetime
import os

def procesar_data_limpia_v2():
    """
    Procesa el archivo Excel con estructura multi-header para extraer solo los px_last 
    de todas las empresas desde el 28-08-2020
    """
    
    # Ruta del archivo original
    archivo_original = r'bloomberg_ipsa\data_trabajo_diaria.xlsx'
    
    # Verificar si el archivo existe
    if not os.path.exists(archivo_original):
        print(f"Error: No se encuentra el archivo {archivo_original}")
        return
    
    try:
        # Leer con estructura multi-header
        print("Leyendo archivo Excel con estructura multi-header...")
        df = pd.read_excel(archivo_original, sheet_name='DATA_PX (2)', header=[0,1])
        
        print(f"Dimensiones del archivo original: {df.shape}")
        
        # Obtener la columna de fechas (primera columna)
        fecha_col = df.columns[0]
        print(f"Columna de fechas: {fecha_col}")
        
        # Convertir fechas
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
        
        # Filtrar desde 28-08-2020
        fecha_inicio = pd.to_datetime('2020-08-28')
        df_filtrado = df[df[fecha_col] >= fecha_inicio].copy()
        
        print(f"Filas después del filtro de fecha: {len(df_filtrado)}")
        
        # Identificar columnas que contienen px_last
        columnas_px_last = []
        empresas_info = []
        
        for col in df.columns:
            if len(col) == 2:  # MultiIndex con 2 niveles
                empresa, tipo_dato = col
                if 'px_last(dates=range(-3000D,0D))' in str(tipo_dato):
                    columnas_px_last.append(col)
                    # Limpiar nombre de empresa
                    empresa_limpia = str(empresa).replace(' CC Equity', '').strip()
                    empresas_info.append((col, empresa_limpia))
        
        print(f"Columnas px_last encontradas: {len(columnas_px_last)}")
        print(f"Empresas: {[info[1] for info in empresas_info[:10]]}...")  # Mostrar primeras 10
        
        # Crear el DataFrame limpio con fechas y solo px_last
        columnas_finales = [fecha_col] + columnas_px_last
        df_limpio = df_filtrado[columnas_finales].copy()
        
        # Crear nuevos nombres de columnas más legibles
        nuevos_nombres = {fecha_col: 'DATES'}
        for col_original, empresa_limpia in empresas_info:
            nuevos_nombres[col_original] = f"{empresa_limpia}_px_last"
        
        # Renombrar columnas (aplanar MultiIndex)
        df_limpio.columns = [nuevos_nombres.get(col, f"{col[0]}_{col[1]}" if len(col) == 2 else col) 
                            for col in df_limpio.columns]
        
        # Guardar el archivo limpio
        archivo_salida = 'bloomberg_ipsa/data_limpia.xlsx'
        df_limpio.to_excel(archivo_salida, index=False)
        
        print(f"\n✓ Archivo creado exitosamente: {archivo_salida}")
        print(f"✓ Dimensiones del archivo limpio: {df_limpio.shape}")
        print(f"✓ Rango de fechas: {df_limpio['DATES'].min()} a {df_limpio['DATES'].max()}")
        print(f"✓ Empresas incluidas: {len(columnas_px_last)}")
        
        # Mostrar nombres de columnas
        print(f"\nColumnas en el archivo limpio:")
        for i, col in enumerate(df_limpio.columns):
            print(f"  {i+1:2d}. {col}")
            if i >= 10:  # Mostrar solo las primeras 10
                print(f"  ... y {len(df_limpio.columns) - 11} más")
                break
        
        # Mostrar una muestra de los datos
        print("\nPrimeras 5 filas del archivo limpio:")
        print(df_limpio.head())
        
        # Mostrar estadísticas básicas
        print(f"\nEstadísticas básicas:")
        print(f"- Total de registros: {len(df_limpio)}")
        print(f"- Total de empresas: {len(columnas_px_last)}")
        
        # Contar valores nulos
        nulos_por_empresa = df_limpio.isnull().sum()
        nulos_por_empresa = nulos_por_empresa[nulos_por_empresa > 0]
        if len(nulos_por_empresa) > 0:
            print(f"- Empresas con valores nulos:")
            for col, nulos in nulos_por_empresa.items():
                if col != 'DATES':
                    print(f"  {col}: {nulos} valores nulos ({nulos/len(df_limpio)*100:.1f}%)")
        else:
            print(f"- Sin valores nulos en las columnas de precios")
        
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    procesar_data_limpia_v2()
