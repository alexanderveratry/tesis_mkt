import pandas as pd
import numpy as np
from datetime import datetime
import os

def procesar_data_limpia():
    """
    Procesa el archivo Excel para extraer solo los px_last de todas las empresas
    desde el 28-08-2020 y crear un nuevo archivo 'data limpia'
    """
    
    # Ruta del archivo original
    archivo_original = r'bloomberg_ipsa\data_trabajo_diaria.xlsx'
    
    # Verificar si el archivo existe
    if not os.path.exists(archivo_original):
        print(f"Error: No se encuentra el archivo {archivo_original}")
        return
    
    try:
        # Leer la hoja "DATA_PX (2)"
        print("Leyendo archivo Excel...")
        df = pd.read_excel(archivo_original, sheet_name='DATA_PX (2)')
        
        print(f"Dimensiones del archivo original: {df.shape}")
        print(f"Primeras columnas: {list(df.columns[:10])}")
        
        # Convertir la primera columna a fecha si no lo está
        if 'DATES' in df.columns:
            fecha_col = 'DATES'
        else:
            fecha_col = df.columns[0]  # Asumir que la primera columna son las fechas
        
        # Convertir fechas
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
        
        # Filtrar desde 28-08-2020
        fecha_inicio = pd.to_datetime('2020-08-28')
        df_filtrado = df[df[fecha_col] >= fecha_inicio].copy()
        
        print(f"Filas después del filtro de fecha: {len(df_filtrado)}")
        
        # Identificar columnas que contienen 'px_last(dates=range(-3000D,0D))'
        columnas_px_last = [col for col in df.columns if 'px_last(dates=range(-3000D,0D))' in str(col)]
        
        print(f"Columnas px_last encontradas: {len(columnas_px_last)}")
        print(f"Nombres de columnas px_last: {columnas_px_last[:5]}...")  # Mostrar primeras 5
        
        # Crear el DataFrame limpio con fechas y solo px_last
        columnas_finales = [fecha_col] + columnas_px_last
        df_limpio = df_filtrado[columnas_finales].copy()
        
        # Limpiar nombres de columnas para que sean más legibles
        nuevos_nombres = {}
        for col in columnas_px_last:
            # Extraer el nombre de la empresa del encabezado
            col_str = str(col)
            if 'CC Equity' in col_str and 'px_last(dates=range(-3000D,0D))' in col_str:
                # El formato esperado es: "EMPRESA CC Equity	px_last(dates=range(-3000D,0D))"
                # o similar, extraer la parte de la empresa
                partes = col_str.split()
                empresa = None
                for i, parte in enumerate(partes):
                    if parte.endswith('/A') or parte.endswith('AB') or (parte.isupper() and len(parte) > 1):
                        empresa = parte
                        break
                
                if empresa:
                    nuevos_nombres[col] = f"{empresa}_px_last"
                else:
                    # Si no se puede extraer, usar una versión simplificada
                    empresa_parte = col_str.split('CC Equity')[0].strip().split()[-1]
                    nuevos_nombres[col] = f"{empresa_parte}_px_last"
            else:
                nuevos_nombres[col] = col
        
        # Renombrar columnas
        df_limpio.rename(columns=nuevos_nombres, inplace=True)
        
        # Guardar el archivo limpio
        archivo_salida = 'bloomberg_ipsa/data_limpia.xlsx'
        df_limpio.to_excel(archivo_salida, index=False)
        
        print(f"\n✓ Archivo creado exitosamente: {archivo_salida}")
        print(f"✓ Dimensiones del archivo limpio: {df_limpio.shape}")
        print(f"✓ Rango de fechas: {df_limpio[fecha_col].min()} a {df_limpio[fecha_col].max()}")
        print(f"✓ Empresas incluidas: {len(columnas_px_last)}")
        
        # Mostrar una muestra de los datos
        print("\nPrimeras 5 filas del archivo limpio:")
        print(df_limpio.head())
        
        # Mostrar estadísticas básicas
        print(f"\nEstadísticas básicas:")
        print(f"- Total de registros: {len(df_limpio)}")
        print(f"- Columnas de precios: {len(columnas_px_last)}")
        print(f"- Valores nulos por columna:")
        for col in df_limpio.columns:
            nulos = df_limpio[col].isnull().sum()
            if nulos > 0:
                print(f"  {col}: {nulos} valores nulos")
        
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    procesar_data_limpia()
