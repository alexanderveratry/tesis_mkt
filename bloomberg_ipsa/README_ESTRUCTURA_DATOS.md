# Estructura y Formato de Datos - Análisis IPSA

## Descripción General

Este proyecto analiza datos financieros del índice IPSA (Índice de Precios Selectivo de Acciones) de la Bolsa de Santiago de Chile. Los datos provienen de Bloomberg y contienen precios históricos de las principales acciones que componen el índice.

## Estructura del Proyecto

```
bloomberg_ipsa/
├── data_limpia.xlsx           # Archivo de entrada (datos originales de Bloomberg)
├── trabajo.py                 # Script principal de procesamiento
├── analisis.py                # Script de análisis de correlaciones
├── data.py                    # Utilidades para manejo de datos
├── README_ESTRUCTURA_DATOS.md # Este archivo
├── 
├── Archivos generados por trabajo.py:
├── ├── precios_limpios.xlsx           # Precios procesados y limpios
├── ├── retornos_diarios.xlsx          # Retornos logarítmicos diarios
├── ├── matriz_correlacion_sin_ipsa.xlsx # Matriz de correlación (sin índice)
├── ├── resumen_estadistico.xlsx       # Estadísticas descriptivas
├── ├── resumen_correlaciones.xlsx     # Resumen de correlaciones
├── └── gráficos PNG...
├── 
└── Archivos generados por analisis.py:
    ├── matriz_correlacion_sin_ipsa_detallada.xlsx
    ├── estadisticas_correlacion.xlsx
    ├── acciones_correlacion_promedio.xlsx
    ├── estadisticas_clusters.xlsx
    └── gráficos PNG...
```

## Formato de Datos de Entrada

### Archivo: `data_limpia.xlsx`

**Estructura:**
- **Filas:** Fechas de trading (días hábiles)
- **Columnas:** 
  - `DATES`: Columna de fechas (formato datetime)
  - `[TICKER]_px_last`: Precio de cierre de cada acción
  - `IPSA_px_last`: Valor del índice IPSA (opcional)

**Ejemplo de estructura:**
```
DATES       | ANDINAB_px_last | COPEC_px_last | CMPC_px_last | IPSA_px_last | ...
2020-01-02  | 1250.5         | 6800.0        | 1890.2       | 4950.25     | ...
2020-01-03  | 1255.0         | 6820.5        | 1895.8       | 4965.80     | ...
2020-01-06  | 1248.2         | 6795.0        | 1888.5       | 4940.15     | ...
...         | ...            | ...           | ...          | ...         | ...
```

**Características de los datos originales:**
- **Período típico:** 2020-2024 (aproximadamente 4-5 años)
- **Frecuencia:** Diaria (solo días de trading)
- **Moneda:** Pesos chilenos (CLP)
- **Tipo de precio:** Precio de cierre ajustado
- **Valores faltantes:** Comunes debido a suspensiones, nuevos listados, etc.

## Proceso de Limpieza de Datos

### 1. Carga y Preparación Inicial
```python
# Cargar datos desde Excel
df = pd.read_excel('data_limpia.xlsx')
df.set_index('DATES', inplace=True)

# Limpiar nombres de columnas (eliminar _px_last)
# ANDINAB_px_last → ANDINAB
```

### 2. Análisis de Valores Faltantes
- **Identificación:** Acciones con >50% de datos faltantes
- **Criterio de exclusión:** Acciones con <50% de datos disponibles
- **Tratamiento:** Forward fill para gaps menores

### 3. Filtros Aplicados
- **Filas:** Eliminar días con >70% de valores faltantes
- **Columnas:** Eliminar acciones con <50% de datos disponibles
- **Final:** Solo observaciones completas (sin NaN)

## Estructura de Datos Procesados

### Archivo: `precios_limpios.xlsx`
```
Índice: Fechas (datetime)
Columnas: Tickers de acciones (sin sufijo _px_last)
Valores: Precios de cierre en CLP
Dimensiones típicas: ~1000 filas x 30-40 columnas
```

### Archivo: `retornos_diarios.xlsx`
```
Índice: Fechas (datetime) - una fecha menos que precios
Columnas: Tickers de acciones
Valores: Retornos logarítmicos = ln(P_t / P_t-1)
Rango típico: -0.15 a +0.15 (retornos diarios)
```

## Características de las Acciones IPSA

### Sectores Representados
- **Bancario:** CHILE, BCI, SANTANDER, ITAU
- **Retail:** FALABELLA, RIPLEY, SMU
- **Minería:** ANTOFAGASTA, COLBUN
- **Forestal:** CMPC, MASISA
- **Energía:** COPEC, ENELCHILE
- **Telecomunicaciones:** ENTEL
- **Construcción:** CENCOSUD

### Ejemplo de Tickers Comunes
```
ANDINAB    - Embotelladora Andina
BCI        - Banco de Crédito e Inversiones
CHILE      - Banco de Chile
CMPC       - CMPC
COLBUN     - Colbún
COPEC      - Empresas COPEC
CENCOSUD   - Cencosud
ENELCHILE  - Enel Chile
FALABELLA  - S.A.C.I. Falabella
ITAU       - Itaú Corpbanca
```

## Métricas Calculadas

### 1. Retornos
- **Diarios:** Retornos logarítmicos diarios
- **Anualizados:** Retorno_diario × 252 (días de trading/año)
- **Fórmula:** r_t = ln(P_t / P_t-1)

### 2. Volatilidad
- **Diaria:** Desviación estándar de retornos diarios
- **Anualizada:** Volatilidad_diaria × √252
- **Interpretación:** Medida de riesgo/variabilidad

### 3. Correlaciones
- **Matriz:** Correlaciones entre todos los pares de acciones
- **Exclusión:** Se excluye el índice IPSA del análisis
- **Rango:** -1 a +1 (típicamente 0.2 a 0.8 para acciones del mismo mercado)

### 4. Ratio de Sharpe
```
Sharpe = (Retorno_anualizado - Tasa_libre_riesgo) / Volatilidad_anualizada
Nota: En este análisis se asume tasa libre de riesgo = 0
```

## Consideraciones Especiales

### 1. Tratamiento del Índice IPSA
- **Incluido en:** Datos originales y archivos de precios/retornos
- **Excluido de:** Análisis de correlaciones y clustering
- **Razón:** Evitar sesgo (el índice es combinación de las acciones)

### 2. Datos Faltantes
- **Causa común:** Suspensiones de trading, nuevos listados, delisting
- **Estrategia:** Conservadora - solo mantener datos completos
- **Impacto:** Reduce el período de análisis pero mejora la calidad

### 3. Moneda y Ajustes
- **Moneda:** Pesos chilenos (no ajustado por inflación)
- **Splits/Dividendos:** Se asume que los precios están ajustados
- **Cambios de ticker:** No manejados automáticamente

## Flujo de Trabajo Recomendado

### 1. Preparación
```bash
# Asegurar que data_limpia.xlsx existe en el directorio
# Verificar que contiene columna DATES y columnas [TICKER]_px_last
```

### 2. Procesamiento Principal
```bash
python trabajo.py
# Genera: precios_limpios.xlsx, retornos_diarios.xlsx, matrices, gráficos
```

### 3. Análisis Avanzado
```bash
python analisis.py
# Genera: análisis de correlaciones, clustering, estadísticas detalladas
```

### 4. Interpretación
- Revisar archivos Excel generados
- Analizar gráficos PNG
- Interpretar clusters y correlaciones

## Limitaciones y Consideraciones

### 1. Calidad de Datos
- **Dependiente** de la calidad de datos de Bloomberg
- **Sensible** a errores de precios o fechas faltantes
- **Requiere** validación manual ocasional

### 2. Período de Análisis
- **Limitado** por disponibilidad de datos completos
- **Afectado** por nuevas inclusiones/exclusiones del IPSA
- **Variable** según criterios de limpieza aplicados

### 3. Supuestos Estadísticos
- **Distribución normal** de retornos (no siempre válida)
- **Estacionariedad** de correlaciones en el tiempo
- **Tasa libre de riesgo** asumida como cero

## Salidas Típicas

### Estadísticas Esperadas (Mercado Chileno)
- **Retorno promedio diario:** -0.05% a +0.05%
- **Volatilidad diaria:** 1% a 3%
- **Correlación promedio:** 0.3 a 0.6
- **Número de acciones finales:** 25-40 (de ~40-50 originales)

### Interpretación de Clusters
- **2-4 clusters** típicos basados en sectores
- **Correlación interna alta** (>0.6) dentro de clusters
- **Correlación externa baja** (<0.4) entre clusters

---

*Última actualización: Diciembre 2024*
*Para dudas o mejoras, contactar al equipo de análisis cuantitativo*
