# Estructura y Formato de Datos - Análisis IPSA

## Descripción General

Este proyecto analiza datos financieros del índice IPSA (Índice de Precios Selectivo de Acciones) de la Bolsa de Santiago de Chile. Los datos provienen de Bloomberg y contienen precios históricos de las principales acciones que componen el índice.

## Estructura del Proyecto


## Formato de Datos de Entrada

### Archivo: `precios_limpios.xlsx`

**Estructura:**
- **Filas:** Fechas de trading (días hábiles)
- **Columnas:** 
  - `DATES`: Columna de fechas (formato datetime)
  - `[TICKER]`: Precio de cierre de cada acción
  - `IPSA index`: Valor del índice IPSA (opcional)

**Ejemplo de estructura:**
```
DATES       | ANDINAB_px_last | COPEC_px_last | CMPC_px_last | 'IPSA index'| ...
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

