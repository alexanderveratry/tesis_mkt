import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib para gráficos más atractivos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Leer el archivo JSON
print("Cargando datos...")
df = pd.read_json('response_1755622727310.json')

# Convertir la columna FECHA a tipo datetime y eliminar zona horaria
df['FECHA'] = pd.to_datetime(df['FECHA']).dt.tz_localize(None)

# Filtrar por PERIODO = "DI"
df_filtrado = df[df['PERIODO'] == 'DI'].copy()

print(f"Datos filtrados por PERIODO='DI': {len(df_filtrado)} registros")

# Agrupar por INDICE y DIVIDENDO, ordenar por fecha
df_agrupado = df_filtrado.groupby(['INDICE', 'DIVIDENDO']).apply(
    lambda x: x.sort_values('FECHA')
).reset_index(drop=True)

# Obtener información sobre los grupos disponibles
grupos_disponibles = df_agrupado.groupby(['INDICE', 'DIVIDENDO']).size().reset_index(name='count')
print(f"\nGrupos disponibles (INDICE, DIVIDENDO):")
for _, row in grupos_disponibles.head(10).iterrows():
    print(f"  {row['INDICE']} - {row['DIVIDENDO']}: {row['count']} registros")

# Crear visualizaciones
def crear_graficos(df_data, indice_filtro=None, dividendo_filtro=None):
    """Crear gráficos de evolución temporal"""
    
    # Filtrar datos si se especifica
    if indice_filtro:
        df_data = df_data[df_data['INDICE'] == indice_filtro]
    if dividendo_filtro:
        df_data = df_data[df_data['DIVIDENDO'] == dividendo_filtro]
    
    if len(df_data) == 0:
        print("No hay datos para los filtros especificados")
        return
    
    # Configurar el tamaño de la figura
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Evolución de Índices Financieros a lo largo del tiempo', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Evolución del IND_ACT por INDICE
    ax1 = axes[0, 0]
    for indice in df_data['INDICE'].unique()[:5]:  # Limitar a 5 índices para claridad
        data_indice = df_data[df_data['INDICE'] == indice]
        for dividendo in data_indice['DIVIDENDO'].unique():
            data_grupo = data_indice[data_indice['DIVIDENDO'] == dividendo]
            if len(data_grupo) > 1:
                ax1.plot(data_grupo['FECHA'], data_grupo['IND_ACT'], 
                        marker='o', markersize=3, linewidth=2, alpha=0.8,
                        label=f'{indice} ({dividendo})')
    
    ax1.set_title('Evolución del Índice Actual (IND_ACT)', fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor del Índice')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Evolución del NUM_ACC_COM por INDICE
    ax2 = axes[0, 1]
    for indice in df_data['INDICE'].unique()[:5]:
        data_indice = df_data[df_data['INDICE'] == indice]
        for dividendo in data_indice['DIVIDENDO'].unique():
            data_grupo = data_indice[data_indice['DIVIDENDO'] == dividendo]
            if len(data_grupo) > 1:
                ax2.plot(data_grupo['FECHA'], data_grupo['NUM_ACC_COM'], 
                        marker='s', markersize=3, linewidth=2, alpha=0.8,
                        label=f'{indice} ({dividendo})')
    
    ax2.set_title('Evolución del Número de Acciones en Común (NUM_ACC_COM)', fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Número de Acciones')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # Gráfico 3: Distribución de IND_VAR por INDICE
    ax3 = axes[1, 0]
    indices_top = df_data['INDICE'].value_counts().head(5).index
    data_top = df_data[df_data['INDICE'].isin(indices_top)]
    sns.boxplot(data=data_top, x='INDICE', y='IND_VAR', ax=ax3)
    ax3.set_title('Distribución de Variación del Índice (IND_VAR)', fontweight='bold')
    ax3.set_xlabel('Índice')
    ax3.set_ylabel('Variación (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Correlación entre IND_ACT y NUM_ACC_COM
    ax4 = axes[1, 1]
    for indice in df_data['INDICE'].unique()[:3]:
        data_indice = df_data[df_data['INDICE'] == indice]
        ax4.scatter(data_indice['IND_ACT'], data_indice['NUM_ACC_COM'], 
                   alpha=0.6, s=50, label=indice)
    
    ax4.set_title('Relación entre Valor del Índice y Número de Acciones', fontweight='bold')
    ax4.set_xlabel('Valor del Índice (IND_ACT)')
    ax4.set_ylabel('Número de Acciones (NUM_ACC_COM)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('evolucion_indices_financieros.png', dpi=300, bbox_inches='tight')
    plt.show()

# Crear gráfico general con todos los datos
print("\nGenerando gráficos...")
crear_graficos(df_agrupado)

# Crear gráficos específicos para algunos índices principales
indices_principales = df_agrupado['INDICE'].value_counts().head(3).index

for indice in indices_principales:
    print(f"\nGenerando gráfico específico para {indice}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Evolución detallada: {indice}', fontsize=14, fontweight='bold')
    
    data_indice = df_agrupado[df_agrupado['INDICE'] == indice]
    
    # Gráfico de IND_ACT
    ax1 = axes[0]
    for dividendo in data_indice['DIVIDENDO'].unique():
        data_div = data_indice[data_indice['DIVIDENDO'] == dividendo]
        if len(data_div) > 1:
            ax1.plot(data_div['FECHA'], data_div['IND_ACT'], 
                    marker='o', linewidth=2, markersize=4, alpha=0.8, 
                    label=f'Dividendo: {dividendo}')
    
    ax1.set_title(f'Evolución IND_ACT - {indice}')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor del Índice')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico de NUM_ACC_COM
    ax2 = axes[1]
    for dividendo in data_indice['DIVIDENDO'].unique():
        data_div = data_indice[data_indice['DIVIDENDO'] == dividendo]
        if len(data_div) > 1:
            ax2.plot(data_div['FECHA'], data_div['NUM_ACC_COM'], 
                    marker='s', linewidth=2, markersize=4, alpha=0.8,
                    label=f'Dividendo: {dividendo}')
    
    ax2.set_title(f'Evolución NUM_ACC_COM - {indice}')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Número de Acciones')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'evolucion_{indice.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Crear tabla resumen
print("\nGenerando tabla resumen...")
resumen = df_agrupado.groupby(['INDICE', 'DIVIDENDO']).agg({
    'IND_ACT': ['mean', 'std', 'min', 'max'],
    'NUM_ACC_COM': ['mean', 'std', 'min', 'max'],
    'IND_VAR': ['mean', 'std'],
    'FECHA': ['min', 'max', 'count']
}).round(4)

# Aplanar nombres de columnas
resumen.columns = ['_'.join(col).strip() for col in resumen.columns.values]

# Guardar tabla resumen en Excel
resumen_filename = 'resumen_indices_financieros.xlsx'
with pd.ExcelWriter(resumen_filename, engine='openpyxl') as writer:
    resumen.to_excel(writer, sheet_name='Resumen Estadístico')
    df_agrupado.to_excel(writer, sheet_name='Datos Completos', index=False)

print(f"\nArchivos generados:")
print(f"- evolucion_indices_financieros.png (gráfico general)")
print(f"- evolucion_[INDICE].png (gráficos por índice)")
print(f"- {resumen_filename} (tabla resumen en Excel)")

print(f"\nResumen final:")
print(f"- Total registros procesados: {len(df_agrupado)}")
print(f"- Número de índices únicos: {df_agrupado['INDICE'].nunique()}")
print(f"- Rango de fechas: {df_agrupado['FECHA'].min().strftime('%Y-%m-%d')} a {df_agrupado['FECHA'].max().strftime('%Y-%m-%d')}")
print(f"- Valores de dividendo: {df_agrupado['DIVIDENDO'].unique()}")
