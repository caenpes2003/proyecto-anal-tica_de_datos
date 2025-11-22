"""
Script para generar visualizaciones del preprocesamiento para el informe.

Genera gráficos que documentan el proceso de preprocesamiento:
- Comparación antes/después
- Distribución de features creadas
- Análisis de valores faltantes
- Estadísticas clave
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Crear carpeta para visualizaciones
output_dir = Path('data/processed/visualizaciones')
output_dir.mkdir(parents=True, exist_ok=True)


def crear_comparacion_shape():
    """Gráfico 1: Comparación de dimensiones antes/después."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Datos
    categorias = ['Registros', 'Columnas']
    original = [196162, 10]
    final = [196152, 40]

    x = np.arange(len(categorias))
    width = 0.35

    # Gráfico de barras
    bars1 = ax1.bar(x - width/2, original, width, label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, final, width, label='Final', color='#2ecc71', alpha=0.8)

    ax1.set_xlabel('Dimensión')
    ax1.set_ylabel('Cantidad')
    ax1.set_title('Comparación: Dataset Original vs Final', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categorias)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9)

    # Porcentajes de cambio
    cambios = ['Registros\nRetenidos', 'Features\nCreadas']
    valores = [99.99, 300]  # 99.99% registros, 300% incremento en features
    colores = ['#2ecc71', '#3498db']

    bars = ax2.barh(cambios, valores, color=colores, alpha=0.8)
    ax2.set_xlabel('Porcentaje (%)')
    ax2.set_title('Métricas de Transformación', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)

    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, valores)):
        ax2.text(val, i, f'  {val:.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '01_comparacion_dimensiones.png', bbox_inches='tight')
    print("OK Grafico 1 guardado: 01_comparacion_dimensiones.png")
    plt.close()


def crear_pipeline_flujo():
    """Gráfico 2: Flujo del pipeline con estadísticas."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Datos de cada paso
    pasos = [
        {'nombre': 'Dataset Original', 'registros': 196162, 'columnas': 10, 'color': '#95a5a6'},
        {'nombre': '5.1. Limpieza', 'registros': 196152, 'columnas': 11, 'color': '#e74c3c'},
        {'nombre': '5.2. Feature Engineering', 'registros': 196152, 'columnas': 34, 'color': '#f39c12'},
        {'nombre': '5.4. Transformación', 'registros': 196152, 'columnas': 46, 'color': '#9b59b6'},
        {'nombre': '5.3. Reducción Dim.', 'registros': 196152, 'columnas': 40, 'color': '#2ecc71'}
    ]

    # Posiciones
    y_pos = np.arange(len(pasos))

    # Gráfico de columnas
    bars = ax.barh(y_pos, [p['columnas'] for p in pasos],
                   color=[p['color'] for p in pasos], alpha=0.7, height=0.6)

    # Etiquetas
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p['nombre'] for p in pasos], fontsize=11)
    ax.set_xlabel('Número de Columnas', fontsize=11, fontweight='bold')
    ax.set_title('Pipeline de Preprocesamiento: Evolución de Features',
                fontweight='bold', fontsize=13, pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Añadir valores y estadísticas
    for i, (bar, paso) in enumerate(zip(bars, pasos)):
        # Número de columnas
        ax.text(paso['columnas'] + 1, i,
               f"{paso['columnas']} cols\n{paso['registros']:,} reg",
               va='center', fontsize=9, fontweight='bold')

        # Flecha al siguiente paso (excepto el último)
        if i < len(pasos) - 1:
            ax.annotate('', xy=(0, i - 0.5), xytext=(0, i + 0.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / '02_flujo_pipeline.png', bbox_inches='tight')
    print("OK Grafico 2 guardado: 02_flujo_pipeline.png")
    plt.close()


def crear_valores_faltantes():
    """Gráfico 3: Tratamiento de valores faltantes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Datos de valores faltantes ANTES
    columnas_antes = ['OBJETO_FIJO', 'CHOQUE', 'FECHA', 'HORA', 'CODIGO_ACC', 'Otros']
    valores_antes = [189473, 28252, 10, 10, 10, 0]
    porcentajes_antes = [96.59, 14.40, 0.01, 0.01, 0.01, 0]

    # Gráfico ANTES
    bars1 = ax1.barh(columnas_antes, valores_antes, color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Valores Faltantes', fontweight='bold')
    ax1.set_title('ANTES: Valores Faltantes por Columna', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)

    # Añadir porcentajes
    for bar, val, pct in zip(bars1, valores_antes, porcentajes_antes):
        if val > 0:
            ax1.text(val, bar.get_y() + bar.get_height()/2,
                    f'  {val:,} ({pct:.1f}%)',
                    va='center', fontsize=9)

    # Gráfico DESPUÉS (círculo verde = 100% completo)
    ax2.text(0.5, 0.6, 'OK', fontsize=80, ha='center', va='center',
            color='#2ecc71', fontweight='bold')
    ax2.text(0.5, 0.25, '0 Valores Faltantes', fontsize=16, ha='center',
            va='center', fontweight='bold', color='#2ecc71')
    ax2.text(0.5, 0.15, '100% Completitud', fontsize=14, ha='center',
            va='center', color='#27ae60')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('DESPUES: Dataset Completo', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / '03_valores_faltantes.png', bbox_inches='tight')
    print("OK Grafico 3 guardado: 03_valores_faltantes.png")
    plt.close()


def crear_features_creadas():
    """Gráfico 4: Features creadas por categoría."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Datos
    categorias = ['Temporales', 'Geograficas', 'Severidad', 'Interaccion', 'Agregadas', 'Encoding']
    cantidad = [12, 3, 3, 3, 3, 6]
    colores = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

    bars = ax.bar(categorias, cantidad, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Numero de Features', fontweight='bold', fontsize=11)
    ax.set_title('Features Creadas por Categoria (5.2 Feature Engineering)',
                fontweight='bold', fontsize=13, pad=15)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # Añadir valores en las barras
    for bar, val in zip(bars, cantidad):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'{val}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Total
    total = sum(cantidad)
    ax.text(0.95, 0.95, f'Total: {total} features',
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()
    plt.savefig(output_dir / '04_features_por_categoria.png', bbox_inches='tight')
    print("OK Grafico 4 guardado: 04_features_por_categoria.png")
    plt.close()


def crear_distribucion_tipos():
    """Gráfico 5: Distribución de tipos de datos."""
    # Leer diccionario de features
    dict_path = Path('data/processed/feature_dictionary.csv')
    if not dict_path.exists():
        print("ADVERTENCIA: No se encontro feature_dictionary.csv, saltando grafico 5")
        return

    df_dict = pd.read_csv(dict_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Por tipo de dato
    tipo_counts = df_dict['tipo'].value_counts()
    colors = plt.cm.Set3(range(len(tipo_counts)))

    ax1.pie(tipo_counts.values, labels=tipo_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Distribucion por Tipo de Dato', fontweight='bold', fontsize=12)

    # Gráfico 2: Por origen
    origen_counts = df_dict['origen'].value_counts()
    colors2 = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax2.bar(origen_counts.index, origen_counts.values, color=colors2, alpha=0.8)
    ax2.set_ylabel('Numero de Features', fontweight='bold')
    ax2.set_title('Features por Origen', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Añadir valores
    for bar, val in zip(bars, origen_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.5,
               f'{val}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '05_distribucion_tipos.png', bbox_inches='tight')
    print("OK Grafico 5 guardado: 05_distribucion_tipos.png")
    plt.close()


def crear_resumen_tiempos():
    """Gráfico 6: Tiempo de ejecución por paso."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Datos (de tu ejecución)
    pasos = ['Limpieza', 'Feature\nEngineering', 'Transformacion', 'Reduccion\nDimensionalidad']
    tiempos = [0.48, 1.99, 0.90, 0.41]
    colores = ['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']

    bars = ax.bar(pasos, tiempos, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Tiempo (segundos)', fontweight='bold', fontsize=11)
    ax.set_title('Tiempo de Ejecucion por Paso del Pipeline',
                fontweight='bold', fontsize=13, pad=15)
    ax.grid(axis='y', alpha=0.3)

    # Añadir valores
    for bar, val in zip(bars, tiempos):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{val:.2f}s',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Tiempo total
    total = sum(tiempos)
    ax.text(0.95, 0.95, f'Tiempo Total: {total:.2f}s\n(~196k registros)',
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
           verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()
    plt.savefig(output_dir / '06_tiempos_ejecucion.png', bbox_inches='tight')
    print("OK Grafico 6 guardado: 06_tiempos_ejecucion.png")
    plt.close()


def crear_resumen_metricas():
    """Gráfico 7: Resumen de métricas clave."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Título
    fig.suptitle('Resumen de Metricas del Preprocesamiento',
                fontweight='bold', fontsize=16, y=0.95)

    # Crear tabla de métricas
    metricas = [
        ['METRICA', 'VALOR'],
        ['', ''],
        ['Registros procesados', '196,152'],
        ['Registros eliminados', '10 (0.01%)'],
        ['Porcentaje retenido', '99.99%'],
        ['', ''],
        ['Columnas originales', '10'],
        ['Columnas finales', '40'],
        ['Features creadas', '30 (+300%)'],
        ['', ''],
        ['Valores faltantes (antes)', '217,750 (11.10%)'],
        ['Valores faltantes (despues)', '0 (100% completo)'],
        ['', ''],
        ['Duplicados eliminados', '5'],
        ['Outliers corregidos', 'Validados'],
        ['Features correlacionadas eliminadas', '6'],
        ['', ''],
        ['Tiempo total de ejecucion', '10.28 segundos'],
        ['Memoria del dataset final', '51.07 MB'],
    ]

    # Colores alternados
    colors = []
    for i, row in enumerate(metricas):
        if row[0] == '':
            colors.append(['white', 'white'])
        elif row[0] == 'METRICA':
            colors.append(['#3498db', '#3498db'])
        else:
            colors.append(['#ecf0f1', '#ecf0f1'])

    table = ax.table(cellText=metricas, cellLoc='left',
                    loc='center', cellColours=colors,
                    colWidths=[0.6, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Estilo de la tabla
    for i, row in enumerate(metricas):
        if row[0] == 'METRICA':
            for j in range(2):
                cell = table[(i, j)]
                cell.set_text_props(weight='bold', color='white', fontsize=12)
                cell.set_facecolor('#3498db')
        elif row[0] != '':
            cell = table[(i, 0)]
            cell.set_text_props(weight='bold')
            cell = table[(i, 1)]
            cell.set_text_props(color='#2c3e50', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / '07_resumen_metricas.png', bbox_inches='tight')
    print("OK Grafico 7 guardado: 07_resumen_metricas.png")
    plt.close()


def main():
    """Genera todas las visualizaciones."""
    print("="*70)
    print("GENERANDO VISUALIZACIONES PARA EL INFORME")
    print("="*70)
    print(f"\nGuardando en: {output_dir}")
    print()

    crear_comparacion_shape()
    crear_pipeline_flujo()
    crear_valores_faltantes()
    crear_features_creadas()
    crear_distribucion_tipos()
    crear_resumen_tiempos()
    crear_resumen_metricas()

    print()
    print("="*70)
    print("OK TODAS LAS VISUALIZACIONES GENERADAS EXITOSAMENTE")
    print("="*70)
    print(f"\nUbicacion: {output_dir.absolute()}")
    print("\nArchivos generados:")
    for i in range(1, 8):
        print(f"  {i}. 0{i}_*.png")


if __name__ == "__main__":
    main()
