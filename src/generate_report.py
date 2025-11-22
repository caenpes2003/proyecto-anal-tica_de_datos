"""
Script para generar un reporte completo de calidad de datos.

Este script ejecuta todos los análisis y genera un reporte en texto
con las observaciones iniciales del dataset.
"""

from pathlib import Path
from datetime import datetime
from data_loader import DataLoader
from data_quality import DataQualityAnalyzer
import pandas as pd


def generate_full_report(output_path: str = None):
    """
    Genera un reporte completo de análisis inicial de datos.

    Args:
        output_path: Ruta donde guardar el reporte. Si es None, usa ubicación por defecto.
    """
    # Configurar rutas
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "reporte_inicial.txt"
    else:
        output_path = Path(output_path)

    # Cargar datos
    print("Cargando datos...")
    data_path = Path(__file__).parent.parent / "data" / "siniestros_viales_consolidados_bogota.xlsx"
    loader = DataLoader(data_path)
    df = loader.load_data()

    # Inicializar analizador
    analyzer = DataQualityAnalyzer(df)

    # Abrir archivo para escritura
    with open(output_path, 'w', encoding='utf-8') as f:
        # Encabezado del reporte
        f.write("="*100 + "\n")
        f.write("REPORTE DE ANÁLISIS INICIAL DE DATOS\n")
        f.write("Proyecto: Análisis de Siniestros Viales - Bogotá\n")
        f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")

        # 4.1. FUENTES DE DATOS Y MÉTODOS DE RECOLECCIÓN
        f.write("4.1. FUENTES DE DATOS Y MÉTODOS DE RECOLECCIÓN\n")
        f.write("-"*100 + "\n\n")

        f.write("Fuente de datos:\n")
        f.write(f"  - Nombre del archivo: {data_path.name}\n")
        f.write(f"  - Ubicación: {data_path}\n")
        f.write(f"  - Tamaño del archivo: {data_path.stat().st_size / 1024**2:.2f} MB\n")
        f.write(f"  - Formato: Excel (.xlsx)\n\n")

        f.write("Método de recolección:\n")
        f.write("  - Dataset de siniestros viales consolidados de Bogotá\n")
        f.write("  - Fuente oficial de registros de eventos de tránsito\n")
        f.write("  - Recopilación sistemática de información de siniestros viales\n\n")

        # 4.2. DESCRIPCIÓN DE LOS CONJUNTOS DE DATOS CRUDOS
        f.write("\n" + "="*100 + "\n")
        f.write("4.2. DESCRIPCIÓN DE LOS CONJUNTOS DE DATOS CRUDOS\n")
        f.write("-"*100 + "\n\n")

        # Información básica
        info = loader.get_basic_info()
        f.write("Información general del dataset:\n")
        f.write(f"  - Total de registros: {info['total_registros']:,}\n")
        f.write(f"  - Total de columnas: {info['total_columnas']}\n")
        f.write(f"  - Memoria utilizada: {info['memoria_uso_mb']:.2f} MB\n\n")

        if info['rango_fechas']:
            f.write(f"Rango temporal de los datos:\n")
            f.write(f"  - Campo de fecha: {info['rango_fechas']['columna']}\n")
            f.write(f"  - Fecha inicial: {info['rango_fechas']['fecha_minima']}\n")
            f.write(f"  - Fecha final: {info['rango_fechas']['fecha_maxima']}\n\n")

        # Estructura del dataset
        f.write("Columnas del dataset:\n")
        for i, col in enumerate(info['columnas'], 1):
            dtype = info['tipos_datos'][col]
            f.write(f"  {i:2d}. {col} ({dtype})\n")
        f.write("\n")

        # Estadísticas descriptivas
        numeric_desc, categorical_desc = loader.describe_data()

        if not numeric_desc.empty:
            f.write("Estadísticas descriptivas - Variables numéricas:\n")
            f.write(numeric_desc.to_string())
            f.write("\n\n")

        if not categorical_desc.empty:
            f.write("Estadísticas descriptivas - Variables categóricas:\n")
            f.write(categorical_desc.to_string())
            f.write("\n\n")

        # 4.3. PROBLEMAS DE CALIDAD DE LOS DATOS Y OBSERVACIONES INICIALES
        f.write("\n" + "="*100 + "\n")
        f.write("4.3. PROBLEMAS DE CALIDAD DE LOS DATOS Y OBSERVACIONES INICIALES\n")
        f.write("-"*100 + "\n\n")

        # Resumen de calidad
        summary = analyzer.generate_quality_summary()
        f.write("Resumen de calidad de datos:\n")
        f.write(f"  - Completitud general: {summary['porcentaje_completitud']:.2f}%\n")
        f.write(f"  - Total de valores faltantes: {summary['valores_faltantes_totales']:,}\n")
        f.write(f"  - Registros duplicados: {summary['registros_duplicados']:,}\n")
        f.write(f"  - Porcentaje de duplicados: {(summary['registros_duplicados']/summary['total_registros']*100):.2f}%\n\n")

        # Valores faltantes
        f.write("Análisis de valores faltantes:\n")
        missing_values = analyzer.analyze_missing_values()
        if len(missing_values) > 0:
            f.write(f"  Se encontraron {len(missing_values)} columnas con valores faltantes:\n\n")
            for _, row in missing_values.iterrows():
                f.write(f"  - {row['columna']}:\n")
                f.write(f"      Valores faltantes: {row['valores_faltantes']:,}\n")
                f.write(f"      Porcentaje: {row['porcentaje']:.2f}%\n")
                f.write(f"      Tipo de dato: {row['tipo_dato']}\n\n")
        else:
            f.write("  ✓ No se encontraron valores faltantes en el dataset\n\n")

        # Duplicados
        f.write("Análisis de registros duplicados:\n")
        duplicates = analyzer.analyze_duplicates()
        f.write(f"  - Total de duplicados: {duplicates['total_duplicados']:,}\n")
        f.write(f"  - Porcentaje de duplicados: {duplicates['porcentaje_duplicados']:.2f}%\n")
        if duplicates['total_duplicados'] > 0:
            f.write(f"  ⚠ Se recomienda revisar y eliminar duplicados\n")
        f.write("\n")

        # Cardinalidad
        f.write("Análisis de cardinalidad de variables categóricas:\n")
        cardinality = analyzer.analyze_cardinality()
        if len(cardinality) > 0:
            for _, row in cardinality.iterrows():
                f.write(f"\n  - {row['columna']}:\n")
                f.write(f"      Valores únicos: {row['valores_unicos']:,}\n")
                f.write(f"      Ratio de cardinalidad: {row['ratio_cardinalidad']:.4f}\n")
                f.write(f"      Top 5 valores más frecuentes:\n")
                for value, count in list(row['top_5_valores'].items())[:5]:
                    f.write(f"        • {value}: {count:,}\n")
        else:
            f.write("  No hay variables categóricas en el dataset\n")
        f.write("\n")

        # Outliers
        f.write("Detección de valores atípicos (método IQR):\n")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numeric_cols) > 0:
            outliers = analyzer.analyze_outliers_iqr()
            outliers_found = False
            for col, info in outliers.items():
                if info['total_outliers'] > 0:
                    outliers_found = True
                    f.write(f"\n  - {col}:\n")
                    f.write(f"      Outliers detectados: {info['total_outliers']:,} ({info['porcentaje']:.2f}%)\n")
                    f.write(f"      Rango normal (Q1-1.5*IQR, Q3+1.5*IQR): [{info['limite_inferior']:.2f}, {info['limite_superior']:.2f}]\n")
                    f.write(f"      Q1: {info['Q1']:.2f} | Q3: {info['Q3']:.2f} | IQR: {info['IQR']:.2f}\n")

            if not outliers_found:
                f.write("  ✓ No se detectaron valores atípicos significativos\n")
        else:
            f.write("  No hay variables numéricas para analizar\n")
        f.write("\n")

        # Observaciones y recomendaciones
        f.write("\n" + "="*100 + "\n")
        f.write("OBSERVACIONES Y RECOMENDACIONES\n")
        f.write("-"*100 + "\n\n")

        f.write("Problemas identificados:\n")
        if len(missing_values) > 0:
            f.write(f"  1. Se encontraron valores faltantes en {len(missing_values)} columnas\n")
        if duplicates['total_duplicados'] > 0:
            f.write(f"  2. Existen {duplicates['total_duplicados']:,} registros duplicados\n")
        if len(missing_values) == 0 and duplicates['total_duplicados'] == 0:
            f.write("  ✓ No se identificaron problemas críticos de calidad\n")
        f.write("\n")

        f.write("Próximos pasos recomendados:\n")
        f.write("  1. Limpiar y procesar valores faltantes según estrategia definida\n")
        f.write("  2. Gestionar registros duplicados\n")
        f.write("  3. Validar y convertir tipos de datos si es necesario\n")
        f.write("  4. Realizar análisis exploratorio detallado por variable\n")
        f.write("  5. Generar visualizaciones de distribuciones y relaciones\n")
        f.write("  6. Documentar hallazgos y decisiones de preprocesamiento\n")
        f.write("\n")

        f.write("="*100 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*100 + "\n")

    print(f"\n✓ Reporte generado exitosamente: {output_path}")
    return output_path


def main():
    """Función principal."""
    print("Generando reporte de análisis inicial...")
    report_path = generate_full_report()
    print(f"\nPuedes consultar el reporte en: {report_path}")


if __name__ == "__main__":
    main()
