"""
Módulo para análisis de calidad de datos.

Este módulo contiene funciones para:
- Detectar valores faltantes
- Identificar valores duplicados
- Detectar valores atípicos
- Analizar consistencia de datos
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class DataQualityAnalyzer:
    """Clase para analizar la calidad de los datos."""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el analizador de calidad.

        Args:
            df: DataFrame a analizar
        """
        self.df = df
        self.quality_report = {}

    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analiza valores faltantes en el dataset.

        Returns:
            DataFrame con estadísticas de valores faltantes
        """
        missing_stats = pd.DataFrame({
            'columna': self.df.columns,
            'valores_faltantes': self.df.isnull().sum().values,
            'porcentaje': (self.df.isnull().sum() / len(self.df) * 100).values,
            'tipo_dato': self.df.dtypes.values
        })

        missing_stats = missing_stats[missing_stats['valores_faltantes'] > 0].sort_values(
            'porcentaje', ascending=False
        )

        self.quality_report['missing_values'] = missing_stats
        return missing_stats

    def analyze_duplicates(self) -> Dict:
        """
        Analiza registros duplicados.

        Returns:
            Diccionario con información sobre duplicados
        """
        total_duplicates = self.df.duplicated().sum()
        duplicate_percentage = (total_duplicates / len(self.df)) * 100

        # Duplicados completos
        duplicate_rows = self.df[self.df.duplicated(keep=False)]

        duplicates_info = {
            'total_duplicados': int(total_duplicates),
            'porcentaje_duplicados': float(duplicate_percentage),
            'filas_duplicadas': duplicate_rows
        }

        self.quality_report['duplicates'] = duplicates_info
        return duplicates_info

    def analyze_data_types(self) -> pd.DataFrame:
        """
        Analiza tipos de datos y posibles inconsistencias.

        Returns:
            DataFrame con análisis de tipos de datos
        """
        type_analysis = pd.DataFrame({
            'columna': self.df.columns,
            'tipo_actual': self.df.dtypes.values,
            'valores_unicos': [self.df[col].nunique() for col in self.df.columns],
            'ejemplo_valor': [self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else None
                             for col in self.df.columns]
        })

        self.quality_report['data_types'] = type_analysis
        return type_analysis

    def analyze_outliers_iqr(self, columns: List[str] = None) -> Dict:
        """
        Detecta valores atípicos usando el método IQR.

        Args:
            columns: Lista de columnas numéricas a analizar. Si es None, analiza todas.

        Returns:
            Diccionario con información de outliers por columna
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        outliers_info = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]

            outliers_info[col] = {
                'total_outliers': len(outliers),
                'porcentaje': (len(outliers) / len(self.df)) * 100,
                'limite_inferior': float(lower_bound),
                'limite_superior': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR)
            }

        self.quality_report['outliers'] = outliers_info
        return outliers_info

    def analyze_cardinality(self) -> pd.DataFrame:
        """
        Analiza la cardinalidad de las variables categóricas.

        Returns:
            DataFrame con análisis de cardinalidad
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        cardinality_data = []
        for col in categorical_cols:
            total_unique = self.df[col].nunique()
            total_values = len(self.df[col])
            cardinality_ratio = total_unique / total_values

            cardinality_data.append({
                'columna': col,
                'valores_unicos': total_unique,
                'total_valores': total_values,
                'ratio_cardinalidad': cardinality_ratio,
                'top_5_valores': self.df[col].value_counts().head(5).to_dict()
            })

        cardinality_df = pd.DataFrame(cardinality_data)
        self.quality_report['cardinality'] = cardinality_df
        return cardinality_df

    def generate_quality_summary(self) -> Dict:
        """
        Genera un resumen completo de calidad de datos.

        Returns:
            Diccionario con resumen de calidad
        """
        summary = {
            'total_registros': len(self.df),
            'total_columnas': len(self.df.columns),
            'valores_faltantes_totales': int(self.df.isnull().sum().sum()),
            'porcentaje_completitud': float((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100),
            'registros_duplicados': int(self.df.duplicated().sum()),
            'columnas_numericas': len(self.df.select_dtypes(include=[np.number]).columns),
            'columnas_categoricas': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'columnas_fecha': len(self.df.select_dtypes(include=['datetime64']).columns)
        }

        self.quality_report['summary'] = summary
        return summary

    def print_quality_report(self):
        """Imprime un reporte completo de calidad de datos."""
        print("\n" + "="*80)
        print("REPORTE DE CALIDAD DE DATOS")
        print("="*80)

        # Resumen general
        summary = self.generate_quality_summary()
        print(f"\nTotal de registros: {summary['total_registros']:,}")
        print(f"Total de columnas: {summary['total_columnas']}")
        print(f"Completitud de datos: {summary['porcentaje_completitud']:.2f}%")
        print(f"Registros duplicados: {summary['registros_duplicados']:,}")

        # Valores faltantes
        print("\n" + "-"*80)
        print("VALORES FALTANTES")
        print("-"*80)
        missing_values = self.analyze_missing_values()
        if len(missing_values) > 0:
            print(f"\nColumnas con valores faltantes: {len(missing_values)}")
            print(missing_values.to_string(index=False))
        else:
            print("✓ No se encontraron valores faltantes")

        # Duplicados
        print("\n" + "-"*80)
        print("REGISTROS DUPLICADOS")
        print("-"*80)
        duplicates = self.analyze_duplicates()
        print(f"Total de duplicados: {duplicates['total_duplicados']:,}")
        print(f"Porcentaje de duplicados: {duplicates['porcentaje_duplicados']:.2f}%")

        # Tipos de datos
        print("\n" + "-"*80)
        print("ANÁLISIS DE TIPOS DE DATOS")
        print("-"*80)
        type_analysis = self.analyze_data_types()
        print(type_analysis.to_string(index=False))

        # Cardinalidad
        print("\n" + "-"*80)
        print("CARDINALIDAD DE VARIABLES CATEGÓRICAS")
        print("-"*80)
        if len(self.df.select_dtypes(include=['object', 'category']).columns) > 0:
            cardinality = self.analyze_cardinality()
            for _, row in cardinality.iterrows():
                print(f"\n{row['columna']}:")
                print(f"  Valores únicos: {row['valores_unicos']:,}")
                print(f"  Ratio: {row['ratio_cardinalidad']:.4f}")
                print(f"  Top 5 valores más frecuentes:")
                for value, count in list(row['top_5_valores'].items())[:5]:
                    print(f"    - {value}: {count:,}")
        else:
            print("No hay variables categóricas en el dataset")

        # Outliers
        print("\n" + "-"*80)
        print("DETECCIÓN DE VALORES ATÍPICOS (IQR)")
        print("-"*80)
        if len(self.df.select_dtypes(include=[np.number]).columns) > 0:
            outliers = self.analyze_outliers_iqr()
            for col, info in outliers.items():
                if info['total_outliers'] > 0:
                    print(f"\n{col}:")
                    print(f"  Outliers detectados: {info['total_outliers']:,} ({info['porcentaje']:.2f}%)")
                    print(f"  Rango normal: [{info['limite_inferior']:.2f}, {info['limite_superior']:.2f}]")
        else:
            print("No hay variables numéricas para analizar outliers")


def main():
    """Función principal para ejecutar el análisis de calidad."""
    from pathlib import Path
    from data_loader import DataLoader

    # Cargar datos
    data_path = Path(__file__).parent.parent / "data" / "siniestros_viales_consolidados_bogota.xlsx"
    loader = DataLoader(data_path)
    df = loader.load_data()

    # Analizar calidad
    analyzer = DataQualityAnalyzer(df)
    analyzer.print_quality_report()

    return analyzer


if __name__ == "__main__":
    analyzer = main()
