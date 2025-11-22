"""
Módulo para la carga y exploración inicial de datos de siniestros viales.

Este módulo contiene funciones para:
- Cargar datos desde archivos Excel
- Realizar análisis exploratorio inicial
- Identificar problemas de calidad de datos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """Clase para gestionar la carga y exploración inicial de datos."""

    def __init__(self, data_path: str):
        """
        Inicializa el cargador de datos.

        Args:
            data_path: Ruta al archivo de datos
        """
        self.data_path = Path(data_path)
        self.df = None
        self.metadata = {}

    def load_data(self, sheet_name: int = 0) -> pd.DataFrame:
        """
        Carga datos desde archivo Excel.

        Args:
            sheet_name: Nombre o índice de la hoja a cargar

        Returns:
            DataFrame con los datos cargados
        """
        print(f"Cargando datos desde: {self.data_path}")

        try:
            self.df = pd.read_excel(self.data_path, sheet_name=sheet_name)
            print(f"✓ Datos cargados exitosamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
            return self.df
        except Exception as e:
            print(f"✗ Error al cargar datos: {e}")
            raise

    def get_basic_info(self) -> Dict:
        """
        Obtiene información básica del dataset.

        Returns:
            Diccionario con información del dataset
        """
        if self.df is None:
            raise ValueError("Primero debe cargar los datos usando load_data()")

        info = {
            'total_registros': len(self.df),
            'total_columnas': len(self.df.columns),
            'columnas': list(self.df.columns),
            'tipos_datos': self.df.dtypes.to_dict(),
            'memoria_uso_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'rango_fechas': self._get_date_range() if self._has_date_column() else None
        }

        self.metadata = info
        return info

    def _has_date_column(self) -> bool:
        """Verifica si existe alguna columna de fecha."""
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        return len(date_columns) > 0

    def _get_date_range(self) -> Dict:
        """Obtiene el rango de fechas del dataset."""
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            date_col = date_columns[0]
            return {
                'columna': date_col,
                'fecha_minima': self.df[date_col].min(),
                'fecha_maxima': self.df[date_col].max()
            }
        return None

    def describe_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Genera descripción estadística de los datos.

        Returns:
            Tupla con (descripción_numérica, descripción_categórica)
        """
        if self.df is None:
            raise ValueError("Primero debe cargar los datos usando load_data()")

        # Descripción de variables numéricas
        numeric_desc = self.df.describe(include=[np.number])

        # Descripción de variables categóricas
        categorical_desc = self.df.describe(include=['object', 'category'])

        return numeric_desc, categorical_desc

    def print_basic_info(self):
        """Imprime información básica del dataset de forma legible."""
        info = self.get_basic_info()

        print("\n" + "="*60)
        print("INFORMACIÓN BÁSICA DEL DATASET")
        print("="*60)
        print(f"Total de registros: {info['total_registros']:,}")
        print(f"Total de columnas: {info['total_columnas']}")
        print(f"Memoria utilizada: {info['memoria_uso_mb']:.2f} MB")

        if info['rango_fechas']:
            print(f"\nRango de fechas ({info['rango_fechas']['columna']}):")
            print(f"  Desde: {info['rango_fechas']['fecha_minima']}")
            print(f"  Hasta: {info['rango_fechas']['fecha_maxima']}")

        print("\nColumnas del dataset:")
        for i, col in enumerate(info['columnas'], 1):
            dtype = info['tipos_datos'][col]
            print(f"  {i:2d}. {col} ({dtype})")


def main():
    """Función principal para ejecutar la carga de datos."""
    # Ruta al archivo de datos
    data_path = Path(__file__).parent.parent / "data" / "siniestros_viales_consolidados_bogota.xlsx"

    # Crear instancia del cargador
    loader = DataLoader(data_path)

    # Cargar datos
    df = loader.load_data()

    # Mostrar información básica
    loader.print_basic_info()

    # Descripción estadística
    print("\n" + "="*60)
    print("DESCRIPCIÓN ESTADÍSTICA - VARIABLES NUMÉRICAS")
    print("="*60)
    numeric_desc, categorical_desc = loader.describe_data()
    print(numeric_desc)

    if not categorical_desc.empty:
        print("\n" + "="*60)
        print("DESCRIPCIÓN ESTADÍSTICA - VARIABLES CATEGÓRICAS")
        print("="*60)
        print(categorical_desc)

    return df, loader


if __name__ == "__main__":
    df, loader = main()
