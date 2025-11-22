"""
Módulo para limpieza de datos.

Maneja valores faltantes, duplicados y outliers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clase para limpieza de datos de siniestros viales."""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el limpiador de datos.

        Args:
            df: DataFrame a limpiar
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []

    def handle_missing_values(self, strategy: str = 'smart') -> pd.DataFrame:
        """
        Maneja valores faltantes según estrategia definida.

        Args:
            strategy: 'smart' (estrategia inteligente por columna), 'drop', 'fill'

        Returns:
            DataFrame con valores faltantes procesados
        """
        logger.info("Procesando valores faltantes...")
        initial_missing = self.df.isnull().sum().sum()

        if strategy == 'smart':
            # 1. OBJETO_FIJO (96.59% faltantes) - Crear columna binaria
            self.df['INVOLUCRA_OBJETO_FIJO'] = self.df['OBJETO_FIJO'].notna().astype(int)
            self.cleaning_log.append(
                "OBJETO_FIJO: Creada columna binaria INVOLUCRA_OBJETO_FIJO (1=Sí, 0=No)"
            )

            # 2. CHOQUE (14.40% faltantes) - Imputar con moda
            if self.df['CHOQUE'].isnull().sum() > 0:
                choque_moda = self.df['CHOQUE'].mode()[0] if len(self.df['CHOQUE'].mode()) > 0 else 0
                self.df['CHOQUE'].fillna(choque_moda, inplace=True)
                self.cleaning_log.append(
                    f"CHOQUE: Imputado con moda ({choque_moda})"
                )

            # 3. Registros con FECHA, HORA o CODIGO_ACCIDENTE faltantes - Eliminar
            critical_cols = ['FECHA', 'HORA', 'CODIGO_ACCIDENTE']
            before_drop = len(self.df)
            self.df.dropna(subset=critical_cols, inplace=True)
            after_drop = len(self.df)
            if before_drop - after_drop > 0:
                self.cleaning_log.append(
                    f"Eliminados {before_drop - after_drop} registros con valores críticos faltantes"
                )

            # 4. DIRECCION faltantes - Marcar como "DESCONOCIDA"
            if self.df['DIRECCION'].isnull().sum() > 0:
                self.df['DIRECCION'].fillna('DESCONOCIDA', inplace=True)
                self.cleaning_log.append("DIRECCION: Valores faltantes marcados como 'DESCONOCIDA'")

            # 5. Otras columnas numéricas - Imputar con moda o eliminar filas
            numeric_cols = ['GRAVEDAD', 'CLASE', 'DISENO_LUGAR', 'CODIGO_LOCALIDAD']
            for col in numeric_cols:
                if col in self.df.columns and self.df[col].isnull().sum() > 0:
                    before = len(self.df)
                    self.df.dropna(subset=[col], inplace=True)
                    after = len(self.df)
                    if before - after > 0:
                        self.cleaning_log.append(
                            f"{col}: Eliminados {before - after} registros con valores faltantes"
                        )

        elif strategy == 'drop':
            self.df.dropna(inplace=True)
            self.cleaning_log.append("Eliminadas todas las filas con valores faltantes")

        final_missing = self.df.isnull().sum().sum()
        logger.info(f"Valores faltantes reducidos: {initial_missing} → {final_missing}")

        return self.df

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Elimina registros duplicados.

        Args:
            subset: Columnas a considerar para duplicados. Si None, usa todas.
            keep: 'first', 'last', o False

        Returns:
            DataFrame sin duplicados
        """
        logger.info("Eliminando duplicados...")
        initial_count = len(self.df)

        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)

        final_count = len(self.df)
        removed = initial_count - final_count

        if removed > 0:
            self.cleaning_log.append(f"Eliminados {removed} registros duplicados")
            logger.info(f"Duplicados eliminados: {removed}")
        else:
            logger.info("No se encontraron duplicados")

        return self.df

    def handle_outliers(self, columns: Optional[List[str]] = None,
                       method: str = 'validate', action: str = 'keep') -> pd.DataFrame:
        """
        Maneja valores atípicos.

        Args:
            columns: Lista de columnas a analizar. Si None, usa todas las numéricas.
            method: 'iqr', 'zscore', 'validate' (validar si son valores legítimos)
            action: 'keep', 'remove', 'cap' (limitar a umbrales)

        Returns:
            DataFrame procesado
        """
        logger.info("Procesando valores atípicos...")

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Excluir ID y columnas binarias
            columns = [col for col in columns if col not in
                      ['CODIGO_ACCIDENTE', 'INVOLUCRA_OBJETO_FIJO']]

        if method == 'validate':
            # CODIGO_LOCALIDAD - Bogotá tiene 20 localidades (códigos 1-20)
            if 'CODIGO_LOCALIDAD' in self.df.columns:
                invalid_localities = self.df[
                    (self.df['CODIGO_LOCALIDAD'] < 1) |
                    (self.df['CODIGO_LOCALIDAD'] > 20)
                ]
                if len(invalid_localities) > 0:
                    logger.warning(f"Encontrados {len(invalid_localities)} códigos de localidad inválidos")
                    if action == 'remove':
                        before = len(self.df)
                        self.df = self.df[
                            (self.df['CODIGO_LOCALIDAD'] >= 1) &
                            (self.df['CODIGO_LOCALIDAD'] <= 20)
                        ]
                        after = len(self.df)
                        self.cleaning_log.append(
                            f"CODIGO_LOCALIDAD: Eliminados {before - after} registros con códigos inválidos"
                        )

            # Para GRAVEDAD, CLASE, CHOQUE, DISENO_LUGAR - Son categorías, no outliers
            # Solo validar que no sean negativos
            categorical_numeric = ['GRAVEDAD', 'CLASE', 'CHOQUE', 'DISENO_LUGAR', 'OBJETO_FIJO']
            for col in categorical_numeric:
                if col in self.df.columns:
                    invalid = self.df[self.df[col] < 0]
                    if len(invalid) > 0:
                        logger.warning(f"Encontrados {len(invalid)} valores negativos en {col}")
                        if action == 'remove':
                            before = len(self.df)
                            self.df = self.df[self.df[col] >= 0]
                            after = len(self.df)
                            self.cleaning_log.append(
                                f"{col}: Eliminados {before - after} registros con valores negativos"
                            )

        elif method == 'iqr':
            for col in columns:
                if col not in self.df.columns:
                    continue

                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outliers_count = outliers_mask.sum()

                if outliers_count > 0:
                    if action == 'remove':
                        before = len(self.df)
                        self.df = self.df[~outliers_mask]
                        after = len(self.df)
                        self.cleaning_log.append(
                            f"{col}: Eliminados {before - after} outliers (IQR)"
                        )
                    elif action == 'cap':
                        self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                        self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                        self.cleaning_log.append(
                            f"{col}: {outliers_count} outliers limitados a umbrales IQR"
                        )

        return self.df

    def convert_data_types(self) -> pd.DataFrame:
        """
        Convierte tipos de datos a formatos apropiados.

        Returns:
            DataFrame con tipos de datos corregidos
        """
        logger.info("Convirtiendo tipos de datos...")

        # FECHA - Convertir a datetime
        if 'FECHA' in self.df.columns:
            try:
                self.df['FECHA'] = pd.to_datetime(self.df['FECHA'], format='%d/%m/%Y', errors='coerce')
                self.cleaning_log.append("FECHA: Convertida a datetime")
            except Exception as e:
                logger.warning(f"Error al convertir FECHA: {e}")

        # HORA - Mantener como string por ahora (se procesará en feature engineering)

        # Variables numéricas categóricas - Convertir a int
        categorical_numeric = ['GRAVEDAD', 'CLASE', 'CHOQUE', 'DISENO_LUGAR',
                              'CODIGO_LOCALIDAD', 'OBJETO_FIJO']
        for col in categorical_numeric:
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype('Int64')  # Int64 permite NaN
                except Exception as e:
                    logger.warning(f"Error al convertir {col}: {e}")

        return self.df

    def clean(self, remove_duplicates: bool = True,
              missing_strategy: str = 'smart',
              outlier_method: str = 'validate',
              outlier_action: str = 'remove') -> pd.DataFrame:
        """
        Pipeline completo de limpieza.

        Args:
            remove_duplicates: Si eliminar duplicados
            missing_strategy: Estrategia para valores faltantes
            outlier_method: Método para detectar outliers
            outlier_action: Acción para outliers

        Returns:
            DataFrame limpio
        """
        logger.info("="*60)
        logger.info("INICIANDO LIMPIEZA DE DATOS")
        logger.info("="*60)
        logger.info(f"Shape inicial: {self.original_shape}")

        # 1. Convertir tipos de datos
        self.convert_data_types()

        # 2. Eliminar duplicados
        if remove_duplicates:
            self.remove_duplicates()

        # 3. Manejar valores faltantes
        self.handle_missing_values(strategy=missing_strategy)

        # 4. Manejar outliers
        self.handle_outliers(method=outlier_method, action=outlier_action)

        # Resumen final
        final_shape = self.df.shape
        logger.info("="*60)
        logger.info("LIMPIEZA COMPLETADA")
        logger.info("="*60)
        logger.info(f"Shape final: {final_shape}")
        logger.info(f"Registros eliminados: {self.original_shape[0] - final_shape[0]}")
        logger.info(f"Columnas agregadas: {final_shape[1] - self.original_shape[1]}")

        return self.df

    def get_cleaning_report(self) -> Dict:
        """
        Genera reporte de limpieza.

        Returns:
            Diccionario con estadísticas de limpieza
        """
        return {
            'shape_original': self.original_shape,
            'shape_final': self.df.shape,
            'registros_eliminados': self.original_shape[0] - self.df.shape[0],
            'porcentaje_retenido': (self.df.shape[0] / self.original_shape[0]) * 100,
            'valores_faltantes_restantes': int(self.df.isnull().sum().sum()),
            'log_acciones': self.cleaning_log
        }

    def print_cleaning_report(self):
        """Imprime reporte de limpieza."""
        report = self.get_cleaning_report()

        print("\n" + "="*80)
        print("REPORTE DE LIMPIEZA DE DATOS")
        print("="*80)
        print(f"\nShape original: {report['shape_original']}")
        print(f"Shape final: {report['shape_final']}")
        print(f"Registros eliminados: {report['registros_eliminados']}")
        print(f"Porcentaje retenido: {report['porcentaje_retenido']:.2f}%")
        print(f"Valores faltantes restantes: {report['valores_faltantes_restantes']}")

        print("\nAcciones realizadas:")
        for i, action in enumerate(report['log_acciones'], 1):
            print(f"  {i}. {action}")


def main():
    """Función principal para pruebas."""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data_loader import DataLoader

    # Cargar datos
    data_path = Path(__file__).parent.parent.parent / "data" / "siniestros_viales_consolidados_bogota.xlsx"
    loader = DataLoader(data_path)
    df = loader.load_data()

    # Limpiar datos
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean()

    # Mostrar reporte
    cleaner.print_cleaning_report()

    return df_clean, cleaner


if __name__ == "__main__":
    df_clean, cleaner = main()
