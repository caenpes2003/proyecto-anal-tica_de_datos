"""
Módulo para transformación y encoding de datos.

Normalización, estandarización y encoding de variables categóricas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransformer:
    """Clase para transformación y encoding de datos."""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el transformador de datos.

        Args:
            df: DataFrame con features creadas
        """
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}
        self.transform_log = []
        self.encoded_columns = {}

    def encode_categorical_variables(self, method: str = 'auto') -> pd.DataFrame:
        """
        Codifica variables categóricas.

        Args:
            method: 'auto', 'label', 'onehot', 'frequency'

        Returns:
            DataFrame con variables codificadas
        """
        logger.info("Codificando variables categóricas...")

        # Variables categóricas de texto
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # Excluir FECHA y HORA (ya procesadas), CODIGO_ACCIDENTE
        categorical_cols = [col for col in categorical_cols if col not in
                           ['FECHA', 'HORA', 'CODIGO_ACCIDENTE', 'DIRECCION']]

        if method == 'auto':
            # Estrategia automática según cardinalidad
            for col in categorical_cols:
                unique_count = self.df[col].nunique()

                if unique_count <= 5:
                    # One-hot encoding para baja cardinalidad
                    self._onehot_encode(col)
                elif unique_count <= 20:
                    # Label encoding para cardinalidad media
                    self._label_encode(col)
                else:
                    # Frequency encoding para alta cardinalidad
                    self._frequency_encode(col)

        elif method == 'label':
            for col in categorical_cols:
                self._label_encode(col)

        elif method == 'onehot':
            for col in categorical_cols:
                self._onehot_encode(col)

        elif method == 'frequency':
            for col in categorical_cols:
                self._frequency_encode(col)

        logger.info("✓ Variables categóricas codificadas")
        return self.df

    def _label_encode(self, column: str):
        """Aplica Label Encoding a una columna."""
        le = LabelEncoder()
        self.df[f'{column}_encoded'] = le.fit_transform(self.df[column].astype(str))
        self.encoders[column] = le
        self.encoded_columns[column] = f'{column}_encoded'
        self.transform_log.append(f"{column}: Label Encoding aplicado")

    def _onehot_encode(self, column: str):
        """Aplica One-Hot Encoding a una columna."""
        dummies = pd.get_dummies(self.df[column], prefix=column, drop_first=True)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.encoded_columns[column] = dummies.columns.tolist()
        self.transform_log.append(f"{column}: One-Hot Encoding aplicado ({len(dummies.columns)} columnas)")

    def _frequency_encode(self, column: str):
        """Aplica Frequency Encoding a una columna."""
        freq_map = self.df[column].value_counts(normalize=True).to_dict()
        self.df[f'{column}_freq'] = self.df[column].map(freq_map)
        self.encoded_columns[column] = f'{column}_freq'
        self.transform_log.append(f"{column}: Frequency Encoding aplicado")

    def encode_ordinal_variables(self) -> pd.DataFrame:
        """
        Codifica variables ordinales con orden específico.

        Returns:
            DataFrame con variables ordinales codificadas
        """
        logger.info("Codificando variables ordinales...")

        # GRAVEDAD: 1=CON MUERTOS (más grave), 2=CON HERIDOS, 3=SOLO DAÑOS (menos grave)
        # Ya procesado como puntaje_gravedad en feature engineering

        # PERIODO_DIA: orden temporal
        if 'periodo_dia' in self.df.columns:
            periodo_orden = {'MADRUGADA': 0, 'MANANA': 1, 'TARDE': 2, 'NOCHE': 3, 'DESCONOCIDO': -1}
            self.df['periodo_dia_ord'] = self.df['periodo_dia'].map(periodo_orden)
            self.transform_log.append("periodo_dia: Encoding ordinal aplicado")

        # DIA_SEMANA: ya es numérico (0-6)

        # FRANJA_HORARIA: binaria
        if 'franja_horaria' in self.df.columns:
            self.df['franja_horaria_bin'] = (self.df['franja_horaria'] == 'PICO').astype(int)
            self.transform_log.append("franja_horaria: Encoding binario aplicado")

        logger.info("✓ Variables ordinales codificadas")
        return self.df

    def scale_numeric_features(self, method: str = 'standard',
                               exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Escala variables numéricas.

        Args:
            method: 'standard', 'minmax', 'robust'
            exclude_cols: Columnas a excluir del escalado

        Returns:
            DataFrame con variables escaladas
        """
        logger.info(f"Escalando variables numéricas con {method}...")

        # Seleccionar columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Excluir columnas
        default_exclude = ['CODIGO_ACCIDENTE', 'anio', 'mes', 'dia_mes', 'dia_semana',
                          'es_fin_semana', 'hora_num', 'trimestre', 'semana_anio',
                          'INVOLUCRA_OBJETO_FIJO', 'choque_multiple', 'riesgo_alto']
        exclude_cols = exclude_cols or []
        exclude_cols.extend(default_exclude)

        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        if not cols_to_scale:
            logger.warning("No hay columnas para escalar")
            return self.df

        # Seleccionar scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Método {method} no reconocido, usando standard")
            scaler = StandardScaler()

        # Escalar
        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
        self.scalers['numeric'] = scaler
        self.transform_log.append(f"Escalado {method} aplicado a {len(cols_to_scale)} columnas")

        logger.info("✓ Variables numéricas escaladas")
        return self.df

    def create_binary_features(self) -> pd.DataFrame:
        """
        Convierte variables categóricas binarias a 0/1.

        Returns:
            DataFrame con features binarias
        """
        logger.info("Creando features binarias...")

        # es_fin_semana ya está como binaria

        # ZONA_BOGOTA - Crear binarias para análisis específico
        if 'zona_bogota' in self.df.columns:
            zonas = self.df['zona_bogota'].unique()
            for zona in zonas:
                if zona != 'DESCONOCIDA':
                    self.df[f'zona_{zona.lower()}'] = (self.df['zona_bogota'] == zona).astype(int)

        logger.info("✓ Features binarias creadas")
        return self.df

    def reduce_high_cardinality(self, threshold: int = 50) -> pd.DataFrame:
        """
        Reduce cardinalidad de variables categóricas agrupando valores poco frecuentes.

        Args:
            threshold: Frecuencia mínima para mantener una categoría

        Returns:
            DataFrame con cardinalidad reducida
        """
        logger.info("Reduciendo cardinalidad...")

        # DIRECCION tiene 92,522 valores únicos - ya se extrajo tipo_via
        # Podemos eliminarla para reducir dimensionalidad
        if 'DIRECCION' in self.df.columns:
            logger.info("Eliminando DIRECCION (alta cardinalidad, ya procesada)")
            self.df.drop('DIRECCION', axis=1, inplace=True)
            self.transform_log.append("DIRECCION eliminada (reemplazada por tipo_via)")

        # Para tipo_via, agrupar categorías poco frecuentes
        if 'tipo_via' in self.df.columns:
            via_counts = self.df['tipo_via'].value_counts()
            rare_categories = via_counts[via_counts < threshold].index
            if len(rare_categories) > 0:
                self.df.loc[self.df['tipo_via'].isin(rare_categories), 'tipo_via'] = 'OTRA'
                self.transform_log.append(
                    f"tipo_via: {len(rare_categories)} categorías poco frecuentes agrupadas en 'OTRA'"
                )

        logger.info("✓ Cardinalidad reducida")
        return self.df

    def drop_unnecessary_columns(self) -> pd.DataFrame:
        """
        Elimina columnas innecesarias después de encoding.

        Returns:
            DataFrame sin columnas redundantes
        """
        logger.info("Eliminando columnas innecesarias...")

        # Columnas a eliminar después de encoding
        cols_to_drop = []

        # Si se crearon versiones encoded, eliminar originales
        for original, encoded in self.encoded_columns.items():
            if original in self.df.columns:
                cols_to_drop.append(original)

        # Eliminar HORA y FECHA originales si ya se extrajeron features
        if 'hora_num' in self.df.columns and 'HORA' in self.df.columns:
            cols_to_drop.append('HORA')

        # No eliminar FECHA aún, puede ser útil para análisis temporal
        # cols_to_drop.append('FECHA')

        # Eliminar OBJETO_FIJO (reemplazada por INVOLUCRA_OBJETO_FIJO)
        if 'INVOLUCRA_OBJETO_FIJO' in self.df.columns and 'OBJETO_FIJO' in self.df.columns:
            cols_to_drop.append('OBJETO_FIJO')

        # Eliminar duplicados en lista
        cols_to_drop = list(set(cols_to_drop))

        # Verificar que existen antes de eliminar
        cols_to_drop = [col for col in cols_to_drop if col in self.df.columns]

        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.transform_log.append(f"Eliminadas {len(cols_to_drop)} columnas redundantes")
            logger.info(f"Eliminadas {len(cols_to_drop)} columnas")

        logger.info("✓ Columnas innecesarias eliminadas")
        return self.df

    def transform(self, encode_method: str = 'auto',
                 scale_method: str = 'standard',
                 drop_redundant: bool = True) -> pd.DataFrame:
        """
        Pipeline completo de transformación.

        Args:
            encode_method: Método de encoding ('auto', 'label', 'onehot', 'frequency')
            scale_method: Método de escalado ('standard', 'minmax', 'robust')
            drop_redundant: Si eliminar columnas redundantes

        Returns:
            DataFrame transformado
        """
        logger.info("="*60)
        logger.info("INICIANDO TRANSFORMACIÓN DE DATOS")
        logger.info("="*60)

        # 1. Reducir cardinalidad
        self.reduce_high_cardinality()

        # 2. Crear features binarias
        self.create_binary_features()

        # 3. Encoding ordinal
        self.encode_ordinal_variables()

        # 4. Encoding categórico
        self.encode_categorical_variables(method=encode_method)

        # 5. Escalar variables numéricas
        # Solo escalar si no vamos a hacer análisis exploratorio después
        # self.scale_numeric_features(method=scale_method)

        # 6. Eliminar columnas redundantes
        if drop_redundant:
            self.drop_unnecessary_columns()

        logger.info("="*60)
        logger.info("TRANSFORMACIÓN COMPLETADA")
        logger.info("="*60)
        logger.info(f"Shape final: {self.df.shape}")

        return self.df

    def get_transform_report(self) -> Dict:
        """
        Genera reporte de transformaciones.

        Returns:
            Diccionario con información de transformaciones
        """
        return {
            'shape_final': self.df.shape,
            'encoders_usados': len(self.encoders),
            'columnas_encoded': len(self.encoded_columns),
            'log_acciones': self.transform_log
        }

    def print_transform_report(self):
        """Imprime reporte de transformaciones."""
        report = self.get_transform_report()

        print("\n" + "="*80)
        print("REPORTE DE TRANSFORMACIÓN Y ENCODING")
        print("="*80)
        print(f"\nShape final: {report['shape_final']}")
        print(f"Encoders utilizados: {report['encoders_usados']}")
        print(f"Columnas codificadas: {report['columnas_encoded']}")

        print("\nAcciones realizadas:")
        for i, action in enumerate(report['log_acciones'], 1):
            print(f"  {i}. {action}")


def main():
    """Función principal para pruebas."""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data_loader import DataLoader
    from preprocessing.cleaner import DataCleaner
    from preprocessing.feature_engineering import FeatureEngineer

    # Pipeline completo
    data_path = Path(__file__).parent.parent.parent / "data" / "siniestros_viales_consolidados_bogota.xlsx"
    loader = DataLoader(data_path)
    df = loader.load_data()

    cleaner = DataCleaner(df)
    df_clean = cleaner.clean()

    engineer = FeatureEngineer(df_clean)
    df_features = engineer.engineer_features()

    transformer = DataTransformer(df_features)
    df_transformed = transformer.transform()

    transformer.print_transform_report()

    return df_transformed, transformer


if __name__ == "__main__":
    df_transformed, transformer = main()
