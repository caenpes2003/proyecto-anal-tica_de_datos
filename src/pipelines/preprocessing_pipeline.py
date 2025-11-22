"""
Pipeline automatizado completo de preprocesamiento.

Integra limpieza, feature engineering, transformación y reducción de dimensionalidad.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import logging
import sys

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.transformers import DataTransformer
from preprocessing.dimensionality import DimensionalityReducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Pipeline automatizado de preprocesamiento de datos."""

    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        """
        Inicializa el pipeline.

        Args:
            df: DataFrame con datos crudos
            config: Configuración del pipeline (opcional)
        """
        self.df_original = df.copy()
        self.df_processed = None
        self.config = config or self._default_config()

        # Objetos de procesamiento
        self.cleaner = None
        self.engineer = None
        self.transformer = None
        self.reducer = None

        # Reportes
        self.pipeline_log = []
        self.execution_time = {}

    def _default_config(self) -> Dict:
        """
        Configuración por defecto del pipeline.

        Returns:
            Diccionario con configuración
        """
        return {
            'cleaning': {
                'remove_duplicates': True,
                'missing_strategy': 'smart',
                'outlier_method': 'validate',
                'outlier_action': 'remove'
            },
            'feature_engineering': {
                'temporal': True,
                'severity': True,
                'geographic': True,
                'interaction': True,
                'aggregated': True
            },
            'transformation': {
                'encode_method': 'auto',
                'scale_method': 'standard',
                'drop_redundant': True
            },
            'dimensionality': {
                'enabled': True,
                'method': 'correlation',
                'correlation_threshold': 0.95,
                'variance_threshold': 0.01
            },
            'output': {
                'save_intermediate': True,
                'save_final': True,
                'output_dir': 'data/processed'
            }
        }

    def step_1_clean_data(self) -> pd.DataFrame:
        """
        Paso 1: Limpieza de datos.

        Returns:
            DataFrame limpio
        """
        logger.info("\n" + "="*80)
        logger.info("PASO 1: LIMPIEZA DE DATOS")
        logger.info("="*80)

        start_time = datetime.now()

        self.cleaner = DataCleaner(self.df_original)
        df_clean = self.cleaner.clean(
            remove_duplicates=self.config['cleaning']['remove_duplicates'],
            missing_strategy=self.config['cleaning']['missing_strategy'],
            outlier_method=self.config['cleaning']['outlier_method'],
            outlier_action=self.config['cleaning']['outlier_action']
        )

        self.execution_time['cleaning'] = (datetime.now() - start_time).total_seconds()
        self.pipeline_log.append(f"Paso 1 completado en {self.execution_time['cleaning']:.2f}s")

        # Guardar intermedio si está configurado
        if self.config['output']['save_intermediate']:
            self._save_intermediate(df_clean, 'cleaned')

        return df_clean

    def step_2_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 2: Feature Engineering.

        Args:
            df: DataFrame limpio

        Returns:
            DataFrame con features
        """
        logger.info("\n" + "="*80)
        logger.info("PASO 2: FEATURE ENGINEERING")
        logger.info("="*80)

        start_time = datetime.now()

        self.engineer = FeatureEngineer(df)
        df_features = self.engineer.engineer_features(
            include_temporal=self.config['feature_engineering']['temporal'],
            include_severity=self.config['feature_engineering']['severity'],
            include_geographic=self.config['feature_engineering']['geographic'],
            include_interaction=self.config['feature_engineering']['interaction'],
            include_aggregated=self.config['feature_engineering']['aggregated']
        )

        self.execution_time['feature_engineering'] = (datetime.now() - start_time).total_seconds()
        self.pipeline_log.append(
            f"Paso 2 completado en {self.execution_time['feature_engineering']:.2f}s"
        )

        if self.config['output']['save_intermediate']:
            self._save_intermediate(df_features, 'features')

        return df_features

    def step_3_transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 3: Transformación y Encoding.

        Args:
            df: DataFrame con features

        Returns:
            DataFrame transformado
        """
        logger.info("\n" + "="*80)
        logger.info("PASO 3: TRANSFORMACIÓN Y ENCODING")
        logger.info("="*80)

        start_time = datetime.now()

        self.transformer = DataTransformer(df)
        df_transformed = self.transformer.transform(
            encode_method=self.config['transformation']['encode_method'],
            scale_method=self.config['transformation']['scale_method'],
            drop_redundant=self.config['transformation']['drop_redundant']
        )

        self.execution_time['transformation'] = (datetime.now() - start_time).total_seconds()
        self.pipeline_log.append(
            f"Paso 3 completado en {self.execution_time['transformation']:.2f}s"
        )

        if self.config['output']['save_intermediate']:
            self._save_intermediate(df_transformed, 'transformed')

        return df_transformed

    def step_4_reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 4: Reducción de Dimensionalidad.

        Args:
            df: DataFrame transformado

        Returns:
            DataFrame con dimensionalidad reducida
        """
        if not self.config['dimensionality']['enabled']:
            logger.info("Reducción de dimensionalidad deshabilitada")
            return df

        logger.info("\n" + "="*80)
        logger.info("PASO 4: REDUCCIÓN DE DIMENSIONALIDAD")
        logger.info("="*80)

        start_time = datetime.now()

        self.reducer = DimensionalityReducer(df)
        df_reduced = self.reducer.reduce_dimensions(
            method=self.config['dimensionality']['method'],
            correlation_threshold=self.config['dimensionality']['correlation_threshold'],
            variance_threshold=self.config['dimensionality']['variance_threshold']
        )

        self.execution_time['dimensionality'] = (datetime.now() - start_time).total_seconds()
        self.pipeline_log.append(
            f"Paso 4 completado en {self.execution_time['dimensionality']:.2f}s"
        )

        return df_reduced

    def run(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo.

        Returns:
            DataFrame procesado final
        """
        logger.info("\n" + "="*80)
        logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        logger.info("="*80)
        logger.info(f"Shape inicial: {self.df_original.shape}")

        overall_start = datetime.now()

        # Ejecutar pasos
        df_clean = self.step_1_clean_data()
        df_features = self.step_2_engineer_features(df_clean)
        df_transformed = self.step_3_transform_data(df_features)
        df_final = self.step_4_reduce_dimensionality(df_transformed)

        self.df_processed = df_final

        # Tiempo total
        total_time = (datetime.now() - overall_start).total_seconds()
        self.execution_time['total'] = total_time

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETADO")
        logger.info("="*80)
        logger.info(f"Shape final: {df_final.shape}")
        logger.info(f"Tiempo total: {total_time:.2f}s")

        # Guardar resultado final
        if self.config['output']['save_final']:
            self._save_final(df_final)

        # Generar reporte
        self._generate_report()

        return df_final

    def _save_intermediate(self, df: pd.DataFrame, step_name: str):
        """Guarda dataset intermedio."""
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"siniestros_{step_name}.csv"
        filepath = output_dir / filename

        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"✓ Dataset intermedio guardado: {filepath}")

    def _save_final(self, df: pd.DataFrame):
        """Guarda dataset final procesado."""
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = output_dir / "siniestros_final.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"✓ Dataset final guardado (CSV): {csv_path}")

        # Parquet (más eficiente para datasets grandes)
        parquet_path = output_dir / "siniestros_final.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"✓ Dataset final guardado (Parquet): {parquet_path}")

    def _generate_report(self):
        """Genera reporte completo del pipeline."""
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "preprocessing_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("REPORTE DE PREPROCESAMIENTO DE DATOS\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")

            # Resumen general
            f.write("RESUMEN GENERAL\n")
            f.write("-"*100 + "\n")
            f.write(f"Shape original: {self.df_original.shape}\n")
            f.write(f"Shape final: {self.df_processed.shape}\n")
            f.write(f"Registros eliminados: {self.df_original.shape[0] - self.df_processed.shape[0]}\n")
            f.write(f"Features creadas: {self.df_processed.shape[1] - self.df_original.shape[1]}\n")
            f.write(f"Tiempo total: {self.execution_time['total']:.2f} segundos\n\n")

            # Detalles por paso
            if self.cleaner:
                f.write("\n5.1. LIMPIEZA DE DATOS\n")
                f.write("-"*100 + "\n")
                report = self.cleaner.get_cleaning_report()
                for action in report['log_acciones']:
                    f.write(f"  • {action}\n")
                f.write(f"\nTiempo: {self.execution_time.get('cleaning', 0):.2f}s\n")

            if self.engineer:
                f.write("\n5.2. FEATURE ENGINEERING\n")
                f.write("-"*100 + "\n")
                report = self.engineer.get_feature_report()
                for action in report['log_acciones']:
                    f.write(f"  • {action}\n")
                f.write(f"\nNuevas features ({len(report['columnas_creadas'])}):\n")
                for col in report['columnas_creadas']:
                    f.write(f"  - {col}\n")
                f.write(f"\nTiempo: {self.execution_time.get('feature_engineering', 0):.2f}s\n")

            if self.transformer:
                f.write("\n5.4. TRANSFORMACIÓN Y ENCODING\n")
                f.write("-"*100 + "\n")
                report = self.transformer.get_transform_report()
                for action in report['log_acciones']:
                    f.write(f"  • {action}\n")
                f.write(f"\nTiempo: {self.execution_time.get('transformation', 0):.2f}s\n")

            if self.reducer:
                f.write("\n5.3. REDUCCIÓN DE DIMENSIONALIDAD\n")
                f.write("-"*100 + "\n")
                report = self.reducer.get_reduction_report()
                for action in report['log_acciones']:
                    f.write(f"  • {action}\n")
                f.write(f"\nTiempo: {self.execution_time.get('dimensionality', 0):.2f}s\n")

            # Columnas finales
            f.write("\nCOLUMNAS FINALES DEL DATASET\n")
            f.write("-"*100 + "\n")
            for i, col in enumerate(self.df_processed.columns, 1):
                dtype = self.df_processed[col].dtype
                f.write(f"  {i:3d}. {col} ({dtype})\n")

            f.write("\n" + "="*100 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("="*100 + "\n")

        logger.info(f"✓ Reporte generado: {report_path}")

        # Guardar configuración usada
        config_path = output_dir / "preprocessing_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Configuración guardada: {config_path}")

    def get_feature_dictionary(self) -> pd.DataFrame:
        """
        Genera diccionario de features.

        Returns:
            DataFrame con descripción de features
        """
        feature_dict = []

        for col in self.df_processed.columns:
            dtype = self.df_processed[col].dtype
            unique_count = self.df_processed[col].nunique()
            missing_count = self.df_processed[col].isnull().sum()

            # Determinar origen
            if col in self.df_original.columns:
                origin = 'Original'
            elif '_encoded' in col or '_freq' in col or '_ord' in col or '_bin' in col:
                origin = 'Transformada'
            else:
                origin = 'Creada'

            feature_dict.append({
                'columna': col,
                'tipo': str(dtype),
                'origen': origin,
                'valores_unicos': unique_count,
                'valores_faltantes': missing_count
            })

        df_dict = pd.DataFrame(feature_dict)

        # Guardar
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        dict_path = output_dir / "feature_dictionary.csv"
        df_dict.to_csv(dict_path, index=False, encoding='utf-8')
        logger.info(f"✓ Diccionario de features guardado: {dict_path}")

        return df_dict

    def print_summary(self):
        """Imprime resumen del pipeline."""
        print("\n" + "="*80)
        print("RESUMEN DEL PIPELINE DE PREPROCESAMIENTO")
        print("="*80)
        print(f"\nShape original: {self.df_original.shape}")
        print(f"Shape final: {self.df_processed.shape}")
        print(f"\nRegistros:")
        print(f"  Originales: {self.df_original.shape[0]:,}")
        print(f"  Finales: {self.df_processed.shape[0]:,}")
        print(f"  Eliminados: {self.df_original.shape[0] - self.df_processed.shape[0]:,}")
        print(f"  Retenidos: {(self.df_processed.shape[0]/self.df_original.shape[0]*100):.2f}%")
        print(f"\nFeatures:")
        print(f"  Originales: {self.df_original.shape[1]}")
        print(f"  Finales: {self.df_processed.shape[1]}")
        print(f"  Creadas/Modificadas: {self.df_processed.shape[1] - self.df_original.shape[1]}")
        print(f"\nTiempo de ejecución:")
        for step, time in self.execution_time.items():
            print(f"  {step.capitalize()}: {time:.2f}s")


def main():
    """Función principal para ejecutar el pipeline."""
    from data_loader import DataLoader

    # Cargar datos
    data_path = Path(__file__).parent.parent.parent / "data" / "siniestros_viales_consolidados_bogota.xlsx"
    logger.info(f"Cargando datos desde: {data_path}")

    loader = DataLoader(data_path)
    df = loader.load_data()

    # Ejecutar pipeline
    pipeline = PreprocessingPipeline(df)
    df_final = pipeline.run()

    # Resumen
    pipeline.print_summary()

    # Diccionario de features
    feature_dict = pipeline.get_feature_dictionary()
    print(f"\n✓ Diccionario de features generado con {len(feature_dict)} columnas")

    return df_final, pipeline


if __name__ == "__main__":
    df_final, pipeline = main()
