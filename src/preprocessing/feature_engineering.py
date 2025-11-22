"""
Módulo para feature engineering.

Crea nuevas características a partir de los datos existentes
para análisis de patrones y factores de riesgo.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Clase para crear nuevas características."""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el ingeniero de características.

        Args:
            df: DataFrame con datos limpios
        """
        self.df = df.copy()
        self.feature_log = []

    def create_temporal_features(self) -> pd.DataFrame:
        """
        Crea características temporales desde FECHA y HORA.

        Returns:
            DataFrame con features temporales
        """
        logger.info("Creando features temporales...")

        # Asegurar que FECHA sea datetime
        if self.df['FECHA'].dtype != 'datetime64[ns]':
            self.df['FECHA'] = pd.to_datetime(self.df['FECHA'], errors='coerce')

        # Features de fecha
        self.df['anio'] = self.df['FECHA'].dt.year
        self.df['mes'] = self.df['FECHA'].dt.month
        self.df['dia_mes'] = self.df['FECHA'].dt.day
        self.df['dia_semana'] = self.df['FECHA'].dt.dayofweek  # 0=Lunes, 6=Domingo
        self.df['nombre_dia'] = self.df['FECHA'].dt.day_name()
        self.df['trimestre'] = self.df['FECHA'].dt.quarter
        self.df['semana_anio'] = self.df['FECHA'].dt.isocalendar().week

        # Es fin de semana (Sábado=5, Domingo=6)
        self.df['es_fin_semana'] = (self.df['dia_semana'] >= 5).astype(int)

        # Features de hora
        if 'HORA' in self.df.columns:
            # Extraer hora como número
            try:
                self.df['hora_num'] = pd.to_datetime(self.df['HORA'], format='%H:%M:%S', errors='coerce').dt.hour
            except:
                try:
                    self.df['hora_num'] = pd.to_datetime(self.df['HORA'], errors='coerce').dt.hour
                except:
                    logger.warning("No se pudo extraer hora numérica")
                    self.df['hora_num'] = 12  # Valor por defecto

            # Periodo del día
            def get_periodo_dia(hora):
                if pd.isna(hora):
                    return 'DESCONOCIDO'
                if 0 <= hora < 6:
                    return 'MADRUGADA'
                elif 6 <= hora < 12:
                    return 'MANANA'
                elif 12 <= hora < 18:
                    return 'TARDE'
                else:
                    return 'NOCHE'

            self.df['periodo_dia'] = self.df['hora_num'].apply(get_periodo_dia)

            # Franja horaria (horas pico vs valle)
            def get_franja_horaria(hora):
                if pd.isna(hora):
                    return 'DESCONOCIDO'
                # Horas pico: 6-9 AM y 5-8 PM
                if (6 <= hora < 9) or (17 <= hora < 20):
                    return 'PICO'
                else:
                    return 'VALLE'

            self.df['franja_horaria'] = self.df['hora_num'].apply(get_franja_horaria)

        self.feature_log.append("Creadas 12 features temporales")
        logger.info("✓ Features temporales creadas")

        return self.df

    def create_severity_features(self) -> pd.DataFrame:
        """
        Crea características de severidad y riesgo.

        Returns:
            DataFrame con features de severidad
        """
        logger.info("Creando features de severidad...")

        # Puntaje de gravedad (ya existe GRAVEDAD, crear versión normalizada)
        if 'GRAVEDAD' in self.df.columns:
            # Mapeo: 1=CON MUERTOS (más grave), 2=CON HERIDOS, 3=SOLO DAÑOS
            gravedad_map = {1: 3, 2: 2, 3: 1}
            self.df['puntaje_gravedad'] = self.df['GRAVEDAD'].map(gravedad_map).fillna(0)

        # Tipo de choque simplificado
        if 'CHOQUE' in self.df.columns:
            # Agrupar tipos de choque en categorías más generales
            # Necesitaremos investigar los códigos específicos, por ahora crear categorías básicas
            self.df['choque_multiple'] = (self.df['CHOQUE'] > 1).astype(int)

        # Indicador de riesgo alto (combina gravedad + objeto fijo + choque)
        if all(col in self.df.columns for col in ['puntaje_gravedad', 'INVOLUCRA_OBJETO_FIJO']):
            self.df['riesgo_alto'] = (
                (self.df['puntaje_gravedad'] >= 2) |
                (self.df['INVOLUCRA_OBJETO_FIJO'] == 1)
            ).astype(int)

        self.feature_log.append("Creadas features de severidad")
        logger.info("✓ Features de severidad creadas")

        return self.df

    def create_geographic_features(self) -> pd.DataFrame:
        """
        Crea características geográficas.

        Returns:
            DataFrame con features geográficas
        """
        logger.info("Creando features geográficas...")

        # Mapeo de códigos de localidad a nombres (Bogotá)
        localidad_map = {
            1: 'USAQUEN', 2: 'CHAPINERO', 3: 'SANTA FE', 4: 'SAN CRISTOBAL',
            5: 'USME', 6: 'TUNJUELITO', 7: 'BOSA', 8: 'KENNEDY',
            9: 'FONTIBON', 10: 'ENGATIVA', 11: 'SUBA', 12: 'BARRIOS UNIDOS',
            13: 'TEUSAQUILLO', 14: 'MARTIRES', 15: 'ANTONIO NARINO',
            16: 'PUENTE ARANDA', 17: 'CANDELARIA', 18: 'RAFAEL URIBE',
            19: 'CIUDAD BOLIVAR', 20: 'SUMAPAZ'
        }

        if 'CODIGO_LOCALIDAD' in self.df.columns:
            self.df['localidad_nombre'] = self.df['CODIGO_LOCALIDAD'].map(localidad_map)
            self.df['localidad_nombre'].fillna('DESCONOCIDA', inplace=True)

            # Agrupar por zonas de Bogotá
            zona_map = {
                'USAQUEN': 'NORTE', 'CHAPINERO': 'NORTE', 'SUBA': 'NORTE',
                'ENGATIVA': 'NORTE', 'BARRIOS UNIDOS': 'NORTE',
                'SANTA FE': 'CENTRO', 'TEUSAQUILLO': 'CENTRO', 'MARTIRES': 'CENTRO',
                'CANDELARIA': 'CENTRO', 'ANTONIO NARINO': 'CENTRO',
                'SAN CRISTOBAL': 'SUR', 'USME': 'SUR', 'TUNJUELITO': 'SUR',
                'RAFAEL URIBE': 'SUR', 'CIUDAD BOLIVAR': 'SUR', 'SUMAPAZ': 'SUR',
                'KENNEDY': 'OCCIDENTE', 'FONTIBON': 'OCCIDENTE', 'BOSA': 'OCCIDENTE',
                'PUENTE ARANDA': 'OCCIDENTE'
            }
            self.df['zona_bogota'] = self.df['localidad_nombre'].map(zona_map)
            self.df['zona_bogota'].fillna('DESCONOCIDA', inplace=True)

        # Extraer tipo de vía de DIRECCION
        if 'DIRECCION' in self.df.columns:
            def extract_via_type(direccion):
                if pd.isna(direccion) or direccion == 'DESCONOCIDA':
                    return 'DESCONOCIDA'
                direccion_upper = str(direccion).upper()
                if 'KR' in direccion_upper or 'CARRERA' in direccion_upper:
                    return 'CARRERA'
                elif 'CL' in direccion_upper or 'CALLE' in direccion_upper:
                    return 'CALLE'
                elif 'AV' in direccion_upper or 'AVENIDA' in direccion_upper:
                    return 'AVENIDA'
                elif 'DG' in direccion_upper or 'DIAGONAL' in direccion_upper:
                    return 'DIAGONAL'
                elif 'TV' in direccion_upper or 'TRANSVERSAL' in direccion_upper:
                    return 'TRANSVERSAL'
                else:
                    return 'OTRA'

            self.df['tipo_via'] = self.df['DIRECCION'].apply(extract_via_type)

        self.feature_log.append("Creadas features geográficas")
        logger.info("✓ Features geográficas creadas")

        return self.df

    def create_interaction_features(self) -> pd.DataFrame:
        """
        Crea características de interacción.

        Returns:
            DataFrame con features de interacción
        """
        logger.info("Creando features de interacción...")

        # Gravedad x Periodo del día
        if all(col in self.df.columns for col in ['puntaje_gravedad', 'periodo_dia']):
            self.df['gravedad_x_periodo'] = (
                self.df['puntaje_gravedad'].astype(str) + '_' + self.df['periodo_dia']
            )

        # Fin de semana x Franja horaria
        if all(col in self.df.columns for col in ['es_fin_semana', 'franja_horaria']):
            def get_momento_semana(row):
                if row['es_fin_semana'] == 1:
                    return f"FDS_{row['franja_horaria']}"
                else:
                    return f"LABORAL_{row['franja_horaria']}"

            self.df['momento_semana'] = self.df.apply(get_momento_semana, axis=1)

        # Zona x Periodo
        if all(col in self.df.columns for col in ['zona_bogota', 'periodo_dia']):
            self.df['zona_x_periodo'] = (
                self.df['zona_bogota'] + '_' + self.df['periodo_dia']
            )

        self.feature_log.append("Creadas features de interacción")
        logger.info("✓ Features de interacción creadas")

        return self.df

    def create_aggregated_features(self) -> pd.DataFrame:
        """
        Crea características agregadas (frecuencias, conteos).

        Returns:
            DataFrame con features agregadas
        """
        logger.info("Creando features agregadas...")

        # Frecuencia de siniestros por localidad
        if 'CODIGO_LOCALIDAD' in self.df.columns:
            localidad_freq = self.df['CODIGO_LOCALIDAD'].value_counts().to_dict()
            self.df['siniestros_localidad'] = self.df['CODIGO_LOCALIDAD'].map(localidad_freq)

        # Frecuencia por tipo de vía
        if 'tipo_via' in self.df.columns:
            via_freq = self.df['tipo_via'].value_counts().to_dict()
            self.df['siniestros_tipo_via'] = self.df['tipo_via'].map(via_freq)

        # Frecuencia por periodo del día
        if 'periodo_dia' in self.df.columns:
            periodo_freq = self.df['periodo_dia'].value_counts().to_dict()
            self.df['siniestros_periodo'] = self.df['periodo_dia'].map(periodo_freq)

        self.feature_log.append("Creadas features agregadas")
        logger.info("✓ Features agregadas creadas")

        return self.df

    def engineer_features(self, include_temporal: bool = True,
                         include_severity: bool = True,
                         include_geographic: bool = True,
                         include_interaction: bool = True,
                         include_aggregated: bool = True) -> pd.DataFrame:
        """
        Pipeline completo de feature engineering.

        Args:
            include_temporal: Incluir features temporales
            include_severity: Incluir features de severidad
            include_geographic: Incluir features geográficas
            include_interaction: Incluir features de interacción
            include_aggregated: Incluir features agregadas

        Returns:
            DataFrame con nuevas features
        """
        logger.info("="*60)
        logger.info("INICIANDO FEATURE ENGINEERING")
        logger.info("="*60)
        initial_cols = len(self.df.columns)

        if include_temporal:
            self.create_temporal_features()

        if include_severity:
            self.create_severity_features()

        if include_geographic:
            self.create_geographic_features()

        if include_interaction:
            self.create_interaction_features()

        if include_aggregated:
            self.create_aggregated_features()

        final_cols = len(self.df.columns)
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING COMPLETADO")
        logger.info("="*60)
        logger.info(f"Columnas iniciales: {initial_cols}")
        logger.info(f"Columnas finales: {final_cols}")
        logger.info(f"Features creadas: {final_cols - initial_cols}")

        return self.df

    def get_feature_report(self) -> Dict:
        """
        Genera reporte de features creadas.

        Returns:
            Diccionario con información de features
        """
        return {
            'total_features': len(self.df.columns),
            'log_acciones': self.feature_log,
            'columnas_creadas': [col for col in self.df.columns if col not in
                                ['CODIGO_ACCIDENTE', 'FECHA', 'HORA', 'GRAVEDAD',
                                 'CLASE', 'CHOQUE', 'OBJETO_FIJO', 'DIRECCION',
                                 'CODIGO_LOCALIDAD', 'DISENO_LUGAR']]
        }

    def print_feature_report(self):
        """Imprime reporte de features."""
        report = self.get_feature_report()

        print("\n" + "="*80)
        print("REPORTE DE FEATURE ENGINEERING")
        print("="*80)
        print(f"\nTotal de features: {report['total_features']}")
        print(f"Features creadas: {len(report['columnas_creadas'])}")

        print("\nAcciones realizadas:")
        for i, action in enumerate(report['log_acciones'], 1):
            print(f"  {i}. {action}")

        print("\nNuevas columnas:")
        for i, col in enumerate(report['columnas_creadas'], 1):
            print(f"  {i:2d}. {col}")


def main():
    """Función principal para pruebas."""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data_loader import DataLoader
    from preprocessing.cleaner import DataCleaner

    # Cargar y limpiar datos
    data_path = Path(__file__).parent.parent.parent / "data" / "siniestros_viales_consolidados_bogota.xlsx"
    loader = DataLoader(data_path)
    df = loader.load_data()

    cleaner = DataCleaner(df)
    df_clean = cleaner.clean()

    # Feature engineering
    engineer = FeatureEngineer(df_clean)
    df_features = engineer.engineer_features()

    # Mostrar reporte
    engineer.print_feature_report()

    return df_features, engineer


if __name__ == "__main__":
    df_features, engineer = main()
