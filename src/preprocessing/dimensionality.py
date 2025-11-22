"""
Módulo para reducción de dimensionalidad.

Análisis de correlación, varianza y selección de features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Clase para reducción de dimensionalidad."""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el reductor de dimensionalidad.

        Args:
            df: DataFrame transformado
        """
        self.df = df.copy()
        self.reduction_log = []
        self.removed_features = []
        self.selected_features = []

    def remove_low_variance_features(self, threshold: float = 0.01) -> pd.DataFrame:
        """
        Elimina features con varianza muy baja.

        Args:
            threshold: Umbral de varianza mínima

        Returns:
            DataFrame sin features de baja varianza
        """
        logger.info("Analizando varianza de features...")

        # Solo analizar columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Excluir IDs y fechas
        exclude = ['CODIGO_ACCIDENTE', 'anio']
        numeric_cols = [col for col in numeric_cols if col not in exclude]

        low_variance_cols = []

        for col in numeric_cols:
            variance = self.df[col].var()
            if variance < threshold:
                low_variance_cols.append(col)

        if low_variance_cols:
            self.df.drop(columns=low_variance_cols, inplace=True)
            self.removed_features.extend(low_variance_cols)
            self.reduction_log.append(
                f"Eliminadas {len(low_variance_cols)} features con varianza < {threshold}"
            )
            logger.info(f"Eliminadas {len(low_variance_cols)} features de baja varianza")
        else:
            logger.info("No se encontraron features con varianza muy baja")

        return self.df

    def remove_highly_correlated_features(self, threshold: float = 0.95) -> pd.DataFrame:
        """
        Elimina features altamente correlacionadas.

        Args:
            threshold: Umbral de correlación (default: 0.95)

        Returns:
            DataFrame sin features redundantes
        """
        logger.info("Analizando correlación entre features...")

        # Solo columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Excluir IDs
        exclude = ['CODIGO_ACCIDENTE']
        numeric_cols = [col for col in numeric_cols if col not in exclude]

        if len(numeric_cols) < 2:
            logger.info("No hay suficientes features numéricas para análisis de correlación")
            return self.df

        # Calcular matriz de correlación
        corr_matrix = self.df[numeric_cols].corr().abs()

        # Seleccionar triángulo superior
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Encontrar features con correlación > threshold
        to_drop = [column for column in upper_triangle.columns
                  if any(upper_triangle[column] > threshold)]

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.removed_features.extend(to_drop)
            self.reduction_log.append(
                f"Eliminadas {len(to_drop)} features con correlación > {threshold}"
            )
            logger.info(f"Eliminadas {len(to_drop)} features altamente correlacionadas")
        else:
            logger.info(f"No se encontraron features con correlación > {threshold}")

        return self.df

    def select_k_best_features(self, target_col: str, k: int = 20,
                              score_func: str = 'f_classif') -> pd.DataFrame:
        """
        Selecciona las K mejores features según una función de score.

        Args:
            target_col: Columna objetivo
            k: Número de features a seleccionar
            score_func: 'f_classif', 'chi2', 'mutual_info'

        Returns:
            DataFrame con K mejores features
        """
        logger.info(f"Seleccionando {k} mejores features...")

        if target_col not in self.df.columns:
            logger.error(f"Columna objetivo {target_col} no encontrada")
            return self.df

        # Separar features y target
        X = self.df.drop(columns=[target_col, 'CODIGO_ACCIDENTE', 'FECHA'],
                        errors='ignore')
        y = self.df[target_col]

        # Solo columnas numéricas
        X_numeric = X.select_dtypes(include=[np.number])

        if X_numeric.empty:
            logger.warning("No hay features numéricas para selección")
            return self.df

        # Seleccionar función de score
        if score_func == 'f_classif':
            func = f_classif
        elif score_func == 'chi2':
            func = chi2
            # chi2 requiere valores no negativos
            X_numeric = X_numeric - X_numeric.min() + 1e-5
        elif score_func == 'mutual_info':
            func = mutual_info_classif
        else:
            logger.warning(f"Función {score_func} no reconocida, usando f_classif")
            func = f_classif

        # Seleccionar K best
        k = min(k, len(X_numeric.columns))  # No más de las disponibles
        selector = SelectKBest(score_func=func, k=k)

        try:
            selector.fit(X_numeric, y)

            # Obtener features seleccionadas
            selected_features_mask = selector.get_support()
            selected_features = X_numeric.columns[selected_features_mask].tolist()

            # Features no seleccionadas
            not_selected = X_numeric.columns[~selected_features_mask].tolist()

            self.selected_features = selected_features
            self.reduction_log.append(
                f"Seleccionadas {len(selected_features)} mejores features usando {score_func}"
            )

            # Mantener solo features seleccionadas + columnas importantes
            keep_cols = selected_features + [target_col, 'CODIGO_ACCIDENTE', 'FECHA']
            keep_cols = [col for col in keep_cols if col in self.df.columns]

            self.df = self.df[keep_cols]

            logger.info(f"Seleccionadas {len(selected_features)} features")

        except Exception as e:
            logger.error(f"Error en selección de features: {e}")

        return self.df

    def apply_pca(self, n_components: float = 0.95,
                  exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, PCA]:
        """
        Aplica PCA para reducción de dimensionalidad.

        Args:
            n_components: Número de componentes o varianza explicada (0-1)
            exclude_cols: Columnas a excluir de PCA

        Returns:
            Tupla (DataFrame con componentes principales, objeto PCA)
        """
        logger.info("Aplicando PCA...")

        # Columnas a excluir
        default_exclude = ['CODIGO_ACCIDENTE', 'FECHA', 'GRAVEDAD', 'puntaje_gravedad']
        exclude_cols = exclude_cols or []
        exclude_cols.extend(default_exclude)

        # Solo columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols_for_pca = [col for col in numeric_cols if col not in exclude_cols]

        if len(cols_for_pca) < 2:
            logger.warning("No hay suficientes features para PCA")
            return self.df, None

        # Extraer datos para PCA
        X = self.df[cols_for_pca]

        # Aplicar PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X)

        # Crear DataFrame con componentes
        n_components_actual = principal_components.shape[1]
        component_names = [f'PC{i+1}' for i in range(n_components_actual)]

        pca_df = pd.DataFrame(
            data=principal_components,
            columns=component_names,
            index=self.df.index
        )

        # Agregar columnas excluidas
        for col in exclude_cols:
            if col in self.df.columns:
                pca_df[col] = self.df[col].values

        # Información de varianza explicada
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        self.reduction_log.append(
            f"PCA aplicado: {n_components_actual} componentes "
            f"(varianza explicada: {cumulative_var[-1]*100:.2f}%)"
        )

        logger.info(f"PCA completado: {n_components_actual} componentes")
        logger.info(f"Varianza explicada acumulada: {cumulative_var[-1]*100:.2f}%")

        return pca_df, pca

    def reduce_dimensions(self, method: str = 'correlation',
                         correlation_threshold: float = 0.95,
                         variance_threshold: float = 0.01,
                         target_col: Optional[str] = None,
                         k_best: int = 30) -> pd.DataFrame:
        """
        Pipeline de reducción de dimensionalidad.

        Args:
            method: 'correlation', 'variance', 'kbest', 'pca', 'all'
            correlation_threshold: Umbral para correlación
            variance_threshold: Umbral para varianza
            target_col: Columna objetivo (para kbest)
            k_best: Número de features a seleccionar (para kbest)

        Returns:
            DataFrame con dimensionalidad reducida
        """
        logger.info("="*60)
        logger.info("INICIANDO REDUCCIÓN DE DIMENSIONALIDAD")
        logger.info("="*60)
        initial_shape = self.df.shape

        if method in ['variance', 'all']:
            self.remove_low_variance_features(threshold=variance_threshold)

        if method in ['correlation', 'all']:
            self.remove_highly_correlated_features(threshold=correlation_threshold)

        if method in ['kbest'] and target_col:
            self.select_k_best_features(target_col=target_col, k=k_best)

        # PCA se aplica por separado ya que cambia completamente las features
        # if method == 'pca':
        #     self.df, _ = self.apply_pca()

        final_shape = self.df.shape
        logger.info("="*60)
        logger.info("REDUCCIÓN DE DIMENSIONALIDAD COMPLETADA")
        logger.info("="*60)
        logger.info(f"Shape inicial: {initial_shape}")
        logger.info(f"Shape final: {final_shape}")
        logger.info(f"Features eliminadas: {initial_shape[1] - final_shape[1]}")

        return self.df

    def get_reduction_report(self) -> Dict:
        """
        Genera reporte de reducción.

        Returns:
            Diccionario con información de reducción
        """
        return {
            'shape_final': self.df.shape,
            'features_eliminadas': len(self.removed_features),
            'features_seleccionadas': len(self.selected_features) if self.selected_features else 0,
            'lista_eliminadas': self.removed_features,
            'lista_seleccionadas': self.selected_features,
            'log_acciones': self.reduction_log
        }

    def print_reduction_report(self):
        """Imprime reporte de reducción."""
        report = self.get_reduction_report()

        print("\n" + "="*80)
        print("REPORTE DE REDUCCIÓN DE DIMENSIONALIDAD")
        print("="*80)
        print(f"\nShape final: {report['shape_final']}")
        print(f"Features eliminadas: {report['features_eliminadas']}")

        if report['features_seleccionadas'] > 0:
            print(f"Features seleccionadas: {report['features_seleccionadas']}")

        print("\nAcciones realizadas:")
        for i, action in enumerate(report['log_acciones'], 1):
            print(f"  {i}. {action}")

        if report['lista_eliminadas']:
            print(f"\nFeatures eliminadas ({len(report['lista_eliminadas'])}):")
            for i, feat in enumerate(report['lista_eliminadas'][:10], 1):
                print(f"  {i}. {feat}")
            if len(report['lista_eliminadas']) > 10:
                print(f"  ... y {len(report['lista_eliminadas']) - 10} más")


def main():
    """Función principal para pruebas."""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data_loader import DataLoader
    from preprocessing.cleaner import DataCleaner
    from preprocessing.feature_engineering import FeatureEngineer
    from preprocessing.transformers import DataTransformer

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

    reducer = DimensionalityReducer(df_transformed)
    df_reduced = reducer.reduce_dimensions(method='correlation')

    reducer.print_reduction_report()

    return df_reduced, reducer


if __name__ == "__main__":
    df_reduced, reducer = main()
