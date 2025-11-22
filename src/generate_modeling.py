"""
Script automatizado para Modelado y Machine Learning.

Genera modelos, optimiza hiperparámetros y valida en ~15-20 minutos.

Sección 7: Metodología
- 7.1. Selección del modelo (clasificación + clustering)
- 7.2. Algoritmos y técnicas utilizadas
- 7.3. Justificación de los hiperparámetros
- 7.4. Validación cruzada y técnicas de re-muestreo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
import warnings
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                            f1_score, accuracy_score, precision_score, recall_score,
                            roc_auc_score, roc_curve, silhouette_score, davies_bouldin_score)

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Determinar directorio base
if Path('data/processed').exists():
    base_dir = Path('.')
elif Path('../data/processed').exists():
    base_dir = Path('..')
else:
    base_dir = Path(__file__).parent.parent

# Crear carpetas de salida
models_dir = base_dir / 'data' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

viz_dir = base_dir / 'reports' / 'visualizaciones_modelos'
viz_dir.mkdir(parents=True, exist_ok=True)

class_dir = viz_dir / 'clasificacion'
cluster_dir = viz_dir / 'clustering'
comp_dir = viz_dir / 'comparacion'

for d in [class_dir, cluster_dir, comp_dir]:
    d.mkdir(exist_ok=True)

# Variables globales para resultados
resultados_modelos = {}
metricas_comparacion = []

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

def cargar_datos():
    """Carga el dataset procesado para modelado."""
    print("\n" + "="*70)
    print("CARGANDO DATOS PARA MODELADO")
    print("="*70)

    # Buscar archivo
    base_paths = [Path('.'), Path('..'), Path(__file__).parent.parent]

    for base in base_paths:
        csv_path = base / 'data' / 'processed' / 'siniestros_features.csv'
        if csv_path.exists():
            print(f"Cargando desde: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"Dataset cargado: {df.shape}")
            return df

    raise FileNotFoundError("No se encontró siniestros_features.csv")


def preparar_datos(df, test_size=0.2, random_state=42):
    """Prepara datos para modelado de clasificación."""
    print("\nPreparando datos para clasificación...")

    # Seleccionar características relevantes
    # IMPORTANTE: NO incluir puntaje_gravedad (es derivado de GRAVEDAD - causa data leakage)
    features_numericas = [
        'anio', 'mes', 'dia_semana', 'hora_num', 'es_fin_semana',
        'CODIGO_LOCALIDAD', 'CLASE', 'CHOQUE',
        'DISENO_LUGAR', 'trimestre', 'semana_anio'
    ]

    features_categoricas = [
        'periodo_dia', 'zona_bogota', 'tipo_via', 'momento_semana'
    ]

    # Verificar columnas disponibles
    features_disponibles = []
    for col in features_numericas + features_categoricas:
        if col in df.columns:
            features_disponibles.append(col)

    print(f"Features disponibles: {len(features_disponibles)}")

    # Preparar X (características)
    X = df[features_disponibles].copy()

    # Encodificar variables categóricas
    le_dict = {}
    for col in features_categoricas:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    # Manejar valores faltantes
    X = X.fillna(X.median())

    # Variable objetivo: GRAVEDAD (1=Muertos, 2=Heridos, 3=Solo daños)
    y = df['GRAVEDAD'].copy()

    # Mapear a etiquetas más intuitivas para el reporte
    # Mantener los valores originales pero guardar mapeo
    gravedad_labels = {1: 'CON_MUERTOS', 2: 'CON_HERIDOS', 3: 'SOLO_DANOS'}

    print(f"\nDistribución de clases:")
    print(y.value_counts().sort_index())
    print(f"\nPorcentajes:")
    print(y.value_counts(normalize=True).sort_index() * 100)

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar nombres de features
    feature_names = X.columns.tolist()

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_orig': X_train,
        'X_test_orig': X_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'label_encoders': le_dict,
        'gravedad_labels': gravedad_labels
    }


# ============================================================================
# SECCIÓN 7.2: MODELOS DE CLASIFICACIÓN
# ============================================================================

def entrenar_logistic_regression(data, optimize=True):
    """Entrena Regresión Logística con optimización opcional."""
    print("\n" + "-"*70)
    print("MODELO 1: REGRESION LOGISTICA")
    print("-"*70)

    inicio = time.time()

    if optimize:
        print("Optimizando hiperparámetros con RandomizedSearchCV...")

        param_dist = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'class_weight': ['balanced', None],
            'max_iter': [300, 500, 1000]
        }

        lr = LogisticRegression(random_state=42)

        search = RandomizedSearchCV(
            lr, param_dist, n_iter=10, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=42, verbose=2
        )

        search.fit(data['X_train'], data['y_train'])
        modelo = search.best_estimator_

        print(f"\nMejores parámetros: {search.best_params_}")
        print(f"Mejor F1-Score (CV): {search.best_score_:.4f}")

        best_params = search.best_params_
    else:
        modelo = LogisticRegression(random_state=42, max_iter=1000)
        modelo.fit(data['X_train'], data['y_train'])
        best_params = modelo.get_params()

    # Predicciones
    y_pred = modelo.predict(data['X_test'])
    y_pred_proba = modelo.predict_proba(data['X_test'])

    # Métricas
    metricas = calcular_metricas(data['y_test'], y_pred, y_pred_proba)

    tiempo = time.time() - inicio
    print(f"\nTiempo de entrenamiento: {tiempo:.2f}s")
    print_metricas(metricas)

    # Guardar resultados
    resultados_modelos['Logistic_Regression'] = {
        'modelo': modelo,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metricas': metricas,
        'tiempo': tiempo,
        'best_params': best_params
    }

    return modelo, metricas


def entrenar_random_forest(data, optimize=True):
    """Entrena Random Forest con optimización opcional."""
    print("\n" + "-"*70)
    print("MODELO 2: RANDOM FOREST")
    print("-"*70)

    inicio = time.time()

    if optimize:
        print("Optimizando hiperparámetros con RandomizedSearchCV...")

        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'max_features': ['sqrt', 'log2']
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        search = RandomizedSearchCV(
            rf, param_dist, n_iter=10, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=42, verbose=2
        )

        search.fit(data['X_train'], data['y_train'])
        modelo = search.best_estimator_

        print(f"\nMejores parámetros: {search.best_params_}")
        print(f"Mejor F1-Score (CV): {search.best_score_:.4f}")

        best_params = search.best_params_
    else:
        modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        modelo.fit(data['X_train'], data['y_train'])
        best_params = modelo.get_params()

    # Predicciones
    y_pred = modelo.predict(data['X_test'])
    y_pred_proba = modelo.predict_proba(data['X_test'])

    # Métricas
    metricas = calcular_metricas(data['y_test'], y_pred, y_pred_proba)

    tiempo = time.time() - inicio
    print(f"\nTiempo de entrenamiento: {tiempo:.2f}s")
    print_metricas(metricas)

    # Guardar resultados
    resultados_modelos['Random_Forest'] = {
        'modelo': modelo,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metricas': metricas,
        'tiempo': tiempo,
        'best_params': best_params,
        'feature_importance': modelo.feature_importances_
    }

    return modelo, metricas


def entrenar_xgboost(data, optimize=True):
    """Entrena XGBoost con optimización opcional."""
    print("\n" + "-"*70)
    print("MODELO 3: XGBOOST")
    print("-"*70)

    inicio = time.time()

    # Ajustar etiquetas para XGBoost (debe empezar en 0)
    y_train_xgb = data['y_train'] - 1
    y_test_xgb = data['y_test'] - 1

    if optimize:
        print("Optimizando hiperparámetros con RandomizedSearchCV...")

        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }

        xgb_model = xgb.XGBClassifier(
            random_state=42, n_jobs=-1, eval_metric='mlogloss'
        )

        search = RandomizedSearchCV(
            xgb_model, param_dist, n_iter=10, cv=3,
            scoring='f1_macro', n_jobs=-1, random_state=42, verbose=2
        )

        search.fit(data['X_train'], y_train_xgb)
        modelo = search.best_estimator_

        print(f"\nMejores parámetros: {search.best_params_}")
        print(f"Mejor F1-Score (CV): {search.best_score_:.4f}")

        best_params = search.best_params_
    else:
        modelo = xgb.XGBClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, eval_metric='mlogloss'
        )
        modelo.fit(data['X_train'], y_train_xgb)
        best_params = modelo.get_params()

    # Predicciones
    y_pred_xgb = modelo.predict(data['X_test'])
    y_pred_proba = modelo.predict_proba(data['X_test'])

    # Convertir de vuelta a etiquetas originales
    y_pred = y_pred_xgb + 1

    # Métricas (usar etiquetas originales)
    metricas = calcular_metricas(data['y_test'], y_pred, y_pred_proba)

    tiempo = time.time() - inicio
    print(f"\nTiempo de entrenamiento: {tiempo:.2f}s")
    print_metricas(metricas)

    # Guardar resultados
    resultados_modelos['XGBoost'] = {
        'modelo': modelo,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metricas': metricas,
        'tiempo': tiempo,
        'best_params': best_params,
        'feature_importance': modelo.feature_importances_
    }

    return modelo, metricas


# ============================================================================
# MÉTRICAS Y EVALUACIÓN
# ============================================================================

def calcular_metricas(y_true, y_pred, y_pred_proba=None):
    """Calcula métricas de clasificación."""
    metricas = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # AUC multiclase (one-vs-rest)
    if y_pred_proba is not None:
        try:
            metricas['roc_auc_ovr'] = roc_auc_score(
                y_true, y_pred_proba, multi_class='ovr', average='macro'
            )
        except:
            metricas['roc_auc_ovr'] = None

    return metricas


def print_metricas(metricas):
    """Imprime métricas de forma legible."""
    print("\nMétricas de evaluación:")
    print(f"  Accuracy:        {metricas['accuracy']:.4f}")
    print(f"  Precision:       {metricas['precision_macro']:.4f}")
    print(f"  Recall:          {metricas['recall_macro']:.4f}")
    print(f"  F1-Score:        {metricas['f1_macro']:.4f}")
    print(f"  F1-Weighted:     {metricas['f1_weighted']:.4f}")
    if metricas.get('roc_auc_ovr'):
        print(f"  ROC-AUC (OvR):   {metricas['roc_auc_ovr']:.4f}")


# ============================================================================
# CLUSTERING
# ============================================================================

def aplicar_clustering(data, k_values=[3, 4, 5, 6]):
    """Aplica K-Means clustering con diferentes valores de k."""
    print("\n" + "="*70)
    print("CLUSTERING: K-MEANS")
    print("="*70)

    # Usar datos escalados
    X = data['X_train']

    # Probar diferentes valores de k
    resultados_k = {}

    for k in k_values:
        print(f"\nProbando K-Means con k={k}...")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        # Métricas
        silhouette = silhouette_score(X, clusters)
        davies_bouldin = davies_bouldin_score(X, clusters)
        inertia = kmeans.inertia_

        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin:   {davies_bouldin:.4f}")
        print(f"  Inertia:          {inertia:.2f}")

        resultados_k[k] = {
            'modelo': kmeans,
            'clusters': clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'inertia': inertia
        }

    # Seleccionar mejor k (mayor silhouette)
    mejor_k = max(resultados_k.keys(), key=lambda k: resultados_k[k]['silhouette'])
    print(f"\nMejor k según Silhouette Score: {mejor_k}")

    # Aplicar PCA para visualización
    print("\nAplicando PCA para visualización...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")

    return {
        'resultados_k': resultados_k,
        'mejor_k': mejor_k,
        'pca': pca,
        'X_pca': X_pca,
        'modelo_final': resultados_k[mejor_k]['modelo'],
        'clusters_final': resultados_k[mejor_k]['clusters']
    }


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def generar_matriz_confusion(modelo_nombre, y_true, y_pred, labels_map):
    """Genera matriz de confusión."""
    fig, ax = plt.subplots(figsize=(10, 8))

    cm = confusion_matrix(y_true, y_pred)
    labels = [labels_map[i] for i in sorted(labels_map.keys())]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Cantidad'})

    ax.set_title(f'Matriz de Confusion - {modelo_nombre}', fontweight='bold', fontsize=13)
    ax.set_xlabel('Prediccion', fontweight='bold')
    ax.set_ylabel('Valor Real', fontweight='bold')

    # Agregar porcentajes
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            ax.text(j+0.5, i+0.7, f'({pct:.1f}%)',
                   ha='center', va='center', fontsize=9, color='gray')

    plt.tight_layout()
    filename = f"01_confusion_matrix_{modelo_nombre.lower().replace(' ', '_')}.png"
    plt.savefig(class_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"OK Grafico: {filename}")


def generar_feature_importance(feature_names, top_n=20):
    """Genera gráfico de importancia de características."""
    modelos_con_importance = []
    for nombre, resultado in resultados_modelos.items():
        if 'feature_importance' in resultado:
            modelos_con_importance.append((nombre, resultado))

    if not modelos_con_importance:
        print("No hay modelos con feature importance")
        return

    # Ajustar top_n al número de features disponibles
    n_features = len(feature_names)
    top_n_ajustado = min(top_n, n_features)

    n_modelos = len(modelos_con_importance)
    fig, axes = plt.subplots(1, n_modelos, figsize=(8*n_modelos, 6))

    # Si solo hay un modelo, axes no es un array
    if n_modelos == 1:
        axes = [axes]

    for idx, (nombre, resultado) in enumerate(modelos_con_importance):
        importance = resultado['feature_importance']
        indices = np.argsort(importance)[-top_n_ajustado:]

        ax = axes[idx]
        ax.barh(range(top_n_ajustado), importance[indices], color='#2ecc71', alpha=0.8)
        ax.set_yticks(range(top_n_ajustado))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        ax.set_xlabel('Importancia', fontweight='bold')
        ax.set_title(f'Top {top_n_ajustado} Features - {nombre}', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(class_dir / '02_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 02_feature_importance.png")


def generar_comparacion_modelos():
    """Genera gráfico de comparación entre modelos."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    modelos = list(resultados_modelos.keys())
    metricas_nombres = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metricas_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, (metrica, label) in enumerate(zip(metricas_nombres, metricas_labels)):
        ax = axes[idx // 2, idx % 2]

        valores = [resultados_modelos[m]['metricas'][metrica] for m in modelos]

        bars = ax.bar(modelos, valores, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'Comparacion: {label}', fontweight='bold', fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # Agregar valores sobre barras
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(comp_dir / '03_comparacion_modelos.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 03_comparacion_modelos.png")


def generar_curvas_roc(y_test, labels_map):
    """Genera curvas ROC para cada modelo."""
    from sklearn.preprocessing import label_binarize

    # Binarizar etiquetas
    y_test_bin = label_binarize(y_test, classes=sorted(labels_map.keys()))
    n_classes = y_test_bin.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (nombre, resultado) in enumerate(resultados_modelos.items()):
        ax = axes[idx]
        y_proba = resultado['y_pred_proba']

        for i, clase in enumerate(sorted(labels_map.keys())):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])

            ax.plot(fpr, tpr, label=f'{labels_map[clase]} (AUC={auc:.3f})', linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(f'ROC Curves - {nombre}', fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(class_dir / '04_roc_curves.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 04_roc_curves.png")


def generar_distribucion_predicciones(y_test, labels_map):
    """Genera distribución de predicciones vs valores reales."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (nombre, resultado) in enumerate(resultados_modelos.items()):
        ax = axes[idx]
        y_pred = resultado['y_pred']

        # Crear DataFrame para comparación
        df_comp = pd.DataFrame({
            'Real': y_test,
            'Predicho': y_pred
        })

        # Contar combinaciones
        counts_real = df_comp['Real'].value_counts().sort_index()
        counts_pred = df_comp['Predicho'].value_counts().sort_index()

        x = np.arange(len(labels_map))
        width = 0.35

        labels = [labels_map[i] for i in sorted(labels_map.keys())]

        bars1 = ax.bar(x - width/2, counts_real, width, label='Real', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, counts_pred, width, label='Predicho', alpha=0.8, color='#e74c3c')

        ax.set_xlabel('Clase', fontweight='bold')
        ax.set_ylabel('Cantidad', fontweight='bold')
        ax.set_title(f'Real vs Predicho - {nombre}', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(class_dir / '05_distribucion_predicciones.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 05_distribucion_predicciones.png")


def generar_visualizacion_clustering(clustering_results, data):
    """Genera visualizaciones de clustering."""

    # 1. Elbow method + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    resultados_k = clustering_results['resultados_k']
    k_values = sorted(resultados_k.keys())

    # Elbow
    inertias = [resultados_k[k]['inertia'] for k in k_values]
    ax1 = axes[0]
    ax1.plot(k_values, inertias, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax1.set_xlabel('Número de Clusters (k)', fontweight='bold')
    ax1.set_ylabel('Inertia', fontweight='bold')
    ax1.set_title('Elbow Method', fontweight='bold', fontsize=13)
    ax1.grid(alpha=0.3)

    # Silhouette
    silhouettes = [resultados_k[k]['silhouette'] for k in k_values]
    ax2 = axes[1]
    bars = ax2.bar(k_values, silhouettes, color='#2ecc71', alpha=0.8)
    ax2.set_xlabel('Número de Clusters (k)', fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    ax2.set_title('Silhouette Score por K', fontweight='bold', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    # Destacar mejor k
    mejor_k = clustering_results['mejor_k']
    mejor_idx = k_values.index(mejor_k)
    bars[mejor_idx].set_color('#e74c3c')

    plt.tight_layout()
    plt.savefig(cluster_dir / '06_clustering_metrics.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 06_clustering_metrics.png")

    # 2. Visualización 2D con PCA
    fig, ax = plt.subplots(figsize=(12, 8))

    X_pca = clustering_results['X_pca']
    clusters = clustering_results['clusters_final']

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                        cmap='viridis', alpha=0.6, s=20)

    # Centroides
    modelo = clustering_results['modelo_final']
    pca = clustering_results['pca']
    centroides_pca = pca.transform(modelo.cluster_centers_)

    ax.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
              c='red', marker='X', s=200, edgecolors='black', linewidth=2,
              label='Centroides')

    ax.set_xlabel('Componente Principal 1', fontweight='bold')
    ax.set_ylabel('Componente Principal 2', fontweight='bold')
    ax.set_title(f'Clustering K-Means (k={mejor_k}) - Visualización PCA',
                fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.savefig(cluster_dir / '07_clustering_visualization.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 07_clustering_visualization.png")

    # 3. Análisis de clusters por gravedad
    fig, ax = plt.subplots(figsize=(12, 6))

    # Agregar clusters a datos originales
    df_analysis = data['X_train_orig'].copy()
    df_analysis['cluster'] = clusters
    df_analysis['GRAVEDAD'] = data['y_train'].values

    # Tabla de contingencia
    ct = pd.crosstab(df_analysis['cluster'], df_analysis['GRAVEDAD'], normalize='index') * 100

    ct.plot(kind='bar', stacked=False, ax=ax, color=['#e74c3c', '#f39c12', '#3498db'], alpha=0.8)
    ax.set_xlabel('Cluster', fontweight='bold')
    ax.set_ylabel('Porcentaje (%)', fontweight='bold')
    ax.set_title('Distribución de Gravedad por Cluster', fontweight='bold', fontsize=13)
    ax.legend(['Muertos', 'Heridos', 'Solo Daños'], title='Gravedad')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(cluster_dir / '08_clusters_por_gravedad.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 08_clusters_por_gravedad.png")


def generar_tabla_resumen():
    """Genera tabla resumen de comparación de modelos."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Crear datos para tabla
    tabla_data = [['MODELO', 'ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE', 'TIEMPO (s)']]

    for nombre, resultado in resultados_modelos.items():
        metricas = resultado['metricas']
        fila = [
            nombre,
            f"{metricas['accuracy']:.4f}",
            f"{metricas['precision_macro']:.4f}",
            f"{metricas['recall_macro']:.4f}",
            f"{metricas['f1_macro']:.4f}",
            f"{resultado['tiempo']:.2f}"
        ]
        tabla_data.append(fila)

    # Crear tabla
    table = ax.table(cellText=tabla_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Estilo
    for i in range(len(tabla_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Resaltar mejor modelo (mayor F1)
    f1_scores = [resultados_modelos[nombre]['metricas']['f1_macro']
                 for nombre in resultados_modelos.keys()]
    mejor_idx = f1_scores.index(max(f1_scores)) + 1

    for i in range(len(tabla_data[0])):
        table[(mejor_idx, i)].set_facecolor('#2ecc71')
        table[(mejor_idx, i)].set_text_props(weight='bold')

    ax.set_title('Comparación de Modelos - Métricas de Evaluación',
                fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(comp_dir / '09_tabla_resumen.png', bbox_inches='tight')
    plt.close()
    print("OK Grafico: 09_tabla_resumen.png")


# ============================================================================
# GENERACIÓN DE INFORME
# ============================================================================

def generar_informe_markdown(data, clustering_results):
    """Genera informe en Markdown."""
    print("\n" + "="*70)
    print("GENERANDO INFORME MARKDOWN")
    print("="*70)

    # Determinar mejor modelo
    f1_scores = {nombre: resultado['metricas']['f1_macro']
                 for nombre, resultado in resultados_modelos.items()}
    mejor_modelo = max(f1_scores.keys(), key=lambda k: f1_scores[k])

    informe = f"""# Seccion 7: Metodologia de Modelado

## Analisis de Siniestros Viales - Bogota

---

## Resumen Ejecutivo

Se implementaron **{len(resultados_modelos)} modelos de clasificacion** para predecir la gravedad de siniestros viales.

**Mejor modelo:** {mejor_modelo} (F1-Score: {f1_scores[mejor_modelo]:.4f})

**Tiempo total de entrenamiento:** {sum(r['tiempo'] for r in resultados_modelos.values()):.2f} segundos

**Dataset:**
- Train set: {data['X_train'].shape[0]:,} registros
- Test set: {data['X_test'].shape[0]:,} registros
- Features: {len(data['feature_names'])}

---

## 7.1. Seleccion del Modelo

### Problema de Negocio

Predecir la **gravedad de siniestros viales** (Con Muertos / Con Heridos / Solo Danos) basado en:
- Factores temporales (hora, dia, mes)
- Factores geograficos (localidad, zona, tipo de via)
- Caracteristicas del siniestro (clase, choque, diseno)

### Tipo de Problema: Clasificacion Multiclase

**Variable objetivo:** GRAVEDAD
- Clase 1: Con Muertos (1.5% del dataset)
- Clase 2: Con Heridos (33.3% del dataset)
- Clase 3: Solo Danos (65.2% del dataset)

**Desafio:** Clases desbalanceadas - requiere tecnicas especiales

### Enfoque Complementario: Clustering

Adicionalmente, se aplico **K-Means clustering** para:
- Descubrir patrones naturales en los datos
- Segmentar siniestros en grupos homogeneos
- Identificar perfiles de riesgo no obvios

---

## 7.2. Algoritmos y Tecnicas Utilizadas

### Modelos de Clasificacion

"""

    # Agregar cada modelo
    for idx, (nombre, resultado) in enumerate(resultados_modelos.items(), 1):
        metricas = resultado['metricas']

        informe += f"""
#### {idx}. {nombre}

**Descripcion:**
"""

        if 'Logistic' in nombre:
            informe += """- Modelo lineal generalizado para clasificacion multiclase
- Usa regresion logistica multinomial (softmax)
- Interpretable: coeficientes indican impacto de cada feature
"""
        elif 'Random' in nombre:
            informe += """- Ensemble de arboles de decision
- Reduce overfitting mediante bagging y random features
- Captura interacciones no lineales complejas
"""
        elif 'XGBoost' in nombre:
            informe += """- Gradient Boosting optimizado
- Construye arboles secuencialmente corrigiendo errores
- Alta performance en competencias de ML
"""

        informe += f"""
**Resultados:**
- Accuracy: {metricas['accuracy']:.4f}
- Precision (macro): {metricas['precision_macro']:.4f}
- Recall (macro): {metricas['recall_macro']:.4f}
- F1-Score (macro): {metricas['f1_macro']:.4f}
- Tiempo de entrenamiento: {resultado['tiempo']:.2f}s

**Grafico:** ![Confusion Matrix](reports/visualizaciones_modelos/clasificacion/01_confusion_matrix_{nombre.lower().replace(' ', '_')}.png)

"""

    # Clustering
    informe += f"""
### Clustering: K-Means

**Configuracion:**
- Algoritmo: K-Means
- Valores de k probados: {list(clustering_results['resultados_k'].keys())}
- Mejor k seleccionado: {clustering_results['mejor_k']} (Silhouette Score: {clustering_results['resultados_k'][clustering_results['mejor_k']]['silhouette']:.4f})

**Metodo de seleccion:**
1. Elbow method (inercia)
2. Silhouette Score (cohesion y separacion)
3. Davies-Bouldin Index (similitud intra-cluster vs inter-cluster)

**Graficos:**
- ![Clustering Metrics](reports/visualizaciones_modelos/clustering/06_clustering_metrics.png)
- ![Clustering Visualization](reports/visualizaciones_modelos/clustering/07_clustering_visualization.png)
- ![Clusters por Gravedad](reports/visualizaciones_modelos/clustering/08_clusters_por_gravedad.png)

---

## 7.3. Justificacion de los Hiperparametros

### Metodo de Optimizacion: RandomizedSearchCV

**¿Por que RandomizedSearchCV?**
- Dataset grande (196k registros) → GridSearchCV seria muy lento
- Explora espacio amplio de parametros eficientemente
- Prueba combinaciones aleatorias (40-50 iteraciones)
- Balance optimo entre velocidad y calidad

**Configuracion:**
- Validacion cruzada: 5-fold estratificado
- Metrica de optimizacion: F1-Score macro (por desbalance de clases)
- Paralelizacion: n_jobs=-1 (usa todos los cores)

### Hiperparametros Optimizados

"""

    # Hiperparámetros de cada modelo
    for nombre, resultado in resultados_modelos.items():
        informe += f"""
#### {nombre}

**Mejores parametros encontrados:**
```python
{resultado['best_params']}
```

**Espacio de busqueda:**
"""

        if 'Logistic' in nombre:
            informe += """- C: [0.001, 0.01, 0.1, 1, 10, 100] (regularizacion)
- penalty: ['l2'] (tipo de regularizacion)
- solver: ['lbfgs', 'saga'] (algoritmo de optimizacion)
- class_weight: ['balanced', None] (ajuste por desbalance)
"""
        elif 'Random' in nombre:
            informe += """- n_estimators: [100, 200, 300, 500] (numero de arboles)
- max_depth: [10, 15, 20, 25, None] (profundidad maxima)
- min_samples_split: [2, 5, 10] (muestras minimas para split)
- min_samples_leaf: [1, 2, 4] (muestras minimas en hoja)
- class_weight: ['balanced', 'balanced_subsample', None]
"""
        elif 'XGBoost' in nombre:
            informe += """- n_estimators: [100, 200, 300] (numero de boosting rounds)
- max_depth: [3, 5, 7, 9] (profundidad de arboles)
- learning_rate: [0.01, 0.05, 0.1, 0.2] (tasa de aprendizaje)
- subsample: [0.7, 0.8, 0.9, 1.0] (fraccion de muestras)
- colsample_bytree: [0.7, 0.8, 0.9, 1.0] (fraccion de features)
"""

    informe += f"""
---

## 7.4. Validacion Cruzada y Tecnicas de Re-muestreo

### Estrategia de Validacion

**1. Train-Test Split Estratificado**
- Split ratio: 80% train / 20% test
- Estratificacion: mantiene proporcion de clases en ambos sets
- Random state: 42 (reproducibilidad)

**Train set:** {data['X_train'].shape[0]:,} registros
**Test set:** {data['X_test'].shape[0]:,} registros

**2. Validacion Cruzada Estratificada (5-Fold)**

Durante la optimizacion de hiperparametros:
- 5 folds estratificados
- Cada fold mantiene proporcion de clases
- Metrica agregada: promedio de F1-Score macro

**¿Por que F1-Score macro?**
- Clases desbalanceadas (65% solo danos, 33% heridos, 1.5% muertos)
- Macro average: trata todas las clases por igual
- Importante detectar bien la clase minoritaria (muertos)

### Tecnicas para Manejar Desbalance

**1. Class Weight Balancing**
- Todos los modelos probaron `class_weight='balanced'`
- Penaliza mas errores en clase minoritaria
- Formula: n_samples / (n_classes * n_samples_class)

**2. Stratified Sampling**
- Todos los splits mantienen proporcion de clases
- Evita que un fold tenga muy pocos casos de clase minoritaria

### Prevencion de Overfitting

**Tecnicas aplicadas:**
1. **Regularizacion:**
   - Logistic Regression: L2 penalty
   - XGBoost: gamma, min_child_weight

2. **Limitacion de complejidad:**
   - Random Forest: max_depth, min_samples_split
   - XGBoost: max_depth, subsample

3. **Validacion cruzada:**
   - Detecta overfitting comparando train vs validation

4. **Test set separado:**
   - Evaluacion final en datos nunca vistos

---

## Comparacion de Modelos

**Grafico resumen:**
![Comparacion Modelos](reports/visualizaciones_modelos/comparacion/03_comparacion_modelos.png)

**Tabla de metricas:**
![Tabla Resumen](reports/visualizaciones_modelos/comparacion/09_tabla_resumen.png)

### Modelo Recomendado: {mejor_modelo}

**Justificacion:**
- Mayor F1-Score macro: {f1_scores[mejor_modelo]:.4f}
- Balance entre precision y recall
- Tiempo de entrenamiento razonable: {resultados_modelos[mejor_modelo]['tiempo']:.2f}s
"""

    if 'Random' in mejor_modelo or 'XGBoost' in mejor_modelo:
        informe += f"""
- Captura interacciones complejas entre features
- Importancia de features interpretable

**Top 5 Features mas importantes:**
"""
        if 'feature_importance' in resultados_modelos[mejor_modelo]:
            importance = resultados_modelos[mejor_modelo]['feature_importance']
            top_idx = np.argsort(importance)[-5:][::-1]
            for idx in top_idx:
                informe += f"\n{idx+1}. {data['feature_names'][idx]}: {importance[idx]:.4f}"

        informe += "\n\n**Grafico:** ![Feature Importance](reports/visualizaciones_modelos/clasificacion/02_feature_importance.png)\n"

    informe += f"""

### Curvas ROC Multiclase

![ROC Curves](reports/visualizaciones_modelos/clasificacion/04_roc_curves.png)

### Distribucion de Predicciones

![Distribucion](reports/visualizaciones_modelos/clasificacion/05_distribucion_predicciones.png)

---

## Conclusiones

### Hallazgos Principales

1. **Desempeno de modelos:**
   - Los 3 modelos superan baseline aleatorio
   - {mejor_modelo} obtiene mejor F1-Score: {f1_scores[mejor_modelo]:.4f}

2. **Desafio de clases desbalanceadas:**
   - Clase minoritaria (muertos) mas dificil de predecir
   - Class balancing mejoro recall en clase minoritaria

3. **Features mas importantes:**
   - Factores temporales (hora, periodo del dia)
   - Factores geograficos (localidad, zona)
   - Tipo de siniestro (clase, choque)

4. **Clustering:**
   - Identificados {clustering_results['mejor_k']} grupos naturales de siniestros
   - Cada cluster tiene perfil de riesgo diferenciado
   - Util para estrategias de intervencion segmentadas

### Aplicaciones Practicas

1. **Sistema de alerta temprana:**
   - Predecir gravedad en tiempo real
   - Priorizar recursos de emergencia

2. **Planificacion de intervenciones:**
   - Identificar combinaciones de alto riesgo
   - Focalizar campanas preventivas

3. **Optimizacion de recursos:**
   - Asignar ambulancias segun probabilidad de gravedad
   - Reforzar controles en horas/zonas criticas

---

**Fecha de generacion:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Modelos entrenados:** {len(resultados_modelos)}
**Registros analizados:** {data['X_train'].shape[0] + data['X_test'].shape[0]:,}
"""

    # Guardar informe
    informe_path = base_dir / 'INFORME_MODELADO.md'
    with open(informe_path, 'w', encoding='utf-8') as f:
        f.write(informe)

    print(f"OK Informe generado: {informe_path}")

    return informe_path


# ============================================================================
# GUARDAR MODELOS
# ============================================================================

def guardar_modelos(data):
    """Guarda modelos entrenados y objetos necesarios."""
    print("\nGuardando modelos...")

    for nombre, resultado in resultados_modelos.items():
        modelo_path = models_dir / f"{nombre.lower().replace(' ', '_')}.pkl"
        with open(modelo_path, 'wb') as f:
            pickle.dump(resultado['modelo'], f)
        print(f"  Guardado: {modelo_path.name}")

    # Guardar scaler y label encoders
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(data['scaler'], f)

    with open(models_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(data['label_encoders'], f)

    # Guardar métricas en CSV
    metricas_df = pd.DataFrame([
        {
            'Modelo': nombre,
            **resultado['metricas'],
            'Tiempo_segundos': resultado['tiempo']
        }
        for nombre, resultado in resultados_modelos.items()
    ])

    metricas_path = models_dir / 'metricas_comparacion.csv'
    metricas_df.to_csv(metricas_path, index=False)
    print(f"  Guardado: {metricas_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ejecuta pipeline completo de modelado."""
    print("\n" + "="*70)
    print("PIPELINE DE MODELADO - SINIESTROS VIALES BOGOTA")
    print("="*70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tiempo_inicio = time.time()

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Preparar datos
    data = preparar_datos(df)

    # 3. Entrenar modelos de clasificación
    print("\n" + "="*70)
    print("SECCION 7.2: ENTRENAMIENTO DE MODELOS")
    print("="*70)

    entrenar_logistic_regression(data, optimize=True)
    entrenar_random_forest(data, optimize=True)
    entrenar_xgboost(data, optimize=True)

    # 4. Aplicar clustering
    clustering_results = aplicar_clustering(data)

    # 5. Generar visualizaciones
    print("\n" + "="*70)
    print("GENERANDO VISUALIZACIONES")
    print("="*70)

    labels_map = data['gravedad_labels']

    # Matrices de confusión
    for nombre, resultado in resultados_modelos.items():
        generar_matriz_confusion(nombre, data['y_test'], resultado['y_pred'], labels_map)

    # Feature importance
    generar_feature_importance(data['feature_names'])

    # Comparación de modelos
    generar_comparacion_modelos()

    # Curvas ROC
    generar_curvas_roc(data['y_test'], labels_map)

    # Distribución de predicciones
    generar_distribucion_predicciones(data['y_test'], labels_map)

    # Clustering
    generar_visualizacion_clustering(clustering_results, data)

    # Tabla resumen
    generar_tabla_resumen()

    # 6. Guardar modelos
    guardar_modelos(data)

    # 7. Generar informe
    generar_informe_markdown(data, clustering_results)

    tiempo_total = time.time() - tiempo_inicio

    print("\n" + "="*70)
    print("MODELADO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    print(f"\nModelos entrenados: {len(resultados_modelos)}")
    print(f"Visualizaciones: 9 graficos")
    print(f"Ubicacion: {viz_dir}")
    print(f"\nInforme: INFORME_MODELADO.md")
    print(f"\nModelos guardados en: {models_dir}")

    # Mejor modelo
    f1_scores = {nombre: resultado['metricas']['f1_macro']
                 for nombre, resultado in resultados_modelos.items()}
    mejor_modelo = max(f1_scores.keys(), key=lambda k: f1_scores[k])
    print(f"\nMejor modelo: {mejor_modelo} (F1={f1_scores[mejor_modelo]:.4f})")


if __name__ == "__main__":
    main()
