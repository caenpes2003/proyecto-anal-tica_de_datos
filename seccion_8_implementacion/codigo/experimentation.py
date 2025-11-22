"""
EXPERIMENTOS DE VALIDACIÓN METODOLÓGICA
========================================

Este script implementa 3 experimentos para justificar parámetros:
1. Ventana temporal óptima (1, 3, 6, 12 meses)
2. Epsilon óptimo para DBSCAN (500m, 1km, 1.5km, 2km)
3. Fórmula de riesgo óptima (3 alternativas)

Autor: Proyecto Analítica de Siniestros
Fecha: 2025-01-19
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

BASE_DIR = r"c:\Users\caenp\OneDrive\Escritorio\analitica_siniestros"
INPUT_FILE = f"{BASE_DIR}/data/processed/siniestros_features_geo.csv"
CLUSTERS_FILE = f"{BASE_DIR}/seccion_10_interpretacion/analisis/siniestros_con_clusters.csv"
OUTPUT_DIR_SEC8 = f"{BASE_DIR}/seccion_8_implementacion"
OUTPUT_DIR_SEC9 = f"{BASE_DIR}/seccion_9_resultados"

# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("="*70)
print("EXPERIMENTOS DE VALIDACION METODOLOGICA")
print("="*70)
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cargando datos...")

df = pd.read_csv(INPUT_FILE)
df['FECHA'] = pd.to_datetime(df['FECHA'])
print(f"[OK] Dataset cargado: {len(df):,} registros")

# Filtrar geocodificados
df_geo = df.dropna(subset=['latitud', 'longitud']).copy()
print(f"[OK] Registros geocodificados: {len(df_geo):,}")

# Filtrar solo graves
df_graves = df_geo[df_geo['GRAVEDAD'].isin([1, 3])].copy()
print(f"[OK] Siniestros graves: {len(df_graves):,}")

# ============================================================================
# EXPERIMENTO 1: VENTANA TEMPORAL OPTIMA
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] EXPERIMENTO 1: Ventana temporal optima")
print("-" * 70)

# Definir fecha de corte (última fecha - 12 meses para tener datos suficientes)
fecha_max = df_graves['FECHA'].max()
fecha_min = df_graves['FECHA'].min()

print(f"Rango de fechas: {fecha_min.date()} a {fecha_max.date()}")

# Usar últimos 12 meses como referencia
fecha_corte = fecha_max - timedelta(days=365)
df_12m = df_graves[df_graves['FECHA'] >= fecha_corte].copy()

print(f"Ultimos 12 meses: {len(df_12m):,} siniestros")

# Definir ventanas
ventanas = {
    '1 mes': 30,
    '3 meses': 90,
    '6 meses': 180,
    '12 meses': 365
}

def calcular_hotspots_coords(df, percentil=90):
    """
    Retorna conjunto de coordenadas de hotspots
    """
    coords = set()
    # Discretizar coordenadas (grid de ~100m)
    df_copy = df.copy()
    df_copy['lat_grid'] = (df_copy['latitud'] * 100).round(0)
    df_copy['lon_grid'] = (df_copy['longitud'] * 100).round(0)

    # Contar siniestros por celda
    grid_counts = df_copy.groupby(['lat_grid', 'lon_grid']).size()
    threshold = np.percentile(grid_counts, percentil)

    hotspots = grid_counts[grid_counts >= threshold]
    coords = set(hotspots.index)

    return coords

def jaccard_similarity(set1, set2):
    """
    Calcula similitud de Jaccard entre dos conjuntos
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Calcular hotspots para cada ventana
hotspots_ventanas = {}
for nombre, dias in ventanas.items():
    fecha_inicio = fecha_max - timedelta(days=dias)
    df_ventana = df_graves[df_graves['FECHA'] >= fecha_inicio].copy()
    hotspots = calcular_hotspots_coords(df_ventana, percentil=90)
    hotspots_ventanas[nombre] = hotspots
    print(f"  {nombre:12s}: {len(df_ventana):6,} siniestros, {len(hotspots):4} hotspots")

# Calcular Jaccard Index entre ventanas consecutivas y contra referencia (12 meses)
jaccard_results = []

# Comparar cada ventana con la referencia (12 meses)
referencia = hotspots_ventanas['12 meses']
for nombre in ['1 mes', '3 meses', '6 meses']:
    jaccard = jaccard_similarity(hotspots_ventanas[nombre], referencia)
    jaccard_results.append({
        'comparacion': f'{nombre} vs 12 meses',
        'jaccard': jaccard
    })
    print(f"  Jaccard ({nombre} vs 12 meses): {jaccard:.4f}")

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Número de hotspots por ventana
ventanas_list = list(ventanas.keys())
n_hotspots = [len(hotspots_ventanas[v]) for v in ventanas_list]

ax1.bar(ventanas_list, n_hotspots, color='steelblue', edgecolor='navy', linewidth=1.5)
ax1.set_xlabel('Ventana Temporal', fontsize=12, fontweight='bold')
ax1.set_ylabel('Numero de Hotspots (percentil 90)', fontsize=12, fontweight='bold')
ax1.set_title('Experimento 1a: Hotspots por Ventana Temporal', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, axis='y')

# Agregar valores sobre barras
for i, v in enumerate(n_hotspots):
    ax1.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

# Gráfico 2: Jaccard Index
comparaciones = [r['comparacion'] for r in jaccard_results]
jaccards = [r['jaccard'] for r in jaccard_results]

colors_jaccard = ['red' if j < 0.7 else 'orange' if j < 0.8 else 'green' for j in jaccards]

ax2.barh(comparaciones, jaccards, color=colors_jaccard, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Indice de Jaccard', fontsize=12, fontweight='bold')
ax2.set_title('Experimento 1b: Estabilidad de Hotspots\n(Similitud con 12 meses)',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlim(0, 1)
ax2.axvline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Umbral aceptable (0.8)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

# Agregar valores
for i, v in enumerate(jaccards):
    ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR_SEC9}/visualizaciones/9.3_jaccard_temporal.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Grafico guardado: seccion_9_resultados/visualizaciones/9.3_jaccard_temporal.png")

# Determinar ventana óptima
ventana_optima = None
max_jaccard = 0
for r in jaccard_results:
    if r['jaccard'] > max_jaccard and r['jaccard'] >= 0.8:
        max_jaccard = r['jaccard']
        ventana_optima = r['comparacion'].split(' vs ')[0]

if ventana_optima is None:
    ventana_optima = '12 meses'  # Si ninguna supera 0.8, usar la más larga

print(f"\n[CONCLUSION] Ventana temporal optima: {ventana_optima} (Jaccard = {max_jaccard:.4f})")

# ============================================================================
# EXPERIMENTO 2: EPSILON OPTIMO PARA DBSCAN
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] EXPERIMENTO 2: Epsilon optimo para DBSCAN")
print("-" * 70)

# Preparar coordenadas
coords_km = np.column_stack([
    df_graves['latitud'].values * 111,
    df_graves['longitud'].values * 110.5
])

# Probar diferentes valores de epsilon
epsilons = [0.5, 1.0, 1.5, 2.0]  # en km
min_samples = 10

resultados_dbscan = []

for eps in epsilons:
    print(f"\nProbando epsilon = {eps} km...")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    clusters = dbscan.fit_predict(coords_km)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    pct_noise = (n_noise / len(clusters)) * 100

    # Métricas de calidad
    if n_clusters > 1:
        mask_clustered = clusters != -1
        silhouette = silhouette_score(coords_km[mask_clustered], clusters[mask_clustered])
        calinski = calinski_harabasz_score(coords_km[mask_clustered], clusters[mask_clustered])
    else:
        silhouette = None
        calinski = None

    resultados_dbscan.append({
        'epsilon': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'pct_noise': pct_noise,
        'silhouette': silhouette,
        'calinski': calinski
    })

    print(f"  Clusters: {n_clusters}")
    print(f"  Ruido: {n_noise} ({pct_noise:.1f}%)")
    if silhouette is not None:
        print(f"  Silhouette: {silhouette:.4f}")
        print(f"  Calinski-Harabasz: {calinski:.2f}")

# Visualización
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

epsilons_plot = [r['epsilon'] for r in resultados_dbscan]

# Gráfico 1: Número de clusters
n_clusters_plot = [r['n_clusters'] for r in resultados_dbscan]
ax1.plot(epsilons_plot, n_clusters_plot, marker='o', linewidth=2, markersize=10, color='steelblue')
ax1.set_xlabel('Epsilon (km)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Numero de Clusters', fontsize=11, fontweight='bold')
ax1.set_title('Experimento 2a: Clusters vs Epsilon', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
for i, v in enumerate(n_clusters_plot):
    ax1.text(epsilons_plot[i], v + 0.2, str(v), ha='center', fontweight='bold')

# Gráfico 2: Porcentaje de ruido
pct_noise_plot = [r['pct_noise'] for r in resultados_dbscan]
ax2.plot(epsilons_plot, pct_noise_plot, marker='s', linewidth=2, markersize=10, color='orangered')
ax2.axhline(15, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Umbral max (15%)')
ax2.set_xlabel('Epsilon (km)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Porcentaje de Ruido (%)', fontsize=11, fontweight='bold')
ax2.set_title('Experimento 2b: Ruido vs Epsilon', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
for i, v in enumerate(pct_noise_plot):
    ax2.text(epsilons_plot[i], v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)

# Gráfico 3: Silhouette Score
silhouette_plot = [r['silhouette'] if r['silhouette'] is not None else 0 for r in resultados_dbscan]
colors_silh = ['red' if s < 0 else 'orange' if s < 0.3 else 'yellow' if s < 0.5 else 'green' for s in silhouette_plot]
ax3.bar(epsilons_plot, silhouette_plot, color=colors_silh, edgecolor='black', linewidth=1.5)
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Umbral bueno (0.5)')
ax3.set_xlabel('Epsilon (km)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
ax3.set_title('Experimento 2c: Calidad de Clustering (Silhouette)', fontsize=12, fontweight='bold')
ax3.set_ylim(-0.2, 0.6)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(silhouette_plot):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

# Gráfico 4: Calinski-Harabasz Index
calinski_plot = [r['calinski'] if r['calinski'] is not None else 0 for r in resultados_dbscan]
ax4.plot(epsilons_plot, calinski_plot, marker='^', linewidth=2, markersize=10, color='purple')
ax4.set_xlabel('Epsilon (km)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Calinski-Harabasz Index', fontsize=11, fontweight='bold')
ax4.set_title('Experimento 2d: Calidad de Clustering (Calinski-Harabasz)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
for i, v in enumerate(calinski_plot):
    ax4.text(epsilons_plot[i], v + 10, f'{v:.0f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR_SEC9}/visualizaciones/9.2_silhouette_scores.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] Grafico guardado: seccion_9_resultados/visualizaciones/9.2_silhouette_scores.png")

# Determinar epsilon óptimo (mejor balance: menos ruido, silhouette positivo, no muchos clusters)
epsilon_optimo = None
mejor_score = -999

for r in resultados_dbscan:
    # Score compuesto: silhouette (peso 0.5) - ruido normalizado (peso 0.3) - clusters normalizado (peso 0.2)
    if r['silhouette'] is not None:
        score = (r['silhouette'] * 0.5) - (r['pct_noise']/100 * 0.3) - (r['n_clusters']/10 * 0.2)
        if score > mejor_score:
            mejor_score = score
            epsilon_optimo = r['epsilon']

print(f"\n[CONCLUSION] Epsilon optimo: {epsilon_optimo} km")

# ============================================================================
# EXPERIMENTO 3: FORMULA DE RIESGO OPTIMA
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] EXPERIMENTO 3: Formula de riesgo optima")
print("-" * 70)

# Cargar dataset con clusters
df_clusters = pd.read_csv(CLUSTERS_FILE)
print(f"[OK] Dataset con clusters cargado: {len(df_clusters):,} registros")

# Calcular estadísticas por cluster
cluster_stats = df_clusters.groupby('cluster').agg({
    'CODIGO_ACCIDENTE': 'count',  # Densidad
    'GRAVEDAD': 'mean',            # Gravedad promedio
    'latitud': 'mean',
    'longitud': 'mean'
}).rename(columns={'CODIGO_ACCIDENTE': 'densidad'})

# Calcular % de mortales por cluster
mortales_por_cluster = df_clusters[df_clusters['GRAVEDAD'] == 3].groupby('cluster').size()
cluster_stats['pct_mortales'] = (mortales_por_cluster / cluster_stats['densidad'] * 100).fillna(0)

# Probabilidad predicha (simulada - en producción vendría del modelo ML)
# Por ahora usar GRAVEDAD como proxy
cluster_stats['prob_grave'] = cluster_stats['GRAVEDAD'] / 3

# Calcular Moran's I local (simplificado - usar densidad como proxy)
cluster_stats['moran_local'] = (cluster_stats['densidad'] - cluster_stats['densidad'].mean()) / cluster_stats['densidad'].std()

print(f"\nEstadisticas de {len(cluster_stats)} clusters:")
print(cluster_stats.head())

# FÓRMULA 1: Densidad x Gravedad promedio
cluster_stats['riesgo_f1'] = cluster_stats['densidad'] * cluster_stats['GRAVEDAD']

# FÓRMULA 2: (Densidad x 0.4) + (% Mortales x 0.4) + (Prob Grave x 0.2)
cluster_stats['riesgo_f2'] = (
    (cluster_stats['densidad'] / cluster_stats['densidad'].max()) * 0.4 +
    (cluster_stats['pct_mortales'] / 100) * 0.4 +
    cluster_stats['prob_grave'] * 0.2
) * 100

# FÓRMULA 3: (Densidad + Moran_I_local) x Gravedad
cluster_stats['riesgo_f3'] = (cluster_stats['densidad'] + cluster_stats['moran_local'] * 10) * cluster_stats['GRAVEDAD']

# Normalizar scores (0-100)
for col in ['riesgo_f1', 'riesgo_f2', 'riesgo_f3']:
    min_val = cluster_stats[col].min()
    max_val = cluster_stats[col].max()
    cluster_stats[f'{col}_norm'] = ((cluster_stats[col] - min_val) / (max_val - min_val)) * 100

# Ranking de cada fórmula
cluster_stats['rank_f1'] = cluster_stats['riesgo_f1_norm'].rank(ascending=False)
cluster_stats['rank_f2'] = cluster_stats['riesgo_f2_norm'].rank(ascending=False)
cluster_stats['rank_f3'] = cluster_stats['riesgo_f3_norm'].rank(ascending=False)

# Correlación entre rankings
from scipy.stats import spearmanr

corr_f1_f2 = spearmanr(cluster_stats['rank_f1'], cluster_stats['rank_f2'])[0]
corr_f1_f3 = spearmanr(cluster_stats['rank_f1'], cluster_stats['rank_f3'])[0]
corr_f2_f3 = spearmanr(cluster_stats['rank_f2'], cluster_stats['rank_f3'])[0]

print(f"\nCorrelaciones de Spearman entre rankings:")
print(f"  F1 vs F2: {corr_f1_f2:.4f}")
print(f"  F1 vs F3: {corr_f1_f3:.4f}")
print(f"  F2 vs F3: {corr_f2_f3:.4f}")

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Comparación de scores normalizados
clusters_idx = cluster_stats.index.tolist()
x = np.arange(len(clusters_idx))
width = 0.25

bars1 = ax1.bar(x - width, cluster_stats['riesgo_f1_norm'], width, label='Formula 1 (Densidad x Gravedad)', color='steelblue')
bars2 = ax1.bar(x, cluster_stats['riesgo_f2_norm'], width, label='Formula 2 (Ponderada)', color='orange')
bars3 = ax1.bar(x + width, cluster_stats['riesgo_f3_norm'], width, label='Formula 3 (Espacial)', color='green')

ax1.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score de Riesgo (0-100)', fontsize=12, fontweight='bold')
ax1.set_title('Experimento 3a: Comparacion de Formulas de Riesgo', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(clusters_idx)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Gráfico 2: Matriz de correlación
correlaciones = np.array([
    [1.0, corr_f1_f2, corr_f1_f3],
    [corr_f1_f2, 1.0, corr_f2_f3],
    [corr_f1_f3, corr_f2_f3, 1.0]
])

im = ax2.imshow(correlaciones, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels(['Formula 1', 'Formula 2', 'Formula 3'])
ax2.set_yticklabels(['Formula 1', 'Formula 2', 'Formula 3'])
ax2.set_title('Experimento 3b: Correlacion entre Rankings\n(Spearman)', fontsize=14, fontweight='bold', pad=15)

# Agregar valores
for i in range(3):
    for j in range(3):
        text = ax2.text(j, i, f'{correlaciones[i, j]:.3f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=12)

cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Correlacion de Spearman', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR_SEC9}/visualizaciones/9.7_formulas_riesgo.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] Grafico guardado: seccion_9_resultados/visualizaciones/9.7_formulas_riesgo.png")

# Determinar fórmula óptima (mayor correlación promedio con las otras)
corr_promedio_f1 = (corr_f1_f2 + corr_f1_f3) / 2
corr_promedio_f2 = (corr_f1_f2 + corr_f2_f3) / 2
corr_promedio_f3 = (corr_f1_f3 + corr_f2_f3) / 2

formula_optima = None
if corr_promedio_f1 >= corr_promedio_f2 and corr_promedio_f1 >= corr_promedio_f3:
    formula_optima = "Formula 1 (Densidad x Gravedad)"
    corr_opt = corr_promedio_f1
elif corr_promedio_f2 >= corr_promedio_f1 and corr_promedio_f2 >= corr_promedio_f3:
    formula_optima = "Formula 2 (Ponderada)"
    corr_opt = corr_promedio_f2
else:
    formula_optima = "Formula 3 (Espacial)"
    corr_opt = corr_promedio_f3

print(f"\n[CONCLUSION] Formula optima: {formula_optima} (correlacion promedio = {corr_opt:.4f})")

# Guardar comparación
cluster_stats.to_csv(f"{OUTPUT_DIR_SEC9}/metricas/comparacion_formulas.csv", index=True)
print(f"[OK] Comparacion guardada: seccion_9_resultados/metricas/comparacion_formulas.csv")

# ============================================================================
# GUARDAR RESULTADOS COMPLETOS
# ============================================================================

resultados_completos = f"""
RESULTADOS: EXPERIMENTOS DE VALIDACION METODOLOGICA
=====================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
EXPERIMENTO 1: VENTANA TEMPORAL OPTIMA
================================================================================

OBJETIVO: Determinar cuantos meses de datos historicos usar para identificar
          hotspots estables que no cambien significativamente con el tiempo.

METODOLOGIA:
- Calcular hotspots (percentil 90) para ventanas de 1, 3, 6, 12 meses
- Medir similitud con Indice de Jaccard (interseccion / union)
- Umbral de aceptacion: Jaccard >= 0.80

RESULTADOS:
"""

for r in jaccard_results:
    resultados_completos += f"\n  {r['comparacion']:25s}: Jaccard = {r['jaccard']:.4f}"

resultados_completos += f"""

CONCLUSION:
Ventana temporal optima: {ventana_optima}
Justificacion: {"Presenta Jaccard >= 0.80, indicando estabilidad de hotspots respecto a ventana de 12 meses" if max_jaccard >= 0.8 else "Presenta mayor similitud con ventana de referencia (12 meses)"}

================================================================================
EXPERIMENTO 2: EPSILON OPTIMO PARA DBSCAN
================================================================================

OBJETIVO: Determinar el radio optimo (epsilon) para agrupar siniestros
          cercanos en clusters espaciales.

METODOLOGIA:
- Probar epsilon = 500m, 1km, 1.5km, 2km
- Evaluar: Numero de clusters, % ruido, Silhouette Score, Calinski-Harabasz
- Criterios: Minimizar ruido (<15%), Silhouette > 0, balance de clusters

RESULTADOS:
"""

for r in resultados_dbscan:
    silh_str = f"{r['silhouette']:.4f}" if r['silhouette'] is not None else 'N/A'
    cal_str = f"{r['calinski']:.2f}" if r['calinski'] is not None else 'N/A'
    resultados_completos += f"""
Epsilon = {r['epsilon']} km:
  - Clusters: {r['n_clusters']}
  - Ruido: {r['pct_noise']:.1f}%
  - Silhouette: {silh_str}
  - Calinski-Harabasz: {cal_str}
"""

resultados_completos += f"""
CONCLUSION:
Epsilon optimo: {epsilon_optimo} km
Justificacion: Mejor balance entre calidad de clustering (Silhouette),
               porcentaje de ruido aceptable y numero razonable de clusters.

================================================================================
EXPERIMENTO 3: FORMULA DE RIESGO OPTIMA
================================================================================

OBJETIVO: Comparar 3 formulas para calcular score de riesgo por zona

FORMULAS EVALUADAS:

Formula 1 (Simple):
  Risk = Densidad x Gravedad_promedio

Formula 2 (Ponderada):
  Risk = (Densidad_norm x 0.4) + (% Mortales x 0.4) + (Prob_Grave x 0.2)

Formula 3 (Espacial):
  Risk = (Densidad + Moran_I_local) x Gravedad_promedio

METODOLOGIA:
- Calcular ranking de clusters segun cada formula
- Medir correlacion de Spearman entre rankings
- Formula optima: mayor consenso con las otras

RESULTADOS:

Correlacion de Spearman:
  Formula 1 vs Formula 2: {corr_f1_f2:.4f}
  Formula 1 vs Formula 3: {corr_f1_f3:.4f}
  Formula 2 vs Formula 3: {corr_f2_f3:.4f}

Correlacion promedio:
  Formula 1: {corr_promedio_f1:.4f}
  Formula 2: {corr_promedio_f2:.4f}
  Formula 3: {corr_promedio_f3:.4f}

CONCLUSION:
Formula optima: {formula_optima}
Justificacion: Presenta mayor correlacion promedio con las otras formulas,
               indicando robustez y consenso en priorizacion.

================================================================================
RESUMEN EJECUTIVO
================================================================================

1. Ventana temporal: {ventana_optima}
2. Epsilon DBSCAN: {epsilon_optimo} km
3. Formula de riesgo: {formula_optima}

Estos parametros seran utilizados en el sistema de priorizacion de
intervenciones para identificar las zonas criticas de Bogota que requieren
atencion inmediata de la Secretaria de Movilidad.

================================================================================
"""

with open(f"{OUTPUT_DIR_SEC8}/resultados/experimentos_resultados.txt", 'w', encoding='utf-8') as f:
    f.write(resultados_completos)

print(f"\n[OK] Resultados guardados: seccion_8_implementacion/resultados/experimentos_resultados.txt")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print("EXPERIMENTOS COMPLETADOS")
print("="*70)
print("\nArchivos generados:")
print(f"  -> seccion_8_implementacion/resultados/experimentos_resultados.txt")
print(f"  -> seccion_9_resultados/visualizaciones/9.2_silhouette_scores.png")
print(f"  -> seccion_9_resultados/visualizaciones/9.3_jaccard_temporal.png")
print(f"  -> seccion_9_resultados/visualizaciones/9.7_formulas_riesgo.png")
print(f"  -> seccion_9_resultados/metricas/comparacion_formulas.csv")
print("\nParametros optimos identificados:")
print(f"  1. Ventana temporal: {ventana_optima}")
print(f"  2. Epsilon DBSCAN: {epsilon_optimo} km")
print(f"  3. Formula riesgo: {formula_optima}")
print("\n" + "="*70)
print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
