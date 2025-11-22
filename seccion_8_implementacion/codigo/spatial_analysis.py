"""
ANÁLISIS ESPACIAL AVANZADO DE SINIESTRALIDAD VIAL EN BOGOTÁ
=============================================================

Este script implementa tres técnicas de análisis espacial:
1. Moran's I - Autocorrelación espacial
2. KDE (Kernel Density Estimation) - Identificación de hotspots
3. DBSCAN - Clustering espacial de siniestros

Autor: Proyecto Analítica de Siniestros
Fecha: 2025-01-19
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import gaussian_kde
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
OUTPUT_DIR_SEC8 = f"{BASE_DIR}/seccion_8_implementacion"
OUTPUT_DIR_SEC9 = f"{BASE_DIR}/seccion_9_resultados"
OUTPUT_DIR_SEC10 = f"{BASE_DIR}/seccion_10_interpretacion"

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

print("="*70)
print("ANALISIS ESPACIAL AVANZADO - SINIESTRALIDAD VIAL BOGOTA")
print("="*70)
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cargando datos...")

df = pd.read_csv(INPUT_FILE)
print(f"[OK] Dataset cargado: {len(df):,} registros")

# Filtrar solo registros geocodificados válidos
df_geo = df.dropna(subset=['latitud', 'longitud']).copy()
print(f"[OK] Registros geocodificados: {len(df_geo):,} ({len(df_geo)/len(df)*100:.1f}%)")

# Validar rango de Bogotá
BOGOTA_BOUNDS = {
    'south': 4.471,
    'north': 4.835,
    'west': -74.224,
    'east': -73.983
}

df_geo = df_geo[
    (df_geo['latitud'] >= BOGOTA_BOUNDS['south']) &
    (df_geo['latitud'] <= BOGOTA_BOUNDS['north']) &
    (df_geo['longitud'] >= BOGOTA_BOUNDS['west']) &
    (df_geo['longitud'] <= BOGOTA_BOUNDS['east'])
].copy()

print(f"[OK] Registros validados en Bogota: {len(df_geo):,}")

# Crear GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_geo,
    geometry=gpd.points_from_xy(df_geo.longitud, df_geo.latitud),
    crs="EPSG:4326"
)

print(f"[OK] GeoDataFrame creado con {len(gdf):,} puntos")

# ============================================================================
# 2. MORAN'S I - AUTOCORRELACIÓN ESPACIAL
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Calculando Moran's I...")

def calculate_morans_i(gdf, variable='GRAVEDAD', k_neighbors=8):
    """
    Calcula el Índice de Moran Global para detectar autocorrelación espacial.

    Parámetros:
    -----------
    gdf : GeoDataFrame
        Datos geoespaciales
    variable : str
        Variable a analizar (default: GRAVEDAD)
    k_neighbors : int
        Número de vecinos más cercanos para matriz de pesos (default: 8)

    Retorna:
    --------
    dict con I, expected_I, variance, z_score, p_value
    """

    # Extraer coordenadas y variable
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    y = gdf[variable].values

    # Estandarizar variable
    y_mean = y.mean()
    y_std = y.std()
    y_standardized = (y - y_mean) / y_std

    n = len(y)

    # Calcular matriz de distancias (usar muestra si es muy grande)
    if n > 5000:
        print(f"  -> Usando muestra de 5000 puntos para Moran's I (dataset muy grande)")
        sample_idx = np.random.choice(n, 5000, replace=False)
        coords_sample = coords[sample_idx]
        y_sample = y_standardized[sample_idx]
        n = 5000
    else:
        coords_sample = coords
        y_sample = y_standardized

    # Matriz de distancias
    dist_matrix = squareform(pdist(coords_sample, metric='euclidean'))

    # Matriz de pesos espaciales (k-nearest neighbors)
    W = np.zeros_like(dist_matrix)
    for i in range(n):
        # Encontrar k vecinos más cercanos (excluyendo el punto mismo)
        neighbors = np.argsort(dist_matrix[i])[1:k_neighbors+1]
        W[i, neighbors] = 1

    # Hacer matriz simétrica
    W = (W + W.T) / 2

    # Normalizar por filas
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Evitar división por cero
    W = W / row_sums[:, np.newaxis]

    # Calcular Moran's I
    numerator = np.sum(W * np.outer(y_sample, y_sample))
    denominator = np.sum(y_sample ** 2)

    I = (n / W.sum()) * (numerator / denominator)

    # Valor esperado bajo hipótesis nula (distribución aleatoria)
    expected_I = -1 / (n - 1)

    # Varianza (fórmula simplificada)
    S0 = W.sum()
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)

    b2 = (n * np.sum(y_sample ** 4)) / (np.sum(y_sample ** 2) ** 2)

    variance = ((n * ((n**2 - 3*n + 3) * S1 - n*S2 + 3*S0**2)) -
                (b2 * ((n**2 - n) * S1 - 2*n*S2 + 6*S0**2))) / \
               ((n - 1) * (n - 2) * (n - 3) * S0**2) - expected_I**2

    # Z-score y p-value
    z_score = (I - expected_I) / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

    return {
        'I': I,
        'expected_I': expected_I,
        'variance': variance,
        'z_score': z_score,
        'p_value': p_value,
        'n': n,
        'k_neighbors': k_neighbors
    }

# Calcular Moran's I
moran_result = calculate_morans_i(gdf, variable='GRAVEDAD', k_neighbors=8)

print(f"[OK] Indice de Moran calculado:")
print(f"  I = {moran_result['I']:.4f}")
print(f"  Esperado (aleatorio) = {moran_result['expected_I']:.4f}")
print(f"  Z-score = {moran_result['z_score']:.4f}")
print(f"  p-value = {moran_result['p_value']:.6f}")

# Interpretación
if moran_result['p_value'] < 0.001:
    significance = "p < 0.001 (altamente significativo)"
elif moran_result['p_value'] < 0.01:
    significance = "p < 0.01 (muy significativo)"
elif moran_result['p_value'] < 0.05:
    significance = "p < 0.05 (significativo)"
else:
    significance = f"p = {moran_result['p_value']:.4f} (no significativo)"

if moran_result['I'] > moran_result['expected_I']:
    pattern = "CLUSTERING ESPACIAL (siniestros graves agrupados)"
elif moran_result['I'] < moran_result['expected_I']:
    pattern = "DISPERSIÓN ESPACIAL (siniestros graves dispersos)"
else:
    pattern = "DISTRIBUCIÓN ALEATORIA"

print(f"  Interpretación: {pattern}")
print(f"  Significancia: {significance}")

# Guardar resultados
moran_output = f"""
RESULTADOS: ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL (MORAN'S I)
================================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Variable analizada: GRAVEDAD
Muestra: {moran_result['n']:,} siniestros geocodificados

ÍNDICE DE MORAN GLOBAL
-----------------------
I observado:        {moran_result['I']:.6f}
I esperado (H0):    {moran_result['expected_I']:.6f}
Varianza:           {moran_result['variance']:.6f}
Z-score:            {moran_result['z_score']:.4f}
P-value:            {moran_result['p_value']:.6f}

INTERPRETACIÓN
--------------
Patrón espacial: {pattern}
Significancia estadística: {significance}

CONCLUSIÓN:
{
'Los siniestros graves en Bogotá muestran autocorrelación espacial POSITIVA significativa, ' +
'lo que indica que tienden a agruparse en zonas específicas de la ciudad. ' +
'Esto valida la pertinencia de estrategias de intervención focalizada en lugar de dispersa.'
if moran_result['I'] > moran_result['expected_I'] and moran_result['p_value'] < 0.05
else 'No se encontró evidencia significativa de clustering espacial.'
}

PARÁMETROS TÉCNICOS
-------------------
Método de pesos espaciales: K-Nearest Neighbors
K vecinos: {moran_result['k_neighbors']}
Proyección: EPSG:4326 (WGS84)
"""

with open(f"{OUTPUT_DIR_SEC8}/resultados/moran_i_results.txt", 'w', encoding='utf-8') as f:
    f.write(moran_output)

print(f"[OK] Resultados guardados en: seccion_8_implementacion/resultados/moran_i_results.txt")

# ============================================================================
# 3. KDE - KERNEL DENSITY ESTIMATION (HOTSPOTS)
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Calculando KDE Hotspots...")

# Filtrar solo siniestros graves (GRAVEDAD = 1 o 3)
df_graves = gdf[gdf['GRAVEDAD'].isin([1, 3])].copy()
print(f"[OK] Siniestros graves para KDE: {len(df_graves):,}")

# Usar muestra si es muy grande
if len(df_graves) > 10000:
    df_graves_sample = df_graves.sample(10000, random_state=42)
    print(f"  -> Usando muestra de 10,000 para visualización KDE")
else:
    df_graves_sample = df_graves

# Calcular KDE
coords_graves = np.array(list(zip(df_graves_sample.geometry.y, df_graves_sample.geometry.x)))

# Bandwidth óptimo (regla de Scott)
bandwidth = len(df_graves_sample) ** (-1/6)
print(f"[OK] Bandwidth KDE (Scott): {bandwidth:.4f}")

kde = gaussian_kde(coords_graves.T, bw_method=bandwidth)

# Crear grid para evaluar KDE
lat_min, lat_max = BOGOTA_BOUNDS['south'], BOGOTA_BOUNDS['north']
lon_min, lon_max = BOGOTA_BOUNDS['west'], BOGOTA_BOUNDS['east']

grid_lat = np.linspace(lat_min, lat_max, 100)
grid_lon = np.linspace(lon_min, lon_max, 100)
grid_lat_mesh, grid_lon_mesh = np.meshgrid(grid_lat, grid_lon)

grid_coords = np.vstack([grid_lat_mesh.ravel(), grid_lon_mesh.ravel()])
density = kde(grid_coords).reshape(grid_lat_mesh.shape)

# Identificar hotspots (percentil 90)
threshold_90 = np.percentile(density, 90)
hotspot_mask = density > threshold_90

print(f"[OK] Hotspots identificados (percentil 90): {hotspot_mask.sum()} celdas")

# Crear mapa interactivo con Folium
m_kde = folium.Map(
    location=[4.6533, -74.0836],  # Centro de Bogotá
    zoom_start=11,
    tiles='OpenStreetMap'
)

# Agregar heatmap
heat_data = [[row.geometry.y, row.geometry.x] for idx, row in df_graves_sample.iterrows()]
HeatMap(heat_data, radius=15, blur=25, max_zoom=13, gradient={
    0.0: 'blue',
    0.5: 'yellow',
    0.7: 'orange',
    1.0: 'red'
}).add_to(m_kde)

# Agregar leyenda
legend_html = '''
<div style="position: fixed;
            top: 10px; right: 10px; width: 250px; height: 120px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px">
<p style="margin:0; font-weight:bold;">Mapa de Densidad (KDE)</p>
<p style="margin:5px 0; font-size:12px;">
<span style="color:blue;">●</span> Baja densidad<br>
<span style="color:yellow;">●</span> Densidad media<br>
<span style="color:orange;">●</span> Densidad alta<br>
<span style="color:red;">●</span> HOTSPOT crítico<br>
</p>
<p style="margin:5px 0; font-size:11px; color:grey;">
n = ''' + f"{len(df_graves_sample):,} siniestros graves" + '''
</p>
</div>
'''
m_kde.get_root().html.add_child(folium.Element(legend_html))

# Guardar mapa
kde_map_path = f"{OUTPUT_DIR_SEC9}/visualizaciones/9.4_kde_hotspots.html"
m_kde.save(kde_map_path)
print(f"[OK] Mapa KDE guardado en: seccion_9_resultados/visualizaciones/9.4_kde_hotspots.html")

# ============================================================================
# 4. DBSCAN - CLUSTERING ESPACIAL
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Ejecutando DBSCAN clustering...")

# Preparar coordenadas (convertir grados a metros aproximados)
# 1 grado de latitud ~ 111 km
# 1 grado de longitud en Bogotá (4.6 degN) ~ 111 * cos(4.6 deg) ~ 110.5 km

coords_km = np.column_stack([
    df_graves.geometry.y * 111,  # Latitud a km
    df_graves.geometry.x * 110.5  # Longitud a km
])

# DBSCAN con epsilon = 1 km, min_samples = 10
epsilon_km = 1.0  # 1 km
min_samples = 10

dbscan = DBSCAN(eps=epsilon_km, min_samples=min_samples, metric='euclidean')
clusters = dbscan.fit_predict(coords_km)

df_graves['cluster'] = clusters

# Estadísticas
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"[OK] Clustering completado:")
print(f"  Clusters identificados: {n_clusters}")
print(f"  Puntos de ruido (outliers): {n_noise} ({n_noise/len(clusters)*100:.1f}%)")

# Métricas de calidad (solo para puntos no-ruido)
if n_clusters > 1:
    mask_clustered = clusters != -1
    silhouette = silhouette_score(coords_km[mask_clustered], clusters[mask_clustered])
    calinski = calinski_harabasz_score(coords_km[mask_clustered], clusters[mask_clustered])
    print(f"  Silhouette Score: {silhouette:.4f} (rango: -1 a 1, >0.5 es bueno)")
    print(f"  Calinski-Harabasz Index: {calinski:.2f} (mayor es mejor)")
else:
    silhouette = None
    calinski = None
    print(f"  [!] Solo 1 cluster, métricas no aplicables")

# Formatear métricas
silh_str = f"{silhouette:.4f}" if silhouette is not None else 'N/A'
cal_str = f"{calinski:.2f}" if calinski is not None else 'N/A'

if silhouette is not None:
    if silhouette > 0.5:
        silh_interp = 'Buena separación entre clusters'
    elif silhouette > 0.3:
        silh_interp = 'Moderada separación'
    else:
        silh_interp = 'Débil separación'
else:
    silh_interp = 'N/A'

if calinski is not None:
    cal_interp = 'bien definidos' if calinski > 100 else 'moderadamente definidos'
else:
    cal_interp = 'N/A'

outlier_text = (f'El {n_noise/len(clusters)*100:.1f}% de siniestros son outliers espaciales (no pertenecen a clusters), ' +
                'lo que sugiere que también existen siniestros graves aislados fuera de las zonas críticas.'
                if n_noise > 0 else 'Todos los siniestros están asignados a clusters.')

# Guardar resultados
dbscan_output = f"""
RESULTADOS: CLUSTERING ESPACIAL (DBSCAN)
==========================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: Siniestros GRAVES (GRAVEDAD = 1 o 3)
Muestra: {len(df_graves):,} siniestros

PARÁMETROS DBSCAN
-----------------
Epsilon (radio): {epsilon_km} km
Min samples: {min_samples} puntos
Métrica: Euclidean (coordenadas convertidas a km)

RESULTADOS
----------
Clusters identificados: {n_clusters}
Puntos de ruido (outliers): {n_noise} ({n_noise/len(clusters)*100:.1f}%)
Puntos clustered: {len(clusters) - n_noise} ({(len(clusters) - n_noise)/len(clusters)*100:.1f}%)

MÉTRICAS DE CALIDAD
-------------------
Silhouette Score: {silh_str}
  -> Interpretación: {silh_interp}

Calinski-Harabasz Index: {cal_str}
  -> Interpretación: Clusters {cal_interp}

INTERPRETACIÓN
--------------
Se identificaron {n_clusters} zonas de concentración espacial de siniestros graves en Bogotá.
{outlier_text}

Cada cluster representa una "zona de alto riesgo" que podría beneficiarse de intervenciones
específicas de infraestructura, señalización o control de tránsito.
"""

with open(f"{OUTPUT_DIR_SEC8}/resultados/dbscan_results.txt", 'w', encoding='utf-8') as f:
    f.write(dbscan_output)

print(f"[OK] Resultados guardados en: seccion_8_implementacion/resultados/dbscan_results.txt")

# Crear mapa de clusters
m_clusters = folium.Map(
    location=[4.6533, -74.0836],
    zoom_start=11,
    tiles='CartoDB positron'
)

# Colores para clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
          'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink']

# Agregar clusters
for cluster_id in range(n_clusters):
    cluster_points = df_graves[df_graves['cluster'] == cluster_id]
    color = colors[cluster_id % len(colors)]

    # Crear feature group para cada cluster
    fg = folium.FeatureGroup(name=f'Cluster {cluster_id + 1} (n={len(cluster_points)})')

    for idx, row in cluster_points.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=f"Cluster {cluster_id + 1}<br>Gravedad: {row['GRAVEDAD']}<br>Fecha: {row['FECHA']}"
        ).add_to(fg)

    fg.add_to(m_clusters)

# Agregar outliers
if n_noise > 0:
    outliers = df_graves[df_graves['cluster'] == -1]
    fg_noise = folium.FeatureGroup(name=f'Outliers (n={n_noise})', show=False)

    for idx, row in outliers.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color='gray',
            fill=True,
            fillColor='gray',
            fillOpacity=0.3,
            popup=f"Outlier<br>Gravedad: {row['GRAVEDAD']}<br>Fecha: {row['FECHA']}"
        ).add_to(fg_noise)

    fg_noise.add_to(m_clusters)

# Agregar control de capas
folium.LayerControl().add_to(m_clusters)

# Guardar mapa
clusters_map_path = f"{OUTPUT_DIR_SEC9}/visualizaciones/9.5_dbscan_clusters.html"
m_clusters.save(clusters_map_path)
print(f"[OK] Mapa de clusters guardado en: seccion_9_resultados/visualizaciones/9.5_dbscan_clusters.html")

# Guardar dataset con clusters
df_graves.to_csv(f"{OUTPUT_DIR_SEC10}/analisis/siniestros_con_clusters.csv", index=False)
print(f"[OK] Dataset con clusters guardado en: seccion_10_interpretacion/analisis/siniestros_con_clusters.csv")

# ============================================================================
# 5. VISUALIZACIÓN: MORAN'S I SCATTERPLOT
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generando visualización Moran's I...")

# Crear scatterplot de Moran (usando muestra)
sample_size = min(2000, len(gdf))
gdf_sample = gdf.sample(sample_size, random_state=42)

coords_sample = np.array(list(zip(gdf_sample.geometry.x, gdf_sample.geometry.y)))
y_sample = gdf_sample['GRAVEDAD'].values

# Calcular lag espacial (promedio de vecinos)
dist_matrix_sample = squareform(pdist(coords_sample, metric='euclidean'))
k = 8
spatial_lag = np.zeros(sample_size)

for i in range(sample_size):
    neighbors = np.argsort(dist_matrix_sample[i])[1:k+1]
    spatial_lag[i] = y_sample[neighbors].mean()

# Estandarizar
y_standardized = (y_sample - y_sample.mean()) / y_sample.std()
lag_standardized = (spatial_lag - spatial_lag.mean()) / spatial_lag.std()

# Crear gráfico
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(y_standardized, lag_standardized, alpha=0.5, s=20, color='steelblue', edgecolors='navy', linewidth=0.5)
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Línea de regresión
m, b = np.polyfit(y_standardized, lag_standardized, 1)
ax.plot(y_standardized, m*y_standardized + b, color='red', linewidth=2, label=f'Pendiente = {m:.3f}')

ax.set_xlabel('GRAVEDAD (estandarizada)', fontsize=12, fontweight='bold')
ax.set_ylabel('Spatial Lag de GRAVEDAD (estandarizada)', fontsize=12, fontweight='bold')
ax.set_title(f"Moran's I Scatterplot\nI = {moran_result['I']:.4f}, p-value = {moran_result['p_value']:.6f}",
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Anotar cuadrantes
ax.text(0.95, 0.95, 'Alto-Alto', transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color='darkred', alpha=0.7)
ax.text(0.05, 0.05, 'Bajo-Bajo', transform=ax.transAxes, ha='left', va='bottom',
        fontsize=10, color='darkblue', alpha=0.7)
ax.text(0.05, 0.95, 'Bajo-Alto', transform=ax.transAxes, ha='left', va='top',
        fontsize=10, color='gray', alpha=0.7)
ax.text(0.95, 0.05, 'Alto-Bajo', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=10, color='gray', alpha=0.7)

plt.tight_layout()
moran_plot_path = f"{OUTPUT_DIR_SEC9}/visualizaciones/9.1_moran_scatterplot.png"
plt.savefig(moran_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Gráfico Moran's I guardado en: seccion_9_resultados/visualizaciones/9.1_moran_scatterplot.png")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print("ANÁLISIS ESPACIAL COMPLETADO")
print("="*70)
print("\nArchivos generados:")
print(f"  -> seccion_8_implementacion/resultados/moran_i_results.txt")
print(f"  -> seccion_8_implementacion/resultados/dbscan_results.txt")
print(f"  -> seccion_9_resultados/visualizaciones/9.1_moran_scatterplot.png")
print(f"  -> seccion_9_resultados/visualizaciones/9.4_kde_hotspots.html")
print(f"  -> seccion_9_resultados/visualizaciones/9.5_dbscan_clusters.html")
print(f"  -> seccion_10_interpretacion/analisis/siniestros_con_clusters.csv")
print("\n" + "="*70)
print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
