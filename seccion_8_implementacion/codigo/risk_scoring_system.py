"""
SISTEMA DE PRIORIZACION DE INTERVENCIONES
==========================================

Este script implementa el sistema de scoring de riesgo para identificar
las intersecciones y zonas criticas que requieren intervencion inmediata.

Formula optima (validada en experimentos): Risk = Densidad x Gravedad_promedio

Autor: Proyecto Analitica de Siniestros
Fecha: 2025-01-20
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuracion de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURACION DE RUTAS
# ============================================================================

BASE_DIR = r"c:\Users\caenp\OneDrive\Escritorio\analitica_siniestros"
INPUT_FILE = f"{BASE_DIR}/data/processed/siniestros_features_geo.csv"
CLUSTERS_FILE = f"{BASE_DIR}/seccion_10_interpretacion/analisis/siniestros_con_clusters.csv"
OUTPUT_DIR_SEC9 = f"{BASE_DIR}/seccion_9_resultados"
OUTPUT_DIR_SEC10 = f"{BASE_DIR}/seccion_10_interpretacion"
OUTPUT_DIR_SEC11 = f"{BASE_DIR}/seccion_11_dashboard"

# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("="*70)
print("SISTEMA DE PRIORIZACION DE INTERVENCIONES")
print("="*70)
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cargando datos...")

df = pd.read_csv(INPUT_FILE)
df['FECHA'] = pd.to_datetime(df['FECHA'])
print(f"[OK] Dataset cargado: {len(df):,} registros")

df_clusters = pd.read_csv(CLUSTERS_FILE)
print(f"[OK] Dataset con clusters cargado: {len(df_clusters):,} registros")

# ============================================================================
# 1. CALCULAR SCORES DE RIESGO POR CLUSTER
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Calculando scores de riesgo por cluster...")

# Estadisticas por cluster
cluster_stats = df_clusters.groupby('cluster').agg({
    'CODIGO_ACCIDENTE': 'count',  # Densidad
    'GRAVEDAD': 'mean',            # Gravedad promedio
    'latitud': 'mean',             # Centroide
    'longitud': 'mean',
    'CODIGO_LOCALIDAD': lambda x: x.mode()[0] if len(x) > 0 else None,  # Localidad mas frecuente
    'localidad_nombre': lambda x: x.mode()[0] if len(x) > 0 else None
}).rename(columns={'CODIGO_ACCIDENTE': 'densidad'})

# Formula 1: Densidad x Gravedad (validada como optima)
cluster_stats['riesgo_score'] = cluster_stats['densidad'] * cluster_stats['GRAVEDAD']

# Normalizar score (0-100)
min_score = cluster_stats['riesgo_score'].min()
max_score = cluster_stats['riesgo_score'].max()
cluster_stats['riesgo_norm'] = ((cluster_stats['riesgo_score'] - min_score) / (max_score - min_score)) * 100

# Clasificar nivel de riesgo
def clasificar_riesgo(score):
    if score >= 80:
        return 'CRITICO'
    elif score >= 60:
        return 'ALTO'
    elif score >= 40:
        return 'MEDIO'
    else:
        return 'BAJO'

cluster_stats['nivel_riesgo'] = cluster_stats['riesgo_norm'].apply(clasificar_riesgo)

# Ranking
cluster_stats['ranking'] = cluster_stats['riesgo_norm'].rank(ascending=False, method='dense').astype(int)

cluster_stats = cluster_stats.sort_values('riesgo_norm', ascending=False)

print(f"\nEstadisticas de {len(cluster_stats)} clusters:")
print("\nTop 5 clusters mas criticos:")
print(cluster_stats.head()[['densidad', 'GRAVEDAD', 'riesgo_norm', 'nivel_riesgo', 'ranking']])

# Guardar analisis de clusters
cluster_stats.to_csv(f"{OUTPUT_DIR_SEC10}/analisis/perfiles_clusters.csv", index=True)
print(f"\n[OK] Perfiles guardados: seccion_10_interpretacion/analisis/perfiles_clusters.csv")

# ============================================================================
# 2. IDENTIFICAR TOP 20 INTERSECCIONES CRITICAS
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Identificando Top 20 intersecciones criticas...")

# Filtrar solo siniestros graves
df_graves = df_clusters[df_clusters['GRAVEDAD'].isin([1, 3])].copy()

# Agrupar por direccion (intersecciones)
intersecciones = df_graves.groupby('DIRECCION').agg({
    'CODIGO_ACCIDENTE': 'count',
    'GRAVEDAD': 'mean',
    'latitud': 'first',
    'longitud': 'first',
    'localidad_nombre': 'first',
    'cluster': 'first'
}).rename(columns={'CODIGO_ACCIDENTE': 'num_siniestros'})

# Filtrar direcciones con al menos 5 siniestros (intersecciones recurrentes)
intersecciones = intersecciones[intersecciones['num_siniestros'] >= 5].copy()

# Calcular score de riesgo
intersecciones['riesgo_score'] = intersecciones['num_siniestros'] * intersecciones['GRAVEDAD']

# Normalizar
min_int = intersecciones['riesgo_score'].min()
max_int = intersecciones['riesgo_score'].max()
intersecciones['riesgo_norm'] = ((intersecciones['riesgo_score'] - min_int) / (max_int - min_int)) * 100

# Clasificar
intersecciones['nivel_riesgo'] = intersecciones['riesgo_norm'].apply(clasificar_riesgo)

# Top 20
top20 = intersecciones.nlargest(20, 'riesgo_norm').copy()
top20['ranking'] = range(1, 21)

print(f"\n{len(intersecciones):,} intersecciones analizadas (>= 5 siniestros)")
print(f"\nTop 20 Intersecciones Criticas:")
print(top20[['num_siniestros', 'GRAVEDAD', 'riesgo_norm', 'nivel_riesgo', 'localidad_nombre']])

# Guardar Top 20
top20_export = top20.copy()
top20_export.index.name = 'DIRECCION'
top20_export = top20_export.reset_index()
top20_export = top20_export[['ranking', 'DIRECCION', 'localidad_nombre', 'num_siniestros',
                               'GRAVEDAD', 'riesgo_norm', 'nivel_riesgo', 'latitud', 'longitud']]
top20_export.columns = ['Ranking', 'Interseccion', 'Localidad', 'Num_Siniestros',
                        'Gravedad_Promedio', 'Score_Riesgo', 'Nivel_Riesgo', 'Latitud', 'Longitud']

top20_export.to_csv(f"{OUTPUT_DIR_SEC11}/reporte_top20.csv", index=False, encoding='utf-8-sig')
print(f"\n[OK] Top 20 guardado: seccion_11_dashboard/reporte_top20.csv")

# ============================================================================
# 3. ESTADISTICAS DE IMPACTO POTENCIAL
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Calculando impacto potencial de intervenciones...")

# Siniestros en Top 20 intersecciones
siniestros_top20 = top20['num_siniestros'].sum()
total_graves = len(df_graves)
pct_top20 = (siniestros_top20 / total_graves) * 100

# Siniestros en clusters criticos
clusters_criticos = cluster_stats[cluster_stats['nivel_riesgo'] == 'CRITICO'].index.tolist()
siniestros_clusters_criticos = df_graves[df_graves['cluster'].isin(clusters_criticos)]['CODIGO_ACCIDENTE'].count()
pct_clusters = (siniestros_clusters_criticos / total_graves) * 100

print(f"\nIMPACTO POTENCIAL DE INTERVENCIONES:")
print(f"  Total siniestros graves: {total_graves:,}")
print(f"  Siniestros en Top 20 intersecciones: {siniestros_top20:,} ({pct_top20:.1f}%)")
print(f"  Siniestros en clusters CRITICOS: {siniestros_clusters_criticos:,} ({pct_clusters:.1f}%)")
print(f"\n  -> Interviniendo Top 20 se impactaria: {pct_top20:.1f}% de siniestros graves")
print(f"  -> Interviniendo clusters CRITICOS se impactaria: {pct_clusters:.1f}% de siniestros graves")

# ============================================================================
# 4. MAPA INTERACTIVO: TOP 20 INTERSECCIONES
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generando mapa interactivo Top 20...")

# Crear mapa base
m_top20 = folium.Map(
    location=[4.6533, -74.0836],
    zoom_start=11,
    tiles='CartoDB positron'
)

# Colores por nivel de riesgo
color_map = {
    'CRITICO': 'darkred',
    'ALTO': 'red',
    'MEDIO': 'orange',
    'BAJO': 'yellow'
}

# Agregar marcadores para Top 20
for idx, row in top20.iterrows():
    color = color_map.get(row['nivel_riesgo'], 'gray')

    popup_html = f"""
    <div style="font-family: Arial; width: 250px;">
        <h4 style="margin:0; color:{color};">#{row['ranking']} - {row['nivel_riesgo']}</h4>
        <hr style="margin:5px 0;">
        <b>Interseccion:</b> {idx}<br>
        <b>Localidad:</b> {row['localidad_nombre']}<br>
        <b>Siniestros:</b> {row['num_siniestros']}<br>
        <b>Gravedad prom:</b> {row['GRAVEDAD']:.2f}<br>
        <b>Score riesgo:</b> {row['riesgo_norm']:.1f}/100
    </div>
    """

    # Icono personalizado
    icon_html = f'''
    <div style="font-size: 16px; font-weight: bold; color: white;
                background-color: {color}; border-radius: 50%;
                width: 30px; height: 30px; display: flex;
                align-items: center; justify-content: center;
                border: 2px solid white;">
        {row['ranking']}
    </div>
    '''

    folium.Marker(
        location=[row['latitud'], row['longitud']],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.DivIcon(html=icon_html)
    ).add_to(m_top20)

# Leyenda
legend_html = '''
<div style="position: fixed;
            top: 10px; right: 10px; width: 220px; height: 180px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px;">
<p style="margin:0; font-weight:bold;">Top 20 Intersecciones Criticas</p>
<hr style="margin:5px 0;">
<p style="margin:5px 0;">
<span style="color:darkred; font-size:20px;">●</span> CRITICO (80-100)<br>
<span style="color:red; font-size:20px;">●</span> ALTO (60-80)<br>
<span style="color:orange; font-size:20px;">●</span> MEDIO (40-60)<br>
<span style="color:yellow; font-size:20px;">●</span> BAJO (0-40)
</p>
<hr style="margin:5px 0;">
<p style="margin:5px 0; font-size:11px; color:grey;">
Score = Densidad x Gravedad<br>
''' + f"Impacto: {pct_top20:.1f}% siniestros graves" + '''
</p>
</div>
'''
m_top20.get_root().html.add_child(folium.Element(legend_html))

# Guardar mapa
m_top20.save(f"{OUTPUT_DIR_SEC9}/visualizaciones/9.6_top20_intersecciones.html")
print(f"[OK] Mapa guardado: seccion_9_resultados/visualizaciones/9.6_top20_intersecciones.html")

# ============================================================================
# 5. VISUALIZACION: DISTRIBUCION DE NIVELES DE RIESGO
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generando visualizaciones de distribucion...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Grafico 1: Distribucion de niveles de riesgo en clusters
nivel_counts = cluster_stats['nivel_riesgo'].value_counts()
colors_bar = [color_map.get(nivel, 'gray') for nivel in nivel_counts.index]

ax1.bar(nivel_counts.index, nivel_counts.values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Nivel de Riesgo', fontsize=12, fontweight='bold')
ax1.set_ylabel('Numero de Clusters', fontsize=12, fontweight='bold')
ax1.set_title('Distribucion de Niveles de Riesgo por Cluster', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, axis='y')

# Agregar valores
for i, v in enumerate(nivel_counts.values):
    ax1.text(i, v + 0.05, str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)

# Grafico 2: Top 10 intersecciones (barras horizontales)
top10 = top20.head(10).copy()
top10 = top10.sort_values('riesgo_norm', ascending=True)

# Simplificar nombres de direcciones
direcciones_simplificadas = []
for direccion in top10.index:
    # Tomar primeras 40 caracteres
    if len(direccion) > 40:
        direcciones_simplificadas.append(direccion[:37] + '...')
    else:
        direcciones_simplificadas.append(direccion)

colors_top10 = [color_map.get(nivel, 'gray') for nivel in top10['nivel_riesgo']]

ax2.barh(range(len(top10)), top10['riesgo_norm'], color=colors_top10, edgecolor='black', linewidth=1.5)
ax2.set_yticks(range(len(top10)))
ax2.set_yticklabels(direcciones_simplificadas, fontsize=9)
ax2.set_xlabel('Score de Riesgo (0-100)', fontsize=12, fontweight='bold')
ax2.set_title('Top 10 Intersecciones Mas Criticas', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='x')

# Agregar valores
for i, v in enumerate(top10['riesgo_norm']):
    ax2.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR_SEC10}/visualizaciones/10.4_priorizacion_zonas.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Visualizacion guardada: seccion_10_interpretacion/visualizaciones/10.4_priorizacion_zonas.png")

# ============================================================================
# 6. REPORTE EJECUTIVO
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generando reporte ejecutivo...")

reporte = f"""
================================================================================
REPORTE EJECUTIVO: SISTEMA DE PRIORIZACION DE INTERVENCIONES
================================================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Secretaria de Movilidad de Bogota D.C.

================================================================================
1. RESUMEN EJECUTIVO
================================================================================

Este reporte identifica las zonas criticas de siniestralidad vial en Bogota
que requieren intervencion INMEDIATA. La priorizacion se basa en:

  - Formula de riesgo: Score = Densidad de siniestros x Gravedad promedio
  - Periodo analizado: 2015-2020
  - Siniestros graves analizados: {total_graves:,}
  - Clusters espaciales identificados: {len(cluster_stats)}
  - Intersecciones recurrentes (>=5 siniestros): {len(intersecciones):,}

================================================================================
2. TOP 20 INTERSECCIONES CRITICAS
================================================================================

Las siguientes 20 intersecciones concentran {pct_top20:.1f}% de los siniestros
graves de la ciudad. Se recomienda INTERVENCION PRIORITARIA:

"""

for idx, row in top20.iterrows():
    reporte += f"\n{row['ranking']}. {idx}"
    reporte += f"\n   Localidad: {row['localidad_nombre']}"
    reporte += f"\n   Siniestros: {row['num_siniestros']} | Gravedad: {row['GRAVEDAD']:.2f} | Score: {row['riesgo_norm']:.1f}/100"
    reporte += f"\n   Nivel: {row['nivel_riesgo']}\n"

reporte += f"""
================================================================================
3. CLUSTERS ESPACIALES - ZONAS DE ALTO RIESGO
================================================================================

Se identificaron {len(cluster_stats)} clusters espaciales con los siguientes
niveles de riesgo:

"""

for nivel in ['CRITICO', 'ALTO', 'MEDIO', 'BAJO']:
    count = len(cluster_stats[cluster_stats['nivel_riesgo'] == nivel])
    if count > 0:
        reporte += f"  {nivel:10s}: {count} cluster(es)\n"

clusters_criticos_info = cluster_stats[cluster_stats['nivel_riesgo'] == 'CRITICO']
if len(clusters_criticos_info) > 0:
    reporte += f"\nClusters CRITICOS (requieren atencion urgente):\n"
    for cluster_id, row in clusters_criticos_info.iterrows():
        reporte += f"\n  Cluster {cluster_id}:"
        reporte += f"\n    - Siniestros: {row['densidad']}"
        reporte += f"\n    - Gravedad promedio: {row['GRAVEDAD']:.2f}"
        reporte += f"\n    - Localidad: {row['localidad_nombre']}"
        reporte += f"\n    - Score: {row['riesgo_norm']:.1f}/100\n"

reporte += f"""
================================================================================
4. IMPACTO ESTIMADO DE INTERVENCIONES
================================================================================

ESCENARIO 1: Intervencion en Top 20 intersecciones
  - Siniestros impactados: {siniestros_top20:,} ({pct_top20:.1f}%)
  - Estrategia: Intervencion puntual en 20 ubicaciones
  - Recursos: BAJOS (20 puntos criticos)
  - Tiempo estimado: 3-6 meses

ESCENARIO 2: Intervencion en clusters CRITICOS
  - Siniestros impactados: {siniestros_clusters_criticos:,} ({pct_clusters:.1f}%)
  - Estrategia: Intervencion por zonas amplias
  - Recursos: ALTOS ({len(clusters_criticos_info)} cluster(es) extenso(s))
  - Tiempo estimado: 12-18 meses

RECOMENDACION: Implementar ESCENARIO 1 (Top 20) en fase inmediata,
               seguido de ESCENARIO 2 (Clusters) en fase de mediano plazo.

================================================================================
5. RECOMENDACIONES POR TIPO DE INTERVENCION
================================================================================

Para Top 20 Intersecciones:
  1. Instalacion de semaforos inteligentes / camaras
  2. Rediseno geometrico de intersecciones
  3. Senalizacion horizontal y vertical reforzada
  4. Reductores de velocidad en zonas criticas
  5. Operativos de control de transito (horarios pico)

Para Clusters Criticos:
  1. Auditoria de seguridad vial (Road Safety Audit)
  2. Mejoramiento de iluminacion
  3. Separacion de flujos vehiculares/peatonales
  4. Ciclorutas segregadas en zonas de alto riesgo
  5. Campanas de educacion vial focalizada

================================================================================
6. SIGUIENTES PASOS
================================================================================

INMEDIATO (0-3 meses):
  [ ] Visitas de campo a Top 20 intersecciones
  [ ] Evaluacion tecnica de viabilidad de intervenciones
  [ ] Estimacion de presupuesto por interseccion
  [ ] Priorizacion final segun presupuesto disponible

CORTO PLAZO (3-6 meses):
  [ ] Diseno de intervenciones
  [ ] Aprobacion de proyectos
  [ ] Ejecucion de obras en Top 5 intersecciones

MEDIANO PLAZO (6-12 meses):
  [ ] Ejecucion Top 6-20 intersecciones
  [ ] Evaluacion de impacto de intervenciones (antes/despues)
  [ ] Inicio de estudios para clusters criticos

LARGO PLAZO (12-24 meses):
  [ ] Intervenciones en clusters criticos
  [ ] Actualizacion de modelo con datos 2021-2024
  [ ] Sistema de monitoreo continuo de nuevas zonas de riesgo

================================================================================
CONTACTO
================================================================================

Para mas informacion sobre este analisis, contactar:
Proyecto Analitica de Siniestros
Secretaria de Movilidad de Bogota D.C.

Archivos tecnicos:
  - Mapa interactivo: seccion_9_resultados/visualizaciones/9.6_top20_intersecciones.html
  - Top 20 detallado: seccion_11_dashboard/reporte_top20.csv
  - Perfiles de clusters: seccion_10_interpretacion/analisis/perfiles_clusters.csv

================================================================================
NOTA: Este reporte es generado automaticamente por el Sistema de Priorizacion
      de Intervenciones basado en analisis de 196,152 siniestros (2015-2020).
================================================================================
"""

with open(f"{OUTPUT_DIR_SEC11}/recomendaciones.txt", 'w', encoding='utf-8') as f:
    f.write(reporte)

print(f"[OK] Reporte guardado: seccion_11_dashboard/recomendaciones.txt")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print("SISTEMA DE PRIORIZACION COMPLETADO")
print("="*70)
print("\nArchivos generados:")
print(f"  -> seccion_9_resultados/visualizaciones/9.6_top20_intersecciones.html")
print(f"  -> seccion_10_interpretacion/analisis/perfiles_clusters.csv")
print(f"  -> seccion_10_interpretacion/visualizaciones/10.4_priorizacion_zonas.png")
print(f"  -> seccion_11_dashboard/reporte_top20.csv")
print(f"  -> seccion_11_dashboard/recomendaciones.txt")
print("\nResultados clave:")
print(f"  - Top 20 intersecciones impactan: {pct_top20:.1f}% siniestros graves")
print(f"  - Clusters criticos impactan: {pct_clusters:.1f}% siniestros graves")
print(f"  - {len(clusters_criticos_info)} cluster(es) clasificado(s) como CRITICO")
print("\n" + "="*70)
print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
