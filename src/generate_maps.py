"""
Generacion de mapas interactivos con Folium
Muestra siniestros y POIs en mapa real de Bogota
"""

import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import pickle
import numpy as np

print("="*80)
print("GENERACION DE MAPAS INTERACTIVOS CON FOLIUM")
print("="*80)

# 1. Cargar datos
print("\n[1/4] Cargando datos...")
df = pd.read_csv('data/processed/siniestros_features_geo.csv')
df_geo = df[df['latitud'].notna()].copy()
print(f"Registros con coordenadas: {len(df_geo):,}")

with open('data/processed/poi_cache.pkl', 'rb') as f:
    pois_dict = pickle.load(f)
print(f"POIs cargados: {sum(len(v) for v in pois_dict.values())}")

# Centro de Bogota
BOGOTA_CENTER = [4.65, -74.08]

# 2. Mapa 1: Heatmap de siniestros
print("\n[2/4] Generando mapa de calor de siniestros...")

# Tomar muestra para visualizacion (max 5000 puntos para rendimiento)
sample_size = min(5000, len(df_geo))
df_sample = df_geo.sample(n=sample_size, random_state=42)

mapa_heat = folium.Map(location=BOGOTA_CENTER, zoom_start=11, tiles='OpenStreetMap')

# Crear heatmap
heat_data = [[row['latitud'], row['longitud']] for idx, row in df_sample.iterrows()]
HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(mapa_heat)

# Guardar
output_file = 'data/processed/eda_visualizaciones/mapa_01_heatmap_siniestros.html'
mapa_heat.save(output_file)
print(f"  Guardado: {output_file.split('/')[-1]}")

# 3. Mapa 2: Siniestros + POIs (solo siniestros graves)
print("\n[3/4] Generando mapa interactivo: Siniestros graves + POIs...")

mapa_pois = folium.Map(location=BOGOTA_CENTER, zoom_start=11, tiles='OpenStreetMap')

# Agregar siniestros GRAVES (muertos) con clusters
siniestros_graves = df_geo[df_geo['GRAVEDAD'] == 3].copy()
print(f"  Siniestros con muertos: {len(siniestros_graves)}")

if len(siniestros_graves) > 0:
    # Limitar a 500 para rendimiento
    if len(siniestros_graves) > 500:
        siniestros_graves = siniestros_graves.sample(n=500, random_state=42)

    # Cluster de siniestros
    marker_cluster_siniestros = MarkerCluster(name="Siniestros con Muertos").add_to(mapa_pois)

    for idx, row in siniestros_graves.iterrows():
        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=5,
            color='red',
            fill=True,
            fill_color='darkred',
            fill_opacity=0.6,
            popup=f"Siniestro GRAVE<br>Fecha: {row['FECHA']}<br>Hora: {row['HORA']}",
            tooltip="Siniestro con muertos"
        ).add_to(marker_cluster_siniestros)

# Agregar POIs por categoria
colores_pois = {
    'centros_comerciales': 'blue',
    'estadios': 'green',
    'bares': 'orange',
    'transmilenio': 'purple'
}

nombres_pois = {
    'centros_comerciales': 'Centros Comerciales',
    'estadios': 'Estadios',
    'bares': 'Bares/Pubs',
    'transmilenio': 'TransMilenio'
}

for categoria, coords in pois_dict.items():
    if categoria in colores_pois:
        # Crear grupo de features para control de capas
        feature_group = folium.FeatureGroup(name=nombres_pois[categoria])

        for lat, lon in coords:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=colores_pois[categoria],
                fill=True,
                fill_color=colores_pois[categoria],
                fill_opacity=0.7,
                popup=f"{nombres_pois[categoria]}",
                tooltip=nombres_pois[categoria]
            ).add_to(feature_group)

        feature_group.add_to(mapa_pois)

# Agregar control de capas
folium.LayerControl().add_to(mapa_pois)

# Guardar
output_file = 'data/processed/eda_visualizaciones/mapa_02_siniestros_pois.html'
mapa_pois.save(output_file)
print(f"  Guardado: {output_file.split('/')[-1]}")

# 4. Mapa 3: Analisis nocturno (bares + siniestros nocturnos)
print("\n[4/4] Generando mapa: Siniestros nocturnos vs Bares...")

mapa_nocturno = folium.Map(location=BOGOTA_CENTER, zoom_start=12, tiles='CartoDB dark_matter')

# Siniestros nocturnos (20:00 - 04:00)
if 'hora_num' in df_geo.columns:
    siniestros_noche = df_geo[
        ((df_geo['hora_num'] >= 20) | (df_geo['hora_num'] <= 4))
    ].copy()
else:
    # Extraer hora del string HORA si existe
    df_geo['hora_temp'] = pd.to_datetime(df_geo['HORA']).dt.hour
    siniestros_noche = df_geo[
        ((df_geo['hora_temp'] >= 20) | (df_geo['hora_temp'] <= 4))
    ].copy()

print(f"  Siniestros nocturnos: {len(siniestros_noche)}")

# Limitar muestra
if len(siniestros_noche) > 1000:
    siniestros_noche = siniestros_noche.sample(n=1000, random_state=42)

# Agregar siniestros nocturnos
for idx, row in siniestros_noche.iterrows():
    color = 'darkred' if row['GRAVEDAD'] == 3 else 'orange' if row['GRAVEDAD'] == 2 else 'yellow'
    folium.CircleMarker(
        location=[row['latitud'], row['longitud']],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        tooltip="Siniestro nocturno"
    ).add_to(mapa_nocturno)

# Agregar bares
if 'bares' in pois_dict:
    for lat, lon in pois_dict['bares']:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='cyan',
            fill=True,
            fill_color='cyan',
            fill_opacity=0.6,
            tooltip="Bar/Pub/Discoteca"
        ).add_to(mapa_nocturno)

# Guardar
output_file = 'data/processed/eda_visualizaciones/mapa_03_nocturnos_bares.html'
mapa_nocturno.save(output_file)
print(f"  Guardado: {output_file.split('/')[-1]}")

print("\n" + "="*80)
print("MAPAS INTERACTIVOS GENERADOS")
print("="*80)
print("\nArchivos HTML creados:")
print("  1. mapa_01_heatmap_siniestros.html - Mapa de calor general")
print("  2. mapa_02_siniestros_pois.html - Siniestros graves + POIs (interactivo)")
print("  3. mapa_03_nocturnos_bares.html - Analisis nocturno")
print("\nAbre los archivos en tu navegador para explorar los mapas interactivos.")
print("="*80)
