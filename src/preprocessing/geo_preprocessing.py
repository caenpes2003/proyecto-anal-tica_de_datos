"""
Preprocesamiento Geoespacial - Seccion 5.5
Geocodificacion de direcciones y extraccion de POIs
"""

import pandas as pd
import numpy as np
import googlemaps
import time
import os
from datetime import datetime
import pickle
import osmnx as ox
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# Configuracion
GOOGLE_API_KEY = "AIzaSyBVQdSDyAtVwXl0PLIjosjF1AXzpgGlJ4Q"
MUESTRA_SIZE = 30000  # Tamano de muestra para prueba
RADIO_PROXIMIDAD = 1000  # 1 km en metros
CACHE_FILE = 'data/processed/geocoding_cache.pkl'
POI_CACHE_FILE = 'data/processed/poi_cache.pkl'

# Coordenadas de Bogota para limitar busqueda
BOGOTA_BOUNDS = {
    'north': 4.835,
    'south': 4.471,
    'east': -73.983,
    'west': -74.224
}

print("="*80)
print("PREPROCESAMIENTO GEOESPACIAL - SECCION 5.5")
print("="*80)

# 1. CARGAR DATASET
print("\n[1/8] Cargando dataset...")
df = pd.read_csv('data/processed/siniestros_features.csv')
print(f"Dataset cargado: {len(df):,} registros")

# 2. SELECCIONAR MUESTRA ESTRATIFICADA
print(f"\n[2/8] Seleccionando muestra estratificada de {MUESTRA_SIZE:,} registros...")
print("Estratificacion por GRAVEDAD y CODIGO_LOCALIDAD")

# Estratificar por gravedad y localidad
df_muestra = df.groupby(['GRAVEDAD', 'CODIGO_LOCALIDAD'], group_keys=False).apply(
    lambda x: x.sample(frac=MUESTRA_SIZE/len(df), random_state=42)
).reset_index(drop=True)

# Si la muestra es menor, completar aleatoriamente
if len(df_muestra) < MUESTRA_SIZE:
    faltantes = MUESTRA_SIZE - len(df_muestra)
    df_extra = df[~df.index.isin(df_muestra.index)].sample(n=faltantes, random_state=42)
    df_muestra = pd.concat([df_muestra, df_extra]).reset_index(drop=True)

df_muestra = df_muestra.head(MUESTRA_SIZE)
print(f"Muestra seleccionada: {len(df_muestra):,} registros")
print(f"Distribucion por gravedad:")
print(df_muestra['GRAVEDAD'].value_counts().sort_index())

# 3. EXTRAER DIRECCIONES UNICAS
print(f"\n[3/8] Extrayendo direcciones unicas...")
direcciones_unicas = df_muestra['DIRECCION'].unique()
print(f"Direcciones unicas en la muestra: {len(direcciones_unicas):,}")

# 4. GEOCODIFICACION CON GOOGLE MAPS API
print(f"\n[4/8] Geocodificando direcciones con Google Maps API...")
print(f"API Key configurada: {GOOGLE_API_KEY[:20]}...")

# Cargar cache si existe
if os.path.exists(CACHE_FILE):
    print("Cargando cache de geocodificacion existente...")
    with open(CACHE_FILE, 'rb') as f:
        geocoding_cache = pickle.load(f)
    print(f"Cache cargado: {len(geocoding_cache):,} direcciones")
else:
    geocoding_cache = {}

# Inicializar cliente de Google Maps
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Geocodificar direcciones que no esten en cache
direcciones_por_geocodificar = [d for d in direcciones_unicas if d not in geocoding_cache]
print(f"Direcciones por geocodificar: {len(direcciones_por_geocodificar):,}")
print(f"Direcciones en cache: {len(direcciones_unicas) - len(direcciones_por_geocodificar):,}")

if len(direcciones_por_geocodificar) > 0:
    print(f"\nEstimacion de costo: ${len(direcciones_por_geocodificar) * 0.005:.2f} USD")
    print(f"Tiempo estimado: {len(direcciones_por_geocodificar) * 0.2:.0f} segundos (~{len(direcciones_por_geocodificar) * 0.2 / 60:.1f} minutos)")
    print("\nIniciando geocodificacion...")

    geocodificadas = 0
    errores = 0
    start_time = time.time()

    for i, direccion in enumerate(direcciones_por_geocodificar):
        try:
            # Agregar "Bogota, Colombia" para mejor precision
            direccion_completa = f"{direccion}, Bogota, Colombia"

            # Geocodificar
            resultado = gmaps.geocode(direccion_completa)

            if resultado:
                location = resultado[0]['geometry']['location']
                lat = location['lat']
                lng = location['lng']

                # Validar que este dentro de Bogota
                if (BOGOTA_BOUNDS['south'] <= lat <= BOGOTA_BOUNDS['north'] and
                    BOGOTA_BOUNDS['west'] <= lng <= BOGOTA_BOUNDS['east']):
                    geocoding_cache[direccion] = {
                        'lat': lat,
                        'lng': lng,
                        'formatted_address': resultado[0]['formatted_address']
                    }
                    geocodificadas += 1
                else:
                    geocoding_cache[direccion] = None
                    errores += 1
            else:
                geocoding_cache[direccion] = None
                errores += 1

            # Progreso cada 100 direcciones
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                tasa = (i + 1) / elapsed
                restante = (len(direcciones_por_geocodificar) - (i + 1)) / tasa
                print(f"Progreso: {i+1}/{len(direcciones_por_geocodificar)} "
                      f"({(i+1)/len(direcciones_por_geocodificar)*100:.1f}%) - "
                      f"Exitosas: {geocodificadas} - Errores: {errores} - "
                      f"Tiempo restante: {restante/60:.1f} min")

                # Guardar cache periodicamente
                with open(CACHE_FILE, 'wb') as f:
                    pickle.dump(geocoding_cache, f)

            # Pequena pausa para no saturar API
            time.sleep(0.05)

        except Exception as e:
            print(f"Error geocodificando '{direccion}': {e}")
            geocoding_cache[direccion] = None
            errores += 1

    # Guardar cache final
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(geocoding_cache, f)

    elapsed_time = time.time() - start_time
    print(f"\nGeocoding completado en {elapsed_time:.1f} segundos ({elapsed_time/60:.1f} minutos)")
    print(f"Exitosas: {geocodificadas} ({geocodificadas/len(direcciones_por_geocodificar)*100:.1f}%)")
    print(f"Errores: {errores} ({errores/len(direcciones_por_geocodificar)*100:.1f}%)")

# Agregar coordenadas al dataframe
print(f"\n[5/8] Agregando coordenadas al dataset...")
df_muestra['latitud'] = df_muestra['DIRECCION'].map(lambda x: geocoding_cache.get(x, {}).get('lat') if geocoding_cache.get(x) else None)
df_muestra['longitud'] = df_muestra['DIRECCION'].map(lambda x: geocoding_cache.get(x, {}).get('lng') if geocoding_cache.get(x) else None)

registros_con_coords = df_muestra['latitud'].notna().sum()
print(f"Registros con coordenadas: {registros_con_coords:,} ({registros_con_coords/len(df_muestra)*100:.1f}%)")

# 5. EXTRAER POIs DE OPENSTREETMAP
print(f"\n[6/8] Extrayendo POIs de OpenStreetMap para Bogota...")

# Cargar cache de POIs si existe
if os.path.exists(POI_CACHE_FILE):
    print("Cargando cache de POIs...")
    with open(POI_CACHE_FILE, 'rb') as f:
        pois_dict = pickle.load(f)
    print(f"POIs cargados desde cache")
else:
    print("Descargando POIs de OpenStreetMap (puede tomar varios minutos)...")

    # Definir bounding box de Bogota
    north, south, east, west = (BOGOTA_BOUNDS['north'], BOGOTA_BOUNDS['south'],
                                 BOGOTA_BOUNDS['east'], BOGOTA_BOUNDS['west'])

    pois_dict = {}

    # Tags de OSM por categoria
    tags_pois = {
        'centros_comerciales': {'shop': 'mall'},
        'estadios': {'leisure': 'stadium'},
        'bares': {'amenity': ['bar', 'pub', 'nightclub']},
        'colegios': {'amenity': 'school'},
        'hospitales': {'amenity': 'hospital'},
        'universidades': {'amenity': 'university'},
        'transmilenio': {'network': 'TransMilenio'}
    }

    for categoria, tags in tags_pois.items():
        try:
            print(f"  Descargando: {categoria}...")
            pois = ox.features_from_bbox(north, south, east, west, tags=tags)

            # Extraer coordenadas
            coords = []
            for idx, row in pois.iterrows():
                try:
                    if row.geometry.geom_type == 'Point':
                        coords.append((row.geometry.y, row.geometry.x))
                    elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                        centroid = row.geometry.centroid
                        coords.append((centroid.y, centroid.x))
                except:
                    continue

            pois_dict[categoria] = coords
            print(f"    {len(coords)} POIs encontrados")
        except Exception as e:
            print(f"    Error descargando {categoria}: {e}")
            pois_dict[categoria] = []

    # Guardar cache
    with open(POI_CACHE_FILE, 'wb') as f:
        pickle.dump(pois_dict, f)
    print("Cache de POIs guardado")

# Resumen de POIs
print("\nResumen de POIs:")
for categoria, coords in pois_dict.items():
    print(f"  {categoria}: {len(coords)} POIs")

# 6. CALCULAR DISTANCIAS Y CREAR FEATURES
print(f"\n[7/8] Calculando distancias a POIs y creando features...")

# Solo calcular para registros con coordenadas validas
df_con_coords = df_muestra[df_muestra['latitud'].notna()].copy()
print(f"Calculando distancias para {len(df_con_coords):,} registros con coordenadas validas...")

# Funcion para calcular distancia minima a POIs
def calcular_distancia_minima(lat, lng, pois_coords):
    if not pois_coords:
        return np.nan
    punto = (lat, lng)
    distancias = [geodesic(punto, poi).meters for poi in pois_coords]
    return min(distancias) if distancias else np.nan

# Funcion para contar POIs en radio
def contar_pois_radio(lat, lng, pois_coords, radio_metros):
    if not pois_coords:
        return 0
    punto = (lat, lng)
    count = sum(1 for poi in pois_coords if geodesic(punto, poi).meters <= radio_metros)
    return count

categorias = list(pois_dict.keys())

# Calcular distancias minimas
for categoria in categorias:
    print(f"  Calculando distancia a {categoria}...")
    df_con_coords[f'dist_{categoria}'] = df_con_coords.apply(
        lambda row: calcular_distancia_minima(row['latitud'], row['longitud'], pois_dict[categoria]),
        axis=1
    )

# Crear features binarias (cerca = dentro de 1km)
for categoria in categorias:
    df_con_coords[f'cerca_{categoria}'] = (df_con_coords[f'dist_{categoria}'] <= RADIO_PROXIMIDAD).astype(int)

# Contar POIs en radio de 1km
for categoria in categorias:
    print(f"  Contando {categoria} en radio de 1km...")
    df_con_coords[f'num_{categoria}_1km'] = df_con_coords.apply(
        lambda row: contar_pois_radio(row['latitud'], row['longitud'], pois_dict[categoria], RADIO_PROXIMIDAD),
        axis=1
    )

# Merge con registros sin coordenadas (rellenar con NaN)
columnas_geo = ['latitud', 'longitud'] + \
               [f'dist_{cat}' for cat in categorias] + \
               [f'cerca_{cat}' for cat in categorias] + \
               [f'num_{cat}_1km' for cat in categorias]

df_muestra = df_muestra.merge(
    df_con_coords[['CODIGO_ACCIDENTE'] + columnas_geo],
    on='CODIGO_ACCIDENTE',
    how='left',
    suffixes=('', '_new')
)

# Reemplazar columnas duplicadas
for col in columnas_geo:
    if col + '_new' in df_muestra.columns:
        df_muestra[col] = df_muestra[col + '_new']
        df_muestra.drop(col + '_new', axis=1, inplace=True)

print("\nNuevas features creadas:")
print(f"  - 2 features de ubicacion: latitud, longitud")
print(f"  - {len(categorias)} features de distancia: dist_{{categoria}}")
print(f"  - {len(categorias)} features binarias: cerca_{{categoria}}")
print(f"  - {len(categorias)} features de densidad: num_{{categoria}}_1km")
print(f"  Total: {2 + len(categorias)*3} nuevas features geoespaciales")

# 7. GUARDAR DATASET CON FEATURES GEOESPACIALES
print(f"\n[8/8] Guardando dataset con features geoespaciales...")

output_file = 'data/processed/siniestros_features_geo.csv'
df_muestra.to_csv(output_file, index=False)
print(f"Dataset guardado: {output_file}")
print(f"Dimensiones: {df_muestra.shape[0]:,} registros x {df_muestra.shape[1]} columnas")

# Estadisticas finales
print("\n" + "="*80)
print("RESUMEN DE PREPROCESAMIENTO GEOESPACIAL")
print("="*80)

print(f"\nDataset original: {len(df):,} registros")
print(f"Muestra procesada: {len(df_muestra):,} registros ({len(df_muestra)/len(df)*100:.1f}%)")
print(f"\nGeocoding:")
print(f"  - Direcciones unicas: {len(direcciones_unicas):,}")
print(f"  - Geocodificadas exitosamente: {registros_con_coords:,} ({registros_con_coords/len(df_muestra)*100:.1f}%)")
print(f"  - Total en cache: {len(geocoding_cache):,} direcciones")

print(f"\nPOIs extraidos de OpenStreetMap:")
for categoria, coords in pois_dict.items():
    print(f"  - {categoria}: {len(coords)} POIs")

print(f"\nFeatures geoespaciales: {2 + len(categorias)*3} nuevas columnas")
print(f"  - Coordenadas: latitud, longitud")
print(f"  - Distancias: dist_* ({len(categorias)} features)")
print(f"  - Proximidad binaria: cerca_* ({len(categorias)} features)")
print(f"  - Densidad: num_*_1km ({len(categorias)} features)")

print(f"\nArchivos generados:")
print(f"  - {output_file}")
print(f"  - {CACHE_FILE} (cache de geocoding)")
print(f"  - {POI_CACHE_FILE} (cache de POIs)")

print("\n" + "="*80)
print("PREPROCESAMIENTO GEOESPACIAL COMPLETADO")
print("="*80)
