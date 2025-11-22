"""
Script para descargar POIs de OpenStreetMap
Corrige error de sintaxis de OSMnx
"""

import osmnx as ox
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DESCARGA DE POIs DE OPENSTREETMAP")
print("="*80)

# Coordenadas de Bogota
BOGOTA_BOUNDS = {
    'north': 4.835,
    'south': 4.471,
    'east': -73.983,
    'west': -74.224
}

north, south, east, west = (BOGOTA_BOUNDS['north'], BOGOTA_BOUNDS['south'],
                             BOGOTA_BOUNDS['east'], BOGOTA_BOUNDS['west'])

pois_dict = {}

# Tags de OSM por categoria
print("\nDescargando POIs de Bogota (puede tomar varios minutos)...\n")

# 1. Centros Comerciales
print("[1/7] Descargando centros comerciales...")
try:
    pois = ox.features_from_bbox(bbox=(north, south, east, west), tags={'shop': 'mall'})
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
    pois_dict['centros_comerciales'] = coords
    print(f"  OK: {len(coords)} centros comerciales encontrados")
except Exception as e:
    print(f"  Error: {e}")
    pois_dict['centros_comerciales'] = []

# 2. Estadios
print("[2/7] Descargando estadios...")
try:
    pois = ox.features_from_bbox(bbox=(north, south, east, west), tags={'leisure': 'stadium'})
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
    pois_dict['estadios'] = coords
    print(f"  OK: {len(coords)} estadios encontrados")
except Exception as e:
    print(f"  Error: {e}")
    pois_dict['estadios'] = []

# 3. Bares - Multiple tags
print("[3/7] Descargando bares/pubs/discotecas...")
coords_bares = []
for amenity_type in ['bar', 'pub', 'nightclub']:
    try:
        pois = ox.features_from_bbox(bbox=(north, south, east, west), tags={'amenity': amenity_type})
        for idx, row in pois.iterrows():
            try:
                if row.geometry.geom_type == 'Point':
                    coords_bares.append((row.geometry.y, row.geometry.x))
                elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = row.geometry.centroid
                    coords_bares.append((centroid.y, centroid.x))
            except:
                continue
    except Exception as e:
        print(f"    Error descargando {amenity_type}: {e}")
pois_dict['bares'] = list(set(coords_bares))  # Eliminar duplicados
print(f"  OK: {len(pois_dict['bares'])} bares/pubs/discotecas encontrados")

# 4. Colegios
print("[4/7] Descargando colegios...")
try:
    pois = ox.features_from_bbox(bbox=(north, south, east, west), tags={'amenity': 'school'})
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
    pois_dict['colegios'] = coords
    print(f"  OK: {len(coords)} colegios encontrados")
except Exception as e:
    print(f"  Error: {e}")
    pois_dict['colegios'] = []

# 5. Hospitales
print("[5/7] Descargando hospitales...")
try:
    pois = ox.features_from_bbox(bbox=(north, south, east, west), tags={'amenity': 'hospital'})
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
    pois_dict['hospitales'] = coords
    print(f"  OK: {len(coords)} hospitales encontrados")
except Exception as e:
    print(f"  Error: {e}")
    pois_dict['hospitales'] = []

# 6. Universidades
print("[6/7] Descargando universidades...")
try:
    pois = ox.features_from_bbox(bbox=(north, south, east, west), tags={'amenity': 'university'})
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
    pois_dict['universidades'] = coords
    print(f"  OK: {len(coords)} universidades encontradas")
except Exception as e:
    print(f"  Error: {e}")
    pois_dict['universidades'] = []

# 7. TransMilenio - probar diferentes tags
print("[7/7] Descargando estaciones TransMilenio...")
coords_tm = []
# Intentar con diferentes tags
tags_tm = [
    {'network': 'TransMilenio'},
    {'public_transport': 'station', 'bus': 'yes'},
    {'highway': 'bus_stop', 'network': 'TransMilenio'}
]
for tags in tags_tm:
    try:
        pois = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
        for idx, row in pois.iterrows():
            try:
                if row.geometry.geom_type == 'Point':
                    coords_tm.append((row.geometry.y, row.geometry.x))
                elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = row.geometry.centroid
                    coords_tm.append((centroid.y, centroid.x))
            except:
                continue
    except Exception as e:
        pass
pois_dict['transmilenio'] = list(set(coords_tm))  # Eliminar duplicados
print(f"  OK: {len(pois_dict['transmilenio'])} estaciones TransMilenio encontradas")

# Guardar cache
POI_CACHE_FILE = 'data/processed/poi_cache.pkl'
with open(POI_CACHE_FILE, 'wb') as f:
    pickle.dump(pois_dict, f)

print("\n" + "="*80)
print("RESUMEN DE POIs DESCARGADOS")
print("="*80)
total_pois = 0
for categoria, coords in pois_dict.items():
    print(f"  {categoria}: {len(coords)} POIs")
    total_pois += len(coords)

print(f"\nTotal POIs: {total_pois}")
print(f"Cache guardado en: {POI_CACHE_FILE}")
print("="*80)
