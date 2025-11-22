"""
Script rapido para descargar POIs usando Overpass API directamente
Mucho mas eficiente que OSMnx
"""

import requests
import pickle
import time
import json

print("="*80)
print("DESCARGA RAPIDA DE POIs - OVERPASS API")
print("="*80)

# Coordenadas de Bogota
BOGOTA_BBOX = "4.471,-74.224,4.835,-73.983"  # south,west,north,east

# Overpass API endpoint
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

pois_dict = {}

def query_overpass(query, categoria):
    """Consulta Overpass API y extrae coordenadas"""
    print(f"  Consultando Overpass API...")
    try:
        response = requests.post(OVERPASS_URL, data={'data': query}, timeout=180)
        if response.status_code == 200:
            data = response.json()
            coords = []

            for element in data.get('elements', []):
                if element.get('type') == 'node':
                    lat = element.get('lat')
                    lon = element.get('lon')
                    if lat and lon:
                        coords.append((lat, lon))
                elif element.get('type') == 'way' or element.get('type') == 'relation':
                    # Usar centro del bbox o primer nodo
                    if 'center' in element:
                        lat = element['center'].get('lat')
                        lon = element['center'].get('lon')
                        if lat and lon:
                            coords.append((lat, lon))

            return coords
        else:
            print(f"  Error HTTP: {response.status_code}")
            return []
    except Exception as e:
        print(f"  Error: {e}")
        return []

# 1. Centros Comerciales
print(f"\n[1/7] Descargando centros comerciales...")
query = f"""
[out:json][timeout:180];
(
  node["shop"="mall"]({BOGOTA_BBOX});
  way["shop"="mall"]({BOGOTA_BBOX});
  relation["shop"="mall"]({BOGOTA_BBOX});
);
out center;
"""
coords = query_overpass(query, 'centros_comerciales')
pois_dict['centros_comerciales'] = coords
print(f"  OK: {len(coords)} centros comerciales")
time.sleep(2)

# 2. Estadios
print(f"[2/7] Descargando estadios...")
query = f"""
[out:json][timeout:180];
(
  node["leisure"="stadium"]({BOGOTA_BBOX});
  way["leisure"="stadium"]({BOGOTA_BBOX});
  relation["leisure"="stadium"]({BOGOTA_BBOX});
);
out center;
"""
coords = query_overpass(query, 'estadios')
pois_dict['estadios'] = coords
print(f"  OK: {len(coords)} estadios")
time.sleep(2)

# 3. Bares/Pubs/Discotecas
print(f"[3/7] Descargando bares/pubs/discotecas...")
query = f"""
[out:json][timeout:180];
(
  node["amenity"="bar"]({BOGOTA_BBOX});
  node["amenity"="pub"]({BOGOTA_BBOX});
  node["amenity"="nightclub"]({BOGOTA_BBOX});
  way["amenity"="bar"]({BOGOTA_BBOX});
  way["amenity"="pub"]({BOGOTA_BBOX});
  way["amenity"="nightclub"]({BOGOTA_BBOX});
);
out center;
"""
coords = query_overpass(query, 'bares')
pois_dict['bares'] = coords
print(f"  OK: {len(coords)} bares/pubs/discotecas")
time.sleep(2)

# 4. Colegios
print(f"[4/7] Descargando colegios...")
query = f"""
[out:json][timeout:180];
(
  node["amenity"="school"]({BOGOTA_BBOX});
  way["amenity"="school"]({BOGOTA_BBOX});
  relation["amenity"="school"]({BOGOTA_BBOX});
);
out center;
"""
coords = query_overpass(query, 'colegios')
pois_dict['colegios'] = coords
print(f"  OK: {len(coords)} colegios")
time.sleep(2)

# 5. Hospitales
print(f"[5/7] Descargando hospitales...")
query = f"""
[out:json][timeout:180];
(
  node["amenity"="hospital"]({BOGOTA_BBOX});
  way["amenity"="hospital"]({BOGOTA_BBOX});
  relation["amenity"="hospital"]({BOGOTA_BBOX});
);
out center;
"""
coords = query_overpass(query, 'hospitales')
pois_dict['hospitales'] = coords
print(f"  OK: {len(coords)} hospitales")
time.sleep(2)

# 6. Universidades
print(f"[6/7] Descargando universidades...")
query = f"""
[out:json][timeout:180];
(
  node["amenity"="university"]({BOGOTA_BBOX});
  way["amenity"="university"]({BOGOTA_BBOX});
  relation["amenity"="university"]({BOGOTA_BBOX});
);
out center;
"""
coords = query_overpass(query, 'universidades')
pois_dict['universidades'] = coords
print(f"  OK: {len(coords)} universidades")
time.sleep(2)

# 7. TransMilenio (estaciones de bus)
print(f"[7/7] Descargando estaciones TransMilenio...")
query = f"""
[out:json][timeout:180];
(
  node["public_transport"="station"]["bus"="yes"]({BOGOTA_BBOX});
  node["highway"="bus_stop"]["network"="TransMilenio"]({BOGOTA_BBOX});
  way["public_transport"="station"]({BOGOTA_BBOX});
);
out center;
"""
coords = query_overpass(query, 'transmilenio')
pois_dict['transmilenio'] = coords
print(f"  OK: {len(coords)} estaciones TransMilenio")

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
