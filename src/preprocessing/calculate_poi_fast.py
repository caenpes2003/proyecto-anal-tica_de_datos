"""
Calculo RAPIDO de distancias a POIs (solo categorias clave)
Optimizado con numpy vectorization
"""

import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CALCULO RAPIDO DE DISTANCIAS A POIs (CATEGORIAS CLAVE)")
print("="*80)

RADIO_PROXIMIDAD = 1000  # 1 km en metros

# Categorias clave (las mas relevantes para el analisis)
CATEGORIAS_CLAVE = ['centros_comerciales', 'estadios', 'bares', 'transmilenio']

# 1. Cargar dataset con coordenadas
print("\n[1/4] Cargando dataset geocodificado...")
df = pd.read_csv('data/processed/siniestros_features_geo.csv')
print(f"Dataset cargado: {len(df):,} registros")

# Filtrar solo registros con coordenadas validas
df_coords = df[df['latitud'].notna() & df['longitud'].notna()].copy()
print(f"Registros con coordenadas validas: {len(df_coords):,}")

# 2. Cargar POIs
print("\n[2/4] Cargando POIs...")
with open('data/processed/poi_cache.pkl', 'rb') as f:
    pois_dict = pickle.load(f)

# Filtrar solo categorias clave
pois_dict_clave = {k: v for k, v in pois_dict.items() if k in CATEGORIAS_CLAVE}

print(f"POIs cargados (categorias clave):")
for categoria, coords in pois_dict_clave.items():
    print(f"  - {categoria}: {len(coords)}")

# 3. Calcular distancias usando vectorizacion
print("\n[3/4] Calculando distancias (metodo vectorizado rapido)...")

def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    """Calcula distancia haversine vectorizada (mas rapida que geodesic)"""
    R = 6371000  # Radio de la Tierra en metros

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# Calcular para cada categoria
for i, categoria in enumerate(CATEGORIAS_CLAVE, 1):
    print(f"  [{i}/{len(CATEGORIAS_CLAVE)}] {categoria}...")

    pois_coords = pois_dict_clave[categoria]
    if not pois_coords:
        df_coords[f'dist_{categoria}'] = np.nan
        df_coords[f'cerca_{categoria}'] = 0
        df_coords[f'num_{categoria}_1km'] = 0
        continue

    # Convertir POIs a arrays numpy
    pois_lats = np.array([p[0] for p in pois_coords])
    pois_lons = np.array([p[1] for p in pois_coords])

    # Calcular distancias para cada registro
    distancias_min = []
    conteos = []

    for idx, row in df_coords.iterrows():
        lat = row['latitud']
        lon = row['longitud']

        # Calcular distancias a todos los POIs de esta categoria
        dists = haversine_vectorized(lat, lon, pois_lats, pois_lons)

        # Distancia minima
        dist_min = np.min(dists)
        distancias_min.append(dist_min)

        # Contar POIs dentro de 1km
        count = np.sum(dists <= RADIO_PROXIMIDAD)
        conteos.append(count)

    # Asignar al dataframe
    df_coords[f'dist_{categoria}'] = distancias_min
    df_coords[f'cerca_{categoria}'] = (np.array(distancias_min) <= RADIO_PROXIMIDAD).astype(int)
    df_coords[f'num_{categoria}_1km'] = conteos

# Hacer merge con el dataframe original (incluir registros sin coordenadas)
print("\n  Integrando con registros sin coordenadas...")
columnas_nuevas = []
for cat in CATEGORIAS_CLAVE:
    columnas_nuevas.extend([f'dist_{cat}', f'cerca_{cat}', f'num_{cat}_1km'])

# Eliminar columnas antiguas si existen
for col in columnas_nuevas:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Merge
df = df.merge(
    df_coords[['CODIGO_ACCIDENTE'] + columnas_nuevas],
    on='CODIGO_ACCIDENTE',
    how='left'
)

# Rellenar NaN en features binarias y conteos con 0
for cat in CATEGORIAS_CLAVE:
    df[f'cerca_{cat}'].fillna(0, inplace=True)
    df[f'num_{cat}_1km'].fillna(0, inplace=True)

# 4. Guardar
print("\n[4/4] Guardando dataset final...")

output_file = 'data/processed/siniestros_features_geo.csv'
df.to_csv(output_file, index=False)

print(f"\nDataset guardado: {output_file}")
print(f"Dimensiones: {df.shape[0]:,} registros x {df.shape[1]} columnas")

# Estadisticas
print("\n" + "="*80)
print("ESTADISTICAS DE FEATURES GEOESPACIALES")
print("="*80)

print(f"\nNuevas features creadas: {2 + len(CATEGORIAS_CLAVE)*3}")
print(f"  - Coordenadas: 2 (latitud, longitud)")
print(f"  - Distancias minimas: {len(CATEGORIAS_CLAVE)}")
print(f"  - Proximidad binaria: {len(CATEGORIAS_CLAVE)}")
print(f"  - Densidad (conteo en 1km): {len(CATEGORIAS_CLAVE)}")

print(f"\nCategorias incluidas: {', '.join(CATEGORIAS_CLAVE)}")
print(f"Registros con coordenadas validas: {df['latitud'].notna().sum():,} ({df['latitud'].notna().sum()/len(df)*100:.1f}%)")

print(f"\nEstadisticas de distancias (metros):")
for categoria in CATEGORIAS_CLAVE:
    col = f'dist_{categoria}'
    if col in df.columns:
        validos = df[col].notna().sum()
        if validos > 0:
            print(f"\n  {categoria}:")
            print(f"    Registros con distancia calculada: {validos:,}")
            print(f"    Distancia minima: {df[col].min():.0f}m")
            print(f"    Distancia media: {df[col].mean():.0f}m")
            print(f"    Distancia mediana: {df[col].median():.0f}m")
            print(f"    Distancia maxima: {df[col].max():.0f}m")
            cerca = df[f'cerca_{categoria}'].sum()
            print(f"    Siniestros cerca (<1km): {cerca:,} ({cerca/validos*100:.1f}%)")
            if df[f'num_{categoria}_1km'].sum() > 0:
                promedio_pois = df[f'num_{categoria}_1km'].mean()
                print(f"    Promedio POIs en 1km: {promedio_pois:.1f}")

print("\n" + "="*80)
print("CALCULO COMPLETADO")
print("="*80)
