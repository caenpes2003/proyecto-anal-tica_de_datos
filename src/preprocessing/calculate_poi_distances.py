"""
Calcular distancias a POIs y crear features geoespaciales
"""

import pandas as pd
import numpy as np
import pickle
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CALCULO DE DISTANCIAS A POIs")
print("="*80)

RADIO_PROXIMIDAD = 1000  # 1 km en metros

# 1. Cargar dataset con coordenadas
print("\n[1/4] Cargando dataset geocodificado...")
df = pd.read_csv('data/processed/siniestros_features_geo.csv')
print(f"Dataset cargado: {len(df):,} registros")

# 2. Cargar POIs
print("\n[2/4] Cargando POIs...")
with open('data/processed/poi_cache.pkl', 'rb') as f:
    pois_dict = pickle.load(f)

total_pois = sum(len(v) for v in pois_dict.values())
print(f"POIs cargados: {total_pois:,} en total")
for categoria, coords in pois_dict.items():
    print(f"  - {categoria}: {len(coords)}")

# 3. Calcular distancias
print("\n[3/4] Calculando distancias a POIs...")

# Funcion optimizada para calcular distancia minima
def calcular_distancia_minima(lat, lng, pois_coords):
    if pd.isna(lat) or pd.isna(lng) or not pois_coords:
        return np.nan
    punto = (lat, lng)
    distancias = [geodesic(punto, poi).meters for poi in pois_coords]
    return min(distancias) if distancias else np.nan

# Funcion para contar POIs en radio
def contar_pois_radio(lat, lng, pois_coords, radio_metros):
    if pd.isna(lat) or pd.isna(lng) or not pois_coords:
        return 0
    punto = (lat, lng)
    count = sum(1 for poi in pois_coords if geodesic(punto, poi).meters <= radio_metros)
    return count

categorias = list(pois_dict.keys())

# Calcular distancias minimas para cada categoria
for i, categoria in enumerate(categorias, 1):
    print(f"  [{i}/{len(categorias)}] Distancias a {categoria}...")
    df[f'dist_{categoria}'] = df.apply(
        lambda row: calcular_distancia_minima(row['latitud'], row['longitud'], pois_dict[categoria]),
        axis=1
    )

# Crear features binarias (cerca = dentro de 1km)
print(f"\n  Creando features binarias (cerca = < {RADIO_PROXIMIDAD}m)...")
for categoria in categorias:
    df[f'cerca_{categoria}'] = (df[f'dist_{categoria}'] <= RADIO_PROXIMIDAD).astype(int)
    # Donde dist es NaN, cerca debe ser 0
    df.loc[df[f'dist_{categoria}'].isna(), f'cerca_{categoria}'] = 0

# Contar POIs en radio de 1km
print(f"\n  Contando POIs en radio de {RADIO_PROXIMIDAD}m...")
for i, categoria in enumerate(categorias, 1):
    print(f"    [{i}/{len(categorias)}] Contando {categoria}...")
    df[f'num_{categoria}_1km'] = df.apply(
        lambda row: contar_pois_radio(row['latitud'], row['longitud'], pois_dict[categoria], RADIO_PROXIMIDAD),
        axis=1
    )

# 4. Estadisticas y guardar
print("\n[4/4] Guardando dataset final...")

output_file = 'data/processed/siniestros_features_geo.csv'
df.to_csv(output_file, index=False)

print(f"\nDataset guardado: {output_file}")
print(f"Dimensiones: {df.shape[0]:,} registros x {df.shape[1]} columnas")

# Estadisticas de features geoespaciales
print("\n" + "="*80)
print("ESTADISTICAS DE FEATURES GEOESPACIALES")
print("="*80)

print(f"\nNuevas features creadas: {2 + len(categorias)*3}")
print(f"  - Coordenadas: 2 (latitud, longitud)")
print(f"  - Distancias minimas: {len(categorias)}")
print(f"  - Proximidad binaria: {len(categorias)}")
print(f"  - Densidad (conteo en 1km): {len(categorias)}")

print(f"\nRegistros con coordenadas validas: {df['latitud'].notna().sum():,} ({df['latitud'].notna().sum()/len(df)*100:.1f}%)")

print(f"\nEstadisticas de distancias (metros):")
for categoria in categorias:
    col = f'dist_{categoria}'
    if col in df.columns:
        validos = df[col].notna().sum()
        if validos > 0:
            print(f"\n  {categoria}:")
            print(f"    Registros con distancia calculada: {validos:,}")
            print(f"    Distancia minima: {df[col].min():.0f}m")
            print(f"    Distancia media: {df[col].mean():.0f}m")
            print(f"    Distancia maxima: {df[col].max():.0f}m")
            cerca = df[f'cerca_{categoria}'].sum()
            print(f"    Siniestros cerca (<1km): {cerca:,} ({cerca/validos*100:.1f}%)")

print("\n" + "="*80)
print("CALCULO COMPLETADO")
print("="*80)
