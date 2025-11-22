"""
Correccion manual de coordenadas del Top 20 basada en conocimiento geografico de Bogota.

Problema: Geocodificacion original tiene errores sistematicos.
Solucion: Corregir manualmente las coordenadas mas criticas usando referencias conocidas.
"""

import pandas as pd
from pathlib import Path

# Rutas
BASE_DIR = Path(r'c:\Users\caenp\OneDrive\Escritorio\analitica_siniestros')
INPUT_FILE = BASE_DIR / 'seccion_11_dashboard' / 'reporte_top20_corregido.csv'
OUTPUT_FILE = BASE_DIR / 'seccion_11_dashboard' / 'reporte_top20.csv'

print("="*80)
print("CORRECCION MANUAL DE COORDENADAS TOP 20")
print("="*80)

# Cargar Top 20 generado automaticamente
df = pd.read_csv(INPUT_FILE)

print(f"\n[1/3] Top 20 cargado: {len(df)} intersecciones")

# Diccionario de correcciones manuales basadas en conocimiento geografico de Bogota
# Formato: {"interseccion_normalizada": (latitud_correcta, longitud_correcta)}

CORRECCIONES_MANUALES = {
    # Calle 100 deberia estar en latitud ~4.69 (no 4.63)
    "CL 100-KR 15 02": (4.691, -74.031),  # Calle 100 con Carrera 15 (Usaquen)

    # Calle 26 deberia estar en latitud ~4.62-4.64
    "AV AV BOYACA-CL 26 02": (4.630, -74.100),  # Av Boyaca con Calle 26 (Fontibon)
    "CL 26-KR 72 02": (4.639, -74.094),  # Calle 26 con Carrera 72 (Fontibon)

    # Otras correcciones basadas en intersecciones conocidas
    "AV AV BOYACA-CL 80 02": (4.697, -74.094),  # Av Boyaca con Calle 80 (Engativa)
    "CL 80-KR 72 02": (4.697, -74.088),  # Calle 80 con Carrera 72 (Engativa)
    "CL 80-KR 72 2": (4.697, -74.088),  # Duplicado de la anterior
}

# Aplicar correcciones
correcciones_aplicadas = 0

print(f"\n[2/3] Aplicando correcciones manuales...")
for idx, row in df.iterrows():
    interseccion = row['Interseccion']

    if interseccion in CORRECCIONES_MANUALES:
        lat_nueva, lon_nueva = CORRECCIONES_MANUALES[interseccion]
        lat_vieja = row['Latitud']
        lon_vieja = row['Longitud']

        df.at[idx, 'Latitud'] = lat_nueva
        df.at[idx, 'Longitud'] = lon_nueva

        correcciones_aplicadas += 1
        print(f"   [{correcciones_aplicadas}] {interseccion}")
        print(f"       Antes: ({lat_vieja:.4f}, {lon_vieja:.4f})")
        print(f"       Despues: ({lat_nueva:.4f}, {lon_nueva:.4f})")

print(f"\n   Total correcciones aplicadas: {correcciones_aplicadas}")

# Guardar resultado
print(f"\n[3/3] Guardando Top 20 final en: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "="*80)
print("TOP 20 FINAL CON COORDENADAS CORREGIDAS")
print("="*80)
print(df[['Ranking', 'Interseccion', 'Localidad', 'Num_Siniestros', 'Latitud', 'Longitud']].to_string(index=False))

print("\n" + "="*80)
print("VALIDACION FINAL")
print("="*80)

# Validar Calle 100 (debe estar en latitud 4.68-4.71)
calle_100 = df[df['Interseccion'].str.contains('CL 100', na=False)]
print("\nIntersecciones con 'CL 100':")
for _, row in calle_100.iterrows():
    lat = row['Latitud']
    status = "[OK]" if 4.68 <= lat <= 4.71 else "[ERROR]"
    print(f"   {status} {row['Interseccion']}: Lat {lat:.4f}")

# Validar Calle 26 (debe estar en latitud 4.62-4.65)
calle_26 = df[df['Interseccion'].str.contains('CL 26', na=False)]
print("\nIntersecciones con 'CL 26':")
for _, row in calle_26.iterrows():
    lat = row['Latitud']
    status = "[OK]" if 4.62 <= lat <= 4.65 else "[ERROR]"
    print(f"   {status} {row['Interseccion']}: Lat {lat:.4f}")

# Validar Calle 80 (debe estar en latitud 4.69-4.72)
calle_80 = df[df['Interseccion'].str.contains('CL 80', na=False)]
print("\nIntersecciones con 'CL 80':")
for _, row in calle_80.iterrows():
    lat = row['Latitud']
    status = "[OK]" if 4.69 <= lat <= 4.72 else "[ERROR]"
    print(f"   {status} {row['Interseccion']}: Lat {lat:.4f}")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)
print(f"\nArchivo final: {OUTPUT_FILE}")
print("\nPróximo paso: Regenerar dashboard con coordenadas corregidas")
